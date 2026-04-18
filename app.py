from __future__ import annotations

import io
import ipaddress
import json
import re
import threading
import time
import uuid
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone

import cv2
import gspread
import numpy as np
import streamlit as st
from PIL import Image
from streamlit.errors import StreamlitSecretNotFoundError

# -- Output spec --
OUTPUT_W = 630
OUTPUT_H = 810
OUTPUT_ASPECT = OUTPUT_W / OUTPUT_H
MAX_OUTPUT_BYTES = 250 * 1024

# -- Upload validation --
MAX_UPLOAD_BYTES = 350 * 1024 * 1024
MIN_DIM = 300
MAX_PIXELS = 40_000_000
MAX_ASPECT = 2.0
MIN_ASPECT = 0.4
MIN_FACE_AREA = 0.02
ALLOWED_FORMATS = {"JPEG", "PNG", "MPO"}

# -- Crop geometry --
# We estimate the full head box from the detected facial box and crop so the
# final image stays at passport aspect ratio while keeping the head inside the
# acceptance band Passport Seva typically expects.
TARGET_HEAD_HEIGHT_RATIO = 0.68
MIN_HEAD_HEIGHT_RATIO = 0.62
MAX_HEAD_HEIGHT_RATIO = 0.76
TOP_MARGIN_RATIO = 0.09
MIN_SECOND_FACE_RATIO = 0.45
CENTER_DISTANCE_RATIO = 0.30
PRIMARY_CENTER_WEIGHT = 0.20
MIN_PRIMARY_DOMINANCE_RATIO = 2.2

# -- Rate limits --
UPLOAD_COOLDOWN = 2
FEEDBACK_COOLDOWN = 10
MAX_UPLOADS_HR = 120
MAX_FEEDBACK_HR = 30

# -- Cache --
CACHE_TTL = 1800
CACHE_MAX = 16

# -- Google Sheets analytics --
SESSION_WINDOW = 300
SHEET_RETRIES = 3
SHEET_RETRY_DELAY = 0.4
BG_QUEUE_LIMIT = 200
IP_LOOKUP_TIMEOUT_SECONDS = 4

# -- Feedback --
FB_MIN_CHARS = 10
FB_MAX_CHARS = 1000

# -- Custom CSS --
_GREEN_BTN_CSS = """<style>
div[data-testid="stDownloadButton"]>button{background:#22c55e;color:#fff;border:0;font-weight:600}
div[data-testid="stDownloadButton"]>button:hover{background:#16a34a;color:#fff;border:0}
div[data-testid="stDownloadButton"]>button:active{background:#15803d;color:#fff}
</style>"""
EMAIL_PATTERN = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")


# =========================================================================
# Image processing
# =========================================================================

@dataclass(frozen=True, slots=True)
class FaceBox:
    x: int; y: int; w: int; h: int


@dataclass(frozen=True, slots=True)
class FaceSelection:
    primary: FaceBox
    extras: tuple[FaceBox, ...]


def _pil_to_bgr(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)


def _bgr_to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))


def _detect_faces(bgr: np.ndarray) -> list[FaceBox]:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    eq = cv2.equalizeHist(gray)
    candidates: list[FaceBox] = []

    for img in (gray, eq):
        rects = cascade.detectMultiScale(
            img,
            scaleFactor=1.08,
            minNeighbors=8,
            minSize=(90, 90),
        )
        candidates.extend(FaceBox(int(x), int(y), int(w), int(h)) for x, y, w, h in rects)

    if not candidates:
        return []

    return _filter_faces(candidates, bgr.shape[1], bgr.shape[0])


def _largest_face(bgr: np.ndarray) -> FaceBox | None:
    faces = _detect_faces(bgr)
    return max(faces, key=lambda f: f.w * f.h) if faces else None


def _select_faces(faces: list[FaceBox], img_w: int, img_h: int) -> FaceSelection | None:
    if not faces:
        return None

    img_cx = img_w / 2
    img_cy = img_h / 2
    diag = (img_w ** 2 + img_h ** 2) ** 0.5

    def _score(face: FaceBox) -> float:
        area = _face_area(face)
        cx = face.x + face.w / 2
        cy = face.y + face.h / 2
        center_dist = ((cx - img_cx) ** 2 + (cy - img_cy) ** 2) ** 0.5 / diag
        return area * (1.0 - PRIMARY_CENTER_WEIGHT * center_dist)

    ranked = sorted(faces, key=_score, reverse=True)
    primary = ranked[0]
    primary_area = _face_area(primary)
    extras: list[FaceBox] = []

    for face in ranked[1:]:
        area_ratio = _face_area(face) / primary_area
        cx = face.x + face.w / 2
        cy = face.y + face.h / 2
        primary_cx = primary.x + primary.w / 2
        primary_cy = primary.y + primary.h / 2
        center_dx = abs(cx - primary_cx) / img_w
        center_dy = abs(cy - primary_cy) / img_h
        vertical_offset = (cy - primary_cy) / img_h
        horizontal_overlap = max(
            0,
            min(primary.x + primary.w, face.x + face.w) - max(primary.x, face.x),
        ) / max(1, min(primary.w, face.w))

        if area_ratio < 0.32:
            continue
        if vertical_offset > 0.20 and horizontal_overlap > 0.55:
            # Common false positive on beards, collars, shirt prints, or torso texture
            # directly below the real face.
            continue
        if area_ratio < MIN_SECOND_FACE_RATIO and (center_dx > CENTER_DISTANCE_RATIO or center_dy > 0.22):
            continue
        extras.append(face)

    if extras and primary_area / max(_face_area(face) for face in extras) >= MIN_PRIMARY_DOMINANCE_RATIO:
        extras = []

    return FaceSelection(primary=primary, extras=tuple(extras))


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(v, hi))


def _face_area(f: FaceBox) -> int:
    return f.w * f.h


def _iou(a: FaceBox, b: FaceBox) -> float:
    x1 = max(a.x, b.x)
    y1 = max(a.y, b.y)
    x2 = min(a.x + a.w, b.x + b.w)
    y2 = min(a.y + a.h, b.y + b.h)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    union = _face_area(a) + _face_area(b) - inter
    return inter / union if union else 0.0


def _contains(big: FaceBox, small: FaceBox) -> bool:
    return (
        small.x >= big.x
        and small.y >= big.y
        and small.x + small.w <= big.x + big.w
        and small.y + small.h <= big.y + big.h
    )


def _filter_faces(candidates: list[FaceBox], img_w: int, img_h: int) -> list[FaceBox]:
    if not candidates:
        return []

    candidates = sorted(candidates, key=_face_area, reverse=True)
    deduped: list[FaceBox] = []
    for cand in candidates:
        if any(_iou(cand, kept) > 0.35 or _contains(kept, cand) for kept in deduped):
            continue
        deduped.append(cand)

    if not deduped:
        return []

    largest = deduped[0]
    largest_area = _face_area(largest)
    img_center_x = img_w / 2
    filtered: list[FaceBox] = []

    for face in deduped:
        area_ratio = _face_area(face) / largest_area
        face_center_x = face.x + face.w / 2
        center_ratio = abs(face_center_x - img_center_x) / img_w

        # Ignore tiny or edge detections that commonly appear in shadows,
        # patterns, or background objects.
        if area_ratio < 0.18:
            continue
        if area_ratio < MIN_SECOND_FACE_RATIO and center_ratio > CENTER_DISTANCE_RATIO:
            continue
        filtered.append(face)

    return filtered or [largest]


def _max_crop_inside_image(img_w: int, img_h: int) -> tuple[float, float]:
    if img_w / img_h > OUTPUT_ASPECT:
        crop_h = float(img_h)
        crop_w = crop_h * OUTPUT_ASPECT
    else:
        crop_w = float(img_w)
        crop_h = crop_w / OUTPUT_ASPECT
    return crop_w, crop_h


def _crop_around_face(img_w: int, img_h: int, f: FaceBox) -> tuple[int, int, int, int]:
    est_head_top = f.y - 0.22 * f.h
    est_head_h = 1.34 * f.h
    est_head_w = 1.48 * f.w

    ch = est_head_h / TARGET_HEAD_HEIGHT_RATIO
    cw = ch * OUTPUT_ASPECT
    cw = max(cw, est_head_w * 1.55)
    ch = cw / OUTPUT_ASPECT

    max_cw, max_ch = _max_crop_inside_image(img_w, img_h)
    scale = min(max_cw / cw, max_ch / ch, 1.0)
    cw *= scale
    ch *= scale

    face_center_x = f.x + f.w / 2
    left = _clamp(face_center_x - cw / 2, 0, img_w - cw)
    top = _clamp(est_head_top - ch * TOP_MARGIN_RATIO, 0, img_h - ch)
    return int(round(left)), int(round(top)), int(round(left + cw)), int(round(top + ch))


def _crop_from_box(bgr: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    l, t, r, b = box
    return bgr[t:b, l:r]


def _refine_crop_box(img_w: int, img_h: int, face: FaceBox, box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    l, t, r, b = box
    crop_h = max(1, b - t)
    ratio = _head_height_ratio(face, crop_h)

    if MIN_HEAD_HEIGHT_RATIO <= ratio <= MAX_HEAD_HEIGHT_RATIO:
        return box

    target_crop_h = (1.34 * face.h) / TARGET_HEAD_HEIGHT_RATIO
    target_crop_w = target_crop_h * OUTPUT_ASPECT

    if ratio > MAX_HEAD_HEIGHT_RATIO:
        target_crop_h = max(target_crop_h, crop_h)
        target_crop_w = max(target_crop_w, crop_h * OUTPUT_ASPECT, r - l)
    else:
        target_crop_h = min(target_crop_h, crop_h)
        target_crop_w = min(target_crop_w, crop_h * OUTPUT_ASPECT, r - l)

    max_cw, max_ch = _max_crop_inside_image(img_w, img_h)
    target_crop_w = min(target_crop_w, max_cw)
    target_crop_h = min(target_crop_h, max_ch)
    target_crop_h = target_crop_w / OUTPUT_ASPECT

    cx = face.x + face.w / 2
    est_head_top = face.y - 0.22 * face.h
    left = _clamp(cx - target_crop_w / 2, 0, img_w - target_crop_w)
    top = _clamp(est_head_top - target_crop_h * TOP_MARGIN_RATIO, 0, img_h - target_crop_h)
    return (
        int(round(left)),
        int(round(top)),
        int(round(left + target_crop_w)),
        int(round(top + target_crop_h)),
    )


def _center_crop(bgr: np.ndarray) -> np.ndarray:
    h, w = bgr.shape[:2]
    if w / h > OUTPUT_ASPECT:
        nw = int(h * OUTPUT_ASPECT)
        s = (w - nw) // 2
        return bgr[:, s:s + nw]
    nh = int(w / OUTPUT_ASPECT)
    s = (h - nh) // 2
    return bgr[s:s + nh, :]


def _whiten_bg(bgr: np.ndarray, face: FaceBox | None) -> tuple[np.ndarray, bool]:
    h, w = bgr.shape[:2]
    mask = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
    bg_mdl = np.zeros((1, 65), np.float64)
    fg_mdl = np.zeros((1, 65), np.float64)

    bx, by = max(8, int(w * 0.03)), max(8, int(h * 0.03))
    mask[:by, :] = cv2.GC_BGD
    mask[-by:, :] = cv2.GC_BGD
    mask[:, :bx] = cv2.GC_BGD
    mask[:, -bx:] = cv2.GC_BGD

    mx, my = max(20, int(w * 0.18)), max(20, int(h * 0.12))
    mask[my:h - my, mx:w - mx] = cv2.GC_PR_FGD

    if face is not None:
        fl = int(_clamp(face.x - face.w * 0.9, 0, w - 1))
        ft = int(_clamp(face.y - face.h * 1.2, 0, h - 1))
        fr = int(_clamp(face.x + face.w * 1.9, 1, w))
        fb = int(_clamp(face.y + face.h * 3.2, 1, h))
        if fr > fl and fb > ft:
            mask[ft:fb, fl:fr] = cv2.GC_FGD

    try:
        cv2.grabCut(bgr, mask, None, bg_mdl, fg_mdl, 5, cv2.GC_INIT_WITH_MASK)
        fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    except cv2.error:
        fg = np.full((h, w), 255, np.uint8)

    k = np.ones((5, 5), np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k)
    fg = cv2.dilate(fg, k)
    fg = cv2.GaussianBlur(fg, (7, 7), 0)

    if np.count_nonzero(fg) / (h * w) < 0.18:
        return bgr, False

    a = (fg.astype(np.float32) / 255.0)[..., None]
    white = np.full_like(bgr, 252)
    return (bgr.astype(np.float32) * a + white.astype(np.float32) * (1 - a)).astype(np.uint8), True


def _build_passport(img: Image.Image) -> tuple[Image.Image, bool, bool]:
    bgr = _pil_to_bgr(img)
    faces = _detect_faces(bgr)
    selection = _select_faces(faces, bgr.shape[1], bgr.shape[0])
    face = selection.primary if selection else None

    if face:
        crop_box = _crop_around_face(bgr.shape[1], bgr.shape[0], face)
        crop_box = _refine_crop_box(bgr.shape[1], bgr.shape[0], face, crop_box)
        bgr = _crop_from_box(bgr, crop_box)
    else:
        bgr = _center_crop(bgr)

    cropped_face = _largest_face(bgr)
    bgr, bg_ok = _whiten_bg(bgr, cropped_face)
    bgr = cv2.resize(bgr, (OUTPUT_W, OUTPUT_H), interpolation=cv2.INTER_CUBIC)
    return _bgr_to_pil(bgr), face is not None, bg_ok


def _head_height_ratio(face: FaceBox, frame_h: int) -> float:
    estimated_head_h = 1.34 * face.h
    return estimated_head_h / frame_h if frame_h else 0.0


def _meaningful_face_count(faces: list[FaceBox], img_w: int, img_h: int) -> tuple[int, FaceBox | None]:
    selection = _select_faces(faces, img_w, img_h)
    if not selection:
        return 0, None
    return 1 + len(selection.extras), selection.primary


def _analyze_output(img: Image.Image) -> tuple[int, float]:
    bgr = _pil_to_bgr(img)
    faces = _detect_faces(bgr)
    count, primary = _meaningful_face_count(faces, bgr.shape[1], bgr.shape[0])
    ratio = _head_height_ratio(primary, bgr.shape[0]) if primary else 0.0
    return count, ratio


def _encode_jpeg(img: Image.Image, limit: int = MAX_OUTPUT_BYTES) -> tuple[bytes, int]:
    rgb = img.convert("RGB")
    rgb.info.clear()
    for q in range(95, 24, -5):
        buf = io.BytesIO()
        rgb.save(buf, format="JPEG", quality=q, optimize=True, exif=b"")
        if len(buf.getvalue()) <= limit:
            return buf.getvalue(), q
    buf = io.BytesIO()
    rgb.save(buf, format="JPEG", quality=25, optimize=True, exif=b"")
    return buf.getvalue(), 25


def _normalize(raw: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(raw))
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        return Image.alpha_composite(bg, img.convert("RGBA")).convert("RGB")
    return img.convert("RGB")


@st.cache_data(show_spinner=False, ttl=CACHE_TTL, max_entries=CACHE_MAX)
def _process(raw: bytes) -> tuple[bytes, int, bool, bool, bytes, int, float]:
    result, face_ok, bg_ok = _build_passport(_normalize(raw))
    out_face_count, out_head_ratio = _analyze_output(result)
    encoded, quality = _encode_jpeg(result)
    buf = io.BytesIO()
    result.save(buf, format="JPEG", quality=88, optimize=True)
    return encoded, quality, face_ok, bg_ok, buf.getvalue(), out_face_count, out_head_ratio


def _adjust(img: Image.Image, br: int, ct: int, zoom: int, bg_pct: int) -> Image.Image:
    arr = np.array(img, dtype=np.float32)
    if br:
        arr = np.clip(arr + br, 0, 255)
    if ct:
        f = (100 + ct) / 100.0
        arr = np.clip(128 + f * (arr - 128), 0, 255)
    out = Image.fromarray(arr.astype(np.uint8))

    if zoom != 100:
        w, h = out.size
        s = zoom / 100.0
        nw, nh = int(w * s), int(h * s)
        r = out.resize((nw, nh), Image.LANCZOS)
        lx, ly = (nw - w) // 2, (nh - h) // 2
        out = r.crop((lx, ly, lx + w, ly + h))

    if bg_pct < 100:
        a = np.array(out, dtype=np.float32)
        m = np.all(a > 240, axis=2)
        b = bg_pct / 100.0
        a[m] = a[m] * b + 252 * (1 - b)
        out = Image.fromarray(a.astype(np.uint8))

    return out


# =========================================================================
# Validation
# =========================================================================

def _validate(raw: bytes) -> tuple[bool, str]:
    if len(raw) > MAX_UPLOAD_BYTES:
        return False, "Image must be under 350 MB."

    try:
        with Image.open(io.BytesIO(raw)) as img:
            img.verify()
    except Exception:
        return False, "Uploaded file is not a valid image."

    # Reopen after verify() (which consumes the stream)
    img = Image.open(io.BytesIO(raw))
    w, h = img.size
    fmt = (img.format or "").upper()

    if fmt not in ALLOWED_FORMATS:
        return False, "Only JPG, JPEG, and PNG files are allowed."
    if w < MIN_DIM or h < MIN_DIM:
        return False, f"Image must be at least {MIN_DIM} x {MIN_DIM} pixels."
    if w * h > MAX_PIXELS:
        return False, "Resolution too high. Please upload up to 40 megapixels."

    aspect = w / h
    if aspect > MAX_ASPECT:
        return False, "Landscape/panoramic images are not supported. Upload a portrait photo."
    if aspect < MIN_ASPECT:
        return False, "Image is too narrow. Upload a standard portrait photo."

    try:
        bgr = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
        faces = _detect_faces(bgr)
        face_count, primary = _meaningful_face_count(faces, w, h)
        if not primary:
            return False, (
                "No face detected. Upload a clear, front-facing portrait "
                "with good lighting and no obstructions."
            )
        if face_count > 1:
            return False, (
                f"Multiple faces detected ({face_count}). "
                "Passport photos must contain exactly one person."
            )
        f = primary
        if (f.w * f.h) / (w * h) < MIN_FACE_AREA:
            return False, "Face is too small. Move closer so your face fills the frame."
        head_ratio = _head_height_ratio(f, h)
        if head_ratio < 0.54:
            return False, (
                "Face is too small for passport framing. Move slightly closer and keep your "
                "head and upper shoulders clearly visible."
            )
        if head_ratio > 0.88:
            return False, (
                "Face is too large for passport framing. Step a little farther back so the head "
                "and top of the shoulders fit naturally in the photo."
            )
    except Exception:
        pass

    return True, ""


def _validate_fb(text: str) -> tuple[bool, str]:
    t = text.strip()
    if len(t) < FB_MIN_CHARS:
        return False, f"Feedback must be at least {FB_MIN_CHARS} characters."
    if len(t) > FB_MAX_CHARS:
        return False, f"Feedback must be under {FB_MAX_CHARS} characters."
    return True, ""


def _is_valid_email(value: str) -> bool:
    return bool(EMAIL_PATTERN.fullmatch(value.strip()))


# =========================================================================
# Rate limiting
# =========================================================================

def _rate_ok(action: str, cooldown: int, max_hr: int) -> tuple[bool, str]:
    now = time.time()
    st_ = st.session_state.setdefault(f"rl_{action}", {"ts": 0.0, "ev": []})
    st_["ev"] = [t for t in st_["ev"] if now - t < 3600]

    gap = now - float(st_["ts"])
    if gap < cooldown:
        return False, f"Please wait {int(cooldown - gap) + 1}s before trying again."
    if len(st_["ev"]) >= max_hr:
        return False, "Rate limit reached. Please try again later."

    st_["ts"] = now
    st_["ev"].append(now)
    return True, ""


# =========================================================================
# Google Sheets helpers
# =========================================================================

def _secret(key: str, default: str | None = None) -> str | None:
    try:
        return st.secrets.get(key, default)
    except StreamlitSecretNotFoundError:
        return default


def _service_account() -> tuple[dict | None, str | None]:
    raw = _secret("GOOGLE_SERVICE_ACCOUNT_JSON")
    sid = _secret("GOOGLE_SHEET_ID")
    if not raw or not sid:
        return None, None
    return (json.loads(raw) if isinstance(raw, str) else dict(raw)), sid


@st.cache_resource(show_spinner=False)
def _gsheet_client(info: dict) -> gspread.Client:
    return gspread.service_account_from_dict(info)


def _append_retry(ws: gspread.Worksheet, row: list[str]) -> None:
    for i in range(SHEET_RETRIES):
        try:
            ws.append_row(row, value_input_option="RAW")
            return
        except Exception:
            if i == SHEET_RETRIES - 1:
                raise
            time.sleep(SHEET_RETRY_DELAY * (2 ** i))


def _ensure_header(ws: gspread.Worksheet, header: list[str]) -> None:
    current = ws.row_values(1)
    normalized = [str(c).strip() for c in current]
    if normalized != header:
        ws.update("A1", [header], value_input_option="RAW")


def _worksheet(sheet_id: str, sa: dict, name: str, header: list[str]) -> gspread.Worksheet:
    sp = _gsheet_client(sa).open_by_key(sheet_id)
    try:
        ws = sp.worksheet(name)
    except gspread.WorksheetNotFound:
        ws = sp.add_worksheet(title=name, rows=2000, cols=len(header))
    _ensure_header(ws, header)
    return ws


def _write_feedback(text: str) -> tuple[bool, str]:
    sa, sid = _service_account()
    if not sa or not sid:
        return False, "Feedback storage is not configured."
    try:
        ws = _worksheet(sid, sa, _secret("GOOGLE_SHEET_WORKSHEET", "feedback") or "feedback",
                        ["submitted_at_utc", "feedback"])
        _append_retry(ws, [datetime.now(timezone.utc).isoformat(), text.strip()])
        return True, "Thank you for your feedback."
    except Exception as e:
        print(f"Feedback write error: {e}")
        return False, "Could not submit feedback right now."


def _write_user_email(email: str, session_id: str, source: str = "download_gate") -> tuple[bool, str]:
    sa, sid = _service_account()
    if not sa or not sid:
        return False, "Email storage is not configured."
    try:
        client_ip, location = _get_client_ip_and_location()
        ws = _worksheet(
            sid,
            sa,
            _secret("GOOGLE_EMAIL_WORKSHEET", "email_leads") or "email_leads",
            [
                "submitted_at_utc",
                "session_id",
                "email",
                "source",
                "client_ip",
                "country",
                "region",
                "city",
                "timezone",
                "latitude",
                "longitude",
            ],
        )
        _append_retry(
            ws,
            [
                datetime.now(timezone.utc).isoformat(),
                session_id,
                email.strip().lower(),
                source,
                client_ip,
                location.get("country", ""),
                location.get("region", ""),
                location.get("city", ""),
                location.get("timezone", ""),
                location.get("latitude", ""),
                location.get("longitude", ""),
            ],
        )
        return True, "Email saved."
    except Exception as e:
        print(f"Email write error: {e}")
        return False, "Could not save email right now."


def _write_traffic(event: str, session_id: str, details: str = "") -> None:
    sa, sid = _service_account()
    if not sa or not sid:
        return
    try:
        ws = _worksheet(sid, sa, _secret("GOOGLE_TRAFFIC_WORKSHEET", "traffic") or "traffic",
                        ["submitted_at_utc", "session_id", "event_name", "details"])
        _append_retry(ws, [datetime.now(timezone.utc).isoformat(), session_id, event, details])
    except Exception as e:
        print(f"Traffic write error: {e}")


def _request_headers() -> dict[str, str]:
    ctx = getattr(st, "context", None)
    headers = getattr(ctx, "headers", None)
    if not headers:
        return {}
    return {str(k).lower(): str(v) for k, v in dict(headers).items()}


def _extract_client_ip(headers: dict[str, str]) -> str:
    ip_keys = [
        "x-forwarded-for",
        "x-real-ip",
        "cf-connecting-ip",
        "x-client-ip",
        "true-client-ip",
        "fly-client-ip",
    ]
    for key in ip_keys:
        raw = headers.get(key, "")
        if not raw:
            continue
        first = raw.split(",")[0].strip()
        if first.startswith("[") and "]" in first:
            first = first[1:first.find("]")]
        elif ":" in first and first.count(":") == 1 and "." in first:
            first = first.split(":")[0]
        try:
            ip = ipaddress.ip_address(first)
            return str(ip)
        except ValueError:
            continue
    return "unknown"


@st.cache_data(show_spinner=False, ttl=3600)
def _lookup_ip_location(client_ip: str) -> dict[str, str]:
    if client_ip == "unknown":
        return {}
    try:
        ip = ipaddress.ip_address(client_ip)
        if ip.is_private or ip.is_loopback or ip.is_reserved or ip.is_link_local:
            return {}
    except ValueError:
        return {}

    url = (
        f"http://ip-api.com/json/{client_ip}"
        "?fields=status,country,regionName,city,lat,lon,timezone,query"
    )
    try:
        with urllib.request.urlopen(url, timeout=IP_LOOKUP_TIMEOUT_SECONDS) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return {}

    if payload.get("status") != "success":
        return {}
    return {
        "country": str(payload.get("country", "")),
        "region": str(payload.get("regionName", "")),
        "city": str(payload.get("city", "")),
        "timezone": str(payload.get("timezone", "")),
        "latitude": str(payload.get("lat", "")),
        "longitude": str(payload.get("lon", "")),
    }


def _get_client_ip_and_location() -> tuple[str, dict[str, str]]:
    if "client_ip_meta" in st.session_state:
        meta = st.session_state["client_ip_meta"]
        return str(meta.get("ip", "unknown")), dict(meta.get("location", {}))
    headers = _request_headers()
    client_ip = _extract_client_ip(headers)
    location = _lookup_ip_location(client_ip)
    st.session_state["client_ip_meta"] = {"ip": client_ip, "location": location}
    return client_ip, location


# =========================================================================
# Background traffic (fire-and-forget via thread pool)
# =========================================================================

@st.cache_resource(show_spinner=False)
def _bg_pool() -> ThreadPoolExecutor:
    return ThreadPoolExecutor(max_workers=2, thread_name_prefix="analytics")


_bg_slots = threading.Semaphore(BG_QUEUE_LIMIT)


def _log_event(event: str, session_id: str, details: str = "") -> None:
    if not _bg_slots.acquire(blocking=False):
        return
    def _task() -> None:
        try:
            _write_traffic(event, session_id, details)
        finally:
            _bg_slots.release()
    _bg_pool().submit(_task)


# =========================================================================
# Session & traffic counters
# =========================================================================

@st.cache_resource(show_spinner=False)
def _counters() -> dict:
    return {"seen": {}, "total": {}, "lock": threading.Lock()}


def _session_id() -> str:
    if "sid" not in st.session_state:
        st.session_state["sid"] = str(uuid.uuid4())
    return st.session_state["sid"]


def _tick(sid: str) -> tuple[int, int]:
    c = _counters()
    now = time.time()
    with c["lock"]:
        c["seen"][sid] = now
        c["total"][sid] = True
        stale = [s for s, t in c["seen"].items() if now - t > SESSION_WINDOW]
        for s in stale:
            del c["seen"][s]
        return len(c["seen"]), len(c["total"])


# =========================================================================
# UI sections
# =========================================================================

def _ui_hero() -> None:
    st.title("Convert Any Photo to Indian Passport Seva Format")
    st.markdown(
        "**630 x 810 JPEG under 250 KB** — exactly what Passport Seva accepts. "
        "Upload a portrait, get a compliant file in seconds."
    )
    c1, c2, c3 = st.columns(3)
    c1.markdown("**Exact 630 x 810 px**")
    c2.markdown("**JPEG under 250 KB**")
    c3.markdown("**White background**")


def _ui_checklist() -> None:
    with st.expander("Before you upload — photo checklist", expanded=False):
        st.markdown(
            "- Face centered and looking straight at the camera\n"
            "- Plain white or light background\n"
            "- No shadows on face or wall\n"
            "- No glasses\n"
            "- Dark clothing preferred\n"
            "- Head and shoulders visible"
        )


def _ui_disclaimer() -> None:
    st.info(
        "This tool resizes and compresses your image to Passport Seva format. "
        "Upload a clear, front-facing photo with proper lighting for best results. "
        "Official acceptance depends on photo quality and compliance, not only file dimensions."
    )


def _ui_adjustments() -> tuple[int, int, int, int]:
    with st.expander("Manual adjustments (optional)", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            br = st.slider("Brightness", -50, 50, 0, key="adj_br")
            ct = st.slider("Contrast", -50, 50, 0, key="adj_ct")
        with c2:
            zm = st.slider("Zoom", 80, 120, 100, format="%d%%", key="adj_zm")
            bg = st.slider("Background whiteness", 0, 100, 100, format="%d%%", key="adj_bg")
    return br, ct, zm, bg


def _ui_compliance(
    data: bytes,
    quality: int,
    face_ok: bool,
    bg_ok: bool,
    out_face_count: int,
    out_head_ratio: float,
) -> None:
    st.subheader("Compliance Report")
    kb = len(data) / 1024
    r1, r2 = st.columns(2)
    with r1:
        st.metric("Dimensions", f"{OUTPUT_W} x {OUTPUT_H} px")
        st.metric("File Size", f"{kb:.1f} KB", delta="OK" if kb <= 250 else "Over limit")
    with r2:
        st.metric("Format", f"JPEG (quality {quality})")
        st.metric("Background", "White" if bg_ok else "Needs improvement")
    face_status = "Single face" if out_face_count == 1 else ("Not detected" if out_face_count == 0 else f"{out_face_count} faces")
    ratio_pct = f"{out_head_ratio * 100:.0f}%" if out_head_ratio else "Unavailable"
    st.metric("Single-Person Check", face_status)
    st.metric("Head Size", ratio_pct)
    st.metric("Face Position", "Centered" if face_ok else "Not detected — adjust recommended")
    if not face_ok:
        st.warning("No face detected — centered crop was used. Upload a clear front-facing portrait.")
    if out_face_count > 1:
        st.warning(
            "The generated image still appears to contain more than one face-like region. "
            "Try a cleaner background photo with only one person in frame."
        )
    if out_face_count == 1 and not (MIN_HEAD_HEIGHT_RATIO <= out_head_ratio <= MAX_HEAD_HEIGHT_RATIO):
        st.warning(
            "Head size is close to the Passport Seva rejection band. Try a photo where your "
            "head and top of shoulders are clearly visible with some space above the head."
        )
    if not bg_ok:
        st.warning("Background cleanup was partial. Try a photo with better lighting and contrast.")
    st.caption(
        "The app crops to the passport aspect ratio first and only then resizes to `630 x 810`, "
        "so the face is not stretched or squashed."
    )


def _ui_privacy() -> None:
    st.markdown("---")
    st.subheader("Privacy & Trust")
    p1, p2, p3, p4 = st.columns(4)
    p1.markdown("**Temporary processing**\n\nPhotos processed in memory only")
    p2.markdown("**No storage**\n\nImages never stored permanently")
    p3.markdown("**No account**\n\nNo sign-up or login required")
    p4.markdown("**Free to use**\n\nCompletely free, no hidden charges")


def _ui_feedback() -> None:
    st.divider()
    st.subheader("Feedback")
    st.write("Please let us know if you face any issues.")
    st.session_state.setdefault("fb_n", 0)
    text = st.text_area(
        "Share your feedback",
        key=f"fb_{st.session_state['fb_n']}",
        placeholder="Tell us what went wrong or how we can improve...",
    )
    if st.button("Submit feedback"):
        ok, msg = _rate_ok("fb", FEEDBACK_COOLDOWN, MAX_FEEDBACK_HR)
        if not ok:
            st.warning(msg); return
        ok, msg = _validate_fb(text)
        if not ok:
            st.warning(msg); return
        ok, msg = _write_feedback(text)
        if ok:
            _log_event("feedback_submitted", _session_id())
            st.session_state["fb_toast"] = "Feedback submitted successfully."
            st.session_state["fb_n"] += 1
            st.rerun()
        else:
            st.error(msg)


def _ui_footer(active: int, visits: int) -> None:
    st.markdown("---")
    st.markdown(
        "For queries, write to us at "
        "[supportpassportphotoconversion@gmail.com](mailto:supportpassportphotoconversion@gmail.com)"
    )
    st.caption(f"Live active users: `{active}` | Visits (runtime): `{visits}`")


def _on_download(email: str, upload_signature: str, session_id: str) -> None:
    st.session_state["downloaded"] = True
    st.session_state["pending_reset_nonce"] = True

    # Avoid duplicate rows when users click download repeatedly on same file.
    lead_sig = f"{upload_signature}:{email.strip().lower()}"
    if st.session_state.get("last_saved_lead_sig") == lead_sig:
        return

    if _is_valid_email(email):
        ok, msg = _write_user_email(email, session_id)
        if ok:
            st.session_state["last_saved_lead_sig"] = lead_sig
        else:
            st.session_state["lead_error"] = msg


# =========================================================================
# Main
# =========================================================================

def main() -> None:
    st.set_page_config(page_title="Indian Passport Photo Converter", page_icon="📷", layout="centered")
    st.markdown(_GREEN_BTN_CSS, unsafe_allow_html=True)
    st.session_state.setdefault("nonce", 0)

    if st.session_state.pop("downloaded", False):
        st.toast("Passport photo downloaded!", icon="✅")
    if st.session_state.pop("pending_reset_nonce", False):
        st.session_state["nonce"] += 1
    if "fb_toast" in st.session_state:
        st.toast(st.session_state.pop("fb_toast"), icon="✅")
    if "lead_error" in st.session_state:
        st.warning(st.session_state.pop("lead_error"))

    sid = _session_id()
    active, visits = _tick(sid)
    if not st.session_state.get("v_logged"):
        _log_event("app_visit", sid)
        st.session_state["v_logged"] = True

    _ui_hero()
    st.markdown("---")
    _ui_checklist()

    uploaded = st.file_uploader(
        "Upload a JPG, JPEG, or PNG image",
        type=["jpg", "jpeg", "png"],
        key=f"uploader_{st.session_state['nonce']}",
    )
    _ui_disclaimer()

    if uploaded:
        raw = uploaded.getvalue()
        sig = f"{uploaded.name}:{len(raw)}"

        if st.session_state.get("_ul_sig") != sig:
            ok, msg = _rate_ok("upload", UPLOAD_COOLDOWN, MAX_UPLOADS_HR)
            if not ok:
                st.error(msg); _ui_feedback(); return
            st.session_state["_ul_sig"] = sig

        ok, msg = _validate(raw)
        if not ok:
            st.error(msg); _ui_feedback(); return

        encoded, quality, face_ok, bg_ok, preview, out_face_count, out_head_ratio = _process(raw)

        if st.session_state.get("_trk_sig") != sig:
            _log_event("photo_processed", sid, f"face={face_ok},bg={bg_ok}")
            st.session_state["_trk_sig"] = sig

        br, ct, zm, bg = _ui_adjustments()
        if br or ct or zm != 100 or bg != 100:
            adj = _adjust(Image.open(io.BytesIO(preview)), br, ct, zm, bg)
            encoded, quality = _encode_jpeg(adj)
            buf = io.BytesIO()
            adj.save(buf, format="JPEG", quality=88, optimize=True)
            preview = buf.getvalue()
            out_face_count, out_head_ratio = _analyze_output(adj)

        st.markdown("---")
        c1, c2 = st.columns(2)
        c1.subheader("Original")
        c1.image(raw, use_container_width=True)
        c2.subheader("Passport-Ready")
        c2.image(preview, use_container_width=True)

        _ui_compliance(encoded, quality, face_ok, bg_ok, out_face_count, out_head_ratio)

        email_value = st.text_input(
            "Enter your email to enable download",
            key=f"download_email_{st.session_state['nonce']}",
            placeholder="name@example.com",
            help="We only use this to ensure a valid recipient format before enabling download.",
        )
        email_ok = _is_valid_email(email_value)
        if not email_value.strip():
            st.caption("Add a valid email address to enable the download button.")
        elif not email_ok:
            st.warning("Please enter a valid email address.")

        dl_col, reset_col = st.columns([3, 1])
        with dl_col:
            st.download_button(
                "Download Passport JPEG", encoded,
                file_name="passport_photo_630x810.jpg", mime="image/jpeg",
                on_click=_on_download,
                args=(email_value, sig, sid),
                disabled=not email_ok,
            )
        with reset_col:
            if st.button("Start Over"):
                st.session_state["nonce"] += 1
                st.rerun()

    _ui_privacy()
    _ui_feedback()
    _ui_footer(active, visits)


if __name__ == "__main__":
    main()
