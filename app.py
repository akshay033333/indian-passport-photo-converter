from __future__ import annotations

import io
import hashlib
import json
import re
import secrets
import smtplib
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from email.message import EmailMessage

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
OTP_SEND_COOLDOWN = 60
OTP_EXPIRY_SECONDS = 10 * 60
MAX_OTP_ATTEMPTS = 5
OTP_LENGTH = 6

# -- Cache --
CACHE_TTL = 1800
CACHE_MAX = 16

# -- Google Sheets analytics --
SESSION_WINDOW = 300
SHEET_RETRIES = 3
SHEET_RETRY_DELAY = 0.4
BG_QUEUE_LIMIT = 200

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
DEBUG_LOG_PATH = "/Users/akshaykailasa/Documents/photo_passport_app/.cursor/debug-7fd596.log"
DEBUG_SESSION_ID = "7fd596"


def _debug_log(run_id: str, hypothesis_id: str, location: str, message: str, data: dict) -> None:
    # #region agent log
    try:
        payload = {
            "sessionId": DEBUG_SESSION_ID,
            "runId": run_id,
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")
    except Exception:
        pass
    # #endregion


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


def _normalize_email(value: str) -> str:
    return value.strip().lower()


def _mask_email(value: str) -> str:
    email = _normalize_email(value)
    if "@" not in email:
        return email
    local, domain = email.split("@", 1)
    if len(local) <= 2:
        masked_local = local[0] + "*"
    else:
        masked_local = local[:2] + "*" * (len(local) - 2)
    return f"{masked_local}@{domain}"


def _hash_otp(email: str, otp: str, salt: str) -> str:
    payload = f"{_normalize_email(email)}|{otp}|{salt}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _clear_otp_state() -> None:
    for key in (
        "otp_email",
        "otp_hash",
        "otp_salt",
        "otp_expires_at",
        "otp_last_sent_at",
        "otp_attempts",
        "otp_verified_at",
    ):
        st.session_state.pop(key, None)


def _send_otp_email(recipient_email: str, otp_code: str) -> tuple[bool, str]:
    smtp_email = _secret("SMTP_EMAIL")
    smtp_password = _secret("SMTP_PASSWORD")
    if not smtp_email or not smtp_password:
        return False, "OTP email is not configured. Please set SMTP_EMAIL and SMTP_PASSWORD."

    msg = EmailMessage()
    msg["From"] = smtp_email
    msg["To"] = recipient_email
    msg["Subject"] = "Your OTP for Passport Photo Download"
    msg.set_content(
        "Your one-time verification code is:\n\n"
        f"{otp_code}\n\n"
        f"This code expires in {OTP_EXPIRY_SECONDS // 60} minutes.\n"
        "If you did not request this, you can ignore this email."
    )
    try:
        with smtplib.SMTP("smtp.gmail.com", 587, timeout=30) as server:
            server.starttls()
            server.login(smtp_email, smtp_password)
            server.send_message(msg)
        return True, f"OTP sent to {_mask_email(recipient_email)}."
    except Exception:
        return False, "Could not send OTP right now. Please try again."


def _request_otp(email: str) -> tuple[bool, str]:
    normalized = _normalize_email(email)
    if not _is_valid_email(normalized):
        return False, "Please enter a valid email address before requesting OTP."

    now = time.time()
    last_sent = float(st.session_state.get("otp_last_sent_at", 0.0))
    otp_email = st.session_state.get("otp_email", "")
    if otp_email == normalized and now - last_sent < OTP_SEND_COOLDOWN:
        wait = int(OTP_SEND_COOLDOWN - (now - last_sent)) + 1
        return False, f"Please wait {wait}s before requesting another OTP."

    otp_code = f"{secrets.randbelow(10 ** OTP_LENGTH):0{OTP_LENGTH}d}"
    otp_salt = secrets.token_hex(8)
    otp_hash = _hash_otp(normalized, otp_code, otp_salt)
    ok, message = _send_otp_email(normalized, otp_code)
    if not ok:
        return False, message

    st.session_state["otp_email"] = normalized
    st.session_state["otp_hash"] = otp_hash
    st.session_state["otp_salt"] = otp_salt
    st.session_state["otp_expires_at"] = now + OTP_EXPIRY_SECONDS
    st.session_state["otp_last_sent_at"] = now
    st.session_state["otp_attempts"] = 0
    st.session_state["email_verified"] = False
    st.session_state["verified_email"] = ""
    return True, message


def _verify_otp(email: str, entered_otp: str) -> tuple[bool, str]:
    normalized = _normalize_email(email)
    otp_email = st.session_state.get("otp_email", "")
    _debug_log(
        "otp-ui-debug",
        "H1",
        "app.py:_verify_otp:entry",
        "verify_otp called",
        {
            "email_matches_otp_email": otp_email == normalized,
            "has_otp_hash": bool(st.session_state.get("otp_hash")),
            "has_otp_salt": bool(st.session_state.get("otp_salt")),
            "has_expiry": bool(st.session_state.get("otp_expires_at")),
            "entered_len": len(entered_otp.strip()),
            "entered_isdigit": entered_otp.strip().isdigit(),
        },
    )
    if otp_email != normalized:
        return False, "Please request OTP for this email first."

    otp_hash = st.session_state.get("otp_hash")
    otp_salt = st.session_state.get("otp_salt")
    expires_at = float(st.session_state.get("otp_expires_at", 0.0))
    if not otp_hash or not otp_salt or not expires_at:
        return False, "OTP session not found. Please request OTP again."

    if time.time() > expires_at:
        _clear_otp_state()
        return False, "OTP has expired. Please request a new one."

    attempts = int(st.session_state.get("otp_attempts", 0))
    if attempts >= MAX_OTP_ATTEMPTS:
        _clear_otp_state()
        return False, "Too many invalid OTP attempts. Please request a new OTP."

    otp_value = entered_otp.strip()
    if not (otp_value.isdigit() and len(otp_value) == OTP_LENGTH):
        _debug_log(
            "otp-ui-debug",
            "H2",
            "app.py:_verify_otp:format_guard",
            "otp format guard rejected input",
            {
                "entered_len": len(otp_value),
                "entered_isdigit": otp_value.isdigit(),
                "otp_length_required": OTP_LENGTH,
            },
        )
        return False, f"Enter a {OTP_LENGTH}-digit OTP."

    if _hash_otp(normalized, otp_value, otp_salt) != otp_hash:
        attempts += 1
        st.session_state["otp_attempts"] = attempts
        remaining = max(0, MAX_OTP_ATTEMPTS - attempts)
        if remaining == 0:
            _clear_otp_state()
            return False, "Too many invalid OTP attempts. Please request a new OTP."
        return False, f"Invalid OTP. {remaining} attempt(s) remaining."

    st.session_state["email_verified"] = True
    st.session_state["verified_email"] = normalized
    st.session_state["otp_verified_at"] = time.time()
    st.session_state.pop("otp_hash", None)
    st.session_state.pop("otp_salt", None)
    st.session_state.pop("otp_attempts", None)
    return True, "Email verified successfully. Download is now enabled."


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
        ws = _worksheet(
            sid,
            sa,
            _secret("GOOGLE_EMAIL_WORKSHEET", "email_leads") or "email_leads",
            ["submitted_at_utc", "session_id", "email", "source"],
        )
        _append_retry(
            ws,
            [datetime.now(timezone.utc).isoformat(), session_id, email.strip().lower(), source],
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
        st.image(
            "assets/requirements.png",
            caption="Official photo requirements illustration",
            use_container_width=True,
        )
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


def _ui_support_section() -> None:
    st.markdown("---")
    st.subheader("Your photo is ready!")
    st.markdown("### Keep this tool free for everyone")
    st.caption(
        "Indian Passport Photo Converter is a solo project to make passport photo conversion "
        "simple and accessible. If this tool saved you time, a small tip helps keep it running."
    )
    st.link_button(
        "☕ Buy me a coffee",
        "https://ko-fi.com/akshay4206",
        type="primary",
        use_container_width=True,
    )


def _on_download(email: str, upload_signature: str, session_id: str) -> None:
    normalized_email = _normalize_email(email)
    is_verified = (
        st.session_state.get("email_verified", False)
        and st.session_state.get("verified_email", "") == normalized_email
    )
    if not is_verified:
        st.session_state["lead_error"] = "Please verify your email with OTP before downloading."
        return

    st.session_state["downloaded"] = True
    st.session_state["pending_reset_nonce"] = True
    st.session_state["show_support_cta"] = True

    # Avoid duplicate rows when users click download repeatedly on same file.
    lead_sig = f"{upload_signature}:{normalized_email}"
    if st.session_state.get("last_saved_lead_sig") == lead_sig:
        return

    ok, msg = _write_user_email(normalized_email, session_id)
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

        st.subheader("Please enter your email address to download the Passport-ready photo")
        email_value = st.text_input(
            "Enter your email to request OTP",
            key=f"download_email_{st.session_state['nonce']}",
            placeholder="name@example.com",
            help="We'll send a one-time password (OTP) to verify email ownership before download.",
        )
        email_normalized = _normalize_email(email_value)
        email_ok = _is_valid_email(email_normalized)
        if st.session_state.get("verified_email") != email_normalized:
            st.session_state["email_verified"] = False
        if not email_value.strip():
            st.caption("Add a valid email address to request OTP.")
        elif not email_ok:
            st.warning("Please enter a valid email address.")
        else:
            otp_cols = st.columns([1, 1, 1])
            with otp_cols[0]:
                if st.button("Send OTP", key=f"send_otp_{st.session_state['nonce']}"):
                    ok, message = _request_otp(email_normalized)
                    if ok:
                        st.success(message)
                    else:
                        st.warning(message)
            otp_value = otp_cols[1].text_input(
                "Enter OTP",
                key=f"otp_value_{st.session_state['nonce']}",
                max_chars=OTP_LENGTH,
                placeholder="6-digit code",
            )
            _debug_log(
                "otp-ui-debug",
                "H3",
                "app.py:main:otp_input_render",
                "otp input rendered",
                {
                    "otp_value_len": len(otp_value),
                    "otp_value_isdigit": otp_value.isdigit() if otp_value else False,
                    "email_ok": email_ok,
                    "email_verified_flag": bool(st.session_state.get("email_verified", False)),
                },
            )
            with otp_cols[2]:
                if st.button("Verify OTP", key=f"verify_otp_{st.session_state['nonce']}"):
                    _debug_log(
                        "otp-ui-debug",
                        "H4",
                        "app.py:main:verify_click",
                        "verify button pressed",
                        {
                            "otp_value_len": len(otp_value.strip()),
                            "otp_value_isdigit": otp_value.strip().isdigit(),
                            "email_matches_verified_email": st.session_state.get("verified_email", "") == email_normalized,
                        },
                    )
                    ok, message = _verify_otp(email_normalized, otp_value)
                    _debug_log(
                        "otp-ui-debug",
                        "H5",
                        "app.py:main:verify_result",
                        "verify result received",
                        {
                            "verify_ok": ok,
                            "message_key": (
                                "invalid_format" if "6-digit" in message.lower()
                                else "request_first" if "request otp" in message.lower()
                                else "expired" if "expired" in message.lower()
                                else "attempts" if "attempt" in message.lower()
                                else "other"
                            ),
                        },
                    )
                    if ok:
                        st.success(message)
                    else:
                        st.warning(message)

            email_verified = (
                st.session_state.get("email_verified", False)
                and st.session_state.get("verified_email", "") == email_normalized
            )
            if email_verified:
                st.caption(f"Verified email: {_mask_email(email_normalized)}")
            else:
                st.caption("Verify your OTP to enable download.")

        dl_col, reset_col = st.columns([3, 1])
        with dl_col:
            email_verified_for_download = (
                st.session_state.get("email_verified", False)
                and st.session_state.get("verified_email", "") == email_normalized
            )
            st.download_button(
                "Download Passport JPEG", encoded,
                file_name="passport_photo_630x810.jpg", mime="image/jpeg",
                on_click=_on_download,
                args=(email_value, sig, sid),
                disabled=not email_verified_for_download,
            )
        with reset_col:
            if st.button("Start Over"):
                st.session_state["nonce"] += 1
                st.rerun()

    _ui_privacy()
    _ui_feedback()
    if st.session_state.get("show_support_cta", False):
        _ui_support_section()
    _ui_footer(active, visits)


if __name__ == "__main__":
    main()
