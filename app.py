from __future__ import annotations

import io
import json
import threading
import time
import uuid
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
ALLOWED_FORMATS = {"JPEG", "PNG"}

# -- Crop geometry --
# We estimate the full head box from the detected facial box and crop so the
# head occupies a passport-style portion of the final frame without distortion.
TARGET_HEAD_HEIGHT_RATIO = 0.74
TOP_MARGIN_RATIO = 0.07
MIN_SECOND_FACE_RATIO = 0.38
CENTER_DISTANCE_RATIO = 0.42

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

# -- Feedback --
FB_MIN_CHARS = 10
FB_MAX_CHARS = 1000

# -- Custom CSS --
_GREEN_BTN_CSS = """<style>
div[data-testid="stDownloadButton"]>button{background:#22c55e;color:#fff;border:0;font-weight:600}
div[data-testid="stDownloadButton"]>button:hover{background:#16a34a;color:#fff;border:0}
div[data-testid="stDownloadButton"]>button:active{background:#15803d;color:#fff}
</style>"""


# =========================================================================
# Image processing
# =========================================================================

@dataclass(frozen=True, slots=True)
class FaceBox:
    x: int; y: int; w: int; h: int


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
    face = max(faces, key=_face_area) if faces else None

    if face:
        l, t, r, b = _crop_around_face(bgr.shape[1], bgr.shape[0], face)
        bgr = bgr[t:b, l:r]
    else:
        bgr = _center_crop(bgr)

    bgr, bg_ok = _whiten_bg(bgr, _largest_face(bgr))
    bgr = cv2.resize(bgr, (OUTPUT_W, OUTPUT_H), interpolation=cv2.INTER_CUBIC)
    return _bgr_to_pil(bgr), face is not None, bg_ok


def _head_height_ratio(face: FaceBox, frame_h: int) -> float:
    estimated_head_h = 1.34 * face.h
    return estimated_head_h / frame_h if frame_h else 0.0


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
def _process(raw: bytes) -> tuple[bytes, int, bool, bool, bytes]:
    result, face_ok, bg_ok = _build_passport(_normalize(raw))
    encoded, quality = _encode_jpeg(result)
    buf = io.BytesIO()
    result.save(buf, format="JPEG", quality=88, optimize=True)
    return encoded, quality, face_ok, bg_ok, buf.getvalue()


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
        if not faces:
            return False, (
                "No face detected. Upload a clear, front-facing portrait "
                "with good lighting and no obstructions."
            )
        if len(faces) > 1:
            return False, (
                f"Multiple faces detected ({len(faces)}). "
                "Passport photos must contain exactly one person."
            )
        f = faces[0]
        if (f.w * f.h) / (w * h) < MIN_FACE_AREA:
            return False, "Face is too small. Move closer so your face fills the frame."
        head_ratio = _head_height_ratio(f, h)
        if head_ratio < 0.58:
            return False, (
                "Face is too small for passport framing. Move closer, keep the camera about "
                "1.2 to 1.5 meters away, and make sure the head fills most of the frame."
            )
        if head_ratio > 0.82:
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


def _worksheet(sheet_id: str, sa: dict, name: str, header: list[str]) -> gspread.Worksheet:
    sp = _gsheet_client(sa).open_by_key(sheet_id)
    try:
        ws = sp.worksheet(name)
    except gspread.WorksheetNotFound:
        ws = sp.add_worksheet(title=name, rows=2000, cols=len(header))
    if not ws.row_values(1):
        _append_retry(ws, header)
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


def _ui_compliance(data: bytes, quality: int, face_ok: bool, bg_ok: bool) -> None:
    st.subheader("Compliance Report")
    kb = len(data) / 1024
    r1, r2 = st.columns(2)
    with r1:
        st.metric("Dimensions", f"{OUTPUT_W} x {OUTPUT_H} px")
        st.metric("File Size", f"{kb:.1f} KB", delta="OK" if kb <= 250 else "Over limit")
    with r2:
        st.metric("Format", f"JPEG (quality {quality})")
        st.metric("Background", "White" if bg_ok else "Needs improvement")
    st.metric("Face Position", "Centered" if face_ok else "Not detected — adjust recommended")
    if not face_ok:
        st.warning("No face detected — centered crop was used. Upload a clear front-facing portrait.")
    if not bg_ok:
        st.warning("Background cleanup was partial. Try a photo with better lighting and contrast.")
    st.caption(
        "The crop preserves the target passport aspect ratio before resizing, so the final "
        "630 x 810 image is cropped and scaled without stretching the face."
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


# =========================================================================
# Main
# =========================================================================

def main() -> None:
    st.set_page_config(page_title="Indian Passport Photo Converter", page_icon="📷", layout="centered")
    st.markdown(_GREEN_BTN_CSS, unsafe_allow_html=True)
    st.session_state.setdefault("nonce", 0)

    if st.session_state.pop("downloaded", False):
        st.toast("Passport photo downloaded!", icon="✅")
        st.session_state["nonce"] += 1
    if "fb_toast" in st.session_state:
        st.toast(st.session_state.pop("fb_toast"), icon="✅")

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

        encoded, quality, face_ok, bg_ok, preview = _process(raw)

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

        st.markdown("---")
        c1, c2 = st.columns(2)
        c1.subheader("Original")
        c1.image(raw, use_container_width=True)
        c2.subheader("Passport-Ready")
        c2.image(preview, use_container_width=True)

        _ui_compliance(encoded, quality, face_ok, bg_ok)

        dl_col, reset_col = st.columns([3, 1])
        with dl_col:
            st.download_button(
                "Download Passport JPEG", encoded,
                file_name="passport_photo_630x810.jpg", mime="image/jpeg",
                on_click=lambda: st.session_state.update(downloaded=True),
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
