from __future__ import annotations

import io
import json
import time
import uuid
from datetime import datetime, timezone
from dataclasses import dataclass

import cv2
import gspread
import numpy as np
import streamlit as st
from PIL import Image
from streamlit.errors import StreamlitSecretNotFoundError


OUTPUT_WIDTH = 630
OUTPUT_HEIGHT = 810
MAX_FILE_SIZE_BYTES = 250 * 1024
MAX_UPLOAD_SIZE_BYTES = 350 * 1024 * 1024
MIN_UPLOAD_WIDTH = 300
MIN_UPLOAD_HEIGHT = 300
MIN_FEEDBACK_CHARS = 10
MAX_FEEDBACK_CHARS = 1000
SESSION_ACTIVE_WINDOW_SECONDS = 5 * 60


@dataclass
class FaceBox:
    x: int
    y: int
    w: int
    h: int


def pil_to_bgr(image: Image.Image) -> np.ndarray:
    rgb = np.array(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def bgr_to_pil(image: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def detect_primary_face(image_bgr: np.ndarray) -> FaceBox | None:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80),
    )

    if len(faces) == 0:
        return None

    x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
    return FaceBox(int(x), int(y), int(w), int(h))


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(value, maximum))


def compute_crop_box(image_w: int, image_h: int, face: FaceBox) -> tuple[int, int, int, int]:
    target_aspect = OUTPUT_WIDTH / OUTPUT_HEIGHT

    crop_h = face.h / 0.52
    crop_w = crop_h * target_aspect

    crop_h = max(crop_h, face.h * 1.75)
    crop_w = max(crop_w, face.w * 1.55)

    if crop_w / crop_h < target_aspect:
        crop_w = crop_h * target_aspect
    else:
        crop_h = crop_w / target_aspect

    center_x = face.x + face.w / 2
    center_y = face.y + face.h * 0.60

    left = center_x - crop_w / 2
    top = center_y - crop_h * 0.43

    left = clamp(left, 0, image_w - crop_w)
    top = clamp(top, 0, image_h - crop_h)

    right = left + crop_w
    bottom = top + crop_h

    return int(round(left)), int(round(top)), int(round(right)), int(round(bottom))


def whiten_background(image_bgr: np.ndarray, face_hint: FaceBox | None = None) -> tuple[np.ndarray, bool]:
    h, w = image_bgr.shape[:2]
    mask = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)

    border_x = max(8, int(w * 0.03))
    border_y = max(8, int(h * 0.03))
    mask[:border_y, :] = cv2.GC_BGD
    mask[-border_y:, :] = cv2.GC_BGD
    mask[:, :border_x] = cv2.GC_BGD
    mask[:, -border_x:] = cv2.GC_BGD

    center_margin_x = max(20, int(w * 0.18))
    center_margin_y = max(20, int(h * 0.12))
    mask[
        center_margin_y:h - center_margin_y,
        center_margin_x:w - center_margin_x,
    ] = cv2.GC_PR_FGD

    if face_hint is not None:
        left = int(clamp(face_hint.x - face_hint.w * 0.9, 0, w - 1))
        top = int(clamp(face_hint.y - face_hint.h * 1.2, 0, h - 1))
        right = int(clamp(face_hint.x + face_hint.w * 1.9, 1, w))
        bottom = int(clamp(face_hint.y + face_hint.h * 3.2, 1, h))
        if right > left and bottom > top:
            mask[top:bottom, left:right] = cv2.GC_FGD

    try:
        cv2.grabCut(image_bgr, mask, None, bg_model, fg_model, 5, cv2.GC_INIT_WITH_MASK)
        foreground = np.where((mask == 1) | (mask == 3), 255, 0).astype(np.uint8)
    except cv2.error:
        foreground = np.full((h, w), 255, dtype=np.uint8)

    kernel = np.ones((5, 5), np.uint8)
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel, iterations=1)
    foreground = cv2.dilate(foreground, kernel, iterations=1)
    foreground = cv2.GaussianBlur(foreground, (7, 7), 0)

    fg_ratio = float(np.count_nonzero(foreground)) / float(h * w)
    if fg_ratio < 0.18:
        return image_bgr, False

    alpha = foreground.astype(np.float32) / 255.0
    alpha = alpha[..., None]

    white_bg = np.full_like(image_bgr, 252)
    blended = image_bgr.astype(np.float32) * alpha + white_bg.astype(np.float32) * (1 - alpha)
    return blended.astype(np.uint8), True


def build_passport_photo(image: Image.Image) -> tuple[Image.Image, bool, bool]:
    image_bgr = pil_to_bgr(image)
    face = detect_primary_face(image_bgr)

    face_found = face is not None
    if face_found:
        left, top, right, bottom = compute_crop_box(image_bgr.shape[1], image_bgr.shape[0], face)
        image_bgr = image_bgr[top:bottom, left:right]
    else:
        image_bgr = center_crop_to_aspect(image_bgr, OUTPUT_WIDTH / OUTPUT_HEIGHT)

    face_after_crop = detect_primary_face(image_bgr)
    image_bgr, background_refined = whiten_background(image_bgr, face_after_crop)
    image_bgr = cv2.resize(image_bgr, (OUTPUT_WIDTH, OUTPUT_HEIGHT), interpolation=cv2.INTER_CUBIC)
    return bgr_to_pil(image_bgr), face_found, background_refined


def center_crop_to_aspect(image_bgr: np.ndarray, target_aspect: float) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    current_aspect = w / h

    if current_aspect > target_aspect:
        new_w = int(h * target_aspect)
        start_x = (w - new_w) // 2
        return image_bgr[:, start_x:start_x + new_w]

    new_h = int(w / target_aspect)
    start_y = (h - new_h) // 2
    return image_bgr[start_y:start_y + new_h, :]


def strip_image_metadata(image: Image.Image) -> Image.Image:
    # Rebuild the image from raw pixels so source metadata/EXIF is not carried over.
    pixel_data = np.array(image.convert("RGB"))
    return Image.fromarray(pixel_data, mode="RGB")


def encode_under_limit(image: Image.Image, size_limit: int = MAX_FILE_SIZE_BYTES) -> tuple[bytes, int]:
    clean_image = strip_image_metadata(image)
    for quality in range(95, 29, -5):
        buffer = io.BytesIO()
        clean_image.save(buffer, format="JPEG", quality=quality, optimize=True, exif=b"")
        data = buffer.getvalue()
        if len(data) <= size_limit:
            return data, quality

    buffer = io.BytesIO()
    clean_image.save(buffer, format="JPEG", quality=25, optimize=True, exif=b"")
    return buffer.getvalue(), 25


def normalize_uploaded_image(uploaded_file: io.BytesIO) -> Image.Image:
    uploaded_file.seek(0)
    image = Image.open(uploaded_file)
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        rgba_image = image.convert("RGBA")
        white_bg = Image.new("RGBA", rgba_image.size, (255, 255, 255, 255))
        return Image.alpha_composite(white_bg, rgba_image).convert("RGB")

    return image.convert("RGB")


def clear_photo_session() -> None:
    st.session_state["uploader_nonce"] = st.session_state.get("uploader_nonce", 0) + 1
    st.rerun()


@st.cache_resource(show_spinner=False)
def get_google_sheet_client(service_account_info: dict) -> gspread.Client:
    return gspread.service_account_from_dict(service_account_info)


def safe_get_secret(key: str, default: str | None = None) -> str | None:
    try:
        return st.secrets.get(key, default)
    except StreamlitSecretNotFoundError:
        return default


@st.cache_resource(show_spinner=False)
def get_runtime_traffic_state() -> dict[str, dict]:
    return {"sessions_last_seen": {}, "visited_sessions": {}}


def get_or_create_session_id() -> str:
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = str(uuid.uuid4())
    return st.session_state["session_id"]


def register_runtime_traffic(session_id: str) -> tuple[int, int]:
    state = get_runtime_traffic_state()
    now_ts = time.time()
    state["sessions_last_seen"][session_id] = now_ts
    state["visited_sessions"][session_id] = True

    stale_sessions = [
        sid
        for sid, last_seen in state["sessions_last_seen"].items()
        if now_ts - last_seen > SESSION_ACTIVE_WINDOW_SECONDS
    ]
    for sid in stale_sessions:
        state["sessions_last_seen"].pop(sid, None)

    active_count = len(state["sessions_last_seen"])
    total_visits = len(state["visited_sessions"])
    return active_count, total_visits


def append_traffic_to_google_sheet(event_name: str, session_id: str, details: str = "") -> tuple[bool, str]:
    service_account_secret = safe_get_secret("GOOGLE_SERVICE_ACCOUNT_JSON")
    sheet_id = safe_get_secret("GOOGLE_SHEET_ID")
    worksheet_name = safe_get_secret("GOOGLE_TRAFFIC_WORKSHEET", "traffic")

    if not service_account_secret or not sheet_id:
        return (
            False,
            "Traffic storage is not configured. Please set GOOGLE_SERVICE_ACCOUNT_JSON and GOOGLE_SHEET_ID.",
        )

    try:
        if isinstance(service_account_secret, str):
            service_account_info = json.loads(service_account_secret)
        else:
            service_account_info = dict(service_account_secret)

        client = get_google_sheet_client(service_account_info)
        spreadsheet = client.open_by_key(sheet_id)

        try:
            worksheet = spreadsheet.worksheet(worksheet_name)
        except gspread.WorksheetNotFound:
            worksheet = spreadsheet.add_worksheet(title=worksheet_name, rows=2000, cols=4)

        first_row = worksheet.row_values(1)
        if not first_row:
            worksheet.append_row(
                ["submitted_at_utc", "session_id", "event_name", "details"],
                value_input_option="RAW",
            )

        worksheet.append_row(
            [datetime.now(timezone.utc).isoformat(), session_id, event_name, details],
            value_input_option="RAW",
        )
        return True, "Traffic event saved."
    except Exception as exc:
        return False, f"Could not save traffic event: {exc}"


def validate_uploaded_file(uploaded_file: io.BytesIO) -> tuple[bool, str]:
    allowed_mime_types = {"image/jpeg", "image/jpg", "image/png"}
    file_size = getattr(uploaded_file, "size", None)
    if file_size is None:
        file_size = len(uploaded_file.getbuffer())

    if file_size > MAX_UPLOAD_SIZE_BYTES:
        return False, "Image size must be under 350 MB."

    mime_type = getattr(uploaded_file, "type", "")
    if mime_type and mime_type not in allowed_mime_types:
        return False, "Only JPG, JPEG, and PNG files are allowed."

    try:
        uploaded_file.seek(0)
        with Image.open(uploaded_file) as image:
            width, height = image.size
    except Exception:
        return False, "Uploaded file is not a valid image."
    finally:
        uploaded_file.seek(0)

    if width < MIN_UPLOAD_WIDTH or height < MIN_UPLOAD_HEIGHT:
        return False, "Image resolution must be at least 300 x 300 pixels."

    return True, ""


def validate_feedback_text(feedback_text: str) -> tuple[bool, str]:
    feedback = feedback_text.strip()
    if len(feedback) < MIN_FEEDBACK_CHARS:
        return False, "Feedback must be at least 10 characters."
    if len(feedback) > MAX_FEEDBACK_CHARS:
        return False, "Feedback must be under 1000 characters."
    return True, ""


def append_feedback_to_google_sheet(feedback_text: str) -> tuple[bool, str]:
    service_account_secret = safe_get_secret("GOOGLE_SERVICE_ACCOUNT_JSON")
    sheet_id = safe_get_secret("GOOGLE_SHEET_ID")
    worksheet_name = safe_get_secret("GOOGLE_SHEET_WORKSHEET", "feedback")

    if not service_account_secret or not sheet_id:
        return (
            False,
            "Feedback storage is not configured yet. Please set GOOGLE_SERVICE_ACCOUNT_JSON and GOOGLE_SHEET_ID.",
        )

    try:
        if isinstance(service_account_secret, str):
            service_account_info = json.loads(service_account_secret)
        else:
            service_account_info = dict(service_account_secret)

        client = get_google_sheet_client(service_account_info)
        spreadsheet = client.open_by_key(sheet_id)

        try:
            worksheet = spreadsheet.worksheet(worksheet_name)
        except gspread.WorksheetNotFound:
            worksheet = spreadsheet.add_worksheet(title=worksheet_name, rows=1000, cols=2)

        first_row = worksheet.row_values(1)
        if not first_row:
            worksheet.append_row(["submitted_at_utc", "feedback"], value_input_option="RAW")

        worksheet.append_row(
            [datetime.now(timezone.utc).isoformat(), feedback_text.strip()],
            value_input_option="RAW",
        )
        return True, "Thank you for your feedback. It has been saved."
    except Exception as exc:
        return False, f"Could not submit feedback to Google Sheets: {exc}"


def render_feedback_section() -> None:
    st.divider()
    st.subheader("Feedback")
    st.write("Please let us know if you face any issues.")
    st.session_state.setdefault("feedback_nonce", 0)
    feedback_key = f"feedback_text_{st.session_state['feedback_nonce']}"
    feedback_text = st.text_area(
        "Share your feedback",
        key=feedback_key,
        placeholder="Tell us what went wrong or how we can improve...",
    )
    if st.button("Submit feedback"):
        is_valid, validation_message = validate_feedback_text(feedback_text)
        if not is_valid:
            st.warning(validation_message)
            return

        ok, message = append_feedback_to_google_sheet(feedback_text)
        if ok:
            session_id = get_or_create_session_id()
            append_traffic_to_google_sheet("feedback_submitted", session_id, "feedback_form")
            st.session_state["feedback_submitted_toast"] = "Feedback submitted successfully."
            st.session_state["feedback_nonce"] += 1
            st.rerun()
        else:
            st.error(message)


def main() -> None:
    st.set_page_config(page_title="Passport Photo Formatter", page_icon="📷", layout="centered")
    st.session_state.setdefault("uploader_nonce", 0)
    if "feedback_submitted_toast" in st.session_state:
        st.toast(st.session_state.pop("feedback_submitted_toast"), icon="✅")
    session_id = get_or_create_session_id()
    active_count, total_visits = register_runtime_traffic(session_id)
    st.caption(f"Live active users: `{active_count}` | Visits (runtime): `{total_visits}`")
    if not st.session_state.get("visit_logged"):
        append_traffic_to_google_sheet("app_visit", session_id, "home_loaded")
        st.session_state["visit_logged"] = True

    st.title("Passport Photo Formatter")
    st.write(
        "Upload a JPG, JPEG, or PNG portrait and export a `630 x 810` image with a white background, "
        "tight head-and-shoulders framing, and a JPEG size target under `250 KB`."
    )
    uploaded_file = st.file_uploader(
        "Upload a JPG, JPEG, or PNG image",
        type=["jpg", "jpeg", "png"],
        key=f"passport_uploader_{st.session_state['uploader_nonce']}",
    )

    st.caption(
        "Target spec: `630x810 px`, plain white/off-white background, "
        "face framed to roughly `80-85%` passport-style coverage."
    )

    if uploaded_file:
        is_valid, validation_message = validate_uploaded_file(uploaded_file)
        if not is_valid:
            st.error(validation_message)
            render_feedback_section()
            return

        source_image = normalize_uploaded_image(uploaded_file)
        result_image, face_found, background_refined = build_passport_photo(source_image)
        encoded_bytes, quality = encode_under_limit(result_image)
        upload_signature = f"{uploaded_file.name}:{getattr(uploaded_file, 'size', 0)}"
        if st.session_state.get("last_tracked_upload") != upload_signature:
            details = f"face_found={face_found},background_refined={background_refined}"
            append_traffic_to_google_sheet("photo_processed", session_id, details)
            st.session_state["last_tracked_upload"] = upload_signature

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original")
            st.image(source_image, use_container_width=True)

        with col2:
            st.subheader("Formatted")
            st.image(result_image, use_container_width=True)

        st.success(
            f"Export ready: `630x810 px`, `{len(encoded_bytes) / 1024:.1f} KB`, JPEG quality `{quality}`."
        )

        if face_found:
            st.info("Face detection succeeded and the crop was aligned around the detected face.")
        else:
            st.warning(
                "No face was detected, so the app used a centered crop. "
                "For best results, upload a clear front-facing portrait."
            )

        if not background_refined:
            st.warning(
                "Background cleanup was partially skipped to preserve hair/body details. "
                "Try a photo with better lighting and clearer contrast for best white-background results."
            )

        st.download_button(
            label="Download formatted JPEG",
            data=encoded_bytes,
            file_name="passport_photo_630x810.jpg",
            mime="image/jpeg",
        )

        action_col1, action_col2 = st.columns(2)
        with action_col1:
            if st.button("Clear"):
                clear_photo_session()
        with action_col2:
            if st.button("I'm done"):
                clear_photo_session()

    render_feedback_section()


if __name__ == "__main__":
    main()
