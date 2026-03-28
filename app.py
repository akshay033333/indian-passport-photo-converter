from __future__ import annotations

import io
from dataclasses import dataclass

import cv2
import numpy as np
import streamlit as st
from PIL import Image


OUTPUT_WIDTH = 630
OUTPUT_HEIGHT = 810
MAX_FILE_SIZE_BYTES = 250 * 1024


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


def whiten_background(image_bgr: np.ndarray) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)

    margin_x = max(10, int(w * 0.08))
    margin_y = max(10, int(h * 0.06))
    rect = (margin_x, margin_y, max(1, w - 2 * margin_x), max(1, h - 2 * margin_y))

    try:
        cv2.grabCut(image_bgr, mask, rect, bg_model, fg_model, 4, cv2.GC_INIT_WITH_RECT)
        foreground = np.where((mask == 1) | (mask == 3), 255, 0).astype(np.uint8)
    except cv2.error:
        foreground = np.full((h, w), 255, dtype=np.uint8)

    kernel = np.ones((5, 5), np.uint8)
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel, iterations=1)
    foreground = cv2.GaussianBlur(foreground, (7, 7), 0)

    alpha = foreground.astype(np.float32) / 255.0
    alpha = alpha[..., None]

    white_bg = np.full_like(image_bgr, 252)
    blended = image_bgr.astype(np.float32) * alpha + white_bg.astype(np.float32) * (1 - alpha)
    return blended.astype(np.uint8)


def build_passport_photo(image: Image.Image) -> tuple[Image.Image, bool]:
    image_bgr = pil_to_bgr(image)
    face = detect_primary_face(image_bgr)

    face_found = face is not None
    if face_found:
        left, top, right, bottom = compute_crop_box(image_bgr.shape[1], image_bgr.shape[0], face)
        image_bgr = image_bgr[top:bottom, left:right]
    else:
        image_bgr = center_crop_to_aspect(image_bgr, OUTPUT_WIDTH / OUTPUT_HEIGHT)

    image_bgr = whiten_background(image_bgr)
    image_bgr = cv2.resize(image_bgr, (OUTPUT_WIDTH, OUTPUT_HEIGHT), interpolation=cv2.INTER_CUBIC)
    return bgr_to_pil(image_bgr), face_found


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


def encode_under_limit(image: Image.Image, size_limit: int = MAX_FILE_SIZE_BYTES) -> tuple[bytes, int]:
    for quality in range(95, 29, -5):
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality, optimize=True)
        data = buffer.getvalue()
        if len(data) <= size_limit:
            return data, quality

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=25, optimize=True)
    return buffer.getvalue(), 25


def main() -> None:
    st.set_page_config(page_title="Passport Photo Formatter", page_icon="📷", layout="centered")

    st.title("Passport Photo Formatter")
    st.write(
        "Upload a JPEG portrait and export a `630 x 810` image with a white background, "
        "tight head-and-shoulders framing, and a JPEG size target under `250 KB`."
    )

    uploaded_file = st.file_uploader("Upload a JPEG image", type=["jpg", "jpeg"])

    st.caption(
        "Target spec: `630x810 px`, plain white/off-white background, "
        "face framed to roughly `80-85%` passport-style coverage."
    )

    if not uploaded_file:
        return

    source_image = Image.open(uploaded_file).convert("RGB")
    result_image, face_found = build_passport_photo(source_image)
    encoded_bytes, quality = encode_under_limit(result_image)

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

    st.download_button(
        label="Download formatted JPEG",
        data=encoded_bytes,
        file_name="passport_photo_630x810.jpg",
        mime="image/jpeg",
    )


if __name__ == "__main__":
    main()
