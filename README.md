# Passport Photo Formatter

This Streamlit app accepts a JPEG portrait and converts it into a passport-style image with:

- `630 x 810` pixels
- JPEG output under `250 KB`
- plain white or off-white background
- head-and-shoulders framing aimed at `80-85%` face coverage

## Run locally

```bash
cd /Users/akshaykailasa/Documents/photo_passport_app
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Notes

- The app uses OpenCV face detection to crop around the face automatically.
- It then applies a foreground extraction step and composites the result over an off-white background.
- If a face is not detected, it falls back to a centered crop.
