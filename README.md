<div align="center">

# Indian Passport Photo Converter

**Convert any portrait to Passport Seva format — in under a minute.**

`630 x 810 px` &nbsp;&middot;&nbsp; `White background` &nbsp;&middot;&nbsp; `Under 250 KB`

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)](https://indianpassportphoto-converter-594qkvflp9pkfixcgakszh.streamlit.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e)](LICENSE)
[![Open Source](https://img.shields.io/badge/Open%20Source-%E2%9D%A4-red)](#)

[**Try the Live App**](https://indianpassportphoto-converter-594qkvflp9pkfixcgakszh.streamlit.app/) &nbsp;&middot;&nbsp; [View on GitHub](https://github.com/akshay033333/indian-passport-photo-converter) &nbsp;&middot;&nbsp; [Star this Repo](https://github.com/akshay033333/indian-passport-photo-converter/stargazers)

</div>

---

## The Problem

Uploading a passport photo to **Passport Seva** should be simple — but it isn't. The portal rejects images that don't meet its exact specifications: **630 x 810 pixels, JPEG format, under 250 KB, white background**. Most people waste hours resizing, compressing, and re-uploading before getting it right.

This is especially painful for **NRIs, Indian students abroad, and anyone renewing their passport** who can't easily visit a local photo studio.

## The Solution

Upload a portrait. Get a **Passport Seva–compliant JPEG** back in seconds. No signup, no payment, no data stored.

## Features

- **Exact output format** — 630 x 810 px JPEG, always under 250 KB
- **Auto face detection** — crops and centers your face using OpenCV
- **White background** — replaces colored backgrounds with clean white via GrabCut
- **Smart validation** — rejects landscape images, multi-face photos, and non-portraits before processing
- **Manual adjustments** — fine-tune brightness, contrast, zoom, and background whiteness
- **Before-upload checklist** — includes an official requirements illustration to help users capture a valid photo
- **Side-by-side preview** — compare original and passport-ready images before downloading
- **Compliance report** — shows dimensions, file size, format, background status, and face position
- **Privacy-first** — photos are processed in memory only, never stored
- **Completely free** — no account, no watermark, no hidden charges

## How It Works

1. **Upload** a JPG or PNG portrait
2. The app **detects your face**, crops to passport proportions, and whitens the background
3. The result is **compressed** to meet the 250 KB limit
4. **Preview** the output, adjust if needed, and **download**

## Tech Stack

| Component | Technology |
|---|---|
| Frontend & framework | [Streamlit](https://streamlit.io/) |
| Face detection & image processing | [OpenCV](https://opencv.org/), [Pillow](https://pillow.readthedocs.io/), NumPy |
| Feedback & analytics | Google Sheets via [gspread](https://docs.gspread.org/) |
| Hosting | [Streamlit Community Cloud](https://streamlit.io/cloud) |
| Daily reports | GitHub Actions + openpyxl |
| Language | Python 3.12 |

## Local Setup

**Prerequisites:** Python 3.11 or 3.12

```bash
git clone https://github.com/akshay033333/indian-passport-photo-converter.git
cd indian-passport-photo-converter
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

### Run with Docker

```bash
docker build -t passport-photo-app .
docker run --rm -p 8501:8501 passport-photo-app
```

## Usage

1. Open the [live app](https://indianpassportphoto-converter-594qkvflp9pkfixcgakszh.streamlit.app/) or run locally
2. Upload a clear, front-facing portrait (JPG or PNG)
3. Review the side-by-side preview and compliance report
4. Optionally adjust brightness, contrast, zoom, or background whiteness
5. Click **Download Passport JPEG**
6. Upload the downloaded file directly to Passport Seva

## Project Structure

```
indian-passport-photo-converter/
├── app.py                  # Streamlit application
├── report.py               # Daily dashboard report (KPIs, charts, email)
├── requirements.txt        # Python dependencies
├── runtime.txt             # Python version for Streamlit Cloud
├── Dockerfile              # Docker containerization
├── .github/workflows/      # GitHub Actions (daily report scheduling)
├── .streamlit/             # Streamlit secrets (git-ignored)
├── assets/                 # App/README image assets
│   ├── poster.png
│   └── requirements.png
├── output/                 # Generated reports (git-ignored)
├── LICENSE
└── README.md
```

## Why This Matters

Every year, **millions of Indians** apply for or renew their passports through Passport Seva. A significant portion of applications face delays due to **rejected photos** that don't meet the portal's strict format requirements. This tool eliminates that friction — especially for people applying from abroad who may not have access to a local photo studio that knows the exact Indian specifications.

## Contributing

Contributions are welcome. If you'd like to improve the app:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Make your changes and commit (`git commit -m "Add your feature"`)
4. Push to your fork (`git push origin feature/your-feature`)
5. Open a Pull Request

## License

This project is licensed under the [MIT License](LICENSE).

## Support

For questions or issues, email **[supportpassportphotoconversion@gmail.com](mailto:supportpassportphotoconversion@gmail.com)** or [open an issue](https://github.com/akshay033333/indian-passport-photo-converter/issues).

---

If this tool saved you time, consider giving it a **star** on GitHub — it helps others find it.

[![Star on GitHub](https://img.shields.io/github/stars/akshay033333/indian-passport-photo-converter?style=social)](https://github.com/akshay033333/indian-passport-photo-converter/stargazers)

---

## Preview

<div align="center">
  <img src="assets/poster.png" alt="Indian Passport Photo Converter" width="480">
</div>
