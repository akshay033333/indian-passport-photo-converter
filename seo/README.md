# Google Search Console verification

## Add your verification file

1. In [Google Search Console](https://search.google.com/search-console), add property `https://passportphoto-converter.streamlit.app/`.
2. Choose **HTML file** verification and download the file (e.g. `google1234567890abcdef.html`).
3. Place that file in this folder (do not rename it).
4. Push to GitHub and wait for Streamlit Cloud to redeploy.

`server.py` automatically serves any `google*.html` file here at the site root.

The file is public by design; it is not a secret.

## After deploy — Streamlit property

1. Confirm the verification file loads:
   `https://passportphoto-converter.streamlit.app/<your-filename>.html`
2. Click **Verify** in Search Console.
3. Submit sitemap: `https://passportphoto-converter.streamlit.app/sitemap.xml`
4. Use **URL Inspection** → **Request indexing** for `https://passportphoto-converter.streamlit.app/`
5. Check indexing progress with: `site:passportphoto-converter.streamlit.app`

## Streamlit Cloud settings

- Main file path: `server.py` (not `app.py`)
- App visibility: **Public**
- Custom subdomain: `passportphoto-converter`

## GitHub Pages (optional second property)

1. Repo **Settings → Pages** → Source: `main` branch, `/docs` folder.
2. Add a separate GSC property for `https://akshay033333.github.io/indian-passport-photo-converter/`
3. Verify with HTML file or meta tag on `docs/index.html` (full HTML control).
