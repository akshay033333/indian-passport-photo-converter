"""
Dynamic daily dashboard generator.

This script REPLACES the previous Excel report pipeline.
It now:
1) Reads traffic + feedback from Google Sheets
2) Builds a dynamic HTML dashboard (`output/preview.html`) with Chart.js
3) Emails a daily summary + the HTML file
4) Runs via GitHub Actions schedule
"""
from __future__ import annotations

import argparse
import json
import os
import smtplib
import sys
import time
from collections import Counter
from datetime import date, datetime
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

import gspread

SECRETS_PATH = Path(__file__).parent / ".streamlit" / "secrets.toml"
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_FILE = OUTPUT_DIR / "preview.html"
APP_URL = "https://indianpassportphoto-converter-594qkvflp9pkfixcgakszh.streamlit.app/"
SMTP_MAX_RETRIES = 5
SMTP_RETRY_BASE_SECONDS = 5


def _load_toml_secrets() -> dict[str, str]:
    if not SECRETS_PATH.exists():
        return {}
    text = SECRETS_PATH.read_text()
    out: dict[str, str] = {}
    i = 0
    lines = text.splitlines()
    while i < len(lines):
        line = lines[i].strip()
        if "=" not in line or line.startswith("#"):
            i += 1
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip()
        if val.startswith("'''") or val.startswith('"""'):
            delim = val[:3]
            parts = [val[3:]]
            i += 1
            while i < len(lines):
                if delim in lines[i]:
                    parts.append(lines[i].split(delim)[0])
                    break
                parts.append(lines[i])
                i += 1
            out[key] = "\n".join(parts)
        else:
            out[key] = val.strip("'").strip('"')
        i += 1
    return out


def _secret(key: str) -> str | None:
    return os.environ.get(key) or _load_toml_secrets().get(key)


def _service_account() -> tuple[dict, str]:
    raw = _secret("GOOGLE_SERVICE_ACCOUNT_JSON")
    sheet_id = _secret("GOOGLE_SHEET_ID")
    if not raw or not sheet_id:
        sys.exit("Missing GOOGLE_SERVICE_ACCOUNT_JSON or GOOGLE_SHEET_ID")
    if isinstance(raw, str):
        raw = raw.strip().strip("'").strip('"').strip()
        return json.loads(raw), sheet_id
    return dict(raw), sheet_id


def _fetch_rows(sheet_id: str, client: gspread.Client, name: str) -> list[dict]:
    try:
        ws = client.open_by_key(sheet_id).worksheet(name)
        return ws.get_all_records()
    except gspread.WorksheetNotFound:
        return []


def _parse_day(iso_str: str) -> date | None:
    try:
        return datetime.fromisoformat(iso_str.replace("Z", "+00:00")).date()
    except Exception:
        return None


def compute_dashboard_payload(traffic: list[dict], feedback: list[dict]) -> dict:
    today = date.today()
    visits_by_day: Counter[date] = Counter()
    photos_by_day: Counter[date] = Counter()
    feedback_by_day: Counter[date] = Counter()
    unique_sessions: set[str] = set()
    today_sessions: set[str] = set()

    for row in traffic:
        day = _parse_day(str(row.get("submitted_at_utc", "")))
        if day is None:
            continue
        event = str(row.get("event_name", ""))
        sid = str(row.get("session_id", ""))
        if event == "app_visit":
            visits_by_day[day] += 1
            unique_sessions.add(sid)
            if day == today:
                today_sessions.add(sid)
        elif event == "photo_processed":
            photos_by_day[day] += 1

    for row in feedback:
        day = _parse_day(str(row.get("submitted_at_utc", "")))
        if day:
            feedback_by_day[day] += 1

    all_days = sorted(set(visits_by_day) | set(photos_by_day) | set(feedback_by_day))
    labels = [d.strftime("%b %d") for d in all_days]
    visits = [visits_by_day[d] for d in all_days]
    photos = [photos_by_day[d] for d in all_days]
    feedback_vals = [feedback_by_day[d] for d in all_days]

    total_visits = sum(visits)
    total_photos = sum(photos)
    conversion_rate = (total_photos / total_visits * 100) if total_visits else 0

    def _insight(series: list[int], label: str) -> str:
        if not series:
            return "No data yet."
        idx = series.index(max(series))
        insight = f"Peak: {max(series)} {label} on {labels[idx]}."
        if len(series) >= 2:
            delta = series[-1] - series[-2]
            direction = "up" if delta > 0 else ("down" if delta < 0 else "flat")
            insight += f" Latest trend: {direction} ({delta:+d} vs prior day)."
        return insight

    return {
        "report_date": today.isoformat(),
        "kpis": {
            "total_visits": total_visits,
            "total_photos": total_photos,
            "conversion_rate": f"{conversion_rate:.1f}%",
            "feedback_entries": len(feedback),
            "active_users_today": len(today_sessions),
            "unique_sessions": len(unique_sessions),
            "today_visits": visits_by_day.get(today, 0),
            "today_photos": photos_by_day.get(today, 0),
        },
        "labels": labels,
        "visits": visits,
        "photos": photos,
        "feedback": feedback_vals,
        "insights": {
            "visits": _insight(visits, "visits"),
            "photos": _insight(photos, "photos"),
            "feedback": _insight(feedback_vals, "feedback entries"),
        },
    }


def render_html(payload: dict) -> str:
    data_json = json.dumps(payload)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Passport Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
* {{ margin:0; padding:0; box-sizing:border-box }}
body {{ font-family:'Inter',sans-serif; background:#f1f5f9; color:#1e293b; padding:36px; max-width:980px; margin:0 auto }}
h1 {{ font-size:24px; font-weight:700; color:#1e3a5f; margin-bottom:4px }}
.sub {{ font-size:13px; color:#94a3b8; margin-bottom:28px }}
.kpi-row {{ display:grid; grid-template-columns:repeat(4,1fr); gap:16px; margin-bottom:30px }}
.kpi {{ background:#fff; border-radius:12px; padding:18px 20px; border:1px solid #e2e8f0 }}
.kpi .label {{ font-size:11px; font-weight:600; color:#94a3b8; text-transform:uppercase; letter-spacing:.7px; margin-bottom:8px }}
.kpi .value {{ font-size:30px; font-weight:700; line-height:1 }}
.kpi .delta {{ font-size:11px; margin-top:6px; color:#64748b }}
.teal{{color:#0e7490}} .green{{color:#16a34a}} .amber{{color:#d97706}} .navy{{color:#1e3a5f}}
.card {{ background:#fff; border-radius:12px; padding:24px; margin-bottom:22px; border:1px solid #e2e8f0 }}
.card h2 {{ font-size:15px; font-weight:600; color:#1e3a5f; margin-bottom:12px }}
.chart-wrap {{ position:relative; height:260px }}
.insight {{ font-size:12px; color:#64748b; font-style:italic; margin-top:14px; padding-top:12px; border-top:1px solid #f1f5f9 }}
</style>
</head>
<body>
<h1>Passport Photo Converter — Daily Report</h1>
<p class="sub">{payload["report_date"]} · Auto-generated daily from production analytics</p>
<div class="kpi-row">
  <div class="kpi"><div class="label">Total Visits</div><div class="value teal" id="kpi-visits"></div><div class="delta" id="kpi-visits-delta"></div></div>
  <div class="kpi"><div class="label">Photos Processed</div><div class="value green" id="kpi-photos"></div><div class="delta" id="kpi-photos-delta"></div></div>
  <div class="kpi"><div class="label">Conversion Rate</div><div class="value amber" id="kpi-conversion"></div><div class="delta">photos / visits</div></div>
  <div class="kpi"><div class="label">Feedback Entries</div><div class="value navy" id="kpi-feedback"></div><div class="delta" id="kpi-feedback-delta"></div></div>
</div>
<div class="card"><h2>Daily Visits</h2><div class="chart-wrap"><canvas id="c1"></canvas></div><div class="insight" id="ins1"></div></div>
<div class="card"><h2>Daily Photos Processed</h2><div class="chart-wrap"><canvas id="c2"></canvas></div><div class="insight" id="ins2"></div></div>
<div class="card"><h2>Daily Feedback Volume</h2><div class="chart-wrap"><canvas id="c3"></canvas></div><div class="insight" id="ins3"></div></div>
<script>
const payload = {data_json};
const labels = payload.labels;
const visits = payload.visits;
const photos = payload.photos;
const feedback = payload.feedback;
const k = payload.kpis;
document.getElementById('kpi-visits').textContent = k.total_visits.toLocaleString();
document.getElementById('kpi-photos').textContent = k.total_photos.toLocaleString();
document.getElementById('kpi-conversion').textContent = k.conversion_rate;
document.getElementById('kpi-feedback').textContent = k.feedback_entries.toLocaleString();
document.getElementById('kpi-visits-delta').textContent = `+${{k.today_visits}} today`;
document.getElementById('kpi-photos-delta').textContent = `+${{k.today_photos}} today`;
document.getElementById('kpi-feedback-delta').textContent = `${{k.active_users_today}} active users today`;
document.getElementById('ins1').textContent = payload.insights.visits;
document.getElementById('ins2').textContent = payload.insights.photos;
document.getElementById('ins3').textContent = payload.insights.feedback;
const base = {{
  responsive:true, maintainAspectRatio:false,
  plugins: {{
    legend: {{display:false}},
    tooltip: {{ backgroundColor:'#1e3a5f', cornerRadius:8, padding:10, displayColors:false }}
  }},
  scales: {{
    x: {{ grid: {{display:false}}, border: {{display:false}}, ticks: {{color:'#94a3b8'}} }},
    y: {{ grid: {{color:'#f1f5f9'}}, border: {{display:false}}, ticks: {{color:'#94a3b8'}}, beginAtZero:true }}
  }}
}};
new Chart(document.getElementById('c1'), {{
  type:'line',
  data:{{labels,datasets:[{{data:visits,borderColor:'#0e7490',backgroundColor:'rgba(14,116,144,.07)',fill:true,tension:.35,borderWidth:2.5,pointRadius:4,pointBackgroundColor:'#0e7490',pointBorderColor:'#fff',pointBorderWidth:2}}]}},
  options:base
}});
new Chart(document.getElementById('c2'), {{
  type:'bar',
  data:{{labels,datasets:[{{data:photos,backgroundColor:'#16a34a',borderRadius:7,maxBarThickness:48}}]}},
  options:base
}});
new Chart(document.getElementById('c3'), {{
  type:'bar',
  data:{{labels,datasets:[{{data:feedback,backgroundColor:'#d97706',borderRadius:7,maxBarThickness:48}}]}},
  options:base
}});
</script>
</body>
</html>"""


def send_email(html_path: Path, payload: dict, recipient: str, smtp_email: str, smtp_password: str) -> None:
    msg = MIMEMultipart()
    msg["From"] = smtp_email
    msg["To"] = recipient
    msg["Subject"] = f"Passport Dashboard — {payload['report_date']}"

    k = payload["kpis"]
    body = (
        f"Daily dashboard is generated.\n\n"
        f"Report Date: {payload['report_date']}\n"
        f"Total Visits: {k['total_visits']}\n"
        f"Photos Processed: {k['total_photos']}\n"
        f"Conversion Rate: {k['conversion_rate']}\n"
        f"Feedback Entries: {k['feedback_entries']}\n\n"
        f"App: {APP_URL}\n"
    )
    msg.attach(MIMEText(body, "plain"))

    with html_path.open("rb") as f:
        part = MIMEBase("text", "html")
        part.set_payload(f.read())
    encoders.encode_base64(part)
    part.add_header("Content-Disposition", f"attachment; filename={html_path.name}")
    msg.attach(part)

    for attempt in range(SMTP_MAX_RETRIES):
        try:
            with smtplib.SMTP("smtp.gmail.com", 587, timeout=30) as server:
                server.starttls()
                server.login(smtp_email, smtp_password)
                server.send_message(msg)
            return
        except smtplib.SMTPDataError as exc:
            is_transient = exc.smtp_code in {421, 450, 451, 452}
            if not is_transient or attempt == SMTP_MAX_RETRIES - 1:
                raise
            delay = SMTP_RETRY_BASE_SECONDS * (2 ** attempt)
            print(f"Temporary SMTP issue ({exc.smtp_code}). Retrying in {delay}s...")
            time.sleep(delay)
        except (smtplib.SMTPServerDisconnected, smtplib.SMTPConnectError) as exc:
            if attempt == SMTP_MAX_RETRIES - 1:
                raise
            delay = SMTP_RETRY_BASE_SECONDS * (2 ** attempt)
            print(f"SMTP connection issue ({exc}). Retrying in {delay}s...")
            time.sleep(delay)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate dynamic HTML dashboard and email it.")
    parser.add_argument("--no-email", action="store_true", help="Skip email")
    args = parser.parse_args()

    sa, sheet_id = _service_account()
    client = gspread.service_account_from_dict(sa)
    traffic_ws = _secret("GOOGLE_TRAFFIC_WORKSHEET") or "traffic"
    feedback_ws = _secret("GOOGLE_SHEET_WORKSHEET") or "feedback"

    print(f"Fetching traffic from '{traffic_ws}'...")
    traffic = _fetch_rows(sheet_id, client, traffic_ws)
    print(f"Traffic rows: {len(traffic)}")

    print(f"Fetching feedback from '{feedback_ws}'...")
    feedback = _fetch_rows(sheet_id, client, feedback_ws)
    print(f"Feedback rows: {len(feedback)}")

    payload = compute_dashboard_payload(traffic, feedback)
    html = render_html(payload)
    OUTPUT_DIR.mkdir(exist_ok=True)
    OUTPUT_FILE.write_text(html, encoding="utf-8")
    print(f"Saved: {OUTPUT_FILE}")

    if args.no_email:
        print("Skipping email (--no-email)")
        return

    smtp_email = _secret("SMTP_EMAIL")
    smtp_password = _secret("SMTP_PASSWORD")
    recipient = _secret("REPORT_RECIPIENT") or "supportpassportphotoconversion@gmail.com"
    if not smtp_email or not smtp_password:
        print("SMTP_EMAIL/SMTP_PASSWORD missing; skipping email.")
        return
    send_email(OUTPUT_FILE, payload, recipient, smtp_email, smtp_password)
    print(f"Emailed report to {recipient}")


if __name__ == "__main__":
    main()
