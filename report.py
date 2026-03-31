"""
Daily dashboard report for Passport Photo Converter.

Reads traffic + feedback from Google Sheets, builds an Excel workbook with
KPI summary, daily metrics, charts, and raw feedback, then emails the file.

Usage:
    python report.py                 # generate + email
    python report.py --no-email      # generate only (no email)

Secrets are read from .streamlit/secrets.toml (same as the app) with
env-var overrides for CI/cron:
    GOOGLE_SERVICE_ACCOUNT_JSON, GOOGLE_SHEET_ID,
    SMTP_EMAIL, SMTP_PASSWORD, REPORT_RECIPIENT
"""
from __future__ import annotations

import argparse
import json
import os
import smtplib
import sys
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

import gspread
from openpyxl import Workbook
from openpyxl.chart import BarChart, LineChart, Reference
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

SECRETS_PATH = Path(__file__).parent / ".streamlit" / "secrets.toml"
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_FILE = OUTPUT_DIR / "passport_photo_dashboard.xlsx"

HEADER_FONT = Font(bold=True, size=12, color="FFFFFF")
HEADER_FILL = PatternFill(start_color="2563EB", end_color="2563EB", fill_type="solid")
KPI_LABEL_FONT = Font(bold=True, size=11)
KPI_VALUE_FONT = Font(bold=True, size=14, color="16A34A")


# =========================================================================
# Secrets
# =========================================================================

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
    val = os.environ.get(key)
    if val:
        return val
    return _load_toml_secrets().get(key)


def _service_account() -> tuple[dict, str]:
    raw = _secret("GOOGLE_SERVICE_ACCOUNT_JSON")
    sheet_id = _secret("GOOGLE_SHEET_ID")
    if not raw or not sheet_id:
        sys.exit("Missing GOOGLE_SERVICE_ACCOUNT_JSON or GOOGLE_SHEET_ID")
    if isinstance(raw, str):
        raw = raw.strip().strip("'").strip('"').strip()
        sa = json.loads(raw)
    else:
        sa = dict(raw)
    return sa, sheet_id


# =========================================================================
# Google Sheets data fetch
# =========================================================================

def _fetch_rows(sheet_id: str, client: gspread.Client, name: str) -> list[dict]:
    try:
        ws = client.open_by_key(sheet_id).worksheet(name)
        return ws.get_all_records()
    except gspread.WorksheetNotFound:
        return []


def _parse_date(iso_str: str) -> date | None:
    try:
        return datetime.fromisoformat(iso_str.replace("Z", "+00:00")).date()
    except Exception:
        return None


# =========================================================================
# KPI computation
# =========================================================================

def compute_metrics(
    traffic: list[dict], feedback: list[dict],
) -> tuple[dict, list[tuple], list[tuple]]:
    today = date.today()

    visits_by_day: Counter[date] = Counter()
    photos_by_day: Counter[date] = Counter()
    feedback_by_day: Counter[date] = Counter()
    unique_sessions: set[str] = set()
    today_sessions: set[str] = set()

    for row in traffic:
        d = _parse_date(str(row.get("submitted_at_utc", "")))
        event = row.get("event_name", "")
        sid = str(row.get("session_id", ""))
        if d is None:
            continue
        if event == "app_visit":
            visits_by_day[d] += 1
            unique_sessions.add(sid)
            if d == today:
                today_sessions.add(sid)
        elif event == "photo_processed":
            photos_by_day[d] += 1

    for row in feedback:
        d = _parse_date(str(row.get("submitted_at_utc", "")))
        if d:
            feedback_by_day[d] += 1

    total_visits = sum(visits_by_day.values())
    total_photos = sum(photos_by_day.values())
    total_feedback = len(feedback)
    conversion_rate = (total_photos / total_visits * 100) if total_visits else 0

    kpis = {
        "Report Date": today.isoformat(),
        "Total Visits (all time)": total_visits,
        "Total Photos Processed": total_photos,
        "Conversion Rate": f"{conversion_rate:.1f}%",
        "Total Feedback Entries": total_feedback,
        "Unique Sessions (all time)": len(unique_sessions),
        "Today's Active Users": len(today_sessions),
        "Today's Visits": visits_by_day.get(today, 0),
        "Today's Photos": photos_by_day.get(today, 0),
    }

    all_days = sorted(set(visits_by_day) | set(photos_by_day) | set(feedback_by_day))
    daily = [
        (d.isoformat(), visits_by_day[d], photos_by_day[d], feedback_by_day[d])
        for d in all_days
    ]

    fb_rows = []
    for row in feedback:
        d = _parse_date(str(row.get("submitted_at_utc", "")))
        fb_rows.append((
            d.isoformat() if d else str(row.get("submitted_at_utc", "")),
            str(row.get("feedback", "")),
        ))

    return kpis, daily, fb_rows


# =========================================================================
# Excel workbook
# =========================================================================

def _style_header(ws, cols: int) -> None:
    for c in range(1, cols + 1):
        cell = ws.cell(row=1, column=c)
        cell.font = HEADER_FONT
        cell.fill = HEADER_FILL
        cell.alignment = Alignment(horizontal="center")
        ws.column_dimensions[get_column_letter(c)].width = 22


def build_workbook(
    kpis: dict, daily: list[tuple], feedback_rows: list[tuple],
) -> Workbook:
    wb = Workbook()

    # -- Sheet 1: KPI Summary --
    ws_kpi = wb.active
    ws_kpi.title = "KPI Summary"
    ws_kpi.sheet_properties.tabColor = "2563EB"
    ws_kpi.column_dimensions["A"].width = 30
    ws_kpi.column_dimensions["B"].width = 25

    ws_kpi.cell(row=1, column=1, value="Passport Photo Converter — Daily Report").font = Font(bold=True, size=16)
    ws_kpi.merge_cells("A1:B1")
    row = 3
    for label, value in kpis.items():
        ws_kpi.cell(row=row, column=1, value=label).font = KPI_LABEL_FONT
        ws_kpi.cell(row=row, column=2, value=value).font = KPI_VALUE_FONT
        row += 1

    # -- Sheet 2: Daily Metrics --
    ws_daily = wb.create_sheet("Daily Metrics")
    ws_daily.sheet_properties.tabColor = "16A34A"
    headers = ["Date", "Visits", "Photos Processed", "Feedback"]
    ws_daily.append(headers)
    _style_header(ws_daily, len(headers))
    for r in daily:
        ws_daily.append(list(r))

    # -- Sheet 3: Charts --
    ws_charts = wb.create_sheet("Charts")
    ws_charts.sheet_properties.tabColor = "F59E0B"

    if len(daily) >= 2:
        data_rows = len(daily) + 1

        visits_chart = LineChart()
        visits_chart.title = "Daily Visits"
        visits_chart.y_axis.title = "Visits"
        visits_chart.x_axis.title = "Date"
        visits_chart.width = 28
        visits_chart.height = 14
        cats = Reference(ws_daily, min_col=1, min_row=2, max_row=data_rows)
        vals = Reference(ws_daily, min_col=2, min_row=1, max_row=data_rows)
        visits_chart.add_data(vals, titles_from_data=True)
        visits_chart.set_categories(cats)
        visits_chart.style = 10
        ws_charts.add_chart(visits_chart, "A1")

        photos_chart = BarChart()
        photos_chart.title = "Daily Photos Processed"
        photos_chart.y_axis.title = "Photos"
        photos_chart.x_axis.title = "Date"
        photos_chart.width = 28
        photos_chart.height = 14
        vals2 = Reference(ws_daily, min_col=3, min_row=1, max_row=data_rows)
        photos_chart.add_data(vals2, titles_from_data=True)
        photos_chart.set_categories(cats)
        photos_chart.style = 10
        ws_charts.add_chart(photos_chart, "A18")

        fb_chart = BarChart()
        fb_chart.title = "Daily Feedback Volume"
        fb_chart.y_axis.title = "Feedback"
        fb_chart.x_axis.title = "Date"
        fb_chart.width = 28
        fb_chart.height = 14
        vals3 = Reference(ws_daily, min_col=4, min_row=1, max_row=data_rows)
        fb_chart.add_data(vals3, titles_from_data=True)
        fb_chart.set_categories(cats)
        fb_chart.style = 10
        ws_charts.add_chart(fb_chart, "A35")
    else:
        ws_charts.cell(row=1, column=1, value="Not enough data for charts (need at least 2 days).")

    # -- Sheet 4: Raw Feedback --
    ws_fb = wb.create_sheet("Raw Feedback")
    ws_fb.sheet_properties.tabColor = "EF4444"
    fb_headers = ["Date", "Feedback"]
    ws_fb.append(fb_headers)
    _style_header(ws_fb, len(fb_headers))
    ws_fb.column_dimensions["B"].width = 80
    for r in feedback_rows:
        ws_fb.append(list(r))

    return wb


# =========================================================================
# Email
# =========================================================================

def send_email(filepath: Path, recipient: str, smtp_email: str, smtp_password: str) -> None:
    msg = MIMEMultipart()
    msg["From"] = smtp_email
    msg["To"] = recipient
    msg["Subject"] = f"Passport Photo Dashboard — {date.today().isoformat()}"

    body = (
        f"Hi,\n\n"
        f"Attached is the daily dashboard report for {date.today().isoformat()}.\n\n"
        f"— Passport Photo Converter\n"
    )
    msg.attach(MIMEText(body, "plain"))

    with open(filepath, "rb") as f:
        part = MIMEBase("application", "vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        part.set_payload(f.read())
    encoders.encode_base64(part)
    part.add_header("Content-Disposition", f"attachment; filename={filepath.name}")
    msg.attach(part)

    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(smtp_email, smtp_password)
        server.send_message(msg)

    print(f"Email sent to {recipient}")


# =========================================================================
# Main
# =========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate & email the daily dashboard report.")
    parser.add_argument("--no-email", action="store_true", help="Skip sending the email")
    args = parser.parse_args()

    print("Connecting to Google Sheets...")
    sa, sheet_id = _service_account()
    client = gspread.service_account_from_dict(sa)

    traffic_ws = _secret("GOOGLE_TRAFFIC_WORKSHEET") or "traffic"
    feedback_ws = _secret("GOOGLE_SHEET_WORKSHEET") or "feedback"

    print(f"Fetching traffic from '{traffic_ws}'...")
    traffic = _fetch_rows(sheet_id, client, traffic_ws)
    print(f"  → {len(traffic)} rows")

    print(f"Fetching feedback from '{feedback_ws}'...")
    feedback = _fetch_rows(sheet_id, client, feedback_ws)
    print(f"  → {len(feedback)} rows")

    print("Computing KPIs...")
    kpis, daily, fb_rows = compute_metrics(traffic, feedback)
    for k, v in kpis.items():
        print(f"  {k}: {v}")

    print("Building workbook...")
    wb = build_workbook(kpis, daily, fb_rows)
    OUTPUT_DIR.mkdir(exist_ok=True)
    wb.save(str(OUTPUT_FILE))
    print(f"Saved → {OUTPUT_FILE}")

    if args.no_email:
        print("Skipping email (--no-email).")
        return

    smtp_email = _secret("SMTP_EMAIL")
    smtp_password = _secret("SMTP_PASSWORD")
    recipient = _secret("REPORT_RECIPIENT") or "supportpassportphotoconversion@gmail.com"

    if not smtp_email or not smtp_password:
        print("SMTP_EMAIL / SMTP_PASSWORD not set — skipping email.")
        print("Add them to .streamlit/secrets.toml or set as env vars.")
        return

    print(f"Emailing to {recipient}...")
    send_email(OUTPUT_FILE, recipient, smtp_email, smtp_password)
    print("Done.")


if __name__ == "__main__":
    main()
