"""
Session Report Generator — PDF and HTML export.

FIXES vs. original:
- Removed false DPDP compliance claim
- Added proper system limitations section
- Added privacy/ethical notice (accurate, not a legal claim)
- Added state distribution summary
- HTML report now includes per-student table and alerts
- All errors caught gracefully
"""
import os
import logging
from datetime import datetime

import pandas as pd

logger = logging.getLogger(__name__)

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable,
    )
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False

# ── Privacy / limitations boilerplate ────────────────────────────────────────

LIMITATIONS_TEXT = (
    "SYSTEM LIMITATIONS: This system estimates visual proxies for attention "
    "(head orientation, approximate gaze direction, eye-openness).  It does "
    "NOT measure cognitive attention, learning, or comprehension.  The "
    "XGBoost model was trained on synthetically generated data; real-world "
    "accuracy has not been independently validated.  Results should be used "
    "only as a rough teaching aid, not as evidence of individual student "
    "performance or conduct."
)

PRIVACY_TEXT = (
    "PRIVACY SAFEGUARDS: Video is processed entirely on-device.  No raw "
    "video frames are stored.  Only numeric metrics and anonymous student IDs "
    "are logged.  Facial recognition is not implemented.  Institutional "
    "ethical review and informed consent from students and parents/guardians "
    "are REQUIRED before any real-world deployment.  Compliance with "
    "applicable data-protection laws (e.g., DPDP Act, GDPR) must be verified "
    "by qualified legal counsel — this software alone does not constitute "
    "legal compliance."
)

# ── Public entry point ────────────────────────────────────────────────────────

def generate_pdf_report(
    session_id: str,
    session_df: pd.DataFrame,
    alerts_df: pd.DataFrame,
    output_path: str | None = None,
) -> str:
    """
    Generate a PDF session report.  Falls back to HTML if reportlab is missing.
    Returns the path of the generated file.
    """
    if output_path is None:
        base_dir    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_path = os.path.join(base_dir, "assets", f"Session_Report_{session_id}.pdf")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if not HAS_REPORTLAB:
        html_path = output_path.replace(".pdf", ".html")
        return generate_html_report(session_id, session_df, alerts_df, html_path)

    try:
        return _build_pdf(session_id, session_df, alerts_df, output_path)
    except Exception as exc:
        logger.error("PDF generation failed: %s — falling back to HTML.", exc)
        html_path = output_path.replace(".pdf", ".html")
        return generate_html_report(session_id, session_df, alerts_df, html_path)


# ── PDF builder ───────────────────────────────────────────────────────────────

def _build_pdf(
    session_id: str,
    session_df: pd.DataFrame,
    alerts_df: pd.DataFrame,
    output_path: str,
) -> str:
    doc    = SimpleDocTemplate(output_path, pagesize=letter,
                                rightMargin=40, leftMargin=40,
                                topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "DocTitle", parent=styles["Heading1"],
        fontName="Helvetica-Bold", fontSize=20,
        textColor=colors.HexColor("#0F172A"), spaceAfter=6,
    )
    sub_style = ParagraphStyle(
        "Sub", parent=styles["Normal"],
        fontName="Helvetica", fontSize=9,
        textColor=colors.HexColor("#64748B"), spaceAfter=12,
    )
    h2 = ParagraphStyle(
        "H2", parent=styles["Heading2"],
        fontName="Helvetica-Bold", fontSize=13,
        textColor=colors.HexColor("#1E293B"), spaceBefore=14, spaceAfter=6,
    )
    body = ParagraphStyle(
        "Body", parent=styles["Normal"],
        fontName="Helvetica", fontSize=9,
        textColor=colors.HexColor("#334155"), leading=13,
    )
    warn = ParagraphStyle(
        "Warn", parent=styles["Normal"],
        fontName="Helvetica-Oblique", fontSize=8,
        textColor=colors.HexColor("#7F1D1D"), leading=12,
    )

    story = []

    # ── Header ────────────────────────────────────────────────────────────
    story.append(Paragraph("Classroom Attention Analytics Report", title_style))
    story.append(Paragraph(
        f"Session ID: <b>{session_id}</b> | "
        f"Generated: <b>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</b>",
        sub_style,
    ))
    story.append(HRFlowable(width="100%", thickness=1.5,
                             color=colors.HexColor("#38BDF8"), spaceAfter=12))

    # ── Summary metrics ───────────────────────────────────────────────────
    total   = len(session_df)
    avg_att = session_df["attention_score"].mean() if not session_df.empty else 0.0
    n_stu   = session_df["student_id"].nunique()  if not session_df.empty else 0

    def _pct(col_val):
        if total == 0:
            return "0  (0.0%)"
        cnt = int(col_val)
        return f"{cnt}  ({cnt / total * 100:.1f}%)"

    att_cnt  = (session_df["status"] == "Attentive").sum()  if not session_df.empty else 0
    dis_cnt  = (session_df["status"] == "Distracted").sum() if not session_df.empty else 0
    dro_cnt  = (session_df["status"] == "Drowsy").sum()     if not session_df.empty else 0
    unk_cnt  = total - att_cnt - dis_cnt - dro_cnt

    summary_rows = [
        ["Metric", "Value"],
        ["Average Class Attention Score",   f"{avg_att:.1f} / 100"],
        ["Unique Students (anonymous IDs)", str(n_stu)],
        ["Attentive observations",          _pct(att_cnt)],
        ["Distracted observations",         _pct(dis_cnt)],
        ["Drowsy observations",             _pct(dro_cnt)],
        ["Unknown / insufficient data",     _pct(unk_cnt)],
        ["Total Alerts",                    str(len(alerts_df))],
    ]

    t_sum = _make_table(summary_rows, [260, 240],
                        header_bg="#0F172A", row_bg="#F8FAFC")
    story.append(Paragraph("Executive Summary", h2))
    story.append(t_sum)
    story.append(Spacer(1, 12))

    # ── Per-student breakdown ─────────────────────────────────────────────
    story.append(Paragraph("Anonymous Per-Student Summary", h2))
    if not session_df.empty:
        grp = session_df.groupby("student_id").agg(
            avg_score=("attention_score", "mean"),
            dominant=("status", lambda x: x.mode()[0] if not x.empty else "N/A"),
        ).reset_index()

        stu_rows = [["Student ID", "Avg Attention", "Dominant State", "Indicator"]]
        for _, r in grp.iterrows():
            score = r["avg_score"]
            ind   = ("Good" if score >= 70
                     else ("Monitor gaze & posture" if r["dominant"] == "Distracted"
                           else "Consider breaks / fatigue"))
            stu_rows.append([
                f"Student #{int(r['student_id'])}",
                f"{score:.1f}",
                r["dominant"],
                ind,
            ])
        story.append(_make_table(stu_rows, [80, 110, 120, 190],
                                 header_bg="#1E293B", row_bg="#F1F5F9"))
    else:
        story.append(Paragraph("No data recorded for this session.", body))

    story.append(Spacer(1, 12))

    # ── Alerts ────────────────────────────────────────────────────────────
    story.append(Paragraph("Sustained Low-Attention Alerts", h2))
    if not alerts_df.empty:
        # Clean & deduplicate repetitive alerts
        clean_alerts = alerts_df.drop_duplicates(subset=["student_id", "alert_type", "message"])
        al_rows = [["Time", "Student", "Type", "Message"]]
        for _, r in clean_alerts.head(15).iterrows():
            ts = str(r["timestamp"]).split("T")[-1][:8]
            al_rows.append([ts, f"Student #{int(r['student_id'])}", r["alert_type"], r["message"]])
        story.append(_make_table(al_rows, [70, 70, 110, 250],
                                 header_bg="#991B1B", row_bg="#FFF5F5"))
    else:
        story.append(Paragraph("No alerts triggered during this session.", body))

    story.append(Spacer(1, 16))
    story.append(HRFlowable(width="100%", thickness=0.8,
                             color=colors.HexColor("#94A3B8"), spaceAfter=8))

    # ── Limitations & privacy ─────────────────────────────────────────────
    story.append(Paragraph("System Limitations", h2))
    story.append(Paragraph(LIMITATIONS_TEXT, warn))
    story.append(Spacer(1, 8))
    story.append(Paragraph("Privacy Safeguards & Legal Notice", h2))
    story.append(Paragraph(PRIVACY_TEXT, warn))

    doc.build(story)
    return output_path


def _make_table(data, col_widths, header_bg="#0F172A", row_bg="#F8FAFC"):
    t = Table(data, colWidths=col_widths)
    t.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1,  0), colors.HexColor(header_bg)),
        ("TEXTCOLOR",   (0, 0), (-1,  0), colors.white),
        ("FONTNAME",    (0, 0), (-1,  0), "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 7),
        ("BACKGROUND",  (0, 1), (-1, -1), colors.HexColor(row_bg)),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor(row_bg)]),
        ("GRID",        (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")),
        ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return t


# ── HTML fallback ─────────────────────────────────────────────────────────────

def generate_html_report(
    session_id: str,
    session_df: pd.DataFrame,
    alerts_df: pd.DataFrame,
    output_path: str,
) -> str:
    avg_att  = session_df["attention_score"].mean() if not session_df.empty else 0.0
    n_stu    = session_df["student_id"].nunique()  if not session_df.empty else 0
    n_alerts = len(alerts_df)
    total    = len(session_df)

    # Per-student table
    stu_html = ""
    if not session_df.empty:
        grp = session_df.groupby("student_id").agg(
            avg_score=("attention_score", "mean"),
            dominant=("status", lambda x: x.mode()[0] if not x.empty else "N/A"),
        ).reset_index()
        rows = "".join(
            f"<tr><td>Student #{int(r['student_id'])}</td>"
            f"<td>{r['avg_score']:.1f}</td><td>{r['dominant']}</td></tr>"
            for _, r in grp.iterrows()
        )
        stu_html = f"""
        <h2>Per-Student Summary</h2>
        <table><thead><tr><th>Student</th><th>Avg Score</th><th>Dominant State</th></tr></thead>
        <tbody>{rows}</tbody></table>"""

    # Alerts table
    al_html = ""
    if not alerts_df.empty:
        rows = "".join(
            f"<tr><td>{str(r['timestamp']).split('T')[-1][:8]}</td>"
            f"<td>Student #{int(r['student_id'])}</td>"
            f"<td>{r['alert_type']}</td><td>{r['message']}</td></tr>"
            for _, r in alerts_df.head(15).iterrows()
        )
        al_html = f"""
        <h2>Alerts</h2>
        <table><thead><tr><th>Time</th><th>Student</th><th>Type</th><th>Message</th></tr></thead>
        <tbody>{rows}</tbody></table>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8">
<title>Attention Report — {session_id}</title>
<style>
  body{{font-family:'Segoe UI',sans-serif;background:#0f172a;color:#f8fafc;padding:2rem;}}
  .box{{max-width:960px;margin:auto;background:#1e293b;padding:2rem;border-radius:1rem;}}
  h1{{color:#38bdf8;border-bottom:2px solid #38bdf8;padding-bottom:.5rem;}}
  h2{{color:#94a3b8;margin-top:1.5rem;}}
  .metric{{font-size:2.5rem;font-weight:bold;color:#34d399;}}
  table{{width:100%;border-collapse:collapse;margin-top:1rem;}}
  th,td{{padding:.6rem 1rem;border:1px solid #334155;text-align:left;font-size:.85rem;}}
  th{{background:#0f172a;color:#38bdf8;}}
  tr:nth-child(even){{background:#1e293b;}}
  .warn{{background:#450a0a;color:#fca5a5;padding:1rem;border-radius:.5rem;font-size:.8rem;margin-top:1rem;}}
</style></head>
<body><div class="box">
<h1>Classroom Attention Analytics Report</h1>
<p>Session: <b>{session_id}</b> &nbsp;|&nbsp; Generated: <b>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</b></p>
<p>Average Class Attention: <span class="metric">{avg_att:.1f}%</span></p>
<p>Students detected: <b>{n_stu}</b> &nbsp;|&nbsp; Log entries: <b>{total}</b> &nbsp;|&nbsp; Alerts: <b>{n_alerts}</b></p>
{stu_html}
{al_html}
<div class="warn"><b>System Limitations:</b> {LIMITATIONS_TEXT}<br><br>
<b>Privacy Notice:</b> {PRIVACY_TEXT}</div>
</div></body></html>"""

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
    except Exception as exc:
        logger.error("HTML report write failed: %s", exc)

    return output_path
