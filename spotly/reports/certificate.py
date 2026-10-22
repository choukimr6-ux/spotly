"""Single-page "Broadcast Airing Certificate" (proof of performance) as an in-memory PDF.

    from spotly.reports.certificate import certificate_for
    pdf_bytes = certificate_for(db_path, occurrence_id)

The integrity line is recomputed at issue time (core/proof.verify_row), so a
certificate never shows PASS for a record or evidence file edited after detection.
"""
import io
from datetime import datetime, timezone
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, Paragraph, SimpleDocTemplate, Spacer,
                                Table, TableStyle)

from spotly.config import LIVE_MATCH_THRESHOLD, MATCH_THRESHOLD
from spotly.core.proof import ALGERIA_TZ, HASH_VERSION, verify_row
from spotly.database.schema import init_db

INK = colors.HexColor("#0b0b0b")
MUTED = colors.HexColor("#52514e")
RULE = colors.HexColor("#d9d8d4")
ACCENT = colors.HexColor("#2a78d6")
REVIEW = {1: "Verified by human reviewer", 0: "Not yet reviewed by a human", -1: "Rejected by human reviewer"}

ROW_SQL = """
SELECT o.*, r.brand_name, r.campaign_name, r.duration_sec
FROM ad_occurrences o JOIN reference_ads r USING (ad_id)
WHERE o.occurrence_id = ?
"""


def _styles():
    ss = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("t", parent=ss["Title"], fontSize=18, leading=22, textColor=INK, spaceAfter=2),
        "sub": ParagraphStyle("s", parent=ss["Normal"], fontSize=9.5, textColor=MUTED, alignment=TA_CENTER),
        "h": ParagraphStyle("h", parent=ss["Heading3"], fontSize=10.5, textColor=ACCENT, spaceBefore=8, spaceAfter=4),
        "cell": ParagraphStyle("c", parent=ss["Normal"], fontSize=9.5, leading=12, textColor=INK),
        "label": ParagraphStyle("l", parent=ss["Normal"], fontSize=8.5, leading=11, textColor=MUTED),
        "mono": ParagraphStyle("m", parent=ss["Normal"], fontName="Courier", fontSize=8.5, leading=11, textColor=INK),
        "foot": ParagraphStyle("f", parent=ss["Normal"], fontSize=7.5, leading=9.5, textColor=MUTED),
    }


def _kv_table(rows, st):
    data = [[Paragraph(k, st["label"]), v if not isinstance(v, str) else Paragraph(v, st["cell"])] for k, v in rows]
    t = Table(data, colWidths=[4.6 * cm, 12.4 * cm])
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LINEBELOW", (0, 0), (-1, -2), 0.4, RULE),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


def _airing_window(row):
    if row["algeria_timestamp"]:
        start = datetime.fromtimestamp(row["airing_start"], ALGERIA_TZ)
        end = datetime.fromtimestamp(row["airing_end"], ALGERIA_TZ)
        utc = datetime.fromtimestamp(row["airing_start"], timezone.utc)
        return (f"<b>{start:%Y-%m-%d %H:%M:%S}.{start.microsecond // 1000:03d} → "
                f"{end:%H:%M:%S}.{end.microsecond // 1000:03d}</b> Algeria time (CET, UTC+1)<br/>"
                f"<font color='#52514e' size='8.5'>{utc:%Y-%m-%d %H:%M:%S} UTC · stream capture time; "
                f"HLS delivery typically lags on-air time by 10–30 s</font>")
    return (f"<b>{row['airing_start']:.3f} s → {row['airing_end']:.3f} s</b> from start of recorded source<br/>"
            f"<font color='#52514e' size='8.5'>File scan: no wall-clock broadcast time available</font>")


def _evidence(row, max_w=17 * cm, max_h=7.5 * cm):
    path = row["evidence_frame_path"]
    if not path or not Path(path).exists():
        return None
    img = Image(path)
    scale = min(max_w / img.imageWidth, max_h / img.imageHeight)
    img.drawWidth, img.drawHeight = img.imageWidth * scale, img.imageHeight * scale
    return img


def build_certificate(row) -> bytes:
    """Render one ad_occurrences row (joined with reference_ads) to PDF bytes."""
    st = _styles()
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                            topMargin=1.6 * cm, bottomMargin=1.5 * cm,
                            title=f"Spotly Airing Certificate #{row['occurrence_id']}", author="Spotly")
    channel = row["source_name"].removeprefix("live:")
    threshold = LIVE_MATCH_THRESHOLD if row["source_name"].startswith("live:") else MATCH_THRESHOLD
    issued = datetime.now(ALGERIA_TZ)
    integrity = verify_row(row)
    if integrity is None:
        integrity_txt = "<font color='#52514e'>No verification hash was recorded for this occurrence.</font>"
    elif integrity:
        integrity_txt = ("<font color='#0ca30c'><b>PASS</b></font> — record and "
                         "evidence file match the hash recorded at detection time.")
    else:
        integrity_txt = ("<font color='#d03b3b'><b>FAIL</b></font> — the record or "
                         "its evidence file has changed since detection. Do not rely on this certificate.")
    vhash = row["verification_hash"] or "—"
    hash_txt = vhash if vhash == "—" else f"{vhash[:32]}<br/>{vhash[32:]}"

    story = [
        Paragraph("Spotly Certified Ad Occurrence Certificate", st["title"]),
        Paragraph(f"Broadcast Airing Certificate · Proof of Performance · No. SPOTLY-{row['occurrence_id']:06d}",
                  st["sub"]),
        Spacer(1, 10),
        Paragraph("Airing", st["h"]),
        _kv_table([
            ("Brand", f"<b>{row['brand_name']}</b>"),
            ("Campaign", row["campaign_name"]),
            ("Channel", f"<b>{channel}</b>"),
            ("Airing window", _airing_window(row)),
            ("Detected duration", f"{row['duration_detected']:.3f} s "
                                  f"<font color='#52514e'>(reference creative {row['duration_sec']:.2f} s)</font>"),
        ], st),
        Paragraph("Evidence", st["h"]),
    ]
    img = _evidence(row)
    story.append(img if img else Paragraph("Evidence file not available.", st["label"]))
    story.append(Paragraph(f"<font color='#52514e' size='8'>{Path(row['evidence_frame_path'] or '').name}"
                           f"</font>", st["label"]))
    story += [
        Paragraph("Match metrics & audit", st["h"]),
        _kv_table([
            ("Match confidence", f"<b>{row['confidence_score'] * 100:.2f}%</b> "
                                 f"<font color='#52514e'>(1 − bit error rate of the audio fingerprint; "
                                 f"detection threshold {threshold * 100:.0f}%, unrelated audio ≈ 50%)</font>"),
            ("Human review", REVIEW.get(row["human_verified"], str(row["human_verified"]))),
            ("Detected at", f"{row['detected_at']} (UTC)"),
            ("Verification hash<br/>(SHA-256)", Paragraph(hash_txt, st["mono"])),
            ("Integrity check", integrity_txt),
        ], st),
        Spacer(1, 10),
        Paragraph(
            f"Issued {issued:%Y-%m-%d %H:%M} Algeria time by Spotly. This certificate records a detected "
            f"<b>airing</b> of the creative, identified by audio fingerprint matching; it is not a measurement of "
            f"audience, reach or impressions. The SHA-256 hash ({HASH_VERSION}) covers the ad id, channel, airing "
            f"time, airing end, duration, confidence and the evidence image bytes; it makes later edits detectable "
            f"but is not a digital signature.", st["foot"]),
    ]
    doc.build(story)
    return buf.getvalue()


def certificate_for(db_path, occurrence_id: int) -> bytes:
    conn = init_db(db_path)
    try:
        row = conn.execute(ROW_SQL, (int(occurrence_id),)).fetchone()
    finally:
        conn.close()
    if row is None:
        raise KeyError(f"occurrence {occurrence_id} not found")
    return build_certificate(row)
