"""Phase 17: tamper-evident Proof-of-Performance certificates (single-page A4 PDF, rendered in memory).

    pdf = generate_airing_certificate_pdf(airing_id)            # bytes
    report = verify_certificate_pdf(pdf, db_path)               # {"status": "PASS" | "FAIL", "checks": {...}}

Seal. A deterministic SHA-256 is computed over exactly
    [airing_id, channel, timestamp_cet, exact_duration, brand, matched_confidence, calculated_cost_dzd]
(canonical JSON with sorted keys, prefixed by the version tag). The payload, the hash and an HMAC-SHA256 are embedded
in the PDF's /Keywords (base64, so no escaping issues) and the hash is printed on the page and in the QR code.
  * hash   - any change to the embedded payload or hash is caught by anyone, with no key;
  * HMAC   - keyed with this installation's secret (env KIHC_POP_KEY or kihc/data/state/pop_signing.key), so a forger
             who edits the payload AND recomputes the hash still fails on the issuing system;
  * record - the payload is re-derived from the database, so a certificate whose underlying airing was edited later
             no longer matches. (Cost is reported as drift, not failure: the rate card may legitimately change.)
It is a tamper-evidence seal, not a qualified digital signature, and certifies an airing, not audience.
"""
import base64
import hashlib
import hmac
import io
import itertools
import json
import logging
import math
import os
import re
import secrets
import sqlite3
import time
import zipfile
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from xml.sax.saxutils import escape

from reportlab import rl_config
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Flowable, Image, KeepInFrame, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle)

from kihc.analytics import compliance_audit as ca
from kihc.analytics import contracts as ctr
from kihc.analytics.rate_engine import RateCardEngine, compute_airing_financials, money
from kihc.config import DATA_DIR, DEFAULT_DB_PATH
from kihc.core.analytics import channel_of
from kihc.stream.time_synchronizer import LOCKED, broadcast_epoch, lock_status
from kihc.core.pod_analyzer import cluster_occurrences_into_pods
from kihc.core.proof import ALGERIA_TZ, record_timestamp, verify_row
from kihc.database.schema import get_db

CERT_VERSION = "KIHC-POP-CERT-v2"
KEY_PATH = DATA_DIR / "state" / "pop_signing.key"
PAYLOAD_FIELDS = ("airing_id", "channel", "timestamp_cet", "exact_duration", "brand", "matched_confidence",
                  "calculated_cost_dzd")
log = logging.getLogger("kihc.certificate")
MAX_BATCH = 500

EMERALD, EMERALD_D, EMERALD_L = colors.HexColor("#059669"), colors.HexColor("#047857"), colors.HexColor("#E6F7F0")
SLATE, SLATE_D, SLATE_M, RULE = (colors.HexColor("#0F172A"), colors.HexColor("#1E293B"), colors.HexColor("#64748B"),
                                 colors.HexColor("#E2E8F0"))
RED = colors.HexColor("#B91C1C")
THUMB_MAX_PX = (800, 450)
rl_config.useA85 = 0                  # embed JPEGs as raw DCT: ASCII85 in pure Python cost ~1 s per image

ROW_SQL = """
SELECT o.*, r.brand_name, r.campaign_name, r.duration_sec, r.industry, r.ad_type
FROM ad_occurrences o JOIN reference_ads r USING (ad_id) WHERE o.occurrence_id = ?
"""


# ------------------------------------------------------------------------------------------ seal
def signing_key() -> bytes:
    env = os.environ.get("KIHC_POP_KEY")
    if env:
        return env.encode()
    try:
        return KEY_PATH.read_text().strip().encode()
    except OSError:
        pass
    key = secrets.token_hex(32)
    try:
        KEY_PATH.parent.mkdir(parents=True, exist_ok=True)
        KEY_PATH.write_text(key)
    except OSError:
        pass                                                   # read-only install: seal still carries the hash
    return key.encode()


def canonical_payload(p: dict) -> bytes:
    rec = {k: str(p[k]) for k in PAYLOAD_FIELDS}
    return (CERT_VERSION + "\n" + json.dumps(rec, sort_keys=True, separators=(",", ":"))).encode("utf-8")


def seal(p: dict, key: bytes | None = None) -> tuple[str, str]:
    body = canonical_payload(p)
    digest = hashlib.sha256(body).hexdigest()
    return digest, hmac.new(key or signing_key(), body + digest.encode(), hashlib.sha256).hexdigest()


def payload_for(row, cost) -> dict:
    return dict(airing_id=int(row["occurrence_id"]), channel=channel_of(row["source_name"]),
                timestamp_cet=record_timestamp(row), exact_duration=f"{float(row['duration_detected']):.3f}",
                brand=row["brand_name"], matched_confidence=f"{float(row['confidence_score']):.4f}",
                calculated_cost_dzd=f"{money(cost):.2f}")


# ------------------------------------------------------------------------------------------ data
def _rate_card(db_path, rate_card=None):
    return rate_card or RateCardEngine.load(db_path=db_path)


def _pod_for(db_path, row, pods_cache: dict | None):
    live = row["source_name"].startswith("live:")
    bucket = int(row["airing_start"] // 21600)                       # 6 h slices, so a batch shares one pod load
    key = (row["source_name"], bucket) if live else (row["source_name"], None)
    if pods_cache is not None and key in pods_cache:
        pods = pods_cache[key]
    else:
        window = (bucket * 21600 - 7200, bucket * 21600 + 21600 + 3600) if live else None
        pods = cluster_occurrences_into_pods(channel=row["source_name"] if not live else channel_of(row["source_name"]),
                                             date_or_window=window, db_path=db_path)
        if pods_cache is not None:
            pods_cache[key] = pods
    for p in pods:
        for a in p.ads:
            if a["occurrence_id"] == row["occurrence_id"]:
                return p, a
    return None, None


def _separation(conn, row, window_s, rate_card, contract, net):
    """('breach'|'compliant'|'not contracted'|'not assessable...', nearest competitor gap s, credit Decimal)."""
    if not window_s:
        return "not contracted", None, Decimal(0)
    sector = ca._sector(row["industry"])
    if not sector:
        return "not assessable: sector unknown", None, Decimal(0)
    near = conn.execute(
        "SELECT o.airing_start, o.airing_end, r.brand_name FROM ad_occurrences o JOIN reference_ads r USING (ad_id) "
        "WHERE o.source_name = ? AND o.human_verified != -1 AND o.occurrence_id != ? AND lower(r.industry) = ? "
        "AND lower(r.brand_name) != lower(?) AND o.airing_start < ? AND o.airing_end > ?",
        (row["source_name"], row["occurrence_id"], sector, row["brand_name"], row["airing_end"] + window_s,
         row["airing_start"] - window_s)).fetchall()
    if not near:
        return "compliant", None, Decimal(0)
    gap = min(ca._gap(dict(airing_start=row["airing_start"], airing_end=row["airing_end"]), dict(n)) for n in near)
    if gap < window_s:
        return "breach", gap, money(net * Decimal(str(contract.get("separation_credit", 1.0))))
    return "compliant", gap, Decimal(0)


def gather(db_path, airing_id: int, rate_card=None, pods_cache: dict | None = None) -> dict:
    with get_db(db_path) as conn:
        row = conn.execute(ROW_SQL, (int(airing_id),)).fetchone()
        if row is None:
            raise KeyError(f"airing {airing_id} not found")
        row = dict(row)
        live = row["source_name"].startswith("live:")
        card = _rate_card(db_path, rate_card)
        pod, pod_ad = _pod_for(db_path, row, pods_cache)
        position = pod_ad["position"] if pod_ad else None
        fin = compute_airing_financials(dict(
            brand=row["brand_name"], campaign=row["campaign_name"], channel=channel_of(row["source_name"]),
            start_epoch=broadcast_epoch(row["airing_start"], row.get("mcr_drift_delta_seconds")) if live else None,
            spot_seconds=row["duration_sec"], pod_position=position,
            placement_status=(pod_ad or {}).get("placement_status")),
            card)
        contract = card.contract(row["brand_name"], row["campaign_name"]) or {}
        bound = ctr.match_contract(ctr.load_active(db_path), channel_of(row["source_name"]), row["brand_name"],
                                   row["airing_start"]) if live and fin["priced"] else None
        if bound:                                       # Track A: the ingested contract replaces rate-card pricing
            track = {}
            ca._bind_contract(track, dict(spot_seconds=row["duration_sec"], delivered_position=position,
                                          airing_start=row["airing_start"], airing_end=row["airing_end"]), bound)
            fin.update(gross_spend=track["estimated_net_spend"], estimated_net_spend=track["estimated_net_spend"],
                       booked_position=track["booked_position"], position_compliance=track["position_compliance"],
                       contract_loss_credit=track["position_credit"], contract_loss_credit_gross=track["position_credit"],
                       agency_discount=0.0)
            contract = dict(separation_credit=bound["separation_breach_penalty_pct"])
            window = float(bound["separation_window_seconds"] or 0)
        else:
            window = float(contract.get("competitor_separation_seconds") or 0)
        sep, gap, sep_credit = _separation(conn, row, window, card, contract,
                                           fin["estimated_net_spend"] if fin["priced"] else Decimal(0))
        try:
            shot = conn.execute(
                "SELECT s.visual_p_hash AS ref_phash, a.packshot_path FROM reference_signatures s "
                "LEFT JOIN discovered_assets a ON a.asset_id = s.asset_id "
                "WHERE s.ad_id = ? OR a.promoted_ad_id = ? ORDER BY a.packshot_path IS NULL LIMIT 1",
                (row["ad_id"], row["ad_id"])).fetchone()
        except sqlite3.OperationalError:                          # database predates the reference registry
            shot = None
        if shot is None:
            shot = conn.execute("SELECT NULL AS ref_phash, packshot_path FROM discovered_assets WHERE promoted_ad_id = ? "
                                "AND packshot_path IS NOT NULL LIMIT 1", (row["ad_id"],)).fetchone()
    cost = fin["gross_spend"] if fin["priced"] else Decimal(0)
    return dict(row=row, live=live, rate_card=card, fin=fin, pod=pod, position=position, contract=contract,
                window=window or None, separation=sep, bound_contract=bound, sep_gap=gap, sep_credit=sep_credit,
                ref_phash=shot["ref_phash"] if shot else None, packshot_path=shot["packshot_path"] if shot else None,
                payload=payload_for(row, cost))


# ------------------------------------------------------------------------------------------ drawing
def _styles():
    ss = getSampleStyleSheet()
    mk = lambda n, **k: ParagraphStyle(n, parent=ss["Normal"], **k)
    return dict(
        label=mk("l", fontSize=7, leading=9, textColor=SLATE_M, fontName="Helvetica-Bold"),
        val=mk("v", fontSize=9.5, leading=12, textColor=SLATE),
        h=mk("h", fontSize=8.5, leading=11, textColor=EMERALD_D, fontName="Helvetica-Bold", spaceBefore=7, spaceAfter=3),
        mono=mk("m", fontName="Courier", fontSize=6.6, leading=8, textColor=colors.white),
        monodark=mk("md", fontName="Courier", fontSize=8, leading=10, textColor=SLATE),
        foot=mk("f", fontSize=6.8, leading=8.6, textColor=SLATE_M),
        small=mk("s", fontSize=7.5, leading=9.5, textColor=SLATE_M))


def _grid(cells, st, cols=3, total_w=17 * cm):
    """cells = [(label, value-html)] laid out cols across."""
    rows = []
    for i in range(0, len(cells), cols):
        chunk = cells[i:i + cols] + [("", "")] * (cols - len(cells[i:i + cols]))
        rows.append([[Paragraph(l, st["label"]), Paragraph(v, st["val"])] for l, v in chunk])
    t = Table(rows, colWidths=[total_w / cols] * cols)
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LINEBELOW", (0, 0), (-1, -2), 0.4, RULE),
                           ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                           ("LEFTPADDING", (0, 0), (-1, -1), 0)]))
    return t


def _thumbnail(path):
    """(reportlab Image, phash of the frame) from a packshot file, downscaled in memory; (None, None) if unusable."""
    if not path or not Path(path).exists():
        return None, None
    try:
        from PIL import Image as PILImage

        from kihc.analytics.reference_matcher import phash_image
        with PILImage.open(path) as im:
            im = im.convert("RGB")
            im.thumbnail(THUMB_MAX_PX)
            ph = phash_image(im)
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=86)
        buf.seek(0)
        w, h = im.size
        scale = min(9.2 * cm / w, 5.4 * cm / h)
        return Image(buf, width=w * scale, height=h * scale), ph
    except Exception as exc:               # half-written / corrupt / unreadable frame: the certificate renders without it
        log.warning("certificate: packshot %s unusable (%s: %s)", path, type(exc).__name__, exc)
        return None, None


class _QR(Flowable):
    """QR code drawn as ONE vector path (reportlab's QrCodeWidget emits a node per run: ~3x slower)."""

    def __init__(self, text: str, size: float = 2.6 * cm):
        super().__init__()
        from reportlab.graphics.barcode import qrencoder
        q = qrencoder.QRCode(None, qrencoder.QRErrorCorrectLevel.L)
        q.addData(text)
        q.make()
        self.modules, self.width, self.height = q.modules, size, size

    def draw(self):
        n, border = len(self.modules), 2
        box = self.width / (n + 2 * border)
        path = self.canv.beginPath()
        for r, row in enumerate(self.modules):
            c = 0
            for dark, run in itertools.groupby(map(bool, row)):
                k = len(list(run))
                if dark:
                    path.rect((c + border) * box, self.height - (r + border + 1) * box, k * box, box)
                c += k
        self.canv.setFillColor(colors.white)
        self.canv.rect(0, 0, self.width, self.height, stroke=0, fill=1)
        self.canv.setFillColor(colors.black)
        self.canv.drawPath(path, stroke=0, fill=1)


def _qr(text: str, size=2.6 * cm) -> Flowable:
    return _QR(text, size)


def _money(v) -> str:
    return f"{Decimal(v):,.2f}"


ARABIC_FONT = "KIHC-Arabic"
_ARABIC_RE = re.compile("[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]")
_FONT_CANDIDATES = (DATA_DIR / "fonts" / "NotoSansArabic-Regular.ttf", DATA_DIR / "fonts" / "Amiri-Regular.ttf",
                    DATA_DIR / "fonts" / "DejaVuSans.ttf",
                    Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "arial.ttf",
                    Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "tahoma.ttf",
                    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
                    Path("/usr/share/fonts/truetype/noto/NotoSansArabic-Regular.ttf"),
                    Path("/Library/Fonts/Arial Unicode.ttf"))
_arabic_ready = None


def _arabic_support() -> bool:
    """Register the first available Unicode TTF with Arabic glyphs (once) and report whether reshaping + font are usable."""
    global _arabic_ready
    if _arabic_ready is None:
        _arabic_ready = False
        try:
            import arabic_reshaper  # noqa: F401
            from bidi.algorithm import get_display  # noqa: F401
            from reportlab.pdfbase import pdfmetrics
            from reportlab.pdfbase.ttfonts import TTFont
            for path in _FONT_CANDIDATES:
                if path.exists():
                    try:
                        pdfmetrics.registerFont(TTFont(ARABIC_FONT, str(path)))
                        _arabic_ready = True
                        break
                    except Exception as exc:
                        log.warning("certificate: font %s unusable (%s)", path, exc)
        except ImportError:
            log.warning("certificate: arabic_reshaper/python-bidi not installed - Arabic falls back to '?'")
    return _arabic_ready


def safe_text(value) -> str:
    """Text for a ReportLab Paragraph: markup escaped. Arabic is reshaped (joined forms), put in visual bidi order and set in a
    registered Unicode TTF; any other glyph Helvetica cannot draw (emoji) becomes '?' - the certificate never fails on a name
    it cannot typeset."""
    text = str(value)
    if _ARABIC_RE.search(text) and _arabic_support():
        import arabic_reshaper
        from bidi.algorithm import get_display
        shaped = get_display(arabic_reshaper.reshape(text))
        out = []
        for ch in escape(shaped):
            out.append(ch if ch.isprintable() or ch.isspace() else "?")
        return f"<font name='{ARABIC_FONT}'>{''.join(out)}</font>"
    out = []
    for ch in escape(text):
        try:
            ch.encode("cp1252")
            out.append(ch)
        except UnicodeEncodeError:
            out.append("?")
    return "".join(out)


def drift_text(row) -> str:
    """' CET ± 1.2 s broadcast latency' (+ the measured capture lag) when the airing carries a verified clock reading."""
    try:
        m, d = float(row.get("clock_drift_margin_s")), float(row.get("clock_drift_seconds"))
        if not (math.isfinite(m) and math.isfinite(d)):
            raise ValueError
    except (TypeError, ValueError):
        return " CET <font color='#64748B'>(broadcast-clock latency not measured)</font>"
    basis = {"program-date-time": "HLS program-date-time", "pts-relative": "MPEG-TS PTS (relative)"}.get(
        row.get("clock_drift_basis"), row.get("clock_drift_basis") or "")
    d = float(d)
    rel = (f"capture lags the broadcaster clock by {d:.1f} s" if d >= 0 else
           f"broadcaster clock runs {-d:.1f} s ahead of the NTP capture clock")
    try:
        mcr = (f"; broadcaster clock at start ≈ {datetime.fromtimestamp(float(row['airing_start']) - d, ALGERIA_TZ):%H:%M:%S}"
               if row.get("airing_start") else "")
    except (OverflowError, OSError, ValueError, TypeError):
        mcr = ""
    return (f" CET ± {float(m):.1f} s broadcast latency<br/><font color='#64748B' size='7'>{rel} ({basis}){mcr}</font>")


def mcr_delta(row) -> float | None:
    """broadcast - ingest in seconds: the stored Phase 26 column, else derived from the Phase 22 reading (capture - broadcaster)."""
    for key, sign in (("mcr_drift_delta_seconds", 1.0), ("clock_drift_seconds", -1.0)):
        try:
            v = float(row.get(key))
        except (TypeError, ValueError):
            continue
        if math.isfinite(v):
            return sign * v
    return None


def mcr_air_time_text(row, when) -> str:
    """'22:59:58 CET ±0.3 s · MCR Drift -5.2 s' + the raw ingest stamp and the sync lock, or '' when the clock is not calibrated.
    The ingest time is shown unaltered next to the calibrated one: the certificate carries both ends of the audit trail."""
    delta = mcr_delta(row)
    if delta is None or when is None:
        return ""
    try:
        air = datetime.fromtimestamp(float(row["airing_start"]) + delta, ALGERIA_TZ)
        end = datetime.fromtimestamp(float(row["airing_end"]) + delta, ALGERIA_TZ)
        margin = float(row.get("clock_drift_margin_s"))
        margin = margin if math.isfinite(margin) else None
    except (OverflowError, OSError, ValueError, TypeError, KeyError):
        return ""
    lock = lock_status(row.get("clock_drift_basis"), margin)
    colour = "#047857" if lock == LOCKED else "#B45309"
    pm = f" ±{margin:.1f} s" if margin is not None else ""
    return (f"<b>{air:%H:%M:%S} CET</b>{pm} · MCR Drift {delta:+.1f} s<br/>"
            f"<font color='#64748B' size='7'>{air:%Y-%m-%d} to {end:%H:%M:%S} CET broadcast · ingest clock "
            f"{when:%H:%M:%S}.{when.microsecond // 1000:03d} CET (raw, unaltered) · MCR sync </font>"
            f"<font color='{colour}' size='7'><b>{lock}</b></font><br/>")


def status_text(row) -> str:
    s = row.get("status") or "CONFIRMED"
    if s == "CONFIRMED":
        return "<b>CONFIRMED</b>"
    why = ("music-dominant spot without a packshot match" if "MUSIC" in s
           else "cut-down ended without the trailing level dip")
    return (f"<b><font color='#B91C1C'>{s.replace('_', ' ')}</font></b> <font color='#64748B'>({why}; "
            f"excluded from penalty claims until visually resolved)</font>")


def render(data: dict, digest: str, sig: str) -> bytes:
    st = _styles()
    row, fin, p = data["row"], data["fin"], data["payload"]
    record_ok = verify_row(row)                                     # detection-time hash vs record + evidence file
    banner_ok = record_ok is not False
    raw_p = p                                                       # the seal covers the raw payload, display gets safe text
    row = {**row, **{k: safe_text(row.get(k)) for k in ("brand_name", "campaign_name", "detected_at", "source_name",
                                                       "evidence_sha256", "industry") if row.get(k) is not None}}
    p = {**p, "channel": safe_text(p["channel"]), "brand": safe_text(p["brand"])}
    try:
        when = datetime.fromtimestamp(row["airing_start"], ALGERIA_TZ) if data["live"] else None
    except (OverflowError, OSError, ValueError, TypeError):
        when = None
    pod = data["pod"]
    clutter = (f"<b>{pod.clutter}</b> · {pod.total_ads_in_pod} ad(s) in a {pod.pod_duration_seconds:.0f} s break"
               if pod else "—")
    shot_img, frame_ph = _thumbnail(row.get("evidence_frame_path") if row.get("evidence_frame_path") and
                                    Path(row["evidence_frame_path"]).exists() else data["packshot_path"])
    ref_ph = data["ref_phash"]
    try:
        dist = bin(int(frame_ph, 16) ^ int(ref_ph, 16)).count("1") if frame_ph and ref_ph else None
    except (TypeError, ValueError):
        dist = None

    buf = io.BytesIO()
    kw = f"{CERT_VERSION} b64={base64.b64encode(canonical_payload(raw_p)).decode()} sha256={digest} hmac={sig}"
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.2 * cm, bottomMargin=1.1 * cm,
                            title=f"KIHC Airing Certificate KIHC-{p['airing_id']:06d}", author="KIHC",
                            subject="Proof of Performance - broadcast airing certificate", keywords=kw, pageCompression=0)
    header = Table([[Paragraph("<font color='white' size='15'><b>KIHC Regulatory Broadcast Verification<br/>"
                               "&amp; Airing Certificate</b></font>", ParagraphStyle("hd", parent=st["val"], leading=20)),
                     Paragraph(f"<font color='#A7F3D0' size='8'>PROOF OF PERFORMANCE</font><br/>"
                               f"<font color='white' size='11'><b>No. KIHC-{p['airing_id']:06d}</b></font><br/>"
                               f"<font color='#A7F3D0' size='7.5'>Issued {datetime.now(ALGERIA_TZ):%Y-%m-%d %H:%M} CET</font>",
                               ParagraphStyle("r", parent=st["val"], alignment=2, leading=13))]],
                    colWidths=[11 * cm, 6 * cm])
    header.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), SLATE), ("LINEBELOW", (0, 0), (-1, -1), 3, EMERALD),
                                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 10),
                                ("BOTTOMPADDING", (0, 0), (-1, -1), 10), ("LEFTPADDING", (0, 0), (-1, -1), 12),
                                ("RIGHTPADDING", (0, 0), (-1, -1), 12)]))

    status = Table([[Paragraph(
        ("<font name='ZapfDingbats' color='#059669'>l</font> <font color='#047857'><b>PASS / VERIFIED</b></font> &nbsp;<font color='#475569'>Record integrity verified at "
         "issue" + (" (detection hash and evidence match)." if record_ok else " (no detection-time hash on file).") + "</font>")
        if banner_ok else "<font name='ZapfDingbats' color='#B91C1C'>l</font> <font color='#B91C1C'><b>FAIL</b></font> <font color='#475569'>The airing record or its evidence "
        "changed since detection. Do not rely on this certificate.</font>", st["val"])]], colWidths=[17 * cm])
    status.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), EMERALD_L if banner_ok else colors.HexColor("#FEE2E2")),
                                ("BOX", (0, 0), (-1, -1), 0.6, EMERALD if banner_ok else RED),
                                ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))

    meta = _grid([
        ("BRAND · CAMPAIGN", f"<b>{row['brand_name']}</b> · {row['campaign_name']}"),
        ("CHANNEL", f"<b>{p['channel']}</b>"),
        ("BROADCAST AIR TIME (MCR CALIBRATED, CET)" if mcr_air_time_text(row, when) else "BROADCAST AIRING TIME (CET, UTC+1)",
         (mcr_air_time_text(row, when) + "<font size='7' color='#64748B'>Clock audit: " + drift_text(row).strip().removeprefix("CET ")
          + "</font>" if mcr_air_time_text(row, when) else
          f"<b>{when:%Y-%m-%d %H:%M:%S}.{when.microsecond // 1000:03d}</b>" + drift_text(row)) if when else
         f"{row['airing_start']:.3f} s into recorded file"),
        ("EXACT MEASURED DURATION", f"<b>{float(row['duration_detected']):.3f} s</b> "
                                    f"<font color='#64748B'>(creative {row['duration_sec']:.2f} s)</font>"),
        ("POD POSITION", f"<b>{data['position'] or 'n/a'}</b>" + (f" · {pod_ad_of(pod, row)}" if pod else "")),
        ("CLUTTER INDEX", clutter),
        ("MATCH CONFIDENCE", f"<b>{float(row['confidence_score']) * 100:.2f}%</b>"),
        ("DETECTED AT (UTC)", str(row["detected_at"])),
        ("AIRING STATUS", status_text(row)),
        ("HUMAN REVIEW", {1: "Verified by reviewer", 0: "Not yet reviewed", -1: "Rejected"}.get(row["human_verified"], "—")),
    ], st)

    proof_cells = [Paragraph("PERCEPTUAL HASH (pHash, 64-bit)", st["label"]),
                   Paragraph(f"<font name='Courier'><b>{frame_ph or '—'}</b></font>", st["val"]),
                   Spacer(1, 4), Paragraph("REFERENCE pHash", st["label"]),
                   Paragraph(f"<font name='Courier'>{ref_ph or '—'}</font>", st["val"]), Spacer(1, 4),
                   Paragraph("VISUAL AGREEMENT", st["label"]),
                   Paragraph((f"<b>{dist}/64 bits apart</b> " + ("(same packshot)" if dist <= 10 else "(differs)"))
                             if dist is not None else "not comparable", st["val"]), Spacer(1, 4),
                   Paragraph("Frame captured from the verified broadcast of this creative; shown at reduced resolution.",
                             st["small"])]
    proof = Table([[shot_img or Paragraph("Packshot frame not available for this airing.", st["small"]), proof_cells]],
                  colWidths=[9.6 * cm, 7.4 * cm])
    proof.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("BOX", (0, 0), (0, 0), 0.5, RULE),
                               ("LEFTPADDING", (1, 0), (1, 0), 12), ("ALIGN", (0, 0), (0, 0), "CENTER")]))

    c = data["contract"]
    booked = fin.get("booked_position") or "none booked"
    mg = []
    if fin.get("position_compliance") == "breach":
        mg.append(f"Position breach: booked {booked}, delivered {data['position']}; credit {_money(fin['contract_loss_credit'])} DZD net")
    if data["separation"] == "breach":
        mg.append(f"Separation breach: competitor {data['sep_gap']:.0f} s away (< {data['window']:.0f} s); "
                  f"credit {_money(data['sep_credit'])} DZD net")
    mg_txt = ("<font color='#B91C1C'><b>BREACH</b></font> — " + "; ".join(mg)) if mg else \
        "<font color='#047857'><b>No breach</b></font> — no make-good due"
    fin_grid = _grid([
        ("GROSS VALUE (DZD)", f"<b>{_money(fin['gross_spend'])}</b>" if fin["priced"] else "unpriced"),
        ("NET VALUE (DZD)", f"<b>{_money(fin['estimated_net_spend'])}</b>" if fin["priced"] else "unpriced"),
        ("RATE CARD", f"{data['rate_card'].status}" + (" <font color='#B45309'>(illustrative)</font>"
                                                       if data["rate_card"].is_illustrative else "")),
        ("APPLIED SLA — SEPARATION", f"{data['window']:.0f} s competitor separation" if data["window"] else "none contracted"),
        ("APPLIED SLA — POSITION", booked + (f" · delivered {data['position']}" if data["position"] else "")),
        ("SEPARATION STATUS", data["separation"]),
        ("MAKE-GOOD / BREACH CREDIT", mg_txt),
    ], st, cols=3)
    if not fin["priced"]:
        fin_grid = _grid([("VALUE", f"unpriced — {fin['unpriced_reason']}"),
                          ("APPLIED SLA", f"separation {data['window'] or 'none'} · position {booked}"),
                          ("MAKE-GOOD", mg_txt)], st)

    seal_txt = Table([[[Paragraph("<font color='#6EE7B7' size='7.5'><b>TAMPER-EVIDENT SEAL · SHA-256</b></font>", st["mono"]),
                        Spacer(1, 3), Paragraph(f"{digest}", st["mono"]), Spacer(1, 4),
                        Paragraph(f"<font color='#94A3B8'>covers: {' + '.join(PAYLOAD_FIELDS)}</font>", st["mono"]),
                        Spacer(1, 3), Paragraph(f"<font color='#94A3B8'>HMAC {sig[:32]}…</font>", st["mono"]),
                        Spacer(1, 4), Paragraph("<font color='#6EE7B7'><b>INTEGRITY: PASS / VERIFIED at issue</b></font>"
                                                 if banner_ok else "<font color='#FCA5A5'><b>INTEGRITY: FAIL</b></font>",
                                                 st["mono"])],
                       _qr(f"kihc-pop:{CERT_VERSION}:{p['airing_id']}:{digest}")]], colWidths=[13.6 * cm, 3.4 * cm])
    seal_txt.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), SLATE_D), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                                  ("TOPPADDING", (0, 0), (-1, -1), 9), ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
                                  ("LEFTPADDING", (0, 0), (-1, -1), 12), ("LINEBEFORE", (0, 0), (0, 0), 4, EMERALD)]))

    bc = data.get("bound_contract")
    banner_txt, banner_bg = ((f"CERTIFIED CONTRACT AUDIT \u2014 Bound to Order #{bc['contract_id']}", EMERALD_D) if bc else
                             ("MARKET ESTIMATE \u2014 Independent Benchmark Telemetry", colors.HexColor("#B45309")))
    track_banner = Table([[Paragraph(f"<font color='white'><b>{escape(banner_txt)}</b></font>",
                                     ParagraphStyle("trk", parent=st["val"], alignment=1, fontSize=10, leading=13))]],
                         colWidths=[17 * cm])
    track_banner.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), banner_bg), ("TOPPADDING", (0, 0), (-1, -1), 5),
                                      ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
    story = [header, Spacer(1, 7), track_banner, Spacer(1, 5), status, Paragraph("AIRING METADATA", st["h"]), meta,
             Paragraph("VISUAL PROOF", st["h"]), proof, Paragraph("FINANCIAL &amp; CONTRACT AUDIT", st["h"]), fin_grid,
             *([Spacer(1, 4), Paragraph(
                 "<b>BREACH EVIDENCE CLIP</b> (airing −3 s .. +3 s, audio) · play/download: "
                 f"/api/evidence/play/{p['airing_id']} · SHA-256 <font name='Courier' size='6.5'>"
                 f"{row['evidence_sha256']}</font>", st["foot"])] if row.get("evidence_sha256") else []),
             Paragraph("CRYPTOGRAPHIC VERIFICATION", st["h"]), seal_txt, Spacer(1, 6),
             Paragraph("Factual electronic broadcast telemetry recorded via SHA-256 tamper-evident logging. This document "
                       "certifies verified physical airing and timecode (NTP CET / UTC+1) as technical ground truth for "
                       "contract reconciliation.", st["foot"]), Spacer(1, 3),
             Paragraph("This certificate attests that the creative above was detected on air on the stated channel at the stated "
                       "time and for the stated duration, identified by audio fingerprint matching. It is proof of broadcast "
                       "occurrence and airtime only; it is NOT a measurement of audience, reach or impressions. Values are "
                       "derived from " + ("the contract bound above" if bc else "the configured rate card, as illustrative "
                       "benchmarks,") + " and become claims only after verification against the "
                       "broadcaster's as-run log. The seal makes any later edit of the listed fields detectable; it is not a "
                       "qualified digital signature. Verify with KIHC: certificate_authority.verify_certificate_pdf().",
                       st["foot"])]
    doc.build([KeepInFrame(17 * cm, 26.2 * cm, story, mode="shrink")])
    return buf.getvalue()


def pod_ad_of(pod, row):
    for a in pod.ads:
        if a["occurrence_id"] == row["occurrence_id"]:
            return a["position_of"]
    return ""


# ------------------------------------------------------------------------------------------ public API
def generate_airing_certificate_pdf(airing_id: int, db_path=None, *, rate_card=None, pods_cache: dict | None = None) -> bytes:
    """Never raises for a rendering problem: KeyError only when the airing does not exist. If the full layout fails for any
    reason (bad optional metadata, unreadable image, odd text) a simplified but still sealed certificate is produced."""
    try:
        return _generate_full(airing_id, db_path, rate_card=rate_card, pods_cache=pods_cache)
    except KeyError:
        raise
    except Exception as exc:
        log.exception("certificate %s: full render failed (%s: %s) - issuing the simplified certificate", airing_id,
                      type(exc).__name__, exc)
        return fallback_pdf(airing_id, db_path, f"{type(exc).__name__}: {exc}")


def fallback_pdf(airing_id: int, db_path=None, reason: str = "") -> bytes:
    """Plain, dependency-light certificate from the stored row alone (no images, contracts, pods or rate card)."""
    from reportlab.pdfgen import canvas
    from kihc.database.schema import get_db
    row = None
    try:
        with get_db(db_path) as conn:
            r = conn.execute(ROW_SQL, (int(airing_id),)).fetchone()
            row = dict(r) if r else None
    except Exception:
        log.exception("certificate %s: fallback could not read the row", airing_id)
    if row is None:
        raise KeyError(f"airing {airing_id} not found")

    def num(key, fmt, default="n/a"):
        try:
            return format(float(row[key]), fmt)
        except (TypeError, ValueError, KeyError):
            return default

    try:
        digest, sig = seal(payload_for(row, Decimal(0)))
    except Exception:
        digest = sig = "unavailable"
    plain = lambda v: "".join(c if ord(c) < 256 else "?" for c in str(v if v is not None else "n/a"))        # noqa: E731
    lines = [("Airing", f"KIHC-{int(airing_id):06d}"), ("Brand", plain(row.get("brand_name"))),
             ("Campaign", plain(row.get("campaign_name"))), ("Channel", plain(channel_of(row.get("source_name") or ""))),
             ("Aired (CET, UTC+1)", plain(row.get("algeria_timestamp") or f"{num('airing_start', '.3f')} s into recording")),
             ("Measured duration", num("duration_detected", ".3f") + " s"),
             ("Match confidence", num("confidence_score", ".4f")), ("Airing status", plain(row.get("status") or "CONFIRMED")),
             ("Broadcast air (MCR)", (datetime.fromtimestamp(float(row["airing_start"]) + mcr_delta(row), ALGERIA_TZ)
                                      .strftime("%Y-%m-%d %H:%M:%S") + f" CET (drift {mcr_delta(row):+.1f} s)")
              if row.get("airing_start") is not None and mcr_delta(row) is not None else "not calibrated"),
             ("Clock offset", (num("clock_drift_seconds", "+.1f") + " s +/- " + num("clock_drift_margin_s", ".1f") + " s")
              if row.get("clock_drift_seconds") is not None else "not measured"),
             ("Detection hash", plain(row.get("verification_hash"))), ("Seal SHA-256", digest), ("Seal HMAC", sig)]
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4, pageCompression=0)
    w, h = A4
    c.setFillColor(SLATE)
    c.rect(0, h - 3.2 * cm, w, 3.2 * cm, stroke=0, fill=1)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(2 * cm, h - 1.7 * cm, "KIHC Airing Certificate")
    c.setFont("Helvetica", 9)
    c.drawString(2 * cm, h - 2.4 * cm, "Proof of performance - simplified layout")
    y = h - 4.6 * cm
    for k, v in lines:
        c.setFillColor(SLATE_M)
        c.setFont("Helvetica-Bold", 8)
        c.drawString(2 * cm, y, k.upper())
        c.setFillColor(SLATE)
        c.setFont("Courier" if "hash" in k.lower() or "seal" in k.lower() else "Helvetica", 9.5)
        c.drawString(6.2 * cm, y, v[:78])
        if len(v) > 78:
            c.drawString(6.2 * cm, y - 11, v[78:156])
            y -= 11
        y -= 20
    c.setFillColor(SLATE_M)
    c.setFont("Helvetica", 7.5)
    c.drawString(2 * cm, 2.2 * cm, "Some optional details (images, contract, pod context) were unavailable and are omitted; the airing "
                                   "facts above are taken from the sealed record.")
    c.drawString(2 * cm, 1.7 * cm, "Proof of broadcast occurrence only - not a measure of audience, reach or impressions.")
    c.showPage()
    c.save()
    return buf.getvalue()


def _generate_full(airing_id: int, db_path=None, *, rate_card=None, pods_cache: dict | None = None) -> bytes:
    """One-page certificate PDF for ad_occurrences.occurrence_id. Raises KeyError if the airing does not exist."""
    db_path = db_path or DEFAULT_DB_PATH
    data = gather(db_path, airing_id, rate_card, pods_cache)
    digest, sig = seal(data["payload"])
    return render(data, digest, sig)


_KW = re.compile(rb"/Keywords \(" + re.escape(CERT_VERSION.encode()) + rb" b64=([A-Za-z0-9+/=]+) sha256=([0-9a-f]{64}) hmac=([0-9a-f]{64})\)")


def extract_seal(pdf: bytes) -> dict | None:
    m = _KW.search(pdf)
    if not m:
        return None
    try:
        body = base64.b64decode(m[1], validate=True)
        head, _, js = body.partition(b"\n")
        return dict(body=body, payload=json.loads(js), version=head.decode(), sha256=m[2].decode(), hmac=m[3].decode())
    except (ValueError, UnicodeDecodeError):
        return dict(corrupt=True)


def verify_certificate_pdf(pdf: bytes, db_path=None, key: bytes | None = None) -> dict:
    """Integrity report. status PASS only if every applicable check passes; any single-byte change of the sealed
    payload, hash or HMAC yields FAIL."""
    checks, notes = {}, []
    s = extract_seal(pdf)
    if s is None or s.get("corrupt"):
        return dict(status="FAIL", checks={"seal_present": False}, notes=["no readable KIHC seal in this PDF"], payload=None)
    checks["seal_present"] = True
    payload = s["payload"]
    checks["payload_complete"] = set(payload) == set(PAYLOAD_FIELDS) and s["version"] == CERT_VERSION
    body = s["body"]
    digest = hashlib.sha256(body).hexdigest()
    checks["sha256_matches_payload"] = hmac.compare_digest(digest, s["sha256"])
    checks["sha256_printed_on_page"] = pdf.count(s["sha256"].encode()) >= 2          # /Keywords + the printed seal line
    k = key or signing_key()
    checks["hmac_valid"] = hmac.compare_digest(hmac.new(k, body + s["sha256"].encode(), hashlib.sha256).hexdigest(), s["hmac"])
    if db_path is not None and checks["payload_complete"]:
        try:
            data = gather(db_path, int(payload["airing_id"]))
            now = data["payload"]
            same = {f: str(now[f]) == str(payload[f]) for f in PAYLOAD_FIELDS if f != "calculated_cost_dzd"}
            checks["matches_current_record"] = all(same.values())
            if not all(same.values()):
                notes.append("record changed since issue: " + ", ".join(f for f, ok in same.items() if not ok))
            if now["calculated_cost_dzd"] != payload["calculated_cost_dzd"]:
                notes.append(f"cost drift (rate card changed?): issued {payload['calculated_cost_dzd']} now {now['calculated_cost_dzd']}")
        except KeyError:
            checks["matches_current_record"] = False
            notes.append("airing no longer exists in the database")
    return dict(status="PASS" if all(checks.values()) else "FAIL", checks=checks, notes=notes, payload=payload,
                sha256=s["sha256"])


# ------------------------------------------------------------------------------------------ batch
def batch_airing_ids(db_path, start_date=None, end_date=None, brand=None, channel=None) -> list[int]:
    sql = ("SELECT o.occurrence_id, o.source_name, o.airing_start FROM ad_occurrences o JOIN reference_ads r USING (ad_id) "
           "WHERE o.human_verified != -1")
    args = []
    if brand:
        sql += " AND lower(r.brand_name) = lower(?)"
        args.append(brand.strip())
    with get_db(db_path) as conn:
        rows = conn.execute(sql + " ORDER BY o.airing_start, o.occurrence_id", args).fetchall()
    d0 = datetime.fromisoformat(start_date).date() if start_date else None
    d1 = datetime.fromisoformat(end_date).date() if end_date else None
    out = []
    for r in rows:
        if channel and channel_of(r["source_name"]).casefold() != channel.strip().casefold():
            continue
        if d0 or d1:
            if not r["source_name"].startswith("live:"):
                continue                                         # file scans carry no calendar date
            day = datetime.fromtimestamp(r["airing_start"], ALGERIA_TZ).date()
            if (d0 and day < d0) or (d1 and day > d1):
                continue
        out.append(r["occurrence_id"])
    return out


def export_batch_zip(db_path, start_date=None, end_date=None, brand=None, channel=None, limit: int = MAX_BATCH) -> tuple[bytes, dict]:
    """ZIP of one certificate per airing + MANIFEST.csv (id, brand, channel, time, sha256 seal, pdf sha256) in memory."""
    ids = batch_airing_ids(db_path, start_date, end_date, brand, channel)
    total, ids = len(ids), ids[:limit]
    card, pods = RateCardEngine.load(db_path=db_path), {}
    buf = io.BytesIO()
    lines = ["airing_id,brand,channel,timestamp_cet,duration_s,confidence,cost_dzd,seal_sha256,pdf_sha256,file"]
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for i in ids:
            pdf = generate_airing_certificate_pdf(i, db_path, rate_card=card, pods_cache=pods)
            s = extract_seal(pdf)["payload"]
            name = f"KIHC-certificate-{i:06d}.pdf"
            z.writestr(name, pdf)
            lines.append(",".join([str(i), '"%s"' % s["brand"].replace('"', '""'), '"%s"' % s["channel"], s["timestamp_cet"],
                                   s["exact_duration"], s["matched_confidence"], s["calculated_cost_dzd"],
                                   extract_seal(pdf)["sha256"], hashlib.sha256(pdf).hexdigest(), name]))
        z.writestr("MANIFEST.csv", "\n".join(lines) + "\n")
    return buf.getvalue(), dict(count=len(ids), matched=total, truncated=total > len(ids))
