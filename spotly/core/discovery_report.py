"""Enrich discovered airings with daypart, break placement and a provisional rate-card value.

Breaks are reconstructed per channel from discovered airings *and* known (reference-ad) airings, with
the same gap rule as pod_analyzer. Values are PROVISIONAL: an unclassified asset may be a promo or
ident, not a paid spot, so these figures never enter spend totals until the asset is classified.
"""
import json
import sqlite3
from collections import defaultdict
from datetime import datetime

from spotly.analytics.brand_extractor import display_label, strip_internal
from spotly.analytics.transcriber import SOUNDTRACK_LABEL, assess_transcript
from spotly.analytics.rate_engine import RateCardEngine, compute_airing_financials
from spotly.core.pod_analyzer import MAX_GAP_SECONDS, Pod, calculate_ad_positions
from spotly.core.proof import ALGERIA_TZ, algeria_iso
from spotly.database.schema import get_db

ASSETS_SQL = """
SELECT asset_id, channel, duration_sec, nominal_duration_sec, duration_basis, audio_path, audio_post_pad_sec,
       status, label, promoted_ad_id, first_seen, last_seen, occurrences, created_at,
       transcript, transcript_language, transcription_model, transcribed_at, transcript_logprob,
       suggested_brand, suggested_industry, brand_confidence, brand_evidence, content_hint,
       merged_into, related_assets, suggestion_source, visual_status, packshot_path, packshot_at, packshot_conf,
       packshot_attempts, packshot_last_error, ocr_text, ocr_at, visual_features, logo_path
FROM discovered_assets {where} ORDER BY last_seen DESC, asset_id
"""
MUSIC_VISUAL_NEEDED = "[🎵 Music / Audio Spot - Visual Needed]"


def review_label(r: dict) -> str | None:
    """What a reviewer should know first about an unclassified asset (derived, never stored)."""
    if r["status"] != "unclassified":
        return None
    if r.get("content_hint"):
        return r["content_hint"]
    if r.get("suggested_brand"):
        return None                                   # the brand badge says it
    if r.get("transcribed_at") and not (r.get("transcript") or "").strip() and not r.get("packshot_path"):
        return MUSIC_VISUAL_NEEDED                    # no usable speech and no frame yet
    if r.get("visual_status") == "ocr_no_brand":
        return "Packshot captured - no brand text recognised; check the frame"
    if r.get("visual_status") == "needed":
        return "No brand heard - packshot will be captured at its next airing"
    return None


def list_assets(db_path, channel=None, include_ignored=True, include_merged=False, with_metrics=False) -> list[dict]:
    clauses, params = [], []
    if channel:
        clauses.append("channel = ?")
        params.append(channel)
    if not include_ignored:
        clauses.append("status != 'ignored'")
    if not include_merged:
        clauses.append("status != 'merged'")          # its airings count under merged_into
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    with get_db(db_path) as conn:
        rows = [dict(r) for r in conn.execute(ASSETS_SQL.format(where=where), params)]
    ads = {}
    with get_db(db_path) as conn:
        ads = {x["ad_id"]: dict(x) for x in conn.execute("SELECT ad_id, brand_name, campaign_name, duration_sec FROM reference_ads")}
    for r in rows:
        ad = ads.get(r["promoted_ad_id"]) if r.get("promoted_ad_id") else None
        if ad:                                             # classified: one clean label, never ids or stitched fragments
            r["label"] = display_label(ad["brand_name"], ad["campaign_name"], ad["duration_sec"])
        elif r.get("label"):
            r["label"] = strip_internal(r["label"]) or r["label"]
        show, why = assess_transcript(r.get("transcript"), r.get("transcript_logprob"))
        r["transcript_quality"] = ("none" if not (r.get("transcript") or "").strip() else "speech" if show else "soundtrack")
        r["transcript_note"] = why
        if r["transcript_quality"] == "soundtrack":        # hallucinated / ungrounded text never reaches the UI
            r["transcript"], r["transcript_display"] = "", SOUNDTRACK_LABEL
        else:
            r["transcript_display"] = r.get("transcript") or ""
        r["first_seen_cet"] = algeria_iso(r["first_seen"]) if r["first_seen"] else None
        r["last_seen_cet"] = algeria_iso(r["last_seen"]) if r["last_seen"] else None
        r["has_audio"] = bool(r["audio_path"])
        r["has_packshot"] = bool(r["packshot_path"])
        r["has_logo"] = bool(r["logo_path"])
        r["review_label"] = review_label(r)
        try:
            r["visual_features"] = json.loads(r["visual_features"]) if r["visual_features"] else None
        except ValueError:
            r["visual_features"] = None
    if with_metrics and rows:
        from spotly.analytics.creative_metrics import compute, packshot_distance
        try:
            with get_db(db_path) as conn:
                ref_hash = {x["asset_id"]: x["visual_p_hash"] for x in conn.execute(
                    "SELECT asset_id, visual_p_hash FROM reference_signatures WHERE visual_p_hash IS NOT NULL")}
        except sqlite3.OperationalError:                   # registry not built yet on this database
            ref_hash = {}
        placements = defaultdict(list)
        for o in discovered_airings(db_path, channel, limit=0):
            placements[o["asset_id"]].append(o)
        for r in rows:
            music_only = bool(r.get("transcribed_at")) and r["transcript_quality"] != "speech"
            dist = packshot_distance(r.get("packshot_path"), ref_hash.get(r["asset_id"])) if music_only else None
            r["creative"] = compute(r, r["visual_features"], placements.get(r["asset_id"], []), dist)
    return rows


def discovered_airings(db_path, channel=None, limit: int = 200, rate_card: RateCardEngine | None = None,
                       max_gap_seconds: float = MAX_GAP_SECONDS) -> list[dict]:
    rate_card = rate_card or RateCardEngine.load(db_path=db_path)
    with get_db(db_path) as conn:
        q = ("SELECT o.*, a.status, a.label FROM discovered_occurrences o JOIN discovered_assets a USING (asset_id) "
             "WHERE a.status NOT IN ('ignored', 'merged')" +(" AND o.channel = ?" if channel else "") + " ORDER BY o.airing_start")
        disc = [dict(r) for r in conn.execute(q, (channel,) if channel else ())]
        channels = sorted({d["channel"] for d in disc})
        known = []
        for ch in channels:
            known += [dict(r, channel=ch) for r in conn.execute(
                "SELECT o.occurrence_id, o.airing_start, o.airing_end, r.brand_name, r.campaign_name "
                "FROM ad_occurrences o JOIN reference_ads r USING (ad_id) "
                "WHERE o.source_name = ? AND o.human_verified != -1", (f"live:{ch}",))]
    # ---- break reconstruction (discovered + known airings, per channel)
    events = defaultdict(list)
    for d in disc:
        if d["status"] == "classified":
            continue                                  # its airings were backfilled into ad_occurrences
        events[d["channel"]].append(dict(occurrence_id=f"D{d['id']}", kind="discovered", brand=d["asset_id"],
                                         ad_type="unclassified", start=d["airing_start"], end=d["airing_end"],
                                         duration=d["duration_sec"]))
    for k in known:
        events[k["channel"]].append(dict(occurrence_id=f"K{k['occurrence_id']}", kind="known",
                                         brand=f"{k['brand_name']} – {k['campaign_name']}", ad_type="known",
                                         start=k["airing_start"], end=k["airing_end"],
                                         duration=k["airing_end"] - k["airing_start"]))
    placement = {}
    for ch, evs in events.items():
        evs.sort(key=lambda e: (e["start"], e["end"]))
        pods, cur = [], None
        for e in evs:
            if cur and e["start"] - cur.pod_end <= max_gap_seconds:
                cur.ads.append(e)
                cur.pod_end = max(cur.pod_end, e["end"])
            else:
                cur = Pod("", ch, f"live:{ch}", True, e["start"], e["end"], [e])
                pods.append(cur)
        per_day = defaultdict(int)
        for p in pods:
            calculate_ad_positions(p)
            day = datetime.fromtimestamp(p.pod_start, ALGERIA_TZ)
            per_day[day.date()] += 1
            idx = per_day[day.date()]
            for a in p.ads:
                placement[a["occurrence_id"]] = dict(
                    break_id=f"BRK-{day:%Y%m%d}-{idx:03d}", break_index=idx, break_ads=p.total_ads_in_pod,
                    break_seconds=p.pod_duration_seconds, position=a["position"], position_label=a["position_label"],
                    position_of=a["position_of"], clutter=p.clutter)
    out = []
    for d in reversed(disc[-limit:] if limit else disc):
        if d["status"] == "classified":
            continue
        pl = placement.get(f"D{d['id']}", {})
        fin = compute_airing_financials(dict(brand=d["asset_id"], campaign="(unclassified)", channel=d["channel"],
                                             start_epoch=d["airing_start"], spot_seconds=d["duration_sec"],
                                             pod_position=pl.get("position")), rate_card)
        out.append(dict(
            id=d["id"], asset_id=d["asset_id"], status=d["status"], channel=d["channel"],
            airing_start=d["airing_start"], airing_end=d["airing_end"],
            start_cet=d["algeria_timestamp"] or algeria_iso(d["airing_start"]), end_cet=algeria_iso(d["airing_end"]),
            duration_sec=d["duration_sec"], confidence=d["confidence"], daypart=fin["daypart"], season=fin["season"],
            **pl, provisional_gross=float(fin["gross_spend"]), provisional_net=float(fin["estimated_net_spend"]),
            valuation_basis=f"PROVISIONAL (unclassified asset; rate card {rate_card.status})"))
    return out
