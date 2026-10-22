"""Executive-overview figures shared by every UI (FastAPI web app and Streamlit dashboard).

Pure Python over SQLite rows - no web framework, no pandas - so both front ends show the
same numbers. Human-rejected occurrences (human_verified = -1) are excluded, matching the
analytics modules.
"""
from collections import Counter
from datetime import datetime, timedelta, timezone

from spotly.core.analytics import channel_of, compute_share_of_voice
from spotly.core.pod_analyzer import (channel_clutter,
                                    cluster_occurrences_into_pods,
                                    position_distribution)
from spotly.core.proof import ALGERIA_TZ
from spotly.database.schema import init_db

AIRINGS_SQL = """
SELECT o.occurrence_id, o.ad_id, o.source_name, o.airing_start, o.airing_end, o.duration_detected,
       o.confidence_score, o.human_verified, o.detected_at, o.algeria_timestamp, o.verification_hash,
       o.evidence_frame_path, r.brand_name, r.campaign_name, r.industry, r.ad_type
FROM ad_occurrences o JOIN reference_ads r USING (ad_id)
WHERE o.human_verified != -1
ORDER BY o.detected_at DESC, o.occurrence_id DESC
"""


def _parse_ts(value) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def load_airings(db_path) -> list[dict]:
    conn = init_db(db_path)
    try:
        return [dict(r) for r in conn.execute(AIRINGS_SQL)]
    finally:
        conn.close()


def detections_per_day(rows, days: int = 14, now: datetime | None = None) -> list[dict]:
    """Detections logged per UTC day (by detected_at) for the last `days` days, oldest first."""
    now = now or datetime.now(timezone.utc)
    today = now.astimezone(timezone.utc).date()
    counts = Counter(ts.astimezone(timezone.utc).date() for ts in (_parse_ts(r["detected_at"]) for r in rows) if ts)
    return [dict(date=(today - timedelta(days=i)).isoformat(), count=counts.get(today - timedelta(days=i), 0))
            for i in range(days - 1, -1, -1)]


def confidence_summary(rows, now: datetime | None = None) -> dict:
    """Mean confidence overall, and last 7 days vs the 7 before (None when either window is empty)."""
    if not rows:
        return dict(avg=None, delta_pts=None, recent_avg=None, prior_avg=None, basis="no detections yet")
    now = now or datetime.now(timezone.utc)
    recent, prior = [], []
    for r in rows:
        ts = _parse_ts(r["detected_at"])
        if ts is None:
            continue
        age = now - ts
        if age <= timedelta(days=7):
            recent.append(r["confidence_score"])
        elif age <= timedelta(days=14):
            prior.append(r["confidence_score"])
    avg = sum(r["confidence_score"] for r in rows) / len(rows)
    if not recent or not prior:
        return dict(avg=avg, delta_pts=None, recent_avg=None, prior_avg=None,
                    basis="no prior 7-day period to compare")
    ra, pa = sum(recent) / len(recent), sum(prior) / len(prior)
    return dict(avg=avg, delta_pts=(ra - pa) * 100, recent_avg=ra, prior_avg=pa, basis="last 7 days vs prior 7 days")


def channel_distribution(rows, top: int = 5) -> list[dict]:
    counts = Counter(channel_of(r["source_name"]) for r in rows)
    total = sum(counts.values())
    if not total:
        return []
    ranked = counts.most_common()
    out = [dict(name=n, count=c, pct=100.0 * c / total) for n, c in ranked[:top]]
    rest = sum(c for _, c in ranked[top:])
    if rest:
        out.append(dict(name="Other", count=rest, pct=100.0 * rest / total))
    return out


def premium_share(pods) -> dict:
    """% of in-break (multi-ad pod) client airings in First/Last; all brands if nothing is tagged as client."""
    dist = position_distribution(pods)
    scope = [d for d in dist if d["role"] == "self"] or dist
    in_breaks = sum(d["first"] + d["middle"] + d["last"] for d in scope)
    premium = sum(d["first"] + d["last"] for d in scope)
    return dict(pct=100.0 * premium / in_breaks if in_breaks else None, in_break_airings=in_breaks,
                premium_airings=premium,
                scope="client brands" if any(d["role"] == "self" for d in dist) else "all brands")


def clutter_summary(pods) -> dict:
    """Live pods -> by Algeria hour (with peak); recordings only -> per source; else none."""
    live = [p for p in pods if p.is_live]
    if live:
        by_hour: dict[int, list] = {}
        for p in live:
            by_hour.setdefault(datetime.fromtimestamp(p.pod_start, ALGERIA_TZ).hour, []).append(p)
        rows = [dict(hour=h, pods=len(ps), avg_ads=sum(p.total_ads_in_pod for p in ps) / len(ps),
                     avg_seconds=sum(p.pod_duration_seconds for p in ps) / len(ps))
                for h, ps in sorted(by_hour.items())]
        peak = max(rows, key=lambda r: (r["avg_ads"], r["pods"]))
        return dict(mode="hourly", rows=rows, peak=peak)
    if pods:
        rows = [dict(channel=c["channel"], pods=c["pods"], avg_ads=c["avg_ads_per_pod"],
                     avg_seconds=c["avg_pod_seconds"]) for c in channel_clutter(pods)]
        return dict(mode="recordings", rows=rows, peak=max(rows, key=lambda r: r["avg_ads"]))
    return dict(mode="none", rows=[], peak=None)


def industries_by_volume(db_path) -> list[str]:
    conn = init_db(db_path)
    try:
        names = [r[0] for r in conn.execute("SELECT DISTINCT industry FROM reference_ads WHERE industry IS NOT NULL")]
    finally:
        conn.close()
    return sorted(names, key=lambda i: (-compute_share_of_voice(i, db_path=db_path)["total_airings"], i.casefold()))


def overview(db_path, industry: str | None = "__auto__", days: int = 14) -> dict:
    """Everything the executive cards need, in one JSON-serialisable dict."""
    rows = load_airings(db_path)
    industries = industries_by_volume(db_path)
    if industry == "__auto__":
        industry = industries[0] if industries else None
    sov = compute_share_of_voice(industry, db_path=db_path)
    leader = sov["brands"][0] if sov["brands"] else None
    pods = cluster_occurrences_into_pods(db_path=db_path)
    return dict(
        total_airings=len(rows),
        per_day=detections_per_day(rows, days),
        confidence=confidence_summary(rows),
        sov=dict(industry=industry, industries=industries, client_pct=sov["self_sov_airtime_pct"],
                 competitor_pct=sov["competitor_sov_airtime_pct"], total_airings=sov["total_airings"],
                 airtime_min=sov["total_airtime_sec"] / 60,
                 leader=dict(brand=leader["brand"], pct=leader["sov_airtime_pct"]) if leader else None),
        channels=channel_distribution(rows),
        premium=premium_share(pods),
        clutter=clutter_summary(pods),
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
