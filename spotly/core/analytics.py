"""Competitive intelligence: Share of Voice, channel placement, dayparts and competitor surges.

Definitions (shown in the UI too):
  * Share of Voice (SOV) = a brand's share of *detected* airtime (and of detected airings)
    among indexed creatives on the channels/recordings Spotly monitors. It is not share of
    spend or of audience; brands without an indexed creative are invisible to it.
  * Airtime = sum of duration_detected. Human-rejected occurrences (human_verified = -1) are
    always excluded; verified_only=True keeps only human-verified ones.
  * Wall-clock filters (date range, time of day, surges) need live occurrences, whose
    airing_start is epoch seconds. File-scan rows only know an offset into a recording, so
    they are excluded whenever a date range is set (counted in `excluded_undated`).
"""
from collections import defaultdict
from datetime import date, datetime, time as dtime, timedelta

from spotly.core.proof import ALGERIA_TZ
from spotly.database.schema import init_db

AIRINGS_SQL = """
SELECT o.occurrence_id, o.ad_id, o.source_name, o.airing_start, o.airing_end, o.duration_detected,
       o.human_verified, r.brand_name, r.campaign_name, r.industry, r.ad_type, r.target_competitor_of
FROM ad_occurrences o JOIN reference_ads r USING (ad_id)
WHERE o.human_verified {review}
"""


def channel_of(source_name: str) -> str:
    """'live:Echorouk TV' -> 'Echorouk TV'; 'Echorouk TV 20:00 · rec.mp4' -> 'Echorouk TV 20:00'."""
    if source_name.startswith("live:"):
        return source_name[5:]
    return source_name.split(" · ", 1)[0]


def _bound(value, end: bool) -> float | None:
    """date / datetime / ISO string -> epoch seconds. Bare dates are Algeria calendar days; date_to is inclusive."""
    if value is None or value == "":
        return None
    if isinstance(value, str):
        value = datetime.fromisoformat(value) if "T" in value or ":" in value else date.fromisoformat(value)
    if isinstance(value, datetime):
        dt = value if value.tzinfo else value.replace(tzinfo=ALGERIA_TZ)
        return dt.timestamp()
    day = datetime.combine(value, dtime.min, ALGERIA_TZ)
    return (day + timedelta(days=1)).timestamp() if end else day.timestamp()


def load_airings(industry=None, channel=None, date_from=None, date_to=None, *, db_path=None,
                 verified_only: bool = False, client_brand: str | None = None) -> tuple[list[dict], int]:
    """Filtered airings plus the number of file-scan rows dropped because a date range was set."""
    t0, t1 = _bound(date_from, end=False), _bound(date_to, end=True)
    dated = t0 is not None or t1 is not None
    want_ind = industry.strip().casefold() if industry else None
    want_ch = channel.strip().casefold() if channel else None
    client = client_brand.strip().casefold() if client_brand else None
    conn = init_db(db_path)
    try:
        cur = conn.execute(AIRINGS_SQL.format(review="= 1" if verified_only else "!= -1"))
        rows, undated = [], 0
        for r in cur:
            if want_ind and (r["industry"] or "").strip().casefold() != want_ind:
                continue
            ch = channel_of(r["source_name"])
            if want_ch and ch.casefold() != want_ch:
                continue
            ad_type = r["ad_type"] or "self"
            if client is not None:
                if ad_type == "self" and r["brand_name"].casefold() != client:
                    continue
                if ad_type == "competitor" and r["target_competitor_of"] \
                        and r["target_competitor_of"].casefold() != client:
                    continue
            live = r["source_name"].startswith("live:")
            if dated:
                if not live:
                    undated += 1
                    continue
                if (t0 is not None and r["airing_start"] < t0) or (t1 is not None and r["airing_start"] >= t1):
                    continue
            rows.append(dict(r) | dict(channel=ch, live=live, ad_type=ad_type,
                                       epoch=r["airing_start"] if live else None))
    finally:
        conn.close()
    return rows, undated


def _brand_roles(rows) -> dict:
    roles = defaultdict(set)
    for r in rows:
        roles[r["brand_name"]].add(r["ad_type"])
    return {b: ("self" if "self" in t else "competitor") for b, t in roles.items()}


def compute_share_of_voice(industry=None, channel=None, date_from=None, date_to=None, *, db_path=None,
                           verified_only: bool = False, client_brand: str | None = None) -> dict:
    """Per-brand airings, airtime and % SOV (by airtime and by airings) within an industry."""
    rows, undated = load_airings(industry, channel, date_from, date_to, db_path=db_path,
                                 verified_only=verified_only, client_brand=client_brand)
    agg = defaultdict(lambda: dict(airings=0, airtime_sec=0.0, channels=set(), campaigns=set()))
    for r in rows:
        a = agg[r["brand_name"]]
        a["airings"] += 1
        a["airtime_sec"] += float(r["duration_detected"])
        a["channels"].add(r["channel"])
        a["campaigns"].add(r["campaign_name"])
    total_air = sum(a["airtime_sec"] for a in agg.values())
    total_n = sum(a["airings"] for a in agg.values())
    roles = _brand_roles(rows)
    brands = [dict(brand=b, role=roles[b], airings=a["airings"], airtime_sec=round(a["airtime_sec"], 3),
                   sov_airtime_pct=100.0 * a["airtime_sec"] / total_air if total_air else 0.0,
                   sov_airings_pct=100.0 * a["airings"] / total_n if total_n else 0.0,
                   channels=sorted(a["channels"]), campaigns=sorted(a["campaigns"]))
              for b, a in agg.items()]
    brands.sort(key=lambda x: (-x["airtime_sec"], x["brand"]))
    return dict(industry=industry, channel=channel, total_airings=total_n, total_airtime_sec=round(total_air, 3),
                brands=brands, excluded_undated=undated,
                self_sov_airtime_pct=sum(b["sov_airtime_pct"] for b in brands if b["role"] == "self"),
                competitor_sov_airtime_pct=sum(b["sov_airtime_pct"] for b in brands if b["role"] == "competitor"))


def channel_placement(industry=None, date_from=None, date_to=None, **kwargs) -> list[dict]:
    """Where each brand's airtime runs: per brand x channel airtime and % of that brand's airtime."""
    rows, _ = load_airings(industry, None, date_from, date_to, **kwargs)
    roles = _brand_roles(rows)
    per = defaultdict(lambda: [0, 0.0])
    for r in rows:
        p = per[(r["brand_name"], r["channel"])]
        p[0] += 1
        p[1] += float(r["duration_detected"])
    brand_total = defaultdict(float)
    for (b, _), (_, sec) in per.items():
        brand_total[b] += sec
    return sorted((dict(brand=b, role=roles[b], channel=ch, airings=n, airtime_sec=round(sec, 3),
                        share_of_brand_pct=100.0 * sec / brand_total[b] if brand_total[b] else 0.0)
                   for (b, ch), (n, sec) in per.items()), key=lambda x: (x["brand"], -x["airtime_sec"]))


def daypart_distribution(industry=None, date_from=None, date_to=None, **kwargs) -> list[dict]:
    """Airings per brand per hour of day (Algeria time). Live occurrences only (file scans have no clock)."""
    rows, _ = load_airings(industry, None, date_from, date_to, **kwargs)
    roles = _brand_roles(rows)
    per = defaultdict(int)
    for r in rows:
        if r["epoch"] is not None:
            per[(r["brand_name"], datetime.fromtimestamp(r["epoch"], ALGERIA_TZ).hour)] += 1
    return sorted((dict(brand=b, role=roles[b], hour=h, airings=n) for (b, h), n in per.items()),
                  key=lambda x: (x["brand"], x["hour"]))


def detect_competitor_surges(window_hours: float = 24, *, now: datetime | float | None = None, db_path=None,
                             industry=None, threshold: float = 0.30, min_airings: int = 3) -> list[dict]:
    """Competitor brands whose airings in the last window exceed the previous window by > threshold.

    A surge needs at least `min_airings` in the current window, so tiny counts (1 -> 2 = +100%)
    don't raise alarms. A brand with none in the previous window but >= min_airings now is
    reported as new activity (change_pct = None).
    """
    now_ts = (now.timestamp() if isinstance(now, datetime) else float(now)) if now is not None \
        else datetime.now(ALGERIA_TZ).timestamp()
    w = window_hours * 3600.0
    rows, _ = load_airings(industry, db_path=db_path)
    cur, prev = defaultdict(lambda: [0, 0.0]), defaultdict(lambda: [0, 0.0])
    meta = {}
    for r in rows:
        if r["ad_type"] != "competitor" or r["epoch"] is None:
            continue
        e = r["epoch"]
        bucket = cur if now_ts - w < e <= now_ts else prev if now_ts - 2 * w < e <= now_ts - w else None
        if bucket is None:
            continue
        bucket[r["brand_name"]][0] += 1
        bucket[r["brand_name"]][1] += float(r["duration_detected"])
        meta[r["brand_name"]] = (r["industry"], r["target_competitor_of"])
    surges = []
    for brand, (n_cur, sec_cur) in cur.items():
        n_prev, sec_prev = prev.get(brand, (0, 0.0))
        if n_cur < min_airings:
            continue
        change = (n_cur - n_prev) / n_prev if n_prev else None
        if change is None or change > threshold:
            ind, target = meta[brand]
            surges.append(dict(brand=brand, industry=ind, target_competitor_of=target, window_hours=window_hours,
                               current_airings=n_cur, previous_airings=n_prev,
                               change_pct=None if change is None else round(100 * change, 1),
                               current_airtime_sec=round(sec_cur, 3), previous_airtime_sec=round(sec_prev, 3),
                               kind="new activity" if change is None else "surge"))
    surges.sort(key=lambda s: (-(s["change_pct"] or float("inf")), s["brand"]))
    return surges
