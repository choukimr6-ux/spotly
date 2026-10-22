"""Phase 21: campaign families (Master 30 s + cut-downs / teasers / tagged edits) and parent-level rollups.

Each edit is its own reference signature and its own `reference_ads` row, so every airing keeps the exact variant
that aired (ad_id, duration_detected) in the audit log. Reporting rolls them up to the Parent Campaign:

    family_map(db_path)        ad_id -> {asset_id, root_asset_id, root_ad_id, parent_campaign, cut_type, nominal}
    campaign_rollup(db_path)   per parent campaign: airings, airtime, SOV share, and the per-variant breakdown
"""
from collections import defaultdict

from spotly.data import reference_db as rdb
from spotly.database.schema import get_db


def family_map(db_path=None) -> dict:
    with get_db(db_path) as conn:
        rdb.ensure_schema(conn)
        rows = conn.execute("SELECT s.asset_id, s.master_asset_id, s.cut_type, s.nominal_duration, s.ad_id, "
                            "a.campaign_name FROM reference_signatures s LEFT JOIN reference_ads a ON a.ad_id = s.ad_id"
                            ).fetchall()
    parent = {r["asset_id"]: r["master_asset_id"] for r in rows}
    by_asset = {r["asset_id"]: r for r in rows}
    out = {}
    for r in rows:
        if r["ad_id"] is None:
            continue
        root, seen = r["asset_id"], {r["asset_id"]}
        while parent.get(root) in by_asset and parent[root] not in seen:
            root = parent[root]
            seen.add(root)
        rr = by_asset[root]
        out[r["ad_id"]] = dict(asset_id=r["asset_id"], root_asset_id=root, root_ad_id=rr["ad_id"],
                               parent_campaign=rr["campaign_name"] or root, cut_type=r["cut_type"] or "MASTER",
                               nominal=float(r["nominal_duration"]))
    return out


def campaign_rollup(db_path=None, *, since_epoch: float | None = None) -> dict:
    """Airings / airtime / SOV per parent campaign with every variant listed under it (live and file airings)."""
    fam = family_map(db_path)
    sql = ("SELECT o.ad_id, o.duration_detected, r.brand_name, r.campaign_name FROM ad_occurrences o "
           "JOIN reference_ads r USING (ad_id) WHERE o.human_verified != -1")
    args = ()
    if since_epoch is not None:
        sql, args = sql + " AND o.airing_start >= ?", (since_epoch,)
    with get_db(db_path) as conn:
        rows = conn.execute(sql, args).fetchall()
    camp = defaultdict(lambda: dict(airings=0, airtime_sec=0.0, variants=defaultdict(
        lambda: dict(airings=0, airtime_sec=0.0))))
    for r in rows:
        f = fam.get(r["ad_id"])
        key = (r["brand_name"], f["parent_campaign"] if f else r["campaign_name"])
        c = camp[key]
        v = c["variants"][(f["asset_id"] if f else f"ad:{r['ad_id']}", f["cut_type"] if f else "MASTER",
                           f["nominal"] if f else None)]
        for d in (c, v):
            d["airings"] += 1
            d["airtime_sec"] += float(r["duration_detected"])
    total = sum(c["airtime_sec"] for c in camp.values())
    out = []
    for (brand, name), c in camp.items():
        out.append(dict(brand=brand, parent_campaign=name, airings=c["airings"], airtime_sec=round(c["airtime_sec"], 3),
                        sov_airtime_pct=100.0 * c["airtime_sec"] / total if total else 0.0,
                        variants=sorted((dict(asset_id=a, cut_type=t, nominal_s=n, airings=v["airings"],
                                              airtime_sec=round(v["airtime_sec"], 3))
                                         for (a, t, n), v in c["variants"].items()),
                                        key=lambda x: (-(x["nominal_s"] or 0), x["asset_id"]))))
    out.sort(key=lambda x: (-x["airtime_sec"], x["brand"], x["parent_campaign"]))
    return dict(total_airings=sum(c["airings"] for c in camp.values()), total_airtime_sec=round(total, 3), campaigns=out)
