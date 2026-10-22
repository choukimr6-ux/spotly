"""Retroactive catalogue consolidation (hotfix 2026-10-21): canonical brands, clean campaign names, campaign families.

    python -m spotly.cli consolidate-catalog [--dry-run]

What it does, in one transaction after an online backup:
  1. brand names -> registry spelling; stitched fragments repaired ('MON' + 'miral NOW EXCELLENCE MON' -> 'Amiral');
  2. campaign names containing internal bookkeeping ('Auto-discovered ELHEDDAF_AD_117') or stitched junk -> '<n> s spot'
     (numbered when a brand has several of that length);
  3. creatives of one brand whose audio is wholly contained in a longer creative of the same brand become CUT_DOWN children of
     it (reference_signatures.master_asset_id / cut_type, the Phase 21 hierarchy), so reports roll them up to the parent
     campaign while every airing row keeps its exact variant;
  4. registry rows get their reference ad (reference_signatures.ad_id) filled in from the promoted asset so the rollup can
     see the families, and their created_at is bumped so a running matcher hot-reloads the new names.

It never merges two different creatives into one and never rewrites an airing row (their SHA-256 seals include ad_id, not the
names), so every previously issued certificate hash stays valid. Different creatives of the same brand (two Dayzy spots that
share no audio) stay separate campaigns under one brand.
"""
import json
import logging
import sqlite3
from datetime import datetime

from spotly.analytics import auto_resolver
from spotly.analytics import brand_extractor as be
from spotly.config import DATA_DIR, DEFAULT_DB_PATH
from spotly.data import reference_db as rdb
from spotly.database.schema import get_db

log = logging.getLogger("spotly.cleanup")


def _default_name(duration: float, taken: set[str]) -> str:
    base = f"{round(float(duration))} s spot"
    name, n = base, 1
    while name in taken:
        n += 1
        name = f"{base} #{n}"
    return name


def plan(db_path=None) -> dict:
    """Everything consolidate_catalog() would change, computed read-only."""
    db_path = db_path or DEFAULT_DB_PATH
    with get_db(db_path) as conn:
        rdb.ensure_schema(conn)
        ads = [dict(r) for r in conn.execute("SELECT ad_id, brand_name, campaign_name, duration_sec FROM reference_ads ORDER BY ad_id")]
        sigs = [dict(r) for r in conn.execute("SELECT asset_id, brand_name, nominal_duration, ad_id, master_asset_id, cut_type "
                                              "FROM reference_signatures ORDER BY asset_id")]
        promoted = {r["asset_id"]: r["promoted_ad_id"] for r in conn.execute(
            "SELECT asset_id, promoted_ad_id FROM discovered_assets WHERE promoted_ad_id IS NOT NULL")}
    for sg in sigs:                                    # registered masters carry their reference ad on the signature itself
        if sg["ad_id"] is not None:
            promoted.setdefault(sg["asset_id"], sg["ad_id"])
    spot = {}                                         # an ad's length = its longest registered creative (not a cut-down's)
    for sg in sigs:
        ad = promoted.get(sg["asset_id"], sg["ad_id"])
        if ad is not None:
            spot[ad] = max(spot.get(ad, 0.0), float(sg["nominal_duration"]))
    ad_changes, taken = {}, {}
    for a in ads:
        a["duration_sec"] = spot.get(a["ad_id"], a["duration_sec"])
        brand, camp = be.repair_stitched_brand(a["brand_name"], a["campaign_name"])
        repaired = brand != be.canonical_brand(a["brand_name"])
        junk = repaired or camp != " ".join((a["campaign_name"] or "").split()) or not camp
        taken.setdefault(brand, set())
        ad_changes[a["ad_id"]] = dict(old_brand=a["brand_name"], old_campaign=a["campaign_name"], brand=brand,
                                      campaign=None if junk else camp, duration=a["duration_sec"])
    for ch in ad_changes.values():                     # names that stay as they are claim their slot first
        if ch["campaign"]:
            taken[ch["brand"]].add(ch["campaign"])
    for ad_id in sorted(ad_changes):
        ch = ad_changes[ad_id]
        if ch["campaign"] is None:
            ch["campaign"] = _default_name(ch["duration"], taken[ch["brand"]])
            taken[ch["brand"]].add(ch["campaign"])
    # families: contained, same brand, strictly shorter -> child of the LONGEST container
    from spotly.analytics.reference_matcher import ReferenceIndex
    brand_of = {s["asset_id"]: ad_changes[promoted.get(s["asset_id"], s["ad_id"])]["brand"] if promoted.get(s["asset_id"], s["ad_id"]) in ad_changes
                else be.canonical_brand(s["brand_name"]) for s in sigs}
    idx = ReferenceIndex.from_db(db_path)
    nominal = {m.asset_id: m.nominal for m in idx.meta}
    families = []
    for i, m in enumerate(idx.meta):
        best = None
        for y, _off in idx.containers(i):
            p = idx.meta[y]
            if brand_of.get(p.asset_id) == brand_of.get(m.asset_id) and p.nominal > m.nominal + 1.0:
                if best is None or p.nominal > nominal[best]:
                    best = p.asset_id
        if best:
            families.append(dict(asset_id=m.asset_id, master=best, cut_type="CUT_DOWN"))
    return dict(ads=ad_changes, families=families, promoted=promoted, brands=brand_of)


def consolidate_catalog(db_path=None, *, dry_run: bool = False) -> dict:
    db_path = db_path or DEFAULT_DB_PATH
    p = plan(db_path)
    report = dict(dry_run=dry_run, backup=None, renamed_ads=[], brand_fixes=[], families=[], registry_updated=0)
    if not dry_run:
        report["backup"] = str(auto_resolver.backup_db(db_path))
    ad_parent = {}                                                   # child ad -> parent ad (for the campaign suffix)
    with get_db(db_path) as real:
        conn = _DryRun() if dry_run else real                        # a dry run executes nothing
        for fam in p["families"]:
            ca, pa = p["promoted"].get(fam["asset_id"]), p["promoted"].get(fam["master"])
            if ca and pa and ca != pa:
                ad_parent[ca] = pa
        names = {ad: (ch["brand"], ch["campaign"]) for ad, ch in p["ads"].items()}
        for child, parent in ad_parent.items():                       # '<parent campaign> [CUT_DOWN]' (Phase 21 convention)
            names[child] = (names[child][0], f"{names[parent][1]} [CUT_DOWN]")
        for ad_id, (brand, camp) in names.items():
            ch = p["ads"][ad_id]
            if (brand, camp) != (ch["old_brand"], ch["old_campaign"]):
                conn.execute("UPDATE reference_ads SET brand_name = ?, campaign_name = ? WHERE ad_id = ?", (brand, camp, ad_id))
                conn.execute("UPDATE discovered_assets SET label = ? WHERE promoted_ad_id = ?", (f"{brand} – {camp}", ad_id))
                rec = dict(ad_id=ad_id, before=f"{ch['old_brand']} / {ch['old_campaign']}", after=f"{brand} / {camp}")
                report["renamed_ads"].append(rec)
                if brand != ch["old_brand"]:
                    report["brand_fixes"].append(rec)
                    conn.execute("UPDATE compliance_alerts SET brand = ? WHERE brand = ?", (brand, ch["old_brand"]))
                    conn.execute("UPDATE reference_ads SET target_competitor_of = ? WHERE target_competitor_of = ?",
                                 (brand, ch["old_brand"]))
        for asset_id, brand in p["brands"].items():                   # registry: brand, reference ad link, hot-reload bump
            conn.execute("UPDATE reference_signatures SET brand_name = ?, ad_id = COALESCE(ad_id, ?), "
                         "created_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') WHERE asset_id = ?",
                         (brand, p["promoted"].get(asset_id), asset_id))
            report["registry_updated"] += 1
        for fam in p["families"]:
            conn.execute("UPDATE reference_signatures SET master_asset_id = ?, cut_type = ? WHERE asset_id = ?",
                         (fam["master"], fam["cut_type"], fam["asset_id"]))
            report["families"].append(fam)
        report["contracts_merged"] = 0 if dry_run else _merge_contracts(conn)
    if not dry_run:
        out = DATA_DIR / "backups" / f"consolidation-{datetime.now():%Y%m%d-%H%M%S}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        report["audit_file"] = str(out)
        log.info("catalogue consolidated: %d ad(s) renamed, %d brand fix(es), %d family link(s)",
                 len(report["renamed_ads"]), len(report["brand_fixes"]), len(report["families"]))
    return report


class _DryRun:
    def execute(self, *a, **k):
        return self


def _merge_contracts(conn) -> int:
    """One contract clause per (brand, campaign): spellings that collapse to the same canonical brand are de-duplicated."""
    try:
        row = conn.execute("SELECT id, json FROM rate_card_overrides WHERE id = 1").fetchone()
    except sqlite3.OperationalError:
        return 0
    if not row:
        return 0
    doc = json.loads(row["json"])
    seen, out = set(), []
    for c in doc.get("contracts", []):
        c = {**c, "brand": be.canonical_brand(c.get("brand"))}
        key = (c["brand"], c.get("campaign", "*"))
        if key not in seen:
            seen.add(key)
            out.append(c)
    merged = len(doc.get("contracts", [])) - len(out)
    if merged or any(a != b for a, b in zip(doc.get("contracts", []), out)):
        doc["contracts"] = out
        conn.execute("UPDATE rate_card_overrides SET json = ? WHERE id = 1", (json.dumps(doc, ensure_ascii=False),))
    return merged
