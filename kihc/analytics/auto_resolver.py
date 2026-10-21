"""Zero-input brand resolution of discovered assets (Phase 11.5).

    python -m kihc.analytics.auto_resolver --db kihc/data/kihc.db            # dry run: what would change
    python -m kihc.analytics.auto_resolver --db kihc/data/kihc.db --apply    # back up, resolve, promote, re-audit

For every *unclassified* asset the stored packshot OCR text and Whisper transcript are fused
(brand_extractor.resolve_brand). A result at or above CONFIRM_MIN is promoted to a classified asset with its
canonical brand and sector, no click needed; weaker results are stored as suggestions. Assets whose speech was
never transcribed are not treated as silent - OCR alone stays a suggestion until the transcript exists.

Confirmed brands that have no contract yet receive the standard clauses (180 s competitor separation with a
100 % credit; first-or-last-in-pod protection with a 30 % credit). These are DEFAULT TERMS, tagged
"auto-default" in the contract list, not signed ones: the credits they produce are claimable only after a person
compares them with the real agreement and the broadcaster's as-run log. A contract entered by a human is never
overwritten.

Pure lookups and SQLite: no model is loaded, so the pass adds a few MB over the host process.
"""
import argparse
import json
import re
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

from kihc.analytics.brand_extractor import CONFIRM_MIN, resolve_brand
from kihc.config import DATA_DIR, DEFAULT_DB_PATH
from kihc.database.schema import get_db

AUTO_CAMPAIGN = "Auto-resolved"
DEFAULT_CLAUSES = dict(campaign="*", competitor_separation_seconds=180, separation_credit=1.0,
                       booked_position="FIRST_OR_LAST", pod_edge_credit=0.30,
                       source="auto-default (assumed standard terms, not a signed contract)")


def backup_db(db_path, dest_dir: Path | None = None) -> Path:
    """Consistent copy of the live (WAL) database via SQLite's online backup API."""
    dest_dir = Path(dest_dir or DATA_DIR / "backups")
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"kihc-pre-resolve-{datetime.now():%Y%m%d-%H%M%S}.db"
    src, dst = sqlite3.connect(str(db_path)), sqlite3.connect(str(dest))
    try:
        src.backup(dst)
    finally:
        dst.close()
        src.close()
    return dest


def _candidates(db_path) -> list[dict]:
    with get_db(db_path) as conn:
        return [dict(r) for r in conn.execute(
            "SELECT asset_id, channel, transcript, transcribed_at, ocr_text, content_hint, visual_status, audio_path "
            "FROM discovered_assets WHERE status = 'unclassified' ORDER BY asset_id")]


def _store_suggestion(conn, asset_id: str, res) -> None:
    ev = json.dumps([dict(brand=res.brand, confidence=res.confidence, basis=res.basis, source="fusion",
                          sources=res.evidence)], ensure_ascii=False)
    conn.execute("UPDATE discovered_assets SET suggested_brand = ?, suggested_industry = ?, brand_confidence = ?, "
                 "brand_evidence = ?, suggestion_source = 'fusion' WHERE asset_id = ? AND status = 'unclassified'",
                 (res.brand, res.sector or res.industry, res.confidence, ev, asset_id))


def ensure_contracts(db_path, brands) -> list[str]:
    """Add the default clauses for brands without any contract. -> brands that received them."""
    from kihc.analytics.compliance_audit import get_contracts, save_contracts
    contracts = list(get_contracts(db_path))
    have = {str(c.get("brand", "")).casefold() for c in contracts}
    added = []
    for brand in sorted(set(brands)):
        if brand.casefold() not in have:
            contracts.append(dict(brand=brand, **DEFAULT_CLAUSES))
            added.append(brand)
    if added:
        save_contracts(db_path, contracts)
    return added


def _reclaimable(db_path):
    from kihc.analytics.compliance_audit import compliance_report
    k = compliance_report(db_path)["kpis"]
    return dict(reclaimable_net=str(k["reclaimable_net"]), position_credit=str(k["position_credit"]),
                separation_credit=str(k["separation_credit"]), breaches=k["breaches"], airings=k["airings"])


def resolve_pending(db_path=None, *, apply: bool = True, promote: bool = True, contracts: bool = True,
                    backup: bool = True) -> dict:
    """Fuse OCR + speech for every unclassified asset. With apply=False nothing is written."""
    from kihc.core.discovery import promote_asset
    db_path = str(db_path or DEFAULT_DB_PATH)
    started = time.perf_counter()
    todo = _candidates(db_path)
    backup_path = backup_db(db_path) if apply and backup and todo else None
    before = _reclaimable(db_path) if apply and promote else None
    decisions, confirmed_brands = [], []
    for a in todo:
        res = resolve_brand(a["ocr_text"], a["transcript"], speech_checked=bool(a["transcribed_at"]),
                            programme=a["visual_status"] == "programme",
                            station_promo=bool(a["content_hint"] and a["content_hint"].startswith("station self-promo")))
        if not res.brand:
            continue
        d = dict(asset_id=a["asset_id"], brand=res.brand, sector=res.sector, confidence=res.confidence,
                 status=res.status, basis=res.basis, promoted=False)
        if apply:
            with get_db(db_path) as conn:
                _store_suggestion(conn, a["asset_id"], res)
            if res.status == "CONFIRMED" and promote:
                try:
                    out = promote_asset(db_path, a["asset_id"], res.brand, "",
                                        industry=res.sector or res.industry)
                    d.update(promoted=True, ad_id=out["ad_id"], occurrences=out["occurrences_backfilled"])
                    confirmed_brands.append(res.brand)
                except Exception as exc:          # one asset without audio must not stop the pass
                    d["error"] = f"{type(exc).__name__}: {exc}"
        decisions.append(d)
    added = ensure_contracts(db_path, confirmed_brands) if apply and contracts and confirmed_brands else []
    after = _reclaimable(db_path) if apply and promote else None
    return dict(applied=apply, scanned=len(todo), resolved=len(decisions),
                confirmed=sum(d["status"] == "CONFIRMED" for d in decisions),
                promoted=sum(d["promoted"] for d in decisions),
                suggested=sum(d["status"] == "SUGGESTED" for d in decisions),
                contracts_added=added, reclaimable_before=before, reclaimable_after=after,
                backup=str(backup_path) if backup_path else None, decisions=decisions,
                seconds=round(time.perf_counter() - started, 1))


# ------------------------------------------------------------------------------- Phase 24: unconfirmed airings
UNCONFIRMED_STATUSES = ("UNCONFIRMED_CUTDOWN_NO_DIP", "UNCONFIRMED_MUSIC_NO_PACKSHOT")
DEFINITIVE_MAX_BITS = 8          # packshot pHash Hamming distance (of 64): same packshot, no further proof needed
CROSSCHECK_MAX_BITS = 12         # 9-12 bits: only with an exact acoustic alignment (the matcher's own accept confidence)


def _recorded_phash(row) -> str | None:
    """pHash of the airing's recorded closing packshot: the evidence frame file (one image in memory at a time), else a hash
    stored with the boundary record. None when nothing was recorded."""
    path = row["evidence_frame_path"]
    if path and Path(path).is_file():
        try:
            from PIL import Image

            from kihc.analytics.reference_matcher import phash_image
            with Image.open(path) as im:
                return phash_image(im.convert("RGB"))
        except (OSError, ValueError):
            pass
    try:
        h = (json.loads(row["boundary_json"]) if row["boundary_json"] else {}).get("packshot_phash")
    except (TypeError, ValueError, AttributeError):
        h = None
    return str(h) if h and re.fullmatch(r"[0-9a-fA-F]{16}", str(h)) else None


def _reference_hashes(conn, ad_id) -> list[tuple[str, str]]:
    """(asset_id, visual_p_hash) of the ad's reference signature(s) and of their master creative."""
    try:
        rows = conn.execute("SELECT asset_id, master_asset_id, visual_p_hash FROM reference_signatures WHERE ad_id = ?",
                            (ad_id,)).fetchall()
    except sqlite3.OperationalError:                      # registry not built on this database: nothing to compare with
        return []
    out = {r["asset_id"]: r["visual_p_hash"] for r in rows if r["visual_p_hash"]}
    for r in rows:
        if r["master_asset_id"] and r["master_asset_id"] not in out:
            m = conn.execute("SELECT visual_p_hash FROM reference_signatures WHERE asset_id = ?",
                             (r["master_asset_id"],)).fetchone()
            if m and m["visual_p_hash"]:
                out[r["master_asset_id"]] = m["visual_p_hash"]
    return list(out.items())


def judge(distance: int | None, acoustic_confidence: float | None) -> tuple[bool, str]:
    """(resolve?, reason). <= 8 bits is definitive; 9-12 needs the acoustic cross-check; anything else stays unconfirmed."""
    from kihc.analytics.reference_matcher import ACCEPT_CONF
    if distance is None:
        return False, "no recorded packshot / reference hash to compare"
    if distance <= DEFINITIVE_MAX_BITS:
        return True, f"packshot match {distance}/64 bits (definitive)"
    if distance <= CROSSCHECK_MAX_BITS:
        if acoustic_confidence is not None and acoustic_confidence >= ACCEPT_CONF:
            return True, f"packshot {distance}/64 bits + acoustic alignment {acoustic_confidence:.2f} (cross-checked)"
        return False, f"packshot {distance}/64 bits needs acoustic alignment >= {ACCEPT_CONF} (have {acoustic_confidence})"
    return False, f"packshot differs ({distance}/64 bits)"


def _status_counts(db_path) -> dict:
    with get_db(db_path) as conn:
        return {r[0]: r[1] for r in conn.execute("SELECT status, COUNT(*) FROM ad_occurrences GROUP BY status")}


def resolve_unconfirmed(db_path=None, *, apply: bool = True, backup: bool = True) -> dict:
    """Promote UNCONFIRMED_CUTDOWN_NO_DIP / UNCONFIRMED_MUSIC_NO_PACKSHOT airings whose recorded closing packshot matches a
    reference signature of the same ad. Each promotion is one guarded UPDATE (only while still unconfirmed), so a re-run or a
    concurrent matcher cannot double-apply; the compliance alerts are then recomputed so the credit reaches the ledger."""
    from kihc.analytics.compliance_audit import refresh_alerts
    from kihc.analytics.reference_matcher import hamming_hex
    db_path = str(db_path or DEFAULT_DB_PATH)
    started = time.perf_counter()
    marks = ",".join("?" * len(UNCONFIRMED_STATUSES))
    with get_db(db_path) as conn:
        todo = [dict(r) for r in conn.execute(
            f"SELECT occurrence_id, ad_id, status, confidence_score, evidence_frame_path, boundary_json "
            f"FROM ad_occurrences WHERE status IN ({marks}) ORDER BY occurrence_id", UNCONFIRMED_STATUSES)]
    before = dict(statuses=_status_counts(db_path), **_reclaimable(db_path))
    backup_path = backup_db(db_path) if apply and backup and todo else None
    decisions, resolved = [], []
    for row in todo:                                          # one candidate at a time: no image batch in RAM
        with get_db(db_path) as conn:
            refs = _reference_hashes(conn, row["ad_id"])
        mine = _recorded_phash(row)
        best = min(((hamming_hex(mine, h), a) for a, h in refs), default=None) if mine else None
        ok, why = judge(best[0] if best else None, row["confidence_score"])
        d = dict(occurrence_id=row["occurrence_id"], ad_id=row["ad_id"], from_status=row["status"],
                 distance=best[0] if best else None, reference_asset=best[1] if best else None, resolved=ok, reason=why)
        if ok and apply:
            try:
                meta = json.loads(row["boundary_json"]) if row["boundary_json"] else {}
                meta = meta if isinstance(meta, dict) else {}
            except ValueError:
                meta = {}
            meta["resolution"] = dict(resolved_by="auto_resolver.resolve_unconfirmed",
                                      resolved_at=datetime.now().isoformat(timespec="seconds"),
                                      previous_status=row["status"], packshot_distance=best[0],
                                      reference_asset=best[1], reason=why)
            with get_db(db_path) as conn:                     # one transaction; busy_timeout/WAL are set by the connection
                n = conn.execute(f"UPDATE ad_occurrences SET status = 'CONFIRMED', boundary_json = ? "
                                 f"WHERE occurrence_id = ? AND status IN ({marks})",
                                 (json.dumps(meta, ensure_ascii=False), row["occurrence_id"],
                                  *UNCONFIRMED_STATUSES)).rowcount
            d["resolved"] = bool(n)
            if n:
                resolved.append(row["occurrence_id"])
        decisions.append(d)
    alerts = 0
    if resolved:                                              # re-audit from the oldest resolved airing so its credit lands
        with get_db(db_path) as conn:
            oldest = conn.execute(f"SELECT MIN(airing_start) FROM ad_occurrences WHERE occurrence_id IN "
                                  f"({','.join('?' * len(resolved))})", resolved).fetchone()[0]
        days = max(2, min(3650, int((time.time() - oldest) // 86400) + 2))
        alerts = refresh_alerts(db_path, lookback_days=days)
    after = dict(statuses=_status_counts(db_path), **_reclaimable(db_path))
    return dict(applied=apply, scanned=len(todo), resolved=sum(d["resolved"] for d in decisions), alerts_refreshed=alerts,
                before=before, after=after, decisions=decisions, backup=str(backup_path) if backup_path else None,
                seconds=round(time.perf_counter() - started, 2))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=None)
    ap.add_argument("--apply", action="store_true", help="write results (default: dry run)")
    ap.add_argument("--no-promote", action="store_true", help="store suggestions only; do not classify")
    ap.add_argument("--no-contracts", action="store_true", help="do not add the default contract clauses")
    a = ap.parse_args(argv)
    r = resolve_pending(a.db, apply=a.apply, promote=not a.no_promote, contracts=not a.no_contracts)
    for d in r["decisions"]:
        tail = "PROMOTED" if d["promoted"] else d.get("error", "")
        print(f"{d['asset_id']}: {d['brand']} [{d['sector']}] {d['confidence']:.2f} {d['status']} - {d['basis']} {tail}")
    print(f"auto_resolver ({'applied' if r['applied'] else 'dry run'}): scanned {r['scanned']}, resolved {r['resolved']}, "
          f"confirmed {r['confirmed']}, promoted {r['promoted']}, suggestions {r['suggested']} "
          f"(threshold {CONFIRM_MIN:.2f}) in {r['seconds']}s")
    if r["contracts_added"]:
        print("default contracts added for:", ", ".join(r["contracts_added"]))
    if r["reclaimable_before"]:
        print("reclaimable (net DZD):", r["reclaimable_before"]["reclaimable_net"], "->", r["reclaimable_after"]["reclaimable_net"])
    if r["backup"]:
        print("backup:", r["backup"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
