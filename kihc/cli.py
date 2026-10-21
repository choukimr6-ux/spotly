"""KIHC command line.

  python -m kihc.cli ingest <file> --brand B --campaign C
  python -m kihc.cli scan <file-or-url> [--source-name NAME] [--threshold 0.82]
  python -m kihc.cli export [--out kihc/reports/occurrences.csv]
  python -m kihc.cli demo        # synth assets -> ingest -> scan, into the default DB
"""
import argparse
import csv
from datetime import datetime, timezone

from kihc.config import DEFAULT_DB_PATH, MATCH_THRESHOLD, REPORTS_DIR
from kihc.core.ingest import ingest_reference
from kihc.core.matcher import scan_source
from kihc.database.schema import get_db

EXPORT_SQL = """
SELECT o.occurrence_id, o.ad_id, r.brand_name, r.campaign_name, o.source_name, o.airing_start, o.airing_end,
       o.duration_detected, o.confidence_score, o.human_verified, o.evidence_frame_path, o.detected_at,
       o.algeria_timestamp, o.verification_hash
FROM ad_occurrences o JOIN reference_ads r USING (ad_id)
ORDER BY o.source_name, o.airing_start
"""


def _print_detections(dets):
    if not dets:
        print("No occurrences at or above threshold.")
    for d in dets:
        print(f"  {d.brand_name} / {d.campaign_name}: {d.airing_start:.3f}s -> {d.airing_end:.3f}s "
              f"conf={d.confidence_score:.4f} evidence={d.evidence_frame_path}")


def export_csv(db_path=None, out=None):
    out = out or REPORTS_DIR / f"occurrences_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with get_db(db_path) as conn:
        rows = conn.execute(EXPORT_SQL).fetchall()
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(rows[0].keys() if rows else [])
        w.writerows([tuple(r) for r in rows])
    return out, len(rows)


def _tag_args(parser):
    parser.add_argument("--industry", help="e.g. Telecom, Dairy, Electronics")
    parser.add_argument("--role", choices=["self", "competitor"], default="self",
                        help="self = client brand (default), competitor = competitor ad")
    parser.add_argument("--competes-with", help="client brand this competitor ad competes with")


def main(argv=None):
    p = argparse.ArgumentParser(prog="kihc")
    p.add_argument("--db", default=str(DEFAULT_DB_PATH))
    sub = p.add_subparsers(dest="cmd", required=True)
    pi = sub.add_parser("ingest")
    pi.add_argument("file")
    pi.add_argument("--brand", required=True)
    pi.add_argument("--campaign", required=True)
    pi.add_argument("--expected-duration", type=float, help="seconds; warns if the audio length differs")
    pi.add_argument("--allow-duplicate", action="store_true",
                    help="index even if it matches an existing reference (risks double-counting airings)")
    _tag_args(pi)
    pt = sub.add_parser("tag-ad", help="set industry / role tags on an indexed reference ad")
    pt.add_argument("ad_id", type=int)
    _tag_args(pt)
    pv = sub.add_parser("sov", help="Share of Voice report for an industry")
    pv.add_argument("--industry")
    pv.add_argument("--channel")
    pv.add_argument("--from", dest="date_from", help="YYYY-MM-DD (Algeria time, live airings only)")
    pv.add_argument("--to", dest="date_to", help="YYYY-MM-DD inclusive")
    pv.add_argument("--client", help="limit to this client brand and its competitors")
    pv.add_argument("--verified-only", action="store_true")
    pp = sub.add_parser("pods", help="reconstruct commercial breaks (pods), positions and clutter")
    pp.add_argument("--channel", help="channel name (live) or recording source name; default all")
    pp.add_argument("--date", help="Algeria calendar day YYYY-MM-DD (live airings); default everything")
    pp.add_argument("--max-gap", type=float, default=12.0, help="max seconds between ads in one pod")
    pp.add_argument("--materialize", action="store_true",
                    help="also cache pod_id / pod_position / pod_density on ad_occurrences")
    pg = sub.add_parser("surges", help="competitor volume surges vs the previous window")
    pg.add_argument("--window-hours", type=float, default=24)
    pg.add_argument("--industry")
    ps = sub.add_parser("scan")
    ps.add_argument("source")
    ps.add_argument("--source-name")
    ps.add_argument("--threshold", type=float, default=MATCH_THRESHOLD)
    pe = sub.add_parser("export")
    pe.add_argument("--out")
    sub.add_parser("demo")
    prs = sub.add_parser("resolve-unconfirmed-assets",
                         help="confirm UNCONFIRMED_* airings whose closing packshot matches the reference (<= 8 bits; 9-12 "
                              "with acoustic cross-check) and recompute the compliance ledger")
    prs.add_argument("--dry-run", action="store_true", help="report what would be confirmed; write nothing")
    prs.add_argument("--db", default=argparse.SUPPRESS)
    pr = sub.add_parser("run-service", help="24/7 headless live monitoring (no Streamlit)")
    pr.add_argument("--channel", action="append", dest="channels",
                    help="national key (el_heddaf_tv, echorouk_tv, …), IPTV channel name, or 'Name=URL'; "
                         "repeatable. Default: data/service_channels.json")
    pr.add_argument("--config", default=None, help="JSON file with {\"channels\": [...]}")
    pr.add_argument("--tick", type=float, default=5.0, help="supervisor interval (s)")
    pr.add_argument("--freeze-after", type=float, default=20.0, help="restart a stream after N s without audio data")
    pr.add_argument("--connect-timeout", type=float, default=45.0,
                    help="restart a stream stuck connecting (no first audio) after N s")
    pr.add_argument("--gc-every", type=float, default=3600.0, help="gc.collect() interval (s)")
    pr.add_argument("--verbose", action="store_true")
    pr.add_argument("--db", default=argparse.SUPPRESS)
    pst = sub.add_parser("stop-service", help="ask the running service to shut down gracefully")
    pst.add_argument("--db", default=argparse.SUPPRESS)
    pst.add_argument("--wait", type=float, default=30.0, help="seconds to wait for confirmation")
    pp2 = sub.add_parser("pilot-run", help="background ingest + discovery and the web dashboard in one process")
    pp2.add_argument("--channel", default="elheddaf", help="elheddaf (direct web feed), a national key, or 'Name=URL'")
    pp2.add_argument("--port", type=int, default=8502)
    pp2.add_argument("--host", default="127.0.0.1")
    pp2.add_argument("--history-hours", type=float, default=4.0, help="discovery look-back window")
    pp2.add_argument("--gc-every", type=float, default=1800.0, help="gc.collect() interval (s)")
    pp2.add_argument("--db", default=argparse.SUPPRESS)
    pp2.add_argument("--transcribe", choices=["on", "off"], default="on",
                     help="background speech-to-text of discovered assets (separate process; off = strict RAM)")
    pp2.add_argument("--transcribe-model", default="base", help="faster-whisper model: tiny | base | small")
    pp2.add_argument("--transcribe-memory-cap-mb", type=float, default=480.0)
    pp2.add_argument("--packshots", choices=["on", "off"], default="on",
                     help="live packshot capture (one ~3 MB video segment per grab) + on-screen text OCR")
    pp2.add_argument("--ocr-memory-cap-mb", type=float, default=300.0)
    pp2.add_argument("--frames", choices=["on", "off"], default="on",
                     help="El-Heddaf 2 fps / 480p video frames for multi-cue pod boundary sensing (default on)")
    pp2.add_argument("--reference-matcher", choices=["on", "off"], default="on",
                     help="exact acoustic reference matcher (needs build-reference-registry; falls back to discovery)")
    pdd = sub.add_parser("dedupe-assets", help="find duplicate discovered assets (dry run unless --apply)")
    pdd.add_argument("--apply", action="store_true", help="merge same-airing / same-creative duplicates")
    pdd.add_argument("--db", default=argparse.SUPPRESS)
    pum = sub.add_parser("unmerge-asset", help="undo the merge of a duplicate asset")
    pum.add_argument("asset_id")
    pum.add_argument("--db", default=argparse.SUPPRESS)
    pocr = sub.add_parser("ocr-packshots", help="read on-screen text of captured packshots (one-off batch)")
    pocr.add_argument("--retry", action="store_true")
    pocr.add_argument("--memory-cap-mb", type=float, default=300.0)
    pocr.add_argument("--db", default=argparse.SUPPRESS)
    pt2 = sub.add_parser("transcribe-assets", help="transcribe discovered assets and suggest brands (one-off batch)")
    pt2.add_argument("--model", default="base")
    pt2.add_argument("--asset", action="append", dest="assets")
    pt2.add_argument("--retranscribe", action="store_true")
    pt2.add_argument("--reanalyse", action="store_true",
                     help="only re-score stored transcripts/OCR text against the brand directory (no Whisper)")
    pt2.add_argument("--memory-cap-mb", type=float, default=480.0)
    pt2.add_argument("--db", default=argparse.SUPPRESS)
    prr = sub.add_parser("build-reference-registry",
                         help="build exact-match acoustic signatures from the confirmed (classified) assets")
    prr.add_argument("--dry-run", action="store_true", help="report only; write nothing")
    prr.add_argument("--db", default=argparse.SUPPRESS)
    pcc = sub.add_parser("consolidate-catalog", help="canonical brands, clean campaign names and campaign families for the "
                                                       "existing catalogue (backs the database up first)")
    pcc.add_argument("--dry-run", action="store_true", help="show what would change, change nothing")
    pcc.add_argument("--db", default=argparse.SUPPRESS)
    pra = sub.add_parser("register-ad", help="onboard a master creative (.mp4/.wav/.mp3) before it airs; the live "
                                              "monitor hot-reloads it with no restart")
    pra.add_argument("--file", required=True, help="path to the master media file")
    pra.add_argument("--brand", required=True)
    pra.add_argument("--sector", required=True)
    pra.add_argument("--campaign", default=None, help="default: 'Master <file name>'")
    pra.add_argument("--separation-window", type=float, default=180.0, help="competitor separation, seconds (default 180)")
    pra.add_argument("--position", default="ANY", choices=["FIRST", "LAST", "FIRST_OR_LAST", "ANY"],
                     help="booked pod position (default ANY = none booked)")
    pra.add_argument("--rate-card-override", type=float, default=None, metavar="DZD",
                     help="negotiated 30 s rate in DZD replacing the daypart card for this brand/campaign")
    pra.add_argument("--parent-id", default=None, metavar="ASSET_ID",
                     help="register this file as an edit of an already registered master (rolls up to its campaign)")
    pra.add_argument("--cut-type", default=None, choices=["MASTER", "CUT_DOWN", "TEASER", "TAGGED"],
                     help="how this file relates to its parent (default CUT_DOWN with --parent-id, else MASTER)")
    pra.add_argument("--music-dominant", default="auto", choices=["auto", "on", "off"],
                     help="music-dominant creatives (voice-over < 20 %%) need a packshot match before an airing is CONFIRMED; "
                          "auto measures it, on/off overrides the measurement")
    pra.add_argument("--allow-duplicate", action="store_true")
    pra.add_argument("--db", default=argparse.SUPPRESS)
    pw = sub.add_parser("web", help="FastAPI + bento web app (default http://127.0.0.1:8502)")
    pw.add_argument("--host", default="127.0.0.1")
    pw.add_argument("--port", type=int, default=8502)
    pw.add_argument("--db", default=argparse.SUPPRESS)
    pss = sub.add_parser("service-status")
    pss.add_argument("--db", default=argparse.SUPPRESS)
    a = p.parse_args(argv)

    if a.cmd == "ingest":
        from kihc.core.library import ingest_media
        res = ingest_media(a.file, a.brand, a.campaign, expected_duration=a.expected_duration,
                           db_path=a.db, allow_duplicate=a.allow_duplicate, industry=a.industry,
                           ad_type=a.role, target_competitor_of=a.competes_with)
        for w in res.warnings:
            print(f"WARNING: {w}")
        if res.committed:
            print(f"Indexed ad_id={res.ad_id} ({res.measured_duration:.3f}s)")
        else:
            raise SystemExit(1)
    elif a.cmd == "scan":
        _print_detections(scan_source(a.source, a.source_name, a.db, a.threshold))
    elif a.cmd == "export":
        from pathlib import Path
        out, n = export_csv(a.db, Path(a.out) if a.out else None)
        print(f"Wrote {n} rows to {out}")
    elif a.cmd == "tag-ad":
        from kihc.core.library import set_ad_tags
        set_ad_tags(a.ad_id, a.industry, a.role, a.competes_with, db_path=a.db)
        print(f"Tagged ad_id={a.ad_id}: industry={a.industry!r} role={a.role} competes_with={a.competes_with!r}")
    elif a.cmd == "sov":
        from kihc.core.analytics import compute_share_of_voice
        r = compute_share_of_voice(a.industry, a.channel, a.date_from, a.date_to, db_path=a.db,
                                   verified_only=a.verified_only, client_brand=a.client)
        print(f"Share of Voice · industry={a.industry or 'all'} channel={a.channel or 'all'} · "
              f"{r['total_airings']} airings, {r['total_airtime_sec']:.0f}s detected airtime")
        for b in r["brands"]:
            print(f"  {b['brand']:<24} {b['role']:<10} {b['sov_airtime_pct']:6.2f}% airtime  "
                  f"{b['sov_airings_pct']:6.2f}% airings  ({b['airings']} airings, {b['airtime_sec']:.0f}s)")
        if r["excluded_undated"]:
            print(f"  ({r['excluded_undated']} file-scan airings excluded: no broadcast date)")
    elif a.cmd == "pods":
        from kihc.core.pod_analyzer import cluster_occurrences_into_pods, materialize_pods
        pods = cluster_occurrences_into_pods(a.channel, a.date, a.max_gap, db_path=a.db)
        if not pods:
            print("No pods (no detected airings in this scope).")
        for p in pods:
            print(f"{p.pod_id}  {p.channel}  {p.total_ads_in_pod} ad(s)  {p.pod_duration_seconds:.1f}s  "
                  f"clutter={p.clutter}{'  [overlapping detections]' if p.has_overlap else ''}")
            for ad in p.ads:
                print(f"    {ad['position_of']:>7}  {ad['position_label']:<14} +{ad['offset_in_pod']:7.1f}s  "
                      f"{ad['brand']} – {ad['campaign']} ({ad['duration']:.1f}s)")
        if a.materialize:
            print(f"Cached pod fields on {materialize_pods(a.channel, a.date, a.max_gap, db_path=a.db)} occurrence(s).")
    elif a.cmd == "surges":
        from kihc.core.analytics import detect_competitor_surges
        found = detect_competitor_surges(a.window_hours, db_path=a.db, industry=a.industry)
        if not found:
            print("No competitor surges.")
        for s in found:
            change = "new activity" if s["change_pct"] is None else f"+{s['change_pct']:.0f}%"
            print(f"  {s['brand']}: {s['previous_airings']} -> {s['current_airings']} airings ({change}) "
                  f"in the last {s['window_hours']:g}h")
    elif a.cmd == "resolve-unconfirmed-assets":
        from kihc.analytics.auto_resolver import resolve_unconfirmed
        r = resolve_unconfirmed(getattr(a, "db", None), apply=not a.dry_run)
        for d in r["decisions"]:
            print(f"  occurrence {d['occurrence_id']} ({d['from_status']}): "
                  f"{'CONFIRMED' if d['resolved'] else 'kept'} - {d['reason']}")
        print(f"resolve-unconfirmed-assets ({'dry run' if a.dry_run else 'applied'}): scanned {r['scanned']}, "
              f"resolved {r['resolved']} in {r['seconds']}s")
        print("status counts before:", r["before"]["statuses"], "after:", r["after"]["statuses"])
        print("reclaimable net DZD:", r["before"]["reclaimable_net"], "->", r["after"]["reclaimable_net"],
              "| breaches", r["before"]["breaches"], "->", r["after"]["breaches"])
        if r["backup"]:
            print("backup:", r["backup"])
    elif a.cmd == "run-service":
        from kihc.service.headless_runner import CONFIG_FILE, run_service
        raise SystemExit(run_service(a.channels, a.db, config=a.config or CONFIG_FILE, tick=a.tick,
                                     freeze_after=a.freeze_after, connect_timeout=a.connect_timeout,
                                     gc_every=a.gc_every, verbose=a.verbose))
    elif a.cmd == "stop-service":
        import time
        from kihc.service.headless_runner import read_service_status
        from kihc.service.lock import request_stop
        st = read_service_status(a.db)
        if not st or not st["lock_held"]:
            print("No KIHC service is running for this database.")
            return
        request_stop(a.db)
        print(f"Stop requested (pid {st['pid']}); waiting up to {a.wait:.0f}s…")
        deadline = time.monotonic() + a.wait
        while time.monotonic() < deadline:
            st = read_service_status(a.db)
            if st and st["state"] == "stopped" and not st["lock_held"]:
                print(f"Service stopped: {st['message']}")
                return
            time.sleep(0.5)
        raise SystemExit("Service did not confirm shutdown in time.")
    elif a.cmd == "pilot-run":
        from kihc.service.pilot import run_pilot
        raise SystemExit(run_pilot(a.channel, a.port, a.host, a.db, history_hours=a.history_hours,
                                   gc_every=a.gc_every, transcribe=a.transcribe == "on",
                                   transcribe_model=a.transcribe_model,
                                   transcribe_memory_cap_mb=a.transcribe_memory_cap_mb,
                                   packshots=a.packshots == "on", ocr_memory_cap_mb=a.ocr_memory_cap_mb,
                                   reference_matcher=a.reference_matcher == "on", frames=a.frames == "on"))
    elif a.cmd == "dedupe-assets":
        from kihc.core.discovery import dedupe_assets
        r = dedupe_assets(a.db, apply=a.apply)
        if not r["pairs"]:
            print("No assets share audio.")
        for p in r["pairs"]:
            action = ("MERGE" if a.apply else "would merge") if p["kind"] != "related" else "flag only"
            print(f"  {p['kind']:<14} {p['duplicate']} -> {p['master']}  similarity {p['similarity']:.3f}  "
                  f"coverage {p['coverage']:.0%}  offset {p['offset_sec']:+.1f}s  overlapping airings "
                  f"{len(p['overlapping_airings'])}  [{action}]")
        for m in r["merged"]:
            print(f"Merged {m['duplicate']} into {m['master']}: {m['same_airings']} same airing(s) not re-counted, "
                  f"{m['airings_added']} added -> {m['master_occurrences']} airing(s)")
        if r["pairs"] and not a.apply:
            print("Dry run: nothing changed (use --apply).")
    elif a.cmd == "unmerge-asset":
        from kihc.core.discovery import unmerge_asset
        r = unmerge_asset(a.db, a.asset_id)
        print(f"Unmerged {r['duplicate']} from {r['master']} ({r['airings_removed']} airing(s) removed from the master)")
    elif a.cmd == "build-reference-registry":
        from kihc.analytics.reference_matcher import build_registry
        r = build_registry(a.db, dry_run=a.dry_run)
        for b in r.built:
            print(f"  {b['asset_id']:<17} {b['brand']:<10} {str(b['sector'] or '-'):<18} nominal {b['nominal']:>6.2f}s  "
                  f"trimmed {b['trimmed']:>6.2f}s (lead {b['lead_trim']:.2f}s)  {b['hashes']:>5} hashes  "
                  f"packshot {'pHash ' + b['phash'] if b['phash'] else 'none'}")
        for asset, why in r.skipped:
            print(f"  SKIPPED {asset}: {why}")
        for x, y, conf, kind, off in r.confusable:
            what = {"contained": f"is a cut inside {y} (from {off:.1f}s)", "duplicate": f"is the same audio as {y}",
                    "overlap": f"shares part of its audio with {y}"}[kind]
            print(f"  NOTE {x} {what} ({conf:.0%}); the matcher logs one airing for the pair")
        print(f"{'Would build' if a.dry_run else 'Built'} {len(r.built)} signature(s), {len(r.skipped)} skipped; "
              f"in-memory index {r.index_mb:.2f} MB (budget 50 MB).")
    elif a.cmd == "consolidate-catalog":
        from kihc.analytics.catalog_cleanup import consolidate_catalog
        r = consolidate_catalog(a.db, dry_run=a.dry_run)
        print(("DRY RUN - nothing changed" if a.dry_run else f"backup: {r['backup']}"))
        for x in r["renamed_ads"]:
            print(f"  ad {x['ad_id']:>3}: {x['before']}  ->  {x['after']}")
        for f in r["families"]:
            print(f"  family: {f['asset_id']} is a {f['cut_type']} of {f['master']}")
        print(f"{len(r['renamed_ads'])} ad(s) renamed, {len(r['brand_fixes'])} brand fix(es), {len(r['families'])} family link(s)")
    elif a.cmd == "register-ad":
        from kihc.analytics.ad_onboarding import OnboardingError, register_ad
        try:
            r = register_ad(a.file, a.brand, a.sector, campaign=a.campaign, separation_window=a.separation_window,
                            position=a.position, rate_override=a.rate_card_override, db_path=a.db,
                            allow_duplicate=a.allow_duplicate, parent_id=a.parent_id, cut_type=a.cut_type,
                            music_dominant={"auto": None, "on": True, "off": False}[a.music_dominant])
        except OnboardingError as exc:
            raise SystemExit(f"register-ad failed: {exc}")
        shot = r["packshot"]
        print(f"Registered {r['brand']} - {r['campaign']} as {r['asset_id']} (ad_id {r['ad_id']}, {r['sector']})")
        if r["parent_id"]:
            print(f"  family   {r['cut_type']} of {r['parent_id']} (rolls up to the parent campaign)")
        print(f"  voice    {r['voice_ratio'] * 100:.0f}% voice-over -> "
              + ("MUSIC-DOMINANT: airings need a packshot match to be CONFIRMED" if r["music_dominant"] else "speech-led"))
        print(f"  audio    {r['duration_s']:.2f}s; trimmed {r['lead_trim_s']:.2f}s lead / {r['tail_trim_s']:.2f}s tail "
              f"(< -50 dBFS) -> {r['trimmed_s']:.2f}s, {r['hashes']} landmarks")
        print("  packshot " + (f"pHash {shot['phash']} from t={shot['at']:.2f}s (entropy {shot['entropy']:.2f} bits)"
                               if shot else "none (audio-only or no usable frame): 0.85-0.96 matches will need discovery"))
        c = r["contract"]
        print(f"  SLA      separation {c['competitor_separation_seconds']:g}s, position {c['booked_position']}"
              + (f", rate {c['rate_override_dzd']:,.0f} DZD" if "rate_override_dzd" in c else ""))
        print("  The running monitor hot-reloads the signature within seconds (no restart).")
    elif a.cmd == "ocr-packshots":
        from kihc.analytics import visual_ocr
        args = ["--db", str(a.db), "--memory-cap-mb", str(a.memory_cap_mb)] + (["--retry"] if a.retry else [])
        raise SystemExit(visual_ocr.main(args))
    elif a.cmd == "transcribe-assets":
        from kihc.analytics import transcriber
        from kihc.core.discovery import backfill_nominal_durations
        for asset_id, nominal, basis in backfill_nominal_durations(a.db):
            print(f"{asset_id}: nominal duration {nominal:g} s ({basis})")
        if a.reanalyse:
            changed = transcriber.reanalyse(a.db)
            for c in changed:
                print(f"{c['asset_id']}: suggestion {c['before'] or '-'} -> {c['after'] or '-'}")
            print(f"Re-analysed stored text: {len(changed)} suggestion(s) changed.")
            return
        args =["--db", str(a.db), "--model", a.model, "--memory-cap-mb", str(a.memory_cap_mb)]
        for x in a.assets or []:
            args += ["--asset", x]
        if a.retranscribe:
            args.append("--retranscribe")
        raise SystemExit(transcriber.main(args))
    elif a.cmd == "web":
        from kihc.app.api_server import main as web_main
        web_main(["--host", a.host, "--port", str(a.port), "--db", a.db])
    elif a.cmd == "service-status":
        from kihc.service.headless_runner import read_service_status
        st = read_service_status(a.db)
        if not st:
            print("No service has run against this database.")
            return
        print(f"{'ACTIVE' if st['active'] else 'INACTIVE'} · state={st['state']} pid={st['pid']} "
              f"heartbeat {st['heartbeat_age_s']}s ago · RAM {st['rss_mb']} MB (+ffmpeg {st['ffmpeg_rss_mb']} MB) "
              f"· restarts {st['restarts']} · gc runs {st['gc_runs']}")
        for c in st["channel_status"]:
            print(f"  {c['name']}: {c['state']} level={c.get('level_dbfs')} dBFS "
                  f"received={c['received_seconds']}s detections={c['detections']} restarts={c['restarts']}")
    elif a.cmd == "demo":
        from kihc.tests.generate_test_assets import generate
        ad_a, ad_b, broadcast = generate()
        ingest_reference(ad_a, "Acme", "Ad A", a.db)
        ingest_reference(ad_b, "Decoy Co", "Ad B (never aired)", a.db)
        _print_detections(scan_source(broadcast, "synthetic_broadcast_60s", a.db))


if __name__ == "__main__":
    main()
