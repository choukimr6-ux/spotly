"""Phase 11.3: asset de-duplication, live packshot capture (offline, synthetic TS), packshot OCR."""
import json
import sqlite3
import subprocess
import threading
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from kihc.analytics import visual_ocr as vo
from kihc.config import SAMPLE_RATE as SR
from kihc.core import discovery as D
from kihc.core.audio import ffmpeg_exe
from kihc.core.discovery_report import MUSIC_VISUAL_NEEDED, discovered_airings, list_assets
from kihc.core.fingerprint import StreamingFingerprinter
from kihc.database.schema import get_db
from kihc.stream import elheddaf_stream as heddaf
from kihc.stream import packshot as P
from kihc.tests.synthetic_broadcast import creative, programme

CH = "El-Heddaf TV"
FPS = D.FPS


def fp_of(x):
    return StreamingFingerprinter().feed(x)


def noisy(x, seed, level=0.012):
    """Another airing of the same audio: own noise floor and gain. 0.012 gives ~0.85 window agreement, in the
    range measured between two real El-Heddaf airings of one creative (0.80-0.85)."""
    return (0.9 * x + level * np.random.default_rng(seed).standard_normal(x.size)).astype(np.float32)


@pytest.fixture(scope="module")
def audio():
    a = creative(101, 30)
    return dict(a=a, other=creative(202, 20), jingle_host=np.concatenate([creative(303, 15), a[:8 * SR]]))


# ================================================================== shared audio measure

def test_shared_audio_subsegment_offset_and_unrelated(audio):
    fa = fp_of(audio["a"])
    sub = fa[int(5 * FPS):int(25 * FPS)]                                 # same captured frames, 5 s in
    sh = D.shared_audio(sub, fa)
    assert sh["similarity"] == pytest.approx(1.0) and sh["coverage"] == 1.0
    assert sh["offset_sec"] == pytest.approx(5.0, abs=0.1)              # sub starts 5 s into a
    assert D.shared_audio(fa, sub)["offset_sec"] == pytest.approx(-5.0, abs=0.1)
    other = D.shared_audio(fp_of(audio["other"]), fa)
    assert other["similarity"] < 0.65 and other["coverage"] == 0.0     # unrelated audio ~ chance


def test_shared_audio_ignores_silence():
    z = np.zeros(int(20 * FPS), np.uint32)
    assert D.shared_audio(z, z) == dict(similarity=0.0, coverage=0.0, offset_sec=0.0)


def test_another_airing_of_the_same_creative_scores_below_identical(audio):
    fa, fb = fp_of(audio["a"]), fp_of(noisy(audio["a"], 7))
    sh = D.shared_audio(fb, fa)
    assert 0.75 <= sh["similarity"] < 1.0 and sh["coverage"] >= 0.9


# ================================================================== duplicate finder + merge

def _asset(conn, aid, fp, dur, airings, status="unclassified"):
    conn.execute("INSERT INTO discovered_assets (asset_id, channel, duration_sec, fingerprint, first_seen, last_seen, "
                 "occurrences, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                 (aid, CH, dur, fp.astype("<u4").tobytes(), airings[0], airings[-1], len(airings), status))
    for t in airings:
        conn.execute("INSERT INTO discovered_occurrences (asset_id, channel, airing_start, airing_end, duration_sec, "
                     "confidence, algeria_timestamp) VALUES (?, ?, ?, ?, ?, 0.9, ?)", (aid, CH, t, t + dur, dur, D.algeria_iso(t)))


@pytest.fixture
def dup_db(tmp_path, audio):
    db = tmp_path / "d.db"
    fa = fp_of(audio["a"])
    with get_db(db) as conn:
        _asset(conn, "ELHEDDAF_AD_001", fa, 30.0, [1000.0, 9000.0])                       # master airings
        _asset(conn, "ELHEDDAF_AD_002", fa[int(5 * FPS):int(25 * FPS)], 20.0, [1005.0, 9005.0, 20005.0])
        _asset(conn, "ELHEDDAF_AD_003", fp_of(audio["other"]), 20.0, [3000.0, 12000.0])  # same length, other ad
        _asset(conn, "ELHEDDAF_AD_004", fp_of(noisy(audio["jingle_host"], 3)), 23.0, [5000.0, 15000.0])
    return db


def _table(db, sql):
    with get_db(db) as conn:
        return [tuple(r) for r in conn.execute(sql)]


def test_finder_merges_only_same_airings_never_same_length(dup_db):
    pairs = {(p["duplicate"], p["master"]): p for p in D.find_duplicate_assets(dup_db)}
    p = pairs[("ELHEDDAF_AD_002", "ELHEDDAF_AD_001")]
    assert p["kind"] == "same_airings" and p["offset_sec"] == pytest.approx(5.0, abs=0.1)
    assert len(p["overlapping_airings"]) == 2
    assert not any("ELHEDDAF_AD_003" in k for k in pairs)              # 20 s like AD_002, different audio
    rel = [v for k, v in pairs.items() if "ELHEDDAF_AD_004" in k]
    assert rel and all(v["kind"] == "related" for v in rel)             # shares 8 s (a jingle): flag only


def test_merge_aggregates_without_double_counting_and_unmerge_restores(dup_db):
    before_occ = _table(dup_db, "SELECT * FROM discovered_occurrences ORDER BY id")
    before_assets = _table(dup_db, "SELECT asset_id, status, occurrences, first_seen, last_seen, merged_into "
                                   "FROM discovered_assets ORDER BY asset_id")
    r = D.dedupe_assets(dup_db)
    assert [(m["duplicate"], m["master"]) for m in r["merged"]] == [("ELHEDDAF_AD_002", "ELHEDDAF_AD_001")]
    m = r["merged"][0]
    assert m["same_airings"] == 2 and m["airings_added"] == 1 and m["master_occurrences"] == 3
    with get_db(dup_db) as conn:
        added = conn.execute("SELECT airing_start, duration_sec FROM discovered_occurrences WHERE asset_id = "
                             "'ELHEDDAF_AD_001' ORDER BY airing_start").fetchall()
        dup = conn.execute("SELECT status, merged_into FROM discovered_assets WHERE asset_id = 'ELHEDDAF_AD_002'").fetchone()
        audit = conn.execute("SELECT * FROM asset_merges").fetchone()
        rel4 = json.loads(conn.execute("SELECT related_assets FROM discovered_assets WHERE asset_id = 'ELHEDDAF_AD_004'").fetchone()[0])
    assert [round(x[0], 1) for x in added] == [1000.0, 9000.0, 20000.0]   # 20005 - 5 s offset, master length
    assert added[-1][1] == 30.0 and tuple(dup) == ("merged", "ELHEDDAF_AD_001")
    assert audit["same_airings"] == 2 and len(json.loads(audit["copied_ids"])) == 1
    assert rel4 and rel4[0]["asset_id"] == "ELHEDDAF_AD_001"
    # the duplicate's own rows are kept (audit); reports and listings no longer count them
    assert len(_table(dup_db, "SELECT 1 FROM discovered_occurrences WHERE asset_id = 'ELHEDDAF_AD_002'")) == 3
    assert "ELHEDDAF_AD_002" not in {a["asset_id"] for a in list_assets(dup_db)}
    assert "ELHEDDAF_AD_002" in {a["asset_id"] for a in list_assets(dup_db, include_merged=True)}
    assert not any(o["asset_id"] == "ELHEDDAF_AD_002" for o in discovered_airings(dup_db, limit=0))
    assert D.dedupe_assets(dup_db)["merged"] == []                       # idempotent
    D.unmerge_asset(dup_db, "ELHEDDAF_AD_002")
    assert _table(dup_db, "SELECT * FROM discovered_occurrences ORDER BY id") == before_occ
    assert _table(dup_db, "SELECT asset_id, status, occurrences, first_seen, last_seen, merged_into "
                          "FROM discovered_assets ORDER BY asset_id") == before_assets


def test_dry_run_changes_nothing(dup_db):
    snap = _table(dup_db, "SELECT * FROM discovered_assets ORDER BY asset_id")
    r = D.dedupe_assets(dup_db, apply=False)
    assert r["merged"] == [] and any(p["kind"] == "same_airings" for p in r["pairs"])
    assert _table(dup_db, "SELECT * FROM discovered_assets ORDER BY asset_id") == snap


def test_merge_refuses_classified_assets(dup_db):
    with get_db(dup_db) as conn:
        conn.execute("UPDATE discovered_assets SET status = 'classified' WHERE asset_id = 'ELHEDDAF_AD_001'")
    with pytest.raises(ValueError):
        D.merge_assets(dup_db, "ELHEDDAF_AD_001", "ELHEDDAF_AD_002", similarity=1, coverage=1, offset_sec=5, reason="t")


# ================================================================== engine: prevention + live status

def test_engine_attributes_a_resegmented_airing_to_the_existing_asset(dup_db, audio):
    eng = D.DiscoveryEngine(CH, dup_db, audio_dir=dup_db.parent / "ads")
    seen = []
    eng.on_airing = lambda aid, fp, end: seen.append((aid, round(end, 1)))
    fa = fp_of(audio["a"])
    seg = fa[int(10 * FPS):int(26 * FPS)]                                 # 16 s cut of AD_001's airings
    eng._record(seg, 16.0, 0.95, [(1010.0, None), (9010.0, (0, 0))])
    assert eng.counters["assets_created"] == 0 and eng.counters.get("same_airing_resegmented") == 1
    assert len(_table(dup_db, "SELECT 1 FROM discovered_occurrences WHERE asset_id = 'ELHEDDAF_AD_001'")) == 2
    assert seen == [("ELHEDDAF_AD_001", 9026.0)]                          # packshot hook: later airing's end


def test_engine_respects_status_changed_while_running(dup_db, audio):
    eng = D.DiscoveryEngine(CH, dup_db, audio_dir=dup_db.parent / "ads")
    fo = fp_of(audio["other"])
    D.ignore_asset(dup_db, "ELHEDDAF_AD_003", "station self-promo")       # reviewer acts after the engine loaded
    eng._record(fo, 20.0, 0.9, [(30000.0, None), (40000.0, (0, 0))])
    assert len(_table(dup_db, "SELECT 1 FROM discovered_occurrences WHERE asset_id = 'ELHEDDAF_AD_003'")) == 2
    # merged duplicate -> its new airings go to the master
    D.dedupe_assets(dup_db)
    sub = fp_of(audio["a"])[int(5 * FPS):int(25 * FPS)]
    eng._record(sub, 20.0, 0.9, [(50005.0, None), (60005.0, (0, 0))])
    assert len(_table(dup_db, "SELECT 1 FROM discovered_occurrences WHERE asset_id = 'ELHEDDAF_AD_002'")) == 3
    assert len(_table(dup_db, "SELECT 1 FROM discovered_occurrences WHERE asset_id = 'ELHEDDAF_AD_001' "
                              "AND airing_start >= 50000")) == 2


# ================================================================== packshot capture

def test_elheddaf_ingest_starts_near_the_live_edge():
    # Phase 12.1: -1 (newest) measured 10.8 disconnects/h; one segment back (-2) for stability
    args = heddaf.FFMPEG_INPUT_ARGS
    assert args[args.index("-live_start_index") + 1] == "-2"
    assert heddaf.VIDEO_HLS.endswith("tracks-v1a1/mono.m3u8")


def test_parse_playlist_pdt_and_segment_names():
    text = ("#EXTM3U\n#EXT-X-TARGETDURATION:6\n#EXT-X-PROGRAM-DATE-TIME:2026-10-13T15:50:15.469Z\n"
            "#EXTINF:6.000,\n2026/10/13/15/50/15-06000.ts\n#EXTINF:6.000,\n2026/10/13/15/50/21-06000.ts\n")
    segs = P.parse_playlist(text, "https://h:8081/elheddaftv/tracks-v1a1/mono.m3u8")
    assert [s.url.rsplit("/", 1)[1] for s in segs] == ["15-06000.ts", "21-06000.ts"]
    assert segs[1].pdt - segs[0].pdt == pytest.approx(6.0)
    assert P.segment_name(segs[0].url) == P.segment_name(segs[0].url.replace("tracks-v1a1", "tracks-a1"))


def test_encrypted_playlist_is_refused(monkeypatch):
    class R:
        text = "#EXTM3U\n#EXT-X-KEY:METHOD=AES-128,URI=\"k\"\n"

        def raise_for_status(self):
            pass
    monkeypatch.setattr(P.requests, "get", lambda *a, **k: R())
    with pytest.raises(PermissionError):
        P.fetch_playlist("https://h/x.m3u8")


def test_align_end_with_partial_overlap(audio):
    fa = fp_of(audio["a"])
    hay = fa[int(20 * FPS):int(26 * FPS)]                                 # a 6 s "segment" holding part of the tail
    tail = fa[int(24 * FPS):int(28 * FPS)]                                # spot ends at 28 s: 2 s stick out of hay
    shift, conf = P.align_end(tail, hay, int(28 * FPS) - int(20 * FPS) - 30, int(2 * FPS))   # estimate 30 early
    assert conf > 0.95 and shift == 30


@pytest.fixture(scope="module")
def synthetic_ts(audio):
    """One 6 s MPEG-TS segment (video + AAC audio) whose audio holds the last 5 s of a 12 s spot, then programme.
    (A 4 s fingerprint tail needs a little audio before it: real 6 s segments always have that.)"""
    spot = creative(404, 12)
    pcm = np.concatenate([spot[-5 * SR:], programme(1, 5)])
    raw = (np.clip(pcm, -1, 1) * 32767).astype("<i2").tobytes()
    p = subprocess.run([ffmpeg_exe(), "-nostdin", "-loglevel", "error", "-f", "lavfi", "-i",
                        "testsrc=size=320x180:rate=25", "-f", "s16le", "-ar", str(SR), "-ac", "1", "-i", "pipe:0",
                        "-shortest", "-c:v", "mpeg2video", "-c:a", "aac", "-b:a", "128k", "-f", "mpegts", "pipe:1"],
                       input=raw, capture_output=True, timeout=120)
    assert p.returncode == 0, p.stderr[-300:]
    return dict(ts=p.stdout, spot_fp=fp_of(spot))


def _fake_feed(monkeypatch, ts, pdt=5000.0, live=True):
    calls = []

    def playlist(url, **kw):
        calls.append(("playlist", url))
        name = "tracks-a1" if "/tracks-a1/" in url else "tracks-v1a1"
        return [P.Segment(f"https://h/elheddaftv/{name}/2026/10/13/16/00/00-06000.ts", 6.0, pdt)] if live else []

    def fetch(segs, **kw):
        calls.append(("bytes", [s.url for s in segs]))
        return ts
    monkeypatch.setattr(P, "fetch_playlist", playlist)
    monkeypatch.setattr(P, "fetch_bytes", fetch)
    return calls


def test_ts_pipes_fingerprint_and_single_frame(synthetic_ts):
    hay = P.ts_fingerprint(synthetic_ts["ts"])
    assert abs(hay.size / FPS - 6.0) < 0.3
    tail = synthetic_ts["spot_fp"][-int(4 * FPS):]
    idx, conf = P.locate(tail, hay)
    assert conf > 0.8 and abs((idx + tail.size) / FPS - 5.0) < 0.15     # spot ends 5 s into the segment
    jpeg = P.extract_frame(synthetic_ts["ts"], 3.5, width=320)
    assert jpeg[:2] == b"\xff\xd8" and jpeg[-2:] == b"\xff\xd9"


def test_calibrated_grab_refines_and_fetches_one_video_segment(monkeypatch, synthetic_ts, tmp_path):
    calls = _fake_feed(monkeypatch, synthetic_ts["ts"])
    g = P.PackshotGrabber(tmp_path / "g.db", heddaf.AUDIO_HLS, heddaf.VIDEO_HLS, tmp_path)
    g.offset = 4005.0 - 0.6                                              # true end: pdt 5005.0 at capture 1000.0
    ps = g.grab(synthetic_ts["spot_fp"], 1000.0)
    assert ps.jpeg[:2] == b"\xff\xd8" and ps.offset == pytest.approx(3.5, abs=0.15)
    assert g.offset == pytest.approx(4005.0, abs=0.15)                   # learned from the verified position
    assert [c for c in calls if c[0] == "bytes"] == [("bytes", ["https://h/elheddaftv/tracks-v1a1/2026/10/13/16/00/00-06000.ts"])]


def test_uncalibrated_grab_locates_via_audio_rendition(monkeypatch, synthetic_ts, tmp_path):
    calls = _fake_feed(monkeypatch, synthetic_ts["ts"])
    g = P.PackshotGrabber(tmp_path / "g.db", heddaf.AUDIO_HLS, heddaf.VIDEO_HLS, tmp_path)
    ps = g.grab(synthetic_ts["spot_fp"], 1000.0)
    assert ps.offset == pytest.approx(3.5, abs=0.15) and g.offset == pytest.approx(4005.0, abs=0.15)
    fetched = [c[1][0] for c in calls if c[0] == "bytes"]
    assert "tracks-a1" in fetched[0] and "tracks-v1a1" in fetched[1] and len(fetched) == 2


def test_expired_segment_is_a_clean_miss(monkeypatch, synthetic_ts, tmp_path):
    _fake_feed(monkeypatch, synthetic_ts["ts"], pdt=6000.0)                 # window moved past the spot
    g = P.PackshotGrabber(tmp_path / "g.db", heddaf.AUDIO_HLS, heddaf.VIDEO_HLS, tmp_path)
    g.offset = 4005.0
    with pytest.raises(P.PackshotMiss, match="no longer published"):
        g.grab(synthetic_ts["spot_fp"], 1000.0)


def test_grabber_thread_saves_one_packshot_and_records_misses(monkeypatch, synthetic_ts, tmp_path):
    db = tmp_path / "g.db"
    with get_db(db) as conn:
        _asset(conn, "ELHEDDAF_AD_001", synthetic_ts["spot_fp"], 12.0, [100.0, 900.0])
        _asset(conn, "ELHEDDAF_AD_002", synthetic_ts["spot_fp"], 12.0, [200.0, 950.0])
        conn.execute("UPDATE discovered_assets SET content_hint = 'station self-promo (El-Heddaf TV)' "
                     "WHERE asset_id = 'ELHEDDAF_AD_002'")
    _fake_feed(monkeypatch, synthetic_ts["ts"])
    g = P.PackshotGrabber(db, heddaf.AUDIO_HLS, heddaf.VIDEO_HLS, tmp_path / "shots", max_age=1e12).start()
    try:
        g.offset = 4005.0
        g.submit("ELHEDDAF_AD_002", synthetic_ts["spot_fp"], 1000.0)          # promo: skipped
        g.submit("ELHEDDAF_AD_001", synthetic_ts["spot_fp"], 1000.0)
        g.submit("ELHEDDAF_AD_001", synthetic_ts["spot_fp"], 1000.0)          # already captured: skipped
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline and g._q.unfinished_tasks and g.stats["captured"] == 0:
            time.sleep(0.2)
        time.sleep(1.0)
    finally:
        g.stop()
    with get_db(db) as conn:
        a1 = dict(conn.execute("SELECT * FROM discovered_assets WHERE asset_id = 'ELHEDDAF_AD_001'").fetchone())
        a2 = dict(conn.execute("SELECT * FROM discovered_assets WHERE asset_id = 'ELHEDDAF_AD_002'").fetchone())
    assert g.stats["captured"] == 1 and a1["visual_status"] == "captured" and a1["packshot_attempts"] == 1
    assert (tmp_path / "shots" / "ELHEDDAF_AD_001_packshot.jpg").read_bytes()[:2] == b"\xff\xd8"
    assert a2["packshot_path"] is None and a2["packshot_attempts"] == 0
    # a late detection is recorded honestly as a miss, not retried forever
    g2 = P.PackshotGrabber(db, heddaf.AUDIO_HLS, heddaf.VIDEO_HLS, tmp_path / "shots", max_age=45).start()
    with get_db(db) as conn:
        conn.execute("UPDATE discovered_assets SET content_hint = NULL WHERE asset_id = 'ELHEDDAF_AD_002'")
    try:
        g2.submit("ELHEDDAF_AD_002", synthetic_ts["spot_fp"], time.time() - 120)
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline and g2.stats["missed"] == 0:
            time.sleep(0.1)
    finally:
        g2.stop()
    with get_db(db) as conn:
        a2 = dict(conn.execute("SELECT * FROM discovered_assets WHERE asset_id = 'ELHEDDAF_AD_002'").fetchone())
    assert a2["packshot_attempts"] == 1 and "expired" in a2["packshot_last_error"]


# ================================================================== OCR

W, H = 1280, 720


def box(cx, cy, w=200, h=40):
    return [[cx - w / 2, cy - h / 2], [cx + w / 2, cy - h / 2], [cx + w / 2, cy + h / 2], [cx - w / 2, cy + h / 2]]


def test_ocr_region_and_noise_filters():
    assert vo.keep_box(box(640, 300), "DJEZZY", 0.9, W, H)
    assert not vo.keep_box(box(1200, 60), "ELHEDDAF", 0.9, W, H)               # channel bug corner
    assert not vo.keep_box(box(640, 690), "Ligue 1 Mobilis", 0.9, W, H)        # ticker band: not a URL
    assert vo.keep_box(box(640, 690), "www.djezzy.dz", 0.9, W, H)              # ticker band URL kept
    for noise in ("17:04", "2-3", "GMT+1", "--"):
        assert not vo.keep_box(box(640, 300), noise, 0.9, W, H)
    assert vo.keep_box(box(640, 300), "جيزي", 0.9, W, H)                       # Phase 11.4: Arabic recogniser
    assert not vo.keep_box(box(640, 300), "DJEZZY", 0.4, W, H)


@pytest.mark.parametrize("lines, brand", [
    (["DJEZZY", "Internet illimité"], "Djezzy"),
    (["www.ooredoo.dz"], "Ooredoo"),
    (["OOREDOO4G", "OOREDOO"], "Ooredoo"),
    (["Condor", "Smart TV"], "Condor"),
    (["USMA", "20:15", "PSG"], None),
])
def test_ocr_brand_suggestion(lines, brand):
    assert vo.analyse_ocr(lines)["suggested_brand"] == brand


def _ocr_db(tmp_path, jpeg=b"\xff\xd8fake\xff\xd9", audio_brand=None):
    db = tmp_path / "o.db"
    shot = tmp_path / "ELHEDDAF_AD_001_packshot.jpg"
    shot.write_bytes(jpeg)
    with get_db(db) as conn:
        _asset(conn, "ELHEDDAF_AD_001", np.ones(100, np.uint32), 20.0, [1.0, 2.0])
        conn.execute("UPDATE discovered_assets SET packshot_path = ?, suggested_brand = ?, suggestion_source = ? "
                     "WHERE asset_id = 'ELHEDDAF_AD_001'", (str(shot), audio_brand, "audio" if audio_brand else None))
    return db


def _fake_frame(lines):
    img = np.full((720, 1280, 3), 90, np.uint8)
    return lambda engine, jpeg: (img, [(box(640, 300 + 40 * i), t, 0.9) for i, t in enumerate(lines)], list(lines))


@pytest.fixture(autouse=False)
def no_opencv(monkeypatch):
    """Visual analysis needs OpenCV, which production only ever loads in the OCR child process; these text-path
    tests stub it so the pytest process stays within the memory budget measured by test_real_ingest."""
    monkeypatch.setattr(vo, "analyse_frame", lambda img, raw, brand_boxes, library=None: dict(
        programme_overlay=False, logo_box=None, logo_basis=None, logo_area=0.0, sharpness=0.0, dhash="0" * 16,
        logo_match=None))


def test_ocr_batch_sets_packshot_suggestion(tmp_path, monkeypatch, no_opencv):
    db = _ocr_db(tmp_path)
    monkeypatch.setattr(vo, "ocr_frame", _fake_frame(["DJEZZY", "www.djezzy.dz"]))
    out = vo.run_batch(db, engine_factory=lambda: object(), logo_root=tmp_path / "logos")
    assert out[0]["suggested_brand"] == "Djezzy"
    a = list_assets(db)[0]
    assert a["suggested_brand"] == "Djezzy" and a["suggestion_source"] == "packshot" and a["visual_status"] == "ocr_brand"
    assert json.loads(a["brand_evidence"])[0]["source"] == "packshot" and vo.pending(db) == []


def test_ocr_never_overrides_a_speech_suggestion(tmp_path, monkeypatch, no_opencv):
    db = _ocr_db(tmp_path, audio_brand="Mobilis")
    monkeypatch.setattr(vo, "ocr_frame", _fake_frame(["DJEZZY"]))
    vo.run_batch(db, engine_factory=lambda: object(), logo_root=tmp_path / "logos")
    a = list_assets(db)[0]
    assert a["suggested_brand"] == "Mobilis" and a["suggestion_source"] == "audio" and a["ocr_text"] == "DJEZZY"


def test_music_spot_label_and_visual_needed(tmp_path):
    db = tmp_path / "m.db"
    with get_db(db) as conn:
        _asset(conn, "ELHEDDAF_AD_001", np.ones(100, np.uint32), 20.0, [1.0, 2.0])
        conn.execute("UPDATE discovered_assets SET transcript = '', transcribed_at = 'x' WHERE asset_id = 'ELHEDDAF_AD_001'")
    assert vo.mark_visual_needed(db) == 1
    a = list_assets(db)[0]
    assert a["visual_status"] == "needed" and a["review_label"] == MUSIC_VISUAL_NEEDED


def test_real_ocr_batch_process_reads_rendered_brand_within_budget(tmp_path):
    """The real engine, run exactly as the pilot runs it: its own process, memory watchdog at 300 MB."""
    import importlib.util
    import sys
    if importlib.util.find_spec("rapidocr_onnxruntime") is None:
        pytest.skip("rapidocr-onnxruntime not installed")
    from PIL import Image, ImageDraw, ImageFont
    img = Image.new("RGB", (1280, 720), "white")
    ImageDraw.Draw(img).text((380, 300), "DJEZZY", fill="black", font=ImageFont.load_default(size=110))
    path = tmp_path / "ELHEDDAF_AD_001_packshot.jpg"
    img.save(path, quality=90)
    db = _ocr_db(tmp_path, jpeg=path.read_bytes())
    p = subprocess.run([sys.executable, "-m", "kihc.analytics.visual_ocr", "--db", str(db), "--memory-cap-mb", "300"],
                       capture_output=True, text=True, timeout=300)
    assert p.returncode == 0, p.stdout + p.stderr
    assert "brand Djezzy" in p.stdout, p.stdout
    peak = float(p.stdout.rsplit("peak RSS", 1)[1].split("MB")[0])
    print(f"\n[OCR] batch process peak RSS {peak:.0f} MB")
    assert peak < 300
    a = list_assets(db)[0]
    assert a["suggested_brand"] == "Djezzy" and a["suggestion_source"] == "packshot"


# ================================================================== API + scheduler slot

def test_packshot_endpoint_and_listing(tmp_path):
    from kihc.app.api_server import create_app
    db = _ocr_db(tmp_path)
    with get_db(db) as conn:
        _asset(conn, "ELHEDDAF_AD_009", np.ones(100, np.uint32), 20.0, [5.0])
        conn.execute("UPDATE discovered_assets SET status = 'merged', merged_into = 'ELHEDDAF_AD_001', "
                     "packshot_path = 'C:/Windows/win.ini' WHERE asset_id = 'ELHEDDAF_AD_009'")
    app = create_app(db, channel_loader=lambda: [])
    app.state.audio_root = tmp_path
    with TestClient(app) as c:
        assets = c.get("/api/discovery/assets").json()["assets"]
        assert [a["asset_id"] for a in assets] == ["ELHEDDAF_AD_001"] and "packshot_path" not in assets[0]
        r = c.get(assets[0]["packshot_url"])
        assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
        assert len(c.get("/api/discovery/assets", params={"include_merged": True}).json()["assets"]) == 2
        assert c.get("/api/discovery/assets/ELHEDDAF_AD_009/packshot").status_code == 404     # outside cache dir
        js = c.get("/static/app.js").text
        assert "packshot_url" in js and "review_label" in js


def test_background_batches_never_run_concurrently(tmp_path):
    from kihc.service.pilot import TranscriptionScheduler
    active, peak = [0], [0]
    lock = threading.Lock()

    class Proc:
        pid = 0

        def __init__(self):
            self.t = time.monotonic()

        def terminate(self):
            self.t = -1e9                                                 # finishes at the next poll

        def poll(self):
            if time.monotonic() - self.t < 0.6:
                return None
            with lock:
                if not getattr(self, "done", False):
                    self.done = True
                    active[0] -= 1
            return 0

    class S(TranscriptionScheduler):
        runs = 0

        def _launch(self, n):
            with lock:
                active[0] += 1
                peak[0] = max(peak[0], active[0])
            self.runs += 1
            self._proc = Proc()
    a = S(tmp_path / "x.db", interval=0.4, pending_fn=lambda db: [1], extra_args=[], name="a")
    b = S(tmp_path / "x.db", interval=0.4, pending_fn=lambda db: [1], extra_args=[], name="b")
    a.start(), b.start()
    time.sleep(8)
    a.stop(), b.stop()
    assert peak[0] == 1 and a.runs + b.runs >= 3
