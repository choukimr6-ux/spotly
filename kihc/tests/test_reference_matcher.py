"""Phase 13.1: constellation fingerprint, exact-match engine, hybrid gate, registry build, WAL safety.

All audio is synthetic (voiced notes + harmonics + noise bursts), so nothing depends on production data.
"""
import sqlite3
import threading
import time
import wave

import numpy as np
import pytest

from kihc.analytics import reference_matcher as rm
from kihc.analytics.compliance_audit import list_alerts
from kihc.core.live_worker import _Chunk, decimate
from kihc.core.pod_analyzer import cluster_occurrences_into_pods
from kihc.data import reference_db as rdb
from kihc.database.schema import get_db, init_db

SR = rm.SR


def synth(seed: int, dur: float, sr: int = SR) -> np.ndarray:
    """Speech/jingle-like test signal: overlapping voiced notes with harmonics, vibrato and fricative bursts."""
    rng = np.random.default_rng(seed)
    n = int(dur * sr)
    out = np.zeros(n, np.float32)
    for voice in range(2):
        pos = int(rng.uniform(0, 0.2) * sr) if voice else 0
        while pos < n:
            seg = int(rng.uniform(0.12, 0.45) * sr)
            tt = np.arange(min(seg, n - pos)) / sr
            f0 = rng.uniform(170, 650)
            amps = rng.uniform(0.15, 1.0, 7)
            vib = 1 + 0.01 * np.sin(2 * np.pi * rng.uniform(4, 7) * tt)
            sig = sum(a * np.sin(2 * np.pi * f0 * (k + 1) * vib * tt) for k, a in enumerate(amps))
            env = np.abs(np.sin(np.pi * tt / max(tt[-1], 1e-3))) ** 0.5 if tt.size > 1 else 1.0
            out[pos:pos + tt.size] += (sig * env * (0.5 if voice else 1.0)).astype(np.float32)
            pos += seg + int(rng.uniform(0.0, 0.12) * sr)
    for _ in range(int(dur * 1.5)):                              # fricative bursts
        a = int(rng.uniform(0, max(1, n - 2000)))
        out[a:a + 1500] += (rng.standard_normal(min(1500, n - a)) * 0.15).astype(np.float32)
    return (out / np.abs(out).max() * 0.5).astype(np.float32)


def noise(sec: float, amp: float = 0.03, seed: int = 99) -> np.ndarray:
    return (np.random.default_rng(seed).standard_normal(int(sec * SR)) * amp).astype(np.float32)


def speed(x: np.ndarray, factor: float) -> np.ndarray:
    """Tape-style speed change: pitch and tempo move together (factor 1.05 = 5 % faster and higher)."""
    return np.interp(np.arange(0, x.size - 1, factor), np.arange(x.size), x).astype(np.float32)


def signature(asset_id, x, brand="Acme", sector="Telecom", nominal=None):
    codes, t_cs, lead, trimmed = rm.signature_from_audio(x)
    return rdb.Signature(asset_id, brand, sector, nominal or x.size / SR, codes, t_cs, lead, trimmed)


@pytest.fixture(scope="module")
def ads():
    return {name: synth(seed, dur) for name, seed, dur in (("A", 1, 30.0), ("B", 2, 20.0), ("C", 3, 25.0))}


@pytest.fixture(scope="module")
def index(ads):
    return rm.ReferenceIndex([signature(n, ads[n], brand=f"Brand{n}") for n in ("A", "B")])


def run_stream(index, audio, epoch0=1_700_000_000.0, piece=0.2):
    m = rm.StreamMatcher(index)
    out, step = [], int(piece * SR)
    for i in range(0, audio.size, step):
        out += m.feed(audio[i:i + step], epoch0 + i / SR)
    return out + m.flush(), m


# ----------------------------------------------------------------------------- schema / blob
def test_blob_roundtrip_and_corruption():
    codes = np.array([5, 1 << 18, 77], np.uint32)
    t_cs = np.array([0, 150, 65535], np.uint16)
    blob = rdb.pack_blob(codes, t_cs, 0.12, 29.5)
    c2, t2, lead, trimmed = rdb.unpack_blob(blob)
    assert np.array_equal(c2, codes) and np.array_equal(t2, t_cs) and (round(lead, 2), trimmed) == (0.12, 29.5)
    assert len(blob) == 16 + 6 * 3
    for bad in (blob[:-1], b"XXXX" + blob[4:], b""):
        with pytest.raises(ValueError):
            rdb.unpack_blob(bad)
    with pytest.raises(ValueError):
        rdb.pack_blob(codes, np.array([0, 1, 70000]), 0.0, 1.0)


def test_schema_columns(tmp_path):
    with get_db(tmp_path / "k.db") as conn:
        rdb.ensure_schema(conn)
        cols = {r["name"]: (r["type"], r["pk"], r["dflt_value"]) for r in conn.execute("PRAGMA table_info(reference_signatures)")}
    assert list(cols) == ["asset_id", "brand_name", "canonical_sector", "nominal_duration", "audio_fingerprint_blob",
                          "visual_p_hash", "created_at", "last_matched_at", "match_count", "ad_id",
                          "master_asset_id", "cut_type", "voice_ratio", "is_music_dominant"]   # Phase 21 families, Phase 22 voice gate
    assert cols["asset_id"][1] == 1 and cols["match_count"][2] == "0" and cols["audio_fingerprint_blob"][0] == "BLOB"


# ----------------------------------------------------------------------------- matching
@pytest.mark.parametrize("lead_in", [0.0, 2.0, 7.37])
def test_time_offset_invariance(index, ads, lead_in):
    """The same airing placed at different stream positions (variable broadcast latency) is found at that position."""
    audio = np.concatenate([noise(lead_in + 1.0), ads["A"], noise(6)])
    out, _ = run_stream(index, audio)
    assert [c.asset_id for c in out] == ["A"]
    c = out[0]
    assert c.confidence >= rm.ACCEPT_CONF and c.complete
    assert abs((c.airing_start - 1_700_000_000.0) - (lead_in + 1.0)) < 0.1
    assert c.airing_end - c.airing_start == pytest.approx(30.0, abs=0.01)         # snapped to the nominal duration


def test_two_second_delay_shifts_the_start_by_exactly_two_seconds(index, ads):
    a, _ = run_stream(index, np.concatenate([noise(4), ads["A"], noise(6)]))
    b, _ = run_stream(index, np.concatenate([noise(6), ads["A"], noise(6)]))      # same airing, 2.0 s later
    assert a and b and b[0].confidence >= rm.ACCEPT_CONF
    assert (b[0].airing_start - a[0].airing_start) == pytest.approx(2.0, abs=0.08)


def test_capture_that_joins_the_ad_midway(index, ads):
    """Stream starts 11 s into the ad: still matched, start extrapolated to before the capture began."""
    audio = np.concatenate([ads["A"][11 * SR:], noise(6)])
    out, _ = run_stream(index, audio)
    assert [c.asset_id for c in out] == ["A"]
    assert out[0].confidence >= rm.ACCEPT_CONF
    assert (out[0].airing_start - 1_700_000_000.0) == pytest.approx(-11.0, abs=0.12)


@pytest.mark.parametrize("factor", [0.95, 0.97, 1.03, 1.05])
def test_resilient_to_5_percent_pitch_speed_jitter(index, ads, factor):
    audio = np.concatenate([noise(4), speed(ads["A"], factor), noise(6)])
    out, _ = run_stream(index, audio)
    assert [c.asset_id for c in out] == ["A"], f"speed x{factor} not matched"
    c = out[0]
    assert c.confidence >= rm.ACCEPT_CONF
    assert c.speed_ratio == pytest.approx(factor, abs=0.01)
    assert abs((c.airing_start - 1_700_000_000.0) - 4.0) < 0.2


def test_resilient_to_added_noise_and_level_change(index, ads):
    quiet = ads["B"] * 0.2 + noise(ads["B"].size / SR, 0.005, seed=5)      # -14 dB and a noisy channel
    out, _ = run_stream(index, np.concatenate([noise(3), quiet, noise(4)]))
    assert [c.asset_id for c in out] == ["B"] and out[0].confidence >= rm.ACCEPT_CONF
    # at ~0 dB SNR the audio is still found, but its score drops into the band that demands the packshot check
    awful = ads["B"] * 0.2 + noise(ads["B"].size / SR, 0.05, seed=6)
    out, _ = run_stream(index, np.concatenate([noise(3), awful, noise(4)]))
    assert out == [] or (out[0].asset_id == "B" and rm.VERIFY_FLOOR <= out[0].confidence)


def test_two_different_ads_back_to_back(index, ads):
    out, _ = run_stream(index, np.concatenate([noise(2), ads["A"], ads["B"], noise(5)]))
    assert sorted(c.asset_id for c in out) == ["A", "B"]
    a, b = sorted(out, key=lambda c: c.airing_start)
    assert b.airing_start - a.airing_start == pytest.approx(30.0, abs=0.15)


def test_no_false_positives(index, ads):
    """Unknown ad, pure noise, silence and a different ad that is not indexed claim nothing."""
    for audio in (np.concatenate([noise(3), ads["C"], noise(4)]), noise(40), np.zeros(30 * SR, np.float32),
                  np.concatenate([noise(3), synth(77, 30.0), noise(4)])):
        out, _ = run_stream(index, audio)
        assert out == []


def test_cut_off_ad_is_not_claimed(index, ads):
    """Only the first third of an ad aired: not enough of it to bill (needs half of the creative)."""
    out, _ = run_stream(index, np.concatenate([noise(3), ads["A"][:10 * SR], noise(8)]))
    assert out == []


def test_ad_spliced_with_other_audio_scores_below_acoustic_threshold(index, ads):
    """Half the creative then different content: never auto-accepted at >= 0.96."""
    out, _ = run_stream(index, np.concatenate([noise(3), ads["A"][:16 * SR], ads["C"][:14 * SR], noise(4)]))
    assert all(c.confidence < rm.ACCEPT_CONF for c in out)


def test_silence_trim_and_lead_back_in(ads):
    padded = np.concatenate([np.zeros(int(1.5 * SR), np.float32), ads["B"], np.zeros(int(2.0 * SR), np.float32)])
    sig = signature("B", padded, nominal=20.0 + 1.5)
    assert sig.lead_trim_s == pytest.approx(1.5 - 0.04, abs=0.03)
    assert sig.trimmed_s == pytest.approx(20.0 + 0.08, abs=0.1)
    out, _ = run_stream(rm.ReferenceIndex([sig]), np.concatenate([noise(3), padded, noise(4)]))
    # the creative's first (silent) frame is snapped to where the untrimmed file would have started: 3.0 s
    assert out and abs((out[0].airing_start - 1_700_000_000.0) - 3.0) < 0.12


# ----------------------------------------------------------------------------- resource boundary
def test_index_ram_and_lookup_latency_budget():
    rng = np.random.default_rng(0)
    sigs = []
    for i in range(250):                         # 250 spots x 55 s x ~80 hashes/s = 1.1 M hashes
        n = 4400
        sigs.append(rdb.Signature(f"X{i}", "b", None, 55.0, rng.integers(0, 1 << 19, n, dtype=np.uint32),
                                  np.sort(rng.integers(0, 5500, n)).astype(np.uint16), 0.0, 55.0))
    idx = rm.ReferenceIndex(sigs)
    mb = idx.memory_bytes() / 2 ** 20
    assert idx.keys.size == 1_100_000 and mb < rm.INDEX_BUDGET_MB and mb < 12
    q = rng.integers(0, 1 << 19, 2500, dtype=np.uint32)          # one second of probed query hashes is ~ 1000-2500
    times = []
    for _ in range(40):
        t0 = time.perf_counter()
        idx.lookup(q)
        times.append((time.perf_counter() - t0) * 1000)
    assert np.percentile(times, 95) < rm.LOOKUP_BUDGET_MS


def test_real_stream_step_latency_under_budget(index, ads):
    _, m = run_stream(index, np.concatenate([noise(3), ads["A"], noise(5)]))
    steps = m.lookup_ms[:-1]                                      # last one is the flush of the final seconds
    assert np.median(steps) < rm.LOOKUP_BUDGET_MS and np.percentile(steps, 90) < rm.LOOKUP_BUDGET_MS


def test_streaming_buffer_stays_bounded(index, ads):
    m = rm.StreamMatcher(index)
    for i in range(0, 5 * 60 * SR, SR):                           # 5 minutes of audio
        m.feed(noise(1.0, seed=i % 7), 1e9 + i / SR)
    assert m._buf.size < 4 * SR and m._pk_f.size < 4000
    assert sum(t.size for t, _ in m._hits.values()) < 20000          # chance hits are pruned with the history window


def test_clock_jump_resets_the_stream(index, ads):
    m = rm.StreamMatcher(index)
    m.feed(ads["A"][:5 * SR], 1000.0)
    m.feed(noise(1.0), 5000.0)                                    # reconnect: capture clock jumped
    assert m.epoch0 == 5000.0 and m.stream_seconds == pytest.approx(1.0)


# ----------------------------------------------------------------------------- shared audio between assets
@pytest.fixture(scope="module")
def nested(ads):
    """X (20 s) is a cut inside Y (= 8 s intro + X + 7 s tag). Real broadcasts re-use creatives like this."""
    x = synth(5, 20.0)
    y = np.concatenate([synth(6, 8.0), x, synth(7, 7.0)])
    return x, y, rm.ReferenceIndex([signature("X", x, brand="BX"), signature("Y", y, brand="BY")])


def test_relations_detect_a_cut_inside_a_longer_creative(nested):
    x, y, idx = nested
    rel = {(idx.meta[r["x"]].asset_id, idx.meta[r["y"]].asset_id): r for r in idx.relations()}
    assert rel[("X", "Y")]["kind"] == "contained" and rel[("X", "Y")]["x_in_y"] == pytest.approx(8.0, abs=0.2)
    assert ("Y", "X") not in rel or rel[("Y", "X")]["kind"] == "overlap"


def test_longer_spot_aired_is_logged_once_not_twice(nested):
    x, y, idx = nested
    out, _ = run_stream(idx, np.concatenate([noise(3), y, noise(8)]))
    assert [c.asset_id for c in out] == ["Y"]                       # X lies inside it: no second airing


def test_cut_down_aired_alone_is_logged_as_the_cut_down(nested):
    x, y, idx = nested
    out, _ = run_stream(idx, np.concatenate([noise(3), x, noise(8)]))
    assert [c.asset_id for c in out] == ["X"] and out[0].confidence >= rm.ACCEPT_CONF
    assert (out[0].airing_start - 1_700_000_000.0) == pytest.approx(3.0, abs=0.15)


def test_cut_down_followed_by_other_content_is_not_claimed_as_the_long_spot(nested):
    x, y, idx = nested
    out, _ = run_stream(idx, np.concatenate([noise(12), x, synth(40, 15.0), noise(5)]))      # Y's intro would be on air
    assert [c.asset_id for c in out] == ["X"]                       # Y's first 8 s are missing: Y never explains the audio


def test_cut_down_then_the_long_spot_are_each_logged_once(nested):
    """Hits the cut-down left on the long spot must not corrupt the long spot's own airing a few seconds later."""
    x, y, idx = nested
    out, _ = run_stream(idx, np.concatenate([noise(12), x, noise(5), y, noise(8)]))
    assert [c.asset_id for c in sorted(out, key=lambda c: c.airing_start)] == ["X", "Y"]
    assert all(c.confidence >= rm.ACCEPT_CONF for c in out)
    starts = sorted(c.airing_start - 1_700_000_000.0 for c in out)
    assert starts[0] == pytest.approx(12.0, abs=0.15) and starts[1] == pytest.approx(12.0 + 20.0 + 5.0, abs=0.15)


def test_same_audio_indexed_twice_is_one_airing(ads):
    twin = rm.ReferenceIndex([signature("A1", ads["A"], brand="B1"), signature("A2", ads["A"], brand="B2")])
    assert {r["kind"] for r in twin.relations()} == {"duplicate"}
    out, _ = run_stream(twin, np.concatenate([noise(2), ads["A"], noise(6)]))
    assert len(out) == 1


# ----------------------------------------------------------------------------- DB + hybrid gate + logging
@pytest.fixture
def env(tmp_path, ads):
    db = tmp_path / "k.db"
    with get_db(db) as conn:
        conn.execute("INSERT INTO reference_ads (ad_id, brand_name, campaign_name, duration_sec, fingerprint, fp_algorithm, "
                     "industry) VALUES (1, 'BrandA', 'camp', 30, x'00', 'x', 'Telecom')")
        conn.execute("INSERT INTO discovered_assets (asset_id, channel, duration_sec, fingerprint, status, promoted_ad_id) "
                     "VALUES ('A', 'Ch', 30, x'00', 'classified', 1)")
        conn.execute("INSERT INTO discovered_assets (asset_id, channel, duration_sec, fingerprint, status) "
                     "VALUES ('B', 'Ch', 20, x'00', 'classified')")             # confirmed but never promoted
        rdb.ensure_schema(conn)
        for n in ("A", "B"):
            s = signature(n, ads[n], brand=f"Brand{n}")
            rdb.upsert_signature(conn, n, s.brand_name, "Telecom", s.nominal_duration,
                                 rdb.pack_blob(s.codes, s.t_cs, s.lead_trim_s, s.trimmed_s),
                                 "f0f0f0f0f0f0f0f0" if n == "A" else None)
    return db


def occurrences(db):
    with get_db(db) as conn:
        return [dict(r) for r in conn.execute("SELECT * FROM ad_occurrences ORDER BY airing_start")]


def test_confirmed_airing_is_logged_with_zero_heavy_inference(env, ads, monkeypatch):
    import sys
    heavy = {"faster_whisper", "paddleocr", "kihc.analytics.transcriber", "kihc.analytics.visual_ocr"}
    before = heavy & set(sys.modules)
    rmx = rm.ReferenceMatcher(env, compliance=False)
    out = rmx.feed("Ch One", np.concatenate([noise(2), ads["A"], noise(6)]), 1_700_000_000.0)
    out += rmx.end_stream("Ch One")
    assert [c.decision for c in out] == ["confirmed"] and out[0].ad_id == 1
    rows = occurrences(env)
    assert len(rows) == 1 and rows[0]["source_name"] == "live:Ch One" and rows[0]["confidence_score"] >= 0.96
    assert rows[0]["airing_end"] - rows[0]["airing_start"] == pytest.approx(30.0, abs=0.01)
    assert rows[0]["verification_hash"] and rows[0]["algeria_timestamp"].endswith("+01:00")
    with get_db(env) as conn:
        r = conn.execute("SELECT match_count, last_matched_at FROM reference_signatures WHERE asset_id='A'").fetchone()
    assert r["match_count"] == 1 and r["last_matched_at"]
    assert rmx.in_reference_airing("Ch One", rows[0]["airing_start"] + 10)
    assert not rmx.in_reference_airing("Ch One", rows[0]["airing_end"] + 20)
    assert (heavy & set(sys.modules)) == before                                # STT / OCR never imported


def test_redelivery_is_a_duplicate_not_a_second_airing(env, ads):
    audio = np.concatenate([noise(2), ads["A"], noise(6)])
    rmx = rm.ReferenceMatcher(env, compliance=False)
    rmx.feed("Ch One", audio, 1_700_000_000.0)
    rmx.end_stream("Ch One")
    again = rmx.feed("Ch One", audio, 1_700_000_000.4) + rmx.end_stream("Ch One")   # same airing seen 0.4 s later
    assert [c.decision for c in again] == ["duplicate"]
    assert len(occurrences(env)) == 1
    with get_db(env) as conn:
        assert conn.execute("SELECT match_count FROM reference_signatures WHERE asset_id='A'").fetchone()[0] == 1


def test_asset_without_promoted_ad_is_not_billed(env, ads):
    rmx = rm.ReferenceMatcher(env, compliance=False)
    out = rmx.feed("Ch One", np.concatenate([noise(2), ads["B"], noise(6)]), 1.7e9) + rmx.end_stream("Ch One")
    assert [c.decision for c in out] == ["unlinked"] and occurrences(env) == []


def _candidate(asset_id="A", conf=0.9):
    return rm.Candidate(asset_id, "BrandA", "Telecom", 30.0, conf, 1_700_000_000.0, 1_700_000_030.0, 500, conf, 28.0,
                        1.0, 10, True, packshot_at=1_700_000_028.5)


@pytest.mark.parametrize("conf,grab,expected,logged", [
    (0.97, "boom", "confirmed", 1),                 # acoustic alone: the grabber is never even called
    (0.90, "f0f0f0f0f0f0f0f1", "confirmed_visual", 1),   # 1 bit away
    (0.90, "0f0f0f0f0f0f0f0f", "rejected_visual", 0),    # all bits flipped
    (0.90, None, "needs_visual", 0),                # no frame obtainable
    (0.855, "f0f0f0f0f0f0f0f0", "confirmed_visual", 1),
    (0.955, "0f0f0f0f0f0f0f0f", "rejected_visual", 0),   # 0.95 .. 0.96 also needs the visual check
])
def test_hybrid_safety_gate(env, conf, grab, expected, logged):
    calls = []

    def grabber(channel, at, asset_id=None):
        calls.append((channel, at))
        if grab == "boom":
            raise AssertionError("must not be called")
        return grab                                  # a str is taken as an already computed pHash
    rmx = rm.ReferenceMatcher(env, frame_grabber=grabber, compliance=False)
    c = rmx._gate("Ch One", _candidate(conf=conf))
    assert c.decision == expected and len(occurrences(env)) == logged
    assert bool(calls) == (conf < rm.ACCEPT_CONF)
    if conf < rm.ACCEPT_CONF:
        assert calls[0][1] == 1_700_000_028.5        # packshot position: 1.5 s before the snapped end


def test_phash_matches_hamming_semantics():
    from PIL import Image, ImageDraw
    a = Image.new("RGB", (320, 180), "white")
    ImageDraw.Draw(a).rectangle([40, 30, 200, 120], fill="navy")
    same = a.resize((640, 360))                                      # rescaled copy of the same packshot
    other = Image.new("RGB", (320, 180), "black")
    ImageDraw.Draw(other).ellipse([100, 20, 300, 160], fill="orange")
    h = rm.phash_image(a)
    assert len(h) == 16 and rm.hamming_hex(h, rm.phash_image(same)) <= rm.VISUAL_MAX_BITS
    assert rm.hamming_hex(h, rm.phash_image(other)) > rm.VISUAL_MAX_BITS


def test_confirmed_airing_feeds_pods_and_compliance(env, ads):
    rmx = rm.ReferenceMatcher(env, compliance=True, compliance_min_interval=0)
    rmx.feed("Ch One", np.concatenate([noise(2), ads["A"], noise(6)]), time.time() - 600)
    rmx.end_stream("Ch One")
    pods = cluster_occurrences_into_pods("Ch One", db_path=env)
    assert len(pods) == 1 and pods[0].ads[0]["brand"] == "BrandA"
    with get_db(env) as conn:
        pos = conn.execute("SELECT pod_id, pod_position FROM ad_occurrences").fetchone()
    assert pos["pod_id"] and pos["pod_position"]                    # position cached for the compliance engine
    list_alerts(env)                                                # alert generation ran without error


def test_registry_hot_reload_and_ignored_legacy_ids(env, ads):
    rmx = rm.ReferenceMatcher(env, compliance=False)
    assert rmx.covered_ad_ids() == {1} and rmx.reload_if_changed() is False
    with get_db(env) as conn:
        s = signature("C", ads["C"])
        rdb.upsert_signature(conn, "C", "BrandC", None, 25.0, rdb.pack_blob(s.codes, s.t_cs, s.lead_trim_s, s.trimmed_s), None)
    assert rmx.reload_if_changed() is True and len(rmx.index) == 3


# ----------------------------------------------------------------------------- WAL / transaction safety
def _airing(db, **kw):
    base = dict(asset_id="A", ad_id=1, source_name="live:Ch", airing_start=1000.0, airing_end=1030.0, duration=30.0,
                confidence=0.99, algeria_timestamp="2026-01-01T00:00:00.000+01:00", verification_hash="h",
                detected_at="2026-01-01T00:00:00+00:00")
    base.update(kw)
    return rdb.commit_airing(db, **base)


def test_wal_mode_and_atomic_rollback(env):
    with get_db(env) as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] >= 10_000
    assert _airing(env) is not None
    with pytest.raises(sqlite3.IntegrityError):                       # unknown ad_id: FOREIGN KEY fails mid-transaction
        _airing(env, ad_id=999, airing_start=5000.0, airing_end=5030.0)
    with get_db(env) as conn:
        n = conn.execute("SELECT COUNT(*) FROM ad_occurrences").fetchone()[0]
        mc = conn.execute("SELECT match_count FROM reference_signatures WHERE asset_id='A'").fetchone()[0]
    assert (n, mc) == (1, 1)                                          # nothing half-written


def test_reader_never_sees_a_half_committed_airing(env):
    """A writer holds its transaction open; a concurrent reader (WAL) sees the old state, then the whole new one."""
    started, release = threading.Event(), threading.Event()

    def writer():
        conn = init_db(env)
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("INSERT INTO ad_occurrences (ad_id, source_name, airing_start, airing_end, duration_detected, "
                         "confidence_score) VALUES (1, 'live:Ch', 1.0, 31.0, 30, 0.99)")
            conn.execute("UPDATE reference_signatures SET match_count = match_count + 1 WHERE asset_id = 'A'")
            started.set()
            release.wait(10)
            conn.commit()
        finally:
            conn.close()
    t = threading.Thread(target=writer)
    t.start()
    assert started.wait(10)
    with get_db(env) as conn:                                         # does not block, sees neither change
        assert conn.execute("SELECT COUNT(*) FROM ad_occurrences").fetchone()[0] == 0
        assert conn.execute("SELECT match_count FROM reference_signatures WHERE asset_id='A'").fetchone()[0] == 0
    release.set()
    t.join(10)
    with get_db(env) as conn:
        assert conn.execute("SELECT COUNT(*) FROM ad_occurrences").fetchone()[0] == 1
        assert conn.execute("SELECT match_count FROM reference_signatures WHERE asset_id='A'").fetchone()[0] == 1


def test_concurrent_writers_do_not_lose_or_duplicate_airings(env):
    errors, results = [], []

    def work(k):
        try:
            for j in range(8):
                results.append(_airing(env, airing_start=10_000.0 + 100 * (k * 8 + j),
                                       airing_end=10_030.0 + 100 * (k * 8 + j)))
            results.append(_airing(env, airing_start=500.0, airing_end=530.0))        # all threads race on the same airing
        except Exception as exc:                                                       # noqa: BLE001
            errors.append(exc)
    ts = [threading.Thread(target=work, args=(k,)) for k in range(4)]
    [t.start() for t in ts]
    [t.join(30) for t in ts]
    assert errors == []
    with get_db(env) as conn:
        n = conn.execute("SELECT COUNT(*) FROM ad_occurrences").fetchone()[0]
        mc = conn.execute("SELECT match_count FROM reference_signatures WHERE asset_id='A'").fetchone()[0]
    assert n == 33 and mc == 33                                       # 32 distinct + the contested one exactly once
    assert sum(r is None for r in results) == 3


# ----------------------------------------------------------------------------- registry build / CLI
def write_wav(path, x, sr=SR):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())


@pytest.fixture
def registry_db(tmp_path):
    db = tmp_path / "reg.db"
    audio = {"G1": synth(11, 20.0), "G2": synth(12, 15.0)}
    with get_db(db) as conn:
        for ad_id, (aid, brand, sector) in enumerate((("G1", "Alpha", "Dairy"), ("G2", "Beta", None)), 1):
            conn.execute("INSERT INTO reference_ads (ad_id, brand_name, campaign_name, duration_sec, fingerprint, fp_algorithm, "
                         "industry) VALUES (?, ?, 'c', 20, x'00', 'x', ?)", (ad_id, brand, sector))
            # clip = 1.2 s leading silence + creative + 4 s of the NEXT spot (post-roll) beyond the nominal length
            clip = np.concatenate([np.zeros(int(1.2 * SR), np.float32), audio[aid], synth(50, 4.0)])
            write_wav(tmp_path / f"{aid}.wav", clip)
            conn.execute("INSERT INTO discovered_assets (asset_id, channel, duration_sec, nominal_duration_sec, fingerprint, "
                         "audio_path, status, promoted_ad_id) VALUES (?, 'Ch', ?, ?, x'00', ?, 'classified', ?)",
                         (aid, audio[aid].size / SR + 1.2, audio[aid].size / SR + 1.2, str(tmp_path / f"{aid}.wav"), ad_id))
        # not eligible: unclassified, classified without audio, and an exact duplicate of G1's audio
        conn.execute("INSERT INTO discovered_assets (asset_id, channel, duration_sec, fingerprint, audio_path, status) "
                     "VALUES ('U1', 'Ch', 20, x'00', ?, 'unclassified')", (str(tmp_path / "G1.wav"),))
        conn.execute("INSERT INTO discovered_assets (asset_id, channel, duration_sec, fingerprint, status, promoted_ad_id) "
                     "VALUES ('NOAUDIO', 'Ch', 20, x'00', 'classified', 1)")
    return db, tmp_path


def test_build_registry_only_clean_confirmed_assets(registry_db):
    db, _ = registry_db
    rep = rm.build_registry(db, dry_run=True)
    assert [b["asset_id"] for b in rep.built] == ["G1", "G2"]
    assert rep.skipped == [("NOAUDIO", "no cached audio")]
    g1 = rep.built[0]
    assert g1["lead_trim"] == pytest.approx(1.2 - 0.04, abs=0.05) and g1["trimmed"] < 20.5   # post-roll cut, silence stripped
    with get_db(db) as conn:
        rdb.ensure_schema(conn)
        assert conn.execute("SELECT COUNT(*) FROM reference_signatures").fetchone()[0] == 0   # dry run wrote nothing
    rep = rm.build_registry(db)
    sigs = {s.asset_id: s for s in rdb.load_signatures(db)}
    assert set(sigs) == {"G1", "G2"} and sigs["G1"].brand_name == "Alpha" and sigs["G1"].canonical_sector == "Dairy"
    assert sigs["G2"].canonical_sector is None and sigs["G1"].codes.size > 500
    # rebuild keeps statistics
    with get_db(db) as conn:
        conn.execute("UPDATE reference_signatures SET match_count = 7 WHERE asset_id = 'G1'")
    rm.build_registry(db)
    assert {s.asset_id: s.match_count for s in rdb.load_signatures(db)}["G1"] == 7


def test_built_signature_matches_a_realistic_stream(registry_db):
    db, tmp = registry_db
    rm.build_registry(db)
    audio = np.concatenate([noise(5), synth(11, 20.0), noise(5)])
    out, _ = run_stream(rm.ReferenceIndex.from_db(db), audio)
    assert [c.asset_id for c in out] == ["G1"] and out[0].confidence >= 0.96
    # the reference was cut after 1.2 s of leading silence; the snapped start gives that silence back (5.0 s - 1.16 s)
    assert out[0].airing_start - 1_700_000_000.0 == pytest.approx(5.0 - 1.16, abs=0.15)


def test_build_registry_flags_duplicate_audio(registry_db):
    db, tmp = registry_db
    with get_db(db) as conn:
        conn.execute("INSERT INTO discovered_assets (asset_id, channel, duration_sec, nominal_duration_sec, fingerprint, "
                     "audio_path, status, promoted_ad_id) VALUES ('G1B', 'Ch', 21.2, 21.2, x'00', ?, 'classified', 1)",
                     (str(tmp / "G1.wav"),))
    rep = rm.build_registry(db, dry_run=True)
    pairs = {(a, b): kind for a, b, _, kind, _ in rep.confusable}
    assert pairs[("G1", "G1B")] == pairs[("G1B", "G1")] == "duplicate" and not any("G2" in p for p in pairs)


def test_cli_build_reference_registry(registry_db, capsys):
    from kihc import cli
    db, _ = registry_db
    cli.main(["--db", str(db), "build-reference-registry", "--dry-run"])
    out = capsys.readouterr().out
    assert "Would build 2 signature(s), 1 skipped" in out and "Alpha" in out and "SKIPPED NOAUDIO" in out
    cli.main(["--db", str(db), "build-reference-registry"])
    assert "Built 2 signature(s)" in capsys.readouterr().out and len(rdb.load_signatures(db)) == 2


# ----------------------------------------------------------------------------- live worker wiring
def test_decimator_is_seamless():
    from kihc.core.reference_worker import Decimator
    x = (np.random.default_rng(3).standard_normal(22050 * 3) * 0.2).astype(np.float32)
    pcm = (x * 32768).astype(np.int16)
    d, parts, t_first = Decimator(), [], []
    for i in range(0, pcm.size, 4410):                                # the live worker's 0.2 s chunks
        y, t0 = d.process(pcm[i:i + 4410])
        if y.size:
            parts.append(y)
            t_first.append(t0)
    got = np.concatenate(parts)
    ref = decimate(pcm.astype(np.float32) / 32768.0)
    n = min(got.size, ref.size)
    assert n > 30000 and np.allclose(got[:n][60:], ref[:n][60:], atol=1e-5)    # identical away from the zero-padded start


def test_reference_worker_matches_live_audio_and_disables_legacy_matcher(env, ads):
    from kihc.core.reference_worker import ReferenceDiscoveryWorker
    w = ReferenceDiscoveryWorker("Ch Live", "fake://stream", env, reference=rm.ReferenceMatcher(env, compliance=False))
    assert w._ignored_ad_ids == {1}
    x22 = np.interp(np.arange(0, ads["A"].size * 2) / 2, np.arange(ads["A"].size), ads["A"])      # 11025 -> 22050
    stream = np.concatenate([np.zeros(22050 * 2), x22, np.zeros(22050 * 6)])
    pcm = (np.clip(stream, -1, 1) * 32767).astype(np.int16)
    t0, g, chunks = 1_700_000_000.0, 0, []
    for i in range(0, pcm.size, 4410):
        chunks.append(_Chunk(g, t0 + g / 22050, pcm[i:i + 4410]))
        g += 4410
        if len(chunks) % 15 == 0:                                      # one analysis pass every 3 s of audio
            w._feed_reference(chunks)
    w._feed_reference(chunks)
    w._restart_reference()                                             # connection ends
    rows = occurrences(env)
    assert len(rows) == 1 and rows[0]["source_name"] == "live:Ch Live"
    assert abs(rows[0]["airing_start"] - (t0 + 2.0)) < 0.2
    assert [c.decision for c in w.reference_events] == ["confirmed"]
