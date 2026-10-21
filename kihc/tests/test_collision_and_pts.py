"""Phase 22: royalty-free audio collision guard (voice ratio + mandatory packshot) and broadcast-clock (PTS) calibration."""
import struct
import time
from datetime import datetime, timezone

import numpy as np
import pytest
from fastapi.testclient import TestClient

from kihc.analytics import brand_extractor as be
from kihc.analytics import certificate_authority as cert
from kihc.analytics import compliance_audit as ca
from kihc.analytics import reference_matcher as rm
from kihc.app.api_server import create_app
from kihc.data import reference_db as rdb
from kihc.database.schema import get_db
from kihc.stream import clock_sync as cs
from kihc.tests.test_api_server import FakeChannel, FakeWorker
from kihc.tests.test_compliance import CH, T, _ad, _air
from kihc.tests.test_reference_matcher import noise, synth

SR = rm.SR
EPOCH = 1_700_000_000.0
PH_A, PH_B = "f0f0f0f0f0f0f0f0", "0f0f0f0f0f0f0f0f"          # 64 bits apart


def flip(phash: str, bits: int) -> str:
    return f"{int(phash, 16) ^ ((1 << bits) - 1):016x}"


@pytest.fixture(autouse=True)
def _fresh_clocks():
    cs.reset()
    yield
    cs.reset()


# ============================================================================================ voice ratio
def test_voice_ratio_edge_cases_and_threshold():
    assert be.compute_voice_ratio(np.zeros(SR * 5, np.float32)) == 0.0              # silence
    assert be.compute_voice_ratio(np.zeros(100, np.float32)) == 0.0                 # shorter than one frame
    assert be.compute_voice_ratio(noise(6, amp=0.1)) == 0.0                         # broadband hiss is not a voice
    r = be.compute_voice_ratio(synth(3, 8.0))
    assert 0.0 <= r <= 1.0
    assert be.is_music_dominant(0.19) and not be.is_music_dominant(0.20) and not be.is_music_dominant(None)


def test_voice_ratio_costs_milliseconds_per_spot():
    x = synth(4, 30.0)
    t0 = time.perf_counter()
    be.compute_voice_ratio(x)
    ms = (time.perf_counter() - t0) * 1000
    print(f"\n[bench] compute_voice_ratio(30 s spot): {ms:.1f} ms")
    assert ms < 400                                  # registration-time only; never on the live audio path


def test_flag_is_stored_kept_on_rebuild_and_loaded(tmp_path):
    blob = rdb.pack_blob(np.array([1], np.uint32), np.array([5], np.uint16), 0.0, 1.0)
    with get_db(tmp_path / "v.db") as conn:
        rdb.ensure_schema(conn)
        rdb.upsert_signature(conn, "M", "B", "S", 15.0, blob, None, voice_ratio=0.07, music_dominant=True)
        rdb.upsert_signature(conn, "M", "B", "S", 15.0, blob, None)                  # a rebuild that passes neither
    sig = rdb.load_signatures(tmp_path / "v.db")[0]
    assert sig.is_music_dominant is True and sig.voice_ratio == pytest.approx(0.07)
    assert rm.ReferenceIndex([sig]).meta[0].music_dominant is True


def test_register_ad_measures_and_honours_the_override(tmp_path):
    from kihc.analytics import ad_onboarding as ob
    from kihc.tests.test_ad_onboarding import make_mp4, master_audio
    full, _body = master_audio(5)
    f = tmp_path / "a.mp4"
    make_mp4(f, full)
    auto = ob.register_ad(f, "BrandM", "Telecom", db_path=tmp_path / "r.db")
    assert 0.0 <= auto["voice_ratio"] <= 1.0 and auto["music_dominant"] == be.is_music_dominant(auto["voice_ratio"])
    forced = ob.register_ad(f, "BrandN", "Telecom", db_path=tmp_path / "r.db", allow_duplicate=True, music_dominant=True)
    assert forced["music_dominant"] is True
    got = {s.asset_id: s for s in rdb.load_signatures(tmp_path / "r.db")}
    assert got[forced["asset_id"]].is_music_dominant is True


# ============================================================================================ collision guard
MUSIC = synth(21, 15.0)              # one royalty-free 15 s track, sold to two brands


def two_brand_db(tmp_path, music_dominant=True):
    db = tmp_path / "c.db"
    codes, t_cs, lead, trimmed = rm.signature_from_audio(MUSIC)
    blob = rdb.pack_blob(codes, t_cs, lead, trimmed)
    with get_db(db) as conn:
        rdb.ensure_schema(conn)
        for i, (asset, brand, ph) in enumerate((("SPOT_A", "BrandA", PH_A), ("SPOT_B", "BrandB", PH_B)), 1):
            conn.execute("INSERT INTO reference_ads (ad_id, brand_name, campaign_name, duration_sec, fingerprint, "
                         "fp_algorithm, industry) VALUES (?, ?, 'music', 15, x'00', 't', 'Telecom')", (i, brand))
            rdb.upsert_signature(conn, asset, brand, "Telecom", 15.0, blob, ph, ad_id=i, voice_ratio=0.02,
                                 music_dominant=music_dominant)
    return db


def run_stream(db, grabber):
    rmx = rm.ReferenceMatcher(db, frame_grabber=grabber, compliance=False)
    out = rmx.feed("Ch One", np.concatenate([noise(2), MUSIC, noise(6)]), EPOCH) + rmx.end_stream("Ch One")
    return rmx, out


def rows(db):
    with get_db(db) as conn:
        return [dict(r) for r in conn.execute(
            "SELECT o.*, r.brand_name FROM ad_occurrences o JOIN reference_ads r USING (ad_id) ORDER BY occurrence_id")]


def test_other_brands_packshot_prevents_false_cross_attribution(tmp_path):
    """Identical audio is registered for BrandA and BrandB. Whichever the matcher picks, a screen showing the OTHER brand's
    packshot must not be attributed to it."""
    db = two_brand_db(tmp_path)
    other = {"SPOT_A": PH_B, "SPOT_B": PH_A}
    rmx, out = run_stream(db, lambda ch, at, asset_id=None: other[asset_id])
    assert out and {c.decision for c in out} == {"rejected_visual"} and all(c.visual_distance == 64 for c in out)
    assert rows(db) == [] and rmx.stats["rejected_visual"] == len(out)


def test_own_packshot_confirms_even_at_exact_acoustic_match(tmp_path):
    db = two_brand_db(tmp_path)
    own = {"SPOT_A": PH_A, "SPOT_B": PH_B}
    seen = []
    rmx, out = run_stream(db, lambda ch, at, asset_id=None: seen.append(asset_id) or own[asset_id])
    assert seen, "a music-dominant spot must ask for the packshot even though confidence is >= 0.96"
    assert [c.decision for c in out] == ["confirmed_visual"] and out[0].visual_distance == 0
    r = rows(db)
    assert len(r) == 1 and r[0]["status"] == "CONFIRMED" and r[0]["brand_name"] == {"SPOT_A": "BrandA", "SPOT_B": "BrandB"}[out[0].asset_id]


def test_music_dominant_without_any_packshot_is_logged_but_never_confirmed(tmp_path):
    db = two_brand_db(tmp_path)
    rmx, out = run_stream(db, None)
    assert [c.decision for c in out] == ["confirmed"] and out[0].status == "UNCONFIRMED_MUSIC_NO_PACKSHOT"
    assert [r["status"] for r in rows(db)] == ["UNCONFIRMED_MUSIC_NO_PACKSHOT"]
    assert rmx.stats["unconfirmed_cutdown"] == 1


def test_speech_led_spot_is_unchanged_by_the_guard(tmp_path):
    db = two_brand_db(tmp_path, music_dominant=False)
    called = []
    rmx, out = run_stream(db, lambda *a, **k: called.append(1) or PH_B)
    assert [c.decision for c in out] == ["confirmed"] and not called               # acoustic alone, grabber never touched
    assert [r["status"] for r in rows(db)] == ["CONFIRMED"]


@pytest.mark.parametrize("bits,music,expected", [(12, True, "confirmed_visual"), (13, True, "rejected_visual"),
                                                 (10, False, "confirmed_visual"), (11, False, "rejected_visual")])
def test_hamming_limits_are_12_bits_for_music_and_10_for_the_rest(tmp_path, bits, music, expected):
    db = two_brand_db(tmp_path, music_dominant=music)
    cand = rm.Candidate("SPOT_A", "BrandA", "Telecom", 15.0, 0.90 if not music else 0.99, EPOCH, EPOCH + 15, 400, 0.9, 14.0,
                        1.0, 5, True, packshot_at=EPOCH + 13.5)
    rmx = rm.ReferenceMatcher(db, frame_grabber=lambda *a, **k: flip(PH_A, bits), compliance=False)
    assert rmx._gate("Ch One", cand).decision == expected


# ============================================================================================ cut-down policy
def family_db(tmp_path, cut_phash=PH_A):
    db = tmp_path / "f.db"
    master = synth(11, 30.0) + noise(30, amp=0.03, seed=3)       # a production-like bed: no digital gaps inside the spot
    blob = lambda x: rdb.pack_blob(*rm.signature_from_audio(x))                         # noqa: E731
    with get_db(db) as conn:
        rdb.ensure_schema(conn)
        for i, (asset, dur, name) in enumerate((("MASTER_30", 30, "Ramadan"), ("CUT_15", 15, "Ramadan [CUT_DOWN]")), 1):
            conn.execute("INSERT INTO reference_ads (ad_id, brand_name, campaign_name, duration_sec, fingerprint, "
                         "fp_algorithm, industry) VALUES (?, 'Djezzy', ?, ?, x'00', 't', 'Telecom')", (i, name, dur))
        rdb.upsert_signature(conn, "MASTER_30", "Djezzy", "Telecom", 30.0, blob(master), None, ad_id=1)
        rdb.upsert_signature(conn, "CUT_15", "Djezzy", "Telecom", 15.0, blob(master[:15 * SR]), cut_phash, ad_id=2,
                             master_asset_id="MASTER_30", cut_type="CUT_DOWN")
    return db, master[:15 * SR]


def feed_live(rmx, audio, piece=0.2):
    """Feed like the live worker does: consecutive short chunks on a continuous clock."""
    step, out = int(piece * SR), []
    for i in range(0, audio.size, step):
        out += rmx.feed("Ch One", audio[i:i + step], EPOCH + i / SR)
    return out


def no_dip_audio(cut):
    """The cut-down runs straight into other content at the SAME level: nothing dips at its trailing edge."""
    rms = float(np.sqrt(np.mean(cut[-2 * SR:] ** 2)))
    other = noise(18, amp=1.0, seed=7)
    return np.concatenate([noise(3), cut, other * (rms / float(other.std()))])


def test_cutdown_without_a_dip_is_logged_unconfirmed_and_visual_resolution_confirms_it(tmp_path):
    db, cut = family_db(tmp_path)
    rmx = rm.ReferenceMatcher(db, compliance=False)                                    # no video at all
    out = feed_live(rmx, no_dip_audio(cut)) + rmx.end_stream("Ch One")
    assert [c.asset_id for c in out] == ["CUT_15"] and out[0].no_dip
    r = rows(db)
    assert len(r) == 1 and r[0]["status"] == "UNCONFIRMED_CUTDOWN_NO_DIP"             # logged, not dropped

    (tmp_path / "x").mkdir()
    db2, cut = family_db(tmp_path / "x")
    rmx2 = rm.ReferenceMatcher(db2, frame_grabber=lambda *a, **k: PH_A, compliance=False)   # packshot agrees
    feed_live(rmx2, no_dip_audio(cut))
    rmx2.end_stream("Ch One")
    assert [r["status"] for r in rows(db2)] == ["CONFIRMED"]


def test_cutdown_with_the_dip_is_confirmed_outright(tmp_path):
    db, cut = family_db(tmp_path)
    rmx = rm.ReferenceMatcher(db, compliance=False)
    out = feed_live(rmx, np.concatenate([noise(3), cut, noise(4, amp=3e-4, seed=5), noise(6)])) + rmx.end_stream("Ch One")
    assert [c.asset_id for c in out] == ["CUT_15"] and not out[0].no_dip
    assert [r["status"] for r in rows(db)] == ["CONFIRMED"]


def test_unconfirmed_airings_raise_no_penalty_claims(tmp_path):
    db = tmp_path / "p.db"
    with get_db(db) as conn:
        _ad(conn, 1, "Mobilis", "Telecom", role="self")
        _ad(conn, 2, "Djezzy", "Telecom")
    ca.save_contracts(db, [dict(brand="Mobilis", competitor_separation_seconds=180, separation_credit=1.0)])
    with get_db(db) as conn:
        _air(conn, 1, T)
        _air(conn, 2, T + 90)                               # 60 s after Mobilis ended: a separation breach
    assert ca.refresh_alerts(db, lookback_days=1, now=T + 500) >= 1
    with get_db(db) as conn:
        conn.execute("DELETE FROM compliance_alerts")
        conn.execute("UPDATE ad_occurrences SET status = 'UNCONFIRMED_CUTDOWN_NO_DIP' WHERE ad_id = 1")
    assert ca.refresh_alerts(db, lookback_days=1, now=T + 500) == 0         # until visually resolved
    with get_db(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM ad_occurrences").fetchone()[0] == 2     # still on record
    assert rdb.set_airing_status(db, 1, "CONFIRMED") is True
    assert ca.refresh_alerts(db, lookback_days=1, now=T + 500) >= 1
    with pytest.raises(ValueError):
        rdb.set_airing_status(db, 1, "MAYBE")


# ============================================================================================ PTS / PCR parsing
def ts_packet(pid, *, pcr=None, pts=None, pusi=False):
    head = bytes([0x47, (0x40 if pusi else 0) | (pid >> 8), pid & 0xFF])
    if pcr is not None:                                  # adaptation-field-only packet carrying a PCR
        pcr_b = struct.pack(">Q", (int(pcr * 90_000) << 15) | (0x3F << 9))[2:]
        return head + bytes([0x20, 183, 0x10]) + pcr_b + bytes([0xFF]) * 176
    p = int(pts * 90_000)
    pes = (b"\x00\x00\x01\xe0\x00\x00\x80\x80\x05" +
           bytes([0x21 | ((p >> 30) & 7) << 1, (p >> 22) & 0xFF, ((p >> 15) & 0x7F) << 1 | 1, (p >> 7) & 0xFF,
                  (p & 0x7F) << 1 | 1]))
    return head + bytes([0x10]) + pes.ljust(184, b"\xff")


def test_pcr_and_pts_are_parsed_from_raw_mpeg_ts():
    data = b"\x01\x02\x03" + ts_packet(0x100, pcr=1234.5) + ts_packet(0x101, pts=1234.64, pusi=True) + ts_packet(0x101, pts=9.0)
    # garbage prefix is skipped; sync is only trusted when three packets line up
    got = cs.ts_clock_refs(data)
    assert got["pcr_s"] == pytest.approx(1234.5, abs=1e-5) and got["pts_s"] == pytest.approx(1234.64, abs=1e-5)
    assert got["packets"] >= 2
    big = 2 ** 33 / 90_000 - 1.0                                                      # near the 33-bit wrap
    assert cs.ts_clock_refs(ts_packet(0x101, pts=big, pusi=True) * 3)["pts_s"] == pytest.approx(big, abs=1e-4)


def test_ts_parser_survives_garbage_and_truncation():
    assert cs.ts_clock_refs(b"") == dict(pcr_s=None, pts_s=None, packets=0)
    assert cs.ts_clock_refs(b"\x00" * 5000)["pts_s"] is None
    assert cs.ts_clock_refs(ts_packet(0x101, pts=5.0, pusi=True)[:100])["pts_s"] is None   # truncated packet


PLAYLIST = """#EXTM3U
#EXT-X-TARGETDURATION:6
#EXT-X-MEDIA-SEQUENCE:134531
#EXT-X-PROGRAM-DATE-TIME:2026-10-20T22:09:56.010Z
#EXTINF:6.000,
2026/10/20/22/09/56-06000.ts
#EXTINF:6.000,
2026/10/20/22/10/02-06000.ts
#EXTINF:6.000,
2026/10/20/22/10/08-06000.ts
#EXTINF:6.000,
2026/10/20/22/10/14-06000.ts
"""
PDT0 = datetime(2026, 10, 20, 22, 9, 56, 10000, tzinfo=timezone.utc).timestamp()      # 2026-10-20T22:09:56.010Z


def test_playlist_program_date_time_is_propagated_and_live_start_index_applied():
    segs = cs.parse_playlist(PLAYLIST)
    assert [s.start_epoch for s in segs] == pytest.approx([PDT0, PDT0 + 6, PDT0 + 12, PDT0 + 18], abs=1e-3)
    assert cs.segment_at(segs, -2).uri.endswith("22/10/08-06000.ts")                    # ffmpeg -live_start_index -2
    assert cs.segment_at([], -2) is None
    assert cs.parse_playlist("#EXTM3U\n#EXTINF:6.0,\na.ts\n")[0].start_epoch is None     # no PDT: unknown, not guessed


# ============================================================================================ drift storage
def test_calibrator_offset_margin_and_implausible_readings():
    cal = cs.get_calibrator("Ch One")
    assert cal.snapshot() is None
    for d in (14.0, 14.2, 13.9, 14.1, 14.0):
        cal.observe(1000.0 + d, 1000.0, rtt_s=0.1)
    s = cal.snapshot()
    assert s["offset_s"] == pytest.approx(14.0, abs=0.01) and s["samples"] == 5 and s["basis"] == "program-date-time"
    assert 0.2 <= s["margin_s"] <= 0.5
    assert cal.observe(1000.0, 5000.0) is None and cal.observe(9000.0, 1000.0) is None      # absurd (> 10 min): ignored
    assert cal.observe(1000.0, 1010.0) == pytest.approx(-10.0)                       # broadcaster clock ahead: kept, signed
    assert cal.snapshot()["samples"] == 6


def test_pts_only_source_reports_relative_drift():
    cal = cs.get_calibrator("Ch Two")
    cal.observe_pts(100.0, 5000.0)                    # reference point
    cal.observe_pts(130.0, 5029.0)                    # 30 s of wall time, only 29 s of PTS: the buffer grew by 1 s
    s = cal.snapshot()
    assert s["basis"] == "pts-relative" and s["samples"] == 2 and s["offset_s"] == pytest.approx(1.0)


def test_ffmpeg_log_line_names_the_segment_it_really_started_with():
    line = "[hls @ 0000021ce2ca0640] [info] Opening 'https://x.example/a/2026/10/20/22/29/20-06000.ts' for reading"
    assert cs.opened_segment(line) == "https://x.example/a/2026/10/20/22/29/20-06000.ts"
    assert cs.opened_segment("[hls @ 0x1] [info] Opening 'https://x.example/a/mono.m3u8' for reading") is None   # playlists aren't segments
    assert cs.opened_segment("[error] Connection reset") is None


def test_connect_probe_calibrates_from_the_playlist_and_publishes(tmp_path, monkeypatch):
    class Resp:
        ok, text, content = True, PLAYLIST, b""
    monkeypatch.setattr(cs.requests, "get", lambda *a, **k: Resp())
    db = tmp_path / "t.db"
    probe = cs.probe_for("El-Heddaf TV", "https://x.example/a/mono.m3u8", db)
    assert probe is not None and cs.probe_for("X", "rtmp://x/live", db) is None and cs.probe_for("X", "a.wav") is None
    probe.connecting()
    probe.segment_opened("https://x.example/a/2026/10/20/22/10/08-06000.ts")           # ffmpeg started on the 3rd segment
    probe.segment_opened("https://x.example/a/2026/10/20/22/10/14-06000.ts")           # later segments don't re-anchor
    assert probe._ready.wait(3)
    probe._calibrate(PDT0 + 12 + 3.2)                  # first audio arrived 3.2 s after that segment's program-date-time
    s = cs.get_calibrator("El-Heddaf TV").snapshot()
    assert s["offset_s"] == pytest.approx(3.2, abs=1e-3) and s["basis"] == "program-date-time"
    saved = cs.load_all(db)
    assert len(saved) == 1 and saved[0]["channel"] == "El-Heddaf TV" and saved[0]["offset_s"] == pytest.approx(3.2, abs=1e-3)


def test_broadcaster_clock_ahead_of_ntp_is_a_signed_negative_offset(tmp_path, monkeypatch):
    class Resp:
        ok, text, content = True, PLAYLIST, b""
    monkeypatch.setattr(cs.requests, "get", lambda *a, **k: Resp())
    probe = cs.ConnectProbe("Ahead TV", "https://x.example/a/mono.m3u8", tmp_path / "n.db")
    probe.connecting()
    probe.segment_opened("https://x.example/a/2026/10/20/22/10/02-06000.ts")
    probe._ready.wait(3)
    probe._calibrate(PDT0 + 6 - 12.5)                   # the audio reached us 12.5 s BEFORE its labelled broadcaster time
    s = cs.get_calibrator("Ahead TV").snapshot()
    assert s["offset_s"] == pytest.approx(-12.5, abs=1e-3)
    text = cert.drift_text(dict(clock_drift_seconds=-12.5, clock_drift_margin_s=0.8, clock_drift_basis="program-date-time",
                                airing_start=EPOCH))
    assert "12.5 s ahead of the NTP capture clock" in text and "± 0.8 s broadcast latency" in text


def test_connect_probe_falls_back_to_segment_pts_without_pdt(tmp_path, monkeypatch):
    class Resp:
        ok = True
        text = "#EXTM3U\n#EXTINF:6.0,\na.ts\n#EXTINF:6.0,\nb.ts\n#EXTINF:6.0,\nc.ts\n"
        content = ts_packet(0x101, pts=777.0, pusi=True) * 3
    monkeypatch.setattr(cs.requests, "get", lambda *a, **k: Resp())
    probe = cs.ConnectProbe("Ch P", "https://x.example/p.m3u8", tmp_path / "p.db")
    probe.connecting()
    probe.segment_opened("https://x.example/b.ts")
    probe._ready.wait(3)
    probe._calibrate(2000.0)
    assert cs.get_calibrator("Ch P").snapshot()["basis"] == "pts-relative"


def test_segment_that_left_the_playlist_gives_no_reading_instead_of_a_guess(tmp_path, monkeypatch):
    class Resp:
        ok, text, content = True, PLAYLIST, b""
    monkeypatch.setattr(cs.requests, "get", lambda *a, **k: Resp())
    probe = cs.ConnectProbe("Gone TV", "https://x.example/a/mono.m3u8", tmp_path / "g.db")
    probe.connecting()
    probe.segment_opened("https://x.example/a/2026/10/20/21/00/00-06000.ts")
    probe._ready.wait(3)
    probe._calibrate(PDT0 + 5)
    assert cs.get_calibrator("Gone TV").snapshot() is None


def test_unmeasured_clock_is_stored_as_null_never_zero(tmp_path):
    db = two_brand_db(tmp_path, music_dominant=False)
    run_stream(db, None)
    r = rows(db)[0]
    assert r["clock_drift_seconds"] is None and r["clock_drift_margin_s"] is None and r["clock_drift_basis"] is None


def test_airing_row_stores_the_clock_drift_and_certificate_prints_the_margin(tmp_path):
    db = two_brand_db(tmp_path, music_dominant=False)
    cal = cs.get_calibrator("Ch One")
    for d in (3.0, 3.1, 2.9, 3.0):
        cal.observe(EPOCH + d, EPOCH, rtt_s=2.0)                                     # rtt/2 dominates the margin: 1.0 s
    _, out = run_stream(db, None)
    r = rows(db)[0]
    assert r["clock_drift_seconds"] == pytest.approx(3.0, abs=0.01) and r["clock_drift_margin_s"] == pytest.approx(1.1, abs=0.05)
    assert r["clock_drift_basis"] == "program-date-time"
    text = cert.drift_text(r)
    assert "± 1.1 s broadcast latency" in text and "3.0 s" in text and "program-date-time" in text
    assert "not measured" in cert.drift_text(dict(clock_drift_seconds=None, clock_drift_margin_s=None))
    pdf = cert.generate_airing_certificate_pdf(r["occurrence_id"], db)
    assert pdf.startswith(b"%PDF-") and b"broadcast latency" in pdf and b"AIRING STATUS" in pdf


def test_certificate_flags_an_unconfirmed_status(tmp_path):
    db = two_brand_db(tmp_path)
    run_stream(db, None)                                                          # music-dominant, no packshot
    r = rows(db)[0]
    assert "UNCONFIRMED MUSIC NO PACKSHOT" in cert.status_text(r) and "excluded from penalty claims" in cert.status_text(r)
    assert b"UNCONFIRMED" in cert.generate_airing_certificate_pdf(r["occurrence_id"], db)


def test_telemetry_exposes_the_calibrated_offset(tmp_path):
    db = tmp_path / "api.db"
    cal = cs.get_calibrator("El-Heddaf TV")
    cal.observe(EPOCH + 14.0, EPOCH, rtt_s=0.2)
    cs.persist(db, "El-Heddaf TV")
    app = create_app(db, worker_factory=FakeWorker, channel_loader=lambda: [FakeChannel("TV2", "http://x/index.m3u8")])
    with TestClient(app) as c:
        body = c.get("/api/telemetry").json()
    clock = body["clock"]
    assert len(clock) == 1 and clock[0]["channel"] == "El-Heddaf TV" and clock[0]["offset_s"] == pytest.approx(14.0)
    assert clock[0]["margin_s"] >= 0.2 and clock[0]["basis"] == "program-date-time" and clock[0]["age_s"] >= 0


def test_bulk_fed_cutdown_gets_the_same_trailing_edge_policy(tmp_path):
    """A file scan / flush decides without waiting for the boundary; the dip is still measured, never assumed."""
    db, cut = family_db(tmp_path)
    rmx = rm.ReferenceMatcher(db, compliance=False)
    out = rmx.feed("Ch One", no_dip_audio(cut), EPOCH) + rmx.end_stream("Ch One")
    assert [(c.asset_id, c.status) for c in out] == [("CUT_15", "UNCONFIRMED_CUTDOWN_NO_DIP")]
    assert "without a level dip" in out[0].family_note
    (tmp_path / "y").mkdir()
    db2, cut = family_db(tmp_path / "y")
    rmx2 = rm.ReferenceMatcher(db2, compliance=False)
    out2 = rmx2.feed("Ch One", np.concatenate([noise(3), cut, noise(4, amp=3e-4, seed=5), noise(6)]), EPOCH) + rmx2.end_stream("Ch One")
    assert [(c.asset_id, c.status) for c in out2] == [("CUT_15", "CONFIRMED")] and "level dip" in out2[0].family_note
