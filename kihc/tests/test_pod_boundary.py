"""Phase 20: multi-cue break-boundary sensing, strict first/last guards and conservative demotion."""
import time
from datetime import datetime
from decimal import Decimal

import numpy as np
import pytest

from kihc.analytics import compliance_audit as ca
from kihc.analytics import contracts as ctr
from kihc.analytics import pod_boundary_detector as pbd
from kihc.core.pod_analyzer import cluster_occurrences_into_pods
from kihc.core.proof import ALGERIA_TZ
from kihc.database.schema import get_db

CH = "El-Heddaf TV"
T = datetime(2026, 10, 13, 20, 0, tzinfo=ALGERIA_TZ).timestamp()
SR = 22050
LOUD, QUIET = 1843.0, 33.0                                   # ~ -25 dBFS programme / ~ -60 dBFS gap
CONTRACT = dict(contract_id="CTR-FIRST-001", brand_name="Mobilis", channel_id=CH, booked_position="FIRST_IN_POD",
                net_rate_per_second=10.0, separation_window_seconds=0, valid_from="2026-10-01", valid_to="2026-10-31")


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    monkeypatch.delenv("KIHC_STRICT_PLACEMENT", raising=False)
    pbd.reset()
    yield
    pbd.flush(5)
    pbd.reset()


def audio(seconds_levels, total, seed=1):
    """seconds_levels: [(t0, t1, amplitude)] relative to T, default LOUD. -> (wall0, int16 pcm @ 22.05 kHz)."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(int(total * SR))
    amp = np.full(x.size, LOUD)
    for a, b, level in seconds_levels:
        amp[int(a * SR):int(b * SR)] = level
    return (x * amp).astype(np.int16)


def feed_audio(pcm, t_from=T, chunk=0.2):
    n = int(chunk * SR)
    for i in range(0, pcm.size, n):
        pbd.feed_audio(CH, t_from + i / SR, pcm[i:i + n])


def feed_video(events, total, ticker_until=None, ticker_back=None, black=(), has_ticker=True, t_from=T):
    """2 fps frames; ticker present before `ticker_until` and from `ticker_back`; black frames at the given seconds."""
    det = pbd.get_detector(CH)
    for i in range(int(total * 2)):
        t = i / 2
        tk = None
        if has_ticker:
            tk = not (ticker_until is not None and t >= ticker_until and (ticker_back is None or t < ticker_back))
        luma = 3.0 if any(abs(t - b) < 0.01 for b in black) else 120.0
        det.feed_frame(t_from + t, luma, 0.0, tk)


def make_db(tmp_path, airings):
    db = tmp_path / "c.db"
    with get_db(db) as conn:
        conn.execute("INSERT INTO reference_ads (ad_id, brand_name, campaign_name, duration_sec, fingerprint, "
                     "fp_algorithm, industry, ad_type) VALUES (1, 'Mobilis', 'Spot', 30, x'00', 't', 'Telecom', 'self')")
        for start, end in airings:
            conn.execute("INSERT INTO ad_occurrences (ad_id, source_name, airing_start, airing_end, duration_detected, "
                         "confidence_score, human_verified) VALUES (1, ?, ?, ?, ?, 0.95, 0)",
                         (f"live:{CH}", T + start, T + end, end - start))
    return db


def settle(db, airings):
    for i, (s, e) in enumerate(airings, 1):
        pbd.note_airing(CH, i, T + s, T + e, db)
    assert pbd.flush(20)


def ad_of(db, occ=1):
    return next(a for p in cluster_occurrences_into_pods(db_path=db) for a in p.ads if a["occurrence_id"] == occ)


def break_scene(promo_before: bool, promo_after: bool):
    """Programme until 60 s; break edge at 60.0 (silence dip + black frame + ticker gone); optional untracked promos."""
    ad_s = 65.5 if promo_before else 60.5
    ad_e = ad_s + 30.0
    exit_t = ad_e + (5.5 if promo_after else 0.5)
    levels = [(60.0, 60.5, QUIET), (exit_t - 0.5, exit_t, QUIET), (exit_t + 3, 200, LOUD)]
    return ad_s, ad_e, exit_t, levels


def build(promo_before, promo_after, *, crossfade=False, has_ticker=True, black_entry=True):
    ad_s, ad_e, exit_t, levels = break_scene(promo_before, promo_after)
    total = exit_t + 8
    if crossfade:
        levels = []                                              # audio keeps rolling through both edges: no dip
    feed_audio(audio(levels, total))
    feed_video(None, total, ticker_until=60.0, ticker_back=exit_t, black=(60.0, exit_t) if black_entry else (),
               has_ticker=has_ticker)
    return ad_s, ad_e, exit_t


# ------------------------------------------------------------------------------------------ the headline cases
def test_untracked_promo_before_client_ad_makes_it_strictly_middle(tmp_path):
    s, e, _ = build(promo_before=True, promo_after=True)
    db = make_db(tmp_path, [(s, e)])
    settle(db, [(s, e)])
    ad = ad_of(db)
    assert ad["position"] == "MIDDLE" and ad["placement_status"] == "demoted_confirmed"
    assert "unclassified material" in ad["placement_note"]
    ctx = pbd.get_pod_boundary_context(T + s, CH)                       # the break edge IS proven, but 5.5 s away
    assert ctx["too_far"] and ctx["unclassified_material"] and not ctx["is_break_entry"]


def test_clean_break_first_and_last_spot_is_certified_with_cues(tmp_path):
    s, e, _ = build(promo_before=False, promo_after=False)
    db = make_db(tmp_path, [(s, e)])
    settle(db, [(s, e)])
    ad = ad_of(db)
    assert ad["position"] == "SOLO" and ad["placement_status"] == "confirmed"
    assert ad["boundary_confidence"] >= pbd.CERTIFY_CONFIDENCE
    assert {"audio_dip", "visual_dip", "ticker_absent"} <= set(ad["boundary_cues"])
    ctx = pbd.get_pod_boundary_context(T + s, CH)
    assert ctx["is_break_entry"] and ctx["boundary_confidence"] >= 0.95 and len(ctx["detected_cues"]) == 3


def test_crossfaded_audio_is_survived_by_visual_and_ticker_vote(tmp_path):
    s, e, _ = build(promo_before=False, promo_after=False, crossfade=True)
    ctx = pbd.get_pod_boundary_context(T + s, CH)
    assert "audio_dip" not in ctx["detected_cues"]                      # the cross-fade hid the audio cue
    assert set(ctx["detected_cues"]) == {"visual_dip", "ticker_absent"}
    assert ctx["is_break_entry"] and ctx["boundary_confidence"] >= pbd.CERTIFY_CONFIDENCE
    db = make_db(tmp_path, [(s, e)])
    settle(db, [(s, e)])
    assert ad_of(db)["placement_status"] == "confirmed"


def test_one_cue_is_not_enough_two_out_of_three_required(tmp_path):
    s, e, _ = build(promo_before=False, promo_after=False, crossfade=True, black_entry=False)   # ticker flip alone
    ctx = pbd.get_pod_boundary_context(T + s, CH)
    assert not ctx["is_break_entry"] and ctx["boundary_confidence"] == 0.0
    db = make_db(tmp_path, [(s, e)])
    settle(db, [(s, e)])
    assert ad_of(db)["placement_status"] == "demoted_ambiguous" and ad_of(db)["position"] == "MIDDLE"


def test_audio_plus_visual_without_ticker_feed_never_certifies(tmp_path):
    s, e, _ = build(promo_before=False, promo_after=False, has_ticker=False)
    ctx = pbd.get_pod_boundary_context(T + s, CH)
    assert 0 < ctx["boundary_confidence"] <= pbd.NO_TICKER_CAP < pbd.CERTIFY_CONFIDENCE and not ctx["is_break_entry"]
    db = make_db(tmp_path, [(s, e)])
    settle(db, [(s, e)])
    assert ad_of(db)["placement_status"] == "demoted_ambiguous"


def test_separator_inside_a_break_is_not_a_boundary():
    levels = [(60.0, 60.5, QUIET), (80.0, 80.5, QUIET)]                  # 2nd dip + black between two ads, ticker still gone
    feed_audio(audio(levels, 120))
    feed_video(None, 120, ticker_until=60.0, black=(60.0, 80.0))
    votes = pbd.get_detector(CH).boundaries(T + 70, T + 100)["votes"]
    assert votes == []                                                   # no ticker flip -> separator, not an edge
    assert pbd.get_pod_boundary_context(T + 80.5, CH)["boundary_confidence"] == 0.0


def test_state_machine_walks_programme_to_pod_and_back():
    s, e, exit_t = build(promo_before=False, promo_after=False)
    tl = pbd.get_detector(CH).state_timeline(T + 30, T + 120)
    assert [(x["from_state"], x["to_state"]) for x in tl] == [
        ("PROGRAMME", "POD_PRE_ROLL"), ("POD_PRE_ROLL", "IN_POD"), ("IN_POD", "POD_POST_ROLL"), ("POD_POST_ROLL", "PROGRAMME")]
    lone = pbd.BoundaryDetector("x")                                     # a lone ticker flip never reaches IN_POD
    for i in range(40):
        lone.feed_frame(T + i / 2, 120.0, 0.0, i < 10)
    assert [x["to_state"] for x in lone.state_timeline(T, T + 20)] == ["POD_PRE_ROLL"]


# ------------------------------------------------------------------------------------------ claims & alerts
def test_ambiguous_demotion_withholds_the_claim_confirmed_demotion_makes_it(tmp_path):
    s, e, _ = build(promo_before=True, promo_after=True)
    db = make_db(tmp_path, [(s, e)])
    settle(db, [(s, e)])
    ctr.ingest_contract(db, CONTRACT)
    row = next(r for r in ca.compliance_report(db)["airings"])
    assert row["position_compliance"] == "breach" and row["position_credit"] == Decimal("90.00")   # proven: promo opened it
    assert row["placement_status"] == "demoted_confirmed" and row["boundary_confidence"] >= 0.85

    pbd.reset()                                                          # same airing, cues too weak to prove anything
    build(promo_before=True, promo_after=True, has_ticker=False)
    with get_db(db) as conn:
        conn.execute("UPDATE ad_occurrences SET boundary_json = NULL")
    settle(db, [(s, e)])
    row = next(r for r in ca.compliance_report(db)["airings"])
    assert row["placement_status"] == "demoted_ambiguous"
    assert row["position_compliance"] == "unverified placement" and row["position_credit"] == Decimal("0.00")
    assert ca.compliance_report(db)["kpis"]["position_breaches"] == 0


def test_alert_metadata_carries_boundary_confidence_and_cues(tmp_path):
    import json
    s, e, _ = build(promo_before=True, promo_after=True)
    db = make_db(tmp_path, [(s, e)])
    settle(db, [(s, e)])
    ctr.ingest_contract(db, CONTRACT)
    ca.refresh_alerts(db, now=T + 3600)
    alert = next(a for a in ca.list_alerts(db) if a["kind"] == "position_breach")
    meta = json.loads(alert["metadata"])
    assert meta["placement_status"] == "demoted_confirmed" and meta["contract_id"] == "CTR-FIRST-001"
    assert meta["boundary_confidence"] >= 0.85 and "ticker_absent" in meta["boundary_cues"]


def test_unsensed_airings_keep_legacy_position_unless_strict(tmp_path, monkeypatch):
    db = make_db(tmp_path, [(60.5, 90.5)])                               # no audio / video ever fed for this channel
    ad = ad_of(db)
    assert ad["position"] == "SOLO" and ad["placement_status"] == "unsensed" and ad["boundary_confidence"] is None
    monkeypatch.setenv("KIHC_STRICT_PLACEMENT", "1")
    ad = ad_of(db)
    assert ad["position"] == "MIDDLE" and ad["placement_status"] == "demoted_ambiguous"


# ------------------------------------------------------------------------------------------ analysis helpers & perf
def test_frame_analysis_black_ticker_and_cut_contrast():
    det = pbd.BoundaryDetector("y")
    rng = np.random.default_rng(0)
    base = rng.integers(90, 130, (36, 64)).astype(np.float32)
    with_ticker = base.copy()
    with_ticker[30:35, :] = (np.arange(64) % 2) * 255.0                 # high-contrast text bar in the lower third
    assert det.analyze_frame(T, with_ticker)[2] is True
    assert det.analyze_frame(T + 0.5, base)[2] is False
    luma, cut, _ = det.analyze_frame(T + 1.0, np.zeros((36, 64)), ticker_band=None)
    assert luma < pbd.LUMA_BLACK and cut > pbd.CUT_SPIKE
    assert det.analyze_frame(T + 1.5, base, ticker_band=None)[2] is None


def test_ffmpeg_sampler_reads_black_at_zero(tmp_path):
    import subprocess

    from kihc.core.audio import ffmpeg_exe
    clip = tmp_path / "blk.mp4"
    subprocess.run([ffmpeg_exe(), "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=128x72:d=2:r=10",
                    "-pix_fmt", "yuv420p", str(clip)], check=True)
    frames = list(pbd.sample_frames(str(clip), T))
    assert 3 <= len(frames) <= 5 and all(f[1] < pbd.LUMA_BLACK for f in frames)   # limited-range black read as ~0
    assert frames[1][0] - frames[0][0] == pytest.approx(0.5)


def test_benchmark_latency_and_memory():
    total = 1200                                                         # a full 20 min buffer of audio + frames
    det = pbd.get_detector(CH)
    rng = np.random.default_rng(2)
    for i in range(int(total / 0.2)):
        det.feed_audio(T + i * 0.2, (rng.standard_normal(4410) * LOUD).astype(np.int16))
    for i in range(int(total * 2)):
        det.feed_frame(T + i / 2, 120.0, 0.0, i < 2000 or i > 2100)
    t0 = time.perf_counter()
    for k in range(20):
        det.context(T + 1000 + k, "entry")
    ctx_ms = (time.perf_counter() - t0) / 20 * 1000
    t0 = time.perf_counter()
    det.state_timeline(T + 900, T + 1200)
    tl_ms = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    for i in range(50):
        det.feed_audio(T + 2000 + i * 0.2, (rng.standard_normal(4410) * LOUD).astype(np.int16))
    feed_ms = (time.perf_counter() - t0) / 50 * 1000
    mem = det.memory_bytes() / 1e6
    print(f"\nBOUNDARY BENCHMARK  context() {ctx_ms:.1f} ms | 5-min state timeline {tl_ms:.1f} ms | "
          f"feed_audio per 0.2 s chunk {feed_ms:.2f} ms | detector buffers ~{mem:.1f} MB for 20 min")
    assert ctx_ms < 250 and feed_ms < 10 and mem < 20
