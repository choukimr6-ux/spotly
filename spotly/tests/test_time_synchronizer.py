"""Phase 26: MCR timecode synchronisation - PDT primary, PTS+EMA fallback, calibrated columns and daypart evaluation."""
import json
import random
import statistics
from datetime import datetime

import pytest

from spotly.analytics import certificate_authority as cert
from spotly.analytics.financial_intelligence import build_financial_market_report
from spotly.analytics.rate_engine import DEFAULT_PATH, RateCardEngine
from spotly.core.proof import ALGERIA_TZ
from spotly.database.schema import get_db
from spotly.stream import clock_sync
from spotly.stream import time_synchronizer as ts

CH = "El-Heddaf TV"


def at(h, mi=0, s=0.0, day=13):
    return datetime(2026, 10, day, h, mi, 0, tzinfo=ALGERIA_TZ).timestamp() + s


@pytest.fixture(autouse=True)
def clean():
    clock_sync.reset()
    yield
    clock_sync.reset()


# ------------------------------------------------------------------------------------------ primary: PROGRAM-DATE-TIME
MANIFEST = """#EXTM3U
#EXT-X-TARGETDURATION:6
#EXT-X-MEDIA-SEQUENCE:100
#EXT-X-PROGRAM-DATE-TIME:2026-10-13T21:59:50.400Z
#EXTINF:6.000,
a.ts
#EXTINF:6.000,
b.ts
#EXTINF:6.000,
c.ts
"""


def test_manifest_pdt_gives_sub_second_alignment():
    segs = clock_sync.parse_playlist(MANIFEST)
    seg = clock_sync.segment_at(segs, -2)                             # ffmpeg starts at the second-newest: b.ts = 21:59:56.4Z
    assert seg.uri == "b.ts"
    broadcaster = seg.start_epoch                                     # 22:59:56.4 CET
    capture = broadcaster + 5.2                                       # first audio reached us 5.2 s later
    rng = random.Random(7)
    sync = ts.MCRTimeSynchronizer()
    for i in range(8):                                                # reconnects: each reading carries +/- 0.2 s of jitter
        sync.observe_manifest(CH, capture + i * 600 + rng.uniform(-0.2, 0.2), broadcaster + i * 600, rtt_s=0.4)
    ingest = capture + 4000 + 17.0
    cal, margin, drift = sync.calibrate_timestamp(ingest, CH, now=time_now(sync))
    assert drift == pytest.approx(-5.2, abs=0.25)                     # broadcast - ingest
    assert cal == pytest.approx(ingest - 5.2, abs=0.25) and 0.2 <= margin <= 0.5
    st = sync.status(CH, now=time_now(sync))
    assert st["lock"] == ts.LOCKED and st["basis"] == ts.BASIS_PDT and st["absolute"]


def time_now(sync):
    return sync._ch[CH].pdt_updated + 1.0


def test_calibrator_readings_reach_the_synchronizer_and_columns_keep_both_clocks():
    cal = clock_sync.get_calibrator(CH)
    cal.observe(at(23, 0, 2.0), at(22, 59, 57.0), rtt_s=0.2)          # broadcaster 5.0 s earlier than our capture clock
    f = ts.occurrence_fields(CH, at(23, 0, 2.0), at(23, 0, 32.0))
    assert f["mcr_drift_delta_seconds"] == pytest.approx(-5.0)
    assert f["broadcast_start_cet"].startswith("2026-10-13T22:59:57") and f["broadcast_end_cet"].startswith("2026-10-13T23:00:27")
    assert clock_sync.occurrence_fields(CH)["clock_drift_seconds"] == pytest.approx(5.0)     # Phase 22 sign: capture - MCR


def test_uncalibrated_channel_is_unknown_not_zero():
    cal_ts, margin, drift = ts.get_synchronizer().calibrate_timestamp(1000.0, "Nowhere")
    assert (cal_ts, margin, drift) == (1000.0, None, None)
    assert ts.occurrence_fields("Nowhere", 1.0, 2.0) == dict(broadcast_start_cet=None, broadcast_end_cet=None,
                                                             mcr_drift_delta_seconds=None)


def test_static_manifest_clock_is_not_trusted():
    sync = ts.MCRTimeSynchronizer()
    assert sync.observe_manifest(CH, 1000.0, 990.0)
    assert not sync.observe_manifest(CH, 1030.0, 990.0)               # 30 s later the tag still says the same instant
    assert sync.status(CH)["static_pdt_ignored"] == 1 and len(sync._ch[CH].pdt) == 1
    assert not sync.observe_manifest(CH, 5000.0, 100.0)               # 80 min apart: implausible, ignored


# ------------------------------------------------------------------------------------------ fallback: PTS + EMA
def run_pts(noise, n=200, latency=5.2, seed=3, sync=None):
    sync = sync or ts.MCRTimeSynchronizer()
    rng, out = random.Random(seed), []
    for i in range(n):
        pts = 1000.0 + i * 6.0                                         # one segment every 6 s of media time
        jitter = rng.uniform(-noise, noise)
        arrival = 50_000.0 + pts + latency + jitter                    # constant true latency + transport jitter
        out.append(sync.observe_pts(CH, arrival, pts, rtt_s=0.2))
    return sync, out


def test_ema_damps_network_jitter_to_a_stable_estimate():
    sync, ema = run_pts(noise=1.5)
    tail = ema[40:]
    raw_sd = 1.5 / 3 ** 0.5                                            # sd of uniform(-1.5, 1.5)
    assert statistics.pstdev(tail) < 0.45 * raw_sd                    # ~ sqrt(alpha / (2 - alpha)) = 0.29 of the raw spread
    assert max(abs(b - a) for a, b in zip(tail, tail[1:])) < 0.6      # no step as large as the +/-1.5 s raw jitter
    assert abs(statistics.mean(tail)) < 0.6                           # constant latency: relative drift ~ 0 (reference = mean of 5)
    st = sync.status(CH, now=sync._ch[CH].ema_updated + 1)
    assert st["basis"] == ts.BASIS_EMA and not st["absolute"] and st["lock"] == ts.TRACKING     # relative only: never "locked"
    assert 0.2 <= st["margin_s"] < 2.0


def test_ema_follows_a_real_latency_ramp_without_overshoot():
    sync = ts.MCRTimeSynchronizer()
    for i in range(120):                                               # latency grows 0.05 s per segment (buffering drift)
        pts = 1000.0 + i * 6.0
        sync.observe_pts(CH, 9e4 + pts + 5.0 + 0.05 * i, pts)
    est = sync._ch[CH].ema
    assert -6.0 < est < -4.5                                           # true relative delta -5.95; EMA lags a little, never leads


def test_pts_drift_is_added_to_the_last_absolute_pdt_anchor():
    sync = ts.MCRTimeSynchronizer()
    sync.observe_manifest(CH, 1000.0, 995.0)                           # absolute anchor: -5.0 s
    sync._ch[CH].pdt_updated -= 10_000                                 # ... measured long ago: stale for stamping
    for i in range(30):
        pts = 10.0 + i * 6.0
        sync.observe_pts(CH, 2000.0 + pts + 0.1 * i, pts)              # latency grew 0.1 s per segment since the first
    cal, margin, drift = sync.calibrate_timestamp(5000.0, CH)
    # growth since the reference (mean of the first 5 pairs, index 2) = 2.7 s, less the EMA's ramp lag 0.1 * (1 - a) / a = 0.57 s
    assert drift == pytest.approx(-5.0 - 2.13, abs=0.2) and sync.status(CH)["absolute"]


# ------------------------------------------------------------------------------------------ daypart edge case
def prime_sports_card(tmp_path):
    cfg = json.loads(DEFAULT_PATH.read_text(encoding="utf-8"))
    cfg["defaults"]["dayparts"] = [{"name": "Prime Sports", "start": "22:30", "end": "23:00", "rate": 300000},
                                   {"name": "Standard", "start": "23:00", "end": "22:30", "rate": 100000}]
    p = tmp_path / "rates.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")
    return RateCardEngine.load(path=p)


def _db(tmp_path, delta, name="d.db"):
    db = tmp_path / name
    start = at(23, 0, 2.0)                                              # detected 23:00:02 on the ingest clock
    with get_db(db) as conn:
        conn.execute("INSERT INTO reference_ads (ad_id, brand_name, campaign_name, duration_sec, fingerprint, fp_algorithm, "
                     "industry, ad_type) VALUES (1, 'Mobilis', 'Spot', 30, x'00', 't', 'Telecom', 'self')")
        conn.execute("INSERT INTO ad_occurrences (ad_id, source_name, airing_start, airing_end, duration_detected, "
                     "confidence_score, human_verified, mcr_drift_delta_seconds) VALUES (1, ?, ?, ?, 30, 0.97, 0, ?)",
                     (f"live:{CH}", start, start + 30, delta))
    return db


def test_ad_detected_at_23_00_02_with_minus_5_s_drift_is_priced_inside_the_22_30_to_23_00_band(tmp_path):
    eng = prime_sports_card(tmp_path)
    calibrated = build_financial_market_report(_db(tmp_path, -5.0), rate_card=eng)["airings"][0]
    assert calibrated["daypart"] == "Prime Sports" and calibrated["base_rate_30s"] == 300000
    assert datetime.fromtimestamp(calibrated["broadcast_start"], ALGERIA_TZ).strftime("%H:%M:%S") == "22:59:57"
    assert calibrated["airing_start"] == at(23, 0, 2.0)                # the raw ingest stamp is untouched
    legacy = build_financial_market_report(_db(tmp_path, None, "legacy.db"), rate_card=eng)
    assert legacy["airings"][0]["daypart"] == "Standard"               # uncalibrated rows keep the ingest-time rule


def test_certificate_shows_calibrated_time_margin_drift_and_lock():
    row = dict(airing_start=at(23, 0, 2.0), airing_end=at(23, 0, 32.0), clock_drift_seconds=5.2, clock_drift_margin_s=0.3,
               clock_drift_basis="program-date-time", mcr_drift_delta_seconds=-5.2)
    when = datetime.fromtimestamp(row["airing_start"], ALGERIA_TZ)
    text = cert.mcr_air_time_text(row, when)
    assert "22:59:56 CET" in text and "±0.3 s" in text and "MCR Drift -5.2 s" in text and "LOCKED" in text
    assert "23:00:02.000 CET (raw, unaltered)" in text                 # both ends of the audit trail
    assert cert.mcr_air_time_text(dict(airing_start=1.0), when) == ""  # not calibrated: the old wording is kept
    loose = cert.mcr_air_time_text(dict(row, clock_drift_basis="pts-relative", clock_drift_margin_s=2.5), when)
    assert "TRACKING" in loose


def test_airing_mcr_lock_states():
    base = dict(airing_start=at(23, 0, 2.0), airing_end=at(23, 0, 32.0))
    assert ts.airing_mcr(dict(base, clock_drift_seconds=5.0, clock_drift_margin_s=0.4,
                              clock_drift_basis="program-date-time"))["lock"] == ts.LOCKED
    assert ts.airing_mcr(dict(base, clock_drift_seconds=5.0, clock_drift_margin_s=0.4,
                              clock_drift_basis="pts-relative"))["lock"] == ts.TRACKING
    legacy = ts.airing_mcr(base)
    assert legacy["lock"] == ts.UNSYNCED and legacy["mcr_drift_delta_seconds"] is None
    assert ts.airing_mcr(dict(base, mcr_drift_delta_seconds=-5.0))["broadcast_start_cet"].startswith("2026-10-13T22:59:57")


def test_synchronizer_footprint_is_negligible():
    import time
    import tracemalloc
    tracemalloc.start()
    sync = ts.MCRTimeSynchronizer()
    t0 = time.perf_counter()
    for i in range(5000):
        sync.observe_manifest(CH, 1000.0 + i * 6, 995.0 + i * 6, 0.1)
        sync.observe_pts(CH, 2000.0 + i * 6, i * 6.0)
        sync.calibrate_timestamp(1000.0 + i * 6, CH)
    took = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak < 1 * 2 ** 20 and took / 5000 < 0.002                   # < 1 MB, < 2 ms per full cycle (one cycle per 6 s segment: < 0.04 % of a core)
