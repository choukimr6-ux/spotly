"""Phase 25: live 2 fps frame ingest -> PodBoundaryDetector -> physical placement verdicts, with audio kept immune."""
import json
import threading
import time
import tracemalloc

import numpy as np
import pytest

from spotly.analytics import pod_boundary_detector as pbd
from spotly.core.pod_analyzer import PHYSICAL_STATUS
from spotly.database import telemetry_store
from spotly.database.schema import get_db
from spotly.service.headless_runner import ChannelSpec, Supervisor
from spotly.stream import elheddaf_stream as hs
from spotly.tests import test_pod_boundary as pb
from spotly.tests.test_campaign_hierarchy import FakeProc, LazyFrames, yuv_frame

CH, T = pb.CH, pb.T


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    monkeypatch.delenv("SPOTLY_STRICT_PLACEMENT", raising=False)
    pbd.reset()
    yield
    pbd.flush(5)
    pbd.reset()


class PacedPipe:
    """ffmpeg stdout stand-in: yields the scene's frames in order, but only once the analyser has caught up, so a test that
    wants every frame analysed loses none (the real pipe is paced by the 2 fps wall clock instead)."""

    def __init__(self, frames, ing):
        self.frames, self.ing, self.i = frames, ing, 0
        self.buf = b""

    def read(self, size):
        if not self.buf:
            if self.i >= len(self.frames):
                return b""
            while self.ing.frames_processed + self.ing.ring.dropped < self.ing.frames_read:
                time.sleep(0.0005)
            if self.i == 0:
                self.ing.offset_s = T - time.time()                   # frame n is stamped T + n / 2 s, like the audio
            self.buf = self.frames[self.i]
            self.i += 1
        out, self.buf = self.buf[:size], self.buf[size:]
        return out


def scene_frames(total, ticker_until, ticker_back, black):
    on, off, blk = yuv_frame(120, True), yuv_frame(120, False), yuv_frame(16, False)
    out = []
    for i in range(int(total * 2)):
        t = i / 2
        if any(abs(t - b) < 0.01 for b in black):
            out.append(blk)
        else:
            out.append(off if ticker_until <= t < ticker_back else on)
    return out


def run_scene(promo_before, promo_after, with_video=True):
    ad_s, ad_e, exit_t, levels = pb.break_scene(promo_before, promo_after)
    total = exit_t + 8
    pb.feed_audio(pb.audio(levels, total))
    ing = None
    if with_video:
        ing = hs.FrameIngest(CH, source=lambda: "x", backoff=30.0)
        frames = scene_frames(total, 60.0, exit_t, black=(60.0, exit_t))
        ing._spawn = lambda url: FakeProc(PacedPipe(frames, ing))
        ing.start()
        deadline = time.time() + 60
        while ing.frames_processed < len(frames) and time.time() < deadline:
            time.sleep(0.02)
        ing.stop()
        assert ing.frames_processed == len(frames) and ing.ring.dropped == 0
    return ad_s, ad_e, ing


def test_audio_only_stays_unsensed(tmp_path):
    s, e, _ = run_scene(False, False, with_video=False)
    db = pb.make_db(tmp_path, [(s, e)])
    pb.settle(db, [(s, e)])
    ad = pb.ad_of(db)
    assert ad["placement_status"] == "unsensed" and ad["physical_status"] == "UNSENSED"     # the legacy audio-only mocks


def test_live_frames_move_first_in_pod_out_of_unsensed_to_confirmed_physical(tmp_path):
    s, e, ing = run_scene(False, False)
    det = pbd.get_detector(CH)
    assert det.sensing_status()["video_frames"] == len(det._video) > 100
    db = pb.make_db(tmp_path, [(s, e)])
    pb.settle(db, [(s, e)])
    ad = pb.ad_of(db)
    assert ad["placement_status"] == "confirmed" and ad["physical_status"] == "CONFIRMED_PHYSICAL"
    assert ad["position"] == "SOLO" and ad["boundary_confidence"] >= pbd.CERTIFY_CONFIDENCE
    assert {"audio_dip", "visual_dip", "ticker_absent"} <= set(ad["boundary_cues"])      # black frame + ticker flip + silence
    assert ing.status()["frames_dropped"] == 0


def test_unclassified_promo_before_the_ad_demotes_it_to_middle(tmp_path):
    s, e, _ = run_scene(True, True)
    db = pb.make_db(tmp_path, [(s, e)])
    pb.settle(db, [(s, e)])
    ad = pb.ad_of(db)
    assert ad["position"] == "MIDDLE" and ad["placement_status"] == "demoted_confirmed"
    assert ad["physical_status"] == "DEMOTED_CONFIRMED" and "unclassified material" in ad["placement_note"]
    assert ad["position"] != "FIRST"                                                      # no false First-in-Pod claim


def test_every_placement_status_has_a_physical_name():
    assert set(PHYSICAL_STATUS) >= {"confirmed", "demoted_confirmed", "demoted_ambiguous", "unsensed"}
    assert PHYSICAL_STATUS["confirmed"] == "CONFIRMED_PHYSICAL" and PHYSICAL_STATUS["demoted_ambiguous"] == "DEMOTED_AMBIGUOUS"


# ------------------------------------------------------------------------------------------ back-pressure
def test_stalled_consumer_drops_frames_without_touching_audio_or_memory():
    n = 400
    gate = threading.Event()
    ing = hs.FrameIngest("Stall", source=lambda: "x", spawn=lambda u: FakeProc(LazyFrames(yuv_frame(), n)), backoff=30.0,
                         analyze=lambda ts, g: gate.wait(10))               # the consumer is stuck on its first frame
    pcm = (np.random.default_rng(3).standard_normal(int(0.2 * 22050)) * 1500).astype(np.int16)
    tracemalloc.start()
    ing.start()
    chunks, worst, t0 = 0, 0.0, time.perf_counter()
    while ing.frames_read < n and time.perf_counter() - t0 < 15:
        t = time.perf_counter()
        pbd.feed_audio("Stall", 2_000_000.0 + chunks * 0.2, pcm)
        worst = max(worst, time.perf_counter() - t)
        chunks += 1
        time.sleep(0.005)
    _, peak = tracemalloc.get_traced_memory()
    assert ing.frames_read == n                                            # the reader never waited for the stuck analyser
    assert len(ing.ring) <= hs.RING_DEPTH and ing.ring.dropped >= n - hs.RING_DEPTH - 1
    assert len(pbd.get_detector("Stall")._audio) == chunks * 4              # every audio chunk landed: 0 lost, 0 delayed
    assert worst < 0.05, f"audio feed stalled {worst * 1000:.1f} ms"
    gate.set()
    ing.stop()
    tracemalloc.stop()
    assert peak < 8 * 2 ** 20, f"heap peak {peak / 2 ** 20:.1f} MB"          # 400 frames of 194 KB went through; 4 were held


def test_slow_analysis_is_counted_and_status_reports_fps_keys():
    ing = hs.FrameIngest("Slow", source=lambda: "x", spawn=lambda u: FakeProc(LazyFrames(yuv_frame(), 6)), backoff=30.0,
                         analyze=lambda ts, g: time.sleep(hs.SLOW_FRAME_MS / 1000 + 0.05))
    ing.start()
    deadline = time.time() + 10
    while ing.slow_frames < 1 and time.time() < deadline:
        time.sleep(0.05)
    ing.stop()
    st = ing.status()
    assert st["slow_frames"] >= 1 and {"fps", "frames_read", "frames_dropped", "analyse_ms_p95"} <= set(st)


# ------------------------------------------------------------------------------------------ telemetry
def test_video_status_rides_on_the_channel_telemetry(tmp_path):
    class Worker:
        def start(self): return self
        def stop(self): pass
        def status(self): return dict(state="listening", level_history=[])

    db = tmp_path / "t.db"
    sup = Supervisor([ChannelSpec(CH, "x")], db, worker_factory=lambda s: Worker())
    ing = hs.FrameIngest(CH, source=lambda: "x", spawn=lambda u: FakeProc(LazyFrames(yuv_frame(), 30)), backoff=30.0)
    ing.start()
    deadline = time.time() + 10
    while ing.frames_processed + ing.ring.dropped < 30 and time.time() < deadline:
        time.sleep(0.02)
    ing.stop()
    sup.workers, sup.frame_ingests = {CH: Worker()}, {CH: ing}
    sup._write_channels()
    conn = telemetry_store.connect(db)
    try:
        status = json.loads(conn.execute("SELECT status_json FROM service_channel_status").fetchone()[0])
    finally:
        conn.close()
    assert status["video"]["frames_read"] == 30 and status["video"]["sensing"]["video_frames"] > 0


def test_watchdog_does_not_kill_a_fresh_connection_for_a_previous_connections_silence():
    ing = hs.FrameIngest("Dog", source=lambda: "x", backoff=30.0)
    ing.last_frame_at = time.time() - 600                      # the PREVIOUS ffmpeg died 10 minutes ago
    proc = FakeProc(b"")
    t = threading.Thread(target=ing._watchdog, args=(proc,), daemon=True)
    t.start()
    time.sleep(2.6)
    assert not proc.dead                                       # the new connection has had 2.6 s, not 600 s
    ing._stop.set()
    t.join(5)


def test_starved_link_drops_stale_frames_then_pauses_instead_of_competing_with_audio(monkeypatch):
    """A link slower than the 2 fps video rendition: frames fall behind real time, so none reach the detector and the
    download is dropped for an escalating pause (the audio ingest owns the bandwidth)."""
    monkeypatch.setattr(hs, "STALE_DRIFT_S", -1.0)             # every frame counts as behind real time
    monkeypatch.setattr(hs, "STARVE_AFTER_S", 0.0)
    monkeypatch.setattr(hs, "PAUSE_STEPS_S", (30.0,))
    seen = []
    ing = hs.FrameIngest("Starved", source=lambda: "x", spawn=lambda u: FakeProc(LazyFrames(yuv_frame(), 4000)), backoff=30.0,
                         analyze=lambda ts, g: seen.append(ts))
    ing.start()
    deadline = time.time() + 10
    while not ing.paused() and time.time() < deadline:
        time.sleep(0.02)
    st = ing.status()
    ing.stop()
    assert st["paused"] and st["starved_events"] == 1 and "link too slow" in st["pause_reason"]
    assert st["stale_dropped"] > 0 and not seen and ing.ring.put_count == 0          # nothing misaligned was ever analysed
    assert st["pause_remaining_s"] > 0
