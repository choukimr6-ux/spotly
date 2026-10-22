"""Multi-stream ingestion: offline fleet simulation with real child processes (a Python stand-in for FFmpeg)."""
import json
import sys
import time

import pytest

from spotly.ingest import multi_stream_manager as msm
from spotly.ingest.multi_stream_manager import (AudioSegment, ChannelProfile, MultiStreamManager, SegmentQueue,
                                              backoff_delay, ffmpeg_command, load_catalogue)

SEG = 0.1                                   # seconds per segment in tests
FAKE = r"""
import sys, time
mode, seg = sys.argv[1], int(sys.argv[2])
out = sys.stdout.buffer
n = 0
while True:
    out.write(b"\x01\x00" * seg); out.flush(); n += 1
    if mode == "crash" and n == 2: sys.exit(1)
    if mode == "stall" and n == 2: time.sleep(60)
    time.sleep(0.02)
"""


def _catalogue(modes: dict, tier="priority"):
    chans = {cid: ChannelProfile(cid, cid.upper(), tier if not cid.startswith("s_") else "secondary",
                                 source={"type": "hls", "url": f"fake://{mode}"}) for cid, mode in modes.items()}
    return msm.Catalogue(chans, dict(msm.DEFAULTS), ("priority",))


def _manager(modes, **kw):
    cat = _catalogue(modes)
    seg_samples = int(msm.PCM_RATE * SEG)
    settings = dict(segment_seconds=SEG, stall_timeout_sec=2.5, backoff_base_sec=0.1, backoff_max_sec=0.4,
                    healthy_after_sec=30, unresolved_retry_sec=0.3, max_active=30, resolve_timeout_sec=5)
    settings.update(kw.pop("settings", {}))

    def resolver(ch):
        if ch.source["url"] == "fake://unresolved":
            raise msm.ChannelUnresolved("no url")
        return ch.source["url"]

    return MultiStreamManager(cat, resolver=resolver, settings=settings, **kw,
                              command_builder=lambda url, ch: [sys.executable, "-c", FAKE, url[7:], str(seg_samples)])


def _wait(cond, timeout=40.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.05)
    return False


def _drain(mgr):
    out = []
    while True:
        try:
            out.append(mgr.queue.get_nowait())
        except Exception:
            return out


def test_one_channel_failing_never_disturbs_the_others():
    mgr = _manager({"good": "ok", "crashy": "crash", "frozen": "stall", "ghost": "unresolved"}).start()
    try:
        assert _wait(lambda: mgr.status()["crashy"]["connects"] >= 3 and mgr.status()["frozen"]["connects"] >= 2
                     and mgr.status()["good"]["segments"] >= 40)
        st = mgr.status()
        assert st["good"]["connects"] == 1 and st["good"]["failures"] == 0 and st["good"]["state"] == "live"
        assert "exited rc=1" in st["crashy"]["last_error"] and st["crashy"]["failures"] >= 2
        assert "stall" in st["frozen"]["last_error"]
        assert st["ghost"]["state"] == "unresolved" and st["ghost"]["connects"] == 0 and st["ghost"]["failures"] >= 1
        segs = _drain(mgr)
    finally:
        mgr.stop()
    assert {s.channel_id for s in segs} == {"good", "crashy", "frozen"}
    good = [s for s in segs if s.channel_id == "good"]
    assert [s.seq for s in good] == list(range(good[0].seq, good[0].seq + len(good)))      # no gap, no reorder
    crashy = [s.seq for s in segs if s.channel_id == "crashy"]
    assert crashy == sorted(crashy) and len(set(crashy)) == len(crashy)                     # seq survives restarts
    assert all(v["ffmpeg_pid"] is None for v in mgr.status().values())                      # every child reaped


def test_segments_are_tagged_raw_pcm_and_timestamped_on_the_capture_clock():
    mgr = _manager({"good": "ok"}).start()
    try:
        assert _wait(lambda: mgr.status()["good"]["segments"] >= 20)
        segs = _drain(mgr)
    finally:
        mgr.stop()
    now = time.time()
    assert all(isinstance(s, AudioSegment) and s.sample_rate == 16000 and len(s.pcm) == 1600 * 2 for s in segs)
    assert all(abs(s.duration - SEG) < 1e-9 for s in segs)
    starts = [s.wall_start for s in segs]
    assert starts == sorted(starts) and starts[-1] <= now
    assert all(abs((b - a) - SEG) < 0.05 or b - a > 0 for a, b in zip(starts, starts[1:]))


def test_pools_standby_activation_failover_and_cap():
    mgr = _manager({"p1": "ok", "p2": "ok", "s_news": "ok", "s_public": "ok"}, settings=dict(max_active=3)).start()
    try:
        assert mgr.active() == ["p1", "p2"]                                  # secondary tier stays in stand-by
        assert mgr.status()["s_news"]["state"] == "standby" and mgr.status()["s_news"]["ffmpeg_pid"] is None
        assert mgr.activate("s_news") and not mgr.activate("s_news")         # already running
        assert not mgr.activate("s_public")                                  # max_active reached
        assert not mgr.activate("nope")
        assert mgr.failover("p1", "s_public") and mgr.active() == ["p2", "s_news", "s_public"]
        assert _wait(lambda: mgr.status()["s_public"]["segments"] > 3)
        assert mgr.status()["p1"]["state"] == "standby" and mgr.status()["p1"]["ffmpeg_pid"] is None
        assert mgr.deactivate("s_news") and not mgr.deactivate("s_news")
    finally:
        mgr.stop()
    assert mgr.active() == []


def test_full_fleet_of_25_runs_in_one_process_with_small_python_footprint():
    cat = load_catalogue()
    assert len(cat.channels) == 25 and len(cat.select(["priority"])) == 10
    seg_samples = int(msm.PCM_RATE * SEG)
    mgr = MultiStreamManager(cat, pools=["priority", "secondary"], resolver=lambda ch: "fake://ok",
                             settings=dict(segment_seconds=SEG, max_active=25),
                             command_builder=lambda url, ch: [sys.executable, "-c", FAKE, "ok", str(seg_samples)]).start()
    try:
        assert _wait(lambda: all(v["segments"] >= 5 for v in mgr.status().values()), 30)
        rep = mgr.memory_report()
        for _ in range(3):                           # consumer keeps up
            _drain(mgr)
            time.sleep(0.2)
    finally:
        mgr.stop()
    assert rep["active_channels"] == 25 and rep["manager_rss_mb"] < 250 and rep["queue_mb"] < 5
    assert len(rep["ffmpeg_mb_per_channel"]) == 25                              # child processes reported separately


def test_queue_never_blocks_and_drops_oldest():
    q = SegmentQueue(5)
    for i in range(12):
        q.put(AudioSegment("a" if i % 2 else "b", i, float(i), 16000, b"\0\0"))
    assert q.qsize() == 5 and [q.get_nowait().seq for _ in range(5)] == [7, 8, 9, 10, 11]
    assert q.dropped["a"] + q.dropped["b"] == 7


def test_backoff_is_exponential_jittered_and_capped():
    class R:                                    # deterministic "random": 0 -> lower bound, 1 -> upper bound
        def __init__(self, v): self.v = v
        def random(self): return self.v
    lo = [backoff_delay(i, 2.0, 60.0, R(0.0)) for i in range(8)]
    hi = [backoff_delay(i, 2.0, 60.0, R(1.0)) for i in range(8)]
    assert hi[:5] == [2, 4, 8, 16, 32] and hi[5:] == [60, 60, 60]
    assert lo == [h / 2 for h in hi]
    assert len({backoff_delay(3, 2.0, 60.0) for _ in range(20)}) > 1            # real jitter


def test_ffmpeg_is_audio_only_16k_mono_pcm_with_hls_live_start_index():
    cmd = ffmpeg_command("https://x/y.m3u8", stall_timeout=20, live_start_index=-2, ffmpeg="ffmpeg")
    flat = " ".join(cmd)
    for token in ("-vn", "-acodec pcm_s16le", "-ar 16000", "-ac 1", "-f s16le pipe:1", "-live_start_index -2"):
        assert token in flat
    assert cmd.index("-live_start_index") < cmd.index("-i")                       # input option
    assert "-live_start_index" not in ffmpeg_command("/tmp/a.wav", stall_timeout=20, live_start_index=-2, ffmpeg="f")


def test_catalogue_lists_the_25_official_channels_and_validates(tmp_path):
    cat = load_catalogue()
    expected = """tv1_nationale canal_algerie tv3_algerie3 tv4_tamazight tv5_coran tv6_jeunesse tv7_el_maarifa tv8_edhakira
        tv9_parlementaire al24_news echorouk_tv echorouk_news ennahar_tv el_bilad_tv el_hayat_tv samira_tv el_heddaf_tv
        bahia_tv el_adjwaa_tv lina_tv berbere_tv cna_chaine_nord_africaine dzair_news el_djazairia_one
        al_magharibia_tv""".split()
    assert list(cat.channels) == expected
    assert {"el_heddaf_tv", "echorouk_tv", "ennahar_tv", "samira_tv", "el_bilad_tv"} <= {c.id for c in cat.select(["priority"])}
    assert cat.channels["el_heddaf_tv"].name == "El-Heddaf TV"                     # same name as the live pipeline's rows
    with pytest.raises(KeyError):
        cat.select(only=["nope"])
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"channels": [{"id": "A B"}]}))
    with pytest.raises(ValueError):
        load_catalogue(bad)


def test_default_resolver_prefers_overrides_and_never_invents_urls(tmp_path):
    ov = tmp_path / "o.json"
    ov.write_text(json.dumps({"samira_tv": "http://example.invalid/s.m3u8"}))
    r = msm.DefaultResolver(ov)
    cat = load_catalogue()
    assert r(cat.channels["samira_tv"]) == "http://example.invalid/s.m3u8"
    assert r(cat.channels["el_heddaf_tv"]).endswith("tracks-a1/mono.m3u8")
    r._playlist, r._playlist_at = [], time.time()
    r._playlist = [type("C", (), dict(name="Bahia TV HD", url="http://b/x.m3u8"))()]
    assert r(cat.channels["bahia_tv"]) == "http://b/x.m3u8"
    with pytest.raises(msm.ChannelUnresolved):
        r(cat.channels["lina_tv"])
