"""Phase 2: connectors + resilient live worker.

Offline by default. Set KIHC_NETWORK_TESTS=1 to also hit the real iptv-org playlist
and a real Algerian stream.
"""
import ctypes
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pytest
import requests

from kihc.connectors import iptv_manager
from kihc.connectors.iptv_manager import (FALLBACK_CHANNELS, check_stream,
                                          clean_name, fetch_channels,
                                          parse_m3u)
from kihc.connectors.youtube_live import YouTubeLiveError, resolve_hls_url
from kihc.core.audio import ffmpeg_exe
from kihc.core.ingest import ingest_reference
from kihc.core.live_worker import LiveStreamWorker, backoff_delay
from kihc.database.schema import init_db
from kihc.tests.generate_test_assets import generate

NETWORK = os.environ.get("KIHC_NETWORK_TESTS") == "1"
WORKER_RAM_BUDGET = 300 * 1024 ** 2

SAMPLE_M3U = """#EXTM3U
#EXTINF:-1 tvg-id="TV2.dz@SD" tvg-logo="https://x/tv2.png" group-title="General",TV2 (Algeria) (1080p)
http://185.9.2.18/chid_347/index.m3u8
#EXTINF:-1 tvg-id="CNA.dz@SD" group-title="Religious",CNA (Algeria) (360p) [Not 24/7]
#EXTVLCOPT:http-referrer=https://example.dz/
#EXTVLCOPT:http-user-agent=TestAgent/1.0
https://live.creacast.com/cna/stream/playlist.m3u8

#EXTINF:-1,Bare Channel
https://example.com/bare.m3u8
"""


# ------------------------------------------------------------------ helpers

def _proc_peak_ws(pid: int) -> int:
    """Peak working set of another process (Windows); 0 if unavailable."""
    if sys.platform != "win32" or not pid:
        return 0
    from ctypes import wintypes

    class PMC(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("a", ctypes.c_size_t), ("b", ctypes.c_size_t), ("c", ctypes.c_size_t),
                    ("d", ctypes.c_size_t), ("e", ctypes.c_size_t), ("f", ctypes.c_size_t)]
    k32 = ctypes.WinDLL("kernel32")
    k32.OpenProcess.restype = wintypes.HANDLE
    k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    k32.K32GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
    k32.CloseHandle.argtypes = [wintypes.HANDLE]
    h = k32.OpenProcess(0x1000 | 0x0010, False, pid)   # QUERY_LIMITED_INFORMATION | VM_READ
    if not h:
        return 0
    try:
        pmc = PMC()
        pmc.cb = ctypes.sizeof(PMC)
        return pmc.PeakWorkingSetSize if k32.K32GetProcessMemoryInfo(h, ctypes.byref(pmc), pmc.cb) else 0
    finally:
        k32.CloseHandle(h)


def _wait_for(pred, timeout):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.05)
    return False


@pytest.fixture(scope="module")
def assets():
    return generate()


@pytest.fixture
def short_clip(tmp_path):
    """1-second synthetic 'stream' that ends quickly -> forces disconnect/reconnect cycles."""
    path = tmp_path / "one_second.wav"
    subprocess.run([ffmpeg_exe(), "-nostdin", "-v", "error", "-y", "-f", "lavfi",
                    "-i", "sine=frequency=1000:duration=1", str(path)], check=True)
    return path


# ------------------------------------------------------------------ IPTV manager

def test_parse_m3u_names_urls_headers():
    chs = parse_m3u(SAMPLE_M3U)
    assert [c.name for c in chs] == ["TV2 (Algeria)", "CNA (Algeria)", "Bare Channel"]
    assert chs[0].url == "http://185.9.2.18/chid_347/index.m3u8"
    assert chs[0].tvg_id == "TV2.dz@SD" and chs[0].group == "General"
    assert chs[1].headers == {"Referer": "https://example.dz/", "User-Agent": "TestAgent/1.0"}
    assert chs[2].headers == {} and chs[2].group == ""


def test_clean_name_strips_quality_and_tags():
    assert clean_name("CNA (Algeria) (360p) [Not 24/7]") == "CNA (Algeria)"
    assert clean_name("Elwatania TV (1080i)") == "Elwatania TV"


def test_fetch_falls_back_when_offline(monkeypatch, tmp_path):
    def offline(*a, **k):
        raise requests.ConnectionError("offline")
    monkeypatch.setattr(iptv_manager.requests, "get", offline)
    monkeypatch.setattr(iptv_manager, "CACHE_FILE", tmp_path / "missing.m3u")
    chs = fetch_channels()
    assert {c.name for c in chs} == set(FALLBACK_CHANNELS)
    assert all(c.source == "fallback" and c.url.startswith("http") for c in chs)


def test_fetch_prefers_cache_over_fallback(monkeypatch, tmp_path):
    cache = tmp_path / "dz.m3u"
    cache.write_text(SAMPLE_M3U, encoding="utf-8")
    monkeypatch.setattr(iptv_manager.requests, "get",
                        lambda *a, **k: (_ for _ in ()).throw(requests.Timeout("slow")))
    monkeypatch.setattr(iptv_manager, "CACHE_FILE", cache)
    chs = fetch_channels()
    assert [c.source for c in chs] == ["cache"] * 3


def test_check_stream_dead_endpoint_is_inactive():
    t0 = time.monotonic()
    ok, detail = check_stream("http://127.0.0.1:9/dead.m3u8", timeout=1.0)
    assert not ok and detail
    assert time.monotonic() - t0 < 5


@pytest.mark.skipif(not NETWORK, reason="set KIHC_NETWORK_TESTS=1")
def test_live_playlist_parses():
    chs = fetch_channels()
    print(f"\n[NETWORK] {len(chs)} channels from {chs[0].source}: {[c.name for c in chs]}")
    assert chs and all(c.url.startswith("http") for c in chs)


# ------------------------------------------------------------------ YouTube (offline, fake session)

class _FakeStream:
    def __init__(self, url):
        self.url = url


class _FakeSession:
    def __init__(self, streams):
        self._streams = streams

    def streams(self, url):
        return self._streams


def test_youtube_resolver_prefers_lowest_variant():
    s = _FakeSession({"worst": _FakeStream("https://hls/144p.m3u8"), "best": _FakeStream("https://hls/1080p.m3u8")})
    assert resolve_hls_url("https://www.youtube.com/@x/live", session=s) == "https://hls/144p.m3u8"


def test_youtube_resolver_offline_channel_raises():
    with pytest.raises(YouTubeLiveError):
        resolve_hls_url("https://www.youtube.com/@x/live", session=_FakeSession({}))


# ------------------------------------------------------------------ live worker

def test_backoff_schedule():
    assert [backoff_delay(i) for i in range(7)] == [2, 4, 8, 16, 30, 30, 30]


def test_ring_buffer_is_bounded_to_30s(assets, tmp_path):
    _, _, broadcast = assets
    w = LiveStreamWorker("RingTest", str(broadcast), tmp_path / "empty.db", max_connects=1,
                         evidence_dir=tmp_path).start()
    assert w.wait(60)
    st = w.status()
    print(f"\n[RING] received={st['received_seconds']}s max_buffered={st['max_buffered_seconds']}s")
    assert abs(st["received_seconds"] - 60.0) < 0.5
    assert 29.0 <= st["max_buffered_seconds"] <= 30.0


def test_reconnects_with_exponential_backoff(short_clip, tmp_path):
    w = LiveStreamWorker("FlakyTV", str(short_clip), tmp_path / "r.db", evidence_dir=tmp_path,
                         backoff_base=0.05, backoff_max=0.2, healthy_after=999).start()
    try:
        assert _wait_for(lambda: len(w.backoff_history) >= 4, 30), w.status()
        assert w.is_alive()
        assert w.backoff_history[:4] == [0.05, 0.1, 0.2, 0.2]
        assert w.connect_attempts >= 4
        assert w.samples_received >= 3 * 22050          # each reconnect delivered audio again
        print(f"\n[RECONNECT] attempts={w.connect_attempts} delays={w.backoff_history[:6]}")
    finally:
        w.stop()
    assert not w.is_alive() and w.state == "stopped"


def test_unreachable_stream_never_crashes(tmp_path):
    w = LiveStreamWorker("DeadTV", str(tmp_path / "does_not_exist.m3u8"), tmp_path / "d.db",
                         evidence_dir=tmp_path, backoff_base=0.05, backoff_max=0.1).start()
    try:
        assert _wait_for(lambda: w.connect_attempts >= 3, 30), w.status()
        assert w.is_alive() and w.last_error.startswith("ffmpeg exited")
    finally:
        t0 = time.monotonic()
        w.stop()
        assert time.monotonic() - t0 < 5
    assert not w.is_alive()


def test_resolver_exceptions_are_retried(tmp_path):
    calls = []

    def resolver():
        calls.append(1)
        raise YouTubeLiveError("channel offline")
    w = LiveStreamWorker("YTOffline", resolver, tmp_path / "y.db", evidence_dir=tmp_path,
                         backoff_base=0.05, backoff_max=0.1).start()
    try:
        assert _wait_for(lambda: len(calls) >= 3, 30)
        assert w.is_alive() and "channel offline" in w.last_error
    finally:
        w.stop()


def test_live_detection_on_synthetic_stream(assets, tmp_path):
    ad_a, ad_b, broadcast = assets
    db = tmp_path / "live.db"
    ad_a_id = ingest_reference(ad_a, "Acme", "Ad A", db_path=db)
    ingest_reference(ad_b, "Decoy Co", "Ad B (never aired)", db_path=db)

    w = LiveStreamWorker("SyntheticTV", str(broadcast), db, evidence_dir=tmp_path / "ev", max_connects=1).start()
    ffmpeg_peak = 0
    while w.is_alive():
        ffmpeg_peak = max(ffmpeg_peak, _proc_peak_ws(w.status()["ffmpeg_pid"]))
        time.sleep(0.02)

    dets = list(w.detections)
    assert len(dets) == 1, dets
    d = dets[0]
    print(f"\n[LIVE] {d.brand_name}/{d.campaign_name} offset={d.stream_offset_sec:.3f}s "
          f"utc={d.airing_start_utc} -> {d.airing_end_utc} conf={d.confidence_score:.4f} "
          f"evidence={Path(d.evidence_frame_path).name}")
    assert d.ad_id == ad_a_id
    assert 24.0 <= d.stream_offset_sec <= 26.0
    assert d.confidence_score >= 0.80
    assert 9.0 <= d.duration_detected <= 10.5
    assert Path(d.evidence_frame_path).stat().st_size > 1000

    conn = init_db(db)
    rows = conn.execute("SELECT * FROM ad_occurrences").fetchall()
    conn.close()
    assert len(rows) == 1                                     # one airing -> one row, no duplicate alerts
    row = rows[0]
    assert row["source_name"] == "live:SyntheticTV"
    assert datetime.fromisoformat(row["detected_at"]).tzinfo is not None
    assert row["airing_end"] - row["airing_start"] == pytest.approx(d.duration_detected, abs=0.01)
    assert row["airing_start"] > 1.6e9                         # epoch seconds, not stream-relative

    print(f"[MEMORY] ffmpeg peak working set {ffmpeg_peak / 1024 ** 2:.1f} MiB")
    if sys.platform == "win32":
        assert 0 < ffmpeg_peak < WORKER_RAM_BUDGET


@pytest.mark.skipif(not NETWORK, reason="set KIHC_NETWORK_TESTS=1")
def test_real_algerian_stream_listens(tmp_path):
    chans = iptv_manager.check_streams(fetch_channels())
    up = [c for c in chans if c.active]
    if not up:
        pytest.skip("no Algerian stream reachable right now")
    ch = up[0]
    w = LiveStreamWorker(ch.name, ch.url, tmp_path / "net.db", headers=ch.headers, evidence_dir=tmp_path).start()
    try:
        ok = _wait_for(lambda: w.status()["received_seconds"] >= 10, 60)
        print(f"\n[NETWORK] {ch.name}: {w.status()}")
        assert ok
    finally:
        w.stop()
