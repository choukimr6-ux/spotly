"""National broadcaster registry + resolve_channel_stream (offline, deterministic).

Streamlink and HTTP probes are mocked. Set KIHC_NETWORK_TESTS=1 to also resolve
the real channels (asserting only that resolution never raises).
"""
import json
import os
import time

import pytest
from streamlink.exceptions import PluginError

from kihc.connectors import iptv_manager
from kihc.connectors.iptv_manager import (FEATURED_KEYS, NATIONAL_BROADCASTERS,
                                          ChannelUnavailable, channel_resolver,
                                          resolve_channel_stream)
from kihc.connectors.youtube_live import live_url
from kihc.core.live_worker import LiveStreamWorker

NATIONAL = ("echorouk_tv", "ennahar_tv", "el_bilad_tv")


class _Stream:
    def __init__(self, url):
        self.url = url


class FakeSession:
    """streams(url) -> {quality: stream}; records every URL asked for."""

    def __init__(self, live: dict | None = None, error: Exception | None = None):
        self.live, self.error, self.asked = live or {}, error, []

    def streams(self, url):
        self.asked.append(url)
        if self.error:
            raise self.error
        return {"worst": _Stream(self.live[url])} if url in self.live else {}


@pytest.fixture(autouse=True)
def no_network(monkeypatch, tmp_path):
    """Every HLS probe is 'unreachable' and there are no overrides unless a test says otherwise."""
    monkeypatch.setattr(iptv_manager, "check_stream", lambda url, timeout=3.0, headers=None: (False, "mock: down"))
    monkeypatch.setattr(iptv_manager, "OVERRIDES_FILE", tmp_path / "channel_overrides.json")


def _first_live_url(key):
    return live_url(NATIONAL_BROADCASTERS[key].youtube[0])


# ------------------------------------------------------------------ registry

def test_registry_has_featured_channels():
    assert FEATURED_KEYS == ("echorouk_tv", "ennahar_tv", "el_bilad_tv", "el_heddaf_tv")
    assert [NATIONAL_BROADCASTERS[k].name for k in FEATURED_KEYS] == \
        ["Echorouk TV", "Ennahar TV", "El Bilad TV", "El-Heddaf TV"]
    for key in NATIONAL:
        b = NATIONAL_BROADCASTERS[key]
        assert b.youtube, key
        assert all(ref.startswith("UC") and len(ref) == 24 for ref in b.youtube)   # stable channel ids


def test_live_url_forms():
    assert live_url("UCd9ox6D3VbhNp3kJNTxQzCQ") == "https://www.youtube.com/channel/UCd9ox6D3VbhNp3kJNTxQzCQ/live"
    assert live_url("@elbiladtv1") == "https://www.youtube.com/@elbiladtv1/live"
    assert live_url("https://www.youtube.com/@x/") == "https://www.youtube.com/@x/live"


# ------------------------------------------------------------------ resolution

@pytest.mark.parametrize("key", NATIONAL)
def test_resolves_youtube_live_when_no_hls(key):
    target = _first_live_url(key)
    session = FakeSession(live={target: f"https://manifest.googlevideo.com/{key}.m3u8"})
    r = resolve_channel_stream(key, session=session)
    assert r.ok and r.stream_type == "youtube_live"
    assert r.url == f"https://manifest.googlevideo.com/{key}.m3u8"
    assert session.asked[0] == target


def test_second_youtube_candidate_used_when_first_offline():
    second = live_url(NATIONAL_BROADCASTERS["echorouk_tv"].youtube[1])
    session = FakeSession(live={second: "https://manifest.googlevideo.com/echorouk-news.m3u8"})
    r = resolve_channel_stream("echorouk_tv", session=session)
    assert r.stream_type == "youtube_live" and r.url.endswith("echorouk-news.m3u8")
    assert [a[0] for a in r.attempts] == ["youtube", "youtube"]


def test_direct_hls_preferred_when_reachable(monkeypatch):
    monkeypatch.setattr(iptv_manager, "check_stream", lambda url, timeout=3.0, headers=None: (True, "mock: up"))
    session = FakeSession()
    r = resolve_channel_stream("el_heddaf_tv", session=session)
    assert r.stream_type == "hls" and r.url == NATIONAL_BROADCASTERS["el_heddaf_tv"].hls[0]
    assert session.asked == []                                   # streamlink not touched


def test_authorised_override_is_tried_first(monkeypatch):
    feed = "https://licensed.example.dz/echorouk/index.m3u8"
    iptv_manager.OVERRIDES_FILE.write_text(json.dumps({"echorouk_tv": feed}), encoding="utf-8")
    monkeypatch.setattr(iptv_manager, "check_stream",
                        lambda url, timeout=3.0, headers=None: (url == feed, "mock"))
    r = resolve_channel_stream("echorouk_tv", session=FakeSession())
    assert (r.stream_type, r.url) == ("hls", feed)


def test_unreachable_hls_falls_back_to_youtube():
    r = resolve_channel_stream("el_heddaf_tv", session=FakeSession())
    assert r.stream_type == "unavailable"                       # El-Heddaf has no YouTube fallback
    assert r.attempts == [("hls", NATIONAL_BROADCASTERS["el_heddaf_tv"].hls[0], "mock: down")]


# ------------------------------------------------------------------ graceful failure

@pytest.mark.parametrize("key", NATIONAL)
@pytest.mark.parametrize("session", [FakeSession(),                                   # channel offline
                                     FakeSession(error=PluginError("YouTube changed")),  # plugin breakage
                                     FakeSession(error=RuntimeError("socket died"))],    # anything else
                         ids=["offline", "plugin-error", "unexpected-error"])
def test_unresolved_channels_never_raise(key, session):
    r = resolve_channel_stream(key, session=session)
    assert not r.ok and r.url is None and r.stream_type == "unavailable"
    assert len(r.attempts) == len(NATIONAL_BROADCASTERS[key].youtube)
    assert all(a[0] == "youtube" for a in r.attempts)


def test_unknown_channel_is_unavailable_not_keyerror():
    r = resolve_channel_stream("tv_does_not_exist")
    assert r.stream_type == "unavailable" and r.attempts[0][2] == "unknown channel key"


def test_corrupt_overrides_file_is_ignored():
    iptv_manager.OVERRIDES_FILE.write_text("{not json", encoding="utf-8")
    r = resolve_channel_stream("el_heddaf_tv", session=FakeSession())
    assert r.stream_type == "unavailable"


def test_channel_resolver_raises_for_worker_backoff(monkeypatch):
    monkeypatch.setattr(iptv_manager, "resolve_hls_url",
                        lambda url, qualities=None, session=None: (_ for _ in ()).throw(RuntimeError("offline")))
    with pytest.raises(ChannelUnavailable):
        channel_resolver("el_bilad_tv")()


def test_worker_waits_for_offline_channel_without_crashing(monkeypatch, tmp_path):
    monkeypatch.setattr(iptv_manager, "resolve_hls_url",
                        lambda url, qualities=None, session=None: (_ for _ in ()).throw(RuntimeError("offline")))
    w = LiveStreamWorker("El Bilad TV", channel_resolver("el_bilad_tv"), tmp_path / "w.db",
                         evidence_dir=tmp_path, backoff_base=0.05, backoff_max=0.1).start()
    try:
        deadline = time.monotonic() + 20
        while w.connect_attempts < 3 and time.monotonic() < deadline:
            time.sleep(0.05)
        assert w.connect_attempts >= 3 and w.is_alive()
        assert "ChannelUnavailable" in w.last_error and "El Bilad TV" in w.last_error
    finally:
        w.stop()
    assert not w.is_alive()


# ------------------------------------------------------------------ optional: real network

@pytest.mark.skipif(os.environ.get("KIHC_NETWORK_TESTS") != "1", reason="set KIHC_NETWORK_TESTS=1")
def test_real_featured_resolution_never_raises(monkeypatch):
    monkeypatch.undo()   # restore the real check_stream / overrides path
    for key in FEATURED_KEYS:
        r = resolve_channel_stream(key)
        print(f"\n[NETWORK] {r.name}: {r.stream_type} {(r.url or '')[:60]}")
        assert r.stream_type in ("hls", "youtube_live", "unavailable")
