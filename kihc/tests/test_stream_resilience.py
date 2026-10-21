"""Phase 28: dedicated IPTV overrides resolve to direct M3U8 feeds and bypass every web / YouTube scraper."""
import json
import re

import pytest

from kihc.connectors import iptv_manager as iptv
from kihc.ingest import multi_stream_manager as msm
from kihc.service import headless_runner as hr

HOST = "http://iptv.test:8080/live/USER/PASS"
OVERRIDES = {"elheddaf_tv": {"stream_url": f"{HOST}/407.m3u8"}, "samira_tv": {"stream_url": f"{HOST}/408.m3u8"},
             "tv3_algerie3": f"{HOST}/406.m3u8",                                  # legacy plain-string form still works
             "ennahar_tv": {"stream_url": "  "}, "_meta": {"stream_url": "x"}, "bad": 42}


@pytest.fixture
def overrides(tmp_path, monkeypatch):
    f = tmp_path / "channel_overrides.json"
    f.write_text(json.dumps(OVERRIDES), encoding="utf-8")
    monkeypatch.setattr(iptv, "OVERRIDES_FILE", f)

    def scraper(*a, **k):
        raise AssertionError("an external scraper / probe was consulted although a dedicated feed exists")
    monkeypatch.setattr(iptv, "check_stream", scraper)
    monkeypatch.setattr(iptv, "resolve_hls_url", scraper)
    monkeypatch.setattr(iptv, "fetch_channels", scraper)
    return f


def test_both_override_shapes_parse_and_junk_is_skipped():
    assert iptv.parse_overrides(OVERRIDES) == {"elheddaf_tv": f"{HOST}/407.m3u8", "samira_tv": f"{HOST}/408.m3u8",
                                               "tv3_algerie3": f"{HOST}/406.m3u8"}
    assert iptv.parse_overrides(None) == {} and iptv.parse_overrides([1]) == {}


def test_keys_match_without_separators_or_case(overrides):
    assert iptv.override_url("el_heddaf_tv") == iptv.override_url("elheddaf_tv") == f"{HOST}/407.m3u8"
    assert iptv.override_url("El-Heddaf TV") == f"{HOST}/407.m3u8" and iptv.override_url("nobody") is None


def test_registry_channel_resolves_to_the_direct_m3u8_without_any_scraper(overrides):
    r = iptv.resolve_channel_stream("el_heddaf_tv")                  # in NATIONAL_BROADCASTERS: web HLS + YouTube candidates
    assert r.ok and r.stream_type == "hls" and r.url == f"{HOST}/407.m3u8" and r.name == "El-Heddaf TV"
    assert [a[0] for a in r.attempts] == ["override"]               # nothing else was even tried
    assert "PASS" not in repr(r.attempts) and "USER" not in repr(r.attempts)   # credentials never reach logs / attempts


def test_channel_outside_the_registry_resolves_too(overrides):
    r = iptv.resolve_channel_stream("samira_tv")
    assert r.ok and r.url.endswith("/408.m3u8") and r.name == "Samira TV"


def test_without_an_override_the_old_chain_is_still_used(tmp_path, monkeypatch):
    monkeypatch.setattr(iptv, "OVERRIDES_FILE", tmp_path / "missing.json")
    monkeypatch.setattr(iptv, "check_stream", lambda url, t: (False, "down"))
    monkeypatch.setattr(iptv, "resolve_hls_url", lambda t, session=None: (_ for _ in ()).throw(RuntimeError("no live")))
    assert {a[0] for a in iptv.resolve_channel_stream("el_heddaf_tv").attempts} == {"hls"}
    r = iptv.resolve_channel_stream("ennahar_tv")
    assert not r.ok and {a[0] for a in r.attempts} == {"youtube"}


def test_worker_resolver_returns_the_url_and_follows_file_edits(overrides):
    resolve = iptv.channel_resolver("samira_tv")
    assert resolve() == f"{HOST}/408.m3u8"
    overrides.write_text(json.dumps({"samira_tv": {"stream_url": f"{HOST}/999.m3u8"}}), encoding="utf-8")
    assert resolve() == f"{HOST}/999.m3u8"                           # re-read on every (re)connect


def test_headless_service_channel_arg_uses_the_override(overrides):
    for arg in ("samira_tv", "elheddaf_tv", "el_heddaf_tv"):
        spec = hr.resolve_channel_arg(arg)
        assert spec.feed_label == "Dedicated IPTV feed" and callable(spec.source)
        assert spec.source().startswith(HOST) and spec.source().endswith(".m3u8")
    assert hr.resolve_channel_arg("elheddaf_tv").name == "El-Heddaf TV"


def test_pilot_uses_the_iptv_feed_and_drops_the_web_only_video_helpers(overrides):
    from kihc.service import pilot
    spec, kw = pilot.pilot_channel("elheddaf")
    assert spec.feed_label == "Dedicated IPTV feed" and spec.source() == f"{HOST}/407.m3u8"
    assert "ffmpeg_input_args" not in kw                             # the web feed's -live_start_index tuning is not applied


def test_multi_stream_resolver_prefers_the_override_over_the_catalogue_source(overrides):
    ch = msm.ChannelProfile("samira_tv", "Samira TV", "priority", source={"type": "playlist", "match": ["Samira"]})
    assert msm.DefaultResolver(overrides)(ch) == f"{HOST}/408.m3u8"          # the playlist scraper is never reached
    ch2 = msm.ChannelProfile("el_heddaf_tv", "El-Heddaf TV", "priority",
                             source={"type": "hls", "url": "https://live.elheddaftv.com:8081/x.m3u8"})
    assert msm.DefaultResolver(overrides)(ch2) == f"{HOST}/407.m3u8"          # the web HLS source is superseded


def test_ingest_command_stays_audio_only(overrides):
    cmd = msm.ffmpeg_command(f"{HOST}/407.m3u8", stall_timeout=15, live_start_index=-2, ffmpeg="ffmpeg")
    assert "-vn" in cmd and "-sn" in cmd and f"{HOST}/407.m3u8" in cmd


def test_credentials_are_masked_for_display():
    assert iptv.mask_url(f"{HOST}/407.m3u8") == "http://iptv.test:8080/live/<user>/<pass>/407.m3u8"


# ------------------------------------------------------------------------------------------ the shipped file
def test_shipped_overrides_cover_every_mapped_channel_with_a_direct_m3u8():
    data = json.loads(iptv.OVERRIDES_FILE.read_text(encoding="utf-8"))
    urls = iptv.parse_overrides(data)
    assert len(urls) == len(data) == 17 and not {"el_watania_tv", "dzair_tv"} & set(urls)    # secondary extras dropped
    assert all(re.fullmatch(r"http://[\w.\-]+:\d+/live/[^/]+/[^/]+/\d+\.m3u8", u) for u in urls.values())
    assert len(set(urls.values())) == 17                                   # one distinct provider stream per channel
    cat = msm.load_catalogue()
    assert {iptv.squash_key(k) for k in urls} <= {iptv.squash_key(c) for c in cat.channels}   # every key is a catalogue channel
    core = {"elheddaf_tv": "407", "echorouk_tv": "400", "ennahar_tv": "262", "samira_tv": "408", "el_bilad_tv": "282"}
    assert all(urls[k].endswith(f"/{sid}.m3u8") for k, sid in core.items())               # the Tier-1 commercial fleet
    assert json.loads((iptv.DATA_DIR / "service_channels.json").read_text())["channels"] == [
        "el_heddaf_tv", "echorouk_tv", "ennahar_tv", "samira_tv", "el_bilad_tv"]
    resolver = msm.DefaultResolver()
    assert all(resolver(ch) == iptv.override_url(ch.id) for ch in cat.channels.values() if iptv.override_url(ch.id))


def test_audio_probe_reports_failure_cleanly_for_an_unreachable_feed():
    r = iptv.probe_audio("http://127.0.0.1:9/none.m3u8", seconds=1, timeout=8)
    assert not r["ok"] and r["pcm_bytes"] == 0 and r["rc"] != 0


def test_iptv_only_mode_never_falls_back_to_scrapers(tmp_path, monkeypatch):
    monkeypatch.setattr(iptv, "OVERRIDES_FILE", tmp_path / "o.json")
    (tmp_path / "o.json").write_text(json.dumps({"samira_tv": {"stream_url": f"{HOST}/408.m3u8"}}), encoding="utf-8")
    monkeypatch.setenv("KIHC_IPTV_ONLY", "1")

    def scraper(*a, **k):
        raise AssertionError("scraper consulted")
    monkeypatch.setattr(iptv, "check_stream", scraper)
    monkeypatch.setattr(iptv, "resolve_hls_url", scraper)
    assert iptv.resolve_channel_stream("samira_tv").ok
    gone = iptv.resolve_channel_stream("ennahar_tv")                         # a YouTube-backed channel with no override
    assert not gone.ok and [a[0] for a in gone.attempts] == ["override"]
    ch = msm.ChannelProfile("ennahar_tv", "Ennahar", "priority", source={"type": "broadcaster", "key": "ennahar_tv"})
    with pytest.raises(msm.ChannelUnresolved):
        msm.DefaultResolver(tmp_path / "o.json")(ch)


def test_every_ingest_command_caps_probe_memory_and_stays_audio_only():
    from kihc.config import FFMPEG_LOW_MEM_ARGS
    args = " ".join(FFMPEG_LOW_MEM_ARGS)
    assert "-probesize 262144" in args and "-analyzeduration 1000000" in args and "-max_delay 500000" in args
    cmd = " ".join(msm.ffmpeg_command(f"{HOST}/407.m3u8", stall_timeout=15, live_start_index=-2, ffmpeg="ffmpeg"))
    assert "-probesize 262144" in cmd and "-vn" in cmd and "-acodec pcm_s16le -ar 16000 -ac 1 -f s16le" in cmd


# ------------------------------------------------------------------------------------------ one-time-token provider relay
@pytest.fixture
def token_provider():
    """Stand-in for the IPTV provider: the playlist URL 302s to an edge URL whose token works exactly once (else HTTP 406)."""
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    used, counter = set(), [0]

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.path.startswith("/live/u/p/"):
                counter[0] += 1
                self.send_response(302)
                self.send_header("Location", f"/edge/408.m3u8?token=t{counter[0]}")
                self.end_headers()
            elif self.path.startswith("/edge/408.m3u8"):
                tok = self.path.split("token=")[1]
                if tok in used:
                    self.send_response(406)
                    self.end_headers()
                    return
                used.add(tok)
                body = f"#EXTM3U\n#EXT-X-TARGETDURATION:10\n#EXT-X-MEDIA-SEQUENCE:{counter[0]}\n#EXTINF:10.0,\n/hls/abc/408_{counter[0]}.ts\n".encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_response(404)
                self.end_headers()
    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def test_relay_reloads_a_one_time_token_playlist_that_breaks_a_direct_reload(token_provider):
    import requests

    from kihc.stream.iptv_relay import PlaylistRelay
    upstream = f"{token_provider}/live/u/p/408.m3u8"
    first = requests.get(upstream, timeout=5, allow_redirects=False).headers["Location"]
    assert requests.get(token_provider + first, timeout=5).status_code == 200
    assert requests.get(token_provider + first, timeout=5).status_code == 406          # why ffmpeg's own reload failed
    relay = PlaylistRelay()
    try:
        local = relay.register(upstream)
        assert relay.register(upstream) == local                                       # stable URL per upstream
        seqs = []
        for _ in range(4):                                                             # four reloads, four fresh tokens
            r = requests.get(local, timeout=5)
            assert r.status_code == 200 and r.text.startswith("#EXTM3U")
            seg = [ln for ln in r.text.splitlines() if ln.endswith(".ts")][0]
            assert seg.startswith(token_provider + "/hls/abc/408_")                    # absolute edge URL: media bypasses Python
            seqs.append(seg)
        assert len(set(seqs)) == 4
        assert requests.get(local.rsplit("/", 1)[0] + "/nope.m3u8", timeout=5).status_code == 404
    finally:
        relay.close()


def test_worker_resolver_hands_out_the_relay_url_only_when_enabled(overrides, monkeypatch):
    from kihc.stream import iptv_relay
    assert iptv.channel_resolver("samira_tv")() == f"{HOST}/408.m3u8"
    monkeypatch.setenv("KIHC_IPTV_RELAY", "1")
    local = iptv.channel_resolver("samira_tv")()
    try:
        assert local.startswith("http://127.0.0.1:") and local.endswith(".m3u8") and "PASS" not in local
    finally:
        iptv_relay.get_relay().close()


def test_telemetry_survives_dedicated_feed_channels_missing_from_the_broadcaster_registry():
    from kihc.app.components.visual_player import resolve_preview_source
    assert resolve_preview_source("Samira TV", "channel:samira_tv") == (None, False)     # was a KeyError -> HTTP 500
