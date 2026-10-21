"""Phase 5.1: client-side visual preview component (HTML generation only; no media decoding)."""
import json
import re
from html.parser import HTMLParser

import pytest

from kihc.app.components import visual_player as vp
from kihc.app.components.visual_player import (DISCLAIMER, HLS_JS_URL,
                                               build_player_html,
                                               is_youtube_url,
                                               resolve_preview_source,
                                               youtube_embed_url)

HLS = "https://live.elheddaftv.com:8081/elheddaftv/index.m3u8"


class _Collector(HTMLParser):
    """Parses the document and records tags/attrs; raises on unbalanced container tags."""
    VOID = {"meta", "br", "img", "input", "link", "source", "hr"}

    def __init__(self):
        super().__init__()
        self.stack, self.tags = [], []

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))
        if tag not in self.VOID:
            self.stack.append(tag)

    def handle_endtag(self, tag):
        assert self.stack and self.stack[-1] == tag, f"unbalanced </{tag}> (open: {self.stack[-3:]})"
        self.stack.pop()


def parse(doc):
    p = _Collector()
    p.feed(doc)
    p.close()
    assert p.stack == [], f"unclosed tags: {p.stack}"
    return p


def tags(doc, name):
    return [a for t, a in parse(doc).tags if t == name]


# ------------------------------------------------------------------ HLS player

def test_hls_player_structure_and_dormancy():
    doc = build_player_html(HLS, "El-Heddaf TV")
    assert doc.startswith("<!doctype html>")
    (video,) = tags(doc, "video")
    assert "autoplay" not in video                                   # never autoplays
    assert video["preload"] == "none" and "muted" in video and "controls" in video and "playsinline" in video
    assert "src" not in video                                        # nothing attached until the click
    assert HLS_JS_URL in doc and "@1.5.13" in HLS_JS_URL             # pinned CDN build
    assert not [s for s in tags(doc, "script") if s.get("src")]      # hls.js injected on demand, not preloaded
    assert json.dumps(HLS) in doc                                    # URL passed as a JS string literal
    buttons = [b.get("id", "") for b in tags(doc, "button")]
    assert any(b.endswith("-go") for b in buttons) and any(b.endswith("-mute") for b in buttons)
    assert "Load preview" in doc and "Unmute" in doc
    assert "CORS" in doc and "audio monitoring is unaffected" in doc  # fallback explanation present
    assert DISCLAIMER in doc


def test_html_is_deterministic_for_refreshes():
    assert build_player_html(HLS, "El-Heddaf TV") == build_player_html(HLS, "El-Heddaf TV")
    assert build_player_html(HLS, "El-Heddaf TV") != build_player_html(HLS, "Other TV")


def test_badge_can_be_suppressed():
    assert DISCLAIMER not in build_player_html(HLS, "El-Heddaf TV", show_badge=False)


# ------------------------------------------------------------------ YouTube

@pytest.mark.parametrize("url, expected", [
    ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "https://www.youtube-nocookie.com/embed/dQw4w9WgXcQ?"),
    ("https://youtu.be/dQw4w9WgXcQ", "https://www.youtube-nocookie.com/embed/dQw4w9WgXcQ?"),
    ("https://www.youtube.com/live/dQw4w9WgXcQ", "https://www.youtube-nocookie.com/embed/dQw4w9WgXcQ?"),
    ("https://www.youtube.com/channel/UCd9ox6D3VbhNp3kJNTxQzCQ/live",
     "https://www.youtube.com/embed/live_stream?channel=UCd9ox6D3VbhNp3kJNTxQzCQ&"),
    ("https://www.youtube.com/@echorouktvofficiel/live", None),          # handle: needs a channel id
    ("https://www.youtube.com/watch?v=not-an-id", None),
])
def test_youtube_embed_urls(url, expected):
    embed = youtube_embed_url(url)
    if expected is None:
        assert embed is None
    else:
        assert embed.startswith(expected) and "autoplay=0" in embed and "mute=1" in embed


def test_youtube_player_is_click_to_load_iframe():
    doc = build_player_html("https://youtu.be/dQw4w9WgXcQ", "Echorouk TV")
    assert not tags(doc, "iframe")                                   # dormant: iframe only created on click
    assert "createElement('iframe')" in doc and "autoplay=0" in doc and "Load preview" in doc


def test_is_youtube_flag_and_detection():
    assert is_youtube_url("https://m.youtube.com/watch?v=dQw4w9WgXcQ") and is_youtube_url("https://youtu.be/x")
    assert not is_youtube_url(HLS) and not is_youtube_url("https://notyoutube.com.evil.io/watch?v=x")
    doc = build_player_html("https://www.youtube.com/channel/UCd9ox6D3VbhNp3kJNTxQzCQ/live", "Echorouk TV",
                            is_youtube=True)
    assert "live_stream?channel=UCd9ox6D3VbhNp3kJNTxQzCQ" in doc and not tags(doc, "video")


# ------------------------------------------------------------------ fallbacks + safety

@pytest.mark.parametrize("url", [None, "", "C:\\recordings\\broadcast.mp4", "rtmp://host/live",
                                 "javascript:alert(1)", "data:text/html,<script>x</script>"])
def test_non_playable_sources_fall_back(url):
    doc = build_player_html(url, "Local TV")
    parse(doc)
    assert not tags(doc, "video") and not tags(doc, "iframe")
    assert "No browser-playable stream" in doc and "javascript:alert" not in doc


def test_unsupported_http_format_offers_link_only():
    doc = build_player_html("https://example.com/live/stream.mpd", "DASH TV")
    (a,) = tags(doc, "a")
    assert a["href"] == "https://example.com/live/stream.mpd" and a["rel"] == "noopener noreferrer"
    assert "HLS .m3u8 or YouTube only" in doc


def test_untrusted_text_is_escaped():
    evil_name = '<img src=x onerror=alert(1)>"</script>'
    evil_url = 'https://e.com/a.m3u8?x="</script><script>alert(1)</script>'
    doc = build_player_html(evil_url, evil_name)
    parse(doc)
    assert "<img src=x" not in doc and "&lt;img src=x" in doc
    assert "</script><script>alert(1)" not in doc
    assert not [a for t, a in parse(doc).tags if t == "img"]


def test_render_live_player_embeds_html(monkeypatch):
    import streamlit as st
    seen = {}
    monkeypatch.setattr(st, "iframe", lambda src, height: seen.update(src=src, height=height))
    vp.render_live_player(HLS, "El-Heddaf TV", height=280)
    assert seen["height"] == 280 and "<video" in seen["src"]


def test_render_live_player_falls_back_on_older_streamlit(monkeypatch):
    import streamlit as st
    import streamlit.components.v1 as components
    seen = {}
    monkeypatch.delattr(st, "iframe", raising=False)
    monkeypatch.setattr(components, "html", lambda body, height, scrolling: seen.update(body=body, height=height))
    vp.render_live_player(HLS, "El-Heddaf TV")
    assert seen["height"] == vp.PLAYER_HEIGHT and "<video" in seen["body"]


# ------------------------------------------------------------------ source selection (no network)

def test_preview_source_selection():
    # national HLS channel -> registry HLS
    assert resolve_preview_source("El-Heddaf TV", "channel:el_heddaf_tv") == (
        "https://live.elheddaftv.com:8081/elheddaftv/index.m3u8", False)
    # override wins
    assert resolve_preview_source("Echorouk TV", None, overrides={"echorouk_tv": "https://lic.dz/e.m3u8"}) == (
        "https://lic.dz/e.m3u8", False)
    # YouTube-only national channel -> its first verified channel id, as an embed (never googlevideo HLS)
    url, yt = resolve_preview_source("Echorouk TV", "channel:echorouk_tv")
    assert yt and url == "https://www.youtube.com/channel/UCd9ox6D3VbhNp3kJNTxQzCQ/live"
    # resolver result: live on the second candidate
    res = dict(stream_type="youtube_live", url="https://manifest.googlevideo.com/x.m3u8",
               attempts=[("youtube", "https://www.youtube.com/channel/UC1/live", "offline"),
                         ("youtube", "https://www.youtube.com/channel/UCR8l6OQCNzsAA26AEcCKcKw/live", "live")])
    url, yt = resolve_preview_source("Echorouk TV", None, resolver=lambda k: res)
    assert yt and "UCR8l6OQCNzsAA26AEcCKcKw" in url and "googlevideo" not in url
    # in-app YouTube URL worker, plain IPTV URL, IPTV list lookup, local file
    assert resolve_preview_source("YouTube x", "youtube:https://youtu.be/dQw4w9WgXcQ") == (
        "https://youtu.be/dQw4w9WgXcQ", True)
    assert resolve_preview_source("TV2 (Algeria)", "http://185.9.2.18/chid_347/index.m3u8")[0].endswith(".m3u8")
    assert resolve_preview_source("StoryChannel TV", None, iptv_urls={"StoryChannel TV": "https://s/x.m3u8"}) == (
        "https://s/x.m3u8", False)
    assert resolve_preview_source("SynthTV", "C:\\data\\broadcast.mp4") == (None, False)


def test_dashboard_wires_preview_into_radar():
    src = (vp.__file__.replace("components\\visual_player.py", "dashboard.py")
           .replace("components/visual_player.py", "dashboard.py"))
    text = open(src, encoding="utf-8").read()
    assert re.search(r'st\.expander\("📺 Live Video Preview \(Visual Check\)", expanded=False\)', text)
    assert "_video_preview(name, stream_ref)" in text and "render_live_player(" in text
