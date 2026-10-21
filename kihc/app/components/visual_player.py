"""Client-side live video preview for the radar (visual sanity check only).

Nothing here decodes media in Python: the dashboard only emits HTML. The browser plays
HLS via hls.js (or Safari's native HLS) or YouTube's own embed. The player is dormant:
no network request is made until the viewer clicks "Load preview", nothing autoplays,
and audio starts muted.

Browser limits (shown in the fallback UI): many IPTV servers send no CORS headers, some
require a Referer the browser cannot set, and an https dashboard cannot load http streams.
In all those cases the preview fails while KIHC's audio monitoring is unaffected.
"""
import hashlib
import html
import json
import re
from urllib.parse import parse_qs, urlparse

HLS_JS_URL = "https://cdn.jsdelivr.net/npm/hls.js@1.5.13/dist/hls.min.js"   # pinned release
PLAYER_HEIGHT = 300
DISCLAIMER = "Client-side preview only · In-memory audio engine operates independently"

_YT_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")
_YT_CHANNEL = re.compile(r"^UC[A-Za-z0-9_-]{22}$")


def is_youtube_url(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return host.endswith("youtube.com") or host.endswith("youtube-nocookie.com") or host == "youtu.be"


def youtube_embed_url(url: str) -> str | None:
    """Watch / short / live / embed / channel-live URL -> embeddable URL with autoplay=0 (None if unknown)."""
    p = urlparse(url)
    host = (p.hostname or "").lower()
    parts = [s for s in p.path.split("/") if s]
    vid = None
    if host == "youtu.be" and parts:
        vid = parts[0]
    elif parts[:1] == ["watch"]:
        vid = parse_qs(p.query).get("v", [None])[0]
    elif len(parts) >= 2 and parts[0] in ("live", "embed", "shorts") and parts[1] != "live_stream":
        vid = parts[1]
    if vid and _YT_ID.match(vid):
        return f"https://www.youtube-nocookie.com/embed/{vid}?autoplay=0&mute=1&playsinline=1&rel=0"
    channel = None
    if len(parts) >= 2 and parts[0] == "channel":
        channel = parts[1]
    elif parts[:2] == ["embed", "live_stream"]:
        channel = parse_qs(p.query).get("channel", [None])[0]
    if channel and _YT_CHANNEL.match(channel):
        return f"https://www.youtube.com/embed/live_stream?channel={channel}&autoplay=0&mute=1&playsinline=1"
    return None                                    # e.g. an @handle: the embed API needs a channel id


def _js(value) -> str:
    """JSON literal safe inside an inline <script>: '</script>', '<!--' and '&' can't break out."""
    return json.dumps(value).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def _is_hls(url: str) -> bool:
    return ".m3u8" in urlparse(url).path.lower()


def _safe_http(url) -> bool:
    return isinstance(url, str) and urlparse(url.strip()).scheme in ("http", "https")


def _dom_id(channel_name: str, url: str) -> str:
    """Deterministic id so the radar's 3-second refresh produces identical HTML (no remount)."""
    return "kp" + hashlib.sha1(f"{channel_name}|{url}".encode()).hexdigest()[:10]


_STYLE = """
<style>
  :root { --ink:#0b0b0b; --muted:#52514e; --rule:#d9d8d4; --surface:#fcfcfb; --accent:#2a78d6; }
  @media (prefers-color-scheme: dark) { :root { --ink:#f0efec; --muted:#c3c2b7; --rule:#383835;
    --surface:#1a1a19; --accent:#3987e5; } }
  * { box-sizing:border-box; }
  body { margin:0; font:13px/1.4 system-ui,-apple-system,"Segoe UI",sans-serif; color:var(--ink);
         background:transparent; }
  .kp { border:1px solid var(--rule); border-radius:8px; overflow:hidden; background:var(--surface); }
  .frame { position:relative; width:100%; aspect-ratio:16/9; background:#000; }
  .frame video, .frame iframe { position:absolute; inset:0; width:100%; height:100%; border:0; }
  .gate { position:absolute; inset:0; display:flex; flex-direction:column; align-items:center;
          justify-content:center; gap:6px; color:#f0efec; background:#111; text-align:center; padding:12px; }
  .bar { display:flex; gap:8px; align-items:center; padding:6px 8px; }
  button { font:inherit; cursor:pointer; border:1px solid var(--rule); background:transparent; color:var(--ink);
           border-radius:6px; padding:4px 10px; }
  button.primary { background:var(--accent); border-color:var(--accent); color:#fff; }
  .msg { color:var(--muted); font-size:12px; }
  .err { display:none; padding:8px; font-size:12px; color:var(--ink); border-top:1px solid var(--rule); }
  .badge { display:inline-block; font-size:11px; color:var(--muted); border:1px solid var(--rule);
           border-radius:999px; padding:1px 8px; margin:6px 8px 8px; }
</style>
"""


def _badge(show: bool) -> str:
    return f'<span class="badge">{html.escape(DISCLAIMER)}</span>' if show else ""


def _fallback_html(channel_name: str, reason: str, link: str | None = None, badge: str = "") -> str:
    link_html = (f'<div class="msg"><a href="{html.escape(link, quote=True)}" target="_blank" '
                 f'rel="noopener noreferrer">Open the source in a new tab</a></div>') if link else ""
    return (f"<!doctype html><html><head><meta charset='utf-8'>{_STYLE}</head><body>"
            f'<div class="kp"><div class="frame"><div class="gate"><div>📺 {html.escape(channel_name)}</div>'
            f'<div class="msg" style="color:#c3c2b7">{html.escape(reason)}</div>{link_html}</div></div>'
            f'{badge}</div></body></html>')


def _youtube_html(channel_name: str, embed: str, uid: str, badge: str = "") -> str:
    return (f"<!doctype html><html><head><meta charset='utf-8'>{_STYLE}</head><body>"
            f'<div class="kp"><div class="frame" id="{uid}-f"><div class="gate">'
            f'<div>▶ {html.escape(channel_name)} · YouTube Live</div>'
            f'<button class="primary" id="{uid}-go" type="button">Load preview</button>'
            f'<div class="msg" style="color:#c3c2b7">Nothing loads until you click. Starts muted.</div></div></div>'
            f'{badge}</div>'
            "<script>(function(){"
            f"var f=document.getElementById({_js(uid + '-f')});"
            f"document.getElementById({_js(uid + '-go')}).addEventListener('click',function(){{"
            "var i=document.createElement('iframe');"
            f"i.src={_js(embed)};"
            "i.allow='encrypted-media; picture-in-picture; fullscreen';"
            "i.referrerPolicy='strict-origin-when-cross-origin';"
            f"i.title={_js(channel_name + ' live preview')};"
            "f.innerHTML='';f.appendChild(i);});"
            "})();</script></body></html>")


def _hls_html(channel_name: str, url: str, uid: str, badge: str = "") -> str:
    return f"""<!doctype html><html><head><meta charset="utf-8">{_STYLE}</head><body>
<div class="kp">
  <div class="frame" id="{uid}-f">
    <video id="{uid}-v" controls muted playsinline preload="none" aria-label="{html.escape(channel_name, quote=True)} live preview"></video>
    <div class="gate" id="{uid}-g">
      <div>▶ {html.escape(channel_name)} · HLS</div>
      <button class="primary" id="{uid}-go" type="button">Load preview</button>
      <div class="msg" style="color:#c3c2b7">Nothing loads until you click. Starts muted.</div>
    </div>
  </div>
  <div class="bar">
    <button id="{uid}-mute" type="button" disabled>🔇 Unmute</button>
    <button id="{uid}-stop" type="button" disabled>■ Stop</button>
    <span class="msg" id="{uid}-s">Idle</span>
  </div>
  <div class="err" id="{uid}-e"></div>
  {badge}
</div>
<script>
(function(){{
  var SRC = {_js(url)}, HLSJS = {_js(HLS_JS_URL)};
  var v = document.getElementById({_js(uid + '-v')}), gate = document.getElementById({_js(uid + '-g')});
  var go = document.getElementById({_js(uid + '-go')}), mute = document.getElementById({_js(uid + '-mute')});
  var stop = document.getElementById({_js(uid + '-stop')}), st = document.getElementById({_js(uid + '-s')});
  var err = document.getElementById({_js(uid + '-e')}), hls = null;
  function fail(reason) {{
    err.style.display = 'block';
    err.textContent = 'Preview unavailable in the browser (' + reason + '). Common causes: the stream is offline, ' +
      'the server sends no CORS headers or needs a Referer, or an https page cannot load an http stream. ' +
      'KIHC audio monitoring is unaffected.';
    st.textContent = 'Unavailable'; teardown();
  }}
  function teardown() {{
    if (hls) {{ hls.destroy(); hls = null; }}
    v.removeAttribute('src'); v.load(); stop.disabled = true; mute.disabled = true;
  }}
  function start() {{
    gate.style.display = 'none'; err.style.display = 'none'; st.textContent = 'Connecting…';
    stop.disabled = false; mute.disabled = false;
    if (v.canPlayType('application/vnd.apple.mpegurl')) {{ v.src = SRC; }}
    else if (window.Hls && window.Hls.isSupported()) {{
      hls = new window.Hls({{ lowLatencyMode: false, maxBufferLength: 10, backBufferLength: 0 }});
      hls.on(window.Hls.Events.ERROR, function(_, d) {{ if (d.fatal) fail(d.type + (d.details ? ': ' + d.details : '')); }});
      hls.loadSource(SRC); hls.attachMedia(v);
    }} else {{ fail('this browser cannot play HLS'); return; }}
    v.play().catch(function(){{}});
  }}
  go.addEventListener('click', function() {{
    if (window.Hls || v.canPlayType('application/vnd.apple.mpegurl')) {{ start(); return; }}
    st.textContent = 'Loading player…';
    var s = document.createElement('script'); s.src = HLSJS; s.async = true;
    s.onload = start; s.onerror = function() {{ fail('player library could not be loaded'); }};
    document.head.appendChild(s);
  }});
  mute.addEventListener('click', function() {{ v.muted = !v.muted; mute.textContent = v.muted ? '🔇 Unmute' : '🔊 Mute'; }});
  stop.addEventListener('click', function() {{ teardown(); gate.style.display = 'flex'; st.textContent = 'Stopped'; }});
  v.addEventListener('playing', function() {{ st.textContent = 'Playing (muted until you unmute)'; }});
  v.addEventListener('error', function() {{ if (!hls) fail('media error'); }});
}})();
</script></body></html>"""


def build_player_html(stream_url, channel_name: str, is_youtube: bool = False, show_badge: bool = True) -> str:
    """Pure HTML for the preview (testable without Streamlit)."""
    channel_name = channel_name or "Channel"
    badge = _badge(show_badge)
    if not _safe_http(stream_url):
        return _fallback_html(channel_name, "No browser-playable stream for this channel "
                                            "(local file, unresolved channel, or non-http source).", badge=badge)
    url = stream_url.strip()
    if is_youtube or is_youtube_url(url):
        embed = youtube_embed_url(url)
        if not embed:
            return _fallback_html(channel_name, "This YouTube URL cannot be embedded directly "
                                                "(an @handle needs its channel id).", link=url, badge=badge)
        return _youtube_html(channel_name, embed, _dom_id(channel_name, embed), badge)
    if _is_hls(url):
        return _hls_html(channel_name, url, _dom_id(channel_name, url), badge)
    return _fallback_html(channel_name, "Stream format not supported in the browser preview (HLS .m3u8 "
                                        "or YouTube only).", link=url, badge=badge)


def render_live_player(stream_url, channel_name: str, is_youtube: bool = False, height: int = PLAYER_HEIGHT,
                       show_badge: bool = True):
    """Render the dormant client-side preview inside the current Streamlit container.

    st.iframe runs this HTML with same-origin access to the app, which is why every untrusted
    value (stream URLs, channel names) is HTML-escaped or emitted through _js().
    """
    import streamlit as st
    doc = build_player_html(stream_url, channel_name, is_youtube, show_badge)
    if hasattr(st, "iframe"):                              # Streamlit >= 1.5x
        st.iframe(doc, height=height)
    else:                                                  # older Streamlit
        import streamlit.components.v1 as components
        components.html(doc, height=height, scrolling=False)


def resolve_preview_source(channel_name: str, stream_ref=None, *, overrides: dict | None = None,
                           resolver=None, iptv_urls: dict | None = None) -> tuple[str | None, bool]:
    """Pick what the browser should play for a monitored channel -> (url, is_youtube).

    Order: channel_overrides.json -> national broadcaster resolution (direct HLS, else the live
    YouTube channel's embed) -> the channel's configured URL (IPTV list or stream_ref).
    The signed googlevideo HLS used by the audio engine is never handed to the browser (IP-bound,
    CORS-blocked); YouTube channels play through YouTube's own embed instead.
    """
    from kihc.connectors.iptv_manager import NATIONAL_BROADCASTERS
    if isinstance(stream_ref, str) and stream_ref.startswith("youtube:"):
        return stream_ref[len("youtube:"):], True
    key = next((k for k, b in NATIONAL_BROADCASTERS.items() if b.name == channel_name), None)
    if isinstance(stream_ref, str) and stream_ref.startswith("channel:"):
        key = stream_ref[len("channel:"):]
    if key:
        override = (overrides or {}).get(key)
        if override and _safe_http(override):
            return override, False
        if resolver is not None:
            res = resolver(key)                          # dict(url, stream_type, attempts)
            if res and res.get("stream_type") == "hls" and res.get("url"):
                return res["url"], False
            if res and res.get("stream_type") == "youtube_live":
                target = next((a[1] for a in res.get("attempts", []) if a[0] == "youtube" and a[2] == "live"), None)
                if target:
                    return target, True
        b = NATIONAL_BROADCASTERS.get(key)               # Phase 28: dedicated-feed channels (samira_tv, ...) are not in the registry
        if b and b.hls:
            return b.hls[0], False
        if b and b.youtube:                              # not live now: YouTube's embed shows the offline slate
            return f"https://www.youtube.com/channel/{b.youtube[0]}/live", True
    if isinstance(stream_ref, str) and _safe_http(stream_ref):
        return stream_ref, is_youtube_url(stream_ref)
    url = (iptv_urls or {}).get(channel_name)
    if url:
        return url, is_youtube_url(url)
    return None, False
