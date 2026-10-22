"""Algerian free-to-air IPTV channel discovery.

Resolution order for the channel list:
  1. live fetch of the iptv-org Algeria playlist
  2. last successfully fetched copy cached under data/iptv_cache/ (small text file)
  3. FALLBACK_CHANNELS below (streams verified to decode audio on the date noted)

Public IPTV URLs rotate often; treat every list as a candidate set and call
check_streams() before monitoring.
"""
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import requests

from spotly.config import DATA_DIR, IPTV_CACHE_DIR
from spotly.connectors.youtube_live import live_url, resolve_hls_url

PLAYLIST_URL = "https://iptv-org.github.io/iptv/countries/dz.m3u"
CACHE_FILE = IPTV_CACHE_DIR / "dz.m3u"
USER_AGENT = "Mozilla/5.0 (Spotly broadcast monitor)"

# Verified 2026-10-12: FFmpeg decoded >= 4 s of audio from each URL.
# Not in the upstream list on that date (so not included): Echorouk, El Bilad, Ennahar, TV1.
# Upstream entries that failed on that date: AL24 News (404), TV4 (401), Elwatania TV (timeout).
FALLBACK_VERIFIED_ON = "2026-10-12"
FALLBACK_CHANNELS = {
    "Amou Yazid TV": "http://mkstream.servehttp.com:1940/AmouYazid/ay1/playlist.m3u8",
    "CNA (Algeria)": "https://live.creacast.com/cna/stream/playlist.m3u8",
    "El-Heddaf TV": "https://live.elheddaftv.com:8081/elheddaftv/index.m3u8",
    "StoryChannel TV": "https://136044159.r.cdnsun.net/storychannel.m3u8",
    "TV2 (Algeria)": "http://185.9.2.18/chid_347/index.m3u8",
}


@dataclass
class Channel:
    name: str
    url: str
    tvg_id: str = ""
    group: str = ""
    logo: str = ""
    headers: dict = field(default_factory=dict)   # e.g. Referer / User-Agent from #EXTVLCOPT
    source: str = "remote"                         # remote | cache | fallback
    active: bool | None = None                     # None = not checked yet
    check_detail: str = ""


_ATTR = re.compile(r'([\w-]+)="([^"]*)"')
_QUALITY = re.compile(r"\s*\((\d{3,4}[pi])\)|\s*\[[^\]]*\]")


def clean_name(raw: str) -> str:
    """'TV2 (Algeria) (1080p) [Not 24/7]' -> 'TV2 (Algeria)'"""
    return _QUALITY.sub("", raw).strip()


def parse_m3u(text: str, source: str = "remote") -> list[Channel]:
    channels, info, headers = [], None, {}
    for line in (l.strip() for l in text.splitlines()):
        if not line:
            continue
        if line.startswith("#EXTINF"):
            attrs = dict(_ATTR.findall(line))
            title = line.rsplit(",", 1)[-1].strip() if "," in line else ""
            info, headers = (title, attrs), {}
        elif line.startswith("#EXTVLCOPT:"):
            key, _, val = line[len("#EXTVLCOPT:"):].partition("=")
            if key == "http-referrer":
                headers["Referer"] = val
            elif key == "http-user-agent":
                headers["User-Agent"] = val
        elif line.startswith("#"):
            continue
        elif info is not None:
            title, attrs = info
            channels.append(Channel(name=clean_name(title) or attrs.get("tvg-name", line), url=line,
                                    tvg_id=attrs.get("tvg-id", ""), group=attrs.get("group-title", ""),
                                    logo=attrs.get("tvg-logo", ""), headers=headers, source=source))
            info, headers = None, {}
    return channels


def fallback_channels() -> list[Channel]:
    return [Channel(name=n, url=u, source="fallback") for n, u in FALLBACK_CHANNELS.items()]


def fetch_channels(url: str = PLAYLIST_URL, timeout: float = 10.0) -> list[Channel]:
    """Remote playlist -> cached copy -> built-in fallback. Never raises."""
    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT})
        r.raise_for_status()
        channels = parse_m3u(r.text, "remote")
        if channels:
            try:
                CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
                CACHE_FILE.write_text(r.text, encoding="utf-8")
            except OSError:
                pass
            return channels
    except requests.RequestException:
        pass
    try:
        channels = parse_m3u(CACHE_FILE.read_text(encoding="utf-8"), "cache")
        if channels:
            return channels
    except OSError:
        pass
    return fallback_channels()


def check_stream(url: str, timeout: float = 3.0, headers: dict | None = None) -> tuple[bool, str]:
    """Lightweight liveness probe: HEAD, then a small GET if HEAD is refused.

    An HLS URL counts as active when it answers 2xx and (for GET) the body starts
    with #EXTM3U. This proves the playlist is served, not that audio decodes.
    """
    hdrs = {"User-Agent": USER_AGENT, **(headers or {})}
    t0 = time.monotonic()
    try:
        if ".m3u8" not in url:   # HLS needs the body check below, so skip straight to GET
            r = requests.head(url, timeout=timeout, headers=hdrs, allow_redirects=True)
            if r.ok:
                return True, f"HEAD {r.status_code} in {time.monotonic() - t0:.1f}s"
        r = requests.get(url, timeout=timeout, headers=hdrs, stream=True)
        try:
            head = next(r.iter_content(64), b"")
        finally:
            r.close()
        if not r.ok:
            return False, f"HTTP {r.status_code}"
        if ".m3u8" in url and not head.lstrip().startswith(b"#EXTM3U"):
            return False, "not an HLS playlist"
        return True, f"GET {r.status_code} in {time.monotonic() - t0:.1f}s"
    except requests.RequestException as exc:
        return False, type(exc).__name__


def check_streams(channels: list[Channel], timeout: float = 3.0, max_workers: int = 8) -> list[Channel]:
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        results = list(ex.map(lambda c: check_stream(c.url, timeout, c.headers), channels))
    for ch, (ok, detail) in zip(channels, results):
        ch.active, ch.check_detail = ok, detail
    return channels


def active_channels(timeout: float = 3.0) -> list[Channel]:
    return [c for c in check_streams(fetch_channels(), timeout) if c.active]


# --------------------------------------------------------------------------- national broadcasters
#
# YouTube channels verified 2026-10-12 by fetching each channel page (title, canonical channel id).
# Channel ids are used rather than handles because handles can be renamed.
#   - The handles '@echorouktv', '@ennahartv', '@elbilad' do NOT exist (HTTP 404 on 2026-10-12).
#   - El Bilad's channel id is the one linked from its own site, elbilad.net.
#   - Echorouk / Ennahar web players (player.dzsecurity.net) are access-controlled (Ennahar: 403
#     outside the publisher's site). Spotly does not bypass that, so these channels ship with no
#     direct HLS. An authorised feed URL can be supplied in data/channel_overrides.json:
#         {"echorouk_tv": "https://.../index.m3u8"}

OVERRIDES_FILE = DATA_DIR / "channel_overrides.json"


@dataclass(frozen=True)
class Broadcaster:
    key: str
    name: str
    youtube: tuple = ()           # ordered candidates: channel ids / handles / URLs
    hls: tuple = ()               # verified direct HLS URLs, tried first
    website: str = ""
    note: str = ""


NATIONAL_BROADCASTERS = {
    "echorouk_tv": Broadcaster(
        "echorouk_tv", "Echorouk TV",
        youtube=("UCd9ox6D3VbhNp3kJNTxQzCQ",      # 'EchoroukTV' (@echorouktvofficiel)
                 "UCR8l6OQCNzsAA26AEcCKcKw"),     # 'Echorouk Online' (@EchoroukNews)
        website="https://www.echoroukonline.com/live"),
    "ennahar_tv": Broadcaster(
        "ennahar_tv", "Ennahar TV",
        youtube=("UC57OCoLoU6zAtBdJOmwg2vA",      # 'EnnaharTv' (youtube.com/user/ENNAHARTVLIVE)
                 "UCISILw7UC9gYmVtlyQOS9aw"),     # 'Ennahar Tv Plus' (@ennahartvplus)
        website="https://www.ennaharonline.com/live/"),
    "el_bilad_tv": Broadcaster(
        "el_bilad_tv", "El Bilad TV",
        youtube=("UCfF6CkUvCBleM19CkgNyrYw",),    # 'ELBILAD TV' (@elbiladtv1), linked from elbilad.net
        website="https://www.elbilad.net"),
    "el_heddaf_tv": Broadcaster(
        "el_heddaf_tv", "El-Heddaf TV",
        hls=(FALLBACK_CHANNELS["El-Heddaf TV"],),
        note="direct HLS verified to decode audio 2026-10-12"),
}
FEATURED_KEYS = ("echorouk_tv", "ennahar_tv", "el_bilad_tv", "el_heddaf_tv")


@dataclass
class ResolvedStream:
    channel_key: str
    name: str
    url: str | None
    stream_type: str                 # 'hls' | 'youtube_live' | 'unavailable'
    attempts: list = field(default_factory=list)   # (method, target, outcome) in order tried

    @property
    def ok(self) -> bool:
        return self.url is not None


def squash_key(key: str) -> str:
    """'el_heddaf_tv' == 'elheddaf_tv' == 'El-Heddaf TV': channel keys compare without separators or case."""
    return "".join(ch for ch in str(key).lower() if ch.isalnum())


def parse_overrides(data) -> dict:
    """{channel_key: url} from the parsed channel_overrides.json. An entry is a URL string (legacy) or
    {"stream_url": "..."} (Phase 28, the dedicated IPTV feeds); blank or malformed entries are skipped."""
    out = {}
    if not isinstance(data, dict):
        return out
    for key, v in data.items():
        url = v.get("stream_url") if isinstance(v, dict) else v
        if isinstance(url, str) and url.strip() and not str(key).startswith("_"):
            out[str(key)] = url.strip()
    return out


def _overrides() -> dict:
    try:
        return parse_overrides(json.loads(OVERRIDES_FILE.read_text(encoding="utf-8")))
    except (OSError, ValueError, AttributeError):
        return {}


def load_overrides() -> dict:
    """{channel_key: url} from data/channel_overrides.json (empty if missing or invalid)."""
    return _overrides()


def iptv_only() -> bool:
    """SPOTLY_IPTV_ONLY=1: the dedicated subscription is the only source; a channel without an override is unavailable."""
    return os.environ.get("SPOTLY_IPTV_ONLY", "").lower() in ("1", "true", "yes", "on")


def override_url(channel_key: str) -> str | None:
    """The dedicated feed for a channel (separator-insensitive key match), or None."""
    want = squash_key(channel_key)
    return next((u for k, u in _overrides().items() if squash_key(k) == want), None)


_CHANNEL_NAMES = {"elheddaf_tv": "El-Heddaf TV", "echorouk_tv": "Echorouk TV", "echorouk_news": "Echorouk News",
                 "ennahar_tv": "Ennahar TV", "samira_tv": "Samira TV", "elbilad_tv": "El Bilad TV",
                 "elhayat_tv": "El Hayat TV", "al24_news": "AL24 News", "elwatania_tv": "El Watania TV",
                 "bahia_tv": "Bahia TV", "dzair_tv": "Dzair TV", "tv3_algerie3": "TV3 (Algerie 3)",
                 "tv4_tamazight": "TV4 (Tamazight)", "tv5_coran": "TV5 (Coran)", "tv6_jeunesse": "TV6 (Jeunesse)",
                 "tv7_elmaarifa": "TV7 El Maarifa", "tv8_edhakira": "TV8 Edhakira",
                 "tv9_parlementaire": "TV9 (Parlementaire)", "berbere_tv": "Berbere TV"}


CHANNEL_NAMES = {squash_key(k): v for k, v in _CHANNEL_NAMES.items()}


def display_name(channel_key: str) -> str:
    return CHANNEL_NAMES.get(squash_key(channel_key)) or str(channel_key).replace("_", " ").title()


def mask_url(url: str) -> str:
    """The URL with the subscription credentials hidden: http://host:8080/live/<user>/<pass>/407.m3u8."""
    return re.sub(r"(/live/)[^/]+/[^/]+/", r"\1<user>/<pass>/", url or "")


def resolve_channel_stream(channel_key: str, *, timeout: float = 3.0, session=None) -> ResolvedStream:
    """Direct HLS if reachable, else the channel's YouTube Live HLS via streamlink.

    Never raises: an unresolvable channel comes back with stream_type='unavailable'
    and the reason for every attempt in `.attempts`.
    """
    direct = override_url(channel_key)
    if direct:                  # Phase 28: the dedicated IPTV subscription is the ONLY source - no probe request (providers
        b = NATIONAL_BROADCASTERS.get(channel_key)       # cap concurrent connections), no web page, no YouTube
        return ResolvedStream(channel_key, b.name if b else display_name(channel_key), direct, "hls",
                              [("override", mask_url(direct), "dedicated IPTV feed")])
    if iptv_only():
        return ResolvedStream(channel_key, display_name(channel_key), None, "unavailable",
                              [("override", channel_key, "no dedicated IPTV feed (SPOTLY_IPTV_ONLY: no web / YouTube fallback)")])
    b = NATIONAL_BROADCASTERS.get(channel_key)
    if b is None:
        return ResolvedStream(channel_key, channel_key, None, "unavailable",
                              [("registry", channel_key, "unknown channel key")])
    result = ResolvedStream(b.key, b.name, None, "unavailable")
    override = _overrides().get(b.key)
    for url in ((override,) if override else ()) + b.hls:
        try:
            ok, detail = check_stream(url, timeout)
        except Exception as exc:             # defensive: a probe must never break resolution
            ok, detail = False, f"{type(exc).__name__}: {exc}"
        result.attempts.append(("hls", url, detail))
        if ok:
            result.url, result.stream_type = url, "hls"
            return result
    for ref in b.youtube:
        target = live_url(ref)
        try:
            result.url = resolve_hls_url(target, session=session)
            result.stream_type = "youtube_live"
            result.attempts.append(("youtube", target, "live"))
            return result
        except Exception as exc:              # YouTubeLiveError, network errors, plugin breakage
            result.attempts.append(("youtube", target, f"{type(exc).__name__}: {exc}"))
    return result


class ChannelUnavailable(RuntimeError):
    pass


def channel_resolver(channel_key: str, timeout: float = 3.0):
    """Callable for LiveStreamWorker(stream_url=...): re-resolves on every (re)connect.

    Raises ChannelUnavailable when nothing is live, which the worker treats like a
    dropped connection (exponential backoff, then try again) - so an offline channel
    is picked up automatically once it goes live.
    """
    def resolve() -> str:
        r = resolve_channel_stream(channel_key, timeout=timeout)
        if not r.ok:
            raise ChannelUnavailable(f"{r.name}: not live ({'; '.join(a[2][:60] for a in r.attempts)})")
        from spotly.stream import iptv_relay
        if iptv_relay.relay_enabled() and r.attempts and r.attempts[0][0] == "override":
            return iptv_relay.get_relay().register(r.url)     # one-time-token provider: stable local playlist (see iptv_relay)
        return r.url
    resolve.__name__ = f"channel:{channel_key}"
    return resolve


# ---------------------------------------------------------------------------------------------- Phase 28: audio probe
PROBE_RATE = 16_000


def probe_audio(url: str, seconds: float = 3.0, timeout: float = 30.0) -> dict:
    """Pull `seconds` of audio only (-vn, mono s16le @ 16 kHz) and let ffmpeg end the stream itself (-t).

    -> {ok, pcm_bytes, audio_seconds, rc, elapsed_s, peak_rss_mb, error}. ok = ffmpeg exited 0 and delivered at least 90 % of the
    requested samples. Run probes one at a time: a single IPTV subscription usually allows few concurrent connections."""
    import subprocess
    import threading
    import time

    from spotly.core.audio import ffmpeg_exe
    from spotly.config import FFMPEG_LOW_MEM_ARGS
    cmd = [ffmpeg_exe(), "-nostdin", "-hide_banner", "-loglevel", "error", *FFMPEG_LOW_MEM_ARGS,
           "-rw_timeout", str(int(timeout * 0.6 * 1e6)), "-i", url, "-vn", "-sn", "-dn", "-t", f"{seconds:g}", "-ac", "1", "-ar", str(PROBE_RATE), "-f", "s16le", "pipe:1"]
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    peak = [0.0]

    def watch():
        try:
            import psutil
            p = psutil.Process(proc.pid)
            while proc.poll() is None:
                peak[0] = max(peak[0], p.memory_info().rss / 2 ** 20)
                time.sleep(0.1)
        except Exception:
            pass
    threading.Thread(target=watch, daemon=True).start()
    try:
        out, err = proc.communicate(timeout=timeout)
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        proc.kill()
        out, err = proc.communicate()
        rc = -9
    got = len(out) / 2 / PROBE_RATE
    text = err.decode(errors="replace").strip().splitlines()
    return dict(ok=rc == 0 and got >= 0.9 * seconds, pcm_bytes=len(out), audio_seconds=round(got, 2), rc=rc,
                elapsed_s=round(time.time() - t0, 1), peak_rss_mb=round(peak[0], 1),
                error=mask_url(text[-1])[:160] if text and rc != 0 else "")
