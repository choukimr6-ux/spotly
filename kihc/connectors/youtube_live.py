"""YouTube Live -> direct HLS URL via streamlink (no browser).

YouTube HLS URLs are signed and expire after a few hours, so pass
`youtube_resolver(url)` to LiveStreamWorker: it re-resolves on every reconnect.
"""
from streamlink import Streamlink
from streamlink.exceptions import NoPluginError, PluginError

# Lowest video variant first: KIHC only fingerprints audio, which every HLS variant carries.
DEFAULT_QUALITIES = ("worst", "360p", "best")


class YouTubeLiveError(RuntimeError):
    pass


def resolve_hls_url(url: str, qualities=DEFAULT_QUALITIES, session: Streamlink | None = None) -> str:
    session = session or Streamlink()
    try:
        streams = session.streams(url)
    except NoPluginError as exc:
        raise YouTubeLiveError(f"Not a supported URL: {url}") from exc
    except PluginError as exc:
        raise YouTubeLiveError(f"streamlink failed for {url}: {exc}") from exc
    if not streams:
        raise YouTubeLiveError(f"No live streams found at {url} (channel may be offline)")
    for q in qualities:
        stream = streams.get(q)
        if stream is not None and hasattr(stream, "url"):
            return stream.url
    raise YouTubeLiveError(f"No HLS variant among {list(streams)} for {url}")


def live_url(channel_ref: str) -> str:
    """'@handle', 'UC…' channel id, 'user/name' or a full URL -> the channel's /live URL."""
    ref = channel_ref.strip().rstrip("/")
    if ref.startswith("http"):
        return ref if ref.endswith("/live") else ref + "/live"
    if ref.startswith("UC") and len(ref) == 24:
        ref = f"channel/{ref}"
    return f"https://www.youtube.com/{ref}/live"


def youtube_resolver(url: str, qualities=DEFAULT_QUALITIES):
    """Callable for LiveStreamWorker(stream_url=...) that resolves a fresh HLS URL each connect."""
    def resolve() -> str:
        return resolve_hls_url(url, qualities)
    resolve.__name__ = f"youtube:{url}"
    return resolve
