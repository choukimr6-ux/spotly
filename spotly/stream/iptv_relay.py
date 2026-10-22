"""Playlist relay for token-redirecting IPTV providers (Phase 28.2).

The dedicated IPTV provider answers `GET .../407.m3u8` with a 302 to a load-balanced edge host and a ONE-TIME token
(`...m3u8?token=...`; a second request with the same token is HTTP 406). FFmpeg's HLS demuxer refreshes a live playlist from the
URL it ended up on, so after the first segment every reload failed ("Failed to reload playlist 0") and the stream ended after
10 s - a reconnect storm with an audio gap each time.

The relay gives FFmpeg a STABLE local playlist URL. Each time FFmpeg reloads it, the relay asks the provider's original URL again
(fresh redirect, fresh token), rewrites the segment URIs to absolute edge-host URLs and returns the playlist. Segments are still
downloaded by FFmpeg directly from the edge, so no media passes through Python. One loopback thread-per-request HTTP server per
process; a few KB of memory.

    local = get_relay().register("http://provider/live/u/p/407.m3u8")      # -> http://127.0.0.1:PORT/r/<id>.m3u8
"""
import hashlib
import logging
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urljoin

import requests

log = logging.getLogger("spotly.iptv_relay")
UA = {"User-Agent": "Mozilla/5.0 (Spotly broadcast monitor)"}
FETCH_TIMEOUT_S = 12.0


def relay_enabled() -> bool:
    return os.environ.get("SPOTLY_IPTV_RELAY", "").lower() in ("1", "true", "yes", "on")


def rewrite_playlist(text: str, base_url: str) -> str:
    """Segment / sub-playlist URIs made absolute against the URL the provider finally answered from."""
    out = []
    for line in text.splitlines():
        s = line.strip()
        out.append(urljoin(base_url, s) if s and not s.startswith("#") else line)
    return "\n".join(out) + "\n"


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def log_message(self, *a):                 # silence the default stderr access log
        return

    def do_GET(self):
        upstream = self.server.routes.get(self.path.rsplit("/", 1)[-1].removesuffix(".m3u8"))
        if not upstream:
            self.send_error(404)
            return
        try:
            r = requests.get(upstream, timeout=FETCH_TIMEOUT_S, headers=UA, allow_redirects=True)
            if not r.ok or not r.text.lstrip().startswith("#EXTM3U"):
                raise RuntimeError(f"HTTP {r.status_code}")
            body = rewrite_playlist(r.text, r.url).encode()
        except Exception as exc:
            log.warning("relay: upstream playlist failed (%s: %s)", type(exc).__name__, str(exc)[:80])
            self.send_error(502)
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/vnd.apple.mpegurl")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


class PlaylistRelay:
    def __init__(self):
        self._lock = threading.Lock()
        self._server: ThreadingHTTPServer | None = None
        self._routes: dict[str, str] = {}

    def _ensure(self) -> ThreadingHTTPServer:
        with self._lock:
            if self._server is None:
                srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
                srv.daemon_threads = True
                srv.routes = self._routes
                threading.Thread(target=srv.serve_forever, name="spotly-iptv-relay", daemon=True).start()
                self._server = srv
            return self._server

    def register(self, upstream_url: str) -> str:
        """Stable local playlist URL for an upstream playlist URL (the same upstream always gets the same local URL)."""
        key = hashlib.sha1(upstream_url.encode()).hexdigest()[:12]
        srv = self._ensure()
        with self._lock:
            self._routes[key] = upstream_url
        return f"http://127.0.0.1:{srv.server_address[1]}/r/{key}.m3u8"

    def close(self) -> None:
        with self._lock:
            srv, self._server = self._server, None
        if srv is not None:
            srv.shutdown()
            srv.server_close()


_RELAY = PlaylistRelay()


def get_relay() -> PlaylistRelay:
    return _RELAY
