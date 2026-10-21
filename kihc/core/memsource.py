"""Serve in-memory bytes to FFmpeg over a private loopback HTTP endpoint.

FFmpeg cannot seek a pipe, so MP4/MOV files whose 'moov' index sits at the end
(the common, non-faststart layout) fail to open from stdin. A throwaway
127.0.0.1 server with HTTP Range support lets FFmpeg seek inside an uploaded
file without it ever being written to disk.

    with InMemorySource(data, "ad.mp4") as url:
        fingerprint_file(url)
"""
import re
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import quote

_RANGE = re.compile(r"bytes=(\d*)-(\d*)$")
_BLOCK = 1 << 20


class InMemorySource:
    def __init__(self, data: bytes, filename: str = "upload.bin"):
        self._data = memoryview(data)
        self._name = re.sub(r"[^A-Za-z0-9_.-]+", "_", filename) or "upload.bin"
        self._token = secrets.token_urlsafe(16)        # unguessable path: other local processes can't read it
        self._server = None

    def __enter__(self) -> str:
        data, path = self._data, f"/{self._token}/{quote(self._name)}"

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):          # silence per-request logging
                pass

            def _serve(self, body: bool):
                if self.path != path:
                    self.send_error(404)
                    return
                size = len(data)
                start, end = 0, size - 1
                m = _RANGE.match(self.headers.get("Range", ""))
                if m and (m.group(1) or m.group(2)):
                    if m.group(1):
                        start = int(m.group(1))
                        end = min(int(m.group(2)), size - 1) if m.group(2) else size - 1
                    else:                           # suffix range: last N bytes
                        start = max(0, size - int(m.group(2)))
                    if start >= size or start > end:
                        self.send_response(416)
                        self.send_header("Content-Range", f"bytes */{size}")
                        self.send_header("Content-Length", "0")
                        self.end_headers()
                        return
                    self.send_response(206)
                    self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                else:
                    self.send_response(200)
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(end - start + 1))
                self.end_headers()
                if body:
                    try:
                        for off in range(start, end + 1, _BLOCK):
                            self.wfile.write(data[off:min(off + _BLOCK, end + 1)])
                    except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                        pass                        # FFmpeg closes connections when it seeks

            def do_GET(self):
                self._serve(True)

            def do_HEAD(self):
                self._serve(False)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        return f"http://127.0.0.1:{self._server.server_port}{path}"

    def __exit__(self, *exc):
        if self._server:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
