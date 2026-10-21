"""El-Heddaf TV direct web feed (no YouTube).

Verified 2026-10-13:
  * The official live page https://elheddaftv.com/elheddaf-live/ embeds
    https://live.elheddaftv.com:8081/elheddaftv/embed.html, a player for the publicly served HLS
    master https://live.elheddaftv.com:8081/elheddaftv/index.m3u8 (no token, no #EXT-X-KEY,
    Access-Control-Allow-Origin: *). Nothing is bypassed: this is the broadcaster's open feed.
  * The server also publishes an audio-only rendition, tracks-a1/mono.m3u8: ~112 KiB per 6 s segment
    (~150 kbit/s) versus ~3 MiB for the 1080p rendition, i.e. ~27x less bandwidth for identical audio.
    KIHC only fingerprints audio, so it ingests that rendition; the master stays for video preview.

FFmpeg input flags: -reconnect 1 -reconnect_streamed 1 -reconnect_on_network_error 1
-reconnect_delay_max 5 plus a 15 s I/O timeout. NOT -reconnect_at_eof: measured on this feed it treats
every segment's normal end-of-file as an error ("Will reconnect ... error=End of file"), delivering
20 s of audio in 29.6 s of wall time (slower than real time, so a live monitor falls behind).

-live_start_index (Phase 11.3 -> 12.1): -1 (newest segment) put our audio ~6 s behind the newest published
video instead of ~18 s, which is what lets the packshot grabber fetch a just-detected spot's frame (the playlist
keeps 4 x 6 s; a segment 404s ~30 s after publication). Measured cost over 14.8 h (2026-10-13/14): 10.8
disconnects/h and 43 supervisor restarts vs 1.3/h before - with no buffer, any slow playlist refresh ends the
stream ("EOF after 6.0 s"). Stepped back one segment (-2, ~12 s behind): stability first, fewer packshots.
"""
import requests

from kihc.connectors.iptv_manager import ChannelUnavailable

OFFICIAL_LIVE_PAGE = "https://elheddaftv.com/elheddaf-live/"
EMBED_PLAYER = "https://live.elheddaftv.com:8081/elheddaftv/embed.html"
MASTER_HLS = "https://live.elheddaftv.com:8081/elheddaftv/index.m3u8"
AUDIO_HLS = "https://live.elheddaftv.com:8081/elheddaftv/tracks-a1/mono.m3u8"
VIDEO_HLS = "https://live.elheddaftv.com:8081/elheddaftv/tracks-v1a1/mono.m3u8"   # 1080p (packshots only)
CHANNEL_NAME = "El-Heddaf TV"
FEED_LABEL = "Direct Web Feed"

FFMPEG_INPUT_ARGS = ["-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_on_network_error", "1",
                     "-reconnect_delay_max", "5", "-rw_timeout", "15000000", "-live_start_index", "-2"]


def _is_live_playlist(url: str, timeout: float) -> bool:
    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0 (KIHC broadcast monitor)"})
        return r.ok and r.text.lstrip().startswith("#EXTM3U")
    except requests.RequestException:
        return False


def resolve_elheddaf_stream(prefer_audio: bool = True, timeout: float = 6.0) -> tuple[str, str]:
    """-> (url, kind) with kind 'audio-only' or 'master'; raises ChannelUnavailable if neither answers."""
    order = [(AUDIO_HLS, "audio-only"), (MASTER_HLS, "master")] if prefer_audio else [(MASTER_HLS, "master")]
    for url, kind in order:
        if _is_live_playlist(url, timeout):
            return url, kind
    raise ChannelUnavailable(f"{CHANNEL_NAME}: direct web feed not reachable ({', '.join(u for u, _ in order)})")


def elheddaf_resolver(prefer_audio: bool = True):
    """Callable for LiveStreamWorker(stream_url=...): re-resolves on every (re)connect."""
    def resolve() -> str:
        return resolve_elheddaf_stream(prefer_audio)[0]
    resolve.__name__ = "elheddaf:direct-web-feed"
    return resolve


# ------------------------------------------------------------------------------------------------ Phase 21: 2 fps frames
import collections
import functools
import logging
import os
import subprocess
import sys
import threading
import time

import numpy as np

log = logging.getLogger("kihc.frames")

FRAME_W, FRAME_H = 480, 270            # 16:9 at 480p; fixed so the raw pipe has a known frame size
FRAME_FPS = 2
FRAME_VF = f"fps={FRAME_FPS},scale={FRAME_W}:{FRAME_H},format=yuv420p"
FRAME_BYTES = FRAME_W * FRAME_H * 3 // 2          # yuv420p: only the leading W*H bytes (luma) are analysed
RING_DEPTH = 4
SLOW_FRAME_MS = 500.0                              # an analysis slower than this is counted (frames behind it are dropped)
# Bandwidth guard. The only video rendition is 1080p at ~4.3 Mbit/s; on a slower link ffmpeg delivers frames slower than the
# 2 fps they stand for, so the video clock (anchor + n / 2 s) falls behind the audio clock AND the download starves the audio
# ingest. Frames more than STALE_DRIFT_S behind real time are not fed to the detector (misaligned cues are worse than none), and
# after STARVE_AFTER_S of that the video connection is dropped and retried only after an escalating pause.
STALE_DRIFT_S = 4.0
STARVE_AFTER_S = 45.0
PAUSE_STEPS_S = (300.0, 900.0, 1800.0, 3600.0)
STALL_S = 20.0                                    # no frame for this long -> kill and reconnect


# El-Heddaf's lower third (calibrated on the live feed, 2026-10-20): a full-width WHITE news-ticker strip sitting directly
# above a DARK-BLUE scores bar. Scrolling text changes inside the strip but never its brightness structure, which is what
# an edge-density test could not tell apart from programme detail (it flapped on 64 % of frames). Override the bands as
# KIHC_TICKER_BANDS="top0,top1,bot0,bot1" (fractions of frame height) for another layout.
TICKER_BANDS = tuple(float(v) for v in os.environ.get("KIHC_TICKER_BANDS", "0.86,0.895,0.915,0.965").split(","))
TICKER_MIN_TOP = 170.0                # mean luma of the white strip
TICKER_MIN_CONTRAST = 90.0            # strip minus the dark bar below it


def ticker_present(gray: np.ndarray) -> bool:
    h, w = gray.shape
    x0, x1 = int(w * 0.04), int(w * 0.94)
    t0, t1, b0, b1 = (int(h * f) for f in TICKER_BANDS)
    top = float(gray[t0:max(t1, t0 + 1), x0:x1].mean())
    bot = float(gray[b0:max(b1, b0 + 1), x0:x1].mean())
    return top >= TICKER_MIN_TOP and top - bot >= TICKER_MIN_CONTRAST


class FrameRing:
    """Bounded, non-blocking hand-over between the pipe reader and the analyser (depth 4).

    put() never waits: when the analyser is behind, the OLDEST frame is discarded at once, so the reader keeps
    draining ffmpeg's pipe (no back-pressure into the decoder, no latency drift) and nothing here can stall the audio
    ingest, which lives in a different process and thread entirely."""

    def __init__(self, depth: int = RING_DEPTH):
        self.depth = depth
        self._d = collections.deque()
        self._cv = threading.Condition()
        self.put_count = 0
        self.dropped = 0

    def put(self, item) -> None:
        with self._cv:
            if len(self._d) >= self.depth:
                self._d.popleft()
                self.dropped += 1
            self._d.append(item)
            self.put_count += 1
            self._cv.notify()

    def get(self, timeout: float = 0.5):
        with self._cv:
            if not self._d and not self._cv.wait(timeout):
                return None
            return self._d.popleft() if self._d else None

    def __len__(self):
        return len(self._d)


def resolve_video_url(timeout: float = 6.0) -> str:
    for url in (VIDEO_HLS, MASTER_HLS):
        if _is_live_playlist(url, timeout):
            return url
    raise ChannelUnavailable(f"{CHANNEL_NAME}: no video rendition reachable")


def luma_from_yuv420(buf: bytes) -> np.ndarray:
    """Y plane of one yuv420p frame, limited range (16..235) expanded to 0..255 so black reads ~0."""
    y = np.frombuffer(buf, np.uint8, count=FRAME_W * FRAME_H).reshape(FRAME_H, FRAME_W).astype(np.float32)
    return np.clip((y - 16.0) * (255.0 / 219.0), 0.0, 255.0)


class FrameIngest:
    """El-Heddaf monitor: ffmpeg -> 480p 2 fps frames -> FrameRing -> PodBoundaryDetector (luma / cut / ticker cues).

        FrameIngest("El-Heddaf TV").start() ... .stop()      status() for telemetry

    Two threads: the reader pulls whole frames off the pipe and stamps each with the capture clock (anchor + n / 2 s,
    assigned BEFORE the ring, so a dropped frame leaves a gap but never shifts later timestamps); the analyser feeds
    PodBoundaryDetector. `KIHC_VIDEO_OFFSET_S` shifts the video clock against the audio clock (both are anchored when
    their own connection delivers its first data, so they can differ by up to a second or so)."""

    def __init__(self, channel: str = CHANNEL_NAME, source=None, *, spawn=None, analyze=None, depth: int = RING_DEPTH,
                 backoff: float = 5.0, offset_s: float | None = None):
        self.channel = channel
        self._source = source or resolve_video_url
        self._spawn = spawn or self._spawn_ffmpeg
        self._analyze = analyze
        self.ring = FrameRing(depth)
        self.backoff = backoff
        self.offset_s = float(os.environ.get("KIHC_VIDEO_OFFSET_S", "0") or 0) if offset_s is None else offset_s
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._proc = None
        self.frames_read = self.frames_processed = self.restarts = 0
        self.connected = False
        self.last_frame_at: float | None = None
        self._ms = collections.deque(maxlen=600)
        self._arrivals = collections.deque(maxlen=21)        # wall times of the last 21 frames -> measured fps
        self.slow_frames = 0                                 # analyses that took longer than SLOW_FRAME_MS
        self.stale_dropped = self.starved_events = 0
        self.paused_until = 0.0
        self.pause_reason = ""
        self._pause_level = 0
        self.last_error = ""

    # -- lifecycle
    def start(self) -> "FrameIngest":
        if any(t.is_alive() for t in self._threads):
            return self
        self._stop.clear()
        self._threads = [threading.Thread(target=self._read_loop, name="kihc-frames-read", daemon=True),
                         threading.Thread(target=self._analyse_loop, name="kihc-frames-analyse", daemon=True)]
        for t in self._threads:
            t.start()
        return self

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        self._kill()
        for t in self._threads:
            t.join(timeout)

    def active(self, within: float = 10.0) -> bool:
        """True while frames are actually flowing (the condition for physical placement rules)."""
        return self.last_frame_at is not None and time.time() - self.last_frame_at <= within

    def paused(self) -> bool:
        return time.time() < self.paused_until

    def status(self) -> dict:
        ms = sorted(self._ms)
        arr = list(self._arrivals)
        fps = round((len(arr) - 1) / (arr[-1] - arr[0]), 2) if len(arr) >= 3 and arr[-1] > arr[0] and self.active() else 0.0
        return dict(channel=self.channel, connected=self.connected, active=self.active(), fps=fps,
                    slow_frames=self.slow_frames, restarts=self.restarts, stale_dropped=self.stale_dropped,
                    starved_events=self.starved_events, paused=self.paused(), pause_reason=self.pause_reason if self.paused() else "",
                    pause_remaining_s=round(max(0.0, self.paused_until - time.time()), 0),
                    frames_read=self.frames_read, frames_processed=self.frames_processed, frames_dropped=self.ring.dropped,
                    ring_depth=len(self.ring), analyse_ms_p50=round(ms[len(ms) // 2], 3) if ms else None,
                    analyse_ms_p95=round(ms[int(len(ms) * 0.95)], 3) if ms else None, last_error=self.last_error)

    # -- ffmpeg
    def _spawn_ffmpeg(self, url: str):
        from kihc.core.audio import ffmpeg_exe
        cmd = [ffmpeg_exe(), "-nostdin", "-hide_banner", "-loglevel", "error", *FFMPEG_INPUT_ARGS, "-i", url,
               "-an", "-sn", "-dn", "-threads", "2", "-vf", FRAME_VF, "-f", "rawvideo", "pipe:1"]
        flags = 0
        if sys.platform == "win32":                    # below-normal priority: audio matching always wins the CPU
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0)
        return subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                creationflags=flags, bufsize=0)

    def _kill(self) -> None:
        proc = self._proc
        if proc is not None and proc.poll() is None:
            try:
                proc.kill()
            except OSError:
                pass

    @staticmethod
    def _read_exact(stream, n: int) -> bytes:
        parts, left = [], n
        while left:
            chunk = stream.read(left)
            if not chunk:
                return b""
            parts.append(chunk)
            left -= len(chunk)
        return b"".join(parts)

    def _read_loop(self) -> None:
        while not self._stop.is_set():
            try:
                url = self._source() if callable(self._source) else self._source
                self._proc = proc = self._spawn(url)
            except Exception as exc:
                self.last_error = str(exc)
                self.connected = False
                if self._stop.wait(self.backoff):
                    return
                continue
            threading.Thread(target=self._watchdog, args=(proc,), daemon=True).start()
            n, anchor, t_first, stale_since, starved = 0, None, None, None, False
            try:
                while not self._stop.is_set():
                    buf = self._read_exact(proc.stdout, FRAME_BYTES)
                    if not buf:
                        break
                    now = time.time()
                    if anchor is None:
                        anchor, t_first, self.connected = now + self.offset_s, now, True
                    self.last_frame_at = now
                    self._arrivals.append(now)
                    self.frames_read += 1
                    behind = (now - t_first) - n / FRAME_FPS                  # >0: ffmpeg delivers slower than real time
                    if behind > STALE_DRIFT_S:
                        self.stale_dropped += 1
                        stale_since = stale_since or now
                        if now - stale_since > STARVE_AFTER_S:
                            starved = True
                            break
                    else:
                        stale_since = None
                        self.ring.put((anchor + n / FRAME_FPS, buf))
                    n += 1
            except (OSError, ValueError) as exc:
                self.last_error = str(exc)
            finally:
                self.connected = False
                self._kill()
            if self._stop.is_set():
                return
            self.restarts += 1
            if starved:
                step = PAUSE_STEPS_S[min(self._pause_level, len(PAUSE_STEPS_S) - 1)]
                self._pause_level += 1
                self.starved_events += 1
                self.pause_reason = (f"link too slow for the {FRAME_FPS} fps video rendition (video clock {behind:.0f} s behind "
                                     f"real time); video paused {step:.0f} s to protect the audio ingest")
                self.paused_until = time.time() + step
                log.warning("[%s] %s", self.channel, self.pause_reason)
                if self._stop.wait(step):
                    return
                continue
            if n >= 600:
                self._pause_level = 0                                          # a long healthy connection resets the ladder
            if self._stop.wait(self.backoff):
                return

    def _watchdog(self, proc) -> None:
        """Kill ffmpeg when THIS connection has delivered nothing for STALL_S. The clock starts when the connection does:
        comparing with a frame time from a previous connection killed every fresh ffmpeg within seconds (restart loop)."""
        started = time.time()
        while proc.poll() is None and not self._stop.wait(2.0):
            last = self.last_frame_at if self.last_frame_at is not None and self.last_frame_at >= started else started
            if time.time() - last > STALL_S:
                log.warning("[%s] video frames stalled for %.0f s - reconnecting", self.channel, STALL_S)
                try:
                    proc.kill()
                except OSError:
                    pass
                return

    def _analyse_loop(self) -> None:
        from kihc.analytics import pod_boundary_detector as pbd
        while not self._stop.is_set():
            item = self.ring.get(0.5)
            if item is None:
                continue
            ts, buf = item
            t0 = time.perf_counter()
            try:
                analyze = self._analyze or functools.partial(pbd.get_detector(self.channel).analyze_frame,
                                                             ticker_fn=ticker_present)
                analyze(ts, luma_from_yuv420(buf))             # default analyser: ticker_present layout test
            except Exception:
                log.exception("[%s] frame analysis failed", self.channel)
            took = (time.perf_counter() - t0) * 1000.0
            self._ms.append(took)
            if took > SLOW_FRAME_MS:                         # the ring already drops the oldest; this makes it visible
                self.slow_frames += 1
            self.frames_processed += 1
