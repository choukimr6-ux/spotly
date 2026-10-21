"""Resilient live-stream monitor.

FFmpeg decodes the stream URL to 16-bit mono 22050 Hz PCM on stdout; nothing
from the broadcast is written to disk except the evidence spectrogram of a
confirmed airing. PCM lands in a 30 s ring buffer (collections.deque); every
LIVE_MATCH_INTERVAL seconds the buffer is fingerprinted and matched with the
unchanged core matcher (same BER, non-max suppression and start/end refinement
as scan_source).

Threads per worker:
  ingest  - connect, read PCM, reconnect with exponential backoff (2, 4, 8, ... 30 s)
  match   - periodic match pass + stall watchdog
Neither thread lets an exception escape; errors land in status()["last_error"].

Timestamps: airing_start/airing_end of live occurrences are Unix epoch seconds
(UTC) of *capture* time. HLS delivery lags the broadcast by roughly one to three
segment lengths, so capture time is later than the true on-air time by that much.
source_name is "live:<channel>"; detected_at is the UTC ISO time of confirmation.
"""
import collections
import logging
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np

from kihc.config import (EVIDENCE_DIR, FFMPEG_LOW_MEM_ARGS, HOP_SIZE, LIVE_BACKOFF_BASE,
                         LIVE_BACKOFF_MAX, LIVE_BUFFER_SECONDS,
                         LIVE_HEALTHY_AFTER, LIVE_MATCH_INTERVAL,
                         LIVE_MATCH_THRESHOLD, LIVE_SAMPLE_RATE,
                         LIVE_STALL_TIMEOUT, SAMPLE_RATE)
from kihc.analytics import evidence_recorder, pod_boundary_detector
from kihc.stream import clock_sync, time_synchronizer
from kihc.core.audio import ffmpeg_exe
from kihc.core.fingerprint import StreamingFingerprinter, fps_to_seconds
from kihc.core.matcher import _load_refs
from kihc.core.normalize import SILENCE_DBFS, normalize_rms, rms_dbfs
from kihc.core.proof import algeria_iso, compute_verification_hash
from kihc.database.schema import init_db

log = logging.getLogger("kihc.live")

DECIM = LIVE_SAMPLE_RATE // SAMPLE_RATE
assert DECIM * SAMPLE_RATE == LIVE_SAMPLE_RATE, "live rate must be an integer multiple of the fingerprint rate"
CHUNK_SAMPLES = 4410                       # 0.2 s at 22050 Hz (even -> decimation stays phase aligned)
FP_FEED_SAMPLES = 5 * SAMPLE_RATE          # fingerprint in 5 s pieces to bound transient RAM


def _lowpass(n_taps=101, cutoff=0.45 / DECIM):
    k = np.arange(n_taps) - n_taps // 2
    h = 2 * cutoff * np.sinc(2 * cutoff * k) * np.hamming(n_taps)
    return (h / h.sum()).astype(np.float32)


_TAPS = _lowpass()
_EDGE = _TAPS.size // 2                    # samples at 22050 Hz affected by zero padding at each edge


def decimate(x22: np.ndarray) -> np.ndarray:
    return np.convolve(x22, _TAPS, mode="same")[::DECIM]


def backoff_delay(attempt: int, base: float = LIVE_BACKOFF_BASE, cap: float = LIVE_BACKOFF_MAX,
                  factor: float = 2.0) -> float:
    """attempt 0, 1, 2, ... -> 2, 4, 8, 16, 30, 30, ... seconds (defaults); base 5, factor 3 -> 5, 15, 30."""
    return min(cap, base * (factor ** attempt))


@dataclass
class _Chunk:
    g0: int            # global sample index (22050 Hz) of first sample
    wall0: float       # epoch seconds of first sample (capture clock)
    pcm: np.ndarray    # int16


@dataclass
class LiveDetection:
    occurrence_id: int | None
    ad_id: int
    brand_name: str
    campaign_name: str
    channel: str
    airing_start_utc: str
    airing_end_utc: str
    duration_detected: float
    confidence_score: float
    stream_offset_sec: float      # seconds since this connection started delivering audio
    detected_at: str
    evidence_frame_path: str | None
    algeria_timestamp: str = ""
    verification_hash: str = ""


def _iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat(timespec="milliseconds")


class LiveStreamWorker:
    def __init__(self, channel_name: str, stream_url: str | Callable[[], str], db_path=None, *,
                 headers: dict | None = None,
                 threshold: float = LIVE_MATCH_THRESHOLD,
                 buffer_seconds: float = LIVE_BUFFER_SECONDS,
                 match_interval: float = LIVE_MATCH_INTERVAL,
                 evidence_dir=EVIDENCE_DIR,
                 backoff_base: float = LIVE_BACKOFF_BASE,
                 backoff_max: float = LIVE_BACKOFF_MAX,
                 healthy_after: float = LIVE_HEALTHY_AFTER,
                 stall_timeout: float = LIVE_STALL_TIMEOUT,
                 max_connects: int | None = None,
                 backoff_factor: float = 2.0,
                 ffmpeg_input_args: list[str] | None = None):
        self.channel_name = channel_name
        self.backoff_factor = backoff_factor
        # Replaces the default http input options (-rw_timeout) when given, e.g. FFmpeg reconnect flags.
        self.ffmpeg_input_args = list(ffmpeg_input_args) if ffmpeg_input_args else None
        self.source_name = f"live:{channel_name}"
        self._url = stream_url
        self.headers = headers or {}
        self.db_path = db_path
        self.threshold = threshold
        self.buffer_seconds = buffer_seconds
        self.match_interval = match_interval
        self.evidence_dir = Path(evidence_dir)
        self.backoff_base, self.backoff_max = backoff_base, backoff_max
        self.healthy_after, self.stall_timeout = healthy_after, stall_timeout
        self.max_connects = max_connects

        self._stop = threading.Event()
        self._wake = threading.Event()
        self._cond = threading.Condition()
        self._tick_lock = threading.Lock()
        self._ring = collections.deque(maxlen=self._ring_len(buffer_seconds))
        self._g = 0
        self._needed_g = 0                # earliest global sample the matcher still needs
        self._reader_blocked = False
        self._refs = {}                   # ad_id -> _RefScanner
        self._ignored_ad_ids = set()      # reference ads another matcher owns (see core/reference_worker.py)
        self._captured = {}               # (ad_id, offset) -> (pcm int16, g_start, wall_start, conn_anchor)
        self._conn_anchor = (0, 0.0)      # (global sample, epoch) at start of current connection
        self._proc = None
        self._stderr_tail = collections.deque(maxlen=20)
        self._threads = []

        self.state = "idle"
        self.last_error = ""
        self.connect_attempts = 0
        self.backoff_history = []
        self.next_retry_at = None
        self.samples_received = 0
        self.max_buffered_seconds = 0.0
        self.last_data_at = None
        self.started_at = None
        self.detections = collections.deque(maxlen=200)
        # Radar telemetry: raw (pre-normalization) input level per 0.2 s chunk, last 60 s.
        self._levels = collections.deque(maxlen=int(60 * LIVE_SAMPLE_RATE / CHUNK_SAMPLES))
        self._disconnects = collections.deque(maxlen=500)   # epoch times of dropped connections
        self.connected_since = None

    # ------------------------------------------------------------ lifecycle
    def start(self):
        if self._threads and any(t.is_alive() for t in self._threads):
            return self
        self._stop.clear()
        self.started_at = time.time()
        self._threads = [threading.Thread(target=self._ingest_loop, name=f"kihc-ingest-{self.channel_name}", daemon=True),
                         threading.Thread(target=self._match_loop, name=f"kihc-match-{self.channel_name}", daemon=True)]
        for t in self._threads:
            t.start()
        return self

    def stop(self, timeout: float = 5.0):
        self._stop.set()
        self._wake.set()
        with self._cond:
            self._cond.notify_all()
        self._kill_proc()
        for t in self._threads:
            t.join(timeout)
        self.state = "stopped"

    def wait(self, timeout: float | None = None) -> bool:
        """Block until the worker ends on its own (max_connects reached). True if it ended."""
        for t in self._threads:
            t.join(timeout)
        return not self.is_alive()

    def is_alive(self) -> bool:
        return any(t.is_alive() for t in self._threads)

    def status(self) -> dict:
        with self._cond:
            buffered = sum(c.pcm.size for c in self._ring) / LIVE_SAMPLE_RATE
        proc = self._proc
        return dict(channel=self.channel_name, state=self.state, alive=self.is_alive(),
                    buffered_seconds=round(buffered, 1), max_buffered_seconds=round(self.max_buffered_seconds, 1),
                    received_seconds=round(self.samples_received / LIVE_SAMPLE_RATE, 1),
                    connect_attempts=self.connect_attempts, backoff_history=list(self.backoff_history),
                    next_retry_at=self.next_retry_at, last_error=self.last_error,
                    references=len(self._refs), detections=len(self.detections),
                    ffmpeg_pid=proc.pid if proc and proc.poll() is None else None,
                    seconds_since_data=None if self.last_data_at is None
                    else round(time.monotonic() - self.last_data_at, 1),
                    **self.radar())

    def radar(self) -> dict:
        """Signal telemetry: input level (dBFS, before normalization) and connection stability."""
        now = time.time()
        levels = list(self._levels)
        last10 = [l for t, l in levels if now - t <= 10.0]
        # Level of the most recent ~1 s of audio (5 chunks of 0.2 s). Keyed on audio, not arrival time:
        # HLS delivers 6 s segments in bursts, so most wall-clock seconds see no new chunk at all.
        recent = [l for _, l in levels[-5:]] if last10 else []
        level = 10 * np.log10(np.mean([10 ** (l / 10) for l in recent])) if recent else None
        return dict(
            level_dbfs=None if level is None else round(float(level), 1),
            peak_dbfs_10s=round(max(last10), 1) if last10 else None,
            silent=bool(last10) and max(last10) < SILENCE_DBFS,
            level_history=[(round(t, 2), round(l, 1)) for t, l in levels],
            connected_for_s=None if self.connected_since is None else round(now - self.connected_since, 1),
            disconnects_last_hour=sum(1 for t in self._disconnects if now - t <= 3600),
        )

    # ------------------------------------------------------------ ingest
    def _ring_len(self, seconds):
        return int(np.ceil(seconds * LIVE_SAMPLE_RATE / CHUNK_SAMPLES))

    def _resolve_url(self) -> str:
        return self._url() if callable(self._url) else self._url

    def _spawn(self, url):
        self._clock_wants_log = bool(getattr(self, "_clock_probe", None))
        # level+info tags every line so the first-segment "Opening ..." message can be read (clock calibration) while only
        # the error lines still reach the stderr tail
        cmd = [ffmpeg_exe(), "-nostdin", "-hide_banner", "-loglevel", "level+info" if self._clock_wants_log else "error"]
        if re.match(r"https?://", url):
            cmd += self.ffmpeg_input_args or [*FFMPEG_LOW_MEM_ARGS, "-rw_timeout", str(int(self.stall_timeout * 1e6))]
            if "User-Agent" in self.headers:
                cmd += ["-user_agent", self.headers["User-Agent"]]
            if "Referer" in self.headers:
                cmd += ["-referer", self.headers["Referer"]]
        cmd += ["-i", url, "-vn", "-sn", "-dn", "-ac", "1", "-ar", str(LIVE_SAMPLE_RATE), "-f", "s16le", "pipe:1"]
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0
        proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, creationflags=flags)
        threading.Thread(target=self._drain_stderr, args=(proc,), daemon=True).start()
        return proc

    def _drain_stderr(self, proc):
        probe = getattr(self, "_clock_probe", None) if getattr(self, "_clock_wants_log", False) else None
        for line in iter(proc.stderr.readline, b""):
            text = line.decode(errors="replace").rstrip()
            if probe is not None and "[info]" in text:          # clock calibration: which segment did ffmpeg start with?
                seg = clock_sync.opened_segment(text)
                if seg:
                    probe.segment_opened(seg)
                continue                                         # info chatter never reaches the error tail
            self._stderr_tail.append(text)
        proc.stderr.close()

    def _kill_proc(self):
        proc = self._proc
        if proc and proc.poll() is None:
            try:
                proc.kill()
            except OSError:
                pass

    def _ingest_loop(self):
        attempt = 0
        try:
            while not self._stop.is_set():
                if self.max_connects is not None and self.connect_attempts >= self.max_connects:
                    break
                self.state = "connecting"
                self.connect_attempts += 1
                connected_at, got = time.monotonic(), 0
                try:
                    self._stderr_tail.clear()
                    url = self._resolve_url()
                    self._clock_probe = clock_sync.probe_for(self.channel_name, url, self.db_path)
                    if self._clock_probe:
                        self._clock_probe.connecting()
                    self._proc = self._spawn(url)
                    got = self._read_until_eof(self._proc)
                    rc = self._proc.wait()
                    if not self._stop.is_set():
                        tail = " | ".join(list(self._stderr_tail)[-3:])
                        self.last_error = (f"ffmpeg exited rc={rc}: {tail}" if rc else
                                           f"stream ended (EOF) after {got / LIVE_SAMPLE_RATE:.1f}s")
                except Exception as exc:  # resolver / spawn / read failures must never kill the worker
                    self.last_error = f"{type(exc).__name__}: {exc}"
                    log.warning("[%s] connection error: %s", self.channel_name, self.last_error)
                finally:
                    self._kill_proc()
                self._on_disconnect()
                if self._stop.is_set():
                    break
                if self.max_connects is not None and self.connect_attempts >= self.max_connects:
                    break
                if got and time.monotonic() - connected_at >= self.healthy_after:
                    attempt = 0
                delay = backoff_delay(attempt, self.backoff_base, self.backoff_max, self.backoff_factor)
                attempt += 1
                self.backoff_history.append(delay)
                self.next_retry_at = time.time() + delay
                self.state = "backoff"
                log.info("[%s] disconnected (%s); retry in %.1fs", self.channel_name, self.last_error, delay)
                self._stop.wait(delay)
                self.next_retry_at = None
        finally:
            self.state = "stopped"
            self._stop.set()      # also ends the match thread when max_connects is reached
            self._wake.set()

    def _read_until_eof(self, proc) -> int:
        nbytes, got = CHUNK_SAMPLES * 2, 0
        while not self._stop.is_set():
            data = proc.stdout.read(nbytes)
            if not data:
                break
            pcm = np.frombuffer(data[: len(data) - len(data) % 2], dtype="<i2")
            if pcm.size == 0:
                continue
            if got == 0:
                self._conn_anchor = (self._g, time.time() - pcm.size / LIVE_SAMPLE_RATE)
                self.connected_since = time.time()
                if getattr(self, "_clock_probe", None):
                    self._clock_probe.first_audio(self._conn_anchor[1])      # side thread: never delays the audio
                self.state = "listening"
            self._levels.append((time.time(), rms_dbfs(pcm.astype(np.float32) / 32768.0)))
            g_anchor, wall_anchor = self._conn_anchor
            self._append(_Chunk(self._g, wall_anchor + (self._g - g_anchor) / LIVE_SAMPLE_RATE, pcm))
            self._g += pcm.size
            got += pcm.size
            self.samples_received += pcm.size
            self.last_data_at = time.monotonic()
        return got

    def _append(self, chunk):
        with self._cond:
            # Back-pressure: never let the ring evict audio the matcher has not scanned yet.
            while (not self._stop.is_set() and len(self._ring) == self._ring.maxlen
                   and self._ring[0].g0 + self._ring[0].pcm.size > self._needed_g):
                self._reader_blocked = True
                self._wake.set()
                self._cond.wait(0.5)
            self._reader_blocked = False
            self._ring.append(chunk)
            evidence_recorder.feed(self.channel_name, chunk.wall0, chunk.pcm)   # Phase 19 scratch ring
            pod_boundary_detector.feed_audio(self.channel_name, chunk.wall0, chunk.pcm)   # Phase 20 sensing
            self.max_buffered_seconds = max(self.max_buffered_seconds,
                                            sum(c.pcm.size for c in self._ring) / LIVE_SAMPLE_RATE)
            if len(self._ring) == self._ring.maxlen:
                self._wake.set()

    def _on_disconnect(self):
        """Audio across a reconnect is not contiguous: finish pending matches, then reset the ring."""
        if self.connected_since is not None:
            self._disconnects.append(time.time())
        self.connected_since = None
        try:
            self._tick(final=True)
        except Exception as exc:
            self.last_error = f"final match pass failed: {type(exc).__name__}: {exc}"
            log.exception("[%s] final match pass failed", self.channel_name)
        with self._cond:
            self._ring.clear()
            self._needed_g = self._g
            self._cond.notify_all()
        self._captured.clear()

    # ------------------------------------------------------------ matching
    def _match_loop(self):
        while not self._stop.is_set():
            self._wake.wait(self.match_interval)
            self._wake.clear()
            if self._stop.is_set():
                break
            try:
                self._watchdog()
                self._tick(final=False)
            except Exception as exc:
                self.last_error = f"match pass failed: {type(exc).__name__}: {exc}"
                log.exception("[%s] match pass failed", self.channel_name)

    def _watchdog(self):
        if (self.state == "listening" and not self._reader_blocked and self.last_data_at is not None
                and time.monotonic() - self.last_data_at > self.stall_timeout):
            self.last_error = f"no audio for {self.stall_timeout:.0f}s; forcing reconnect"
            self._kill_proc()

    def _refresh_refs(self):
        conn = init_db(self.db_path)
        try:
            ids = {r[0] for r in conn.execute("SELECT ad_id FROM reference_ads")} - self._ignored_ad_ids
            for gone in set(self._refs) - ids:
                del self._refs[gone]
            new = ids - set(self._refs)
            if new:
                for ref in _load_refs(conn, self.threshold, ad_ids=new):
                    self._refs[ref.ad_id] = ref
                longest = max(r.duration for r in self._refs.values()) if self._refs else 0
                # The ring must hold one full ad plus a match interval, or matching could stall.
                needed = max(self.buffer_seconds, longest + self.match_interval + 2.0)
                with self._cond:
                    if self._ring_len(needed) != self._ring.maxlen:
                        self._ring = collections.deque(self._ring, maxlen=self._ring_len(needed))
        finally:
            conn.close()

    def _tick(self, final: bool):
        with self._tick_lock:
            self._refresh_refs()
            with self._cond:
                chunks = list(self._ring)
            if (not self._refs and not self.wants_fingerprints) or not chunks:
                with self._cond:
                    if not self._refs and not self.wants_fingerprints:   # nothing consumes audio: evict freely
                        self._needed_g = float("inf")
                    self._cond.notify_all()
                return

            g0 = chunks[0].g0
            x22 = normalize_rms(np.concatenate([c.pcm for c in chunks]).astype(np.float32) / 32768.0)
            x11 = decimate(x22)
            if not final:   # tail samples see zero padding; they are fingerprinted next pass instead
                x11 = x11[: max(0, x11.size - _EDGE // DECIM - 1)]
            s0 = g0 // DECIM
            s_aligned = -(-s0 // HOP_SIZE) * HOP_SIZE      # keep frame grid identical across passes
            x11 = x11[s_aligned - s0:]
            fper = StreamingFingerprinter()
            parts = [fper.feed(x11[i:i + FP_FEED_SAMPLES]) for i in range(0, x11.size, FP_FEED_SAMPLES)]
            fp = np.concatenate(parts) if parts else np.zeros(0, np.uint32)
            base = s_aligned // HOP_SIZE
            try:   # extension point (e.g. unsupervised discovery); must never break matching
                self._on_fingerprints(fp, base, x11, s_aligned, chunks)
            except Exception as exc:
                self.last_error = f"fingerprint hook failed: {type(exc).__name__}: {exc}"
                log.warning("[%s] fingerprint hook failed: %s", self.channel_name, exc)

            for ref in self._refs.values():
                ref.next_offset = max(ref.next_offset, base)
                ref.scan(fp, base)
                for off, conf, errs in list(ref.found) + ([ref.pending] if ref.pending else []):
                    self._capture(ref, off, x22, chunks)
                # Confirm once a full ad length has passed with no better alignment (same NMS as file scans).
                if ref.pending and (final or ref.next_offset - ref.pending[0] >= ref.L):
                    ref.flush()
                for off, conf, errs in ref.found:
                    self._confirm(ref, off, conf, errs)
                ref.found.clear()

            keep = {(r.ad_id, r.pending[0]) for r in self._refs.values() if r.pending}
            self._captured = {k: v for k, v in self._captured.items() if k in keep}
            # Earliest audio still needed: by the matcher (oldest untested window) and/or by the hook
            # (it has consumed everything fingerprinted so far, keeping the frame overlap).
            next_frame = min([r.next_offset for r in self._refs.values()] + [base + fp.size - 1])
            needed = next_frame * HOP_SIZE * DECIM - 2 * _EDGE
            with self._cond:
                self._needed_g = needed
                self._cond.notify_all()

    # ------------------------------------------------------------ extension point
    wants_fingerprints = False    # subclasses that consume audio without reference ads set this True

    def _on_fingerprints(self, fp, base, x11, s_aligned, chunks):
        """Called every analysis pass with the snapshot's sub-fingerprints.

        fp[i] is global frame index base + i (an airing starting at frame j starts at sample j*HOP_SIZE
        of the 11025 Hz grid); x11 is the snapshot's normalized 11025 Hz audio starting at global sample
        s_aligned; chunks carry capture-clock times. Default: no-op.
        """

    def _capture(self, ref, offset, x22, chunks):
        key = (ref.ad_id, offset)
        if key in self._captured:
            return
        g = offset * HOP_SIZE * DECIM
        i0 = max(0, g - chunks[0].g0)
        pcm = (np.clip(x22[i0:i0 + int(ref.duration * LIVE_SAMPLE_RATE)], -1, 1) * 32767).astype(np.int16)
        wall = next((c.wall0 + (g - c.g0) / LIVE_SAMPLE_RATE for c in chunks if c.g0 <= g < c.g0 + c.pcm.size),
                    chunks[0].wall0)
        self._captured[key] = (pcm, g, wall, self._conn_anchor)

    def _confirm(self, ref, offset, conf, errs):
        pcm, g_start, wall_start, (g_anchor, _) = self._captured.get(
            (ref.ad_id, offset), (np.zeros(0, np.int16), offset * HOP_SIZE * DECIM, time.time(), self._conn_anchor))
        det = ref.detection(offset, conf, errs, self.source_name)
        win_start = fps_to_seconds(offset)
        rel_s, rel_e = det.airing_start - win_start, det.airing_end - win_start
        start_epoch, end_epoch = wall_start + rel_s, wall_start + rel_e
        detected_at = datetime.now(timezone.utc)
        # det times are rounded to ms, so rel_s can be a hair below 0: clamp (a negative index would wrap).
        i_s = max(0, int(round(rel_s * LIVE_SAMPLE_RATE)))
        i_e = max(i_s, int(round(rel_e * LIVE_SAMPLE_RATE)))
        evidence, evidence_bytes = self._spectrogram(pcm[i_s:i_e], ref.ad_id, start_epoch)
        start_epoch, end_epoch = round(start_epoch, 3), round(end_epoch, 3)
        ts_cet = algeria_iso(start_epoch)
        vhash = compute_verification_hash(ref.ad_id, self.source_name, ts_cet, end_epoch,
                                          det.duration_detected, det.confidence_score, evidence_bytes)
        conn = init_db(self.db_path)
        try:
            with conn:
                cur = conn.execute(
                    """INSERT OR IGNORE INTO ad_occurrences
                       (ad_id, source_name, airing_start, airing_end, duration_detected,
                        confidence_score, evidence_frame_path, detected_at,
                        verification_hash, algeria_timestamp,
                        clock_drift_seconds, clock_drift_margin_s, clock_drift_basis,
                        broadcast_start_cet, broadcast_end_cet, mcr_drift_delta_seconds)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (ref.ad_id, self.source_name, start_epoch, end_epoch,
                     det.duration_detected, det.confidence_score, evidence,
                     detected_at.isoformat(timespec="seconds"), vhash, ts_cet,
                     *clock_sync.occurrence_fields(self.channel_name).values(),
                     *time_synchronizer.occurrence_fields(self.channel_name, start_epoch, end_epoch).values()),
                )
                occ_id = cur.lastrowid if cur.rowcount else None
        finally:
            conn.close()
        if occ_id is not None:
            evidence_recorder.note_airing(self.channel_name, occ_id, start_epoch, end_epoch)
            pod_boundary_detector.note_airing(self.channel_name, occ_id, start_epoch, end_epoch, self.db_path)
        rec = LiveDetection(occ_id, ref.ad_id, ref.brand, ref.campaign, self.channel_name,
                            _iso(start_epoch), _iso(end_epoch), det.duration_detected, det.confidence_score,
                            round((g_start - g_anchor) / LIVE_SAMPLE_RATE + rel_s, 3),
                            detected_at.isoformat(timespec="seconds"), evidence, ts_cet, vhash)
        self.detections.append(rec)
        log.info("[%s] DETECTED %s / %s at %s conf=%.4f", self.channel_name, ref.brand, ref.campaign,
                 rec.airing_start_utc, det.confidence_score)
        # Advance past this airing: its audio is no longer needed in the ring.
        g_end = g_start + int(rel_e * LIVE_SAMPLE_RATE)
        with self._cond:
            while self._ring and self._ring[0].g0 + self._ring[0].pcm.size <= g_end:
                self._ring.popleft()
            self._cond.notify_all()

    def _spectrogram(self, pcm: np.ndarray, ad_id: int, start_epoch: float):
        """Evidence = spectrogram of the matched airing's own audio, rendered from memory.

        Returns (path, png bytes); the bytes hashed are exactly the bytes written.
        """
        if pcm.size < LIVE_SAMPLE_RATE // 2:
            return None, None
        cmd = [ffmpeg_exe(), "-nostdin", "-v", "error", "-f", "s16le", "-ar", str(LIVE_SAMPLE_RATE),
               "-ac", "1", "-i", "pipe:0", "-lavfi", "showspectrumpic=s=640x240:legend=0",
               "-f", "image2pipe", "-c:v", "png", "pipe:1"]
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0
        proc = subprocess.run(cmd, input=pcm.tobytes(), capture_output=True, creationflags=flags)
        if proc.returncode != 0 or not proc.stdout:
            return None, None
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.fromtimestamp(start_epoch, timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", self.channel_name)
        out = self.evidence_dir / f"live_{safe}_ad{ad_id}_{stamp}.png"
        out.write_bytes(proc.stdout)
        return str(out), proc.stdout
