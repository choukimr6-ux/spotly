"""Live packshot capture: one still frame from the last seconds of a spot, located acoustically.

Measured on the El-Heddaf feed (2026-10-13): the public 1080p rendition (tracks-v1a1) lists 4 x 6 s segments
and a segment returns 404 ~30 s after publication; #EXT-X-PROGRAM-DATE-TIME runs ~16 s ahead of the local
clock. So a packshot can only be taken *live*, within ~20 s of the spot ending, and timestamps cannot locate
it: we fetch the segments the playlist currently publishes (no URL guessing, no token, no DRM), decode their
audio in memory, find the spot's tail by fingerprint, and pull exactly one frame at (end - 1.5 s).

Nothing but the final JPEG (one per asset, the same kind of still kept as certificate evidence) touches the
disk: segment bytes live in memory for the duration of one grab and FFmpeg reads/writes through pipes.
"""
import logging
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from urllib.parse import urljoin

import numpy as np
import requests

from kihc.config import HOP_SIZE, SAMPLE_RATE
from kihc.core.audio import ffmpeg_exe
from kihc.core.fingerprint import StreamingFingerprinter
from kihc.core.live_worker import LIVE_SAMPLE_RATE, decimate

log = logging.getLogger("kihc.packshot")
FPS = SAMPLE_RATE / HOP_SIZE
UA = {"User-Agent": "Mozilla/5.0 (KIHC broadcast monitor)"}
PACKSHOT_LEAD_SEC = 1.5            # frame taken this long before the spot's end
TAIL_SEC = 4.0                     # length of the spot tail used to locate it in the video segments
LOCATE_MIN_CONF = 0.75             # fingerprint agreement needed to trust the location
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0


@dataclass
class Segment:
    url: str
    duration: float
    pdt: float | None              # broadcaster's PROGRAM-DATE-TIME (informative only: clock is skewed)


def parse_playlist(text: str, base_url: str) -> list[Segment]:
    out, pdt, dur = [], None, None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("#EXT-X-PROGRAM-DATE-TIME:"):
            try:
                pdt = datetime.fromisoformat(line.split(":", 1)[1].replace("Z", "+00:00")).timestamp()
            except ValueError:
                pdt = None
        elif line.startswith("#EXTINF:"):
            dur = float(line[8:].split(",")[0])
        elif line and not line.startswith("#"):
            out.append(Segment(urljoin(base_url, line), dur or 0.0, pdt))
            pdt = pdt + dur if pdt is not None and dur else None
            dur = None
    return out


def fetch_playlist(url: str, timeout: float = 8.0, session=None) -> list[Segment]:
    r = (session or requests).get(url, headers=UA, timeout=timeout)
    r.raise_for_status()
    if "#EXT-X-KEY" in r.text:                      # encrypted: not ours to decode
        raise PermissionError("playlist is encrypted (#EXT-X-KEY); packshot capture not attempted")
    return parse_playlist(r.text, url)


def fetch_bytes(segments: list[Segment], timeout: float = 15.0, session=None) -> bytes:
    """Concatenated MPEG-TS bytes of consecutive segments (TS concatenates at the byte level)."""
    parts = []
    for s in segments:
        r = (session or requests).get(s.url, headers=UA, timeout=timeout)
        r.raise_for_status()
        parts.append(r.content)
    return b"".join(parts)


def _ffmpeg(args: list[str], data: bytes, timeout: float = 60.0) -> bytes:
    p = subprocess.run([ffmpeg_exe(), "-nostdin", "-hide_banner", "-loglevel", "error", *args], input=data,
                       capture_output=True, timeout=timeout, creationflags=_NO_WINDOW)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.decode(errors="replace").strip()[:200] or f"ffmpeg exit {p.returncode}")
    return p.stdout


def ts_fingerprint(ts: bytes) -> np.ndarray:
    """Sub-fingerprints of a TS buffer's audio, through the live worker's exact 22050 -> 11025 Hz path."""
    raw = _ffmpeg(["-i", "pipe:0", "-vn", "-sn", "-dn", "-ac", "1", "-ar", str(LIVE_SAMPLE_RATE),
                   "-f", "s16le", "pipe:1"], ts)
    x22 = np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0
    return StreamingFingerprinter().feed(decimate(x22))


def locate(tail: np.ndarray, hay: np.ndarray) -> tuple[int, float]:
    """(frame index of `tail` inside `hay`, agreement 0..1); (-1, 0.0) if hay is too short."""
    n = tail.size
    if n == 0 or hay.size < n:
        return -1, 0.0
    w = np.lib.stride_tricks.sliding_window_view(hay, n)
    conf = 1.0 - np.bitwise_count(w ^ tail).sum(axis=1) / (n * 32)
    i = int(conf.argmax())
    return i, float(conf[i])


def extract_frame(ts: bytes, at_sec: float, width: int = 1280) -> bytes:
    """Exactly one JPEG frame at `at_sec` into the TS buffer (decoded via pipes, nothing written)."""
    return _ffmpeg(["-i", "pipe:0", "-ss", f"{max(0.0, at_sec):.3f}", "-frames:v", "1", "-an",
                    "-vf", f"scale={width}:-2", "-f", "image2pipe", "-c:v", "mjpeg", "-q:v", "3", "pipe:1"], ts)


def segment_name(url: str) -> str:
    """Rendition-independent segment name: '.../tracks-a1/2026/10/13/16/02/57-06000.ts' -> '2026/.../57-06000.ts'."""
    tail = url.split("/tracks-", 1)[-1]
    return tail.split("/", 1)[1] if "/" in tail else tail


@dataclass
class Packshot:
    jpeg: bytes
    conf: float                    # acoustic agreement of the located tail (audio rendition)
    segment: str                   # published segment name the frame came from
    offset: float                  # seconds into that segment


class PackshotMiss(Exception):
    """The spot's tail could not be located / is no longer published (reason in the message)."""


def _tail(spot_fp: np.ndarray) -> np.ndarray:
    tail = spot_fp[-int(TAIL_SEC * FPS):]
    if tail.size < int(TAIL_SEC * FPS) // 2 or np.count_nonzero(tail) < 0.8 * tail.size:
        raise PackshotMiss("spot tail is silent or too short to locate")
    return tail


def locate_end_pdt(fp: np.ndarray, audio_playlist: str, *, session=None, max_segments: int = 3) -> tuple[float, float]:
    """Broadcaster time (PDT) of the end of `fp`, found in the newest audio-rendition segments -> (pdt, conf).

    Only the newest `max_segments` (~112 KB each) are fetched: on this link even 4 can take > 6 s and the
    oldest expire mid-fetch.
    """
    tail = _tail(fp)
    a_segs = fetch_playlist(audio_playlist, session=session)[-max_segments:]
    if not a_segs or a_segs[0].pdt is None:
        raise PackshotMiss("audio playlist empty or without PROGRAM-DATE-TIME")
    a_ts = fetch_bytes(a_segs, session=session)
    hay = ts_fingerprint(a_ts)
    del a_ts
    idx, conf = locate(tail, hay)
    if idx < 0 or conf < LOCATE_MIN_CONF:
        raise PackshotMiss(f"spot end not in the published window (best agreement {conf:.2f})")
    return a_segs[0].pdt + (idx + tail.size) / FPS, conf


def align_end(tail: np.ndarray, hay: np.ndarray, expected_end: int, search: int) -> tuple[int, float]:
    """Best shift of `tail`'s end around hay index `expected_end`, allowing the tail to stick out of `hay`
    (a 6 s segment may hold only part of it). -> (shift in frames, agreement over >= 1 s of overlap)."""
    n, best = tail.size, (0, 0.0)
    for shift in range(-search, search + 1):
        lo = expected_end + shift - n                          # hay index of tail[0]
        j0, j1 = max(0, -lo), min(n, hay.size - lo)
        if j1 - j0 < FPS:
            continue
        a, b = tail[j0:j1], hay[lo + j0:lo + j1]
        c = 1.0 - np.bitwise_count(a ^ b).sum() / ((j1 - j0) * 32)
        if c > best[1]:
            best = (shift, float(c))
    return best


def frame_at_pdt(spot_fp: np.ndarray, end_pdt: float, video_playlist: str, *, lead_sec: float = PACKSHOT_LEAD_SEC,
                 verify_search_sec: float = 3.0, session=None) -> tuple[Packshot, float]:
    """Download the ONE video segment holding (end_pdt - lead_sec), confirm acoustically that the spot ends
    where expected (its own audio track), and return (packshot, corrected end_pdt)."""
    tail = _tail(spot_fp)
    target = end_pdt - lead_sec
    v_segs = fetch_playlist(video_playlist, session=session)
    seg = next((s for s in v_segs if s.pdt is not None and s.pdt <= target < s.pdt + s.duration), None)
    if seg is None:
        newest = v_segs[-1].pdt + v_segs[-1].duration if v_segs and v_segs[-1].pdt else None
        state = "not yet published" if newest is not None and target >= newest else "no longer published"
        raise PackshotMiss(f"video segment for the packshot {state}")
    v_ts = fetch_bytes([seg], session=session)
    try:
        hay = ts_fingerprint(v_ts)
        shift, conf = align_end(tail, hay, int(round((end_pdt - seg.pdt) * FPS)), int(verify_search_sec * FPS))
        if conf < LOCATE_MIN_CONF:
            raise PackshotMiss(f"spot not found in the expected video segment (agreement {conf:.2f})")
        end_pdt += shift / FPS
        offset = end_pdt - lead_sec - seg.pdt
        if not 0.0 <= offset < seg.duration:
            raise PackshotMiss("packshot instant falls in a neighbouring segment")
        jpeg = extract_frame(v_ts, offset)
    finally:
        del v_ts                                                 # ~3 MB: drop immediately
    if jpeg[:2] != b"\xff\xd8":
        raise PackshotMiss("no frame decoded")
    return Packshot(jpeg, round(conf, 3), segment_name(seg.url), round(offset, 2)), end_pdt


def grab_packshot(spot_fp: np.ndarray, audio_playlist: str, video_playlist: str, *,
                  lead_sec: float = PACKSHOT_LEAD_SEC, session=None) -> Packshot:
    """Uncalibrated grab: locate the spot's end in the audio rendition, then take the frame from video."""
    end_pdt, _ = locate_end_pdt(spot_fp, audio_playlist, session=session)
    return frame_at_pdt(spot_fp, end_pdt, video_playlist, lead_sec=lead_sec, session=session)[0]


class PackshotGrabber:
    """Background thread of the pilot: captures one packshot per unclassified asset, live.

    The discovery engine calls submit() when an asset has just aired (spot fingerprint + capture-clock end).
    Capture clock -> broadcaster time is a constant offset per connection (the worker timestamps samples by
    sample count from the connection anchor), calibrated acoustically against the audio rendition. With it,
    a grab costs one playlist request and one ~3 MB segment, verified on that segment's own audio; without
    it (or if verification fails: e.g. after a reconnect) the slower audio-rendition locate is used and the
    offset re-learned.
    """

    def __init__(self, db_path, audio_playlist: str, video_playlist: str, out_dir, *, max_attempts: int = 6,
                 calibrate_every: float = 1200.0, max_age: float = 45.0):
        import queue
        import threading
        self.db_path, self.audio_playlist, self.video_playlist = str(db_path), audio_playlist, video_playlist
        self.out_dir = out_dir
        self.max_attempts, self.calibrate_every, self.max_age = max_attempts, calibrate_every, max_age
        self.offset = None                              # broadcaster PDT - capture epoch (seconds)
        self.calibrated_at = 0.0
        self._recent = None                             # () -> (fp, end_epoch) | None
        self._q = queue.Queue(maxsize=8)
        self._stop = threading.Event()
        self._session = requests.Session()             # keep-alive: TLS handshakes cost seconds on this link
        self.stats = dict(submitted=0, captured=0, missed=0, calibrations=0, last_error=None, last_capture=None)
        self._thread = threading.Thread(target=self._loop, name="kihc-packshot", daemon=True)

    # -------------------------------------------------------- wiring
    def attach(self, engine):
        engine.on_airing = self.submit
        engine.packshot_stats = self.public_stats      # surfaced in the worker's telemetry
        self._recent = engine.recent_fingerprint
        return self

    def public_stats(self) -> dict:
        return dict(self.stats, clock_offset_s=None if self.offset is None else round(self.offset, 2))

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        self._thread.join(15)

    def submit(self, asset_id: str, spot_fp: np.ndarray, end_epoch: float):
        """Called by the engine under its lock: enqueue only."""
        try:
            self._q.put_nowait((asset_id, spot_fp.copy(), float(end_epoch)))
            self.stats["submitted"] += 1
        except Exception:
            pass                                        # queue full: a later airing will try again

    # -------------------------------------------------------- work
    def _needs(self, asset_id: str) -> bool:
        from kihc.database.schema import get_db
        with get_db(self.db_path) as conn:
            r = conn.execute("SELECT status, packshot_path, packshot_attempts, content_hint, suggested_brand, "
                             "suggestion_source FROM discovered_assets WHERE asset_id = ?", (asset_id,)).fetchone()
        return bool(r and r["status"] == "unclassified" and not r["packshot_path"] and not r["content_hint"]
                    and r["packshot_attempts"] < self.max_attempts
                    and not (r["suggested_brand"] and r["suggestion_source"] in (None, "audio")))

    def _any_needed(self) -> bool:
        from kihc.database.schema import get_db
        with get_db(self.db_path) as conn:
            return conn.execute("SELECT 1 FROM discovered_assets WHERE status = 'unclassified' AND packshot_path IS NULL "
                                "AND content_hint IS NULL AND suggested_brand IS NULL AND packshot_attempts < ? LIMIT 1",
                                (self.max_attempts,)).fetchone() is not None

    def calibrate(self) -> bool:
        recent = self._recent() if self._recent else None
        if recent is None:
            return False
        fp, end_epoch = recent
        try:
            pdt, conf = locate_end_pdt(fp, self.audio_playlist, session=self._session)
        except Exception as exc:
            self.stats["last_error"] = f"calibration: {exc}"
            log.info("packshot clock calibration failed (retry in 2 min): %s", exc)
            return False
        self.offset, self.calibrated_at = pdt - end_epoch, time.time()
        self.stats["calibrations"] += 1
        log.info("packshot clock calibrated: broadcaster time = capture time %+.2f s (agreement %.2f)", self.offset, conf)
        return True

    def grab(self, spot_fp: np.ndarray, end_epoch: float) -> Packshot:
        if self.offset is not None:
            try:
                ps, end_pdt = frame_at_pdt(spot_fp, end_epoch + self.offset, self.video_playlist, session=self._session)
                self.offset = end_pdt - end_epoch                # refined by the verified position
                return ps
            except PackshotMiss as exc:
                if "no longer published" in str(exc):
                    raise
                self.offset = None                              # stale (e.g. reconnect): fall back + re-learn
        end_pdt, _ = locate_end_pdt(spot_fp, self.audio_playlist, session=self._session)
        ps, end_pdt = frame_at_pdt(spot_fp, end_pdt, self.video_playlist, session=self._session)
        self.offset, self.calibrated_at = end_pdt - end_epoch, time.time()
        return ps

    def _save(self, asset_id: str, ps: Packshot | None, error: str | None):
        from pathlib import Path

        from kihc.database.schema import get_db
        with get_db(self.db_path) as conn:
            if ps is None:
                conn.execute("UPDATE discovered_assets SET packshot_attempts = packshot_attempts + 1, "
                             "packshot_last_error = ? WHERE asset_id = ?", (error, asset_id))
                return
            out = Path(self.out_dir)
            out.mkdir(parents=True, exist_ok=True)
            path = out / f"{asset_id}_packshot.jpg"
            path.write_bytes(ps.jpeg)
            conn.execute("UPDATE discovered_assets SET packshot_path = ?, packshot_at = ?, packshot_conf = ?, "
                         "packshot_attempts = packshot_attempts + 1, packshot_last_error = NULL, visual_status = 'captured' "
                         "WHERE asset_id = ?",
                         (str(path), datetime.now().astimezone().isoformat(timespec="seconds"), ps.conf, asset_id))

    def _loop(self):
        import queue
        while not self._stop.is_set():
            try:
                job = self._q.get(timeout=5)
            except queue.Empty:
                try:
                    if time.time() - self.calibrated_at > self.calibrate_every and self._any_needed():
                        self.calibrate()
                        self.calibrated_at = max(self.calibrated_at, time.time() - self.calibrate_every + 120)
                except Exception as exc:                       # never take the pilot down
                    log.warning("packshot calibration: %s", exc)
                continue
            asset_id, spot_fp, end_epoch = job
            try:
                if not self._needs(asset_id):
                    continue
                age = time.time() - end_epoch
                if age > self.max_age:
                    raise PackshotMiss(f"detected {age:.0f} s after the spot ended; segment already expired")
                ps = self.grab(spot_fp, end_epoch)
                self._save(asset_id, ps, None)
                self.stats["captured"] += 1
                self.stats["last_capture"] = asset_id
                log.info("packshot captured for %s (segment %s +%.1f s, agreement %.2f, %d KB)", asset_id,
                         ps.segment, ps.offset, ps.conf, len(ps.jpeg) // 1024)
            except Exception as exc:
                self.stats["missed"] += 1
                self.stats["last_error"] = f"{asset_id}: {exc}"
                log.info("packshot missed for %s: %s", asset_id, exc)
                try:
                    self._save(asset_id, None, str(exc)[:200])
                except Exception:
                    pass
            finally:
                del job, spot_fp
