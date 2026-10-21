"""Phase 20: multi-cue commercial-break boundary sensing and ground-truth placement guards.

Pod position used to be deduced only from *recognised* ads, so an untracked promo opening the break made the next
client ad look "first". This module senses the break edges themselves from three independent cues and votes:

  a) audio  - a dip of >= 12 dB below the running level, or energy < -40 dBFS, lasting >= 100 ms (50 ms frames)
  b) visual - mean luminance < 15/255 (black) or a scene-cut contrast spike, sub-sampled at 2 fps
  c) ticker - the broadcaster's lower-third ticker disappearing (break entry) / returning (break exit)

A boundary is confirmed when >= 2 distinct cues fall inside a 1.5 s window. Audio / black dips also occur *between*
ads inside a break, so an audio+visual coincidence alone is NOT trusted as a break edge when ticker coverage exists
(it is a separator); a confirmed edge needs the ticker flip plus one more cue. Where no ticker feed exists an
audio+visual vote is accepted only as ambiguous (confidence capped below the 0.85 certification bar).

Guards (skeptical by design - withholding a claim is better than a false one):
  FIRST_IN_POD  only if the ad's onset is <= 2.0 s after a confirmed PROGRAMME -> IN_POD transition and no audible
                unclassified segment sits between them;
  LAST_IN_POD   only if a confirmed IN_POD -> PROGRAMME transition follows its trailing edge within <= 2.0 s with
                nothing audible in between.
Otherwise the spot is demoted to MIDDLE_IN_POD. "Confirmed demotion" (a boundary is proven and unclassified material
sits in between) is a real finding; "ambiguous demotion" (cues too weak) only withholds certification.

Sensing needs feeds: audio comes from the live worker; frames (luma / ticker) from `feed_frame()` or the ffmpeg
`sample_frames()` helper. The stock pipeline is audio-only, so until frames are fed placements are reported
`unsensed` and keep the legacy detected-ads-only meaning (set KIHC_STRICT_PLACEMENT=1 to demote them instead).

Memory: all buffers are bounded circular deques (audio 50 ms frames x 20 min ~ 24k floats, video 2 fps x 20 min).
"""
import collections
import json
import logging
import os
import queue
import subprocess
import threading
import time
from datetime import datetime

import numpy as np

from kihc.config import LIVE_SAMPLE_RATE
from kihc.core.proof import ALGERIA_TZ

log = logging.getLogger(__name__)

AUDIO_FRAME_S = 0.05
AUDIO_ABS_DBFS = -40.0
AUDIO_DROP_DB = 12.0
AUDIO_MIN_DIP_S = 0.10
AUDIO_REF_S = 2.0
AUDIO_ACTIVE_DBFS = -45.0          # louder than this counts as audible material in a gap
VIDEO_FPS = 2.0
LUMA_BLACK = 15.0                  # of 255, after limited->full range normalisation
CUT_SPIKE = 0.35                   # normalised mean-abs frame difference
VOTE_WINDOW_S = 1.5
GUARD_S = 2.0                      # max distance between a break edge and the ad's edge
MAX_AFTER_ONSET_S = 0.5            # jitter: an edge may be stamped this much after the ad actually starts
UNCLASSIFIED_S = 0.5               # audible gap material longer than this = an unclassified segment
CERTIFY_CONFIDENCE = 0.85
NO_TICKER_CAP = 0.80
BUFFER_SECONDS = 1200.0
POST_WAIT_S = GUARD_S + 1.0        # an evaluation waits until this much audio exists after the ad
JOB_GIVE_UP_S = 30.0
CUES = ("audio_dip", "visual_dip", "ticker_absent", "ticker_return")


def strict() -> bool:
    return os.environ.get("KIHC_STRICT_PLACEMENT", "") not in ("", "0", "false")


def _epoch(value) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    dt = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    return (dt if dt.tzinfo else dt.replace(tzinfo=ALGERIA_TZ)).timestamp()


class BoundaryDetector:
    """Per-channel sensing buffers + cue extraction + voting + the PROGRAMME/POD state machine."""

    def __init__(self, channel: str, buffer_seconds: float = BUFFER_SECONDS):
        self.channel = channel
        self._lock = threading.Lock()
        self._audio = collections.deque(maxlen=int(buffer_seconds / AUDIO_FRAME_S))     # (ts, dBFS)
        self._video = collections.deque(maxlen=int(buffer_seconds * VIDEO_FPS))          # (ts, luma, cut, ticker|None)
        self._prev_frame: np.ndarray | None = None

    # ---------------------------------------------------------------- feeds
    def feed_audio(self, wall0: float, pcm: np.ndarray, sample_rate: int = LIVE_SAMPLE_RATE) -> None:
        n = int(AUDIO_FRAME_S * sample_rate)
        x = np.asarray(pcm, np.float32)
        blocks = x[: x.size - x.size % n].reshape(-1, n) if x.size >= n else np.zeros((0, n), np.float32)
        if not blocks.shape[0]:
            return
        rms = np.sqrt((blocks ** 2).mean(axis=1)) / 32768.0
        db = 20 * np.log10(np.maximum(rms, 1e-7))
        with self._lock:
            for i, d in enumerate(db):
                self._audio.append((wall0 + i * AUDIO_FRAME_S, float(d)))

    def feed_frame(self, ts: float, luma: float, cut_score: float = 0.0, ticker: bool | None = None) -> None:
        with self._lock:
            self._video.append((float(ts), float(luma), float(cut_score), ticker))

    def analyze_frame(self, ts: float, gray: np.ndarray, ticker_band: tuple[float, float] | None = (0.80, 0.97),
                      ticker_threshold: float = 0.10, ticker_fn=None) -> tuple[float, float, bool | None]:
        """From one 8-bit grayscale frame (any size): mean luma, cut contrast vs the previous frame, ticker present.

        Ticker presence = horizontal edge density inside the lower-third band (a scrolling text bar is edge-rich);
        band=None means this channel has no ticker feed (-> None). Calibrate `ticker_threshold` per broadcaster, or pass
        `ticker_fn(gray) -> bool | None` for a layout-specific test (El-Heddaf: stream/elheddaf_stream.ticker_present)
        """
        g = np.asarray(gray, np.float32)
        luma = float(g.mean())
        prev, self._prev_frame = self._prev_frame, g
        cut = float(np.abs(g - prev).mean() / 255.0) if prev is not None and prev.shape == g.shape else 0.0
        ticker = None
        if ticker_fn is not None:                           # a per-broadcaster layout test replaces the edge heuristic
            ticker = ticker_fn(g)
        elif ticker_band is not None:
            h = g.shape[0]
            band = g[int(h * ticker_band[0]):int(h * ticker_band[1])]
            ticker = bool(band.size and np.abs(np.diff(band, axis=1)).mean() / 255.0 > ticker_threshold)
        self.feed_frame(ts, luma, cut, ticker)
        return luma, cut, ticker

    def sensing_status(self, now: float | None = None) -> dict:
        """What the detector is currently fed with (telemetry): buffered frames, last frame age, last ticker reading."""
        now = time.time() if now is None else now
        with self._lock:
            v = self._video[-1] if self._video else None
            a = self._audio[-1] if self._audio else None
            n_v, n_a = len(self._video), len(self._audio)
        known = [f[3] for f in list(self._video)[-20:] if f[3] is not None] if v else []
        return dict(video_frames=n_v, audio_frames=n_a,
                    video_age_s=round(now - v[0], 1) if v else None, audio_age_s=round(now - a[0], 1) if a else None,
                    ticker=v[3] if v else None, ticker_coverage=bool(known),
                    sensed=bool(v and a and now - v[0] < 15.0 and now - a[0] < 15.0))

    # ---------------------------------------------------------------- cue extraction
    def _window(self, t0, t1):
        with self._lock:
            a = [(t, d) for t, d in self._audio if t0 <= t <= t1]
            v = [f for f in self._video if t0 <= f[0] <= t1]
        return a, v

    @staticmethod
    def audio_dips(frames: list) -> list[float]:
        """Start times of dips: >= 100 ms below -40 dBFS, or >= 12 dB under the median of the previous 2 s."""
        out, run_start, run_len = [], None, 0
        ts = np.array([t for t, _ in frames])
        db = np.array([d for _, d in frames])
        for i in range(len(db)):
            lo = np.searchsorted(ts, ts[i] - AUDIO_REF_S)
            ref = float(np.median(db[lo:i])) if i - lo >= 4 else None
            dip = db[i] < AUDIO_ABS_DBFS or (ref is not None and ref > AUDIO_ABS_DBFS and db[i] <= ref - AUDIO_DROP_DB)
            if dip:
                run_start = ts[i] if run_start is None else run_start
                run_len += 1
            else:
                if run_start is not None and run_len * AUDIO_FRAME_S >= AUDIO_MIN_DIP_S - 1e-9:
                    out.append(float(run_start))
                run_start, run_len = None, 0
        if run_start is not None and run_len * AUDIO_FRAME_S >= AUDIO_MIN_DIP_S - 1e-9:
            out.append(float(run_start))
        return out

    @staticmethod
    def visual_dips(frames: list) -> list[float]:
        return [f[0] for f in frames if f[1] < LUMA_BLACK or f[2] >= CUT_SPIKE]

    @staticmethod
    def ticker_flips(frames: list) -> tuple[list[float], list[float], bool]:
        """(absent-onsets, return-onsets, coverage). A flip needs 2 consecutive frames (0.5 s) in the new state."""
        known = [(t, k) for t, _, _, k in frames if k is not None]
        if len(known) < 3:
            return [], [], False
        absent, back, state = [], [], known[0][1]
        for i in range(1, len(known) - 1):
            t, k = known[i]
            if k != state and known[i + 1][1] == k:
                (back if k else absent).append(t)
                state = k
        return absent, back, True

    # ---------------------------------------------------------------- voting
    def boundaries(self, t0: float, t1: float) -> dict:
        """Cue events and confirmed votes inside [t0, t1]."""
        a, v = self._window(t0 - AUDIO_REF_S, t1)
        a_ev, v_ev = [t for t in self.audio_dips(a) if t >= t0], [t for t in self.visual_dips(v) if t >= t0]
        t_abs, t_ret, ticker_cov = self.ticker_flips([f for f in v if f[0] >= t0])
        cov = dict(audio=bool(a), video=bool(v), ticker=ticker_cov)
        votes = []
        for kind, flips, name in (("ENTRY", t_abs, "ticker_absent"), ("EXIT", t_ret, "ticker_return")):
            for tf in flips:
                cues = {name: tf}
                for nm, evs in (("audio_dip", a_ev), ("visual_dip", v_ev)):
                    near = [e for e in evs if abs(e - tf) <= VOTE_WINDOW_S]
                    if near:
                        cues[nm] = min(near, key=lambda e: abs(e - tf))
                votes.append(self._vote(kind, cues, capped=False))
        if not ticker_cov:                                  # no ticker feed: audio+visual only, direction unknown
            for ta in a_ev:
                near = [e for e in v_ev if abs(e - ta) <= VOTE_WINDOW_S]
                if near:
                    votes.append(self._vote("AMBIGUOUS", {"audio_dip": ta, "visual_dip": near[0]}, capped=True))
        return dict(votes=[x for x in votes if x], coverage=cov)

    @staticmethod
    def _vote(kind: str, cues: dict, capped: bool) -> dict | None:
        if len(cues) < 2:
            return None
        ts = sorted(cues.values())
        spread = ts[-1] - ts[0]
        if spread > VOTE_WINDOW_S:
            return None
        conf = {2: 0.90, 3: 1.0}[len(cues)] * (1 - 0.15 * spread / VOTE_WINDOW_S)
        if capped:
            conf = min(conf, NO_TICKER_CAP)
        return dict(kind=kind, ts=float(np.median(ts)), confidence=round(conf, 3), cues=sorted(cues),
                    cue_times={k: round(v, 3) for k, v in cues.items()})

    # ---------------------------------------------------------------- state machine
    def state_timeline(self, t0: float, t1: float) -> list[dict]:
        """PROGRAMME -> POD_PRE_ROLL -> IN_POD -> POD_POST_ROLL -> PROGRAMME transitions found in [t0, t1].

        PRE_ROLL / POST_ROLL = a single unconfirmed cue (ticker flip alone) was seen; they resolve to IN_POD / back
        to PROGRAMME only when a second cue confirms, or revert after 5 s if none does.
        """
        info = self.boundaries(t0, t1)
        _, v = self._window(t0, t1)
        t_abs, t_ret, _ = self.ticker_flips(v)
        confirmed = {round(x["cue_times"].get("ticker_absent", x["cue_times"].get("ticker_return", -1)), 3)
                     for x in info["votes"] if x["kind"] in ("ENTRY", "EXIT")}
        stream = []                                         # (ts, kind, payload)
        for x in info["votes"]:
            if x["kind"] in ("ENTRY", "EXIT"):
                first, last = min(x["cue_times"].values()), max(x["cue_times"].values())
                stream.append((first, x["kind"] + "_START", x))
                stream.append((last, x["kind"] + "_CONFIRM", x))
        stream += [(t, "ENTRY?", None) for t in t_abs if round(t, 3) not in confirmed]
        stream += [(t, "EXIT?", None) for t in t_ret if round(t, 3) not in confirmed]
        timeline, state, since = [], "PROGRAMME", t0
        for ts, kind, vote in sorted(stream, key=lambda p: (p[0], not p[1].endswith("_START"))):
            if state in ("POD_PRE_ROLL", "POD_POST_ROLL") and ts - since > 5.0 and not kind.endswith("_CONFIRM"):
                state = "IN_POD" if state == "POD_POST_ROLL" else "PROGRAMME"           # unconfirmed cue lapsed
            if kind in ("ENTRY_START", "ENTRY?") and state == "PROGRAMME":
                new = "POD_PRE_ROLL"
            elif kind == "ENTRY_CONFIRM" and state == "POD_PRE_ROLL":
                new = "IN_POD"
            elif kind in ("EXIT_START", "EXIT?") and state == "IN_POD":
                new = "POD_POST_ROLL"
            elif kind == "EXIT_CONFIRM" and state == "POD_POST_ROLL":
                new = "PROGRAMME"
            else:
                continue
            timeline.append(dict(ts=ts, from_state=state, to_state=new,
                                 confidence=vote["confidence"] if vote else None, cues=vote["cues"] if vote else []))
            state, since = new, ts
        return timeline

    # ---------------------------------------------------------------- context & guards
    def _active_gap(self, t0: float, t1: float) -> float:
        """Seconds of audible material (> -45 dBFS) between two instants."""
        if t1 <= t0:
            return 0.0
        a, _ = self._window(t0, t1)
        return sum(AUDIO_FRAME_S for _, d in a if d > AUDIO_ACTIVE_DBFS)

    def context(self, timestamp, edge: str = "entry", lookback: float = 12.0) -> dict:
        """Break-edge context around an instant: is it (right after) a confirmed break entry / (right before) exit?"""
        t = _epoch(timestamp)
        lo, hi = (t - lookback, t + 1.0) if edge == "entry" else (t - 1.0, t + lookback)
        info = self.boundaries(lo, hi + POST_WAIT_S)
        want = "ENTRY" if edge == "entry" else "EXIT"
        cov = info["coverage"]
        sensed = cov["audio"] and cov["video"]                      # >= 2 independent sensing sources are present
        cands = [x for x in info["votes"] if x["kind"] in (want, "AMBIGUOUS")]
        if edge == "entry":
            near = [x for x in cands if -MAX_AFTER_ONSET_S <= t - x["ts"] <= GUARD_S]
            gap = lambda x: self._active_gap(x["ts"] + 0.3, t - 0.05)                           # noqa: E731
        else:
            near = [x for x in cands if -MAX_AFTER_ONSET_S <= x["ts"] - t <= GUARD_S]
            gap = lambda x: self._active_gap(t + 0.05, x["ts"] - 0.3)                           # noqa: E731
        best = max(near, key=lambda x: x["confidence"], default=None)
        out = dict(edge=edge, sensed=sensed, coverage=cov, is_break_entry=False, boundary_confidence=0.0,
                   detected_cues=[], unclassified_material=False, boundary_ts=None, distance_s=None, too_far=False)
        if best is None:                    # a proven edge exists but further away -> something else sits in between
            far = [x for x in cands if x["kind"] == want and x["confidence"] >= CERTIFY_CONFIDENCE
                   and ((t - x["ts"] > GUARD_S) if edge == "entry" else (x["ts"] - t > GUARD_S))]
            if far:
                f = min(far, key=lambda x: abs(t - x["ts"]))
                out.update(boundary_confidence=f["confidence"], detected_cues=f["cues"], boundary_ts=f["ts"],
                           distance_s=round(abs(t - f["ts"]), 3), too_far=True,
                           unclassified_material=(self._active_gap(f["ts"] + 0.3, t - 0.05) if edge == "entry"
                                                  else self._active_gap(t + 0.05, f["ts"] - 0.3)) > UNCLASSIFIED_S)
            return out
        unclassified = gap(best) > UNCLASSIFIED_S
        out.update(boundary_confidence=best["confidence"], detected_cues=best["cues"], boundary_ts=best["ts"],
                   distance_s=round(abs(t - best["ts"]), 3), unclassified_material=unclassified,
                   is_break_entry=best["kind"] == want and best["confidence"] >= CERTIFY_CONFIDENCE)
        return out

    def evaluate_airing(self, start: float, end: float) -> dict:
        """Entry + exit context for one airing -> the record stored on ad_occurrences.boundary_json."""
        return dict(entry=self.context(start, "entry"), exit=self.context(end, "exit"), evaluated_at=time.time())

    def last_ts(self) -> float | None:
        with self._lock:
            return self._audio[-1][0] if self._audio else None

    def memory_bytes(self) -> int:
        return len(self._audio) * 56 + len(self._video) * 90          # rough deque-of-tuples footprint


# ---------------------------------------------------------------- registry, public API
_LOCK = threading.Lock()
_DETECTORS: dict[str, BoundaryDetector] = {}
_JOBS: dict[str, list] = collections.defaultdict(list)
_Q: queue.Queue = queue.Queue(maxsize=2000)
_THREAD: threading.Thread | None = None
_PENDING = 0
_DONE = threading.Condition()


def get_detector(channel: str) -> BoundaryDetector:
    with _LOCK:
        return _DETECTORS.setdefault(channel, BoundaryDetector(channel))


def reset() -> None:
    with _LOCK:
        _DETECTORS.clear()
        _JOBS.clear()


def get_pod_boundary_context(timestamp_cet, channel: str | None = None, edge: str = "entry") -> dict:
    """{is_break_entry, boundary_confidence (0.0-1.0), detected_cues, ...} for an instant on `channel`.

    `timestamp_cet`: ISO-8601 (naive = Algeria time) or epoch. With no channel the single registered detector is used.
    edge='exit' answers the mirror question (a confirmed break exit right after the instant)."""
    with _LOCK:
        det = _DETECTORS.get(channel) if channel else (next(iter(_DETECTORS.values())) if len(_DETECTORS) == 1 else None)
    if det is None:
        return dict(edge=edge, sensed=False, is_break_entry=False, boundary_confidence=0.0, detected_cues=[],
                    coverage=dict(audio=False, video=False, ticker=False), unclassified_material=False,
                    boundary_ts=None, distance_s=None)
    return det.context(timestamp_cet, edge)


def feed_audio(channel: str, wall0: float, pcm_22k: np.ndarray) -> None:
    """Live-worker hook (per 0.2 s chunk): sense audio and settle any airing whose trailing context is now available."""
    det = get_detector(channel)
    det.feed_audio(wall0, pcm_22k)
    if _JOBS.get(channel):
        _release_ready(channel, det)


def feed_frame(channel: str, ts: float, luma: float, cut_score: float = 0.0, ticker: bool | None = None) -> None:
    get_detector(channel).feed_frame(ts, luma, cut_score, ticker)


def note_airing(channel: str, occurrence_id: int, start: float, end: float, db_path) -> None:
    """Live-worker hook: queue the airing for boundary evaluation once end + POST_WAIT_S of audio exists."""
    if occurrence_id is None:
        return
    with _LOCK:
        _JOBS[channel].append((int(occurrence_id), float(start), float(end), db_path, time.time()))
    _release_ready(channel, get_detector(channel))


def _release_ready(channel: str, det: BoundaryDetector) -> None:
    last = det.last_ts()
    with _LOCK:
        pending, keep = _JOBS[channel], []
        _JOBS[channel] = keep
        ready = []
        for job in pending:
            (ready if (last is not None and last >= job[2] + POST_WAIT_S) or time.time() - job[4] > JOB_GIVE_UP_S
             else keep).append(job)
    for occ, start, end, db_path, _ in ready:
        _enqueue(det, occ, start, end, db_path)


def _enqueue(det, occ, start, end, db_path) -> None:
    global _THREAD, _PENDING
    with _DONE:
        _PENDING += 1
    try:
        _Q.put_nowait((det, occ, start, end, db_path))
    except queue.Full:
        with _DONE:
            _PENDING -= 1
        return
    with _LOCK:
        if _THREAD is None or not _THREAD.is_alive():
            _THREAD = threading.Thread(target=_worker, name="kihc-pod-boundary", daemon=True)
            _THREAD.start()


def _worker() -> None:
    global _PENDING
    while True:
        try:
            det, occ, start, end, db_path = _Q.get(timeout=30)
        except queue.Empty:
            return
        try:
            persist(db_path, occ, det.evaluate_airing(start, end))
        except Exception:
            log.exception("boundary evaluation failed for occurrence %s", occ)
        finally:
            with _DONE:
                _PENDING -= 1
                _DONE.notify_all()


def persist(db_path, occurrence_id: int, record: dict) -> None:
    from kihc.database.schema import get_db
    with get_db(db_path) as conn:
        conn.execute("UPDATE ad_occurrences SET boundary_json = ? WHERE occurrence_id = ?",
                     (json.dumps(record, default=float), int(occurrence_id)))


def flush(timeout: float = 30.0) -> bool:
    end = time.monotonic() + timeout
    with _DONE:
        while _PENDING:
            left = end - time.monotonic()
            if left <= 0:
                return False
            _DONE.wait(left)
    return True


# ---------------------------------------------------------------- ffmpeg 2 fps frame sampler (opt-in)
def sample_frames(source: str, start_epoch: float, *, width: int = 64, height: int = 36, max_seconds: float | None = None,
                  ffmpeg: str | None = None):
    """Yield (epoch, luma_0_255, gray_frame) at 2 fps from a file/stream. Limited-range video is expanded so true
    black reads ~0 (ffmpeg black is Y=16). Decoding still costs CPU on the full stream; run it only where needed."""
    from kihc.core.audio import ffmpeg_exe
    cmd = [ffmpeg or ffmpeg_exe(), "-nostdin", "-v", "error", "-i", source, "-vf",
           f"fps={VIDEO_FPS},scale={width}:{height},format=gray", "-f", "rawvideo", "-"]
    if max_seconds:
        cmd[1:1] = ["-t", str(max_seconds)]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    size, i = width * height, 0
    try:
        while True:
            buf = proc.stdout.read(size)
            if len(buf) < size:
                return
            g = np.frombuffer(buf, np.uint8).reshape(height, width).astype(np.float32)
            g = np.clip((g - 16.0) * (255.0 / 219.0), 0, 255)
            yield start_epoch + i / VIDEO_FPS, float(g.mean()), g
            i += 1
    finally:
        proc.kill()
        proc.wait()
