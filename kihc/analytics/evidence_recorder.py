"""Phase 19: autonomous dispute-evidence recorder - tight, SHA-256-sealed audio clips for contract / pod breaches.

No continuous recording. Per channel the live worker feeds a small circular in-memory PCM ring (RING_SECONDS).
When a detection is confirmed, a *candidate* for that airing is cut from the ring as soon as the post-roll is
available (airing start - PAD_SECONDS .. airing end + PAD_SECONDS) and kept in a bounded in-memory stash. Only
when a breach is flagged is the candidate written to disk (`<EVIDENCE_DIR>/breaches/*.wav`), hashed, and bound
to the record: `ad_occurrences.evidence_path / evidence_sha256` plus a hash-chained `contract_audit_log` entry.
Unflagged airings never touch the disk and the stash expires on its own.

The ingest path is audio-only (FFmpeg -> mono PCM), so the clip is audio (.wav); a video clip would need a
second video capture the pipeline does not have. The seal makes later edits detectable; it is not a qualified
signature (anyone who can rewrite both the file and the database can re-seal).
"""
import collections
import hashlib
import logging
import queue
import re
import threading
import time
import wave
from datetime import datetime
from pathlib import Path

import numpy as np

from kihc.analytics import contracts as ctr
from kihc.config import EVIDENCE_DIR, LIVE_SAMPLE_RATE
from kihc.core.proof import ALGERIA_TZ
from kihc.database.schema import get_db

log = logging.getLogger(__name__)

PAD_SECONDS = 3.0                 # pre-roll and post-roll
CLIP_RATE = LIVE_SAMPLE_RATE // 2   # 11 025 Hz mono 16-bit (~22 kB/s): plenty for speech/jingle evidence
RING_SECONDS = 60.0               # circular scratch buffer per channel (~1.3 MB at CLIP_RATE)
STASH_MAX = 48                    # candidate clips held in memory across channels (~36 s each, ~0.8 MB)
STASH_TTL = 3600.0
JOB_GIVE_UP = 30.0                # cut a partial candidate if the post-roll never arrives within this long
TOLERANCE = 0.25                  # seconds of missing audio tolerated at either edge of a clip
BREACH_DIR = "breaches"


class EvidenceUnavailable(RuntimeError):
    pass


def _to_epoch(value) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    dt = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    return (dt if dt.tzinfo else dt.replace(tzinfo=ALGERIA_TZ)).timestamp()


def _halve(pcm: np.ndarray) -> np.ndarray:
    """22 050 -> 11 025 Hz by averaging sample pairs (a 2-tap low-pass; speech/jingle band stays intact)."""
    n = pcm.size - pcm.size % 2
    return ((pcm[:n:2].astype(np.int32) + pcm[1:n:2]) // 2).astype(np.int16)


class ChannelRing:
    """Circular PCM buffer: contiguous-or-gapped chunks keyed by capture wall clock, bounded in seconds."""

    def __init__(self, seconds: float = RING_SECONDS):
        self.seconds = seconds
        self._chunks: collections.deque = collections.deque()      # (wall0, int16 @ CLIP_RATE)
        self._lock = threading.Lock()

    def feed(self, wall0: float, pcm_22k: np.ndarray) -> None:
        pcm = _halve(np.asarray(pcm_22k))
        with self._lock:
            self._chunks.append((float(wall0), pcm))
            end = wall0 + pcm.size / CLIP_RATE
            while self._chunks and self._chunks[0][0] + self._chunks[0][1].size / CLIP_RATE < end - self.seconds:
                self._chunks.popleft()

    def end(self) -> float | None:
        with self._lock:
            return self._chunks[-1][0] + self._chunks[-1][1].size / CLIP_RATE if self._chunks else None

    def slice(self, t0: float, t1: float) -> tuple[np.ndarray, float]:
        """Samples for [t0, t1) at CLIP_RATE (silence where the ring has a gap) and the seconds actually covered."""
        n = int(round((t1 - t0) * CLIP_RATE))
        out = np.zeros(n, np.int16)
        covered = 0
        with self._lock:
            for wall0, pcm in self._chunks:
                a = int(round((wall0 - t0) * CLIP_RATE))
                lo, hi = max(a, 0), min(a + pcm.size, n)
                if hi > lo:
                    out[lo:hi] = pcm[lo - a:hi - a]
                    covered += hi - lo
        return out, covered / CLIP_RATE


class _Candidate:
    __slots__ = ("pcm", "t0", "t1", "covered", "stashed_at")

    def __init__(self, pcm, t0, t1, covered):
        self.pcm, self.t0, self.t1, self.covered, self.stashed_at = pcm, t0, t1, covered, time.time()


_LOCK = threading.Lock()
_RINGS: dict[str, ChannelRing] = {}
_JOBS: dict[str, list] = collections.defaultdict(list)             # channel -> [(occ_id, start, end, queued_at)]
_STASH: "collections.OrderedDict[int, _Candidate]" = collections.OrderedDict()


def reset() -> None:
    with _LOCK:
        _RINGS.clear()
        _JOBS.clear()
        _STASH.clear()


def feed(channel: str, wall0: float, pcm_22k: np.ndarray) -> None:
    """Live worker hook: append capture audio to the channel ring and cut any candidate whose post-roll is in."""
    with _LOCK:
        ring = _RINGS.setdefault(channel, ChannelRing())
    ring.feed(wall0, pcm_22k)
    jobs = _JOBS.get(channel)
    if jobs:
        _cut_ready(channel, ring)


def _cut_ready(channel: str, ring: ChannelRing, now: float | None = None) -> None:
    now = time.time() if now is None else now
    end = ring.end()
    with _LOCK:
        pending, _JOBS[channel] = _JOBS[channel], []
    keep = []
    for occ, start, stop, queued in pending:
        if end is not None and end >= stop + PAD_SECONDS or now - queued > JOB_GIVE_UP:
            pcm, covered = ring.slice(start - PAD_SECONDS, stop + PAD_SECONDS)
            with _LOCK:
                _STASH[occ] = _Candidate(pcm, start - PAD_SECONDS, stop + PAD_SECONDS, covered)
                while len(_STASH) > STASH_MAX:
                    _STASH.popitem(last=False)
        else:
            keep.append((occ, start, stop, queued))
    with _LOCK:
        _JOBS[channel] = keep + _JOBS[channel]


def note_airing(channel: str, occurrence_id: int, start: float, end: float) -> None:
    """Live worker hook: a detection was confirmed; remember to cut its contextual snippet once the post-roll lands."""
    if occurrence_id is None:
        return
    with _LOCK:
        _JOBS[channel].append((int(occurrence_id), float(start), float(end), time.time()))
        ring = _RINGS.get(channel)
    if ring is not None:
        _cut_ready(channel, ring)


def _expire(now: float) -> None:
    with _LOCK:
        for k in [k for k, c in _STASH.items() if now - c.stashed_at > STASH_TTL]:
            del _STASH[k]


def write_wav(path: Path, pcm: np.ndarray) -> None:
    """Deterministic 16-bit mono WAV: identical samples always give byte-identical files (stable SHA-256)."""
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(CLIP_RATE)
        w.writeframes(np.asarray(pcm, "<i2").tobytes())


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def capture_breach_evidence(channel_id: str, start_time_cet, end_time_cet, breach_type: str, *,
                            occurrence_id: int | None = None, evidence_dir=None, ring: ChannelRing | None = None) -> dict:
    """Cut [start - 3.0 s, end + 3.0 s], write it under <evidence>/breaches/, hash it.

    Source order: the candidate already cut for `occurrence_id`, else the channel's live ring.
    -> dict(path=<relative to the evidence dir>, sha256, start_epoch, end_epoch, clip_start_epoch, clip_seconds,
            sample_rate, covered_seconds, complete)
    Raises EvidenceUnavailable when the audio is no longer (or was never) in memory.
    """
    start, end = _to_epoch(start_time_cet), _to_epoch(end_time_cet)
    if end <= start:
        raise ValueError("end_time_cet must be after start_time_cet")
    _expire(time.time())
    t0, t1 = start - PAD_SECONDS, end + PAD_SECONDS
    with _LOCK:
        cand = _STASH.get(int(occurrence_id)) if occurrence_id is not None else None
        ring = ring or _RINGS.get(channel_id)
    if cand is not None:
        pcm, covered, t0 = cand.pcm, cand.covered, cand.t0
    elif ring is not None:
        pcm, covered = ring.slice(t0, t1)
    else:
        raise EvidenceUnavailable(f"no buffered audio for {channel_id}: the airing left the scratch buffer")
    total = t1 - t0
    if covered < total - 2 * TOLERANCE and covered < (end - start):
        raise EvidenceUnavailable(f"only {covered:.1f} s of {total:.1f} s available for {channel_id}")
    root = Path(evidence_dir or EVIDENCE_DIR)
    folder = root / BREACH_DIR
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.fromtimestamp(start, ALGERIA_TZ).strftime("%Y%m%dT%H%M%S")
    slug = re.sub(r"[^A-Za-z0-9]+", "_", f"{channel_id}")[:40].strip("_")
    kind = re.sub(r"[^a-z0-9]+", "_", breach_type.lower()).strip("_")[:20]
    name = f"{slug}_{occurrence_id if occurrence_id is not None else 'x'}_{kind}_{stamp}.wav"
    target = folder / name
    write_wav(target, pcm)
    return dict(path=f"{BREACH_DIR}/{name}", sha256=sha256_file(target), start_epoch=start, end_epoch=end,
                clip_start_epoch=t0, clip_seconds=round(len(pcm) / CLIP_RATE, 3), sample_rate=CLIP_RATE,
                covered_seconds=round(covered, 3), complete=covered >= total - 2 * TOLERANCE)


# ------------------------------------------------------------------------------------------ persistence
def bind_evidence(db_path, occurrence_id: int, ev: dict, breach_type: str, contract_id: str | None = None) -> None:
    """Store path + SHA-256 on the occurrence and append a hash-chained audit entry (WAL, one transaction)."""
    with get_db(db_path) as conn:
        conn.execute("UPDATE ad_occurrences SET evidence_path = ?, evidence_sha256 = ? WHERE occurrence_id = ?",
                     (ev["path"], ev["sha256"], int(occurrence_id)))
        ctr._append_log(conn, f"occ:{occurrence_id}", "EVIDENCE", dict(
            occurrence_id=int(occurrence_id), breach_type=breach_type, contract_id=contract_id,
            evidence_path=ev["path"], evidence_sha256=ev["sha256"], clip_seconds=ev["clip_seconds"],
            complete=ev["complete"]), "evidence-recorder", entity="breach_evidence")


def evidence_for(db_path, occurrence_id: int) -> dict | None:
    with get_db(db_path) as conn:
        r = conn.execute("SELECT evidence_path, evidence_sha256 FROM ad_occurrences WHERE occurrence_id = ?",
                         (int(occurrence_id),)).fetchone()
    return dict(path=r["evidence_path"], sha256=r["evidence_sha256"]) if r and r["evidence_path"] else None


def resolve_path(rel: str, evidence_dir=None) -> Path:
    """Absolute clip path, refusing anything that escapes <evidence>/breaches/."""
    base = (Path(evidence_dir or EVIDENCE_DIR) / BREACH_DIR).resolve()
    p = (Path(evidence_dir or EVIDENCE_DIR) / rel).resolve()
    if base not in p.parents:
        raise ValueError("evidence path outside the breach folder")
    return p


def verify_evidence(db_path, occurrence_id: int, evidence_dir=None) -> dict:
    """Recompute the clip's SHA-256 and compare with the recorded one: status PASS | FAIL | MISSING | NONE."""
    rec = evidence_for(db_path, occurrence_id)
    if not rec:
        return dict(status="NONE", recorded=None, actual=None)
    try:
        p = resolve_path(rec["path"], evidence_dir)
    except ValueError:
        return dict(status="FAIL", recorded=rec["sha256"], actual=None)
    if not p.is_file():
        return dict(status="MISSING", recorded=rec["sha256"], actual=None)
    actual = sha256_file(p)
    return dict(status="PASS" if actual == rec["sha256"] else "FAIL", recorded=rec["sha256"], actual=actual)


# ------------------------------------------------------------------------------------------ async trigger
_Q: queue.Queue = queue.Queue(maxsize=1000)
_THREAD: threading.Thread | None = None
_TLOCK = threading.Lock()
_DONE = threading.Condition()
_PENDING = 0


def _worker() -> None:
    global _PENDING
    while True:
        try:
            job = _Q.get(timeout=30)
        except queue.Empty:
            return
        try:
            _run_job(**job)
        except Exception:                                    # never let one failed clip stop the recorder
            log.exception("evidence capture failed for %s", job.get("occurrence_id"))
        finally:
            with _DONE:
                _PENDING -= 1
                _DONE.notify_all()


def _run_job(db_path, occurrence_id, channel, start, end, breach_type, contract_id, evidence_dir=None) -> None:
    if evidence_for(db_path, occurrence_id):                   # one clip per airing covers every breach type
        return
    try:
        ev = capture_breach_evidence(channel, start, end, breach_type, occurrence_id=occurrence_id,
                                     evidence_dir=evidence_dir)
    except EvidenceUnavailable as exc:
        log.info("no evidence clip for occurrence %s: %s", occurrence_id, exc)
        return
    bind_evidence(db_path, occurrence_id, ev, breach_type, contract_id)


def submit_breach(db_path, occurrence_id: int, channel: str, start: float, end: float, breach_type: str,
                  contract_id: str | None = None, evidence_dir=None) -> None:
    """Fire-and-forget: the detection / alert loop only enqueues; a background thread cuts, hashes and stores."""
    global _THREAD, _PENDING
    job = dict(db_path=db_path, occurrence_id=int(occurrence_id), channel=channel, start=start, end=end,
               breach_type=breach_type, contract_id=contract_id, evidence_dir=evidence_dir)
    with _DONE:
        _PENDING += 1
    try:
        _Q.put_nowait(job)
    except queue.Full:
        with _DONE:
            _PENDING -= 1
        log.warning("evidence queue full; dropping clip for occurrence %s", occurrence_id)
        return
    with _TLOCK:
        if _THREAD is None or not _THREAD.is_alive():
            _THREAD = threading.Thread(target=_worker, name="kihc-evidence", daemon=True)
            _THREAD.start()


def flush(timeout: float = 30.0) -> bool:
    end = time.monotonic() + timeout
    with _DONE:
        while _PENDING:
            left = end - time.monotonic()
            if left <= 0:
                return False
            _DONE.wait(left)
    return True
