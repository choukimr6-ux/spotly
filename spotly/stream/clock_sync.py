"""Broadcast-clock calibration (Phase 22): how far behind the master control room (MCR) is our capture clock?

Every live airing is timestamped on the *capture* clock: the wall time (NTP, then shown as CET / UTC+1) at which its audio
reached us. The broadcaster's own clock ran earlier - HLS packaging, CDN and the `-live_start_index` look-behind put the
audio we receive many seconds after it left the MCR. For a contract audit that gap, and how well it is known, belongs on
the certificate:

    drift = capture_time - broadcaster_time          (seconds; > 0: we stamp it later than the MCR clock says, < 0: the
                                                      broadcaster's own clock runs AHEAD of our NTP clock)

The sign is kept: El-Heddaf's program-date-time was measured 2026-10-20 running ~10-20 s ahead of NTP UTC, which is exactly
what a contract audit against the broadcaster's as-run log needs to know.

Sources, best first (the basis is stored with every reading):
  program-date-time  the HLS playlist's #EXT-X-PROGRAM-DATE-TIME is the broadcaster's UTC clock for each segment. At connect we
                     read the playlist, work out which segment ffmpeg starts with (live_start_index) and compare its PDT with
                     the wall time of the first audio we received.
  pts-relative       no PDT in the playlist: MPEG-TS PTS / PCR only give a relative clock, so the reading is the change of
                     (capture - PTS) since the first observation. It tracks buffering drift but not the absolute offset.

`ts_clock_refs()` parses raw MPEG-TS (188-byte packets): PCR from adaptation fields (27 MHz) and PTS from PES headers
(90 kHz). `ClockCalibrator` keeps the running offset (median) and its tolerance (rtt/2 + 95th-percentile deviation, never
below 0.2 s); `get_calibrator(channel)` is the per-channel instance the matcher reads when it logs an airing.
"""
import collections
import logging
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import requests

log = logging.getLogger("spotly.clock")

PCR_HZ, PTS_HZ = 27_000_000, 90_000
TS_PACKET = 188
MIN_MARGIN_S = 0.2
MAX_PLAUSIBLE_DRIFT_S = 600.0          # |reading| beyond 10 min is a wrong segment pick or a bad playlist, not a clock offset
HISTORY = 240
UA = {"User-Agent": "Mozilla/5.0 (Spotly broadcast monitor)"}


# ------------------------------------------------------------------------------------------- MPEG-TS PTS / PCR
def ts_clock_refs(data: bytes) -> dict:
    """First PCR and first PTS found in a chunk of MPEG-TS -> {'pcr_s': float|None, 'pts_s': float|None, 'packets': int}.

    Skips garbage until a 0x47 sync byte lines up three packets in a row; tolerates a truncated tail."""
    n = len(data)
    start = next((i for i in range(min(n, TS_PACKET * 4))
                  if all(i + k * TS_PACKET < n and data[i + k * TS_PACKET] == 0x47 for k in range(3))), None)
    out = dict(pcr_s=None, pts_s=None, packets=0)
    if start is None:
        return out
    for off in range(start, n - TS_PACKET + 1, TS_PACKET):
        p = data[off:off + TS_PACKET]
        if p[0] != 0x47:
            continue
        out["packets"] += 1
        pusi, afc = bool(p[1] & 0x40), (p[3] >> 4) & 0x3
        pos = 4
        if afc & 0x2:                                           # adaptation field
            alen = p[4]
            if out["pcr_s"] is None and alen >= 7 and p[5] & 0x10:
                base = (p[6] << 25) | (p[7] << 17) | (p[8] << 9) | (p[9] << 1) | (p[10] >> 7)
                ext = ((p[10] & 0x01) << 8) | p[11]
                out["pcr_s"] = (base * 300 + ext) / PCR_HZ
            pos = 5 + alen
        if out["pts_s"] is None and pusi and afc & 0x1 and p[pos:pos + 3] == b"\x00\x00\x01" and pos + 14 <= TS_PACKET:
            if p[pos + 7] & 0x80:                               # PTS_DTS_flags: PTS present
                b = p[pos + 9:pos + 14]
                out["pts_s"] = (((b[0] >> 1) & 0x07) << 30 | b[1] << 22 | (b[2] >> 1) << 15 | b[3] << 7 | (b[4] >> 1)) / PTS_HZ
        if out["pcr_s"] is not None and out["pts_s"] is not None:
            break
    return out


# ------------------------------------------------------------------------------------------- playlist
@dataclass
class Segment:
    uri: str
    start_epoch: float | None          # broadcaster UTC from #EXT-X-PROGRAM-DATE-TIME (propagated), None if absent
    duration: float


def parse_playlist(text: str) -> list[Segment]:
    segs, pdt, dur = [], None, None
    for line in (ln.strip() for ln in text.splitlines()):
        if line.startswith("#EXT-X-PROGRAM-DATE-TIME:"):
            try:
                pdt = datetime.fromisoformat(line.split(":", 1)[1].strip().replace("Z", "+00:00")).timestamp()
            except ValueError:
                pdt = None
        elif line.startswith("#EXTINF:"):
            m = re.match(r"#EXTINF:([0-9.]+)", line)
            dur = float(m.group(1)) if m else None
        elif line and not line.startswith("#") and dur is not None:
            segs.append(Segment(line, pdt, dur))
            pdt = pdt + dur if pdt is not None else None
            dur = None
    return segs


def segment_at(segs: list[Segment], live_start_index: int = -2) -> Segment | None:
    """The segment ffmpeg's -live_start_index selects (negative: counted from the newest)."""
    if not segs:
        return None
    i = len(segs) + live_start_index if live_start_index < 0 else live_start_index
    return segs[max(0, min(len(segs) - 1, i))]


# ------------------------------------------------------------------------------------------- calibrator
class ClockCalibrator:
    """Running capture-vs-broadcaster offset for one channel. Thread-safe; readers never block on the network."""

    def __init__(self, channel: str, history: int = HISTORY):
        self.channel = channel
        self._lock = threading.Lock()
        self._deltas = collections.deque(maxlen=history)
        self._pts_ref: tuple[float, float] | None = None       # (capture epoch, pts s) of the first PTS-only observation
        self.basis = "none"
        self.rtt_s = 0.0
        self.updated_epoch: float | None = None
        self.pts_vs_pdt_ppm: float | None = None

    def observe(self, capture_epoch: float, broadcaster_epoch: float, *, basis: str = "program-date-time",
                rtt_s: float = 0.0) -> float | None:
        """delta = capture_time - broadcaster_time. Implausible readings are ignored; returns the stored delta or None."""
        delta = capture_epoch - broadcaster_epoch
        if abs(delta) > MAX_PLAUSIBLE_DRIFT_S:
            log.warning("[%s] clock reading %.2f s outside +/-%.0f s - ignored", self.channel, delta, MAX_PLAUSIBLE_DRIFT_S)
            return None
        with self._lock:
            self._deltas.append(delta)
            self.basis, self.rtt_s, self.updated_epoch = basis, float(rtt_s), time.time()
        if basis == "program-date-time":                      # Phase 26: the MCR synchronizer sees the same reading
            from spotly.stream import time_synchronizer
            time_synchronizer.get_synchronizer().observe_manifest(self.channel, capture_epoch, broadcaster_epoch, rtt_s)
        return delta

    def observe_pts(self, capture_epoch: float, pts_s: float, rtt_s: float = 0.0) -> float | None:
        """PTS-only source: delta = growth of (capture - PTS) since the first observation (absolute offset unknown)."""
        with self._lock:
            if self._pts_ref is None:
                self._pts_ref = (capture_epoch, pts_s)
            ref_cap, ref_pts = self._pts_ref
            delta = (capture_epoch - ref_cap) - (pts_s - ref_pts)
            if self.basis in ("none", "pts-relative"):
                self._deltas.append(delta)
                self.basis, self.rtt_s, self.updated_epoch = "pts-relative", float(rtt_s), time.time()
        from spotly.stream import time_synchronizer
        time_synchronizer.get_synchronizer().observe_pts(self.channel, capture_epoch, pts_s, rtt_s)
        return delta

    def snapshot(self) -> dict | None:
        """{'offset_s', 'margin_s', 'samples', 'basis', 'updated_epoch'} or None while nothing was measured."""
        with self._lock:
            d = sorted(self._deltas)
            if not d:
                return None
            med = d[len(d) // 2]
            dev = sorted(abs(x - med) for x in d)
            p95 = dev[min(len(dev) - 1, int(len(dev) * 0.95))]
            return dict(channel=self.channel, offset_s=round(med, 3), margin_s=round(max(MIN_MARGIN_S, self.rtt_s / 2 + p95), 3),
                        samples=len(d), basis=self.basis, updated_epoch=self.updated_epoch,
                        pts_vs_pdt_ppm=self.pts_vs_pdt_ppm)


_calibrators: dict[str, ClockCalibrator] = {}
_reg_lock = threading.Lock()


def get_calibrator(channel: str) -> ClockCalibrator:
    with _reg_lock:
        return _calibrators.setdefault(channel, ClockCalibrator(channel))


def reset() -> None:
    with _reg_lock:
        _calibrators.clear()
    from spotly.stream import time_synchronizer
    time_synchronizer.reset()


# ------------------------------------------------------------------------------------------- storage
DDL = """
CREATE TABLE IF NOT EXISTS clock_calibration (
    channel       TEXT PRIMARY KEY,
    offset_s      REAL NOT NULL,
    margin_s      REAL NOT NULL,
    samples       INTEGER NOT NULL,
    basis         TEXT NOT NULL,
    updated_epoch REAL NOT NULL
);
"""


def persist(db_path, channel: str) -> None:
    """Publish the channel's calibration for other processes (the API server reads it for /api/telemetry)."""
    snap = get_calibrator(channel).snapshot()
    if not snap:
        return
    from spotly.database.schema import get_db
    try:
        with get_db(db_path) as conn:
            conn.execute(DDL)
            conn.execute("INSERT INTO clock_calibration (channel, offset_s, margin_s, samples, basis, updated_epoch) "
                         "VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(channel) DO UPDATE SET offset_s = excluded.offset_s, "
                         "margin_s = excluded.margin_s, samples = excluded.samples, basis = excluded.basis, "
                         "updated_epoch = excluded.updated_epoch",
                         (channel, snap["offset_s"], snap["margin_s"], snap["samples"], snap["basis"], snap["updated_epoch"]))
    except Exception as exc:                                   # telemetry must never break the audio path
        log.warning("[%s] clock calibration not stored: %s", channel, exc)


def load_all(db_path) -> list[dict]:
    from spotly.database.schema import get_db
    try:
        with get_db(db_path) as conn:
            conn.execute(DDL)
            return [dict(r) for r in conn.execute("SELECT * FROM clock_calibration ORDER BY channel")]
    except Exception:
        return []


def occurrence_fields(channel: str, max_age_s: float = 600.0) -> dict:
    """Columns for an ad_occurrences row: {'clock_drift_seconds', 'clock_drift_margin_s', 'clock_drift_basis'} (None when
    the channel has no fresh calibration - an unmeasured clock is stored as unknown, never as zero)."""
    snap = get_calibrator(channel).snapshot()
    if not snap or (snap["updated_epoch"] and time.time() - snap["updated_epoch"] > max_age_s):
        return dict(clock_drift_seconds=None, clock_drift_margin_s=None, clock_drift_basis=None)
    return dict(clock_drift_seconds=snap["offset_s"], clock_drift_margin_s=snap["margin_s"], clock_drift_basis=snap["basis"])


# ------------------------------------------------------------------------------------------- live probe
OPENING_RE = re.compile(r"Opening '([^']+)' for reading")


def opened_segment(line: str) -> str | None:
    """URL of a media segment from an ffmpeg log line ("Opening '...ts' for reading"); playlists are not segments."""
    m = OPENING_RE.search(line)
    return m.group(1) if m and ".m3u8" not in m.group(1).lower() else None


class ConnectProbe:
    """Used by the audio worker around each (re)connect, without ever delaying it.

        probe = ConnectProbe(channel, playlist_url, db_path)
        probe.connecting()                    # a new connection starts
        probe.segment_opened(url)             # ffmpeg's log: the first segment it really opened (-live_start_index decided)
        probe.first_audio(anchor_epoch)       # wall time the first PCM sample was received

    The segment ffmpeg opened is looked up in a freshly fetched playlist to get its #EXT-X-PROGRAM-DATE-TIME: the
    broadcaster's own clock for exactly the audio we start with, however long ffmpeg took to connect."""

    def __init__(self, channel: str, playlist_url: str, db_path=None, timeout: float = 4.0):
        self.channel, self.url, self.db_path, self.timeout = channel, playlist_url, db_path, timeout
        self._segment: Segment | None = None
        self._segment_url: str | None = None
        self._rtt = 0.0
        self._ready = threading.Event()
        self._seen = False

    def connecting(self) -> None:
        self._segment = self._segment_url = None
        self._rtt, self._seen = 0.0, False
        self._ready.clear()

    def segment_opened(self, url: str) -> None:
        if self._seen:
            return                                              # only the first segment fixes the clock relation
        self._seen = True
        self._segment_url = url
        threading.Thread(target=self._lookup, args=(url,), daemon=True, name="spotly-clock-fetch").start()

    LOOKUP_ATTEMPTS = 3

    def _lookup(self, url: str) -> None:
        """Find the opened segment in a fresh playlist. A slow link (the playlist GET timing out) is retried: Phase 26 found a
        live service whose single connect-time lookup had timed out and which therefore never stored a clock reading."""
        try:
            for attempt in range(self.LOOKUP_ATTEMPTS):
                try:
                    t0 = time.time()
                    r = requests.get(self.url, timeout=self.timeout, headers=UA)
                    self._rtt = time.time() - t0
                    if r.ok:
                        from urllib.parse import urljoin
                        for seg in parse_playlist(r.text):
                            if urljoin(self.url, seg.uri) == url or url.endswith(seg.uri):
                                self._segment = seg
                                return
                        return                                          # playlist answered but the segment has expired
                except requests.RequestException as exc:
                    log.debug("[%s] clock playlist fetch failed (attempt %d): %s", self.channel, attempt + 1, exc)
                time.sleep(1.0)
        finally:
            self._ready.set()

    def first_audio(self, anchor_epoch: float) -> None:
        threading.Thread(target=self._calibrate, args=(anchor_epoch,), daemon=True, name="spotly-clock-cal").start()

    def _calibrate(self, anchor_epoch: float) -> None:
        self._ready.wait(self.LOOKUP_ATTEMPTS * (self.timeout + 1) + 2)
        seg = self._segment
        if seg is None:
            return                                              # the segment already left the playlist: no reading, no guess
        cal = get_calibrator(self.channel)
        if seg.start_epoch is None:                             # no PDT: fall back to the segment's own MPEG-TS PTS
            pts = self._segment_pts(self._segment_url)
            if pts is not None:
                cal.observe_pts(anchor_epoch, pts, self._rtt)
                persist(self.db_path, self.channel)
            return
        if cal.observe(anchor_epoch, seg.start_epoch, basis="program-date-time", rtt_s=self._rtt) is not None:
            snap = cal.snapshot()
            log.info("[%s] broadcast clock: capture - MCR = %+.2f s +/- %.2f s (%s)", self.channel, snap["offset_s"],
                     snap["margin_s"], snap["basis"])
            persist(self.db_path, self.channel)

    def _segment_pts(self, url: str | None) -> float | None:
        """PTS (s) of the first audio / video access unit of a segment: only its first 48 KB is downloaded."""
        if not url:
            return None
        try:
            r = requests.get(url, timeout=self.timeout, headers={**UA, "Range": "bytes=0-49151"})
            refs = ts_clock_refs(r.content) if r.ok else {}
        except requests.RequestException:
            return None
        return refs.get("pts_s") if refs.get("pts_s") is not None else refs.get("pcr_s")


def probe_for(channel: str, url, db_path=None) -> "ConnectProbe | None":
    """A ConnectProbe for HLS sources (a media playlist URL); None for anything else (files, RTMP, ...)."""
    if isinstance(url, str) and url.lower().split("?")[0].endswith(".m3u8") and url.startswith("http"):
        return ConnectProbe(channel, url, db_path)
    return None
