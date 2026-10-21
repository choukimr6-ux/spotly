"""DiscoveryWorker that also runs the exact reference matcher on the live audio (Phase 13.1).

Known, verified commercials are recognised from the peak-constellation index and logged straight into
ad_occurrences (-> pods -> compliance/DZD), with no Whisper and no OCR. Everything the registry does not know
keeps flowing through the unchanged discovery loop (same worker, same audio), so an unknown creative is never lost.

To keep one airing from being counted twice, the legacy BER matcher is switched off for the reference ads the
registry covers (keep_legacy=True restores it; commit_airing then de-duplicates by overlap).
"""
import logging

import numpy as np

from kihc.analytics.reference_matcher import ReferenceMatcher
from kihc.config import LIVE_SAMPLE_RATE
from kihc.core.discovery_worker import DiscoveryWorker
from kihc.core.live_worker import _EDGE, _TAPS, DECIM

log = logging.getLogger("kihc.reference")


class Decimator:
    """Streaming 22050 -> 11025 Hz, same filter as live_worker.decimate but without seams between pieces."""

    def __init__(self):
        self.reset()

    def reset(self):
        self._pending = np.zeros(0, np.float32)
        self._first = True

    def process(self, pcm_i16: np.ndarray):
        """-> (samples at 11025 Hz, time of the first one relative to the first sample of pcm_i16 in seconds)."""
        before = self._pending.size
        r = np.concatenate([self._pending, pcm_i16.astype(np.float32) / 32768.0])
        h = (_EDGE + DECIM - 1) // DECIM
        i0 = 0 if self._first else h
        i1 = (r.size - _EDGE) // DECIM
        if i1 <= i0:
            self._pending = r
            return np.zeros(0, np.float32), 0.0
        y = np.convolve(r, _TAPS, mode="same")[::DECIM]
        self._pending = r[DECIM * (i1 - h):]
        self._first = False
        return y[i0:i1], (DECIM * i0 - before) / LIVE_SAMPLE_RATE


class ReferenceDiscoveryWorker(DiscoveryWorker):
    def __init__(self, channel_name, stream_url, db_path=None, *, reference: ReferenceMatcher | None = None,
                 keep_legacy: bool = False, **kwargs):
        super().__init__(channel_name, stream_url, db_path, **kwargs)
        self.reference = reference or ReferenceMatcher(db_path)
        if getattr(self.reference, "evidence_dir", None) is None:
            self.reference.evidence_dir = self.evidence_dir      # confirmed-airing clips land beside this worker's evidence
        self.keep_legacy = keep_legacy
        self.reference_events = []
        self._dec = Decimator()
        self._next_g = None            # next global 22050 Hz sample the reference matcher expects
        self._apply_exclusions()

    def _apply_exclusions(self):
        self._ignored_ad_ids = set() if self.keep_legacy else self.reference.covered_ad_ids()
        for ad_id in self._ignored_ad_ids:
            self._refs.pop(ad_id, None)

    def _on_fingerprints(self, fp, base, x11, s_aligned, chunks):
        try:
            super()._on_fingerprints(fp, base, x11, s_aligned, chunks)      # discovery of unknown creatives
        finally:
            self._feed_reference(chunks)

    def _feed_reference(self, chunks):
        try:
            if self.reference.reload_if_changed():
                self._apply_exclusions()
            for c in chunks:
                if self._next_g is not None and c.g0 < self._next_g:
                    continue                                              # already fed on an earlier pass
                if self._next_g is not None and c.g0 != self._next_g:     # gap: audio is not contiguous
                    self._restart_reference()
                self._next_g = c.g0 + c.pcm.size
                y, t_first = self._dec.process(c.pcm)
                if y.size:
                    self._record(self.reference.feed(self.channel_name, y, c.wall0 + t_first))
            self.reference.run_compliance()
        except Exception as exc:                                          # never break the discovery loop
            self.last_error = f"reference matcher failed: {type(exc).__name__}: {exc}"
            log.exception("[%s] reference matcher failed", self.channel_name)

    def _record(self, cands):
        for c in cands:
            self.reference_events.append(c)
            log.info("[%s] REFERENCE %s %s %s conf=%.3f", self.channel_name, c.decision, c.brand_name,
                     c.asset_id, c.confidence)
        del self.reference_events[:-200]

    def _restart_reference(self):
        self._record(self.reference.end_stream(self.channel_name))
        self._dec.reset()

    def _on_disconnect(self):
        super()._on_disconnect()
        try:
            self._restart_reference()
        finally:
            self._next_g = None

    def status(self) -> dict:
        s = super().status()
        s["reference"] = dict(self.reference.stats, **{k: v for k, v in self.reference.memory_report().items()
                                                       if k in ("assets", "index_mb", "lookup_ms_p50", "lookup_ms_max")})
        return s
