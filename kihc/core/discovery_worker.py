"""LiveStreamWorker that also feeds the unsupervised discovery engine (no extra decoding)."""
import numpy as np

from kihc.config import HOP_SIZE, LIVE_SAMPLE_RATE
from kihc.core.discovery import DiscoveryEngine
from kihc.core.live_worker import DECIM, LiveStreamWorker


class DiscoveryWorker(LiveStreamWorker):
    """Matches known reference ads (as LiveStreamWorker) and discovers recurring unknown segments."""

    wants_fingerprints = True

    def __init__(self, channel_name, stream_url, db_path=None, *, discovery: DiscoveryEngine | None = None,
                 discovery_kwargs: dict | None = None, **kwargs):
        super().__init__(channel_name, stream_url, db_path, **kwargs)
        self.discovery = discovery or DiscoveryEngine(channel_name, db_path, **(discovery_kwargs or {}))
        self.discovery.begin_new_source(self)          # reused engine: keep its history, shift our indices

    def _on_fingerprints(self, fp, base, x11, s_aligned, chunks):
        if fp.size == 0 or not chunks:
            return
        # capture time of each frame: frame j starts at 11025-grid sample j*HOP -> 22050-grid sample 2*j*HOP
        g = (np.arange(base, base + fp.size, dtype=np.int64) * HOP_SIZE * DECIM)
        g0s = np.array([c.g0 for c in chunks], dtype=np.int64)
        walls = np.array([c.wall0 for c in chunks], dtype=np.float64)
        idx = np.clip(np.searchsorted(g0s, g, side="right") - 1, 0, len(chunks) - 1)
        epochs = walls[idx] + (g - g0s[idx]) / LIVE_SAMPLE_RATE
        pcm = (np.clip(x11, -1.0, 1.0) * 32767).astype(np.int16)
        self.discovery.feed(fp, base, epochs, pcm, s_aligned, source=self)

    def _on_disconnect(self):
        super()._on_disconnect()
        self.discovery.flush()          # audio is not contiguous across a reconnect

    def status(self) -> dict:
        s = super().status()
        s["discovery"] = self.discovery.stats()
        return s
