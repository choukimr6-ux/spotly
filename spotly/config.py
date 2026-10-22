"""Central paths and tunables.

Writable locations can be redirected with environment variables (the test suite points them at a
temporary directory so it never touches production data): SPOTLY_DB_PATH, SPOTLY_EVIDENCE_DIR,
SPOTLY_DISCOVERED_DIR, SPOTLY_REPORTS_DIR.
"""
import os
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent
DATA_DIR = PKG_DIR / "data"
REFERENCE_DIR = DATA_DIR / "reference_ads"
SAMPLES_DIR = DATA_DIR / "samples"
EVIDENCE_DIR = Path(os.environ.get("SPOTLY_EVIDENCE_DIR") or DATA_DIR / "evidence")
DISCOVERED_DIR = Path(os.environ.get("SPOTLY_DISCOVERED_DIR") or DATA_DIR / "discovered_ads")
REPORTS_DIR = Path(os.environ.get("SPOTLY_REPORTS_DIR") or PKG_DIR / "reports")
DEFAULT_DB_PATH = Path(os.environ.get("SPOTLY_DB_PATH") or DATA_DIR / "spotly.db")

# Fingerprint parameters (all audio is resampled to mono SAMPLE_RATE by FFmpeg).
SAMPLE_RATE = 11025
FRAME_SIZE = 2048          # ~186 ms analysis window
HOP_SIZE = 128             # ~11.6 ms hop -> fine alignment, robust to phase offsets
N_BANDS = 33               # 33 log-spaced bands -> 32 bits per sub-fingerprint
BAND_LOW_HZ = 300.0
BAND_HIGH_HZ = 3000.0
FP_ALGORITHM = "kihc-subband-v2-masked"
MASK_KEEP_FRACTION = 0.30  # fraction of reference bits (strongest derivatives) used for BER

# Matching
MATCH_THRESHOLD = 0.82     # confidence = 1 - BER; random audio sits near 0.5
CHUNK_SECONDS = 30         # streaming decode chunk; bounds RAM independent of stream length

# Live monitoring (Phase 2)
# FFmpeg input options that cap probe / demux buffering for every audio ingest (Phase 28.2). probesize 32768 (the first value
# tried) makes ffmpeg fail with "Output file does not contain any stream" on these MPEG-TS HLS feeds (audio is not found in the
# first 32 KB of a video-carrying segment); 262144 / 1 s is the smallest setting measured to work on all five Tier-1 feeds.
FFMPEG_LOW_MEM_ARGS = ["-probesize", "262144", "-analyzeduration", "1000000", "-max_delay", "500000", "-threads", "1"]
LIVE_SAMPLE_RATE = 22050       # PCM rate piped from FFmpeg; decimated 2:1 to SAMPLE_RATE for fingerprinting
LIVE_BUFFER_SECONDS = 30       # PCM ring buffer length
LIVE_MATCH_INTERVAL = 3.0      # seconds between match passes
LIVE_MATCH_THRESHOLD = 0.80
LIVE_BACKOFF_BASE = 2.0        # reconnect delays: 2, 4, 8, 16, 30, 30, ...
LIVE_BACKOFF_MAX = 30.0
LIVE_HEALTHY_AFTER = 30.0      # a connection that lasts this long resets the backoff
LIVE_STALL_TIMEOUT = 20.0      # no audio for this long -> kill FFmpeg and reconnect
IPTV_CACHE_DIR = DATA_DIR / "iptv_cache"
