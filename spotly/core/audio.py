"""FFmpeg discovery and in-memory PCM streaming (no temp WAV files)."""
import functools
import shutil
import subprocess

import numpy as np

from spotly.config import SAMPLE_RATE


@functools.lru_cache(maxsize=1)
def ffmpeg_exe() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "FFmpeg not found. Install it (winget install Gyan.FFmpeg / apt install ffmpeg) "
            "or `pip install imageio-ffmpeg`."
        ) from exc


def iter_pcm(source, chunk_seconds: float, sample_rate: int = SAMPLE_RATE):
    """Yield float32 mono chunks decoded by FFmpeg through a stdout pipe.

    Memory is bounded by chunk size regardless of input length.
    """
    cmd = [ffmpeg_exe(), "-nostdin", "-v", "error", "-i", str(source),
           "-vn", "-ac", "1", "-ar", str(sample_rate), "-f", "s16le", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    chunk_bytes = int(chunk_seconds * sample_rate) * 2
    try:
        while True:
            buf = proc.stdout.read(chunk_bytes)
            if not buf:
                break
            usable = len(buf) - (len(buf) % 2)
            yield np.frombuffer(buf[:usable], dtype="<i2").astype(np.float32) / 32768.0
    finally:
        proc.stdout.close()
        err = proc.stderr.read().decode(errors="replace").strip()
        proc.stderr.close()
        rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"FFmpeg failed decoding {source}: {err}")


def has_video_stream(source) -> bool:
    proc = subprocess.run([ffmpeg_exe(), "-nostdin", "-hide_banner", "-i", str(source)],
                          capture_output=True, text=True, errors="replace")
    return "Video:" in proc.stderr
