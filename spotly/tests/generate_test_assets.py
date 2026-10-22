"""Generate self-contained synthetic test media with FFmpeg (no downloads).

Outputs:
  data/reference_ads/ad_a.mp4         10 s "Ad A": 440 Hz tremolo bed + distinct chime melody, SMPTE bars video
  data/reference_ads/ad_b_decoy.mp4   10 s decoy ad (different melody) that never airs -> negative control
  data/samples/broadcast_60s.mp4      60 s broadcast: pink-noise ambience, Ad A embedded at 00:25-00:35,
                                      broadband channel noise over everything, AAC re-encode
"""
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from spotly.config import REFERENCE_DIR, SAMPLES_DIR
from spotly.core.audio import ffmpeg_exe

AD_A = REFERENCE_DIR / "ad_a.mp4"
AD_B = REFERENCE_DIR / "ad_b_decoy.mp4"
BROADCAST = SAMPLES_DIR / "broadcast_60s.mp4"
AD_START, AD_DURATION, BROADCAST_DURATION = 25.0, 10.0, 60.0
VIDEO = "s=320x180:r=10"


def _chime_expr(bed_hz, notes, step, duration):
    terms = [f"0.22*sin(2*PI*{bed_hz}*t)*(0.6+0.4*sin(2*PI*3*t))"]
    t, i = 0.0, 0
    while t < duration:
        f = notes[i % len(notes)]
        terms.append(f"gte(t,{t:.3f})*exp(-6*(t-{t:.3f}))*(0.30*sin(2*PI*{f}*t)+0.12*sin(2*PI*{2 * f}*t))")
        t += step
        i += 1
    return "+".join(terms)


AD_A_EXPR = _chime_expr(440, [1318.5, 1760.0, 1568.0, 2093.0, 1174.7, 1975.5, 1396.9, 2349.3], 0.625, AD_DURATION)
AD_B_EXPR = _chime_expr(523.3, [987.8, 1661.2, 1244.5, 1864.7, 1108.7], 0.8, AD_DURATION)


def _run(args):
    proc = subprocess.run([ffmpeg_exe(), "-nostdin", "-v", "error", "-y", *args], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr)


def make_ad(path, expr, video_src):
    path.parent.mkdir(parents=True, exist_ok=True)
    _run(["-f", "lavfi", "-i", f"aevalsrc=exprs='{expr}':s=44100:d={AD_DURATION}",
          "-f", "lavfi", "-i", f"{video_src}=d={AD_DURATION}:{VIDEO}",
          "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
          "-c:a", "aac", "-b:a", "128k", "-shortest", str(path)])


def make_broadcast(path, ad_path):
    path.parent.mkdir(parents=True, exist_ok=True)
    pre, post = AD_START, BROADCAST_DURATION - AD_START - AD_DURATION
    graph = (
        # program audio: ambient pink noise / ad / ambient pink noise
        f"anoisesrc=c=pink:a=0.15:r=44100:d={pre}:seed=7[n1];"
        f"anoisesrc=c=pink:a=0.15:r=44100:d={post}:seed=11[n2];"
        "[0:a]aformat=sample_rates=44100:channel_layouts=mono[ada];"
        "[n1][ada][n2]concat=n=3:v=0:a=1[prog];"
        # channel noise across the whole broadcast (~ -26 dB relative to ad peaks)
        f"anoisesrc=c=white:a=0.02:r=44100:d={BROADCAST_DURATION}:seed=3[hiss];"
        "[prog][hiss]amix=inputs=2:duration=first:normalize=0[aout];"
        # program video: test pattern / ad bars / test pattern
        f"testsrc2=d={pre}:{VIDEO},format=yuv420p,setsar=1[v1];"
        f"testsrc2=d={post}:{VIDEO},format=yuv420p,setsar=1[v2];"
        "[0:v]fps=10,scale=320:180,format=yuv420p,setsar=1[adv];"
        "[v1][adv][v2]concat=n=3:v=1:a=0[vout]"
    )
    _run(["-i", str(ad_path), "-filter_complex", graph, "-map", "[vout]", "-map", "[aout]",
          "-c:v", "libx264", "-preset", "veryfast", "-c:a", "aac", "-b:a", "96k", str(path)])


def generate(force=False):
    if force or not AD_A.exists():
        make_ad(AD_A, AD_A_EXPR, "smptebars")
    if force or not AD_B.exists():
        make_ad(AD_B, AD_B_EXPR, "rgbtestsrc")
    if force or not BROADCAST.exists():
        make_broadcast(BROADCAST, AD_A)
    return AD_A, AD_B, BROADCAST


if __name__ == "__main__":
    for p in generate(force=True):
        print(f"{p}  ({p.stat().st_size / 1024:.0f} KiB)")
