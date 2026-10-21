"""KIHC commercial pilot: an isolated, non-destructive end-to-end run.

    python run_market_pilot.py            # run, print the report, clean up
    python run_market_pilot.py --keep     # keep pilot_test.db + evidence for inspection

What it does (nothing touches the production database, evidence folder or source code):
  1. Synthesizes two distinct telecom commercials in memory: Mobilis (30 s) and Ooredoo (20 s).
     All records are labelled "SYNTHETIC PILOT" - they are test tones, not real airings.
  2. Fingerprints and indexes them into pilot_test.db inside a fresh temporary directory.
  3. Builds a 3-minute El-Heddaf TV segment: programme intro, a break of
     Mobilis -> 2 s -> Ooredoo -> 2 s -> Mobilis, then outro, under channel noise, and passes it
     through a real AAC encode (FFmpeg pipes, in memory).
  4. Scans it with KIHC's in-memory matcher, reconstructs the pod, checks positions
     (First / Middle / Last), verifies every SHA-256 hash, proves tampering is detected,
     and renders a PDF certificate per airing (in memory).
  5. Prints a summary table; exits 0 only if every check passed.
"""
import argparse
import hashlib
import io
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from kihc.config import DEFAULT_DB_PATH, EVIDENCE_DIR  # noqa: E402
from kihc.core.audio import ffmpeg_exe  # noqa: E402
from kihc.core.library import ingest_bytes, scan_bytes, timecode  # noqa: E402
from kihc.core.pod_analyzer import cluster_occurrences_into_pods  # noqa: E402
from kihc.core.proof import verify_row  # noqa: E402
from kihc.database.schema import init_db  # noqa: E402
from kihc.reports.certificate import certificate_for  # noqa: E402
from kihc.service.sysinfo import memory_mb  # noqa: E402

SR = 44100
CHANNEL = "El-Heddaf TV (synthetic pilot)"
LABEL = "SYNTHETIC PILOT"
INTRO, GAP, TOTAL = 40.0, 2.0, 180.0
RAM_BUDGET_MB = 300.0
START_TOLERANCE = 0.5            # s; AAC priming shifts ADTS timing by ~20 ms


# ---------------------------------------------------------------- synthesis (all in memory)

CHIME_TAIL = 1.5                 # s; exp(-6 * 1.5) ~ 1e-4, so each chime is computed only over its decay
NOISE_BLOCK = 10.0               # s; noise is synthesized blockwise to keep FFT temporaries small


def commercial(duration, bed_hz, notes, step, seed):
    """Jingle-like creative: tremolo bed + decaying chime melody + a little texture noise (float32)."""
    n = int(duration * SR)
    t = np.arange(n, dtype=np.float32) / SR
    x = (0.20 * np.sin(2 * np.pi * bed_hz * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t))).astype(np.float32)
    for i, onset in enumerate(np.arange(0, duration, step)):
        a, b = int(onset * SR), min(n, int((onset + CHIME_TAIL) * SR))
        tt = t[a:b]
        f = notes[i % len(notes)]
        x[a:b] += np.exp(-6 * (tt - onset)) * (0.30 * np.sin(2 * np.pi * f * tt) + 0.12 * np.sin(4 * np.pi * f * tt))
    x += 0.01 * np.random.default_rng(seed).standard_normal(n, dtype=np.float32)
    return x / np.abs(x).max() * 0.7


def pink_noise(seconds, amp, seed):
    """Approximately pink (1/f) noise, generated in NOISE_BLOCK pieces, float32 out."""
    n, rng = int(seconds * SR), np.random.default_rng(seed)
    out = np.empty(n, dtype=np.float32)
    block = int(NOISE_BLOCK * SR)
    for a in range(0, n, block):
        m = min(block, n - a)
        spec = np.fft.rfft(rng.standard_normal(m, dtype=np.float32))
        spec /= np.sqrt(np.maximum(np.arange(spec.size), 1))
        out[a:a + m] = np.fft.irfft(spec, m)
    return out / np.abs(out).max() * amp


def wav_bytes(x):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())
    return buf.getvalue()


def aac_bytes(wav: bytes) -> bytes:
    """Real broadcast-style lossy encode, entirely through pipes."""
    p = subprocess.run([ffmpeg_exe(), "-nostdin", "-v", "error", "-f", "wav", "-i", "pipe:0",
                        "-c:a", "aac", "-b:a", "96k", "-f", "adts", "pipe:1"], input=wav, capture_output=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.decode(errors="replace"))
    return p.stdout


def build_break(mobilis, ooredoo):
    """Intro programme -> Mobilis -> 2 s -> Ooredoo -> 2 s -> Mobilis -> outro; channel noise over all."""
    plan, t = [], INTRO
    for brand, clip in (("Mobilis", mobilis), ("Ooredoo", ooredoo), ("Mobilis", mobilis)):
        plan.append((brand, t, t + len(clip) / SR))
        t += len(clip) / SR + GAP
    x = pink_noise(TOTAL, 0.02, 3)                                           # channel noise floor
    x[: int(INTRO * SR)] += pink_noise(INTRO, 0.25, 1)                         # programme before the break
    outro = int((plan[-1][2] + GAP) * SR)
    x[outro:] += pink_noise((x.size - outro) / SR, 0.25, 2)[: x.size - outro]  # programme after the break
    for (brand, start, _), clip in zip(plan, (mobilis, ooredoo, mobilis)):
        i = int(round(start * SR))
        x[i:i + clip.size] += clip
    return x, plan


# ---------------------------------------------------------------- isolation guards

def fingerprint_production():
    db = Path(DEFAULT_DB_PATH)
    digest = hashlib.sha256(db.read_bytes()).hexdigest() if db.exists() else "absent"
    evidence = sorted(p.name for p in Path(EVIDENCE_DIR).glob("*")) if Path(EVIDENCE_DIR).exists() else []
    return digest, evidence


# ---------------------------------------------------------------- run

class Report:
    def __init__(self):
        self.rows = []

    def check(self, step, ok, detail):
        self.rows.append((step, bool(ok), detail))
        return ok

    @property
    def passed(self):
        return all(ok for _, ok, _ in self.rows)


def run(workdir: Path) -> tuple[Report, list, list]:
    rep = Report()
    db = workdir / "pilot_test.db"
    evidence = workdir / "evidence"
    t0 = time.perf_counter()

    # 1. creatives
    mobilis = commercial(30.0, 330.0, [1046.5, 1318.5, 1568.0, 2093.0, 1760.0, 1396.9], 0.75, seed=11)
    ooredoo = commercial(20.0, 587.3, [880.0, 1975.5, 1174.7, 2349.3, 1661.2], 0.55, seed=22)
    rep.check("Synthesize creatives", True, f"Mobilis {mobilis.size / SR:.1f}s, Ooredoo {ooredoo.size / SR:.1f}s")

    # 2. index into the isolated pilot DB
    ids = {}
    for brand, clip, campaign in (("Mobilis", mobilis, f"Telecom spot 30s [{LABEL}]"),
                                  ("Ooredoo", ooredoo, f"Telecom spot 20s [{LABEL}]")):
        res = ingest_bytes(wav_bytes(clip), f"{brand.lower()}_pilot.wav", brand, campaign, db_path=db,
                           expected_duration=clip.size / SR, industry="Telecom")
        ids[brand] = res.ad_id
        rep.check(f"Index {brand}", res.committed and not res.warnings,
                  f"ad_id={res.ad_id}, {res.measured_duration:.2f}s"
                  + (f" WARN: {res.warnings}" if res.warnings else ""))

    # 3. broadcast segment (AAC, in memory)
    signal_, plan = build_break(mobilis, ooredoo)
    segment = aac_bytes(wav_bytes(signal_))
    rep.check("Build 3-min break", len(segment) > 0,
              f"{TOTAL:.0f}s AAC 96k ({len(segment) / 1024:.0f} KiB in memory), break {plan[0][1]:.0f}s–{plan[-1][2]:.0f}s")

    # 4. scan
    rows = scan_bytes(segment, "pilot_break.aac", f"{CHANNEL} · pilot_break.aac", db_path=db, evidence_dir=evidence)
    rows.sort(key=lambda r: r["airing_start"])
    found = [(r["brand_name"], r["airing_start"]) for r in rows]
    rep.check("Detect airings", len(rows) == 3 and [b for b, _ in found] == [b for b, _, _ in plan],
              f"{len(rows)} detected: " + ", ".join(f"{b}@{timecode(s)}" for b, s in found))
    timing_ok = len(rows) == 3 and all(abs(r["airing_start"] - p[1]) <= START_TOLERANCE for r, p in zip(rows, plan))
    rep.check("Timing accuracy", timing_ok, "max start error " + (
        f"{max(abs(r['airing_start'] - p[1]) for r, p in zip(rows, plan)) * 1000:.0f} ms" if len(rows) == 3 else "n/a"))
    rep.check("Confidence ≥ 0.85", rows and min(r["confidence_score"] for r in rows) >= 0.85,
              "min " + (f"{min(r['confidence_score'] for r in rows):.4f}" if rows else "n/a"))

    # 5. pod positions
    pods = cluster_occurrences_into_pods(db_path=db)
    pod = pods[0] if pods else None
    positions = [a["position_label"] for a in pod.ads] if pod else []
    rep.check("Reconstruct pod", len(pods) == 1 and pod.total_ads_in_pod == 3,
              f"{len(pods)} pod(s)" + (f": {pod.pod_id}, {pod.pod_duration_seconds:.1f}s, clutter {pod.clutter}"
                                       if pod else ""))
    rep.check("Pod positions", positions == ["First-in-Break", "Middle", "Last-in-Break"], " → ".join(
        f"{a['brand']} {a['position_of']}" for a in pod.ads) if pod else "no pod")

    # 6. integrity + tamper detection + certificates
    conn = init_db(db)
    stored = conn.execute("SELECT * FROM ad_occurrences ORDER BY airing_start").fetchall()
    mem = sqlite3.connect(":memory:")
    conn.backup(mem)
    conn.close()
    rep.check("SHA-256 verification", len(stored) == 3 and all(verify_row(r) is True for r in stored),
              f"{sum(verify_row(r) is True for r in stored)}/{len(stored)} hashes verify")
    mem.row_factory = sqlite3.Row
    mem.execute("UPDATE ad_occurrences SET confidence_score = 0.999 WHERE occurrence_id = ?",
                (stored[0]["occurrence_id"],))
    tampered = mem.execute("SELECT * FROM ad_occurrences WHERE occurrence_id = ?",
                           (stored[0]["occurrence_id"],)).fetchone()
    mem.close()
    rep.check("Tamper detection", verify_row(tampered) is False, "edited confidence (in-memory copy) → FAIL detected")
    pdfs = [certificate_for(db, r["occurrence_id"]) for r in stored]
    rep.check("PDF certificates", all(p.startswith(b"%PDF-") for p in pdfs) and len(pdfs) == 3,
              f"{len(pdfs)} rendered in memory, {sum(map(len, pdfs)) / 1024:.0f} KiB total")

    peak = memory_mb()[1]
    rep.check("Peak memory", peak <= RAM_BUDGET_MB, f"{peak:.1f} MB (budget {RAM_BUDGET_MB:.0f} MB)")
    rep.check("Runtime", True, f"{time.perf_counter() - t0:.1f}s")
    return rep, rows, (pod.ads if pod else [])


def print_report(rep, rows, ads, workdir, kept):
    line = "─" * 96
    print(f"\n{line}\n KIHC MARKET PILOT · {CHANNEL} · {LABEL} (test tones, not real airings)\n{line}")
    print(f" {'Step':<30} {'Result':<7} Detail")
    print(f" {'-' * 30} {'-' * 7} {'-' * 56}")
    for step, ok, detail in rep.rows:
        print(f" {step:<30} {'PASS' if ok else 'FAIL':<7} {detail}")
    print(f"\n {'#':<5} {'Brand':<9} {'Start':<13} {'End':<13} {'Dur(s)':>7} {'Conf':>7}  {'Position':<16} SHA-256")
    print(f" {'-' * 5} {'-' * 9} {'-' * 13} {'-' * 13} {'-' * 7} {'-' * 7}  {'-' * 16} {'-' * 16}")
    pos = {a["occurrence_id"]: a for a in ads}
    for r in rows:
        a = pos.get(r["occurrence_id"], {})
        print(f" {r['occurrence_id']:<5} {r['brand_name']:<9} {timecode(r['airing_start']):<13} "
              f"{timecode(r['airing_end']):<13} {r['duration_detected']:>7.2f} {r['confidence_score']:>7.4f}  "
              f"{a.get('position_of', '?') + ' ' + a.get('position', ''):<16} {(r['verification_hash'] or '')[:16]}…")
    print(f"\n Pilot workspace: {workdir} ({'kept' if kept else 'removed'})")
    print(f" OVERALL: {'PASS' if rep.passed else 'FAIL'} ({sum(ok for _, ok, _ in rep.rows)}/{len(rep.rows)} checks)")
    print(line)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--keep", action="store_true", help="keep pilot_test.db and evidence after the run")
    args = ap.parse_args(argv)

    before = fingerprint_production()
    workdir = Path(tempfile.mkdtemp(prefix="kihc_pilot_"))
    try:
        rep, rows, ads = run(workdir)
    except Exception as exc:                       # report, never leave a half-written workspace silently
        rep, rows, ads = Report(), [], []
        rep.check("Pilot run", False, f"{type(exc).__name__}: {exc}")
    after = fingerprint_production()
    rep.check("Production DB untouched", before[0] == after[0], f"sha256 {after[0][:16]}… unchanged"
              if before[0] == after[0] else "PRODUCTION DATABASE CHANGED")
    rep.check("Production evidence untouched", before[1] == after[1], f"{len(after[1])} files, no additions"
              if before[1] == after[1] else f"evidence folder changed: {set(after[1]) ^ set(before[1])}")
    if not args.keep and workdir.name.startswith("kihc_pilot_"):
        shutil.rmtree(workdir, ignore_errors=True)
    print_report(rep, rows, ads, workdir, args.keep)
    return 0 if rep.passed else 1


if __name__ == "__main__":
    sys.exit(main())
