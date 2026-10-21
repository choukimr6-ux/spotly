"""Offline speech-to-text for discovered assets (faster-whisper, CPU, int8) + brand suggestion.

Runs as a short-lived, low-priority subprocess, never inside the monitoring process:

    python -m kihc.analytics.transcriber --pending            # every unclassified asset without a transcript
    python -m kihc.analytics.transcriber --asset ELHEDDAF_AD_003 --retranscribe
    python -m kihc.cli transcribe-assets                       # same, via the CLI

Why a separate process: measured on this machine, Whisper 'base' int8 peaks at ~415-440 MB and even
'tiny' at ~290 MB, so loading it into the ~95 MB pilot would break its memory budget for good. Here the
model lives only for the batch (about 5-40 s per clip on one core), a watchdog exits the process if it
passes --memory-cap-mb, and all memory is returned to the OS when it finishes.

Audio is decoded by FFmpeg straight into memory at 16 kHz (faster-whisper's own PyAV decoder is
incompatible with the installed PyAV 19). Whisper hallucinates on music and jingles (repetition
loops, invented English or foreign-script text), so segments are kept only if no_speech_prob < 0.6,
avg_logprob > -1.0 and compression_ratio < 2.4; text in other scripts is dropped and repeated words
or phrases are collapsed. Darija is transcribed as Arabic and is only partly intelligible: the brand
suggestion is advisory and needs a human to confirm it.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone

import numpy as np

from kihc.analytics.brand_extractor import detect_station_promo, extract_brands, suggest_brand
from kihc.config import DATA_DIR, DEFAULT_DB_PATH
from kihc.core.audio import ffmpeg_exe
from kihc.database.schema import get_db
from kihc.service.sysinfo import memory_mb

MODEL_DIR = DATA_DIR / "models"
DEFAULT_MODEL = "base"
MEMORY_CAP_MB = 480
EXIT_MEMORY_CAP = 75
_ALLOWED = re.compile(r"[^\w\s؀-ۿ.,!?'\-%:/()«»“”\"]", re.UNICODE)
_FOREIGN_SCRIPT = re.compile(r"[Ͱ-ϿЀ-ӿ֐-׿ऀ-෿฀-࿿"
                             r"぀-ヿ一-鿿가-힯]")


def decode_16k(path) -> np.ndarray:
    p = subprocess.run([ffmpeg_exe(), "-nostdin", "-v", "error", "-i", str(path), "-ac", "1", "-ar", "16000",
                        "-f", "f32le", "-"], capture_output=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.decode(errors="replace").strip()[:200])
    return np.frombuffer(p.stdout, np.float32)


def collapse_repeats(text: str, max_ngram: int = 4) -> str:
    """'a a a b c b c b c' -> 'a b c' (Whisper repetition loops): drop immediate repeats of 1..max_ngram words."""
    toks, out, i = text.split(), [], 0
    while i < len(toks):
        for n in range(1, max_ngram + 1):
            if len(toks[i:i + n]) == n and toks[i:i + n] == toks[i + n:i + 2 * n]:
                del toks[i + n:i + 2 * n]                 # remove the repeat, re-examine the same position
                break
        else:
            out.append(toks[i])
            i += 1
    return " ".join(out)


SOUNDTRACK_LABEL = "Audio: Commercial Soundtrack & Sound Effects"
MIN_LOGPROB = -0.75            # stricter than the per-segment keep rule (-1.0): the transcript as a whole must be grounded
MIN_WORDS = 3
_WORD = re.compile(r"\w+", re.UNICODE)


def assess_transcript(text: str | None, avg_logprob: float | None = None) -> tuple[bool, str]:
    """Perceptual coherence gate for what the UI may show of a transcript -> (show_text, reason).

    Whisper invents fluent-looking Darija on music, jingles and sound effects. The raw text is hidden (the UI shows
    SOUNDTRACK_LABEL instead) when
      * the transcript's mean logprob is poor (< -0.75),
      * it has fewer than 3 words,
      * it is a repetition loop: unique words < 55 % of 8+ words, or one word makes up 30 %+ of 6+ words,
      * its words are ungrounded noise: a majority are 1-2 letter fragments, or 40 %+ are digits / symbols.
    The rule needs no model: it also works on transcripts stored before logprobs were kept."""
    words = _WORD.findall(text or "")
    if avg_logprob is not None and avg_logprob < MIN_LOGPROB:
        return False, f"low confidence (mean logprob {avg_logprob:.2f})"
    if len(words) < MIN_WORDS:
        if words and suggest_brand(text or ""):          # a short line that names a registry brand is grounded
            return True, "names a known brand"
        return False, "too little speech"
    low = [w.casefold() for w in words]
    if len(low) >= 8 and len(set(low)) / len(low) < 0.55:
        return False, "repetition loop"
    top = max(low.count(w) for w in set(low))
    if len(low) >= 6 and top / len(low) >= 0.30:
        return False, "one word dominates"
    if sum(len(w) <= 2 for w in words) / len(words) > 0.5:
        return False, "fragmentary words"
    if sum(w.isdigit() for w in words) / len(words) >= 0.4:
        return False, "numeric noise"
    return True, "coherent"


def keep_segment(seg) -> bool:
    text = seg.text.strip()
    return (bool(text) and seg.no_speech_prob < 0.6 and seg.avg_logprob > -1.0 and seg.compression_ratio < 2.4
            and not _FOREIGN_SCRIPT.search(text))


def clean_text(segments) -> str:
    text = " ".join(_ALLOWED.sub(" ", s.text.strip()) for s in segments if keep_segment(s))
    return collapse_repeats(" ".join(text.split()))


class Transcriber:
    def __init__(self, model: str = DEFAULT_MODEL, threads: int = 1):
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        from faster_whisper import WhisperModel
        kw = dict(device="cpu", compute_type="int8", cpu_threads=threads, num_workers=1,
                  download_root=str(MODEL_DIR))
        try:
            self.model = WhisperModel(model, local_files_only=True, **kw)      # no network if already cached
        except Exception:
            self.model = WhisperModel(model, **kw)
        self.name = f"faster-whisper {model} int8"

    def transcribe(self, path, max_seconds: float | None = None) -> tuple[str, str | None, float]:
        audio = decode_16k(path)
        if max_seconds:                          # spot only: the review clip's post-roll may hold the next ad
            audio = audio[:int(max_seconds * 16000)]
        kept, langs = [], []
        for i in range(0, audio.size, 30 * 16000):                           # <= 30 s windows
            segs, info = self.model.transcribe(audio[i:i + 30 * 16000], beam_size=1, vad_filter=False,
                                               condition_on_previous_text=False)
            segs = list(segs)
            kept.extend(segs)
            langs.append((info.language, info.language_probability))
        lang, prob = max(langs, key=lambda x: x[1]) if langs else (None, 0.0)
        good = [x for x in kept if keep_segment(x)]
        total = sum(max(0.01, x.end - x.start) for x in good)
        self.last_logprob = (sum(x.avg_logprob * max(0.01, x.end - x.start) for x in good) / total) if good else None
        return clean_text(kept), lang, float(prob)


def analyse_text(text: str, channel: str) -> dict:
    """Brand suggestion + station-promo hint for a transcript (pure; used by the batch and tests)."""
    promo = detect_station_promo(text, channel)
    best = None if promo else suggest_brand(text)
    matches = extract_brands(text)
    return dict(content_hint=promo,
                suggested_brand=best.brand if best else None,
                suggested_industry=best.industry if best else None,
                brand_confidence=best.confidence if best else (matches[0].confidence if matches else None),
                brand_evidence=json.dumps([dict(brand=m.brand, confidence=m.confidence, evidence=m.evidence)
                                           for m in matches[:3]], ensure_ascii=False) if matches else None)


def save_result(db_path, asset_id: str, channel: str, text: str, language: str | None, model_name: str,
                avg_logprob: float | None = None) -> dict:
    res = analyse_text(text, channel)
    with get_db(db_path) as conn:
        conn.execute(
            "UPDATE discovered_assets SET transcript = ?, transcript_language = ?, transcription_model = ?, "
            "transcribed_at = ?, content_hint = ?, transcript_logprob = ? WHERE asset_id = ?",
            (text, language, model_name, datetime.now(timezone.utc).isoformat(timespec="seconds"),
             res["content_hint"], avg_logprob, asset_id))
        # speech replaces any suggestion it has an opinion on; with none, an on-screen (packshot) one is kept
        conn.execute(
            "UPDATE discovered_assets SET suggested_brand = ?, suggested_industry = ?, brand_confidence = ?, "
            "brand_evidence = ?, suggestion_source = ? WHERE asset_id = ? AND (? IS NOT NULL OR "
            "coalesce(suggestion_source, 'audio') = 'audio')",
            (res["suggested_brand"], res["suggested_industry"], res["brand_confidence"], res["brand_evidence"],
             "audio" if res["suggested_brand"] else None, asset_id, res["suggested_brand"]))
    return res


def reanalyse(db_path) -> list[dict]:
    """Re-score stored transcripts (and packshot OCR text) after the brand directory changed - no Whisper.

    Only suggestion fields of unclassified assets change; an OCR suggestion is kept unless speech now yields one.
    -> [{asset_id, before, after}] for assets whose suggestion changed.
    """
    from kihc.analytics.visual_ocr import analyse_ocr
    changed = []
    with get_db(db_path) as conn:
        rows = conn.execute("SELECT asset_id, channel, transcript, transcribed_at, ocr_text, suggested_brand, "
                            "suggestion_source FROM discovered_assets WHERE status = 'unclassified' "
                            "AND (transcribed_at IS NOT NULL OR ocr_text IS NOT NULL)").fetchall()
        for r in rows:
            res = analyse_text(r["transcript"] or "", r["channel"]) if r["transcribed_at"] else None
            source = "audio" if res and res["suggested_brand"] else None
            if not source and r["ocr_text"]:
                o = analyse_ocr(r["ocr_text"].split(" "))
                if o["suggested_brand"]:
                    res, source = dict(o, content_hint=res["content_hint"] if res else None), "packshot"
            res = res or dict(content_hint=None, suggested_brand=None, suggested_industry=None,
                              brand_confidence=None, brand_evidence=None)
            if res["suggested_brand"] != r["suggested_brand"]:
                changed.append(dict(asset_id=r["asset_id"], before=r["suggested_brand"], after=res["suggested_brand"]))
            conn.execute("UPDATE discovered_assets SET suggested_brand = ?, suggested_industry = ?, brand_confidence = ?, "
                         "brand_evidence = ?, content_hint = ?, suggestion_source = ? WHERE asset_id = ?",
                         (res["suggested_brand"], res["suggested_industry"], res["brand_confidence"],
                          res["brand_evidence"], res["content_hint"], source, r["asset_id"]))
    return changed


def pending_assets(db_path, asset_ids=None, retranscribe=False) -> list:
    q = ("SELECT asset_id, channel, audio_path, duration_sec, nominal_duration_sec FROM discovered_assets "
         "WHERE audio_path IS NOT NULL")
    args = []
    if asset_ids:
        q += f" AND asset_id IN ({','.join('?' * len(asset_ids))})"
        args = list(asset_ids)
    else:
        q += " AND status = 'unclassified'"
    if not retranscribe:
        q += " AND transcribed_at IS NULL"
    with get_db(db_path) as conn:
        return [dict(r) for r in conn.execute(q + " ORDER BY asset_id", args)]


def _memory_watchdog(cap_mb: float, interval: float = 0.5):
    def run():
        while True:
            rss = memory_mb()[0]
            if rss > cap_mb:
                print(f"transcriber: memory {rss:.0f} MB above cap {cap_mb:.0f} MB, exiting", flush=True)
                os._exit(EXIT_MEMORY_CAP)
            time.sleep(interval)
    threading.Thread(target=run, daemon=True, name="kihc-transcriber-memguard").start()


def _lower_priority():
    try:
        if sys.platform == "win32":
            import ctypes
            ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000)
        else:
            os.nice(10)
    except Exception:
        pass


def run_batch(db_path=None, model: str = DEFAULT_MODEL, asset_ids=None, retranscribe=False,
              transcriber_factory=None) -> list[dict]:
    db_path = str(db_path or DEFAULT_DB_PATH)
    todo = pending_assets(db_path, asset_ids, retranscribe)
    if not todo:
        return []
    tr = (transcriber_factory or Transcriber)(model)
    out = []
    for a in todo:
        t0 = time.perf_counter()
        try:
            spot = max(a["nominal_duration_sec"] or 0, a["duration_sec"] or 0) or None
            text, lang, prob = tr.transcribe(a["audio_path"], max_seconds=spot)
        except Exception as exc:                          # one bad clip must not stop the batch
            out.append(dict(asset_id=a["asset_id"], error=f"{type(exc).__name__}: {exc}"))
            continue
        res = save_result(db_path, a["asset_id"], a["channel"], text, lang, tr.name, getattr(tr, "last_logprob", None))
        out.append(dict(asset_id=a["asset_id"], language=lang, language_prob=round(prob, 2), text=text,
                        seconds=round(time.perf_counter() - t0, 1), **res))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=None)
    ap.add_argument("--model", default=DEFAULT_MODEL, help="tiny | base | small (int8, CPU)")
    ap.add_argument("--asset", action="append", dest="assets", help="asset id (repeatable); default: pending")
    ap.add_argument("--pending", action="store_true", help="all unclassified assets without a transcript")
    ap.add_argument("--retranscribe", action="store_true", help="redo assets that already have a transcript")
    ap.add_argument("--memory-cap-mb", type=float, default=MEMORY_CAP_MB)
    a = ap.parse_args(argv)
    _lower_priority()
    _memory_watchdog(a.memory_cap_mb)
    started = time.perf_counter()
    results = run_batch(a.db, a.model, a.assets, a.retranscribe)
    for r in results:
        if "error" in r:
            print(f"{r['asset_id']}: ERROR {r['error']}", flush=True)
            continue
        verdict = r["content_hint"] or (f"suggested brand {r['suggested_brand']} ({r['brand_confidence']:.2f})"
                                        if r["suggested_brand"] else "no brand identified")
        print(f"{r['asset_id']}: {r['language']} ({r['language_prob']:.2f}) {r['seconds']}s | {verdict} | "
              f"{r['text'][:120]!r}", flush=True)
    print(f"transcriber: {len(results)} asset(s) in {time.perf_counter() - started:.0f}s, "
          f"peak RSS {memory_mb()[1]:.0f} MB", flush=True)
    return 2 if any("error" in r for r in results) else 0     # non-zero -> the pilot scheduler backs off


if __name__ == "__main__":
    sys.exit(main())
