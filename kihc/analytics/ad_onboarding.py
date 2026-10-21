"""Phase 16: register a master ad creative BEFORE it airs (no cold start).

    register_ad("spot.mp4", "Djezzy", "Telecom", separation_window=180)

Pipeline
  1. FFmpeg extracts the audio track (-vn, mono PCM; 16 kHz for the library fingerprint, 11.025 kHz for the
     constellation) - the source file is never modified.
  2. Digital silence below -50 dBFS is trimmed from both ends of the master before the peak constellation is
     computed; the removed lead is stored (`lead_trim_s`) so the matcher puts the airing start where the creative's
     first frame was on air.
  3. Packshot: the frame at (duration - 1.5 s) is sampled and its visual entropy checked; a fade-to-black or
     near-flat frame is rejected and earlier candidates are tried, so the stored pHash is of the real end card.
  4. Everything is persisted in one go: reference ad (legacy fingerprint), reference_signatures row (constellation +
     pHash + ad_id link) and the SLA clause (separation window, booked position, optional DZD rate) in the
     contract list. The running monitor picks the new signature up through the copy-on-write registry reload.
"""
import io
import logging
import re
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np

from kihc.analytics import reference_matcher as rm
from kihc.analytics.brand_extractor import compute_voice_ratio, is_music_dominant
from kihc.core.audio import ffmpeg_exe
from kihc.data import reference_db as rdb
from kihc.database.schema import get_db

log = logging.getLogger("kihc.onboarding")

ALLOWED_EXT = (".mp4", ".wav", ".mp3")
SILENCE_FLOOR_DBFS = -50.0
PACKSHOT_SECONDS_BEFORE_END = 1.5
PACKSHOT_FALLBACK_STEP_S = 0.5
PACKSHOT_FALLBACK_SPAN_S = 6.0
MIN_ENTROPY_BITS = 2.0              # Shannon entropy of the 8-bit luma histogram; black / flat frames sit near 0
MIN_MEAN_LUMA = 20.0                # a frame still fading to black has a low mean
MIN_HASHES = 60
POSITIONS = ("FIRST", "LAST", "FIRST_OR_LAST", "ANY")
DEFAULT_SEPARATION_S = 180


class OnboardingError(ValueError):
    pass


def _run(cmd: list, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, **kw)


def probe(path) -> dict:
    """{duration, has_video, has_audio} from ffmpeg's banner (no ffprobe dependency)."""
    p = _run([ffmpeg_exe(), "-nostdin", "-hide_banner", "-i", str(path)], text=True, errors="replace")
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", p.stderr)
    dur = int(m[1]) * 3600 + int(m[2]) * 60 + float(m[3]) if m else 0.0
    return dict(duration=dur, has_video=bool(re.search(r"Stream #.*Video:(?!.*attached pic)", p.stderr)),
                has_audio="Audio:" in p.stderr)


def _read_wav(path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0


def extract_audio(path, workdir: Path) -> tuple[np.ndarray, np.ndarray, bytes]:
    """-> (pcm 11025 Hz float, pcm 16 kHz float, 16 kHz wav bytes). Mono 16-bit PCM, video stripped."""
    out16, out11 = Path(workdir) / "a16.wav", Path(workdir) / "a11.wav"
    p = _run([ffmpeg_exe(), "-nostdin", "-v", "error", "-y", "-i", str(path),
              "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(out16),
              "-vn", "-ac", "1", "-ar", str(rm.SR), "-c:a", "pcm_s16le", str(out11)])
    if p.returncode != 0 or not out16.exists():
        raise OnboardingError("FFmpeg could not decode an audio track: " + p.stderr.decode(errors="replace")[-300:])
    return _read_wav(out11), _read_wav(out16), out16.read_bytes()


def frame_quality(img) -> tuple[float, float]:
    """(entropy bits, mean luma) of a PIL image."""
    g = np.asarray(img.convert("L"))
    hist = np.bincount(g.ravel(), minlength=256).astype(np.float64)
    p = hist[hist > 0] / hist.sum()
    return float(-(p * np.log2(p)).sum()), float(g.mean())


def _frame_at(path, t: float):
    from PIL import Image
    p = _run([ffmpeg_exe(), "-nostdin", "-v", "error", "-ss", f"{max(0.0, t):.3f}", "-i", str(path),
              "-frames:v", "1", "-f", "image2pipe", "-c:v", "png", "-"])
    if p.returncode != 0 or not p.stdout:
        return None
    return Image.open(io.BytesIO(p.stdout)).convert("RGB")


def pick_packshot(path, duration: float) -> dict | None:
    """Frame at (duration - 1.5 s) if it has real content, else the nearest earlier one that does.
    -> {at, entropy, luma, phash, jpeg-less image} or None when no sampled frame passes (never hashes a black card)."""
    t = duration - PACKSHOT_SECONDS_BEFORE_END
    tried = []
    while t >= max(0.0, duration - PACKSHOT_SECONDS_BEFORE_END - PACKSHOT_FALLBACK_SPAN_S):
        img = _frame_at(path, t)
        if img is not None:
            ent, luma = frame_quality(img)
            tried.append((round(t, 2), round(ent, 2), round(luma, 1)))
            if ent >= MIN_ENTROPY_BITS and luma >= MIN_MEAN_LUMA:
                return dict(at=round(t, 3), entropy=round(ent, 3), luma=round(luma, 1), phash=rm.phash_image(img),
                            tried=tried)
        t -= PACKSHOT_FALLBACK_STEP_S
    log.warning("no usable packshot frame in %s (tried %s)", path, tried)
    return None


def normalise_peak(x: np.ndarray, target: float = 0.89) -> np.ndarray:
    peak = float(np.abs(x).max()) if x.size else 0.0
    return x * (target / peak) if peak > 1e-4 else x


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_").upper()[:40] or "AD"


def register_ad(file, brand: str, sector: str, *, campaign: str | None = None, separation_window: float | None = None,
                position: str | None = None, rate_override: float | None = None, db_path=None,
                allow_duplicate: bool = False, original_name: str | None = None,
                parent_id: str | None = None, cut_type: str | None = None, music_dominant: bool | None = None) -> dict:
    """Index a master creative end to end. Raises OnboardingError with a human-readable reason on bad input.

    Phase 21 campaign families: `parent_id` (the master's asset_id) registers this file as a cut-down / teaser / tagged
    edit of that master. It is indexed as its own signature (so the matcher can tell 15 s from 30 s) but rolls up to
    the parent campaign and inherits its brand, sector and contract clause."""
    cut_type = (cut_type or ("CUT_DOWN" if parent_id else "MASTER")).upper()
    if cut_type not in rdb.CUT_TYPES:
        raise OnboardingError(f"cut type must be one of {', '.join(rdb.CUT_TYPES)}")
    if parent_id and cut_type == "MASTER":
        raise OnboardingError("a MASTER cannot have a parent; use CUT_DOWN, TEASER or TAGGED with --parent-id")
    if cut_type != "MASTER" and not parent_id:
        raise OnboardingError(f"cut type {cut_type} needs --parent-id <master asset id>")
    parent = None
    if parent_id:
        with get_db(db_path) as conn:
            rdb.ensure_schema(conn)
            parent = conn.execute("SELECT s.asset_id, s.brand_name, s.ad_id, a.campaign_name FROM reference_signatures s "
                                  "LEFT JOIN reference_ads a ON a.ad_id = s.ad_id WHERE s.asset_id = ?",
                                  (parent_id,)).fetchone()
        if parent is None:
            raise OnboardingError(f"parent asset {parent_id!r} is not registered; register the master first")
    path = Path(file)
    brand, sector = " ".join((brand or "").split()), " ".join((sector or "").split())
    if not brand or not sector:
        raise OnboardingError("brand and sector are required")
    if (original_name and Path(original_name).suffix.lower() or path.suffix.lower()) not in ALLOWED_EXT:
        raise OnboardingError(f"unsupported file type; use one of {', '.join(ALLOWED_EXT)}")
    if not path.is_file():
        raise OnboardingError(f"file not found: {path}")
    position = (position or "ANY").upper()
    if position not in POSITIONS:
        raise OnboardingError(f"position must be one of {', '.join(POSITIONS)}")
    sep = DEFAULT_SEPARATION_S if separation_window is None else float(separation_window)
    if not 0 < sep <= 3600:
        raise OnboardingError("separation window must be in (0, 3600] seconds")
    if rate_override is not None and float(rate_override) < 0:
        raise OnboardingError("rate override must not be negative")
    info = probe(path)
    if not info["has_audio"]:
        raise OnboardingError("the file has no audio track; the acoustic matcher needs one")
    if parent is not None and " ".join((brand or "").split()).casefold() != (parent["brand_name"] or "").casefold():
        raise OnboardingError(f"brand {brand!r} differs from the parent's ({parent['brand_name']!r})")
    campaign = campaign or (f"{parent['campaign_name'] or parent_id} [{cut_type}]" if parent
                            else f"Master {Path(original_name or path.name).stem}")[:160]

    with tempfile.TemporaryDirectory(prefix="kihc_master_") as tmp:
        x11, _x16, wav16 = extract_audio(path, Path(tmp))
        duration = x11.size / rm.SR
        codes, t_cs, lead, trimmed = rm.signature_from_audio(normalise_peak(x11), floor_dbfs=SILENCE_FLOOR_DBFS)
        voice_ratio = compute_voice_ratio(x11)
        dominant = is_music_dominant(voice_ratio) if music_dominant is None else bool(music_dominant)
        if codes.size < MIN_HASHES:
            raise OnboardingError(f"only {codes.size} acoustic landmarks after trimming silence "
                                  f"({trimmed:.1f}s of sound): too sparse or short to match exactly")
        shot = pick_packshot(path, info["duration"] or duration) if info["has_video"] else None

        from kihc.core.library import ingest_media
        # a declared edit of its parent is expected to share audio with it: the duplicate guard is for accidents
        res = ingest_media(path, brand, campaign, db_path=db_path, industry=sector,
                           allow_duplicate=allow_duplicate or parent is not None)
    if not res.committed:
        raise OnboardingError("; ".join(res.warnings) or "could not index the creative")
    ad_id = res.ad_id
    asset_id = f"MASTER_{ad_id:04d}_{_slug(brand)}"

    with get_db(db_path) as conn:                                # WAL; one transaction for signature + link
        rdb.ensure_schema(conn)
        rdb.upsert_signature(conn, asset_id, brand, sector, float(duration), rdb.pack_blob(codes, t_cs, lead, trimmed),
                             shot["phash"] if shot else None, ad_id=ad_id,
                             master_asset_id=parent_id, cut_type=cut_type, voice_ratio=voice_ratio,
                             music_dominant=dominant)

    from kihc.analytics import compliance_audit as ca
    clause = dict(brand=brand, campaign=campaign, booked_position=position, competitor_separation_seconds=sep)
    if rate_override is not None:
        clause["rate_override_dzd"] = float(rate_override)
    contracts = [c for c in ca.get_contracts(db_path) if not (c.get("brand") == brand and c.get("campaign") == campaign)]
    if parent is not None:                  # the edit airs under the parent's booking: same position / separation / rate
        pc = next((c for c in contracts if c.get("brand") == brand and c.get("campaign") == parent["campaign_name"]), None)
        if pc:
            clause = {**pc, "campaign": campaign}
    ca.save_contracts(db_path, contracts + [clause])

    return dict(asset_id=asset_id, ad_id=ad_id, brand=brand, sector=sector, campaign=campaign,
                duration_s=round(duration, 2), trimmed_s=round(trimmed, 2), lead_trim_s=round(lead, 3),
                tail_trim_s=round(max(0.0, duration - lead - trimmed), 3), hashes=int(codes.size),
                packshot=shot, contract=clause, parent_id=parent_id, cut_type=cut_type,
                voice_ratio=voice_ratio, music_dominant=dominant)
