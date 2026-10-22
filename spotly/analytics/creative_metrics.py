"""Creative health of a discovered asset: four transparent components and their weighted mean.

These are Spotly heuristics computed from what the monitor actually observed (one packshot frame, the speech
transcript, and the breaks the asset aired in). They are not audience measurements and never use ratings
or impressions. A component that cannot be assessed is reported as None with the reason, and the total is
the weighted mean of the assessable components only (at least two are required).

  * Brand visual dominance - size of the brand's wordmark/logo on the packshot (6 % of the frame or more =
    full marks) and frame sharpness. Needs the brand to be identified on screen (text or logo library).
  * Audio-visual sync - is the brand both said and shown? said+shown 100, only one of the two 50,
    two different brands 0; no brand either way: share of on-screen words that are spoken. A music-only spot has nothing
    to align, so a closing packshot confirmed against the reference pHash (<= 8 bits) scores 95 (Sonic-Visual Lock).
  * Pod quality - per airing: alone in its break 100 (Solo Airing - Zero Clutter), first 85, last 75, middle 45; unconfirmed
    boundaries are scored from the contiguous ad density; -15 in breaks of 6+ spots.
  * Brand identification - 100 once a human classified it, else the suggestion's confidence; not scored
    while the advertiser is unknown.
"""
from collections import Counter

from spotly.analytics.brand_extractor import suggest_brand

WEIGHTS = dict(visual_dominance=0.30, av_sync=0.20, pod_quality=0.30, identification=0.20)
FULL_AREA = 0.06
FULL_SHARPNESS = 400.0
POSITION_SCORE = {"FIRST": 85, "LAST": 75, "MIDDLE": 45}
STANDARD_BRACKETS = (10, 15, 20, 30, 45, 60)


def visual_dominance(features: dict | None, visual_brand: str | None):
    if not features:
        return None, "no packshot yet"
    if features.get("programme_overlay"):
        return None, "packshot shows programme overlay"
    if not visual_brand or features.get("logo_basis") != "brand text" and not features.get("logo_match"):
        return None, "brand not identified on screen"
    size = min(1.0, (features.get("logo_area") or 0.0) / FULL_AREA)
    clarity = min(1.0, (features.get("sharpness") or 0.0) / FULL_SHARPNESS)
    return round(100 * (0.75 * size + 0.25 * clarity)), f"wordmark {features.get('logo_area', 0) * 100:.1f}% of frame"


SONIC_LOCK_SCORE = 95                # music-only spot whose closing packshot is confirmed against the reference
SONIC_LOCK_MAX_BITS = 8              # packshot pHash Hamming distance (of 64) that counts as the same packshot
LOCK_LABEL = "Sonic-Visual Lock"
SOLO_LABEL = "Solo Airing - Zero Clutter"


def _distinctive(text: str | None) -> set[str]:
    from spotly.analytics.brand_extractor import _tokens
    return {t for t in _tokens(text or "") if len(t) >= 3 and not t.isdigit()}


def token_alignment(transcript: str | None, ocr_text: str | None) -> int | None:
    """% of the distinctive on-screen tokens that are also spoken (RapidFuzz-tolerant); None when either side is empty."""
    from spotly.analytics.brand_extractor import fuzzy_ratio
    seen, said = _distinctive(ocr_text), _distinctive(transcript)
    if not seen or not said:
        return None
    hit = sum(1 for t in seen if t in said or any(fuzzy_ratio(t, w) > 85 for w in said))
    return round(100 * hit / len(seen))


def av_sync(transcript: str | None, transcribed: bool, audio_brand: str | None, visual_brand: str | None,
            packshot_distance: int | None = None, ocr_text: str | None = None):
    """Verbal spot: brand said+shown, else the spoken/on-screen token alignment. Music-only spot: no speech to align, so
    the closing packshot decides - confirmed against the reference pHash it is a 'Sonic-Visual Lock'."""
    speech = transcribed and bool((transcript or "").strip())
    if not speech:
        if transcribed and packshot_distance is not None and packshot_distance <= SONIC_LOCK_MAX_BITS:
            return SONIC_LOCK_SCORE, f"{LOCK_LABEL}: closing packshot matches the reference ({packshot_distance}/64 bits)"
        return None, "no speech to compare"
    if visual_brand is None and audio_brand is None:
        al = token_alignment(transcript, ocr_text)
        return (al, "spoken words vs on-screen text") if al is not None else (None, "brand neither heard nor read")
    if audio_brand and visual_brand:
        return (100, "brand said and shown") if audio_brand == visual_brand else (0, f"heard {audio_brand}, shown {visual_brand}")
    return 50, "brand shown, not said" if visual_brand else "brand said, not shown"


DENSITY_STEP = 11                    # points lost per additional contiguous ad when the position is not confirmed


def _placement_score(p: dict) -> int | None:
    ads = p.get("break_ads")
    if ads is not None and ads <= 1:
        return 100                                    # alone in its break: zero clutter whatever the boundary sensing says
    if p.get("position"):
        s = POSITION_SCORE.get(p["position"], 45)
    elif ads:                                         # boundaries unconfirmed: score the verified contiguous ad density
        s = max(45, 100 - DENSITY_STEP * (ads - 1))
    else:
        return None
    return s - 15 if (ads or 0) >= 6 else s


def pod_quality(placements: list[dict]):
    scores = [s for s in map(_placement_score, placements) if s is not None]
    if not scores:
        return None, "no break placement yet"
    solo = sum(1 for p in placements if p.get("break_ads") is not None and p["break_ads"] <= 1)
    note = f"{len(scores)} airing(s), {solo} solo"
    if solo == len(scores):
        note = SOLO_LABEL
    return round(sum(scores) / len(scores)), note


def packshot_distance(packshot_path, ref_hash: str | None) -> int | None:
    """Hamming distance (of 64) between the stored closing packshot and the reference pHash; None if either is unusable.
    One image is opened and released at a time."""
    if not packshot_path or not ref_hash:
        return None
    try:
        from pathlib import Path

        from PIL import Image

        from spotly.analytics.reference_matcher import hamming_hex, phash_image
        if not Path(packshot_path).is_file():
            return None
        with Image.open(packshot_path) as im:
            return hamming_hex(phash_image(im.convert("RGB")), ref_hash)
    except (OSError, ValueError):
        return None


def display_score(name: str, score, note: str | None = None) -> str:
    """The user-facing text of a component: never 'n/a' for a score that exists."""
    if score is None:
        return "not assessable"
    if name == "pod_quality" and note == SOLO_LABEL:
        return f"{score}% ({SOLO_LABEL})"
    if name == "av_sync" and note and note.startswith(LOCK_LABEL):
        return f"{score}% ({LOCK_LABEL})"
    return f"{score}%"


def identification(status: str, suggested_brand: str | None, confidence: float | None):
    if status == "classified":
        return 100, "confirmed by a reviewer"
    if suggested_brand:
        return round(100 * (confidence or 0)), "auto-suggested"
    # Not knowing the advertiser yet is not a creative weakness: not scored (a 0 here dragged every
    # unidentified asset's health down to roughly pod quality x 0.6 - measured on 2026-10-14).
    return None, "advertiser not identified yet"


def duration_label(nominal: float | None, measured: float | None) -> str:
    d = nominal or measured or 0
    std = any(abs(d - b) < 0.05 for b in STANDARD_BRACKETS)
    return f"{d:g}s {'Standard' if std else 'Non-standard'}"


def main_daypart(placements: list[dict]) -> str | None:
    c = Counter(p.get("daypart") for p in placements if p.get("daypart"))
    return c.most_common(1)[0][0] if c else None


def compute(asset: dict, features: dict | None, placements: list[dict], packshot_distance: int | None = None) -> dict:
    """asset: a list_assets() row. -> {components: {name: {score, note}}, health, assessed, label, daypart}."""
    audio_brand = None
    if asset.get("transcribed_at") and asset.get("transcript"):
        s = suggest_brand(asset["transcript"])
        audio_brand = s.brand if s else None
    visual_brand = None
    if asset.get("ocr_text"):
        s = suggest_brand(asset["ocr_text"])
        visual_brand = s.brand if s else None
    if not visual_brand and features and features.get("logo_match"):
        visual_brand = features["logo_match"]["brand"]
    parts = dict(visual_dominance=visual_dominance(features, visual_brand),
                 av_sync=av_sync(asset.get("transcript"), bool(asset.get("transcribed_at")), audio_brand, visual_brand,
                         packshot_distance, asset.get("ocr_text")),
                 pod_quality=pod_quality(placements),
                 identification=identification(asset.get("status"), asset.get("suggested_brand"), asset.get("brand_confidence")))
    have = {k: v[0] for k, v in parts.items() if v[0] is not None}
    health = None
    if len(have) >= 2:
        w = sum(WEIGHTS[k] for k in have)
        health = round(sum(WEIGHTS[k] * v for k, v in have.items()) / w)
    return dict(components={k: dict(score=v[0], note=v[1], display=display_score(k, v[0], v[1])) for k, v in parts.items()},
                health=health,
                assessed=f"{len(have)}/{len(parts)}", audio_brand=audio_brand, visual_brand=visual_brand,
                label=duration_label(asset.get("nominal_duration_sec"), asset.get("duration_sec")),
                daypart=main_daypart(placements))
