"""On-screen text and visuals of captured packshots -> brand suggestion (fallback when speech gave none).

    python -m kihc.analytics.visual_ocr --db kihc/data/kihc.db      # every unclassified asset with an un-read packshot

Runs as its own short-lived, low-priority process (started by the pilot when work is pending), never inside the
monitoring process; a watchdog exits at the memory cap (default 300 MB).

Recogniser (Phase 11.4): PaddlePaddle's arabic_PP-OCRv5_mobile_rec (Apache-2.0, 8 MB ONNX, in
kihc/data/models/ocr-arabic) replaces rapidocr's Latin/Chinese one when present. Measured on El-Heddaf packshots
it reads Arabic (in visual order: reversed here) *and* reads Latin better than the default ("Messäd" vs
"Messaid", "DAYZY" vs "DAZY", "VANILL CREAM" vs "VANBLL CREN"); peak RSS ~280 MB. The detector still misses
some large stylised Arabic headlines; logos without text need the logo library (creative_visual.LogoLibrary).

Frame regions: the channel bug (top-right) would make every frame look like station self-promo, so OCR text is
never used for promo detection and that corner is ignored. The bottom band carries El-Heddaf's news/score
ticker (which can mention sponsors, e.g. 'Ligue 1 Mobilis'): text there only counts if it is a URL/domain, and a
ticker *clock* there marks the frame as programme content (creative_visual.programme_overlay).
"""
import argparse
import gc
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

from kihc.analytics.brand_extractor import _tokens, extract_brands, sanitize_ocr, suggest_brand
from kihc.analytics.creative_visual import (LOGO_DIR_NAME, analyse_frame, hamming, programme_text,
                                            visual_to_logical)
from kihc.analytics.transcriber import _lower_priority, _memory_watchdog
from kihc.config import DATA_DIR, DEFAULT_DB_PATH
from kihc.database.schema import get_db
from kihc.service.sysinfo import memory_mb

MEMORY_CAP_MB = 300
MIN_BOX_CONF = 0.6
TICKER_TOP = 0.78                  # boxes centred below this fraction of the height are ticker/score bar
BUG_LEFT, BUG_BOTTOM = 0.82, 0.22  # top-right channel logo region
ARABIC_REC = DATA_DIR / "models" / "ocr-arabic"
LOGO_ROOT = Path(os.environ.get("KIHC_LOGO_DIR") or DATA_DIR / LOGO_DIR_NAME)   # <Brand>/*.png + _learned/
SAME_PACKSHOT_BITS = 6             # dHash distance for "identical end frame" (64 bits)
PROGRAMME_HINT = "likely programme content (channel ticker on screen)"
_DOMAIN = re.compile(r"(?:www\.)?([a-z0-9][a-z0-9-]{2,})\.(?:dz|com|net|org|app)\b", re.I)
_NOISE = re.compile(r"^(?:\d{1,2}[:.h]\d{2}|\d+\s*-\s*\d+|gmt.*|\W+)$", re.I)     # clocks, scores, punctuation
_CAMEL_SPLIT = re.compile(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Za-z])(?=[0-9])|(?<=[0-9])(?=[A-Za-z])")   # OOREDOO4G


DET_MAX_SIDE = int(os.environ.get("KIHC_OCR_DET_MAX_SIDE", "960"))


def make_engine():
    """rapidocr detector + the Arabic/Latin PP-OCRv5 recogniser when its model is installed.

    Memory (measured on El-Heddaf packshots): rapidocr's defaults upscale a 720p frame to 1308x736 for detection
    ('min' side 736) and run a rotation classifier; with the Arabic recogniser that peaked at 298-339 MB. Packshot
    text is upright and large, so detection runs on at most DET_MAX_SIDE px and the classifier is off."""
    from rapidocr_onnxruntime import RapidOCR
    eng = RapidOCR(intra_op_num_threads=1, inter_op_num_threads=1, use_angle_cls=False,
                   det_limit_type="max", det_limit_side_len=DET_MAX_SIDE, det_use_cuda=False, det_model_path=None)
    eng.kihc_recognizer = "rapidocr default (Latin)"
    model = ARABIC_REC / "inference.onnx"
    if model.is_file() and (ARABIC_REC / "inference.yml").is_file():
        import yaml
        from rapidocr_onnxruntime.ch_ppocr_v3_rec import TextRecognizer
        chars = yaml.safe_load((ARABIC_REC / "inference.yml").read_text("utf-8"))["PostProcess"]["character_dict"]
        eng.text_recognizer = TextRecognizer(dict(model_path=str(model), keys_path=list(chars), rec_batch_num=6,
                                                  rec_img_shape=[3, 48, 320], use_cuda=False))
        eng.kihc_recognizer = "PP-OCRv5 arabic mobile (Arabic + Latin)"
        gc.collect()                                     # drop the default recogniser now
    return eng


def keep_box(box, text: str, conf: float, width: int, height: int) -> bool:
    if conf < MIN_BOX_CONF or len(text.strip()) < 2 or _NOISE.match(text.strip()):
        return False
    xs, ys = [p[0] for p in box], [p[1] for p in box]
    cx, cy = sum(xs) / 4 / width, sum(ys) / 4 / height
    if cx > BUG_LEFT and cy < BUG_BOTTOM:
        return False                                    # channel bug
    if cy > TICKER_TOP:
        return bool(_DOMAIN.search(text))               # ticker band: URLs only
    return True


def analyse_ocr(lines: list[str]) -> dict:
    """Brand suggestion from OCR lines (pure; tests use it directly)."""
    text = " ".join(lines)
    domains = [m.group(1) for m in _DOMAIN.finditer(text)]
    expanded = " ".join([text, " ".join(_CAMEL_SPLIT.sub(" ", t) for t in lines), " ".join(domains)])
    best = suggest_brand(expanded)
    display_name = sanitize_ocr(text)
    matches = extract_brands(expanded)
    return dict(ocr_text=text, domains=domains, display_name=display_name,
                suggested_brand=best.brand if best else None,
                suggested_industry=best.industry if best else None,
                brand_confidence=best.confidence if best else (matches[0].confidence if matches else None),
                brand_evidence=json.dumps([dict(brand=m.brand, confidence=m.confidence, evidence=m.evidence,
                                                product=m.product, source="packshot") for m in matches[:3]],
                                          ensure_ascii=False) if matches else None)


def ocr_frame(engine, jpeg: bytes):
    """-> (decoded image, raw boxes [(box, text in reading order, conf)], kept lines)."""
    import cv2
    import numpy as np
    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("packshot is not a decodable image")
    h, w = img.shape[:2]
    res, _ = engine(img)
    raw = [(box, visual_to_logical(text), float(conf)) for box, text, conf in res or []]
    kept = [t.strip() for box, t, c in raw if keep_box(box, t, c, w, h)]
    return img, raw, kept


def ocr_lines(engine, jpeg: bytes) -> list[str]:
    return ocr_frame(engine, jpeg)[2]


def _brand_boxes(raw, res: dict, width: int, height: int) -> list:
    """Boxes holding the matched brand's wordmark (for the logo crop / visual dominance)."""
    if not res.get("brand_evidence") or not res.get("suggested_brand"):
        return []
    ev = next((e for e in json.loads(res["brand_evidence"]) if e["brand"] == res["suggested_brand"]), None)
    # distinctive tokens of the matched aliases: a multi-word name ("flux care") may span several OCR boxes
    want = {t for a, *_ in (ev or {}).get("evidence", []) for t in _tokens(a) if len(t) >= 3}
    want |= {t for t in _tokens(res["suggested_brand"]) if len(t) >= 3}
    out = []
    for box, text, conf in raw:
        toks = set(_tokens(text)) | set(_tokens(_CAMEL_SPLIT.sub(" ", text)))   # "RicamaR" and "OOREDOO 4G"
        if keep_box(box, text, conf, width, height) and toks & want:
            out.append(box)
    return out


def save_logo_crop(img, box, path: Path, pad: float = 0.08) -> str | None:
    import cv2
    if not box:
        return None
    h, w = img.shape[:2]
    x0, y0, x1, y1 = box
    px, py = int((x1 - x0) * pad) + 4, int((y1 - y0) * pad) + 4
    crop = img[max(0, y0 - py):min(h, y1 + py), max(0, x0 - px):min(w, x1 + px)]
    if crop.size == 0:
        return None
    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        return None
    path.write_bytes(buf.tobytes())
    return str(path)


def save_result(db_path, asset_id: str, res: dict, features: dict | None = None, logo_path: str | None = None) -> None:
    """Store OCR text + visual features. A text or logo brand becomes the asset's suggestion only if nothing
    (speech) suggested one already; a ticker clock marks the asset as likely programme content."""
    features = features or {}
    logo = features.get("logo_match")
    brand, source = res["suggested_brand"], "packshot"
    industry, conf, evidence = res["suggested_industry"], res["brand_confidence"], res["brand_evidence"]
    if not brand and logo:
        from kihc.analytics.brand_extractor import BRANDS
        b = next((x for x in BRANDS if x.name == logo["brand"]), None)
        brand, source, conf = logo["brand"], "logo", logo["confidence"]
        industry = b.industry if b else None
        evidence = json.dumps([dict(brand=brand, confidence=conf, evidence=[["logo", f"ORB {logo['inliers']} inliers", 1]],
                                    product=b.product if b else "", source="logo")])
    status = ("programme" if features.get("programme_overlay") else
              "ocr_brand" if res["suggested_brand"] else "logo_match" if brand else "ocr_no_brand")
    with get_db(db_path) as conn:
        conn.execute("UPDATE discovered_assets SET ocr_text = ?, ocr_at = ?, visual_status = ?, visual_features = ?, "
                     "logo_path = coalesce(?, logo_path) WHERE asset_id = ?",
                     (res["ocr_text"], datetime.now().astimezone().isoformat(timespec="seconds"), status,
                      json.dumps(features) if features else None, logo_path, asset_id))
        if brand:
            conn.execute("UPDATE discovered_assets SET suggested_brand = ?, suggested_industry = ?, brand_confidence = ?, "
                         "brand_evidence = ?, suggestion_source = ? WHERE asset_id = ? AND status = 'unclassified' "
                         "AND (suggested_brand IS NULL OR suggestion_source IN ('packshot', 'logo'))",
                         (brand, industry, conf, evidence, source, asset_id))
        if features.get("programme_overlay"):
            conn.execute("UPDATE discovered_assets SET content_hint = ? WHERE asset_id = ? AND content_hint IS NULL",
                         (PROGRAMME_HINT, asset_id))
            # Permanent rejection: an 'ignored' asset stays in the catalogue as a tombstone, so discovery skips every
            # later airing of this audio (DiscoveryEngine._record) and it never returns to the review card.
            conn.execute("UPDATE discovered_assets SET status = 'ignored', label = ? "
                         "WHERE asset_id = ? AND status = 'unclassified'",
                         (f"rejected: programme content ({features.get('programme_reason') or 'ticker clock'})", asset_id))


def pending(db_path, retry=False) -> list[dict]:
    q = ("SELECT asset_id, packshot_path FROM discovered_assets WHERE status = 'unclassified' "
         "AND packshot_path IS NOT NULL" + ("" if retry else " AND ocr_at IS NULL") + " ORDER BY asset_id")
    with get_db(db_path) as conn:
        return [dict(r) for r in conn.execute(q)]


def learn_logos(db_path, engine, logo_root: Path = LOGO_ROOT) -> int:
    """Reference images from packshots of assets a human classified: logos/_learned/<Brand>/<asset>.jpg.
    Uses the wordmark crop when it is large enough, else the packshot's central region."""
    with get_db(db_path) as conn:
        rows = conn.execute("SELECT a.asset_id, a.packshot_path, r.brand_name FROM discovered_assets a "
                            "JOIN reference_ads r ON r.ad_id = a.promoted_ad_id WHERE a.status = 'classified' "
                            "AND a.packshot_path IS NOT NULL").fetchall()
    added = 0
    if not rows:
        return 0
    import cv2
    for r in rows:
        dest = Path(logo_root) / "_learned" / r["brand_name"] / f"{r['asset_id']}.jpg"
        if dest.exists() or not Path(r["packshot_path"]).is_file():
            continue
        img, raw, _ = ocr_frame(engine, Path(r["packshot_path"]).read_bytes())
        h, w = img.shape[:2]
        name = " ".join(_tokens(r["brand_name"]))
        boxes = [b for b, t, c in raw if keep_box(b, t, c, w, h) and name and name in " ".join(_tokens(t))]
        dest.parent.mkdir(parents=True, exist_ok=True)
        crop = None
        if boxes:
            xs = [p[0] for b in boxes for p in b]
            ys = [p[1] for b in boxes for p in b]
            x0, y0, x1, y1 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
            if (x1 - x0) * (y1 - y0) >= 0.01 * w * h:
                crop = img[max(0, y0 - 12):y1 + 12, max(0, x0 - 12):x1 + 12]
        if crop is None:
            crop = img[int(0.1 * h):int(0.76 * h), int(0.13 * w):int(0.82 * w)]
        ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if ok:
            dest.write_bytes(buf.tobytes())
            added += 1
        del img, crop
    return added


def cluster_packshots(db_path) -> list[tuple]:
    """Assets whose end frames are the same image (dHash within SAME_PACKSHOT_BITS): typically cut-downs of one
    campaign. A human-classified asset's brand is offered to the others as a suggestion."""
    with get_db(db_path) as conn:
        rows = [dict(r) for r in conn.execute(
            "SELECT a.asset_id, a.status, a.visual_features, a.suggested_brand, r.brand_name, r.industry "
            "FROM discovered_assets a LEFT JOIN reference_ads r ON r.ad_id = a.promoted_ad_id "
            "WHERE a.visual_features IS NOT NULL AND a.status IN ('unclassified', 'classified')")]
        feats = {r["asset_id"]: json.loads(r["visual_features"]) for r in rows}
        usable = [r for r in rows if not feats[r["asset_id"]].get("programme_overlay")
                  and feats[r["asset_id"]].get("sharpness", 0) >= 50]
        pairs = []
        for i, a in enumerate(usable):
            for b in usable[i + 1:]:
                d = hamming(int(feats[a["asset_id"]]["dhash"], 16), int(feats[b["asset_id"]]["dhash"], 16))
                if d <= SAME_PACKSHOT_BITS:
                    pairs.append((a["asset_id"], b["asset_id"], d))
        for r in rows:
            f = feats[r["asset_id"]]
            f["same_packshot_as"] = sorted({y if x == r["asset_id"] else x for x, y, _ in pairs if r["asset_id"] in (x, y)})
            conn.execute("UPDATE discovered_assets SET visual_features = ? WHERE asset_id = ?", (json.dumps(f), r["asset_id"]))
        by_id = {r["asset_id"]: r for r in rows}
        for x, y, d in pairs:
            for src, dst in ((x, y), (y, x)):
                s, t = by_id[src], by_id[dst]
                if s["status"] == "classified" and s["brand_name"] and t["status"] == "unclassified" and not t["suggested_brand"]:
                    ev = json.dumps([dict(brand=s["brand_name"], confidence=0.85, source="same packshot",
                                          evidence=[[f"same end frame as {src}", f"dHash distance {d}", 1]])])
                    conn.execute("UPDATE discovered_assets SET suggested_brand = ?, suggested_industry = ?, "
                                 "brand_confidence = 0.85, brand_evidence = ?, suggestion_source = 'same packshot' "
                                 "WHERE asset_id = ? AND suggested_brand IS NULL", (s["brand_name"], s["industry"], ev, dst))
    return pairs


def run_batch(db_path=None, retry=False, engine_factory=None, logo_root: Path | None = None) -> list[dict]:
    from kihc.analytics.creative_visual import LogoLibrary
    db_path = str(db_path or DEFAULT_DB_PATH)
    todo = pending(db_path, retry)
    if not todo:
        return []
    engine = engine_factory() if engine_factory else make_engine()
    logo_root = Path(logo_root or LOGO_ROOT)
    try:
        learn_logos(db_path, engine, logo_root)
    except Exception as exc:                            # learning is best-effort; never block the batch
        print(f"logo learning skipped: {exc}", flush=True)
    has_refs = logo_root.is_dir() and any(p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")
                                          for p in logo_root.rglob("*"))
    library = LogoLibrary(logo_root) if has_refs else None
    out = []
    for a in todo:
        t0 = time.perf_counter()
        try:
            img, raw, lines = ocr_frame(engine, Path(a["packshot_path"]).read_bytes())
            res = analyse_ocr(lines)
            h, w = img.shape[:2]
            features = analyse_frame(img, raw, _brand_boxes(raw, res, w, h), library)
            why = programme_text(raw, w, h)
            if why:                                     # sports ticker / match clock / programme title
                features["programme_overlay"], features["programme_reason"] = True, why
            logo_path = None
            if features["logo_box"] and not features["programme_overlay"]:
                logo_path = save_logo_crop(img, features["logo_box"],
                                           Path(a["packshot_path"]).with_name(f"{a['asset_id']}_logo.jpg"))
            del img
        except Exception as exc:
            out.append(dict(asset_id=a["asset_id"], error=f"{type(exc).__name__}: {exc}"))
            continue
        save_result(db_path, a["asset_id"], res, features, logo_path)
        out.append(dict(asset_id=a["asset_id"], seconds=round(time.perf_counter() - t0, 1), features=features, **res))
        gc.collect()
    cluster_packshots(db_path)
    return out


def mark_visual_needed(db_path) -> int:
    """Unclassified assets with no brand from speech and no packshot yet: visual capture needed at next airing."""
    with get_db(db_path) as conn:
        return conn.execute("UPDATE discovered_assets SET visual_status = 'needed' WHERE status = 'unclassified' "
                            "AND suggested_brand IS NULL AND content_hint IS NULL AND packshot_path IS NULL "
                            "AND (visual_status IS NULL OR visual_status = 'needed')").rowcount


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=None)
    ap.add_argument("--retry", action="store_true", help="re-read packshots that were already OCR'd")
    ap.add_argument("--memory-cap-mb", type=float, default=MEMORY_CAP_MB)
    a = ap.parse_args(argv)
    _lower_priority()
    _memory_watchdog(a.memory_cap_mb)
    started = time.perf_counter()
    results = run_batch(a.db, a.retry)
    for r in results:
        if "error" in r:
            print(f"{r['asset_id']}: ERROR {r['error']}", flush=True)
            continue
        verdict = f"brand {r['suggested_brand']} ({r['brand_confidence']:.2f})" if r["suggested_brand"] else "no brand"
        print(f"{r['asset_id']}: {r['seconds']}s | {verdict} | {r['ocr_text'][:120]!r}", flush=True)
    print(f"visual_ocr: {len(results)} packshot(s) in {time.perf_counter() - started:.0f}s, "
          f"peak RSS {memory_mb()[1]:.0f} MB", flush=True)
    return 2 if any("error" in r for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
