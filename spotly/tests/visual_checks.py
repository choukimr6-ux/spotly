"""OpenCV-dependent checks of packshot visual analysis (Phase 11.4).

Not collected directly (no test_ prefix): test_creative_intel.test_visual_checks_in_isolated_process runs this
file in a child pytest process, exactly as production runs visual analysis in the OCR child process, so OpenCV
never inflates the main test process (whose peak memory test_real_ingest audits).
"""
import cv2
import numpy as np
import pytest

from spotly.analytics import creative_visual as cv
from spotly.analytics import visual_ocr as vo
from spotly.database.schema import get_db

CH = "El-Heddaf TV"


def _logo(seed=1, size=(220, 120)):
    rng = np.random.default_rng(seed)
    img = np.full((size[1], size[0], 3), 255, np.uint8)
    for _ in range(14):                                                  # distinctive artwork with corners
        x, y = int(rng.integers(5, size[0] - 40)), int(rng.integers(5, size[1] - 30))
        color = tuple(int(c) for c in rng.integers(0, 200, 3))
        cv2.rectangle(img, (x, y), (x + int(rng.integers(10, 40)), y + int(rng.integers(8, 25))), color, -1)
    cv2.putText(img, "ACME", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (20, 40, 160), 4)
    return img


def _packshot(logo=None, at=(500, 260), scale=1.6, seed=9):
    rng = np.random.default_rng(seed)
    img = (rng.random((720, 1280, 3)) * 40 + 60).astype(np.uint8)       # textured background
    if logo is not None:
        lg = cv2.resize(logo, None, fx=scale, fy=scale)
        img[at[1]:at[1] + lg.shape[0], at[0]:at[0] + lg.shape[1]] = lg
    return img


def test_logo_library_matches_scaled_artwork_and_rejects_others(tmp_path):
    (tmp_path / "Acme").mkdir()
    cv2.imwrite(str(tmp_path / "Acme" / "logo.png"), _logo())
    lib = cv.LogoLibrary(tmp_path)
    hit = lib.match(_packshot(_logo()))
    assert hit and hit[0] == "Acme" and hit[2] >= cv.LogoLibrary.MIN_INLIERS
    assert lib.match(_packshot(_logo(seed=2), seed=3)) is None
    assert lib.match(_packshot(None)) is None


def test_analyse_frame_dominance_hash_and_overlays():
    img = _packshot(_logo())
    box = [[500, 260], [852, 260], [852, 452], [500, 452]]
    f = cv.analyse_frame(img, [(box, "ACME", 0.95)], [box])
    assert f["logo_basis"] == "brand text" and f["logo_area"] == pytest.approx(352 * 192 / (1280 * 720), rel=0.01)
    assert not f["programme_overlay"] and f["sharpness"] > 0 and len(f["dhash"]) == 16
    same = cv.analyse_frame(img.copy(), [], [])
    assert cv.hamming(int(f["dhash"], 16), int(same["dhash"], 16)) == 0
    other = cv.analyse_frame(_packshot(_logo(seed=5), at=(100, 100), seed=4), [], [])
    assert cv.hamming(int(f["dhash"], 16), int(other["dhash"], 16)) > vo.SAME_PACKSHOT_BITS
    assert same["logo_basis"] == "salient region"                      # no brand text: best visual cluster


def test_learn_logos_from_classified_packshots(tmp_path, monkeypatch):
    db = tmp_path / "l.db"
    shot = tmp_path / "ELHEDDAF_AD_001_packshot.jpg"
    img = _packshot(_logo())
    cv2.imwrite(str(shot), img)
    with get_db(db) as conn:
        conn.execute("INSERT INTO reference_ads (ad_id, brand_name, campaign_name, duration_sec, fingerprint, fp_algorithm) "
                     "VALUES (3, 'Acme', 'x', 30, x'00', 't')")
        conn.execute("INSERT INTO discovered_assets (asset_id, channel, duration_sec, fingerprint, status, packshot_path, "
                     "promoted_ad_id) VALUES ('ELHEDDAF_AD_001', ?, 30, x'01000000', 'classified', ?, 3)", (CH, str(shot)))
    word = [[530, 330], [800, 330], [800, 420], [530, 420]]
    monkeypatch.setattr(vo, "ocr_frame", lambda engine, jpeg: (img, [(word, "ACME", 0.95)], ["ACME"]))
    assert vo.learn_logos(db, object(), tmp_path / "logos") == 1
    learned = tmp_path / "logos" / "_learned" / "Acme" / "ELHEDDAF_AD_001.jpg"
    assert learned.is_file()
    assert vo.learn_logos(db, object(), tmp_path / "logos") == 0                     # once per asset
    (tmp_path / "logos" / "Other").mkdir()                                            # a user-supplied folder too
    cv2.imwrite(str(tmp_path / "logos" / "Other" / "o.png"), _logo(seed=8))
    lib = cv.LogoLibrary(tmp_path / "logos")                                          # learned refs: brand, not "_learned"
    assert sorted({r.brand for r in lib.refs}) == ["Acme", "Other"]
    assert lib.match(_packshot(_logo(), at=(300, 200), scale=1.3, seed=21))[0] == "Acme"
