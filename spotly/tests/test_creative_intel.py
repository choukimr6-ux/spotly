"""Phase 11.4: phonetic Darija brand matching, Arabic OCR order, packshot visual analysis + logo library,
creative health metrics, discovery continuity across worker/pilot restarts, agency card API."""
import json
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from spotly.analytics import creative_metrics as cm
from spotly.analytics import creative_visual as cv
from spotly.analytics import visual_ocr as vo
from spotly.analytics.brand_extractor import extract_brands, skeleton_ar, skeleton_lat, suggest_brand
from spotly.config import HOP_SIZE
from spotly.config import SAMPLE_RATE as SR
from spotly.core.discovery import FPS, DiscoveryEngine
from spotly.core.discovery_report import list_assets
from spotly.core.fingerprint import StreamingFingerprinter
from spotly.database.schema import get_db
from spotly.tests.synthetic_broadcast import build_schedule, creative, render

CH = "El-Heddaf TV"
T0 = 1_791_900_000.0


# ================================================================== phonetic Darija matching

@pytest.mark.parametrize("latin, arabic", [("mobilis", "موبيليس"), ("condor", "كوندور"), ("cevital", "سيفيتال"),
                                           ("naftal", "نفطال"), ("samsung", "سامسونغ"), ("castrol", "كاسترول")])
def test_skeletons_agree_across_scripts(latin, arabic):
    assert skeleton_lat(latin) == skeleton_ar(arabic) and len(skeleton_lat(latin)) >= 4


def test_phonetic_spelling_variant_counts_like_a_misspelling():
    m = extract_brands("زيت كاسطرول")                                    # ط for ت: not a listed alias
    assert m[0].brand == "Castrol" and m[0].evidence[0][1].startswith("phonetic")
    assert suggest_brand("زيت كاسطرول") is None                          # one phonetic mention: not enough
    assert suggest_brand("كاسطرول زيت المحرك كاسطرول").brand == "Castrol"


def test_everyday_darija_never_matches_phonetically():
    text = ("واش راك خويا معاك حتى الدقيقة تسعين عيش المباريات بأعلى دقة جيل جديد عروض الصيف "
            "نقاوة بركة ذوق ما يتنساش حرية أكثر وقيمة أكبر الجزائر بلادي")
    assert suggest_brand(text) is None
    assert not [m for m in extract_brands(text) if m.strong]


# ================================================================== OCR text order + regions

def test_arabic_visual_order_is_reversed_to_reading_order():
    assert cv.visual_to_logical("ةكربةواقن") == "نقاوةبركة"
    assert cv.visual_to_logical("Messäd") == "Messäd"
    assert cv.visual_to_logical("ريفولوشن") != ""                       # Arabic-only strings are reversed
    assert cv.visual_to_logical("4G يديدج") == "جديدي 4G"               # Latin runs stay readable


def _box(cx, cy, w=200, h=40):
    return [[cx - w / 2, cy - h / 2], [cx + w / 2, cy - h / 2], [cx + w / 2, cy + h / 2], [cx - w / 2, cy + h / 2]]


def test_ticker_clock_marks_programme_content():
    assert cv.programme_overlay([(_box(1200, 690), "19:43 GMT+1", 0.9)], 1280, 720)
    assert cv.programme_overlay([(_box(1200, 690), "17:04GMT+1", 0.9)], 1280, 720)
    assert not cv.programme_overlay([(_box(640, 200), "20:00", 0.9)], 1280, 720)   # a time in the creative itself
    assert not cv.programme_overlay([(_box(640, 690), "www.djezzy.dz", 0.9)], 1280, 720)


# ================================================================== visual analysis (OpenCV) in an isolated process

def test_visual_checks_in_isolated_process():
    """Logo library, frame analysis and logo learning (spotly/tests/visual_checks.py) need OpenCV, which production
    only loads in the OCR child process; running them in a child pytest keeps this process within its budget."""
    import subprocess
    import sys
    from pathlib import Path
    here = Path(__file__).resolve()
    p = subprocess.run([sys.executable, "-m", "pytest", str(here.with_name("visual_checks.py")), "-q", "-p", "no:cacheprovider"],
                       cwd=here.parents[2], capture_output=True, text=True, timeout=600)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-1000:]
    assert " passed" in p.stdout

# ================================================================== creative metrics

def test_metric_components_and_weighted_health():
    feats = dict(programme_overlay=False, logo_basis="brand text", logo_area=0.06, sharpness=400.0)
    assert cm.visual_dominance(feats, "Messad")[0] == 100
    assert cm.visual_dominance(dict(feats, logo_area=0.003), "Messad")[0] < 30
    assert cm.visual_dominance(feats, None)[0] is None
    assert cm.visual_dominance(dict(feats, programme_overlay=True), "Messad")[0] is None
    assert cm.av_sync("Messad ...", True, "Messad", "Messad")[0] == 100
    assert cm.av_sync("...", True, None, "Messad") == (50, "brand shown, not said")
    assert cm.av_sync("...", True, "Djezzy", "Mobilis")[0] == 0
    assert cm.av_sync("", True, None, "Messad")[0] is None
    assert cm.pod_quality([dict(position="FIRST", break_ads=3), dict(position="SOLO", break_ads=1)])[0] == 92
    assert cm.pod_quality([dict(position="MIDDLE", break_ads=7)])[0] == 30
    assert cm.pod_quality([])[0] is None
    asset = dict(status="unclassified", suggested_brand="Messad", brand_confidence=0.7, transcript="", transcribed_at="x",
                 ocr_text="Messäd", nominal_duration_sec=30.0, duration_sec=28.0)
    r = cm.compute(asset, feats, [dict(position="FIRST", break_ads=3, daypart="Prime Time")])
    assert r["components"]["av_sync"]["score"] is None and r["assessed"] == "3/4"
    expected = round((0.3 * 100 + 0.3 * 85 + 0.2 * 70) / 0.8)
    assert r["health"] == expected and r["label"] == "30s Standard" and r["daypart"] == "Prime Time"
    assert cm.compute(dict(asset, ocr_text=None, suggested_brand=None), None, [])["health"] is None   # 1 component only


# ================================================================== OCR persistence

def _asset(conn, aid, status="unclassified", packshot=None, fp=None):
    conn.execute("INSERT INTO discovered_assets (asset_id, channel, duration_sec, fingerprint, first_seen, last_seen, "
                 "occurrences, status, packshot_path) VALUES (?, ?, 30, ?, 1, 2, 2, ?, ?)",
                 (aid, CH, (fp if fp is not None else np.ones(50, np.uint32)).astype("<u4").tobytes(), status, packshot))


def test_save_result_programme_logo_and_precedence(tmp_path):
    db = tmp_path / "s.db"
    with get_db(db) as conn:
        for i in (1, 2, 3):
            _asset(conn, f"ELHEDDAF_AD_00{i}")
        conn.execute("UPDATE discovered_assets SET suggested_brand = 'Mobilis', suggestion_source = 'audio' "
                     "WHERE asset_id = 'ELHEDDAF_AD_003'")
    empty = vo.analyse_ocr([])
    vo.save_result(db, "ELHEDDAF_AD_001", empty, dict(programme_overlay=True, dhash="0" * 16))
    logo = dict(programme_overlay=False, logo_match=dict(brand="Amiral", confidence=0.6, inliers=24), dhash="0" * 16)
    vo.save_result(db, "ELHEDDAF_AD_002", empty, logo)
    vo.save_result(db, "ELHEDDAF_AD_003", empty, logo)
    a = {r["asset_id"]: r for r in list_assets(db)}
    assert a["ELHEDDAF_AD_001"]["content_hint"] == vo.PROGRAMME_HINT and a["ELHEDDAF_AD_001"]["visual_status"] == "programme"
    assert (a["ELHEDDAF_AD_002"]["suggested_brand"], a["ELHEDDAF_AD_002"]["suggestion_source"]) == ("Amiral", "logo")
    assert json.loads(a["ELHEDDAF_AD_002"]["brand_evidence"])[0]["product"] == "Biscuits"
    assert (a["ELHEDDAF_AD_003"]["suggested_brand"], a["ELHEDDAF_AD_003"]["suggestion_source"]) == ("Mobilis", "audio")


def test_same_packshot_propagates_a_confirmed_brand(tmp_path):
    db = tmp_path / "c.db"
    with get_db(db) as conn:
        conn.execute("INSERT INTO reference_ads (ad_id, brand_name, campaign_name, duration_sec, fingerprint, fp_algorithm, "
                     "industry) VALUES (7, 'Bellat', 'x', 30, x'00', 't', 'Food & FMCG')")
        _asset(conn, "ELHEDDAF_AD_001", status="classified")
        _asset(conn, "ELHEDDAF_AD_002")
        _asset(conn, "ELHEDDAF_AD_003")
        conn.execute("UPDATE discovered_assets SET promoted_ad_id = 7 WHERE asset_id = 'ELHEDDAF_AD_001'")
        f = dict(programme_overlay=False, sharpness=300.0)
        for aid, h in (("ELHEDDAF_AD_001", "00ff00ff00ff00ff"), ("ELHEDDAF_AD_002", "00ff00ff00ff00fe"),
                       ("ELHEDDAF_AD_003", "ff00ff00ff00ff00")):
            conn.execute("UPDATE discovered_assets SET visual_features = ? WHERE asset_id = ?", (json.dumps(dict(f, dhash=h)), aid))
    pairs = vo.cluster_packshots(db)
    assert [(x, y) for x, y, _ in pairs] == [("ELHEDDAF_AD_001", "ELHEDDAF_AD_002")]
    a = {r["asset_id"]: r for r in list_assets(db)}
    assert (a["ELHEDDAF_AD_002"]["suggested_brand"], a["ELHEDDAF_AD_002"]["suggestion_source"]) == ("Bellat", "same packshot")
    assert a["ELHEDDAF_AD_003"]["suggested_brand"] is None
    assert a["ELHEDDAF_AD_002"]["visual_features"]["same_packshot_as"] == ["ELHEDDAF_AD_001"]


# ================================================================== discovery continuity across restarts

def _fp_chunks(chunks):
    out, fper, piece = [], StreamingFingerprinter(), 5 * SR
    for c0, x in chunks:
        for p0 in range(0, x.size, piece):
            y = x[p0:p0 + piece]
            out.append((fper.feed(y), (np.clip(y, -1, 1) * 32767).astype(np.int16)))
    return out


def _feed(eng, pieces, source, wall0):
    g = 0
    for fp, pcm in pieces:
        eng.feed(fp, g, wall0 + np.arange(g, g + fp.size) * HOP_SIZE / SR, pcm, g * HOP_SIZE, source=source)
        g += fp.size


def test_repeat_is_discovered_across_a_worker_restart(tmp_path):
    spots = {"A20": creative(11, 20)}
    events, total = build_schedule(spots, n_breaks=2, spacing=300.0)
    pieces = _fp_chunks(render(spots, events, total))
    half = len(pieces) // 2
    eng = DiscoveryEngine(CH, tmp_path / "r.db", audio_dir=tmp_path / "ads")
    old, new = object(), object()
    eng.begin_new_source(old)
    _feed(eng, pieces[:half], old, T0)                                   # 1st airing heard by worker #1
    eng.begin_new_source(new)                                            # supervisor restart: indices restart at 0
    _feed(eng, pieces[half:], new, T0 + half * 5)                        # 2nd airing heard by worker #2
    eng.feed(np.ones(10, np.uint32), 999_999, np.full(10, T0), source=old)   # late feed from worker #1: ignored
    eng.flush()
    with get_db(tmp_path / "r.db") as conn:
        rows = conn.execute("SELECT asset_id, occurrences, duration_sec FROM discovered_assets").fetchall()
    assert len(rows) == 1 and rows[0]["occurrences"] == 2 and rows[0]["duration_sec"] == pytest.approx(20, abs=1.0)


def test_history_survives_a_pilot_restart_via_state_file(tmp_path):
    a = DiscoveryEngine(CH, tmp_path / "h.db", audio_dir=tmp_path / "ads")
    a.begin_new_source("w1")
    fp = ((np.arange(int(60 * FPS), dtype=np.uint64) * 2654435761) % 2**32).astype(np.uint32)
    a.feed(fp.astype(np.uint32), 0, T0 + np.arange(fp.size) / FPS, source="w1")
    path = tmp_path / "state" / "d.npz"
    assert a.save_state(path)
    b = DiscoveryEngine(CH, tmp_path / "h.db", audio_dir=tmp_path / "ads")
    assert b.load_state(path) == pytest.approx(60, abs=0.5)
    b.begin_new_source("w2")
    b.feed(fp[:100].astype(np.uint32), 0, T0 + 100 + np.arange(100) / FPS, source="w2")
    assert b._fp.size == fp.size + 1 + 100                               # history + 1-frame gap + new audio
    assert np.array_equal(b._fp[:fp.size], fp)
    assert DiscoveryEngine("Other TV", tmp_path / "h.db", audio_dir=tmp_path / "ads").load_state(path) == 0.0
    assert b.load_state(path, max_age_s=-1) == 0.0                       # too old


def test_pilot_factory_reuses_one_engine(tmp_path):
    from spotly.service.pilot import build_supervisor
    sup, spec = build_supervisor("Synth TV=https://example.invalid/x.m3u8", tmp_path / "p.db")
    w1, w2 = sup._factory(spec), sup._factory(spec)                    # what a supervisor restart does
    assert w1.discovery is w2.discovery is sup.discovery_engines[spec.name]


# ================================================================== API: agency card fields

def test_assets_api_returns_creative_metrics_and_logo(tmp_path):
    from spotly.app.api_server import create_app
    db = tmp_path / "a.db"
    logo = tmp_path / "ELHEDDAF_AD_001_logo.jpg"
    from PIL import Image
    Image.new("RGB", (120, 60), "white").save(logo, quality=90)
    with get_db(db) as conn:
        _asset(conn, "ELHEDDAF_AD_001")
        conn.execute("UPDATE discovered_assets SET logo_path = ?, suggested_brand = 'Messad', brand_confidence = 0.7, "
                     "ocr_text = 'Messäd', visual_features = ?, nominal_duration_sec = 30",
                     (str(logo), json.dumps(dict(programme_overlay=False, logo_basis="brand text", logo_area=0.03,
                                                 sharpness=300.0, dhash="0" * 16))))
    app = create_app(db, channel_loader=lambda: [])
    app.state.audio_root = tmp_path
    with TestClient(app) as c:
        a = c.get("/api/discovery/assets").json()["assets"][0]
        assert "logo_path" not in a and a["logo_url"].endswith("/logo")
        assert a["creative"]["components"]["visual_dominance"]["score"] > 0 and a["creative"]["label"] == "30s Standard"
        r = c.get(a["logo_url"])
        assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
        html = c.get("/").text
        assert 'id="disc-body"' in html and 'id="disc-filter"' in html
