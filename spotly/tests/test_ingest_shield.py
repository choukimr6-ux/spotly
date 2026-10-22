"""Phase 13.1 ingest shield: commercial-duration gate and permanent rejection of ticker / programme content."""
import numpy as np
import pytest

from spotly.analytics import creative_visual as cv
from spotly.analytics import visual_ocr as vo
from spotly.core import discovery as disc
from spotly.database.schema import get_db

W, H = 1280, 720


def box(cy_frac, cx_frac=0.5):
    cy, cx = cy_frac * H, cx_frac * W
    return [(cx - 50, cy - 10), (cx + 50, cy - 10), (cx + 50, cy + 10), (cx - 50, cy + 10)]


@pytest.mark.parametrize("seconds,ok", [
    (30.0, True), (29.2, True), (35.45, True), (40.94, True), (54.92, True), (55.41, True), (45.0, True),
    (7.5, True), (3.2, True),                      # bumpers / stings
    (22.96, False), (27.5, False), (33.0, False), (47.6, False), (71.9, False),   # interview-style arbitrary lengths
])
def test_commercial_duration_gate(seconds, ok):
    assert disc.commercial_duration_ok(seconds) is ok


def test_every_verified_production_length_passes_the_gate_except_known_cut():
    verified = [54.92, 30.0, 35.45, 54.92, 45.0, 40.94, 35.45, 30.0, 55.41, 20.0, 45.0, 45.0, 15.0, 20.0]
    assert all(disc.commercial_duration_ok(d) for d in verified)


@pytest.mark.parametrize("text,cy,expected", [
    ("MCA 2-1 USMA", 0.92, True),                  # score on the ticker bar
    ("67' MCA", 0.90, True),                       # match clock
    ("90+3'", 0.90, True),
    ("أخبار الرياضة", 0.2, True),                   # programme title anywhere
    ("نشرة الأخبار", 0.3, True),
    ("2-1", 0.4, False),                           # a score-looking pair mid-frame is not ticker evidence
    ("Djezzy 4G 100 DA", 0.5, False),
    ("www.djezzy.dz", 0.92, False),                # the sponsor URL on the ticker band stays allowed
    ("Amiral 30 jours", 0.3, False),
    ("الرياضة للجميع", 0.3, False),                 # a bare 'sport' word also appears in genuine spots
])
def test_programme_text_detector(text, cy, expected):
    got = cv.programme_text([(box(cy), text, 0.9)], W, H)
    assert bool(got) is expected


def test_low_confidence_ocr_is_not_evidence():
    assert cv.programme_text([(box(0.9), "MCA 2-1 USMA", 0.3)], W, H) is None


def _asset(conn, asset_id, status="unclassified"):
    conn.execute("INSERT INTO discovered_assets (asset_id, channel, duration_sec, fingerprint, status) "
                 "VALUES (?, 'Ch', 30, x'00', ?)", (asset_id, status))


def test_programme_frame_is_rejected_permanently(tmp_path):
    db = tmp_path / "s.db"
    with get_db(db) as conn:
        _asset(conn, "P1")
        _asset(conn, "OK1")
        _asset(conn, "C1", "classified")
    empty = vo.analyse_ocr([])
    why = cv.programme_text([(box(0.9), "MCA 2-1 USMA", 0.9)], W, H)
    vo.save_result(db, "P1", empty, dict(programme_overlay=True, programme_reason=why, dhash="0" * 16))
    vo.save_result(db, "OK1", empty, dict(programme_overlay=False, dhash="0" * 16))
    with get_db(db) as conn:
        rows = {r["asset_id"]: dict(r) for r in conn.execute("SELECT asset_id, status, label, content_hint FROM discovered_assets")}
    assert rows["P1"]["status"] == "ignored" and "score" in rows["P1"]["label"] and rows["P1"]["content_hint"] == vo.PROGRAMME_HINT
    assert rows["OK1"]["status"] == "unclassified" and rows["C1"]["status"] == "classified"


def test_rejection_never_touches_a_classified_asset(tmp_path):
    db = tmp_path / "s.db"
    with get_db(db) as conn:
        _asset(conn, "C1", "classified")
    vo.save_result(db, "C1", vo.analyse_ocr([]), dict(programme_overlay=True, dhash="0" * 16))
    with get_db(db) as conn:
        assert conn.execute("SELECT status FROM discovered_assets WHERE asset_id='C1'").fetchone()[0] == "classified"


def test_ignored_tombstone_makes_discovery_skip_the_audio(tmp_path):
    """DiscoveryEngine._record returns early for a non-unclassified match: the rejected audio never comes back."""
    import inspect
    src = inspect.getsource(disc.DiscoveryEngine._record)
    assert 'asset["status"] != "unclassified"' in src and "return" in src.split('asset["status"] != "unclassified"')[1][:120]
