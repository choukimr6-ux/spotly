"""Phase 24: solo pod quality, music-only sonic-visual lock, OCR sanitising and automatic resolution of UNCONFIRMED airings."""
import json
from datetime import datetime

import pytest

from kihc.analytics import auto_resolver as ar
from kihc.analytics import compliance_audit as ca
from kihc.analytics import creative_metrics as cm
from kihc.analytics.brand_extractor import clean_campaign, sanitize_ocr
from kihc.analytics.visual_ocr import analyse_ocr
from kihc.data import reference_db as rdb
from kihc.core.proof import ALGERIA_TZ
from kihc.database.schema import get_db

CH = "El-Heddaf TV"
T = datetime(2026, 10, 13, 20, 0, tzinfo=ALGERIA_TZ).timestamp()
REF = "f0f0f0f0f0f0f0f0"


# ------------------------------------------------------------------------------------------ pod quality / sync
def test_solo_airing_scores_100_with_label_not_na():
    score, note = cm.pod_quality([dict(position="SOLO", break_ads=1), dict(position=None, break_ads=1)])
    assert score == 100 and note == cm.SOLO_LABEL
    assert cm.display_score("pod_quality", score, note) == "100% (Solo Airing - Zero Clutter)"


def test_unconfirmed_boundaries_scored_from_contiguous_density():
    s2, _ = cm.pod_quality([dict(position=None, break_ads=2)])
    s4, _ = cm.pod_quality([dict(position=None, break_ads=4)])
    s20, _ = cm.pod_quality([dict(position=None, break_ads=20)])
    assert s2 > s4 > s20 >= 0 and s2 == 89 and s20 == 30             # bounded, deterministic, monotone in density
    assert cm.pod_quality([])[0] is None                              # no airing at all stays honestly unassessed


def test_music_only_spot_gets_sonic_visual_lock_from_packshot():
    score, note = cm.av_sync("", True, None, "Messad", packshot_distance=3)
    assert score == 95 and note.startswith(cm.LOCK_LABEL)
    assert cm.display_score("av_sync", score, note) == "95% (Sonic-Visual Lock)"
    assert cm.av_sync("", True, None, "Messad", packshot_distance=11)[0] is None     # not the same packshot: no lock
    assert cm.av_sync("", True, None, "Messad")[0] is None                           # nothing to confirm it with


def test_verbal_spot_without_brand_uses_token_alignment():
    assert cm.av_sync("promotion exclusive iphone maintenant", True, None, None, ocr_text="iphone exclusive")[0] == 100
    assert cm.av_sync("bonjour tout le monde", True, None, None, ocr_text="samsung galaxy")[0] == 0


def test_compute_exposes_display_text():
    r = cm.compute(dict(transcribed_at=1, transcript="", status="classified", ocr_text="Messad", suggested_brand="Messad"),
                   None, [dict(position="SOLO", break_ads=1)], packshot_distance=2)
    assert r["components"]["pod_quality"]["display"] == "100% (Solo Airing - Zero Clutter)"
    assert r["components"]["av_sync"]["display"] == "95% (Sonic-Visual Lock)"


# ------------------------------------------------------------------------------------------ OCR sanitiser
def test_noisy_ocr_string_resolves_to_canonical_entity():
    assert sanitize_ocr("new SILICONE FLUX/ FLUX MEN EDITION care") == "Flux Care - Men Edition"
    assert sanitize_ocr("FLUXO KARE 16:9") == "Flux Care"
    assert sanitize_ocr("MOBILIS 4G //") == "Mobilis - 4G"


def test_sanitiser_protects_arabic_and_never_invents_a_brand():
    assert "موبيليس" in sanitize_ocr("موبيليس / ///") or sanitize_ocr("موبيليس / ///") == "Mobilis"
    assert sanitize_ocr("XQZ /// foo foo") == "XQZ foo"                 # unknown: tidied only, repeated words collapsed
    assert sanitize_ocr(None) == "" and sanitize_ocr("   ") == ""


def test_ui_campaign_label_never_exposes_raw_scrape():
    assert clean_campaign("Flux Care", "new SILICONE FLUX/ FLUX MEN EDITION care") == "Men Edition"
    assert analyse_ocr(["new SILICONE FLUX/", "FLUX MEN EDITION care"])["display_name"] == "Flux Care - Men Edition"


# ------------------------------------------------------------------------------------------ resolver
@pytest.fixture
def db(tmp_path):
    path = tmp_path / "r.db"
    with get_db(path) as conn:
        rdb.ensure_schema(conn)
        conn.execute("INSERT INTO reference_ads (ad_id, brand_name, campaign_name, duration_sec, fingerprint, fp_algorithm, "
                     "industry, ad_type) VALUES (1, 'Mobilis', 'Spot', 30, x'00', 't', 'Telecom', 'self')")
        conn.execute("INSERT INTO reference_signatures (asset_id, brand_name, nominal_duration, audio_fingerprint_blob, "
                     "visual_p_hash, ad_id) VALUES ('MASTER_0001', 'Mobilis', 30, x'00', ?, 1)", (REF,))
    return path


def _occ(db, status, packshot_hex=None, conf=0.97, start=T):
    with get_db(db) as conn:
        return conn.execute(
            "INSERT INTO ad_occurrences (ad_id, source_name, airing_start, airing_end, duration_detected, confidence_score, "
            "human_verified, status, boundary_json) VALUES (1, ?, ?, ?, 30, ?, 0, ?, ?)",
            (f"live:{CH}", start, start + 30, conf, status,
             json.dumps(dict(packshot_phash=packshot_hex)) if packshot_hex else None)).lastrowid


def _flip(hexs, bits):
    return f"{int(hexs, 16) ^ ((1 << bits) - 1):016x}"                 # exactly `bits` bits away


def _status(db, oid):
    with get_db(db) as conn:
        return conn.execute("SELECT status, boundary_json FROM ad_occurrences WHERE occurrence_id = ?", (oid,)).fetchone()


def test_judge_thresholds():
    assert ar.judge(0, 0.1)[0] and ar.judge(8, 0.1)[0]
    assert ar.judge(9, 0.97)[0] and ar.judge(12, 0.96)[0]
    assert not ar.judge(9, 0.80)[0] and not ar.judge(9, None)[0]
    assert not ar.judge(13, 1.0)[0] and not ar.judge(None, 1.0)[0]


def _alerts(db):
    with get_db(db) as conn:
        return conn.execute("SELECT COUNT(*) FROM compliance_alerts").fetchone()[0]


def test_cutdown_with_matching_packshot_is_confirmed_and_credited(db):
    with get_db(db) as conn:                       # a competitor airs 60 s after the Mobilis cut-down: separation breach
        conn.execute("INSERT INTO reference_ads (ad_id, brand_name, campaign_name, duration_sec, fingerprint, fp_algorithm, "
                     "industry, ad_type) VALUES (2, 'Djezzy', 'Spot', 30, x'00', 't', 'Telecom', 'competitor')")
        conn.execute("INSERT INTO ad_occurrences (ad_id, source_name, airing_start, airing_end, duration_detected, "
                     "confidence_score, human_verified, status) VALUES (2, ?, ?, ?, 30, 0.97, 0, 'CONFIRMED')",
                     (f"live:{CH}", T + 90, T + 120))
    ca.save_contracts(db, [dict(brand="Mobilis", campaign="*", competitor_separation_seconds=180, separation_credit=1.0,
                                booked_position="ANY", pod_edge_credit=0.0)])
    oid = _occ(db, "UNCONFIRMED_CUTDOWN_NO_DIP", _flip(REF, 4))
    assert ca.refresh_alerts(db, lookback_days=3650, now=T + 3600) == 0 and _alerts(db) == 0    # logged, never claimed
    r = ar.resolve_unconfirmed(db, backup=False)
    row = _status(db, oid)
    assert r["scanned"] == 1 and r["resolved"] == 1 and row["status"] == "CONFIRMED"
    assert json.loads(row["boundary_json"])["resolution"]["packshot_distance"] == 4
    assert r["after"]["statuses"] == {"CONFIRMED": 2} and "UNCONFIRMED_CUTDOWN_NO_DIP" in r["before"]["statuses"]
    assert r["alerts_refreshed"] >= 1 and _alerts(db) >= 1               # credit reaches the ledger
    assert ar.resolve_unconfirmed(db, backup=False)["scanned"] == 0       # idempotent


def test_music_airing_resolved_by_packshot_hash(db):
    oid = _occ(db, "UNCONFIRMED_MUSIC_NO_PACKSHOT", REF)
    assert ar.resolve_unconfirmed(db, backup=False)["resolved"] == 1 and _status(db, oid)["status"] == "CONFIRMED"


def test_borderline_needs_acoustic_crosscheck_and_far_packshot_is_kept(db):
    weak = _occ(db, "UNCONFIRMED_CUTDOWN_NO_DIP", _flip(REF, 10), conf=0.80, start=T)
    strong = _occ(db, "UNCONFIRMED_CUTDOWN_NO_DIP", _flip(REF, 10), conf=0.99, start=T + 600)
    far = _occ(db, "UNCONFIRMED_MUSIC_NO_PACKSHOT", _flip(REF, 20), conf=0.99, start=T + 1200)
    none = _occ(db, "UNCONFIRMED_MUSIC_NO_PACKSHOT", None, conf=0.99, start=T + 1800)
    r = ar.resolve_unconfirmed(db, backup=False)
    assert r["scanned"] == 4 and r["resolved"] == 1
    assert [_status(db, o)["status"] for o in (weak, strong, far, none)] == [
        "UNCONFIRMED_CUTDOWN_NO_DIP", "CONFIRMED", "UNCONFIRMED_MUSIC_NO_PACKSHOT", "UNCONFIRMED_MUSIC_NO_PACKSHOT"]


def test_dry_run_writes_nothing(db):
    oid = _occ(db, "UNCONFIRMED_CUTDOWN_NO_DIP", REF)
    r = ar.resolve_unconfirmed(db, apply=False)
    assert r["resolved"] == 1 and r["backup"] is None and _status(db, oid)["status"] == "UNCONFIRMED_CUTDOWN_NO_DIP"


def test_evidence_frame_file_is_hashed(db, tmp_path):
    from PIL import Image
    from kihc.analytics.reference_matcher import phash_image
    img = Image.effect_noise((64, 64), 60).convert("RGB")
    frame = tmp_path / "f.png"
    img.save(frame)
    with get_db(db) as conn:
        conn.execute("UPDATE reference_signatures SET visual_p_hash = ?", (phash_image(img),))
    oid = _occ(db, "UNCONFIRMED_CUTDOWN_NO_DIP")
    with get_db(db) as conn:
        conn.execute("UPDATE ad_occurrences SET evidence_frame_path = ? WHERE occurrence_id = ?", (str(frame), oid))
    assert ar.resolve_unconfirmed(db, backup=False)["resolved"] == 1
