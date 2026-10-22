"""Phase 11.5: autonomous multi-modal brand resolution (OCR + speech fusion, carrier phrases, retro audit)."""
import wave
from datetime import datetime
from decimal import Decimal

import numpy as np
import pytest

from spotly.analytics import auto_resolver as ar
from spotly.analytics import compliance_audit as ca
from spotly.analytics.brand_extractor import (CONFIRM_MIN, FUZZY_MIN, clean_carriers, extract_brands, fuzzy_ratio,
                                            resolve_brand, sector_of)
from spotly.analytics.rate_engine import RateCardEngine, RateCardError, compute_airing_financials, validate
from spotly.core.proof import ALGERIA_TZ
from spotly.database.schema import get_db

CH = "El-Heddaf TV"
T = datetime(2026, 10, 13, 20, 0, tzinfo=ALGERIA_TZ).timestamp()


def test_carrier_phrases_are_removed_and_do_not_hide_the_brand():
    for phrase in ("لا بروموسيون", "كاين جديد", "ما تفوتش", "تخفيضات", "على خاطر"):
        assert clean_carriers(f"{phrase} messad") == "messad"
    assert extract_brands("لا بروموسيون كاين جديد ما تفوتش مسعد messad")[0].brand == "Messad"
    assert extract_brands("تخفيضات على خاطر") == []


def test_rapidfuzz_threshold_and_near_names():
    assert FUZZY_MIN == 82 and fuzzy_ratio("mobiliss", "mobilis") > FUZZY_MIN
    assert extract_brands("FLUXO KARE")[0].brand == "Flux Care"            # noisy two-word OCR
    assert extract_brands("amirall")[0].brand == "Amiral"
    assert all(m.brand != "Mobilis" for m in extract_brands("mobil"))      # another brand's exact name is not a typo


def test_sectors_separate_water_from_biscuits_and_cold_cuts():
    assert (sector_of("Messad"), sector_of("Amiral"), sector_of("Bellat")) == \
        ("Mineral water", "Biscuits", "Cold cuts")
    assert extract_brands("renault dacia")[0].product == "Cars"


@pytest.mark.parametrize("ocr,speech,checked,conf,status", [
    ("MESSAD", "ماء مسعد messad", True, 0.99, "CONFIRMED"),            # packshot + matching speech
    ("AMIRAL", "", True, 0.95, "CONFIRMED"),                            # silent / music spot
    ("AMIRAL", "ديرو ليكم بسكويت جديد للعائلة كلها", True, 0.90, "CONFIRMED"),
    ("", "موبيليس عرض جديد للجميع", True, 0.80, "SUGGESTED"),           # speech only: needs a human
    ("AMIRAL", "", False, 0.80, "SUGGESTED"),                          # transcript missing != silent
])
def test_fusion_confidence_ladder(ocr, speech, checked, conf, status):
    r = resolve_brand(ocr, speech, speech_checked=checked)
    assert (r.confidence, r.status) == (conf, status)
    assert (r.status == "CONFIRMED") == (r.confidence >= CONFIRM_MIN)


def test_fusion_refuses_conflicts_programme_frames_and_station_promos():
    assert resolve_brand("AMIRAL", "ماء مسعد messad messad", speech_checked=True).brand is None
    assert resolve_brand("MOBILIS", "", speech_checked=True, programme=True).brand is None
    assert resolve_brand("MESSAD", "", speech_checked=True, station_promo=True).brand is None


def test_first_or_last_clause_credits_30_percent_when_aired_mid_pod():
    card = RateCardEngine.load(overrides=dict(contracts=[dict(brand="Messad", booked_position="FIRST_OR_LAST")]))
    a = dict(brand="Messad", campaign="x", channel=CH, start_epoch=T, spot_seconds=30)
    mid = compute_airing_financials(dict(a, pod_position="MIDDLE"), card)
    assert mid["position_compliance"] == "breach"
    assert mid["contract_loss_credit_gross"] == (mid["spot_value"] * Decimal("0.30")).quantize(Decimal("0.01"))
    for ok in ("FIRST", "LAST", "SOLO"):
        assert compute_airing_financials(dict(a, pod_position=ok), card)["position_compliance"] == "compliant"
    with pytest.raises(RateCardError):
        validate({**RateCardEngine.load().config, "contracts": [dict(brand="X", pod_edge_credit=1.5)]})


# ---------------------------------------------------------------- retroactive audit on a database

def _wav(path, seed, secs=30, sr=16000):
    pcm = (np.random.default_rng(seed).uniform(-1, 1, secs * sr) * 12000).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return str(path)


def _asset(conn, tmp, n, ocr, transcript, transcribed=True, visual="ocr_brand", starts=()):
    aid = f"ELHEDDAF_AD_{n:03d}"
    conn.execute("INSERT INTO discovered_assets (asset_id, channel, duration_sec, nominal_duration_sec, fingerprint, "
                 "audio_path, first_seen, last_seen, occurrences, status, ocr_text, transcript, transcribed_at, "
                 "visual_status) VALUES (?, ?, 30, 30, ?, ?, 1, 2, ?, 'unclassified', ?, ?, ?, ?)",
                 (aid, CH, np.ones(50, np.uint32).tobytes(), _wav(tmp / f"{aid}.wav", n), len(starts), ocr, transcript,
                  "2026-10-13T00:00:00Z" if transcribed else None, visual))
    for s in starts:
        conn.execute("INSERT INTO discovered_occurrences (asset_id, channel, airing_start, airing_end, duration_sec, "
                     "confidence) VALUES (?, ?, ?, ?, 30, 0.9)", (aid, CH, s, s + 30))
    return aid


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "r.db"
    with get_db(path) as conn:
        _asset(conn, tmp_path, 1, "MESSAD", "", starts=(T, T + 3600))                  # silent -> 0.95
        _asset(conn, tmp_path, 2, "AMIRAL", "", starts=(T + 90, T + 7200))
        _asset(conn, tmp_path, 3, "", "موبيليس عرض جديد للجميع")                         # speech only -> suggestion
        _asset(conn, tmp_path, 4, "MOBILIS", "", visual="programme")                   # ticker frame -> nothing
        _asset(conn, tmp_path, 5, "DAYZY", "", transcribed=False)                      # no transcript yet -> suggestion
    return path


def test_dry_run_changes_nothing(db):
    r = ar.resolve_pending(db, apply=False)
    assert r["confirmed"] == 2 and r["promoted"] == 0 and r["backup"] is None
    with get_db(db) as conn:
        assert conn.execute("SELECT count(*) FROM discovered_assets WHERE status = 'classified'").fetchone()[0] == 0
        assert conn.execute("SELECT count(*) FROM reference_ads").fetchone()[0] == 0


def test_apply_confirms_promotes_adds_contracts_and_keeps_weak_matches_as_suggestions(db, tmp_path, monkeypatch):
    monkeypatch.setattr(ar, "DATA_DIR", tmp_path)
    r = ar.resolve_pending(db, apply=True)
    assert (r["confirmed"], r["promoted"], r["suggested"]) == (2, 2, 2)
    assert sorted(r["contracts_added"]) == ["Amiral", "Messad"] and r["backup"]
    with get_db(db) as conn:
        ads = {x["brand_name"]: x["industry"] for x in conn.execute("SELECT brand_name, industry FROM reference_ads")}
        st = {x["asset_id"]: (x["status"], x["suggested_brand"], x["suggestion_source"])
              for x in conn.execute("SELECT * FROM discovered_assets")}
        occ = conn.execute("SELECT count(*) FROM ad_occurrences").fetchone()[0]
    assert ads == {"Messad": "Mineral water", "Amiral": "Biscuits"} and occ == 4
    assert st["ELHEDDAF_AD_001"][0] == "classified" and st["ELHEDDAF_AD_003"] == ("unclassified", "Mobilis", "fusion")
    assert st["ELHEDDAF_AD_004"] == ("unclassified", None, None)                       # programme frame untouched
    assert st["ELHEDDAF_AD_005"][:2] == ("unclassified", "Dayzy")                      # waits for its transcript
    clauses = {c["brand"]: c for c in ca.get_contracts(db)}
    assert clauses["Messad"]["competitor_separation_seconds"] == 180 and clauses["Messad"]["separation_credit"] == 1.0
    assert clauses["Messad"]["pod_edge_credit"] == 0.30 and clauses["Messad"]["source"].startswith("auto-default")
    again = ar.resolve_pending(db, apply=True, backup=False)                           # idempotent
    assert again["promoted"] == 0 and again["contracts_added"] == []


def test_human_contract_is_never_overwritten_and_penalty_is_recomputed(db):
    ca.save_contracts(db, [dict(brand="Messad", competitor_separation_seconds=60, separation_credit=0.5)])
    r = ar.resolve_pending(db, apply=True, backup=False)
    assert r["contracts_added"] == ["Amiral"]
    assert {c["brand"]: c for c in ca.get_contracts(db)}["Messad"]["competitor_separation_seconds"] == 60
    assert r["reclaimable_before"] and r["reclaimable_after"]
    rep = ca.compliance_report(db)
    assert rep["kpis"]["airings"] == 4
