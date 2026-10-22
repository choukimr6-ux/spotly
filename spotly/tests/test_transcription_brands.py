"""Phase 11.2: spot boundary hardening, brand extraction, background transcription, one-click brand confirm."""
import json
import subprocess
import sys
import time
import wave
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from spotly.analytics import transcriber as tx
from spotly.analytics.brand_extractor import (BRANDS, detect_station_promo, extract_brands, normalize,
                                            suggest_brand)
from spotly.config import SAMPLE_RATE as SR
from spotly.core.discovery import FPS, DiscoveryEngine, snap_duration
from spotly.database.schema import get_db

ROOT = Path(__file__).resolve().parents[2]
CH = "El-Heddaf TV"


# ================================================================== duration snapping

@pytest.mark.parametrize("measured, nominal", [
    (17.97, 20), (27.46, 30), (29.6, 30), (30.45, 30), (20.47, 20), (14.2, 15), (44.1, 45), (60.8, 60),
    (38.94, 38.94), (54.92, 54.92), (31.5, 31.5), (25.0, 25.0),
])
def test_snap_duration_brackets(measured, nominal):
    got, basis = snap_duration(measured)
    assert got == pytest.approx(nominal)
    assert basis.startswith("snapped") == (got != round(measured, 2))


# ================================================================== fade-tolerant tail

def _tone(seconds, dbfs, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(int(seconds * SR))
    x *= 10 ** (dbfs / 20) / np.sqrt(np.mean(x ** 2))
    return (x * 32767).astype(np.int16)


def _engine_over(tmp_path, pcm):
    eng = DiscoveryEngine(CH, tmp_path / "f.db", audio_dir=tmp_path / "ads")
    eng._pcm = deque([(0, pcm)])
    eng._base = 0
    eng._fp = np.zeros(int(pcm.size / SR * FPS), np.uint32)
    return eng


@pytest.mark.parametrize("tail, expect_s", [
    ([(-22, 0.5), (-27, 0.5), (-32, 0.5), (-37, 0.5), (-70, 3)], 2.0),     # gradual fade: kept
    ([(-22, 0.5), (-27, 0.5), (-15, 3)], 1.0),                             # fade, then the next spot starts
    ([(-46, 4)], 0.0),                                                     # cliff into the noise-floor gap
    ([(-70, 4)], 0.0),                                                     # digital silence
    ([(-14, 4)], 0.0),                                                     # loud new content (next spot)
    ([(-18, 4)], 0.0),                                                     # next spot at similar loudness
    ([(-22, 4)], 0.0),                                                     # quieter programme plateau: not a fade
    ([(-22, 0.5), (-22, 3)], 0.0),                                         # one quieter step is not a fade
    ([(-20 - 3 * i, 0.5) for i in range(8)], 2.5),                         # long fade: capped
])
def test_fade_tail_extension(tmp_path, tail, expect_s):
    spot = _tone(20, -15)
    pcm = np.concatenate([spot] + [_tone(s, db, i + 1) for i, (db, s) in enumerate(tail)])
    eng = _engine_over(tmp_path, pcm)
    step = int(round(0.5 * FPS))
    e = int(20 * FPS)
    got = (eng._extend_fade_tail(0, e, step) - e) / FPS
    assert got == pytest.approx(expect_s, abs=0.1)


# ================================================================== brand extraction

def test_normalization_arabic_and_latin():
    assert normalize("Mobilisé — ÉTÉ!") == "mobilise ete"
    assert normalize("أُورِيدُو") == normalize("اوريدو")
    assert normalize("جـــازي") == "جازي"
    assert normalize("إفري") == normalize("افري") and normalize("آيدوم") == normalize("ايدوم")


@pytest.mark.parametrize("text, brand", [
    ("Djezzy, internet illimité pour tous", "Djezzy"),
    ("مع جيزي الانترنت بلا حدود", "Djezzy"),
    ("عروض موبيليس الجديدة", "Mobilis"),
    ("وموبيليس معاك", "Mobilis"),                                  # conjunction prefix stemmed
    ("Ooredoo Algérie, la 4G", "Ooredoo"),
    ("téléviseur Condor smart, Condor la qualité", "Condor"),
    ("حليب كانديا", "Candia"),
    ("l'huile Cevital", "Cevital"),
    ("telechargez baridimob", "BaridiMob"),
    ("mobilisse offre", "Mobilis"),                                # Whisper spelling variant
    ("mobiliss offres, mobiliss", "Mobilis"),                      # fuzzy, heard twice
    ("Mobil 1 huile moteur", "Mobil"),
])
def test_suggests_the_named_brand(text, brand):
    s = suggest_brand(text)
    assert s is not None and s.brand == brand and s.confidence >= 0.55


def test_single_fuzzy_hit_is_found_but_not_suggested():
    assert extract_brands("mobiliss offres")[0].brand == "Mobilis"
    assert suggest_brand("mobiliss offres") is None


def test_mobil_is_not_mobilis_and_vice_versa():
    assert suggest_brand("Mobilis") .brand == "Mobilis"
    assert all(m.brand != "Mobilis" for m in extract_brands("mobil super huile"))


@pytest.mark.parametrize("text", [
    "",
    "le match de ce soir",                                        # no brand
    "total total total",                                           # ambiguous word only (weak), never suggested
    "رامي سجل هدف",                                                # 'Ramy' is a first name: weak only
    "Djezzy et Mobilis",                                           # two brands, too close to call
])
def test_no_suggestion_when_absent_weak_or_ambiguous(text):
    assert suggest_brand(text) is None


def test_common_arabic_words_never_become_a_brand():
    # real El-Heddaf transcript fragment (2026-10-13): "جيل جديد" (new generation) heard as "جيلي جيدي"
    assert suggest_brand("عيش المباريات بأعلى دقة تكامل أجهزة جيلي جيدي وخدمة") is None
    assert suggest_brand("ياسير سجل هدف") is None                                   # first name
    assert suggest_brand("Geely Coolray, Geely Algérie").brand == "Geely"
    assert suggest_brand("جيزي معاك").brand == "Djezzy"


def test_reanalyse_rescoring_and_source_precedence(tmp_path):
    db = _db_with_asset(tmp_path)
    tx.save_result(db, "ELHEDDAF_AD_001", CH, "عيش المباريات أجهزة جيلي جيدي", "ar", "fake")
    with get_db(db) as conn:                                                       # as stored by the old list
        conn.execute("UPDATE discovered_assets SET suggested_brand = 'Geely', suggestion_source = NULL")
    assert tx.reanalyse(db) == [dict(asset_id="ELHEDDAF_AD_001", before="Geely", after=None)]
    with get_db(db) as conn:
        conn.execute("UPDATE discovered_assets SET ocr_text = 'DJEZZY www.djezzy.dz'")
    assert tx.reanalyse(db)[0]["after"] == "Djezzy"
    with get_db(db) as conn:
        assert conn.execute("SELECT suggestion_source FROM discovered_assets").fetchone()[0] == "packshot"
    tx.save_result(db, "ELHEDDAF_AD_001", CH, "", None, "fake")                    # speech: nothing -> OCR kept
    with get_db(db) as conn:
        assert tuple(conn.execute("SELECT suggested_brand, suggestion_source FROM discovered_assets").fetchone()) == ("Djezzy", "packshot")
    tx.save_result(db, "ELHEDDAF_AD_001", CH, "عروض موبيليس الجديدة", "ar", "fake")  # speech names a brand: wins
    with get_db(db) as conn:
        assert tuple(conn.execute("SELECT suggested_brand, suggestion_source FROM discovered_assets").fetchone()) == ("Mobilis", "audio")


def test_extract_counts_each_position_once_and_reports_evidence():
    m = extract_brands("Djezzy djezzy DJEZZY djezzy djezzy")[0]
    assert m.brand == "Djezzy" and m.score == pytest.approx(3.0) and m.strong       # capped at 3 mentions
    assert m.evidence and m.evidence[0][0] == "djezzy"


def test_brand_list_covers_requested_categories():
    names = {b.name for b in BRANDS}
    assert {"Mobilis", "Djezzy", "Ooredoo", "Condor", "Iris", "Cevital", "Soummam", "Naftal", "BaridiMob"} <= names
    assert {b.category for b in BRANDS} >= {"Telecom", "Electronics & Home", "Food & FMCG", "Automotive & Energy",
                                            "Services & Banking"}


def test_station_self_promo_detection():
    assert detect_station_promo("حملوا تطبيق الهداف الآن", CH)
    assert detect_station_promo("Regardez El Heddaf TV", CH)
    assert detect_station_promo("سجل الهدف في الدقيقة 90", CH) is None                 # 'the goal' in sports talk
    assert detect_station_promo("حملوا تطبيق الهداف", "Other TV") is None
    res = tx.analyse_text("تطبيق الهداف مع جيزي", CH)
    assert res["content_hint"] and res["suggested_brand"] is None                       # promo wins: no paid brand


# ================================================================== transcriber (fake model: no Whisper load)

def test_collapse_repeats_and_segment_filters():
    assert tx.collapse_repeats("a a a b c b c b c d") == "a b c d"
    assert tx.collapse_repeats("مرحبا بكم مرحبا بكم مرحبا بكم") == "مرحبا بكم"
    seg = lambda text, ns=0.1, lp=-0.3, cr=1.2: SimpleNamespace(text=text, no_speech_prob=ns, avg_logprob=lp,
                                                                compression_ratio=cr)
    kept = [seg(" عروض جيزي "), seg("silence", ns=0.9), seg("??", lp=-1.5), seg("la la la la", cr=3.0),
            seg("字幕 by someone"), seg("Djezzy Djezzy Djezzy")]
    assert tx.clean_text(kept) == "عروض جيزي Djezzy"


def _db_with_asset(tmp_path, seconds=6.0, pad=3.5):
    db = tmp_path / "t.db"
    wav = tmp_path / "ELHEDDAF_AD_001.wav"
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(SR)
        w.writeframes(_tone(seconds + pad, -20).tobytes())
    with get_db(db) as conn:
        conn.execute("INSERT INTO discovered_assets (asset_id, channel, duration_sec, nominal_duration_sec, "
                     "duration_basis, fingerprint, audio_path, audio_post_pad_sec, first_seen, last_seen, occurrences) "
                     "VALUES ('ELHEDDAF_AD_001', ?, ?, ?, 'measured', x'00000000', ?, ?, 100, 500, 2)",
                     (CH, seconds, seconds, str(wav), pad))
        for i, t in enumerate((100.0, 500.0)):
            conn.execute("INSERT INTO discovered_occurrences (asset_id, channel, airing_start, airing_end, duration_sec, "
                         "confidence, detected_at, algeria_timestamp) VALUES ('ELHEDDAF_AD_001', ?, ?, ?, ?, 0.9, "
                         "'2026-10-13T12:00:00Z', ?)", (CH, t, t + seconds, seconds, f"2026-10-13T13:0{i}:00+01:00"))
    return db


class FakeTranscriber:
    def __init__(self, text):
        self.text, self.calls, self.name = text, [], "fake whisper"

    def __call__(self, model):
        return self

    def transcribe(self, path, max_seconds=None):
        self.calls.append((path, max_seconds))
        return self.text, "ar", 0.93


def test_batch_saves_transcript_and_suggestion_without_touching_occurrences(tmp_path):
    db = _db_with_asset(tmp_path)
    with get_db(db) as conn:
        before = [tuple(r) for r in conn.execute("SELECT * FROM discovered_occurrences ORDER BY id")]
    fake = FakeTranscriber("عروض جيزي الجديدة، جيزي معاك")
    out = tx.run_batch(db, "base", transcriber_factory=fake)
    assert out[0]["suggested_brand"] == "Djezzy" and out[0]["suggested_industry"] == "Telecom"
    assert fake.calls[0][1] == pytest.approx(6.0)                    # spot only, not the post-roll
    with get_db(db) as conn:
        a = dict(conn.execute("SELECT * FROM discovered_assets").fetchone())
        after = [tuple(r) for r in conn.execute("SELECT * FROM discovered_occurrences ORDER BY id")]
    assert a["transcript"].startswith("عروض") and a["transcript_language"] == "ar" and a["transcribed_at"]
    assert a["transcription_model"] == "fake whisper" and a["brand_confidence"] >= 0.55
    assert json.loads(a["brand_evidence"])[0]["brand"] == "Djezzy"
    assert after == before and a["occurrences"] == 2 and a["first_seen"] == 100 and a["status"] == "unclassified"
    assert tx.pending_assets(db) == [] and tx.run_batch(db, transcriber_factory=fake) == []   # done once
    assert len(tx.pending_assets(db, retranscribe=True)) == 1


def test_batch_survives_a_bad_clip(tmp_path):
    db = _db_with_asset(tmp_path)

    class Boom(FakeTranscriber):
        def transcribe(self, path, max_seconds=None):
            raise RuntimeError("decode failed")
    out = tx.run_batch(db, transcriber_factory=Boom(""))
    assert "error" in out[0] and len(tx.pending_assets(db)) == 1     # left pending for a later retry


def test_memory_watchdog_exits_with_cap_code():
    p = subprocess.run([sys.executable, "-c", "import time; from spotly.analytics.transcriber import _memory_watchdog;"
                        " _memory_watchdog(1, 0.05); time.sleep(20)"], cwd=ROOT, capture_output=True, text=True,
                       timeout=60)
    assert p.returncode == tx.EXIT_MEMORY_CAP and "above cap" in p.stdout


def test_scheduler_launches_only_when_work_is_pending(tmp_path):
    from spotly.service.pilot import TranscriptionScheduler
    db = _db_with_asset(tmp_path)
    launched = []

    class Done:
        pid = 0

        def poll(self):
            return 0

    class S(TranscriptionScheduler):
        def _launch(self, n):
            launched.append(n)
            tx.save_result(self.db_path, "ELHEDDAF_AD_001", CH, "", None, "fake")   # the "batch" did its work
            self._proc = Done()
    s = S(db, interval=0.2).start()
    try:
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline and not launched:
            time.sleep(0.1)
        time.sleep(1.0)
    finally:
        s.stop()
    assert launched == [1]                                           # nothing pending afterwards: no relaunch


# ================================================================== API: transcript fields + one-click confirm

@pytest.fixture
def api(tmp_path):
    from spotly.app.api_server import create_app
    db = _db_with_asset(tmp_path)
    app = create_app(db, channel_loader=lambda: [])
    app.state.audio_root = tmp_path
    with TestClient(app) as c:
        yield c, db


def test_assets_api_exposes_transcript_and_suggestion(api):
    c, db = api
    tx.save_result(db, "ELHEDDAF_AD_001", CH, "Djezzy internet", "fr", "fake")
    a = c.get("/api/discovery/assets").json()["assets"][0]
    assert a["transcript"] == "Djezzy internet" and a["suggested_brand"] == "Djezzy"
    assert {"nominal_duration_sec", "duration_basis", "audio_post_pad_sec", "content_hint", "brand_confidence"} <= set(a)


def test_confirm_brand_flow(api):
    c, db = api
    url = "/api/discovery/assets/ELHEDDAF_AD_001/confirm-brand"
    assert c.post(url, json={"role": "self"}).status_code == 422                      # no suggestion yet
    tx.save_result(db, "ELHEDDAF_AD_001", CH, "عروض جيزي الجديدة", "ar", "fake")
    assert c.post(url, json={}).status_code == 422                                    # role required
    assert c.post(url, json={"role": "boss"}).status_code == 422
    assert c.post(url, json={"role": "self"}, headers={"origin": "https://evil.example"}).status_code == 403
    r = c.post(url, json={"role": "competitor"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["occurrences_backfilled"] == 2
    with get_db(db) as conn:
        ref = conn.execute("SELECT brand_name, duration_sec FROM reference_ads WHERE ad_id = ?", (body["ad_id"],)).fetchone()
        occ = conn.execute("SELECT COUNT(*), MIN(algeria_timestamp) FROM ad_occurrences WHERE ad_id = ?",
                           (body["ad_id"],)).fetchone()
        st = conn.execute("SELECT status FROM discovered_assets").fetchone()[0]
    assert ref["brand_name"] == "Djezzy" and ref["duration_sec"] == pytest.approx(6.0, abs=0.05)
    assert occ[0] == 2 and occ[1] == "2026-10-13T13:00:00+01:00" and st == "classified"
    assert c.post(url, json={"role": "self"}).status_code == 409                      # already classified


def test_dashboard_js_renders_transcript_and_confirm_button(api):
    c, _ = api
    js = c.get("/static/app.js").text
    # Phase 11.4: confirm is one click per role ([🏢 My Brand] / [⚔️ Competitor]) on the agency card
    assert "🏢 My Brand" in js and "⚔️ Competitor" in js and "/confirm-brand" in js and "transcript" in js
