"""Phase 21: campaign family trees (master / cut-down), trailing-edge disambiguation, live 2 fps frame ingest."""
import io
import sqlite3
import time
import tracemalloc

import numpy as np
import pytest

from spotly.analytics import ad_onboarding as ob
from spotly.analytics import campaign_family as cf
from spotly.analytics import pod_boundary_detector as pbd
from spotly.analytics import reference_matcher as rm
from spotly.data import reference_db as rdb
from spotly.database.schema import get_db
from spotly.service.sysinfo import memory_mb
from spotly.stream import elheddaf_stream as hs
from spotly.tests import test_pod_boundary as pb
from spotly.tests.test_ad_onboarding import make_mp4, master_audio
from spotly.tests.test_reference_matcher import noise, synth

SR = rm.SR
EPOCH = 1_700_000_000.0


def family_index(master, cut, parent="MASTER_30", child="CUTDOWN_15"):
    def sig(asset, x, **kw):
        codes, t_cs, lead, trimmed = rm.signature_from_audio(x)
        return rdb.Signature(asset, "Djezzy", "Telecom", x.size / SR, codes, t_cs, lead, trimmed, **kw)
    return rm.ReferenceIndex([sig(parent, master), sig(child, cut, master_asset_id=parent, cut_type="CUT_DOWN")])


def stream(index, audio, piece=0.2):
    """Feed `audio`; -> (candidates, seconds after the audio's end at which each was emitted)."""
    m = rm.StreamMatcher(index)
    out, step = [], int(piece * SR)
    for i in range(0, audio.size, step):
        for c in m.feed(audio[i:i + step], EPOCH + i / SR):
            out.append((c, m.stream_seconds))
    for c in m.flush():
        out.append((c, m.stream_seconds))
    return out


@pytest.fixture(scope="module")
def audio30():
    return synth(11, 30.0)


@pytest.fixture(scope="module")
def fam(audio30):
    return family_index(audio30, audio30[:15 * SR])


def quiet(sec):
    return noise(sec, amp=3e-4, seed=5)                    # ~ -70 dBFS: a real gap, not digital zero


# ------------------------------------------------------------------------------ schema & registry tree
def test_schema_has_family_columns_and_rejects_bad_cut_type(tmp_path):
    with get_db(tmp_path / "a.db") as conn:
        rdb.ensure_schema(conn)
        cols = {r[1]: r for r in conn.execute("PRAGMA table_info(reference_signatures)")}
        assert {"master_asset_id", "cut_type"} <= set(cols)
        blob = rdb.pack_blob(np.array([1], np.uint32), np.array([5], np.uint16), 0.0, 1.0)
        rdb.upsert_signature(conn, "M", "B", "S", 30.0, blob, None)
        rdb.upsert_signature(conn, "C", "B", "S", 15.0, blob, None, master_asset_id="M", cut_type="CUT_DOWN")
        rows = {r["asset_id"]: (r["master_asset_id"], r["cut_type"]) for r in conn.execute("SELECT * FROM reference_signatures")}
        assert rows == {"M": (None, "MASTER"), "C": ("M", "CUT_DOWN")}
        rdb.upsert_signature(conn, "C", "B", "S", 15.0, blob, None)               # a registry rebuild keeps the family
        assert conn.execute("SELECT master_asset_id, cut_type FROM reference_signatures WHERE asset_id='C'").fetchone()[:] == ("M", "CUT_DOWN")
        with pytest.raises(ValueError):
            rdb.upsert_signature(conn, "X", "B", "S", 9.0, blob, None, cut_type="BOGUS")
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("UPDATE reference_signatures SET cut_type = 'BOGUS' WHERE asset_id = 'C'")


def test_old_table_is_migrated_in_place(tmp_path):
    with get_db(tmp_path / "old.db") as conn:
        conn.execute("CREATE TABLE reference_signatures (asset_id TEXT PRIMARY KEY, brand_name TEXT NOT NULL, "
                     "canonical_sector TEXT, nominal_duration REAL NOT NULL, audio_fingerprint_blob BLOB NOT NULL, "
                     "visual_p_hash TEXT, created_at TIMESTAMP, last_matched_at TIMESTAMP, match_count INTEGER NOT NULL "
                     "DEFAULT 0, ad_id INTEGER)")
        conn.execute("INSERT INTO reference_signatures (asset_id, brand_name, nominal_duration, audio_fingerprint_blob) "
                     "VALUES ('OLD', 'B', 30, x'00')")
        rdb.ensure_schema(conn)
        assert conn.execute("SELECT master_asset_id, cut_type FROM reference_signatures").fetchone()[:] == (None, "MASTER")


def test_family_tree_resolves_depth_and_survives_bad_rows():
    S = lambda a, p=None, t="MASTER": rdb.Signature(a, "B", "S", 30, np.zeros(0, np.uint32), np.zeros(0, np.uint16), 0, 0,  # noqa: E731
                                                    master_asset_id=p, cut_type=t)
    tree = rdb.family_tree([S("M"), S("C", "M", "CUT_DOWN"), S("T", "C", "TEASER"), S("ORPHAN", "GONE", "CUT_DOWN"),
                            S("X", "Y", "CUT_DOWN"), S("Y", "X", "CUT_DOWN")])
    assert tree["T"]["root"] == "M" and tree["T"]["depth"] == 2 and tree["C"]["parent"] == "M"
    assert tree["ORPHAN"]["root"] == "ORPHAN" and tree["ORPHAN"]["parent"] is None        # dangling parent: own root
    assert tree["X"]["root"] in ("X", "Y") and tree["Y"]["root"] in ("X", "Y")             # cycle: no hang, no crash


def test_index_builds_the_tree_and_hot_reload_picks_up_a_new_variant(tmp_path, audio30):
    db = tmp_path / "h.db"
    codes, t_cs, lead, trimmed = rm.signature_from_audio(audio30)
    blob = rdb.pack_blob(codes, t_cs, lead, trimmed)
    with get_db(db) as conn:
        rdb.ensure_schema(conn)
        rdb.upsert_signature(conn, "MASTER_30", "Djezzy", "Telecom", 30.0, blob, None)
    matcher = rm.ReferenceMatcher(db, compliance=False)
    assert matcher.index.families() == {}
    with get_db(db) as conn:
        rdb.upsert_signature(conn, "CUT_15", "Djezzy", "Telecom", 15.0, blob, None, master_asset_id="MASTER_30",
                             cut_type="CUT_DOWN")
    assert matcher.reload_if_changed()
    assert {k: sorted(v) for k, v in matcher.index.families().items()} == {"MASTER_30": ["CUT_15", "MASTER_30"]}
    m = matcher.index.meta[matcher.index.position["CUT_15"]]
    assert (m.cut_type, m.parent, matcher.index.meta[m.root].asset_id) == ("CUT_DOWN", "MASTER_30", "MASTER_30")


# ------------------------------------------------------------------------------ register-ad: parent / cut type
def test_register_ad_with_parent_and_cut_type(tmp_path):
    from spotly import cli
    db = tmp_path / "r.db"
    full, body = master_audio(31)
    master_mp4, cut_mp4 = tmp_path / "m.mp4", tmp_path / "c.mp4"
    make_mp4(master_mp4, full)
    make_mp4(cut_mp4, np.concatenate([np.zeros(int(ob.rm.SR * 0.2), np.float32), body[:int(3.5 * SR)]]))
    m = ob.register_ad(master_mp4, "Djezzy", "Telecom", campaign="Ramadan 2026", position="FIRST", db_path=db)
    assert m["cut_type"] == "MASTER" and m["parent_id"] is None
    with pytest.raises(ob.OnboardingError, match="not registered"):
        ob.register_ad(cut_mp4, "Djezzy", "Telecom", db_path=db, parent_id="NOPE")
    with pytest.raises(ob.OnboardingError, match="cannot have a parent"):
        ob.register_ad(cut_mp4, "Djezzy", "Telecom", db_path=db, parent_id=m["asset_id"], cut_type="MASTER")
    with pytest.raises(ob.OnboardingError, match="needs --parent-id"):
        ob.register_ad(cut_mp4, "Djezzy", "Telecom", db_path=db, cut_type="TEASER")
    with pytest.raises(ob.OnboardingError, match="differs from the parent"):
        ob.register_ad(cut_mp4, "Mobilis", "Telecom", db_path=db, parent_id=m["asset_id"])
    # through the real CLI: a declared edit shares audio with its parent and must not trip the duplicate guard
    cli.main(["register-ad", "--file", str(cut_mp4), "--brand", "Djezzy", "--sector", "Telecom", "--parent-id",
              m["asset_id"], "--cut-type", "CUT_DOWN", "--db", str(db)])
    with get_db(db) as conn:
        row = conn.execute("SELECT asset_id, master_asset_id, cut_type FROM reference_signatures WHERE master_asset_id IS NOT NULL").fetchone()
    assert (row["master_asset_id"], row["cut_type"]) == (m["asset_id"], "CUT_DOWN")
    from spotly.analytics.compliance_audit import get_contracts
    clauses = {c["campaign"]: c for c in get_contracts(db)}
    child = next(c for k, c in clauses.items() if k != "Ramadan 2026")
    assert child["booked_position"] == "FIRST"                      # the edit airs under the parent's booking
    assert {k: sorted(v) for k, v in rm.ReferenceIndex.from_db(db).families().items()} == {m["asset_id"]: sorted([m["asset_id"], row["asset_id"]])}


# ------------------------------------------------------------------------------ trailing-edge disambiguation
def test_clean_15s_cut_snaps_to_the_cut_down_and_not_the_master(fam):
    x = np.concatenate([noise(3), fam_cut(fam), quiet(4), noise(6)])
    got = stream(fam, x)
    assert [c.asset_id for c, _ in got] == ["CUTDOWN_15"]
    c, emitted_at = got[0]
    assert c.cut_type == "CUT_DOWN" and c.family_root == "MASTER_30" and "level dip" in c.family_note
    assert c.nominal_duration == pytest.approx(15.0, abs=0.05) and c.airing_start - EPOCH == pytest.approx(3.0, abs=0.15)
    print(f"\n[bench] cut-down decision {emitted_at - 18.0:.2f} s after its trailing edge")


def fam_cut(fam):
    return synth(11, 30.0)[:15 * SR]


def test_master_aired_in_full_keeps_master_tracking_and_no_cut_down(fam, audio30):
    got = stream(fam, np.concatenate([noise(3), audio30, noise(6)]))
    assert [c.asset_id for c, _ in got] == ["MASTER_30"] and got[0][0].cut_type == "MASTER"
    assert got[0][0].family_note == ""


def test_cut_down_straight_into_other_loud_content_is_still_logged_not_mistaken_for_the_master(fam):
    x = np.concatenate([noise(3), fam_cut(fam), noise(14, amp=0.12, seed=7), noise(4)])    # no gap: a steady level carries on
    got = stream(fam, x)
    assert [c.asset_id for c, _ in got] == ["CUTDOWN_15"]
    assert got[0][0].family_note.startswith("snapped to CUT_DOWN 15s")        # logged (never dropped), with its basis on record


def test_back_to_back_cut_down_then_master_are_two_airings(fam, audio30):
    x = np.concatenate([noise(3), fam_cut(fam), quiet(3), audio30, quiet(4)])
    got = stream(fam, x)
    assert [c.asset_id for c, _ in got] == ["CUTDOWN_15", "MASTER_30"]


def test_energy_dip_needs_12_db_or_silence(fam):
    m = rm.StreamMatcher(fam)
    loud = (synth(3, 8.0))
    m.feed(np.concatenate([loud, loud[:SR] * 0.5, loud]), EPOCH)                 # -6 dB only at t=8..9
    assert m.energy_dip(8.5)[0] is False
    m = rm.StreamMatcher(fam)
    m.feed(np.concatenate([loud, loud[:SR] * 0.1, loud]), EPOCH)                 # -20 dB
    dip, depth = m.energy_dip(8.5)
    assert dip and depth >= 12.0


# ------------------------------------------------------------------------------ rollup: parent campaign, exact variants
def test_rollup_aggregates_at_parent_campaign_and_keeps_variants(tmp_path):
    db = tmp_path / "f.db"
    blob = rdb.pack_blob(np.array([1], np.uint32), np.array([5], np.uint16), 0.0, 1.0)
    with get_db(db) as conn:
        rdb.ensure_schema(conn)
        for ad_id, camp in ((1, "Ramadan"), (2, "Ramadan [CUT_DOWN]"), (3, "Other")):
            conn.execute("INSERT INTO reference_ads (ad_id, brand_name, campaign_name, duration_sec, fingerprint, "
                         "fp_algorithm, industry, ad_type) VALUES (?, 'Djezzy', ?, 30, x'00', 't', 'Telecom', 'self')",
                         (ad_id, camp))
        rdb.upsert_signature(conn, "M", "Djezzy", "Telecom", 30.0, blob, None, ad_id=1)
        rdb.upsert_signature(conn, "C", "Djezzy", "Telecom", 15.0, blob, None, ad_id=2, master_asset_id="M", cut_type="CUT_DOWN")
        rdb.upsert_signature(conn, "O", "Djezzy", "Telecom", 30.0, blob, None, ad_id=3)
        for i, (ad, dur) in enumerate(((1, 30.0), (2, 15.0), (2, 15.0), (3, 30.0))):
            conn.execute("INSERT INTO ad_occurrences (ad_id, source_name, airing_start, airing_end, duration_detected, "
                         "confidence_score, human_verified) VALUES (?, 'live:X', ?, ?, ?, 0.97, 0)",
                         (ad, 1000.0 + i * 100, 1000.0 + i * 100 + dur, dur))
    out = cf.campaign_rollup(db)
    top = {c["parent_campaign"]: c for c in out["campaigns"]}
    assert set(top) == {"Ramadan", "Other"}
    r = top["Ramadan"]
    assert r["airings"] == 3 and r["airtime_sec"] == pytest.approx(60.0)
    assert r["sov_airtime_pct"] == pytest.approx(100.0 * 60 / 90)
    assert [(v["asset_id"], v["cut_type"], v["nominal_s"], v["airings"], v["airtime_sec"]) for v in r["variants"]] == \
        [("M", "MASTER", 30.0, 1, 30.0), ("C", "CUT_DOWN", 15.0, 2, 30.0)]
    with get_db(db) as conn:       # the audit rows still name the exact variant that aired
        assert [r_[0] for r_ in conn.execute("SELECT duration_detected FROM ad_occurrences WHERE ad_id = 2")] == [15.0, 15.0]


# ------------------------------------------------------------------------------ 2 fps frame ingest
def yuv_frame(luma=120, ticker=True):
    y = np.full((hs.FRAME_H, hs.FRAME_W), luma, np.uint8)
    if ticker:
        h = hs.FRAME_H
        y[int(h * .86):int(h * .895) + 1] = 235
        y[int(h * .915):int(h * .965) + 1] = 40
    return y.tobytes() + bytes(hs.FRAME_W * hs.FRAME_H // 2)


class LazyFrames:
    """Pipe-like source of `n` identical frames, produced on demand (so the test itself holds almost nothing)."""

    def __init__(self, frame, n):
        self.frame, self.left, self.pos = frame, n, 0

    def read(self, size):
        if self.pos == 0 and self.left == 0:
            return b""
        out = self.frame[self.pos:self.pos + size]
        self.pos += len(out)
        if self.pos >= len(self.frame):
            self.pos, self.left = 0, self.left - 1
        return out


class FakeProc:
    def __init__(self, data):
        self.stdout = io.BytesIO(data) if isinstance(data, bytes) else data
        self.dead = False

    def poll(self):
        return 0 if self.dead else None

    def kill(self):
        self.dead = True


def test_ticker_layout_and_black_calibration():
    on, off = hs.luma_from_yuv420(yuv_frame(120, True)), hs.luma_from_yuv420(yuv_frame(120, False))
    assert hs.ticker_present(on) and not hs.ticker_present(off)
    black = hs.luma_from_yuv420(yuv_frame(16, False))
    assert float(black.mean()) == pytest.approx(0.0, abs=0.5)                      # limited-range black reads 0
    assert not hs.ticker_present(hs.luma_from_yuv420(yuv_frame(235, False)))        # a white screen is not a ticker


def test_ring_drops_the_oldest_and_never_blocks():
    ring = hs.FrameRing(depth=4)
    t0 = time.perf_counter()
    for i in range(1000):
        ring.put(i)
    assert (time.perf_counter() - t0) < 0.5 and len(ring) == 4 and ring.dropped == 996
    assert [ring.get(0.01) for _ in range(5)] == [996, 997, 998, 999, None]      # newest four survive, in order


def test_slow_analyser_drops_frames_but_stalls_neither_reader_nor_audio():
    pbd.reset()
    frames, n = yuv_frame(), 120
    seen = []
    ing = hs.FrameIngest("Test", source=lambda: "x", spawn=lambda u: FakeProc(LazyFrames(frames, n)), backoff=30.0,
                         analyze=lambda ts, g: (time.sleep(0.05), seen.append(ts)))
    chunks, worst, t_audio = 0, 0.0, time.perf_counter()
    pcm = (np.random.default_rng(1).standard_normal(int(0.2 * 22050)) * 1500).astype(np.int16)
    ing.start()
    t_start = time.perf_counter()
    while time.perf_counter() - t_start < 3.0:                  # the audio loop, at 10x real time, while frames are slow
        t = time.perf_counter()
        pbd.feed_audio("Test", 1_000_000.0 + chunks * 0.2, pcm)
        worst = max(worst, time.perf_counter() - t)
        chunks += 1
        time.sleep(0.02)
    ing.stop()
    st = ing.status()
    assert st["frames_read"] == n                                # the reader drained the whole pipe immediately
    assert st["frames_dropped"] > 0 and st["frames_processed"] < n          # the analyser could not keep up: drops, not stalls
    assert all(b > a for a, b in zip(seen, seen[1:]))            # timestamps strictly increase; no drift after drops
    det = pbd.get_detector("Test")
    assert len(det._audio) == chunks * 4                          # 0 dropped audio chunks (4 x 50 ms frames per 0.2 s)
    assert worst < 0.05, f"audio feed stalled for {worst * 1000:.1f} ms"
    print(f"\n[bench] {chunks} audio chunks in 3 s with a 50 ms/frame analyser: 0 lost, worst feed_audio "
          f"{worst * 1000:.2f} ms; frames read {st['frames_read']}, processed {st['frames_processed']}, dropped {st['frames_dropped']}")


def test_frame_pipeline_memory_is_flat():
    pbd.reset()
    frames = yuv_frame()
    ing = hs.FrameIngest("Mem", source=lambda: "x", spawn=lambda u: FakeProc(LazyFrames(frames, 600)), backoff=30.0)
    before = memory_mb()[0]
    tracemalloc.start()
    ing.start()
    deadline = time.time() + 20
    while ing.status()["frames_read"] < 600 and time.time() < deadline:
        time.sleep(0.05)
    time.sleep(1.0)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    ing.stop()
    st = ing.status()
    assert st["frames_read"] == 600 and st["frames_processed"] + st["frames_dropped"] >= 590
    assert len(ing.ring) <= hs.RING_DEPTH
    assert peak < 8 * 2 ** 20, f"python heap peak {peak / 2 ** 20:.1f} MB"       # 600 frames x 194 KB streamed, 4 held
    assert memory_mb()[0] < 450 and memory_mb()[0] - before < 60
    print(f"\n[bench] 600 frames ({600 * hs.FRAME_BYTES / 2 ** 20:.0f} MB through the pipe): python peak {peak / 2 ** 20:.2f} MB, "
          f"RSS {memory_mb()[0]:.0f} MB, analyse p50 {st['analyse_ms_p50']} ms p95 {st['analyse_ms_p95']} ms")


# ------------------------------------------------------------------------------ conditional strict placement
def feed_frames_scene(ad_s, exit_t, total):
    """The Phase 20 break scene as REAL 480p frames through analyze_frame + the El-Heddaf ticker layout test."""
    det = pbd.get_detector(pb.CH)
    for i in range(int(total * 2)):
        t = i / 2
        in_break = 60.0 <= t < exit_t
        black = abs(t - 60.0) < 0.01 or abs(t - exit_t) < 0.01
        g = hs.luma_from_yuv420(yuv_frame(16 if black else 120, ticker=not in_break and not black))
        det.analyze_frame(pb.T + t, g, ticker_fn=hs.ticker_present)


def test_physical_rules_apply_only_when_video_frames_flow(tmp_path, monkeypatch):
    monkeypatch.delenv("SPOTLY_STRICT_PLACEMENT", raising=False)
    pbd.reset()
    s, e, exit_t, levels = pb.break_scene(promo_before=True, promo_after=False)
    total = exit_t + 8
    db = pb.make_db(tmp_path, [(s, e)])
    pb.feed_audio(pb.audio(levels, total))                                          # audio only: legacy fallback
    pb.settle(db, [(s, e)])
    ad = pb.ad_of(db)
    assert ad["placement_status"] == "unsensed" and ad["position"] in ("SOLO", "FIRST", "LAST")
    feed_frames_scene(s, exit_t, total)                                              # frames now flow: rules apply
    pb.settle(db, [(s, e)])
    ad = pb.ad_of(db)
    assert ad["placement_status"] == "demoted_confirmed" and ad["position"] == "LAST"        # entry failed, exit proven
    assert "ticker_absent" in ad["boundary_cues"] or "ticker_return" in ad["boundary_cues"]
    pbd.reset()
