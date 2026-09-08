import numpy as np
import pytest
from tldw_chatbook.Audio import diarizer_worker as w


def test_parse_args_defaults_to_speechbrain_and_reads_engine():
    assert w._parse_args([]) == (0, "speechbrain")
    assert w._parse_args(["--start-id", "7", "--engine", "onnx"]) == (7, "onnx")
    with pytest.raises(ValueError):
        w._parse_args(["--engine", "bogus"])


def test_map_final_clusters_reuses_matches_and_mints_past_start_id():
    # Deviation from the brief (see task-1-report.md): the brief's fixture had
    # a SECOND live centroid ("S2": [0.0, 1.0]) which -- verified against the
    # real, untouched `reconcile()` -- greedily claims F1 in the very first
    # (threshold-blind) matching pass whenever every live slot has a final to
    # fill (`reconcile(live, final, 0.25) == {"F0": "S1", "F1": "S2"}`, not a
    # mint). One live centroid makes F1 a genuine surplus no live id claims
    # within threshold, which is what "unmatched -> minted" actually needs.
    live = {"S1": np.array([1.0, 0.0])}
    finals = [("F0", np.array([0.99, 0.01]), 4.0), ("F1", np.array([0.7, 0.7]), 2.0)]
    out = {}
    mapping = w._map_final_clusters(finals, live, threshold=0.25, start_id=5, out_centroids=out)
    assert mapping["F0"] == "S1"
    assert mapping["F1"] == "S6"          # unmatched -> minted past start_id
    assert set(out) == {"S1", "S6"} and abs(np.linalg.norm(out["S1"]) - 1.0) < 1e-6


def test_speechbrain_engine_module_imports_without_torch(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "torch", None)   # importing torch would raise
    import importlib
    mod = importlib.import_module("tldw_chatbook.Audio.diarizer_engine_speechbrain")
    assert mod.MODEL_ID == "speechbrain/spkrec-ecapa-voxceleb@unpinned"
    assert callable(mod.load)


def test_loaded_engine_defaults_its_live_threshold_to_the_ecapa_tuned_value():
    """Task 8 (31827): the live clusterer's threshold becomes per-engine.

    0.25 is the value every engine used before this field existed, so an
    engine that does not set one (the SpeechBrain engine) keeps it.
    """
    loaded = w.LoadedEngine(lambda pcm, sr: [1.0], lambda *a: ([], {}), "fake@1")
    assert loaded.live_threshold == 0.25


def test_main_builds_the_clusterer_at_the_engines_live_threshold(monkeypatch):
    """The measured reason for this field (task-8-context, controller smoke):
    ECAPA's 0.25 minted S1..S8 in 30 s on a titanet_small stream, so each
    engine has to be able to hand `main()` its own tuned threshold."""
    import io
    import sys
    import types

    seen = {}

    def fake_load(live, max_speakers):
        seen["live"] = live
        seen["max_speakers"] = max_speakers
        return w.LoadedEngine(lambda pcm, sr: [1.0], lambda *a: ([], {}), "fake@1", live_threshold=0.55)

    monkeypatch.setitem(sys.modules, "tldw_chatbook.Audio.fake_engine_for_test", types.SimpleNamespace(load=fake_load))
    monkeypatch.setitem(w.ENGINES, "fake", "tldw_chatbook.Audio.fake_engine_for_test")
    monkeypatch.setattr(sys, "argv", ["worker", "--engine", "fake"])
    monkeypatch.setattr(sys, "stdin", types.SimpleNamespace(buffer=io.BytesIO(b"")))
    monkeypatch.setattr(sys, "stdout", types.SimpleNamespace(buffer=io.BytesIO()))

    assert w.main() == 0
    assert seen["live"].threshold == 0.55
