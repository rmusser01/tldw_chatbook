import numpy as np
from tldw_chatbook.Audio.diarizer_cluster import OnlineClusterer, reconcile

def _v(*xs): return np.array(xs, dtype=np.float32)

def test_two_distinct_voices_get_two_stable_ids():
    c = OnlineClusterer(threshold=0.25, max_speakers=8)
    a1 = c.assign(_v(1, 0, 0)); b1 = c.assign(_v(0, 1, 0))
    a2 = c.assign(_v(0.95, 0.05, 0)); b2 = c.assign(_v(0.02, 0.98, 0))
    assert a1 == a2 and b1 == b2 and a1 != b1

def test_cap_folds_extra_speaker_into_nearest():
    c = OnlineClusterer(threshold=0.01, max_speakers=2)
    c.assign(_v(1, 0, 0)); c.assign(_v(0, 1, 0))
    third = c.assign(_v(0, 0, 1))
    assert third in {"S1", "S2"}  # folded, no S3

def test_pinned_cluster_is_never_merged_away():
    c = OnlineClusterer(threshold=0.9, max_speakers=8)
    a = c.assign(_v(1, 0, 0)); c.pin(a)
    # a near-identical later vector must still resolve to the pinned id
    assert c.assign(_v(0.99, 0.01, 0)) == a

def test_reconcile_maps_final_to_live_by_nearest_centroid():
    live = {"S1": _v(1, 0, 0), "S2": _v(0, 1, 0)}
    final = [("F0", _v(0, 0.9, 0)), ("F1", _v(0.9, 0, 0))]
    assert reconcile(live, final) == {"F0": "S2", "F1": "S1"}
