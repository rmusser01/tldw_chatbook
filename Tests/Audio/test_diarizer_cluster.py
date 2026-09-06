import numpy as np
from tldw_chatbook.Audio.diarizer_cluster import (
    OnlineClusterer,
    merged_speaker_names,
    reconcile,
)

def _v(*xs): return np.array(xs, dtype=np.float32)

def test_two_distinct_voices_get_two_stable_ids():
    c = OnlineClusterer(threshold=0.25, max_speakers=8)
    a1 = c.assign(_v(1, 0, 0)); b1 = c.assign(_v(0, 1, 0))
    a2 = c.assign(_v(0.95, 0.05, 0)); b2 = c.assign(_v(0.02, 0.98, 0))
    assert a1 == a2 and b1 == b2 and a1 != b1

def test_cap_fold_goes_to_the_nearest_cluster():
    c = OnlineClusterer(threshold=0.01, max_speakers=2)
    s1 = c.assign(_v(1, 0, 0)); c.assign(_v(0, 1, 0))
    assert c.assign(_v(0.9, 0.2, 0)) == s1  # clearly nearer s1

def test_pinned_cluster_centroid_survives_a_distinct_fold_at_cap():
    c = OnlineClusterer(threshold=0.2, max_speakers=2)
    a = c.assign(_v(1, 0, 0)); b = c.assign(_v(0, 1, 0))
    c.pin(a)
    before = c.centroids()[a].copy()
    folded = c.assign(_v(0.6, 0.1, 0.79))  # distinct 3rd voice, nearest to a, at cap
    assert np.allclose(before, c.centroids()[a])  # pinned centroid untouched
    assert folded == b  # folded into the non-pinned cluster instead

def test_pinned_cluster_still_joined_by_a_similar_voice():
    c = OnlineClusterer(threshold=0.3, max_speakers=2)
    a = c.assign(_v(1, 0, 0)); c.assign(_v(0, 1, 0)); c.pin(a)
    assert c.assign(_v(0.98, 0.02, 0)) == a  # a near-match still joins the pinned cluster

def test_reconcile_maps_final_to_live_by_nearest_centroid():
    live = {"S1": _v(1, 0, 0), "S2": _v(0, 1, 0)}
    final = [("F0", _v(0, 0.9, 0)), ("F1", _v(0.9, 0, 0))]
    assert reconcile(live, final) == {"F0": "S2", "F1": "S1"}

def test_reconcile_with_no_live_clusters_returns_empty():
    assert reconcile({}, [("F0", _v(1, 0, 0))]) == {}

def test_reconcile_never_gives_two_final_clusters_the_same_live_id():
    # Qodo Q11 completion: the Stop pass separated two speakers the live pass
    # had merged into one cluster. Mapping each final id to its nearest live
    # id independently handed BOTH of them "S1" and undid the correction.
    live = {"S1": _v(1, 0, 0)}
    final = [("F0", _v(0.99, 0.01, 0)), ("F1", _v(0.9, 0.1, 0))]
    mapping = reconcile(live, final)
    assert mapping == {"F0": "S1"}                 # the closest match wins it
    assert "F1" not in mapping                     # ... the other is left to be minted


# ---- spec §4/§8: many-to-one merge keeps both names and flags --------------

def test_merge_of_two_named_clusters_keeps_both_names_and_flags():
    # The Stop pass folded S2's segments entirely into S1; both were named.
    transitions = [("S1", "S1"), ("S2", "S1")]
    merged, flagged = merged_speaker_names(transitions, {"S1": "Alice", "S2": "Bob"})
    assert merged == {"S1": "Alice / Bob"} and flagged == ["S1"]

def test_merge_into_an_unnamed_survivor_still_keeps_both_names():
    # S3 (unnamed) absorbed both named clusters -> combine onto S3.
    transitions = [("S1", "S3"), ("S2", "S3")]
    merged, flagged = merged_speaker_names(transitions, {"S1": "Alice", "S2": "Bob"})
    assert merged == {"S3": "Alice / Bob"} and flagged == ["S3"]

def test_no_flag_when_an_absorbed_cluster_keeps_some_of_its_own_segments():
    # S2 kept one segment, so it was not fully merged -> its name is not at risk.
    transitions = [("S1", "S1"), ("S2", "S1"), ("S2", "S2")]
    merged, flagged = merged_speaker_names(transitions, {"S1": "Alice", "S2": "Bob"})
    assert merged == {} and flagged == []

def test_same_name_on_both_clusters_is_not_a_collision():
    transitions = [("S1", "S1"), ("S2", "S1")]
    merged, flagged = merged_speaker_names(transitions, {"S1": "Alice", "S2": "Alice"})
    assert merged == {} and flagged == []

def test_merge_of_unnamed_clusters_is_not_flagged():
    transitions = [("S1", "S1"), ("S2", "S1")]
    merged, flagged = merged_speaker_names(transitions, {})
    assert merged == {} and flagged == []
