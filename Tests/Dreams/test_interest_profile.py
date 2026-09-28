# Tests/Dreams/test_interest_profile.py
"""Interest profile weight math: decay, merge, and the snapshot read."""
import time
from datetime import datetime, timezone

import pytest

from tldw_chatbook.DB.Dreams_DB import DreamsDB
from tldw_chatbook.Dreams.interest_profile import decay_weights, merge_signals, snapshot

DAY = 86400.0


def test_decay_moves_weight_toward_floor_with_half_life():
    topics = [{"facet": "topic", "text": "rust", "weight": 0.9,
               "last_boosted_at": 0.0}]
    out = decay_weights(topics, now_epoch=14 * DAY)  # one half-life
    assert out[0]["weight"] == pytest.approx(0.05 + (0.9 - 0.05) / 2)


def test_decay_never_reaches_zero_and_caps_at_one():
    fresh = [{"facet": "topic", "text": "a", "weight": 1.0,
              "last_boosted_at": 99 * DAY}]
    stale = [{"facet": "topic", "text": "b", "weight": 1.0,
              "last_boosted_at": 0.0}]
    assert decay_weights(fresh, now_epoch=100 * DAY)[0]["weight"] <= 1.0
    assert decay_weights(stale, now_epoch=100 * DAY)[0]["weight"] >= 0.05


def test_merge_sums_same_topic_and_ranks():
    merged = merge_signals(
        [[{"facet": "topic", "text": " Rust ", "weight": 0.4},
          {"facet": "topic", "text": "html", "weight": 0.2}],
         [{"facet": "topic", "text": "rust", "weight": 0.3}]],
        top_n=2,
    )
    assert merged[0]["text"] == "rust"
    assert merged[0]["weight"] == pytest.approx(0.7)
    assert [t["text"] for t in merged] == ["rust", "html"]


def test_snapshot_decays_stale_topics_keeps_fresh_and_reads_region(
    tmp_path, monkeypatch
):
    """Ruling R3: snapshot() against a real DreamsDB on tmp_path."""
    db = DreamsDB(tmp_path / "dreams.sqlite", "test-client")
    try:
        db.upsert_profile_entry(
            "topic", "rust", weight=0.9, searchable=1, source="user"
        )
        db.upsert_profile_entry(
            "topic", "kubernetes", weight=0.9, searchable=1, source="user"
        )
        db.upsert_profile_entry(
            "goal", "visit japan", weight=1.0, searchable=1, source="user"
        )
        now_epoch = time.time()
        # upsert leaves last_boosted_at NULL (owned by other flows), so stamp
        # the fresh topic directly and leave the stale one never-boosted.
        with db.transaction() as conn:
            conn.execute(
                "UPDATE dream_interest_profile SET last_boosted_at = ?"
                " WHERE text = 'kubernetes'",
                (datetime.fromtimestamp(now_epoch, timezone.utc).isoformat(),),
            )
        monkeypatch.setattr(
            "tldw_chatbook.Dreams.settings.get_cli_setting",
            lambda section, key, default: (
                "near Seattle" if (section, key) == ("dreams", "region") else default
            ),
        )
        snap = snapshot(db, now_epoch=now_epoch)
    finally:
        db.close()
    topics = {t["text"]: t for t in snap["topics"]}
    assert set(topics) == {"rust", "kubernetes"}  # goals never join topics
    assert topics["rust"]["weight"] < 0.9  # stale topic decayed toward floor
    assert topics["rust"]["weight"] >= 0.05
    assert topics["kubernetes"]["weight"] == pytest.approx(0.9, abs=0.01)
    assert snap["region"] == "near Seattle"
