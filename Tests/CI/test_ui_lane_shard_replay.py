"""scripts/ui_lane_shard_replay.py: the timings the UI lane's census order is tuned on."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "ui_lane_shard_replay.py"
_spec = importlib.util.spec_from_file_location("ui_lane_shard_replay", _SCRIPT)
replay = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(replay)

_LOG = """\
UI Fast Lane (1)\tUNKNOWN STEP\t2026-10-10T00:00:00.0000000Z collected 9 items
UI Fast Lane (1)\tUNKNOWN STEP\t2026-10-10T00:00:10.0000000Z Tests/UI/test_a.py ...     [ 33%]
UI Fast Lane (1)\tUNKNOWN STEP\t2026-10-10T00:00:40.0000000Z Tests/UI/test_b.py ....... [ 80%]
UI Fast Lane (1)\tUNKNOWN STEP\t2026-10-10T00:01:10.0000000Z ..                         [100%]
UI Fast Lane (1)\tUNKNOWN STEP\t2026-10-10T00:01:11.0000000Z ===== 9 passed in 71.00s =====
"""


def test_a_file_is_timed_from_the_previous_files_last_line_to_its_own():
    assert replay.file_seconds(_LOG) == {
        "Tests/UI/test_a.py": 10.0,
        "Tests/UI/test_b.py": 60.0,  # its wrapped second line counts
    }


def test_each_file_is_scored_at_its_slowest_run():
    runs = [{"a": 10.0, "b": 60.0}, {"a": 25.0}]
    assert replay.worst_case(runs) == {"a": 25.0, "b": 60.0}


def test_shards_are_round_robin_and_node_ids_share_their_files_time():
    seconds = {"Tests/UI/test_a.py": 100.0, "Tests/UI/test_b.py": 40.0}
    entries = [
        "Tests/UI/test_a.py",
        "Tests/UI/test_b.py::test_one",
        "Tests/UI/test_new.py",
        "Tests/UI/test_b.py::test_two",
    ]
    # Shard 1 gets entries 1 and 3, shard 2 gets 2 and 4; test_b.py's 40 s is
    # charged once, to its first entry, and an untimed file costs the default.
    assert replay.shard_loads(entries, seconds, 2) == [
        100.0 + replay.UNTIMED_SECONDS,
        40.0,
    ]


def test_the_tail_search_moves_the_heavy_file_off_the_heavy_shard():
    seconds = {"h": 100.0, "t1": 90.0, "t2": 5.0}
    head = ["h", "x"]  # shard 1 already carries 100 s
    order = replay.flattest_tail(head, ["t1", "t2"], seconds, 2)
    assert order == ["t2", "t1"]  # the 90 s file goes to shard 2
