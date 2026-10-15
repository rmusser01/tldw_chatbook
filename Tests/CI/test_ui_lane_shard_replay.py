"""scripts/ui_lane_shard_replay.py: the timings the UI lane's census order is tuned on."""

from __future__ import annotations

import importlib.util
import re
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
        "Tests/UI/test_b.py::test_two",
        "Tests/UI/test_new.py",
    ]
    # Shard 1 gets entries 1 and 3, shard 2 gets 2 and 4. test_b.py's two node
    # ids run in different shards, so each shard carries half of its 40 s (a
    # log times a file, not a node id); an untimed file costs the default.
    assert replay.shard_loads(entries, seconds, 2) == [
        100.0 + 20.0,
        20.0 + replay.UNTIMED_SECONDS,
    ]


def test_a_file_split_over_shards_is_timed_in_every_shard_log(tmp_path):
    """A file whose node ids land in two shards is in both logs: its seconds
    are their sum, not whichever log was read last."""
    line = (
        "UI Fast Lane (1)\tUNKNOWN STEP\t2026-10-10T00:0{m}:{s:02d}.0000000Z {body}\n"
    )
    (tmp_path / "shard1.log").write_text(
        line.format(m=0, s=0, body="collected 2 items")
        + line.format(m=0, s=30, body="Tests/UI/test_b.py ..    [100%]"),
        encoding="utf-8",
    )
    (tmp_path / "shard2.log").write_text(
        line.format(m=0, s=0, body="collected 1 item")
        + line.format(m=0, s=12, body="Tests/UI/test_b.py .     [100%]"),
        encoding="utf-8",
    )
    assert replay.run_seconds(tmp_path) == {"Tests/UI/test_b.py": 42.0}


def test_the_first_file_of_a_log_with_no_collected_line_is_still_timed():
    """pytest -q prints no "collected" line: the first file then runs from the
    last line stamped before it, instead of costing the untimed default."""
    log = (
        "x\ty\t2026-10-10T00:00:00.0000000Z ===== test session starts =====\n"
        "x\ty\t2026-10-10T00:00:05.0000000Z rootdir: /repo\n"
        "x\ty\t2026-10-10T00:00:25.0000000Z Tests/UI/test_a.py ...   [ 50%]\n"
        "x\ty\t2026-10-10T00:00:55.0000000Z Tests/UI/test_b.py ...   [100%]\n"
    )
    assert replay.file_seconds(log) == {
        "Tests/UI/test_a.py": 20.0,
        "Tests/UI/test_b.py": 30.0,
    }


def test_the_tail_search_moves_the_heavy_file_off_the_heavy_shard():
    seconds = {"h": 100.0, "t1": 90.0, "t2": 5.0}
    head = ["h", "x"]  # shard 1 already carries 100 s
    order = replay.flattest_tail(head, ["t1", "t2"], seconds, 2)
    assert order == ["t2", "t1"]  # the 90 s file goes to shard 2


_CENSUS = """\
# a comment that stays where it is
Tests/UI/test_x.py
Tests/UI/test_h.py

# the tail block

# what test_t2.py pins
Tests/UI/test_t2.py
Tests/UI/test_t1.py
"""


def _run_dir(tmp_path: Path) -> Path:
    """One CI run of two shards: test_x.py 6 s, test_h.py 102 s, test_t1.py
    90 s and test_t2.py 6 s."""
    stamp = (
        "UI Fast Lane\tUNKNOWN STEP\t2026-10-10T00:{m:02d}:{s:02d}.0000000Z {body}\n"
    )
    run = tmp_path / "run1"
    run.mkdir()
    (run / "shard1.log").write_text(
        stamp.format(m=0, s=0, body="collected 2 items")
        + stamp.format(m=0, s=6, body="Tests/UI/test_x.py .   [ 50%]")
        + stamp.format(m=1, s=36, body="Tests/UI/test_t1.py . [100%]"),
        encoding="utf-8",
    )
    (run / "shard2.log").write_text(
        stamp.format(m=0, s=0, body="collected 2 items")
        + stamp.format(m=1, s=42, body="Tests/UI/test_h.py .  [ 50%]")
        + stamp.format(m=1, s=48, body="Tests/UI/test_t2.py . [100%]"),
        encoding="utf-8",
    )
    return run


def test_main_replays_the_runs_over_the_census_it_is_given(tmp_path, capsys):
    census = tmp_path / "census.txt"
    census.write_text(_CENSUS, encoding="utf-8")

    code = replay.main(
        ["--shards", "2", "--census", str(census), str(_run_dir(tmp_path))]
    )

    assert code == 0
    # Shard 1: test_x.py 6 s + test_t2.py 6 s; shard 2: test_h.py 102 s + 90 s.
    assert capsys.readouterr().out == "2 shards, as listed:  0.2 3.2  max 3.2 min\n"
    assert census.read_text(encoding="utf-8") == _CENSUS  # read, never written


def test_write_reorders_only_the_tail_of_the_census_it_is_given(tmp_path, capsys):
    """--write moves the heavy tail file off the heavy shard, in the census
    passed in: the head, its order and the block header stay, and a tail
    entry's own comment moves with it, write after write."""
    census = tmp_path / "census.txt"
    census.write_text(_CENSUS, encoding="utf-8")
    real_census = replay.CENSUS_PATH.read_bytes()
    argv = ["--shards", "2", "--tail", "2", "--write", "--census", str(census)]
    run = str(_run_dir(tmp_path))

    code = replay.main(argv + [run])

    assert code == 0
    written = census.read_text(encoding="utf-8")
    assert written == (
        "# a comment that stays where it is\n"
        "Tests/UI/test_x.py\n"
        "Tests/UI/test_h.py\n"
        "\n"
        "# the tail block\n"
        "\n"
        "Tests/UI/test_t1.py\n"
        "# what test_t2.py pins\n"
        "Tests/UI/test_t2.py\n"
    )
    out = capsys.readouterr().out
    assert "2 shards, best tail:  1.6 1.8  max 1.8 min" in out
    assert "rewrote the last 2 entries of census.txt" in out
    assert replay.main(argv + [run]) == 0
    assert census.read_text(encoding="utf-8") == written  # a second write keeps it
    assert replay.CENSUS_PATH.read_bytes() == real_census


def test_the_census_tail_block_keeps_the_shape_write_relies_on():
    """--write keeps the PR block's header in place only because one blank line ends it.

    A blank line among the block's own entries would travel with the entry
    after it and drift towards the header over repeated writes, so the block
    has none. Only the entries the header announces are checked: anything
    appended after them is not this block and may be laid out as usual.
    """
    census = _SCRIPT.parent / "ui_pr_gate_census.txt"
    lines = census.read_text(encoding="utf-8").splitlines()
    header = next(
        (
            i
            for i, line in enumerate(lines)
            if line.startswith("# TASK-33007 (PR #3057")
        ),
        None,
    )
    assert header is not None, "the PR block's header is missing"
    announced = re.search(r"the (\d+)$", lines[header])
    assert announced, "the header no longer says how many entries the block has"
    count = int(announced.group(1))
    blank = next((i for i in range(header, len(lines)) if not lines[i].strip()), None)
    assert blank is not None, "no blank line ends the block header"
    assert all(line.lstrip().startswith("#") for line in lines[header:blank])
    rows = [
        i
        for i in range(blank + 1, len(lines))
        if lines[i].strip() and not lines[i].lstrip().startswith("#")
    ]
    assert len(rows) >= count, "the block has fewer entries than its header announces"
    block = lines[blank + 1 : rows[count - 1] + 1]
    assert block[0].lstrip().startswith("#"), "the block's first entry lost its comment"
    assert all(line.strip() for line in block), "a blank line among the block's entries"


def test_main_refuses_runs_that_timed_no_file(tmp_path, capsys):
    (tmp_path / "empty-run").mkdir()
    census = tmp_path / "census.txt"
    census.write_text(_CENSUS, encoding="utf-8")

    code = replay.main(
        ["--shards", "2", "--census", str(census), str(tmp_path / "empty-run")]
    )

    assert code == 2
    assert "no shard logs" in capsys.readouterr().err
