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


def test_the_census_header_stays_put_when_every_entry_moves():
    """--deal --write moves every entry with the lines above it; the census's
    own header stays first only because a blank line ends it before the first
    entry's comments (TASK-36514). Without that line, the first entry would
    carry the header off with it."""
    census = _SCRIPT.parent / "ui_pr_gate_census.txt"
    lines = census.read_text(encoding="utf-8").splitlines(keepends=True)
    blank = next(i for i, line in enumerate(lines) if not line.strip())
    assert blank > 0 and all(line.startswith("#") for line in lines[:blank]), (
        "the census must open with a comment header that a blank line ends"
    )
    entries = replay.read_census(census)
    reversed_text = replay._reordered(census, entries, entries[::-1])
    assert reversed_text.splitlines(keepends=True)[: blank + 1] == lines[: blank + 1]
    moved = [
        line.strip()
        for line in reversed_text.splitlines()
        if line.strip() and not line.startswith("#")
    ]
    assert moved == entries[::-1]


def _deal_seconds(**costs: float) -> dict[str, float]:
    return {f"Tests/UI/test_{name}.py": spent for name, spent in costs.items()}


def test_the_deal_puts_the_heaviest_shard_at_the_floor_and_keeps_census_order():
    """Largest-first fills the 2-entry shard with both 3 s files and leaves 7 s
    in the 3-entry one; a swap evens it to 6 and 6. Each shard's entries stay
    in census order, and shard s still takes entries s, s+2, ..."""
    seconds = _deal_seconds(a=3.0, b=2.0, c=3.01, d=2.01, e=2.02)
    entries = list(seconds)

    order = replay.dealt(entries, seconds, 2)

    loads = replay.shard_loads(order, seconds, 2)
    assert max(loads) < 6.1 and sorted(order) == sorted(entries)
    for shard in (order[0::2], order[1::2]):
        assert shard == sorted(shard, key=entries.index)
    assert replay.dealt(entries, seconds, 2) == order  # same logs, same census


def test_deal_write_moves_every_entry_with_its_comment(tmp_path, capsys):
    census = tmp_path / "census.txt"
    census.write_text(
        "# the header\n"
        "\n"
        "# what test_h.py pins\n"
        "Tests/UI/test_h.py\n"
        "# what test_x.py pins\n"
        "Tests/UI/test_x.py\n"
        "Tests/UI/test_t1.py\n"
        "Tests/UI/test_t2.py\n",
        encoding="utf-8",
    )
    argv = ["--shards", "2", "--deal", "--write", "--census", str(census)]
    run = str(_run_dir(tmp_path))  # h 102 s, t1 90 s, x 6 s, t2 6 s

    assert replay.main(argv + [run]) == 0

    written = census.read_text(encoding="utf-8")
    assert written == (  # test_x.py moves to shard 1 and takes its comment
        "# the header\n"
        "\n"
        "# what test_h.py pins\n"
        "Tests/UI/test_h.py\n"
        "Tests/UI/test_t1.py\n"
        "# what test_x.py pins\n"
        "Tests/UI/test_x.py\n"
        "Tests/UI/test_t2.py\n"
    )
    out = capsys.readouterr().out
    assert "2 shards, as listed:  3.2 0.2  max 3.2 min" in out
    assert "2 shards, dealt:      1.8 1.6  max 1.8 min" in out
    assert "re-dealt every entry of census.txt" in out
    assert replay.main(argv + [run]) == 0
    # test_x.py and test_t2.py tie at 6 s: ties go by name, not census order,
    # so a second deal over the same logs changes nothing.
    assert census.read_text(encoding="utf-8") == written


def test_main_refuses_runs_that_timed_no_file(tmp_path, capsys):
    (tmp_path / "empty-run").mkdir()
    census = tmp_path / "census.txt"
    census.write_text(_CENSUS, encoding="utf-8")

    code = replay.main(
        ["--shards", "2", "--census", str(census), str(tmp_path / "empty-run")]
    )

    assert code == 2
    assert "no shard logs" in capsys.readouterr().err


def test_repeated_file_occurrences_keep_their_own_end_timestamps():
    log = "\n".join(
        [
            "x 2026-10-20T00:00:00.0000000Z collected 3 items",
            "x 2026-10-20T00:00:10.0000000Z Tests/UI/test_a.py . [ 33%]",
            "x 2026-10-20T00:00:30.0000000Z Tests/UI/test_b.py . [ 66%]",
            "x 2026-10-20T00:01:00.0000000Z Tests/UI/test_a.py . [100%]",
        ]
    )
    assert replay.file_seconds(log) == {
        "Tests/UI/test_a.py": 40.0,
        "Tests/UI/test_b.py": 20.0,
    }


def test_run_timings_keep_all_node_phases_and_parameter_cases(tmp_path):
    log = "\n".join(
        [
            "2026-10-20T00:00:00.0000000Z collected 3 items",
            "2026-10-20T00:01:20.0000000Z Tests/UI/test_a.py ... [100%]",
            "2026-10-20T00:01:21.0000000Z 5.00s setup Tests/UI/test_a.py::test_slow[x]",
            "2026-10-20T00:01:21.0000000Z 20.00s call Tests/UI/test_a.py::test_slow[x]",
            "2026-10-20T00:01:21.0000000Z 5.00s teardown Tests/UI/test_a.py::test_slow[x]",
            "2026-10-20T00:01:21.0000000Z 30.00s call Tests/UI/test_a.py::test_slow[y]",
            "2026-10-20T00:01:21.0000000Z 10.00s call Tests/UI/test_a.py::test_fast",
        ]
    )
    (tmp_path / "shard.log").write_text(log, encoding="utf-8")
    seconds = replay.run_seconds(tmp_path)
    assert seconds["Tests/UI/test_a.py::test_slow[x]"] == 30.0
    assert replay.shard_loads(
        [
            "Tests/UI/test_a.py::test_slow",
            "Tests/UI/test_a.py::test_fast",
        ],
        seconds,
        2,
    ) == [65.0, 15.0]  # 10s file overhead shared, not 40/40


def test_partial_node_timings_only_share_the_unmeasured_file_remainder():
    assert replay.shard_loads(
        [
            "Tests/UI/test_a.py::test_slow",
            "Tests/UI/test_a.py::test_other",
            "Tests/UI/test_a.py::test_last",
        ],
        {
            "Tests/UI/test_a.py": 90.0,
            "Tests/UI/test_a.py::test_slow": 50.0,
        },
        3,
    ) == [50.0, 20.0, 20.0]


def test_a_parameter_entry_does_not_charge_its_sibling_cases():
    assert replay.shard_loads(
        [
            "Tests/UI/test_a.py::test_rows[small]",
            "Tests/UI/test_a.py::test_rows[large]",
        ],
        {
            "Tests/UI/test_a.py": 60.0,
            "Tests/UI/test_a.py::test_rows[small]": 10.0,
            "Tests/UI/test_a.py::test_rows[large]": 50.0,
        },
        2,
    ) == [10.0, 50.0]


def test_duration_rows_keep_spaces_inside_parameter_ids(tmp_path):
    log = "\n".join(
        [
            "2026-10-20T00:00:00.0000000Z collected 2 items",
            "2026-10-20T00:01:00.0000000Z Tests/UI/test_a.py .. [100%]",
            "2026-10-20T00:01:01.0000000Z 10.00s call Tests/UI/test_a.py::test_rows[small value]",
            "2026-10-20T00:01:01.0000000Z 50.00s call Tests/UI/test_a.py::test_rows[large value]",
        ]
    )
    (tmp_path / "shard.log").write_text(log, encoding="utf-8")
    assert replay.shard_loads(
        [
            "Tests/UI/test_a.py::test_rows[small value]",
            "Tests/UI/test_a.py::test_rows[large value]",
        ],
        replay.run_seconds(tmp_path),
        2,
    ) == [10.0, 50.0]
