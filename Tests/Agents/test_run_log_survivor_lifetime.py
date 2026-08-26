# Tests/Agents/test_run_log_survivor_lifetime.py
"""A survivor's run-log records belong to ITS OWN run tree, forever.

PR3a-1 Task 3. Since Task 2 a sub-agent that is still working when its
turn returns KEEPS RUNNING -- and keeps emitting run-log records long
after `run_turn` came back. Where those records land is this file's
subject.

The bug this file was written against is NOT "a survivor appends to a
CLOSED writer" (measured: `close()` only fsyncs the final segment, it
does not deactivate the writer -- see
``test_closing_a_writer_does_not_stop_a_survivor_from_appending`` below,
which is why merely DEFERRING `close()` would have fixed nothing). It is
that `on_record` used to read `self.run_log_writer` **at call time**,
while `run_turn` REPLACES that attribute with a fresh writer bound to the
next turn's primary. A survivor therefore recorded through whatever
writer the service happened to be holding when it got to its model call:

- turn 1's child wrote nothing at all into turn 1's tree (its "Full run
  log" in the Console renders empty -- `load_run_log_text` filters the
  OWNING primary's directory by the child's run id and finds none), and
- turn 2's tree carried a FOREIGN run's records, reachable through
  `search_run_log`/`run_log_slice` scoped to that tree -- the exact
  inverse of the isolation `test_run_log_sandbox_isolation.py` and
  `test_run_log_workspace_isolation.py` defend.

The fix resolves the writer PER RUN (passed down at spawn, on the
parent's thread, so no scheduling race can hand a child the next turn's
writer) instead of reading it off the service. These tests pin the
observable consequence, not the mechanism: a survivor's records are in
its own tree, absent from every other tree, and still invisible to
`grep_files` when they are written after the turn ended.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from tldw_chatbook.Agents import run_log as run_log_module
from tldw_chatbook.Agents.agent_models import (
    RUN_DONE,
    RUN_RUNNING,
    SPAWN_TOOL_NAME,
)
from tldw_chatbook.Agents.run_log import (
    MANIFEST_NAME,
    RunLogWriter,
    resolve_existing_log_dir,
)
from tldw_chatbook.Agents.run_log_search import load_records
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.Tools.file_operation_tools import GrepFiles

from Tests.Agents.test_agent_service import fence
from Tests.Agents.test_fleet_runtime import FLEET_CFG, make_fleet_service

#: What the survivor answers. Distinctive enough that finding it in the
#: wrong tree is unambiguous.
SURVIVOR_MARKER = "SURVIVOR_RECORD_4c1a late answer"

_JOIN_TIMEOUT = 10.0


@pytest.fixture()
def db(tmp_path):
    return AgentRunsDB(tmp_path / "runs.db", client_id="test")


@pytest.fixture()
def log_root(tmp_path, monkeypatch):
    """Point every writer this test builds at a private root."""
    monkeypatch.setattr(run_log_module, "resolve_log_root", lambda: tmp_path)
    return tmp_path


def _records_of(run_id: str):
    """Every record in ``run_id``'s own run-tree directory.

    Resolved through the SAME production reader the Console's run-log
    viewer uses (`resolve_existing_log_dir` + `load_records`), so these
    assertions are about what a user can actually see, not about a path
    the test computed for itself.
    """
    log_dir = resolve_existing_log_dir(run_id)
    if log_dir is None:
        return []
    return load_records(log_dir)


def _gated_child(entered: threading.Event, release: threading.Event, reply: str):
    """A child provider reply blocked until the test releases it."""

    def child():
        entered.set()
        if not release.wait(_JOIN_TIMEOUT):
            raise AssertionError("child was never released by the test")
        return reply

    return child


def _after(entered: threading.Event, reply: str):
    """A parent reply held until the child is provably at its model call."""

    def parent():
        if not entered.wait(_JOIN_TIMEOUT):
            raise AssertionError("the child never reached its model call")
        return reply

    return parent


def _turn(service, config=FLEET_CFG):
    return service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "go"}],
        config=config,
        api_endpoint="llama_cpp",
    )


def _child_run_id(db) -> str:
    rows = [
        row
        for row in db.list_runs("c", include_superseded=True)
        if row["agent_kind"] == "subagent"
    ]
    assert len(rows) == 1, f"expected exactly one sub-agent row, got {len(rows)}"
    return rows[0]["id"]


def _join(threads, timeout=_JOIN_TIMEOUT):
    for thread in threads:
        thread.join(timeout)
        assert not thread.is_alive(), f"{thread.name} never finished"


def test_a_survivors_records_stay_out_of_the_next_turns_tree(db, log_root):
    """THE reproduction: turn 2's tree must not carry turn 1's child.

    Turn 1 leaves a child gated at its model call. Turn 2 runs to
    completion -- replacing `self.run_log_writer` on the way in -- and only
    then is the child released. Before the fix its `model` record was
    appended to turn 2's writer, landing in turn 2's directory tagged with
    turn 1's child run id.
    """
    entered = threading.Event()
    release = threading.Event()
    service, _chat, coordinator = make_fleet_service(
        db,
        [
            fence(SPAWN_TOOL_NAME, {"task": "slow task"}),
            _after(entered, "turn one answer"),
            "turn two answer",
        ],
        {"slow task": [_gated_child(entered, release, SURVIVOR_MARKER)]},
    )
    try:
        turn_one_id, outcome_one = _turn(service)
        assert outcome_one.status == RUN_DONE
        assert entered.is_set(), "precondition: the child reached its model call"
        child_id = _child_run_id(db)
        # Captured BEFORE turn 2: `run_turn` resets `_fleet_threads`, and a
        # survivor is deliberately dropped from it.
        survivor_threads = list(service._fleet_threads.values())
        assert survivor_threads, "precondition: the child is on its own thread"

        turn_two_id, outcome_two = _turn(service)
        assert outcome_two.status == RUN_DONE
        assert turn_two_id != turn_one_id
    finally:
        release.set()
    _join(survivor_threads)
    assert coordinator.all_finished()

    # 1. Turn 2's tree holds ONLY turn 2's own primary records.
    turn_two_records = _records_of(turn_two_id)
    assert turn_two_records, "turn 2 logged nothing at all -- test is vacuous"
    foreign = [r for r in turn_two_records if r.run_id != turn_two_id]
    assert foreign == [], (
        "turn 2's run tree carries a foreign run's records: "
        f"{[(r.run_id, r.kind, r.content[:40]) for r in foreign]}"
    )
    assert not any(SURVIVOR_MARKER in r.content for r in turn_two_records)

    # 2. Turn 1's child log is not empty -- its records are in ITS tree.
    child_records = [r for r in _records_of(turn_one_id) if r.run_id == child_id]
    assert child_records, (
        "the survivor's records vanished from its own run tree -- this is "
        "what makes the Console's 'Full run log' render empty for the child"
    )
    assert any(SURVIVOR_MARKER in r.content for r in child_records)
    assert {r.kind for r in child_records} == {"subagent"}


def test_a_survivor_recording_DURING_the_next_turn_still_files_its_own_tree(
    db, log_root
):
    """The sharpest window: both trees live and appending at once.

    Turn 2's own provider call releases the survivor and does not answer
    until it has finished, so the survivor's append happens while turn 2's
    writer is the one on the service -- exactly the moment the old
    call-time lookup misfiled into. Neither tree may pick up the other's
    records.
    """
    entered = threading.Event()
    release = threading.Event()

    def release_and_wait():
        release.set()
        deadline = time.monotonic() + _JOIN_TIMEOUT
        while time.monotonic() < deadline:
            if coordinator.all_finished():
                return "turn two answer"
            time.sleep(0.01)
        raise AssertionError("the released survivor never finished")

    service, _chat, coordinator = make_fleet_service(
        db,
        [
            fence(SPAWN_TOOL_NAME, {"task": "slow task"}),
            _after(entered, "turn one answer"),
            release_and_wait,
        ],
        {"slow task": [_gated_child(entered, release, SURVIVOR_MARKER)]},
    )
    try:
        turn_one_id, outcome_one = _turn(service)
        assert outcome_one.status == RUN_DONE
        child_id = _child_run_id(db)
        survivor_threads = list(service._fleet_threads.values())

        turn_two_id, outcome_two = _turn(service)
        assert outcome_two.status == RUN_DONE
    finally:
        release.set()
    _join(survivor_threads)

    assert [r.run_id for r in _records_of(turn_two_id)] == [turn_two_id]
    child_records = [r for r in _records_of(turn_one_id) if r.run_id == child_id]
    assert any(SURVIVOR_MARKER in r.content for r in child_records)


def test_a_survivors_record_numbers_stay_unique_within_its_own_tree(db, log_root):
    """One counter per tree, still, with a survivor writing after the turn.

    A "give the survivor its own writer" fix would have satisfied the two
    tests above while silently restarting record numbering inside turn 1's
    directory -- `run_log_slice`'s `from_record`/`to_record` addressing and
    the truncation trailer's "see record N" pointer both assume one
    monotonic counter per tree, so a duplicate number is a real defect.
    """
    entered = threading.Event()
    release = threading.Event()
    service, _chat, _coordinator = make_fleet_service(
        db,
        [
            fence(SPAWN_TOOL_NAME, {"task": "slow task"}),
            _after(entered, "turn one answer"),
            "turn two answer",
        ],
        {"slow task": [_gated_child(entered, release, SURVIVOR_MARKER)]},
    )
    try:
        turn_one_id, _outcome = _turn(service)
        survivor_threads = list(service._fleet_threads.values())
        _turn(service)
    finally:
        release.set()
    _join(survivor_threads)

    numbers = [r.number for r in _records_of(turn_one_id)]
    assert numbers == sorted(set(numbers)), (
        f"record numbers collided or went backwards in turn 1's tree: {numbers}"
    )


def test_closing_a_writer_does_not_stop_a_survivor_from_appending(log_root):
    """Why deferring `close()` was never the fix, pinned as a fact.

    `RunLogWriter.close()` fsyncs the final segment and nothing else: it
    leaves `is_active` true and every later `append` lands normally
    (records are written with a fresh `open(..., "ab")` per record, so
    there is no descriptor to close in the first place). The records a
    survivor lost were never lost to closure -- they were written to a
    DIFFERENT writer.
    """
    writer = RunLogWriter()
    writer.bind("run-1")
    assert writer.is_active
    writer.append(run_id="run-1", kind="primary", type="model", content="before")
    writer.write_manifest({"run_id": "run-1"})
    writer.close()

    assert writer.is_active, "close() must not deactivate the writer"
    number = writer.append(
        run_id="child-1", kind="subagent", type="model", content="after close"
    )
    assert number is not None, "an append after close() must still be recorded"
    contents = [r.content for r in load_records(Path(writer.log_dir))]
    assert contents == ["before", "after close"]


def test_a_survivors_post_turn_records_stay_hidden_from_grep_files(
    db, tmp_path, monkeypatch
):
    """The isolation property, re-asserted with a survivor in flight.

    `test_run_log_sandbox_isolation.py` proves a sub-agent cannot
    `grep_files` its parent's log because `bind()` dots the directory. A
    survivor writes AFTER its turn ended, through a writer nobody is
    holding any more -- so that guarantee has to be re-established for the
    records it writes then, not inherited from the ones written during the
    turn. Same plant-a-secret-and-grep shape as that file.
    """
    import tldw_chatbook.Tools.file_operation_tools as file_tools
    import tldw_chatbook.Tools.workspace_file_roots as ws_roots

    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    monkeypatch.setattr(file_tools, "_tool_sandbox_root", lambda: sandbox)
    monkeypatch.setattr(
        ws_roots,
        "allowed_file_roots",
        lambda write=False, sandbox_root=None: (sandbox,),
    )

    import asyncio

    def grep(pattern: str) -> list[dict]:
        return asyncio.run(GrepFiles().execute(pattern=pattern)).get("matches", [])

    # Positive control: grep_files really does scan this sandbox.
    (sandbox / "control.txt").write_text("CONTROL_MARKER_7d3f\n", encoding="utf-8")
    assert grep("CONTROL_MARKER_7d3f"), (
        "positive control failed: grep_files must find visible sandbox content"
    )

    secret = "SURVIVOR_SECRET_API_KEY=sk-live-survivor1"
    entered = threading.Event()
    release = threading.Event()
    service, _chat, _coordinator = make_fleet_service(
        db,
        [
            fence(SPAWN_TOOL_NAME, {"task": "slow task"}),
            _after(entered, "turn one answer"),
            "turn two answer",
        ],
        {"slow task": [_gated_child(entered, release, f"noting {secret}")]},
    )
    try:
        turn_one_id, _outcome = _turn(service)
        child_id = _child_run_id(db)
        survivor_threads = list(service._fleet_threads.values())
        _turn(service)
    finally:
        release.set()
    _join(survivor_threads)

    # The record really was written (or the grep below proves nothing), but
    # capture-time privacy removes the credential before it reaches disk.
    child_records = [r for r in _records_of(turn_one_id) if r.run_id == child_id]
    assert child_records, (
        "the survivor's post-turn record was never written -- a silent drop "
        "is not an acceptable answer to misfiling"
    )
    assert all(secret not in record.content for record in child_records)
    assert all("sk-live-survivor1" not in record.content for record in child_records)
    assert grep("SURVIVOR_SECRET_API_KEY") == [], (
        "a survivor's post-turn records must stay as unreadable to "
        "grep_files as the ones written during the turn"
    )


def test_the_next_turn_binds_its_own_tree_while_a_survivor_writes(db, log_root):
    """Two turns, two directories, and the survivor in exactly one of them.

    The structural half of the property: a writer is scoped to ONE run
    tree (`bind()` latches permanently), so the fix must not be "reuse
    turn 1's writer for turn 2" -- that would append turn 2's records into
    turn 1's directory and overwrite its manifest.
    """
    entered = threading.Event()
    release = threading.Event()
    service, _chat, _coordinator = make_fleet_service(
        db,
        [
            fence(SPAWN_TOOL_NAME, {"task": "slow task"}),
            _after(entered, "turn one answer"),
            "turn two answer",
        ],
        {"slow task": [_gated_child(entered, release, SURVIVOR_MARKER)]},
    )
    try:
        turn_one_id, _one = _turn(service)
        survivor_threads = list(service._fleet_threads.values())
        turn_two_id, _two = _turn(service)
    finally:
        release.set()
    _join(survivor_threads)

    one_dir = resolve_existing_log_dir(turn_one_id)
    two_dir = resolve_existing_log_dir(turn_two_id)
    assert one_dir is not None and two_dir is not None
    assert one_dir != two_dir
    # Each manifest names its own run, and turn 1's was not overwritten.
    import json

    assert json.loads((one_dir / MANIFEST_NAME).read_text())["run_id"] == turn_one_id
    assert json.loads((two_dir / MANIFEST_NAME).read_text())["run_id"] == turn_two_id
    assert {r.run_id for r in load_records(two_dir)} == {turn_two_id}


def test_a_survivors_row_is_running_while_it_logs(db, log_root):
    """Sanity: these tests really are exercising a SURVIVOR.

    Without this the whole file could pass against a build where the child
    settled inside the turn (which files its records correctly for the
    boring reason that no writer swap ever happened in between).
    """
    entered = threading.Event()
    release = threading.Event()
    service, _chat, _coordinator = make_fleet_service(
        db,
        [
            fence(SPAWN_TOOL_NAME, {"task": "slow task"}),
            _after(entered, "turn one answer"),
        ],
        {"slow task": [_gated_child(entered, release, SURVIVOR_MARKER)]},
    )
    try:
        _turn(service)
        rows = [
            row
            for row in db.list_runs("c", include_superseded=True)
            if row["agent_kind"] == "subagent"
        ]
        assert [row["status"] for row in rows] == [RUN_RUNNING]
        survivor_threads = list(service._fleet_threads.values())
    finally:
        release.set()
    _join(survivor_threads)


def test_a_child_scheduled_after_the_next_turn_begins_files_its_own_tree(
    db, log_root
):
    """The race the spawn-time capture closes, made deterministic.

    Capturing the writer at `_run_one`'s ENTRY is not enough on its own:
    `spawn` returns the moment the thread is started, and nothing
    guarantees that thread runs a single line before `run_turn` returns and
    the NEXT turn replaces `self.run_log_writer`. Here the child's
    `_run_one` is held until turn 2 is already in flight -- so a fix that
    resolved the writer on the CHILD's thread would file turn 1's child
    into turn 2's tree, exactly as the original bug did. The writer is
    instead captured on the parent's thread at spawn and passed down.
    """
    scheduled = threading.Event()
    finished = threading.Event()

    def turn_two_parent():
        # Turn 2's writer is already bound by the time a provider call
        # happens, so releasing the child here puts its whole run inside
        # the window where `self.run_log_writer` is the WRONG writer.
        scheduled.set()
        if not finished.wait(_JOIN_TIMEOUT):
            raise AssertionError("the late child never finished")
        return "turn two answer"

    service, _chat, _coordinator = make_fleet_service(
        db,
        [
            fence(SPAWN_TOOL_NAME, {"task": "slow task"}),
            "turn one answer",
            turn_two_parent,
        ],
        {"slow task": [SURVIVOR_MARKER]},
    )
    real_run_one = service._run_one

    def late_run_one(**kwargs):
        if kwargs.get("agent_kind") == "subagent":
            if not scheduled.wait(_JOIN_TIMEOUT):
                raise AssertionError("turn 2 never started")
        try:
            return real_run_one(**kwargs)
        finally:
            if kwargs.get("agent_kind") == "subagent":
                finished.set()

    service._run_one = late_run_one

    turn_one_id, outcome_one = _turn(service)
    assert outcome_one.status == RUN_DONE
    survivor_threads = list(service._fleet_threads.values())
    turn_two_id, outcome_two = _turn(service)
    assert outcome_two.status == RUN_DONE
    _join(survivor_threads)

    child_id = _child_run_id(db)
    assert [r.run_id for r in _records_of(turn_two_id)] == [turn_two_id]
    assert any(
        SURVIVOR_MARKER in r.content
        for r in _records_of(turn_one_id)
        if r.run_id == child_id
    )
