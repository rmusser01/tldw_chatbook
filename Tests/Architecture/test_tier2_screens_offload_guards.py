"""Off-loop / off-thread guards for `UI/Screens/`, tier-2 review S18+S19 P2s.

Deliberately gate-free: every test here drives a screen method on a bare
instance (`cls.__new__`) with hand-built doubles, so none of them needs the
app fixture that `Backup_Recovery`'s ADR-126 recovery gate blocks in a clean
worktree. What they pin is thread/loop topology, which is exactly the
property that every end-state assertion in the mounted suites is blind to.
"""

from __future__ import annotations

import ast
import asyncio
import threading
from pathlib import Path
from typing import Any

import pytest

from tldw_chatbook.UI.Screens.scheduling.schedules_workbench import (
    SchedulesWorkbench,
)
from tldw_chatbook.UI.Screens.watchlists_collections_screen import (
    WatchlistsCollectionsScreen,
)


class _BareWatchlists(WatchlistsCollectionsScreen):
    """Shadow the `runtime_backend` reactive with a plain class attribute.

    A reactive is a data descriptor, so an instance attribute cannot shadow
    it; redefining it on a subclass can, which is what lets this run without
    a mounted Textual app.
    """

    runtime_backend = "local"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_item_status_write_does_not_spin_a_throwaway_event_loop() -> None:
    """S18 P2: `_update_item_status_off_loop` must await the controller on
    THIS loop.

    The service layer took itself off the loop when
    `Subscriptions/db_offload.run_db_off_loop` landed
    (`LocalWatchlistsService.update_item` -> `run_db_off_loop` ->
    `asyncio.to_thread`), so the screen's own
    `asyncio.to_thread(lambda: asyncio.run(...))` wrapper added a second
    thread and a throwaway event loop per keystroke in the reader. The
    end state -- status written, cell repainted -- is identical either way;
    only the running loop identity can tell the two apart.
    """
    screen = _BareWatchlists.__new__(_BareWatchlists)
    seen: dict[str, Any] = {}

    class _Controller:
        async def update_item_status(
            self, *, runtime_backend: str, item_id: Any, status: str
        ) -> dict[str, Any]:
            seen["loop"] = asyncio.get_running_loop()
            seen["thread"] = threading.get_ident()
            seen["args"] = (runtime_backend, item_id, status)
            return {"success": True, "status": status}

    screen._controller = _Controller()

    result = await screen._update_item_status_off_loop(item_id=7, status="ingested")

    assert result == {"success": True, "status": "ingested"}
    assert seen["args"] == ("local", 7, "ingested")
    assert seen["loop"] is asyncio.get_running_loop(), (
        "the controller chain must be awaited on the caller's loop -- a "
        "nested asyncio.run() in a worker thread builds a throwaway loop "
        "(and a throwaway executor) on top of the off-loop hop "
        "run_db_off_loop already performs"
    )
    assert seen["thread"] == threading.get_ident()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_orphaned_transfer_sweep_runs_off_the_event_loop_thread() -> None:
    """S19 P2: `_settle_orphaned_transfers` must not sweep on the loop.

    `SyncEngine._settle_orphaned_transfer_mutations` is a plain `def` doing
    `get_pending_mutations`/`set_transfer_state`. `run_worker(coroutine)`
    only reschedules onto the same loop -- it is not a thread -- so the
    docstring's "deferred onto the fire-and-forget worker path" moved the
    ORDERING off `on_mount` and left the blocking work exactly where it was,
    against the file's own stated `asyncio.to_thread` discipline for every
    `service.db.*` read.
    """
    loop_thread_id = threading.get_ident()
    sweep_thread_ids: list[int] = []
    dispatched: list[Any] = []

    class _SyncEngine:
        def _settle_orphaned_transfer_mutations(self, target_owner: Any) -> None:
            sweep_thread_ids.append(threading.get_ident())

    class _Service:
        owner_id = "local"
        sync_engine = _SyncEngine()

    class _Bench(SchedulesWorkbench):
        def _service(self) -> Any:
            return _Service()

        def _active_server_id(self) -> Any:
            return None

        def run_worker(self, work: Any, **kwargs: Any) -> Any:
            dispatched.append(work)
            return None

    bench = _Bench.__new__(_Bench)
    bench._settle_orphaned_transfers()

    assert dispatched, "the sweep must still be dispatched as a worker"
    await dispatched[0]()

    assert sweep_thread_ids, "the sweep must have run at all"
    assert sweep_thread_ids[0] != loop_thread_id, (
        "the synchronous DB sweep must run on a worker thread "
        "(asyncio.to_thread), not inline on the event loop"
    )


def _function_named(tree: ast.AST, name: str) -> ast.AST:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"{name} not found; this guard's anchor is stale.")


@pytest.mark.unit
def test_skill_eval_run_persists_its_66_rows_off_the_event_loop() -> None:
    """S18 P2: the skill-eval persistence block must sit on a thread.

    `save_artifact` -> `EvalsDB.store_result` commits one transaction per
    call, and a deep run writes 16 judge cells plus `deep_sim_total` (50)
    sim cells plus the report. Measured against a file-backed `EvalsDB`,
    66 serial `store_result` calls cost 108-137 ms. `_run_skill_eval_worker`
    is dispatched as a bare coroutine (no `thread=True`), so every one of
    those commits ran on the event loop.

    Structural rather than behavioural on purpose: driving the worker end to
    end needs a bench, a subject, two targets, a provider and a runner, none
    of which this defect depends on. What the defect IS, exactly, is a
    persistence call reachable on the worker's own frame.
    """
    source = (
        Path(__file__).resolve().parents[2]
        / "tldw_chatbook"
        / "UI"
        / "Screens"
        / "evals_screen.py"
    ).read_text(encoding="utf-8")
    worker = _function_named(ast.parse(source), "_run_skill_eval_worker")

    owners: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(worker):
        for child in ast.iter_child_nodes(node):
            owners[child] = node

    def _enclosing_function(node: ast.AST) -> ast.AST:
        cursor = owners.get(node)
        while cursor is not None:
            if isinstance(cursor, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return cursor
            cursor = owners.get(cursor)
        return worker

    on_the_loop = [
        node.lineno
        for node in ast.walk(worker)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"save_artifact", "save_report"}
        and _enclosing_function(node) is worker
    ]
    assert not on_the_loop, (
        "save_artifact/save_report called directly on _run_skill_eval_worker's "
        f"own frame at line(s) {on_the_loop}; each is a separate committed "
        "sqlite transaction and the worker is a coroutine, not a thread. "
        "Persist inside the closure handed to asyncio.to_thread."
    )

    offloaded = [
        node
        for node in ast.walk(worker)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "to_thread"
    ]
    assert offloaded, "the persistence closure must be run via asyncio.to_thread"


class _RecordingStatic:
    """A `Static` stand-in that counts real repaints."""

    def __init__(self) -> None:
        self.content = ""
        self.updates = 0

    def update(self, text: Any) -> None:
        self.updates += 1
        self.content = text


class _FakeTimer:
    def __init__(self) -> None:
        self.paused = 0
        self.resumed = 0

    def pause(self) -> None:
        self.paused += 1

    def resume(self) -> None:
        self.resumed += 1


def _bare_backup_screen(status: _RecordingStatic) -> Any:
    from tldw_chatbook.UI.Screens.backup_restore_screen import BackupRestoreScreen

    screen = BackupRestoreScreen.__new__(BackupRestoreScreen)

    class _Cancel:
        disabled = False

    cancel = _Cancel()

    def _query_one(selector: str, _type: Any = None) -> Any:
        return status if selector == "#backup-status" else cancel

    screen.query_one = _query_one  # type: ignore[method-assign]
    screen.service = _FakeBackupService()
    screen._summary_requested = None
    screen._dismissed_inspection_id = None
    screen._review_codes_seen = ()
    screen._restore_review_codes_seen = ()
    screen._later_review_codes_seen = ()
    screen._requested_backup_operation = None
    screen._requested_restore_operation = None
    screen._requested_rollback_operation = None
    screen._rollback_selection = None
    screen._mode = "copies"
    screen._last_terminal = None
    screen._revision = 0
    return screen


class _FakeBackupService:
    def current(self) -> dict[str, Any]:
        return {
            "kind": "backup",
            "state": "running",
            "phase": "writing_archive",
            "operation_id": "op-1",
            "issues": (),
            "review_issues": (),
            "result": {},
        }

    def issue_message(self, code: str) -> str:  # pragma: no cover - unused here
        return code


@pytest.mark.unit
def test_backup_status_poll_repaints_only_when_the_text_changes() -> None:
    """S19 P2: the 0.2 s poller must not re-`update()` identical copy.

    `Static.update` ends in `refresh(layout=True)` by default, so an
    unconditional write laid the screen out five times a second for the
    life of the screen. `SchedulesWorkbench._update_static_content` is the
    same rule, in the same slice.
    """
    status = _RecordingStatic()
    screen = _bare_backup_screen(status)

    screen._refresh_status()
    assert status.updates == 1, "the first tick must paint"
    assert status.content == "Running: writing archive"

    for _ in range(10):
        screen._refresh_status()
    assert status.updates == 1, (
        "ten further ticks with identical copy must not repaint; "
        f"saw {status.updates} update() calls"
    )


@pytest.mark.unit
def test_backup_status_poll_pauses_while_the_screen_is_covered() -> None:
    """S19 P2: the poll must stop while a pushed modal covers the screen.

    Textual SUSPENDS an installed/covered screen rather than unmounting it,
    and this screen pushes `FileOpen`/`FileSave`/`SelectDirectory` over
    itself, so the poll ticked behind an opaque screen (TASK-23022).
    """
    status = _RecordingStatic()
    screen = _bare_backup_screen(status)
    screen._poller = _FakeTimer()

    screen.on_screen_suspend()
    assert screen._poller.paused == 1
    assert screen._poller.resumed == 0

    screen.on_screen_resume()
    assert screen._poller.resumed == 1
    assert status.updates == 1, "resume must repaint immediately, not wait a tick"
