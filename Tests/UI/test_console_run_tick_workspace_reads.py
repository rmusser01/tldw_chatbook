"""The Console 5 Hz run tick must not touch the Workspace DB.

TASK-22201. The holistic perf review (Docs/Design/2026-08-24-holistic-perf-review.md,
finding 22201) measured that PR #2034 reintroduced the TASK-21118 hot-path shape on
the run tick: ``_sync_native_console_chat_ui`` builds
``_build_console_workspace_context_state()`` three times per 0.2 s tick, each build
reaching ``_console_browser_workspace_records()`` twice (browser labels + the new
``workspace_tree_projection``), and each of those calls ran
``ensure_default_workspace()`` (SELECT + bindings probe + occasional DELETE write
transaction) plus ``list_workspaces()`` — synchronous SQLite on the event loop,
roughly 45 extra statements/second while a reply streams — plus the O(materialized
rows) merge/canonical-owner/overlay pipeline twice per build.

These tests replicate the review's counter probe as a permanent gate, the same
shape as ``test_console_keystroke_workspace_reads.py`` (TASK-21118):

* zero registry-service reads AND zero WorkspaceDB round-trips AND zero traced SQL
  statements across settled run ticks;
* a control proving the counters still see real registry traffic;
* the staleness guard: a registry mutation performed directly on the service (no
  Console seam) must be reflected by the very next tick;
* the row merge / canonical-owner / overlay pipeline runs at most once per state
  build, and a settled tick builds the workspace context state at most once.
"""

from __future__ import annotations

import time

import pytest

from Tests.UI.test_console_dictation import _mounted_console, _ready_host
from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.Workspaces.registry_service import LocalWorkspaceRegistryService

APP_SIZE = (140, 42)

#: Enough ticks that per-tick work cannot hide in warmup noise; at the real
#: cadence this window is one second of streaming.
SETTLED_TICKS = 5


async def _settled_console(host, pilot):
    """Mount the ready Console and let its own sync passes settle.

    The mount's first passes may legitimately read the registry to warm the
    generation-keyed memos; one explicit tick afterwards guarantees the memo
    is warm before a counter window opens.
    """
    console = await _mounted_console(host, pilot)
    for _ in range(3):
        await pilot.pause()
    await console._sync_native_console_chat_ui()
    for _ in range(2):
        await pilot.pause()
    return console


class _WorkspaceReadCounter:
    """Count registry-service reads, WorkspaceDB round-trips, and statements.

    Patches at the class so every instance-bound call the app makes is
    counted, and counts THREE layers: the display-read service methods, the
    ``WorkspaceDB.connection`` / ``WorkspaceDB.transaction`` context
    factories, and — via ``sqlite3``'s own trace callback installed on every
    connection those factories yield — the literal SQL statements executed
    (the review's probe recipe). Restored on exit, including the trace
    callbacks.
    """

    _READ_METHODS = (
        "ensure_default_workspace",
        "get_active_workspace",
        "list_workspaces",
        "list_runtime_bindings",
        "list_workspace_memberships",
    )

    def __init__(self) -> None:
        self.read_calls: dict[str, int] = {name: 0 for name in self._READ_METHODS}
        self.db_connections = 0
        self.db_transactions = 0
        self.statements: list[str] = []
        self._original_reads = {
            name: getattr(LocalWorkspaceRegistryService, name)
            for name in self._READ_METHODS
        }
        self._original_connection = WorkspaceDB.connection
        self._original_transaction = WorkspaceDB.transaction
        self._traced_connections: list = []

    def __enter__(self) -> "_WorkspaceReadCounter":
        counter = self

        def _counting_read(name, original):
            def wrapper(service, *args, **kwargs):
                counter.read_calls[name] += 1
                return original(service, *args, **kwargs)

            return wrapper

        for name, original in self._original_reads.items():
            setattr(
                LocalWorkspaceRegistryService, name, _counting_read(name, original)
            )

        original_connection = self._original_connection
        original_transaction = self._original_transaction

        def _trace(statement: str) -> None:
            counter.statements.append(statement)

        def _tracing_context(original, attr):
            from contextlib import contextmanager

            @contextmanager
            def wrapper(db):
                setattr(counter, attr, getattr(counter, attr) + 1)
                with original(db) as conn:
                    conn.set_trace_callback(_trace)
                    if conn not in counter._traced_connections:
                        counter._traced_connections.append(conn)
                    try:
                        yield conn
                    finally:
                        conn.set_trace_callback(None)

            return wrapper

        WorkspaceDB.connection = _tracing_context(
            original_connection, "db_connections"
        )
        WorkspaceDB.transaction = _tracing_context(
            original_transaction, "db_transactions"
        )
        return self

    def __exit__(self, *_exc) -> None:
        for name, original in self._original_reads.items():
            setattr(LocalWorkspaceRegistryService, name, original)
        WorkspaceDB.connection = self._original_connection
        WorkspaceDB.transaction = self._original_transaction
        for conn in self._traced_connections:
            try:
                conn.set_trace_callback(None)
            except Exception:
                pass

    @property
    def registry_reads(self) -> int:
        return sum(self.read_calls.values())

    @property
    def db_round_trips(self) -> int:
        return self.db_connections + self.db_transactions


# ---------------------------------------------------------------------------
# 1. Settled run ticks: zero Workspace-DB work
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_settled_run_ticks_do_zero_workspace_registry_sql():
    """The review's probe, as a gate: settled ticks -> 0 registry SQL.

    On the pre-fix code each tick counts ~6 ``ensure_default_workspace`` +
    ~6 ``list_workspaces`` + 3 x (``get_active_workspace`` +
    ``list_runtime_bindings`` + ``list_workspaces`` +
    ``list_workspace_memberships``) and their SQLite statements.
    """
    _, host = _ready_host()
    async with host.run_test(size=APP_SIZE) as pilot:
        console = await _settled_console(host, pilot)

        with _WorkspaceReadCounter() as counter:
            started = time.perf_counter()
            for _ in range(SETTLED_TICKS):
                await console._sync_native_console_chat_ui()
                await pilot.pause()
            elapsed = time.perf_counter() - started

        # Measurement metric for the task record; the assertions carry the gate.
        print(
            f"\n[t22201] {SETTLED_TICKS} settled ticks: "
            f"{elapsed * 1000.0:.1f} ms wall, "
            f"reads={counter.read_calls}, "
            f"round_trips={counter.db_round_trips}, "
            f"statements={len(counter.statements)}"
        )
        assert counter.read_calls["ensure_default_workspace"] == 0
        assert counter.registry_reads == 0
        assert counter.db_round_trips == 0
        assert counter.statements == []


@pytest.mark.asyncio
async def test_the_counter_still_sees_real_registry_traffic():
    """Control: "zero" above must not be satisfiable by an unwired counter."""
    app, host = _ready_host()
    async with host.run_test(size=APP_SIZE) as pilot:
        await _settled_console(host, pilot)
        registry = app.workspace_registry_service
        assert registry is not None

        with _WorkspaceReadCounter() as counter:
            active = registry.get_active_workspace()
            listed = registry.list_workspaces()
            registry.ensure_default_workspace()

        assert active is not None
        assert listed
        assert counter.read_calls["get_active_workspace"] >= 1
        assert counter.read_calls["list_workspaces"] >= 1
        assert counter.read_calls["ensure_default_workspace"] == 1
        assert counter.db_round_trips >= 3
        assert any("workspace_records" in s for s in counter.statements)


# ---------------------------------------------------------------------------
# 2. The memo must not go stale
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_registry_mutation_reflects_in_the_next_tick():
    """A mutation with no Console seam must reach the next tick's state.

    The create/rename/set-active trio is performed directly on the registry
    service — the way Settings or Library do it — so only the memo's own
    ``mutation_generation`` revalidation can keep the Console truthful.
    """
    app, host = _ready_host()
    async with host.run_test(size=APP_SIZE) as pilot:
        console = await _settled_console(host, pilot)
        registry = app.workspace_registry_service
        assert registry is not None
        controller = console._workspace

        before_records = controller._console_browser_workspace_records()
        assert all(
            record.workspace_id != "workspace-gamma" for record in before_records
        )

        registry.create_workspace(workspace_id="workspace-gamma", name="Gamma")
        registry.set_active_workspace("workspace-gamma")

        await console._sync_native_console_chat_ui()
        await pilot.pause()

        after_records = controller._console_browser_workspace_records()
        assert any(
            record.workspace_id == "workspace-gamma" for record in after_records
        )
        context = controller._current_console_workspace_context()
        assert context.active_workspace_id == "workspace-gamma"
        state = controller._build_console_workspace_context_state()
        assert state.active_workspace_id == "workspace-gamma"
        assert state.workspace_name == "Gamma"

        registry.rename_workspace("workspace-gamma", "Gamma Renamed")
        await console._sync_native_console_chat_ui()
        await pilot.pause()
        renamed = controller._build_console_workspace_context_state()
        assert renamed.workspace_name == "Gamma Renamed"


# ---------------------------------------------------------------------------
# 3. The row pipeline runs once per build; a settled tick builds once
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_row_pipeline_runs_once_per_state_build():
    """canonical-owner + overlay must each run once for a no-query build."""
    _, host = _ready_host()
    async with host.run_test(size=APP_SIZE) as pilot:
        console = await _settled_console(host, pilot)
        controller = console._workspace

        calls = {"canonical": 0, "overlay": 0}
        original_canonical = controller._rows_with_latest_canonical_owner
        original_overlay = controller._overlay_current_console_browser_markers

        def counting_canonical(rows):
            calls["canonical"] += 1
            return original_canonical(rows)

        def counting_overlay(rows, current_conversation_id=None):
            calls["overlay"] += 1
            return original_overlay(rows, current_conversation_id)

        controller._rows_with_latest_canonical_owner = counting_canonical
        controller._overlay_current_console_browser_markers = counting_overlay
        try:
            controller._build_console_workspace_context_state()
        finally:
            del controller._rows_with_latest_canonical_owner
            del controller._overlay_current_console_browser_markers

        assert calls["canonical"] == 1, calls
        assert calls["overlay"] == 1, calls


@pytest.mark.asyncio
async def test_settled_tick_builds_workspace_state_once(monkeypatch):
    """One settled tick must pay for the context build at most once.

    Counted at the uncached seam (``build_console_workspace_state`` as
    imported by the workspace controller module) rather than the memoized
    front method, which the tick legitimately CALLS several times -- the
    stack probe that motivated the tick scope measured six calls per tick
    (rail states x2, workspace push, control bar + agent section inspector
    legs, settings summary).
    """
    from tldw_chatbook.UI.Console_Modules import workspace as workspace_module

    _, host = _ready_host()
    async with host.run_test(size=APP_SIZE) as pilot:
        console = await _settled_console(host, pilot)

        builds = {"count": 0}
        original_build = workspace_module.build_console_workspace_state

        def counting_build(**kwargs):
            builds["count"] += 1
            return original_build(**kwargs)

        monkeypatch.setattr(
            workspace_module, "build_console_workspace_state", counting_build
        )
        await console._sync_native_console_chat_ui()

        assert builds["count"] <= 1, builds


@pytest.mark.asyncio
async def test_tick_scope_rebuilds_when_registry_changes_mid_scope():
    """Inputs changing inside one tick scope must rebuild, not reuse."""
    app, host = _ready_host()
    async with host.run_test(size=APP_SIZE) as pilot:
        console = await _settled_console(host, pilot)
        registry = app.workspace_registry_service
        assert registry is not None
        controller = console._workspace

        with controller.tick_workspace_build_scope():
            first = controller._build_console_workspace_context_state()
            again = controller._build_console_workspace_context_state()
            assert again is first

            registry.create_workspace(workspace_id="workspace-delta", name="Delta")
            registry.set_active_workspace("workspace-delta")

            refreshed = controller._build_console_workspace_context_state()
            assert refreshed is not first
            assert refreshed.active_workspace_id == "workspace-delta"
