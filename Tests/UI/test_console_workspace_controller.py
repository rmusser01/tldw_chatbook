"""Characterisation + boundary tests for the Console workspace cluster.

Originally written before the wave-2 Task 2 extraction of `ConsoleWorkspace
Controller` (`tldw_chatbook/UI/Console_Modules/workspace.py`) landed, driving
the resume flow and the conversation-search debounce through REAL
interactions against the (at the time) still-monolithic `ChatScreen` -- the
same "real production coroutine, not a rebuilt double" discipline
`test_console_native_chat_flow.py`'s own resume coverage uses. Search-token/
error state is exactly where a snapshot-vs-live binding bug would hide (see
wave 1's `ConsoleDictationController` review history), so these assert the
token/error lifecycle explicitly, not just the visible rows.

Now that the extraction has landed, the moved-method calls below go through
`console._workspace.<method>(...)`; the six `_console_workspace_conversation_
*` state reads/writes stay exactly as they were pre-move -- `ChatScreen`
keeps get/set proxy properties for them, exactly as `ConsoleDictation
Controller`'s cluster did in wave 1 (see that module's docstring), so no
test here needed to change for the state accesses, only the method calls.
"""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import replace
import time
from threading import Event
from types import SimpleNamespace

import pytest
from textual.widgets import Input

from tldw_chatbook.Chat.console_chat_models import (
    CONSOLE_RUN_MARKER_GLYPHS,
    ConsoleMessageRole,
    ConsoleRunMarker,
    ConsoleWorkspaceContext,
)
from tldw_chatbook.Constants import (
    CONSOLE_NAV_CONTEXT_RESUME_LOCAL_CONVERSATION_ID,
)
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
from Tests.UI.background_signals import (
    await_background_task,
    wait_for_background_signal,
    wait_for_signal,
)
from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_destination_shells import _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.UI.Console_Modules.workspace import ConsoleWorkspaceController
import tldw_chatbook.UI.Console_Modules.workspace as workspace_module
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.Widgets.glyph_fallback import resolve_glyph
from tldw_chatbook.Widgets.Console.console_workspace_files_modal import (
    ConsoleWorkspaceFilesModal,
    WorkspaceFilesAttention,
    WorkspaceFilesBinding,
)
from tldw_chatbook.Workspaces import (
    CONSOLE_CONVERSATION_BROWSER_GROUP_ROW_LIMIT,
    CONSOLE_CONVERSATION_BROWSER_RESULT_LIMIT,
    CONSOLE_RAIL_SECTION_MIN_BUDGET_LINES,
    DEFAULT_WORKSPACE_ID,
    ConsoleConversationBrowserInputRow,
    ConsoleConversationBrowserRow,
    RuntimeBindingKind,
    RuntimeBindingStatus,
    WorkspaceRecord,
    WorkspaceRuntimeBinding,
    console_conversation_browser_group_row_limit,
    console_rail_section_height_budget,
)
from tldw_chatbook.Workspaces.file_inspector import BindingScope, ScopeCaptureError
from tldw_chatbook.Workspaces.display_state import (
    ConsoleWorkspaceContextState,
    ConsoleWorkspaceConversationRow,
)


def _noop(*args, **kwargs):
    return None


class _NoMountScreen:
    def __init__(self) -> None:
        self.after_refresh: list[object] = []
        self.workers: list[tuple[object, dict[str, object]]] = []
        self._pending_console_launch_context = None
        self._console_agent_drilldown_run_id = None
        self.size = SimpleNamespace(width=120, height=40)
        self.app = SimpleNamespace(push_screen=lambda _modal: None)

    def call_after_refresh(self, callback) -> None:
        self.after_refresh.append(callback)

    def run_worker(self, work, **kwargs) -> None:
        self.workers.append((work, kwargs))
        coroutine = work() if callable(work) else work
        coroutine.close()


class _AsyncWorkerScreen(_NoMountScreen):
    """Run controller-owned workers without mounting a Textual app."""

    def run_worker(self, coroutine, **kwargs):
        task = asyncio.create_task(coroutine)
        self.workers.append((task, kwargs))
        return task


class _FakeTimer:
    def __init__(self) -> None:
        self.stop_calls = 0

    def stop(self) -> None:
        self.stop_calls += 1


def _workspace_controller(
    *,
    screen=None,
    app_instance=None,
    **overrides,
) -> ConsoleWorkspaceController:
    """Build the real Workspace controller without mounting Textual."""
    dependencies = dict(
        app_instance=app_instance or SimpleNamespace(),
        chat_store_accessor=_noop,
        current_chat_store_accessor=_noop,
        current_conversation_id_accessor=_noop,
        native_session_rows_accessor=_noop,
        capture_draft_switch_snapshot=_noop,
        sync_chat_core_state=_noop,
        sync_native_console_chat_ui=_noop,
        sync_temporary_chip=_noop,
        default_session_settings_accessor=_noop,
        scope_picker_listers_accessor=_noop,
        active_native_session_accessor=_noop,
        refresh_effective_scope_and_sync=_noop,
        messages_from_conversation_tree_accessor=_noop,
        session_settings_for_resume_accessor=_noop,
        inject_resume_agent_markers_accessor=_noop,
        resolve_effective_scope_state=_noop,
        sync_retrieval_scope_row=_noop,
        note_follow_intent=_noop,
        focus_composer_if_needed=_noop,
        conversation_section_config_accessor=_noop,
        conversation_browser_config=_noop,
        focus_conversation_search=_noop,
        sync_workspace_context=_noop,
        schedule_timer=_noop,
        screen_running_accessor=lambda: True,
        current_chat_controller_accessor=_noop,
        fleet_unseen_ids_accessor=lambda: frozenset(),
        run_marker_with_unseen=_noop,
        broken_conversation_ids_accessor=lambda: set(),
        ensure_agent_bridge=_noop,
        subagent_counts_for_rows=lambda _bridge, _rows: {},
        conversation_browser_collapse_preferences=lambda: {},
    )
    dependencies.update(overrides)
    return ConsoleWorkspaceController(screen or _NoMountScreen(), **dependencies)


def _minimal_workspace_context_state() -> ConsoleWorkspaceContextState:
    return ConsoleWorkspaceContextState(
        heading="Workspaces",
        workspace_label="Workspace: A",
        authority_label="Authority: ready",
        sync_label="Sync: ready",
        runtime_label="Runtime: ready",
        conversation_rows=(),
        conversation_empty_copy="",
        change_workspace_enabled=True,
        change_workspace_recovery="",
        new_conversation_enabled=True,
        new_conversation_recovery="",
        recovery_copy="",
    )


def _availability_controller(monkeypatch, registry, screen, records, sync_calls):
    """Build the real cache owner with all unrelated state derivation inert."""
    controller = _workspace_controller(
        screen=screen,
        app_instance=SimpleNamespace(workspace_registry_service=registry),
        sync_workspace_context=lambda: sync_calls.append("sync"),
    )
    monkeypatch.setattr(
        workspace_module,
        "build_console_workspace_state",
        lambda **_kwargs: _minimal_workspace_context_state(),
    )
    monkeypatch.setattr(
        ConsoleWorkspaceController,
        "_with_native_console_session_rows",
        lambda _self, state: state,
    )
    monkeypatch.setattr(
        ConsoleWorkspaceController,
        "_with_console_conversation_browser_state",
        lambda _self, state, **_kwargs: state,
    )
    monkeypatch.setattr(
        ConsoleWorkspaceController,
        "_console_browser_workspace_records",
        lambda _self: tuple(records[0]),
    )
    return controller


@pytest.mark.asyncio
async def test_workspace_files_availability_snapshot_never_blocks_context_build(
    monkeypatch,
):
    """Filesystem status work is owned by one off-loop, fail-closed snapshot."""
    entered = Event()
    release = Event()

    class _Registry:
        def list_folder_bindings(self, workspace_id):
            entered.set()
            release.wait(timeout=2)
            return (
                SimpleNamespace(
                    binding_kind=RuntimeBindingKind.LOCAL_FILESYSTEM,
                    status=RuntimeBindingStatus.READY,
                ),
            )

    screen = _AsyncWorkerScreen()
    records = [[SimpleNamespace(workspace_id="ws-a")]]
    sync_calls: list[str] = []
    controller = _availability_controller(
        monkeypatch, _Registry(), screen, records, sync_calls
    )

    started = time.monotonic()
    state = controller._build_console_workspace_context_state()
    assert time.monotonic() - started < 0.1
    assert state.workspace_files_available_by_id == {"ws-a": False}
    with pytest.raises(TypeError):
        state.workspace_files_available_by_id["ws-a"] = True
    assert len(screen.workers) == 1
    await asyncio.sleep(0)
    assert entered.is_set()

    # A cadence tick while disk work is blocked is still cache-only and does
    # not mint a duplicate worker or mutate the Console context.
    state = controller._build_console_workspace_context_state()
    assert state.workspace_files_available_by_id == {"ws-a": False}
    assert len(screen.workers) == 1
    assert sync_calls == []

    release.set()
    await screen.workers[0][0]
    assert controller._workspace_files_availability_by_id == {"ws-a": True}
    assert sync_calls == ["sync"]
    assert controller._build_console_workspace_context_state().workspace_files_available_by_id == {
        "ws-a": True
    }


@pytest.mark.asyncio
async def test_workspace_files_availability_snapshot_discards_stale_generation(
    monkeypatch,
):
    """Only the current workspace set can publish; absent entries stay false."""
    first_entered = Event()
    first_release = Event()
    second_entered = Event()
    second_release = Event()

    class _Registry:
        def list_folder_bindings(self, workspace_id):
            if workspace_id == "ws-a":
                first_entered.set()
                first_release.wait(timeout=2)
            else:
                second_entered.set()
                second_release.wait(timeout=2)
            return (
                SimpleNamespace(
                    binding_kind=RuntimeBindingKind.LOCAL_FILESYSTEM,
                    status=RuntimeBindingStatus.READY,
                ),
            )

    screen = _AsyncWorkerScreen()
    records = [[SimpleNamespace(workspace_id="ws-a")]]
    sync_calls: list[str] = []
    controller = _availability_controller(
        monkeypatch, _Registry(), screen, records, sync_calls
    )
    assert controller._build_console_workspace_context_state().workspace_files_available_by_id == {
        "ws-a": False
    }
    await asyncio.sleep(0)
    assert first_entered.is_set()

    records[0] = [SimpleNamespace(workspace_id="ws-b")]
    assert controller._build_console_workspace_context_state().workspace_files_available_by_id == {
        "ws-b": False
    }
    first_release.set()
    while not second_entered.is_set():
        await asyncio.sleep(0)
    second_release.set()
    await screen.workers[0][0]

    assert controller._workspace_files_availability_by_id == {"ws-b": True}
    assert controller._build_console_workspace_context_state().workspace_files_available_by_id == {
        "ws-b": True
    }
    assert sync_calls == ["sync"]


@pytest.mark.asyncio
async def test_active_files_availability_uses_the_off_loop_binding_snapshot(
    monkeypatch,
):
    """The real active-state builder cannot reach synchronous filesystem status."""
    entered = Event()
    release = Event()
    active = WorkspaceRecord(workspace_id="ws-active", name="Active")
    binding = WorkspaceRuntimeBinding(
        workspace_id="ws-active",
        binding_id="folder-active",
        binding_kind=RuntimeBindingKind.LOCAL_FILESYSTEM,
        label="Folder",
        locator="/blocked-by-test",
        status=RuntimeBindingStatus.READY,
        metadata={"access": "ro"},
    )

    class _Registry:
        def get_active_workspace(self):
            return active

        def list_workspaces(self, **_kwargs):
            return (active,)

        def list_memberships(self, *_args, **_kwargs):
            return ()

        def list_folder_bindings(self, _workspace_id):
            entered.set()
            release.wait(timeout=2)
            return (binding,)

        def list_runtime_bindings(self, _workspace_id):
            return (binding,)

    screen = _AsyncWorkerScreen()
    sync_calls: list[str] = []
    controller = _workspace_controller(
        screen=screen,
        app_instance=SimpleNamespace(workspace_registry_service=_Registry()),
        sync_workspace_context=lambda: sync_calls.append("sync"),
    )
    monkeypatch.setattr(
        ConsoleWorkspaceController,
        "_with_native_console_session_rows",
        lambda _self, state: state,
    )
    monkeypatch.setattr(
        ConsoleWorkspaceController,
        "_with_console_conversation_browser_state",
        lambda _self, state, **_kwargs: state,
    )
    monkeypatch.setattr(
        ConsoleWorkspaceController,
        "_console_browser_workspace_records",
        lambda _self: (active,),
    )

    started = time.monotonic()
    state = controller._build_console_workspace_context_state()
    assert time.monotonic() - started < 0.1
    assert state.workspace_files_available is False
    assert state.workspace_files_available_by_id == {"ws-active": False}
    await asyncio.sleep(0)
    assert entered.is_set()

    release.set()
    await screen.workers[0][0]
    refreshed = controller._build_console_workspace_context_state()
    assert refreshed.workspace_files_available is True
    assert refreshed.workspace_files_available_by_id == {"ws-active": True}
    assert sync_calls == ["sync"]


@pytest.mark.asyncio
async def test_workspace_files_cancelled_resolution_releases_its_admission_claim():
    """Textual worker cancellation cannot permanently lock later Files visits."""
    entered = Event()
    release = Event()
    notices: list[str] = []
    controller = _workspace_controller(
        app_instance=SimpleNamespace(
            workspace_registry_service=object(),
            notify=lambda message, **_kwargs: notices.append(message),
        )
    )

    def resolve(_workspace_id: str):
        entered.set()
        release.wait()
        return None

    controller._resolve_workspace_files_visit = resolve  # type: ignore[method-assign]
    pending = asyncio.create_task(controller.request_workspace_files("ws-a"))
    while not entered.is_set():
        await asyncio.sleep(0)
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert controller._workspace_files_admission_claim is None
    release.set()


@pytest.mark.asyncio
async def test_workspace_files_claim_blocks_different_request_and_promotes_once():
    """The real admission gate, rather than worker cancellation, owns races."""
    entered = Event()
    release = Event()
    notices: list[str] = []
    promotions: list[str] = []

    class _Modal:
        is_mounted = True

        @staticmethod
        def query_one(_selector):
            return SimpleNamespace(focus=lambda: None)

    app = SimpleNamespace(
        workspace_registry_service=object(),
        notify=lambda message, **_kwargs: notices.append(message),
    )
    controller = _workspace_controller(app_instance=app)

    def resolve(workspace_id: str):
        entered.set()
        release.wait()
        return SimpleNamespace(
            workspace_id=workspace_id,
            workspace_name="Workspace A",
            active_workspace_id="active",
            active_workspace_name="Active",
            bindings=(),
            had_bindings=True,
        )

    controller._resolve_workspace_files_visit = resolve  # type: ignore[method-assign]
    controller.open_workspace_files_modal = lambda **_kwargs: (  # type: ignore[method-assign]
        promotions.append("open") or _Modal()
    )
    first = asyncio.create_task(controller.request_workspace_files("ws-a"))
    while not entered.is_set():
        await asyncio.sleep(0)
    await controller.request_workspace_files("ws-b")
    assert notices == ["Close Workspace Files before inspecting another workspace."]
    release.set()
    await first
    await controller.request_workspace_files("ws-a")
    assert promotions == ["open"]
    await controller.request_workspace_files("ws-b")
    assert notices[-1] == "Close Workspace Files before inspecting another workspace."


@pytest.mark.asyncio
async def test_workspace_files_push_failure_leaves_no_ledger_or_claim():
    app = SimpleNamespace(workspace_registry_service=object(), notify=_noop)
    controller = _workspace_controller(app_instance=app)
    controller._resolve_workspace_files_visit = lambda workspace_id: SimpleNamespace(  # type: ignore[method-assign]
        workspace_id=workspace_id, workspace_name="A", active_workspace_id="A",
        active_workspace_name="A", bindings=(), had_bindings=True,
    )
    controller.open_workspace_files_modal = lambda **_kwargs: (_ for _ in ()).throw(  # type: ignore[method-assign]
        RuntimeError("push failed")
    )
    with pytest.raises(RuntimeError, match="push failed"):
        await controller.request_workspace_files("ws-a")
    assert controller._workspace_files_admission_claim is None
    assert controller._workspace_files_modal is None
    assert controller._workspace_files_visit_workspace_id is None


@pytest.mark.asyncio
async def test_workspace_files_no_folder_blocks_but_stale_available_request_opens_recovery():
    notices: list[str] = []
    opened: list[dict[str, object]] = []
    app = SimpleNamespace(workspace_registry_service=object(), notify=lambda text, **_kw: notices.append(text))
    controller = _workspace_controller(app_instance=app)
    controller._resolve_workspace_files_visit = lambda workspace_id: SimpleNamespace(  # type: ignore[method-assign]
        workspace_id=workspace_id, workspace_name="A", active_workspace_id="A",
        active_workspace_name="A", bindings=(), had_bindings=False,
    )
    controller.open_workspace_files_modal = lambda **kwargs: (  # type: ignore[method-assign]
        opened.append(kwargs) or SimpleNamespace(is_mounted=True)
    )
    await controller.request_workspace_files("")
    assert notices == ["No local folders are attached. Add one in Settings."]
    assert opened == []
    await controller.request_workspace_files("ws-a", expected_available=True)
    assert len(opened) == 1
    closed = opened[0]["on_visit_closed"]
    assert callable(closed)
    closed()
    closed()
    assert controller._workspace_files_modal is None
    assert controller._workspace_files_visit_workspace_id is None
    await controller.request_workspace_files("ws-a", expected_available=True)
    assert len(opened) == 2


def test_workspace_files_resolution_preserves_ro_and_rw_binding_access_labels(monkeypatch):
    """The inspector must show the captured folder mode, even when unavailable."""
    workspace = SimpleNamespace(workspace_id="ws-a", name="A", archived=False)
    bindings = (
        SimpleNamespace(
            binding_id="folder-ro", label="Read only", metadata={"access": "ro"}
        ),
        SimpleNamespace(
            binding_id="folder-rw", label="Read write", metadata={"access": "rw"}
        ),
    )

    class _Registry:
        def get_workspace(self, _workspace_id):
            return workspace

        def list_folder_bindings(self, _workspace_id):
            return bindings

        def get_active_workspace(self):
            return workspace

    class _Inspector:
        def __init__(self, _registry):
            pass

        def capture_binding(self, _workspace_id, binding_id):
            if binding_id == "folder-ro":
                raise ScopeCaptureError("changed")
            return BindingScope("ws-a", binding_id, "fingerprint", "/safe", 1, 1)

    monkeypatch.setattr(workspace_module, "WorkspaceFileInspector", _Inspector)
    controller = _workspace_controller(
        app_instance=SimpleNamespace(workspace_registry_service=_Registry())
    )
    resolution = controller._resolve_workspace_files_visit("ws-a")

    assert resolution is not None
    assert [(binding.access_label, binding.available) for binding in resolution.bindings] == [
        ("Read-only", False),
        ("Read/write", True),
    ]


@pytest.mark.asyncio
async def test_workspace_files_minimum_refusal_does_not_resolve_or_claim() -> None:
    """The exact small-terminal warning must not create a worker or visit."""
    notices: list[tuple[str, str | None]] = []
    screen = _NoMountScreen()
    screen.size = SimpleNamespace(width=79, height=23)
    controller = _workspace_controller(
        screen=screen,
        app_instance=SimpleNamespace(
            workspace_registry_service=object(),
            notify=lambda text, **kwargs: notices.append((text, kwargs.get("severity"))),
        ),
    )
    calls: list[str] = []
    controller._resolve_workspace_files_visit = lambda workspace_id: calls.append(workspace_id)  # type: ignore[method-assign]

    await controller.request_workspace_files("ws-a")

    assert notices == [("Workspace Files needs at least 80 × 24 terminal cells.", "warning")]
    assert calls == []
    assert screen.workers == []
    assert controller._workspace_files_admission_claim is None
    assert controller._workspace_files_modal is None


def _rich_row() -> ConsoleConversationBrowserInputRow:
    return ConsoleConversationBrowserInputRow(
        row_key="conversation-7",
        conversation_id="conversation-7",
        native_session_id=None,
        title="Canonical row",
        scope_type="workspace",
        workspace_id="workspace-7",
        workspace_label="Workspace 7",
        status="active",
        selected=True,
        source_kind="persisted",
    )


def _browser_row(
    row_key: str,
    title: str,
    *,
    conversation_id: str | None = None,
    source_kind: str = "persisted",
    workspace_id: str | None = "workspace-7",
) -> ConsoleConversationBrowserInputRow:
    return ConsoleConversationBrowserInputRow(
        row_key=row_key,
        conversation_id=conversation_id or row_key,
        native_session_id=None,
        title=title,
        scope_type="workspace",
        workspace_id=workspace_id,
        workspace_label="Workspace 7",
        status="saved",
        selected=False,
        source_kind=source_kind,
    )


def _membership_registry(*conversation_ids: str):
    return SimpleNamespace(
        list_workspace_conversations=lambda _workspace_id: tuple(
            SimpleNamespace(item_id=conversation_id)
            for conversation_id in conversation_ids
        )
    )


def _workspace_state() -> ConsoleWorkspaceContextState:
    return ConsoleWorkspaceContextState(
        heading="Workspace",
        workspace_label="Workspace: New",
        authority_label="",
        sync_label="",
        runtime_label="",
        conversation_rows=(),
        conversation_empty_copy="No conversations.",
        change_workspace_enabled=True,
        change_workspace_recovery="",
        new_conversation_enabled=True,
        new_conversation_recovery="",
        recovery_copy="",
    )


def test_workspace_controller_initializes_canonical_browser_state_without_mount():
    controller = _workspace_controller()

    assert controller._console_persisted_rows_cache is None
    assert controller._console_persisted_rows_cache_key is None
    assert controller._console_persisted_rows_cache_at == 0.0
    assert controller._console_conversation_browser_query == ""
    assert controller._console_conversation_browser_search_timer is None
    assert controller._console_conversation_browser_search_token == 0
    assert controller._console_conversation_browser_rows == ()
    assert controller._console_conversation_browser_total is None
    assert controller._console_conversation_browser_error == ""


def test_workspace_controller_initializes_independent_projection_attempt_state():
    controller = _workspace_controller()

    assert controller._workspace_tree_search.query == ""
    assert controller._flat_conversation_search.query == ""
    assert controller._workspace_tree_search is not controller._flat_conversation_search
    assert controller._workspace_page_attempts == {}


def test_workspace_tree_expansion_preferences_round_trip_exact_empty_state() -> None:
    config: dict[str, object] = {}
    persisted: list[list[str]] = []
    controller = _workspace_controller(
        conversation_browser_config=lambda: config,
        persist_workspace_tree_expansion_preferences=persisted.append,
    )

    assert controller.workspace_tree_expansion_preferences() is None

    controller.set_workspace_tree_expansion_preferences(frozenset({"w2", "w1"}))
    assert config["expanded_workspace_ids"] == ["w1", "w2"]
    assert controller.workspace_tree_expansion_preferences() == frozenset({"w1", "w2"})

    controller.set_workspace_tree_expansion_preferences(frozenset())
    assert config["expanded_workspace_ids"] == []
    assert controller.workspace_tree_expansion_preferences() == frozenset()
    assert persisted == [["w1", "w2"], []]


def test_clearing_workspace_search_settles_visible_loading_immediately() -> None:
    controller = _workspace_controller()
    controller._workspace_tree_search.query = "old"
    controller._workspace_tree_search.request_key = ("workspaces", "old", 1)

    controller.transition_workspace_tree_search("", disabled=False)

    assert controller._workspace_tree_search.request_key is None
    assert controller._workspace_tree_search.worker is None
    state = controller._with_console_conversation_browser_state(_workspace_state())
    assert state.workspace_query == ""
    assert state.workspace_loading is False


def test_expanding_each_unloaded_named_workspace_schedules_one_loading_page() -> None:
    screen = _NoMountScreen()
    registry = SimpleNamespace(
        ensure_default_workspace=lambda: SimpleNamespace(workspace_id="active"),
        list_workspaces=lambda: (
            SimpleNamespace(workspace_id="active", name="Active", archived=False),
            SimpleNamespace(workspace_id="other", name="Other", archived=False),
        ),
        list_workspace_conversations=lambda _workspace_id: (),
    )
    controller = _workspace_controller(
        screen=screen,
        app_instance=SimpleNamespace(workspace_registry_service=registry),
    )

    controller.transition_workspace_tree_expansion("other", expanded=True)
    controller.transition_workspace_tree_expansion("other", expanded=True)
    controller.transition_workspace_tree_expansion("active", expanded=True)

    assert len(screen.workers) == 2
    assert controller._workspace_page_attempts["other"].loading is True
    assert controller._workspace_page_attempts["active"].loading is True
    projected = {
        workspace.workspace_id: workspace
        for workspace in controller.workspace_tree_projection()
    }
    assert projected["other"].loading is True
    assert projected["other"].conversations == ()
    assert projected["active"].loading is True
    assert projected["active"].conversations == ()


def test_active_named_workspace_memberships_wait_for_the_bounded_page_lane() -> None:
    registry = SimpleNamespace(
        ensure_default_workspace=lambda: SimpleNamespace(workspace_id="workspace-7"),
        list_workspaces=lambda: (
            SimpleNamespace(workspace_id="workspace-7", name="Seven", archived=False),
        ),
    )
    controller = _workspace_controller(
        app_instance=SimpleNamespace(workspace_registry_service=registry)
    )
    legacy_rows = tuple(
        ConsoleWorkspaceConversationRow(
            conversation_id=f"conversation-{index}",
            title=f"Conversation {index}",
        )
        for index in range(100)
    )
    state = replace(
        _workspace_state(),
        workspace_name="Seven",
        workspace_label="Workspace: Seven",
        active_workspace_id="workspace-7",
        conversation_rows=legacy_rows,
    )

    projected = controller._with_console_conversation_browser_state(state)

    assert len(projected.workspace_tree) == 1
    assert projected.workspace_tree[0].active is True
    assert projected.workspace_tree[0].conversations == ()


def test_workspace_context_state_build_does_not_enumerate_active_memberships() -> None:
    membership_reads: list[str] = []
    workspace = WorkspaceRecord(
        workspace_id="workspace-7",
        name="Seven",
        archived=False,
    )
    registry = SimpleNamespace(
        get_active_workspace=lambda: workspace,
        ensure_default_workspace=lambda: workspace,
        list_workspaces=lambda: (workspace,),
        list_runtime_bindings=lambda _workspace_id: (),
        list_workspace_memberships=lambda _workspace_id: (),
        list_workspace_conversations=lambda workspace_id: (
            membership_reads.append(workspace_id) or ()
        ),
    )
    controller = _workspace_controller(
        app_instance=SimpleNamespace(workspace_registry_service=registry),
        native_session_rows_accessor=lambda state: state,
    )

    state = controller._build_console_workspace_context_state()

    assert state.active_workspace_id == "workspace-7"
    assert membership_reads == []


@pytest.mark.asyncio
async def test_workspace_search_exhausts_every_service_page() -> None:
    items = [
        {
            "id": f"conversation-{index}",
            "title": f"Needle {index}",
            "scope_type": "workspace",
            "workspace_id": "workspace-7",
        }
        for index in range(80)
    ]
    offsets: list[int] = []

    async def list_conversations(**kwargs):
        offset = int(kwargs["offset"])
        limit = int(kwargs["limit"])
        offsets.append(offset)
        return {"items": items[offset : offset + limit], "total": len(items)}

    controller = _workspace_controller(
        app_instance=SimpleNamespace(
            workspace_registry_service=SimpleNamespace(
                list_workspaces=lambda: (
                    SimpleNamespace(
                        workspace_id="workspace-7", name="Seven", archived=False
                    ),
                )
            ),
            chat_conversation_scope_service=SimpleNamespace(
                list_conversations=list_conversations
            ),
        )
    )

    await controller.refresh_workspace_tree_search("needle")

    assert offsets == [0, CONSOLE_CONVERSATION_BROWSER_RESULT_LIMIT]
    assert controller._workspace_tree_search.total == 80
    assert len(controller._workspace_tree_search.rows) == 80


@pytest.mark.asyncio
async def test_workspace_search_includes_unsaved_native_rows_and_clear_restores_tree() -> (
    None
):
    native_row = ConsoleConversationBrowserInputRow(
        row_key="native:session-7",
        conversation_id=None,
        native_session_id="session-7",
        title="Needle roleplay",
        scope_type="workspace",
        workspace_id="workspace-7",
        workspace_label="Seven",
        status="open session",
        star_enabled=False,
        source_kind="native",
    )

    async def list_conversations(**_kwargs):
        return {"items": [], "total": 0}

    controller = _workspace_controller(
        app_instance=SimpleNamespace(
            workspace_registry_service=SimpleNamespace(
                list_workspaces=lambda: (
                    SimpleNamespace(
                        workspace_id="workspace-7", name="Seven", archived=False
                    ),
                )
            ),
            chat_conversation_scope_service=SimpleNamespace(
                list_conversations=list_conversations
            ),
        )
    )
    controller._native_console_browser_rows = lambda _current=None: [native_row]

    await controller.refresh_workspace_tree_search("needle")

    assert [row.row_key for row in controller._workspace_tree_search.rows] == [
        "native:session-7"
    ]
    assert (
        controller.workspace_tree_projection()[0].conversations[0].conversation_id
        == "native:session-7"
    )

    controller.transition_workspace_tree_search("", disabled=False)

    assert controller._workspace_tree_search.rows == ()
    assert controller._workspace_tree_search.query == ""
    restored = controller.workspace_tree_projection((native_row,))
    assert restored[0].conversations[0].conversation_id == "native:session-7"


@pytest.mark.asyncio
async def test_workspace_search_settles_current_key_without_clearing_newer_attempt() -> (
    None
):
    controller = _workspace_controller()
    old_started = asyncio.Event()
    release_old = asyncio.Event()
    new_started = asyncio.Event()
    release_new = asyncio.Event()

    async def load(query):
        if query == "old":
            old_started.set()
            await release_old.wait()
            return (_browser_row("old", "Old"),), 1
        new_started.set()
        await release_new.wait()
        return (_browser_row("new", "New"),), 1

    controller._load_workspace_tree_search_rows = load
    old_task = asyncio.create_task(controller.refresh_workspace_tree_search("old"))
    await wait_for_signal(old_started, what="old workspace search start")
    new_task = asyncio.create_task(controller.refresh_workspace_tree_search("new"))
    await wait_for_signal(new_started, what="replacement workspace search start")
    newer_key = controller._workspace_tree_search.request_key

    release_old.set()
    await old_task
    assert controller._workspace_tree_search.request_key == newer_key
    assert controller._workspace_tree_search.rows == ()

    release_new.set()
    await new_task
    assert controller._workspace_tree_search.request_key is None
    assert [row.conversation_id for row in controller._workspace_tree_search.rows] == [
        "new"
    ]


@pytest.mark.asyncio
async def test_workspace_search_failure_settles_loading_and_exposes_retry_state() -> (
    None
):
    controller = _workspace_controller()

    async def fail(_query):
        raise RuntimeError("private")

    controller._load_workspace_tree_search_rows = fail
    await controller.refresh_workspace_tree_search("needle")

    assert controller._workspace_tree_search.request_key is None
    state = controller._with_console_conversation_browser_state(_workspace_state())
    assert state.workspace_loading is False
    assert state.workspace_error == "Workspace search is unavailable."
    assert state.workspace_retry_available is True


def test_browser_row_cap_expands_to_fill_measured_rail_height():
    """The browser's visible-row cap follows the measured rail body height.

    The Chats list's budget is half the rail (its peer share alongside the
    Workspaces section), minus the tray's fixed chrome converted to rows:
    with 200 terminal lines available the cap is (100 - 8) // 3 = 30 rows
    instead of the historical 12-row default.
    """
    controller = _workspace_controller(
        rail_body_height_accessor=lambda: 200,
    )
    # Pin the workspace so the section builder's workspace-change reset does
    # not clear the seeded rows before the browser state is built.
    controller._console_workspace_conversation_workspace_id = str(
        controller._current_console_workspace_context().active_workspace_id or ""
    )
    controller._console_conversation_browser_rows = tuple(
        _browser_row(f"chat-{i}", f"Chat {i}", workspace_id=None)
        for i in range(40)
    )

    state = controller._with_console_conversation_browser_state(_workspace_state())

    browser = state.conversation_browser
    assert browser is not None
    (chats,) = browser.sections
    assert chats.section_id == "chats"
    assert len(chats.rows) == 30
    assert chats.hidden_count == 10


def test_browser_row_cap_falls_back_to_default_without_a_measurement():
    """Unmeasured rails keep the 12-row default cap."""
    controller = _workspace_controller()
    controller._console_workspace_conversation_workspace_id = str(
        controller._current_console_workspace_context().active_workspace_id or ""
    )
    controller._console_conversation_browser_rows = tuple(
        _browser_row(f"chat-{i}", f"Chat {i}", workspace_id=None)
        for i in range(CONSOLE_CONVERSATION_BROWSER_GROUP_ROW_LIMIT + 3)
    )

    state = controller._with_console_conversation_browser_state(_workspace_state())

    browser = state.conversation_browser
    assert browser is not None
    (chats,) = browser.sections
    assert len(chats.rows) == CONSOLE_CONVERSATION_BROWSER_GROUP_ROW_LIMIT


@pytest.mark.asyncio
async def test_workspace_and_flat_search_completions_are_independent() -> None:
    controller = _workspace_controller()
    workspace_started = asyncio.Event()
    release_workspace = asyncio.Event()

    async def load_workspace(query):
        workspace_started.set()
        await release_workspace.wait()
        return (_browser_row("workspace-hit", query, workspace_id="workspace-7"),), 1

    async def load_flat(query):
        return (_browser_row("flat-hit", query, workspace_id=DEFAULT_WORKSPACE_ID),), 1

    controller._load_workspace_tree_search_rows = load_workspace
    controller._load_flat_conversation_search_rows = load_flat

    workspace_task = asyncio.create_task(controller.refresh_workspace_tree_search("A"))
    await wait_for_signal(workspace_started, what="workspace search start")
    await controller.refresh_flat_conversation_search("B")
    flat_rows = controller._flat_conversation_search.rows
    release_workspace.set()
    await workspace_task

    assert [row.title for row in controller._workspace_tree_search.rows] == ["A"]
    assert controller._flat_conversation_search.rows == flat_rows
    assert [row.title for row in flat_rows] == ["B"]


@pytest.mark.asyncio
async def test_replacing_workspace_owner_invalidates_only_workspace_search() -> None:
    owners = {"workspace": object(), "flat": object(), "lifecycle": object()}
    controller = _workspace_controller(
        workspace_tree_owner_accessor=lambda: owners["workspace"]
    )
    controller._flat_conversation_owner_accessor = lambda: owners["flat"]
    controller._screen_lifecycle_token_accessor = lambda: owners["lifecycle"]
    workspace_started = asyncio.Event()
    flat_started = asyncio.Event()
    release = asyncio.Event()

    async def load_workspace(_query):
        workspace_started.set()
        await release.wait()
        return (_browser_row("workspace-new", "Workspace new"),), 1

    async def load_flat(_query):
        flat_started.set()
        await release.wait()
        return (
            _browser_row("flat-new", "Flat new", workspace_id=DEFAULT_WORKSPACE_ID),
        ), 1

    controller._load_workspace_tree_search_rows = load_workspace
    controller._load_flat_conversation_search_rows = load_flat
    workspace = asyncio.create_task(controller.refresh_workspace_tree_search("needle"))
    flat = asyncio.create_task(controller.refresh_flat_conversation_search("needle"))
    await wait_for_signal(workspace_started, what="workspace owner search start")
    await wait_for_signal(flat_started, what="flat owner search start")
    owners["workspace"] = object()
    release.set()
    await asyncio.gather(workspace, flat)

    assert controller._workspace_tree_search.rows == ()
    assert [
        row.conversation_id for row in controller._flat_conversation_search.rows
    ] == ["flat-new"]


@pytest.mark.asyncio
async def test_replacing_flat_owner_invalidates_only_flat_search() -> None:
    owners = {"workspace": object(), "flat": object(), "lifecycle": object()}
    controller = _workspace_controller(
        workspace_tree_owner_accessor=lambda: owners["workspace"]
    )
    controller._flat_conversation_owner_accessor = lambda: owners["flat"]
    controller._screen_lifecycle_token_accessor = lambda: owners["lifecycle"]
    workspace_started = asyncio.Event()
    flat_started = asyncio.Event()
    release = asyncio.Event()

    async def load_workspace(_query):
        workspace_started.set()
        await release.wait()
        return (_browser_row("workspace-new", "Workspace new"),), 1

    async def load_flat(_query):
        flat_started.set()
        await release.wait()
        return (
            _browser_row("flat-new", "Flat new", workspace_id=DEFAULT_WORKSPACE_ID),
        ), 1

    controller._load_workspace_tree_search_rows = load_workspace
    controller._load_flat_conversation_search_rows = load_flat
    workspace = asyncio.create_task(controller.refresh_workspace_tree_search("needle"))
    flat = asyncio.create_task(controller.refresh_flat_conversation_search("needle"))
    await wait_for_signal(workspace_started, what="workspace owner search start")
    await wait_for_signal(flat_started, what="flat owner search start")
    owners["flat"] = object()
    release.set()
    await asyncio.gather(workspace, flat)

    assert [row.conversation_id for row in controller._workspace_tree_search.rows] == [
        "workspace-new"
    ]
    assert controller._flat_conversation_search.rows == ()


@pytest.mark.asyncio
async def test_remount_lifecycle_invalidates_old_requests_for_both_search_lanes() -> (
    None
):
    owners = {"workspace": object(), "flat": object(), "lifecycle": object()}
    running = True
    controller = _workspace_controller(
        workspace_tree_owner_accessor=lambda: owners["workspace"],
        screen_running_accessor=lambda: running,
    )
    controller._flat_conversation_owner_accessor = lambda: owners["flat"]
    controller._screen_lifecycle_token_accessor = lambda: owners["lifecycle"]
    workspace_started = asyncio.Event()
    flat_started = asyncio.Event()
    release = asyncio.Event()

    async def load_workspace(_query):
        workspace_started.set()
        await release.wait()
        return (_browser_row("workspace-new", "Workspace new"),), 1

    async def load_flat(_query):
        flat_started.set()
        await release.wait()
        return (
            _browser_row("flat-new", "Flat new", workspace_id=DEFAULT_WORKSPACE_ID),
        ), 1

    controller._load_workspace_tree_search_rows = load_workspace
    controller._load_flat_conversation_search_rows = load_flat
    workspace = asyncio.create_task(controller.refresh_workspace_tree_search("needle"))
    flat = asyncio.create_task(controller.refresh_flat_conversation_search("needle"))
    await wait_for_signal(workspace_started, what="workspace lifecycle search start")
    await wait_for_signal(flat_started, what="flat lifecycle search start")
    running = False
    owners["lifecycle"] = object()
    running = True
    release.set()
    await asyncio.gather(workspace, flat)

    assert controller._workspace_tree_search.rows == ()
    assert controller._flat_conversation_search.rows == ()


@pytest.mark.asyncio
async def test_search_failures_preserve_each_lane_and_expose_scoped_retry() -> None:
    controller = _workspace_controller()
    workspace_settled = (_browser_row("workspace-old", "Workspace settled"),)
    flat_settled = (
        _browser_row("flat-old", "Flat settled", workspace_id=DEFAULT_WORKSPACE_ID),
    )
    controller._workspace_tree_search.rows = workspace_settled
    controller._flat_conversation_search.rows = flat_settled

    async def fail(_query):
        raise RuntimeError("private detail")

    controller._load_workspace_tree_search_rows = fail
    controller._load_flat_conversation_search_rows = fail
    await controller.refresh_workspace_tree_search("workspace query")

    assert controller._workspace_tree_search.rows == workspace_settled
    assert controller._workspace_tree_search.retry_query == "workspace query"
    assert controller._workspace_tree_search.error == "Workspace search is unavailable."
    assert controller._flat_conversation_search.rows == flat_settled
    assert controller._flat_conversation_search.error == ""

    await controller.refresh_flat_conversation_search("flat query")
    assert controller._flat_conversation_search.rows == flat_settled
    assert controller._flat_conversation_search.retry_query == "flat query"
    assert (
        controller._flat_conversation_search.error
        == "Conversation search is unavailable."
    )
    assert "private" not in controller._flat_conversation_search.error


@pytest.mark.asyncio
async def test_normal_search_transitions_preserve_settled_rows_per_lane_on_failure() -> (
    None
):
    registry = SimpleNamespace(
        list_workspaces=lambda: (
            SimpleNamespace(workspace_id="workspace-7", name="Seven", archived=False),
            SimpleNamespace(
                workspace_id=DEFAULT_WORKSPACE_ID, name="Default", archived=False
            ),
        ),
        list_workspace_conversations=lambda workspace_id: (
            (
                SimpleNamespace(
                    item_id="flat-settled",
                    title="Stable ordinary row",
                    role="member",
                ),
            )
            if workspace_id == DEFAULT_WORKSPACE_ID
            else ()
        ),
    )

    async def fail_search(**_kwargs):
        raise RuntimeError("private service detail")

    controller = _workspace_controller(
        app_instance=SimpleNamespace(
            workspace_registry_service=registry,
            chat_conversation_scope_service=SimpleNamespace(
                list_conversations=fail_search,
                local_service=None,
            ),
        )
    )
    workspace_settled = (_browser_row("workspace-settled", "Stable workspace row"),)
    loaded_children = (_browser_row("loaded-child", "Loaded child"),)
    controller._workspace_tree_search.rows = workspace_settled
    controller._workspace_page_attempts["workspace-7"] = (
        controller._new_workspace_page_state(rows=loaded_children, next_cursor=75)
    )
    controller._flat_conversation_search.rows = (
        _browser_row(
            "flat-settled",
            "Stable ordinary row",
            workspace_id=DEFAULT_WORKSPACE_ID,
        ),
    )
    initial_flat = controller._with_console_conversation_browser_state(
        _workspace_state()
    ).conversation_browser
    assert initial_flat is not None
    assert [
        row.conversation_id for section in initial_flat.sections for row in section.rows
    ] == ["flat-settled"]
    flat_settled = controller._flat_conversation_search.settled_rows
    assert [row.conversation_id for row in flat_settled] == ["flat-settled"]

    controller.transition_browser_search("first needle", disabled=False)
    controller.transition_browser_search("needle", disabled=False)
    flat_token = controller._flat_conversation_search.generation
    await controller._refresh_console_conversation_browser_search("needle", flat_token)

    assert controller._flat_conversation_search.rows == flat_settled
    assert controller._flat_conversation_search.total == 1
    assert controller._flat_conversation_search.retry_query == "needle"
    assert (
        controller._flat_conversation_search.error
        == "Conversation search is unavailable."
    )
    assert controller._workspace_tree_search.rows == workspace_settled
    assert controller._workspace_page_attempts["workspace-7"].rows == loaded_children
    flat_projection = controller._with_console_conversation_browser_state(
        _workspace_state()
    ).conversation_browser
    assert flat_projection is not None
    assert [
        row.conversation_id
        for section in flat_projection.sections
        for row in section.rows
    ] == ["flat-settled"]

    flat_after_failure = controller._flat_conversation_search.rows
    controller.transition_workspace_tree_search("workspace needle", disabled=False)
    await controller.refresh_workspace_tree_search("workspace needle")

    assert controller._workspace_tree_search.rows == workspace_settled
    assert controller._workspace_tree_search.retry_query == "workspace needle"
    assert controller._workspace_tree_search.error == "Workspace search is unavailable."
    assert controller._workspace_page_attempts["workspace-7"].rows == loaded_children
    assert controller._flat_conversation_search.rows == flat_after_failure
    assert [
        row.conversation_id
        for node in controller.workspace_tree_projection()
        for row in node.conversations
    ] == ["loaded-child"]


@pytest.mark.asyncio
async def test_workspace_page_retry_generation_rejects_stale_failure() -> None:
    controller = _workspace_controller(
        app_instance=SimpleNamespace(
            workspace_registry_service=_membership_registry("existing", "page-row")
        )
    )
    calls = 0
    release_retry = asyncio.Event()

    async def fetch(_workspace_id, _cursor):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("first failed")
        await release_retry.wait()
        return (_browser_row("page-row", "Page row"),), None

    controller._fetch_workspace_tree_page = fetch
    controller._workspace_page_attempts["workspace-7"] = (
        controller._new_workspace_page_state(
            rows=(_browser_row("existing", "Existing"),), next_cursor=75
        )
    )
    await controller.load_workspace_tree_page("workspace-7", 75)
    attempt = controller._workspace_page_attempts["workspace-7"]
    stale_generation = attempt.generation
    stale_key = attempt.request_key
    assert attempt.error == "Couldn't load conversations."
    assert attempt.retry_cursor == 75
    assert [row.conversation_id for row in attempt.rows] == ["existing"]

    retry = asyncio.create_task(controller.retry_workspace_tree_page("workspace-7"))
    await asyncio.sleep(0)
    controller._commit_workspace_page_failure(
        "workspace-7", stale_generation, stale_key
    )
    assert attempt.error == ""
    release_retry.set()
    await retry

    assert [row.conversation_id for row in attempt.rows] == ["existing", "page-row"]


@pytest.mark.asyncio
async def test_collapsing_workspace_fences_late_page_commit_without_dropping_rows() -> (
    None
):
    controller = _workspace_controller(
        app_instance=SimpleNamespace(
            workspace_registry_service=_membership_registry("settled", "late")
        )
    )
    settled = _browser_row("settled", "Settled")
    attempt = controller._new_workspace_page_state(rows=(settled,), next_cursor=75)
    controller._workspace_page_attempts["workspace-7"] = attempt
    started = asyncio.Event()
    release = asyncio.Event()

    async def fetch(_workspace_id: str, _cursor: int):
        started.set()
        await release.wait()
        return (_browser_row("late", "Late"),), None

    controller._fetch_workspace_tree_page = fetch
    page = asyncio.create_task(controller.load_workspace_tree_page("workspace-7", 75))
    await wait_for_signal(started, what="workspace page collapse request start")

    controller.transition_workspace_tree_expansion("workspace-7", expanded=False)
    release.set()
    await page

    assert [row.conversation_id for row in attempt.rows] == ["settled"]
    assert attempt.loading is False
    assert attempt.request_key is None

    async def fetch_after_collapse(_workspace_id: str, _cursor: int):
        return (_browser_row("late", "Late"),), None

    controller._fetch_workspace_tree_page = fetch_after_collapse
    await controller.load_workspace_tree_page("workspace-7", 75)
    assert [row.conversation_id for row in attempt.rows] == ["settled"]
    assert attempt.error == ""
    assert attempt.retry_cursor is None


@pytest.mark.asyncio
async def test_page_completion_with_unknown_membership_preserves_settled_retry() -> (
    None
):
    membership_unavailable = False

    def list_memberships(workspace_id):
        if workspace_id == "workspace-7" and membership_unavailable:
            raise RuntimeError("registry unavailable")
        conversation_id = "existing" if workspace_id == "workspace-7" else "steady"
        return (SimpleNamespace(item_id=conversation_id),)

    controller = _workspace_controller(
        app_instance=SimpleNamespace(
            workspace_registry_service=SimpleNamespace(
                list_workspace_conversations=list_memberships,
                list_workspaces=lambda: (
                    SimpleNamespace(
                        workspace_id="workspace-7", name="Seven", archived=False
                    ),
                    SimpleNamespace(
                        workspace_id="workspace-8", name="Eight", archived=False
                    ),
                ),
            )
        )
    )
    settled = (_browser_row("existing", "Existing"),)
    other_rows = (_browser_row("steady", "Steady", workspace_id="workspace-8"),)
    controller._workspace_page_attempts["workspace-7"] = (
        controller._new_workspace_page_state(rows=settled, next_cursor=75)
    )
    other_attempt = controller._new_workspace_page_state(
        rows=other_rows, next_cursor=None
    )
    controller._workspace_page_attempts["workspace-8"] = other_attempt
    workspace_rows = (_browser_row("workspace-hit", "Workspace hit"),)
    flat_rows = (
        _browser_row("flat-hit", "Flat hit", workspace_id=DEFAULT_WORKSPACE_ID),
    )
    controller._workspace_tree_search.rows = workspace_rows
    controller._flat_conversation_search.rows = flat_rows
    started = asyncio.Event()
    release = asyncio.Event()

    async def fetch(_workspace_id, _cursor):
        started.set()
        await release.wait()
        return (_browser_row("incoming", "Incoming"),), None

    controller._fetch_workspace_tree_page = fetch
    page = asyncio.create_task(controller.load_workspace_tree_page("workspace-7", 75))
    await wait_for_signal(started, what="unknown-membership page request start")
    membership_unavailable = True
    release.set()
    await page

    attempt = controller._workspace_page_attempts["workspace-7"]
    assert attempt.rows == settled
    assert attempt.next_cursor == 75
    assert attempt.membership_unknown is True
    assert attempt.error == "Workspace access unknown."
    assert attempt.retry_cursor == 75
    public = {
        node.workspace_id: node for node in controller.workspace_tree_projection()
    }["workspace-7"]
    assert (
        public.loading,
        public.error,
        public.retry_cursor,
        public.membership_unknown,
    ) == (False, "Workspace access unknown.", 75, True)
    assert controller._workspace_page_attempts["workspace-8"] is other_attempt
    assert other_attempt.rows == other_rows
    assert controller._workspace_tree_search.rows == workspace_rows
    assert controller._flat_conversation_search.rows == flat_rows


@pytest.mark.asyncio
async def test_page_completion_from_prior_screen_lifecycle_is_discarded_and_settled() -> (
    None
):
    identities = {"owner": object(), "lifecycle": object()}
    syncs: list[str] = []
    controller = _workspace_controller(
        app_instance=SimpleNamespace(
            workspace_registry_service=_membership_registry("settled", "late")
        ),
        workspace_tree_owner_accessor=lambda: identities["owner"],
        screen_lifecycle_token_accessor=lambda: identities["lifecycle"],
        sync_workspace_context=lambda: syncs.append("sync"),
    )
    settled = _browser_row("settled", "Settled")
    attempt = controller._new_workspace_page_state(rows=(settled,), next_cursor=75)
    attempt.membership_token = ("late", "settled")
    controller._workspace_page_attempts["workspace-7"] = attempt
    started = asyncio.Event()
    release = asyncio.Event()

    async def fetch(_workspace_id, _cursor):
        started.set()
        await release.wait()
        return (_browser_row("late", "Old lifecycle"),), None

    controller._fetch_workspace_tree_page = fetch
    page = asyncio.create_task(controller.load_workspace_tree_page("workspace-7", 75))
    await wait_for_signal(started, what="prior-lifecycle page request start")
    loading_sync_count = len(syncs)
    identities["lifecycle"] = object()
    release.set()
    await page

    assert [row.conversation_id for row in attempt.rows] == ["settled"]
    assert attempt.loading is False
    assert len(syncs) > loading_sync_count


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal", ["owner", "cancel", "membership"])
async def test_page_terminal_paths_settle_loading_and_allow_retry(
    terminal: str,
) -> None:
    identities = {"owner": object(), "lifecycle": object()}
    memberships = ["settled", "late"]
    syncs: list[str] = []
    fetch_calls = 0
    controller = _workspace_controller(
        app_instance=SimpleNamespace(
            workspace_registry_service=SimpleNamespace(
                list_workspace_conversations=lambda _workspace_id: tuple(
                    SimpleNamespace(item_id=conversation_id)
                    for conversation_id in memberships
                )
            )
        ),
        workspace_tree_owner_accessor=lambda: identities["owner"],
        screen_lifecycle_token_accessor=lambda: identities["lifecycle"],
        sync_workspace_context=lambda: syncs.append("sync"),
    )
    started = asyncio.Event()
    release = asyncio.Event()

    async def blocked_fetch(_workspace_id, _cursor):
        nonlocal fetch_calls
        fetch_calls += 1
        started.set()
        await release.wait()
        return (_browser_row("late", "Late"),), None

    controller._fetch_workspace_tree_page = blocked_fetch
    page = asyncio.create_task(controller.load_workspace_tree_page("workspace-7", 75))
    await wait_for_signal(started, what="workspace terminal page request start")
    loading_sync_count = len(syncs)
    if terminal == "owner":
        identities["owner"] = object()
    elif terminal == "membership":
        memberships[:] = ["settled"]
    else:
        page.cancel()
    release.set()
    if terminal == "cancel":
        with pytest.raises(asyncio.CancelledError):
            await page
    else:
        await page

    attempt = controller._workspace_page_attempts["workspace-7"]
    assert attempt.loading is False
    if terminal == "owner":
        assert len(syncs) == loading_sync_count
    else:
        assert len(syncs) > loading_sync_count

    async def retry_fetch(_workspace_id, _cursor):
        nonlocal fetch_calls
        fetch_calls += 1
        return (_browser_row("retry", "Retry"),), None

    controller._fetch_workspace_tree_page = retry_fetch
    await controller.load_workspace_tree_page("workspace-7", 75)
    assert fetch_calls == 2
    assert attempt.loading is False


@pytest.mark.asyncio
async def test_stale_page_terminal_does_not_settle_newer_attempt() -> None:
    controller = _workspace_controller(
        app_instance=SimpleNamespace(
            workspace_registry_service=_membership_registry("same")
        )
    )
    started = [asyncio.Event(), asyncio.Event()]
    release = [asyncio.Event(), asyncio.Event()]
    calls = 0

    async def fetch(_workspace_id, _cursor):
        nonlocal calls
        index = calls
        calls += 1
        started[index].set()
        await release[index].wait()
        return (_browser_row("same", f"Attempt {index}"),), None

    controller._fetch_workspace_tree_page = fetch
    old_page = asyncio.create_task(
        controller.load_workspace_tree_page("workspace-7", 75)
    )
    await started[0].wait()
    attempt = controller._workspace_page_attempts["workspace-7"]
    attempt.loading = False
    new_page = asyncio.create_task(
        controller.load_workspace_tree_page("workspace-7", 75)
    )
    await started[1].wait()

    release[0].set()
    await old_page
    assert attempt.loading is True

    release[1].set()
    await new_page
    assert attempt.loading is False


@pytest.mark.asyncio
async def test_repeated_same_cursor_pages_materialize_conversation_id_once() -> None:
    controller = _workspace_controller(
        app_instance=SimpleNamespace(
            workspace_registry_service=_membership_registry("same")
        )
    )
    calls = 0

    async def fetch(_workspace_id, _cursor):
        nonlocal calls
        index = calls
        calls += 1
        return (_browser_row("same", f"attempt-{index}"),), None

    controller._fetch_workspace_tree_page = fetch
    await controller.load_workspace_tree_page("workspace-7", 75)
    await controller.load_workspace_tree_page("workspace-7", 75)

    rows = controller._workspace_page_attempts["workspace-7"].rows
    assert [(row.conversation_id, row.title) for row in rows] == [("same", "attempt-0")]


@pytest.mark.asyncio
async def test_workspace_search_projects_hit_outside_loaded_page() -> None:
    registry = SimpleNamespace(
        list_workspaces=lambda: (
            SimpleNamespace(
                workspace_id="workspace-7", name="Workspace 7", archived=False
            ),
            SimpleNamespace(
                workspace_id="workspace-8", name="Unrelated", archived=False
            ),
        )
    )
    controller = _workspace_controller(
        app_instance=SimpleNamespace(workspace_registry_service=registry)
    )
    loaded = _browser_row("loaded", "Loaded")
    controller._workspace_page_attempts["workspace-7"] = (
        controller._new_workspace_page_state(rows=(loaded,), next_cursor=75)
    )

    async def search(_query):
        return (_browser_row("outside", "Needle outside page"),), 1

    controller._load_workspace_tree_search_rows = search
    await controller.refresh_workspace_tree_search("needle")

    projection = controller.workspace_tree_projection()
    assert [node.workspace_id for node in projection] == ["workspace-7"]
    assert [row.conversation_id for row in projection[0].conversations] == ["outside"]


@pytest.mark.asyncio
async def test_workspace_search_is_one_bounded_all_scope_query() -> None:
    workspace_ids = tuple(f"workspace-{index}" for index in range(50))
    calls: list[dict[str, object]] = []

    async def list_conversations(**kwargs):
        calls.append(kwargs)
        if kwargs.get("scope_type") == "all":
            return {
                "items": [
                    {
                        "id": "hit-1",
                        "title": "Needle one",
                        "scope_type": "workspace",
                        "workspace_id": "workspace-1",
                    },
                    {
                        "id": "hit-2",
                        "title": "Needle two",
                        "scope_type": "workspace",
                        "workspace_id": "workspace-2",
                    },
                    {
                        "id": "default-hit",
                        "title": "Needle default",
                        "scope_type": "workspace",
                        "workspace_id": DEFAULT_WORKSPACE_ID,
                    },
                ],
                "pagination": {"total": 3},
            }
        workspace_id = str(kwargs.get("workspace_id") or "")
        return {
            "items": [
                {
                    "id": f"hit-{workspace_id}",
                    "title": "Needle",
                    "scope_type": "workspace",
                    "workspace_id": workspace_id,
                }
            ],
            "pagination": {"total": 1},
        }

    registry = SimpleNamespace(
        list_workspaces=lambda: tuple(
            SimpleNamespace(
                workspace_id=workspace_id, name=workspace_id, archived=False
            )
            for workspace_id in workspace_ids
        )
    )
    controller = _workspace_controller(
        app_instance=SimpleNamespace(
            workspace_registry_service=registry,
            chat_conversation_scope_service=SimpleNamespace(
                list_conversations=list_conversations,
                local_service=object(),
            ),
        )
    )

    await controller.refresh_workspace_tree_search("needle")

    assert len(calls) == 1
    assert calls[0]["scope_type"] == "all"
    assert calls[0]["limit"] == 75
    assert [row.conversation_id for row in controller._workspace_tree_search.rows] == [
        "hit-1",
        "hit-2",
    ]
    assert [node.workspace_id for node in controller.workspace_tree_projection()] == [
        "workspace-1",
        "workspace-2",
    ]


@pytest.mark.asyncio
async def test_membership_move_discards_inflight_page_and_projects_new_owner_once() -> (
    None
):
    memberships = {
        "workspace-7": [
            SimpleNamespace(item_id="moving", title="Moving", role="workspace-thread")
        ],
        "workspace-8": [],
    }
    registry = SimpleNamespace(
        list_workspaces=lambda: (
            SimpleNamespace(workspace_id="workspace-7", name="Seven", archived=False),
            SimpleNamespace(workspace_id="workspace-8", name="Eight", archived=False),
        ),
        list_workspace_conversations=lambda workspace_id: tuple(
            memberships[workspace_id]
        ),
    )
    controller = _workspace_controller(
        app_instance=SimpleNamespace(workspace_registry_service=registry)
    )
    started = asyncio.Event()
    release = asyncio.Event()

    async def fetch(_workspace_id, _cursor):
        started.set()
        await release.wait()
        return (_browser_row("moving", "Stale owner"),), None

    controller._fetch_workspace_tree_page = fetch
    task = asyncio.create_task(controller.load_workspace_tree_page("workspace-7", 75))
    await wait_for_signal(started, what="workspace membership move start")
    memberships["workspace-7"] = []
    memberships["workspace-8"] = [
        SimpleNamespace(item_id="moving", title="Moved", role="workspace-thread")
    ]
    release.set()
    await task

    moved = _browser_row("moving", "Moved", workspace_id="workspace-8")
    projection = controller.workspace_tree_projection((moved,))
    owners = [
        node.workspace_id
        for node in projection
        if any(row.conversation_id == "moving" for row in node.conversations)
    ]
    assert owners == ["workspace-8"]
    stale_attempt = controller._workspace_page_attempts["workspace-7"]
    assert stale_attempt.rows == ()
    assert stale_attempt.membership_unknown is False
    assert stale_attempt.error == ""
    assert stale_attempt.retry_cursor is None


@pytest.mark.asyncio
async def test_settled_page_membership_move_to_default_has_one_ordinary_owner() -> None:
    memberships = {
        "workspace-7": [
            SimpleNamespace(item_id="moving", title="Moving", role="member")
        ],
        DEFAULT_WORKSPACE_ID: [],
    }
    registry = SimpleNamespace(
        list_workspaces=lambda: (
            SimpleNamespace(workspace_id="workspace-7", name="Seven", archived=False),
            SimpleNamespace(
                workspace_id=DEFAULT_WORKSPACE_ID, name="Default", archived=False
            ),
        ),
        list_workspace_conversations=lambda workspace_id: tuple(
            memberships[workspace_id]
        ),
    )

    async def list_conversations(**_kwargs):
        return {
            "items": [{"id": "moving", "title": "Moving", "state": "saved"}],
            "total": 1,
        }

    controller = _workspace_controller(
        app_instance=SimpleNamespace(
            workspace_registry_service=registry,
            chat_conversation_scope_service=SimpleNamespace(
                list_conversations=list_conversations
            ),
        )
    )
    await controller.load_workspace_tree_page("workspace-7", 0)
    assert [
        row.conversation_id
        for node in controller.workspace_tree_projection()
        for row in node.conversations
    ] == ["moving"]

    memberships["workspace-7"] = []
    memberships[DEFAULT_WORKSPACE_ID] = [
        SimpleNamespace(item_id="moving", title="Moving", role="member")
    ]
    controller.apply_workspace_membership_snapshot(
        {
            "workspace-7": (),
            DEFAULT_WORKSPACE_ID: ("moving",),
        },
        complete=True,
    )
    state = controller._with_console_conversation_browser_state(_workspace_state())
    tree = state.workspace_tree
    flat = state.conversation_browser
    assert flat is not None
    owners = [
        *(
            node.workspace_id
            for node in tree
            for row in node.conversations
            if row.conversation_id == "moving"
        ),
        *(
            "flat"
            for section in flat.sections
            for row in section.rows
            if row.conversation_id == "moving"
        ),
    ]

    assert owners == ["flat"]

    memberships["workspace-7"] = [
        SimpleNamespace(item_id="moving", title="Moving", role="member")
    ]
    memberships[DEFAULT_WORKSPACE_ID] = []
    controller.apply_workspace_membership_snapshot(
        {
            "workspace-7": ("moving",),
            DEFAULT_WORKSPACE_ID: (),
        },
        complete=True,
        workspace_labels={"workspace-7": "Seven"},
    )
    state = controller._with_console_conversation_browser_state(_workspace_state())
    assert state.conversation_browser is not None
    owners = [
        *(
            node.workspace_id
            for node in state.workspace_tree
            for row in node.conversations
            if row.conversation_id == "moving"
        ),
        *(
            "flat"
            for section in state.conversation_browser.sections
            for row in section.rows
            if row.conversation_id == "moving"
        ),
    ]

    assert owners == ["workspace-7"]


@pytest.mark.asyncio
async def test_membership_read_failure_preserves_settled_page_and_retry_state() -> None:
    membership_unavailable = False

    def list_memberships(_workspace_id):
        if membership_unavailable:
            raise RuntimeError("registry unavailable")
        return (SimpleNamespace(item_id="settled", title="Settled", role="member"),)

    registry = SimpleNamespace(
        list_workspaces=lambda: (
            SimpleNamespace(workspace_id="workspace-7", name="Seven", archived=False),
        ),
        list_workspace_conversations=list_memberships,
    )

    async def list_conversations(**_kwargs):
        return {
            "items": [{"id": "settled", "title": "Settled", "state": "saved"}],
            "total": 151,
        }

    controller = _workspace_controller(
        app_instance=SimpleNamespace(
            workspace_registry_service=registry,
            chat_conversation_scope_service=SimpleNamespace(
                list_conversations=list_conversations
            ),
        )
    )
    await controller.load_workspace_tree_page("workspace-7", 0)
    membership_unavailable = True
    await controller.load_workspace_tree_page("workspace-7", 75)

    state = controller._with_console_conversation_browser_state(_workspace_state())
    attempt = controller._workspace_page_attempts["workspace-7"]

    assert [
        row.conversation_id
        for node in state.workspace_tree
        for row in node.conversations
    ] == ["settled"]
    assert state.workspace_tree[0].next_cursor == 75
    assert attempt.error == "Workspace access unknown."
    assert attempt.retry_cursor == 75


def test_projection_uses_canonical_membership_snapshots_without_polling() -> None:
    membership_calls: list[str] = []
    workspace_calls: list[str] = []
    memberships = {
        "workspace-1": ("moving",),
        "workspace-2": ("steady-2",),
        "workspace-3": ("steady-3",),
        DEFAULT_WORKSPACE_ID: (),
    }

    def list_memberships(workspace_id):
        membership_calls.append(workspace_id)
        return tuple(
            SimpleNamespace(item_id=item_id) for item_id in memberships[workspace_id]
        )

    def list_workspaces():
        workspace_calls.append("list")
        return tuple(
            SimpleNamespace(
                workspace_id=workspace_id, name=workspace_id, archived=False
            )
            for workspace_id in memberships
        )

    registry = SimpleNamespace(
        list_workspaces=list_workspaces,
        list_workspace_conversations=list_memberships,
    )
    controller = _workspace_controller(
        app_instance=SimpleNamespace(workspace_registry_service=registry)
    )
    for workspace_id, conversation_id in (
        ("workspace-1", "moving"),
        ("workspace-2", "steady-2"),
        ("workspace-3", "steady-3"),
    ):
        attempt = controller._new_workspace_page_state(
            rows=(
                _browser_row(
                    conversation_id, conversation_id, workspace_id=workspace_id
                ),
            ),
            next_cursor=75,
        )
        attempt.membership_token = memberships[workspace_id]
        controller._workspace_page_attempts[workspace_id] = attempt
    steady_attempts = {
        workspace_id: controller._workspace_page_attempts[workspace_id]
        for workspace_id in ("workspace-2", "workspace-3")
    }

    controller.workspace_tree_projection()
    assert membership_calls == []
    workspace_calls.clear()

    controller.apply_workspace_membership_snapshot(
        {
            "workspace-1": (),
            "workspace-2": memberships["workspace-2"],
            "workspace-3": memberships["workspace-3"],
            DEFAULT_WORKSPACE_ID: ("moving",),
        },
        complete=True,
    )
    assert workspace_calls == []
    projection = controller.workspace_tree_projection()

    assert membership_calls == []
    assert all(
        row.conversation_id != "moving"
        for node in projection
        for row in node.conversations
    )
    for workspace_id, attempt in steady_attempts.items():
        assert controller._workspace_page_attempts[workspace_id] is attempt
        assert [row.conversation_id for row in attempt.rows] == [
            memberships[workspace_id][0]
        ]


def test_production_state_build_uses_canonical_membership_without_fanout() -> None:
    workspace_ids = tuple(f"workspace-{index}" for index in range(50))
    membership_calls: list[str] = []
    registry = SimpleNamespace(
        list_workspaces=lambda: tuple(
            SimpleNamespace(
                workspace_id=workspace_id, name=workspace_id, archived=False
            )
            for workspace_id in workspace_ids
        ),
        list_workspace_conversations=lambda workspace_id: (
            membership_calls.append(workspace_id) or ()
        ),
    )
    controller = _workspace_controller(
        app_instance=SimpleNamespace(workspace_registry_service=registry)
    )
    for index, workspace_id in enumerate(workspace_ids):
        attempt = controller._new_workspace_page_state(
            rows=(
                _browser_row(
                    f"conversation-{index}",
                    f"Conversation {index}",
                    workspace_id=workspace_id,
                ),
            ),
            next_cursor=75,
        )
        attempt.membership_token = (f"conversation-{index}",)
        controller._workspace_page_attempts[workspace_id] = attempt
    untouched = controller._workspace_page_attempts["workspace-49"]

    state = controller._with_console_conversation_browser_state(_workspace_state())

    assert membership_calls == []
    assert sum(len(node.conversations) for node in state.workspace_tree) == 50

    controller.apply_workspace_membership_snapshot(
        {"workspace-0": ()},
        complete=True,
    )
    state = controller._with_console_conversation_browser_state(_workspace_state())

    assert membership_calls == []
    assert all(
        row.conversation_id != "conversation-0"
        for node in state.workspace_tree
        for row in node.conversations
    )
    assert controller._workspace_page_attempts["workspace-49"] is untouched


@pytest.mark.asyncio
async def test_production_publish_reconciles_fresh_default_owner_before_both_projections() -> (
    None
):
    membership_calls: list[str] = []
    registry = SimpleNamespace(
        list_workspaces=lambda: (
            SimpleNamespace(workspace_id="workspace-7", name="Seven", archived=False),
            SimpleNamespace(
                workspace_id=DEFAULT_WORKSPACE_ID, name="Default", archived=False
            ),
        ),
        list_workspace_conversations=lambda workspace_id: (
            membership_calls.append(workspace_id) or ()
        ),
    )
    controller = _workspace_controller(
        app_instance=SimpleNamespace(workspace_registry_service=registry)
    )
    stale_named = _browser_row("moving", "Stale named owner")
    attempt = controller._new_workspace_page_state(rows=(stale_named,), next_cursor=75)
    attempt.membership_token = ("moving",)
    controller._workspace_page_attempts["workspace-7"] = attempt

    async def fresh_flat(_query):
        return (
            (
                _browser_row(
                    "moving",
                    "Fresh Default owner",
                    workspace_id=DEFAULT_WORKSPACE_ID,
                ),
            ),
            1,
        )

    controller._load_flat_conversation_search_rows = fresh_flat
    await controller.refresh_flat_conversation_search("owner")

    state = controller._with_console_conversation_browser_state(_workspace_state())
    assert state.conversation_browser is not None
    owners = [
        *(
            node.workspace_id
            for node in state.workspace_tree
            for row in node.conversations
            if row.conversation_id == "moving"
        ),
        *(
            "flat"
            for section in state.conversation_browser.sections
            for row in section.rows
            if row.conversation_id == "moving"
        ),
    ]

    assert owners == ["flat"]
    assert membership_calls == []


@pytest.mark.asyncio
async def test_production_publish_reconciles_fresh_named_search_owner_before_flat() -> (
    None
):
    membership_calls: list[str] = []
    controller = _workspace_controller(
        app_instance=SimpleNamespace(
            workspace_registry_service=SimpleNamespace(
                list_workspaces=lambda: (
                    SimpleNamespace(
                        workspace_id="workspace-7", name="Seven", archived=False
                    ),
                ),
                list_workspace_conversations=lambda workspace_id: (
                    membership_calls.append(workspace_id) or ()
                ),
            )
        )
    )
    controller._workspace_membership_rows[DEFAULT_WORKSPACE_ID] = (
        _browser_row(
            "moving",
            "Stale Default owner",
            workspace_id=DEFAULT_WORKSPACE_ID,
        ),
    )

    async def fresh_named(_query):
        return ((_browser_row("moving", "Fresh named owner"),), 1)

    controller._load_workspace_tree_search_rows = fresh_named
    await controller.refresh_workspace_tree_search("owner")

    state = controller._with_console_conversation_browser_state(_workspace_state())
    assert state.conversation_browser is not None
    owners = [
        *(
            node.workspace_id
            for node in state.workspace_tree
            for row in node.conversations
            if row.conversation_id == "moving"
        ),
        *(
            "flat"
            for section in state.conversation_browser.sections
            for row in section.rows
            if row.conversation_id == "moving"
        ),
    ]

    assert owners == ["workspace-7"]
    assert membership_calls == []


@pytest.mark.asyncio
async def test_ordinary_refresh_supersedes_cleared_named_search_owner() -> None:
    controller = _workspace_controller(
        app_instance=SimpleNamespace(
            workspace_registry_service=SimpleNamespace(
                list_workspaces=lambda: (
                    SimpleNamespace(
                        workspace_id="workspace-7", name="Seven", archived=False
                    ),
                )
            )
        )
    )

    async def named_search(_query):
        return ((_browser_row("moving", "Old named owner"),), 1)

    controller._load_workspace_tree_search_rows = named_search
    await controller.refresh_workspace_tree_search("owner")
    controller.transition_workspace_tree_search("", disabled=False)

    default_row = _browser_row(
        "moving",
        "Fresh Default owner",
        workspace_id=DEFAULT_WORKSPACE_ID,
    )

    async def ordinary_rows(_query, current_conversation_id=None):
        del current_conversation_id
        return [default_row], 1, ""

    controller._persisted_console_browser_rows = ordinary_rows
    await controller._refresh_console_persisted_rows_cache()

    state = controller._with_console_conversation_browser_state(_workspace_state())
    assert state.conversation_browser is not None
    owners = [
        *(
            node.workspace_id
            for node in state.workspace_tree
            for row in node.conversations
            if row.conversation_id == "moving"
        ),
        *(
            "flat"
            for section in state.conversation_browser.sections
            for row in section.rows
            if row.conversation_id == "moving"
        ),
    ]

    assert owners == ["flat"]


@pytest.mark.asyncio
async def test_current_native_row_supersedes_cleared_named_search_owner() -> None:
    session = SimpleNamespace(
        id="session-moving",
        persisted_conversation_id="moving",
        workspace_id="global",
        title="Fresh native Default owner",
        updated_at="2026-08-22T18:00:00Z",
    )
    store = SimpleNamespace(
        active_session_id="session-moving",
        sessions=lambda: (session,),
    )
    controller = _workspace_controller(
        app_instance=SimpleNamespace(
            workspace_registry_service=SimpleNamespace(
                list_workspaces=lambda: (
                    SimpleNamespace(
                        workspace_id="workspace-7", name="Seven", archived=False
                    ),
                )
            )
        ),
        current_chat_store_accessor=lambda: store,
    )

    async def named_search(_query):
        return ((_browser_row("moving", "Old named owner"),), 1)

    controller._load_workspace_tree_search_rows = named_search
    await controller.refresh_workspace_tree_search("owner")
    controller.transition_workspace_tree_search("", disabled=False)

    state = controller._with_console_conversation_browser_state(_workspace_state())
    assert state.conversation_browser is not None
    owners = [
        *(
            node.workspace_id
            for node in state.workspace_tree
            for row in node.conversations
            if row.conversation_id == "moving"
        ),
        *(
            "flat"
            for section in state.conversation_browser.sections
            for row in section.rows
            if row.conversation_id == "moving"
        ),
    ]

    assert owners == ["flat"]


@pytest.mark.asyncio
async def test_complete_default_snapshot_supersedes_named_observation_without_move() -> (
    None
):
    controller = _workspace_controller(
        app_instance=SimpleNamespace(
            workspace_registry_service=SimpleNamespace(
                list_workspaces=lambda: (
                    SimpleNamespace(
                        workspace_id="workspace-7", name="Seven", archived=False
                    ),
                )
            )
        )
    )
    default_row = _browser_row(
        "moving",
        "Fresh Default owner",
        workspace_id=DEFAULT_WORKSPACE_ID,
    )

    async def ordinary_rows(_query, current_conversation_id=None):
        del current_conversation_id
        return [default_row], 1, ""

    controller._persisted_console_browser_rows = ordinary_rows
    await controller._refresh_console_persisted_rows_cache()

    async def named_search(_query):
        return ((_browser_row("moving", "Old named owner"),), 1)

    controller._load_workspace_tree_search_rows = named_search
    await controller.refresh_workspace_tree_search("owner")

    controller.apply_workspace_membership_snapshot(
        {DEFAULT_WORKSPACE_ID: ("moving",), "workspace-7": ()},
        complete=True,
    )

    state = controller._with_console_conversation_browser_state(_workspace_state())
    assert state.conversation_browser is not None
    owners = [
        *(
            node.workspace_id
            for node in state.workspace_tree
            for row in node.conversations
            if row.conversation_id == "moving"
        ),
        *(
            "flat"
            for section in state.conversation_browser.sections
            for row in section.rows
            if row.conversation_id == "moving"
        ),
    ]

    assert owners == ["flat"]


@pytest.mark.asyncio
async def test_workspace_search_completion_before_complete_owner_move_is_stale() -> (
    None
):
    controller = _workspace_controller(
        app_instance=SimpleNamespace(
            workspace_registry_service=SimpleNamespace(
                list_workspaces=lambda: (
                    SimpleNamespace(
                        workspace_id="workspace-7", name="Seven", archived=False
                    ),
                    SimpleNamespace(
                        workspace_id="workspace-8", name="Eight", archived=False
                    ),
                )
            )
        )
    )
    old_owner = _browser_row("moving", "Old owner")
    attempt = controller._new_workspace_page_state(rows=(old_owner,), next_cursor=75)
    attempt.membership_token = ("moving",)
    controller._workspace_page_attempts["workspace-7"] = attempt
    settled = _browser_row("settled", "Settled search row")
    controller._workspace_tree_search.rows = (settled,)
    controller._workspace_tree_search.settled_rows = (settled,)
    controller._workspace_tree_search.settled_query = "settled"
    started = asyncio.Event()
    release = asyncio.Event()

    async def stale_search(_query):
        started.set()
        await release.wait()
        return (old_owner,), 1

    controller._load_workspace_tree_search_rows = stale_search
    task = asyncio.create_task(controller.refresh_workspace_tree_search("owner"))
    await wait_for_signal(started, what="workspace owner move search start")

    controller.apply_workspace_membership_snapshot(
        {"workspace-7": (), "workspace-8": ("moving",)},
        complete=True,
        workspace_labels={"workspace-7": "Seven", "workspace-8": "Eight"},
    )
    release.set()
    await task

    assert controller._workspace_tree_search.rows == (settled,)
    assert controller._workspace_tree_search.settled_rows == (settled,)
    assert controller._canonical_owner_observations["moving"] == "workspace-8"

    controller.transition_workspace_tree_search("", disabled=False)
    state = controller._with_console_conversation_browser_state(_workspace_state())
    assert state.conversation_browser is not None
    owners = [
        *(
            node.workspace_id
            for node in state.workspace_tree
            for row in node.conversations
            if row.conversation_id == "moving"
        ),
        *(
            "flat"
            for section in state.conversation_browser.sections
            for row in section.rows
            if row.conversation_id == "moving"
        ),
    ]

    assert owners == ["workspace-8"]


@pytest.mark.asyncio
async def test_flat_search_completion_before_complete_owner_move_is_stale() -> None:
    controller = _workspace_controller(
        app_instance=SimpleNamespace(
            workspace_registry_service=SimpleNamespace(
                list_workspaces=lambda: (
                    SimpleNamespace(
                        workspace_id="workspace-8", name="Eight", archived=False
                    ),
                )
            )
        )
    )
    old_owner = _browser_row(
        "moving", "Old Default owner", workspace_id=DEFAULT_WORKSPACE_ID
    )
    controller._workspace_membership_rows[DEFAULT_WORKSPACE_ID] = (old_owner,)
    settled = _browser_row(
        "settled",
        "Settled flat search row",
        workspace_id=DEFAULT_WORKSPACE_ID,
    )
    controller._flat_conversation_search.rows = (settled,)
    controller._flat_conversation_search.settled_rows = (settled,)
    controller._flat_conversation_search.settled_query = "settled"
    started = asyncio.Event()
    release = asyncio.Event()

    async def stale_search(_query):
        started.set()
        await release.wait()
        return (old_owner,), 1

    controller._load_flat_conversation_search_rows = stale_search
    task = asyncio.create_task(controller.refresh_flat_conversation_search("owner"))
    await wait_for_signal(started, what="flat owner move search start")

    controller.apply_workspace_membership_snapshot(
        {DEFAULT_WORKSPACE_ID: (), "workspace-8": ("moving",)},
        complete=True,
        workspace_labels={"workspace-8": "Eight"},
    )
    release.set()
    await task

    assert controller._flat_conversation_search.rows == (settled,)
    assert controller._flat_conversation_search.settled_rows == (settled,)
    assert controller._canonical_owner_observations["moving"] == "workspace-8"

    controller.clear_console_conversation_browser_search()
    state = controller._with_console_conversation_browser_state(_workspace_state())
    assert state.conversation_browser is not None
    owners = [
        *(
            node.workspace_id
            for node in state.workspace_tree
            for row in node.conversations
            if row.conversation_id == "moving"
        ),
        *(
            "flat"
            for section in state.conversation_browser.sections
            for row in section.rows
            if row.conversation_id == "moving"
        ),
    ]

    assert owners == ["workspace-8"]


def test_partial_production_owner_observation_preserves_unobserved_page_members() -> (
    None
):
    controller = _workspace_controller(
        app_instance=SimpleNamespace(
            workspace_registry_service=SimpleNamespace(
                list_workspaces=lambda: (
                    SimpleNamespace(
                        workspace_id="workspace-7", name="Seven", archived=False
                    ),
                )
            )
        )
    )
    attempt = controller._new_workspace_page_state(
        rows=(_browser_row("a", "A"), _browser_row("b", "B")),
        next_cursor=75,
    )
    attempt.membership_token = ("a", "b")
    attempt.generation = 4
    controller._workspace_page_attempts["workspace-7"] = attempt
    controller._console_conversation_browser_rows = (_browser_row("a", "Fresh A"),)

    state = controller._with_console_conversation_browser_state(_workspace_state())

    projected_rows = [
        row for node in state.workspace_tree for row in node.conversations
    ]
    assert [(row.conversation_id, row.title) for row in projected_rows] == [
        ("b", "B"),
        ("a", "Fresh A"),
    ]
    assert attempt.membership_token == ("a", "b")
    assert attempt.next_cursor == 75
    assert attempt.generation == 4


def test_active_workspace_search_overlays_current_star_selection_and_run_marker() -> (
    None
):
    membership_calls: list[str] = []
    current_conversation = "hit"
    unseen_ids = frozenset({"hit"})
    controller = _workspace_controller(
        app_instance=SimpleNamespace(
            workspace_registry_service=SimpleNamespace(
                list_workspaces=lambda: (
                    SimpleNamespace(
                        workspace_id="workspace-7", name="Seven", archived=False
                    ),
                ),
                list_workspace_conversations=lambda workspace_id: (
                    membership_calls.append(workspace_id) or ()
                ),
            ),
            conversation_local_marks_service=SimpleNamespace(
                list_marked_conversation_ids=lambda: ("hit",)
            ),
        ),
        current_conversation_id_accessor=lambda: current_conversation,
        fleet_unseen_ids_accessor=lambda: unseen_ids,
    )
    stale = _browser_row("hit", "Needle", workspace_id="workspace-7")
    controller._workspace_tree_search.query = "needle"
    controller._workspace_tree_search.rows = (stale,)

    projection = controller.workspace_tree_projection()
    row = projection[0].conversations[0]

    assert (row.starred, row.selected, bool(row.run_marker)) == (True, True, True)
    assert controller._workspace_tree_search.rows == (stale,)
    assert membership_calls == []


def test_active_workspace_search_overlays_full_native_marker_truth_without_queries() -> (
    None
):
    service_calls: list[str] = []
    marker_by_session = {
        "session-running": ConsoleRunMarker.RUNNING,
        "session-approval": ConsoleRunMarker.NEEDS_APPROVAL,
        "session-clear": ConsoleRunMarker.NONE,
    }
    sessions = tuple(
        SimpleNamespace(id=session_id, persisted_conversation_id=conversation_id)
        for session_id, conversation_id in (
            ("session-running", "running"),
            ("session-approval", "approval"),
            ("session-clear", "cleared"),
        )
    )
    chat_controller = SimpleNamespace(
        run_marker_for=lambda session_id: marker_by_session[session_id]
    )

    def marker_with_unseen(controller, session, unseen_ids):
        marker = controller.run_marker_for(session.id)
        if (
            marker is ConsoleRunMarker.NONE
            and session.persisted_conversation_id in unseen_ids
        ):
            return ConsoleRunMarker.SUBAGENT_UNSEEN
        return marker

    controller = _workspace_controller(
        app_instance=SimpleNamespace(
            workspace_registry_service=SimpleNamespace(
                list_workspaces=lambda: (
                    SimpleNamespace(
                        workspace_id="workspace-7", name="Seven", archived=False
                    ),
                ),
                list_workspace_conversations=lambda workspace_id: (
                    service_calls.append(workspace_id) or ()
                ),
            ),
            conversation_local_marks_service=SimpleNamespace(
                list_marked_conversation_ids=lambda: ("running",)
            ),
        ),
        current_chat_store_accessor=lambda: SimpleNamespace(sessions=lambda: sessions),
        current_chat_controller_accessor=lambda: chat_controller,
        current_conversation_id_accessor=lambda: "running",
        fleet_unseen_ids_accessor=lambda: frozenset({"unseen"}),
        run_marker_with_unseen=marker_with_unseen,
    )
    rows = (
        _browser_row("running", "Running"),
        _browser_row("approval", "Approval"),
        replace(_browser_row("cleared", "Cleared"), run_marker="✗"),
        _browser_row("unseen", "Unseen"),
        replace(_browser_row("closed", "Closed session"), run_marker="●"),
        replace(_browser_row("acknowledged", "Acknowledged"), run_marker="◈"),
    )
    controller._workspace_tree_search.query = "seven"
    controller._workspace_tree_search.rows = rows

    projected = {
        row.conversation_id: row
        for node in controller.workspace_tree_projection()
        for row in node.conversations
    }

    assert projected["running"].run_marker == resolve_glyph(
        CONSOLE_RUN_MARKER_GLYPHS[ConsoleRunMarker.RUNNING]
    )
    assert projected["approval"].run_marker == resolve_glyph(
        CONSOLE_RUN_MARKER_GLYPHS[ConsoleRunMarker.NEEDS_APPROVAL]
    )
    assert projected["cleared"].run_marker == ""
    assert projected["unseen"].run_marker == resolve_glyph(
        CONSOLE_RUN_MARKER_GLYPHS[ConsoleRunMarker.SUBAGENT_UNSEEN]
    )
    assert projected["closed"].run_marker == ""
    assert projected["acknowledged"].run_marker == ""
    assert (projected["running"].starred, projected["running"].selected) == (
        True,
        True,
    )
    assert service_calls == []


def test_active_flat_search_overlays_current_star_selection_and_run_marker() -> None:
    current_conversation = "flat-hit"
    unseen_ids = frozenset({"flat-hit"})
    controller = _workspace_controller(
        app_instance=SimpleNamespace(
            conversation_local_marks_service=SimpleNamespace(
                list_marked_conversation_ids=lambda: ("flat-hit",)
            )
        ),
        current_conversation_id_accessor=lambda: current_conversation,
        fleet_unseen_ids_accessor=lambda: unseen_ids,
    )
    stale = _browser_row(
        "flat-hit",
        "Needle",
        workspace_id=DEFAULT_WORKSPACE_ID,
    )
    controller._flat_conversation_search.query = "needle"
    controller._flat_conversation_search.rows = (stale,)

    state = controller._with_console_conversation_browser_state(_workspace_state())
    assert state.conversation_browser is not None
    rows = [
        row for section in state.conversation_browser.sections for row in section.rows
    ]

    assert len(rows) == 1
    assert (rows[0].starred, rows[0].selected, bool(rows[0].run_marker)) == (
        True,
        True,
        True,
    )
    assert controller._flat_conversation_search.rows == (stale,)
def test_workspace_files_owner_seam_only_constructs_and_pushes_the_pinned_modal():
    """Task 2's truthful launch seam cannot activate or retarget Console state."""
    pushed: list[ConsoleWorkspaceFilesModal] = []
    screen = _NoMountScreen()
    screen.app = SimpleNamespace(push_screen=lambda modal: pushed.append(modal))
    controller = _workspace_controller(screen=screen)
    inspector = object()
    bindings = (
        WorkspaceFilesBinding("binding-1", "Folder", None, available=False),
    )
    attention = WorkspaceFilesAttention("One generic attention flag.")
    before = (
        controller._console_conversation_browser_query,
        controller._console_conversation_browser_rows,
        controller._console_workspace_conversation_workspace_id,
    )

    controller.open_workspace_files_modal(
        inspector=inspector,
        inspected_workspace_id="inspected-id",
        inspected_workspace_name="Inspected",
        active_workspace_id="active-id",
        active_workspace_name="Active",
        bindings=bindings,
        attention=attention,
    )

    assert len(pushed) == 1
    modal = pushed[0]
    assert modal.inspected_workspace_id == "inspected-id"
    assert modal.inspected_workspace_name == "Inspected"
    assert modal.active_workspace_id == "active-id"
    assert modal.active_workspace_name == "Active"
    assert modal._workspace_bindings == bindings
    assert modal._attention is attention
    assert before == (
        controller._console_conversation_browser_query,
        controller._console_conversation_browser_rows,
        controller._console_workspace_conversation_workspace_id,
    )


def test_workspace_files_owner_seam_fails_closed_for_a_foreign_binding_scope():
    """A modal labeled for A must never receive readable scope authority for B."""
    pushed: list[ConsoleWorkspaceFilesModal] = []
    screen = _NoMountScreen()
    screen.app = SimpleNamespace(push_screen=lambda modal: pushed.append(modal))
    controller = _workspace_controller(screen=screen)
    foreign_scope = BindingScope("workspace-b", "binding-b", "fp", "/b", 1, 2)
    valid_scope = BindingScope("workspace-a", "binding-a", "fp", "/a", 3, 4)

    controller.open_workspace_files_modal(
        inspector=object(),
        inspected_workspace_id="workspace-a",
        inspected_workspace_name="A",
        active_workspace_id="workspace-a",
        active_workspace_name="A",
        bindings=(
            WorkspaceFilesBinding("binding-a", "Valid", valid_scope),
            WorkspaceFilesBinding("binding-b", "Foreign", foreign_scope),
        ),
    )

    valid, foreign = pushed[0]._workspace_bindings
    assert valid.available and valid.scope is valid_scope
    assert not foreign.available and foreign.scope is None
    assert "different workspace" in foreign.availability_copy


def test_workspace_controller_constructor_documents_every_dependency():
    docstring = inspect.getdoc(ConsoleWorkspaceController.__init__) or ""
    parameters = inspect.signature(ConsoleWorkspaceController.__init__).parameters

    assert "Args:" in docstring
    for name in parameters.keys() - {"self"}:
        assert f"{name}:" in docstring


def test_workspace_state_build_never_lists_persisted_conversations_inline():
    calls: list[dict[str, object]] = []

    class LocalService:
        db = SimpleNamespace(is_memory_db=True)

        def list_conversations(self, **kwargs):
            calls.append(kwargs)
            return {"items": [], "pagination": {"total": 0}}

    screen = _NoMountScreen()
    controller = _workspace_controller(
        screen=screen,
        app_instance=SimpleNamespace(
            local_chat_conversation_service=LocalService(),
            chat_conversation_scope_service=None,
        ),
    )

    controller._with_console_conversation_browser_state(_workspace_state())
    controller._with_console_conversation_browser_state(_workspace_state())

    assert calls == []
    assert len(screen.workers) == 1
    assert screen.workers[0][1] == {
        "group": "console-persisted-browser-cache",
        "exclusive": True,
    }


@pytest.mark.asyncio
async def test_workspace_persisted_cache_refresh_awaits_async_provider():
    calls: list[dict[str, object]] = []

    async def list_conversations(**kwargs):
        calls.append(kwargs)
        if kwargs["scope_type"] != "global":
            return {"items": [], "total": 0}
        return {
            "items": [
                {
                    "id": "persisted-1",
                    "title": "Persisted chat",
                    "scope_type": "global",
                }
            ],
            "total": 1,
        }

    controller = _workspace_controller(
        app_instance=SimpleNamespace(
            chat_conversation_scope_service=SimpleNamespace(
                list_conversations=list_conversations,
                local_service=None,
            )
        )
    )

    rows, total, error = await controller._refresh_console_persisted_rows_cache()

    assert [row.conversation_id for row in rows] == ["persisted-1"]
    assert total == 1
    assert error == ""
    assert controller._console_persisted_rows_cache == (rows, total, error)
    assert calls[0]["limit"] == CONSOLE_CONVERSATION_BROWSER_RESULT_LIMIT


@pytest.mark.asyncio
async def test_persisted_cache_refresh_before_complete_owner_move_is_stale() -> None:
    screen = _NoMountScreen()
    controller = _workspace_controller(
        screen=screen,
        app_instance=SimpleNamespace(
            workspace_registry_service=SimpleNamespace(
                list_workspaces=lambda: (
                    SimpleNamespace(
                        workspace_id="workspace-8", name="Eight", archived=False
                    ),
                )
            )
        ),
    )
    old_owner = _browser_row(
        "moving", "Old Default owner", workspace_id=DEFAULT_WORKSPACE_ID
    )
    controller._workspace_membership_rows[DEFAULT_WORKSPACE_ID] = (old_owner,)
    started = asyncio.Event()
    release = asyncio.Event()

    async def stale_rows(_query, current_conversation_id=None):
        del current_conversation_id
        started.set()
        await release.wait()
        return [old_owner], 1, ""

    controller._persisted_console_browser_rows = stale_rows
    refresh_key = ("", None, controller._console_persisted_rows_cache_token)
    controller._console_persisted_rows_refresh_key = refresh_key
    task = asyncio.create_task(
        controller._refresh_console_persisted_rows_cache(refresh_key=refresh_key)
    )
    await wait_for_signal(started, what="persisted cache refresh start")

    controller.apply_workspace_membership_snapshot(
        {DEFAULT_WORKSPACE_ID: (), "workspace-8": ("moving",)},
        complete=True,
        workspace_labels={"workspace-8": "Eight"},
    )
    release.set()
    await task

    assert controller._console_persisted_rows_cache is None
    assert controller._console_persisted_rows_refresh_key is None
    assert controller._canonical_owner_observations["moving"] == "workspace-8"

    state = controller._with_console_conversation_browser_state(_workspace_state())
    controller._with_console_conversation_browser_state(_workspace_state())
    assert state.conversation_browser is not None
    owners = [
        *(
            node.workspace_id
            for node in state.workspace_tree
            for row in node.conversations
            if row.conversation_id == "moving"
        ),
        *(
            "flat"
            for section in state.conversation_browser.sections
            for row in section.rows
            if row.conversation_id == "moving"
        ),
    ]

    assert owners == ["workspace-8"]
    assert len(screen.workers) == 1
    assert controller._console_persisted_rows_refresh_key == refresh_key


def test_workspace_controller_projects_canonical_rows_to_legacy_rows():
    controller = _workspace_controller()
    rich = _rich_row()

    controller._console_conversation_browser_rows = (rich,)

    assert controller._console_workspace_conversation_search_rows == (
        ConsoleWorkspaceConversationRow(
            conversation_id="conversation-7",
            title="Canonical row",
            status="active",
            selected=True,
        ),
    )
    assert controller._console_conversation_browser_rows == (rich,)
    assert controller._console_conversation_browser_rows[0] is rich


def test_workspace_controller_converts_legacy_rows_to_canonical_rich_rows():
    controller = _workspace_controller()
    legacy = ConsoleWorkspaceConversationRow(
        conversation_id="legacy-3",
        title="Legacy row",
        status="saved",
        selected=True,
    )

    controller._console_workspace_conversation_search_rows = (legacy,)

    assert controller._console_conversation_browser_rows == (
        ConsoleConversationBrowserInputRow(
            row_key="legacy-3",
            conversation_id="legacy-3",
            native_session_id=None,
            title="Legacy row",
            scope_type="workspace",
            workspace_id=DEFAULT_WORKSPACE_ID,
            workspace_label="Chats",
            status="saved",
            selected=True,
            source_kind="persisted",
        ),
    )
    controller._console_conversation_browser_rows = (_rich_row(),)
    assert all(
        isinstance(row, ConsoleConversationBrowserInputRow)
        for row in controller._console_conversation_browser_rows
    )


@pytest.mark.asyncio
async def test_session_switcher_rows_include_unmaterialized_named_workspace_history():
    registry = SimpleNamespace(
        ensure_default_workspace=lambda: None,
        list_workspaces=lambda: (
            SimpleNamespace(workspace_id="workspace-8", name="Workspace 8"),
            SimpleNamespace(workspace_id="workspace-9", name="Workspace 9"),
        ),
        list_workspace_conversations=lambda workspace_id: (
            (
                SimpleNamespace(
                    item_id="saved-unmaterialized",
                    title="Saved unmaterialized",
                    role="workspace-thread",
                    created_at="2026-08-23T00:00:00Z",
                ),
            )
            if workspace_id == "workspace-9"
            else ()
        ),
    )
    controller = _workspace_controller(
        app_instance=SimpleNamespace(workspace_registry_service=registry)
    )
    native = _browser_row(
        "open-native",
        "Open native",
        workspace_id="workspace-8",
        source_kind="native",
    )
    named = _browser_row(
        "saved-named",
        "Saved named",
        workspace_id="workspace-8",
    )
    flat = _browser_row(
        "saved-default",
        "Saved default",
        workspace_id=DEFAULT_WORKSPACE_ID,
    )
    controller._native_console_browser_rows = lambda _current=None: [native]
    controller._workspace_membership_rows["workspace-8"] = (named,)

    async def load_flat_history(_current_conversation_id=None):
        return (flat,)

    controller._load_flat_conversation_history_rows = load_flat_history

    rows = await controller.console_session_switcher_rows()

    assert [row.conversation_id for row in rows] == [
        "open-native",
        "saved-unmaterialized",
        "saved-named",
        "saved-default",
    ]


@pytest.mark.asyncio
async def test_session_switcher_rows_load_flat_history_without_a_warm_cache():
    calls: list[dict[str, object]] = []

    global_items = [
        {
            "id": f"cold-global-{index}",
            "title": f"Cold global {index}",
            "scope_type": "global",
        }
        for index in range(80)
    ]
    default_items = [
        {
            "id": f"cold-default-{index}",
            "title": f"Cold default {index}",
            "scope_type": "workspace",
            "workspace_id": DEFAULT_WORKSPACE_ID,
        }
        for index in range(80)
    ]

    async def list_conversations(**kwargs):
        calls.append(kwargs)
        items = global_items if kwargs["scope_type"] == "global" else default_items
        offset = int(kwargs["offset"])
        limit = int(kwargs["limit"])
        return {"items": items[offset : offset + limit], "total": len(items)}

    controller = _workspace_controller(
        app_instance=SimpleNamespace(
            chat_conversation_scope_service=SimpleNamespace(
                list_conversations=list_conversations
            )
        )
    )

    rows = await controller.console_session_switcher_rows()

    conversation_ids = [row.conversation_id for row in rows]
    assert set(conversation_ids) == {
        *(item["id"] for item in global_items),
        *(item["id"] for item in default_items),
    }
    assert len(conversation_ids) == 160
    assert [(call["scope_type"], call["offset"]) for call in calls] == [
        ("global", 0),
        ("global", CONSOLE_CONVERSATION_BROWSER_RESULT_LIMIT),
        ("workspace", 0),
        ("workspace", CONSOLE_CONVERSATION_BROWSER_RESULT_LIMIT),
    ]


@pytest.mark.asyncio
async def test_session_switcher_replaces_stale_named_owner_with_global_history():
    registry = SimpleNamespace(
        ensure_default_workspace=lambda: None,
        list_workspaces=lambda: (
            SimpleNamespace(workspace_id="workspace-old", name="Old workspace"),
        ),
        list_workspace_conversations=lambda _workspace_id: (),
    )

    async def list_conversations(**kwargs):
        if kwargs["scope_type"] != "global":
            return {"items": [], "total": 0}
        return {
            "items": [
                {
                    "id": "moved-global",
                    "title": "Moved global",
                    "scope_type": "global",
                }
            ],
            "total": 1,
        }

    controller = _workspace_controller(
        app_instance=SimpleNamespace(
            workspace_registry_service=registry,
            chat_conversation_scope_service=SimpleNamespace(
                list_conversations=list_conversations
            ),
        )
    )
    controller._canonical_owner_observations["moved-global"] = "workspace-old"

    rows = await controller.console_session_switcher_rows()

    assert [row.conversation_id for row in rows] == ["moved-global"]
    assert (
        controller._canonical_owner_observations["moved-global"] == DEFAULT_WORKSPACE_ID
    )


def test_workspace_controller_replaces_timer_and_filters_rows_immediately():
    scheduled: list[tuple[float, object, _FakeTimer]] = []

    def schedule(delay, callback):
        timer = _FakeTimer()
        scheduled.append((delay, callback, timer))
        return timer

    controller = _workspace_controller(schedule_timer=schedule)
    controller._console_conversation_browser_rows = (
        _browser_row("alpha", "Alpha chat"),
        _browser_row("beta", "Beta chat"),
    )

    controller.transition_browser_search("alpha", disabled=False)
    first_timer = scheduled[-1][2]
    controller.transition_browser_search("alpha ", disabled=False)

    assert first_timer.stop_calls == 1
    assert [delay for delay, _callback, _timer in scheduled] == [0.2, 0.2]
    assert controller._console_conversation_browser_search_token == 2
    assert tuple(
        row.row_key for row in controller._console_conversation_browser_rows
    ) == ("alpha",)

    controller.transition_browser_search("alpha ", disabled=False)
    controller.transition_browser_search("ignored", disabled=True)
    assert len(scheduled) == 2
    assert controller._console_conversation_browser_search_token == 2


@pytest.mark.asyncio
async def test_workspace_controller_stale_token_and_query_refreshes_are_noops():
    sync_calls: list[None] = []
    controller = _workspace_controller(
        sync_workspace_context=lambda: sync_calls.append(None)
    )
    original = (_browser_row("current", "Current chat"),)
    controller._console_conversation_browser_query = "current"
    controller._console_conversation_browser_search_token = 4
    controller._console_conversation_browser_rows = original

    await controller._refresh_console_conversation_browser_search("current", 3)
    await controller._refresh_console_conversation_browser_search("stale", 4)

    assert controller._console_conversation_browser_rows == original
    assert sync_calls == []


def test_workspace_controller_blank_query_clears_without_mount():
    screen = _NoMountScreen()
    sync_calls: list[None] = []
    controller = _workspace_controller(
        screen=screen,
        sync_workspace_context=lambda: sync_calls.append(None),
    )
    controller._console_conversation_browser_search_token = 2
    controller._console_conversation_browser_rows = (_rich_row(),)
    controller._console_conversation_browser_total = 5
    controller._console_conversation_browser_error = "old"
    controller._console_persisted_rows_cache = (_rich_row(),)

    controller._start_console_conversation_browser_search("", 2)

    assert controller._console_conversation_browser_rows == ()
    assert controller._console_conversation_browser_total is None
    assert controller._console_conversation_browser_error == ""
    assert controller._console_persisted_rows_cache is None
    assert sync_calls == [None]
    assert screen.after_refresh == [
        controller._focus_console_workspace_conversation_search
    ]


@pytest.mark.asyncio
async def test_workspace_controller_refresh_stages_local_then_merges_persisted_rows():
    started = asyncio.Event()
    release = asyncio.Event()
    workspace = SimpleNamespace(workspace_id=DEFAULT_WORKSPACE_ID, name="Default")
    registry = SimpleNamespace(
        ensure_default_workspace=lambda: workspace,
        list_workspaces=lambda: (workspace,),
        list_workspace_conversations=lambda _workspace_id: (
            SimpleNamespace(
                item_id="shared",
                title="Shared local",
                role="member",
                created_at="1",
            ),
        ),
    )

    async def list_conversations(**kwargs):
        started.set()
        await release.wait()
        if kwargs["scope_type"] == "global":
            return {"items": [], "total": 0}
        return {
            "items": [
                {
                    "id": "shared",
                    "title": "Shared persisted",
                    "workspace_id": DEFAULT_WORKSPACE_ID,
                    "scope_type": "workspace",
                },
                {
                    "id": "persisted-only",
                    "title": "Persisted only",
                    "workspace_id": DEFAULT_WORKSPACE_ID,
                    "scope_type": "workspace",
                },
            ],
            "total": 7,
        }

    app = SimpleNamespace(
        workspace_registry_service=registry,
        chat_conversation_scope_service=SimpleNamespace(
            list_conversations=list_conversations,
            local_service=None,
        ),
    )
    controller = _workspace_controller(app_instance=app)
    controller._workspace_membership_rows[DEFAULT_WORKSPACE_ID] = (
        _browser_row(
            "shared",
            "Shared local",
            workspace_id=DEFAULT_WORKSPACE_ID,
        ),
    )
    controller._console_conversation_browser_query = "shared"
    controller._console_conversation_browser_search_token = 1

    refresh = asyncio.create_task(
        controller._refresh_console_conversation_browser_search("shared", 1)
    )
    await wait_for_background_signal(
        started,
        refresh,
        what="the persisted conversation refresh to start",
    )

    assert [row.title for row in controller._console_conversation_browser_rows] == [
        "Shared local"
    ]
    assert controller._console_conversation_browser_total is None

    release.set()
    await await_background_task(
        refresh,
        what="the persisted conversation refresh to finish",
    )

    assert [row.title for row in controller._console_conversation_browser_rows] == [
        "Shared local",
        "Persisted only",
    ]
    assert controller._console_conversation_browser_total == 7
    assert controller._console_conversation_browser_error == ""


@pytest.mark.asyncio
async def test_workspace_controller_refresh_preserves_local_rows_on_sanitized_error():
    workspace = SimpleNamespace(workspace_id=DEFAULT_WORKSPACE_ID, name="Default")
    registry = SimpleNamespace(
        ensure_default_workspace=lambda: workspace,
        list_workspaces=lambda: (workspace,),
        list_workspace_conversations=lambda _workspace_id: (
            SimpleNamespace(
                item_id="local",
                title="Local chat",
                role="member",
                created_at="1",
            ),
        ),
    )

    async def fail_search(**_kwargs):
        raise RuntimeError("secret database detail")

    app = SimpleNamespace(
        workspace_registry_service=registry,
        chat_conversation_scope_service=SimpleNamespace(
            list_conversations=fail_search,
            local_service=None,
        ),
    )
    controller = _workspace_controller(app_instance=app)
    controller._workspace_membership_rows[DEFAULT_WORKSPACE_ID] = (
        _browser_row(
            "local",
            "Local chat",
            workspace_id=DEFAULT_WORKSPACE_ID,
        ),
    )
    controller._console_conversation_browser_query = "local"
    controller._console_conversation_browser_search_token = 1

    await controller._refresh_console_conversation_browser_search("local", 1)

    assert [row.title for row in controller._console_conversation_browser_rows] == [
        "Local chat"
    ]
    assert controller._console_conversation_browser_total == 1
    assert (
        controller._console_conversation_browser_error
        == "Conversation search is unavailable."
    )
    assert "secret" not in controller._console_conversation_browser_error


@pytest.mark.asyncio
async def test_workspace_controller_selection_refresh_stops_timer_and_uses_new_token():
    timer = _FakeTimer()
    sync_calls: list[None] = []
    controller = _workspace_controller(
        sync_workspace_context=lambda: sync_calls.append(None)
    )
    controller._console_conversation_browser_query = "alpha"
    controller._console_conversation_browser_search_token = 8
    controller._console_conversation_browser_search_timer = timer

    await controller._refresh_console_conversation_browser_after_selection()

    assert timer.stop_calls == 1
    assert controller._console_conversation_browser_search_timer is None
    assert controller._console_conversation_browser_search_token == 9
    assert sync_calls == [None, None]


def test_workspace_change_preserves_independent_flat_search_state():
    workspace = SimpleNamespace(workspace_id="workspace-new")
    app = SimpleNamespace(
        workspace_registry_service=SimpleNamespace(
            ensure_default_workspace=lambda: workspace
        )
    )
    controller = _workspace_controller(app_instance=app)
    controller._console_workspace_conversation_workspace_id = "workspace-old"
    controller._console_conversation_browser_query = "old"
    controller._console_conversation_browser_search_token = 3
    controller._console_conversation_browser_rows = (_rich_row(),)
    controller._console_conversation_browser_total = 4
    controller._console_conversation_browser_error = "old error"

    state = controller._with_console_workspace_conversation_section(_workspace_state())

    assert controller._console_workspace_conversation_workspace_id == "workspace-new"
    assert controller._console_conversation_browser_query == "old"
    assert controller._console_conversation_browser_search_token == 3
    assert controller._console_conversation_browser_rows == (_rich_row(),)
    assert controller._console_conversation_browser_total == 4
    assert controller._console_conversation_browser_error == "old error"
    assert state.conversation_section is not None
    assert state.conversation_section.workspace_id == "workspace-new"


def test_workspace_change_does_not_cancel_or_invalidate_flat_lane():
    workspace = SimpleNamespace(workspace_id="workspace-new")
    app = SimpleNamespace(
        workspace_registry_service=SimpleNamespace(
            ensure_default_workspace=lambda: workspace
        )
    )
    timer = _FakeTimer()
    controller = _workspace_controller(app_instance=app)
    controller._console_workspace_conversation_workspace_id = "workspace-old"
    controller._console_conversation_browser_query = "old"
    controller._console_conversation_browser_search_token = 3
    controller._console_conversation_browser_search_timer = timer
    controller._console_conversation_browser_rows = (_rich_row(),)
    controller._console_persisted_rows_cache = ([], 0, "")
    controller._console_persisted_rows_cache_token = 5

    state = controller._with_console_conversation_browser_state(_workspace_state())

    assert controller._console_conversation_browser_query == "old"
    assert controller._console_conversation_browser_rows == (_rich_row(),)
    assert controller._console_conversation_browser_search_token == 3
    assert controller._console_conversation_browser_search_timer is timer
    assert timer.stop_calls == 0
    assert controller._console_persisted_rows_cache == ([], 0, "")
    assert controller._console_persisted_rows_cache_token == 5
    assert state.conversation_browser is not None
    assert state.conversation_browser.query == "old"
    browser_rows = tuple(
        row
        for section in state.conversation_browser.sections
        for row in (
            *section.rows,
            *(row for group in section.groups for row in group.rows),
        )
    )
    assert browser_rows == ()
    assert state.conversation_section is not None
    assert state.conversation_section.query == "old"


def test_workspace_controller_clear_transition_stops_once_and_syncs_and_focuses():
    screen = _NoMountScreen()
    timer = _FakeTimer()
    sync_calls: list[None] = []
    controller = _workspace_controller(
        screen=screen,
        sync_workspace_context=lambda: sync_calls.append(None),
    )
    controller._console_conversation_browser_query = "alpha"
    controller._console_conversation_browser_search_token = 10
    controller._console_conversation_browser_search_timer = timer
    controller._console_conversation_browser_rows = (_rich_row(),)
    controller._console_conversation_browser_total = 4
    controller._console_conversation_browser_error = "old"

    controller.clear_console_conversation_browser_search()

    assert timer.stop_calls == 1
    assert controller._console_conversation_browser_search_timer is None
    assert controller._console_conversation_browser_search_token == 11
    assert controller._console_conversation_browser_query == ""
    assert controller._console_conversation_browser_rows == ()
    assert controller._console_conversation_browser_total is None
    assert controller._console_conversation_browser_error == ""
    assert sync_calls == [None]
    assert screen.after_refresh == [
        controller._focus_console_workspace_conversation_search
    ]


@pytest.mark.parametrize(
    ("legacy_name", "canonical_name", "value"),
    (
        (
            "_console_workspace_conversation_query",
            "_console_conversation_browser_query",
            "q",
        ),
        (
            "_console_workspace_conversation_search_timer",
            "_console_conversation_browser_search_timer",
            _FakeTimer(),
        ),
        (
            "_console_workspace_conversation_search_token",
            "_console_conversation_browser_search_token",
            7,
        ),
        (
            "_console_workspace_conversation_search_total",
            "_console_conversation_browser_total",
            8,
        ),
        (
            "_console_workspace_conversation_search_error",
            "_console_conversation_browser_error",
            "error",
        ),
    ),
)
def test_workspace_controller_scalar_legacy_aliases_share_canonical_state(
    legacy_name,
    canonical_name,
    value,
):
    controller = _workspace_controller()

    setattr(controller, legacy_name, value)
    assert getattr(controller, canonical_name) is value

    replacement = {
        "_console_conversation_browser_query": "",
        "_console_conversation_browser_search_timer": _FakeTimer(),
        "_console_conversation_browser_search_token": 0,
        "_console_conversation_browser_total": None,
        "_console_conversation_browser_error": "",
    }[canonical_name]
    setattr(controller, canonical_name, replacement)
    assert getattr(controller, legacy_name) == replacement


def test_workspace_controller_observes_late_bound_dependency_replacement():
    dependencies = SimpleNamespace(controller=lambda: "before")
    controller = _workspace_controller(
        current_chat_controller_accessor=lambda: dependencies.controller()
    )

    assert controller._console_chat_controller == "before"
    dependencies.controller = lambda: "after"
    assert controller._console_chat_controller == "after"


class _InputChangedEvent:
    """Minimal stand-in for `Input.Changed` -- matches what the handler reads
    (`event.value`, `event.input.disabled`, and `event.stop()`)."""

    def __init__(self, value: str, *, input_widget=None) -> None:
        self.value = value
        self.input = input_widget

    def stop(self) -> None:
        return None


def _set_workspace_search(console, query: str) -> None:
    """Drive the real search-changed handler, mirroring a real keystroke."""
    search = console.query_one("#console-workspace-conversation-search", Input)
    search.value = query
    console.on_console_workspace_conversation_search_changed(
        _InputChangedEvent(query, input_widget=search)
    )


def _conversation_tree_payload(
    conversation_id: str,
    *,
    title: str = "Resumed chat",
    workspace_id: str | None = None,
) -> dict:
    conversation: dict = {"id": conversation_id, "title": title}
    if workspace_id is not None:
        conversation["workspace_id"] = workspace_id
    return {"conversation": conversation, "root_threads": []}


@pytest.mark.asyncio
async def test_search_debounce_mirrors_query_and_bumps_token_and_timer():
    """Typing into the rail search mirrors into workspace state and arms a timer."""
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(
            console, pilot, "#console-workspace-conversation-search"
        )

        _set_workspace_search(console, "alpha")
        await pilot.pause()

        assert console._console_workspace_conversation_query == "alpha"
        # `_console_workspace_conversation_search_token` is a pure mirror of
        # `_console_conversation_browser_search_token` inside this handler
        # (see the handler's own docstring / the module docstring for this
        # test file) -- not an independently incrementing counter, so the
        # only thing worth asserting is that the mirror is faithful.
        assert (
            console._console_workspace_conversation_search_token
            == console._console_conversation_browser_search_token
        )
        assert console._console_workspace_conversation_search_timer is not None


@pytest.mark.asyncio
async def test_search_debounce_empty_query_clears_state_and_cancels_the_old_search():
    """Emptying the search box clears the query and abandons the old search.

    TASK-15454 renamed this from `..._clears_state_synchronously` and dropped
    its "no pending timer" assertion. Backspacing to empty is a keystroke like
    any other, and the clear it triggered ran the same full derivation chain
    (workspace records + labels, one membership SELECT per workspace, starred
    ids, then a 3-instance tray recompose) synchronously on the event loop --
    so it is now debounced with the rest. The pending timer that assertion
    forbade was really a proxy for "no stale search survives", and that is
    asserted directly below instead: the timer that does fire carries the
    EMPTY query, and the debounced callback re-checks the token/query before
    doing anything, so the superseded "alpha" search can never run.

    (The "Clear" button is a separate path and still clears immediately --
    see the `console-workspace-conversation-search-clear` branch in
    `on_button_pressed`, unchanged.)
    """
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(
            console, pilot, "#console-workspace-conversation-search"
        )

        _set_workspace_search(console, "alpha")
        await pilot.pause()
        assert console._console_workspace_conversation_query == "alpha"
        stale_token = console._console_workspace_conversation_search_token

        _set_workspace_search(console, "")
        await pilot.pause()

        # The query mirror moves at once; only the DB work waits.
        assert console._console_workspace_conversation_query == ""
        assert console._console_conversation_browser_query == ""
        assert console._console_workspace_conversation_search_token > stale_token

        # A superseded callback cannot revive the old search.
        console._workspace._start_console_conversation_browser_search(
            "alpha", stale_token
        )
        assert console._console_conversation_browser_query == ""

        await pilot.pause(0.35)
        assert console._console_conversation_browser_rows == ()
        assert console._console_conversation_browser_total is None
        assert console._console_conversation_browser_error == ""


@pytest.mark.asyncio
async def test_search_refresh_populates_rows_from_scope_service():
    """The canonical grouped-browser refresh fills rows from the scope service."""
    app = _build_test_app()
    app.chat_conversation_scope_service = SimpleNamespace(
        list_conversations=lambda **kwargs: {
            "items": [
                {
                    "id": "conv-alpha",
                    "title": "Alpha project",
                    "state": "workspace-thread",
                }
            ],
            "total": 1,
        }
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(
            console, pilot, "#console-workspace-conversation-search"
        )

        console._console_workspace_conversation_query = "alpha"
        token = console._console_workspace_conversation_search_token
        await console._workspace._refresh_console_conversation_browser_search(
            "alpha", token
        )
        await pilot.pause()

        rows = console._console_workspace_conversation_search_rows
        assert any(row.conversation_id == "conv-alpha" for row in rows), rows
        assert console._console_workspace_conversation_search_error == ""


@pytest.mark.asyncio
async def test_search_refresh_ignores_stale_token():
    """A refresh whose token no longer matches current state is a no-op.

    Simulates a slow in-flight refresh (token N-1) that lands after a newer
    keystroke already bumped the token to N -- exactly the race the token
    guard exists to close.
    """
    app = _build_test_app()
    app.chat_conversation_scope_service = SimpleNamespace(
        list_conversations=lambda **kwargs: {
            "items": [
                {
                    "id": "conv-late",
                    "title": "Late arrival",
                    "state": "workspace-thread",
                }
            ],
            "total": 1,
        }
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(
            console, pilot, "#console-workspace-conversation-search"
        )

        console._console_workspace_conversation_query = "late"
        current_token = console._console_workspace_conversation_search_token + 1
        console._console_workspace_conversation_search_token = current_token

        await console._workspace._refresh_console_conversation_browser_search(
            "late", current_token - 1
        )

        assert console._console_workspace_conversation_search_rows == ()


@pytest.mark.asyncio
async def test_resume_workspace_conversation_restores_native_session():
    """Resuming a real persisted conversation creates a matching native session."""
    app = _build_test_app()
    active_workspace = app.workspace_registry_service.get_active_workspace()
    app.chat_conversation_scope_service = SimpleNamespace(
        get_conversation_tree=lambda conversation_id, **kwargs: (
            _conversation_tree_payload(
                conversation_id,
                title="Resumed alpha",
                workspace_id=active_workspace.workspace_id,
            )
        )
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")

        resumed = await console._workspace._resume_console_workspace_conversation(
            "conv-resume-1"
        )
        await pilot.pause()

        assert resumed is True
        store = console._ensure_console_chat_store()
        active_session = store.switch_session(store.active_session_id)
        assert active_session.persisted_conversation_id == "conv-resume-1"
        assert active_session.workspace_id == active_workspace.workspace_id


@pytest.mark.asyncio
async def test_resume_workspace_conversation_missing_record_returns_false():
    """A missing conversation record is reported honestly as False."""
    app = _build_test_app()
    app.chat_conversation_scope_service = SimpleNamespace(
        get_conversation_tree=lambda *args, **kwargs: {}
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")

        resumed = await console._workspace._resume_console_workspace_conversation(
            "conv-missing"
        )
        await pilot.pause()

        assert resumed is False


RESUME_FAILURE_COPY = (
    "Couldn't resume this saved conversation: it was deleted or couldn't be read.\n"
    "Your previous Console chat is still active."
)


def _atomic_resume_controller(
    *,
    failure_boundary: str | None = None,
    failure: BaseException | None = None,
):
    class DurableObserver:
        def __init__(self) -> None:
            self.mutations: list[tuple[str, str]] = []

        def delete_conversation(self, conversation_id: str) -> None:
            self.mutations.append(("delete", conversation_id))

        def update_conversation(self, conversation_id: str) -> None:
            self.mutations.append(("update", conversation_id))

    durable = DurableObserver()
    store = ConsoleChatStore(persistence=durable)
    prior_settings = ConsoleSessionSettings(provider="openai", model="prior")
    prior = store.create_session(title="Prior", settings=prior_settings)
    store.set_session_draft(prior.id, "draft stays exact")
    notifications: list[tuple[str, str, dict]] = []
    app = SimpleNamespace(
        chachanotes_db=SimpleNamespace(
            get_conversation_active_leaf=lambda _target: None,
        ),
        chat_conversation_scope_service=SimpleNamespace(
            get_conversation_tree=lambda conversation_id, **_kwargs: (
                _conversation_tree_payload(conversation_id)
            )
        ),
        notify=lambda message, severity="information", **kwargs: notifications.append(
            (message, severity, kwargs)
        ),
    )
    sync_calls = 0

    def inject_markers(messages, _conversation_id):
        if failure_boundary == "marker":
            raise failure  # type: ignore[misc]
        return messages

    async def resolve_scope(_session):
        if failure_boundary == "scope":
            raise failure  # type: ignore[misc]

    async def sync_ui():
        nonlocal sync_calls
        sync_calls += 1
        if failure_boundary == "ui" and sync_calls == 1:
            raise failure  # type: ignore[misc]

    async def refresh_browser():
        if failure_boundary == "browser":
            raise failure  # type: ignore[misc]

    chat_controller = SimpleNamespace(
        store=store,
        switch_session=store.switch_session,
        activity_for=lambda _session_id: SimpleNamespace(queued_count=0),
    )
    controller = _workspace_controller(
        app_instance=app,
        chat_store_accessor=lambda: store,
        current_chat_store_accessor=lambda: store,
        native_session_rows_accessor=lambda state: state,
        current_chat_controller_accessor=lambda: chat_controller,
        ensure_chat_controller=lambda: chat_controller,
        session_settings_for_resume_accessor=lambda _conversation: prior_settings,
        inject_resume_agent_markers_accessor=inject_markers,
        resolve_effective_scope_state=resolve_scope,
        sync_native_console_chat_ui=sync_ui,
    )
    controller._refresh_console_conversation_browser_after_selection = (
        refresh_browser
    )
    return controller, store, prior, prior_settings, durable, notifications


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure_boundary", ("marker", "scope", "ui", "browser")
)
async def test_open_saved_conversation_atomic_rollback_on_presentation_failure(
    failure_boundary,
):
    from loguru import logger as loguru_logger

    controller, store, prior, prior_settings, durable, notifications = (
        _atomic_resume_controller(
            failure_boundary=failure_boundary,
            failure=RuntimeError(f"failed {failure_boundary}"),
        )
    )
    private_target = "resume-target-private"
    warnings: list[str] = []
    sink_id = loguru_logger.add(
        lambda message: warnings.append(message.record["message"]), level="WARNING"
    )

    try:
        result = await controller.open_console_workspace_conversation(private_target)
    finally:
        loguru_logger.remove(sink_id)

    assert result is None
    assert store.active_session_id == prior.id
    assert store.sessions() == [prior]
    assert prior.settings is prior_settings
    assert prior.draft == "draft stays exact"
    assert durable.mutations == []
    assert notifications == [(RESUME_FAILURE_COPY, "error", {"timeout": 15})]
    assert "Unable to present Console saved conversation" in warnings
    assert all(private_target not in warning for warning in warnings)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure_boundary", ("marker", "scope", "ui", "browser")
)
async def test_open_saved_conversation_cancellation_rolls_back_and_propagates(
    failure_boundary,
):
    controller, store, prior, prior_settings, durable, notifications = (
        _atomic_resume_controller(
            failure_boundary=failure_boundary,
            failure=asyncio.CancelledError(),
        )
    )

    with pytest.raises(asyncio.CancelledError):
        await controller.open_console_workspace_conversation("resume-target")

    assert store.active_session_id == prior.id
    assert store.sessions() == [prior]
    assert prior.settings is prior_settings
    assert prior.draft == "draft stays exact"
    assert durable.mutations == []
    assert notifications == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure", "propagates"),
    ((RuntimeError("sync failed"), False), (asyncio.CancelledError(), True)),
    ids=("ordinary", "cancellation"),
)
async def test_open_already_live_session_sync_failure_is_atomic(
    failure,
    propagates,
):
    controller, store, prior, prior_settings, durable, notifications = (
        _atomic_resume_controller()
    )
    target_settings = ConsoleSessionSettings(provider="openai", model="target")
    target = store.create_session(title="Already live", settings=target_settings)
    target.persisted_conversation_id = "resume-target"
    store.set_session_draft(target.id, "target draft")
    store.switch_session(prior.id)
    sync_calls = 0

    async def fail_once_then_sync():
        nonlocal sync_calls
        sync_calls += 1
        if sync_calls == 1:
            raise failure

    controller._sync_native_console_chat_ui_fn = fail_once_then_sync

    if propagates:
        with pytest.raises(asyncio.CancelledError):
            await controller.open_console_workspace_conversation("resume-target")
    else:
        result = await controller.open_console_workspace_conversation("resume-target")
        assert result is None

    assert store.active_session_id == prior.id
    assert store.sessions() == [prior, target]
    assert prior.settings is prior_settings
    assert prior.draft == "draft stays exact"
    assert target.settings is target_settings
    assert target.draft == "target draft"
    assert durable.mutations == []
    assert notifications == (
        []
        if propagates
        else [(RESUME_FAILURE_COPY, "error", {"timeout": 15})]
    )


def test_resume_match_prefers_active_then_creation_order():
    store = ConsoleChatStore()
    first = store.create_session(title="First match")
    first.persisted_conversation_id = "shared-conversation"
    second = store.create_session(title="Second match")
    second.persisted_conversation_id = "shared-conversation"
    other = store.create_session(title="Other")
    controller = _workspace_controller(
        chat_store_accessor=lambda: store,
        current_chat_store_accessor=lambda: store,
    )

    assert (
        controller._console_session_id_for_workspace_conversation(
            "shared-conversation"
        )
        == first.id
    )

    store.switch_session(second.id)
    assert (
        controller._console_session_id_for_workspace_conversation(
            "shared-conversation"
        )
        == second.id
    )
    assert other in store.sessions()


@pytest.mark.asyncio
async def test_id_only_public_opener_keeps_active_matching_duplicate():
    """ID-only navigation must not inherit grouped-browser dedupe order."""
    controller, store, _prior, _settings, _durable, _notifications = (
        _atomic_resume_controller()
    )
    first = store.create_session(title="First matching session")
    first.persisted_conversation_id = "shared-conversation"
    store.set_session_draft(first.id, "first draft")
    active = store.create_session(title="Active matching session")
    active.persisted_conversation_id = "shared-conversation"
    store.set_session_draft(active.id, "active draft")
    deduped_first_row = ConsoleConversationBrowserRow(
        row_key="shared-conversation",
        conversation_id="shared-conversation",
        native_session_id=first.id,
        title="First matching session",
        scope_type="global",
        workspace_id=None,
        workspace_label="Global",
        status="open session",
        updated_label="now",
    )
    controller._find_console_browser_row = (
        lambda *_args, **_kwargs: deduped_first_row
    )
    controller._session_id_for_browser_row_fn = (
        lambda row: row.native_session_id
    )

    result = await controller.open_console_workspace_conversation(
        "shared-conversation"
    )

    assert result is True
    assert store.active_session_id == active.id
    assert first.draft == "first draft"
    assert active.draft == "active draft"


@pytest.mark.asyncio
async def test_fresh_screen_already_active_resume_performs_final_presentation_sync():
    """Ordered mount still paints an already-active matching target."""
    from Tests.UI.test_console_native_chat_flow import (
        _configure_native_ready_console,
    )

    app = _build_test_app()
    _configure_native_ready_console(app)
    store = app.console_runtime.ensure_chat_store()
    target = store.create_session(title="Already active saved chat")
    target.persisted_conversation_id = "already-active"
    message = store.append_message(
        target.id,
        role=ConsoleMessageRole.USER,
        content="Already-active transcript",
    )
    screen = ChatScreen(app)
    screen.apply_navigation_context(
        {
            CONSOLE_NAV_CONTEXT_RESUME_LOCAL_CONVERSATION_ID: (
                target.persisted_conversation_id
            )
        }
    )
    final_focus = asyncio.Event()
    original_focus = screen._workspace._focus_composer_if_needed_fn

    def record_final_focus(*, force: bool = False) -> None:
        original_focus(force=force)
        final_focus.set()

    screen._workspace._focus_composer_if_needed_fn = record_final_focus

    async with app.run_test(size=(160, 48)) as pilot:
        await app.push_screen(screen)
        app._initial_screen_pushed = True
        await wait_for_signal(
            final_focus,
            what="the already-active resume's final presentation",
        )
        await pilot.pause()

        assert store.active_session_id == target.id
        assert screen.query_one(f"#console-message-{message.id}")
        assert app.focused is screen.query_one("#console-native-composer")


@pytest.mark.asyncio
async def test_resume_missing_returns_false_with_id_only_copy():
    controller, store, prior, prior_settings, durable, notifications = (
        _atomic_resume_controller()
    )
    controller.app_instance.chat_conversation_scope_service = SimpleNamespace(
        get_conversation_tree=lambda *_args, **_kwargs: {}
    )
    broken: list[str] = []
    controller._mark_conversation_row_broken_fn = broken.append

    result = await controller.open_console_workspace_conversation("missing-target")

    assert result is False
    assert store.active_session_id == prior.id
    assert store.sessions() == [prior]
    assert prior.settings is prior_settings
    assert prior.draft == "draft stays exact"
    assert durable.mutations == []
    assert broken == ["missing-target"]
    assert notifications == [(RESUME_FAILURE_COPY, "warning", {"timeout": 15})]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure_boundary", "expected_outcome", "expected_severity"),
    (
        ("missing", False, "warning"),
        ("load", None, "error"),
        ("post_sync", None, "error"),
    ),
)
async def test_mounted_failed_open_repaints_prior_and_forces_composer(
    failure_boundary,
    expected_outcome,
    expected_severity,
):
    """Every terminal/transient open failure restores the mounted prior chat."""
    from Tests.UI.test_console_native_chat_flow import (
        _configure_native_ready_console,
    )

    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        prior = store.ensure_session()
        prior_message = store.append_message(
            prior.id,
            role=ConsoleMessageRole.USER,
            content="Exact prior transcript",
        )
        await console._sync_native_console_chat_ui()
        await _wait_for_selector(
            console, pilot, f"#console-message-{prior_message.id}"
        )
        console._stop_console_transcript_sync_timer()
        transcript = console.query_one("#console-native-transcript")
        transcript.focus()
        await pilot.pause()
        assert host.focused is transcript
        repaint_session_ids = []
        original_sync = console._workspace._sync_native_console_chat_ui_fn

        async def observe_sync():
            repaint_session_ids.append(store.active_session_id)
            await original_sync()

        console._workspace._sync_native_console_chat_ui_fn = observe_sync

        if failure_boundary == "missing":
            app.chat_conversation_scope_service = SimpleNamespace(
                get_conversation_tree=lambda *_args, **_kwargs: {}
            )
        elif failure_boundary == "load":

            def fail_load(*_args, **_kwargs):
                raise RuntimeError("unreadable destination")

            app.chat_conversation_scope_service = SimpleNamespace(
                get_conversation_tree=fail_load
            )
        else:
            app.chat_conversation_scope_service = SimpleNamespace(
                get_conversation_tree=lambda conversation_id, **_kwargs: (
                    _conversation_tree_payload(conversation_id)
                )
            )

            async def fail_after_target_sync() -> None:
                raise RuntimeError("post-sync presentation failed")

            console._workspace._refresh_console_conversation_browser_after_selection = (
                fail_after_target_sync
            )

        notifications: list[tuple[str, str, dict]] = []
        app.notify = (
            lambda message, severity="information", **kwargs: notifications.append(
                (message, severity, kwargs)
            )
        )

        outcome = await console._workspace.open_console_workspace_conversation(
            "failed-target"
        )
        await pilot.pause()

        assert outcome is expected_outcome
        assert store.active_session_id == prior.id
        assert store.sessions() == [prior]
        assert repaint_session_ids[-1] == prior.id
        assert console.query_one(f"#console-message-{prior_message.id}")
        assert host.focused is console.query_one("#console-native-composer")
        assert notifications == [
            (RESUME_FAILURE_COPY, expected_severity, {"timeout": 15})
        ]


class _AtomicWorkspaceRegistry:
    def __init__(self) -> None:
        self._records = {
            "workspace-prior": SimpleNamespace(
                workspace_id="workspace-prior", name="Prior workspace"
            ),
            "workspace-target": SimpleNamespace(
                workspace_id="workspace-target", name="Target workspace"
            ),
        }
        self._active = self._records["workspace-prior"]
        self.mutation_generation = 0

    def get_active_workspace(self):
        return self._active

    def set_active_workspace(self, workspace_id: str) -> None:
        self._active = self._records[workspace_id]
        self.mutation_generation += 1


def _cross_workspace_browser_row() -> ConsoleConversationBrowserRow:
    return ConsoleConversationBrowserRow(
        row_key="workspace:workspace-target:resume-target",
        conversation_id="resume-target",
        native_session_id=None,
        title="Target chat",
        scope_type="workspace",
        workspace_id="workspace-target",
        workspace_label="Target workspace",
        status="saved chat",
        updated_label="now",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("load_failure", ("missing", "transient"))
async def test_browser_row_resume_failure_preserves_prior_workspace_context(
    load_failure,
):
    controller, store, prior, prior_settings, durable, notifications = (
        _atomic_resume_controller()
    )
    registry = _AtomicWorkspaceRegistry()
    controller.app_instance.workspace_registry_service = registry
    prior_context = ConsoleWorkspaceContext(
        active_workspace_id="workspace-prior"
    )
    store.set_workspace_context(prior_context)
    row = _cross_workspace_browser_row()
    controller._find_console_browser_row = lambda *_args, **_kwargs: row

    if load_failure == "missing":
        controller.app_instance.chat_conversation_scope_service = SimpleNamespace(
            get_conversation_tree=lambda *_args, **_kwargs: {}
        )
    else:
        def fail_load(*_args, **_kwargs):
            raise RuntimeError("service failed")

        controller.app_instance.chat_conversation_scope_service = SimpleNamespace(
            get_conversation_tree=fail_load
        )

    result = await controller.open_console_workspace_conversation(
        "resume-target", row_key=row.row_key
    )

    assert result is (False if load_failure == "missing" else None)
    assert registry.get_active_workspace().workspace_id == "workspace-prior"
    assert store.workspace_context is prior_context
    assert store.active_session_id == prior.id
    assert prior.settings is prior_settings
    assert durable.mutations == []
    assert notifications == (
        [(RESUME_FAILURE_COPY, "warning", {"timeout": 15})]
        if load_failure == "missing"
        else [(RESUME_FAILURE_COPY, "error", {"timeout": 15})]
    )


@pytest.mark.asyncio
async def test_resume_loading_teardown_failure_cannot_replace_success():
    controller, store, prior, _settings, _durable, _notifications = (
        _atomic_resume_controller()
    )

    def set_loading(_conversation_id: str, loading: bool) -> None:
        if not loading:
            raise RuntimeError("teardown failed")

    controller._set_conversation_row_loading_fn = set_loading

    result = await controller.open_console_workspace_conversation("resume-target")

    assert result is True
    assert store.active_session_id != prior.id
    assert len(store.sessions()) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("resume_outcome", (False, None))
async def test_resume_loading_teardown_failure_cannot_replace_tristate_failure(
    resume_outcome,
):
    controller, _store, _prior, _settings, _durable, notifications = (
        _atomic_resume_controller()
    )

    async def resume(*_args, **_kwargs):
        return resume_outcome

    def set_loading(_conversation_id: str, loading: bool) -> None:
        if not loading:
            raise RuntimeError("teardown failed")

    controller._resume_console_workspace_conversation = resume
    controller._set_conversation_row_loading_fn = set_loading

    result = await controller.open_console_workspace_conversation("resume-target")

    assert result is resume_outcome
    assert notifications == (
        [(RESUME_FAILURE_COPY, "warning", {"timeout": 15})]
        if resume_outcome is False
        else []
    )


@pytest.mark.asyncio
async def test_resume_loading_teardown_failure_cannot_replace_cancellation():
    controller, store, prior, _settings, _durable, notifications = (
        _atomic_resume_controller(
            failure_boundary="marker", failure=asyncio.CancelledError()
        )
    )

    def set_loading(_conversation_id: str, loading: bool) -> None:
        if not loading:
            raise RuntimeError("teardown failed")

    controller._set_conversation_row_loading_fn = set_loading

    with pytest.raises(asyncio.CancelledError):
        await controller.open_console_workspace_conversation("resume-target")

    assert store.active_session_id == prior.id
    assert store.sessions() == [prior]
    assert notifications == []


@pytest.mark.asyncio
async def test_resume_composer_focus_failure_rolls_back_without_scheduling_retry():
    controller, store, prior, _settings, _durable, notifications = (
        _atomic_resume_controller()
    )
    retry_calls: list[str] = []
    controller._wake_retry_poke_fn = lambda: retry_calls.append("retry")

    def fail_focus(*, force: bool) -> None:
        raise RuntimeError(f"focus failed: {force}")

    controller._focus_composer_if_needed_fn = fail_focus

    result = await controller.open_console_workspace_conversation("resume-target")

    assert result is None
    assert store.active_session_id == prior.id
    assert store.sessions() == [prior]
    assert retry_calls == []
    assert notifications == [(RESUME_FAILURE_COPY, "error", {"timeout": 15})]


@pytest.mark.asyncio
async def test_resume_retry_is_scheduled_only_after_successful_composer_focus():
    controller, _store, _prior, _settings, _durable, _notifications = (
        _atomic_resume_controller()
    )
    events: list[str] = []
    controller._focus_composer_if_needed_fn = (
        lambda *, force: events.append(f"focus:{force}")
    )
    controller._wake_retry_poke_fn = lambda: events.append("retry")

    result = await controller.open_console_workspace_conversation("resume-target")

    assert result is True
    assert events == ["focus:True", "retry"]


@pytest.mark.asyncio
async def test_active_workspace_id_for_conversation_search_reads_registry():
    """Falls back to the registry's active workspace when none is staged."""
    app = _build_test_app()
    active_workspace = app.workspace_registry_service.get_active_workspace()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")

        assert (
            console._workspace._active_console_workspace_id_for_conversation_search()
            == active_workspace.workspace_id
        )


@pytest.mark.asyncio
async def test_fresh_console_runtime_starts_in_registry_active_workspace():
    """TASK-2033: a prebuilt empty runtime must not fall back to Default."""
    app = _build_test_app()
    store = app.console_runtime.ensure_chat_store()
    assert store.active_session_id is None

    workspace = app.workspace_registry_service.create_workspace(
        workspace_id="workspace-startup",
        name="Startup Workspace",
    )
    app.workspace_registry_service.set_active_workspace(workspace.workspace_id)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        for _ in range(200):
            if store.active_session_id is not None:
                break
            await pilot.pause(0.01)

        assert store.workspace_context.active_workspace_id == workspace.workspace_id
        active_session = store.ensure_session()
        assert active_session.workspace_id == workspace.workspace_id
        assert len(store.sessions()) == 1


async def _wait_until(pilot, predicate, *, timeout: float = 5.0, detail: str):
    """Poll ``predicate`` between pilot pauses until true or timeout."""
    import time as _time

    deadline = _time.monotonic() + timeout
    last = None
    while _time.monotonic() < deadline:
        last = predicate()
        if last:
            return
        await pilot.pause(0.05)
    raise AssertionError(f"Timed out waiting for {detail}; last value: {last!r}")


def _mounted_section_budgets(console):
    """Return (workspace, conversations) bounded-section ceilings as a dict."""
    from tldw_chatbook.Widgets.Console.console_bounded_section import (
        ConsoleBoundedSection,
    )

    budgets = {}
    for section_id in ("workspace", "conversations"):
        bounded = console.query_one(
            f"#console-bounded-section-{section_id}", ConsoleBoundedSection
        )
        budgets[section_id] = bounded.max_content_lines
    return budgets


@pytest.mark.asyncio
async def test_rail_height_drives_adaptive_cap_through_the_mounted_ui(tmp_path):
    """Mounted integration: measure -> accessor wiring -> caps -> resize.

    The pure budget/row-limit math and the controller seam are unit-tested
    elsewhere; this test exercises the real boundary end to end -- ChatScreen
    measuring the rail body, the late-bound ``rail_body_height_accessor``
    feeding the DOM-free controller, both peer bounded sections' ceilings
    following the shared budget, the mounted Chats list showing more rows
    than the 12-row historical default on a tall terminal, and the whole
    chain re-syncing after a terminal resize.
    """
    from tldw_chatbook.Chat.chat_conversation_scope_service import (
        ChatConversationScopeService,
    )
    from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    chat_db = CharactersRAGDB(tmp_path / "adaptive_cap_chats.db", "test-client")
    chat_service = ChatConversationService(chat_db)
    for index in range(40):
        chat_service.create_conversation(
            id=f"global-chat-{index}",
            title=f"Global chat {index}",
            scope_type="global",
        )
    app = _build_test_app()
    # Same injection seam `test_console_native_chat_flow.py` uses: replace
    # the scope service so the controller's persisted-rows path reads the
    # tmp DB backed conversations above.
    app.chat_conversation_scope_service = ChatConversationScopeService(
        local_service=chat_service,
        server_service=None,
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 160)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(
            console, pilot, "#console-workspace-context"
        )

        # The screen measures the mounted rail body and the budget grows
        # past the historical 20-line ceiling on this tall terminal.
        measured = console._console_rail_body_height()
        assert measured is not None and measured > 0
        expected_budget = console_rail_section_height_budget(measured)
        assert expected_budget > CONSOLE_RAIL_SECTION_MIN_BUDGET_LINES

        # Both peer list sections' ceilings follow the shared budget.
        await _wait_until(
            pilot,
            lambda: _mounted_section_budgets(console)
            == {"workspace": expected_budget, "conversations": expected_budget},
            detail=f"adaptive section ceilings == {expected_budget}",
        )

        # The real wiring reaches the Chats row cap: the mounted browser
        # carries more visible rows than the 12-row historical default
        # (the first state build kicks the persisted-rows cache worker, so
        # poll until the seeded conversations land).
        def _mounted_chats_row_count():
            state = console._workspace._build_console_workspace_context_state()
            browser = state.conversation_browser
            if browser is None:
                return 0
            (chats,) = browser.sections
            return len(chats.rows)

        await _wait_until(
            pilot,
            lambda: _mounted_chats_row_count()
            > CONSOLE_CONVERSATION_BROWSER_GROUP_ROW_LIMIT,
            detail="mounted Chats rows exceed the 12-row default",
        )
        assert _mounted_chats_row_count() == console_conversation_browser_group_row_limit(
            measured
        )

        # Shrinking the terminal re-syncs the whole chain: the measured
        # rail body shrinks and the section ceilings follow it back down.
        await pilot.resize_terminal(140, 30)
        await _wait_until(
            pilot,
            lambda: (
                console._console_rail_body_height() is not None
                and console._console_rail_body_height() < measured
            ),
            detail="measured rail height shrinks after resize",
        )
        small_measured = console._console_rail_body_height()
        expected_small_budget = console_rail_section_height_budget(small_measured)
        await _wait_until(
            pilot,
            lambda: _mounted_section_budgets(console)
            == {
                "workspace": expected_small_budget,
                "conversations": expected_small_budget,
            },
            detail=f"section ceilings shrink to {expected_small_budget}",
        )
        # A 30-line terminal is below every growth step: the historical
        # ceiling is back.
        assert expected_small_budget == CONSOLE_RAIL_SECTION_MIN_BUDGET_LINES
