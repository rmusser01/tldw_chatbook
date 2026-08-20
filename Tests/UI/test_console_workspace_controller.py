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
from types import SimpleNamespace

import pytest
from textual.widgets import Input

from Tests.UI.background_signals import wait_for_background_signal
from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_destination_shells import _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.UI.Console_Modules.workspace import ConsoleWorkspaceController
from tldw_chatbook.Workspaces import (
    CONSOLE_CONVERSATION_BROWSER_RESULT_LIMIT,
    DEFAULT_WORKSPACE_ID,
    ConsoleConversationBrowserInputRow,
)
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

    def call_after_refresh(self, callback) -> None:
        self.after_refresh.append(callback)

    def run_worker(self, coroutine, **kwargs) -> None:
        self.workers.append((coroutine, kwargs))
        coroutine.close()


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
        resolve_resumed_character_name=_noop,
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
    workspace = SimpleNamespace(workspace_id="workspace-7", name="Workspace 7")
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
                    "workspace_id": "workspace-7",
                    "scope_type": "workspace",
                },
                {
                    "id": "persisted-only",
                    "title": "Persisted only",
                    "workspace_id": "workspace-7",
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
    await refresh

    assert [row.title for row in controller._console_conversation_browser_rows] == [
        "Shared local",
        "Persisted only",
    ]
    assert controller._console_conversation_browser_total == 7
    assert controller._console_conversation_browser_error == ""


@pytest.mark.asyncio
async def test_workspace_controller_refresh_preserves_local_rows_on_sanitized_error():
    workspace = SimpleNamespace(workspace_id="workspace-7", name="Workspace 7")
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
    controller._console_conversation_browser_query = "local"
    controller._console_conversation_browser_search_token = 1

    await controller._refresh_console_conversation_browser_search("local", 1)

    assert [row.title for row in controller._console_conversation_browser_rows] == [
        "Local chat"
    ]
    assert controller._console_conversation_browser_total == 1
    assert (
        controller._console_conversation_browser_error
        == "Workspace conversation search is unavailable."
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


def test_workspace_controller_workspace_change_resets_canonical_search_state():
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
    assert controller._console_conversation_browser_query == ""
    assert controller._console_conversation_browser_search_token == 4
    assert controller._console_conversation_browser_rows == ()
    assert controller._console_conversation_browser_total is None
    assert controller._console_conversation_browser_error == ""
    assert state.conversation_section is not None
    assert state.conversation_section.workspace_id == "workspace-new"


def test_workspace_controller_workspace_change_resets_before_rich_browser_snapshot():
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

    assert controller._console_conversation_browser_query == ""
    assert controller._console_conversation_browser_rows == ()
    assert controller._console_conversation_browser_search_token == 4
    assert controller._console_conversation_browser_search_timer is None
    assert timer.stop_calls == 1
    assert controller._console_persisted_rows_cache is None
    assert controller._console_persisted_rows_cache_token == 6
    assert state.conversation_browser is not None
    assert state.conversation_browser.query == ""
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
    assert state.conversation_section.query == ""


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

    replacement = value if isinstance(value, _FakeTimer) else None
    setattr(controller, canonical_name, replacement)
    assert getattr(controller, legacy_name) is replacement


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
