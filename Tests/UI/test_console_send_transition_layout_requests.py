"""TASK-36012: a send's start and end leave same-size widgets un-laid-out.

TASK-33620.15.7 stopped a run's 0.2 s poll ticks from asking for whole-screen
layouts (``Tests/UI/test_console_run_tick_layout_requests.py``). The send's own
two transitions -- Send accepted, reply finished -- still did. Textual answers
every ``refresh(layout=True)`` or recompose with a ``Screen._refresh_layout``
over every widget on the screen, however small the requester.

Measured headless on origin/dev 0d7e418db1 (2026-10-26, 160x45, a warm second
send in a persisted chat, collapsed Inspector): the send's start made 186
layout requests over 13-15 full reflows and its end 123 over 4-5. The
requesters this module pins, and why none of them needed a layout:

* the Conversations tray (``#console-workspace-context``) recomposed every row
  because the active row gained, then lost, its "Running" attention fact. That
  changes the row's action icon (a fixed 4-cell slot), its colour class and two
  tooltips: no box moves. The recompose then cost the alias button's re-mount
  and two bounded-section reconciles, three more full reflows at the start and
  two at the end;
* the Details tray (``#console-workspace-details``) recomposed for the same
  ``conversation_browser`` delta, a field it never renders;
* the collapsed Inspector rail (``#console-right-rail``) asked for a layout
  twice: once when its badge-only rail state changed (``running``), and once
  for its outer-fold reconcile, which measures nothing while the rail is
  hidden;
* the rail state's badge-only change re-ran the whole rail cascade, hiding the
  Settings rows the Settings summary had just re-shown inside the hidden rail,
  so each of the eleven rows flipped twice per transition.

Each pin below has a positive control showing the counter sees a real change.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import replace

import pytest
from rich.text import Text
from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import Button, Static

# The UI fixture imports the app after selecting a per-test profile. Import it
# under the collection bootstrap first, before config freezes its source owner.
import tldw_chatbook.app  # noqa: F401
from Tests.private_profile import private_profile_test
from Tests.UI.consolidated_css import APP_STYLESHEETS, ConsolidatedCSSApp
from Tests.UI.test_console_run_tick_layout_requests import _LayoutRequests, _names
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Widgets.Console.console_workspace_context import (
    ConsoleWorkspaceContextTray,
)
from tldw_chatbook.Widgets.Console.console_workspace_details import (
    ConsoleWorkspaceDetailsTray,
)
from tldw_chatbook.Workspaces.conversation_attention import ConversationAttentionFact
from tldw_chatbook.Workspaces.conversation_browser_state import (
    ConsoleConversationBrowserRow,
    ConsoleConversationBrowserSection,
    ConsoleConversationBrowserState,
)
from tldw_chatbook.Workspaces.display_state import ConsoleWorkspaceContextState

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]


def _within(widget: Widget, owner: Widget) -> bool:
    return widget is owner or owner in widget.ancestors


def _painted(widget: Widget) -> str:
    """Text the widget shows, for an unchanged-copy check."""
    if isinstance(widget, Button):
        return widget.label.plain
    if isinstance(widget, Static):
        return str(widget.render())
    return ""


def _snapshot(owner: Widget) -> dict[int, tuple[bool, str]]:
    return {id(w): (bool(w.display), _painted(w)) for w in owner.query("*")}


# -- A scripted warm send through the real Console ---------------------------


def _console_host(tmp_path):
    from Tests.UI.test_console_fleet_wake_wiring import _attach_real_dbs
    from Tests.UI.test_console_regenerate_feedback import GatedGateway
    from Tests.UI.test_destination_shells import _build_test_app
    from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
        ConsoleHarness,
    )

    app = _build_test_app()
    # A file-backed DB with its conversation services: a second turn reads
    # the persisted conversation (world books, dictionaries) off the UI
    # thread, where a ``:memory:`` DB would be a different, empty database.
    _attach_real_dbs(app, tmp_path)
    app.app_config.setdefault("console", {})["agent_runtime"] = False
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "test-model"
    # ``GatedGateway`` reads ``self.release`` when it waits, so each turn
    # gets a fresh event (``_send_turn``).
    gateway = GatedGateway()
    app.console_provider_gateway_factory = lambda: gateway
    return gateway, ConsoleHarness(app)


async def _wait_for(pilot, predicate, what, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await pilot.pause(0.05)
    assert predicate(), what() if callable(what) else what


def _assistant_turns(store) -> int:
    return sum(
        message.role is ConsoleMessageRole.ASSISTANT
        and message.content.startswith("first-chunk")
        for message in store.messages_for_session(store.active_session_id)
    )


async def _send_turn(console, composer, pilot, gateway, text: str, windows=None):
    """Run one held send; count each transition when ``windows`` is given.

    ``windows`` is ``(requests, start_calls, end_calls)``: the start window runs
    from Send to the first streamed chunk, the end window from the stream's
    release to the end of the run.
    """
    store = console._ensure_console_chat_store()
    send = console.query_one("#console-send-message", Button)
    stop = composer.query_one("#console-stop-generation", Button)
    gateway.release = asyncio.Event()
    # A finished turn can re-sync the session draft after it ends; load until
    # Send has the draft and keeps it.
    for _ in range(40):
        composer.load_draft(text)
        await pilot.pause(0.2)
        if not send.disabled:
            await pilot.pause(0.3)
            if not send.disabled:
                break
    assert not send.disabled, "Send never took the draft"
    before = _assistant_turns(store)

    async def start() -> None:
        send.press()
        await _wait_for(
            pilot, lambda: _assistant_turns(store) > before, "no first chunk", 15.0
        )
        await _wait_for(pilot, lambda: stop.display, "Stop never displayed")
        await pilot.pause()
        await pilot.pause()

    async def end() -> None:
        gateway.release.set()
        await _wait_for(pilot, lambda: not stop.display, "Stop never hid")
        await _wait_for(
            pilot,
            lambda: console._console_transcript_sync_timer is None,
            "the run poll never stopped",
        )
        await pilot.pause(0.3)

    if windows is None:
        await start()
        await pilot.pause(0.5)
        await end()
        return
    requests, start_calls, end_calls, snapshots = windows
    rail = console.query_one("#console-right-rail")
    snapshots["start-before"] = _snapshot(rail)
    with requests.counting() as calls:
        await start()
        start_calls.extend(calls)
    snapshots["start-after"] = _snapshot(rail)
    snapshots["running-icon"] = console.query_one(
        "#console-conversation-actions-0", Button
    ).label.plain
    await pilot.pause(0.5)
    snapshots["end-before"] = _snapshot(rail)
    with requests.counting() as calls:
        await end()
        end_calls.extend(calls)
    snapshots["end-after"] = _snapshot(rail)
    snapshots["idle-icon"] = console.query_one(
        "#console-conversation-actions-0", Button
    ).label.plain


@private_profile_test
async def test_warm_send_transitions_lay_out_no_same_size_rail_widget(
    monkeypatch, request, tmp_path
) -> None:
    """A second send in a chat: its start and end leave unchanged widgets alone."""
    from Tests.UI.test_console_composer_run_controls import _mounted

    requests = _LayoutRequests(monkeypatch)
    gateway, host = _console_host(tmp_path)
    async with host.run_test(size=(160, 45)) as pilot:
        console, composer = await _mounted(host, pilot)
        await pilot.pause(0.5)
        # The first send creates and persists the conversation (its row's
        # title and scope change: a real recompose). The measured one is the
        # everyday case, a later send in that chat.
        await _send_turn(console, composer, pilot, gateway, "stream and hold 0")
        await pilot.pause(0.5)
        tray = console.query_one("#console-workspace-context")
        details = console.query_one("#console-workspace-details")
        rail = console.query_one("#console-right-rail")
        # The layout under test: Context rail open on Conversations, the
        # Inspector collapsed to its handle (the 160-column default).
        assert tray.display and tray.region.height > 0
        assert not rail.display
        start_calls: list[Widget] = []
        end_calls: list[Widget] = []
        snapshots: dict = {}
        await _send_turn(
            console,
            composer,
            pilot,
            gateway,
            "stream and hold 1",
            windows=(requests, start_calls, end_calls, snapshots),
        )
        # Positive controls: the run really changed what these widgets show.
        assert snapshots["running-icon"] != snapshots["idle-icon"]
        assert (
            details.state == console._workspace._build_console_workspace_context_state()
        )

        for window, calls, before, after in (
            ("start", start_calls, "start-before", "start-after"),
            ("end", end_calls, "end-before", "end-after"),
        ):
            assert calls, f"the {window} window counted nothing"
            in_tray = [w for w in calls if _within(w, tray)]
            assert _names(in_tray) == [], (
                f"send {window}: the Conversations tray laid out "
                f"{len(in_tray)} times: {_names(in_tray)}"
            )
            in_details = [w for w in calls if _within(w, details)]
            assert _names(in_details) == [], (
                f"send {window}: the Details tray laid out: {_names(in_details)}"
            )
            assert rail not in calls, (
                f"send {window}: the collapsed Inspector rail asked for a layout"
            )
            # Inside the collapsed rail, only a widget whose display or copy
            # changed across the window may ask (its height can change).
            unchanged = [
                w
                for w in calls
                if w is not rail
                and _within(w, rail)
                and snapshots[before].get(id(w)) == snapshots[after].get(id(w))
                and id(w) in snapshots[before]
            ]
            assert _names(unchanged) == [], (
                f"send {window}: unchanged widgets in the collapsed Inspector "
                f"asked for layouts: {_names(unchanged)}"
            )


# -- The Conversations tray: a running marker repaints in place ---------------


def _row(index: int, *, running: bool = False, approval: bool = False):
    facts = []
    if running:
        facts.append(ConversationAttentionFact(kind="running", label="Running"))
    if approval:
        facts.append(ConversationAttentionFact(kind="approval", label="Needs approval"))
    return ConsoleConversationBrowserRow(
        row_key=f"row-{index}",
        conversation_id=f"conversation-{index}",
        native_session_id=None,
        title=f"Research thread {index}",
        scope_type="conversation",
        workspace_id="workspace-default",
        workspace_label="Default",
        status="active" if index == 0 else "saved",
        updated_label="now",
        selected=index == 0,
        run_marker="⟳" if running else "",
        attention=tuple(facts),
    )


def _tray_state(**row0) -> ConsoleWorkspaceContextState:
    rows = (_row(0, **row0), _row(1), _row(2))
    browser = ConsoleConversationBrowserState(
        query="",
        sections=(
            ConsoleConversationBrowserSection(
                section_id="chats",
                label="Chats",
                collapsed=False,
                rows=rows,
                count=len(rows),
            ),
        ),
        selected_summary="Research thread 0 - Default",
    )
    return ConsoleWorkspaceContextState(
        heading="Conversations",
        workspace_label="Workspace: Default",
        workspace_name="Default",
        authority_label="Authority: local registry ready",
        sync_label="Sync: not configured",
        runtime_label="Local file tools: Private scratch",
        conversation_rows=(),
        conversation_empty_copy="No conversations yet.",
        change_workspace_enabled=False,
        change_workspace_recovery="",
        new_conversation_enabled=True,
        new_conversation_recovery="",
        recovery_copy="",
        conversation_browser=browser,
    )


class _TrayHost(ConsolidatedCSSApp):
    """The tray under the stylesheets the real app loads (its row sizes)."""

    CSS_PATH = [str(path) for path in APP_STYLESHEETS]

    def __init__(self, state: ConsoleWorkspaceContextState) -> None:
        super().__init__()
        self.app_config = {}
        self.tray = ConsoleWorkspaceContextTray(
            state, content="conversations", id="console-workspace-context"
        )

    def compose(self) -> ComposeResult:
        yield self.tray


def _witness(tray: Widget) -> list[tuple]:
    """Every descendant's geometry, copy, classes and tooltip, plus the paint."""
    nodes = [
        (
            type(w).__name__,
            w.id,
            tuple(w.region),
            tuple(w.virtual_size),
            bool(w.display),
            _painted(w),
            tuple(sorted(w.classes)),
            str(w.tooltip.plain if isinstance(w.tooltip, Text) else w.tooltip),
        )
        for w in tray.query("*")
    ]
    painted = [strip.text for strip in tray.render_lines(tray.region.reset_offset)]
    return [*nodes, tuple(painted)]


async def _settled_tray(host, pilot) -> ConsoleWorkspaceContextTray:
    tray = host.tray
    # The rail marks a tray it has pushed state into; the in-place paths
    # trust only a tray that has been synced at least once.
    tray._console_workspace_context_synced = True
    for _ in range(4):
        await pilot.pause()
    return tray


async def test_running_marker_patch_matches_a_recompose(monkeypatch) -> None:
    """Geometry A/B: the in-place patch lands what a recompose would build."""
    requests = _LayoutRequests(monkeypatch)
    idle, running = _tray_state(), _tray_state(running=True)

    patched_host = _TrayHost(idle)
    async with patched_host.run_test(size=(40, 30)) as pilot:
        tray = await _settled_tray(patched_host, pilot)
        before = _witness(tray)
        with requests.counting() as calls:
            tray.sync_state(running)
            await pilot.pause()
            await pilot.pause()
        assert _names(calls) == []
        assert tray.state == running
        patched = _witness(tray)
        # Back to idle: the patch is symmetric.
        tray.sync_state(idle)
        for _ in range(3):
            await pilot.pause()
        assert _witness(tray) == before

    built_host = _TrayHost(running)
    async with built_host.run_test(size=(40, 30)) as pilot:
        built = _witness(await _settled_tray(built_host, pilot))

    assert patched != before, "the running marker changed nothing visible"
    assert patched == built


async def test_marker_that_moves_rows_still_recomposes(monkeypatch) -> None:
    """Positive control: an approval adds an activity line, so rows move."""
    requests = _LayoutRequests(monkeypatch)
    host = _TrayHost(_tray_state())
    async with host.run_test(size=(40, 30)) as pilot:
        tray = await _settled_tray(host, pilot)
        row = tray.query_one("#console-workspace-conversation-0", Button)
        height = row.region.height
        with requests.counting() as calls:
            tray.sync_state(_tray_state(approval=True))
        assert tray in calls
        for _ in range(4):
            await pilot.pause()
        row = tray.query_one("#console-workspace-conversation-0", Button)
        assert row.region.height == height + 1


# -- The Details tray renders no conversation-browser field -------------------


async def test_details_tray_skips_a_delta_it_does_not_render(monkeypatch) -> None:
    requests = _LayoutRequests(monkeypatch)
    idle = _tray_state()
    details = ConsoleWorkspaceDetailsTray(idle, id="console-workspace-details")

    class _Host(ConsolidatedCSSApp):
        CSS_PATH = [str(path) for path in APP_STYLESHEETS]

        def compose(self) -> ComposeResult:
            yield details

    async with _Host().run_test(size=(40, 30)) as pilot:
        for _ in range(3):
            await pilot.pause()
        with requests.counting() as calls:
            details.sync_state(_tray_state(running=True))
        assert _names(calls) == []
        assert details.state == _tray_state(running=True)
        # Positive control: a field the tray renders still rebuilds it.
        with requests.counting() as calls:
            details.sync_state(
                replace(_tray_state(running=True), runtime_label="Runtime: 2 ready")
            )
        assert details in calls
        for _ in range(3):
            await pilot.pause()
        value = details.query_one("#console-workspace-runtime-value", Static)
        assert "2 ready" in str(value.render())


# -- Fixed-box rows in the collapsed Inspector repaint without a layout ------


def _authority_state(run_status: str):
    from tldw_chatbook.Chat.console_display_state import (
        ConsoleDisplayRow,
        ConsoleInspectorState,
    )

    return ConsoleInspectorState(
        rows=(
            ConsoleDisplayRow("Workspace", "Research Lab"),
            ConsoleDisplayRow("Selected conversation", "Research conversation 3"),
            ConsoleDisplayRow("Provider", run_status, status=run_status),
        ),
    )


class _FixedRowHost(ConsolidatedCSSApp):
    CSS_PATH = [str(path) for path in APP_STYLESHEETS]

    def __init__(self, widget: Widget) -> None:
        super().__init__()
        self.app_config = {}
        self.widget = widget

    def compose(self) -> ComposeResult:
        yield self.widget


async def test_authority_fact_rows_repaint_in_a_fixed_box(monkeypatch) -> None:
    """Geometry A/B for ``ConsoleSendAuthoritySummary``'s ``layout=False``."""
    from Tests.UI.test_timer_path_layout_cost import (
        assert_geometry_is_content_independent,
    )
    from tldw_chatbook.Widgets.Console.console_send_authority_summary import (
        ConsoleSendAuthoritySummary,
    )

    requests = _LayoutRequests(monkeypatch)
    summary = ConsoleSendAuthoritySummary(_authority_state("ready"))
    async with _FixedRowHost(summary).run_test(size=(40, 12)) as pilot:
        await pilot.pause()
        run_row = summary.query_one("#console-send-authority-run", Static)
        with requests.counting() as calls:
            summary.sync_state(_authority_state("blocked"))
        assert _names(calls) == []
        await pilot.pause()
        assert "Blocked" in str(run_row.render())
        await assert_geometry_is_content_independent(
            pilot,
            run_row,
            {
                "short": Text("Run: Ready"),
                "long": Text("Run: " + "Generating a very long status " * 4),
                "empty": Text(""),
            },
            list(summary.query(Static)),
        )


async def test_settings_rows_repaint_in_a_fixed_box(monkeypatch) -> None:
    """Geometry A/B for ``ConsoleSettingsSummary``'s ``layout=False``."""
    from Tests.UI.test_timer_path_layout_cost import (
        assert_geometry_is_content_independent,
    )
    from tldw_chatbook.Chat.console_session_settings import (
        ConsoleSettingsSummaryState,
    )
    from tldw_chatbook.Widgets.Console.console_settings_summary import (
        ConsoleSettingsSummary,
    )

    def settings(context: str) -> ConsoleSettingsSummaryState:
        return ConsoleSettingsSummaryState(
            model_row="Model: test-model",
            context_row=context,
            sampling_row="Sampling: default",
            identity_row="Identity: none",
            provider_row="Provider: llama.cpp",
        )

    requests = _LayoutRequests(monkeypatch)
    summary = ConsoleSettingsSummary(
        settings("Context: 1,200 tokens"), id="console-settings-summary"
    )
    async with _FixedRowHost(summary).run_test(size=(40, 20)) as pilot:
        for _ in range(3):
            await pilot.pause()
        context_row = summary.query_one("#console-settings-context-row", Static)
        with requests.counting() as calls:
            summary.sync_state(settings("Context: 1,450 tokens"))
        assert _names(calls) == []
        await pilot.pause()
        assert "1,450" in str(context_row.render())
        # Positive control: a row that appears still lays out (display).
        with requests.counting() as calls:
            summary.sync_state(
                replace(settings("Context: 1,450 tokens"), endpoint_row="Endpoint: x")
            )
        assert "Static#console-settings-endpoint-row" in _names(calls)
        await assert_geometry_is_content_independent(
            pilot,
            context_row,
            {
                "short": "Context: 9",
                "long": "Context: " + "1,234,567 tokens of a long window " * 4,
                "empty-ish": "Context: -",
            },
            list(summary.query(Static)),
        )


# -- The workbench header re-writes only the copy that changed ---------------


async def test_header_title_keeps_its_layout_when_status_changes(monkeypatch) -> None:
    from tldw_chatbook.UI.Workbench.workbench_widgets import (
        DestinationHeader,
        WorkbenchHeaderState,
    )

    requests = _LayoutRequests(monkeypatch)
    state = WorkbenchHeaderState(title="Console", subtitle="Ready", status="ready")
    header = DestinationHeader(state, id="console-workbench-header")
    async with _FixedRowHost(header).run_test(size=(120, 6)) as pilot:
        await pilot.pause()
        with requests.counting() as calls:
            header.sync_state(
                replace(state, subtitle="Generating a reply", status="running")
            )
        names = _names(calls)
        assert "Static#workbench-header-title" not in names
        # Positive control: the copy that changed still lays out (auto width).
        assert "Static#workbench-header-subtitle" in names
        assert "Static#workbench-header-status" in names


# -- plan_in_place_delta: what it patches, and what it leaves to a rebuild ---


def _row_tree(
    *,
    icon: str = "*",
    tip: str = "idle",
    cls: str = "",
    width: int | None = 4,
    extra=False,
):
    from textual.containers import Horizontal

    def build() -> ComposeResult:
        with Horizontal(id="row"):
            button = Button(icon, id="row-action", classes=cls or None, compact=True)
            if width is not None:
                button.styles.width = width
            button.styles.height = 1
            button.tooltip = tip
            button.row_key = "row-1"
            yield button
            if extra:
                yield Static("extra", id="row-extra")

    return build


class _DeltaHost(ConsolidatedCSSApp):
    def __init__(self, build) -> None:
        super().__init__()
        self.app_config = {}
        self._build = build

    def compose(self) -> ComposeResult:
        from textual.containers import Vertical

        with Vertical(id="owner"):
            yield from self._build()


async def _plan(mounted_build, before_build, after_build, mutate=None):
    from textual.compose import compose

    from tldw_chatbook.Widgets.Console.compose_delta import plan_in_place_delta

    host = _DeltaHost(mounted_build)
    async with host.run_test(size=(40, 6)) as pilot:
        await pilot.pause()
        owner = host.query_one("#owner")
        if mutate is not None:
            mutate(host)
        before = compose(owner, before_build())
        after = compose(owner, after_build())
        writes = plan_in_place_delta(list(owner.children), before, after)
        if writes is not None:
            for write in writes:
                write()
            await pilot.pause()
        button = host.query_one("#row-action", Button)
        return writes, button


async def test_delta_patches_copy_classes_and_tooltip_in_place() -> None:
    writes, button = await _plan(
        _row_tree(),
        _row_tree(),
        _row_tree(icon="R", tip="running", cls="conversation-attention-running"),
    )
    assert writes is not None
    assert button.label.plain == "R"
    assert str(button.tooltip) == "running"
    assert button.has_class("conversation-attention-running")


async def test_delta_rebuilds_copy_in_a_content_sized_box() -> None:
    writes, _ = await _plan(
        _row_tree(width=None),
        _row_tree(width=None),
        _row_tree(icon="Running", width=None),
    )
    assert writes is None


def _drift_label(host) -> None:
    host.query_one("#row-action", Button).label = "?"


def _drift_class(host) -> None:
    host.query_one("#row-action", Button).add_class("conversation-attention-running")


def _drift_tooltip(host) -> None:
    host.query_one("#row-action", Button).tooltip = "edited elsewhere"


@pytest.mark.parametrize("drift", [_drift_label, _drift_class, _drift_tooltip])
async def test_delta_rebuilds_a_mounted_tree_that_drifted(drift) -> None:
    """A mounted tree that no longer shows ``before`` is rebuilt, not patched."""
    writes, _ = await _plan(
        _row_tree(),
        _row_tree(),
        _row_tree(icon="R", tip="running", cls="conversation-attention-running"),
        mutate=drift,
    )
    assert writes is None


async def test_delta_rebuilds_a_new_shape_or_style() -> None:
    writes, _ = await _plan(_row_tree(), _row_tree(), _row_tree(extra=True))
    assert writes is None
    writes, _ = await _plan(_row_tree(), _row_tree(), _row_tree(width=5))
    assert writes is None
