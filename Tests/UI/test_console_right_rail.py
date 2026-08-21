"""Characterisation test for the Console right (Inspector) rail, written
BEFORE it becomes its own region widget (wave-1 console decomposition,
task 4, spec rule 6).

Drives the real ``ChatScreen`` through the real Console harness -- the same
idiom ``test_console_left_rail.py`` (task 3's precedent) and
``test_console_shell_regions.py`` use -- and performs real ``pilot.click``s
on the rail's real collapse/expand controls, asserting the outcome survives
a fresh re-query (not just a transient widget attribute) in both
directions.

Unlike the left rail, the Inspector rail has no per-section toggle headers:
the whole rail opens/closes as one unit via ``#console-inspector-rail-open``
(on the collapsed handle, a ``ChatScreen`` sibling that stays outside this
extraction -- see the task-4 report) and ``#console-inspector-rail-collapse``
(inside the rail's own header, the control this extraction actually moves).
This file exercises both.

This file must pass against unmodified code before the right rail is
extracted into ``UI/Console_Modules/right_rail.py``, and must stay green
and byte-identical afterwards (task-4 brief, global constraint 3).
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import replace
import importlib
import threading

import pytest
from textual.containers import Horizontal
from textual.widgets import Button, Static

from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_display_state import (
    ConsoleDisplayRow,
    ConsoleInspectorAction,
    ConsoleInspectorState,
)
from tldw_chatbook.Chat.console_project_instructions import (
    ProjectInstructionControlState,
)
from tldw_chatbook.Widgets.Console.console_conversation_inspector import (
    ConsoleConversationInspector,
)
from tldw_chatbook.Widgets.Console.console_context_modal import ConsoleContextModal
from tldw_chatbook.Widgets.Console.console_run_inspector import ConsoleRunInspector

from Tests.UI.test_console_native_chat_flow import _configure_native_ready_console
from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)


@asynccontextmanager
async def make_console_pilot(*, size=(160, 45)):
    """Mount a fresh, send-ready Console (ChatScreen) via the production harness.

    Mirrors ``test_console_left_rail.py``'s ``make_console_pilot``: rail-click
    tests need the blocking first-run ``ConsoleSetupModal`` dismissed, which
    requires a ready provider, not just a mounted composer. At this size the
    Inspector rail's own responsive auto-open rule
    (``ChatScreen._should_open_standard_width_inspector``, 118-128 available
    columns) does not fire, so the rail starts from its plain persisted
    default (closed) -- the deliberately chosen, unambiguous starting point
    for the toggle tests below.
    """
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)
    async with host.run_test(size=size) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        await pilot.pause(0.2)
        yield pilot


def _right_rail_open(pilot) -> bool:
    right_rail = pilot.app.screen.query_one("#console-right-rail")
    return bool(right_rail.display) and right_rail.styles.display != "none"


def _handle_visible(pilot) -> bool:
    handle = pilot.app.screen.query_one("#console-inspector-rail-handle")
    return bool(handle.display) and handle.styles.display != "none"


_EXPECTED_BOUNDARY_ANCHORS = (
    ("console-project-instruction-status", "Project Instructions"),
    ("console-staged-context-tray", "Sources"),
    ("console-retrieval-scope-row", "Scope"),
    ("console-changed-files-section", "Changed Files"),
    ("console-inspector-run-status-summary", "Run status"),
    ("console-inspector-run-heading", "Run"),
    ("console-inspector-source-readiness-heading", "Source Readiness"),
    ("console-inspector-tools-heading", "Tools"),
    ("console-inspector-approvals-heading", "Approvals"),
    ("console-inspector-artifacts-heading", "Artifacts"),
    (
        "console-inspector-selected-conversation-heading",
        "Selected Conversation",
    ),
    ("console-inspector-session-defaults-heading", "Session Defaults"),
    ("console-inspector-selected-message-heading", "Selected Message"),
    ("console-inspector-changes-heading", "Changes"),
    ("console-inspector-dictionaries-heading", "Chat Dictionaries"),
    ("console-inspector-worldbooks-heading", "World Books"),
    ("console-settings-summary", "Session Settings"),
)
_LIVE_WORK_IDS = {
    "console-pending-launch-card",
    "console-live-work-source-readiness",
}


def _mounted_boundary_ids(rail) -> tuple[str, ...]:
    """Read semantic boundaries from the mounted production hierarchy."""
    body = rail.query_one("#console-inspector-rail-body")
    direct_children = tuple(child.id for child in body.children)
    assert direct_children[:5] == (
        "console-project-instruction-status",
        "console-staged-context-tray",
        "console-retrieval-scope-row",
        "console-changed-files-section",
        "console-run-inspector",
    )
    assert len(direct_children) == 6
    assert direct_children[-1] in _LIVE_WORK_IDS

    run_wrapper = rail.query_one("#console-run-inspector")
    run_wrapper_children = tuple(child.id for child in run_wrapper.children)
    assert run_wrapper_children == (
        "console-run-inspector-state",
        "console-settings-summary",
    )
    inspector = run_wrapper.query_one(
        "#console-run-inspector-state", ConsoleRunInspector
    )
    inspector_boundaries = tuple(
        child.id
        for child in inspector.children
        if child.id == "console-inspector-run-status-summary"
        or child.has_class("console-inspector-group-heading")
    )
    return (
        *direct_children[:4],
        *inspector_boundaries,
        run_wrapper_children[-1],
        direct_children[-1],
    )


def test_inspector_boundary_inventory_has_approved_order_and_specialized_owners():
    assert tuple(owner for _widget_id, owner in _EXPECTED_BOUNDARY_ANCHORS) + (
        "Live Work",
    ) == (
        "Project Instructions",
        "Sources",
        "Scope",
        "Changed Files",
        "Run status",
        "Run",
        "Source Readiness",
        "Tools",
        "Approvals",
        "Artifacts",
        "Selected Conversation",
        "Session Defaults",
        "Selected Message",
        "Changes",
        "Chat Dictionaries",
        "World Books",
        "Session Settings",
        "Live Work",
    )
    assert dict(_EXPECTED_BOUNDARY_ANCHORS[:5]) | {
        "console-settings-summary": "Session Settings",
    } | {live_id: "Live Work" for live_id in _LIVE_WORK_IDS} == {
        "console-project-instruction-status": "Project Instructions",
        "console-staged-context-tray": "Sources",
        "console-retrieval-scope-row": "Scope",
        "console-changed-files-section": "Changed Files",
        "console-inspector-run-status-summary": "Run status",
        "console-settings-summary": "Session Settings",
        "console-pending-launch-card": "Live Work",
        "console-live-work-source-readiness": "Live Work",
    }


@pytest.mark.asyncio
async def test_mounted_inspector_semantic_census_matches_actual_right_rail_order():
    exhaustive_rows = tuple(
        ConsoleDisplayRow(label, "value")
        for label in (
            "Run recipe",
            "Live work",
            "Setup",
            "Send blocked",
            "Recovery action",
            "Blocked impact",
            "Next action",
            "Provider",
            "Sources",
            "RAG/source",
            "Evidence",
            "Authority",
            "Tools",
            "MCP",
            "Approvals",
            "Artifacts",
            "Selected conversation",
            "Conversation source",
            "Workspace",
            "Resume state",
            "Prefill (next send only)",
            "Prefill (pinned)",
            "Session provider",
            "Session model",
            "Session endpoint",
            "Session sampling",
            "Session persona",
            "Selected message",
            "Message actions",
            "Keyboard",
            "Variants",
            "Excerpt",
            "Delete confirmation",
        )
    )
    exhaustive_state = ConsoleInspectorState(
        rows=exhaustive_rows,
        actions=(
            ConsoleInspectorAction(
                "console-inspector-review-approval", "Review approval", True
            ),
            ConsoleInspectorAction(
                "console-inspector-review-changes", "Review changes", True
            ),
            ConsoleInspectorAction(
                "console-inspector-save-chatbook", "Save as Chatbook", True
            ),
        ),
        dictionary_rows=(ConsoleDisplayRow("Dictionary", "attached"),),
        dictionary_actions=(
            ConsoleInspectorAction(
                "console-inspector-dictionaries-attach", "Attach dictionary", True
            ),
        ),
        world_book_rows=(ConsoleDisplayRow("World Book", "attached"),),
        world_book_actions=(
            ConsoleInspectorAction(
                "console-inspector-worldbooks-attach", "Attach World Book", True
            ),
        ),
    )

    async with make_console_pilot() as pilot:
        await pilot.click("#console-inspector-rail-open")
        await pilot.pause()
        rail = pilot.app.screen.query_one("#console-right-rail")
        rail._inspector_state = exhaustive_state
        await rail.recompose()
        await pilot.pause()

        mounted_ids = _mounted_boundary_ids(rail)
        expected_ids = tuple(item[0] for item in _EXPECTED_BOUNDARY_ANCHORS)
        assert mounted_ids[:-1] == expected_ids
        assert mounted_ids[-1] in _LIVE_WORK_IDS


@pytest.mark.asyncio
async def test_new_specialized_sibling_fails_mounted_production_census():
    async with make_console_pilot() as pilot:
        await pilot.click("#console-inspector-rail-open")
        await pilot.pause()
        rail = pilot.app.screen.query_one("#console-right-rail")
        for parent_selector, sibling_id in (
            ("#console-inspector-rail-body", "console-new-specialized-sibling"),
            ("#console-run-inspector", "console-new-run-wrapper-sibling"),
        ):
            sibling = Static("new", id=sibling_id)
            await rail.query_one(parent_selector).mount(sibling)
            with pytest.raises(AssertionError):
                _mounted_boundary_ids(rail)
            await sibling.remove()


@pytest.mark.asyncio
async def test_real_inspector_producer_variants_are_strictly_owned(monkeypatch):
    ownership = importlib.import_module(
        "tldw_chatbook.Widgets.Console.console_inspector_ownership"
    )

    async with make_console_pilot() as pilot:
        screen = pilot.app.screen
        store = screen._ensure_console_chat_store()
        message = store.append_message(
            store.active_session_id,
            role=ConsoleMessageRole.USER,
            content="selected producer message",
        )
        await screen._sync_native_console_chat_ui()
        transcript = screen.query_one("#console-native-transcript")
        transcript.select_message(message.id)
        screen._retrieval._active_dictionaries_summary = {
            "dictionaries": [{"name": "Producer dictionary", "source": "conversation"}]
        }
        screen._retrieval._active_world_books_summary = {
            "world_books": [
                {"name": "Producer world book", "entry_count": 2, "enabled": True}
            ]
        }
        monkeypatch.setattr(
            screen, "_console_provider_blocker_copy", lambda: "Provider setup needed"
        )
        monkeypatch.setattr(
            screen,
            "_console_provider_recovery_action",
            lambda: ("Open Settings", "settings", "Open provider settings"),
        )

        state = screen._build_console_inspector_state(None)
        classified = ownership.classify_inspector_content(
            state, ownership.InspectorOwnershipPolicy.STRICT
        )

        assert not classified.incomplete
        assert {row.label for row in state.rows} >= {
            "Setup",
            "Blocked impact",
            "Next action",
            "Selected conversation",
            "Conversation source",
            "Selected message",
            "Message actions",
            "Keyboard",
        }
        assert state.dictionary_rows
        assert state.dictionary_actions
        assert state.world_book_rows
        assert state.world_book_actions


@pytest.mark.asyncio
async def test_rail_recompose_retains_unknown_fingerprint_deduper(monkeypatch):
    ownership = importlib.import_module(
        "tldw_chatbook.Widgets.Console.console_inspector_ownership"
    )
    inspector_module = importlib.import_module(
        "tldw_chatbook.Widgets.Console.console_run_inspector"
    )
    diagnostics = []
    monkeypatch.setattr(
        inspector_module.logger,
        "warning",
        lambda message, fingerprint: diagnostics.append((message, fingerprint)),
    )

    async with make_console_pilot() as pilot:
        await pilot.click("#console-inspector-rail-open")
        await pilot.pause()
        rail = pilot.app.screen.query_one("#console-right-rail")
        initial = rail._inspector_state
        rail._inspector_state = replace(
            initial,
            rows=initial.rows
            + (ConsoleDisplayRow("Unknown retained", "PRIVATE VALUE"),),
        )
        await rail.recompose()
        await pilot.pause()
        assert len(diagnostics) == 1
        assert (
            rail.query_one(
                "#console-run-inspector-state", ConsoleRunInspector
            ).ownership_policy
            is ownership.InspectorOwnershipPolicy.RESILIENT
        )

        await rail.recompose()
        await pilot.pause()
        assert len(diagnostics) == 1

        rail._inspector_state = replace(
            initial,
            rows=initial.rows
            + (ConsoleDisplayRow("Another unknown", "OTHER PRIVATE VALUE"),),
        )
        await rail.recompose()
        await pilot.pause()
        assert len(diagnostics) == 2
        assert diagnostics[-1][1] == ("row:Another unknown",)
        assert "PRIVATE VALUE" not in repr(diagnostics)


def test_inspector_composition_boundary_resolves_strict_opt_in(monkeypatch):
    right_rail = importlib.import_module("tldw_chatbook.UI.Console_Modules.right_rail")
    ownership = importlib.import_module(
        "tldw_chatbook.Widgets.Console.console_inspector_ownership"
    )

    monkeypatch.delenv("TLDW_CONSOLE_STRICT_INSPECTOR_OWNERSHIP", raising=False)
    assert (
        right_rail._resolve_inspector_ownership_policy()
        is ownership.InspectorOwnershipPolicy.RESILIENT
    )
    monkeypatch.setenv("TLDW_CONSOLE_STRICT_INSPECTOR_OWNERSHIP", "1")
    assert (
        right_rail._resolve_inspector_ownership_policy()
        is ownership.InspectorOwnershipPolicy.STRICT
    )
    monkeypatch.setenv("TLDW_CONSOLE_STRICT_INSPECTOR_OWNERSHIP", "true")
    assert (
        right_rail._resolve_inspector_ownership_policy()
        is ownership.InspectorOwnershipPolicy.RESILIENT
    )


@pytest.mark.asyncio
async def test_inspector_rail_starts_closed_by_default():
    """Fresh harness, no stored rail preferences, 160-column terminal.

    Pins ``CONSOLE_RAIL_RIGHT_DEFAULT_OPEN = False``
    (``Chat/console_rail_state.py``) as observed through the real DOM at a
    terminal width outside the 118-128 responsive auto-open band, not just
    read off the constant -- this is the starting point the toggle tests
    below build on.
    """
    async with make_console_pilot() as pilot:
        assert _right_rail_open(pilot) is False
        assert _handle_visible(pilot) is True


@pytest.mark.asyncio
async def test_clicking_open_then_collapse_toggles_visibility_and_persists():
    """A real click on the handle's Open button opens the rail; a real click
    on the rail's own Collapse button closes it again.

    "Persists" here means what it means in the left-rail characterisation
    test: re-querying the rail/handle display state after each click
    reflects the new state (this is what
    ``ChatScreen.on_console_inspector_rail_collapse``/``_open`` ->
    ``_set_console_rail_preference`` -> ``_sync_console_rail_visibility``
    does today), not merely that the click handler ran without raising.
    """
    async with make_console_pilot() as pilot:
        await _wait_for_selector(
            pilot.app.screen, pilot, "#console-inspector-rail-open"
        )
        open_button = pilot.app.screen.query_one("#console-inspector-rail-open", Button)
        assert str(open_button.label) == "<-Inspect"
        far_end = (
            open_button.region.width - 1,
            open_button.region.height // 2,
        )
        assert await pilot.click(open_button, offset=far_end)
        await pilot.pause(0.2)
        assert _right_rail_open(pilot) is True
        assert _handle_visible(pilot) is False

        # The content this extraction moves is actually mounted once open --
        # pins that every id inside the moved block survived the click path,
        # not just the rail's own root.
        assert pilot.app.screen.query_one("#console-inspector-rail-body")
        project_row = pilot.app.screen.query_one("#console-project-instruction-status")
        staged = pilot.app.screen.query_one("#console-staged-context-tray")
        assert project_row.region.y < staged.region.y
        assert pilot.app.screen.query_one("#console-staged-context-tray")
        assert pilot.app.screen.query_one("#console-run-inspector")
        assert pilot.app.screen.query_one("#console-run-inspector-state")
        assert pilot.app.screen.query_one("#console-settings-summary")
        controller = pilot.app.screen._ensure_console_chat_controller()
        assert controller._confirm_project_instruction_dispatch.__self__ is (
            pilot.app.screen._session
        )
        assert controller._select_project_instruction_binding.__self__ is (
            pilot.app.screen._session
        )

        await pilot.click("#console-project-instruction-status-button")
        await pilot.pause()
        assert isinstance(pilot.app.screen, ConsoleConversationInspector)
        await pilot.press("escape")
        await pilot.pause()

        await pilot.click("#console-inspector-rail-collapse")
        await pilot.pause(0.2)
        assert _right_rail_open(pilot) is False
        assert _handle_visible(pilot) is True


@pytest.mark.asyncio
async def test_inspector_header_is_one_full_width_collapse_button() -> None:
    async with make_console_pilot() as pilot:
        await _wait_for_selector(
            pilot.app.screen, pilot, "#console-inspector-rail-open"
        )
        open_button = pilot.app.screen.query_one("#console-inspector-rail-open", Button)
        assert await pilot.click(open_button)
        await pilot.pause(0.2)

        screen = pilot.app.screen
        button = screen.query_one("#console-inspector-rail-collapse", Button)
        header = button.parent

        assert _right_rail_open(pilot) is True
        assert _handle_visible(pilot) is False
        assert isinstance(header, Horizontal)
        assert list(header.children) == [button]
        assert not screen.query("#console-inspector-rail-title")
        assert str(button.label) == "Inspect|--------->"
        assert button.tooltip == "Collapse Inspector rail"
        assert header.content_region.contains_region(button.region)
        assert button.region.width == header.content_region.width
        assert header.region.height == 1
        assert button.region.height == 1
        assert button.styles.text_align == "left"
        assert button.styles.content_align_horizontal == "left"


@pytest.mark.asyncio
async def test_clicking_inspector_header_title_start_collapses_the_rail() -> None:
    async with make_console_pilot() as pilot:
        await _wait_for_selector(
            pilot.app.screen, pilot, "#console-inspector-rail-open"
        )
        assert await pilot.click("#console-inspector-rail-open")
        await pilot.pause(0.2)

        button = pilot.app.screen.query_one("#console-inspector-rail-collapse", Button)
        assert str(button.label) == "Inspect|--------->"
        title_start = (1, 0)
        assert await pilot.click(button, offset=title_start)
        await pilot.pause(0.2)

        assert _right_rail_open(pilot) is False
        assert _handle_visible(pilot) is True
        assert pilot.app.focused is None


@pytest.mark.asyncio
async def test_clicking_the_collapse_button_clears_focus():
    """Pin today's focus behaviour when the Inspector rail's own Collapse
    button is pressed.

    Unlike the left rail's section-toggle buttons (which stay visible after
    their own click, so Textual's default click-to-focus behaviour lands
    focus ON the button), the Inspector rail's Collapse button click makes
    ITSELF display:none as a direct effect of the click
    (``_sync_console_rail_visibility`` hides ``#console-right-rail``, which
    contains the button that was just clicked) -- observed here, Textual
    clears focus to ``None`` rather than leaving it on a now-hidden widget.
    Pinning the OBSERVED outcome (rather than asserting what "should"
    happen) is the point of a characterisation test.
    """
    async with make_console_pilot() as pilot:
        await _wait_for_selector(
            pilot.app.screen, pilot, "#console-inspector-rail-open"
        )
        await pilot.click("#console-inspector-rail-open")
        await pilot.pause(0.2)

        await pilot.click("#console-inspector-rail-collapse")
        await pilot.pause(0.2)

        assert pilot.app.focused is None


@pytest.mark.asyncio
async def test_context_modal_refresh_factory_keeps_opening_session_after_switch():
    async with make_console_pilot() as pilot:
        console = pilot.app.screen
        store = console._ensure_console_chat_store()
        captured = store.ensure_session(title="Captured")
        store.append_message(
            captured.id,
            role=ConsoleMessageRole.USER,
            content="captured transcript",
        )
        await pilot.click("#console-inspector-rail-open")
        await pilot.pause()
        await pilot.click("#console-project-instruction-status-button")
        await pilot.pause()
        modal = pilot.app.screen
        assert isinstance(modal, ConsoleConversationInspector)
        assert modal._project_instruction_session_id == captured.id

        active = store.create_session(title="Active")
        store.append_message(
            active.id,
            role=ConsoleMessageRole.USER,
            content="wrong active transcript",
        )
        snapshot = await modal._snapshot_factory()

        assert [message.content for message in snapshot.current_messages] == [
            "captured transcript"
        ]

        assert modal._project_instruction_recovery is not None
        main_thread_id = threading.get_ident()
        setter_threads = []
        original_setter = store.set_session_project_instruction_state

        def record_setter(session_id, state):
            setter_threads.append(threading.get_ident())
            return original_setter(session_id, state)

        store.set_session_project_instruction_state = record_setter
        state = await modal._project_instruction_recovery(captured.id, "disable")
        assert state.status == "Off"
        assert setter_threads == [main_thread_id]
        captured_after = next(
            item for item in store.sessions() if item.id == captured.id
        )
        active_after = next(item for item in store.sessions() if item.id == active.id)
        assert captured_after.project_instruction_state == (
            ProjectInstructionControlState.legacy_disabled()
        )
        assert active_after.project_instruction_state != (
            ProjectInstructionControlState.legacy_disabled()
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(100, 30), (140, 40)])
async def test_project_status_remains_visible_in_real_thirty_column_rail(size):
    async with make_console_pilot(size=size) as pilot:
        if not _right_rail_open(pilot):
            await pilot.click("#console-inspector-rail-open")
            await pilot.pause()
        rail = pilot.app.screen.query_one("#console-right-rail")
        rail.styles.width = 30
        rail.styles.min_width = 30
        rail.styles.max_width = 30
        await pilot.pause()
        button = pilot.app.screen.query_one(
            "#console-project-instruction-status-button", Button
        )
        assert button.region.width <= 30
        assert str(button.label).endswith(" · Project")
