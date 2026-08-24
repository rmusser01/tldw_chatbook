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
from types import SimpleNamespace

import pytest
from textual.containers import Horizontal
from textual.widgets import Button, Static

from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_display_state import (
    ConversationFileEntry,
    ConsoleDisplayRow,
    ConsoleInspectorAction,
    ConsoleInspectorState,
    ConsoleRetrievalScopeState,
)
from tldw_chatbook.Widgets.Console.console_bounded_section import (
    ConsoleBoundedSection,
)
from tldw_chatbook.Widgets.Console.console_changed_files_section import (
    ConsoleChangedFilesState,
    ConsoleChangedFilesSection,
)
from tldw_chatbook.Chat.console_project_instructions import (
    ProjectInstructionControlState,
)
from tldw_chatbook.Widgets.Console.console_conversation_inspector import (
    ConsoleConversationInspector,
)
from tldw_chatbook.Chat.console_live_work import ConsoleLiveWorkLaunch
from tldw_chatbook.Widgets.Console.console_run_inspector import ConsoleRunInspector
from tldw_chatbook.Widgets.Console.console_send_authority_summary import (
    ConsoleSendAuthoritySummary,
    project_console_send_authority,
)
from tldw_chatbook.UI.Workbench.help import WorkbenchHelpPanel

from Tests.UI.test_console_native_chat_flow import _configure_native_ready_console
from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)


@asynccontextmanager
async def make_console_pilot(*, size=(160, 45), css: str | None = None):
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
    if css is None:
        host = ConsoleHarness(app)
    else:

        class StyledConsoleHarness(ConsoleHarness):
            CSS = css

        host = StyledConsoleHarness(app)
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


async def _wait_for_right_rail_condition(
    pilot,
    predicate,
    *,
    description: str,
    attempts: int = 30,
) -> None:
    """Bound asynchronous rail reconciliation by observable state."""

    for _ in range(attempts):
        if predicate():
            return
        await pilot.pause()
    pytest.fail(f"Timed out waiting for {description}")


_EXPECTED_BOUNDARY_ANCHORS = (
    ("console-project-instruction-status", "Project Instructions"),
    ("console-send-authority-summary", "Next send authority"),
    ("console-staged-context-tray", "Sources"),
    ("console-retrieval-scope-row", "Scope"),
    ("console-changed-files-section", "Changed Files"),
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


def _expected_run_inspector_child_ids(state) -> tuple[str, ...]:
    """Project every direct child ID from canonical STRICT ownership data."""
    ownership = importlib.import_module(
        "tldw_chatbook.Widgets.Console.console_inspector_ownership"
    )
    projected = ownership.classify_inspector_content(
        state, ownership.InspectorOwnershipPolicy.STRICT
    )
    child_ids = []

    for owner, heading_id, _labels in ownership.ROW_GROUPS:
        rows = projected.rows_for(owner)
        actions = projected.actions_for(owner)
        if not rows and not actions:
            continue
        if rows or any(action.enabled for action in actions):
            child_ids.append(heading_id)
            child_ids.append(
                f"console-bounded-section-{owner.lower().replace(' ', '-')}"
            )

    for heading_id, rows, actions in (
        (
            "console-inspector-dictionaries-heading",
            projected.dictionary_rows,
            projected.dictionary_actions,
        ),
        (
            "console-inspector-worldbooks-heading",
            projected.world_book_rows,
            projected.world_book_actions,
        ),
    ):
        if not rows and not actions:
            continue
        child_ids.append(heading_id)
        child_ids.append(
            "console-bounded-section-"
            + ("chat-dictionaries" if "dictionaries" in heading_id else "world-books")
        )

    return tuple(child_ids)


def _mounted_boundary_ids(rail) -> tuple[str, ...]:
    """Read semantic boundaries from the mounted production hierarchy."""
    body = rail.query_one("#console-inspector-rail-body")
    direct_children = tuple(child.id for child in body.children)
    assert direct_children[:4] == (
        "console-staged-context-tray",
        "console-retrieval-scope-row",
        "console-changed-files-section",
        "console-run-inspector",
    )
    assert len(direct_children) == 5
    assert direct_children[-1] == "console-live-work-section"

    run_wrapper = rail.query_one("#console-run-inspector")
    run_wrapper_children = tuple(child.id for child in run_wrapper.children)
    assert run_wrapper_children == (
        "console-run-inspector-state",
        "console-settings-summary",
    )
    inspector = run_wrapper.query_one(
        "#console-run-inspector-state", ConsoleRunInspector
    )
    expected_inspector_children = _expected_run_inspector_child_ids(inspector.state)
    actual_inspector_children = tuple(child.id for child in inspector.children)
    assert actual_inspector_children == expected_inspector_children
    ownership = importlib.import_module(
        "tldw_chatbook.Widgets.Console.console_inspector_ownership"
    )
    boundary_ids = {
        "console-inspector-dictionaries-heading",
        "console-inspector-worldbooks-heading",
        *(heading_id for _owner, heading_id, _labels in ownership.ROW_GROUPS),
    }
    inspector_boundaries = tuple(
        child_id for child_id in expected_inspector_children if child_id in boundary_ids
    )
    return (
        "console-project-instruction-status",
        "console-send-authority-summary",
        *direct_children[:3],
        *inspector_boundaries,
        run_wrapper_children[-1],
        next(card_id for card_id in _LIVE_WORK_IDS if list(rail.query(f"#{card_id}"))),
    )


def test_inspector_boundary_inventory_has_approved_order_and_specialized_owners():
    assert tuple(owner for _widget_id, owner in _EXPECTED_BOUNDARY_ANCHORS) + (
        "Live Work",
    ) == (
        "Project Instructions",
        "Next send authority",
        "Sources",
        "Scope",
        "Changed Files",
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
        "console-send-authority-summary": "Next send authority",
        "console-staged-context-tray": "Sources",
        "console-retrieval-scope-row": "Scope",
        "console-changed-files-section": "Changed Files",
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
        rail._changed_files_state = ConsoleChangedFilesState(
            entries=(
                ConversationFileEntry(
                    root="/tmp/project",
                    path="changed.py",
                    label="changed.py",
                    status="M",
                    adds=1,
                    dels=0,
                    run_id="run-1",
                    snapshot_id=1,
                    note_count=0,
                ),
            )
        )
        await rail.recompose()
        await pilot.pause()

        mounted_ids = _mounted_boundary_ids(rail)
        expected_ids = tuple(item[0] for item in _EXPECTED_BOUNDARY_ANCHORS)
        assert mounted_ids[:-1] == expected_ids
        assert mounted_ids[-1] in _LIVE_WORK_IDS

        compact_ids = {
            "console-project-instruction-status",
            "console-retrieval-scope-row",
        }
        for compact_id in compact_ids:
            compact = rail.query_one(f"#{compact_id}")
            assert not any(
                isinstance(ancestor, ConsoleBoundedSection)
                for ancestor in compact.ancestors
            )

        specialized = (
            ("#console-staged-context-tray", "sources"),
            ("#console-changed-files-section", "changed-files"),
            ("#console-settings-summary", "session-settings"),
            ("#console-live-work-section", "live-work"),
        )
        for root_selector, section_id in specialized:
            root = rail.query_one(root_selector)
            bodies = list(root.query(ConsoleBoundedSection))
            assert [body.section_id for body in bodies] == [section_id]

        changed = rail.query_one(
            "#console-changed-files-section", ConsoleChangedFilesSection
        )
        assert changed.MAX_VISIBLE_ROWS == 12


@pytest.mark.asyncio
async def test_new_specialized_sibling_fails_mounted_production_census():
    async with make_console_pilot() as pilot:
        await pilot.click("#console-inspector-rail-open")
        await pilot.pause()
        rail = pilot.app.screen.query_one("#console-right-rail")
        for parent_selector, sibling_id in (
            ("#console-inspector-rail-body", "console-new-specialized-sibling"),
            ("#console-run-inspector", "console-new-run-wrapper-sibling"),
            (
                "#console-run-inspector-state",
                "console-new-inspector-content-sibling",
            ),
        ):
            sibling = Static("new", id=sibling_id)
            await rail.query_one(parent_selector).mount(sibling)
            with pytest.raises(AssertionError):
                _mounted_boundary_ids(rail)
            await sibling.remove()


@pytest.mark.asyncio
async def test_sources_use_exact_twenty_line_content_ceiling():
    async with make_console_pilot(size=(235, 52)) as pilot:
        await pilot.click("#console-inspector-rail-open")
        await pilot.pause()
        rail = pilot.app.screen.query_one("#console-right-rail")
        section = rail.query_one(
            "#console-bounded-section-sources", ConsoleBoundedSection
        )
        await section.viewport.remove_children()
        content = Static("\n".join(f"row {index}" for index in range(20)))
        await section.viewport.mount(content)
        section.request_reconcile()
        for _ in range(5):
            await pilot.pause()

        assert section.viewport.content_region.height == 20
        assert section.hint.display is False

        content.update("\n".join(f"row {index}" for index in range(21)))
        content.refresh(layout=True)
        section.request_reconcile()
        for _ in range(5):
            await pilot.pause()

        assert section.viewport.content_region.height == 20
        assert section.hint.display is True
        assert section.hint.region.height == 1


@pytest.mark.parametrize(
    (
        "direction",
        "terminal_width",
        "payload_rows",
        "acp_status",
        "before_demand",
        "after_demand",
    ),
    (
        ("pending-to-readiness", 190, 14, "not_configured", 20, 21),
        ("readiness-to-pending", 202, 15, "running", 20, 21),
    ),
)
@pytest.mark.asyncio
async def test_live_work_widget_swaps_cover_real_twenty_twenty_one_geometry(
    monkeypatch,
    direction,
    terminal_width,
    payload_rows,
    acp_status,
    before_demand,
    after_demand,
):
    async with make_console_pilot(size=(terminal_width, 52)) as pilot:
        await pilot.click("#console-inspector-rail-open")
        screen = pilot.app.screen
        rail = screen.query_one("#console-right-rail")
        live_root = rail.query_one("#console-live-work-section")
        header = rail.query_one("#console-live-work-header")
        pending_header = rail.query_one("#console-live-work-status-badge")
        readiness_header = rail.query_one("#console-live-work-source-readiness-title")
        bounded = rail.query_one(
            "#console-bounded-section-live-work", ConsoleBoundedSection
        )
        viewport = bounded.viewport
        hint = bounded.hint
        pending = ConsoleLiveWorkLaunch.from_values(
            source="test",
            title="physical row boundary",
            payload={f"row-{index:02}": "value" for index in range(payload_rows)},
        )
        screen.app_instance.acp_runtime_process_manager = SimpleNamespace(
            snapshot=lambda: {"status": acp_status}
        )

        if direction == "pending-to-readiness":
            screen._pending_console_launch_context = pending
            await screen._apply_console_live_work_card_swap()
        else:
            screen._pending_console_launch_context = None
            await screen._apply_console_live_work_card_swap()

        def initial_geometry_is_stable() -> bool:
            return (
                not bounded._reconcile_scheduled
                and not rail._outer_reconcile_scheduled
                and bounded.desired_content_lines == before_demand
                and bounded.viewport.content_region.height == 20
                and bounded.hint.display is False
            )

        await _wait_for_right_rail_condition(
            pilot,
            initial_geometry_is_stable,
            description="initial Live Work widget geometry",
        )
        await pilot.pause()
        assert initial_geometry_is_stable()

        assert bounded.desired_content_lines == before_demand
        assert bounded.viewport.content_region.height == 20
        assert bounded.hint.display is False

        order = []
        original_local = bounded.request_reconcile
        original_outer = rail.request_outer_reconcile

        def observe_local() -> None:
            order.append("local")
            original_local()

        def observe_outer() -> None:
            order.append("outer")
            original_outer()

        monkeypatch.setattr(bounded, "request_reconcile", observe_local)
        monkeypatch.setattr(rail, "request_outer_reconcile", observe_outer)
        baseline = rail._outer_owner_reconcile_count

        screen._pending_console_launch_context = (
            None if direction == "pending-to-readiness" else pending
        )
        await screen._apply_console_live_work_card_swap()

        def swapped_geometry_is_stable() -> bool:
            return (
                order == ["local", "outer"]
                and not bounded._reconcile_scheduled
                and not rail._outer_reconcile_scheduled
                and bounded.desired_content_lines == after_demand
                and bounded.viewport.content_region.height == 20
                and bounded.hint.display is True
                and rail.query_one("#console-live-work-section") is live_root
                and rail.query_one("#console-live-work-header") is header
                and rail.query_one("#console-bounded-section-live-work") is bounded
                and bounded.viewport is viewport
                and bounded.hint is hint
            )

        await _wait_for_right_rail_condition(
            pilot,
            swapped_geometry_is_stable,
            description="swapped Live Work widget geometry",
        )
        await pilot.pause()
        assert swapped_geometry_is_stable()

        assert order == ["local", "outer"]
        assert rail.query_one("#console-live-work-section") is live_root
        assert rail.query_one("#console-live-work-header") is header
        assert rail.query_one("#console-live-work-status-badge") is pending_header
        assert (
            rail.query_one("#console-live-work-source-readiness-title")
            is readiness_header
        )
        assert rail.query_one("#console-bounded-section-live-work") is bounded
        assert bounded.viewport is viewport
        assert bounded.hint is hint
        assert bounded.desired_content_lines == after_demand
        assert bounded.viewport.content_region.height == 20
        assert bounded.hint.display is True
        assert bounded.hint.region.height == 1
        assert bounded._reconcile_scheduled is False
        assert rail._outer_owner_reconcile_count == baseline + 1
        assert rail._outer_reconcile_scheduled is False

        if direction == "pending-to-readiness":
            assert (
                rail.query_one("#console-live-work-source-readiness").parent is viewport
            )
            assert pending_header.display is False
            assert readiness_header.display is True
        else:
            assert rail.query_one("#console-pending-launch-card").parent is viewport
            assert pending_header.display is True
            assert readiness_header.display is False


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
        inspector_module,
        "logger",
        SimpleNamespace(
            warning=lambda message, fingerprint: diagnostics.append(
                (message, fingerprint)
            )
        ),
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
@pytest.mark.parametrize("size", [(140, 45), (235, 52)])
async def test_inspector_root_pins_project_and_authority_above_outer_body(
    size,
) -> None:
    async with make_console_pilot(size=size) as pilot:
        await pilot.click("#console-inspector-rail-open")
        await pilot.pause()

        rail = pilot.app.screen.query_one("#console-right-rail")
        assert tuple(child.id for child in rail.children) == (
            "console-inspector-rail-header",
            "console-project-instruction-status",
            "console-send-authority-summary",
            "console-inspector-rail-body",
            "console-inspector-outer-scroll-hint",
        )
        summary = rail.query_one("#console-send-authority-summary")
        body = rail.query_one("#console-inspector-rail-body")
        assert summary.parent is rail
        assert summary not in tuple(body.query("*"))
        assert summary.region.height == 6
        assert rail.content_region.contains_region(summary.region)

        region_before = summary.region
        body.scroll_end(animate=False)
        await pilot.pause()
        assert summary.region == region_before


@pytest.mark.asyncio
async def test_control_sync_shares_one_inspector_snapshot_with_both_consumers(
    monkeypatch,
) -> None:
    async with make_console_pilot() as pilot:
        await pilot.click("#console-inspector-rail-open")
        await pilot.pause()

        screen = pilot.app.screen
        summary = screen.query_one(
            "#console-send-authority-summary", ConsoleSendAuthoritySummary
        )
        inspector = screen.query_one(
            "#console-run-inspector-state", ConsoleRunInspector
        )
        snapshot = screen._build_console_inspector_state(
            screen._pending_console_launch_context
        )
        builds: list[object] = []
        summary_states: list[object] = []
        inspector_states: list[object] = []
        rail_states: list[object] = []
        build_rail_state = screen._build_console_rail_state
        monkeypatch.setattr(
            screen,
            "_build_console_inspector_state",
            lambda _launch: builds.append(snapshot) or snapshot,
        )
        monkeypatch.setattr(summary, "sync_state", summary_states.append)
        monkeypatch.setattr(inspector, "sync_state", inspector_states.append)
        monkeypatch.setattr(
            screen,
            "_build_console_rail_state",
            lambda **kwargs: (
                rail_states.append(kwargs["inspector_state"])
                or build_rail_state(**kwargs)
            ),
        )

        screen._sync_console_control_bar()

        assert builds == [snapshot]
        assert summary_states == [snapshot]
        assert inspector_states == [snapshot]
        assert rail_states == [snapshot]
        assert summary_states[0] is inspector_states[0]
        assert inspector_states[0] is rail_states[0]


@pytest.mark.asyncio
async def test_effective_empty_scope_survives_snapshot_projection_as_no_sources(
    monkeypatch,
) -> None:
    async with make_console_pilot() as pilot:
        screen = pilot.app.screen
        monkeypatch.setattr(
            screen._retrieval,
            "_build_console_retrieval_scope_state",
            lambda: ConsoleRetrievalScopeState.empty(cause="no-workspace-overlap"),
        )

        snapshot = screen._build_console_inspector_state(
            screen._pending_console_launch_context
        )

        assert snapshot.scope_item_count == 0
        assert project_console_send_authority(snapshot).scope == "No sources"


@pytest.mark.asyncio
async def test_strict_inspector_rejection_keeps_authority_summary_at_prior_snapshot(
    monkeypatch,
) -> None:
    ownership = importlib.import_module(
        "tldw_chatbook.Widgets.Console.console_inspector_ownership"
    )
    async with make_console_pilot() as pilot:
        await pilot.click("#console-inspector-rail-open")
        await pilot.pause()

        screen = pilot.app.screen
        summary = screen.query_one(
            "#console-send-authority-summary", ConsoleSendAuthoritySummary
        )
        inspector = screen.query_one(
            "#console-run-inspector-state", ConsoleRunInspector
        )
        inspector.ownership_policy = ownership.InspectorOwnershipPolicy.STRICT
        prior = summary.last_state
        rejected = ConsoleInspectorState(
            rows=(ConsoleDisplayRow("Unknown changed row", "must not publish"),)
        )
        monkeypatch.setattr(
            screen,
            "_build_console_inspector_state",
            lambda _launch: rejected,
        )

        with pytest.raises(ownership.UnownedInspectorContentError):
            screen._sync_console_control_bar()

        assert summary.last_state is prior


@pytest.mark.asyncio
async def test_authority_focus_f1_discloses_all_five_complete_facts() -> None:
    async with make_console_pilot() as pilot:
        await pilot.click("#console-inspector-rail-open")
        await pilot.pause()

        screen = pilot.app.screen
        summary = screen.query_one(
            "#console-send-authority-summary", ConsoleSendAuthoritySummary
        )
        expected = summary.contextual_help_rows()
        summary.focus()
        await pilot.pause()
        await pilot.press("f1")
        await pilot.pause()

        panel = pilot.app.screen
        assert isinstance(panel, WorkbenchHelpPanel)
        rendered = panel.state.render_text()
        assert "What happens if I send now?" in rendered
        for label, value in expected:
            assert f"{label}: {value}" in rendered


@pytest.mark.asyncio
async def test_authority_focus_f1_preserves_literal_rich_markup() -> None:
    async with make_console_pilot() as pilot:
        await pilot.click("#console-inspector-rail-open")
        await pilot.pause()

        screen = pilot.app.screen
        summary = screen.query_one(
            "#console-send-authority-summary", ConsoleSendAuthoritySummary
        )
        summary.sync_state(
            ConsoleInspectorState(
                rows=(
                    ConsoleDisplayRow("Workspace", "[bold]literal[/bold]"),
                    ConsoleDisplayRow("Selected conversation", "Chat"),
                    ConsoleDisplayRow("Provider", "ready", status="ready"),
                )
            )
        )
        summary.focus()
        await pilot.pause()
        await screen.action_show_workbench_help()
        await pilot.pause()

        panel = pilot.app.screen
        assert isinstance(panel, WorkbenchHelpPanel)
        body = panel.query_one("#workbench-help-body", Static)
        assert "[bold]literal[/bold]" in body.render().plain


@pytest.mark.asyncio
async def test_more_toggle_disappearance_recovers_to_next_inspector_boundary(
    monkeypatch,
) -> None:
    async with make_console_pilot() as pilot:
        await pilot.click("#console-inspector-rail-open")
        await pilot.pause()

        screen = pilot.app.screen
        rail = screen.query_one("#console-right-rail")
        inspector = screen.query_one(
            "#console-run-inspector-state", ConsoleRunInspector
        )
        toggle = inspector.query_one("#console-inspector-more-toggle", Button)
        boundaries = rail._mounted_boundaries()
        expected_section_id = "selected-conversation"
        assert expected_section_id in {
            section.section_id for section, _header in boundaries
        }
        toggle.focus()
        await pilot.pause()
        assert pilot.app.focused is toggle
        recovered: list[str | None] = []
        recover_focus = inspector._on_more_focus_removed
        monkeypatch.setattr(
            inspector,
            "_on_more_focus_removed",
            lambda section_id: (
                recovered.append(section_id) or recover_focus(section_id)
            ),
        )

        rows = tuple(
            row
            for row in inspector.state.rows
            if row.label not in {"Tools", "Approvals", "Artifacts"}
        ) + (
            ConsoleDisplayRow("Tools", "1 ready"),
            ConsoleDisplayRow("Approvals", "1 pending", status="blocked"),
            ConsoleDisplayRow("Artifacts", "Chatbook available"),
        )
        inspector.sync_state(replace(inspector.state, rows=rows))
        for _ in range(4):
            await pilot.pause()

        assert not list(inspector.query("#console-inspector-more-toggle"))
        focused = pilot.app.focused
        assert focused is not None
        assert recovered == [expected_section_id]
        assert rail.inspector_active(focused)
        assert focused.id == "console-inspector-rail-body"
        assert focused.id not in {
            "console-inspector-tools-heading",
            "console-inspector-approvals-heading",
            "console-inspector-artifacts-heading",
            "console-native-transcript",
            "console-native-composer",
        }


@pytest.mark.asyncio
async def test_more_disappearance_does_not_steal_newer_context_focus() -> None:
    async with make_console_pilot() as pilot:
        await pilot.click("#console-inspector-rail-open")
        await pilot.pause()

        screen = pilot.app.screen
        inspector = screen.query_one(
            "#console-run-inspector-state", ConsoleRunInspector
        )
        toggle = inspector.query_one("#console-inspector-more-toggle", Button)
        context_focus = screen.query_one("#console-context-rail-collapse", Button)
        toggle.focus()
        await pilot.pause()
        assert pilot.app.focused is toggle

        rows = tuple(
            row
            for row in inspector.state.rows
            if row.label not in {"Tools", "Approvals", "Artifacts"}
        ) + (
            ConsoleDisplayRow("Tools", "1 ready"),
            ConsoleDisplayRow("Approvals", "1 pending", status="blocked"),
            ConsoleDisplayRow("Artifacts", "Chatbook available"),
        )
        inspector.sync_state(replace(inspector.state, rows=rows))
        context_focus.focus()
        for _ in range(4):
            await pilot.pause()

        assert not list(inspector.query("#console-inspector-more-toggle"))
        assert pilot.app.focused is context_focus


@pytest.mark.asyncio
async def test_more_toggle_persists_without_programmatic_repost() -> None:
    async with make_console_pilot() as pilot:
        await pilot.click("#console-inspector-rail-open")
        await pilot.pause()

        screen = pilot.app.screen
        body = screen.query_one("#console-inspector-more-body")
        assert body.display is False
        toggle = screen.query_one("#console-inspector-more-toggle", Button)
        toggle.focus()
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause(0.2)
        assert body.display is True
        stored = next(
            iter(screen.app_instance.app_config["console"]["rail_state"].values())
        )
        assert stored["inspector_more_open"] is True

        rail_state = replace(
            screen._current_console_rail_state(), inspector_more_open=False
        )
        screen._sync_console_rail_visibility(rail_state)
        await pilot.pause()

        assert body.display is False
        assert stored["inspector_more_open"] is True


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
