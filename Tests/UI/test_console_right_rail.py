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
import asyncio
import importlib
import threading
from types import SimpleNamespace

import pytest
from textual.containers import Horizontal
from textual.widgets import Button, Static

from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_display_state import (
    CONSOLE_INSPECTOR_SAVE_CHATBOOK_ID,
    ConsoleDisplayRow,
    ConsoleInspectorAction,
    ConsoleInspectorState,
    ConsoleRetrievalScopeState,
)
from tldw_chatbook.Widgets.Console.console_bounded_section import (
    ConsoleBoundedSection,
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

from Tests.UI.console_rail_section_helpers import (
    _wait_for_console_quiet,
    _wait_for_right_rail_condition,
)
from Tests.UI.test_console_native_chat_flow import (
    _configure_native_ready_console,
    _sync_console_control_bar_now,
)
from Tests.UI.test_console_shell_regions import click_inspect_rail_open
from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)

# TASK-41633: every mounted Console here reads the collection-time config;
# the per-test sandbox refused it (RecoveryRequired: raw_source_selection_changed).
pytestmark = pytest.mark.bootstrap_profile


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
    from tldw_chatbook.config import get_user_data_dir

    # At the profile's own data folder the MCP store binds, as by default in
    # the app. At the factory's folder it stays unbound, and the live-work
    # MCP row names that reason in place of "Not wired" (TASK-41631), which
    # also makes the readiness card one row taller.
    app = _build_test_app(user_data_dir=get_user_data_dir())
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


_EXPECTED_BOUNDARY_ANCHORS = (
    ("console-project-instruction-status", "Project Instructions"),
    ("console-send-authority-summary", "Next send authority"),
    # task-9: Environment and Tasks sections -- mounted at the TOP of the
    # scrollable Inspector body, ahead of the staged-context tray, per the
    # redesign spec's rail ordering.
    ("console-environment-section", "Environment"),
    ("console-tasks-section", "Tasks"),
    ("console-staged-context-tray", "Sources — next send"),
    # TASK-24611: the Library search controls, moved out of the live-work
    # readiness card to sit beside the empty state that names them. It is a
    # real n/p boundary, so it belongs in this inventory.
    ("console-library-search-region", "Library search"),
    ("console-retrieval-scope-row", "Scope"),
    # task-10: the Agent rail's fleet mini-section, moved here from the
    # left rail's Agent section. TASK-41959 (owner ruling 2026-11-25): below
    # the Sources tray's group, ahead of the run inspector.
    ("console-agent-section-subagents", "Agents"),
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
    # task-9: Environment and Tasks are the first two children now, ahead of
    # the staged-context tray -- mounted unconditionally (hidden via
    # `styles.display = "none"` when their projection has no rows, never
    # omitted from the DOM), so they always occupy these two slots.
    #
    # task-10: the Agent rail's fleet mini-section moved here from the left
    # rail's Agent section -- same unconditional-mount/hide-when-empty
    # pattern. TASK-41959 (owner ruling 2026-11-25) put it after the Scope
    # row, so the Sources tray follows Tasks directly.
    #
    # TASK-24611: the Library search region sits directly beneath the Sources
    # tray, because the tray's empty state ("Stage sources from Library.") is
    # the sentence it answers. It used to be the first three children of the
    # readiness card at the very bottom, ~25 rows further down. The readiness
    # card itself still anchors last, which is task-400's placement.
    assert direct_children[:7] == (
        "console-environment-section",
        "console-tasks-section",
        "console-staged-context-tray",
        "console-library-search-region",
        "console-retrieval-scope-row",
        "console-agent-section-subagents",
        "console-run-inspector",
    )
    assert len(direct_children) == 8
    assert direct_children[-1] == "console-live-work-section"

    run_wrapper = rail.query_one("#console-run-inspector")
    run_wrapper_children = tuple(child.id for child in run_wrapper.children)
    assert run_wrapper_children == (
        "console-run-inspector-state",
        "console-selected-turn",
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
        # task-9: five pre-run boundaries, not three -- Environment and
        # Tasks were prepended ahead of the Sources tray/Library
        # search/Scope trio (TASK-24611 added the middle one, "three ...
        # not two", to that trio). task-10 then added the Agent fleet
        # mini-section, making six; TASK-41959 put it after Scope.
        *direct_children[:6],
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
        "Environment",
        "Tasks",
        "Sources — next send",
        "Library search",
        "Scope",
        "Agents",
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
    assert dict(_EXPECTED_BOUNDARY_ANCHORS[:8]) | {
        "console-settings-summary": "Session Settings",
    } | {live_id: "Live Work" for live_id in _LIVE_WORK_IDS} == {
        "console-project-instruction-status": "Project Instructions",
        "console-send-authority-summary": "Next send authority",
        "console-environment-section": "Environment",
        "console-tasks-section": "Tasks",
        "console-staged-context-tray": "Sources — next send",
        "console-library-search-region": "Library search",
        "console-retrieval-scope-row": "Scope",
        "console-agent-section-subagents": "Agents",
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
        await click_inspect_rail_open(pilot)
        screen = pilot.app.screen
        rail = screen.query_one("#console-right-rail")
        # The Console's own control-bar syncs push the screen's built
        # Inspector state into the mounted inspector. One still owed from
        # startup when this test swapped in its own state landed after the
        # swap under load, and the census read the screen's state instead
        # ("More" at index 4, round 24). So swap only once that work is done,
        # and read only once nothing new is owed.
        await _wait_for_console_quiet(pilot, screen)
        rail._inspector_state = exhaustive_state
        await rail.recompose()
        await _wait_for_console_quiet(pilot, screen)
        inspector = rail.query_one("#console-run-inspector-state", ConsoleRunInspector)
        assert inspector.state is exhaustive_state, "a Console sync replaced the state"

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
            ("#console-settings-summary", "session-settings"),
            ("#console-live-work-section", "live-work"),
        )
        for root_selector, section_id in specialized:
            root = rail.query_one(root_selector)
            bodies = list(root.query(ConsoleBoundedSection))
            assert [body.section_id for body in bodies] == [section_id]


@pytest.mark.asyncio
async def test_new_specialized_sibling_fails_mounted_production_census():
    async with make_console_pilot() as pilot:
        await click_inspect_rail_open(pilot)
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
async def test_agent_fleet_section_lives_in_the_right_rail():
    """task-10: the agent-fleet mini-section moved out of the left rail.

    ``#console-agent-section-subagents`` (id unchanged) must now be a
    descendant of the Inspector rail's own scrollable body rather than the
    left (Context) rail's -- the move this task exists to make.
    """
    async with make_console_pilot() as pilot:
        screen = pilot.app.screen
        section = screen.query_one("#console-agent-section-subagents")
        body = screen.query_one("#console-inspector-rail-body")
        node = section
        while node is not None and node is not body:
            node = node.parent
        assert node is body, (
            "fleet section must be a descendant of the Inspect rail body"
        )


@pytest.mark.asyncio
async def test_sources_use_exact_twenty_line_content_ceiling():
    async with make_console_pilot(size=(235, 52)) as pilot:
        await click_inspect_rail_open(pilot)
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
        # TASK-24611 moved the Library search controls out of the readiness
        # card, so its demand fell 21 -> 15 (a 2-row scope label, a 3-row
        # Input and a 1-row Button). With the old 9-row payload the pending
        # card sat at exactly 20 and NEITHER side crossed the 20-row cap any
        # more -- the swap still happened but the test had stopped exercising
        # the hint-on/hint-off boundary it exists for. A 10-row payload puts
        # pending back at 21, so each direction still crosses the cap once.
        ("pending-to-readiness", 250, 10, "not_configured", 21, 15),
        ("readiness-to-pending", 250, 10, "not_configured", 15, 21),
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
        await click_inspect_rail_open(pilot)
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

        # TASK-42415: a swap requests its local and owner reconcile one refresh
        # later (`call_after_refresh`). A readiness-to-readiness first swap
        # leaves the geometry as it was, so a wait on geometry and the two
        # flags alone passed before that request landed; one pause later CI
        # found the owner pass still scheduled. Record each request with the
        # owner-pass count it saw, and wait for a pass counted after it.
        swap_reconcile_owner_counts: list[int] = []
        original_swap_reconcile = screen._request_console_live_work_reconcile

        def observe_swap_reconcile() -> None:
            swap_reconcile_owner_counts.append(rail._outer_owner_reconcile_count)
            original_swap_reconcile()

        monkeypatch.setattr(
            screen, "_request_console_live_work_reconcile", observe_swap_reconcile
        )

        def swap_reconcile_settled(requests: int) -> bool:
            return (
                len(swap_reconcile_owner_counts) == requests
                and rail._outer_owner_reconcile_count > swap_reconcile_owner_counts[-1]
            )

        def rail_report() -> str:
            return (
                f"demand={bounded.desired_content_lines}, "
                f"width={viewport.content_region.width}, "
                f"viewport={viewport.content_region.height}, "
                f"hint={bounded.hint.display}, "
                f"local_scheduled={bounded._reconcile_scheduled}, "
                f"outer_scheduled={rail._outer_reconcile_scheduled}, "
                f"swap_reconcile_requests={swap_reconcile_owner_counts}, "
                f"owner_passes={rail._outer_owner_reconcile_count}"
            )

        if direction == "pending-to-readiness":
            screen._pending_console_launch_context = pending
            await screen._apply_console_live_work_card_swap()
        else:
            screen._pending_console_launch_context = None
            await screen._apply_console_live_work_card_swap()

        def initial_geometry_is_stable() -> bool:
            return (
                swap_reconcile_settled(1)
                # The rail counts requests from every mounted owner. Finish
                # startup work before measuring the second swap's one pass.
                and not screen.console_pending_startup_work()
                and not bounded._reconcile_scheduled
                and not rail._outer_reconcile_scheduled
                and bounded.desired_content_lines == before_demand
                and bounded.viewport.content_region.height
                == min(before_demand, bounded.max_content_lines)
                and bounded.hint.display is (before_demand > bounded.max_content_lines)
            )

        await _wait_for_right_rail_condition(
            pilot,
            initial_geometry_is_stable,
            description=lambda: f"initial Live Work widget geometry: {rail_report()}",
            stable_polls=2,
        )

        assert bounded.desired_content_lines == before_demand
        assert bounded.viewport.content_region.height == min(
            before_demand, bounded.max_content_lines
        )
        assert bounded.hint.display is (before_demand > bounded.max_content_lines)

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
                order[:2] == ["local", "outer"]
                and swap_reconcile_settled(2)
                and not bounded._reconcile_scheduled
                and not rail._outer_reconcile_scheduled
                and rail._outer_owner_reconcile_count == baseline + 1
                and bounded.desired_content_lines == after_demand
                and bounded.viewport.content_region.height
                == min(after_demand, bounded.max_content_lines)
                and bounded.hint.display is (after_demand > bounded.max_content_lines)
                and rail.query_one("#console-live-work-section") is live_root
                and rail.query_one("#console-live-work-header") is header
                and rail.query_one("#console-bounded-section-live-work") is bounded
                and bounded.viewport is viewport
                and bounded.hint is hint
            )

        await _wait_for_right_rail_condition(
            pilot,
            swapped_geometry_is_stable,
            description=lambda: (
                f"swapped Live Work widget geometry: {rail_report()}, "
                f"order={order}, completed_owner_passes="
                f"{rail._outer_owner_reconcile_count - baseline}"
            ),
            stable_polls=2,
        )

        assert order[:2] == ["local", "outer"]
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
        assert bounded.viewport.content_region.height == min(
            after_demand, bounded.max_content_lines
        )
        assert bounded.hint.display is (after_demand > bounded.max_content_lines)
        assert bounded.hint.region.height == int(
            after_demand > bounded.max_content_lines
        )
        assert bounded._reconcile_scheduled is False
        assert rail._outer_owner_reconcile_count == baseline + 1
        assert rail._outer_reconcile_scheduled is False
        # A redundant trailing owner pass would land after settling: re-check
        # the count once more after an idle pause (flags are not re-asserted).
        await pilot.pause()
        assert rail._outer_owner_reconcile_count == baseline + 1, rail_report()

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
    from tldw_chatbook.Widgets.Console.console_background_effect import (
        ConsoleTranscriptSurface,
        active_console_transcript,
    )

    ownership = importlib.import_module(
        "tldw_chatbook.Widgets.Console.console_inspector_ownership"
    )

    async with make_console_pilot() as pilot:
        screen = pilot.app.screen
        store = screen._ensure_console_chat_store()
        initial = store.ensure_session()
        store.append_message(
            initial.id,
            role=ConsoleMessageRole.USER,
            content="retained viewport fixture",
        )
        await screen._sync_native_console_chat_ui()
        await pilot.pause()
        surface = screen.query_one(ConsoleTranscriptSurface)
        retained = surface.transcript
        target = store.create_session(title="Inspector target", activate=False)
        message = store.append_message(
            target.id,
            role=ConsoleMessageRole.USER,
            content="selected producer message",
        )
        await screen._session._activate_native_console_session(target.id)
        await screen._sync_native_console_chat_ui()
        await pilot.pause()
        active = active_console_transcript(screen)
        assert active is not None and active is surface.transcript
        assert active is not retained and active.is_attached and active.display
        assert retained.is_attached and not retained.display
        assert screen.query_one("#console-native-transcript") is retained
        assert active.display_message(message.id) is not None
        assert retained.display_message(message.id) is None
        assert (
            store.session_id_for_message(message.id)
            == store.active_session_id
            == target.id
        )
        transcript = active_console_transcript(screen)
        assert transcript is active
        transcript.select_message(message.id)
        assert transcript.selected_message_id == message.id
        screen._retrieval._active_dictionaries_summary = {
            "dictionaries": [{"name": "Producer dictionary", "source": "conversation"}]
        }
        screen._retrieval._active_world_books_summary = {
            "world_books": [
                {"name": "Producer world book", "entry_count": 2, "enabled": True}
            ]
        }
        monkeypatch.setattr(
            screen,
            "_console_provider_blocker_copy",
            lambda *, settings_readiness=None: "Provider setup needed",
        )
        monkeypatch.setattr(
            screen,
            "_console_provider_recovery_action",
            lambda *, settings_readiness=None: (
                "Open Settings",
                "settings",
                "Open provider settings",
            ),
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
async def test_active_run_alone_never_produces_a_setup_recovery_row(monkeypatch):
    """task-32345: an active turn is not a provider-configuration problem.

    ``build_console_settings_readiness(..., active_run=True)`` blocks an
    OTHERWISE fully-configured provider with ``recovery_action=
    "wait_for_active_run"`` -- the same "wait, don't reconfigure" signal
    ``_console_setup_blocked_reason``/``_console_send_blocked_reason``
    already special-case. ``_console_provider_blocker_copy`` (which feeds
    the Setup/Blocked-impact/Next-action inspector rows) did not, so a
    turn with a pending approval read "Setup: Provider configuration
    required" -- and the authority summary's ``recovery_required`` check
    (any "Next action" row present) rendered "Recovery required" over a
    healthy pending approval.
    """
    from tldw_chatbook.Chat.console_session_settings import (
        ConsoleSessionSettings,
        build_console_settings_readiness,
    )

    async with make_console_pilot() as pilot:
        screen = pilot.app.screen
        settings = ConsoleSessionSettings(
            provider="ollama", model="model", base_url="http://127.0.0.1:11434"
        )
        readiness = build_console_settings_readiness(
            settings,
            app_config={
                "api_settings": {"ollama": {"api_url": "http://127.0.0.1:11434"}}
            },
            environ={},
            active_run=True,
        )
        assert readiness.operability == "not_ready"
        assert readiness.recovery_action == "wait_for_active_run"
        monkeypatch.setattr(
            screen, "_active_console_settings_readiness", lambda: (settings, readiness)
        )

        assert screen._console_provider_blocker_copy() == ""

        state = screen._build_console_inspector_state(None)
        labels = {row.label for row in state.rows}
        assert "Setup" not in labels
        assert "Blocked impact" not in labels
        assert "Next action" not in labels


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
        await click_inspect_rail_open(pilot)
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
        assert str(open_button.label) == "◂ Inspect"
        far_end = (
            open_button.region.width - 1,
            open_button.region.height // 2,
        )
        await click_inspect_rail_open(pilot, offset=far_end)
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
        # b7dd8e53f2: the Console runtime owns both project-instruction
        # callbacks and overwrites any caller copy (TASK-34667).
        runtime = pilot.app.screen._console_runtime()
        assert controller._confirm_project_instruction_dispatch.__self__ is runtime
        assert controller._select_project_instruction_binding.__self__ is runtime

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
        await click_inspect_rail_open(pilot)

        screen = pilot.app.screen
        button = screen.query_one("#console-inspector-rail-collapse", Button)
        header = button.parent

        assert _right_rail_open(pilot) is True
        assert _handle_visible(pilot) is False
        assert isinstance(header, Horizontal)
        assert list(header.children) == [button]
        assert not screen.query("#console-inspector-rail-title")
        # TASK-23195 follow-up: the ASCII arrow became a readable name
        # plus one resolved glyph, mirroring the Context rail. The
        # header is still ONE full-width collapse button.
        assert "Inspect" in str(button.label)
        assert "---" not in str(button.label)
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
        await click_inspect_rail_open(pilot)

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
        await click_inspect_rail_open(pilot)

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

        await _sync_console_control_bar_now(screen)

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
        await click_inspect_rail_open(pilot)

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
            await _sync_console_control_bar_now(screen)

        assert summary.last_state is prior


@pytest.mark.asyncio
async def test_authority_focus_f1_discloses_all_five_complete_facts() -> None:
    async with make_console_pilot() as pilot:
        await click_inspect_rail_open(pilot)

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
        await click_inspect_rail_open(pilot)

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
        assert r"\[bold]literal\[/bold]" in body.render().plain


@pytest.mark.asyncio
async def test_more_toggle_disappearance_recovers_to_next_inspector_boundary(
    monkeypatch,
) -> None:
    async with make_console_pilot() as pilot:
        await click_inspect_rail_open(pilot)

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

        promoted_rows = (
            ConsoleDisplayRow("Tools", "1 ready"),
            ConsoleDisplayRow("Approvals", "1 pending", status="blocked"),
            ConsoleDisplayRow("Artifacts", "Chatbook available"),
        )

        def promote_groups(state: ConsoleInspectorState) -> ConsoleInspectorState:
            rows = tuple(
                row
                for row in state.rows
                if row.label not in {"Tools", "Approvals", "Artifacts"}
            )
            return replace(state, rows=rows + promoted_rows)

        # A native control-bar refresh can publish between the synthetic
        # promotion and its recompose. Project the same three rows there too.
        build_state = screen._build_console_inspector_state

        def build_promoted_state(pending_launch):
            return promote_groups(build_state(pending_launch))

        monkeypatch.setattr(
            screen, "_build_console_inspector_state", build_promoted_state
        )
        inspector.sync_state(promote_groups(inspector.state))
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
        await click_inspect_rail_open(pilot)

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
async def test_conditional_demotion_does_not_steal_newer_context_focus() -> None:
    async with make_console_pilot() as pilot:
        await click_inspect_rail_open(pilot)

        screen = pilot.app.screen
        inspector = screen.query_one(
            "#console-run-inspector-state", ConsoleRunInspector
        )
        context_focus = screen.query_one("#console-context-rail-collapse", Button)
        active_actions = tuple(
            replace(action, enabled=True, disabled_reason="")
            if action.widget_id == CONSOLE_INSPECTOR_SAVE_CHATBOOK_ID
            else action
            for action in inspector.state.actions
        )
        active_state = replace(inspector.state, actions=active_actions)
        before_recompose = inspector.recompose_count
        inspector.sync_state(active_state)
        await _wait_for_right_rail_condition(
            pilot,
            lambda: (
                inspector.recompose_count > before_recompose
                and not inspector.query_one(
                    f"#{CONSOLE_INSPECTOR_SAVE_CHATBOOK_ID}", Button
                ).disabled
                and inspector.query_one("#console-inspector-more-body")
                not in inspector.query_one(
                    f"#{CONSOLE_INSPECTOR_SAVE_CHATBOOK_ID}", Button
                ).ancestors
            ),
            description="promoted Artifacts action before demotion",
        )
        artifact_action = inspector.query_one(
            f"#{CONSOLE_INSPECTOR_SAVE_CHATBOOK_ID}", Button
        )
        artifact_action.focus()
        await pilot.pause()
        assert pilot.app.focused is artifact_action

        disabled_actions = tuple(
            replace(action, enabled=False, disabled_reason="unavailable")
            if action.widget_id == CONSOLE_INSPECTOR_SAVE_CHATBOOK_ID
            else action
            for action in active_actions
        )
        inspector.sync_state(replace(active_state, actions=disabled_actions))
        context_focus.focus()
        for _ in range(4):
            await pilot.pause()

        assert pilot.app.focused is context_focus


@pytest.mark.asyncio
async def test_more_toggle_persists_without_programmatic_repost() -> None:
    async with make_console_pilot() as pilot:
        await click_inspect_rail_open(pilot)

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
        await click_inspect_rail_open(pilot)

        button = pilot.app.screen.query_one("#console-inspector-rail-collapse", Button)
        assert "Inspect" in str(button.label)
        assert "---" not in str(button.label)
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
        await click_inspect_rail_open(pilot)

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
        await click_inspect_rail_open(pilot)
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
            await click_inspect_rail_open(pilot)
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


# --- TASK-24602 -------------------------------------------------------------


@pytest.mark.asyncio
async def test_screen_wires_a_failed_run_into_the_pinned_authority_line():
    """TASK-24602 end to end, not just the projection.

    The controller already recorded a terminal FAILED run state with visible
    copy; the defect was that ``_build_console_inspector_state`` never read
    it, so the pinned line answered "Ready" beside a transcript saying the
    run had failed. A projection-only test would pass with the screen still
    not wired, which is exactly how this shipped.
    """
    from tldw_chatbook.Chat.console_chat_models import (
        ConsoleRunState,
        ConsoleRunStatus,
    )

    async with make_console_pilot() as pilot:
        screen = pilot.app.screen
        await _wait_for_selector(screen, pilot, "#console-inspector-rail-open")

        ready = screen._build_console_inspector_state(None)
        assert ready.run_failed is False
        assert project_console_send_authority(ready).run == "Ready"

        controller = screen._ensure_console_chat_controller()
        # `run_state` is a read-only facade; `_set_run_state` is the only
        # path that mutates the per-session map (parallel-agents spec §2),
        # and it is the same call the agent-run failure path makes.
        controller._set_run_state(
            ConsoleRunState(
                ConsoleRunStatus.FAILED,
                "Agent run failed: provider returned HTTP 401",
            )
        )

        after = screen._build_console_inspector_state(None)
        assert after.run_failed is True, (
            "the screen did not read the controller's FAILED run state"
        )
        assert "401" in after.run_failure_reason
        projected = project_console_send_authority(after).run
        assert projected.startswith("Failed"), projected
        assert "401" in projected


@pytest.mark.asyncio
async def test_a_failed_run_stops_describing_the_next_send_once_it_starts():
    """TASK-24602, the recovery direction (Qodo focus area, third round).

    Pinning the authority line to a terminal FAILED status is only correct
    if something un-pins it, or an old failure keeps describing a send that
    has already recovered. `_clear_terminal_run_state` handles session
    switch and provider change, but NOT a new send on the same session --
    that path relies on the first thing `_send_console_message` does, which
    is to transition the run state to VALIDATING. This asserts that
    transition actually clears the pinned line, so the un-pinning is covered
    by a test rather than by reading the send path.
    """
    from tldw_chatbook.Chat.console_chat_models import (
        ConsoleRunState,
        ConsoleRunStatus,
    )

    async with make_console_pilot() as pilot:
        screen = pilot.app.screen
        await _wait_for_selector(screen, pilot, "#console-inspector-rail-open")
        controller = screen._ensure_console_chat_controller()

        controller._set_run_state(
            ConsoleRunState(ConsoleRunStatus.FAILED, "Agent run failed: HTTP 401")
        )
        assert screen._build_console_inspector_state(None).run_failed is True

        # The send path's own first transition, verbatim.
        controller._set_run_state(
            ConsoleRunState(ConsoleRunStatus.VALIDATING, "Validating provider.")
        )
        recovered = screen._build_console_inspector_state(None)
        assert recovered.run_failed is False, (
            "a new send left the previous failure pinned to the authority line"
        )
        assert recovered.run_failure_reason == ""
        assert not project_console_send_authority(recovered).run.startswith("Failed")


# --- TASK-24611 -------------------------------------------------------------


@pytest.mark.asyncio
async def test_library_search_sits_with_the_empty_state_that_points_at_it():
    """TASK-24611: the control that stages sources lives beside the sentence
    telling you to stage sources.

    The Sources tray's empty state reads "No sources attached. Stage sources
    from Library." The `Ask Library` input and `Search Library` button that
    do exactly that used to be the first children of the live-work readiness
    card at the BOTTOM of the rail -- roughly 25 rows below that sentence,
    behind the fold, under a heading naming a status inventory.

    Deliberately NOT a whole-section swap: the readiness card keeps the
    bottom anchor task-400 chose for it (stated in that task's own test
    docstring), and run state keeps its place above the fold.
    """
    async with make_console_pilot(size=(120, 45)) as pilot:
        screen = pilot.app.screen
        await _wait_for_selector(screen, pilot, "#console-inspector-rail-open")
        await click_inspect_rail_open(pilot)

        tray = screen.query_one("#console-staged-context-tray")
        search = screen.query_one("#console-library-search-region")
        scope = screen.query_one("#console-retrieval-scope-row")
        run_inspector = screen.query_one("#console-run-inspector")

        assert tray.region.y < search.region.y, (
            "the Library search must sit BELOW the Sources tray's empty state"
        )
        assert search.region.y < scope.region.y < run_inspector.region.y, (
            "the search region must sit between the tray and the run "
            f"inspector; got search={search.region.y} scope={scope.region.y} "
            f"run={run_inspector.region.y}"
        )

        # The controls really moved -- they are no longer inside the
        # readiness card, which is now rows only.
        readiness = screen.query_one("#console-live-work-source-readiness")
        assert not list(readiness.query("#console-library-rag-query-input"))
        assert not list(readiness.query("#console-run-library-rag"))
        assert search.query_one("#console-library-rag-query-input")
        assert search.query_one("#console-run-library-rag")

        # And the readiness card still anchors at the bottom (task-400).
        assert run_inspector.region.y < readiness.region.y


# --- TASK-24704 -------------------------------------------------------------


@pytest.mark.asyncio
async def test_readiness_card_picks_up_mcp_after_it_connects():
    """TASK-24704 (Qodo #1): the card must not freeze its MCP sample.

    The readiness card reads MCP and RAG when it is BUILT, and the only
    thing that rebuilds it is the live-work card swap -- scheduled from
    `_sync_console_pending_launch_surfaces`, which is reached only when a
    launch is consumed or evidence is unstaged. A card mounted before MCP
    connected therefore kept saying "Not wired" no matter how many tools the
    catalog later published.
    """
    async with make_console_pilot() as pilot:
        screen = pilot.app.screen
        await _wait_for_selector(screen, pilot, "#console-inspector-rail-open")
        await click_inspect_rail_open(pilot)

        mcp_row = screen.query_one("#console-live-work-source-mcp", Static)
        assert "Not wired" in str(mcp_row.renderable), str(mcp_row.renderable)

        # MCP connects and publishes a catalog, exactly as
        # `_publish_mcp_inspector_counts` does mid-run.
        screen.app_instance.console_mcp_tool_count = 4
        screen._sync_console_live_work_readiness_rows()
        await pilot.pause()

        refreshed = str(
            screen.query_one("#console-live-work-source-mcp", Static).renderable
        )
        assert refreshed == "MCP: Connected - 4 tools ready.", refreshed

        # And back again when the catalog empties, without a card swap.
        screen.app_instance.console_mcp_tool_count = None
        screen._sync_console_live_work_readiness_rows()
        await pilot.pause()
        assert "Not wired" in str(
            screen.query_one("#console-live-work-source-mcp", Static).renderable
        )


@pytest.mark.asyncio
async def test_a_raising_acp_snapshot_cannot_take_down_the_console_sync_tick():
    """TASK-24704 (Qodo focus area, third round): isolate the tick's read.

    `_sync_console_live_work_readiness_rows` runs on the console sync tick,
    which re-raises on a LIVE screen inside a worker whose `exit_on_error`
    is Textual's default True. `snapshot()` reaches `Popen.poll()` and the
    manager is duck-typed off `app_instance`, so an unguarded raise here is
    not "a card renders stale" -- it is the app exiting, five times a second
    during an active run. The card BUILDER makes the same read and is left
    unguarded on purpose: it runs once, at compose, off the tick.

    Only the ACP row may freeze. It keeps its last known text rather than
    falling back to a status nobody measured (TASK-24601's contract), but
    MCP and RAG come from their own probes and have nothing to do with ACP
    -- a PERSISTENT ACP failure must not stop them updating (Qodo #3).
    """
    async with make_console_pilot() as pilot:
        screen = pilot.app.screen
        await _wait_for_selector(screen, pilot, "#console-inspector-rail-open")
        await click_inspect_rail_open(pilot)

        screen.app_instance.console_mcp_tool_count = 4
        screen._sync_console_live_work_readiness_rows()
        await pilot.pause()
        settled = str(
            screen.query_one("#console-live-work-source-mcp", Static).renderable
        )
        assert settled == "MCP: Connected - 4 tools ready.", settled

        acp_before = str(
            screen.query_one("#console-live-work-source-acp", Static).renderable
        )

        class ExplodingManager:
            """Stands in for a process manager whose runtime died mid-tick."""

            def snapshot(self):
                raise RuntimeError("runtime went away mid-tick")

        screen.app_instance.acp_runtime_process_manager = ExplodingManager()
        # Must not raise -- on the real tick any exception here reaches the
        # worker's error handler and exits the app.
        screen._sync_console_live_work_readiness_rows()
        await pilot.pause()

        assert (
            str(screen.query_one("#console-live-work-source-acp", Static).renderable)
            == acp_before
        ), "a failed ACP read overwrote the row it never measured"

        # ...and the rows that ACP has nothing to do with are still live, so
        # a PERSISTENT failure cannot freeze them.
        screen.app_instance.console_mcp_tool_count = 9
        screen._sync_console_live_work_readiness_rows()
        await pilot.pause()
        refreshed = str(
            screen.query_one("#console-live-work-source-mcp", Static).renderable
        )
        assert refreshed == "MCP: Connected - 9 tools ready.", refreshed


def test_a_probed_empty_mcp_catalog_is_not_reported_as_unprobed():
    """TASK-24704 (Qodo #2): `None` cannot mean "nobody looked" here.

    `_publish_mcp_inspector_counts` documents `(None, None)` as the contract
    for the no-service, kill-switch-on, compose-failed AND empty-catalog
    paths alike, so a probed-but-empty catalog is indistinguishable from an
    absent one. Claiming "Not checked" for both asserted something the data
    does not support, and made "Not wired" unreachable for the ordinary
    zero-tool result.
    """
    from tldw_chatbook.Chat.console_live_work import (
        ConsoleLiveWorkSourceReadinessState,
    )

    def mcp_text(count):
        state = ConsoleLiveWorkSourceReadinessState.from_acp_runtime_status(
            "not_configured", mcp_tool_count=count
        )
        return next(
            row.text
            for row in state.rows
            if row.widget_id == "console-live-work-source-mcp"
        )

    assert mcp_text(None) == "MCP: Not wired - MCP servers."
    assert mcp_text(0) == "MCP: Not wired - MCP servers."
    assert mcp_text(1) == "MCP: Connected - 1 tool ready."
    assert mcp_text(4) == "MCP: Connected - 4 tools ready."


@pytest.mark.asyncio
async def test_rag_row_probes_instead_of_trusting_the_unset_registry_flag(
    monkeypatch,
) -> None:
    """TASK-24704 (Qodo #4): the RAG row must not read an unprobed default.

    `DEPENDENCIES_AVAILABLE["embeddings_rag"]` starts False and is only ever
    populated by `check_embeddings_rag_deps`, which nothing calls
    automatically. Reading it raw therefore reported `RAG: Unavailable` on
    installs where the extras are genuinely present -- an unprobed default
    rendered as a measured negative, which is the exact failure mode this
    whole task exists to remove.

    `embeddings_rag_deps_installed` is the repo's cheap `find_spec` probe,
    documented as safe for render paths.
    """
    from tldw_chatbook.Utils import optional_deps

    async with make_console_pilot() as pilot:
        screen = pilot.app.screen

        # The registry says "no" purely because nobody probed.
        monkeypatch.setitem(
            optional_deps.DEPENDENCIES_AVAILABLE, "embeddings_rag", False
        )
        # ...while the packages are in fact importable.
        monkeypatch.setattr(
            optional_deps, "embeddings_rag_deps_installed", lambda: True
        )
        screen._console_rag_extras_cache = None

        assert screen._console_rag_extras_available() is True, (
            "the screen trusted the unprobed registry default instead of probing"
        )

        # And it reports honestly when the extras really are missing.
        monkeypatch.setattr(
            optional_deps, "embeddings_rag_deps_installed", lambda: False
        )
        screen._console_rag_extras_cache = None
        assert screen._console_rag_extras_available() is False


async def _wait_reveal(pilot, predicate, *, description, stable_polls: int = 1):
    """`_wait_for_right_rail_condition` with room for a loaded run.

    At load 60 the 180x48 auto-open took longer than the default 5 s to hold
    for three polls, though the Agents section was in view by the end.
    """
    await _wait_for_right_rail_condition(
        pilot,
        predicate,
        description=description,
        stable_polls=stable_polls,
        timeout=15.0,
    )


async def _start_fleet_with_staged_tray(
    pilot, screen, *, tray_rows: int, history: bool = False, open_rail: bool = False
):
    """Stage tray rows, then let three running sub-agents reach the fleet.

    Drives the real sync path: `_sync_console_agent_section` derives the rows
    from the fake bridge and runs the section's one-shot auto-open. The
    Environment fetches the rail's opening starts are stubbed out, so the
    only landing a test sees is one it makes. ``history`` gives the bridge
    a run-history page, so the section stays shown ("Run history") after
    its rows end. ``open_rail`` applies the rail state in the same tick as
    the rows, with no pause between. Returns the bridge.
    """
    from Tests.UI.test_console_fleet_panel import _FleetBridge
    from tldw_chatbook.Agents.fleet_coordinator import FleetHandle
    from tldw_chatbook.Chat.console_display_state import ConsoleStagedContextState
    from tldw_chatbook.Widgets.Console.console_staged_context import (
        ConsoleStagedContextTray,
    )

    await _wait_for_console_quiet(pilot, screen)
    environment = screen._console_environment
    environment._workspace_root_accessor = lambda: "/w"  # else it lands "unbound"
    environment._dispatch_local = lambda root: None
    environment._dispatch_net = lambda root, **kwargs: None
    screen.query_one(
        "#console-staged-context-tray", ConsoleStagedContextTray
    ).sync_state(
        ConsoleStagedContextState(
            heading="Sources",
            summary="",
            rows=tuple(
                ConsoleDisplayRow(f"Source {index}", f"value {index}")
                for index in range(tray_rows)
            ),
            source_count=tray_rows,
        )
    )
    bridge = _FleetBridge(
        tuple(
            FleetHandle(
                handle_id=f"h{index}",
                run_id=f"run-h{index}",
                agent="researcher",
                task="find pricing",
                status="running",
                started_at=1000.0,
            )
            for index in range(3)
        )
    )
    if history:
        bridge.subagent_history_page = lambda *args, **kwargs: None
    screen._console_agent_bridge = bridge
    screen._console_agent_drilldown_run_id = None
    screen._character._current_console_rail_conversation_id = lambda: "conv-A"
    screen._agent._console_agent_drilldown_conversation_id = "conv-A"
    await _real_fleet_sync(pilot, screen, open_rail=open_rail)
    return bridge


async def _real_fleet_sync(pilot, screen, *, open_rail: bool = False) -> None:
    screen._console_agent_section_last = None
    screen._sync_console_agent_section()
    if open_rail:
        screen._sync_console_rail_visibility_if_changed(
            screen._current_console_rail_state()
        )
    await pilot.pause()


@pytest.fixture
def frozen_reveal_clock(monkeypatch):
    """Hold the fleet reveal's clock still, so a slow loaded run never
    outlasts the replay window; only the bound test moves it."""
    agent_module = importlib.import_module("tldw_chatbook.UI.Console_Modules.agent")
    monkeypatch.setattr(agent_module, "_fleet_reveal_clock", lambda: 1000.0)
    return agent_module


def _fleet_section_inside_rail_body(screen) -> bool:
    """The Agents section shows: all of it, or, when it is taller than the
    body, as much as the body holds from its top -- what a reveal can show."""
    window = screen.query_one("#console-inspector-rail-body").content_region
    region = screen.query_one("#console-agent-section-subagents").region
    return region.height > 0 and window.contains_region(region.crop_size(window.size))


def _fleet_geometry(screen) -> str:
    section = screen.query_one("#console-agent-section-subagents")
    body = screen.query_one("#console-inspector-rail-body")
    return f"section {section.region}, body {body.content_region}"


def _spy_fleet_scroll_visible(screen) -> list[dict]:
    """Record every `scroll_visible` call on the Agents section."""
    section = screen.query_one("#console-agent-section-subagents")
    calls: list[dict] = []
    original = section.scroll_visible

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    section.scroll_visible = spy
    return calls


async def _settle_refreshes(pilot, count: int = 10) -> None:
    """Wait screen refreshes, not wall time: the reveal retries per refresh.

    Each round also lets queued messages run, because a landing's reveal
    waits behind the sections' rebuilds, which `wait_for_refresh` alone does
    not process.
    """
    for _ in range(count):
        await pilot.pause()
        await pilot.app.screen.wait_for_refresh()


async def _settle_landing(pilot, screen) -> None:
    """Wait until a landing's replay, if any, has run or stood down.

    The replay waits behind the Environment and Tasks rebuilds, then behind
    the Agents section's queue and a refresh. A sentinel queued along the
    same path runs after it, so "never called" is checked only once the
    reveal had its chance -- under load, ten refreshes alone were not enough.
    """
    environment = screen.query_one("#console-environment-section")
    tasks = screen.query_one("#console-tasks-section")
    fleet = screen.query_one("#console-agent-section-subagents")
    done = asyncio.Event()
    environment.call_next(
        tasks.call_next, fleet.call_next, fleet.call_after_refresh, done.set
    )
    await asyncio.wait_for(done.wait(), timeout=15.0)
    await _settle_refreshes(pilot)


def _rich_environment_snapshot():
    """Branch, changed files, a PR with checks and a branch task."""
    from tldw_chatbook.Chat.console_environment_state import (
        BranchTaskState,
        EnvironmentSnapshot,
        EnvSourceAvailability,
        GitEnvState,
        PrCheck,
        PrEnvState,
        TasksEnvState,
    )
    from tldw_chatbook.Workspaces.change_tracking import ChangedFile

    files = tuple(
        ChangedFile(path=f"src/module_{index}.py", status="M", adds=5, dels=2)
        for index in range(3)
    )
    return EnvironmentSnapshot(
        git=GitEnvState(
            availability=EnvSourceAvailability.OK,
            root="/w",
            branch="feat/task-1-inspector-order",
            adds=15,
            dels=6,
            files=files,
        ),
        pr=PrEnvState(
            availability=EnvSourceAvailability.OK,
            number=7,
            title="Inspector order",
            state="OPEN",
            url="https://x/pull/7",
            checks=(
                PrCheck("ci-tests", "failure", "https://ci/1"),
                PrCheck("lint", "success", "https://ci/2"),
                PrCheck("docs", "pending", "https://ci/3"),
            ),
        ),
        tasks=TasksEnvState(
            availability=EnvSourceAvailability.OK,
            branch_task=BranchTaskState(
                task_id="1",
                title="Inspector order",
                status="In Progress",
                path="backlog/tasks/task-1.md",
            ),
        ),
    )


def _lean_environment_snapshot():
    """A branch and nothing else: the sections above shrink back."""
    from tldw_chatbook.Chat.console_environment_state import (
        EnvironmentSnapshot,
        EnvSourceAvailability,
        GitEnvState,
    )

    return EnvironmentSnapshot(
        git=GitEnvState(
            availability=EnvSourceAvailability.OK,
            root="/w",
            branch="feat/task-1-inspector-order",
        )
    )


def _land(screen, snapshot=None) -> None:
    """Land an Environment snapshot the way the controller's tiers do."""
    snapshot = snapshot or _rich_environment_snapshot()
    screen._console_environment.snapshot = snapshot
    screen._land_console_environment(snapshot)


async def _auto_open_with_fleet(pilot, screen, *, tray_rows: int) -> None:
    await _start_fleet_with_staged_tray(pilot, screen, tray_rows=tray_rows)
    assert screen.query_one("#console-agent-section-subagents").open
    screen._sync_console_rail_visibility_if_changed(
        screen._current_console_rail_state()
    )
    await _wait_reveal(
        pilot,
        lambda: _fleet_section_inside_rail_body(screen),
        description=lambda: (
            f"the auto-opened rail to show the Agents section: {_fleet_geometry(screen)}"
        ),
        stable_polls=3,
    )
    assert _right_rail_open(pilot)
    assert screen._last_console_rail_state.preferred_right_open is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("size", "tray_rows"),
    [((150, 40), 6), ((180, 48), 12), ((150, 40), 20), ((190, 40), 20)],
)
async def test_fleet_auto_open_scrolls_the_agents_section_into_view(
    size, tray_rows, frozen_reveal_clock
):
    """TASK-41959 review: the fleet's rail auto-open must show the fleet.

    Agents now sits below the Sources tray, Library search and Scope. Its
    one-shot expand ran while the rail was still hidden, so the section's
    `scroll_on_expand` reveal was lost and the auto-opened rail showed no
    fleet: at 150x40 (the narrowest auto-open width) the header sat below
    the rail body's fold, and at 180x48 a 12-row tray pushed it there too.
    With 20 staged sources the opening then shrank the list to its cap and
    the section rose above the top (y=1, body 15-31) until the reveal waited
    for that re-fit. At 190x40 that wait ran out while the re-fit was still
    pending: most of its 20 attempts came while paints were held, which no
    longer count. So the scroll must come once nothing is settling.
    """
    async with make_console_pilot(size=size) as pilot:
        screen = pilot.app.screen
        assert not _right_rail_open(pilot)
        await _start_fleet_with_staged_tray(pilot, screen, tray_rows=tray_rows)
        body = screen.query_one("#console-inspector-rail-body")
        section = screen.query_one("#console-agent-section-subagents")
        settling_at_scroll: list[bool] = []
        original = section.scroll_visible

        def spy(*args, **kwargs):
            settling_at_scroll.append(body.layout_settling())
            return original(*args, **kwargs)

        section.scroll_visible = spy
        screen._sync_console_rail_visibility_if_changed(
            screen._current_console_rail_state()
        )
        await _wait_reveal(
            pilot,
            lambda: _right_rail_open(pilot) and _fleet_section_inside_rail_body(screen),
            description=lambda: (
                f"the auto-opened rail to show the Agents section: {_fleet_geometry(screen)}"
            ),
            stable_polls=3,
        )
        assert settling_at_scroll and not any(settling_at_scroll), (
            f"the reveal scrolled while the rail was still settling: {settling_at_scroll}"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "land_during_replay", [False, True], ids=["resize", "environment-during-resize"]
)
async def test_fleet_reveal_rechecks_the_committed_viewport(
    frozen_reveal_clock, land_during_replay
):
    """Keep the fleet visible across a committed resize and an overlapping landing."""
    async with make_console_pilot(size=(180, 50)) as pilot:
        screen = pilot.app.screen
        section = screen.query_one("#console-agent-section-subagents")
        original = section.scroll_visible
        resize_queued = False
        resize_task = None
        scrolls = 0
        landing_delivered = False

        def resize():
            nonlocal resize_task
            resize_task = asyncio.create_task(pilot.resize_terminal(180, 48))

        def scroll(*args, **kwargs):
            nonlocal resize_queued, scrolls
            on_complete = kwargs.get("on_complete")
            if on_complete is not None:
                scrolls += 1
                if land_during_replay and scrolls == 2:

                    def complete_with_landing():
                        nonlocal landing_delivered
                        if not landing_delivered:
                            landing_delivered = True
                            _land(screen)
                        on_complete()

                    kwargs["on_complete"] = complete_with_landing
            result = original(*args, **kwargs)
            if not resize_queued and kwargs.get("on_complete"):
                resize_queued = True
                section.call_next(resize)
            return result

        section.scroll_visible = scroll
        try:
            await _start_fleet_with_staged_tray(
                pilot, screen, tray_rows=12, open_rail=True
            )
            await _wait_reveal(
                pilot,
                lambda: (
                    pilot.app.size.height == 48
                    and _right_rail_open(pilot)
                    and _fleet_section_inside_rail_body(screen)
                ),
                description=lambda: (
                    f"Agents in the committed viewport: {_fleet_geometry(screen)}"
                ),
                stable_polls=3,
            )
            assert resize_task is not None
            assert screen._last_console_rail_state.preferred_right_open is False
            if land_during_replay:
                assert landing_delivered
                await _settle_landing(pilot, screen)
                assert _fleet_section_inside_rail_body(screen), _fleet_geometry(screen)
                assert scrolls == 1 + frozen_reveal_clock.FLEET_REVEAL_REPLAY_LIMIT
        finally:
            if resize_task is not None:
                await resize_task


@pytest.mark.asyncio
@pytest.mark.parametrize(("size", "tray_rows"), [((150, 40), 6), ((180, 48), 12)])
async def test_fleet_reveal_survives_the_first_environment_landing(
    size, tray_rows, frozen_reveal_clock
):
    """The rail's opening starts the Environment fetch; its landing grew the
    sections above and pushed the fleet's rows and "View all runs" back below
    the fold (150x40: header 21 -> 30, body 15-31). A landing soon after the
    reveal redoes it."""
    async with make_console_pilot(size=size) as pilot:
        screen = pilot.app.screen
        await _auto_open_with_fleet(pilot, screen, tray_rows=tray_rows)

        _land(screen)
        await _settle_refreshes(pilot)
        await _wait_reveal(
            pilot,
            lambda: _fleet_section_inside_rail_body(screen),
            description=lambda: (
                f"the fleet to stay in view after the landing: {_fleet_geometry(screen)}"
            ),
            stable_polls=5,
        )


@pytest.mark.asyncio
async def test_environment_landing_keeps_a_scroll_the_user_made(frozen_reveal_clock):
    """Control: once the user scrolls the body, the landing leaves it alone."""
    async with make_console_pilot(size=(150, 40)) as pilot:
        screen = pilot.app.screen
        await _auto_open_with_fleet(pilot, screen, tray_rows=6)
        body = screen.query_one("#console-inspector-rail-body")
        body.scroll_to(y=0, animate=False)
        await _wait_reveal(
            pilot, lambda: body.scroll_y == 0, description="the user's scroll"
        )
        calls = _spy_fleet_scroll_visible(screen)

        _land(screen)
        await _settle_landing(pilot, screen)

        assert calls == []
        assert body.scroll_y == 0


@pytest.mark.asyncio
async def test_band_open_after_context_closes_reveals_running_sub_agents(
    frozen_reveal_clock,
):
    """Any automatic opening reveals running sub-agents, not only the fleet's.

    At 124 columns the window is too narrow for both rails: the open Context
    rail keeps the Inspector from opening itself, so the section expands
    behind a hidden rail. Closing Context lets the Inspector open -- unasked
    -- and it shows the fleet.
    """
    async with make_console_pilot(size=(124, 40)) as pilot:
        screen = pilot.app.screen
        await _start_fleet_with_staged_tray(pilot, screen, tray_rows=6)
        screen._sync_console_rail_visibility_if_changed(
            screen._current_console_rail_state()
        )
        await pilot.pause()
        assert not _right_rail_open(pilot)

        screen._set_console_rail_preference(left_open=False)
        await _wait_reveal(
            pilot,
            lambda: _right_rail_open(pilot) and _fleet_section_inside_rail_body(screen),
            description=lambda: (
                f"the band-opened rail to show the fleet: {_fleet_geometry(screen)}"
            ),
            stable_polls=3,
        )
        assert screen._last_console_rail_state.preferred_right_open is False


@pytest.mark.asyncio
async def test_manual_inspector_open_keeps_its_scroll_with_fleet_rows(
    frozen_reveal_clock,
):
    """The reveal belongs to automatic opens; a rail the user opens is theirs.

    At 140 columns the fleet auto-open is declined, so the section expands
    behind a closed rail. Opening the rail with Alt+I's action must not
    scroll it: `scroll_visible` is never called, checked after the refreshes
    the reveal would have used rather than after a fixed pause.
    """
    async with make_console_pilot(size=(140, 40)) as pilot:
        screen = pilot.app.screen
        await _start_fleet_with_staged_tray(pilot, screen, tray_rows=6)
        screen._sync_console_rail_visibility_if_changed(
            screen._current_console_rail_state()
        )
        await pilot.pause()
        assert not _right_rail_open(pilot)
        calls = _spy_fleet_scroll_visible(screen)

        await screen.action_toggle_console_inspector_rail()
        body = screen.query_one("#console-inspector-rail-body")
        section = screen.query_one("#console-agent-section-subagents")
        await _wait_reveal(
            pilot,
            lambda: _right_rail_open(pilot) and section.region.height > 0,
            description="the manually opened rail to lay out the Agents section",
        )
        await _settle_refreshes(pilot)
        assert calls == [], "the user's open scrolled to the Agents section"
        assert body.scroll_y == 0
        assert not _fleet_section_inside_rail_body(screen), (
            "precondition: Agents must start below the fold for this to test a jump"
        )


@pytest.mark.asyncio
async def test_reveal_is_dropped_when_the_sub_agents_finish_first(frozen_reveal_clock):
    """Rows that end before the rail opens leave nothing to reveal.

    The section stays shown for its "Run history" summary, so a stale reveal
    would still scroll to it when the rail later opens itself.
    """
    async with make_console_pilot(size=(124, 40)) as pilot:
        screen = pilot.app.screen
        bridge = await _start_fleet_with_staged_tray(
            pilot, screen, tray_rows=6, history=True
        )
        screen._sync_console_rail_visibility_if_changed(
            screen._current_console_rail_state()
        )
        await pilot.pause()
        assert not _right_rail_open(pilot)

        bridge._handles = []
        await _real_fleet_sync(pilot, screen)
        section = screen.query_one("#console-agent-section-subagents")
        assert not section.rows and section.summary == "Run history"
        calls = _spy_fleet_scroll_visible(screen)

        # Closing Context lets the Inspector open automatically at 124 columns.
        state = screen._set_console_rail_preference(left_open=False)
        assert state.right_open
        assert state.preferred_right_open is False
        body = screen.query_one("#console-inspector-rail-body")
        await _wait_reveal(
            pilot,
            lambda: _right_rail_open(pilot) and section.region.height > 0,
            description="the opened rail to lay out the Agents section",
        )
        await _settle_refreshes(pilot)
        assert calls == []
        assert body.scroll_y == 0
        assert not _fleet_section_inside_rail_body(screen), (
            "precondition: the section must start below the fold"
        )


@pytest.mark.asyncio
async def test_environment_scroll_inside_the_rebuild_window_is_kept(
    frozen_reveal_clock,
):
    """A scroll made after a landing but before its reveal runs is the user's.

    The reveal waits behind the landed sections' rebuilds; it compares the
    body's offset with the one the last reveal left, not a fresh reading.
    """
    async with make_console_pilot(size=(150, 40)) as pilot:
        screen = pilot.app.screen
        await _auto_open_with_fleet(pilot, screen, tray_rows=6)
        body = screen.query_one("#console-inspector-rail-body")
        calls = _spy_fleet_scroll_visible(screen)

        _land(screen)
        body.scroll_to(y=0, animate=False, immediate=True)  # same tick
        await _settle_landing(pilot, screen)

        assert calls == []
        assert body.scroll_y == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "gesture",
    [
        "row",
        "row-in-flight",
        "environment-collapse",
        "agents-collapse",
        "focus",
        "mouse-down",
    ],
)
async def test_a_gesture_in_the_rail_stops_the_reveal_for_good(
    gesture, frozen_reveal_clock
):
    """Expanding a row, toggling a section, focusing or pressing inside the
    rail body makes the rail the user's.

    The row case: Enter on the Changes row re-lands the Environment at once,
    and the replay that followed moved the body and the focused row out of
    view (200x70 review probe).
    """
    from tldw_chatbook.Chat.console_environment_state import ENV_ROW_CHANGES

    async with make_console_pilot(size=(150, 40)) as pilot:
        screen = pilot.app.screen
        await _auto_open_with_fleet(pilot, screen, tray_rows=6)
        if gesture.startswith("row"):  # the Changes row needs a landed snapshot
            _land(screen)
            if gesture == "row":  # else the row beats that landing's reveal
                await _settle_refreshes(pilot)
        calls = _spy_fleet_scroll_visible(screen)

        if gesture.startswith("row"):
            screen._handle_console_environment_row("environment", ENV_ROW_CHANGES)
        elif gesture in ("focus", "mouse-down"):
            if gesture == "focus":  # a fleet row already in view: no scroll
                fleet = screen.query_one("#console-agent-section-subagents")
                next(node for node in fleet.query("*") if node.focusable).focus()
            else:  # a press on the Scope row, which takes no focus
                await pilot.mouse_down("#console-retrieval-scope-row")
            await _settle_refreshes(pilot, count=3)
            _land(screen)
        else:
            selector = {
                "environment-collapse": "#console-environment-section",
                "agents-collapse": "#console-agent-section-subagents",
            }[gesture]
            screen.query_one(selector).set_open(False)
            await _settle_refreshes(pilot, count=3)
            _land(screen)
        await _settle_landing(pilot, screen)

        assert calls == []
        # A landing that leaves the fleet in view scrolls nothing either way;
        # the stand-down shows as no replay armed for later landings.
        assert screen._agent._fleet_replay is None, "the rail is the user's now"


@pytest.mark.asyncio
async def test_a_rail_reopened_by_hand_is_not_replayed(frozen_reveal_clock):
    """Closing and reopening the Inspector by hand leaves its scroll alone."""
    async with make_console_pilot(size=(150, 40)) as pilot:
        screen = pilot.app.screen
        await _auto_open_with_fleet(pilot, screen, tray_rows=6)
        calls = _spy_fleet_scroll_visible(screen)

        screen._set_console_rail_preference(right_open=False)
        await _settle_refreshes(pilot, count=3)
        screen._set_console_rail_preference(right_open=True)  # lands at once
        await _settle_landing(pilot, screen)
        _land(screen)
        await _settle_landing(pilot, screen)

        assert screen._last_console_rail_state.preferred_right_open is True
        assert calls == []


@pytest.mark.asyncio
async def test_replays_are_bounded_and_end_with_the_window(
    frozen_reveal_clock, monkeypatch
):
    """At most FLEET_REVEAL_REPLAY_LIMIT replays scroll, and none after the window.

    Rich and lean landings alternate. Rich #0 pushes the fleet below the fold
    and is replayed; lean #1 shrinks the rows above, the fleet rises above the
    top, and that is replayed too; rich #2 pushes it out again with the budget
    spent, so it stays out. The every-landing replay of round 2 scrolled once
    per identical landing (5 of 5).
    """
    agent_module = frozen_reveal_clock
    async with make_console_pilot(size=(150, 40)) as pilot:
        screen = pilot.app.screen
        await _auto_open_with_fleet(pilot, screen, tray_rows=6)
        calls = _spy_fleet_scroll_visible(screen)
        for index in range(5):
            _land(screen, None if index % 2 == 0 else _lean_environment_snapshot())
            await _settle_landing(pilot, screen)
        assert len(calls) == agent_module.FLEET_REVEAL_REPLAY_LIMIT
        assert not _fleet_section_inside_rail_body(screen), (
            "rich #2 came after the budget was spent and must not be replayed"
        )

    async with make_console_pilot(size=(150, 40)) as pilot:
        screen = pilot.app.screen
        await _auto_open_with_fleet(pilot, screen, tray_rows=6)
        calls = _spy_fleet_scroll_visible(screen)
        after = 1000.0 + agent_module.FLEET_REVEAL_REPLAY_SECONDS + 1.0
        monkeypatch.setattr(agent_module, "_fleet_reveal_clock", lambda: after)
        _land(screen)
        await _settle_landing(pilot, screen)
        assert calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(150, 40), (150, 30)])
async def test_landings_that_change_nothing_spend_no_replay(size, frozen_reveal_clock):
    """No-op landings -- the poll, a focus regain, a TTL re-land -- cost nothing.

    Round 3 spent a replay on every landing: two unchanged ones used the
    budget, and the PR tier that followed left the fleet below the fold. At
    150x30 the section is taller than the body; round 4's whole-section test
    read it as out of view and spent the budget the same way.
    """
    async with make_console_pilot(size=size) as pilot:
        screen = pilot.app.screen
        await _auto_open_with_fleet(pilot, screen, tray_rows=6)
        unchanged = screen._console_environment.snapshot
        for _ in range(2):
            _land(screen, unchanged)
            await _settle_landing(pilot, screen)
        _land(screen)  # the rich PR-and-checks tier
        await _settle_landing(pilot, screen)
        await _wait_reveal(
            pilot,
            lambda: _fleet_section_inside_rail_body(screen),
            description=lambda: (
                f"the fleet after the PR tier: {_fleet_geometry(screen)}"
            ),
            stable_polls=3,
        )


@pytest.mark.asyncio
async def test_a_landing_while_hidden_is_revealed_once_laid_out(frozen_reveal_clock):
    """A landing that arrives before the rail opens is laid out by the opening.

    The opening rebuilds inside `batch_update`, where after-refresh callbacks
    run without a layout, so the reveal finds no region at first and retries.
    """
    async with make_console_pilot(size=(140, 40)) as pilot:
        screen = pilot.app.screen
        await _start_fleet_with_staged_tray(pilot, screen, tray_rows=6)
        assert not _right_rail_open(pilot)
        _land(screen)
        await pilot.pause()
        assert not _right_rail_open(pilot)

        await pilot.resize_terminal(150, 40)
        screen._sync_console_rail_visibility_if_changed(
            screen._current_console_rail_state()
        )
        await _wait_reveal(
            pilot,
            lambda: _right_rail_open(pilot) and _fleet_section_inside_rail_body(screen),
            description=lambda: (
                f"the fleet after a hidden landing: {_fleet_geometry(screen)}"
            ),
            stable_polls=3,
        )


@pytest.mark.asyncio
async def test_rows_and_the_rail_opening_in_one_tick(frozen_reveal_clock):
    """The fleet's own open is not a user's collapse toggle, even when the
    rail opens in the same tick as the rows (no pause between them)."""
    async with make_console_pilot(size=(150, 40)) as pilot:
        screen = pilot.app.screen
        await _start_fleet_with_staged_tray(pilot, screen, tray_rows=6, open_rail=True)
        await _wait_reveal(
            pilot,
            lambda: _right_rail_open(pilot) and _fleet_section_inside_rail_body(screen),
            description=lambda: f"same-tick open: {_fleet_geometry(screen)}",
            stable_polls=3,
        )
        _land(screen)
        await _settle_landing(pilot, screen)
        await _wait_reveal(
            pilot,
            lambda: _fleet_section_inside_rail_body(screen),
            description=lambda: (
                f"same-tick open, then a landing: {_fleet_geometry(screen)}"
            ),
            stable_polls=3,
        )


@pytest.mark.asyncio
async def test_one_reveal_arms_the_replay_once(frozen_reveal_clock):
    """`scroll_to_widget` hands `on_complete` to every scrolling ancestor; a
    late duplicate re-armed a replay a landing had already used (6 arms per
    reveal in the review probe)."""
    async with make_console_pilot(size=(150, 40)) as pilot:
        screen = pilot.app.screen
        agent = screen._agent
        arms: list[object] = []

        class CountingArms(type(agent)):  # type: ignore[misc]
            @property
            def _fleet_replay(self):
                return self.__dict__.get("_counted_replay")

            @_fleet_replay.setter
            def _fleet_replay(self, value):
                if value is not None:
                    arms.append(value)
                self.__dict__["_counted_replay"] = value

        agent.__dict__["_counted_replay"] = agent.__dict__.pop("_fleet_replay", None)
        agent.__class__ = CountingArms
        await _auto_open_with_fleet(pilot, screen, tray_rows=6)
        await _settle_refreshes(pilot)
        assert len(arms) == 1

        arms.clear()
        _land(screen)
        await _settle_landing(pilot, screen)
        assert len(arms) == 1


@pytest.mark.asyncio
async def test_a_landing_after_a_manual_close_schedules_nothing(frozen_reveal_clock):
    """A rail the user closed gets no reveal queued for its hidden body."""
    async with make_console_pilot(size=(150, 40)) as pilot:
        screen = pilot.app.screen
        await _auto_open_with_fleet(pilot, screen, tray_rows=6)
        screen._set_console_rail_preference(right_open=False)
        await _settle_refreshes(pilot, count=3)
        assert not _right_rail_open(pilot)
        scheduled: list[object] = []
        agent = screen._agent
        original = agent._schedule_fleet_reveal
        agent._schedule_fleet_reveal = lambda reveal: (
            scheduled.append(reveal),
            original(reveal),
        )
        _land(screen)
        await _settle_landing(pilot, screen)
        assert scheduled == []


#: Enough staged sources that the Sources list hits its cap and scrolls
#: (at 150x40 it scrolls by 20 rows; 12 scrolls by only 4).
CAPPED_TRAY_ROWS = 20
#: At 160x70 this many keeps the capped list (scrolling by 60) on screen
#: beside the revealed fleet.
VISIBLE_CAPPED_TRAY_ROWS = 40


def _sources_viewport(screen):
    from tldw_chatbook.Widgets.Console.console_bounded_section import (
        ConsoleBoundedSection,
    )

    return screen.query_one(
        "#console-bounded-section-sources", ConsoleBoundedSection
    ).viewport


@pytest.mark.asyncio
async def test_a_scroll_inside_a_rail_scroll_area_stops_the_reveal(frozen_reveal_clock):
    """A wheel over the Sources tray's capped list moves no body offset and
    posts no press or focus; it still makes the rail the user's.

    At 160x70 with 40 sources the capped list (y 20-40) and the revealed fleet
    are both on screen after the opening, so the wheel lands where a user's
    would.
    """
    from textual import events

    async with make_console_pilot(size=(160, 70)) as pilot:
        screen = pilot.app.screen
        await _auto_open_with_fleet(pilot, screen, tray_rows=VISIBLE_CAPPED_TRAY_ROWS)
        viewport = _sources_viewport(screen)
        window = screen.query_one("#console-inspector-rail-body").content_region
        assert viewport.max_scroll_y > 0, "precondition: the Sources list is capped"
        assert window.contains_region(viewport.region), (
            "precondition: the Sources list is on screen"
        )
        calls = _spy_fleet_scroll_visible(screen)

        viewport.post_message(
            events.MouseScrollDown(viewport, 1, 1, 0, 1, 0, False, False, False)
        )
        await _wait_reveal(
            pilot, lambda: viewport.scroll_y > 0, description="the wheel's scroll"
        )
        _land(screen)
        await _settle_landing(pilot, screen)

        assert calls == []
        assert screen._agent._fleet_replay is None, "the rail is the user's now"


@pytest.mark.asyncio
async def test_a_landing_rebuild_is_not_an_inner_scroll(frozen_reveal_clock):
    """No false stand-down when a rebuild replaces an inner scroll area.

    Restaging the Sources tray rebuilds it: the capped list the opening
    snapshotted is replaced by a new widget. The old one keeps the offset it
    had, so it reads as unmoved, and the rich landing is still replayed.
    """
    from tldw_chatbook.Chat.console_display_state import ConsoleStagedContextState
    from tldw_chatbook.Widgets.Console.console_staged_context import (
        ConsoleStagedContextTray,
    )

    async with make_console_pilot(size=(150, 40)) as pilot:
        screen = pilot.app.screen
        await _auto_open_with_fleet(pilot, screen, tray_rows=CAPPED_TRAY_ROWS)
        snapshotted = _sources_viewport(screen)
        assert snapshotted.max_scroll_y > 0
        screen.query_one(
            "#console-staged-context-tray", ConsoleStagedContextTray
        ).sync_state(
            ConsoleStagedContextState(
                heading="Sources",
                summary="",
                rows=tuple(
                    ConsoleDisplayRow(f"Restaged {index}", f"value {index}")
                    for index in range(CAPPED_TRAY_ROWS)
                ),
                source_count=CAPPED_TRAY_ROWS,
            )
        )
        await _wait_reveal(
            pilot,
            lambda: _sources_viewport(screen) is not snapshotted,
            description="the restaged tray to replace its capped list",
        )
        await _settle_refreshes(pilot, count=5)
        _land(screen)
        await _settle_landing(pilot, screen)
        await _wait_reveal(
            pilot,
            lambda: _fleet_section_inside_rail_body(screen),
            description=lambda: (
                f"the fleet after the landing: {_fleet_geometry(screen)}"
            ),
            stable_polls=3,
        )


@pytest.mark.asyncio
async def test_rows_ending_as_the_rail_opens_leave_nothing_armed(frozen_reveal_clock):
    """The rows end in the same tick the Inspector opens itself (124 columns):
    the reveal in flight must not arm a replay for a fleet that is gone."""
    async with make_console_pilot(size=(124, 40)) as pilot:
        screen = pilot.app.screen
        bridge = await _start_fleet_with_staged_tray(
            pilot, screen, tray_rows=6, history=True
        )
        screen._sync_console_rail_visibility_if_changed(
            screen._current_console_rail_state()
        )
        await pilot.pause()
        assert not _right_rail_open(pilot)

        screen._set_console_rail_preference(left_open=False)
        bridge._handles = []
        screen._console_agent_section_last = None
        screen._sync_console_agent_section()
        section = screen.query_one("#console-agent-section-subagents")
        await _wait_reveal(
            pilot,
            lambda: _right_rail_open(pilot) and section.region.height > 0,
            description="the rail to open on the fleet's run history",
        )
        await _settle_refreshes(pilot)
        calls = _spy_fleet_scroll_visible(screen)
        _land(screen)
        await _settle_landing(pilot, screen)

        assert calls == []


@pytest.mark.asyncio
async def test_a_section_opened_in_view_owes_no_later_reveal(frozen_reveal_clock):
    """Rows that arrive while the rail is open need no reveal later.

    The user opened the rail, the fleet section opened in view, the user
    closed the rail; a later automatic opening must not scroll to Agents.
    """
    async with make_console_pilot(size=(150, 40)) as pilot:
        screen = pilot.app.screen
        await _wait_for_console_quiet(pilot, screen)
        screen._set_console_rail_preference(right_open=True)
        await _wait_reveal(
            pilot, lambda: _right_rail_open(pilot), description="the user's open"
        )
        await _start_fleet_with_staged_tray(pilot, screen, tray_rows=6)
        assert screen.query_one("#console-agent-section-subagents").open
        screen._set_console_rail_preference(right_open=False)
        await _settle_refreshes(pilot, count=3)
        calls = _spy_fleet_scroll_visible(screen)

        state = replace(screen._current_console_rail_state(), right_open=True)
        assert state.preferred_right_open is False
        screen._sync_console_rail_visibility_if_changed(state)
        section = screen.query_one("#console-agent-section-subagents")
        await _wait_reveal(
            pilot,
            lambda: _right_rail_open(pilot) and section.region.height > 0,
            description="the rail to open",
        )
        await _settle_refreshes(pilot)
        assert calls == []


@pytest.mark.asyncio
async def test_a_cancel_beats_the_openings_late_on_complete(frozen_reveal_clock):
    """`on_complete` can arrive after the user acted: it must not re-arm."""
    async with make_console_pilot(size=(150, 40)) as pilot:
        screen = pilot.app.screen
        await _start_fleet_with_staged_tray(pilot, screen, tray_rows=6)
        section = screen.query_one("#console-agent-section-subagents")
        late: list = []
        section.scroll_visible = lambda *args, **kwargs: late.append(
            kwargs["on_complete"]
        )
        screen._sync_console_rail_visibility_if_changed(
            screen._current_console_rail_state()
        )
        await _wait_reveal(pilot, lambda: bool(late), description="the reveal's scroll")

        screen._agent.cancel_fleet_reveal()
        late[0]()
        assert screen._agent._fleet_replay is None


@pytest.mark.asyncio
async def test_a_reveal_whose_rail_closed_again_shows_nothing(frozen_reveal_clock):
    """The rail closes before the reveal's layout: the wait ends with no
    region, and nothing is scrolled."""
    async with make_console_pilot(size=(150, 40)) as pilot:
        screen = pilot.app.screen
        await _start_fleet_with_staged_tray(pilot, screen, tray_rows=6)
        calls = _spy_fleet_scroll_visible(screen)
        screen._sync_console_rail_visibility_if_changed(
            screen._current_console_rail_state()
        )
        screen._set_console_rail_preference(right_open=False)  # same tick
        await _settle_refreshes(pilot, count=30)
        assert not _right_rail_open(pilot)
        assert calls == []
        assert screen._agent._fleet_replay is None, "a hidden fleet was armed"


@pytest.mark.asyncio
async def test_a_paint_hold_that_never_ends_cannot_stall_the_reveal(
    frozen_reveal_clock, monkeypatch
):
    """Attempts made while paints are held are free only up to the hold cap."""
    agent_module = frozen_reveal_clock
    monkeypatch.setattr(agent_module, "_paints_held", lambda app: True)
    monkeypatch.setattr(agent_module, "FLEET_REVEAL_PAINT_HOLD_CAP_SECONDS", 0.5)
    async with make_console_pilot(size=(150, 40)) as pilot:
        screen = pilot.app.screen
        await _start_fleet_with_staged_tray(pilot, screen, tray_rows=6)
        body = screen.query_one("#console-inspector-rail-body")
        monkeypatch.setattr(body, "layout_settling", lambda: True)  # never settles
        calls = _spy_fleet_scroll_visible(screen)
        screen._sync_console_rail_visibility_if_changed(
            screen._current_console_rail_state()
        )
        await _wait_reveal(
            pilot,
            lambda: bool(calls),
            description="the reveal to give up waiting and scroll",
        )


@pytest.mark.asyncio
async def test_a_paint_hold_is_waited_out_on_a_timer_not_a_spin(
    frozen_reveal_clock, monkeypatch
):
    """Under a real `batch_update` hold, `Screen._on_idle` runs after-refresh
    callbacks at once, so re-queuing the wait with `call_after_refresh` spun
    the loop for the whole hold (2,549 runs in this 0.5 s hold). It re-checks
    on a short timer instead, and the reveal still lands once the hold ends."""
    agent_module = frozen_reveal_clock
    held_runs: list[bool] = []
    real = agent_module._paints_held

    def counting(app) -> bool:
        held = real(app)
        held_runs.append(held)
        return held

    monkeypatch.setattr(agent_module, "_paints_held", counting)
    async with make_console_pilot(size=(190, 40)) as pilot:
        screen = pilot.app.screen
        await _start_fleet_with_staged_tray(pilot, screen, tray_rows=CAPPED_TRAY_ROWS)
        app = pilot.app
        app._begin_batch()  # a hold like `hold_paints`', ended after 0.5 s
        asyncio.get_running_loop().call_later(0.5, app._end_batch)
        screen._sync_console_rail_visibility_if_changed(
            screen._current_console_rail_state()
        )
        await _wait_reveal(
            pilot,
            lambda: _right_rail_open(pilot) and _fleet_section_inside_rail_body(screen),
            description=lambda: f"the fleet after the hold: {_fleet_geometry(screen)}",
            stable_polls=3,
        )
        assert 0 < sum(held_runs) <= 60, f"{sum(held_runs)} wait runs during the hold"
