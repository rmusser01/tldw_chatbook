"""Console Inspector access below the 150-col compact collapse (TASK-2154.2).

Regression coverage for UX-review findings LY-11 and DS-06
(Docs/superpowers/qa/console-ux-review-2026-08/console-ux-review.md):

- LY-11: below 150 cols the Inspector was force-collapsed and clicking its
  handle silently persisted ``right_open=True`` with zero visual change,
  leaving staged Sources/scope/run inspector unreachable. The force rule is
  now the responsive DEFAULT: an explicit toggle is honored at any width,
  with the main column's min-width waived so the grid always resolves.
- DS-06: the Sources/Tools chips were focusable but inert. Both now open
  the Inspector rail -- at narrow widths they are the ONLY route to it
  (handles hide in single-pane mode).

Both rails' manual toggles must always produce visible feedback.
"""

from __future__ import annotations

from copy import deepcopy
import json
import threading
from unittest.mock import AsyncMock

import pytest
from textual.widgets import Button

from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_destination_shells import _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.Chat.console_rail_state import (
    ConsoleRailPreferences,
    build_console_rail_preference_key,
    serialize_console_rail_preferences,
)
import tldw_chatbook.UI.Screens.chat_screen as chat_screen_module


def _stored_rail_preferences(app) -> dict:
    """Return the single stored rail-preference payload for the test workspace."""
    rail_state_config = app.app_config.get("console", {}).get("rail_state", {})
    assert len(rail_state_config) == 1
    return next(iter(rail_state_config.values()))


@pytest.mark.asyncio
async def test_standard_width_keeps_context_and_preserves_its_preference():
    """At 120 columns Context stays, and no rendering rewrites its preference.

    TASK-23197 changed the first half of this. The Inspector used to open
    ITSELF between 118 and 128 columns, which tripped priority resolution and
    evicted Context -- a one-column resize from 117 to 118 swapped which
    sidebar the user had, unasked and unexplained. The automatic open is now
    declined when it would evict a visible Context rail.

    The second half is unchanged and is why this test still exists: whatever
    the renderer decides about visibility, it must never write that decision
    back into the stored preference payload.
    """
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(120, 40)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-workspace-grid")
        await pilot.pause(0.2)

        shared_key = build_console_rail_preference_key(layout_scope="global")
        stored = app.app_config["console"]["rail_state"][shared_key.value]
        before_render = json.dumps(stored, sort_keys=True, separators=(",", ":"))

        rail_state = console._current_console_rail_state(available_columns=120)

        assert rail_state.left_open is True
        assert rail_state.right_open is False
        assert console.query_one("#console-left-rail").display is True
        assert console.query_one("#console-right-rail").display is False
        assert console.query_one("#console-context-rail-handle").display is False
        # The zero min-width here used to be the compact-override waiver that
        # made a three-pane 120-column layout solvable at all. With the
        # automatic open declined there are only two panes plus a handle, so
        # the transcript keeps its ordinary minimum and the row still fits.
        grid = console.query_one("#console-workspace-grid")
        for pane_id in (
            "console-left-rail",
            "console-main-column",
            "console-inspector-rail-handle",
        ):
            pane = console.query_one(f"#{pane_id}")
            assert pane.display, f"{pane_id} is not displayed at 120 columns"
            assert grid.region.contains_region(pane.region), (
                f"{pane_id} escaped the workspace grid: {pane.region} "
                f"vs {grid.region}"
            )
        assert stored["left_open"] is True
        assert "right_open" not in stored
        assert list(app.app_config["console"]["rail_state"]) == [shared_key.value]
        assert (
            json.dumps(stored, sort_keys=True, separators=(",", ":")) == before_render
        )


@pytest.mark.asyncio
async def test_context_reveal_switches_from_inspector_at_120_columns():
    """The Context handle persists an exact Inspector-to-Context switch."""
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(120, 40)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-context-rail-open")
        await pilot.pause(0.2)
        console._set_console_rail_preference(right_open=True)
        await pilot.pause(0.2)

        assert console.query_one("#console-left-rail").display is False
        assert console.query_one("#console-right-rail").display is True

        console.query_one("#console-context-rail-open", Button).press()
        await pilot.pause()
        await pilot.pause(0.2)

        assert console.query_one("#console-left-rail").display is True
        assert console.query_one("#console-right-rail").display is False
        stored = _stored_rail_preferences(app)
        assert stored["left_open"] is True
        assert stored["right_open"] is False


@pytest.mark.asyncio
async def test_visible_attach_context_action_switches_rails_without_file_picker():
    """Only the visible Workbench action reveals Context at 120 columns."""
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(120, 40)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-control-attach-context")
        await pilot.pause(0.2)
        file_picker = AsyncMock()
        console._handle_console_attach_context = file_picker
        console._set_console_rail_preference(right_open=True)
        await pilot.pause(0.2)

        assert not list(console.query("#console-attach-context"))
        assert not list(console.query("#console-staged-context-attach"))
        assert console.query_one("#console-control-attach-context").display is True

        await pilot.click("#console-control-attach-context")
        await pilot.pause(0.2)

        assert console.query_one("#console-left-rail").display is True
        assert console.query_one("#console-right-rail").display is False
        stored = _stored_rail_preferences(app)
        assert stored["left_open"] is True
        assert stored["right_open"] is False
        file_picker.assert_not_awaited()


@pytest.mark.asyncio
async def test_inspector_handle_opens_and_collapses_rail_at_140_cols():
    """LY-11: at 140 cols the Inspector handle visibly opens the rail (and
    the in-rail collapse button visibly closes it) -- never a silent
    preference-only change."""
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-inspector-rail-handle")
        await pilot.pause(0.2)

        right_rail = console.query_one("#console-right-rail")
        handle = console.query_one("#console-inspector-rail-handle")
        assert right_rail.display is False
        assert handle.display is True

        await pilot.click("#console-inspector-rail-open")
        await pilot.pause(0.3)

        assert right_rail.display is True
        assert handle.display is False
        assert _stored_rail_preferences(app)["right_open"] is True
        # Compact override: the honored rail waives the main column's
        # min-width so the grid always resolves below 150 cols.
        main_column = console.query_one("#console-main-column")
        assert main_column.styles.min_width.value == 0
        # AC1: Inspector content is actually reachable.
        assert console.query_one("#console-staged-context-tray").display is True

        await pilot.click("#console-inspector-rail-collapse")
        await pilot.pause(0.3)

        assert right_rail.display is False
        assert handle.display is True
        assert _stored_rail_preferences(app)["right_open"] is False


@pytest.mark.asyncio
async def test_sources_chip_opens_inspector_at_140_cols():
    """DS-06: clicking the Sources chip opens the Inspector rail (the
    staged-sources tray's only surface below 150 cols) and focuses it, so
    activation is visible even when the rail was already open."""
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-sources-chip")
        await pilot.pause(0.2)

        right_rail = console.query_one("#console-right-rail")
        assert right_rail.display is False

        # The ConsoleHarness does not load the app's chip-width CSS, so the
        # chip can lay out beyond the clickable viewport here; keyboard
        # activation (focus + Enter) exercises the same BINDINGS path, and
        # the real-app click path is covered by the UAT (p7c).
        sources_chip = console.query_one("#console-sources-chip")
        sources_chip.focus()
        await pilot.pause(0.1)
        await pilot.press("enter")
        await pilot.pause(0.3)

        assert right_rail.display is True
        assert console.focused is right_rail
        assert _stored_rail_preferences(app)["right_open"] is True

        # Already open: activation must still give feedback (focus frame).
        sources_chip.focus()
        await pilot.pause(0.1)
        await pilot.press("enter")
        await pilot.pause(0.3)
        assert right_rail.display is True
        assert console.focused is right_rail


@pytest.mark.asyncio
async def test_tools_chip_opens_inspector_at_80x24_single_pane():
    """AC1 at 80x24: single-pane mode hides both handles, so the chips are
    the route to the Inspector; keyboard activation (Enter) opens it
    in-grid while the transcript keeps a usable share."""
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(80, 24)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        await pilot.pause(0.2)

        # Single-pane defaults: both handles hidden, both rails closed.
        assert console.query_one("#console-context-rail-handle").display is False
        assert console.query_one("#console-inspector-rail-handle").display is False
        right_rail = console.query_one("#console-right-rail")
        assert right_rail.display is False

        # Chips clip visually at 80 cols (LY-03), so activate from the
        # keyboard: focus + Enter exercises the chip's BINDINGS path.
        tools_chip = console.query_one("#console-tools-chip")
        tools_chip.focus()
        await pilot.pause(0.1)
        await pilot.press("enter")
        await pilot.pause(0.3)

        assert right_rail.display is True
        assert console.focused is right_rail
        main_column = console.query_one("#console-main-column")
        assert main_column.styles.min_width.value == 0
        transcript_region = console.query_one("#console-transcript-region")
        assert transcript_region.outer_size.width > 0
        # Regression guard (2026-08-05 UAT): toggling the RIGHT rail
        # serializes the full payload including left_open -- without the
        # explicit-toggle marker the left rail must stay force-collapsed
        # here, or the LY-08 single-pane protection is silently lost.
        assert console.query_one("#console-left-rail").display is False
        assert "left_open_explicit" not in _stored_rail_preferences(app)

        # The rail's own collapse button is the way back; activate it from
        # the keyboard because the stripped harness clips narrow controls.
        collapse_button = console.query_one("#console-inspector-rail-collapse", Button)
        collapse_button.focus()
        await pilot.press("enter")
        await pilot.pause(0.3)
        assert right_rail.display is False


@pytest.mark.asyncio
async def test_left_handle_opens_left_rail_at_90_cols():
    """AC2 for the LEFT rail: at 90 cols (below the 100-col compact
    collapse) the Context handle visibly opens the rail -- the write-through
    persists left_open=True plus the explicit-toggle marker even though the
    coerced default is already True, which is what the marker-based
    force-collapse rule honors (ADR-043)."""
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(90, 24)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-context-rail-handle")
        await pilot.pause(0.2)

        left_rail = console.query_one("#console-left-rail")
        left_handle = console.query_one("#console-context-rail-handle")
        context_button = console.query_one("#console-context-rail-open", Button)
        assert left_rail.display is False
        assert left_handle.display is True
        assert context_button.label == "Context->"
        assert context_button.tooltip == "Open Context rail"

        context_button.focus()
        await pilot.press("enter")
        await pilot.pause(0.3)

        assert left_rail.display is True
        assert left_handle.display is False
        assert _stored_rail_preferences(app)["left_open"] is True
        # The marker (not the left_open value, which matches the default)
        # is what the narrow force-collapse rule honors (ADR-043).
        assert _stored_rail_preferences(app)["left_open_explicit"] is True
        main_column = console.query_one("#console-main-column")
        assert main_column.styles.min_width.value == 0
        transcript_region = console.query_one("#console-transcript-region")
        assert transcript_region.outer_size.width > 0

        collapse_button = console.query_one("#console-context-rail-collapse", Button)
        collapse_button.focus()
        await pilot.press("enter")
        await pilot.pause(0.3)

        assert left_rail.display is False
        assert left_handle.display is True
        assert _stored_rail_preferences(app)["left_open"] is False


@pytest.mark.asyncio
async def test_section_toggle_preserves_left_open_explicit_marker():
    """A later write that did not touch the left rail (a section toggle
    re-serializing the full payload) must not erase the explicit-toggle
    marker, or the honored left rail would silently re-collapse below 100
    cols on the next state build."""
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(90, 24)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-context-rail-handle")
        await pilot.pause(0.2)

        context_button = console.query_one("#console-context-rail-open", Button)
        context_button.focus()
        await pilot.press("enter")
        await pilot.pause(0.3)
        assert _stored_rail_preferences(app)["left_open_explicit"] is True

        # The Details toggle is scrolled out of view at 90x24; calling the
        # handler directly exercises the same section-only write path
        # (_set_console_rail_preference(section_updates=...)).
        console._toggle_console_rail_section("details")
        await pilot.pause(0.3)

        stored = _stored_rail_preferences(app)
        assert stored["details_open"] is True
        assert stored["left_open_explicit"] is True
        assert console.query_one("#console-left-rail").display is True


@pytest.mark.asyncio
async def test_default_layout_unchanged_at_160_cols():
    """AC4: with no explicit toggles stored, the 160-col default layout is
    exactly what it was -- left rail open, Inspector closed, standard
    main-column min-width."""
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-left-rail")
        await pilot.pause(0.2)

        assert console.query_one("#console-left-rail").display is True
        assert console.query_one("#console-context-rail-handle").display is False
        assert console.query_one("#console-right-rail").display is False
        assert console.query_one("#console-inspector-rail-handle").display is True
        main_column = console.query_one("#console-main-column")
        assert main_column.styles.min_width.value == 56
        assert list(app.app_config["console"]["rail_state"]) == [
            "console_rail_state:global:shared-layout-v1"
        ]


@pytest.mark.asyncio
async def test_console_rail_scope_seeding_is_lossless_one_time_and_responsive_safe(
    monkeypatch,
):
    """Every absent target is seeded once without deleting or rewriting sources."""
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-left-rail")
        await pilot.pause(0.2)

        writes: list[tuple[str, dict[str, bool], bool]] = []

        def record_save(
            _screen,
            key: str,
            serialized: dict[str, bool],
            *,
            notify_on_failure: bool = False,
        ) -> None:
            writes.append((key, deepcopy(serialized), notify_on_failure))

        monkeypatch.setattr(
            type(console), "_save_console_rail_preferences", record_save
        )

        shared_key = build_console_rail_preference_key(layout_scope="global")
        research_key = build_console_rail_preference_key(
            workspace_id="Research Lab", layout_scope="workspace"
        )
        writing_key = build_console_rail_preference_key(
            workspace_id="Writing Room", layout_scope="workspace"
        )
        defaults = serialize_console_rail_preferences(ConsoleRailPreferences())
        seeded_defaults = {
            key: value
            for key, value in defaults.items()
            if key not in {"right_open", "character_open"}
        }
        workspace_payload = {**defaults, "left_open": False, "workspace_open": False}
        legacy_payload = {**defaults, "right_open": True, "details_open": True}
        shared_payload = {**defaults, "agent_open": True, "character_open": False}

        cases = (
            (
                shared_key,
                research_key,
                {research_key.value: deepcopy(workspace_payload)},
                workspace_payload,
            ),
            (
                shared_key,
                research_key,
                {research_key.fallback_value: deepcopy(legacy_payload)},
                legacy_payload,
            ),
            (shared_key, research_key, {}, seeded_defaults),
            (
                research_key,
                research_key,
                {
                    research_key.fallback_value: deepcopy(legacy_payload),
                    shared_key.value: deepcopy(shared_payload),
                },
                legacy_payload,
            ),
            (
                writing_key,
                writing_key,
                {shared_key.value: deepcopy(shared_payload)},
                shared_payload,
            ),
        )

        for selected_key, workspace_key, records, expected in cases:
            writes.clear()
            app.app_config["console"] = {"rail_state": records}
            source_snapshot = deepcopy(records)

            seeded = console._ensure_console_rail_scope_seed(
                selected_key, workspace_key
            )

            assert seeded == expected
            assert records[selected_key.value] == expected
            assert writes == [(selected_key.value, expected, False)]
            for source_key, source_payload in source_snapshot.items():
                assert records[source_key] == source_payload

            console._ensure_console_rail_scope_seed(selected_key, workspace_key)
            assert writes == [(selected_key.value, expected, False)]

        writes.clear()
        existing = {**defaults, "model_open": False}
        records = {
            shared_key.value: deepcopy(existing),
            research_key.value: deepcopy(workspace_payload),
            writing_key.value: deepcopy(shared_payload),
            research_key.fallback_value: deepcopy(legacy_payload),
        }
        app.app_config["console"] = {
            "rail_layout_scope": "global",
            "rail_state": records,
        }
        snapshot = deepcopy(records)

        assert console._ensure_console_rail_scope_seed(shared_key, research_key) == (
            existing
        )
        app.app_config["console"]["rail_layout_scope"] = "workspace"
        assert console._ensure_console_rail_scope_seed(research_key, research_key) == (
            workspace_payload
        )
        assert records == snapshot
        assert writes == []

        app.app_config["console"]["rail_layout_scope"] = "global"
        before_compact = json.dumps(records, sort_keys=True, separators=(",", ":"))
        console._current_console_rail_state(available_columns=80)
        after_compact = json.dumps(records, sort_keys=True, separators=(",", ":"))
        assert after_compact == before_compact
        assert writes == []

        app.app_config["console"] = {"rail_layout_scope": "global", "rail_state": {}}
        writes.clear()
        console._set_console_rail_preference(left_open=False)
        assert len(writes) == 1
        assert writes[0][0] == shared_key.value
        assert writes[0][1]["left_open"] is False
        assert "right_open" not in writes[0][1]


@pytest.mark.asyncio
async def test_console_rail_writes_cannot_finish_with_stale_seed(monkeypatch):
    """A delayed seed writer cannot overwrite a newer in-memory mutation."""
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-left-rail")
        await pilot.app.workers.wait_for_complete()

        shared_key = build_console_rail_preference_key(layout_scope="global")
        app.app_config["console"] = {"rail_layout_scope": "global", "rail_state": {}}
        seed_started = threading.Event()
        release_seed = threading.Event()
        mutation_finished = threading.Event()
        persisted: dict[str, dict[str, bool]] = {}
        write_order: list[bool] = []

        def blocking_save(section: str, key: str, value: dict[str, bool]) -> bool:
            assert section == "console.rail_state"
            if value["left_open"] is True and not seed_started.is_set():
                seed_started.set()
                assert release_seed.wait(5)
            persisted[key] = deepcopy(value)
            write_order.append(value["left_open"])
            if value["left_open"] is False:
                mutation_finished.set()
            return True

        monkeypatch.setattr(
            chat_screen_module,
            "save_setting_to_cli_config",
            blocking_save,
        )

        console._current_console_rail_state(available_columns=160)
        for _ in range(40):
            if seed_started.is_set():
                break
            await pilot.pause(0.05)
        assert seed_started.is_set()

        console._set_console_rail_preference(left_open=False)
        for _ in range(10):
            if mutation_finished.is_set():
                break
            await pilot.pause(0.02)
        release_seed.set()
        await pilot.app.workers.wait_for_complete()

        assert (
            app.app_config["console"]["rail_state"][shared_key.value]["left_open"]
            is False
        )
        assert persisted[shared_key.value]["left_open"] is False
        assert write_order[-1] is False


@pytest.mark.asyncio
async def test_console_rail_scope_seed_preserves_explicit_left_open_intent(
    monkeypatch,
):
    """Scope adoption keeps the stored intent marker but drops transient UI state."""
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-left-rail")
        await pilot.pause(0.2)

        writes: list[tuple[str, dict[str, bool]]] = []

        def record_save(
            _screen,
            key: str,
            serialized: dict[str, bool],
            *,
            notify_on_failure: bool = False,
        ) -> None:
            assert notify_on_failure is False
            writes.append((key, deepcopy(serialized)))

        monkeypatch.setattr(
            type(console), "_save_console_rail_preferences", record_save
        )

        workspace_id = (
            console._workspace._current_console_workspace_context().active_workspace_id
        )
        shared_key = build_console_rail_preference_key(layout_scope="global")
        workspace_key = build_console_rail_preference_key(
            workspace_id=workspace_id,
            layout_scope="workspace",
        )
        base = serialize_console_rail_preferences(ConsoleRailPreferences())
        source = {
            **base,
            "left_open": True,
            "left_open_explicit": True,
            "focused_widget_id": "console-workspace-tree",
        }

        for selected_scope, source_key, target_key in (
            ("global", workspace_key.value, shared_key),
            ("workspace", shared_key.value, workspace_key),
        ):
            records = {source_key: deepcopy(source)}
            source_bytes = json.dumps(records[source_key], separators=(",", ":"))
            app.app_config["console"] = {
                "rail_layout_scope": selected_scope,
                "rail_state": records,
            }
            writes.clear()

            state = console._current_console_rail_state(available_columns=90)

            target = records[target_key.value]
            assert target["left_open_explicit"] is True
            assert "focused_widget_id" not in target
            assert state.left_open is True
            assert state.left_forced_collapsed is False
            assert writes == [(target_key.value, target)]
            assert (
                json.dumps(records[source_key], separators=(",", ":")) == source_bytes
            )
