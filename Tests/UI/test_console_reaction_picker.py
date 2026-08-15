"""Console reaction picker UI and metadata-only boundary contracts."""

from __future__ import annotations

import asyncio
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
from PIL import Image
from textual import events, on
from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Button, Input, Static

from tldw_chatbook.Character_Chat.visual_identity import (
    SAMIRA_EXPRESSION_KEYS,
    SAMIRA_REACTION_LABELS,
)
from tldw_chatbook.Chat.console_rail_state import ConsoleRailState
from tldw_chatbook.Chat.console_session_settings import ConsoleSettingsSummaryState
from tldw_chatbook.UI.Console_Modules.left_rail import ConsoleLeftRail
from tldw_chatbook.Widgets.Console.console_inspector_section import (
    ConsoleInspectorSectionState,
)
from tldw_chatbook.Widgets.Console.console_reaction_picker_modal import (
    FILTER_INPUT_ID,
    PREVIEW_ID,
    RESULTS_CONTAINER_ID,
    ROW_CLASS,
    ROW_HIGHLIGHTED_CLASS,
    ConsoleReactionPickerModal,
    ReactionCleared,
    ReactionOption,
    ReactionPreviewRequested,
    ReactionSelected,
    filter_reaction_options,
)
from tldw_chatbook.Workspaces.conversation_browser_state import (
    build_console_conversation_browser_state,
)
from tldw_chatbook.Workspaces.display_state import ConsoleWorkspaceContextState

FILTER_SETTLE_SECONDS = 0.3
PREVIEW_SETTLE_SECONDS = 0.25


def _samira_options() -> tuple[ReactionOption, ...]:
    return tuple(
        ReactionOption(
            expression_key=SAMIRA_EXPRESSION_KEYS[label],
            display_label=label.title(),
            content_type="image/webp",
            is_animated=False,
        )
        for label in SAMIRA_REACTION_LABELS
    )


class PickerHarness(App[None]):
    def __init__(self, options: tuple[ReactionOption, ...]) -> None:
        super().__init__()
        self._options = options
        self.previews: list[ReactionOption] = []
        self.selected: list[ReactionOption] = []
        self.cleared = 0
        self.dismissed: object = "not-called"

    async def on_mount(self) -> None:
        await self.push_screen(
            ConsoleReactionPickerModal(options=self._options, message_target=self),
            callback=lambda value: setattr(self, "dismissed", value),
        )

    @on(ReactionPreviewRequested)
    def capture_preview(self, event: ReactionPreviewRequested) -> None:
        self.previews.append(event.option)

    @on(ReactionSelected)
    def capture_selection(self, event: ReactionSelected) -> None:
        self.selected.append(event.option)

    @on(ReactionCleared)
    def capture_clear(self, _event: ReactionCleared) -> None:
        self.cleared += 1


class _FilterApplyBarrier:
    def __init__(self, blocked_query: str, latest_query: str | None = None) -> None:
        self.blocked_query = blocked_query
        self.latest_query = latest_query
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.latest_finished = asyncio.Event()
        self.calls: list[str] = []
        self.finished: list[str] = []
        self.active = 0
        self.max_active = 0

    async def run(self, modal, query: str, original_apply) -> None:
        self.calls.append(query)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            if query == self.blocked_query:
                self.started.set()
                try:
                    await self.release.wait()
                except asyncio.CancelledError:
                    self.cancelled.set()
                    raise
            await original_apply(modal, query)
            self.finished.append(query)
            if query == self.latest_query:
                self.latest_finished.set()
        finally:
            self.active -= 1


def _install_filter_apply_barrier(
    monkeypatch,
    *,
    blocked_query: str,
    latest_query: str | None = None,
) -> _FilterApplyBarrier:
    original_apply = ConsoleReactionPickerModal._apply_filter
    barrier = _FilterApplyBarrier(blocked_query, latest_query)

    async def blocked_apply(modal, query: str) -> None:
        await barrier.run(modal, query, original_apply)

    monkeypatch.setattr(ConsoleReactionPickerModal, "_apply_filter", blocked_apply)
    return barrier


def test_reaction_option_is_small_frozen_metadata_only_value() -> None:
    option = _samira_options()[0]

    assert option.expression_key == "custom:admiration"
    assert option.display_label == "Admiration"
    assert option.content_type == "image/webp"
    assert option.is_animated is False
    assert not hasattr(option, "path")
    assert not hasattr(option, "bytes")
    with pytest.raises(FrozenInstanceError):
        option.display_label = "Changed"  # type: ignore[misc]


def test_filter_covers_exact_31_samira_metadata_options() -> None:
    options = _samira_options()

    assert len(options) == 31
    assert tuple(option.display_label.lower() for option in options) == (
        SAMIRA_REACTION_LABELS
    )
    assert filter_reaction_options(options, "") == options
    assert [item.display_label for item in filter_reaction_options(options, "rel")] == [
        "Relief"
    ]
    assert [
        item.display_label
        for item in filter_reaction_options(options, "custom:speaking")
    ] == ["Speaking"]
    assert filter_reaction_options(options, "no-such-reaction") == ()


@pytest.mark.asyncio
async def test_open_focuses_filter_shows_all_rows_count_and_one_preview_request() -> (
    None
):
    app = PickerHarness(_samira_options())

    async with app.run_test(size=(120, 36)) as pilot:
        await pilot.pause(PREVIEW_SETTLE_SECONDS)
        filter_input = app.screen.query_one(f"#{FILTER_INPUT_ID}", Input)
        rows = list(app.screen.query(f".{ROW_CLASS}"))

        assert filter_input.has_focus
        assert len(rows) == 31
        assert len(app.screen.query(f".{ROW_HIGHLIGHTED_CLASS}")) == 1
        assert str(
            app.screen.query_one("#console-reaction-picker-count", Static).renderable
        ) == ("1 / 31 reactions")
        assert app.previews == [_samira_options()[0]]


@pytest.mark.asyncio
async def test_filter_updates_rows_count_empty_state_and_selected_only_preview() -> (
    None
):
    app = PickerHarness(_samira_options())

    async with app.run_test(size=(120, 36)) as pilot:
        await pilot.pause(PREVIEW_SETTLE_SECONDS)
        filter_input = app.screen.query_one(f"#{FILTER_INPUT_ID}", Input)
        filter_input.value = "rel"
        await pilot.pause(FILTER_SETTLE_SECONDS + PREVIEW_SETTLE_SECONDS)

        assert len(app.screen.query(f".{ROW_CLASS}")) == 1
        assert str(
            app.screen.query_one("#console-reaction-picker-count", Static).renderable
        ) == ("1 / 1 reactions")
        assert app.previews[-1].display_label == "Relief"

        filter_input.value = "no-such-reaction"
        await pilot.pause(FILTER_SETTLE_SECONDS)
        empty = app.screen.query_one("#console-reaction-picker-empty", Static)
        assert str(empty.renderable) == "No reactions match."
        assert str(
            app.screen.query_one("#console-reaction-picker-count", Static).renderable
        ) == ("0 / 0 reactions")
        assert len(app.previews) == 2


@pytest.mark.asyncio
async def test_keyboard_down_up_enter_selects_highlight_and_escape_cancels() -> None:
    options = _samira_options()
    app = PickerHarness(options)

    async with app.run_test(size=(120, 36)) as pilot:
        await pilot.pause(PREVIEW_SETTLE_SECONDS)
        await pilot.press("down", "down", "up")
        await pilot.press("enter")
        await pilot.pause()

    assert app.previews == [options[0]]
    assert app.selected == [options[1]]
    assert app.dismissed is None

    cancel_app = PickerHarness(options)
    async with cancel_app.run_test(size=(120, 36)) as pilot:
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()

    assert cancel_app.selected == []
    assert cancel_app.cleared == 0
    assert cancel_app.dismissed is None


@pytest.mark.parametrize(
    ("keys", "expected_index"),
    [(("enter",), 23), (("down", "enter"), 24)],
)
@pytest.mark.asyncio
async def test_pending_filter_is_flushed_before_navigation_and_enter(
    keys: tuple[str, ...], expected_index: int
) -> None:
    options = _samira_options()
    app = PickerHarness(options)

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause(PREVIEW_SETTLE_SECONDS)
        app.screen.query_one(f"#{FILTER_INPUT_ID}", Input).value = "custom:re"
        await pilot.press(*keys)
        await pilot.pause()

    assert app.selected == [options[expected_index]]
    assert app.previews == [options[0]]


@pytest.mark.parametrize("action", ["down", "enter"])
@pytest.mark.asyncio
async def test_timer_filter_and_flush_share_one_inflight_apply(
    monkeypatch,
    action: str,
) -> None:
    barrier = _install_filter_apply_barrier(
        monkeypatch,
        blocked_query="custom:re",
    )
    flush_started = asyncio.Event()
    original_flush = ConsoleReactionPickerModal._flush_pending_filter

    async def observed_flush(modal: ConsoleReactionPickerModal) -> None:
        flush_started.set()
        await original_flush(modal)

    monkeypatch.setattr(
        ConsoleReactionPickerModal,
        "_flush_pending_filter",
        observed_flush,
    )
    options = _samira_options()
    app = PickerHarness(options)

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause(PREVIEW_SETTLE_SECONDS)
        modal = app.screen
        filter_input = modal.query_one(f"#{FILTER_INPUT_ID}", Input)
        filter_input.value = "custom:re"
        await asyncio.wait_for(barrier.started.wait(), timeout=1)
        if action == "down":
            action_task = asyncio.create_task(modal.on_key(events.Key("down", None)))
        else:
            action_task = asyncio.create_task(
                modal._filter_submitted(
                    Input.Submitted(filter_input, filter_input.value)
                )
            )
        await asyncio.wait_for(flush_started.wait(), timeout=1)
        observed_max_active = barrier.max_active
        barrier.release.set()
        await action_task
        await pilot.pause()

        assert observed_max_active == 1
        assert barrier.max_active == 1
        assert barrier.calls.count("custom:re") == 1
        if action == "down":
            assert (
                str(
                    app.screen.query_one(
                        "#console-reaction-picker-count", Static
                    ).renderable
                )
                == "2 / 3 reactions"
            )
            assert app.screen.query_one(f"#{FILTER_INPUT_ID}", Input).has_focus
            await pilot.press("escape")

    if action == "enter":
        assert app.selected == [options[23]]


@pytest.mark.asyncio
async def test_new_query_waits_for_old_apply_then_owns_final_dom(monkeypatch) -> None:
    barrier = _install_filter_apply_barrier(
        monkeypatch,
        blocked_query="rel",
        latest_query="custom:re",
    )
    app = PickerHarness(_samira_options())

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause(PREVIEW_SETTLE_SECONDS)
        filter_input = app.screen.query_one(f"#{FILTER_INPUT_ID}", Input)
        filter_input.value = "rel"
        await asyncio.wait_for(barrier.started.wait(), timeout=1)
        filter_input.value = "custom:re"
        observed_max_active: list[int] = []

        async def release_after_latest_timer() -> None:
            await asyncio.sleep(FILTER_SETTLE_SECONDS)
            observed_max_active.append(barrier.max_active)
            barrier.release.set()

        release_task = asyncio.create_task(release_after_latest_timer())
        await asyncio.wait_for(barrier.latest_finished.wait(), timeout=1)
        await release_task
        await pilot.pause()

        assert observed_max_active == [1]
        assert barrier.max_active == 1
        assert barrier.calls.count("rel") == 1
        assert barrier.calls.count("custom:re") == 1
        assert len(app.screen.query(f".{ROW_CLASS}")) == 3
        assert (
            str(
                app.screen.query_one(
                    "#console-reaction-picker-count", Static
                ).renderable
            )
            == "1 / 3 reactions"
        )
        assert filter_input.has_focus


@pytest.mark.asyncio
async def test_dismiss_cancels_inflight_filter_before_barrier_release(
    monkeypatch,
) -> None:
    barrier = _install_filter_apply_barrier(monkeypatch, blocked_query="rel")
    app = PickerHarness(_samira_options())

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause(PREVIEW_SETTLE_SECONDS)
        app.screen.query_one(f"#{FILTER_INPUT_ID}", Input).value = "rel"
        await asyncio.wait_for(barrier.started.wait(), timeout=1)
        try:
            await app.screen._perform_safe_cancel(source="test")
            await asyncio.wait_for(barrier.cancelled.wait(), timeout=1)
        finally:
            barrier.release.set()
        await asyncio.sleep(0)

    assert "rel" not in barrier.finished


@pytest.mark.asyncio
async def test_narrow_keyboard_keeps_highlight_visible_and_count_synced() -> None:
    options = _samira_options()
    app = PickerHarness(options)

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause(PREVIEW_SETTLE_SECONDS)
        results = app.screen.query_one(f"#{RESULTS_CONTAINER_ID}", VerticalScroll)
        initial_scroll_y = results.scroll_y

        await pilot.press(*(20 * ("down",)))
        await pilot.pause(PREVIEW_SETTLE_SECONDS)

        highlighted = app.screen.query_one(f".{ROW_HIGHLIGHTED_CLASS}", Button)
        assert results.content_region.contains_region(highlighted.region)
        assert results.scroll_y > initial_scroll_y
        assert (
            str(
                app.screen.query_one(
                    "#console-reaction-picker-count", Static
                ).renderable
            )
            == "21 / 31 reactions"
        )
        assert app.previews == [options[0], options[20]]

        await pilot.press("enter")
        await pilot.pause()

    assert app.selected == [options[20]]


@pytest.mark.asyncio
async def test_narrow_up_reversal_and_filtered_count_preserve_filter_focus() -> None:
    app = PickerHarness(_samira_options())

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause(PREVIEW_SETTLE_SECONDS)
        results = app.screen.query_one(f"#{RESULTS_CONTAINER_ID}", VerticalScroll)
        filter_input = app.screen.query_one(f"#{FILTER_INPUT_ID}", Input)

        await pilot.press(*(20 * ("down",)))
        await pilot.pause()
        advanced_scroll_y = results.scroll_y

        await pilot.press(*(20 * ("up",)))
        await pilot.pause()

        highlighted = app.screen.query_one(f".{ROW_HIGHLIGHTED_CLASS}", Button)
        assert results.content_region.contains_region(highlighted.region)
        assert results.scroll_y < advanced_scroll_y
        assert (
            str(
                app.screen.query_one(
                    "#console-reaction-picker-count", Static
                ).renderable
            )
            == "1 / 31 reactions"
        )
        assert filter_input.has_focus

        filter_input.value = "custom:re"
        await pilot.pause(FILTER_SETTLE_SECONDS)
        assert (
            str(
                app.screen.query_one(
                    "#console-reaction-picker-count", Static
                ).renderable
            )
            == "1 / 3 reactions"
        )

        await pilot.press("down")
        await pilot.pause(PREVIEW_SETTLE_SECONDS)
        assert (
            str(
                app.screen.query_one(
                    "#console-reaction-picker-count", Static
                ).renderable
            )
            == "2 / 3 reactions"
        )
        assert filter_input.has_focus


@pytest.mark.asyncio
async def test_rapid_filter_changes_render_only_the_settled_query(
    monkeypatch,
) -> None:
    render_count = 0
    original_render = ConsoleReactionPickerModal._render_results

    async def counted_render(modal: ConsoleReactionPickerModal) -> None:
        nonlocal render_count
        render_count += 1
        await original_render(modal)

    monkeypatch.setattr(
        ConsoleReactionPickerModal,
        "_render_results",
        counted_render,
    )
    options = _samira_options()
    app = PickerHarness(options)

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause(PREVIEW_SETTLE_SECONDS)
        filter_input = app.screen.query_one(f"#{FILTER_INPUT_ID}", Input)
        for query in ("c", "cu", "cus", "custom", "custom:", "custom:re"):
            filter_input.value = query
        await pilot.pause(FILTER_SETTLE_SECONDS + PREVIEW_SETTLE_SECONDS)

        assert render_count == 2
        assert len(app.screen.query(f".{ROW_CLASS}")) == 3
        assert (
            str(
                app.screen.query_one(
                    "#console-reaction-picker-count", Static
                ).renderable
            )
            == "1 / 3 reactions"
        )
        assert filter_input.has_focus
        assert app.previews == [options[0], options[23]]


@pytest.mark.asyncio
async def test_pending_filter_is_cancelled_when_modal_is_dismissed(
    monkeypatch,
) -> None:
    render_count = 0
    original_render = ConsoleReactionPickerModal._render_results

    async def counted_render(modal: ConsoleReactionPickerModal) -> None:
        nonlocal render_count
        render_count += 1
        await original_render(modal)

    monkeypatch.setattr(
        ConsoleReactionPickerModal,
        "_render_results",
        counted_render,
    )
    options = _samira_options()
    app = PickerHarness(options)

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause(PREVIEW_SETTLE_SECONDS)
        app.screen.query_one(f"#{FILTER_INPUT_ID}", Input).value = "custom:re"
        await pilot.press("escape")
        await pilot.pause(FILTER_SETTLE_SECONDS)

    assert render_count == 1
    assert app.previews == [options[0]]


@pytest.mark.asyncio
async def test_rapid_highlights_emit_latest_preview_only_and_none_after_dismiss() -> (
    None
):
    options = _samira_options()
    app = PickerHarness(options)

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause(PREVIEW_SETTLE_SECONDS)
        assert app.previews == [options[0]]

        await pilot.press(*(10 * ("down",)))
        await pilot.pause(PREVIEW_SETTLE_SECONDS / 4)
        await pilot.press(*(20 * ("down",)))
        await pilot.pause(PREVIEW_SETTLE_SECONDS)
        assert app.previews == [options[0], options[30]]

        app.screen._sync_highlight()
        await pilot.pause(PREVIEW_SETTLE_SECONDS)
        assert app.previews == [options[0], options[30]]

        await pilot.press("down")
        await pilot.press("escape")
        await pilot.pause(PREVIEW_SETTLE_SECONDS)

    assert app.previews == [options[0], options[30]]


@pytest.mark.asyncio
async def test_clear_is_explicit_named_message_and_dismisses() -> None:
    app = PickerHarness(_samira_options())

    async with app.run_test(size=(120, 36)) as pilot:
        await pilot.pause()
        await pilot.click("#console-reaction-picker-clear")
        await pilot.pause()

    assert app.cleared == 1
    assert app.selected == []
    assert app.dismissed is None


@pytest.mark.parametrize("size", [(80, 24), (120, 36)])
@pytest.mark.asyncio
async def test_modal_geometry_keeps_filter_list_and_actions_in_bounds(size) -> None:
    app = PickerHarness(_samira_options())

    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        shell = app.screen.query_one("#console-reaction-picker-modal")
        filter_input = app.screen.query_one(f"#{FILTER_INPUT_ID}", Input)
        results = app.screen.query_one(f"#{RESULTS_CONTAINER_ID}")
        clear = app.screen.query_one("#console-reaction-picker-clear", Button)
        select = app.screen.query_one("#console-reaction-picker-select", Button)
        preview = app.screen.query_one(f"#{PREVIEW_ID}")

        assert 0 <= shell.region.x and shell.region.right <= size[0]
        assert 0 <= shell.region.y and shell.region.bottom <= size[1]
        for core in (filter_input, results, clear, select):
            assert shell.region.contains_region(core.region)
            assert core.region.width > 0 and core.region.height > 0
        if size == (80, 24):
            assert preview.display is False
        else:
            assert preview.display is True


@pytest.mark.asyncio
async def test_zero_options_has_usable_filter_clear_and_empty_copy() -> None:
    app = PickerHarness(())

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        assert app.screen.query_one(f"#{FILTER_INPUT_ID}", Input).has_focus
        assert str(
            app.screen.query_one("#console-reaction-picker-empty", Static).renderable
        ) == ("No reactions available.")
        assert app.screen.query_one("#console-reaction-picker-select", Button).disabled
        clear = app.screen.query_one("#console-reaction-picker-clear", Button)
        assert not clear.disabled and clear.display
        assert app.previews == []


@pytest.mark.asyncio
async def test_display_label_markup_is_rendered_as_literal_text() -> None:
    option = ReactionOption(
        expression_key="custom:alarm",
        display_label="[red]Alarm[/red]",
        content_type="image/webp",
        is_animated=False,
    )
    app = PickerHarness((option,))

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        row = app.screen.query_one(f".{ROW_CLASS}", Button)
        assert "[red]Alarm[/red]" in row.label.plain


@pytest.mark.asyncio
async def test_open_filter_and_highlight_never_read_or_decode_all_assets(
    monkeypatch,
) -> None:
    reads: list[object] = []
    decodes: list[object] = []

    def fail_read(path: Path) -> bytes:
        reads.append(path)
        raise AssertionError("picker must not read reaction assets")

    def fail_decode(source, *args, **kwargs):
        decodes.append(source)
        raise AssertionError("picker must not decode reaction assets")

    monkeypatch.setattr(Path, "read_bytes", fail_read)
    monkeypatch.setattr(Image, "open", fail_decode)
    options = _samira_options()
    app = PickerHarness(options)

    async with app.run_test(size=(120, 36)) as pilot:
        await pilot.pause(PREVIEW_SETTLE_SECONDS)
        app.screen.query_one(f"#{FILTER_INPUT_ID}", Input).value = "custom:re"
        await pilot.pause(FILTER_SETTLE_SECONDS + PREVIEW_SETTLE_SECONDS)
        await pilot.press("down")
        await pilot.pause(PREVIEW_SETTLE_SECONDS)

    assert reads == []
    assert decodes == []
    assert app.previews == [options[0], options[23], options[24]]


def _workspace_state() -> ConsoleWorkspaceContextState:
    return ConsoleWorkspaceContextState(
        heading="Context",
        workspace_label="Workspace: Default",
        authority_label="Authority: local",
        sync_label="Sync: not configured",
        runtime_label="Runtime: local",
        conversation_rows=(),
        conversation_empty_copy="No conversations yet.",
        conversation_browser=build_console_conversation_browser_state(
            rows=(), active_workspace_id=None
        ),
        change_workspace_enabled=False,
        change_workspace_recovery="",
        new_conversation_enabled=False,
        new_conversation_recovery="",
        recovery_copy="",
    )


class RailHarness(App[None]):
    def __init__(self, manual_label: str | None) -> None:
        super().__init__()
        self._manual_label = manual_label
        self.requests = 0

    def compose(self) -> ComposeResult:
        yield ConsoleLeftRail(
            rail_state=ConsoleRailState(True, False, True, False),
            workspace_context_state=_workspace_state(),
            settings_summary_state=ConsoleSettingsSummaryState(
                model_row="Model: test",
                context_row="Context: 0",
                sampling_row="T 0.7 · max_tokens 100",
                identity_row="Identity: character",
            ),
            system_line_text="System: none",
            system_line_dim=True,
            fleet_line="",
            agent_status_line="Idle",
            agent_steps_text="",
            agent_fleet_section_state=ConsoleInspectorSectionState(rows=(), summary=""),
            agent_drilldown_active=False,
            agent_full_log_available=False,
            show_character_section=True,
            character_avatar_widget_builder=lambda: Static("avatar"),
            character_avatar_name="Samira",
            manual_reaction_label=self._manual_label,
        )

    @on(ConsoleLeftRail.ReactionPickerRequested)
    def capture_request(self, _event: ConsoleLeftRail.ReactionPickerRequested) -> None:
        self.requests += 1


@pytest.mark.parametrize(
    ("manual_label", "expected"),
    [(None, "Reaction: Automatic"), ("Relief", "Reaction: Relief (manual)")],
)
@pytest.mark.asyncio
async def test_character_rail_shows_reaction_action_and_visible_manual_state(
    manual_label: str | None, expected: str
) -> None:
    app = RailHarness(manual_label)

    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause()
        button = app.screen.query_one("#console-character-reaction-open", Button)
        button.scroll_visible(animate=False)
        await pilot.pause()
        state = app.screen.query_one("#console-character-reaction-state", Static)
        assert str(state.renderable) == expected
        assert str(button.label) == "Reaction…"
        await pilot.click(button)
        await pilot.pause()

    assert app.requests == 1
