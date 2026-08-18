"""Mounted regressions for the collapsible Console composer."""

import inspect
from itertools import pairwise
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from textual.app import App, ComposeResult
from textual.events import Paste
from textual.widgets import Button, Static

from Tests.UI.test_console_native_chat_flow import (
    _configure_native_ready_console,
)
from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Widgets.Console import ConsoleComposerBar, ConsoleTranscript

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_STYLESHEET = _REPO_ROOT / "tldw_chatbook/css/components/_agentic_terminal.tcss"
_BUNDLED_STYLESHEET = _REPO_ROOT / "tldw_chatbook/css/tldw_cli_modular.tcss"


class _ComposerGeometryApp(App[None]):
    """Mount a composer with the production stylesheet for geometry assertions."""

    CSS_PATH = str(_BUNDLED_STYLESHEET)

    def __init__(self, *, collapsed: bool = False) -> None:
        super().__init__()
        self._initially_collapsed = collapsed

    def compose(self) -> ComposeResult:
        yield ConsoleComposerBar(
            id="console-native-composer",
            collapsed=self._initially_collapsed,
        )


class _BundledConsoleGeometryHarness(ConsoleHarness):
    """Mount the full Console with the generated production stylesheet."""

    CSS_PATH = str(_BUNDLED_STYLESHEET)


def _ready_console_host() -> ConsoleHarness:
    app = _build_test_app()
    _configure_native_ready_console(app)
    return ConsoleHarness(app)


async def _mounted_console(host: ConsoleHarness, pilot):
    console = host.screen_stack[-1]
    await _wait_for_selector(console, pilot, "#console-native-composer")
    return console


async def _seed_overflowing_transcript(console, pilot):
    """Populate enough multi-line rows to exercise transcript scrolling."""
    store = console._ensure_console_chat_store()
    selected_message_id = ""
    for index in range(24):
        message = store.append_message(
            store.active_session_id,
            role=(
                ConsoleMessageRole.USER
                if index % 2 == 0
                else ConsoleMessageRole.ASSISTANT
            ),
            content="\n".join(f"message {index} line {line}" for line in range(3)),
        )
        selected_message_id = message.id
    await console._sync_native_console_chat_ui()
    transcript = console.query_one("#console-native-transcript", ConsoleTranscript)
    for _ in range(40):
        if transcript.max_scroll_y > 0:
            break
        await pilot.pause(0.05)
    assert transcript.max_scroll_y > 0
    transcript.select_message(selected_message_id)
    return transcript, selected_message_id


def _transcript_tail_is_anchored(transcript: ConsoleTranscript) -> bool:
    """Return Textual's semantic tail-follow state, including manual release."""
    return bool(
        transcript.is_anchored and not getattr(transcript, "_anchor_released", False)
    )


def test_small_ordinary_paste_keeps_explicit_paste_origin_when_not_collapsed():
    composer = ConsoleComposerBar(paste_collapse_threshold=50)

    composer.insert_pasted_text("small paste")

    snapshot = composer.capture_draft_snapshot()
    assert [
        (segment.origin, segment.collapse_state) for segment in snapshot.segments
    ] == [("paste", "literal")]
    assert composer.has_paste_segments() is True


def test_adjacent_collapsed_pastes_have_one_literal_newline_and_expand_copy():
    composer = ConsoleComposerBar(paste_collapse_threshold=20)
    first = "A" * 21
    second = "B" * 22

    composer.insert_pasted_text(first)
    composer.insert_pasted_text(second)

    assert composer.draft_text() == first + "\n" + second
    assert composer._display_draft_text() == (
        f"Pasted text | {len(first)} characters | Expand\n"
        f"Pasted text | {len(second)} characters | Expand"
    )
    assert [
        (segment.text, segment.origin, segment.collapse_state)
        for segment in composer.capture_draft_snapshot().segments
    ] == [
        (first, "paste", "collapsed"),
        ("\n", "literal", "literal"),
        (second, "paste", "collapsed"),
    ]


@pytest.mark.parametrize(
    ("first_suffix", "second_prefix", "expected_boundary"),
    [
        ("\n", "", ""),
        ("", "\n", ""),
        (" \t", " \t", "\n"),
    ],
)
def test_adjacent_collapsed_paste_boundary_respects_only_literal_newlines(
    first_suffix: str,
    second_prefix: str,
    expected_boundary: str,
):
    composer = ConsoleComposerBar(paste_collapse_threshold=20)
    first = ("A" * 21) + first_suffix
    second = second_prefix + ("B" * 22)

    composer.insert_pasted_text(first)
    composer.insert_pasted_text(second)

    assert composer.draft_text() == first + expected_boundary + second


def test_adjacent_paste_snapshot_restore_preserves_boundary_and_block_states():
    first = "A" * 80
    second = "B" * 90
    composer = ConsoleComposerBar(paste_collapse_threshold=50)
    composer.insert_pasted_text(first)
    composer.insert_pasted_text(second)
    snapshot = composer.capture_draft_snapshot()

    restored = ConsoleComposerBar(paste_collapse_threshold=50)
    restored.restore_snapshot(snapshot)

    assert restored.draft_text() == first + "\n" + second
    assert restored._display_draft_text().count("Pasted text |") == 2
    assert restored.cursor_index == len(first) + 1 + len(second)


def _paste_token(text: str) -> str:
    return f"Pasted text | {len(text)} characters | Expand"


@pytest.mark.parametrize(
    ("first_suffix", "second_prefix", "display_boundary"),
    [
        ("\n", "", "\n"),
        ("\r\n", "", "\r\n"),
        ("", "\n", "\n"),
        ("", "\r\n", "\r\n"),
        ("\n\n", "", "\n\n"),
        ("", "\r\n\r\n", "\r\n\r\n"),
    ],
)
def test_embedded_line_breaks_separate_collapsed_tokens_in_display_without_rewrite(
    first_suffix: str,
    second_prefix: str,
    display_boundary: str,
):
    composer = ConsoleComposerBar(paste_collapse_threshold=20)
    first = ("A" * 21) + first_suffix
    second = second_prefix + ("B" * 22)

    composer.insert_pasted_text(first)
    composer.insert_pasted_text(second)

    assert composer.draft_text() == first + second
    assert composer._display_draft_text() == (
        _paste_token(first) + display_boundary + _paste_token(second)
    )
    assert not any(
        getattr(segment, "generated_boundary", False)
        for segment in composer.capture_draft_snapshot().segments
    )


@pytest.mark.parametrize("line_break", ["\n", "\r\n"])
@pytest.mark.parametrize("position", ["first", "second", "middle"])
def test_all_line_break_paste_keeps_adjacent_display_labels_separated(
    line_break: str,
    position: str,
):
    all_breaks = line_break * 30
    first = "A" * 60
    second = "B" * 70
    payloads = {
        "first": [all_breaks, second],
        "second": [first, all_breaks],
        "middle": [first, all_breaks, second],
    }[position]
    composer = ConsoleComposerBar(paste_collapse_threshold=20)

    for payload in payloads:
        composer.insert_pasted_text(payload)

    display = composer._display_draft_text()
    labels = [_paste_token(payload) for payload in payloads]
    assert composer.draft_text() == "".join(payloads)
    assert display.count("Pasted text |") == len(payloads)
    for left_label, right_label in pairwise(labels):
        left_end = display.index(left_label) + len(left_label)
        right_start = display.index(right_label, left_end)
        assert line_break in display[left_end:right_start]
    leading, trailing = composer._paste_edge_line_breaks(all_breaks)
    assert leading
    assert trailing
    assert leading + trailing == all_breaks


def test_inserting_collapsed_paste_before_existing_block_adds_right_boundary():
    first = "A" * 80
    second = "B" * 90
    composer = ConsoleComposerBar(paste_collapse_threshold=50)
    composer.insert_pasted_text(second)
    composer.position_cursor_from_display_index(0)

    composer.insert_pasted_text(first)

    assert composer.draft_text() == first + "\n" + second
    assert composer._display_draft_text() == (
        _paste_token(first) + "\n" + _paste_token(second)
    )


def test_inserting_collapsed_paste_between_blocks_reuses_and_adds_boundaries():
    first = "A" * 80
    middle = "M" * 85
    last = "Z" * 90
    composer = ConsoleComposerBar(paste_collapse_threshold=50)
    composer.insert_pasted_text(first)
    composer.insert_pasted_text(last)
    composer.position_cursor_from_display_index(len(_paste_token(first)) + 1)

    composer.insert_pasted_text(middle)

    assert composer.draft_text() == first + "\n" + middle + "\n" + last
    snapshot = composer.capture_draft_snapshot()
    assert [segment.text for segment in snapshot.segments] == [
        first,
        "\n",
        middle,
        "\n",
        last,
    ]
    assert [segment.generated_boundary for segment in snapshot.segments] == [
        False,
        True,
        False,
        True,
        False,
    ]


@pytest.mark.parametrize("file_first", [True, False])
def test_file_and_collapsed_paste_never_gain_a_generated_boundary(file_first: bool):
    pasted = "P" * 80
    composer = ConsoleComposerBar(paste_collapse_threshold=50)
    if file_first:
        composer.insert_file_segment("file body", label="file.txt")
        composer.insert_pasted_text(pasted)
        expected = "file body" + pasted
    else:
        composer.insert_pasted_text(pasted)
        composer.insert_file_segment("file body", label="file.txt")
        expected = pasted + "file body"

    assert composer.draft_text() == expected
    assert not any(
        segment.generated_boundary
        for segment in composer.capture_draft_snapshot().segments
    )


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("insertion_kind", ["small_paste", "file", "literal"])
def test_insertion_next_to_generated_boundary_removes_orphan_separator(
    side: str,
    insertion_kind: str,
):
    first = "A" * 80
    second = "B" * 90
    inserted = "small" if insertion_kind == "small_paste" else "inserted"
    composer = ConsoleComposerBar(paste_collapse_threshold=50)
    composer.insert_pasted_text(first)
    composer.insert_pasted_text(second)
    boundary_display_index = len(_paste_token(first))
    if side == "right":
        boundary_display_index += 1
    composer.position_cursor_from_display_index(boundary_display_index)

    if insertion_kind == "small_paste":
        composer.insert_pasted_text(inserted)
    elif insertion_kind == "file":
        composer.insert_file_segment(inserted, label="inserted.txt")
    else:
        composer.insert_text(inserted)

    assert composer.draft_text() == first + inserted + second
    assert not any(
        segment.generated_boundary
        for segment in composer.capture_draft_snapshot().segments
    )


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("insertion_kind", ["small_paste", "file", "literal"])
def test_insertion_next_to_user_authored_newline_preserves_it(
    side: str,
    insertion_kind: str,
):
    first = "A" * 80
    second = "B" * 90
    inserted = "small" if insertion_kind == "small_paste" else "inserted"
    composer = ConsoleComposerBar(paste_collapse_threshold=50)
    composer.insert_pasted_text(first)
    composer.insert_text("\n")
    composer.insert_pasted_text(second)
    boundary_display_index = len(_paste_token(first))
    if side == "right":
        boundary_display_index += 1
    composer.position_cursor_from_display_index(boundary_display_index)

    if insertion_kind == "small_paste":
        composer.insert_pasted_text(inserted)
    elif insertion_kind == "file":
        composer.insert_file_segment(inserted, label="inserted.txt")
    else:
        composer.insert_text(inserted)

    expected = (
        first + inserted + "\n" + second
        if side == "left"
        else first + "\n" + inserted + second
    )
    assert composer.draft_text() == expected
    snapshot = composer.capture_draft_snapshot()
    assert "\n" in "".join(segment.text for segment in snapshot.segments)
    assert all(not segment.generated_boundary for segment in snapshot.segments)


def test_generic_history_collapsed_literal_is_not_a_paste_block_boundary():
    composer = ConsoleComposerBar(paste_collapse_threshold=50)
    huge_literal = "L" * (composer.UNDO_RECOLLAPSE_CHAR_THRESHOLD + 1)
    composer.load_draft(huge_literal)
    composer.insert_text("x")
    assert composer.undo() is True
    assert composer.capture_draft_snapshot().segments[0].origin == "literal"
    assert composer.capture_draft_snapshot().segments[0].collapse_state == "collapsed"

    composer.insert_pasted_text("P" * 80)

    assert composer.draft_text() == huge_literal + ("P" * 80)
    assert not any(
        segment.generated_boundary
        for segment in composer.capture_draft_snapshot().segments
    )


def test_expanded_unedited_paste_retains_block_identity_for_next_paste():
    first = "A" * 80
    second = "B" * 90
    composer = ConsoleComposerBar(paste_collapse_threshold=50)
    composer.insert_pasted_text(first)
    composer._segments[0].collapse_state = "expanded"

    composer.insert_pasted_text(second)

    assert composer.draft_text() == first + "\n" + second
    first_snapshot = composer.capture_draft_snapshot().segments[0]
    assert first_snapshot.collapse_state == "expanded"
    assert first_snapshot.paste_block is True


def test_editing_inside_expanded_paste_uses_literal_split_semantics_for_adjacency():
    first = "A" * 80
    second = "B" * 90
    composer = ConsoleComposerBar(paste_collapse_threshold=50)
    composer.insert_pasted_text(first)
    composer._segments[0].collapse_state = "expanded"
    composer.position_cursor_from_display_index(40)
    composer.insert_text(" edited ")
    composer.move_cursor_end()

    composer.insert_pasted_text(second)

    assert composer.draft_text() == first[:40] + " edited " + first[40:] + second
    assert not any(
        segment.generated_boundary
        for segment in composer.capture_draft_snapshot().segments
    )


@settings(max_examples=24, derandomize=True, deadline=None)
@given(
    first_suffix=st.sampled_from(["", " ", "\t", "\n", "\r\n", "\n\n"]),
    second_prefix=st.sampled_from(["", " ", "\t", "\n", "\r\n", "\r\n\r\n"]),
)
def test_adjacent_paste_boundary_matrix_is_canonical_and_visibly_separated(
    first_suffix: str,
    second_prefix: str,
):
    first = ("A" * 60) + first_suffix
    second = second_prefix + ("B" * 70)
    composer = ConsoleComposerBar(paste_collapse_threshold=50)

    composer.insert_pasted_text(first)
    composer.insert_pasted_text(second)

    has_explicit_break = first.endswith(("\n", "\r\n")) or second.startswith(
        ("\n", "\r\n")
    )
    expected_separator = "" if has_explicit_break else "\n"
    assert composer.draft_text() == first + expected_separator + second
    display = composer._display_draft_text()
    first_label_end = display.index(_paste_token(first)) + len(_paste_token(first))
    second_label_start = display.index(_paste_token(second), first_label_end)
    assert "\n" in display[first_label_end:second_label_start]


def _assert_full_button_label_fits(button: Button, expected_label: str) -> None:
    """Assert the mounted button renders its complete label inside its chrome."""
    rendered_line = button.render_line(0)
    rendered_text = rendered_line.text
    internal_chrome_cells = len(rendered_text) - len(rendered_text.strip())
    rendered_label_capacity = button.content_region.width - internal_chrome_cells

    assert str(button.label) == expected_label
    assert rendered_label_capacity >= button.label.cell_length, (
        f"{button.id} region={button.region.width}, "
        f"content={button.content_region.width}, "
        f"label_capacity={rendered_label_capacity}, "
        f"label_cells={button.label.cell_length}, "
        f"rendered={rendered_text!r}"
    )
    assert rendered_text.strip() == expected_label


@pytest.mark.asyncio
async def test_collapse_button_moves_focus_to_transcript_without_sending():
    host = _ready_console_host()
    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("keep me")
        store = console._ensure_console_chat_store()
        message_count = len(store.messages_for_session(store.active_session_id))
        collapse = composer.query_one("#console-composer-collapse", Button)
        collapse.focus()
        await pilot.press("enter")
        await pilot.pause()

        assert composer.collapsed is True
        assert composer.draft_text() == "keep me"
        assert isinstance(host.focused, ConsoleTranscript)
        assert len(store.messages_for_session(store.active_session_id)) == message_count


@pytest.mark.asyncio
async def test_expand_button_and_one_escape_expand_and_focus_draft():
    host = _ready_console_host()
    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        await pilot.click("#console-composer-collapse")
        await pilot.pause()
        assert composer.collapsed is True

        await pilot.press("escape")
        await pilot.pause()
        assert composer.collapsed is False
        assert host.focused is composer

        await pilot.click("#console-composer-collapse")
        await pilot.pause()
        assert composer.collapsed is True
        expand = composer.query_one("#console-composer-expand", Button)
        expand.focus()
        await pilot.press("enter")
        await pilot.pause()
        assert composer.collapsed is False
        assert host.focused is composer


@pytest.mark.asyncio
async def test_collapsed_composer_hidden_input_and_paste_do_not_mutate_or_send():
    host = _ready_console_host()
    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("keep me")
        store = console._ensure_console_chat_store()
        message_count = len(store.messages_for_session(store.active_session_id))
        console._set_console_composer_collapsed(True)
        await pilot.pause()

        for key in ("x", "backspace", "delete", "enter"):
            await pilot.press(key)
            await pilot.pause()
            assert composer.draft_text() == "keep me"
            assert (
                len(store.messages_for_session(store.active_session_id))
                == message_count
            )

        console.on_paste(Paste("pasted"))
        await pilot.pause()
        assert composer.draft_text() == "keep me"
        assert len(store.messages_for_session(store.active_session_id)) == message_count

        run_worker = Mock()
        console.run_worker = run_worker
        console.on_paste(Paste("/tmp/hidden-image.png"))
        await pilot.pause()
        run_worker.assert_not_called()
        assert composer.draft_text() == "keep me"
        assert len(store.messages_for_session(store.active_session_id)) == message_count


@pytest.mark.asyncio
async def test_collapsed_escape_action_is_dynamic_and_setup_controls_are_inert(
    monkeypatch,
):
    host = _ready_console_host()
    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)

        assert console.check_action("expand_collapsed_console_composer", ()) is False
        with monkeypatch.context() as context:
            context.setattr(console, "_console_setup_modal_blocking", lambda: True)
            console._set_console_composer_collapsed(True)
            assert composer.collapsed is False

        console._set_console_composer_collapsed(True)
        await pilot.pause()
        assert console.check_action("expand_collapsed_console_composer", ()) is True

        with monkeypatch.context() as context:
            context.setattr(console, "_console_setup_modal_blocking", lambda: True)
            assert (
                console.check_action("expand_collapsed_console_composer", ()) is False
            )
            console._set_console_composer_collapsed(False)
            assert composer.collapsed is True


@pytest.mark.asyncio
async def test_expanded_escape_clears_transcript_selection_before_focus_fallback():
    host = _ready_console_host()
    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        store = console._ensure_console_chat_store()
        message = store.append_message(
            store.active_session_id,
            role=ConsoleMessageRole.ASSISTANT,
            content="selected answer",
        )
        await console._sync_native_console_chat_ui()
        transcript = console.query_one("#console-native-transcript", ConsoleTranscript)
        transcript.select_message(message.id)
        transcript.focus()
        await pilot.pause()

        await pilot.press("escape")
        await pilot.pause()

        assert transcript.selected_message_id is None
        assert host.focused is transcript


@pytest.mark.asyncio
async def test_screen_collapse_disarms_unfurl_without_changing_pasted_content():
    host = _ready_console_host()
    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        pasted_text = "pending unfurl paste " * 20
        composer.insert_pasted_text(pasted_text)
        await pilot.click("#console-command-visible-text")
        await pilot.pause()
        assert composer.has_pending_paste_confirmation() is True

        console._set_console_composer_collapsed(True)
        await pilot.pause()

        assert composer.has_pending_paste_confirmation() is False
        assert composer.has_paste_segments() is True
        assert composer.draft_text() == pasted_text


@pytest.mark.asyncio
async def test_collapsed_stop_routes_native_run_control_without_expanding():
    host = _ready_console_host()
    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.sync_action_state(
            has_draft=False,
            run_active=True,
            can_save_chatbook=False,
        )
        console._set_console_composer_collapsed(True)
        await pilot.pause()
        stop_run = AsyncMock()
        console._stop_console_generation_from_visible_action = stop_run

        composer.query_one("#console-collapsed-stop-generation", Button).press()
        await pilot.pause()

        stop_run.assert_awaited_once_with()
        assert composer.collapsed is True


@pytest.mark.asyncio
async def test_collapsed_stop_is_inert_while_setup_blocks(monkeypatch):
    host = _ready_console_host()
    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.sync_action_state(
            has_draft=False,
            run_active=True,
            can_save_chatbook=False,
        )
        console._set_console_composer_collapsed(True)
        await pilot.pause()
        stop_run = AsyncMock()
        console._stop_console_generation_from_visible_action = stop_run
        monkeypatch.setattr(console, "_console_setup_modal_blocking", lambda: True)

        stop = composer.query_one("#console-collapsed-stop-generation", Button)
        await console.handle_console_stop_generation(Button.Pressed(stop))

        stop_run.assert_not_awaited()
        assert composer.collapsed is True


@pytest.mark.asyncio
async def test_stale_composer_layout_revision_does_not_override_current_focus():
    host = _ready_console_host()
    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        console._set_console_composer_collapsed(True)
        await pilot.pause()
        expand = console.query_one("#console-composer-expand", Button)
        expand.focus()
        await pilot.pause()

        console._finish_console_composer_layout_change(
            console._console_composer_layout_revision - 1,
            True,
            console._capture_console_transcript_reading_state(),
        )
        await pilot.pause()

        assert host.focused is expand


@pytest.mark.asyncio
async def test_rapid_toggle_ignores_stale_collapse_focus_callback():
    host = _ready_console_host()
    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        transcript, selected = await _seed_overflowing_transcript(console, pilot)

        console._set_console_composer_collapsed(True)
        console._set_console_composer_collapsed(False)
        await pilot.pause()
        await pilot.pause()

        assert composer.collapsed is False
        assert host.focused is composer
        assert transcript.selected_message_id == selected


@pytest.mark.asyncio
async def test_rapid_collapse_then_priority_escape_ignores_stale_focus_callback():
    host = _ready_console_host()
    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        transcript, selected = await _seed_overflowing_transcript(console, pilot)

        console._set_console_composer_collapsed(True)
        await pilot.press("escape")
        await pilot.pause()
        await pilot.pause()

        assert composer.collapsed is False
        assert host.focused is composer
        assert transcript.selected_message_id == selected


@pytest.mark.asyncio
async def test_anchored_tail_and_selection_survive_collapse_round_trip():
    host = _ready_console_host()
    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        transcript, selected = await _seed_overflowing_transcript(console, pilot)
        transcript.anchor()
        await pilot.pause()

        assert transcript.is_anchored
        assert transcript.scroll_y == transcript.max_scroll_y

        console._set_console_composer_collapsed(True)
        await pilot.pause()
        assert transcript.is_anchored
        assert transcript.scroll_y == transcript.max_scroll_y
        assert transcript.selected_message_id == selected

        console._set_console_composer_collapsed(False)
        await pilot.pause()
        assert transcript.is_anchored
        assert transcript.scroll_y == transcript.max_scroll_y
        assert transcript.selected_message_id == selected


@pytest.mark.asyncio
async def test_manual_reading_position_and_selection_survive_collapse_round_trip():
    host = _ready_console_host()
    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        transcript, selected = await _seed_overflowing_transcript(console, pilot)
        await pilot.pause()
        transcript.release_anchor()
        transcript.scroll_to(y=2, animate=False)
        await pilot.pause()
        assert _transcript_tail_is_anchored(transcript) is False
        reading_y = transcript.scroll_y

        console._set_console_composer_collapsed(True)
        await pilot.pause()
        assert _transcript_tail_is_anchored(transcript) is False
        assert transcript.scroll_y == min(reading_y, transcript.max_scroll_y)
        assert transcript.selected_message_id == selected

        console._set_console_composer_collapsed(False)
        await pilot.pause()
        assert _transcript_tail_is_anchored(transcript) is False
        assert transcript.scroll_y == min(reading_y, transcript.max_scroll_y)
        assert transcript.selected_message_id == selected


@pytest.mark.asyncio
async def test_replacement_composer_inherits_screen_state_and_active_session_draft():
    host = _ready_console_host()
    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        store = console._ensure_console_chat_store()
        composer.load_draft("canonical session draft")
        console._console_composer_collapsed = True
        console._session._sync_console_session_draft()

        await console.recompose()
        await pilot.pause()
        replacement = console.query_one("#console-native-composer", ConsoleComposerBar)

        assert replacement is not composer
        assert replacement.collapsed is True
        assert replacement.draft_text() == "canonical session draft"
        assert store.session_draft(store.active_session_id) == "canonical session draft"


@pytest.mark.asyncio
async def test_console_composer_defaults_expanded_and_toggles_idempotently():
    host = _ready_console_host()

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        expanded = composer.query_one("#console-composer-expanded")
        collapsed = composer.query_one("#console-composer-collapsed")

        assert composer.collapsed is False
        assert expanded.display is True
        assert collapsed.display is False
        assert composer.can_focus is True

        composer.set_collapsed(True)
        composer.set_collapsed(True)
        await pilot.pause()

        assert composer.collapsed is True
        assert expanded.display is False
        assert collapsed.display is True
        assert composer.can_focus is False

        composer.set_collapsed(False)
        composer.set_collapsed(False)
        await pilot.pause()

        assert composer.collapsed is False
        assert expanded.display is True
        assert collapsed.display is False
        assert composer.can_focus is True


@pytest.mark.asyncio
async def test_console_composer_default_geometry_is_single_row():
    """task-17651: zero chrome rows — the composer IS its draft rows.

    The old box (border 2 + padding 2, COMPOSER_CHROME_ROWS = 4) rendered
    5-8 total rows for 1-4 draft rows. The dense-form composer renders
    exactly its draft height: 1 row empty, capped at 4 for long drafts.
    """
    app = _ComposerGeometryApp()

    async with app.run_test(size=(140, 42)) as pilot:
        await pilot.pause()
        composer = app.query_one("#console-native-composer", ConsoleComposerBar)

        assert composer.region.height == 1

        composer.load_draft("x " * 400)
        await pilot.pause()
        assert composer.region.height == 4

        composer.load_draft("short")
        await pilot.pause()
        assert composer.region.height == 1


@pytest.mark.asyncio
async def test_console_bottom_stack_single_separator_contract():
    """task-17651: one border row between transcript text and the chips.

    The workbench frame closes at the grid: the transcript widget and the
    region no longer draw their own bottom edges, and the composer carries
    no border box at all — grid border, chips, 1-row composer, footer.
    """
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = _BundledConsoleGeometryHarness(app)

    async with host.run_test(size=(150, 44)) as pilot:
        console = await _mounted_console(host, pilot)
        transcript = console.query_one(
            "#console-native-transcript", ConsoleTranscript
        )
        region = console.query_one("#console-transcript-region")
        grid = console.query_one("#console-workspace-grid")
        chips = console.query_one("#console-status-chips")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        footer = console.query_one("#screen-footer-status")

        # The transcript draws no border of its own at any size.
        assert transcript.styles.border.top[0] in ("", "none")
        assert transcript.styles.border.bottom[0] in ("", "none")
        # The region's bottom edge is suppressed, so the transcript content
        # runs flush to the region's last row and the grid's own bottom
        # border is the only separator line.
        assert region.styles.border_bottom[0] in ("", "none")
        assert (
            transcript.region.y + transcript.region.height
            == region.region.y + region.region.height
        )
        # Below the grid's closing border: chips, then ONE deliberate blank
        # row on each side of the 1-row composer (task-17657/17659: the bar
        # floats clear of the status row above and the footer below), footer.
        assert chips.region.y == grid.region.y + grid.region.height
        assert composer.region.y == chips.region.y + chips.region.height + 1
        assert composer.region.height == 1
        assert footer.region.y == composer.region.y + composer.region.height + 1
        strips = host.screen._compositor.render_strips()
        for gap_y in (composer.region.y - 1, composer.region.y + composer.region.height):
            gap_row = "".join(seg.text for seg in strips[gap_y])
            assert not gap_row.strip(), (gap_y, repr(gap_row[:20]))


@pytest.mark.asyncio
async def test_console_composer_focus_edge_is_live_and_stable():
    """task-17651: the dense-form focus edge actually renders.

    The inline workbench frame used to override the stylesheet, leaving
    the focused composer visually identical to the rest state. With the
    composer out of the frame grammar, CSS owns the edge: solid at rest,
    thick focus accent when focused, with no dimensional change.
    """
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = _BundledConsoleGeometryHarness(app)

    async with host.run_test(size=(150, 44)) as pilot:
        console = await _mounted_console(host, pilot)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        # The Console auto-focuses the composer on mount; move focus away
        # to capture the genuine rest state first.
        console.query_one(
            "#console-native-transcript", ConsoleTranscript
        ).focus()
        await pilot.pause()
        rest_kind, rest_color = composer.styles.border_left
        rest_region = composer.region

        # Painted evidence for the REST edge too — style reads cannot see
        # the global *:focus outline overpainting the row (task-17651).
        rest_row = "".join(
            seg.text
            for seg in host.screen._compositor.render_strips()[rest_region.y]
        )
        assert rest_row[0] == "│", repr(rest_row[:4])

        composer.focus()
        await pilot.pause()

        focus_kind, focus_color = composer.styles.border_left
        assert rest_kind == "solid"
        assert focus_kind == "thick"
        assert focus_color != rest_color
        assert composer.region == rest_region
        focus_row = "".join(
            seg.text
            for seg in host.screen._compositor.render_strips()[composer.region.y]
        )
        # Thick edge block + padding cell — never the outline's ┌─ corners.
        assert focus_row[0] == "█", repr(focus_row[:4])
        assert focus_row[1] == " ", repr(focus_row[:4])


@pytest.mark.asyncio
async def test_console_transcript_focus_recolors_region_columns():
    """task-17651: the region's column lines carry transcript focus.

    The transcript widget draws no border of its own any more, so the
    TASK-359 pane-stop painter recolors the region's inline column lines
    to the focus accent while the transcript holds focus — and the
    suppressed top/bottom edges survive the repaint (no resurrected
    separator rows), with no layout change.
    """
    from textual.color import Color as _Color

    app = _build_test_app()
    _configure_native_ready_console(app)
    host = _BundledConsoleGeometryHarness(app)

    async with host.run_test(size=(150, 44)) as pilot:
        console = await _mounted_console(host, pilot)
        transcript = console.query_one(
            "#console-native-transcript", ConsoleTranscript
        )
        region = console.query_one("#console-transcript-region")
        region_rect = region.region

        transcript.focus()
        await pilot.pause()

        kind, color = region.styles.border_left
        assert kind == "solid"
        assert color == _Color.parse("#0178D4")
        assert region.styles.border_bottom[0] in ("", "none")
        assert region.styles.border_top[0] in ("", "none")
        assert region.region == region_rect

        console.query_one("#console-native-composer", ConsoleComposerBar).focus()
        await pilot.pause()

        blur_kind, blur_color = region.styles.border_left
        assert blur_kind == "solid"
        assert blur_color == _Color.parse("#6f7782")
        assert region.styles.border_bottom[0] in ("", "none")


@pytest.mark.asyncio
async def test_console_composer_geometry_is_bounded_then_exactly_one_row():
    app = _ComposerGeometryApp()

    async with app.run_test(size=(140, 42)) as pilot:
        composer = app.query_one("#console-native-composer", ConsoleComposerBar)

        # task-17651: dense-form composer — 1-4 rows, no chrome.
        assert 1 <= composer.region.height <= 4

        composer.set_collapsed(True)
        await pilot.pause()

        assert composer.region.height == 1

        composer.set_collapsed(False)
        await pilot.pause()

        assert 1 <= composer.region.height <= 4


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(140, 42), (100, 32)])
async def test_expanded_composer_toggle_renders_full_approved_label(size):
    app = _ComposerGeometryApp()

    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        collapse = app.query_one("#console-composer-collapse", Button)

        _assert_full_button_label_fits(collapse, "Composer ▾")
        # task-2154.14: 14 -> 12 ("Composer ▾" is exactly 10 cells + 2
        # chrome); the 2 freed cells fund the labeled "Menu" button beside
        # it, keeping the left cluster's total footprint unchanged.
        assert collapse.region.width == 12
        assert collapse.content_region.width == 10


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(140, 42), (100, 32)])
async def test_collapsed_composer_toggle_renders_full_approved_label(size):
    app = _ComposerGeometryApp(collapsed=True)

    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        composer = app.query_one("#console-native-composer", ConsoleComposerBar)
        expand = composer.query_one("#console-composer-expand", Button)

        _assert_full_button_label_fits(expand, "Expand ▴")
        assert composer.region.height == 1
        assert expand.region.width == 12
        assert expand.content_region.width == 10


@pytest.mark.asyncio
async def test_console_composer_compact_geometry_keeps_status_and_expand_visible():
    app = _ComposerGeometryApp(collapsed=True)

    async with app.run_test(size=(100, 32)) as pilot:
        await pilot.pause()
        composer = app.query_one("#console-native-composer", ConsoleComposerBar)

        assert composer.region.height == 1
        assert composer.query_one("#console-composer-expand", Button).region.width > 0
        assert (
            composer.query_one(
                "#console-composer-collapsed-status", Static
            ).region.width
            > 0
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(140, 42), (100, 32)])
async def test_console_responsive_collapsed_geometry_preserves_controls_and_rows(size):
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = _BundledConsoleGeometryHarness(app)

    async with host.run_test(size=size) as pilot:
        console = await _mounted_console(host, pilot)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        transcript = console.query_one("#console-native-transcript", ConsoleTranscript)
        composer.load_draft("retained draft")
        composer.set_pending_attachment_label("photo.png · 12 B")
        await pilot.pause()
        expanded_transcript_height = transcript.region.height

        console._set_console_composer_collapsed(True)
        await pilot.pause()
        status = composer.query_one("#console-composer-collapsed-status", Static)
        stop = composer.query_one("#console-collapsed-stop-generation", Button)
        expand = composer.query_one("#console-composer-expand", Button)

        assert composer.region.height == 1
        # task-17651: the expanded composer is already 1 row for a 1-row
        # draft, so collapse is a content swap, not a row-saving lever —
        # the transcript must simply not shrink.
        assert transcript.region.height >= expanded_transcript_height
        assert "Draft retained" in str(status.renderable)
        assert "Attachment retained" in str(status.renderable)
        assert expand.region.right <= status.region.x
        assert status.region.right <= composer.region.right

        composer.sync_action_state(
            has_draft=True,
            run_active=True,
            can_save_chatbook=False,
        )
        await pilot.pause()

        assert composer.region.height == 1
        assert stop.display is True
        assert expand.region.right <= status.region.x
        assert status.region.right <= stop.region.x
        assert stop.region.right <= composer.region.right
        assert stop.region.width == 8
        assert expand.region.width == 12


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("draft", "attachment", "run_active", "expected"),
    [
        ("", None, False, "Composer hidden"),
        (" ", None, False, "Composer hidden · Draft retained"),
        (
            "draft",
            "photo.png · 12 B",
            False,
            "Composer hidden · Draft retained · Attachment retained",
        ),
        ("", None, True, "Composer hidden · Generating"),
        (
            "draft",
            "photo.png · 12 B",
            True,
            "Composer hidden · Generating · Draft retained · Attachment retained",
        ),
    ],
)
async def test_console_collapsed_status_uses_presence_only(
    draft: str,
    attachment: str | None,
    run_active: bool,
    expected: str,
):
    host = _ready_console_host()

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft(draft)
        composer.set_pending_attachment_label(attachment)
        composer.sync_action_state(
            has_draft=bool(draft.strip()),
            run_active=run_active,
            can_save_chatbook=False,
        )
        composer.set_collapsed(True)
        await pilot.pause()

        status = composer.query_one("#console-composer-collapsed-status", Static)
        assert str(status.renderable) == expected
        stop = composer.query_one("#console-collapsed-stop-generation", Button)
        assert stop.display is run_active
        assert "photo.png" not in str(status.renderable)


@pytest.mark.asyncio
async def test_console_collapsed_status_sync_does_not_join_canonical_draft(
    monkeypatch,
):
    host = _ready_console_host()

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("retained draft")
        composer.set_collapsed(True)
        await pilot.pause()

        def _reject_draft_join(_composer: ConsoleComposerBar) -> str:
            raise AssertionError("collapsed status must not materialize the draft")

        monkeypatch.setattr(ConsoleComposerBar, "draft_text", _reject_draft_join)

        composer.sync_action_state(
            has_draft=True,
            run_active=True,
            can_save_chatbook=False,
        )

        status = composer.query_one("#console-composer-collapsed-status", Static)
        assert str(status.renderable) == (
            "Composer hidden · Generating · Draft retained"
        )


@pytest.mark.asyncio
async def test_console_composer_round_trip_preserves_editor_and_attachment_state():
    host = _ready_console_host()

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        pasted_text = "preserved large paste " * 20

        composer.insert_pasted_text(pasted_text)
        composer.move_cursor_left()
        composer.set_pending_attachment_label("photo.png · 12 B")
        before_caret_round_trip = (
            composer.draft_text(),
            composer.cursor_index,
            composer.has_paste_segments(),
            composer.has_full_draft_selection(),
        )

        composer.set_collapsed(True)
        composer.set_collapsed(False)
        await pilot.pause()

        assert (
            composer.draft_text(),
            composer.cursor_index,
            composer.has_paste_segments(),
            composer.has_full_draft_selection(),
        ) == before_caret_round_trip

        assert composer.select_all_draft() is True
        before_selection_round_trip = (
            composer.cursor_index,
            composer.has_full_draft_selection(),
        )

        composer.set_collapsed(True)
        composer.set_collapsed(False)
        await pilot.pause()

        assert (
            composer.cursor_index,
            composer.has_full_draft_selection(),
        ) == before_selection_round_trip
        attachment = composer.query_one("#console-attachment-indicator", Static)
        assert "photo.png · 12 B" in str(attachment.renderable)


@pytest.mark.asyncio
async def test_console_composer_collapse_preserves_pending_unfurl_segment():
    host = _ready_console_host()

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        pasted_text = "pending unfurl paste " * 20

        composer.insert_pasted_text(pasted_text)
        await pilot.click("#console-command-visible-text")
        await pilot.pause()
        assert composer.has_pending_paste_confirmation() is True

        composer.set_collapsed(True)
        await pilot.pause()

        assert composer.draft_text() == pasted_text
        assert composer.has_paste_segments() is True
        assert composer.has_pending_paste_confirmation() is True


@pytest.mark.asyncio
async def test_console_composer_collapse_pauses_cursor_timer_despite_lingering_focus():
    host = _ready_console_host()

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.focus()
        await pilot.pause()
        timer = composer._cursor_blink_timer
        assert timer is not None
        assert timer._active.is_set() is True

        composer.set_collapsed(True)
        composer._sync_cursor_blink_state()

        assert timer._active.is_set() is False


@pytest.mark.parametrize(
    "stylesheet",
    (_SOURCE_STYLESHEET, _BUNDLED_STYLESHEET),
    ids=("source", "bundle"),
)
def test_console_composer_collapsed_styles_are_pinned(stylesheet: Path):
    css = stylesheet.read_text(encoding="utf-8")
    required = (
        "#console-native-composer.console-composer-collapsed",
        "#console-composer-collapsed-status",
        "#console-composer-expand",
        "text-overflow: ellipsis",
        "#console-composer-collapse {\n    width: 12;\n    min-width: 12;\n}",
        "#console-composer-expand {\n    width: 12;\n    min-width: 12;\n}",
    )

    for token in required:
        assert token in css


def test_console_composer_has_no_status_strip_selector_dependency():
    source = inspect.getsource(ConsoleComposerBar)

    assert "console-status-chips" not in source


@pytest.mark.asyncio
async def test_composer_bar_no_longer_owns_the_save_chatbook_button():
    """Save Chatbook moved into the composer's ☰ menu.

    The temporary-chat block this test used to assert here moved with it and
    is covered in `Tests/UI/test_console_composer_menu.py`. What remains
    worth pinning is that the button did not stay behind: two surfaces for
    one action is how this branch previously ended up with save-chatbook
    blocked in one place and reachable in two others.
    """
    app = _ComposerGeometryApp()

    async with app.run_test(size=(140, 42)) as pilot:
        composer = app.query_one("#console-native-composer", ConsoleComposerBar)
        await pilot.pause()

        assert not composer.query("#console-save-chatbook")
        assert not composer.query("#console-attach-context")
        assert composer.query_one("#console-composer-menu", Button)


@pytest.mark.asyncio
async def test_composer_row_menu_left_of_draft_send_beside_draft_mic_gapped():
    """Pin the requested composer row order and the Send/Mic buffer.

    The Menu button sits left of the draft (right of Composer ▾) so overflow
    actions live on the left button cluster; Send hugs the draft's right
    edge; Mic follows Send across a >=2-cell empty gap so a press aimed at
    one cannot land on the other. Stop's budgeted-but-hidden slot sits AFTER
    Mic, so a run starting or stopping never shifts Send or Mic.
    """
    app = _ComposerGeometryApp()

    async with app.run_test(size=(140, 42)) as pilot:
        await pilot.pause()
        composer = app.query_one("#console-native-composer", ConsoleComposerBar)
        # TASK-2154.6: an empty draft shows the Send disabled-reason strip
        # between the draft and Send (by design, borrowing 1fr cells) --
        # probe with a draft present so the row-order assertions below
        # measure the resting layout the way they always have.
        composer.load_draft("x")
        await pilot.pause()
        collapse = app.query_one("#console-composer-collapse", Button)
        menu = app.query_one("#console-composer-menu", Button)
        draft = app.query_one("#console-command-visible-text", Static)
        mic = app.query_one("#console-dictation", Button)
        send = app.query_one("#console-send-message", Button)

        assert collapse.region.right <= menu.region.x
        assert menu.region.right <= draft.region.x
        # Send is adjacent to the draft (the draft keeps its 1-cell margin);
        # a right-aligned actions row would park Stop's hidden 8-cell budget
        # here instead.
        assert draft.region.right <= send.region.x
        assert send.region.x - draft.region.right <= 2
        # The anti-misclick buffer between Send and Mic.
        assert mic.region.x - send.region.right >= 2

        composer = app.query_one("#console-native-composer", ConsoleComposerBar)
        mic_x, send_x = mic.region.x, send.region.x
        composer.sync_action_state(
            has_draft=True,
            run_active=True,
            can_save_chatbook=False,
        )
        await pilot.pause()
        stop = app.query_one("#console-stop-generation", Button)
        assert stop.display
        # Stop appears in its budgeted slot right of Mic without moving
        # Send or Mic.
        assert mic.region.right <= stop.region.x
        assert (mic.region.x, send.region.x) == (mic_x, send_x)
