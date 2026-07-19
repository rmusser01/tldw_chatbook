"""Console composer caret editing: movement, mid-draft edits, word delete,
Shift+Enter newline, click-to-position, and caret validity across mutations.

The composer's caret is an offset into the canonical draft text (the text that
will be sent); collapsed paste tokens are single units for movement, deletion,
and word boundaries.
"""

import pytest
from textual.widgets import Static

from Tests.UI.test_console_native_chat_flow import (
    CapturingGateway,
    _configure_native_ready_console,
    _wait_for_text,
)
from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import ConsoleHarness
from tldw_chatbook.Widgets.Console import ConsoleComposerBar


PASTE_CHUNK = "chunk of pasted console text " * 10
PASTE_TOKEN = f"Pasted Text: {len(PASTE_CHUNK)} Characters"


def _composer_with(*parts: str) -> ConsoleComposerBar:
    """Build an unmounted composer; ``parts`` alternate typed/pasted text."""
    composer = ConsoleComposerBar()
    for index, part in enumerate(parts):
        if index % 2 == 0:
            composer.insert_text(part)
        else:
            composer.insert_pasted_text(part)
    return composer


# ---------------------------------------------------------------------------
# Caret movement (unmounted segment-model tests)
# ---------------------------------------------------------------------------


def test_composer_cursor_moves_by_character_and_clamps_at_edges():
    composer = _composer_with("hello")

    assert composer.cursor_index == 5
    assert composer.move_cursor_left() is True
    assert composer.cursor_index == 4
    assert composer.move_cursor_right() is True
    assert composer.cursor_index == 5
    assert composer.move_cursor_right() is False
    assert composer.cursor_index == 5

    assert composer.move_cursor_home() is True
    assert composer.cursor_index == 0
    assert composer.move_cursor_left() is False
    assert composer.cursor_index == 0
    assert composer.move_cursor_end() is True
    assert composer.cursor_index == 5


def test_composer_cursor_arrows_skip_paste_tokens_as_units():
    composer = _composer_with("ab", PASTE_CHUNK, "cd")
    token_end = 2 + len(PASTE_CHUNK)

    composer.move_cursor_home()
    composer.move_cursor_right()
    composer.move_cursor_right()
    assert composer.cursor_index == 2

    # Right arrow over the token jumps the whole token in one step.
    composer.move_cursor_right()
    assert composer.cursor_index == token_end
    composer.move_cursor_right()
    assert composer.cursor_index == token_end + 1

    # Left arrow over the token also jumps it in one step.
    composer.move_cursor_left()
    composer.move_cursor_left()
    assert composer.cursor_index == 2


# ---------------------------------------------------------------------------
# Mid-draft insertion and deletion
# ---------------------------------------------------------------------------


def test_composer_insert_text_at_cursor_mid_draft():
    composer = _composer_with("hello world")
    for _ in range(5):
        composer.move_cursor_left()
    assert composer.cursor_index == 6

    composer.insert_text("big ")

    assert composer.draft_text() == "hello big world"
    assert composer.cursor_index == 10


def test_composer_delete_left_and_right_at_cursor_mid_draft():
    composer = _composer_with("abcd")
    composer.move_cursor_home()
    composer.move_cursor_right()
    composer.move_cursor_right()
    assert composer.cursor_index == 2

    composer.delete_left()
    assert composer.draft_text() == "acd"
    assert composer.cursor_index == 1

    composer.delete_right()
    assert composer.draft_text() == "ad"
    assert composer.cursor_index == 1

    # Backspace at the draft start is a no-op; delete at the end is a no-op.
    composer.move_cursor_home()
    composer.delete_left()
    assert composer.draft_text() == "ad"
    composer.move_cursor_end()
    composer.delete_right()
    assert composer.draft_text() == "ad"


def test_composer_collapsed_paste_inserts_at_cursor_and_splits_literal_text():
    composer = _composer_with("before after")
    for _ in range(6):
        composer.move_cursor_left()

    composer.insert_pasted_text(PASTE_CHUNK)

    assert composer.draft_text() == "before" + PASTE_CHUNK + " after"
    assert composer.cursor_index == 6 + len(PASTE_CHUNK)
    assert [segment.collapse_state for segment in composer._segments] == [
        "literal",
        "collapsed",
        "literal",
    ]

    # Typing right after the token merges into the right literal neighbour.
    composer.insert_text("X")
    assert composer.draft_text() == "before" + PASTE_CHUNK + "X after"
    assert [segment.collapse_state for segment in composer._segments] == [
        "literal",
        "collapsed",
        "literal",
    ]


def test_composer_deletes_paste_token_as_unit_left_and_right():
    composer = _composer_with("ab", PASTE_CHUNK, "cd")

    # Backspace right after the token deletes the whole token.
    composer.move_cursor_home()
    composer.move_cursor_right()
    composer.move_cursor_right()
    composer.move_cursor_right()  # jumps over the token
    composer.delete_left()
    assert composer.draft_text() == "abcd"
    assert composer.cursor_index == 2

    # Forward delete right before the token deletes the whole token.
    composer = _composer_with("ab", PASTE_CHUNK, "cd")
    composer.move_cursor_home()
    composer.move_cursor_right()
    composer.move_cursor_right()
    composer.delete_right()
    assert composer.draft_text() == "abcd"
    assert composer.cursor_index == 2


# ---------------------------------------------------------------------------
# Ctrl+W (readline word-rubout)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("draft", "expected"),
    [
        ("hello world", "hello "),
        ("hello  ", ""),
        ("hello world  ", "hello "),
        ("hello", ""),
        ("hello world again", "hello world "),
    ],
)
def test_composer_delete_word_left_readline_semantics(draft, expected):
    composer = _composer_with(draft)

    assert composer.delete_word_left() is True
    assert composer.draft_text() == expected
    assert composer.cursor_index == len(expected)


def test_composer_delete_word_left_at_start_is_noop():
    composer = _composer_with("hello")
    composer.move_cursor_home()

    assert composer.delete_word_left() is False
    assert composer.draft_text() == "hello"


def test_composer_delete_word_left_treats_paste_token_as_single_word():
    composer = _composer_with("foo ", PASTE_CHUNK)

    assert composer.delete_word_left() is True
    assert composer.draft_text() == "foo "
    assert composer.cursor_index == 4


def test_composer_delete_word_left_stops_at_token_boundary():
    composer = _composer_with("foo ", PASTE_CHUNK, "bar")

    # The word right of the token deletes without touching the token...
    assert composer.delete_word_left() is True
    assert composer.draft_text() == "foo " + PASTE_CHUNK
    # ...and a second Ctrl+W deletes the token as one word.
    assert composer.delete_word_left() is True
    assert composer.draft_text() == "foo "


# ---------------------------------------------------------------------------
# Caret validity across paste/clear/restore and selection
# ---------------------------------------------------------------------------


def test_composer_cursor_stays_valid_after_paste_and_clear():
    composer = _composer_with("ab", PASTE_CHUNK, "cd")
    expected_end = 2 + len(PASTE_CHUNK) + 2
    assert composer.cursor_index == expected_end

    # Deleting text left of the caret pulls the caret back with it.
    composer.move_cursor_home()
    composer.delete_right()
    assert composer.draft_text() == "b" + PASTE_CHUNK + "cd"
    composer.move_cursor_end()
    assert composer.cursor_index == expected_end - 1

    composer.clear_draft()
    assert composer.cursor_index == 0
    assert composer.draft_text() == ""

    # Edits after a clear start from a clean caret.
    composer.insert_text("fresh")
    assert composer.draft_text() == "fresh"
    assert composer.cursor_index == 5


def test_composer_load_draft_restores_text_with_cursor_at_end():
    """Draft restore (session sync, `/prompt` handoffs) lands the caret at the end."""
    composer = ConsoleComposerBar()

    composer.load_draft("restored session draft")
    assert composer.draft_text() == "restored session draft"
    assert composer.cursor_index == len("restored session draft")

    # Typing after a restore appends at the restored caret position.
    composer.insert_text("!")
    assert composer.draft_text() == "restored session draft!"

    composer.load_draft("")
    assert composer.cursor_index == 0


def test_composer_select_all_then_typing_replaces_and_positions_cursor():
    composer = _composer_with("select me")

    assert composer.select_all_draft() is True
    assert composer.has_full_draft_selection()
    composer.insert_text("replacement")

    assert composer.draft_text() == "replacement"
    assert composer.cursor_index == len("replacement")
    assert not composer.has_full_draft_selection()


# ---------------------------------------------------------------------------
# Caret rendering (pure renderable tests)
# ---------------------------------------------------------------------------


def test_composer_renderable_places_cursor_glyph_at_cursor():
    renderable = ConsoleComposerBar._draft_renderable(
        "hello",
        width=80,
        focused=True,
        cursor_visible=True,
        cursor_index=2,
    )
    assert renderable.plain == "he▌llo"

    # The hidden blink phase reserves the same single cell mid-draft.
    hidden = ConsoleComposerBar._draft_renderable(
        "hello",
        width=80,
        focused=True,
        cursor_visible=False,
        cursor_index=2,
    )
    assert hidden.plain == "he llo"

    # No cursor index keeps the historical caret-at-tail behavior.
    tail = ConsoleComposerBar._draft_renderable(
        "hello",
        width=80,
        focused=True,
        cursor_visible=True,
    )
    assert tail.plain == "hello▌"


def test_composer_renderable_shifts_style_ranges_past_mid_draft_cursor():
    token_style = ConsoleComposerBar.PASTE_TOKEN_STYLE
    renderable = ConsoleComposerBar._draft_renderable(
        f"ab{PASTE_TOKEN}cd",
        width=200,
        focused=True,
        cursor_visible=True,
        cursor_index=4,
        style_ranges=[(2, 2 + len(PASTE_TOKEN), token_style)],
    )

    assert renderable.plain == f"ab{PASTE_TOKEN[:2]}▌{PASTE_TOKEN[2:]}cd"
    spans = [(span.start, span.end, str(span.style)) for span in renderable._spans]
    # The token span grows across the spliced caret cell instead of shifting.
    assert spans == [(2, 2 + len(PASTE_TOKEN) + 1, token_style)]


def test_composer_display_canonical_index_mapping_snaps_over_tokens():
    composer = _composer_with("ab", PASTE_CHUNK, "cd")
    token_display_end = 2 + len(PASTE_TOKEN)

    assert composer._canonical_index_at_display(0) == 0
    assert composer._canonical_index_at_display(2) == 2  # token start
    # Clicking a token snaps to its nearest canonical edge, never inside it.
    assert composer._canonical_index_at_display(3) == 2
    assert composer._canonical_index_at_display(token_display_end - 1) == 2 + len(
        PASTE_CHUNK
    )
    assert composer._canonical_index_at_display(token_display_end) == 2 + len(PASTE_CHUNK)
    assert composer._canonical_index_at_display(token_display_end + 1) == 2 + len(
        PASTE_CHUNK
    ) + 1

    # The reverse mapping renders the caret at the token's display edges.
    composer.move_cursor_home()
    assert composer._cursor_display_index() == 0
    composer.move_cursor_right()
    composer.move_cursor_right()
    composer.move_cursor_right()  # over the token
    assert composer._cursor_display_index() == token_display_end


# ---------------------------------------------------------------------------
# Screen-level key routing and click-to-position (pilot tests)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_console_composer_arrow_home_end_keys_move_caret_and_render_glyph():
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")

        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        visible_draft = composer.query_one("#console-command-visible-text", Static)
        composer.load_draft("hello")
        composer.focus()
        await pilot.pause(0.1)
        # De-flake: own every blink phase so the caret glyph stays visible.
        composer._cursor_blink_timer.pause()

        await pilot.press("left", "left")
        await pilot.pause(0.1)
        assert composer.cursor_index == 3
        assert "hel▌lo" in visible_draft.renderable.plain

        await pilot.press("home")
        await pilot.pause(0.1)
        assert composer.cursor_index == 0
        assert "▌hello" in visible_draft.renderable.plain

        await pilot.press("end")
        await pilot.pause(0.1)
        assert composer.cursor_index == 5
        assert "hello▌" in visible_draft.renderable.plain


@pytest.mark.asyncio
async def test_console_composer_typing_inserts_at_caret_mid_draft():
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")

        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("hell world")
        composer.focus()
        await pilot.pause(0.1)

        await pilot.press("left", "left", "left", "left", "left", "left")
        await pilot.press("o")
        await pilot.pause(0.1)

        assert composer.draft_text() == "hello world"
        assert composer.cursor_index == 5


@pytest.mark.asyncio
async def test_console_composer_ctrl_w_deletes_word_left_of_caret():
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")

        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("delete this word")
        composer.focus()
        await pilot.pause(0.1)

        await pilot.press("ctrl+w")
        await pilot.pause(0.1)

        assert composer.draft_text() == "delete this "
        assert composer.cursor_index == len("delete this ")


@pytest.mark.asyncio
async def test_console_composer_shift_enter_inserts_newline_enter_still_sends():
    gateway = CapturingGateway()
    app = _build_test_app()
    _configure_native_ready_console(app)
    app.console_provider_gateway_factory = lambda: gateway
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")

        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("line one")
        composer.focus()
        await pilot.pause(0.1)

        await pilot.press("shift+enter")
        await pilot.pause(0.1)
        assert composer.draft_text() == "line one\n"

        await pilot.press("l", "i", "n", "e", " ", "t", "w", "o")
        await pilot.pause(0.1)
        assert composer.draft_text() == "line one\nline two"
        assert not gateway.sent_messages

        await pilot.press("enter")
        await _wait_for_text(console, pilot, "accepted")

        assert gateway.sent_messages[-1][-1]["content"] == "line one\nline two"
        assert composer.draft_text() == ""
        assert composer.cursor_index == 0


@pytest.mark.asyncio
async def test_console_composer_click_positions_caret_in_literal_text():
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")

        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        visible_draft = composer.query_one("#console-command-visible-text", Static)
        composer.load_draft("click to place the caret")
        await pilot.pause(0.1)
        assert composer.cursor_index == len("click to place the caret")

        padding_left = getattr(visible_draft.styles.padding, "left", 0)
        target_column = padding_left + 5
        await pilot.click("#console-command-visible-text", offset=(target_column, 0))
        await pilot.pause(0.1)

        assert composer.cursor_index == 5
        # Typing after the click inserts where the caret landed.
        composer.insert_text("X")
        assert composer.draft_text() == "clickX to place the caret"


@pytest.mark.asyncio
async def test_console_composer_screen_coordinate_click_positions_caret():
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")

        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        visible_draft = composer.query_one("#console-command-visible-text", Static)
        composer.load_draft("absolute positioning check")
        await pilot.pause(0.1)

        visible_region = composer._screen_region(visible_draft)
        padding_left = getattr(visible_draft.styles.padding, "left", 0)
        assert composer.activate_visible_draft_screen_position(
            visible_region.x + padding_left + 9,
            visible_region.y,
        )
        await pilot.pause(0.1)

        assert composer.cursor_index == 9


@pytest.mark.asyncio
async def test_console_composer_click_on_paste_token_still_unfurls_not_positions():
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")

        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.insert_pasted_text(PASTE_CHUNK)
        cursor_before = composer.cursor_index
        await pilot.click("#console-command-visible-text", offset=(2, 0))
        await pilot.pause(0.1)

        # Token clicks keep the unfurl flow; the caret is not repositioned.
        assert composer.has_pending_paste_confirmation()
        assert composer.cursor_index == cursor_before


@pytest.mark.asyncio
async def test_console_prompt_insert_appends_at_end_regardless_of_caret():
    """The `/prompt` and Library handoffs keep their append-at-end contract."""
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")

        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("existing draft")
        composer.move_cursor_home()
        await pilot.pause(0.1)
        assert composer.cursor_index == 0

        assert console._insert_prompt_text_into_composer("resolved body", replace=False)

        assert composer.draft_text() == "existing draft\nresolved body"
        assert composer.cursor_index == len("existing draft\nresolved body")
