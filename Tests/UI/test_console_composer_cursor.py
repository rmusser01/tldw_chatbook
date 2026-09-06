"""Console composer caret editing: movement, mid-draft edits, word delete,
Shift+Enter newline, click-to-position, and caret validity across mutations.

The composer's caret is an offset into the canonical draft text (the text that
will be sent); collapsed paste tokens are single units for movement, deletion,
and word boundaries.
"""

from pathlib import Path

import pytest
from rich.cells import cell_len
from textual.screen import ModalScreen
from textual.widgets import Static

from Tests.UI.test_console_native_chat_flow import (
    CapturingGateway,
    _build_console_send_test_app,
    _configure_native_ready_console,
    _wait_for_text,
)
from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.Chat.prompt_history import PromptHistory
from tldw_chatbook.Widgets.Console import ConsoleComposerBar


PASTE_CHUNK = "chunk of pasted console text " * 10
PASTE_TOKEN = f"Pasted text | {len(PASTE_CHUNK)} characters | Expand"


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


def test_cursor_split_preserves_paste_origin_and_typed_text_stays_literal():
    composer = ConsoleComposerBar(collapse_large_pastes=False)
    composer.insert_pasted_text("pasted")
    composer.move_cursor_home()
    for _ in range(3):
        composer.move_cursor_right()

    composer.insert_file_segment("SECRET", "notes.md · 6 B")
    composer.insert_text("typed")

    snapshot = composer.capture_draft_snapshot()
    assert [(segment.text, segment.origin) for segment in snapshot.segments] == [
        ("pas", "paste"),
        ("SECRET", "inline_file"),
        ("typed", "literal"),
        ("ted", "paste"),
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
    assert composer._canonical_index_at_display(token_display_end) == 2 + len(
        PASTE_CHUNK
    )
    assert (
        composer._canonical_index_at_display(token_display_end + 1)
        == 2 + len(PASTE_CHUNK) + 1
    )

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
    # TASK-21590: this is the one test in this module that drives a real send,
    # so it needs the durable persistence the shipping app always has.
    app = _build_console_send_test_app()
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

        assert console._commands._insert_prompt_text_into_composer("resolved body", replace=False)

        assert composer.draft_text() == "existing draft\nresolved body"
        assert composer.cursor_index == len("existing draft\nresolved body")


# ---------------------------------------------------------------------------
# TASK-21692: the blink tick must not arm a layout pass
# ---------------------------------------------------------------------------


class _CssTrueConsoleHarness(ConsoleHarness):
    """ConsoleHarness that loads the real app CSS bundle.

    The shared harness is a bare ``App`` -- none of the app's stylesheet
    applies under it, so geometry conclusions made there are void (see
    ``lessons-testing-evidence.md``). The blink-cost and blink-safety tests
    below both assert about real geometry, so they need the real sheet.
    """

    CSS_PATH = str(
        Path(__file__).resolve().parents[2]
        / "tldw_chatbook"
        / "css"
        / "tldw_cli_modular.tcss"
    )


async def _focused_composer(pilot, console, draft: str) -> ConsoleComposerBar:
    """Return a focused composer holding ``draft`` with the blink timer owned."""
    await _wait_for_selector(console, pilot, "#console-native-composer")
    composer = console.query_one("#console-native-composer", ConsoleComposerBar)
    if draft:
        composer.load_draft(draft)
    composer.focus()
    await pilot.pause(0.1)
    # Own the blink phase: the test drives ticks itself so the free-running
    # 0.53 s timer cannot add uncounted ticks mid-measurement.
    composer._cursor_blink_timer.pause()
    await pilot.pause(0.1)
    await pilot.pause()
    return composer


async def _count_layout_passes(pilot, composer, rounds: int, *, blink: bool) -> int:
    """Count real screen layout passes over ``rounds`` event-loop settles.

    Counts ``Screen._refresh_layout`` -- the call that reflows the whole
    compositor -- rather than asserting on ``refresh(layout=...)`` arguments,
    so the assertion is about work performed, not about how it was requested.
    """
    screen = composer.screen
    real_refresh_layout = screen._refresh_layout
    calls = 0

    def counting_refresh_layout(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real_refresh_layout(*args, **kwargs)

    screen._refresh_layout = counting_refresh_layout
    try:
        for _ in range(rounds):
            if blink:
                composer._toggle_cursor_blink()
            await pilot.pause()
            await pilot.pause()
    finally:
        del screen._refresh_layout
    return calls


@pytest.mark.asyncio
async def test_console_composer_blink_tick_arms_no_layout_pass():
    """A blink phase flip costs no more layout work than an idle settle.

    TASK-21692: ``Static.update`` defaults to ``layout=True``, so the 0.53 s
    cursor-blink tick used to schedule a full ``Screen._refresh_layout`` /
    ``Compositor.reflow`` ~2x/second for as long as the composer merely held
    focus -- exactly what ``_render_visible_draft_only``'s own docstring says
    must not happen.

    The idle arm is the noise floor (this harness measures 0, but asserting
    against the measured ambient rather than a bare 0 keeps the test from
    turning into a flake if some unrelated timer starts firing).
    """
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = _CssTrueConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        composer = await _focused_composer(pilot, console, "hello world")

        rounds = 6
        ambient = await _count_layout_passes(pilot, composer, rounds, blink=False)
        blinking = await _count_layout_passes(pilot, composer, rounds, blink=True)

        assert blinking == ambient, (
            f"{rounds} blink ticks cost {blinking - ambient} extra screen layout "
            f"passes (idle floor {ambient})"
        )

        # The tick must still REPAINT -- a caret that stopped blinking would
        # also score zero layout passes.
        visible = composer.query_one("#console-command-visible-text", Static)
        composer._cursor_visible = True
        composer._toggle_cursor_blink()
        await pilot.pause()
        hidden_text = visible.renderable.plain
        composer._toggle_cursor_blink()
        await pilot.pause()
        shown_text = visible.renderable.plain
        assert "▌" not in hidden_text
        assert "▌" in shown_text


@pytest.mark.asyncio
async def test_console_composer_blink_phases_are_geometry_identical():
    """Both blink phases occupy identical geometry, at every wrap boundary.

    This is the safety half of TASK-21692: ``layout=False`` is only sound
    while the rendered size genuinely cannot change between phases. The
    reserved caret cell (glyph when visible, space when hidden, wrapped in
    the same pass) is what guarantees that, so the boundary cases that would
    break it -- an empty draft, a draft filling the last cell, a draft
    landing exactly at the wrap width so the caret spills a row, and a
    double-width CJK draft -- are all exercised here.
    """
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = _CssTrueConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        composer = await _focused_composer(pilot, console, "seed")
        visible = composer.query_one("#console-command-visible-text", Static)
        width = composer._draft_render_width()
        assert width >= 8

        drafts = {
            "empty": "",
            "short": "hello",
            "fills-last-cell": "x" * (width - 1),
            "exactly-at-width": "x" * width,
            "one-past-width": "x" * (width + 1),
            "wrapped": "word " * 40,
            "cjk-double-width": "漢" * (width // 2),
        }

        for name, draft in drafts.items():
            composer.load_draft(draft)
            composer.focus()
            await pilot.pause(0.1)
            composer._cursor_blink_timer.pause()
            await pilot.pause()

            geometry = []
            cell_counts = []
            for _ in range(2):
                composer._toggle_cursor_blink()
                await pilot.pause()
                await pilot.pause()
                painted = visible.renderable.plain
                geometry.append(
                    (
                        visible.outer_size,
                        composer.outer_size,
                        len(painted.split("\n")),
                    )
                )
                cell_counts.append([cell_len(row) for row in painted.split("\n")])

            assert geometry[0] == geometry[1], (
                f"{name!r}: blink phases differ in size/row count -- "
                f"{geometry[0]} vs {geometry[1]}"
            )
            assert cell_counts[0] == cell_counts[1], (
                f"{name!r}: blink phases differ in painted cell widths -- "
                f"{cell_counts[0]} vs {cell_counts[1]}"
            )


# ---------------------------------------------------------------------------
# TASK-22218: the blink tick must not wrap the draft, scan history, or keep
# working under a modal
# ---------------------------------------------------------------------------


def _seeded_history(*inputs: str) -> PromptHistory:
    """Build a history store with in-memory entries only (no file IO)."""
    history = PromptHistory("/nonexistent/t22218_prompt_history.jsonl")
    history._entries = [{"input": text, "timestamp": 0.0} for text in inputs]
    history._loaded = True
    return history


async def _drive_ticks(pilot, composer, count: int) -> list[bool]:
    """Drive ``count`` blink ticks; return the phase seen after each tick."""
    phases: list[bool] = []
    for _ in range(count):
        composer._toggle_cursor_blink()
        await pilot.pause()
        phases.append(composer._cursor_visible)
    return phases


@pytest.mark.asyncio
async def test_console_composer_idle_blink_ticks_do_no_wrap_or_history_scan(
    monkeypatch,
):
    """A steady-state blink tick performs no draft wrap and no history scan.

    TASK-22218: before the render memo, every 0.53 s tick re-ran the full
    grapheme-aware ``cell_len`` wrap of the ENTIRE draft (a pasted 20 KB
    draft, re-wrapped ~1.89x/s forever) plus ``_ghost_suffix``'s linear
    ``startswith`` scan over up to 1000 history entries -- all to flip one
    caret cell. With the draft, width, and history unchanged, a tick must be
    a memo hit: zero wraps, zero scans.

    The two warm-up ticks are the memo filling its two blink phases -- the
    hidden phase genuinely has never been rendered for a fresh draft, so its
    first render is real work, once.
    """
    app = _build_test_app()
    history = _seeded_history(
        *[f"prompt number {index}" for index in range(999)],
        "word word word final entry",
    )
    app.console_prompt_history_factory = lambda: history
    _configure_native_ready_console(app)
    host = _CssTrueConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        draft = "word " * 4000  # 20,000 characters, wraps to many rows
        composer = await _focused_composer(pilot, console, draft)
        assert composer._prompt_history is history

        counts = {"wrap": 0, "scan": 0}
        real_wrap = ConsoleComposerBar._wrap_draft_line_slices.__func__

        def counting_wrap(cls, text, width):
            counts["wrap"] += 1
            return real_wrap(cls, text, width)

        monkeypatch.setattr(
            ConsoleComposerBar,
            "_wrap_draft_line_slices",
            classmethod(counting_wrap),
        )
        real_complete = history.complete

        def counting_complete(prefix):
            counts["scan"] += 1
            return real_complete(prefix)

        monkeypatch.setattr(history, "complete", counting_complete)

        # Warm-up: one tick per blink phase.
        await _drive_ticks(pilot, composer, 2)

        counts["wrap"] = 0
        counts["scan"] = 0
        rounds = 6
        await _drive_ticks(pilot, composer, rounds)
        assert counts == {"wrap": 0, "scan": 0}, (
            f"{rounds} idle blink ticks with an unchanged draft/width/history "
            f"performed {counts['wrap']} full-draft wraps and {counts['scan']} "
            f"history scans -- a tick must be a render-memo hit"
        )

        # The ticks must still repaint the caret -- a tick that stopped
        # rendering entirely would also score zero.
        visible = composer.query_one("#console-command-visible-text", Static)
        composer._cursor_visible = True
        composer._toggle_cursor_blink()
        await pilot.pause()
        hidden_text = visible.renderable.plain
        composer._toggle_cursor_blink()
        await pilot.pause()
        shown_text = visible.renderable.plain
        assert "▌" not in hidden_text
        assert "▌" in shown_text


class _ComposerCoverModal(ModalScreen[None]):
    """Bare modal used to cover the Console screen in blink-gate tests."""

    def compose(self):
        yield Static("covering modal", id="composer-cover-modal-body")


@pytest.mark.asyncio
async def test_console_composer_blink_freezes_solid_under_modal_and_resumes():
    """Blink ticks stop flipping while the composer's screen is covered.

    TASK-22218: the blink resume gate is ``has_focus_within``, which reads
    the composer's OWN screen's focus memory -- it survives ``push_screen``,
    so every modal left the caret blinking (and re-rendering) underneath.
    The tick now early-outs on ``not self.screen.is_active`` (the TASK-22219
    shape: the timer keeps ticking and IS the resume path), parking the
    caret solid; the first tick after the modal pops blinks again.
    """
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = _CssTrueConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        composer = await _focused_composer(pilot, console, "hello world")

        host.push_screen(_ComposerCoverModal())
        await pilot.pause()
        await pilot.pause()
        assert not console.is_active
        # The pre-existing trap this test pins: per-screen focus memory
        # keeps the old resume gate armed underneath the modal.
        assert composer.has_focus_within

        phases = await _drive_ticks(pilot, composer, 6)
        assert phases == [True] * 6, (
            f"blink kept flipping under a modal: phases {phases} -- the tick "
            f"must park the caret solid while the composer's screen is covered"
        )

        host.pop_screen()
        await pilot.pause()
        await pilot.pause()
        assert console.is_active
        resumed = await _drive_ticks(pilot, composer, 2)
        assert False in resumed, (
            f"blink did not resume after the modal popped: phases {resumed}"
        )


@pytest.mark.asyncio
async def test_console_composer_typing_after_idle_ticks_repaints_new_draft():
    """The render memo invalidates on a draft edit -- no stale caret/text.

    Guard for TASK-22218's memoization: after idle ticks have filled both
    blink-phase memo slots, a typed character must repaint with the new
    draft text and the caret after it (a memo keyed without the draft would
    serve the stale renderable forever).
    """
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = _CssTrueConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        composer = await _focused_composer(pilot, console, "hello")
        visible = composer.query_one("#console-command-visible-text", Static)

        # Fill both phase slots, ending on the solid phase.
        composer._cursor_visible = True
        await _drive_ticks(pilot, composer, 2)
        assert "hello▌" in visible.renderable.plain

        composer.insert_text("!")
        await pilot.pause()
        assert "hello!▌" in visible.renderable.plain

        # And the next ticks keep blinking the caret against the new draft.
        composer._cursor_visible = True
        await _drive_ticks(pilot, composer, 1)
        assert "hello!" in visible.renderable.plain
        assert "▌" not in visible.renderable.plain


@pytest.mark.asyncio
async def test_console_composer_history_append_while_idle_updates_ghost(tmp_path):
    """A history record while the composer idles invalidates the ghost text.

    Guard for TASK-22218's memo key: the ghost suffix is part of the memoized
    OUTPUT, so a new history entry recorded while the composer sits idle
    (e.g. a queued send completing) must reach the next blink tick via the
    history revision in the memo key -- not be served stale until the next
    keystroke.
    """
    app = _build_test_app()
    # A real writable path: `append` below must succeed, not roll back.
    history = PromptHistory(tmp_path / "prompt_history.jsonl")
    history._entries = [{"input": "hello world", "timestamp": 0.0}]
    history._loaded = True
    app.console_prompt_history_factory = lambda: history
    _configure_native_ready_console(app)
    host = _CssTrueConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        composer = await _focused_composer(pilot, console, "hello")
        assert composer._prompt_history is history
        visible = composer.query_one("#console-command-visible-text", Static)

        # Warm both phases; land on the solid phase showing the ghost.
        composer._cursor_visible = True
        await _drive_ticks(pilot, composer, 2)
        assert "hello▌ world" in visible.renderable.plain

        # Record a newer entry the way production does (append bumps the
        # store's revision; most-recent-wins changes the suggestion).
        recorded = await history.append("hello brave")
        assert recorded is True

        composer._cursor_visible = False
        await _drive_ticks(pilot, composer, 1)
        assert "hello▌ brave" in visible.renderable.plain, (
            f"ghost text served stale after a history record: "
            f"{visible.renderable.plain!r}"
        )
