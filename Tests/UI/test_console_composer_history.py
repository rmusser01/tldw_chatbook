"""Console composer prompt-history UX (TASK-1364): ghost text and recall.

Covers the toad-inspired input UX ported onto the composer's segment/caret
model:

- Ghost text: most-recent-wins history prefix match rendered as a dimmed
  suffix after the caret; offered only on the live draft with an empty
  selection, the caret at end, and a draft that does not start with ``/``
  (the slash-command popup owns completion there). Right-arrow accepts.
- Recall: Up/Down walk history only when the caret sits on the first/last
  visual row of the wrapped draft; index 0 is the live draft, stashed while
  navigating and restored on return; clamped at the oldest entry.
"""

import json

import pytest
from textual.widgets import Static

from Tests.UI.test_console_native_chat_flow import (
    _configure_native_ready_console,
)
from Tests.UI.app_factory import attach_chachanotes_db
from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.Chat.prompt_history import PromptHistory
from tldw_chatbook.Widgets.Console import ConsoleComposerBar


def _seeded_history(*inputs: str) -> PromptHistory:
    """Build a history store with in-memory entries only (no file IO)."""
    history = PromptHistory("/nonexistent/prompt_history.jsonl")
    history._entries = [{"input": text, "timestamp": 0.0} for text in inputs]
    history._loaded = True
    return history


def _composer_with_history(*inputs: str) -> ConsoleComposerBar:
    """Build an unmounted composer wired to a seeded history store."""
    composer = ConsoleComposerBar()
    composer.set_prompt_history(_seeded_history(*inputs))
    return composer


# ---------------------------------------------------------------------------
# Ghost-text suffix computation (unmounted segment-model tests)
# ---------------------------------------------------------------------------


def test_ghost_suffix_offers_most_recent_prefix_match():
    composer = _composer_with_history("fix the bug", "find it", "fix the tests")
    composer.insert_text("fi")
    assert composer._ghost_suffix() == "x the tests"

    composer.clear_draft()
    composer.insert_text("fin")
    assert composer._ghost_suffix() == "d it"


def test_ghost_suffix_requires_caret_at_end():
    composer = _composer_with_history("hello world")
    composer.insert_text("hel")
    assert composer._ghost_suffix() == "lo world"
    composer.move_cursor_left()
    assert composer._ghost_suffix() == ""
    composer.move_cursor_end()
    assert composer._ghost_suffix() == "lo world"


def test_ghost_suffix_suppressed_for_slash_command_drafts():
    """The slash-command popup owns completion for '/'-prefixed drafts."""
    composer = _composer_with_history("/help me with this")
    composer.insert_text("/he")
    assert composer._ghost_suffix() == ""


def test_ghost_suffix_suppressed_with_selection_and_empty_draft():
    composer = _composer_with_history("hello world")
    assert composer._ghost_suffix() == ""  # empty draft

    composer.insert_text("hel")
    composer.select_all_draft()
    assert composer._ghost_suffix() == ""  # selection active


def test_ghost_suffix_suppressed_while_navigating_history():
    composer = _composer_with_history("hello world")
    composer.insert_text("hel")
    composer._history_index = -1
    assert composer._ghost_suffix() == ""


def test_ghost_suffix_exact_match_offers_nothing():
    composer = _composer_with_history("hello")
    composer.insert_text("hello")
    assert composer._ghost_suffix() == ""


def test_ghost_suffix_without_history_store_is_empty():
    composer = ConsoleComposerBar()
    composer.insert_text("hel")
    assert composer._ghost_suffix() == ""


# ---------------------------------------------------------------------------
# Ghost-text rendering (the composer's own render pass)
# ---------------------------------------------------------------------------


def test_draft_renderable_renders_dimmed_ghost_after_caret():
    rendered = ConsoleComposerBar._draft_renderable(
        "hel",
        width=40,
        focused=True,
        cursor_visible=True,
        cursor_index=3,
        ghost_suffix="lo world",
    )
    assert rendered.plain == "hel▌lo world"
    ghost_spans = [
        span for span in rendered.spans if span.style == ConsoleComposerBar.GHOST_TEXT_STYLE
    ]
    assert len(ghost_spans) == 1
    assert (ghost_spans[0].start, ghost_spans[0].end) == (4, 12)


def test_draft_renderable_ghost_only_at_caret_end_and_focused():
    # Caret mid-draft: the ghost suffix is dropped entirely.
    rendered = ConsoleComposerBar._draft_renderable(
        "hello",
        width=40,
        focused=True,
        cursor_visible=True,
        cursor_index=2,
        ghost_suffix=" world",
    )
    assert rendered.plain == "he▌llo"

    # Unfocused: no caret, no ghost.
    rendered = ConsoleComposerBar._draft_renderable(
        "hel",
        width=40,
        focused=False,
        ghost_suffix="lo world",
    )
    assert rendered.plain == "hel"


def test_draft_renderable_ghost_wraps_without_affecting_height_math():
    """A long ghost shares the wrap pass but never grows the row count."""
    draft = "hel"
    ghost = "lo " + "word " * 30
    rendered = ConsoleComposerBar._draft_renderable(
        draft,
        width=40,
        focused=True,
        cursor_visible=True,
        cursor_index=3,
        ghost_suffix=ghost,
    )
    assert rendered.plain.splitlines()[0].startswith("hel▌lo")
    # The composer's own height math only ever sees the draft, not the ghost.
    assert ConsoleComposerBar._visible_draft_row_count(
        draft, 40, reserve_trailing_cell=True
    ) == 1


# ---------------------------------------------------------------------------
# Right-arrow acceptance
# ---------------------------------------------------------------------------


def test_accept_ghost_text_inserts_suffix_and_moves_caret_to_end():
    composer = _composer_with_history("hello world")
    composer.insert_text("hello w")

    assert composer.accept_ghost_text() is True
    assert composer.draft_text() == "hello world"
    assert composer.cursor_index == len("hello world")
    # The completed draft now matches the entry exactly -- nothing left to
    # suggest, so a second accept is a no-op (caret movement resumes).
    assert composer.accept_ghost_text() is False


def test_accept_ghost_text_without_suggestion_is_noop():
    composer = _composer_with_history("hello world")
    composer.insert_text("zzz")
    assert composer.accept_ghost_text() is False
    assert composer.draft_text() == "zzz"
    assert composer.cursor_index == 3


# ---------------------------------------------------------------------------
# Recall gating: first/last visual row of the wrapped draft
# ---------------------------------------------------------------------------


def test_recall_gating_single_row_draft():
    composer = _composer_with_history("older prompt")
    composer.insert_text("short")
    # A single-row draft is simultaneously the first and last visual row.
    assert composer._can_recall_history(-1) is True
    # Down only recalls while navigating (index < 0); at the live draft it
    # falls through to ordinary caret movement.
    assert composer._can_recall_history(1) is False


def test_recall_gating_wrapped_draft_rows():
    composer = _composer_with_history("older prompt")
    # Unmounted composers wrap at FALLBACK_DRAFT_WIDTH (80 cells).
    composer.insert_text("x" * 100)  # wraps to 2 visual rows
    # Caret at end -> last row: Up moves the caret, no recall.
    assert composer._can_recall_history(-1) is False
    composer.move_cursor_home()  # first row
    assert composer._can_recall_history(-1) is True


def test_recall_gating_middle_row_recalls_neither_direction():
    composer = _composer_with_history("older prompt", "newer prompt")
    composer.insert_text("x" * 200)  # wraps to 3 visual rows
    composer._move_cursor_to(100)  # middle row
    assert composer._can_recall_history(-1) is False
    composer._history_index = -1  # pretend we are navigating
    assert composer._can_recall_history(1) is False
    composer.move_cursor_end()  # last row
    assert composer._can_recall_history(1) is True
    composer.move_cursor_home()  # first row
    assert composer._can_recall_history(1) is False


def test_recall_gating_requires_history_and_empty_selection():
    composer = ConsoleComposerBar()
    composer.insert_text("short")
    assert composer._can_recall_history(-1) is False  # no store injected

    composer.set_prompt_history(_seeded_history())
    assert composer._can_recall_history(-1) is False  # empty history

    composer.set_prompt_history(_seeded_history("older prompt"))
    composer.select_all_draft()
    assert composer._can_recall_history(-1) is False  # selection active


# ---------------------------------------------------------------------------
# Recall navigation: stash/restore round-trip and clamping
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_move_history_stashes_and_restores_live_draft():
    composer = _composer_with_history("first prompt", "second prompt")
    composer.insert_text("wip draft")
    history = composer._prompt_history

    await composer._move_history(-1)
    assert composer.draft_text() == "second prompt"
    assert composer._history_index == -1
    assert composer.cursor_index == len("second prompt")
    assert history.current == "wip draft"  # live draft stashed

    await composer._move_history(-1)
    assert composer.draft_text() == "first prompt"
    assert composer._history_index == -2

    # Clamped at the oldest entry: no-op, index unchanged.
    await composer._move_history(-1)
    assert composer.draft_text() == "first prompt"
    assert composer._history_index == -2

    await composer._move_history(1)
    assert composer.draft_text() == "second prompt"
    assert composer._history_index == -1

    # Back at the live draft: the stashed in-progress text is restored.
    await composer._move_history(1)
    assert composer.draft_text() == "wip draft"
    assert composer._history_index == 0
    assert composer.cursor_index == len("wip draft")


@pytest.mark.asyncio
async def test_move_history_repeated_stash_leaving_live_draft_once():
    """Re-navigating from the live draft re-stashes the CURRENT text."""
    composer = _composer_with_history("older prompt")
    composer.insert_text("wip one")
    history = composer._prompt_history

    await composer._move_history(-1)
    await composer._move_history(1)
    assert composer.draft_text() == "wip one"

    composer.insert_text(" v2")
    await composer._move_history(-1)
    assert history.current == "wip one v2"
    await composer._move_history(1)
    assert composer.draft_text() == "wip one v2"


def test_clear_history_resets_recall_index_to_live_draft():
    """An accepted send (the clear_history barrier) ends history navigation."""
    composer = _composer_with_history("older prompt")
    composer.insert_text("wip")
    composer._history_index = -1
    composer.clear_history()
    assert composer._history_index == 0


# ---------------------------------------------------------------------------
# Screen-level key routing (pilot tests)
# ---------------------------------------------------------------------------


def _seed_history_file(path, *inputs: str) -> None:
    with open(path, "w", encoding="utf-8") as history_file:
        for index, text in enumerate(inputs):
            history_file.write(
                json.dumps({"input": text, "timestamp": float(index)}) + "\n"
            )


@pytest.mark.asyncio
async def test_console_ghost_text_renders_and_right_arrow_accepts(
    tmp_path, monkeypatch
):
    history_path = tmp_path / "prompt_history.jsonl"
    _seed_history_file(history_path, "explain quantum computing")
    monkeypatch.setattr(
        "tldw_chatbook.UI.Console_Modules.prompts.default_prompt_history_path",
        lambda: history_path,
    )
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")

        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        visible_draft = composer.query_one("#console-command-visible-text", Static)
        # The screen shares ONE history store between composer and controller.
        assert composer._prompt_history is console._prompts._ensure_console_prompt_history()
        composer.focus()
        await pilot.pause(0.3)  # let the mount-time history load land
        composer._cursor_blink_timer.pause()

        composer.insert_text("explain qu")
        await pilot.pause(0.1)
        assert "explain qu▌antum computing" in visible_draft.renderable.plain

        # Typing dismisses the ghost implicitly (prefix no longer matches).
        await pilot.press("x")
        await pilot.pause(0.1)
        assert "antum computing" not in visible_draft.renderable.plain
        composer.delete_left()
        await pilot.pause(0.1)
        assert "explain qu▌antum computing" in visible_draft.renderable.plain

        await pilot.press("right")
        await pilot.pause(0.1)
        assert composer.draft_text() == "explain quantum computing"
        assert composer.cursor_index == len("explain quantum computing")

        # Right at the end with no suggestion falls back to a (no-op) move.
        await pilot.press("right")
        await pilot.pause(0.1)
        assert composer.cursor_index == len("explain quantum computing")


@pytest.mark.asyncio
async def test_console_up_down_recall_gated_to_boundary_rows(tmp_path, monkeypatch):
    history_path = tmp_path / "prompt_history.jsonl"
    _seed_history_file(history_path, "first prompt", "second prompt")
    monkeypatch.setattr(
        "tldw_chatbook.UI.Console_Modules.prompts.default_prompt_history_path",
        lambda: history_path,
    )
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")

        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.focus()
        await pilot.pause(0.3)
        composer._cursor_blink_timer.pause()

        # Two-row draft, caret at end (last row): Up moves the caret, it
        # does NOT recall.
        composer.load_draft("line one\nline two")
        await pilot.pause(0.1)
        await pilot.press("up")
        await pilot.pause(0.1)
        assert composer.draft_text() == "line one\nline two"
        # Caret moved up one row at the same column (row 0, col 8).
        assert composer.cursor_index == 8
        assert composer._caret_visual_row() == (0, 2)

        # Now on the first visual row: Up recalls older history.
        await pilot.press("up")
        await pilot.pause(0.2)
        assert composer.draft_text() == "second prompt"
        await pilot.press("up")
        await pilot.pause(0.2)
        assert composer.draft_text() == "first prompt"
        # Clamped at the oldest entry.
        await pilot.press("up")
        await pilot.pause(0.2)
        assert composer.draft_text() == "first prompt"

        # Down walks back; the stashed live draft returns at index 0.
        await pilot.press("down")
        await pilot.pause(0.2)
        assert composer.draft_text() == "second prompt"
        await pilot.press("down")
        await pilot.pause(0.2)
        assert composer.draft_text() == "line one\nline two"
        assert composer.cursor_index == len("line one\nline two")


@pytest.mark.asyncio
async def test_console_send_records_to_shared_prompt_history(tmp_path, monkeypatch):
    """An accepted send lands once in the JSONL store the composer reads."""
    from Tests.UI.test_console_native_chat_flow import CapturingGateway

    history_path = tmp_path / "prompt_history.jsonl"
    monkeypatch.setattr(
        "tldw_chatbook.UI.Console_Modules.prompts.default_prompt_history_path",
        lambda: history_path,
    )
    gateway = CapturingGateway()
    app = _build_test_app()
    attach_chachanotes_db(app)
    _configure_native_ready_console(app)
    app.console_provider_gateway_factory = lambda: gateway
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")

        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.focus()
        await pilot.pause(0.3)

        composer.load_draft("record this send")
        await pilot.pause(0.1)
        await pilot.press("enter")
        for _ in range(50):
            await pilot.pause(0.1)
            if history_path.exists() and history_path.read_text().strip():
                break

        lines = history_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0])["input"] == "record this send"

        # The accepted send also ended history navigation at the live draft.
        assert composer._history_index == 0
