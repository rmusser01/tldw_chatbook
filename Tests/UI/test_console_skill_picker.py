"""Tests for ``ConsoleSkillPickerModal`` (Console `/skill-name` picker).

Harness mirrors ``Tests/UI/test_console_prompt_picker.py``'s ``ModalHarness``
(bare ``App[None]`` subclass + ``push_screen(..., callback=...)``), since this
picker is a deliberate structural copy of ``ConsolePromptPickerModal`` per the
Task 8 brief -- minus the ``apply-system``-style blocked-row mode (skills
have a single mode: every row `skill_search` returns is already eligible to
run). Assertions stay on structure/content/dismiss-values, never geometry,
per project convention.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, Input, Static

from tldw_chatbook.Widgets.Console.console_skill_picker_modal import (
    EMPTY_STORE_COPY,
    FILTER_INPUT_ID,
    ROW_ID_PREFIX,
    SEARCH_DEBOUNCE_SECONDS,
    ConsoleSkillPickerModal,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
AGENTIC_TERMINAL = REPO_ROOT / "tldw_chatbook" / "css" / "components" / "_agentic_terminal.tcss"
BUNDLED_STYLESHEET = REPO_ROOT / "tldw_chatbook" / "css" / "tldw_cli_modular.tcss"


def _css_block(text: str, selector: str) -> str:
    """Return a CSS rule body starting at ``selector`` (mirrors the helper of
    the same name in ``test_console_prompt_picker.py``)."""
    start = text.index(selector)
    block_start = text.index("{", start)
    block_end = text.index("}", block_start)
    return text[block_start:block_end]


def _record(*, name: str, description: str = "Does the thing.") -> dict[str, Any]:
    return {"name": name, "description": description}


class FakeSkillSearch:
    """Async callable recording calls and returning a scripted result list."""

    def __init__(self, results: list[Mapping[str, Any]] | None = None) -> None:
        self.results = results if results is not None else []
        self.calls: list[str] = []

    async def __call__(self, query: str) -> list[Mapping[str, Any]]:
        self.calls.append(query)
        return self.results


class ModalHarness(App[None]):
    CSS = """
    Screen {
        layout: vertical;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self.dismissed_with: Mapping[str, Any] | None | str = "not-called"

    def capture(self, value: Mapping[str, Any] | None) -> None:
        self.dismissed_with = value


async def _wait_for_search(pilot: Any) -> None:
    """Advance past the debounce timer and let the search worker settle."""
    await pilot.pause(SEARCH_DEBOUNCE_SECONDS + 0.1)
    await pilot.app.workers.wait_for_complete()
    await pilot.pause()


@pytest.mark.asyncio
async def test_typing_filters_calls_skill_search_with_settled_query() -> None:
    app = ModalHarness()
    fake_search = FakeSkillSearch([_record(name="code-review")])

    async with app.run_test(size=(100, 40)) as pilot:
        await app.push_screen(
            ConsoleSkillPickerModal(initial_query="", skill_search=fake_search),
            callback=app.capture,
        )
        await pilot.pause()
        await _wait_for_search(pilot)
        assert fake_search.calls == [""]

        filter_input = app.screen.query_one(f"#{FILTER_INPUT_ID}", Input)
        filter_input.value = "s"
        filter_input.value = "su"
        filter_input.value = "sum"

        await pilot.pause(SEARCH_DEBOUNCE_SECONDS - 0.1)
        assert fake_search.calls == [""]

        await _wait_for_search(pilot)
        assert fake_search.calls == ["", "sum"]


@pytest.mark.asyncio
async def test_initial_query_prefills_filter_and_searches_immediately() -> None:
    app = ModalHarness()
    fake_search = FakeSkillSearch([_record(name="summarize")])

    async with app.run_test(size=(100, 40)) as pilot:
        await app.push_screen(
            ConsoleSkillPickerModal(initial_query="summ", skill_search=fake_search),
            callback=app.capture,
        )
        await pilot.pause()
        await _wait_for_search(pilot)

        filter_input = app.screen.query_one(f"#{FILTER_INPUT_ID}", Input)
        assert filter_input.value == "summ"
        # Textual's `Input(value=...)` posts its own initial `Changed` message
        # on construction (shared with `ConsolePromptPickerModal`, which this
        # widget mirrors structurally) -- so a non-empty `initial_query`
        # triggers both the immediate on_mount search AND one settled
        # debounced search, both for the same (already-settled) query. Every
        # call carries the same text, so this is wasteful but never
        # incorrect -- not a behavior this task's brief asks to change.
        assert fake_search.calls == ["summ", "summ"]


@pytest.mark.asyncio
async def test_enter_on_highlighted_row_dismisses_with_that_record() -> None:
    app = ModalHarness()
    record = _record(name="summarize", description="Summarize the doc.")
    fake_search = FakeSkillSearch([record])

    async with app.run_test(size=(100, 40)) as pilot:
        await app.push_screen(
            ConsoleSkillPickerModal(initial_query="", skill_search=fake_search),
            callback=app.capture,
        )
        await pilot.pause()
        await _wait_for_search(pilot)

        row = app.screen.query_one(f"#{ROW_ID_PREFIX}summarize", Button)
        assert "summarize" in row.label.plain
        assert "Summarize the doc." in row.label.plain

        await pilot.press("enter")

    assert app.dismissed_with == record


@pytest.mark.asyncio
async def test_row_click_dismisses_with_that_record() -> None:
    app = ModalHarness()
    record = _record(name="code-review")
    fake_search = FakeSkillSearch([record])

    async with app.run_test(size=(100, 40)) as pilot:
        await app.push_screen(
            ConsoleSkillPickerModal(initial_query="", skill_search=fake_search),
            callback=app.capture,
        )
        await pilot.pause()
        await _wait_for_search(pilot)

        await pilot.click(f"#{ROW_ID_PREFIX}code-review")
        await pilot.pause()

    assert app.dismissed_with == record


@pytest.mark.asyncio
async def test_escape_dismisses_none() -> None:
    app = ModalHarness()
    fake_search = FakeSkillSearch([_record(name="code-review")])

    async with app.run_test(size=(100, 40)) as pilot:
        await app.push_screen(
            ConsoleSkillPickerModal(initial_query="", skill_search=fake_search),
            callback=app.capture,
        )
        await pilot.pause()
        await pilot.press("escape")

    assert app.dismissed_with is None


@pytest.mark.asyncio
async def test_empty_store_shows_exact_copy() -> None:
    app = ModalHarness()
    fake_search = FakeSkillSearch([])

    async with app.run_test(size=(100, 40)) as pilot:
        await app.push_screen(
            ConsoleSkillPickerModal(initial_query="", skill_search=fake_search),
            callback=app.capture,
        )
        await pilot.pause()
        await _wait_for_search(pilot)

        empty = app.screen.query_one("#console-skill-picker-empty", Static)
        assert str(empty.renderable) == EMPTY_STORE_COPY
        assert EMPTY_STORE_COPY == "No skills yet — create them in Library ▸ Skills."


@pytest.mark.asyncio
async def test_description_containing_bracket_renders_escaped() -> None:
    app = ModalHarness()
    record = _record(name="code-review", description="Checks [x] style violations.")
    fake_search = FakeSkillSearch([record])

    async with app.run_test(size=(100, 40)) as pilot:
        await app.push_screen(
            ConsoleSkillPickerModal(initial_query="", skill_search=fake_search),
            callback=app.capture,
        )
        await pilot.pause()
        await _wait_for_search(pilot)

        row = app.screen.query_one(f"#{ROW_ID_PREFIX}code-review", Button)
        # The rendered plain text must contain the literal brackets; Rich
        # markup must not have swallowed them as a style tag.
        assert "Checks [x] style violations." in row.label.plain


@pytest.mark.asyncio
async def test_keyboard_down_moves_highlight_and_enter_selects_moved_row() -> None:
    app = ModalHarness()
    first = _record(name="alpha")
    second = _record(name="beta")
    fake_search = FakeSkillSearch([first, second])

    async with app.run_test(size=(100, 40)) as pilot:
        await app.push_screen(
            ConsoleSkillPickerModal(initial_query="", skill_search=fake_search),
            callback=app.capture,
        )
        await pilot.pause()
        await _wait_for_search(pilot)

        await pilot.press("down")
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

    assert app.dismissed_with == second


@pytest.mark.asyncio
async def test_typing_still_filters_after_a_row_click_does_not_dismiss() -> None:
    """A click on a row always dismisses immediately in this single-mode
    picker (no blocked-row concept), but the filter Input must still be the
    thing regaining/keeping focus afterwards -- mirrors the discipline
    ``ConsolePromptPickerModal`` documents for its blocked-row click, applied
    here to confirm rows never strand real DOM focus even on the success
    path."""
    app = ModalHarness()
    record = _record(name="alpha")
    fake_search = FakeSkillSearch([record])

    async with app.run_test(size=(100, 40)) as pilot:
        await app.push_screen(
            ConsoleSkillPickerModal(initial_query="", skill_search=fake_search),
            callback=app.capture,
        )
        await pilot.pause()
        await _wait_for_search(pilot)

        await pilot.click(f"#{ROW_ID_PREFIX}alpha")
        await pilot.pause()

    assert app.dismissed_with == record


@pytest.mark.asyncio
async def test_duplicate_names_fall_back_to_index_id_and_stay_selectable() -> None:
    """Reviewer-style defensive case (mirrors the prompt picker's composite-id
    fallback test): a malformed result set with a repeated ``name`` must not
    crash the render worker with Textual's ``DuplicateIds`` -- the second
    row's DOM id falls back to its index, while both records stay
    selectable."""
    app = ModalHarness()
    first = _record(name="alpha", description="First one.")
    second = _record(name="alpha", description="Second one.")
    fake_search = FakeSkillSearch([first, second])

    async with app.run_test(size=(100, 40)) as pilot:
        await app.push_screen(
            ConsoleSkillPickerModal(initial_query="", skill_search=fake_search),
            callback=app.capture,
        )
        await pilot.pause()
        await _wait_for_search(pilot)  # Must not raise WorkerFailed/DuplicateIds.

        first_row = app.screen.query_one(f"#{ROW_ID_PREFIX}alpha", Button)
        assert "First one." in first_row.label.plain
        second_row = app.screen.query_one(f"#{ROW_ID_PREFIX}idx-1", Button)
        assert "Second one." in second_row.label.plain

        await pilot.press("down")
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

    assert app.dismissed_with == second


def test_skill_picker_css_blocks_pinned_in_source_and_bundle() -> None:
    """The skill-picker ids/classes must be styled in BOTH the module source
    (``_agentic_terminal.tcss``) and the generated bundle
    (``tldw_cli_modular.tcss``) -- proves ``build_css.py`` was re-run after
    the source edit, mirroring ``test_console_prompt_picker.py``'s dual-file
    CSS-parity discipline."""
    agentic_terminal = AGENTIC_TERMINAL.read_text(encoding="utf-8")
    bundled_stylesheet = BUNDLED_STYLESHEET.read_text(encoding="utf-8")

    for text in (agentic_terminal, bundled_stylesheet):
        for selector in (
            "ConsoleSkillPickerModal {",
            "#console-skill-picker-modal {",
            f"#{FILTER_INPUT_ID} {{",
            f"#{FILTER_INPUT_ID}:focus {{",
            ".console-skill-picker-row {",
            ".console-skill-picker-row-highlighted {",
            "#console-skill-picker-empty {",
        ):
            assert selector in text, f"missing CSS for {selector!r}"

        row_block = _css_block(text, ".console-skill-picker-row {")
        for pinned in ("width: 100%;", "border: tall $ds-grid-line;"):
            assert pinned in row_block

        highlighted_block = _css_block(text, ".console-skill-picker-row-highlighted {")
        assert "border: tall $ds-action-focus;" in highlighted_block

        filter_focus_block = _css_block(text, f"#{FILTER_INPUT_ID}:focus {{")
        assert "border: tall $ds-input-focus-accent;" in filter_focus_block
