"""Tests for ``ConsolePromptPickerModal`` (Console `/prompt` and `/system` picker).

Harness mirrors ``Tests/UI/test_console_session_settings.py``'s ``ModalHarness``
(bare ``App[None]`` subclass + ``push_screen(..., callback=...)``) since no
dedicated Console modal test file exists for this pattern yet. Assertions stay
on structure/content/dismiss-values, never geometry, per project convention.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, Input, Static

from tldw_chatbook.Widgets.Console.console_prompt_picker_modal import (
    EMPTY_STORE_COPY,
    FILTER_INPUT_ID,
    MODE_APPLY_SYSTEM,
    MODE_INSERT,
    NO_SYSTEM_PART_SUFFIX,
    REASON_STATIC_ID,
    ROW_ID_PREFIX,
    SEARCH_DEBOUNCE_SECONDS,
    ConsolePromptPickerModal,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
AGENTIC_TERMINAL = REPO_ROOT / "tldw_chatbook" / "css" / "components" / "_agentic_terminal.tcss"
BUNDLED_STYLESHEET = REPO_ROOT / "tldw_chatbook" / "css" / "tldw_cli_modular.tcss"


def _css_block(text: str, selector: str) -> str:
    """Return a CSS rule body starting at ``selector`` (mirrors the helper of
    the same name in ``test_library_prompts_canvas.py``)."""
    start = text.index(selector)
    block_start = text.index("{", start)
    block_end = text.index("}", block_start)
    return text[block_start:block_end]


def _record(
    *,
    local_id: int,
    name: str,
    system_prompt: str | None = "You are helpful.",
    user_prompt: str = "Do the thing.",
    keywords: tuple[str, ...] = (),
) -> dict[str, Any]:
    return {
        "id": f"local:prompt:{local_id}",
        "local_id": local_id,
        "name": name,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "keywords": list(keywords),
    }


class FakePromptSearch:
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
async def test_typing_filters_calls_prompt_search_with_settled_query() -> None:
    app = ModalHarness()
    fake_search = FakePromptSearch([_record(local_id=1, name="Alpha")])

    async with app.run_test(size=(100, 40)) as pilot:
        await app.push_screen(
            ConsolePromptPickerModal(
                mode=MODE_INSERT,
                initial_query="",
                prompt_search=fake_search,
            ),
            callback=app.capture,
        )
        await pilot.pause()
        await _wait_for_search(pilot)
        assert fake_search.calls == [""]

        filter_input = app.screen.query_one(f"#{FILTER_INPUT_ID}", Input)
        filter_input.value = "s"
        filter_input.value = "sa"
        filter_input.value = "sam"

        await pilot.pause(SEARCH_DEBOUNCE_SECONDS - 0.1)
        assert fake_search.calls == [""]

        await _wait_for_search(pilot)
        assert fake_search.calls == ["", "sam"]


@pytest.mark.asyncio
async def test_enter_on_highlighted_row_dismisses_with_that_record() -> None:
    app = ModalHarness()
    record = _record(local_id=7, name="Summarize")
    fake_search = FakePromptSearch([record])

    async with app.run_test(size=(100, 40)) as pilot:
        await app.push_screen(
            ConsolePromptPickerModal(
                mode=MODE_INSERT,
                initial_query="",
                prompt_search=fake_search,
            ),
            callback=app.capture,
        )
        await pilot.pause()
        await _wait_for_search(pilot)

        row = app.screen.query_one(f"#{ROW_ID_PREFIX}7", Button)
        assert "Summarize" in str(row.label)

        await pilot.press("enter")

    assert app.dismissed_with == record


@pytest.mark.asyncio
async def test_apply_system_mode_blocks_empty_system_row_without_dismissing() -> None:
    app = ModalHarness()
    record = _record(local_id=3, name="No System", system_prompt="")
    fake_search = FakePromptSearch([record])

    async with app.run_test(size=(100, 40)) as pilot:
        await app.push_screen(
            ConsolePromptPickerModal(
                mode=MODE_APPLY_SYSTEM,
                initial_query="",
                prompt_search=fake_search,
            ),
            callback=app.capture,
        )
        await pilot.pause()
        await _wait_for_search(pilot)

        row = app.screen.query_one(f"#{ROW_ID_PREFIX}3", Button)
        assert NO_SYSTEM_PART_SUFFIX in str(row.label)
        assert row.has_class("console-prompt-picker-row-blocked")

        await pilot.click(f"#{ROW_ID_PREFIX}3")
        await pilot.pause()

        reason = app.screen.query_one(f"#{REASON_STATIC_ID}", Static)
        assert reason.display is True
        assert str(reason.renderable).strip() != ""

    # Dismiss callback must never have fired.
    assert app.dismissed_with == "not-called"


@pytest.mark.asyncio
async def test_escape_dismisses_none() -> None:
    app = ModalHarness()
    fake_search = FakePromptSearch([_record(local_id=1, name="Alpha")])

    async with app.run_test(size=(100, 40)) as pilot:
        await app.push_screen(
            ConsolePromptPickerModal(
                mode=MODE_INSERT,
                initial_query="",
                prompt_search=fake_search,
            ),
            callback=app.capture,
        )
        await pilot.pause()
        await pilot.press("escape")

    assert app.dismissed_with is None


@pytest.mark.asyncio
async def test_empty_store_shows_exact_copy() -> None:
    app = ModalHarness()
    fake_search = FakePromptSearch([])

    async with app.run_test(size=(100, 40)) as pilot:
        await app.push_screen(
            ConsolePromptPickerModal(
                mode=MODE_INSERT,
                initial_query="",
                prompt_search=fake_search,
            ),
            callback=app.capture,
        )
        await pilot.pause()
        await _wait_for_search(pilot)

        empty = app.screen.query_one("#console-prompt-picker-empty", Static)
        assert str(empty.renderable) == EMPTY_STORE_COPY


@pytest.mark.asyncio
async def test_bracket_name_renders_escaped() -> None:
    app = ModalHarness()
    record = _record(local_id=9, name="[draft] Q3 plan [wip]")
    fake_search = FakePromptSearch([record])

    async with app.run_test(size=(100, 40)) as pilot:
        await app.push_screen(
            ConsolePromptPickerModal(
                mode=MODE_INSERT,
                initial_query="",
                prompt_search=fake_search,
            ),
            callback=app.capture,
        )
        await pilot.pause()
        await _wait_for_search(pilot)

        row = app.screen.query_one(f"#{ROW_ID_PREFIX}9", Button)
        # The rendered plain text must contain the literal brackets; Rich
        # markup must not have swallowed them as a style tag.
        assert "[draft] Q3 plan [wip]" in row.label.plain


@pytest.mark.asyncio
async def test_keyboard_nav_survives_a_blocked_row_click() -> None:
    """Reviewer finding: clicking a row (the apply-system BLOCK-REFUSAL click
    is the only row interaction that keeps the modal open) must not strand
    real DOM focus on that row's Button. If it did, Enter would re-trigger
    the *stale focused* row's own built-in ``enter -> press`` binding instead
    of the row the synthetic highlight has since moved to via Down -- a
    silent desync between what's visually highlighted and what Enter
    actually selects."""
    app = ModalHarness()
    blocked = _record(local_id=1, name="Alpha", system_prompt="")
    unblocked = _record(local_id=2, name="Beta", system_prompt="You are helpful.")
    fake_search = FakePromptSearch([blocked, unblocked])

    async with app.run_test(size=(100, 40)) as pilot:
        await app.push_screen(
            ConsolePromptPickerModal(
                mode=MODE_APPLY_SYSTEM,
                initial_query="",
                prompt_search=fake_search,
            ),
            callback=app.capture,
        )
        await pilot.pause()
        await _wait_for_search(pilot)

        # Block-refusal click: modal stays open, but (pre-fix) strands real
        # focus on row 1's Button.
        await pilot.click(f"#{ROW_ID_PREFIX}1")
        await pilot.pause()
        assert app.dismissed_with == "not-called"

        # Down moves the synthetic highlight onto row 2 (Beta, unblocked).
        await pilot.press("down")
        await pilot.pause()

        # Enter must select the now-highlighted row (Beta) -- not silently
        # re-press whatever row last held stale DOM focus.
        await pilot.press("enter")
        await pilot.pause()

    assert app.dismissed_with == unblocked


@pytest.mark.asyncio
async def test_typing_still_filters_after_a_blocked_row_click() -> None:
    """Reviewer finding: after the same block-refusal click above, typed
    characters must keep reaching the filter Input (pre-fix, focus is
    stranded on the clicked row's Button, so keystrokes go nowhere and
    prompt_search is never called again)."""
    app = ModalHarness()
    blocked = _record(local_id=3, name="No System", system_prompt="")
    fake_search = FakePromptSearch([blocked])

    async with app.run_test(size=(100, 40)) as pilot:
        await app.push_screen(
            ConsolePromptPickerModal(
                mode=MODE_APPLY_SYSTEM,
                initial_query="",
                prompt_search=fake_search,
            ),
            callback=app.capture,
        )
        await pilot.pause()
        await _wait_for_search(pilot)
        assert fake_search.calls == [""]

        await pilot.click(f"#{ROW_ID_PREFIX}3")
        await pilot.pause()
        assert app.dismissed_with == "not-called"

        filter_input = app.screen.query_one(f"#{FILTER_INPUT_ID}", Input)
        assert filter_input.has_focus, "filter Input must regain focus after a blocked-row refusal"

        await pilot.press("x")
        await pilot.pause()
        assert filter_input.value == "x"

        await _wait_for_search(pilot)
        assert fake_search.calls == ["", "x"]


@pytest.mark.asyncio
async def test_composite_string_id_falls_back_to_index_and_is_selectable() -> None:
    """Reviewer finding: a record whose only id is a composite string (e.g.
    the Task 6 normalization's ``"local:prompt:7"``) with no ``local_id``
    must not crash the render worker with Textual's ``BadIdentifier`` (colon
    is an illegal widget-id character) -- the row's DOM id must fall back to
    something legal (its index), while the record itself stays selectable."""
    app = ModalHarness()
    record = {
        "id": "local:prompt:7",
        "name": "Composite Id Only",
        "system_prompt": "You are helpful.",
        "user_prompt": "Do the thing.",
        "keywords": [],
    }
    fake_search = FakePromptSearch([record])

    async with app.run_test(size=(100, 40)) as pilot:
        await app.push_screen(
            ConsolePromptPickerModal(
                mode=MODE_INSERT,
                initial_query="",
                prompt_search=fake_search,
            ),
            callback=app.capture,
        )
        await pilot.pause()
        await _wait_for_search(pilot)  # Must not raise WorkerFailed/BadIdentifier.

        row = app.screen.query_one(f"#{ROW_ID_PREFIX}0", Button)
        assert "Composite Id Only" in str(row.label)

        await pilot.press("enter")

    assert app.dismissed_with == record


def test_prompt_picker_css_blocks_pinned_in_source_and_bundle() -> None:
    """The prompt-picker ids/classes must be styled in BOTH the module source
    (``_agentic_terminal.tcss``) and the generated bundle
    (``tldw_cli_modular.tcss``) -- proves ``build_css.py`` was re-run after
    the source edit, mirroring ``test_library_prompts_canvas.py``'s dual-file
    CSS-parity discipline for this feature branch."""
    agentic_terminal = AGENTIC_TERMINAL.read_text(encoding="utf-8")
    bundled_stylesheet = BUNDLED_STYLESHEET.read_text(encoding="utf-8")

    for text in (agentic_terminal, bundled_stylesheet):
        for selector in (
            "ConsolePromptPickerModal {",
            "#console-prompt-picker-modal {",
            f"#{FILTER_INPUT_ID} {{",
            f"#{FILTER_INPUT_ID}:focus {{",
            ".console-prompt-picker-row {",
            ".console-prompt-picker-row-highlighted {",
            ".console-prompt-picker-row-blocked {",
            f"#{REASON_STATIC_ID} {{",
            "#console-prompt-picker-empty {",
        ):
            assert selector in text, f"missing CSS for {selector!r}"

        row_block = _css_block(text, ".console-prompt-picker-row {")
        for pinned in ("width: 100%;", "border: tall $ds-grid-line;"):
            assert pinned in row_block

        highlighted_block = _css_block(text, ".console-prompt-picker-row-highlighted {")
        assert "border: tall $ds-action-focus;" in highlighted_block

        blocked_block = _css_block(text, ".console-prompt-picker-row-blocked {")
        assert "color: $ds-text-muted;" in blocked_block

        filter_focus_block = _css_block(text, f"#{FILTER_INPUT_ID}:focus {{")
        assert "border: tall $ds-input-focus-accent;" in filter_focus_block
