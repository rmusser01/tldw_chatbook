"""Tests for the Library prompts list canvas widget and its screen wiring.

Widget-only tests mount ``LibraryPromptsListCanvas`` directly in a bare
``App`` subclass (mirrors ``test_library_export_cancel.py``'s
``test_cancel_button_visible_only_while_running``) -- this harness has no
app CSS loaded, so assertions stick to structure/content, never geometry
(Horizontal's own `layout: horizontal` is baked into the Textual widget
class itself -- not the app's custom stylesheet -- but pixel/region
assertions are still avoided here per the "no geometry assertions" rule;
"one row" is instead proven structurally via shared Horizontal parentage).

Screen-wiring tests call ``LibraryScreen`` bound methods directly against a
``SimpleNamespace`` stand-in for ``self`` (mirrors
``test_library_export_cancel.py``'s direct-method style), plus one real
``App.run_test()`` integration test reusing the existing
``Tests.UI.test_library_shell`` / ``Tests.UI.test_destination_shells``
harness fixtures to prove the rail row -> snapshot fetch -> canvas mount
path end to end.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from textual.app import App
from textual.widgets import Button, Input

from tldw_chatbook.Library.library_prompts_state import (
    PromptListRow,
    PromptsListState,
    build_prompts_list_state,
)
from tldw_chatbook.Library.library_shell_state import LIBRARY_ROW_BROWSE_PROMPTS
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen
from tldw_chatbook.Widgets.Library.library_prompts_canvas import LibraryPromptsListCanvas

from Tests.UI.test_destination_shells import (
    StaticLibraryConversationScopeService,
    StaticLibraryMediaScopeService,
    StaticLibraryNotesListScopeService,
)
from Tests.UI.test_library_shell import (
    LIBRARY_TEST_SIZE,
    LibraryHarness,
    _active_library_screen,
    _wait_for_library_shell,
)
from Tests.UI.test_screen_navigation import _build_test_app


def _three_row_state(*, sort: str = "newest") -> PromptsListState:
    return PromptsListState(
        rows=(
            PromptListRow(prompt_id=1, name="Summarize", secondary="Alice · 3m"),
            PromptListRow(prompt_id=2, name="[draft] Q3 plan [wip]", secondary="Bob · 1h"),
            PromptListRow(prompt_id=3, name="Translate", secondary="2d"),
        ),
        count=3,
        sort=sort,
    )


class _CanvasHost(App):
    def __init__(self, state: PromptsListState | None, **kwargs: Any) -> None:  # type: ignore[valid-type]
        super().__init__()
        self._state = state
        self._kwargs = kwargs

    def compose(self):
        yield LibraryPromptsListCanvas(
            self._state, id="library-prompts-canvas", **self._kwargs
        )


# ---------------------------------------------------------------------------
# Widget-only tests (Step 2 of the brief)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prompts_canvas_renders_a_button_per_row():
    """A 3-row state renders exactly 3 prompt row buttons, ids
    ``library-prompt-row-<id>`` keyed by the row's ``prompt_id`` (not
    index, unlike the notes canvas)."""
    app = _CanvasHost(_three_row_state())
    async with app.run_test() as pilot:
        for prompt_id in (1, 2, 3):
            button = pilot.app.query_one(f"#library-prompt-row-{prompt_id}", Button)
            assert button.prompt_id == prompt_id
        rows = pilot.app.query(".library-prompt-row")
        assert len(rows) == 3


@pytest.mark.asyncio
async def test_prompts_canvas_escapes_bracket_titles_verbatim():
    """A prompt named "[draft] Q3 plan [wip]" renders its bracket segments
    verbatim in the row label instead of having them consumed as Rich
    markup (the search-history Button-label lesson)."""
    app = _CanvasHost(_three_row_state())
    async with app.run_test() as pilot:
        button = pilot.app.query_one("#library-prompt-row-2", Button)
        first_line = str(button.label).splitlines()[0]
        assert first_line == "[draft] Q3 plan [wip]"


@pytest.mark.asyncio
async def test_prompts_canvas_toolbar_is_one_horizontal_row():
    """sort/Import/Export share a single ``ds-toolbar`` Horizontal parent
    -- proven structurally (shared parentage), not via region/geometry
    (the bare harness has no app CSS loaded)."""
    app = _CanvasHost(_three_row_state())
    async with app.run_test() as pilot:
        sort_button = pilot.app.query_one("#library-prompts-sort", Button)
        import_button = pilot.app.query_one("#library-prompts-import", Button)
        export_button = pilot.app.query_one("#library-prompts-export", Button)
        toolbar = sort_button.parent
        assert toolbar is not None and toolbar.has_class("ds-toolbar")
        assert import_button.parent is toolbar
        assert export_button.parent is toolbar


@pytest.mark.asyncio
async def test_prompts_canvas_filter_input_prefilled():
    app = _CanvasHost(_three_row_state(), filter_value="plan")
    async with app.run_test() as pilot:
        filter_input = pilot.app.query_one("#library-prompts-filter", Input)
        assert filter_input.value == "plan"


@pytest.mark.asyncio
async def test_prompts_canvas_empty_state_renders_empty_copy_not_list():
    empty_state = PromptsListState(rows=(), count=0, sort="newest")
    app = _CanvasHost(empty_state)
    async with app.run_test() as pilot:
        empty = pilot.app.query_one("#library-prompts-empty")
        assert "No prompts yet" in str(empty.renderable)
        assert len(pilot.app.query(".library-prompt-row")) == 0


@pytest.mark.asyncio
async def test_prompts_canvas_sort_label_reflects_sort_mode():
    app = _CanvasHost(_three_row_state(sort="name"), sort_mode="name")
    async with app.run_test() as pilot:
        sort_button = pilot.app.query_one("#library-prompts-sort", Button)
        assert "Name" in str(sort_button.label)


# ---------------------------------------------------------------------------
# Screen-wiring unit tests (direct-method style, mirrors
# test_library_export_cancel.py)
# ---------------------------------------------------------------------------


def test_build_library_prompts_state_reads_local_source_records():
    fake = SimpleNamespace(
        _local_source_records={
            "prompts": (
                2,
                (
                    {"id": 1, "name": "Summarize", "author": "Alice",
                     "last_modified": "2026-07-01T00:00:00+00:00"},
                    {"id": 2, "name": "Translate", "author": "Bob",
                     "last_modified": "2026-07-02T00:00:00+00:00"},
                ),
            )
        },
        _library_prompts_filter="",
        _library_prompts_sort="newest",
    )
    state = LibraryScreen._build_library_prompts_state(fake)
    assert state.count == 2
    assert [row.prompt_id for row in state.rows] == [2, 1]  # newest first


def test_build_library_prompts_state_tolerates_missing_entry():
    fake = SimpleNamespace(
        _local_source_records={},
        _library_prompts_filter="",
        _library_prompts_sort="newest",
    )
    state = LibraryScreen._build_library_prompts_state(fake)
    assert state.rows == ()
    assert state.count == 0


def test_handle_library_prompts_sort_cycles_newest_to_name():
    calls = []
    fake = SimpleNamespace(
        _library_prompts_sort="newest",
        refresh=lambda recompose=False: calls.append(recompose),
    )
    event = SimpleNamespace(stop=lambda: None)
    LibraryScreen.handle_library_prompts_sort(fake, event)
    assert fake._library_prompts_sort == "name"
    assert calls == [True]


def test_handle_library_prompts_sort_cycles_name_back_to_newest():
    fake = SimpleNamespace(
        _library_prompts_sort="name",
        refresh=lambda recompose=False: None,
    )
    LibraryScreen.handle_library_prompts_sort(fake, SimpleNamespace(stop=lambda: None))
    assert fake._library_prompts_sort == "newest"


def test_handle_library_prompts_filter_submitted_sets_filter():
    calls = []
    fake = SimpleNamespace(
        _library_prompts_filter="",
        _safe_text=LibraryScreen._safe_text,
        refresh=lambda recompose=False: calls.append(recompose),
    )
    event = SimpleNamespace(value="plan", stop=lambda: None)
    LibraryScreen.handle_library_prompts_filter(fake, event)
    assert fake._library_prompts_filter == "plan"
    assert calls == [True]


def test_handle_library_prompt_row_records_selected_id():
    calls = []
    fake = SimpleNamespace(
        _selected_prompt_id=None,
        _library_selected_row_id="",
        refresh=lambda recompose=False: calls.append(recompose),
    )
    button = SimpleNamespace(prompt_id=42)
    event = SimpleNamespace(button=button, stop=lambda: None)
    LibraryScreen.handle_library_prompt_row(fake, event)
    assert fake._selected_prompt_id == 42
    assert fake._library_selected_row_id == LIBRARY_ROW_BROWSE_PROMPTS
    assert calls == [True]


# ---------------------------------------------------------------------------
# Real end-to-end integration test
# ---------------------------------------------------------------------------


class _FakePromptScopeServiceWithList:
    """Prompt-scope fake exposing both ``count_prompts`` and ``list_prompts``
    (unlike ``test_library_shell._FakePromptScopeService``, which only
    exposes the count seam) -- shaped like the real
    ``PromptScopeService.list_prompts`` normalized envelope (composite
    string ``id``, integer ``local_id``) so the screen's remap-to-raw-shape
    adapter is exercised for real."""

    def __init__(self, prompts):
        self._prompts = prompts

    async def count_prompts(self, *, mode="local", **kwargs):
        return len(self._prompts)

    async def list_prompts(self, *, mode="local", page=1, per_page=10, **kwargs):
        items = [
            {
                "id": f"local:prompt:{prompt['id']}",
                "backend": "local",
                "local_id": prompt["id"],
                "name": prompt["name"],
                "author": prompt.get("author"),
                "keywords": prompt.get("keywords") or [],
                "last_modified": prompt.get("last_modified"),
                "version": prompt.get("version"),
            }
            for prompt in self._prompts
        ]
        return {
            "items": items,
            "total_pages": 1,
            "current_page": page,
            "total_items": len(items),
            "page": page,
            "per_page": per_page,
        }


@pytest.mark.asyncio
async def test_library_shell_prompts_row_press_renders_list_canvas():
    """Pressing the Prompts rail row -- with a fake service exposing both
    ``count_prompts`` and ``list_prompts`` -- renders
    ``LibraryPromptsListCanvas`` with a row button per fetched prompt,
    replacing the old placeholder-empty canvas fallback."""
    app = _build_test_app()
    app.notes_scope_service = StaticLibraryNotesListScopeService([])
    app.media_reading_scope_service = StaticLibraryMediaScopeService([])
    app.chat_conversation_scope_service = StaticLibraryConversationScopeService([])
    app.prompt_scope_service = _FakePromptScopeServiceWithList(
        [
            {"id": 5, "name": "Summarize", "author": "Alice",
             "last_modified": "2026-07-01T00:00:00+00:00"},
            {"id": 6, "name": "Translate", "author": "Bob",
             "last_modified": "2026-07-02T00:00:00+00:00"},
        ]
    )
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-prompts").press()
        await pilot.pause()
        await pilot.pause()

        assert screen._library_selected_row_id == LIBRARY_ROW_BROWSE_PROMPTS
        canvas = screen.query_one("#library-prompts-canvas", LibraryPromptsListCanvas)
        assert canvas is not None
        assert screen.query_one("#library-prompt-row-5", Button)
        assert screen.query_one("#library-prompt-row-6", Button)
