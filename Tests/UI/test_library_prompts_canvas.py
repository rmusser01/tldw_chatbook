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

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
from textual.app import App
from textual.widgets import Button, Input, Static, TextArea

from tldw_chatbook.Constants import TAB_CHAT
from tldw_chatbook.DB.Prompts_DB import ConflictError, PromptsDatabase
from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
from tldw_chatbook.Library.library_prompts_state import (
    PromptEditorState,
    PromptListRow,
    PromptsListState,
    build_prompts_list_state,
)
from tldw_chatbook.Library.library_shell_state import (
    LIBRARY_ROW_BROWSE_PROMPTS,
    LIBRARY_ROW_CREATE_PROMPT,
)
from tldw_chatbook.Prompt_Management.prompt_markdown_export import render_prompt_markdown
from tldw_chatbook.Prompt_Management.Prompts_Interop import parse_markdown_prompts_from_content
from tldw_chatbook.Prompt_Management.prompt_scope_service import (
    LocalPromptService as ScopeLocalPromptService,
    PromptScopeService,
)
from tldw_chatbook.runtime_policy.enforcement import ServicePolicyEnforcer
from tldw_chatbook.runtime_policy.types import RuntimeSourceState
from tldw_chatbook.Third_Party.textual_fspicker import FileOpen, FileSave
from tldw_chatbook.UI.Screens import library_screen as library_screen_module
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
    _fake_import_dialog_result,
    _wait_for_library_shell,
    _wait_for_selector,
)
from Tests.UI.test_screen_navigation import _build_test_app


REPO_ROOT = Path(__file__).resolve().parents[2]
AGENTIC_TERMINAL = REPO_ROOT / "tldw_chatbook/css/components/_agentic_terminal.tcss"
BUNDLED_STYLESHEET = REPO_ROOT / "tldw_chatbook/css/tldw_cli_modular.tcss"


def _css_block(text: str, selector: str) -> str:
    """Return a CSS rule body starting at ``selector`` (mirrors
    ``test_product_maturity_phase3_library_contract_layout.py``'s helper of
    the same name)."""
    start = text.index(selector)
    block_start = text.index("{", start)
    block_end = text.index("}", block_start)
    return text[block_start:block_end]


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
    """sort/Import share a single ``ds-toolbar`` Horizontal parent -- proven
    structurally (shared parentage), not via region/geometry (the bare
    harness has no app CSS loaded)."""
    app = _CanvasHost(_three_row_state())
    async with app.run_test() as pilot:
        sort_button = pilot.app.query_one("#library-prompts-sort", Button)
        import_button = pilot.app.query_one("#library-prompts-import", Button)
        toolbar = sort_button.parent
        assert toolbar is not None and toolbar.has_class("ds-toolbar")
        assert import_button.parent is toolbar


@pytest.mark.asyncio
async def test_prompts_canvas_list_toolbar_has_no_dead_export_button():
    """D5 (Task 8c): the list-toolbar "Export..." button had no handler
    anywhere -- pressing it silently no-op'd. Bulk export is deferred to
    task-197; per-prompt export lives in the editor's own
    ``#library-prompt-export`` and still works. The dead affordance is
    removed rather than wired to a fake bulk export."""
    app = _CanvasHost(_three_row_state())
    async with app.run_test() as pilot:
        assert len(pilot.app.query("#library-prompts-export")) == 0


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
# Task 5: toolbar Import… row widget tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prompts_canvas_import_row_hidden_by_default():
    """The Import row (path Input, Run/Cancel actions, outcome line) is not
    mounted at all while ``import_open`` is ``False`` (the default)."""
    app = _CanvasHost(_three_row_state())
    async with app.run_test() as pilot:
        assert len(pilot.app.query("#library-prompts-import-path")) == 0
        assert len(pilot.app.query("#library-prompts-import-run")) == 0
        assert len(pilot.app.query("#library-prompts-import-cancel")) == 0


@pytest.mark.asyncio
async def test_prompts_canvas_import_row_renders_when_open():
    """``import_open=True`` renders the path Input (prefilled from
    ``import_path``), Run/Cancel actions, and the outcome Static (showing
    ``import_status`` verbatim)."""
    app = _CanvasHost(
        _three_row_state(),
        import_open=True,
        import_path="/tmp/my-prompts.md",
        import_status="2 imported · 1 skipped (duplicate name)",
    )
    async with app.run_test() as pilot:
        path_input = pilot.app.query_one("#library-prompts-import-path", Input)
        assert path_input.value == "/tmp/my-prompts.md"
        assert pilot.app.query_one("#library-prompts-import-run", Button)
        assert pilot.app.query_one("#library-prompts-import-cancel", Button)
        status = pilot.app.query_one("#library-prompts-import-status", Static)
        assert str(status.renderable) == "2 imported · 1 skipped (duplicate name)"


@pytest.mark.asyncio
async def test_prompts_canvas_import_row_browse_button_shares_toolbar_with_run_cancel():
    """Task 8b D4: Browse… reuses ``.library-canvas-action`` and shares the
    same fixed-width-only ``ds-toolbar`` as Import/Cancel (no per-id CSS
    block needed -- see the canvas family's render-safe-shape docstring)."""
    app = _CanvasHost(_three_row_state(), import_open=True)
    async with app.run_test() as pilot:
        browse_button = pilot.app.query_one("#library-prompts-import-browse", Button)
        run_button = pilot.app.query_one("#library-prompts-import-run", Button)
        assert browse_button.parent is run_button.parent
        assert browse_button.parent.has_class("ds-toolbar")
        assert browse_button.has_class("library-canvas-action")


@pytest.mark.asyncio
async def test_prompts_canvas_import_path_input_is_not_packed_into_a_toolbar_row():
    """The path Input is its own full-width sibling, NOT packed into a
    ``Horizontal`` alongside the Run/Cancel Buttons -- this canvas family's
    documented non-rendering failure mode is a ``Horizontal`` mixing a 1fr
    Input with fixed-width compact Buttons (see
    ``LibraryIngestCanvas``'s docstring)."""
    app = _CanvasHost(_three_row_state(), import_open=True)
    async with app.run_test() as pilot:
        path_input = pilot.app.query_one("#library-prompts-import-path", Input)
        assert path_input.parent is not None
        assert not path_input.parent.has_class("ds-toolbar")
        run_button = pilot.app.query_one("#library-prompts-import-run", Button)
        assert run_button.parent is not None
        assert run_button.parent.has_class("ds-toolbar")


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


## ``test_handle_library_prompt_row_records_selected_id`` (the old
## recording-only, no-editor behavior) was superseded by Task 4, which
## upgrades this handler to open the in-canvas editor -- see the
## ``handle_library_prompt_row`` end-to-end coverage in the "Task 4" section
## at the bottom of this file, which exercises the real handler (row press
## -> editor opens -> fields populated) through a mounted screen instead of
## a bare ``SimpleNamespace`` stand-in.


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
                "details": prompt.get("details"),
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


@pytest.mark.asyncio
async def test_library_shell_prompts_row_secondary_line_shows_details_not_author():
    """Task 8b D2/U1: the list row's secondary line surfaces the prompt's
    PURPOSE (``details``) instead of ``author · age`` -- exercises the full
    pipeline: the screen's ``_prompts_page_records_or_empty`` remap now
    carries ``details`` through, and the pure state builder's secondary
    line uses it instead of ``author``."""
    app = _build_test_app()
    app.notes_scope_service = StaticLibraryNotesListScopeService([])
    app.media_reading_scope_service = StaticLibraryMediaScopeService([])
    app.chat_conversation_scope_service = StaticLibraryConversationScopeService([])
    app.prompt_scope_service = _FakePromptScopeServiceWithList(
        [
            {"id": 5, "name": "Summarize", "author": "Alice", "details": "Summarizes text",
             "last_modified": "2026-07-01T00:00:00+00:00"},
        ]
    )
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-prompts").press()
        await pilot.pause()
        await pilot.pause()

        button = screen.query_one("#library-prompt-row-5", Button)
        label_text = str(button.label)
        assert "Summarizes text" in label_text
        assert "Alice" not in label_text


# ---------------------------------------------------------------------------
# Stylesheet parity pin (review finding: the canvas's ids/classes had no
# stylesheet rules at all, so prompt rows silently rendered as auto-width
# default Buttons instead of matching the sibling notes list's look).
# ---------------------------------------------------------------------------


def test_library_prompt_row_class_matches_notes_row_visual_parity():
    """``.library-prompt-row`` (the row Buttons in
    ``library_prompts_canvas.py``) must have a stylesheet block, with the
    same width/height/border/background as ``.library-notes-row`` -- visual
    parity with the sibling notes list, not default auto-width Buttons."""
    agentic_terminal = AGENTIC_TERMINAL.read_text(encoding="utf-8")
    bundled_stylesheet = BUNDLED_STYLESHEET.read_text(encoding="utf-8")

    for text in (agentic_terminal, bundled_stylesheet):
        assert ".library-prompt-row {" in text
        prompt_row_block = _css_block(text, ".library-prompt-row {")
        notes_row_block = _css_block(text, ".library-notes-row {")
        for pinned in (
            "width: 100%;",
            "height: 2;",
            "border: none;",
            "background: $ds-surface-panel;",
        ):
            assert pinned in prompt_row_block
            assert pinned in notes_row_block


def test_library_prompts_header_filter_empty_have_css_blocks():
    """``#library-prompts-header``/``#library-prompts-filter``
    (+ ``:focus``)/``#library-prompts-empty`` (``library_prompts_canvas.py``)
    must have stylesheet rules matching their ``#library-notes-*`` siblings,
    instead of silently falling back to unstyled defaults."""
    agentic_terminal = AGENTIC_TERMINAL.read_text(encoding="utf-8")
    bundled_stylesheet = BUNDLED_STYLESHEET.read_text(encoding="utf-8")

    for text in (agentic_terminal, bundled_stylesheet):
        assert "#library-prompts-header {" in text
        assert "#library-prompts-filter {" in text
        assert "#library-prompts-filter:focus {" in text
        assert "#library-prompts-empty {" in text

        header_block = _css_block(text, "#library-prompts-header {")
        notes_header_block = _css_block(text, "#library-notes-header {")
        assert "height: auto;" in header_block
        assert "height: auto;" in notes_header_block

        filter_block = _css_block(text, "#library-prompts-filter {")
        notes_filter_block = _css_block(text, "#library-notes-filter {")
        for pinned in (
            "height: 3;",
            "border: tall $ds-grid-line;",
            "background: $ds-surface-raised;",
        ):
            assert pinned in filter_block
            assert pinned in notes_filter_block

        focus_block = _css_block(text, "#library-prompts-filter:focus {")
        notes_focus_block = _css_block(text, "#library-notes-filter:focus {")
        for pinned in ("border: tall $ds-input-focus-accent;", "outline: none;"):
            assert pinned in focus_block
            assert pinned in notes_focus_block

        empty_block = _css_block(text, "#library-prompts-empty {")
        notes_empty_block = _css_block(text, "#library-notes-empty {")
        assert "color: $ds-text-muted;" in empty_block
        assert "color: $ds-text-muted;" in notes_empty_block


def test_library_prompt_editor_field_css_blocks_match_notes_editor_parity():
    """Editor field ids introduced by Task 4 (name/author/details/keywords
    Inputs, system/user TextAreas, meta line, conflict/status Statics) must
    have stylesheet rules matching their ``#library-note-*`` siblings."""
    agentic_terminal = AGENTIC_TERMINAL.read_text(encoding="utf-8")
    bundled_stylesheet = BUNDLED_STYLESHEET.read_text(encoding="utf-8")

    for text in (agentic_terminal, bundled_stylesheet):
        assert "#library-prompt-name," in text
        assert "#library-prompt-keywords {" in text
        input_block = _css_block(text, "#library-prompt-keywords {")
        note_input_block = _css_block(text, "#library-note-keywords {")
        for pinned in ("height: 3;", "border: tall $ds-grid-line;", "background: $ds-surface-raised;"):
            assert pinned in input_block
            assert pinned in note_input_block

        assert "#library-prompt-name:focus," in text
        assert "#library-prompt-keywords:focus {" in text
        focus_block = _css_block(text, "#library-prompt-keywords:focus {")
        for pinned in ("border: tall $ds-input-focus-accent;", "outline: none;"):
            assert pinned in focus_block

        assert ".library-prompt-field-label {" in text
        label_block = _css_block(text, ".library-prompt-field-label {")
        assert "color: $ds-text-muted;" in label_block

        assert "#library-prompt-system," in text
        assert "#library-prompt-user {" in text
        textarea_block = _css_block(text, "#library-prompt-user {")
        assert "min-height: 6;" in textarea_block
        assert "max-height: 14;" in textarea_block

        assert "#library-prompt-meta {" in text
        meta_block = _css_block(text, "#library-prompt-meta {")
        assert "color: $ds-text-muted;" in meta_block

        assert "#library-prompt-conflict-copy," in text
        assert "#library-prompt-save-status {" in text
        status_block = _css_block(text, "#library-prompt-save-status {")
        assert "color: $ds-text-muted;" in status_block


def test_library_prompt_field_hint_css_block_matches_field_label_parity():
    """U7 (Task 8c): ``.library-prompt-field-hint`` (the one-line dim hint
    under the System/User prompt labels) must have a stylesheet rule, same
    muted tier as its ``.library-prompt-field-label`` sibling -- instead of
    silently falling back to unstyled defaults."""
    agentic_terminal = AGENTIC_TERMINAL.read_text(encoding="utf-8")
    bundled_stylesheet = BUNDLED_STYLESHEET.read_text(encoding="utf-8")

    for text in (agentic_terminal, bundled_stylesheet):
        assert ".library-prompt-field-hint {" in text
        hint_block = _css_block(text, ".library-prompt-field-hint {")
        label_block = _css_block(text, ".library-prompt-field-label {")
        assert "color: $ds-text-muted;" in hint_block
        assert "color: $ds-text-muted;" in label_block


def test_library_prompts_import_row_css_blocks_match_filter_status_parity():
    """Toolbar Import… row ids introduced by Task 5 (the path Input, its
    outcome Static) must have stylesheet rules matching their
    ``#library-prompts-filter``/``#library-prompt-save-status`` siblings,
    instead of silently falling back to unstyled defaults."""
    agentic_terminal = AGENTIC_TERMINAL.read_text(encoding="utf-8")
    bundled_stylesheet = BUNDLED_STYLESHEET.read_text(encoding="utf-8")

    for text in (agentic_terminal, bundled_stylesheet):
        assert "#library-prompts-import-path {" in text
        assert "#library-prompts-import-path:focus {" in text
        assert "#library-prompts-import-status {" in text

        input_block = _css_block(text, "#library-prompts-import-path {")
        filter_block = _css_block(text, "#library-prompts-filter {")
        for pinned in (
            "height: 3;",
            "border: tall $ds-grid-line;",
            "background: $ds-surface-raised;",
        ):
            assert pinned in input_block
            assert pinned in filter_block

        focus_block = _css_block(text, "#library-prompts-import-path:focus {")
        filter_focus_block = _css_block(text, "#library-prompts-filter:focus {")
        for pinned in ("border: tall $ds-input-focus-accent;", "outline: none;"):
            assert pinned in focus_block
            assert pinned in filter_focus_block

        status_block = _css_block(text, "#library-prompts-import-status {")
        assert "color: $ds-text-muted;" in status_block


# ---------------------------------------------------------------------------
# Task 4: editor canvas, explicit Save, conflict outcomes, delete
#
# Uses a REAL ``PromptsDatabase`` + ``PromptScopeService`` (mirroring
# ``Tests/Library/test_library_prompts_seam.py``'s Task 1 precedent)
# rather than a hand-rolled fake -- the conflict/name-collision scenarios
# below depend on the real DB's actual exception/return-value shapes
# (``Prompts.name`` is globally UNIQUE regardless of soft-delete state;
# ``update_prompt_by_id`` has no caller-supplied expected-version
# parameter, so the screen's own pre-check, exercised here, is what
# actually detects staleness -- see ``_save_library_prompt``'s docstring).
# ---------------------------------------------------------------------------


def _real_prompt_scope_service(tmp_path):
    db = PromptsDatabase(tmp_path / "prompts.db", client_id="test-client")
    service = PromptScopeService(local_service=ScopeLocalPromptService(db), server_service=None)
    return db, service


def _real_prompt_scope_service_with_production_policy_enforcer(tmp_path):
    """Like ``_real_prompt_scope_service``, but wires the real production
    runtime-policy seam instead of leaving ``policy_enforcer`` unset.

    ``_real_prompt_scope_service`` (used by every other test below) passes
    no ``policy_enforcer`` at all, so ``PromptScopeService._enforce_policy``
    short-circuits and never calls ``require_allowed`` -- that is exactly
    why the Phase-1 gate defect (clicking a prompt row raised
    ``PolicyDeniedError: Unknown runtime-policy action_id:
    prompts.detail.local``) went uncaught by every existing UI test here.

    This mirrors how ``app.py`` (~2345-2350, 2513-2517) actually builds the
    production seam: a ``ServicePolicyEnforcer`` around the real
    ``CAPABILITY_REGISTRY`` (via its default ``PolicyEngine``), fed a
    ``RuntimeSourceState`` in local mode.
    """
    db = PromptsDatabase(tmp_path / "prompts.db", client_id="test-client")
    policy_enforcer = ServicePolicyEnforcer(
        state_provider=lambda: RuntimeSourceState(active_source="local"),
    )
    service = PromptScopeService(
        local_service=ScopeLocalPromptService(db),
        server_service=None,
        policy_enforcer=policy_enforcer,
    )
    return db, service


def _wire_empty_non_prompt_services(app) -> None:
    app.notes_scope_service = StaticLibraryNotesListScopeService([])
    app.media_reading_scope_service = StaticLibraryMediaScopeService([])
    app.chat_conversation_scope_service = StaticLibraryConversationScopeService([])


async def _open_prompt_editor(screen, pilot, prompt_id: int) -> None:
    """Open the rail's Prompts row, then a specific prompt's row."""
    screen.query_one("#library-row-browse-prompts").press()
    await pilot.pause()
    await pilot.pause()
    screen.query_one(f"#library-prompt-row-{prompt_id}", Button).press()
    await pilot.pause()
    for _ in range(150):
        if screen._library_prompt_detail is not None:
            break
        await pilot.pause(0.02)
    await pilot.pause()


async def _wait_for_prompt_status(screen, pilot, *, attempts=150) -> str:
    status_text = ""
    for _ in range(attempts):
        status_text = str(screen.query_one("#library-prompt-save-status").renderable)
        if status_text:
            return status_text
        await pilot.pause(0.02)
    return status_text


@pytest.mark.asyncio
async def test_library_prompt_row_opens_editor_with_six_fields_populated(tmp_path):
    db, service = _real_prompt_scope_service(tmp_path)
    prompt_id, _uuid, _msg = db.add_prompt(
        name="Summarize",
        author="Alice",
        details="A summarizer",
        system_prompt="You are concise.",
        user_prompt="Summarize: {text}",
        keywords=["writing", "summary"],
    )
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompt_editor(screen, pilot, prompt_id)

        assert screen._library_prompts_view == "editor"
        assert screen.query_one("#library-prompt-name", Input).value == "Summarize"
        assert screen.query_one("#library-prompt-author", Input).value == "Alice"
        assert screen.query_one("#library-prompt-details", Input).value == "A summarizer"
        assert screen.query_one("#library-prompt-system", TextArea).text == "You are concise."
        assert screen.query_one("#library-prompt-user", TextArea).text == "Summarize: {text}"
        assert screen.query_one("#library-prompt-keywords", Input).value == "summary, writing"


@pytest.mark.asyncio
async def test_library_prompt_row_opens_editor_with_modified_meta_not_new_prompt(tmp_path):
    """Critical regression: ``handle_library_prompt_row`` ->
    ``_refresh_library_prompt_detail`` fetches through the REAL production
    seam (``PromptScopeService.get_prompt`` -> ``normalize_prompt_record``),
    whose ``detail["id"]`` is a composite string (``"local:prompt:<uuid>"``)
    with the raw int id under ``detail["local_id"]`` instead.
    ``build_prompt_editor_state`` used to read only ``detail["id"]``, so
    ``_to_int`` silently returned ``None`` and every EXISTING saved prompt's
    meta line rendered "New prompt" (the D1 blank-create sentinel) instead
    of "Modified ... · vN". This is the assertion whose absence let that
    slip past ``test_library_prompt_row_opens_editor_with_six_fields_populated``
    above (which never inspects the meta line)."""
    db, service = _real_prompt_scope_service(tmp_path)
    prompt_id, _uuid, _msg = db.add_prompt(
        name="Summarize",
        author="Alice",
        details="A summarizer",
        system_prompt="You are concise.",
        user_prompt="Summarize: {text}",
        keywords=["writing", "summary"],
    )
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompt_editor(screen, pilot, prompt_id)

        meta = screen.query_one("#library-prompt-meta", Static)
        meta_text = str(meta.renderable)
        assert "New prompt" not in meta_text
        assert "Modified" in meta_text
        assert "v1" in meta_text


@pytest.mark.asyncio
async def test_library_prompt_row_opens_editor_under_real_runtime_policy_enforcer(tmp_path):
    """Regression test for the Phase-1 gate defect (live-blocking): clicking
    a Library prompt row raised
    ``PolicyDeniedError: Unknown runtime-policy action_id: prompts.detail.local``
    from ``PromptScopeService.get_prompt`` -> ``_enforce_policy``, which
    ``_refresh_library_prompt_detail`` (``UI/Screens/library_screen.py``)
    swallows via a bare ``except Exception`` and then treats as "prompt no
    longer available" -- the editor never opened.

    Unlike ``test_library_prompt_row_opens_editor_with_six_fields_populated``
    above (and every other test in this module), this wires the *real*
    production runtime-policy seam -- ``ServicePolicyEnforcer`` bound to the
    real ``CAPABILITY_REGISTRY`` -- via
    ``_real_prompt_scope_service_with_production_policy_enforcer`` instead of
    leaving ``policy_enforcer`` unset. That gap (no test exercised the real
    enforcer+registry combination against the Library Prompts screen) is why
    the missing ``prompts.detail.local`` registry row went uncaught.
    """
    db, service = _real_prompt_scope_service_with_production_policy_enforcer(tmp_path)
    prompt_id, _uuid, _msg = db.add_prompt(
        name="Summarize",
        author="Alice",
        details="A summarizer",
        system_prompt="You are concise.",
        user_prompt="Summarize: {text}",
        keywords=["writing", "summary"],
    )
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompt_editor(screen, pilot, prompt_id)

        assert screen._library_prompts_view == "editor"
        assert screen._library_prompt_detail is not None
        assert screen.query_one("#library-prompt-name", Input).value == "Summarize"


@pytest.mark.asyncio
async def test_library_prompt_save_name_already_in_use_shows_status_copy(tmp_path):
    db, service = _real_prompt_scope_service(tmp_path)
    db.add_prompt(name="Alpha", author="A", details="d", user_prompt="x")
    beta_id, _uuid, _msg = db.add_prompt(name="Beta", author="B", details="d", user_prompt="y")
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompt_editor(screen, pilot, beta_id)

        screen.query_one("#library-prompt-name", Input).value = "Alpha"
        await pilot.pause()
        screen.query_one("#library-prompt-save", Button).press()
        await pilot.pause()

        status_text = await _wait_for_prompt_status(screen, pilot)
        assert status_text == "Name already in use — pick another or open the existing prompt."


@pytest.mark.asyncio
async def test_library_prompt_save_onto_soft_deleted_name_shows_status_copy(tmp_path):
    db, service = _real_prompt_scope_service(tmp_path)
    db.add_prompt(name="Gamma", author="A", details="d", user_prompt="x")
    db.soft_delete_prompt("Gamma")
    delta_id, _uuid, _msg = db.add_prompt(name="Delta", author="B", details="d", user_prompt="y")
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompt_editor(screen, pilot, delta_id)

        screen.query_one("#library-prompt-name", Input).value = "Gamma"
        await pilot.pause()
        screen.query_one("#library-prompt-save", Button).press()
        await pilot.pause()

        status_text = await _wait_for_prompt_status(screen, pilot)
        assert status_text == "A deleted prompt holds this name — restore it or choose another."


@pytest.mark.asyncio
async def test_library_prompt_save_stale_version_shows_conflict_bar(tmp_path):
    db, service = _real_prompt_scope_service(tmp_path)
    prompt_id, _uuid, _msg = db.add_prompt(
        name="Epsilon", author="A", details="d1", user_prompt="x"
    )
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompt_editor(screen, pilot, prompt_id)
        assert screen._library_prompt_version == 1

        # A second, real service call bumps the version behind the open
        # editor's back -- simulating another writer, exactly like the
        # brief's "bump version through a second service call" scenario.
        await service.save_prompt(
            mode="local", prompt_identifier=prompt_id, details="changed elsewhere"
        )

        screen.query_one("#library-prompt-save", Button).press()
        await pilot.pause()
        for _ in range(150):
            if len(screen.query("#library-prompt-conflict-overwrite")) > 0:
                break
            await pilot.pause(0.02)

        assert screen.query_one("#library-prompt-conflict-overwrite", Button)
        assert screen.query_one("#library-prompt-conflict-reload", Button)


@pytest.mark.asyncio
async def test_library_prompt_save_write_time_conflict_shows_conflict_bar(tmp_path):
    """A ``ConflictError`` raised by the actual write itself -- a race the
    pre-checks cannot see (a second app instance / external writer landing
    between this save's pre-read and its real write) -- must route into
    the SAME conflict banner as the pre-check staleness path, not the
    generic "Couldn't save this prompt." status line.

    The pre-checks (name lookup, version pre-read) are left alone here --
    only the real write call (``service.save_prompt``) is monkeypatched to
    raise a real ``tldw_chatbook.DB.Prompts_DB.ConflictError`` on its next
    invocation, so this exercises the exception path inside
    ``_save_library_prompt``'s own write attempt, not the earlier
    version-mismatch pre-check.
    """
    db, service = _real_prompt_scope_service(tmp_path)
    prompt_id, _uuid, _msg = db.add_prompt(
        name="Kappa", author="Original", details="d1", user_prompt="x"
    )
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)

    original_save_prompt = service.save_prompt
    calls = {"count": 0}

    async def _raise_once_then_delegate(**kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise ConflictError("Prompt was modified by another writer.")
        return await original_save_prompt(**kwargs)

    service.save_prompt = _raise_once_then_delegate

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompt_editor(screen, pilot, prompt_id)
        assert screen._library_prompt_version == 1

        screen.query_one("#library-prompt-author", Input).value = "Race Author"
        await pilot.pause()
        screen.query_one("#library-prompt-save", Button).press()
        await pilot.pause()
        for _ in range(150):
            if len(screen.query("#library-prompt-conflict-overwrite")) > 0:
                break
            await pilot.pause(0.02)

        assert calls["count"] == 1
        assert screen.query_one("#library-prompt-conflict-overwrite", Button)
        assert screen.query_one("#library-prompt-conflict-reload", Button)
        status_widgets = screen.query("#library-prompt-save-status")
        if len(status_widgets) > 0:
            assert str(status_widgets.first().renderable) != (
                "Couldn't save this prompt. Try again."
            )

        # The stashed snapshot the banner's Overwrite/Reload actions read
        # from must carry this entry path's live-edit fields too, exactly
        # like the pre-check path's snapshot.
        snapshot = screen._library_prompt_conflict_snapshot
        assert snapshot is not None
        assert snapshot.prompt_id == prompt_id
        assert snapshot.author == "Race Author"

        # Overwrite should succeed once the monkeypatch's single-raise
        # budget is spent (the second call delegates to the real write).
        screen.query_one("#library-prompt-conflict-overwrite", Button).press()
        await pilot.pause()
        for _ in range(150):
            if len(screen.query("#library-prompt-conflict-overwrite")) == 0:
                break
            await pilot.pause(0.02)

        assert calls["count"] == 2
        assert len(screen.query("#library-prompt-conflict-overwrite")) == 0
        persisted = db.fetch_prompt_details(prompt_id)
        assert persisted["author"] == "Race Author"


@pytest.mark.asyncio
async def test_library_shell_create_prompt_write_time_conflict_recovers_on_reload(tmp_path):
    """Task 8b Fix wave 1: the CREATE flow (``_selected_prompt_id`` is
    ``None``) must recover from a genuine write-time ``ConflictError`` the
    same way the update flow does above -- NOT silently no-op both
    Overwrite and Reload just because ``prompt_id`` happens to be the
    create-flow's ``None`` sentinel.

    Regression for the finding: ``_resolve_library_prompt_conflict``'s
    ``if not prompt_id or ...: return`` guard treated a create's ``None``
    prompt_id as "nothing to resolve" and returned immediately for BOTH
    buttons, so ``_library_prompt_dirty`` was never cleared either --
    ``flush_pending_work`` (and therefore Back/rail-row/prompt-row/app-tab
    navigation) then vetoed forever, trapping the user in the editor with
    no in-app recovery.
    """
    db, service = _real_prompt_scope_service(tmp_path)
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)

    original_save_prompt = service.save_prompt
    calls = {"count": 0}

    async def _raise_once_then_delegate(**kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise ConflictError("Prompt 'Brand New' already exists.")
        return await original_save_prompt(**kwargs)

    service.save_prompt = _raise_once_then_delegate

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one(f"#library-row-{LIBRARY_ROW_CREATE_PROMPT}").press()
        await _wait_for_selector(screen, pilot, "#library-prompt-name")

        screen.query_one("#library-prompt-name", Input).value = "Brand New"
        await pilot.pause()
        screen.query_one("#library-prompt-user", TextArea).text = "Hello {name}"
        await pilot.pause()
        screen.query_one("#library-prompt-save", Button).press()
        await pilot.pause()
        for _ in range(150):
            if len(screen.query("#library-prompt-conflict-overwrite")) > 0:
                break
            await pilot.pause(0.02)

        assert calls["count"] == 1
        assert screen._selected_prompt_id is None
        assert screen.query_one("#library-prompt-conflict-overwrite", Button)
        assert screen.query_one("#library-prompt-conflict-reload", Button)
        # The trap the finding describes: dirty stuck true, so every other
        # exit is vetoed too -- assert it up front so a regression here is
        # unambiguous, not just inferred from the buttons doing nothing.
        assert screen._library_prompt_dirty is True

        screen.query_one("#library-prompt-conflict-reload", Button).press()
        await pilot.pause()
        for _ in range(150):
            if len(screen.query("#library-prompt-conflict-overwrite")) == 0:
                break
            await pilot.pause(0.02)

        # Reload must land on a usable, blank create state -- not a
        # permanently stuck banner -- and must clear the dirty flag so
        # navigation is no longer vetoed.
        assert len(screen.query("#library-prompt-conflict-overwrite")) == 0
        assert screen._library_prompt_conflict_snapshot is None
        assert screen._library_prompt_dirty is False
        assert screen.query_one("#library-prompt-name", Input).value == ""
        assert screen.query_one("#library-prompt-user", TextArea).text == ""

        allowed = await screen.flush_pending_work()
        assert allowed is True

        # The colliding name was never actually persisted under this
        # editor session -- only one prompt (the pre-existing "Brand New"
        # implied by the monkeypatched race) may exist, and this session's
        # own record was correctly abandoned rather than double-written.
        assert calls["count"] == 1


@pytest.mark.asyncio
async def test_library_shell_create_prompt_write_time_conflict_overwrite_retries_create(tmp_path):
    """Task 8b Fix wave 1: Overwrite on a CREATE-flow conflict retries the
    create with the kept text (rather than the update path's "re-save
    against a fresh version", which a not-yet-persisted record has none
    of) -- once the monkeypatch's single-raise budget is spent, the retry
    delegates to the real write and the prompt is actually persisted.
    """
    db, service = _real_prompt_scope_service(tmp_path)
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)

    original_save_prompt = service.save_prompt
    calls = {"count": 0}

    async def _raise_once_then_delegate(**kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise ConflictError("Prompt 'Brand New' already exists.")
        return await original_save_prompt(**kwargs)

    service.save_prompt = _raise_once_then_delegate

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one(f"#library-row-{LIBRARY_ROW_CREATE_PROMPT}").press()
        await _wait_for_selector(screen, pilot, "#library-prompt-name")

        screen.query_one("#library-prompt-name", Input).value = "Brand New"
        await pilot.pause()
        screen.query_one("#library-prompt-user", TextArea).text = "Hello {name}"
        await pilot.pause()
        screen.query_one("#library-prompt-save", Button).press()
        await pilot.pause()
        for _ in range(150):
            if len(screen.query("#library-prompt-conflict-overwrite")) > 0:
                break
            await pilot.pause(0.02)

        assert calls["count"] == 1
        assert screen._selected_prompt_id is None

        screen.query_one("#library-prompt-conflict-overwrite", Button).press()
        await pilot.pause()
        for _ in range(150):
            if len(screen.query("#library-prompt-conflict-overwrite")) == 0:
                break
            await pilot.pause(0.02)

        assert calls["count"] == 2
        assert len(screen.query("#library-prompt-conflict-overwrite")) == 0
        assert screen._library_prompt_dirty is False
        assert screen._selected_prompt_id is not None
        persisted = db.fetch_prompt_details(screen._selected_prompt_id)
        assert persisted is not None
        assert persisted["name"] == "Brand New"
        assert persisted["user_prompt"] == "Hello {name}"

        allowed = await screen.flush_pending_work()
        assert allowed is True


@pytest.mark.asyncio
async def test_library_prompt_flush_pending_work_vetoes_dirty_editor(tmp_path):
    db, service = _real_prompt_scope_service(tmp_path)
    prompt_id, _uuid, _msg = db.add_prompt(name="Zeta", author="A", details="d", user_prompt="x")
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompt_editor(screen, pilot, prompt_id)

        screen.query_one("#library-prompt-author", Input).value = "Changed mid switch"
        await pilot.pause()
        assert screen._library_prompt_dirty is True

        allowed = await screen.flush_pending_work()

        assert allowed is False
        assert screen._library_prompt_dirty is True


@pytest.mark.asyncio
async def test_library_prompt_delete_returns_to_list_and_decrements_count(tmp_path):
    db, service = _real_prompt_scope_service(tmp_path)
    eta_id, _uuid, _msg = db.add_prompt(name="Eta", author="A", details="d", user_prompt="x")
    db.add_prompt(name="Theta", author="B", details="d", user_prompt="y")
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompt_editor(screen, pilot, eta_id)

        screen.query_one("#library-prompt-delete", Button).press()
        await pilot.pause()
        for _ in range(150):
            if screen._library_prompts_view == "list":
                break
            await pilot.pause(0.02)
        await pilot.pause()

        assert screen._library_prompts_view == "list"
        rail_label = ""
        for _ in range(150):
            rail_label = str(screen.query_one("#library-row-browse-prompts").label)
            if "(1)" in rail_label:
                break
            await pilot.pause(0.02)
        assert "(1)" in rail_label
        assert len(screen.query(f"#library-prompt-row-{eta_id}")) == 0


@pytest.mark.asyncio
async def test_library_prompt_save_success_updates_status_and_persists(tmp_path):
    """Happy-path Save: not one of the brief's six numbered scenarios, but
    foundational coverage the others all rest on (a broken success path
    would make every other Save-outcome test meaningless)."""
    db, service = _real_prompt_scope_service(tmp_path)
    prompt_id, _uuid, _msg = db.add_prompt(
        name="Iota", author="Original", details="d", user_prompt="x"
    )
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompt_editor(screen, pilot, prompt_id)

        screen.query_one("#library-prompt-author", Input).value = "Updated Author"
        await pilot.pause()
        screen.query_one("#library-prompt-save", Button).press()
        await pilot.pause()

        status_text = await _wait_for_prompt_status(screen, pilot)
        assert status_text == "Saved."
        assert screen._library_prompt_dirty is False
        assert screen._library_prompt_version == 2

        persisted = db.fetch_prompt_details(prompt_id)
        assert persisted["author"] == "Updated Author"
        assert persisted["version"] == 2


@pytest.mark.asyncio
async def test_library_prompt_editing_shows_unsaved_marker_and_save_clears_it(tmp_path):
    """U6 (Task 8c): editing a field surfaces a visible unsaved-changes
    marker on the meta line -- previously the dirty flag was invisible
    until the ``flush_pending_work`` veto fired on nav-away. Saving clears
    it. The meta ``Static`` instance itself must never change identity
    across the edit (a full recompose would remount the Input/TextArea
    fields, re-arm-race the editor, and silently re-trigger the mount-time
    ``Changed`` event the arm-delay guards against)."""
    db, service = _real_prompt_scope_service(tmp_path)
    prompt_id, _uuid, _msg = db.add_prompt(
        name="Mu", author="Original", details="d", user_prompt="x"
    )
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompt_editor(screen, pilot, prompt_id)

        meta_before = screen.query_one("#library-prompt-meta", Static)
        assert "Unsaved" not in str(meta_before.renderable)

        screen.query_one("#library-prompt-author", Input).value = "Changed"
        await pilot.pause()

        assert screen._library_prompt_dirty is True
        meta_after_edit = screen.query_one("#library-prompt-meta", Static)
        assert meta_after_edit is meta_before  # no recompose -- same widget instance
        assert "• Unsaved changes" in str(meta_after_edit.renderable)

        screen.query_one("#library-prompt-save", Button).press()
        await pilot.pause()
        status_text = await _wait_for_prompt_status(screen, pilot)
        assert status_text == "Saved."

        assert screen._library_prompt_dirty is False
        meta_after_save = screen.query_one("#library-prompt-meta", Static)
        assert meta_after_save is meta_before
        assert "Unsaved" not in str(meta_after_save.renderable)


# ---------------------------------------------------------------------------
# Task 5: toolbar Import… + editor Export .md, end-to-end (real DB + service)
# ---------------------------------------------------------------------------


async def _open_prompts_list(screen, pilot) -> None:
    """Open the rail's Prompts row (list view, not the editor)."""
    screen.query_one("#library-row-browse-prompts").press()
    await pilot.pause()
    await pilot.pause()


async def _wait_for_import_status(screen, pilot, *, attempts=200) -> str:
    for _ in range(attempts):
        if screen._library_prompts_import_status:
            return screen._library_prompts_import_status
        await pilot.pause(0.02)
    return screen._library_prompts_import_status


async def _run_import(screen, pilot, path: str) -> str:
    """Open the Import row, type ``path``, press Import, and wait for the outcome."""
    screen.query_one("#library-prompts-import", Button).press()
    await pilot.pause()
    screen.query_one("#library-prompts-import-path", Input).value = path
    await pilot.pause()
    screen.query_one("#library-prompts-import-run", Button).press()
    await pilot.pause()
    return await _wait_for_import_status(screen, pilot)


@pytest.mark.asyncio
async def test_library_prompts_import_button_opens_row(tmp_path):
    db, service = _real_prompt_scope_service(tmp_path)
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompts_list(screen, pilot)

        assert len(screen.query("#library-prompts-import-path")) == 0
        screen.query_one("#library-prompts-import", Button).press()
        await pilot.pause()

        assert screen.query_one("#library-prompts-import-path", Input)


@pytest.mark.asyncio
async def test_library_prompts_import_cancel_closes_row(tmp_path):
    db, service = _real_prompt_scope_service(tmp_path)
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompts_list(screen, pilot)

        screen.query_one("#library-prompts-import", Button).press()
        await pilot.pause()
        assert screen.query_one("#library-prompts-import-path", Input)

        screen.query_one("#library-prompts-import-cancel", Button).press()
        await pilot.pause()

        assert len(screen.query("#library-prompts-import-path")) == 0


@pytest.mark.asyncio
async def test_library_prompts_import_from_file_creates_prompt_and_reports_outcome(tmp_path):
    db, service = _real_prompt_scope_service(tmp_path)
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)

    import_file = tmp_path / "imported.md"
    import_file.write_text(
        render_prompt_markdown(
            {
                "name": "Imported Prompt",
                "author": "Importer",
                "details": "from a file",
                "system_prompt": "sys text",
                "user_prompt": "user text",
                "keywords": ["a", "b"],
            }
        ),
        encoding="utf-8",
    )

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompts_list(screen, pilot)

        status_text = await _run_import(screen, pilot, str(import_file))

        assert status_text == "1 imported · 0 skipped (duplicate name)"
        persisted = db.fetch_prompt_details("Imported Prompt")
        assert persisted is not None
        assert persisted["author"] == "Importer"
        assert persisted["details"] == "from a file"
        assert persisted["system_prompt"] == "sys text"
        assert persisted["user_prompt"] == "user text"
        assert sorted(persisted["keywords"]) == ["a", "b"]


@pytest.mark.asyncio
async def test_library_prompts_import_skips_existing_duplicate_name(tmp_path):
    """Duplicate names are SKIPPED, never overwritten -- the pre-existing
    row's content must be untouched after the import runs."""
    db, service = _real_prompt_scope_service(tmp_path)
    db.add_prompt(
        name="Existing", author="Original", details="d", user_prompt="original text"
    )
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)

    import_file = tmp_path / "dup.md"
    import_file.write_text(
        render_prompt_markdown(
            {
                "name": "Existing",
                "author": "Different",
                "details": "d2",
                "system_prompt": "sys2",
                "user_prompt": "different text",
                "keywords": [],
            }
        ),
        encoding="utf-8",
    )

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompts_list(screen, pilot)

        status_text = await _run_import(screen, pilot, str(import_file))

        assert status_text == "0 imported · 1 skipped (duplicate name)"
        persisted = db.fetch_prompt_details("Existing")
        assert persisted["author"] == "Original"
        assert persisted["user_prompt"] == "original text"


@pytest.mark.asyncio
async def test_library_prompts_import_from_folder_aggregates_two_files(tmp_path):
    db, service = _real_prompt_scope_service(tmp_path)
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)

    import_dir = tmp_path / "prompts_to_import"
    import_dir.mkdir()
    (import_dir / "one.md").write_text(
        render_prompt_markdown(
            {"name": "Folder One", "author": "A", "details": "", "system_prompt": "s1", "user_prompt": "u1", "keywords": []}
        ),
        encoding="utf-8",
    )
    (import_dir / "two.md").write_text(
        render_prompt_markdown(
            {"name": "Folder Two", "author": "B", "details": "", "system_prompt": "s2", "user_prompt": "u2", "keywords": []}
        ),
        encoding="utf-8",
    )
    (import_dir / "ignored.dat").write_text("not a supported extension", encoding="utf-8")

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompts_list(screen, pilot)

        status_text = await _run_import(screen, pilot, str(import_dir))

        assert status_text == "2 imported · 0 skipped (duplicate name)"
        assert db.fetch_prompt_details("Folder One") is not None
        assert db.fetch_prompt_details("Folder Two") is not None


@pytest.mark.asyncio
async def test_library_prompts_import_invalid_path_shows_quiet_status(tmp_path):
    db, service = _real_prompt_scope_service(tmp_path)
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompts_list(screen, pilot)

        status_text = await _run_import(screen, pilot, str(tmp_path / "does_not_exist.md"))

        assert status_text == "Could not find that file or folder."
        assert db.list_prompts()[3] == 0  # total_items


@pytest.mark.asyncio
async def test_library_prompts_import_counts_per_prompt_save_failures_as_failed(tmp_path):
    """A per-prompt save failure (a non-duplicate-name exception from the
    scope service's ``save_prompt``) must be tracked as its own ``failed``
    bucket -- previously it fell into neither ``imported`` nor ``skipped``
    and vanished from the outcome line entirely."""
    db, service = _real_prompt_scope_service(tmp_path)
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)

    real_save_prompt = service.save_prompt

    def _flaky_save_prompt(*args, **kwargs):
        if kwargs.get("name") == "Boom":
            raise RuntimeError("simulated save failure")
        return real_save_prompt(*args, **kwargs)

    app.prompt_scope_service = SimpleNamespace(
        get_prompt=service.get_prompt, save_prompt=_flaky_save_prompt
    )
    host = LibraryHarness(app)

    import_dir = tmp_path / "prompts_to_import"
    import_dir.mkdir()
    (import_dir / "ok.md").write_text(
        render_prompt_markdown(
            {"name": "Fine", "author": "A", "details": "", "system_prompt": "s1", "user_prompt": "u1", "keywords": []}
        ),
        encoding="utf-8",
    )
    (import_dir / "boom.md").write_text(
        render_prompt_markdown(
            {"name": "Boom", "author": "B", "details": "", "system_prompt": "s2", "user_prompt": "u2", "keywords": []}
        ),
        encoding="utf-8",
    )

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompts_list(screen, pilot)

        status_text = await _run_import(screen, pilot, str(import_dir))

        assert status_text == "1 imported · 0 skipped (duplicate name) · 1 failed"
        assert db.fetch_prompt_details("Fine") is not None
        assert db.fetch_prompt_details("Boom") is None


@pytest.mark.asyncio
async def test_library_prompts_import_outcome_omits_failed_segment_when_zero(tmp_path):
    """When nothing failed, the outcome line keeps its exact pre-existing
    two-part copy -- no trailing "· K failed" segment at all (not even
    "· 0 failed"). ``test_library_prompts_import_from_file_creates_prompt_and_reports_outcome``
    already pins this exact string for the plain success path; this test
    pins it again explicitly alongside the failed-segment tests above so
    both copies ("N imported · M skipped (duplicate name)" and its
    "· K failed" extension) are each proven by a dedicated assertion."""
    db, service = _real_prompt_scope_service(tmp_path)
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)

    import_file = tmp_path / "clean.md"
    import_file.write_text(
        render_prompt_markdown(
            {"name": "Clean Import", "author": "A", "details": "", "system_prompt": "s", "user_prompt": "u", "keywords": []}
        ),
        encoding="utf-8",
    )

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompts_list(screen, pilot)

        status_text = await _run_import(screen, pilot, str(import_file))

        assert status_text == "1 imported · 0 skipped (duplicate name)"


@pytest.mark.asyncio
async def test_library_prompts_import_folder_permission_error_shows_quiet_status(
    tmp_path, monkeypatch
):
    """A folder enumeration failure (e.g. permissions revoked between the
    Import row's path-validation check and the worker's own ``iterdir()``
    call) must surface an honest status line -- mirroring the per-file read
    try/except below it -- rather than raising out of the worker or
    reporting a misleading "No supported prompt files found."."""
    db, service = _real_prompt_scope_service(tmp_path)
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)

    import_dir = tmp_path / "locked_prompts"
    import_dir.mkdir()

    real_iterdir = Path.iterdir

    def _flaky_iterdir(self):
        if str(self) == str(import_dir):
            raise PermissionError("simulated permission error")
        return real_iterdir(self)

    monkeypatch.setattr(Path, "iterdir", _flaky_iterdir)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompts_list(screen, pilot)

        status_text = await _run_import(screen, pilot, str(import_dir))

        assert status_text == "Could not read that folder."


@pytest.mark.asyncio
async def test_library_prompt_export_pushes_file_save_dialog(tmp_path):
    """Export… pushes a ``FileSave`` dialog pre-filled with a sanitized
    default filename derived from the prompt's current name -- mirrors
    ``test_library_shell_note_export_markdown_pushes_file_save_dialog``."""
    db, service = _real_prompt_scope_service(tmp_path)
    prompt_id, _uuid, _msg = db.add_prompt(
        name="Export Me", author="Author", details="d", system_prompt="s", user_prompt="u"
    )
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompt_editor(screen, pilot, prompt_id)

        screen.query_one("#library-prompt-export", Button).press()
        for _ in range(150):
            if isinstance(host.screen_stack[-1], FileSave):
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError("Export… never pushed a FileSave dialog.")

        dialog = host.screen_stack[-1]
        assert dialog._default_file == "Export Me.md"

        await host.pop_screen()
        await pilot.pause()


@pytest.mark.asyncio
async def test_library_prompt_write_export_file_writes_roundtrippable_markdown(tmp_path):
    """The export write-path (bypassing the dialog UI, exercised separately
    above) writes content that round-trips through the real parser --
    mirrors ``test_library_shell_note_write_export_file_writes_expected_content``."""
    db, service = _real_prompt_scope_service(tmp_path)
    prompt_id, _uuid, _msg = db.add_prompt(
        name="Export Me", author="Author", details="d", system_prompt="s", user_prompt="u",
        keywords=["k1", "k2"],
    )
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    app.notify = Mock()
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompt_editor(screen, pilot, prompt_id)

        destination = tmp_path / "export.md"
        screen._write_library_prompt_export_file(
            destination, "Export Me", "Author", "d", "s", "u", "k1, k2", prompt_id,
        )

        written = destination.read_text(encoding="utf-8")
        parsed = parse_markdown_prompts_from_content(written)
        assert len(parsed) == 1
        p = parsed[0]
        assert (p["name"], p["author"], p["details"], p["system_prompt"], p["user_prompt"]) == (
            "Export Me", "Author", "d", "s", "u",
        )
        assert p["keywords"] == ["k1", "k2"]
        app.notify.assert_called_once()
        assert "exported successfully" in app.notify.call_args.args[0]


@pytest.mark.asyncio
async def test_library_prompt_write_export_file_rejects_invalid_path(tmp_path, monkeypatch):
    """A ``FileSave``-returned path that fails ``validate_path_simple`` must
    be rejected with a quiet warning notice -- no write, no crash."""
    db, service = _real_prompt_scope_service(tmp_path)
    prompt_id, _uuid, _msg = db.add_prompt(name="Export Me", author="A", details="d", user_prompt="u")
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    app.notify = Mock()
    host = LibraryHarness(app)

    def _reject_path(*_args, **_kwargs):
        raise ValueError("rejected for test")

    monkeypatch.setattr(library_screen_module, "validate_path_simple", _reject_path)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompt_editor(screen, pilot, prompt_id)

        destination = tmp_path / "export.md"
        screen._write_library_prompt_export_file(
            destination, "Export Me", "A", "d", "", "u", "", prompt_id,
        )

        assert not destination.exists()
        app.notify.assert_called_once()
        args, kwargs = app.notify.call_args
        assert "Rejected export path" in args[0]
        assert kwargs.get("severity") == "warning"


@pytest.mark.asyncio
async def test_library_prompt_write_export_file_cancelled_dialog_notifies_quietly(tmp_path):
    """A cancelled ``FileSave`` dialog (``selected_path=None``) is a silent
    no-op plus a quiet notice -- no write, no crash."""
    db, service = _real_prompt_scope_service(tmp_path)
    prompt_id, _uuid, _msg = db.add_prompt(name="Export Me", author="A", details="d", user_prompt="u")
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    app.notify = Mock()
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompt_editor(screen, pilot, prompt_id)

        screen._write_library_prompt_export_file(
            None, "Export Me", "A", "d", "", "u", "", prompt_id,
        )

        app.notify.assert_called_once()
        assert "cancelled" in app.notify.call_args.args[0]


# ---------------------------------------------------------------------------
# Task 8b, Group 1: D1 (New prompt create entry), U2 (Author demoted), U3
# (Duplicate), U4 ("Description" label)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prompts_canvas_editor_description_label_replaces_details():
    """U4: the rendered field label reads "Description", not "Details" --
    the DB/record field name (``#library-prompt-details``) is untouched."""
    editor_state = PromptEditorState(
        prompt_id=1, name="X", author="A", details="d", system_prompt="s", user_prompt="u",
        keywords_csv="", version=1, created="", modified="2026-07-07T11:00:00+00:00",
    )
    app = _CanvasHost(None, mode="editor", editor_state=editor_state)
    async with app.run_test() as pilot:
        labels = [
            str(getattr(s.renderable, "plain", s.renderable))
            for s in pilot.app.query(".library-prompt-field-label")
        ]
        assert "Description" in labels
        assert "Details" not in labels
        assert pilot.app.query_one("#library-prompt-details", Input).value == "d"


@pytest.mark.asyncio
async def test_prompts_canvas_editor_field_order_author_last_beside_keywords():
    """U2: compose order is Name, Description, System prompt, User prompt,
    Keywords, Author -- Author moves from 2nd/3rd position to last."""
    editor_state = PromptEditorState(
        prompt_id=1, name="X", author="A", details="d", system_prompt="s", user_prompt="u",
        keywords_csv="kw1, kw2", version=1, created="", modified="2026-07-07T11:00:00+00:00",
    )
    app = _CanvasHost(None, mode="editor", editor_state=editor_state)
    async with app.run_test() as pilot:
        canvas = pilot.app.query_one(LibraryPromptsListCanvas)
        ids = [child.id for child in canvas.children if child.id]
        assert (
            ids.index("library-prompt-name")
            < ids.index("library-prompt-details")
            < ids.index("library-prompt-system")
            < ids.index("library-prompt-user")
            < ids.index("library-prompt-keywords")
            < ids.index("library-prompt-author")
        )


# ---------------------------------------------------------------------------
# Task 8c: U7 (System/User field help) + U8 (Copy vs Duplicate relabel)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prompts_canvas_editor_renders_system_and_user_field_hints():
    """U7: a one-line dim hint renders under each of the System prompt/User
    prompt labels, explaining the two-part prompt model to a new user."""
    editor_state = PromptEditorState(
        prompt_id=1, name="X", author="A", details="d", system_prompt="s", user_prompt="u",
        keywords_csv="", version=1, created="", modified="2026-07-07T11:00:00+00:00",
    )
    app = _CanvasHost(None, mode="editor", editor_state=editor_state)
    async with app.run_test() as pilot:
        hints = [
            str(getattr(s.renderable, "plain", s.renderable))
            for s in pilot.app.query(".library-prompt-field-hint")
        ]
        assert "Instructions the model always follows." in hints
        assert "The message inserted into the composer." in hints


@pytest.mark.asyncio
async def test_prompts_canvas_editor_copy_and_duplicate_relabeled():
    """U8: #library-prompt-copy (clipboard) and #library-prompt-duplicate
    (clone as new prompt) sit adjacent with near-identical labels today --
    relabel to disambiguate. Ids are unchanged."""
    editor_state = PromptEditorState(
        prompt_id=1, name="X", author="A", details="d", system_prompt="s", user_prompt="u",
        keywords_csv="", version=1, created="", modified="2026-07-07T11:00:00+00:00",
    )
    app = _CanvasHost(None, mode="editor", editor_state=editor_state)
    async with app.run_test() as pilot:
        copy_button = pilot.app.query_one("#library-prompt-copy", Button)
        duplicate_button = pilot.app.query_one("#library-prompt-duplicate", Button)
        assert str(copy_button.label) == "Copy text"
        assert str(duplicate_button.label) == "Duplicate prompt"


@pytest.mark.asyncio
async def test_library_shell_create_prompt_row_opens_blank_editor(tmp_path):
    """D1: the Create rail's "New prompt" row opens the in-canvas editor on
    a blank, not-yet-saved record -- empty fields, meta line reads "New
    prompt" (not "Modified … · vN"), prompt_id None."""
    db, service = _real_prompt_scope_service(tmp_path)
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one(f"#library-row-{LIBRARY_ROW_CREATE_PROMPT}").press()
        await _wait_for_selector(screen, pilot, "#library-prompt-name")

        assert screen._library_prompts_view == "editor"
        assert screen._selected_prompt_id is None
        assert screen.query_one("#library-prompt-name", Input).value == ""
        assert screen.query_one("#library-prompt-author", Input).value == ""
        assert screen.query_one("#library-prompt-details", Input).value == ""
        assert screen.query_one("#library-prompt-system", TextArea).text == ""
        assert screen.query_one("#library-prompt-user", TextArea).text == ""
        assert screen.query_one("#library-prompt-keywords", Input).value == ""
        meta = screen.query_one("#library-prompt-meta", Static)
        assert str(meta.renderable) == "New prompt"
        assert len(screen.query("#library-prompt-open-existing")) == 0


@pytest.mark.asyncio
async def test_library_shell_create_prompt_save_creates_and_increments_count(tmp_path):
    """D1: Save with a fresh name CREATES via the scope service's create
    path (not update) -- the Prompts rail count increments and the editor
    adopts the new id + switches to the normal "Modified … · vN" meta."""
    db, service = _real_prompt_scope_service(tmp_path)
    db.add_prompt(name="Existing", author="A", details="d", user_prompt="x")
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one(f"#library-row-{LIBRARY_ROW_CREATE_PROMPT}").press()
        await _wait_for_selector(screen, pilot, "#library-prompt-name")

        screen.query_one("#library-prompt-name", Input).value = "Brand New"
        await pilot.pause()
        screen.query_one("#library-prompt-user", TextArea).text = "Hello {name}"
        await pilot.pause()
        screen.query_one("#library-prompt-save", Button).press()
        await pilot.pause()

        status_text = await _wait_for_prompt_status(screen, pilot)
        assert status_text == "Saved."
        assert screen._selected_prompt_id is not None
        created_id = screen._selected_prompt_id
        persisted = db.fetch_prompt_details(created_id)
        assert persisted is not None
        assert persisted["name"] == "Brand New"
        assert persisted["user_prompt"] == "Hello {name}"

        meta = screen.query_one("#library-prompt-meta", Static)
        for _ in range(150):
            if "Modified" in str(meta.renderable):
                break
            await pilot.pause(0.02)
        assert "Modified" in str(meta.renderable)
        assert "v1" in str(meta.renderable)

        rail_label = ""
        for _ in range(150):
            rail_label = str(screen.query_one("#library-row-browse-prompts").label)
            if "(2)" in rail_label:
                break
            await pilot.pause(0.02)
        assert "(2)" in rail_label


@pytest.mark.asyncio
async def test_library_shell_create_prompt_save_existing_name_shows_name_in_use(tmp_path):
    """D1: the three save outcomes apply to create too -- an existing name
    shows the same name-in-use status the update path uses."""
    db, service = _real_prompt_scope_service(tmp_path)
    db.add_prompt(name="Taken", author="A", details="d", user_prompt="x")
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one(f"#library-row-{LIBRARY_ROW_CREATE_PROMPT}").press()
        await _wait_for_selector(screen, pilot, "#library-prompt-name")

        screen.query_one("#library-prompt-name", Input).value = "Taken"
        await pilot.pause()
        screen.query_one("#library-prompt-user", TextArea).text = "hi"
        await pilot.pause()
        screen.query_one("#library-prompt-save", Button).press()
        await pilot.pause()

        status_text = await _wait_for_prompt_status(screen, pilot)
        assert status_text == "Name already in use — pick another or open the existing prompt."
        assert screen._selected_prompt_id is None
        _prompts, _tp, _cp, total = db.list_prompts()
        assert total == 1


@pytest.mark.asyncio
async def test_library_prompt_duplicate_button_between_copy_and_delete(tmp_path):
    """U3: the Duplicate action sits between Copy and Delete in the editor's
    action row."""
    db, service = _real_prompt_scope_service(tmp_path)
    prompt_id, _uuid, _msg = db.add_prompt(name="X", author="A", details="d", user_prompt="y")
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompt_editor(screen, pilot, prompt_id)

        toolbar = screen.query_one("#library-prompt-copy", Button).parent
        ids = [child.id for child in toolbar.children]
        assert ids.index("library-prompt-copy") < ids.index("library-prompt-duplicate") < ids.index(
            "library-prompt-delete"
        )


@pytest.mark.asyncio
async def test_library_prompt_duplicate_prefills_blank_editor_and_saves_distinct_prompt(tmp_path):
    """U3: Duplicate opens the editor on a NEW blank-id record pre-filled
    from the current prompt's fields, name "<name> (copy)", dirty/unsaved.
    Reuses the D1 create path on Save -- a distinct prompt is created."""
    db, service = _real_prompt_scope_service(tmp_path)
    prompt_id, _uuid, _msg = db.add_prompt(
        name="Original", author="Alice", details="d", system_prompt="sys", user_prompt="usr",
        keywords=["a", "b"],
    )
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompt_editor(screen, pilot, prompt_id)

        screen.query_one("#library-prompt-duplicate", Button).press()
        await pilot.pause()

        assert screen._selected_prompt_id is None
        assert screen._library_prompt_dirty is True
        assert screen.query_one("#library-prompt-name", Input).value == "Original (copy)"
        assert screen.query_one("#library-prompt-system", TextArea).text == "sys"
        assert screen.query_one("#library-prompt-user", TextArea).text == "usr"
        assert screen.query_one("#library-prompt-author", Input).value == "Alice"

        screen.query_one("#library-prompt-save", Button).press()
        await pilot.pause()
        status_text = await _wait_for_prompt_status(screen, pilot)
        assert status_text == "Saved."
        assert screen._selected_prompt_id is not None
        assert screen._selected_prompt_id != prompt_id

        _prompts, _tp, _cp, total = db.list_prompts()
        assert total == 2
        original = db.fetch_prompt_details(prompt_id)
        assert original["name"] == "Original"


# ---------------------------------------------------------------------------
# Task 8b, Group 3: D3 (Open existing) + D4 (import Browse…)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_library_prompt_open_existing_button_shows_only_in_name_in_use_state_and_opens_it(
    tmp_path,
):
    """D3: the "Open existing" button appears ONLY in the name-in-use
    state, and pressing it loads the colliding prompt into the editor."""
    db, service = _real_prompt_scope_service(tmp_path)
    alpha_id, _uuid, _msg = db.add_prompt(name="Alpha", author="A", details="d-alpha", user_prompt="x")
    beta_id, _uuid, _msg = db.add_prompt(name="Beta", author="B", details="d-beta", user_prompt="y")
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompt_editor(screen, pilot, beta_id)

        assert len(screen.query("#library-prompt-open-existing")) == 0

        screen.query_one("#library-prompt-name", Input).value = "Alpha"
        await pilot.pause()
        screen.query_one("#library-prompt-save", Button).press()
        await pilot.pause()

        status_text = await _wait_for_prompt_status(screen, pilot)
        assert status_text == "Name already in use — pick another or open the existing prompt."

        for _ in range(150):
            if len(screen.query("#library-prompt-open-existing")) > 0:
                break
            await pilot.pause(0.02)
        open_existing = screen.query_one("#library-prompt-open-existing", Button)

        open_existing.press()
        await pilot.pause()
        for _ in range(150):
            if screen._selected_prompt_id == alpha_id:
                break
            await pilot.pause(0.02)
        assert screen._selected_prompt_id == alpha_id

        for _ in range(150):
            if screen.query_one("#library-prompt-name", Input).value == "Alpha":
                break
            await pilot.pause(0.02)
        assert screen.query_one("#library-prompt-name", Input).value == "Alpha"
        assert screen.query_one("#library-prompt-details", Input).value == "d-alpha"
        assert len(screen.query("#library-prompt-open-existing")) == 0


@pytest.mark.asyncio
async def test_library_prompt_open_existing_resolves_offending_name_not_drifted_field(
    tmp_path,
):
    """Task 8b Fix wave 1 (Minor): once the name-in-use status is showing,
    "Open existing" stays mounted even if the user keeps typing in the
    Name field without re-saving -- it must still resolve against the
    name that actually collided ("Alpha"), not whatever text is currently
    sitting in the (drifted, never re-saved) Name field."""
    db, service = _real_prompt_scope_service(tmp_path)
    alpha_id, _uuid, _msg = db.add_prompt(name="Alpha", author="A", details="d-alpha", user_prompt="x")
    beta_id, _uuid, _msg = db.add_prompt(name="Beta", author="B", details="d-beta", user_prompt="y")
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompt_editor(screen, pilot, beta_id)

        screen.query_one("#library-prompt-name", Input).value = "Alpha"
        await pilot.pause()
        screen.query_one("#library-prompt-save", Button).press()
        await pilot.pause()
        await _wait_for_prompt_status(screen, pilot)
        for _ in range(150):
            if len(screen.query("#library-prompt-open-existing")) > 0:
                break
            await pilot.pause(0.02)
        assert screen._library_prompt_name_in_use == "Alpha"

        # Drift: the user keeps editing the Name field to something that
        # collides with NEITHER prompt, without pressing Save again -- the
        # status/button never clear (nothing re-checks on plain typing).
        screen.query_one("#library-prompt-name", Input).value = "Not A Real Prompt"
        await pilot.pause()
        assert len(screen.query("#library-prompt-open-existing")) > 0

        screen.query_one("#library-prompt-open-existing", Button).press()
        await pilot.pause()
        for _ in range(150):
            if screen._selected_prompt_id == alpha_id:
                break
            await pilot.pause(0.02)

        # Resolves to the prompt that ACTUALLY collided ("Alpha"), not a
        # failed/empty lookup for the drifted "Not A Real Prompt" text.
        assert screen._selected_prompt_id == alpha_id
        for _ in range(150):
            if screen.query_one("#library-prompt-name", Input).value == "Alpha":
                break
            await pilot.pause(0.02)
        assert screen.query_one("#library-prompt-name", Input).value == "Alpha"


@pytest.mark.asyncio
async def test_library_prompts_import_browse_button_fills_path_input(tmp_path):
    """D4: Browse… (beside the import path Input) opens the same FileOpen
    dialog the media-ingest form's Browse action uses; on pick, it fills
    the path Input."""
    db, service = _real_prompt_scope_service(tmp_path)
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)

    picked_file = tmp_path / "prompts.json"
    picked_file.write_text("[]", encoding="utf-8")

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompts_list(screen, pilot)

        screen.query_one("#library-prompts-import", Button).press()
        await pilot.pause()
        assert screen.query_one("#library-prompts-import-browse", Button)

        push_calls = _fake_import_dialog_result(screen, picked_file)
        screen.query_one("#library-prompts-import-browse", Button).press()
        await pilot.pause()

        assert push_calls and isinstance(push_calls[0], FileOpen)
        assert screen.query_one("#library-prompts-import-path", Input).value == str(picked_file)


# ---------------------------------------------------------------------------
# Task 12: editor "Use in Console" -- ChatHandoffPayload-free direct route
# (stages a bare string via ``TldwCli.stage_console_prompt_insert`` and
# navigates to Chat; ChatScreen itself owns append-vs-replace and the
# provider-setup-blocked gate, tested in ``test_console_command_composer.py``).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_library_prompt_insert_console_stages_live_user_prompt_and_navigates(tmp_path):
    db, service = _real_prompt_scope_service(tmp_path)
    prompt_id, _uuid, _msg = db.add_prompt(
        name="Summarize",
        author="Alice",
        details="A summarizer",
        system_prompt="You are concise.",
        user_prompt="Summarize: {text}",
        keywords=["writing"],
    )
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompt_editor(screen, pilot, prompt_id)
        assert screen._library_prompt_dirty is False

        # ``app`` (the plain ``TldwCli`` built by ``_build_test_app``) is a
        # data/logic container here, never itself run as the active Textual
        # App (``host``/``LibraryHarness`` is) -- so a real ``post_message``
        # would queue into a message pump that is never started, and
        # ``host.seen_routes`` would never observe it. Spying on
        # ``post_message`` directly (mirrors
        # ``test_chat_first_handoffs.py``'s ``open_chat_with_handoff``
        # coverage) proves the navigation intent without depending on that
        # unrelated plumbing.
        post_message_spy = Mock(wraps=app.post_message)
        app.post_message = post_message_spy

        screen.query_one("#library-prompt-insert-console", Button).press()
        await pilot.pause()

        assert app.pending_console_prompt_insert == "Summarize: {text}"
        posted = post_message_spy.call_args.args[0]
        assert isinstance(posted, NavigateToScreen)
        assert posted.screen_name == TAB_CHAT
        # The source prompt itself is never touched by this action.
        assert screen._library_prompts_view == "editor"
        assert screen._selected_prompt_id == prompt_id


@pytest.mark.asyncio
async def test_library_prompt_insert_console_refuses_while_dirty(tmp_path):
    """An unsaved in-progress edit refuses the action outright (rather than
    staging text that a vetoed navigation would later fire unexpectedly on
    some unrelated future Console visit) -- the prompt is never lost either
    way, since the edit simply stays in the still-open editor."""
    db, service = _real_prompt_scope_service(tmp_path)
    prompt_id, _uuid, _msg = db.add_prompt(
        name="Summarize",
        author="",
        details="",
        system_prompt="",
        user_prompt="Summarize: {text}",
        keywords=[],
    )
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompt_editor(screen, pilot, prompt_id)

        screen.query_one("#library-prompt-user", TextArea).text = "Hello {name}"
        await pilot.pause()
        assert screen._library_prompt_dirty is True

        notify_spy = Mock()
        app.notify = notify_spy
        post_message_spy = Mock(wraps=app.post_message)
        app.post_message = post_message_spy
        screen.query_one("#library-prompt-insert-console", Button).press()
        await pilot.pause()

        notify_spy.assert_called_once_with(
            "Save your changes before using this prompt in Console.", severity="warning"
        )
        assert app.pending_console_prompt_insert is None
        post_message_spy.assert_not_called()
        assert screen.query_one("#library-prompt-user", TextArea).text == "Hello {name}"


@pytest.mark.asyncio
async def test_library_prompt_insert_console_notifies_when_user_prompt_is_empty(tmp_path):
    db, service = _real_prompt_scope_service(tmp_path)
    prompt_id, _uuid, _msg = db.add_prompt(
        name="System Only",
        author="",
        details="",
        system_prompt="You are helpful.",
        user_prompt="",
        keywords=[],
    )
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompt_editor(screen, pilot, prompt_id)

        notify_spy = Mock()
        app.notify = notify_spy
        post_message_spy = Mock(wraps=app.post_message)
        app.post_message = post_message_spy
        screen.query_one("#library-prompt-insert-console", Button).press()
        await pilot.pause()

        notify_spy.assert_called_once_with(
            "This prompt has no user prompt text to insert.", severity="warning"
        )
        assert app.pending_console_prompt_insert is None
        post_message_spy.assert_not_called()
