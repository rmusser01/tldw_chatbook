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

from tldw_chatbook.DB.Prompts_DB import ConflictError, PromptsDatabase
from tldw_chatbook.Library.library_prompts_state import (
    PromptListRow,
    PromptsListState,
    build_prompts_list_state,
)
from tldw_chatbook.Library.library_shell_state import LIBRARY_ROW_BROWSE_PROMPTS
from tldw_chatbook.Prompt_Management.prompt_markdown_export import render_prompt_markdown
from tldw_chatbook.Prompt_Management.Prompts_Interop import parse_markdown_prompts_from_content
from tldw_chatbook.Prompt_Management.prompt_scope_service import (
    LocalPromptService as ScopeLocalPromptService,
    PromptScopeService,
)
from tldw_chatbook.Third_Party.textual_fspicker import FileSave
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
    _wait_for_library_shell,
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
