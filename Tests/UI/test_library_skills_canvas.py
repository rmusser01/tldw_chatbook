"""Tests for the Library skills list canvas widget and its screen wiring.

Widget-only tests mount ``LibrarySkillsListCanvas`` directly in a bare
``App`` subclass (mirrors ``test_library_prompts_canvas.py``'s
``_CanvasHost`` harness) -- no app CSS loaded, so assertions stick to
structure/content, never geometry.

Screen-wiring tests call ``LibraryScreen`` bound methods directly against a
``SimpleNamespace`` stand-in for ``self`` (mirrors
``test_library_prompts_canvas.py``'s direct-method style), plus one real
``App.run_test()`` integration test reusing the existing
``Tests.UI.test_library_shell`` harness fixtures to prove the rail row ->
snapshot fetch -> canvas mount path end to end.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from textual.app import App
from textual.widgets import Button, Input, Static

from tldw_chatbook.Library.library_shell_state import LIBRARY_ROW_BROWSE_SKILLS
from tldw_chatbook.Library.library_skills_state import SkillListRow, SkillsListState
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen
from tldw_chatbook.Widgets.Library.library_skills_canvas import LibrarySkillsListCanvas

from Tests.UI.test_destination_shells import (
    StaticLibraryConversationScopeService,
    StaticLibraryMediaScopeService,
    StaticLibraryNotesListScopeService,
)
from Tests.UI.test_library_shell import (
    LIBRARY_TEST_SIZE,
    LibraryHarness,
    _FakeSkillsScopeService,
    _active_library_screen,
    _wait_for_library_shell,
)
from Tests.UI.test_screen_navigation import _build_test_app


REPO_ROOT = Path(__file__).resolve().parents[2]
AGENTIC_TERMINAL = REPO_ROOT / "tldw_chatbook/css/components/_agentic_terminal.tcss"
BUNDLED_STYLESHEET = REPO_ROOT / "tldw_chatbook/css/tldw_cli_modular.tcss"


def _css_block(text: str, selector: str) -> str:
    """Return a CSS rule body starting at ``selector`` (mirrors
    ``test_library_prompts_canvas.py``'s helper of the same name)."""
    start = text.index(selector)
    block_start = text.index("{", start)
    block_end = text.index("}", block_start)
    return text[block_start:block_end]


def _two_row_state(*, sort: str = "name") -> SkillsListState:
    return SkillsListState(
        rows=(
            SkillListRow(
                name="code-review", secondary="user · agent · Reviews a diff",
                trust_glyph="✓", blocked=False,
            ),
            SkillListRow(
                name="summarize", secondary="user · agent needs review [x]",
                trust_glyph="⚠", blocked=True,
            ),
        ),
        count=2,
        sort=sort,
    )


class _CanvasHost(App):
    def __init__(self, state: SkillsListState | None, **kwargs: Any) -> None:  # type: ignore[valid-type]
        super().__init__()
        self._state = state
        self._kwargs = kwargs

    def compose(self):
        yield LibrarySkillsListCanvas(
            self._state, id="library-skills-canvas", **self._kwargs
        )


# ---------------------------------------------------------------------------
# Widget-only tests (Step 2 of the brief)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_skills_canvas_renders_a_button_per_row():
    """A 2-row state (one trusted, one blocked) renders exactly 2 skill row
    buttons, ids ``library-skill-row-<name>``."""
    app = _CanvasHost(_two_row_state())
    async with app.run_test() as pilot:
        for name in ("code-review", "summarize"):
            button = pilot.app.query_one(f"#library-skill-row-{name}", Button)
            assert button.skill_name == name
        rows = pilot.app.query(".library-skill-row")
        assert len(rows) == 2


@pytest.mark.asyncio
async def test_skills_canvas_blocked_row_has_blocked_class():
    """The needs-review (⚠) row carries ``library-skill-row-blocked`` in
    addition to the base ``library-skill-row`` class -- dim, still
    selectable (the trust panel needs it visible)."""
    app = _CanvasHost(_two_row_state())
    async with app.run_test() as pilot:
        trusted = pilot.app.query_one("#library-skill-row-code-review", Button)
        blocked = pilot.app.query_one("#library-skill-row-summarize", Button)
        assert not trusted.has_class("library-skill-row-blocked")
        assert blocked.has_class("library-skill-row-blocked")
        assert blocked.has_class("library-skill-row")


@pytest.mark.asyncio
async def test_skills_canvas_row_label_is_glyph_and_name_only():
    """The row Button's label is exactly ``f"{glyph} {name}"`` -- the
    flags/description line is a SEPARATE secondary Static, not packed into
    the same Button label (unlike the prompts canvas's two-line label)."""
    app = _CanvasHost(_two_row_state())
    async with app.run_test() as pilot:
        button = pilot.app.query_one("#library-skill-row-code-review", Button)
        assert str(button.label) == "✓ code-review"
        blocked_button = pilot.app.query_one("#library-skill-row-summarize", Button)
        assert str(blocked_button.label) == "⚠ summarize"


@pytest.mark.asyncio
async def test_skills_canvas_escapes_secondary_line_bracket_text_verbatim():
    """A skill name is impossible to seed with brackets (names are
    name-shaped), so the escape proof runs on the ``description`` shown in
    the secondary Static instead: unescaped, "[x]" would be silently
    swallowed by Rich markup parsing as an (invalid) tag."""
    app = _CanvasHost(_two_row_state())
    async with app.run_test() as pilot:
        secondaries = pilot.app.query(".library-skill-row-secondary")
        assert len(secondaries) == 2
        texts = [str(s.renderable) for s in secondaries]
        assert any("[x]" in text for text in texts)


@pytest.mark.asyncio
async def test_skills_canvas_toolbar_is_one_horizontal_row():
    """sort/Import share a single ``ds-toolbar`` Horizontal parent -- proven
    structurally (shared parentage), not via region/geometry (the bare
    harness has no app CSS loaded)."""
    app = _CanvasHost(_two_row_state())
    async with app.run_test() as pilot:
        sort_button = pilot.app.query_one("#library-skills-sort", Button)
        import_button = pilot.app.query_one("#library-skills-import", Button)
        toolbar = sort_button.parent
        assert toolbar is not None and toolbar.has_class("ds-toolbar")
        assert import_button.parent is toolbar


@pytest.mark.asyncio
async def test_skills_canvas_filter_input_prefilled():
    app = _CanvasHost(_two_row_state(), filter_value="review")
    async with app.run_test() as pilot:
        filter_input = pilot.app.query_one("#library-skills-filter", Input)
        assert filter_input.value == "review"


@pytest.mark.asyncio
async def test_skills_canvas_sort_label_reflects_sort_mode():
    app = _CanvasHost(_two_row_state(sort="status"), sort_mode="status")
    async with app.run_test() as pilot:
        sort_button = pilot.app.query_one("#library-skills-sort", Button)
        assert "Status" in str(sort_button.label)


@pytest.mark.asyncio
async def test_skills_canvas_empty_state_renders_exact_copy_not_list():
    empty_state = SkillsListState(rows=(), count=0, sort="name")
    app = _CanvasHost(empty_state)
    async with app.run_test() as pilot:
        empty = pilot.app.query_one("#library-skills-empty")
        assert str(empty.renderable) == "No skills yet — create them in Library ▸ Skills."
        assert len(pilot.app.query(".library-skill-row")) == 0


@pytest.mark.asyncio
async def test_skills_canvas_empty_state_with_filter_shows_filter_copy():
    empty_state = SkillsListState(rows=(), count=0, sort="name")
    app = _CanvasHost(empty_state, filter_value="nope")
    async with app.run_test() as pilot:
        empty = pilot.app.query_one("#library-skills-empty")
        assert "match your filter" in str(empty.renderable)


# ---------------------------------------------------------------------------
# Screen-wiring unit tests (direct-method style, mirrors
# test_library_prompts_canvas.py)
# ---------------------------------------------------------------------------


def test_build_library_skills_state_reads_local_source_records():
    fake = SimpleNamespace(
        _local_source_records={
            "skills": (
                2,
                {
                    "available_skills": [{"name": "code-review"}],
                    "blocked_skills": [{"name": "summarize"}],
                },
            )
        },
        _library_skills_filter="",
        _library_skills_sort="name",
    )
    state = LibraryScreen._build_library_skills_state(fake)
    assert state.count == 2
    assert [row.name for row in state.rows] == ["code-review", "summarize"]


def test_build_library_skills_state_tolerates_missing_entry():
    fake = SimpleNamespace(
        _local_source_records={},
        _library_skills_filter="",
        _library_skills_sort="name",
    )
    state = LibraryScreen._build_library_skills_state(fake)
    assert state.rows == ()
    assert state.count == 0


def test_handle_library_skills_sort_cycles_name_to_status():
    calls = []
    fake = SimpleNamespace(
        _library_skills_sort="name",
        refresh=lambda recompose=False: calls.append(recompose),
    )
    event = SimpleNamespace(stop=lambda: None)
    LibraryScreen.handle_library_skills_sort(fake, event)
    assert fake._library_skills_sort == "status"
    assert calls == [True]


def test_handle_library_skills_sort_cycles_status_back_to_name():
    fake = SimpleNamespace(
        _library_skills_sort="status",
        refresh=lambda recompose=False: None,
    )
    LibraryScreen.handle_library_skills_sort(fake, SimpleNamespace(stop=lambda: None))
    assert fake._library_skills_sort == "name"


def test_handle_library_skills_filter_submitted_sets_filter():
    calls = []
    fake = SimpleNamespace(
        _library_skills_filter="",
        _safe_text=LibraryScreen._safe_text,
        refresh=lambda recompose=False: calls.append(recompose),
    )
    event = SimpleNamespace(value="review", stop=lambda: None)
    LibraryScreen.handle_library_skills_filter(fake, event)
    assert fake._library_skills_filter == "review"
    assert calls == [True]


def test_handle_library_skill_row_records_selected_name():
    """Recording-only for now (mirrors the prompts canvas's original,
    pre-editor ``handle_library_prompt_row`` shape): the in-canvas skill
    detail/trust editor lands in a later task, so this just stores the
    selection for that task to pick up rather than navigating anywhere."""
    calls = []
    fake = SimpleNamespace(
        _selected_skill_name="",
        _library_selected_row_id="",
        refresh=lambda recompose=False: calls.append(recompose),
    )
    button = SimpleNamespace(skill_name="code-review")
    event = SimpleNamespace(stop=lambda: None, button=button)
    LibraryScreen.handle_library_skill_row(fake, event)
    assert fake._selected_skill_name == "code-review"
    assert fake._library_selected_row_id == LIBRARY_ROW_BROWSE_SKILLS
    assert calls == [True]


# ---------------------------------------------------------------------------
# Real end-to-end integration test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_library_shell_skills_row_press_renders_list_canvas():
    """Pressing the Skills rail row -- with a fake service exposing
    ``get_context`` (both available and blocked populations) -- renders
    ``LibrarySkillsListCanvas`` with a row button per fetched skill,
    replacing the old placeholder-empty canvas fallback."""
    app = _build_test_app()
    app.notes_scope_service = StaticLibraryNotesListScopeService([])
    app.media_reading_scope_service = StaticLibraryMediaScopeService([])
    app.chat_conversation_scope_service = StaticLibraryConversationScopeService([])
    app.skills_scope_service = _FakeSkillsScopeService(
        available=[{"name": "code-review", "description": "Reviews a diff"}],
        blocked=[{"name": "summarize", "description": "Summarizes text"}],
    )
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-skills").press()
        await pilot.pause()
        await pilot.pause()

        assert screen._library_selected_row_id == LIBRARY_ROW_BROWSE_SKILLS
        canvas = screen.query_one("#library-skills-canvas", LibrarySkillsListCanvas)
        assert canvas is not None
        trusted_row = screen.query_one("#library-skill-row-code-review", Button)
        blocked_row = screen.query_one("#library-skill-row-summarize", Button)
        assert not trusted_row.has_class("library-skill-row-blocked")
        assert blocked_row.has_class("library-skill-row-blocked")


@pytest.mark.asyncio
async def test_library_shell_skills_row_press_selects_row():
    """Pressing a skill row selects it (recording-only for now -- the
    in-canvas editor lands in a later task)."""
    app = _build_test_app()
    app.notes_scope_service = StaticLibraryNotesListScopeService([])
    app.media_reading_scope_service = StaticLibraryMediaScopeService([])
    app.chat_conversation_scope_service = StaticLibraryConversationScopeService([])
    app.skills_scope_service = _FakeSkillsScopeService(
        available=[{"name": "code-review"}],
    )
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-skills").press()
        await pilot.pause()
        await pilot.pause()

        screen.query_one("#library-skill-row-code-review", Button).press()
        await pilot.pause()

        assert screen._selected_skill_name == "code-review"


@pytest.mark.asyncio
async def test_library_shell_skills_sort_toggle_cycles_and_recomposes():
    app = _build_test_app()
    app.notes_scope_service = StaticLibraryNotesListScopeService([])
    app.media_reading_scope_service = StaticLibraryMediaScopeService([])
    app.chat_conversation_scope_service = StaticLibraryConversationScopeService([])
    app.skills_scope_service = _FakeSkillsScopeService(
        available=[{"name": "code-review"}],
        blocked=[{"name": "summarize"}],
    )
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-skills").press()
        await pilot.pause()
        await pilot.pause()

        assert screen._library_skills_sort == "name"
        screen.query_one("#library-skills-sort", Button).press()
        await pilot.pause()
        await pilot.pause()

        assert screen._library_skills_sort == "status"
        sort_button = screen.query_one("#library-skills-sort", Button)
        assert "Status" in str(sort_button.label)


@pytest.mark.asyncio
async def test_library_shell_skills_filter_submitted_rebuilds_state():
    app = _build_test_app()
    app.notes_scope_service = StaticLibraryNotesListScopeService([])
    app.media_reading_scope_service = StaticLibraryMediaScopeService([])
    app.chat_conversation_scope_service = StaticLibraryConversationScopeService([])
    app.skills_scope_service = _FakeSkillsScopeService(
        available=[{"name": "code-review"}, {"name": "translate"}],
    )
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-skills").press()
        await pilot.pause()
        await pilot.pause()

        filter_input = screen.query_one("#library-skills-filter", Input)
        filter_input.value = "review"
        filter_input.focus()
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        await pilot.pause()

        assert screen._library_skills_filter == "review"
        assert screen.query_one("#library-skill-row-code-review", Button)
        assert len(screen.query("#library-skill-row-translate")) == 0


# ---------------------------------------------------------------------------
# Stylesheet parity pin (dual-target: source + regenerated bundle) --
# mirrors test_library_prompts_canvas.py's own pin tests for its sibling
# canvas.
# ---------------------------------------------------------------------------


def test_library_skill_row_class_matches_prompt_row_visual_parity():
    """``.library-skill-row`` (the row Buttons in ``library_skills_canvas.py``)
    must have a stylesheet block, with the same width/height/border/
    background as ``.library-prompt-row`` -- visual parity with the sibling
    prompts list, not default auto-width Buttons."""
    agentic_terminal = AGENTIC_TERMINAL.read_text(encoding="utf-8")
    bundled_stylesheet = BUNDLED_STYLESHEET.read_text(encoding="utf-8")

    for text in (agentic_terminal, bundled_stylesheet):
        assert ".library-skill-row {" in text
        skill_row_block = _css_block(text, ".library-skill-row {")
        prompt_row_block = _css_block(text, ".library-prompt-row {")
        for pinned in (
            "width: 100%;",
            "height: 2;",
            "border: none;",
            "background: $ds-surface-panel;",
        ):
            assert pinned in skill_row_block
            assert pinned in prompt_row_block

        assert ".library-skill-row-blocked {" in text
        blocked_block = _css_block(text, ".library-skill-row-blocked {")
        assert "color: $ds-text-muted;" in blocked_block


def test_library_skills_header_filter_empty_have_css_blocks():
    """``#library-skills-header``/``#library-skills-filter`` (+ ``:focus``)/
    ``#library-skills-empty`` (``library_skills_canvas.py``) must have
    stylesheet rules matching their ``#library-prompts-*`` siblings, instead
    of silently falling back to unstyled defaults."""
    agentic_terminal = AGENTIC_TERMINAL.read_text(encoding="utf-8")
    bundled_stylesheet = BUNDLED_STYLESHEET.read_text(encoding="utf-8")

    for text in (agentic_terminal, bundled_stylesheet):
        assert "#library-skills-header {" in text
        assert "#library-skills-filter {" in text
        assert "#library-skills-filter:focus {" in text
        assert "#library-skills-empty {" in text

        header_block = _css_block(text, "#library-skills-header {")
        prompts_header_block = _css_block(text, "#library-prompts-header {")
        assert "height: auto;" in header_block
        assert "height: auto;" in prompts_header_block

        filter_block = _css_block(text, "#library-skills-filter {")
        prompts_filter_block = _css_block(text, "#library-prompts-filter {")
        for pinned in (
            "height: 3;",
            "border: tall $ds-grid-line;",
            "background: $ds-surface-raised;",
        ):
            assert pinned in filter_block
            assert pinned in prompts_filter_block

        focus_block = _css_block(text, "#library-skills-filter:focus {")
        prompts_focus_block = _css_block(text, "#library-prompts-filter:focus {")
        for pinned in ("border: tall $ds-input-focus-accent;", "outline: none;"):
            assert pinned in focus_block
            assert pinned in prompts_focus_block

        empty_block = _css_block(text, "#library-skills-empty {")
        prompts_empty_block = _css_block(text, "#library-prompts-empty {")
        assert "color: $ds-text-muted;" in empty_block
        assert "color: $ds-text-muted;" in prompts_empty_block
