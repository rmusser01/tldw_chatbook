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
from unittest.mock import AsyncMock

import pytest
from textual.app import App
from textual.widgets import Button, Input, Static, TextArea

from tldw_chatbook.Library.library_shell_state import LIBRARY_ROW_BROWSE_SKILLS
from tldw_chatbook.Library.library_skills_state import (
    SkillEditorState,
    SkillListRow,
    SkillsListState,
)
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen
from tldw_chatbook.Widgets.Library.library_skills_canvas import (
    LibrarySkillsListCanvas,
    skill_context_toggle_label,
    skill_disable_model_label,
    skill_editor_warning_lines,
    skill_trust_state_line,
    skill_user_invocable_label,
)

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
# Editor widget-only tests (Task 4): mounts LibrarySkillsListCanvas directly
# with mode="editor", mirroring test_library_prompts_canvas.py's own
# _compose_editor widget tests.
# ---------------------------------------------------------------------------


def _editor_state(
    *,
    name: str = "code-review",
    description: str = "Reviews a diff",
    argument_hint: str | None = "pr number",
    allowed_tools_csv: str = "git.diff",
    user_invocable: bool = True,
    disable_model_invocation: bool = False,
    context: str = "inline",
    model: str | None = None,
    body: str = "Review the diff.",
    supporting_files: tuple[tuple[str, int], ...] = (("checklist.md", 42),),
    version: int | None = 3,
    trust_status: str = "trusted",
    trust_blocked: bool = False,
    trust_changed_files: tuple[str, ...] = (),
) -> SkillEditorState:
    return SkillEditorState(
        name=name,
        description=description,
        argument_hint=argument_hint,
        allowed_tools_csv=allowed_tools_csv,
        user_invocable=user_invocable,
        disable_model_invocation=disable_model_invocation,
        context=context,
        model=model,
        body=body,
        supporting_files=supporting_files,
        version=version,
        trust_status=trust_status,
        trust_blocked=trust_blocked,
        trust_changed_files=trust_changed_files,
    )


class _EditorHost(App):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self._kwargs = kwargs

    def compose(self):
        yield LibrarySkillsListCanvas(id="library-skills-canvas", **self._kwargs)


@pytest.mark.asyncio
async def test_skill_editor_renders_all_field_ids_populated():
    state = _editor_state()
    app = _EditorHost(mode="editor", editor_state=state)
    async with app.run_test() as pilot:
        assert pilot.app.query_one("#library-skill-name", Input).value == "code-review"
        assert pilot.app.query_one("#library-skill-description", Input).value == "Reviews a diff"
        assert pilot.app.query_one("#library-skill-argument-hint", Input).value == "pr number"
        assert pilot.app.query_one("#library-skill-allowed-tools", Input).value == "git.diff"
        assert pilot.app.query_one("#library-skill-model", Input).value == ""
        assert pilot.app.query_one("#library-skill-model-hint", Static)
        model_hint = str(pilot.app.query_one("#library-skill-model-hint", Static).renderable)
        assert model_hint == "Not applied in v1."
        body_area = pilot.app.query_one("#library-skill-body", TextArea)
        assert body_area.text == "Review the diff."
        supporting = str(pilot.app.query_one("#library-skill-supporting", Static).renderable)
        assert "checklist.md (42 bytes)" in supporting


@pytest.mark.asyncio
async def test_skill_editor_toggle_and_context_button_labels_reflect_state():
    state = _editor_state(
        user_invocable=False, disable_model_invocation=True, context="fork",
    )
    app = _EditorHost(mode="editor", editor_state=state)
    async with app.run_test() as pilot:
        user_button = pilot.app.query_one("#library-skill-user-invocable", Button)
        assert str(user_button.label) == skill_user_invocable_label(False)
        disable_button = pilot.app.query_one("#library-skill-disable-model", Button)
        assert str(disable_button.label) == skill_disable_model_label(True)
        context_button = pilot.app.query_one("#library-skill-context", Button)
        assert str(context_button.label) == skill_context_toggle_label("fork")


@pytest.mark.asyncio
async def test_skill_editor_warnings_static_shows_screen_computed_text():
    state = _editor_state()
    app = _EditorHost(mode="editor", editor_state=state, warnings="a warning line")
    async with app.run_test() as pilot:
        warnings = str(pilot.app.query_one("#library-skill-warnings", Static).renderable)
        assert warnings == "a warning line"


@pytest.mark.asyncio
async def test_skill_editor_non_conflict_mode_renders_save_and_delete():
    state = _editor_state()
    app = _EditorHost(mode="editor", editor_state=state)
    async with app.run_test() as pilot:
        assert pilot.app.query_one("#library-skill-save", Button)
        assert pilot.app.query_one("#library-skill-delete", Button)
        assert len(pilot.app.query("#library-skill-conflict-reload")) == 0
        assert pilot.app.query_one("#library-skill-save-status", Static)


@pytest.mark.asyncio
async def test_skill_editor_conflict_mode_renders_reload_only():
    state = _editor_state()
    app = _EditorHost(mode="editor", editor_state=state, conflict=True)
    async with app.run_test() as pilot:
        assert pilot.app.query_one("#library-skill-conflict-reload", Button)
        assert pilot.app.query_one("#library-skill-conflict-copy", Static)
        assert len(pilot.app.query("#library-skill-save")) == 0
        assert len(pilot.app.query("#library-skill-delete")) == 0
        assert len(pilot.app.query("#library-skill-save-status")) == 0


@pytest.mark.asyncio
async def test_skill_editor_trust_panel_shows_state_line_and_gates_buttons():
    state = _editor_state(
        trust_status="quarantined_modified", trust_blocked=True,
        trust_changed_files=("SKILL.md",),
    )
    app = _EditorHost(mode="editor", editor_state=state)
    async with app.run_test() as pilot:
        state_line = str(pilot.app.query_one("#library-skill-trust-state", Static).renderable)
        assert state_line == skill_trust_state_line("quarantined_modified", ("SKILL.md",))
        assert pilot.app.query_one(
            "#library-skill-trust-state", Static
        ).has_class("library-skill-trust-state-blocked")
        unlock_button = pilot.app.query_one("#library-skill-trust-unlock", Button)
        assert unlock_button.disabled is True
        review_button = pilot.app.query_one("#library-skill-trust-review", Button)
        assert review_button.disabled is False
        approve_button = pilot.app.query_one("#library-skill-trust-approve", Button)
        assert approve_button.disabled is True
        review_files = str(
            pilot.app.query_one("#library-skill-trust-review-files", Static).renderable
        )
        assert review_files == ""


@pytest.mark.asyncio
async def test_skill_editor_trust_panel_approve_enabled_with_active_review():
    state = _editor_state(trust_status="quarantined_modified", trust_blocked=True)
    app = _EditorHost(
        mode="editor",
        editor_state=state,
        active_review={"review_id": "r1", "changed_files": ["SKILL.md"]},
    )
    async with app.run_test() as pilot:
        approve_button = pilot.app.query_one("#library-skill-trust-approve", Button)
        assert approve_button.disabled is False
        review_files = str(
            pilot.app.query_one("#library-skill-trust-review-files", Static).renderable
        )
        assert review_files == "SKILL.md"


def test_skill_trust_state_line_appends_changed_files():
    assert skill_trust_state_line("trusted") == "Trust: trusted"
    line = skill_trust_state_line("quarantined_modified", ("SKILL.md", "notes.md"))
    assert line == "Trust: changed since trusted baseline (SKILL.md, notes.md)"


def test_skill_editor_warning_lines_shadow_and_needs_review():
    assert skill_editor_warning_lines(
        live_name="summarize", trust_status="trusted", trust_blocked=False,
    ) == (
        'Saving marks this skill "needs review" — re-approve it in the trust '
        "panel after saving.",
    )
    assert skill_editor_warning_lines(
        live_name="calculator", trust_status="quarantined_modified", trust_blocked=True,
    ) == (
        'Name shadows a built-in command/tool ("calculator") — it will not be '
        "invocable as /calculator or as an agent tool.",
    )
    assert skill_editor_warning_lines(
        live_name="summarize", trust_status="quarantined_modified", trust_blocked=True,
    ) == ()


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


@pytest.mark.asyncio
async def test_handle_library_skill_row_opens_editor_and_records_selected_name():
    """Task 4 supersedes the recording-only shape ``handle_library_skill_row``
    had in Task 3: pressing a row now also switches into editor mode and
    kicks the detail-fetch worker (the full real-service open flow is
    covered end to end by ``Tests/Skills/test_skills_library_flow.py``;
    this direct-method test pins the SYNCHRONOUS side of the handler --
    state assignment + worker kickoff -- against a minimal fake ``self``)."""
    worker_calls = []
    reset_calls = []
    refresh_calls = []
    fake = SimpleNamespace(
        _selected_skill_name="",
        _library_selected_row_id="",
        _library_skills_view="list",
        _flush_library_skill_save=AsyncMock(return_value=True),
        _reset_library_skill_editor_state=lambda: reset_calls.append(True),
        # ``run_worker``'s first positional arg is evaluated eagerly (it's
        # the coroutine-producing call itself), so this needs a real (but
        # inert) callable rather than a bare attribute -- returning ``None``
        # is fine since the fake ``run_worker`` below never awaits it.
        _refresh_library_skill_detail=lambda name: None,
        run_worker=lambda coro, **kwargs: worker_calls.append(kwargs),
        refresh=lambda recompose=False: refresh_calls.append(recompose),
    )
    button = SimpleNamespace(skill_name="code-review")
    event = SimpleNamespace(stop=lambda: None, button=button)
    await LibraryScreen.handle_library_skill_row(fake, event)
    assert fake._selected_skill_name == "code-review"
    assert fake._library_selected_row_id == LIBRARY_ROW_BROWSE_SKILLS
    assert fake._library_skills_view == "editor"
    assert reset_calls == [True]
    assert worker_calls and worker_calls[0]["group"] == "library_skill_detail"
    assert refresh_calls == [True]


@pytest.mark.asyncio
async def test_handle_library_skill_row_vetoed_while_dirty():
    """Mirrors ``handle_library_prompt_row``'s dirty veto: switching rows
    while the currently-open skill is dirty must NOT reset state or open a
    new fetch."""
    fake = SimpleNamespace(
        _selected_skill_name="already-open",
        _library_selected_row_id=LIBRARY_ROW_BROWSE_SKILLS,
        _library_skills_view="editor",
        _flush_library_skill_save=AsyncMock(return_value=False),
    )
    button = SimpleNamespace(skill_name="code-review")
    event = SimpleNamespace(stop=lambda: None, button=button)
    await LibraryScreen.handle_library_skill_row(fake, event)
    assert fake._selected_skill_name == "already-open"


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


def test_library_skill_name_input_css_blocks_match_prompt_name_parity():
    """``#library-skill-name`` (the editor's Name Input, Task 4) must have a
    stylesheet block matching its ``#library-prompt-name`` sibling's field
    look (same tall-border/focus-accent Input styling), dual-pinned against
    both the source module AND the regenerated bundle -- mirrors
    ``test_library_skills_header_filter_empty_have_css_blocks`` above."""
    agentic_terminal = AGENTIC_TERMINAL.read_text(encoding="utf-8")
    bundled_stylesheet = BUNDLED_STYLESHEET.read_text(encoding="utf-8")

    for text in (agentic_terminal, bundled_stylesheet):
        assert "#library-skill-name," in text or "#library-skill-name {" in text
        skill_name_block = _css_block(text, "#library-skill-name")
        prompt_name_block = _css_block(text, "#library-prompt-name")
        for pinned in (
            "height: 3;",
            "border: tall $ds-grid-line;",
            "background: $ds-surface-raised;",
            "color: $ds-text-primary;",
        ):
            assert pinned in skill_name_block
            assert pinned in prompt_name_block

        assert "#library-skill-warnings {" in text
        warnings_block = _css_block(text, "#library-skill-warnings {")
        assert "color: $ds-status-warning;" in warnings_block

        assert "#library-skill-trust-state.library-skill-trust-state-blocked {" in text
        blocked_state_block = _css_block(
            text, "#library-skill-trust-state.library-skill-trust-state-blocked {"
        )
        assert "color: $ds-text-muted;" in blocked_state_block
