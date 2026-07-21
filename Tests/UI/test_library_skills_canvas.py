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
from textual import events
from textual.app import App
from textual.pilot import _get_mouse_message_arguments
from textual.widgets import Button, Input, Static, TextArea

from tldw_chatbook.Library.library_shell_state import (
    LIBRARY_ROW_BROWSE_SKILLS,
    LIBRARY_ROW_CREATE_SKILL,
)
from tldw_chatbook.Library.library_skills_state import (
    SkillEditorState,
    SkillListRow,
    SkillsListState,
)
from tldw_chatbook.Skills_Interop.local_skills_service import LocalSkillsService
from tldw_chatbook.Skills_Interop.skills_scope_service import SkillsScopeService
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen
from tldw_chatbook.Widgets.Library.library_skills_canvas import (
    _TRUST_SETUP_EXPLANATION_COPY,
    LibrarySkillsListCanvas,
    skill_context_toggle_label,
    skill_disable_model_label,
    skill_editor_warning_lines,
    skill_trust_needs_setup,
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
                name="code-review",
                secondary="user · agent · Reviews a diff",
                trust_glyph="✓",
                blocked=False,
            ),
            SkillListRow(
                name="summarize",
                secondary="user · agent needs review [x]",
                trust_glyph="⚠",
                blocked=True,
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
        assert (
            str(empty.renderable)
            == "No skills yet — use Create ▸ New skill in the rail, or Import… above."
        )
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
        assert (
            pilot.app.query_one("#library-skill-description", Input).value
            == "Reviews a diff"
        )
        assert (
            pilot.app.query_one("#library-skill-argument-hint", Input).value
            == "pr number"
        )
        assert (
            pilot.app.query_one("#library-skill-allowed-tools", Input).value
            == "git.diff"
        )
        assert pilot.app.query_one("#library-skill-model", Input).value == ""
        assert pilot.app.query_one("#library-skill-model-hint", Static)
        model_hint = str(
            pilot.app.query_one("#library-skill-model-hint", Static).renderable
        )
        assert model_hint == "Not applied in v1."
        body_area = pilot.app.query_one("#library-skill-body", TextArea)
        assert body_area.text == "Review the diff."
        supporting = str(
            pilot.app.query_one("#library-skill-supporting", Static).renderable
        )
        assert "checklist.md (42 bytes)" in supporting


@pytest.mark.asyncio
async def test_skill_editor_name_input_disabled_for_existing_skill_with_rename_hint():
    """Fix wave for the review Critical (rename corruption): an existing
    skill has no rename primitive to build on, so its Name Input must be
    disabled (not just visually discouraged) with a dim explanatory hint --
    ``is_create`` defaults to ``False``, matching every editor open via a
    real skill row (as opposed to the Create rail's "New skill" row,
    which passes ``is_create=True`` -- see the next test)."""
    state = _editor_state()
    app = _EditorHost(mode="editor", editor_state=state)
    async with app.run_test() as pilot:
        name_input = pilot.app.query_one("#library-skill-name", Input)
        assert name_input.disabled is True
        assert name_input.value == "code-review"
        hint = str(pilot.app.query_one("#library-skill-name-hint", Static).renderable)
        assert hint == "Rename isn't supported — create a new skill instead."


@pytest.mark.asyncio
async def test_skill_editor_name_input_editable_on_create_branch():
    """The create branch (Create rail's "New skill" row) is the only case
    where a Name is still being chosen -- so it must stay editable, with
    no rename hint shown."""
    state = _editor_state(name="")
    app = _EditorHost(mode="editor", editor_state=state, is_create=True)
    async with app.run_test() as pilot:
        name_input = pilot.app.query_one("#library-skill-name", Input)
        assert name_input.disabled is False
        assert len(pilot.app.query("#library-skill-name-hint")) == 0


@pytest.mark.asyncio
async def test_skill_editor_toggle_and_context_button_labels_reflect_state():
    state = _editor_state(
        user_invocable=False,
        disable_model_invocation=True,
        context="fork",
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
        warnings = str(
            pilot.app.query_one("#library-skill-warnings", Static).renderable
        )
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
        trust_status="quarantined_modified",
        trust_blocked=True,
        trust_changed_files=("SKILL.md",),
    )
    app = _EditorHost(mode="editor", editor_state=state)
    async with app.run_test() as pilot:
        state_line = str(
            pilot.app.query_one("#library-skill-trust-state", Static).renderable
        )
        assert state_line == skill_trust_state_line(
            "quarantined_modified", ("SKILL.md",)
        )
        assert pilot.app.query_one("#library-skill-trust-state", Static).has_class(
            "library-skill-trust-state-blocked"
        )
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


@pytest.mark.asyncio
async def test_skill_editor_trust_panel_shows_setup_state_when_uninitialized():
    """Fix wave (Skills Phase-1 gate, FIX 2): while
    ``trust_status == "trust_uninitialized"`` (a brand-new, never-
    bootstrapped trust store), the trust panel renders the first-run setup
    state -- an explanation line plus a single "Set up skill trust" action
    -- instead of a dead Unlock/Review/Approve row (Unlock only ever
    unlocks an EXISTING manifest, so it would never enable here)."""
    state = _editor_state(trust_status="trust_uninitialized", trust_blocked=True)
    app = _EditorHost(mode="editor", editor_state=state)
    async with app.run_test() as pilot:
        state_line = str(
            pilot.app.query_one("#library-skill-trust-state", Static).renderable
        )
        assert state_line == "Trust: not initialized"
        explanation = str(
            pilot.app.query_one(
                "#library-skill-trust-setup-explanation", Static
            ).renderable
        )
        assert explanation == _TRUST_SETUP_EXPLANATION_COPY
        setup_button = pilot.app.query_one("#library-skill-trust-setup", Button)
        assert setup_button.disabled is False
        assert len(pilot.app.query("#library-skill-trust-unlock")) == 0
        assert len(pilot.app.query("#library-skill-trust-review")) == 0
        assert len(pilot.app.query("#library-skill-trust-approve")) == 0
        assert len(pilot.app.query("#library-skill-trust-review-files")) == 0


@pytest.mark.asyncio
async def test_skill_editor_trust_panel_hides_setup_state_when_not_uninitialized():
    """The inverse of the setup-state test above: any OTHER trust status
    (trusted, locked, quarantined-*) must render the normal panel, never
    the first-run setup state."""
    state = _editor_state(trust_status="quarantined_modified", trust_blocked=True)
    app = _EditorHost(mode="editor", editor_state=state)
    async with app.run_test() as pilot:
        assert len(pilot.app.query("#library-skill-trust-setup")) == 0
        assert len(pilot.app.query("#library-skill-trust-setup-explanation")) == 0
        assert pilot.app.query_one("#library-skill-trust-unlock", Button)
        assert pilot.app.query_one("#library-skill-trust-review", Button)
        assert pilot.app.query_one("#library-skill-trust-approve", Button)


def test_skill_trust_needs_setup_predicate():
    assert skill_trust_needs_setup("trust_uninitialized") is True
    for other in (
        "trusted",
        "trust_locked",
        "quarantined_modified",
        "quarantined_added",
    ):
        assert skill_trust_needs_setup(other) is False


def test_skill_trust_state_line_appends_changed_files():
    assert skill_trust_state_line("trusted") == "Trust: trusted"
    line = skill_trust_state_line("quarantined_modified", ("SKILL.md", "notes.md"))
    assert line == "Trust: changed since trusted baseline (SKILL.md, notes.md)"


def test_skill_editor_warning_lines_shadow_and_needs_review():
    assert skill_editor_warning_lines(
        live_name="summarize",
        trust_status="trusted",
        trust_blocked=False,
    ) == (
        'Saving marks this skill "needs review" — re-approve it in the trust '
        "panel after saving.",
    )
    assert skill_editor_warning_lines(
        live_name="calculator",
        trust_status="quarantined_modified",
        trust_blocked=True,
    ) == (
        'Name shadows a built-in command/tool ("calculator") — it will not be '
        "invocable as /calculator or as an agent tool.",
    )
    assert (
        skill_editor_warning_lines(
            live_name="summarize",
            trust_status="quarantined_modified",
            trust_blocked=True,
        )
        == ()
    )


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
    veto_notices: list[bool] = []
    fake = SimpleNamespace(
        _selected_skill_name="already-open",
        _library_selected_row_id=LIBRARY_ROW_BROWSE_SKILLS,
        _library_skills_view="editor",
        _flush_library_skill_save=AsyncMock(return_value=False),
        # task-413: the veto is no longer silent -- the handler reports it.
        _notify_skill_dirty_veto=lambda: veto_notices.append(True),
    )
    button = SimpleNamespace(skill_name="code-review")
    event = SimpleNamespace(stop=lambda: None, button=button)
    await LibraryScreen.handle_library_skill_row(fake, event)
    assert fake._selected_skill_name == "already-open"
    assert veto_notices == [True]


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


_TAB_BAR_CLICK_BUG_SKILL_CONTENT = (
    "---\ndescription: Summarize notes\n---\n# Summarize\nSummarize body text.\n"
)


@pytest.mark.asyncio
async def test_opening_skill_editor_does_not_break_tab_bar_click_activation(tmp_path):
    """Regression lock for the Phase-2 gate's tab-bar-click finding.

    Root cause: ``Widget.recompose()`` (what ``self.refresh(recompose=True)``
    schedules) unconditionally removes and remounts every child of the
    screen. If ``App.mouse_captured`` happens to reference one of those
    children -- e.g. an ``Input`` mid click/selection whose ``MouseUp``
    hasn't arrived yet (plausible over textual-serve's websocket transport,
    where down/up travel as independently-timed messages) -- ``Input`` has
    no ``_on_hide`` handler to release the mouse on removal, unlike
    ``TextArea``/``ScrollBar`` (Textual's other two mouse-capturing
    widgets), which both do. ``mouse_captured`` is then left referencing a
    removed widget FOREVER: every subsequent mouse event anywhere in the
    app -- routed through ``Screen._forward_event``/``_handle_mouse_move``,
    both of which special-case ``if self.app.mouse_captured: ...
    self.find_widget(widget)`` -- hits ``NoWidget`` and is silently
    swallowed, permanently breaking click dispatch app-wide (including the
    top nav bar's own buttons). The Library skills detail editor is a
    reachable trigger: opening it recomposes the screen (list -> "Loading
    skill..." -> full editor), and any Input on the canvas a user's mouse
    is mid-interacting with when that fires can leak the capture.

    Fixed in ``BaseAppScreen.refresh`` (``tldw_chatbook/UI/Navigation/
    base_app_screen.py``): release any active mouse capture BEFORE a
    recompose, mirroring the same defensive ``capture_mouse(None)`` call
    Textual's own ``push_screen``/``switch_screen``/``_replace_screen``
    already make before swapping screens -- this same-screen content
    recompose path never had that protection.

    This test drives the REAL production path end to end: a real
    ``LocalSkillsService``/``SkillsScopeService``, a real skill row click to
    open the in-canvas editor, a simulated in-flight ``MouseDown`` on the
    Description ``Input`` (capturing the mouse) with no matching
    ``MouseUp``, an ordinary "Back to list" click (recomposing the canvas
    away from under it), and then a real click on the top nav bar's Console
    button -- which must still dispatch. A focused-widget Enter keypress
    (an entirely different, mouse-capture-oblivious dispatch path) is
    asserted too, confirming keyboard activation was never at risk either
    way.
    """
    local_service = LocalSkillsService(
        store_dir=tmp_path,
        trust_service=None,
        allow_untrusted_without_trust_service=True,
        policy_enforcer=None,
    )
    await local_service.create_skill(
        name="summarize-notes", content=_TAB_BAR_CLICK_BUG_SKILL_CONTENT
    )
    service = SkillsScopeService(
        local_service=local_service, server_service=None, policy_enforcer=None
    )

    app = _build_test_app()
    app.notes_scope_service = StaticLibraryNotesListScopeService([])
    app.media_reading_scope_service = StaticLibraryMediaScopeService([])
    app.chat_conversation_scope_service = StaticLibraryConversationScopeService([])
    app.skills_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-skills").press()
        await pilot.pause()
        await pilot.pause()

        screen.query_one("#library-skill-row-summarize-notes", Button).press()
        for _ in range(150):
            if screen._library_skill_detail is not None:
                break
            await pilot.pause(0.02)
        await pilot.pause()
        assert screen._library_skills_view == "editor"

        # Simulate a user starting a click/selection gesture in the
        # Description Input -- MouseDown captures the mouse -- whose
        # MouseUp never arrives before the screen recomposes.
        description_input = screen.query_one("#library-skill-description", Input)
        message_arguments = _get_mouse_message_arguments(
            description_input, (1, 0), button=1
        )
        host.screen._forward_event(events.MouseDown(**message_arguments))
        await pilot.pause()
        assert host.app.mouse_captured is description_input

        # An ordinary "Back to list" click recomposes the canvas, removing
        # the Description Input (and every other editor widget) entirely.
        screen.query_one("#library-skill-back", Button).press()
        await pilot.pause()
        await pilot.pause()
        assert screen._library_skills_view == "list"

        # THE FIX: mouse capture must have been released before that
        # recompose, not left dangling on the now-removed Input.
        assert host.app.mouse_captured is None

        # A real click on the top nav bar's Console button must still work.
        nav_button = screen.query_one("#nav-console", Button)
        await pilot.click(nav_button)
        await pilot.pause()
        await pilot.pause()
        assert "chat" in host.seen_routes

        # Keyboard activation (an entirely separate, mouse-capture-oblivious
        # dispatch path) was never at risk, but is asserted for completeness.
        host.seen_routes.clear()
        nav_button_2 = screen.query_one("#nav-artifacts", Button)
        nav_button_2.focus()
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        await pilot.pause()
        assert "artifacts" in host.seen_routes


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


def test_library_skills_import_row_css_blocks_match_prompt_parity():
    """``#library-skills-import-path`` and ``#library-skills-import-status``
    (the inline import row in ``library_skills_canvas.py``, Task 5) must have
    stylesheet blocks matching their ``#library-prompts-import-*`` siblings,
    with same field look (tall-border/focus-accent Input) and muted status
    line -- dual-pinned against both the source module AND the regenerated
    bundle."""
    agentic_terminal = AGENTIC_TERMINAL.read_text(encoding="utf-8")
    bundled_stylesheet = BUNDLED_STYLESHEET.read_text(encoding="utf-8")

    for text in (agentic_terminal, bundled_stylesheet):
        assert "#library-skills-import-path {" in text
        assert "#library-skills-import-path:focus {" in text
        assert "#library-skills-import-status {" in text

        import_path_block = _css_block(text, "#library-skills-import-path {")
        prompts_import_path_block = _css_block(text, "#library-prompts-import-path {")
        for pinned in (
            "width: 100%;",
            "height: 3;",
            "border: tall $ds-grid-line;",
            "background: $ds-surface-raised;",
            "color: $ds-text-primary;",
        ):
            assert pinned in import_path_block
            assert pinned in prompts_import_path_block

        import_path_focus_block = _css_block(
            text, "#library-skills-import-path:focus {"
        )
        prompts_import_path_focus_block = _css_block(
            text, "#library-prompts-import-path:focus {"
        )
        for pinned in (
            "border: tall $ds-input-focus-accent;",
            "outline: none;",
            "background: $ds-input-focus-bg;",
            "text-style: bold;",
        ):
            assert pinned in import_path_focus_block
            assert pinned in prompts_import_path_focus_block

        import_status_block = _css_block(text, "#library-skills-import-status {")
        prompts_import_status_block = _css_block(
            text, "#library-prompts-import-status {"
        )
        for pinned in (
            "width: 100%;",
            "height: auto;",
            "color: $ds-text-muted;",
        ):
            assert pinned in import_status_block
            assert pinned in prompts_import_status_block


def test_library_skill_trust_setup_explanation_css_block_matches_review_files_parity():
    """``#library-skill-trust-setup-explanation`` (the first-run trust
    setup state's explanation line, gate fix wave FIX 2) must have a
    stylesheet block with the same muted secondary-line look as its
    ``#library-skill-trust-review-files`` sibling -- dual-pinned against
    both the source module AND the regenerated bundle, same pattern as
    every other Skills CSS pin above."""
    agentic_terminal = AGENTIC_TERMINAL.read_text(encoding="utf-8")
    bundled_stylesheet = BUNDLED_STYLESHEET.read_text(encoding="utf-8")

    for text in (agentic_terminal, bundled_stylesheet):
        assert "#library-skill-trust-setup-explanation {" in text
        setup_block = _css_block(text, "#library-skill-trust-setup-explanation {")
        review_files_block = _css_block(text, "#library-skill-trust-review-files {")
        for pinned in (
            "width: 100%;",
            "height: auto;",
            "color: $ds-text-muted;",
        ):
            assert pinned in setup_block
            assert pinned in review_files_block


@pytest.mark.asyncio
async def test_library_shell_rail_switch_vetoed_while_skill_editor_dirty():
    """task-412 P0 regression: switching Library rail rows while the skill
    editor holds an unsaved edit must veto the switch -- the same contract
    ``_select_library_rail_row`` already enforces for dirty note and prompt
    edits. Before the fix the skill flush was omitted from that guard, so a
    rail-row press silently discarded the edit (verified live in the
    2026-07-21 Skills UX review)."""
    app = _build_test_app()
    app.notes_scope_service = StaticLibraryNotesListScopeService([])
    app.media_reading_scope_service = StaticLibraryMediaScopeService([])
    app.chat_conversation_scope_service = StaticLibraryConversationScopeService([])
    app.skills_scope_service = _FakeSkillsScopeService(available=[], blocked=[])
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-create-skill").press()
        await pilot.pause()
        await pilot.pause()
        assert screen._library_skills_view == "editor"
        assert screen._library_selected_row_id == LIBRARY_ROW_CREATE_SKILL

        # Real user edit: the armed Name Input marks the editor dirty.
        screen.query_one("#library-skill-name", Input).value = "dirty-demo"
        await pilot.pause()
        assert screen._library_skill_dirty is True

        screen.query_one("#library-row-browse-media").press()
        await pilot.pause()
        await pilot.pause()

        # Vetoed: still in the skill editor with the unsaved edit intact.
        assert screen._library_selected_row_id == LIBRARY_ROW_CREATE_SKILL
        assert screen._library_skills_view == "editor"
        assert screen.query_one("#library-skill-name", Input).value == "dirty-demo"


@pytest.mark.asyncio
async def test_library_skill_back_veto_notifies_unsaved_changes():
    """task-413: the Back-to-list dirty veto must produce visible feedback
    (a notification) instead of silently doing nothing -- before the fix
    the vetoed click gave zero indication why navigation was blocked."""
    app = _build_test_app()
    app.notes_scope_service = StaticLibraryNotesListScopeService([])
    app.media_reading_scope_service = StaticLibraryMediaScopeService([])
    app.chat_conversation_scope_service = StaticLibraryConversationScopeService([])
    app.skills_scope_service = _FakeSkillsScopeService(available=[], blocked=[])
    notifications: list[tuple[str, dict]] = []
    app.notify = lambda message, **kwargs: notifications.append((message, kwargs))
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-create-skill").press()
        await pilot.pause()
        await pilot.pause()
        screen.query_one("#library-skill-name", Input).value = "dirty-demo"
        await pilot.pause()
        assert screen._library_skill_dirty is True

        screen.query_one("#library-skill-back").press()
        await pilot.pause()
        await pilot.pause()

        assert screen._library_skills_view == "editor"
        assert notifications, "Vetoed Back gave no visible feedback."
        assert "Unsaved skill changes" in notifications[-1][0]


@pytest.mark.asyncio
async def test_library_shell_rail_switch_veto_notifies_unsaved_changes():
    """task-413 companion to the task-412 guard: when the rail switch is
    vetoed by a dirty skill edit the user must be told why."""
    app = _build_test_app()
    app.notes_scope_service = StaticLibraryNotesListScopeService([])
    app.media_reading_scope_service = StaticLibraryMediaScopeService([])
    app.chat_conversation_scope_service = StaticLibraryConversationScopeService([])
    app.skills_scope_service = _FakeSkillsScopeService(available=[], blocked=[])
    notifications: list[tuple[str, dict]] = []
    app.notify = lambda message, **kwargs: notifications.append((message, kwargs))
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-create-skill").press()
        await pilot.pause()
        await pilot.pause()
        screen.query_one("#library-skill-name", Input).value = "dirty-demo"
        await pilot.pause()

        screen.query_one("#library-row-browse-media").press()
        await pilot.pause()
        await pilot.pause()

        assert screen._library_selected_row_id == LIBRARY_ROW_CREATE_SKILL
        assert notifications, "Vetoed rail switch gave no visible feedback."
        assert "Unsaved skill changes" in notifications[-1][0]


@pytest.mark.asyncio
async def test_library_skill_discard_button_leaves_without_saving():
    """task-413: an explicit Discard affordance -- disabled until dirty,
    live-enabled on the first edit, and pressing it leaves the editor
    without saving (list view, dirty cleared). Before the fix the only
    exit from a dirty editor was Save."""
    app = _build_test_app()
    app.notes_scope_service = StaticLibraryNotesListScopeService([])
    app.media_reading_scope_service = StaticLibraryMediaScopeService([])
    app.chat_conversation_scope_service = StaticLibraryConversationScopeService([])
    app.skills_scope_service = _FakeSkillsScopeService(available=[], blocked=[])
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-create-skill").press()
        await pilot.pause()
        await pilot.pause()

        discard = screen.query_one("#library-skill-discard", Button)
        assert discard.disabled is True

        screen.query_one("#library-skill-name", Input).value = "dirty-demo"
        await pilot.pause()
        assert screen._library_skill_dirty is True
        assert discard.disabled is False

        discard.press()
        await pilot.pause()
        await pilot.pause()

        assert screen._library_skills_view == "list"
        assert screen._library_skill_dirty is False


@pytest.mark.asyncio
async def test_library_flush_pending_work_skill_veto_notifies():
    """task-413: a dirty skill edit vetoing screen-level navigation
    (``flush_pending_work``) must surface the same unsaved-changes toast
    as the in-screen exits -- the app-level caller only logs the veto."""
    app = _build_test_app()
    app.notes_scope_service = StaticLibraryNotesListScopeService([])
    app.media_reading_scope_service = StaticLibraryMediaScopeService([])
    app.chat_conversation_scope_service = StaticLibraryConversationScopeService([])
    app.skills_scope_service = _FakeSkillsScopeService(available=[], blocked=[])
    notifications: list[tuple[str, dict]] = []
    app.notify = lambda message, **kwargs: notifications.append((message, kwargs))
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-create-skill").press()
        await pilot.pause()
        await pilot.pause()
        screen.query_one("#library-skill-name", Input).value = "dirty-demo"
        await pilot.pause()

        assert await screen.flush_pending_work() is False
        assert notifications, "Screen-leave veto gave no visible feedback."
        assert "Unsaved skill changes" in notifications[-1][0]


# ---------------------------------------------------------------------------
# task-414: Review changes must show the content under review, and a failed
# approve must say why (snapshot_mismatch) instead of the generic toast.
# ---------------------------------------------------------------------------


def test_skill_trust_review_preview_renders_per_file_content():
    from tldw_chatbook.Widgets.Library.library_skills_canvas import (
        skill_trust_review_preview,
    )

    preview = skill_trust_review_preview(
        {
            "changed_files": ["SKILL.md", "notes.md"],
            "current_files": {
                "SKILL.md": "---\nname: x\n---\nBody text.",
                "notes.md": "supporting notes",
            },
        }
    )
    assert "SKILL.md" in preview
    assert "Body text." in preview
    assert "notes.md" in preview
    assert "supporting notes" in preview


def test_skill_trust_review_preview_labels_deleted_files():
    from tldw_chatbook.Widgets.Library.library_skills_canvas import (
        skill_trust_review_preview,
    )

    preview = skill_trust_review_preview(
        {"changed_files": ["gone.md"], "current_files": {}}
    )
    assert "gone.md" in preview
    assert "deleted" in preview.lower()


def test_skill_trust_review_preview_truncates_huge_files():
    from tldw_chatbook.Widgets.Library.library_skills_canvas import (
        skill_trust_review_preview,
    )

    preview = skill_trust_review_preview(
        {
            "changed_files": ["SKILL.md"],
            "current_files": {"SKILL.md": "x" * 100_000},
        }
    )
    assert len(preview) < 20_000
    assert "truncated" in preview


def test_skill_trust_review_preview_empty_without_review():
    from tldw_chatbook.Widgets.Library.library_skills_canvas import (
        skill_trust_review_preview,
    )

    assert skill_trust_review_preview(None) == ""
    assert skill_trust_review_preview({}) == ""


@pytest.mark.asyncio
async def test_skill_editor_trust_panel_renders_review_content_preview():
    """task-414: with an active review the trust panel must show the actual
    file content under review -- before the fix only the filename list
    rendered, so Approve was blind sign-off."""
    state = _editor_state(
        trust_status="quarantined_modified",
        trust_blocked=True,
        trust_changed_files=("SKILL.md",),
    )
    app = _EditorHost(
        mode="editor",
        editor_state=state,
        active_review={
            "review_id": "r1",
            "changed_files": ["SKILL.md"],
            "current_files": {"SKILL.md": "reviewable body content"},
        },
    )
    async with app.run_test() as pilot:
        content = pilot.app.query_one("#library-skill-trust-review-content", Static)
        text = str(content.renderable)
        assert "reviewable body content" in text


@pytest.mark.asyncio
async def test_render_trust_panel_patches_review_content_in_place():
    """task-414: capturing a review patches the content preview without a
    recompose (same in-place contract as the changed-files line)."""
    app = _build_test_app()
    app.notes_scope_service = StaticLibraryNotesListScopeService([])
    app.media_reading_scope_service = StaticLibraryMediaScopeService([])
    app.chat_conversation_scope_service = StaticLibraryConversationScopeService([])
    app.skills_scope_service = _FakeSkillsScopeService(available=[], blocked=[])
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-create-skill").press()
        await pilot.pause()
        await pilot.pause()
        # task-416: create mode renders no trust panel, so patch-test the
        # existing-skill shape (selected name set -> is_create False).
        screen._selected_skill_name = "code-review"
        screen.refresh(recompose=True)
        await pilot.pause()
        await pilot.pause()

        screen._library_skill_active_review = {
            "review_id": "r1",
            "changed_files": ["SKILL.md"],
            "current_files": {"SKILL.md": "patched preview content"},
        }
        screen._render_library_skill_trust_panel()
        await pilot.pause()

        content = screen.query_one("#library-skill-trust-review-content", Static)
        assert "patched preview content" in str(content.renderable)


@pytest.mark.asyncio
async def test_trust_service_call_uses_failure_copy_override():
    """task-414: a trust-service failure whose message has a registered
    override must toast that specific copy instead of the generic one."""
    notifications: list[str] = []

    class _MismatchService:
        def trust_reviewed_snapshot(self, review_id):
            raise ValueError("snapshot_mismatch")

    fake = SimpleNamespace(
        app_instance=SimpleNamespace(
            local_skill_trust_service=_MismatchService(),
            notify=lambda message, **kwargs: notifications.append(message),
        ),
    )
    result, ok = await LibraryScreen._call_library_skill_trust_service(
        fake,
        "trust_reviewed_snapshot",
        "r1",
        failure_copy={"snapshot_mismatch": "specific mismatch copy"},
    )
    assert ok is False
    assert notifications == ["specific mismatch copy"]


@pytest.mark.asyncio
async def test_approve_failure_discards_stale_review():
    """task-414: the service discards the review on every
    ``trust_reviewed_snapshot`` raise, so a failed approve must not leave
    the UI holding a dead review with Approve still enabled."""
    render_calls: list[bool] = []
    refresh_calls: list[bool] = []

    async def _call(method_name, *args, **kwargs):
        if method_name == "unlock_with_passphrase":
            return None, True
        return None, False

    async def _refresh_status():
        refresh_calls.append(True)

    fake = SimpleNamespace(
        _library_skills_view="editor",
        _library_skill_active_review={"review_id": "r1"},
        _selected_skill_name="code-review",
        _request_library_skill_trust_passphrase=AsyncMock(return_value="pw"),
        _call_library_skill_trust_service=_call,
        _render_library_skill_trust_panel=lambda: render_calls.append(True),
        _refresh_library_skill_trust_status=_refresh_status,
    )
    await LibraryScreen._approve_library_skill_trust(fake)
    assert fake._library_skill_active_review is None
    assert render_calls == [True]
    assert refresh_calls == [True]


# ---------------------------------------------------------------------------
# task-415: Delete needs an explicit confirmation step (inline two-step,
# mirroring the notes/media confirming-delete pattern), and create mode must
# not render a Delete button at all (nothing exists to delete).
# ---------------------------------------------------------------------------


def test_skill_delete_confirm_copy_names_skill_and_supporting_files():
    from tldw_chatbook.Widgets.Library.library_skills_canvas import (
        skill_delete_confirm_copy,
    )

    copy = skill_delete_confirm_copy("code-review", 2)
    assert "code-review" in copy
    assert "2 supporting files" in copy
    bare = skill_delete_confirm_copy("code-review", 0)
    assert "supporting" not in bare
    assert "cannot be undone" in bare


@pytest.mark.asyncio
async def test_skill_editor_confirming_delete_renders_confirm_row():
    state = _editor_state()
    app = _EditorHost(mode="editor", editor_state=state, confirming_delete=True)
    async with app.run_test() as pilot:
        copy = pilot.app.query_one("#library-skill-delete-confirm-copy", Static)
        text = str(copy.renderable)
        assert "code-review" in text
        assert pilot.app.query_one("#library-skill-delete-confirm", Button)
        assert pilot.app.query_one("#library-skill-delete-cancel", Button)
        assert not pilot.app.query("#library-skill-save")
        assert not pilot.app.query("#library-skill-delete")


@pytest.mark.asyncio
async def test_skill_editor_create_mode_renders_no_delete_button():
    """A brand-new unsaved skill has nothing on disk to delete -- the old
    always-rendered Delete was a silent no-op in create mode."""
    state = _editor_state()
    app = _EditorHost(mode="editor", editor_state=state, is_create=True)
    async with app.run_test() as pilot:
        assert pilot.app.query_one("#library-skill-save", Button)
        assert not pilot.app.query("#library-skill-delete")


@pytest.mark.asyncio
async def test_handle_library_skill_delete_enters_confirm_state():
    """First Delete press arms the inline confirmation -- it must NOT kick
    the delete worker anymore."""
    worker_calls: list[dict] = []
    refresh_calls: list[bool] = []
    after_refresh: list[Any] = []
    fake = SimpleNamespace(
        _library_skills_view="editor",
        _selected_skill_name="code-review",
        _library_skill_confirming_delete=False,
        _library_skill_editor_state=_editor_state(),
        _library_skill_editor_armed=True,
        _snapshot_library_skill_live_fields=lambda: None,
        _arm_library_skill_editor=lambda: None,
        run_worker=lambda coro, **kwargs: worker_calls.append(kwargs),
        refresh=lambda recompose=False: refresh_calls.append(recompose),
        call_after_refresh=lambda fn: after_refresh.append(fn),
        is_mounted=True,
    )
    event = SimpleNamespace(stop=lambda: None)
    LibraryScreen.handle_library_skill_delete(fake, event)
    assert fake._library_skill_confirming_delete is True
    assert worker_calls == []
    assert refresh_calls == [True]


@pytest.mark.asyncio
async def test_handle_library_skill_delete_confirm_kicks_delete_worker():
    worker_calls: list[dict] = []
    fake = SimpleNamespace(
        _library_skills_view="editor",
        _selected_skill_name="code-review",
        _library_skill_confirming_delete=True,
        _delete_library_skill=lambda name: None,
        run_worker=lambda coro, **kwargs: worker_calls.append(kwargs),
    )
    event = SimpleNamespace(stop=lambda: None)
    LibraryScreen.handle_library_skill_delete_confirm(fake, event)
    assert worker_calls and worker_calls[0]["group"] == "library_skill_delete"


@pytest.mark.asyncio
async def test_handle_library_skill_delete_cancel_leaves_confirm_state():
    refresh_calls: list[bool] = []
    fake = SimpleNamespace(
        _library_skills_view="editor",
        _library_skill_confirming_delete=True,
        _library_skill_editor_armed=True,
        _arm_library_skill_editor=lambda: None,
        refresh=lambda recompose=False: refresh_calls.append(recompose),
        call_after_refresh=lambda fn: None,
        is_mounted=True,
    )
    event = SimpleNamespace(stop=lambda: None)
    LibraryScreen.handle_library_skill_delete_cancel(fake, event)
    assert fake._library_skill_confirming_delete is False
    assert refresh_calls == [True]


# ---------------------------------------------------------------------------
# task-416: create mode must not render the trust panel -- a never-saved
# skill has no on-disk files, so "Trust: trusted" + Unlock/Review/Approve
# was a false state with dead buttons.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_skill_editor_create_mode_renders_no_trust_panel():
    state = _editor_state()
    app = _EditorHost(mode="editor", editor_state=state, is_create=True)
    async with app.run_test() as pilot:
        assert not pilot.app.query("#library-skill-trust-panel")


@pytest.mark.asyncio
async def test_skill_editor_existing_skill_still_renders_trust_panel():
    state = _editor_state()
    app = _EditorHost(mode="editor", editor_state=state, is_create=False)
    async with app.run_test() as pilot:
        assert pilot.app.query_one("#library-skill-trust-panel")


# ---------------------------------------------------------------------------
# task-417: the create-save recompose must land back on the action row
# (not snap to the top away from the just-pressed Save), and "Saved." must
# not persist as stale status across later edits/trust actions.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_skill_editor_scroll_to_actions_lands_on_save_row():
    """With ``scroll_to_actions`` the freshly-mounted editor canvas scrolls
    its action row into view instead of starting at the top."""
    state = _editor_state()
    app = _EditorHost(mode="editor", editor_state=state, scroll_to_actions=True)
    async with app.run_test(size=(80, 12)) as pilot:
        await pilot.pause()
        await pilot.pause()
        canvas = pilot.app.query_one("#library-skills-canvas")
        assert canvas.scroll_offset.y > 0


@pytest.mark.asyncio
async def test_create_save_success_arms_scroll_to_actions():
    app = _build_test_app()
    app.notes_scope_service = StaticLibraryNotesListScopeService([])
    app.media_reading_scope_service = StaticLibraryMediaScopeService([])
    app.chat_conversation_scope_service = StaticLibraryConversationScopeService([])
    app.skills_scope_service = _FakeSkillsScopeService(available=[], blocked=[])
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-create-skill").press()
        await pilot.pause()
        await pilot.pause()

        screen._apply_library_skill_save_success(
            {"name": "fresh-skill", "description": "d", "body": "b", "version": 1},
            is_create=True,
        )
        assert screen._library_skill_scroll_pending is True


@pytest.mark.asyncio
async def test_mark_dirty_clears_stale_saved_status():
    """Typing after a save must clear the lingering 'Saved.' -- the status
    otherwise stays wrong across any number of later edits."""
    app = _build_test_app()
    app.notes_scope_service = StaticLibraryNotesListScopeService([])
    app.media_reading_scope_service = StaticLibraryMediaScopeService([])
    app.chat_conversation_scope_service = StaticLibraryConversationScopeService([])
    app.skills_scope_service = _FakeSkillsScopeService(available=[], blocked=[])
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-create-skill").press()
        await pilot.pause()
        await pilot.pause()

        screen._update_library_skill_status_static("Saved.")
        screen.query_one("#library-skill-name", Input).value = "editing-again"
        await pilot.pause()

        assert screen._library_skill_status == ""
        status = screen.query_one("#library-skill-save-status", Static)
        assert str(status.renderable) == ""


@pytest.mark.asyncio
async def test_trust_review_press_clears_stale_saved_status():
    cleared: list[str] = []
    fake = SimpleNamespace(
        _update_library_skill_status_static=lambda text: cleared.append(text),
        _review_library_skill_trust=lambda: None,
        run_worker=lambda coro, **kwargs: None,
    )
    event = SimpleNamespace(stop=lambda: None)
    LibraryScreen.handle_library_skill_trust_review(fake, event)
    assert cleared == [""]


# ---------------------------------------------------------------------------
# task-418: copy pass -- self-referential empty state, jargon toggle
# labels, approve-purpose passphrase modal.
# ---------------------------------------------------------------------------


def test_skills_empty_state_copy_names_real_paths():
    from tldw_chatbook.Widgets.Library.library_skills_canvas import (
        _EMPTY_SKILLS_COPY,
    )

    assert "New skill" in _EMPTY_SKILLS_COPY
    assert "Import" in _EMPTY_SKILLS_COPY
    # The old copy pointed at "Library ▸ Skills" -- the list the user is
    # already looking at.
    assert "Library ▸ Skills" not in _EMPTY_SKILLS_COPY


def test_skill_toggle_labels_read_as_plain_statements():
    assert skill_user_invocable_label(True) == "User can invoke: yes ▸"
    assert skill_user_invocable_label(False) == "User can invoke: no ▸"
    # Polarity inverted for display: the stored field stays
    # disable_model_invocation, the label answers the user's question.
    assert skill_disable_model_label(False) == "Agent can invoke: yes ▸"
    assert skill_disable_model_label(True) == "Agent can invoke: no ▸"
    assert (
        skill_context_toggle_label("inline") == "Runs in: inline (this conversation) ▸"
    )
    assert skill_context_toggle_label("fork") == "Runs in: fork (sub-agent) ▸"


@pytest.mark.asyncio
async def test_trust_passphrase_modal_accepts_purpose_copy():
    """task-418: Approve re-uses the passphrase modal, which always said
    'Unlock Local Skill Trust' -- a task/dialog mismatch. The modal now
    takes purpose copy overrides."""
    from tldw_chatbook.UI.Screens.skills_screen import SkillTrustPassphraseModal

    class _Host(App):
        pass

    app = _Host()
    async with app.run_test() as pilot:
        modal = SkillTrustPassphraseModal(
            confirm_bootstrap=False,
            title="Approve Reviewed Skill Version",
            message="Enter the trust passphrase to approve.",
        )
        app.push_screen(modal)
        await pilot.pause()
        title = modal.query_one("#skill-trust-passphrase-title", Static)
        assert str(title.renderable) == "Approve Reviewed Skill Version"
