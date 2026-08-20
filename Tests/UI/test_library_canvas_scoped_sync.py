"""TASK-15457: Library per-click canvas-scoped update evidence."""

from __future__ import annotations

from statistics import median
from time import perf_counter
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from textual.app import ComposeResult

# Harness apps load the consolidated widget CSS the real app loads
# (TASK-15450); without it the widgets under test mount unstyled.
from Tests.UI.consolidated_css import ConsolidatedCSSApp
from textual.widgets import Button, Static

from Tests.UI.app_factory import _build_test_app
from Tests.Skills.test_skills_library_flow import (
    _real_skills_scope_service,
    _skill_content,
    _wire_empty_non_skill_services,
)
from Tests.UI.test_library_prompts_canvas import (
    _real_prompt_scope_service,
    _wire_empty_non_prompt_services,
)
from Tests.UI.test_library_shell import (
    LIBRARY_NAV_CONTEXT_INGEST,
    LIBRARY_TEST_SIZE,
    LibraryHarness,
    _active_library_screen,
    _seed_conversations,
    _two_conversations,
    _two_media_items,
    _two_notes,
    _wait_for_library_shell,
    _wait_for_selector,
)
from tldw_chatbook.Library.library_ingest_state import (
    LibraryIngestFormState,
    build_library_ingest_state,
)
from tldw_chatbook.Library.library_notes_sync_state import auto_sync_label
from tldw_chatbook.UI.Navigation.base_app_screen import BaseAppScreen
from tldw_chatbook.UI.Screens import library_screen as library_screen_module
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen
from tldw_chatbook.Widgets.Library import (
    LibraryIngestCanvas,
    LibraryMediaCanvas,
    LibraryNotesCanvas,
    LibraryPromptsListCanvas,
    LibrarySearchRagPanel,
    LibrarySkillsListCanvas,
)


def _screen_recompose_spy() -> tuple[list[BaseAppScreen], object]:
    """Return a call list and a spy that preserves normal refresh behavior."""
    calls: list[BaseAppScreen] = []
    original = BaseAppScreen.refresh

    def spy(self, *args, **kwargs):
        if kwargs.get("recompose"):
            calls.append(self)
        return original(self, *args, **kwargs)

    return calls, spy


async def _wait_for_widget_text(
    screen: LibraryScreen,
    pilot,
    selector: str,
    expected: str,
    *,
    attempts: int = 80,
) -> None:
    """Wait for a recomposed widget to expose its new visible state."""
    for _ in range(attempts):
        matches = list(screen.query(selector))
        if matches:
            widget = matches[0]
            visible = getattr(widget, "label", getattr(widget, "renderable", ""))
            if expected in str(visible):
                return
        await pilot.pause(0.02)
    raise AssertionError(f"{selector} did not render {expected!r}")


@pytest.mark.asyncio
async def test_loading_surfaces_keep_unicode_copy_and_notes_sync_hook() -> None:
    """Canvas-owned loading views retain their user-facing glyphs and copy."""

    class LoadingCanvasApp(ConsolidatedCSSApp):
        def compose(self) -> ComposeResult:
            yield LibraryNotesCanvas(mode="loading", id="notes")
            yield LibraryPromptsListCanvas(mode="loading", id="prompts")
            yield LibrarySkillsListCanvas(mode="loading", id="skills")

    app = LoadingCanvasApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        notes = app.query_one("#notes", LibraryNotesCanvas)
        assert callable(notes.sync_state)
        assert str(app.query_one("#library-note-back", Button).label) == "‹ Notes"
        assert str(app.query_one("#library-note-loading", Static).renderable) == (
            "Loading note…"
        )
        assert str(app.query_one("#library-prompt-loading", Static).renderable) == (
            "Loading prompt…"
        )
        assert str(app.query_one("#library-skill-loading", Static).renderable) == (
            "Loading skill…"
        )


@pytest.mark.asyncio
async def test_notes_per_click_updates_keep_screen_and_canvas_identity() -> None:
    """Notes toggle/select/sort interactions never remount the Library shell."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-notes").press()
        await _wait_for_selector(screen, pilot, "#library-notes-select-toggle")
        rail_before = screen.query_one("#library-rail")
        canvas_before = screen.query_one("#library-notes-canvas", LibraryNotesCanvas)
        calls, spy = _screen_recompose_spy()

        with patch.object(BaseAppScreen, "refresh", spy):
            screen.query_one("#library-notes-select-toggle").press()
            await _wait_for_selector(screen, pilot, "#library-notes-select-all")
            screen.query_one("#library-notes-select-all").press()
            await _wait_for_widget_text(
                screen, pilot, "#library-notes-selected-count", "2 selected"
            )
            assert screen._library_notes_row_selection.count == 2
            screen.query_one("#library-notes-select-toggle").press()
            await _wait_for_selector(screen, pilot, "#library-notes-sort")
            screen.query_one("#library-notes-sort").press()
            await _wait_for_selector(screen, pilot, "#library-notes-sort-oldest")
            screen.query_one("#library-notes-sort-oldest").press()
            await _wait_for_widget_text(
                screen, pilot, "#library-notes-sort", "Oldest"
            )
            screen.query_one("#library-notes-sync-open").press()
            await _wait_for_selector(screen, pilot, "#library-notes-sync-folder")
            screen.query_one(
                "#library-notes-sync-direction-disk_to_db"
            ).press()
            await _wait_for_widget_text(
                screen,
                pilot,
                "#library-notes-sync-direction-disk_to_db",
                "✓",
            )
            screen.query_one("#library-notes-sync-conflict-disk_wins").press()
            await _wait_for_widget_text(
                screen,
                pilot,
                "#library-notes-sync-conflict-disk_wins",
                "✓",
            )
            screen.query_one("#library-notes-sync-auto").press()
            await _wait_for_widget_text(
                screen, pilot, "#library-notes-sync-auto", auto_sync_label(True)
            )
            screen.query_one("#library-notes-sync-auto").press()
            await _wait_for_widget_text(
                screen, pilot, "#library-notes-sync-auto", auto_sync_label(False)
            )
            screen.query_one("#library-notes-sync-back").press()
            await _wait_for_selector(screen, pilot, "#library-notes-sync-open")
            screen.query_one("#library-notes-row-0").press()
            await _wait_for_selector(screen, pilot, "#library-note-title")

        assert calls == []
        assert screen.query_one("#library-rail") is rail_before
        assert (
            screen.query_one("#library-notes-canvas", LibraryNotesCanvas)
            is canvas_before
        )
        assert screen._library_notes_sort == "oldest"


@pytest.mark.asyncio
async def test_media_choice_and_rag_toggles_are_canvas_scoped() -> None:
    """Media chooser/Escape and Search/RAG toggles preserve shell identity."""
    app = _build_test_app()
    _seed_conversations(
        app,
        _two_conversations(),
        notes=_two_notes(),
        media=_two_media_items(),
    )
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-browse-media").press()
        await _wait_for_selector(screen, pilot, "#library-media-type-filter")
        media_rail = screen.query_one("#library-rail")
        media_canvas = screen.query_one("#library-media-canvas", LibraryMediaCanvas)
        calls, spy = _screen_recompose_spy()
        with patch.object(BaseAppScreen, "refresh", spy):
            screen.query_one("#library-media-type-filter").press()
            await _wait_for_selector(screen, pilot, "#library-media-type-choices")
            await pilot.press("escape")
            await _wait_for_selector(screen, pilot, "#library-media-type-filter")
        assert calls == []
        assert screen.query_one("#library-rail") is media_rail
        assert screen.query_one("#library-media-canvas", LibraryMediaCanvas) is media_canvas

        screen.query_one("#library-row-browse-search").press()
        await _wait_for_selector(screen, pilot, "#library-rag-mode-toggle")
        rag_rail = screen.query_one("#library-rail")
        rag_panel = screen.query_one("#library-search-rag-panel", LibrarySearchRagPanel)
        calls.clear()
        with patch.object(BaseAppScreen, "refresh", spy):
            screen.query_one("#library-rag-mode-toggle").press()
            await pilot.pause()
            screen.query_one("#library-rag-scope-toggle-media").press()
            await pilot.pause()
        assert calls == []
        assert screen.query_one("#library-rail") is rag_rail
        assert (
            screen.query_one("#library-search-rag-panel", LibrarySearchRagPanel)
            is rag_panel
        )


@pytest.mark.asyncio
async def test_prompt_and_skill_row_handlers_route_to_their_canvas() -> None:
    """The two list-to-loading row transitions use their canvas hook."""
    kinds: list[str] = []

    async def permitted() -> bool:
        return True

    skill_screen = SimpleNamespace(
        _flush_library_skill_save=permitted,
        _notify_skill_dirty_veto=Mock(),
        _reset_library_skill_editor_state=Mock(),
        _selected_skill_name="",
        _library_selected_row_id="",
        _library_skills_view="list",
        run_worker=Mock(),
    )
    skill_event = SimpleNamespace(
        stop=Mock(), button=SimpleNamespace(skill_name=None)
    )
    prompt_screen = SimpleNamespace(
        _library_prompts_mutation_in_flight=False,
        _flush_library_prompt_save=permitted,
        _invalidate_library_prompts_browse=Mock(),
        _clear_library_prompt_selection=Mock(),
        _reset_library_prompt_editor_state=Mock(),
        _refresh_library_prompt_detail=Mock(return_value=object()),
        _selected_prompt_id=None,
        _library_prompt_select_mode=False,
        _library_selected_row_id="",
        _library_prompts_view="list",
        run_worker=Mock(),
    )
    prompt_event = SimpleNamespace(
        stop=Mock(), button=SimpleNamespace(prompt_id=1)
    )

    with patch.object(
        library_screen_module,
        "_sync_library_canvas",
        side_effect=lambda _screen, kind: kinds.append(kind),
    ):
        await LibraryScreen.handle_library_skill_row(skill_screen, skill_event)
        await LibraryScreen.handle_library_prompt_row(prompt_screen, prompt_event)

    assert kinds == ["skills", "prompts"]
    assert skill_screen._library_skills_view == "editor"
    assert prompt_screen._library_prompts_view == "editor"


@pytest.mark.asyncio
async def test_real_prompt_and_skill_rows_keep_their_canvas_identity(tmp_path) -> None:
    """Real service-backed list-to-editor loads avoid the screen fallback."""
    prompt_path = tmp_path / "prompts"
    prompt_path.mkdir()
    prompt_db, prompt_service = _real_prompt_scope_service(prompt_path)
    prompt_id, _uuid, _message = prompt_db.add_prompt(
        name="Scoped prompt",
        author="Test",
        details="Canvas identity probe",
        system_prompt="Stay concise.",
        user_prompt="Summarize {text}",
        keywords=["test"],
    )
    prompt_app = _build_test_app()
    _wire_empty_non_prompt_services(prompt_app)
    prompt_app.prompt_scope_service = prompt_service
    prompt_host = LibraryHarness(prompt_app)

    async with prompt_host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(prompt_host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-prompts").press()
        row = await _wait_for_selector(
            screen, pilot, f"#library-prompt-row-{prompt_id}"
        )
        canvas = screen.query_one("#library-prompts-canvas")
        calls, spy = _screen_recompose_spy()
        with patch.object(BaseAppScreen, "refresh", spy):
            row.press()
            await _wait_for_selector(screen, pilot, "#library-prompt-name")
        assert calls == []
        assert screen.query_one("#library-prompts-canvas") is canvas

    skill_path = tmp_path / "skills"
    skill_path.mkdir()
    local_skills, skills_service = _real_skills_scope_service(skill_path)
    await local_skills.create_skill(
        name="scoped-skill",
        content=_skill_content(title="Scoped", description="Identity probe"),
    )
    skills_app = _build_test_app()
    _wire_empty_non_skill_services(skills_app)
    skills_app.skills_scope_service = skills_service
    skills_app.local_skill_trust_service = None
    skills_host = LibraryHarness(skills_app)

    async with skills_host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(skills_host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-skills").press()
        row = await _wait_for_selector(
            screen, pilot, "#library-skill-row-scoped-skill"
        )
        canvas = screen.query_one("#library-skills-canvas")
        calls, spy = _screen_recompose_spy()
        with patch.object(BaseAppScreen, "refresh", spy):
            row.press()
            await _wait_for_selector(screen, pilot, "#library-skill-name")
        assert calls == []
        assert screen.query_one("#library-skills-canvas") is canvas


def test_ingest_checkbox_routes_to_ingest_canvas_sync() -> None:
    """A structural ingest checkbox edit rebuilds only the ingest canvas."""
    kinds: list[str] = []
    form = LibraryIngestFormState()
    screen = SimpleNamespace(
        _library_ingest_form=form,
        _invalidate_library_external_submission=Mock(),
        _disarm_library_ingest_start_confirm=Mock(),
        _disarm_library_ingest_retry_confirm=Mock(),
    )
    event = SimpleNamespace(
        stop=Mock(), group="generic", name="analyze", value=True
    )
    with patch.object(
        library_screen_module,
        "_sync_library_canvas",
        side_effect=lambda _screen, kind: kinds.append(kind),
    ):
        LibraryScreen.handle_library_ingest_option_value_changed(screen, event)

    assert kinds == ["ingest"]
    assert form.analyze is True
    assert form.type_options["generic"]["analyze"] is True


@pytest.mark.asyncio
@pytest.mark.allow_network
async def test_ingest_backend_switch_recomposes_only_the_ingest_canvas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Local/Server selection keeps the Library shell, rail, and footer mounted."""
    backend = {"value": "local"}
    app = _build_test_app()
    app._resolve_ingest_backend = lambda: backend["value"]
    _seed_conversations(app, ())
    screen = LibraryScreen(app)
    screen._build_library_ingest_state = lambda: build_library_ingest_state(
        (), form=screen._library_ingest_form, ingest_backend=backend["value"],
        runtime_source="server", server_ingest_available=True,
    )
    screen.apply_navigation_context({LIBRARY_NAV_CONTEXT_INGEST: True})
    host = LibraryHarness(app, screen=screen)

    monkeypatch.setattr(
        library_screen_module,
        "save_setting_to_cli_config",
        lambda _section, _key, value: backend.__setitem__("value", value) or True,
    )

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _wait_for_selector(screen, pilot, "#library-ingest-backend-switch")
        rail = screen.query_one("#library-rail")
        footer = screen.query_one("#screen-footer-status")
        canvas = screen.query_one("#library-ingest-canvas", LibraryIngestCanvas)
        calls, spy = _screen_recompose_spy()

        with patch.object(BaseAppScreen, "refresh", spy):
            screen.query_one("#library-ingest-backend-switch", Button).press()
            await pilot.pause()

        assert backend["value"] == "server"
        assert calls == []
        assert screen.query_one("#library-rail") is rail
        assert screen.query_one("#screen-footer-status") is footer
        assert screen.query_one("#library-ingest-canvas", LibraryIngestCanvas) is canvas


def test_import_status_lines_patch_the_mounted_static_without_recompose() -> None:
    """One-line Prompt/Skill receipts update their existing Static only."""
    prompt_line = SimpleNamespace(update=Mock())
    prompt_screen = SimpleNamespace(
        is_mounted=True,
        _library_selected_row_id=library_screen_module.LIBRARY_ROW_BROWSE_PROMPTS,
        _library_prompts_mutation_in_flight=False,
        _library_prompts_import_status="",
        query_one=Mock(return_value=prompt_line),
    )
    prompt_screen.app = SimpleNamespace(screen=prompt_screen)
    skill_line = SimpleNamespace(update=Mock())
    skill_screen = SimpleNamespace(
        is_mounted=True,
        _library_selected_row_id=library_screen_module.LIBRARY_ROW_BROWSE_SKILLS,
        _library_skills_import_status="",
        query_one=Mock(return_value=skill_line),
    )
    skill_screen.app = SimpleNamespace(screen=skill_screen)

    with patch.object(
        library_screen_module,
        "_sync_library_canvas",
        side_effect=AssertionError("mounted status line must not recompose"),
    ):
        LibraryScreen._apply_library_prompts_import_status(
            prompt_screen, "2 imported"
        )
        prompt_screen._library_selected_row_id = (
            library_screen_module.LIBRARY_ROW_BROWSE_NOTES
        )
        LibraryScreen._apply_library_prompts_import_status(
            prompt_screen, "Imported after navigation"
        )
        LibraryScreen._apply_library_skills_import_status(skill_screen, "Imported")
        skill_screen._library_selected_row_id = (
            library_screen_module.LIBRARY_ROW_BROWSE_NOTES
        )
        LibraryScreen._apply_library_skills_import_status(
            skill_screen, "Imported after navigation"
        )

    prompt_line.update.assert_called_once_with("2 imported")
    skill_line.update.assert_called_once_with("Imported")
    assert prompt_screen._library_prompts_import_status == "2 imported"
    assert skill_screen._library_skills_import_status == "Imported after navigation"


@pytest.mark.asyncio
async def test_notes_select_toggle_latency_probe() -> None:
    """Measure the mounted Notes toggle with an identical before/after probe.

    The task records the median externally; this test deliberately has no
    timing threshold because wall-clock assertions are not stable CI evidence.
    Its behavioral assertion is that every measured click completes the
    expected list/select-mode transition.
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    host = LibraryHarness(app)
    samples: list[float] = []

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-notes").press()
        await _wait_for_selector(screen, pilot, "#library-notes-select-toggle")

        for index in range(12):
            started = perf_counter()
            screen.query_one("#library-notes-select-toggle").press()
            expected = (
                "#library-notes-select-all"
                if index % 2 == 0
                else "#library-notes-sort"
            )
            await _wait_for_selector(screen, pilot, expected)
            samples.append((perf_counter() - started) * 1000.0)

    assert len(samples) == 12
    assert all(sample > 0 for sample in samples)
    print(
        "TASK-15457 notes-toggle latency: "
        f"median={median(samples):.3f}ms samples="
        + ",".join(f"{sample:.3f}" for sample in samples)
    )
