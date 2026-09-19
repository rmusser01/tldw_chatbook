"""Prompt resize preserves the live field through the production Library shell."""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock

import pytest
from textual.widgets import Input, TextArea

from Tests.UI.test_library_prompts_canvas import (
    _build_test_app,
    _open_prompt_editor,
    _real_prompt_scope_service,
    _wire_empty_non_prompt_services,
)
from Tests.UI.test_library_shell import (
    LibraryProductionCSSHarness,
    _active_library_screen,
    _painted_text,
    _wait_for_library_shell,
)
from tldw_chatbook.Widgets.Library import LibraryAdaptiveReaderShell


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["basic", "advanced"])
@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
async def test_prompt_resize_keeps_live_field_painted_without_data_work(
    tmp_path, monkeypatch, mode, theme
):
    db, service = _real_prompt_scope_service(tmp_path)
    prompt_id, _, _ = db.add_prompt(
        name="Resize focus",
        author="",
        details="",
        user_prompt="Keep this message visible.",
    )
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    app.app_config.setdefault("library", {})["prompt_editor_mode"] = mode
    host = LibraryProductionCSSHarness(app)
    host.theme = theme

    async with host.run_test(size=(170, 48)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompt_editor(screen, pilot, prompt_id)
        selector = (
            "#library-prompt-user" if mode == "basic" else ".prompt-block-content"
        )
        field = screen.query(selector).first(TextArea)
        field.focus()
        await pilot.pause()
        assert "Keep this" in _painted_text(host, field.region)

        reads = {
            name: AsyncMock(wraps=getattr(service, name))
            for name in ("list_prompts", "count_prompts", "get_prompt")
        }
        for name, spy in reads.items():
            monkeypatch.setattr(service, name, spy)
        persistence = AsyncMock(wraps=screen._persist_library_reader_preference)
        monkeypatch.setattr(screen, "_persist_library_reader_preference", persistence)
        snapshot = Mock(wraps=screen._refresh_local_source_snapshot)
        monkeypatch.setattr(screen, "_refresh_local_source_snapshot", snapshot)
        observations = []
        for size in ((80, 24), (170, 48), (170, 24), (170, 48)):
            await pilot.resize_terminal(*size)
            await pilot.pause()
            assert screen.query(selector).first(TextArea) is field
            assert field.text == "Keep this message visible."
            observations.append(
                (
                    size,
                    screen.focused is field,
                    "Keep this" in _painted_text(host, field.region),
                    repr(screen.focused),
                    tuple(field.region),
                )
            )

        assert all(focused and painted for _, focused, painted, *_ in observations), (
            observations
        )
        for spy in reads.values():
            spy.assert_not_called()
        persistence.assert_not_called()
        snapshot.assert_not_called()
        assert db.fetch_prompt_details(prompt_id)["version"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("outside_editor", [False, True])
@pytest.mark.parametrize("size", [(80, 24), (170, 24)])
async def test_newer_focus_wins_over_deferred_prompt_resize(
    tmp_path, monkeypatch, outside_editor, size
):
    _, service = _real_prompt_scope_service(tmp_path)
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryProductionCSSHarness(app)
    async with host.run_test(size=(170, 48)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await screen._select_library_rail_row("create-prompt")
        await pilot.pause()
        field = screen.query_one("#library-prompt-user", TextArea)
        field.text = "Keep this unsaved draft."
        field.focus()
        await pilot.pause()
        shell = screen.query_one(
            "#library-prompts-reader-shell", LibraryAdaptiveReaderShell
        )
        viewport = screen.query_one("#library-prompt-editor-content")

        # Hold the work pane's real after-layout callbacks until a newer focus
        # action has settled. Replaying them must neither restore the old field
        # nor scroll its viewport away from the user's chosen position.
        pending = []

        def defer(callback, *args, **kwargs):
            pending.append((callback, args, kwargs))
            return True

        monkeypatch.setattr(shell.work, "call_after_refresh", defer)
        await pilot.resize_terminal(*size)
        assert pending
        viewport.scroll_home(animate=False)
        target = (
            shell.library_grip
            if outside_editor
            else screen.query_one("#library-prompt-name", Input)
        )
        target.focus()
        await pilot.pause()
        offset = viewport.scroll_offset
        for callback, args, kwargs in pending:
            callback(*args, **kwargs)
        await pilot.pause()
        assert screen.focused is target
        assert viewport.scroll_offset == offset
        assert field.text == "Keep this unsaved draft."
        assert screen._prompts_state.dirty
        if not outside_editor:
            assert "Name" in _painted_text(host, target.region.grow((1, 0, 0, 0)))
