"""Real-storage Prompt save continuity through the production Library shell."""

from __future__ import annotations

import pytest
from textual.widgets import Button, Input, Static, TextArea

from Tests.UI.test_library_prompts_canvas import (
    _build_test_app,
    _real_prompt_scope_service,
    _wire_empty_non_prompt_services,
)
from Tests.UI.test_library_shell import (
    LibraryProductionCSSHarness,
    _active_library_screen,
    _painted_text,
    _wait_for_condition,
    _wait_for_library_shell,
    _wait_for_selector,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("seeded", [False, True])
@pytest.mark.parametrize("size", [(170, 48), (80, 24)])
@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
async def test_first_prompt_save_settles_items_without_leaving_editor(
    tmp_path, seeded, size, theme
):
    db, service = _real_prompt_scope_service(tmp_path)
    if seeded:
        db.add_prompt(
            name="Earlier prompt", author="", details="", user_prompt="Existing content"
        )
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    app.app_config.setdefault("library", {})["prompt_editor_mode"] = "basic"
    host = LibraryProductionCSSHarness(app)
    host.theme = theme

    async with host.run_test(size=size) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-create-prompt", Button).press()
        name = await _wait_for_selector(screen, pilot, "#library-prompt-name")
        assert isinstance(name, Input)
        await _wait_for_condition(
            pilot,
            lambda: (
                screen._library_prompt_browse_controller.result.status
                == ("ready" if seeded else "empty_library")
            ),
            message="New prompt did not load its resident Items pane",
        )
        if not seeded:
            empty = await _wait_for_selector(screen, pilot, "#library-prompts-empty")
            assert "No prompts yet" in str(empty.renderable)
        name.value = "Continuity [bold] café"
        content = screen.query_one("#library-prompt-user", TextArea)
        content.text = "Retain this message template after saving."
        content.focus()
        await pilot.pause()
        screen.query_one("#library-prompt-save", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: (
                screen._prompts_state.selected_prompt_id is not None
                and not screen._prompts_state.dirty
            ),
            message="Prompt save did not settle",
        )
        prompt_id = screen._prompts_state.selected_prompt_id
        persisted = db.fetch_prompt_details(prompt_id)
        assert persisted["user_prompt"] == content.text
        assert persisted["version"] == 1
        await _wait_for_condition(
            pilot,
            lambda: (
                screen._library_snapshot_rendered_generation
                == screen._library_snapshot_state_generation
            ),
            message="Library count projection did not settle",
        )
        await _wait_for_condition(
            pilot,
            lambda: screen._library_prompt_browse_controller.result.status == "ready",
            message=lambda: repr(screen._library_prompt_browse_controller.result),
        )
        row = await _wait_for_selector(
            screen, pilot, f"#library-prompt-row-{prompt_id}"
        )
        assert isinstance(row, Button)
        assert not row.disabled
        assert screen.query_one("#library-prompt-user", TextArea) is content
        assert screen.query_one("#library-prompt-name", Input) is name
        assert not screen.query_one("#library-prompt-save", Button).display
        assert screen.query_one("#library-prompt-insert-console", Button).display
        assert screen.focused is content
        assert "Retain this" in _painted_text(host, content.region)
        assert screen._library_prompt_browse_controller.visible_result.total_items == (
            2 if seeded else 1
        )

        # Both modes expose the same saved content through keyboard controls.
        advanced = screen.query_one("#library-prompt-mode-advanced", Button)
        advanced.focus()
        await pilot.press("enter")
        await _wait_for_condition(
            pilot,
            lambda: screen._prompts_state.editor_mode == "advanced",
            message="Keyboard Advanced activation did not apply",
        )
        block = screen.query(".prompt-block-content").first(TextArea)
        assert block.text == content.text
        assert all(
            "Unsaved changes" not in str(status.renderable)
            for status in screen.query(".prompt-block-status")
        )
        assert "structured format" in str(
            screen.query_one("#library-prompt-artifact-status", Static).renderable
        )
        assert "structured format" in str(
            screen.query_one("#library-prompt-info-provenance", Static).renderable
        )
        block.focus()
        await _wait_for_condition(
            pilot,
            lambda: (
                screen.focused is block
                and "Retain this" in _painted_text(host, block.region)
            ),
            message="Focused Advanced content did not become readable",
        )
        assert "Retain this" in _painted_text(host, block.region)
        basic = screen.query_one("#library-prompt-mode-basic", Button)
        basic.focus()
        await pilot.press("enter")
        await _wait_for_condition(
            pilot,
            lambda: screen._prompts_state.editor_mode == "basic",
            message="Keyboard Basic activation did not apply",
        )
        assert screen.query_one("#library-prompt-user", TextArea) is content
        assert not screen._prompts_state.dirty
        back = screen.query_one("#library-prompt-back", Button)
        back.focus()
        await pilot.press("enter")
        await _wait_for_selector(screen, pilot, "#library-prompt-work-empty")
        await _wait_for_condition(
            pilot,
            lambda: (
                not screen.query_one(
                    f"#library-prompt-row-{prompt_id}", Button
                ).disabled
            ),
            message="Saved Prompt row did not become ready on return",
        )
        row = screen.query_one(f"#library-prompt-row-{prompt_id}", Button)
        row.focus()
        await pilot.press("enter")
        reopened = await _wait_for_selector(screen, pilot, "#library-prompt-user")
        assert isinstance(reopened, TextArea)
        assert reopened.text == persisted["user_prompt"]
        assert (
            screen.query_one("#library-prompt-name", Input).value == persisted["name"]
        )
        assert "v1" in str(screen.query_one("#library-prompt-meta", Static).renderable)
        assert db.fetch_prompt_details(prompt_id)["version"] == 1
