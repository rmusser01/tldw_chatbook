"""Production-CSS keyboard journeys through local Skill editing."""

from __future__ import annotations

import asyncio
import threading

import pytest
import yaml
from textual.widgets import Input, Static, TextArea

from Tests.UI.test_library_prompt_collection_journeys import _focus
from Tests.UI.test_library_shell import (
    LibraryProductionCSSHarness,
    _active_library_screen,
    _wait_for_condition,
    _wait_for_library_shell,
    _wait_for_selector,
)
from Tests.UI.test_library_skills_canvas import _build_test_app, _open_real_skill_editor
from Tests.UI.test_library_skills_reader import _wire_skills
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen
from tldw_chatbook.Widgets.Library.library_skills_canvas import LibrarySkillsListCanvas


async def _activate(screen, host, pilot, selector, label):
    target = await _wait_for_selector(screen, pilot, selector)
    target.focus()
    await _focus(screen, host, pilot, selector, label)
    await _wait_for_condition(
        pilot,
        lambda: not target.has_class("-active"),
        message="Action feedback did not settle",
    )
    await pilot.press("enter")
    await pilot.pause()


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(170, 48), (80, 24)])
@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
async def test_skill_edit_save_and_discard_keyboard_journey(
    tmp_path, monkeypatch, size, theme
):
    app = _build_test_app()
    app.library_new_profile_admission = True
    local = _wire_skills(app, tmp_path)
    await local.create_skill(
        name="editor-journey",
        content=(
            "---\nname: editor-journey\ndescription: Saved description\n"
            "allowed-tools: fs_read fs_read unknown_tool\n---\nOriginal body."
        ),
    )
    original = await local.get_skill("editor-journey")
    host = LibraryProductionCSSHarness(app)
    host.theme = theme
    app.notify = host.notify
    async with host.run_test(size=size) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await screen._select_library_rail_row("browse-skills")
        await _wait_for_selector(screen, pilot, "#library-skills-items-grip")
        shell = screen.query_one("#library-skills-reader-shell")
        await _wait_for_condition(
            pilot,
            lambda: shell.effective_layout.reader_width > 0 and shell.region.width > 0,
            message="Skills layout did not settle",
        )
        if not shell.effective_layout.items_open:
            await _activate(screen, host, pilot, "#library-skills-items-grip", "--->")
        await _activate(
            screen, host, pilot, "#library-skill-row-editor-journey", "editor-journey"
        )
        await _wait_for_selector(screen, pilot, "#library-skill-mode-edit")
        await _activate(screen, host, pilot, "#library-skill-mode-edit", "Edit")
        await _focus(screen, host, pilot, "#library-skill-mode-edit", "Edit")
        body = screen.query_one("#library-skill-body", TextArea)
        description = screen.query_one("#library-skill-description", Input)
        description.value = "Edited [bold] café"
        body.text = "Saved body {literal}."
        await pilot.pause()
        assert screen._skills_state.dirty
        await _activate(
            screen, host, pilot, "#library-skill-editor-mode", "Show advanced"
        )
        assert screen.query_one("#library-skill-body") is body
        assert description.value == "Edited [bold] café"
        await _activate(screen, host, pilot, "#library-skill-editor-mode", "Show basic")
        assert body.text == "Saved body {literal}."
        for selector, label in (
            ("#library-skill-user-invocable", "User can invoke"),
            ("#library-skill-disable-model", "Agent can invoke"),
        ):
            await _activate(screen, host, pilot, selector, label)
            await _activate(screen, host, pilot, selector, label)
            assert screen.query_one("#library-skill-body") is body
            assert body.text == "Saved body {literal}."
            assert description.value == "Edited [bold] café"
        entered, release = threading.Event(), threading.Event()
        update = local.update_skill

        async def held_update(*args, **kwargs):
            entered.set()
            await asyncio.to_thread(release.wait, 10)
            return await update(*args, **kwargs)

        monkeypatch.setattr(local, "update_skill", held_update)
        await _activate(screen, host, pilot, "#library-skill-save", "Save changes")
        await _wait_for_condition(pilot, entered.is_set, message="Save not admitted")
        await pilot.pause()
        release.set()
        await _wait_for_condition(
            pilot,
            lambda: (
                not screen._skills_state.dirty
                and not screen._skills_state.mutation_in_flight
            ),
            message="Skill save did not settle",
        )
        await _focus(screen, host, pilot, "#library-skill-back", "Back to list")
        assert screen.query_one("#library-skill-body") is body
        saved = await local.get_skill("editor-journey")
        assert saved["content"] != original["content"]
        metadata = yaml.safe_load(saved["content"].split("---", 2)[1])
        assert metadata["description"] == "Edited [bold] café"
        assert "Saved body {literal}." in saved["content"]
        assert metadata["allowed_tools"] == ["fs_read", "fs_read", "unknown_tool"]
        shell = screen.query_one("#library-skills-reader-shell")
        if shell.effective_layout.items_open:
            await _activate(screen, host, pilot, "#library-skills-items-grip", "<---")
            assert not screen._skills_state.reader_preferences.items_open
        # Let the handoff see retained old rows before the loading recompose.
        # Focusing one makes its removal look like user navigation, cancelling
        # the pending handoff before the ready rows arrive.
        attempted = asyncio.Event()
        loading_done = threading.Event()
        recomposed = False
        recompose = LibrarySkillsListCanvas.recompose
        focus_entry = LibraryScreen._focus_library_list_entry
        list_skills = app.skills_scope_service.list_skills

        async def loading_recompose(canvas):
            nonlocal recomposed
            if canvas.mode == "list" and not recomposed:
                recomposed = True
                await asyncio.wait_for(attempted.wait(), 3)
                await recompose(canvas)
                canvas.screen.call_after_refresh(loading_done.set)
                return
            await recompose(canvas)

        def focus_during_loading(current):
            focus_entry(current)
            attempted.set()

        async def list_after_loading(*args, **kwargs):
            assert await asyncio.to_thread(loading_done.wait, 3)
            return await list_skills(*args, **kwargs)

        with monkeypatch.context() as back_patch:
            back_patch.setattr(LibrarySkillsListCanvas, "recompose", loading_recompose)
            back_patch.setattr(
                LibraryScreen, "_focus_library_list_entry", focus_during_loading
            )
            back_patch.setattr(
                app.skills_scope_service, "list_skills", list_after_loading
            )
            try:
                await _activate(
                    screen, host, pilot, "#library-skill-back", "Back to list"
                )
                await _focus(
                    screen,
                    host,
                    pilot,
                    "#library-skill-row-editor-journey",
                    "editor-journey",
                )
                assert recomposed and attempted.is_set()
            finally:
                loading_done.set()
        await pilot.press("enter")
        await _wait_for_selector(screen, pilot, "#library-skill-mode-edit")
        await _activate(screen, host, pilot, "#library-skill-mode-edit", "Edit")
        description = screen.query_one("#library-skill-description", Input)
        description.value = "Discard this draft"
        await pilot.pause()
        await pilot.press("escape")
        assert screen._skills_state.dirty
        assert description.value == "Discard this draft"
        await _activate(
            screen, host, pilot, "#library-skill-discard", "Discard changes"
        )
        await _focus(
            screen, host, pilot, "#library-skill-row-editor-journey", "editor-journey"
        )
        assert (await local.get_skill("editor-journey"))["content"] == saved["content"]

        await screen._select_library_rail_row("create-skill")
        name = await _wait_for_selector(screen, pilot, "#library-skill-name")
        name.value = "help"
        screen.query_one("#library-skill-body", TextArea).text = "Never saved."
        await pilot.pause()
        assert "shadows" in str(
            screen.query_one("#library-skill-warnings", Static).renderable
        )
        listed, release_list = threading.Event(), threading.Event()
        list_skills = app.skills_scope_service.list_skills

        async def held_list(*args, **kwargs):
            listed.set()
            await asyncio.to_thread(release_list.wait, 10)
            return await list_skills(*args, **kwargs)

        monkeypatch.setattr(app.skills_scope_service, "list_skills", held_list)
        try:
            await _activate(screen, host, pilot, "#library-skill-cancel", "Cancel")
            await _wait_for_condition(
                pilot, listed.is_set, message="Cancel did not reload Skills"
            )
            await pilot.pause()
        finally:
            release_list.set()
        await _focus(
            screen, host, pilot, "#library-skill-row-editor-journey", "editor-journey"
        )
        assert screen._library_selected_row_id == "browse-skills"
        assert (
            screen.query_one("#library-rail").shell.selected_row_id == "browse-skills"
        )
        assert not (tmp_path / "help").exists()


@pytest.mark.asyncio
async def test_skill_save_preserves_edits_made_while_write_is_running(
    tmp_path, monkeypatch
):
    app = _build_test_app()
    local = _wire_skills(app, tmp_path)
    await local.create_skill(
        name="write-race", content="---\nname: write-race\n---\nOriginal."
    )
    host = LibraryProductionCSSHarness(app)
    entered, release = threading.Event(), threading.Event()
    update = local.update_skill

    async def held_update(*args, **kwargs):
        entered.set()
        await asyncio.to_thread(release.wait, 10)
        return await update(*args, **kwargs)

    monkeypatch.setattr(local, "update_skill", held_update)
    async with host.run_test(size=(170, 48)) as pilot:
        screen = await _open_real_skill_editor(host, pilot, "write-race")
        description = screen.query_one("#library-skill-description", Input)
        description.value = "Submitted draft"
        await pilot.pause()
        await _activate(screen, host, pilot, "#library-skill-save", "Save changes")
        await _wait_for_condition(pilot, entered.is_set, message="Save not admitted")
        try:
            description.focus()
            await _focus(
                screen, host, pilot, "#library-skill-description", "Submitted draft"
            )
            await pilot.press("end")
            await pilot.press("!", "!")
            assert description.value == "Submitted draft!!"
        finally:
            release.set()
        await _wait_for_condition(
            pilot,
            lambda: not screen._skills_state.mutation_in_flight,
            message="Save did not finish",
        )
        assert screen.focused is description
        assert screen._skills_state.dirty
        assert description.value == "Submitted draft!!"
        saved = await local.get_skill("write-race")
        assert (
            yaml.safe_load(saved["content"].split("---", 2)[1])["description"]
            == "Submitted draft"
        )
        await pilot.press("escape")
        assert screen._skills_state.view == "editor"
        assert description.value == "Submitted draft!!"
        await _activate(screen, host, pilot, "#library-skill-save", "Save changes")
        await _wait_for_condition(
            pilot,
            lambda: (
                not screen._skills_state.dirty
                and not screen._skills_state.mutation_in_flight
            ),
            message="Second save did not settle",
        )
        saved = await local.get_skill("write-race")
        assert (
            yaml.safe_load(saved["content"].split("---", 2)[1])["description"]
            == "Submitted draft!!"
        )
        assert saved["version"] == 3
