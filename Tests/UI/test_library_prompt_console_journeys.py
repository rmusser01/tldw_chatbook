"""Production-CSS keyboard journeys from Library to guarded Prompt applications."""

from __future__ import annotations

import pytest
from textual.widgets import Button, Input

from Tests.UI.test_library_prompt_collection_journeys import _focus, _tab_to
from Tests.UI.test_library_prompts_canvas import (
    _build_test_app,
    _library_prompt_target,
    _open_prompt_editor,
    _real_prompt_scope_service,
    _wire_empty_non_prompt_services,
)
from Tests.UI.test_library_shell import (
    LibraryProductionCSSHarness,
    _active_library_screen,
    _wait_for_condition,
    _wait_for_library_shell,
)
from Tests.UI.test_prompt_variables_dialog import _input_for
from tldw_chatbook.Widgets.Console.prompt_variables_dialog import (
    PromptVariablesDialog,
)


def _prompt_host(tmp_path, theme, *, system="", user="Hello {{name}}."):
    db, service = _real_prompt_scope_service(tmp_path)
    prompt_id, _, _ = db.add_prompt(
        name="Console handoff [bold] café",
        author="",
        details="",
        system_prompt=system,
        user_prompt=user,
    )
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryProductionCSSHarness(app)
    host.theme = theme
    applications = []
    app.console_prompt_target_projection = _library_prompt_target
    app.stage_console_prompt_insert = applications.append
    app.push_screen = host.push_screen
    app.notify = host.notify
    return db, prompt_id, host, applications


async def _open_application(screen, host, pilot):
    opener = screen.query_one("#library-prompt-insert-console", Button)
    opener.focus()
    await _focus(
        screen, host, pilot, "#library-prompt-insert-console", "Use in Console"
    )
    await _wait_for_condition(
        pilot,
        lambda: not opener.has_class("-active"),
        message="Prior press still active",
    )
    await pilot.press("enter")


async def _dialog(host, pilot):
    await _wait_for_condition(
        pilot,
        lambda: (
            isinstance(host.screen, PromptVariablesDialog)
            and bool(host.screen.query("#prompt-variables-cancel"))
        ),
        message="Prompt variables dialog did not mount",
    )
    return host.screen


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(170, 48), (80, 24)])
@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
async def test_prompt_variables_cancel_authorize_apply_and_original_keyboard_journey(
    tmp_path, size, theme
):
    system = "Use {tone} for {name}."
    user = "Hello {name}; literal {{name}}."
    db, prompt_id, host, applications = _prompt_host(
        tmp_path, theme, system=system, user=user
    )
    original = db.fetch_prompt_details(prompt_id)
    async with host.run_test(size=size) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompt_editor(screen, pilot, prompt_id)
        fields = tuple(screen.query(Input))
        await _open_application(screen, host, pilot)
        dialog = await _dialog(host, pilot)
        _input_for(dialog, "name").value = "discard me"
        await pilot.press("escape")
        await _wait_for_condition(
            pilot, lambda: host.screen is screen, message="Cancel"
        )
        await _focus(
            screen, host, pilot, "#library-prompt-insert-console", "Use in Console"
        )
        assert applications == []

        await _open_application(screen, host, pilot)
        dialog = await _dialog(host, pilot)
        name_input = _input_for(dialog, "name")
        assert name_input.value == ""
        name_input.focus()
        name_input.value = "Zoë {other} [bold]"
        checkbox = dialog.query_one("#prompt-variables-apply-system")
        assert not checkbox.value
        checkbox.focus()
        await pilot.press("space")
        await _wait_for_condition(
            pilot, lambda: len(dialog._plan.variables) == 2, message="System variables"
        )
        assert _input_for(dialog, "name").value == "Zoë {other} [bold]"
        _input_for(dialog, "tone").value = "plain"
        checkbox.focus()
        await pilot.press("space")
        await _wait_for_condition(
            pilot, lambda: len(dialog._plan.variables) == 1, message="User lane only"
        )
        checkbox.focus()
        await pilot.press("space")
        await _wait_for_condition(
            pilot, lambda: len(dialog._plan.variables) == 2, message="System restored"
        )
        assert _input_for(dialog, "tone").value == "plain"
        await _tab_to(dialog, host, pilot, "#prompt-variables-apply", "Apply")
        await pilot.press("enter")
        await _wait_for_condition(
            pilot, lambda: len(applications) == 1, message="Application not staged"
        )
        application = applications[0]
        assert application.destination == "append_active"
        assert application.system_text == "Use plain for Zoë {other} [bold]."
        assert application.user_text == "Hello Zoë {other} [bold]; literal {name}."
        assert application.apply_system and application.apply_user

        await _open_application(screen, host, pilot)
        dialog = await _dialog(host, pilot)
        assert not dialog.query_one("#prompt-variables-apply-system").value
        await _tab_to(
            dialog,
            host,
            pilot,
            "#prompt-variables-original",
            "Use original placeholders",
        )
        await pilot.press("enter")
        await _wait_for_condition(
            pilot, lambda: len(applications) == 2, message="Original source not staged"
        )
        assert applications[1].user_text == user
        assert applications[1].system_text is None
        assert not applications[1].apply_system
        assert all(field.is_attached for field in fields)
        assert db.fetch_prompt_details(prompt_id) == original


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "source,expected",
    [
        ("Plain [bold] café", "Plain [bold] café"),
        ("Literal {{name}} and }}", "Literal {name} and }"),
        ('{{"key": "value"}}', '{"key": "value"}'),
    ],
)
async def test_direct_library_prompt_decodes_escapes_without_opening_dialog(
    tmp_path, source, expected
):
    db, prompt_id, host, applications = _prompt_host(
        tmp_path, "textual-dark", user=source
    )
    original = db.fetch_prompt_details(prompt_id)
    async with host.run_test(size=(170, 48)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompt_editor(screen, pilot, prompt_id)
        await _open_application(screen, host, pilot)
        await _wait_for_condition(
            pilot, lambda: len(applications) == 1, message="Direct insert"
        )
        assert host.screen is screen
        assert applications[0].user_text == expected
        assert applications[0].destination == "append_active"
        assert not applications[0].apply_system
        assert db.fetch_prompt_details(prompt_id) == original
