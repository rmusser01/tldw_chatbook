"""Conversation Inspector 'Choose folder' drives a real folder picker (TASK-33621.13).

GAP4-01 (Console UX review 2026-09-29): pressing 'Choose folder' (or
'Enable') in the Inspector's Project Instructions panel froze the whole app.
The Inspector's ``RecoveryRequested`` handler was a plain ``@on`` coroutine
that awaited ``ConsoleSessionController._select_project_instruction_binding``,
which awaits ``app.push_screen_wait``. Textual appends the picker to the stack
and only THEN raises ``NoActiveWorker`` outside a worker, so the picker was
painted while the Inspector's own message loop died under it.

These tests drive the REAL production app (``TldwCli`` from the shared
factory), the real Console ``ChatScreen``, the real Inspector opened from the
real rail pin, the real ``ProjectInstructionSetupModal`` pushed through the
real ``push_screen_wait`` seam, and a real ``LocalWorkspaceRegistryService``
holding a real folder binding. Nothing on the push-and-await path is stubbed:
under ``run_test`` the keep-alive is off, so on the pre-fix code the
``NoActiveWorker`` propagates out of ``run_test`` and fails the test.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from textual.widgets import Button, Static

from Tests.private_profile import private_profile_test
from Tests.UI.app_factory import _build_test_app, persist_seeded_config
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.Widgets.Console.console_conversation_inspector import (
    ConsoleConversationInspector,
)
from tldw_chatbook.Widgets.Console.console_project_instructions import (
    ConsoleProjectInstructionContextPanel,
    ProjectInstructionSetupModal,
)
from tldw_chatbook.Workspaces.models import WorkspaceRuntimeBinding

_WORKSPACE_ID = "u3-project-workspace"
_BINDING_ID = "u3-project-folder"
_BINDING_LABEL = "Scratch project"
_PIN = "#console-project-instruction-status-button"


def _ready_console_app(folder: Path):
    """The production app, send-ready, active in a named workspace with one folder."""
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "llama_cpp", "model": "local-model"}
    app.app_config["api_settings"] = {
        "llama_cpp": {"api_url": "http://127.0.0.1:9099", "model": "local-model"}
    }
    persist_seeded_config(app, "chat_defaults", "api_settings.llama_cpp")
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "local-model"
    registry = app.workspace_registry_service
    registry.create_workspace(workspace_id=_WORKSPACE_ID, name="Project work")
    registry.save_runtime_binding(
        WorkspaceRuntimeBinding(
            workspace_id=_WORKSPACE_ID,
            binding_id=_BINDING_ID,
            binding_kind="local-filesystem",
            label=_BINDING_LABEL,
            locator=str(folder),
            status="ready",
            metadata={"access": "ro"},
        )
    )
    registry.set_active_workspace(_WORKSPACE_ID)
    return app


async def _until(pilot, predicate, what: str, *, timeout: float = 8.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await pilot.pause(0.02)
    if predicate():
        return
    pytest.fail(f"Timed out waiting for {what}")


def _pin_label(screen: ChatScreen) -> str:
    return str(screen.query_one(_PIN, Button).label)


def _panel_text(inspector: ConsoleConversationInspector) -> str:
    panel = inspector.query_one(
        "#console-context-project-instructions", ConsoleProjectInstructionContextPanel
    )
    return " ".join(str(item.renderable) for item in panel.query(Static))


async def _open_inspector_from_pin(app, pilot, folder: Path):
    """Boot to Console, start a session in the bound workspace, click the pin."""
    await _until(
        pilot,
        lambda: isinstance(app.screen, ChatScreen) and app.screen.region.width > 0,
        "the production Console screen",
    )
    console = app.screen
    await _until(
        pilot,
        lambda: bool(console.query("#console-native-composer")),
        "the Console composer",
    )
    controller = console._ensure_console_chat_controller()
    session = controller.store.ensure_session(title="Folder picker")
    assert session.workspace_id == _WORKSPACE_ID
    assert session.project_instruction_state.working_folder_binding_id is None
    console._set_console_rail_preference(right_open=True)
    await _until(
        pilot, lambda: bool(console.query(_PIN)), "the Project Instructions pin"
    )
    await _until(
        pilot,
        lambda: _pin_label(console) == "Choose folder · Project",
        "the pin to read 'Choose folder · Project'",
    )
    console.query_one(_PIN, Button).scroll_visible(animate=False, immediate=True)
    await pilot.pause()
    await pilot.click(_PIN)
    await _until(
        pilot,
        lambda: isinstance(app.screen, ConsoleConversationInspector),
        "the Conversation Inspector",
    )
    inspector = app.screen
    await _until(
        pilot,
        lambda: bool(inspector.query("#console-project-instruction-choose")),
        "the 'Choose folder' button",
    )
    return console, controller, session, inspector


async def _press_choose_folder(app, pilot, inspector) -> ProjectInstructionSetupModal:
    choose = inspector.query_one("#console-project-instruction-choose", Button)
    choose.scroll_visible(animate=False, immediate=True, top=True)
    await pilot.pause()
    await pilot.click("#console-project-instruction-choose")
    await _until(
        pilot,
        lambda: isinstance(app.screen, ProjectInstructionSetupModal),
        "the folder picker",
    )
    picker = app.screen

    def laid_out() -> bool:
        rows = picker.query("#console-project-binding-0")
        return bool(rows) and rows.first().region.width > 0 and app.focused is not None

    await _until(pilot, laid_out, "the folder picker's first row to be laid out")
    return picker


@private_profile_test
@pytest.mark.asyncio
async def test_choose_folder_opens_picker_and_applies_the_chosen_binding(
    request, tmp_path
):
    """AC#1/#5: the real button, the real picker, the binding really applied."""
    folder = tmp_path / "project"
    folder.mkdir()
    folder = folder.resolve(strict=True)
    app = _ready_console_app(folder)
    async with app.run_test(size=(160, 45)) as pilot:
        console, controller, session, inspector = await _open_inspector_from_pin(
            app, pilot, folder
        )
        assert "State: Choose folder" in _panel_text(inspector)

        picker = await _press_choose_folder(app, pilot, inspector)
        row = picker.query_one("#console-project-binding-0", Button)
        assert _BINDING_LABEL in str(row.label)
        assert not row.disabled
        await pilot.click("#console-project-binding-0")

        await _until(
            pilot,
            lambda: (
                session.project_instruction_state.working_folder_binding_id
                == _BINDING_ID
            ),
            "the chosen folder to be bound to the session",
        )
        await _until(
            pilot,
            lambda: (
                app.screen is inspector
                and "State: Choose folder" not in _panel_text(inspector)
            ),
            "the Inspector to show the applied binding",
        )
        assert session.project_instruction_state.project_instructions_enabled
        assert _BINDING_LABEL in _panel_text(inspector)
        # The Inspector is live, not a dead pump left on the stack: a key it
        # binds still reaches it and closes it.
        assert inspector.is_running
        await pilot.press("escape")
        await _until(pilot, lambda: app.screen is console, "the Inspector to close")
        await _until(
            pilot,
            lambda: _pin_label(console) != "Choose folder · Project",
            "the Inspect pin to stop reading 'Choose folder · Project'",
        )


@private_profile_test
@pytest.mark.asyncio
@pytest.mark.parametrize("how", ["escape", "cancel-button"])
async def test_cancelling_the_picker_returns_to_a_responsive_inspector(
    request, tmp_path, how
):
    """AC#2: Esc or Cancel leaves the binding unchanged and the Inspector live."""
    folder = tmp_path / "project"
    folder.mkdir()
    folder = folder.resolve(strict=True)
    app = _ready_console_app(folder)
    async with app.run_test(size=(160, 45)) as pilot:
        console, controller, session, inspector = await _open_inspector_from_pin(
            app, pilot, folder
        )
        before = session.project_instruction_state
        await _press_choose_folder(app, pilot, inspector)
        if how == "escape":
            await pilot.press("escape")
        else:
            await pilot.click("#console-project-setup-cancel")
        await _until(pilot, lambda: app.screen is inspector, "the Inspector")
        await pilot.pause(0.1)
        assert session.project_instruction_state == before
        assert "State: Choose folder" in _panel_text(inspector)
        assert inspector.is_running
        # Responsive: the same button opens the picker again...
        picker = await _press_choose_folder(app, pilot, inspector)
        await pilot.press("escape")
        await _until(pilot, lambda: app.screen is inspector, "the Inspector again")
        assert not picker.is_running or app.screen is not picker
        # ...and Esc still closes the Inspector back to the Console.
        await pilot.press("escape")
        await _until(pilot, lambda: app.screen is console, "the Inspector to close")
        assert _pin_label(console) == "Choose folder · Project"
