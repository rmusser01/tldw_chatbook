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

import asyncio
import time
from pathlib import Path

import pytest
from textual.widgets import Button, Static

from Tests.private_profile import private_profile_test
from Tests.UI.app_factory import _build_test_app, persist_seeded_config
from tldw_chatbook.Chat.console_chat_controller import (
    list_project_instruction_bindings,
)
from tldw_chatbook.Chat.console_project_instructions import (
    ProjectInstructionControlState,
)
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


async def _until(pilot, predicate, what: str, *, timeout: float = 30.0) -> None:
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


_CHOOSE_PIN = "Choose folder · Project"


async def _open_inspector_from_pin(app, pilot, folder: Path, *, off: bool = False):
    """Boot to Console, start a session in the bound workspace, click the pin.

    ``off`` first turns project instructions off (the legacy-disabled state
    whose recovery action is 'Enable')."""
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
    if off:
        controller.store.set_session_project_instruction_state(
            session.id, ProjectInstructionControlState.legacy_disabled()
        )
    console._set_console_rail_preference(right_open=True)
    await _until(
        pilot, lambda: bool(console.query(_PIN)), "the Project Instructions pin"
    )
    await _until(
        pilot,
        lambda: _pin_label(console) == ("Off · Project" if off else _CHOOSE_PIN),
        "the pin to show the starting state",
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
    action = "enable" if off else "choose"
    await _until(
        pilot,
        lambda: bool(inspector.query(f"#console-project-instruction-{action}")),
        f"the '{action}' recovery button",
    )
    return console, controller, session, inspector


async def _press_choose_folder(
    app, pilot, inspector, action: str = "choose"
) -> ProjectInstructionSetupModal:
    selector = f"#console-project-instruction-{action}"
    button = inspector.query_one(selector, Button)
    button.scroll_visible(animate=False, immediate=True, top=True)
    await pilot.pause()
    await pilot.click(selector)
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
@pytest.mark.parametrize("action", ["choose", "enable"])
async def test_choose_folder_opens_picker_and_applies_the_chosen_binding(
    request, tmp_path, action
):
    """AC#1/#5: the real button, the real picker, the binding really applied.

    'Enable' (from Off) opens the same picker through the same seam."""
    folder = tmp_path / "project"
    folder.mkdir()
    folder = folder.resolve(strict=True)
    app = _ready_console_app(folder)
    async with app.run_test(size=(160, 45)) as pilot:
        console, controller, session, inspector = await _open_inspector_from_pin(
            app, pilot, folder, off=action == "enable"
        )
        before = "State: Off" if action == "enable" else "State: Choose folder"
        assert before in _panel_text(inspector)

        picker = await _press_choose_folder(app, pilot, inspector, action)
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

        def shows_binding() -> bool:
            # Positive, not just "the old state is gone": the panel is briefly
            # empty while it re-renders the new state.
            text = _panel_text(inspector)
            return before not in text and _BINDING_LABEL in text

        await _until(
            pilot,
            lambda: app.screen is inspector and shows_binding(),
            "the Inspector to show the applied binding",
        )
        assert session.project_instruction_state.project_instructions_enabled
        # The Inspector is live, not a dead pump left on the stack: a key it
        # binds still reaches it and closes it.
        assert inspector.is_running
        await pilot.press("escape")
        await _until(pilot, lambda: app.screen is console, "the Inspector to close")
        await _until(
            pilot,
            lambda: _pin_label(console) not in {_CHOOSE_PIN, "Off · Project"},
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
        # The dismissed picker must actually stop. Textual removes a popped
        # screen in a later call_next'd _replace_screen, so poll for it.
        await _until(pilot, lambda: not picker.is_running, "the picker to close")
        # ...and Esc still closes the Inspector back to the Console.
        await pilot.press("escape")
        await _until(pilot, lambda: app.screen is console, "the Inspector to close")
        assert _pin_label(console) == _CHOOSE_PIN


@private_profile_test
@pytest.mark.asyncio
async def test_inspector_names_a_saved_folder_before_its_preview_is_ready(
    request, tmp_path, monkeypatch
):
    """AC#1 live re-verification: a chat reopened after a restart showed
    'Binding: <raw binding id> · Locator: not checked' for as long as the
    next-send preview took (seconds, live), because the Inspector resolved
    the folder only AFTER that preview. It must name the folder at once.

    The chat's state is what a restart restores: a chosen folder, never
    resolved in this run. The real preview is held behind a gate so "before
    the preview is ready" is deterministic; the folder lookup is not stubbed.
    """
    folder = tmp_path / "project"
    folder.mkdir()
    folder = folder.resolve(strict=True)
    app = _ready_console_app(folder)
    async with app.run_test(size=(160, 45)) as pilot:
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
        session = controller.store.ensure_session(title="Saved folder")
        (selection,) = list_project_instruction_bindings(
            session, app.workspace_registry_service
        )
        controller.store.set_session_project_instruction_state(
            session.id,
            ProjectInstructionControlState(
                project_instructions_enabled=True,
                working_folder_binding_id=_BINDING_ID,
                working_folder_locator_fingerprint=selection.locator_fingerprint,
                project_instruction_notice_key=None,
            ),
        )
        preview_gate = asyncio.Event()
        build_factories = console._context_spend._console_inspector_next_send_factories

        def held_preview(chat_controller, session_id):
            factory, *rest = build_factories(chat_controller, session_id)

            async def after_gate():
                await preview_gate.wait()
                return await factory()

            return (after_gate, *rest)

        monkeypatch.setattr(
            console._context_spend,
            "_console_inspector_next_send_factories",
            held_preview,
        )
        console._set_console_rail_preference(right_open=True)
        await _until(
            pilot, lambda: bool(console.query(_PIN)), "the Project Instructions pin"
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
        named = f"Binding: {_BINDING_LABEL} · Locator: match"
        await _until(
            pilot,
            lambda: named in _panel_text(inspector),
            "the folder's name while the preview is still preparing",
            timeout=10.0,
        )
        assert not inspector._snapshot_ready
        assert _BINDING_ID not in _panel_text(inspector)

        preview_gate.set()
        await _until(pilot, lambda: inspector._snapshot_ready, "the preview")
        assert named in _panel_text(inspector)
        assert inspector.is_running
