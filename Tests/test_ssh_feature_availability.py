"""SSH feature availability check (Phase 5b, Task 21).

Spec: ``Docs/superpowers/specs/2026-09-24-ssh-remote-workspace-bindings-design.md``
"User-facing surfaces" — when no ``ssh`` binary is on PATH the SSH-binding
feature surfaces as DISABLED with one clear message at the user entry
points, never as a crash. Two entry points are pinned here:

- The Settings → Workspaces SSH add-form submit refuses (the message
  lands in the form's result line) and adds nothing.
- The Console project-instruction picker marks ssh-filesystem options
  ineligible with the same message as the recovery row; local folder
  options are untouched.

The present-path (``ssh`` found) for both entry points is already covered
by ``Tests/UI/test_settings_ssh_bindings.py`` — these tests only pin the
absent-binary direction plus the helper itself.
"""

from __future__ import annotations

import shutil
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from textual.widgets import Button, Input

from Tests.private_profile import private_profile_test
from Tests.UI.test_settings_configuration_hub import (
    DestinationHarness,
    _active_destination_screen,
    _build_test_app,
    _open_settings_category,
    _visible_text,
)
from tldw_chatbook.Tools.remote_workspace_transport import (
    SSH_UNAVAILABLE_MESSAGE,
    ssh_available,
)

#: The one refusal message every entry point must surface verbatim
#: (ruling: "OpenSSH client (ssh) not found — SSH workspace bindings
#: unavailable").
EXPECTED_MESSAGE = "OpenSSH client (ssh) not found — SSH workspace bindings unavailable"


# --- the helper ----------------------------------------------------------------


def test_message_constant_is_the_ruling_text() -> None:
    """The shared constant carries the exact ruling message."""
    assert SSH_UNAVAILABLE_MESSAGE == EXPECTED_MESSAGE


def test_ssh_available_false_when_which_finds_nothing(monkeypatch) -> None:
    """``shutil.which("ssh")`` -> None means the feature is off."""
    monkeypatch.setattr(shutil, "which", lambda name: None)
    assert ssh_available() is False


def test_ssh_available_true_when_which_finds_a_binary(monkeypatch) -> None:
    """A resolved ssh path means the feature is on."""
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/ssh")
    assert ssh_available() is True


# --- Console picker entry point ------------------------------------------------


@pytest.mark.asyncio
async def test_console_picker_refuses_ssh_options_without_binary(
    monkeypatch,
) -> None:
    """No ssh binary: ssh-filesystem picker options are ineligible with the
    message; local folder options stay eligible and chip-free."""
    from tldw_chatbook.UI.Console_Modules.session import ConsoleSessionController

    monkeypatch.setattr(shutil, "which", lambda name: None)
    controller = ConsoleSessionController.__new__(ConsoleSessionController)
    controller._current_chat_store_accessor = lambda: SimpleNamespace(
        sessions=lambda: [SimpleNamespace(id="session-a")]
    )

    async def capture(modal):
        captured.append(modal)
        return SimpleNamespace(action="cancel", binding_id=None)

    captured: list[object] = []
    controller._screen = SimpleNamespace(
        app=SimpleNamespace(push_screen_wait=AsyncMock())
    )
    controller._screen.app.push_screen_wait.side_effect = capture

    selections = (
        SimpleNamespace(
            binding=SimpleNamespace(
                binding_id="folder-b1",
                display_name="",
                label="project",
                binding_kind="local-filesystem",
            )
        ),
        SimpleNamespace(
            binding=SimpleNamespace(
                binding_id="ssh-b1",
                display_name="",
                label="/srv/app",
                binding_kind="ssh-filesystem",
            )
        ),
    )
    await controller._select_project_instruction_binding(
        "session-a", selections, ""
    )

    (modal,) = captured
    folder_option, ssh_option = modal._options
    # Local folders are unaffected: still selectable, no refusal row.
    assert folder_option.binding_id == "folder-b1"
    assert folder_option.eligible is True
    assert folder_option.recovery == ""
    # SSH options refuse with the exact message, not a crash, and carry
    # no live-status chip (the cache cannot be truthful without ssh).
    assert ssh_option.binding_id == "ssh-b1"
    assert ssh_option.eligible is False
    assert ssh_option.recovery == EXPECTED_MESSAGE
    assert "ssh:" not in ssh_option.label


# --- Settings add-form entry point ---------------------------------------------


@pytest.mark.timeout(240)
@private_profile_test
@pytest.mark.asyncio
async def test_settings_add_form_refuses_without_ssh_binary(
    request, tmp_path, monkeypatch
) -> None:
    """No ssh binary: the add-form submit shows the message inline, keeps
    the draft, and persists no binding."""
    app = _build_test_app()
    registry = app.workspace_registry_service
    registry.create_workspace(workspace_id="ws-ssh-off", name="SSH Off")
    host = DestinationHarness(app, "settings")

    # The whole machine appears ssh-less from the entry point's viewpoint.
    monkeypatch.setattr(shutil, "which", lambda name: None)

    async with host.run_test(size=(170, 48)) as pilot:
        screen = _active_destination_screen(host)
        await _open_settings_category(pilot, "#settings-category-workspaces")
        screen.query_one("#settings-workspace-row-ws-ssh-off", Button).press()
        await pilot.pause(0.3)

        screen.query_one("#settings-workspace-ssh-target", Input).value = "devbox"
        screen.query_one("#settings-workspace-ssh-path", Input).value = "/srv/app"
        screen.query_one("#settings-workspace-ssh-add", Button).press()
        await pilot.pause(0.3)

        # Refusal is a message, not a crash: the exact ruling text lands
        # next to the form, nothing is added, and the draft survives.
        assert EXPECTED_MESSAGE in _visible_text(screen)
        assert registry.list_ssh_bindings("ws-ssh-off") == ()
        assert (
            screen.query_one("#settings-workspace-ssh-path", Input).value == "/srv/app"
        )
