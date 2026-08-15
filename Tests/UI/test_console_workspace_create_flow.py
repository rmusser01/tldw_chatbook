"""Console new-workspace creation flow (setup modal -> create+bind+activate).

The controller's ``_confirm_console_workspace_create`` is exercised on a
bare controller instance (``object.__new__`` -- the repo's established
pattern for testing moved method bodies without wiring 30 constructor
deps) with a REAL ``LocalWorkspaceRegistryService``, covering the happy
path, the None (cancel) path, and the bind-race path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.UI.Console_Modules.workspace import ConsoleWorkspaceController
from tldw_chatbook.Widgets.Console.console_workspace_setup_modal import (
    ConsoleWorkspaceSetupResult,
)
from tldw_chatbook.Workspaces import LocalWorkspaceRegistryService


class _Notifier:
    def __init__(self) -> None:
        self.messages: list[tuple[str, str]] = []

    def notify(self, message: str, severity: str = "information") -> None:
        self.messages.append((severity, message))


class _AppStub:
    def __init__(self, registry: LocalWorkspaceRegistryService) -> None:
        self.workspace_registry_service = registry
        self.notifier = _Notifier()

    def notify(self, message: str, severity: str = "information") -> None:
        self.notifier.notify(message, severity=severity)


class _ControllerHarness:
    """Bare controller + call log for the sync hooks the confirm body uses.

    The three sync hooks are read-only properties on the real class, so a
    plain instance-attribute stub cannot shadow them; a shim subclass
    replaces them with assignable attributes instead.
    """

    def __init__(self, registry: LocalWorkspaceRegistryService) -> None:
        self.calls: list = []
        self.app = _AppStub(registry)

        harness = self

        class _ConfirmShim(ConsoleWorkspaceController):
            _sync_console_chat_core_state = None
            _activate_console_session_for_workspace = None
            _sync_console_workspace_context = None

            def run_worker(self, work, **kwargs):  # noqa: D102
                harness.calls.append("sync-ui")

        self.controller = _ConfirmShim.__new__(_ConfirmShim)
        self.controller.app_instance = self.app
        self.controller._sync_console_chat_core_state = lambda: self.calls.append(
            "sync-core"
        )
        self.controller._activate_console_session_for_workspace = (
            lambda ws_id: self.calls.append(("activate", ws_id))
        )
        self.controller._sync_console_workspace_context = lambda: self.calls.append(
            "sync-context"
        )
        # ``_sync_native_console_chat_ui`` is a property over the stored
        # constructor callable; the confirm body calls it to get the
        # coroutine it hands to ``run_worker``.
        self.controller._sync_native_console_chat_ui_fn = lambda: (lambda: None)


def _registry(tmp_path: Path) -> LocalWorkspaceRegistryService:
    tmp_path.mkdir(parents=True, exist_ok=True)
    registry = LocalWorkspaceRegistryService(
        WorkspaceDB(tmp_path / "workspaces.sqlite", client_id="create-flow-tests")
    )
    registry.ensure_default_workspace()
    return registry


def test_confirm_creates_binds_and_activates(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    harness = _ControllerHarness(registry)
    folder = tmp_path / "proj"
    folder.mkdir()

    harness.controller._confirm_console_workspace_create(
        ConsoleWorkspaceSetupResult(
            name="My Project", folder_path=str(folder), allow_write=True
        )
    )

    workspace = registry.get_workspace("workspace-local-1")
    assert workspace is not None and workspace.name == "My Project"
    bindings = registry.list_folder_bindings("workspace-local-1")
    assert len(bindings) == 1
    assert bindings[0].locator == str(folder.resolve())
    assert bindings[0].metadata["access"] == "rw"
    assert registry.get_active_workspace().workspace_id == "workspace-local-1"
    assert ("activate", "workspace-local-1") in harness.calls
    assert not harness.app.notifier.messages or all(
        sev == "information" for sev, _ in harness.app.notifier.messages
    )


def test_confirm_none_result_creates_nothing(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    harness = _ControllerHarness(registry)

    harness.controller._confirm_console_workspace_create(None)

    assert registry.list_workspaces(include_archived=True) == (
        registry.list_workspaces(include_archived=True)
    )
    # Only the ensured default workspace exists; nothing was created,
    # bound, activated, or synced.
    ids = {ws.workspace_id for ws in registry.list_workspaces()}
    assert ids == {"workspace-default"}
    assert harness.calls == []


def test_confirm_bind_race_keeps_workspace_and_warns(
    tmp_path: Path, monkeypatch: Any
) -> None:
    registry = _registry(tmp_path)
    harness = _ControllerHarness(registry)
    folder = tmp_path / "proj"
    folder.mkdir()

    def _failing_add(workspace_id: str, path: str, *, allow_write: bool = False):
        raise RuntimeError("race: folder vanished between validation and write")

    monkeypatch.setattr(registry, "add_folder_binding", _failing_add)

    harness.controller._confirm_console_workspace_create(
        ConsoleWorkspaceSetupResult(
            name="Raced", folder_path=str(folder), allow_write=False
        )
    )

    workspace = registry.get_workspace("workspace-local-1")
    assert workspace is not None and workspace.name == "Raced"
    warnings = [m for sev, m in harness.app.notifier.messages if sev == "warning"]
    assert any("folder binding failed" in m for m in warnings)
    # The workspace is still activated; no re-prompt, no orphaned modal state.
    assert registry.get_active_workspace().workspace_id == "workspace-local-1"
