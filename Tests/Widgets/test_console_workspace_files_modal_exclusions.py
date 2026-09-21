"""Workspace Files modal exclusion surface: badge + exclude/unexclude action.

Service-level tests drive the real controller and registry (Task 1 CRUD);
widget-level tests mount the real modal over a fake service to pin the
``[excluded]`` marker and the toggle action's arguments.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from textual.app import App
from textual.widgets import Button

from Tests.UI.test_console_workspace_controller import _workspace_controller
from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.Workspaces.file_inspector import (
    BindingScope,
    DirectoryEntry,
    DirectoryPage,
    DirectoryStatus,
)
from tldw_chatbook.Workspaces.registry_service import LocalWorkspaceRegistryService
from tldw_chatbook.Widgets.Console.console_workspace_files_modal import (
    ConsoleWorkspaceFilesModal,
    WorkspaceFilesBinding,
)


@pytest.fixture()
def registry(tmp_path: Path) -> LocalWorkspaceRegistryService:
    db = WorkspaceDB(tmp_path / "workspaces.db")
    try:
        service = LocalWorkspaceRegistryService(db)
        service.create_workspace(workspace_id="ws-files", name="Files WS")
        yield service
    finally:
        db.close()


@pytest.fixture()
def binding_id(registry: LocalWorkspaceRegistryService, tmp_path: Path) -> str:
    root = tmp_path / "repo"
    (root / "secrets").mkdir(parents=True)
    binding = registry.add_folder_binding("ws-files", root, allow_write=True)
    return binding.binding_id


def _files_controller(
    registry: LocalWorkspaceRegistryService,
) -> Any:
    return _workspace_controller(
        app_instance=SimpleNamespace(workspace_registry_service=registry)
    )


# -- Service level: controller over the real registry ----------------------


def test_controller_set_exclusion_round_trips_through_registry(
    registry: LocalWorkspaceRegistryService, binding_id: str
) -> None:
    controller = _files_controller(registry)

    controller.set_exclusion(binding_id, "secrets", True)
    assert [
        entry.path for entry in registry.list_binding_exclusions(binding_id)
    ] == ["secrets"]

    controller.set_exclusion(binding_id, "secrets", False)
    assert registry.list_binding_exclusions(binding_id) == ()


def test_controller_set_exclusion_on_unknown_binding_is_rejected(
    registry: LocalWorkspaceRegistryService,
) -> None:
    from tldw_chatbook.Workspaces.registry_service import WorkspaceRegistryServiceError

    controller = _files_controller(registry)
    with pytest.raises(WorkspaceRegistryServiceError):
        controller.set_exclusion("missing-binding", "secrets", True)


def test_resolve_workspace_files_visit_exposes_binding_exclusions(
    registry: LocalWorkspaceRegistryService, binding_id: str
) -> None:
    registry.add_binding_exclusion("ws-files", binding_id, "secrets")
    controller = _files_controller(registry)

    resolution = controller._resolve_workspace_files_visit("ws-files")

    assert resolution is not None
    assert [binding.binding_id for binding in resolution.bindings] == [binding_id]
    assert resolution.bindings[0].exclusions == ("secrets",)


# -- Widget level: real modal over a fake service --------------------------


def test_is_excluded_folds_case_componentwise() -> None:
    """A differently-cased exclusion must still badge on case-insensitive FS.

    Mirrors the denylist's ``_compare_key`` discipline (Finding 3): tools
    refuse a case-variant excluded path, so the modal's badge must agree --
    ``Docs`` badges on-disk ``docs/...`` while ``docsfoo`` stays unbadged.
    """
    assert ConsoleWorkspaceFilesModal._is_excluded("docs/file.txt", ("Docs",))
    assert ConsoleWorkspaceFilesModal._is_excluded("Docs", ("docs",))
    assert not ConsoleWorkspaceFilesModal._is_excluded("docsfoo/file.txt", ("docs",))
    assert not ConsoleWorkspaceFilesModal._is_excluded("other/file.txt", ("docs",))


class _FakeFilesService:
    """The modal's whole service boundary, recording exclusion toggles."""

    def __init__(self, page: DirectoryPage) -> None:
        self._page = page
        self.exclusion_calls: list[tuple[str, str, bool]] = []

    def list_directory(
        self,
        scope: Any,
        directory_parts: tuple[str, ...] = (),
        *,
        continuation: Any | None = None,
    ) -> DirectoryPage:
        return self._page

    def filter_paths(
        self,
        scope: Any,
        query: str,
        *,
        is_cancelled: Any | None = None,
        on_progress: Any | None = None,
    ) -> Any:
        raise NotImplementedError

    def read_file(
        self,
        scope: Any,
        raw_parts: tuple[str, ...],
        *,
        page_offset: int | None = None,
        expected_revision: Any | None = None,
    ) -> Any:
        raise NotImplementedError

    def set_exclusion(
        self, binding_id: str, relative_path: str, excluded: bool
    ) -> None:
        self.exclusion_calls.append((binding_id, relative_path, excluded))


def _files_modal(
    service: _FakeFilesService, *, exclusions: tuple[str, ...]
) -> ConsoleWorkspaceFilesModal:
    scope = BindingScope(
        workspace_id="ws-files",
        binding_id="binding-1",
        binding_fingerprint="fingerprint",
        canonical_root="/repo",
        root_device=0,
        root_inode=0,
    )
    return ConsoleWorkspaceFilesModal(
        inspector=service,
        inspected_workspace_id="ws-files",
        inspected_workspace_name="Files WS",
        active_workspace_id="ws-active",
        active_workspace_name="Active WS",
        bindings=(
            WorkspaceFilesBinding(
                binding_id="binding-1",
                label="Repo",
                scope=scope,
                exclusions=exclusions,
            ),
        ),
    )


def _root_page() -> DirectoryPage:
    return DirectoryPage(
        status=DirectoryStatus.COMPLETE,
        entries=(
            DirectoryEntry(("secrets",), "secrets", True),
            DirectoryEntry(("notes.md",), "notes.md", False),
        ),
    )


async def test_excluded_entry_renders_marker_and_plain_entries_do_not() -> None:
    service = _FakeFilesService(_root_page())
    modal = _files_modal(service, exclusions=("secrets",))
    app = App[None]()
    async with app.run_test(size=(120, 40)) as pilot:
        app.push_screen(modal)
        for _ in range(50):
            await pilot.pause()
            if list(modal.query(".console-workspace-files-entry")):
                break
        labels = [button.label.plain for button in modal.query(".console-workspace-files-entry")]
        assert any(
            "secrets" in label and " [excluded]" in label for label in labels
        ), labels
        assert any(
            "notes.md" in label and " [excluded]" not in label for label in labels
        ), labels


async def test_toggle_action_unexcludes_selected_excluded_entry() -> None:
    service = _FakeFilesService(_root_page())
    modal = _files_modal(service, exclusions=("secrets",))
    app = App[None]()
    async with app.run_test(size=(120, 40)) as pilot:
        app.push_screen(modal)
        for _ in range(50):
            await pilot.pause()
            if list(modal.query(".console-workspace-files-entry")):
                break
        modal._state = replace(modal._state, selected_tree_parts=("secrets",))

        await modal.action_toggle_exclusion()
        await pilot.pause()

        assert service.exclusion_calls == [("binding-1", "secrets", False)]
        labels = [
            button.label.plain
            for button in modal.query(".console-workspace-files-entry")
        ]
        assert not any(" [excluded]" in label for label in labels), labels


async def test_toggle_action_unexclude_under_parent_targets_covering_exclusion() -> None:
    service = _FakeFilesService(_root_page())
    modal = _files_modal(service, exclusions=("secrets",))
    app = App[None]()
    async with app.run_test(size=(120, 40)) as pilot:
        app.push_screen(modal)
        await pilot.pause()
        modal._state = replace(modal._state, selected_tree_parts=("secrets", "key.pem"))

        await modal.action_toggle_exclusion()
        await pilot.pause()

        # T1 removal is exact-path, so the covering "secrets" entry is the
        # one that must be restored, not the deeper selected path.
        assert service.exclusion_calls == [("binding-1", "secrets", False)]


async def test_toggle_action_excludes_selected_plain_entry_and_badges_it() -> None:
    service = _FakeFilesService(_root_page())
    modal = _files_modal(service, exclusions=())
    app = App[None]()
    async with app.run_test(size=(120, 40)) as pilot:
        app.push_screen(modal)
        for _ in range(50):
            await pilot.pause()
            if list(modal.query(".console-workspace-files-entry")):
                break
        modal._state = replace(modal._state, selected_tree_parts=("notes.md",))

        await modal.action_toggle_exclusion()
        await pilot.pause()

        assert service.exclusion_calls == [("binding-1", "notes.md", True)]
        labels = [
            button.label.plain
            for button in modal.query(".console-workspace-files-entry")
        ]
        assert any(
            "notes.md" in label and " [excluded]" in label for label in labels
        ), labels


async def test_nested_unexclude_strips_deepest_layer_first_with_truthful_status() -> None:
    service = _FakeFilesService(_root_page())
    modal = _files_modal(service, exclusions=("secrets", "secrets/keys"))
    app = App[None]()
    async with app.run_test(size=(120, 40)) as pilot:
        app.push_screen(modal)
        for _ in range(50):
            await pilot.pause()
            if list(modal.query(".console-workspace-files-entry")):
                break
        modal._state = replace(
            modal._state, selected_tree_parts=("secrets", "keys", "a.pem")
        )

        # First press removes only the deepest covering layer; the selected
        # path stays excluded by the parent, and the status must say so.
        await modal.action_toggle_exclusion()
        await pilot.pause()

        assert service.exclusion_calls == [("binding-1", "secrets/keys", False)]
        assert "Still excluded by secrets" in modal.state.status_copy
        labels = [
            button.label.plain
            for button in modal.query(".console-workspace-files-entry")
        ]
        assert any(
            "secrets" in label and " [excluded]" in label for label in labels
        ), labels

        # Second press removes the remaining parent layer and only then
        # claims the path is included.
        await modal.action_toggle_exclusion()
        await pilot.pause()

        assert service.exclusion_calls == [
            ("binding-1", "secrets/keys", False),
            ("binding-1", "secrets", False),
        ]
        assert (
            modal.state.status_copy
            == "Included secrets/keys/a.pem for agent file tools."
        )
        labels = [
            button.label.plain
            for button in modal.query(".console-workspace-files-entry")
        ]
        assert not any(" [excluded]" in label for label in labels), labels


async def test_toggle_action_without_selection_calls_nothing() -> None:
    service = _FakeFilesService(_root_page())
    modal = _files_modal(service, exclusions=("secrets",))
    app = App[None]()
    async with app.run_test(size=(120, 40)) as pilot:
        app.push_screen(modal)
        await pilot.pause()

        await modal.action_toggle_exclusion()
        await pilot.pause()

        assert service.exclusion_calls == []
