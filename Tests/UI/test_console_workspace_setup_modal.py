"""Console new-workspace setup modal: validation gating and dismiss results.

"New Workspace" in the Console must no longer silently create a bare
workspace -- it opens this modal, which requires a validated folder
binding before Create enables, and dismisses with the confirmed
(name, path, access) triple (or ``None`` on cancel).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pytest
from textual.app import App, ComposeResult

from tldw_chatbook.Widgets.Console.console_workspace_setup_modal import (
    ConsoleWorkspaceSetupModal,
    ConsoleWorkspaceSetupResult,
)

CREATE_BTN_ID = "#console-workspace-setup-create"
CANCEL_BTN_ID = "#console-workspace-setup-cancel"
NAME_INPUT_ID = "#console-workspace-setup-name"
PATH_INPUT_ID = "#console-workspace-setup-path"
WRITE_CHECKBOX_ID = "#console-workspace-setup-write"
ERROR_STATIC_ID = "#console-workspace-setup-error"


class _SetupHarness(App[None]):
    """Push the setup modal and record what it dismisses with."""

    def __init__(self, suggested_name: str, validate) -> None:
        super().__init__()
        self._suggested = suggested_name
        self._validate = validate
        self.dismissed_with: object = "--never-dismissed--"

    def compose(self) -> ComposeResult:
        yield from ()

    def on_mount(self) -> None:
        modal = ConsoleWorkspaceSetupModal(
            suggested_name=self._suggested,
            validate=self._validate,
            debounce_seconds=0.001,
        )

        def _record(result: object) -> None:
            self.dismissed_with = result

        self.push_screen(modal, callback=_record)


@pytest.mark.asyncio
async def test_create_disabled_until_folder_valid() -> None:
    app = _SetupHarness(
        "Workspace 4", lambda name, path: "nope" if not path else None
    )
    async with app.run_test() as pilot:
        modal = app.screen
        assert isinstance(modal, ConsoleWorkspaceSetupModal)
        # Path starts empty -> validator errors -> Create disabled.
        await pilot.pause()
        await pilot.pause()
        assert modal.query_one(CREATE_BTN_ID).disabled
        assert "nope" in str(modal.query_one(ERROR_STATIC_ID).render())

        modal.query_one(PATH_INPUT_ID).value = "/tmp/some-project"
        await pilot.pause()
        await pilot.pause()
        assert not modal.query_one(CREATE_BTN_ID).disabled
        assert str(modal.query_one(ERROR_STATIC_ID).render()).strip() == ""


@pytest.mark.asyncio
async def test_cancel_dismisses_none_and_creates_nothing() -> None:
    app = _SetupHarness("Workspace 4", lambda name, path: None)
    async with app.run_test() as pilot:
        modal = app.screen
        await pilot.pause()
        await pilot.pause()
        await pilot.click(CANCEL_BTN_ID)
        await pilot.pause()
        assert app.screen is not modal
        assert app.dismissed_with is None


@pytest.mark.asyncio
async def test_confirm_carries_name_path_and_read_only_default() -> None:
    app = _SetupHarness("Workspace 4", lambda name, path: None)
    async with app.run_test() as pilot:
        modal = app.screen
        assert isinstance(modal, ConsoleWorkspaceSetupModal)
        modal.query_one(NAME_INPUT_ID).value = "My Project"
        modal.query_one(PATH_INPUT_ID).value = "/tmp/some-project"
        await pilot.pause()
        await pilot.pause()
        assert not modal.query_one(CREATE_BTN_ID).disabled
        await pilot.click(CREATE_BTN_ID)
        await pilot.pause()
        assert app.dismissed_with == ConsoleWorkspaceSetupResult(
            name="My Project",
            folder_path="/tmp/some-project",
            allow_write=False,
        )


@pytest.mark.asyncio
async def test_read_write_checkbox_flips_access() -> None:
    app = _SetupHarness("Workspace 4", lambda name, path: None)
    async with app.run_test() as pilot:
        modal = app.screen
        assert isinstance(modal, ConsoleWorkspaceSetupModal)
        modal.query_one(PATH_INPUT_ID).value = "/tmp/some-project"
        await pilot.pause()
        await pilot.pause()
        assert not modal.query_one(WRITE_CHECKBOX_ID).value
        await pilot.click(WRITE_CHECKBOX_ID)
        await pilot.pause()
        assert modal.query_one(WRITE_CHECKBOX_ID).value
        await pilot.click(CREATE_BTN_ID)
        await pilot.pause()
        result = app.dismissed_with
        assert isinstance(result, ConsoleWorkspaceSetupResult)
        assert result.allow_write is True


@pytest.mark.asyncio
async def test_submit_on_invalid_path_does_not_dismiss() -> None:
    def validator(name: str, path: str) -> Optional[str]:
        return "Folder does not exist" if not Path(path).is_dir() else None

    app = _SetupHarness("Workspace 4", validator)
    async with app.run_test() as pilot:
        modal = app.screen
        assert isinstance(modal, ConsoleWorkspaceSetupModal)
        modal.query_one(PATH_INPUT_ID).value = "/definitely/not/a/dir"
        await pilot.pause()
        await pilot.pause()
        assert modal.query_one(CREATE_BTN_ID).disabled
        await pilot.press("enter")
        await pilot.pause()
        assert app.screen is modal
        assert "Folder does not exist" in str(
            modal.query_one(ERROR_STATIC_ID).render()
        )
