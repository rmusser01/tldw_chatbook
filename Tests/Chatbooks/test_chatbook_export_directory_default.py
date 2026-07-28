# test_chatbook_export_directory_default.py
# Description: Regression tests for task-984 -- reconciling the Chatbook
# export directory default across the four windows.
#
"""
Chatbook Export Directory Default
----------------------------------

Covers the runtime behavior half of task-984: ``ChatbookExportManagementWindow``,
``ChatbooksWindowImproved`` and the ``ChatbookCreationWizard`` preview step now
default the visible export directory to ``get_private_chatbooks_dir()`` (the
same accessor ``ChatbookCreationWindow`` already used) instead of the
hardcoded ``~/Documents/Chatbooks`` literal.

Every expected value below is derived by calling the real accessor with its
dependency monkeypatched, never by re-spelling a path literal -- a test that
repeats the literal goes vacuous in lockstep with the next drift instead of
catching it.
"""

from pathlib import Path

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Static

from tldw_chatbook.Chatbooks import database_paths
from tldw_chatbook.UI.ChatbookExportManagementWindow import (
    ChatbookExportManagementWindow,
)
from tldw_chatbook.UI.Chatbooks_Window_Improved import ChatbooksWindowImproved
from tldw_chatbook.UI.Wizards.BaseWizard import WizardStepConfig
from tldw_chatbook.UI.Wizards.ChatbookCreationWizard import PreviewConfirmStep


class _FakeAppInstance:
    """Minimal stand-in for TldwCli; the windows under test only store it."""


class _FakeWizard:
    """Minimal stand-in for WizardContainer; only `.wizard_data` is read."""

    def __init__(self) -> None:
        self.wizard_data: dict = {}


def test_export_management_window_default_uses_private_chatbooks_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The management window's default storage dir is get_private_chatbooks_dir()."""
    user_data_dir = tmp_path / "runtime-data"
    monkeypatch.setattr(
        database_paths.config, "get_user_data_dir", lambda: user_data_dir
    )

    window = ChatbookExportManagementWindow(_FakeAppInstance())

    assert window.chatbooks_dir == database_paths.get_private_chatbooks_dir()


def test_chatbooks_window_improved_default_uses_private_chatbooks_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The landing page's default scan dir is get_private_chatbooks_dir()."""
    user_data_dir = tmp_path / "runtime-data"
    monkeypatch.setattr(
        database_paths.config, "get_user_data_dir", lambda: user_data_dir
    )

    window = ChatbooksWindowImproved(_FakeAppInstance())

    assert window._export_path == database_paths.get_private_chatbooks_dir()


@pytest.mark.asyncio
async def test_wizard_preview_confirm_default_export_path_uses_private_chatbooks_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The creation wizard's previewed export path sits under the private dir.

    Also proves the UI keeps showing the resolved directory rather than
    implying a fixed location: `#export-path` renders the same directory the
    wizard will actually write to.
    """
    user_data_dir = tmp_path / "runtime-data"
    monkeypatch.setattr(
        database_paths.config, "get_user_data_dir", lambda: user_data_dir
    )

    fake_wizard = _FakeWizard()
    step = PreviewConfirmStep(
        wizard=fake_wizard,
        config=WizardStepConfig(
            id="preview-confirm",
            title="Preview & Confirm",
            description="Review your chatbook",
            step_number=4,
        ),
    )

    class WizardStepApp(App):
        def compose(self) -> ComposeResult:
            yield step

    app = WizardStepApp()
    async with app.run_test():
        step._update_preview()
        rendered_path = str(step.query_one("#export-path", Static).content)

    expected_dir = database_paths.get_private_chatbooks_dir()
    export_path = Path(fake_wizard.wizard_data["export_path"])

    assert export_path.parent == expected_dir
    assert str(expected_dir) in rendered_path


@pytest.mark.asyncio
async def test_wizard_preview_confirm_server_mode_never_touches_local_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Server-mode preview must not resolve or create the local chatbooks dir.

    Regression guard: `_update_preview()` used to call
    `get_private_chatbooks_dir()` -- which hardens and *creates* the
    directory -- before checking `execution_mode`, then discarded the
    result for server-mode exports. That made merely opening the preview
    step mutate the filesystem (and able to raise) even though server mode
    never needs a local directory.
    """
    from tldw_chatbook.UI.Wizards import ChatbookCreationWizard as wizard_module

    user_data_dir = tmp_path / "runtime-data"
    monkeypatch.setattr(
        database_paths.config, "get_user_data_dir", lambda: user_data_dir
    )

    call_count = 0
    real_get_private_chatbooks_dir = wizard_module.get_private_chatbooks_dir

    def _counting_get_private_chatbooks_dir():
        nonlocal call_count
        call_count += 1
        return real_get_private_chatbooks_dir()

    monkeypatch.setattr(
        wizard_module, "get_private_chatbooks_dir", _counting_get_private_chatbooks_dir
    )

    fake_wizard = _FakeWizard()
    fake_wizard.wizard_data["export-options"] = {"execution_mode": "server"}
    step = PreviewConfirmStep(
        wizard=fake_wizard,
        config=WizardStepConfig(
            id="preview-confirm",
            title="Preview & Confirm",
            description="Review your chatbook",
            step_number=4,
        ),
    )

    class WizardStepApp(App):
        def compose(self) -> ComposeResult:
            yield step

    app = WizardStepApp()
    async with app.run_test():
        step._update_preview()
        rendered_path = str(step.query_one("#export-path", Static).content)

    assert call_count == 0, "server mode must not resolve the local directory at all"
    assert not user_data_dir.exists(), (
        "server mode must not create the local chatbooks directory as a side effect"
    )
    assert fake_wizard.wizard_data["export_path"] == ""
    assert "Server-side export" in rendered_path


@pytest.mark.parametrize(
    "window_factory",
    [
        lambda: ChatbookExportManagementWindow(_FakeAppInstance()).chatbooks_dir,
        lambda: ChatbooksWindowImproved(_FakeAppInstance())._export_path,
    ],
    ids=["export-management-window", "chatbooks-window-improved"],
)
def test_explicit_data_dir_override_relocates_export_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    window_factory,
) -> None:
    """A user-configured data directory still wins over the default root.

    Neither window has its own export-directory config key; the only lever a
    user has to relocate exports today is the general ``[paths] data_dir``
    setting that ``get_private_chatbooks_dir()`` already resolves through
    ``get_user_data_dir()``. This proves the windows follow wherever that
    resolves rather than a value baked in at construction time.
    """
    first_root = tmp_path / "root-one"
    monkeypatch.setattr(database_paths.config, "get_user_data_dir", lambda: first_root)
    first_resolved = window_factory()
    first_expected = database_paths.get_private_chatbooks_dir()
    assert first_resolved == first_expected

    second_root = tmp_path / "root-two"
    monkeypatch.setattr(
        database_paths.config, "get_user_data_dir", lambda: second_root
    )
    second_resolved = window_factory()
    second_expected = database_paths.get_private_chatbooks_dir()
    assert second_resolved == second_expected

    assert first_expected != second_expected


def test_existing_export_at_old_documents_location_is_left_untouched(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Constructing a window under the new default must not touch old exports.

    Regression guard for task-984's default-only constraint: this change only
    affects what the export directory defaults to. A user who already has
    exports under the pre-existing ``~/Documents/Chatbooks`` literal keeps
    them exactly where they are -- nothing is moved, copied, or deleted.
    """
    # `Path.home()` resolves the per-test isolated HOME set by the autouse
    # `isolate_test_environment` fixture in Tests/conftest.py, never the
    # real user's home directory.
    old_export_dir = Path.home() / "Documents" / "Chatbooks"
    old_export_dir.mkdir(parents=True, mode=0o755)
    marker = old_export_dir / "existing_export.zip"
    marker.write_bytes(b"pre-existing chatbook export")
    marker_mtime_before = marker.stat().st_mtime_ns

    user_data_dir = tmp_path / "runtime-data"
    monkeypatch.setattr(
        database_paths.config, "get_user_data_dir", lambda: user_data_dir
    )

    management_window = ChatbookExportManagementWindow(_FakeAppInstance())
    landing_window = ChatbooksWindowImproved(_FakeAppInstance())

    resolved = database_paths.get_private_chatbooks_dir()
    assert management_window.chatbooks_dir == resolved
    assert landing_window._export_path == resolved
    assert resolved != old_export_dir

    # The old location and its pre-existing contents are exactly as they
    # were -- not read into a listing, not deleted, not written to.
    assert old_export_dir.exists()
    assert [p.name for p in old_export_dir.iterdir()] == ["existing_export.zip"]
    assert marker.read_bytes() == b"pre-existing chatbook export"
    assert marker.stat().st_mtime_ns == marker_mtime_before
