"""Restore from setup: open on Inspect, name the format, recognise the wrong file.

TASK-34100.16 [entry-exit-handoff-16]. Welcome's "Restore a backup" opened
the generic "Backup & Restore" home, whose first action is Create backup. The
Inspect pane never named the archive format; a config.toml failed as
"Failed: inspecting ... (backup_operation_failed)"; a folder failed the same
way; and Create backup stayed disabled after Review with nothing on screen
saying why. These tests pin:

* the setup entry (``initial_mode="inspect"``) opens directly on Inspect,
  titled "Restore from a backup", and the Inspect pane names
  .tldw-backup.zip / .tldw-backup.zip.age;
* a *.toml gets "That's a settings file, not a backup archive" with the
  second-machine guide pointer, and a folder gets "Choose the archive file,
  not a folder." -- neither starts an inspection;
* a real archive still starts an inspection from the setup entry;
* whenever Create backup is disabled in the Create form, the line directly
  above the buttons says why (before Review, after a change voids it,
  Partial not acknowledged, not enough space, unavailable).
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture(autouse=True)
def _disable_model_catalog_refresh():
    """This screen harness has no application startup or catalog refresh.

    Same override as Tests/UI/test_backup_restore_screen.py: the UI conftest's
    version imports the full application inside the per-test sandbox.
    """


# --- pure rules (backup_restore_state) ---------------------------------------


def test_archive_source_problem_names_a_settings_file_and_a_folder(tmp_path):
    from tldw_chatbook.UI.Screens.backup_restore_state import archive_source_problem

    settings = tmp_path / "config.toml"
    settings.write_text("[general]\n")
    message = archive_source_problem(settings)
    assert message.startswith("That's a settings file, not a backup archive.")
    assert "Setting up another machine" in message
    assert "tldw-cli --config" in message
    # Review round 1: no repo path (an installed user has no Docs/), and no
    # path of any length -- the copy must fit the message line at 54 columns.
    assert "Docs/" not in message and str(tmp_path) not in message
    assert archive_source_problem(tmp_path / "a" / ("long" * 20) / "config.toml") == message
    assert archive_source_problem(tmp_path / "Other.TOML") is not None
    assert archive_source_problem(tmp_path) == "Choose the archive file, not a folder."
    assert archive_source_problem(tmp_path / "home.tldw-backup.zip") is None
    assert archive_source_problem(tmp_path / "home.tldw-backup.zip.age") is None


@pytest.mark.skipif(
    not hasattr(os, "geteuid") or os.geteuid() == 0,
    reason="needs POSIX permissions that apply to this user",
)
def test_archive_source_problem_never_raises_for_an_unreadable_folder(tmp_path):
    """Review round 1: ``Path.is_dir()`` raises PermissionError on Python 3.12.

    An archive under a folder the terminal cannot traverse (a macOS
    privacy-protected Downloads, a chmod-000 folder) must fall through to the
    inspection, which reports its own failure, instead of raising.
    """
    from tldw_chatbook.UI.Screens.backup_restore_state import archive_source_problem

    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0)
    try:
        assert archive_source_problem(locked / "home.tldw-backup.zip") is None
        assert archive_source_problem(locked / "config.toml").startswith(
            "That's a settings file"
        )
    finally:
        locked.chmod(0o700)


def _capacity(sufficient=True, path="/Volumes/Backups"):
    return (
        {
            "path": path,
            "required_bytes": 2_000_000,
            "available_bytes": 1_000_000 if not sufficient else 9_000_000,
            "sufficient": sufficient,
        },
    )


def test_create_unavailable_reason_names_each_blocking_condition():
    from tldw_chatbook.UI.Screens.backup_restore_state import create_unavailable_reason

    def reason(**overrides):
        values = dict(
            complete=True,
            allow_partial=False,
            capacity=_capacity(),
            available=True,
            unavailable_message="",
            destination=Path("/Volumes/Backups/home.tldw-backup.zip"),
        )
        values.update(overrides)
        return create_unavailable_reason(**values)

    assert reason() is None
    assert reason(complete=False, allow_partial=True) is None
    partial = reason(complete=False)
    assert "Partial" in partial and "Acknowledge Partial archive" in partial
    assert "Review" in partial
    space = reason(capacity=_capacity(sufficient=False))
    assert space.startswith("Not enough free space") and "/Volumes/Backups" in space
    assert "choose another location" in space
    # Review round 1: the staging volume (the system temporary folder) has no
    # control in the form, so its row never advises moving the destination.
    staging = reason(capacity=_capacity(sufficient=False, path="/private/var/tmp"))
    assert staging.startswith("Not enough free space")
    assert "temporary folder" in staging
    assert "choose another location" not in staging
    unavailable = reason(available=False, unavailable_message="Pause writers first.")
    assert unavailable == "Create backup is unavailable: Pause writers first."
    # The service's own refusal outranks the form's conditions.
    assert reason(available=False, unavailable_message="X", complete=False).endswith("X")


# --- the screen ----------------------------------------------------------------


def _harness(service, **screen_kwargs):
    from Tests.UI.consolidated_css import ConsolidatedCSSApp as App
    from tldw_chatbook.UI.Screens.backup_restore_screen import BackupRestoreScreen

    class Harness(App):
        def on_mount(self):
            self.push_screen(BackupRestoreScreen(service, **screen_kwargs))

    return Harness()


def _text(screen, selector):
    from textual.widgets import Static

    return str(screen.query_one(selector, Static).render())


@pytest.mark.asyncio
async def test_setup_entry_opens_on_inspect_titled_restore_from_a_backup(tmp_path):
    from textual.widgets import Input

    from tldw_chatbook.Backup_Recovery.recovery_service import RecoveryService

    service = RecoveryService(tmp_path / "control")
    app = _harness(service, config_paths=(), initial_mode="inspect")
    try:
        async with app.run_test(size=(100, 36)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert _text(screen, "#backup-title") == "Restore from a backup"
            assert screen.query_one("#backup-inspect-form").display
            assert not screen.query_one("#backup-home").display
            assert not screen.query_one("#backup-create-form").display
            assert screen.query_one("#backup-inspect").display
            assert not screen.query_one("#backup-create").display
            hint = _text(screen, "#backup-archive-format")
            assert ".tldw-backup.zip" in hint and ".tldw-backup.zip.age" in hint
            # Ready to type the archive path straight away.
            assert screen.focused is screen.query_one("#backup-source", Input)
    finally:
        service.close()


@pytest.mark.asyncio
async def test_the_ordinary_entry_keeps_its_home_and_its_inspect_pane_names_the_format(
    tmp_path,
):
    from tldw_chatbook.Backup_Recovery.recovery_service import RecoveryService

    service = RecoveryService(tmp_path / "control")
    app = _harness(service, config_paths=())
    try:
        async with app.run_test(size=(100, 36)) as pilot:
            screen = app.screen
            assert _text(screen, "#backup-title") == "Backup & Restore"
            assert screen.query_one("#backup-home").display
            await pilot.click("#backup-open-inspect")
            hint = _text(screen, "#backup-archive-format")
            assert ".tldw-backup.zip" in hint and ".tldw-backup.zip.age" in hint
    finally:
        service.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["settings", "folder"])
async def test_a_settings_file_or_a_folder_is_named_and_never_inspected(tmp_path, kind):
    from textual.widgets import Input

    from tldw_chatbook.Backup_Recovery.recovery_service import RecoveryService

    if kind == "settings":
        source = tmp_path / "config.toml"
        source.write_text('[general]\nusers_name="copied"\n')
        expected = "That's a settings file, not a backup archive."
    else:
        source = tmp_path / "carried"
        source.mkdir()
        expected = "Choose the archive file, not a folder."
    service = RecoveryService(tmp_path / "control")
    app = _harness(service, config_paths=(), initial_mode="inspect")
    try:
        async with app.run_test(size=(100, 36)) as pilot:
            screen = app.screen
            screen.query_one("#backup-source", Input).value = str(source)
            await pilot.click("#backup-inspect")
            await pilot.pause()
            message = _text(screen, "#backup-message")
            assert message.startswith(expected), message
            assert "backup_operation_failed" not in message
            if kind == "settings":
                assert "Setting up another machine" in message
            assert service.current() is None
    finally:
        service.close()


@pytest.mark.asyncio
async def test_a_real_archive_still_starts_an_inspection_from_the_setup_entry(tmp_path):
    import asyncio

    from textual.widgets import Input

    from Tests.Backup_Recovery.test_archive_reader import archive
    from tldw_chatbook.Backup_Recovery.recovery_service import RecoveryService

    source = archive(tmp_path)
    service = RecoveryService(tmp_path / "control")
    app = _harness(service, config_paths=(), initial_mode="inspect")
    try:
        async with app.run_test(size=(100, 36)) as pilot:
            screen = app.screen
            screen.query_one("#backup-source", Input).value = str(source)
            await pilot.click("#backup-inspect")
            current = service.current()
            assert current is not None and current["kind"].startswith("inspect")
            result = await asyncio.to_thread(
                service.wait, current["operation_id"], timeout=10
            )
            assert result["result"]["archive_verified"]
    finally:
        await asyncio.to_thread(service.close)


def _review_details(*, complete=True, sufficient=True, available=(True, None)):
    preview = SimpleNamespace(
        items=(), complete=complete, issues=(), scope_digest="digest"
    )
    return {
        "inventory": preview,
        "availability": available,
        "whole_profile": True,
        "complete": complete,
        "maintenance": "Writers pause while data is copied.",
        "capacity": _capacity(sufficient),
        "credential_mode": "exclude",
    }


@pytest.mark.asyncio
async def test_a_disabled_create_always_says_why_on_the_line_above_it(tmp_path):
    from textual.widgets import Button, Checkbox

    from tldw_chatbook.Backup_Recovery.recovery_service import RecoveryService
    from tldw_chatbook.UI.Screens.backup_restore_state import (
        CREATE_NEEDS_REVIEW,
        CREATE_STARTED,
    )

    service = RecoveryService(tmp_path / "control")
    app = _harness(service, config_paths=(tmp_path / "config.toml",))
    destination = tmp_path / "home.tldw-backup.zip"
    try:
        async with app.run_test(size=(100, 36)) as pilot:
            await pilot.click("#backup-open-create")
            screen = app.screen
            create = screen.query_one("#backup-create", Button)

            # Before any review.
            assert create.disabled
            assert _text(screen, "#backup-message") == CREATE_NEEDS_REVIEW

            def show(details, *, allow_partial=False):
                screen._show_preview(
                    screen._revision,
                    details,
                    (
                        (),
                        destination,
                        {"allow_partial": allow_partial, "encrypted": False},
                    ),
                )

            show(_review_details(complete=False))
            assert create.disabled
            assert "Partial" in _text(screen, "#backup-message")

            show(_review_details(sufficient=False))
            assert create.disabled
            assert _text(screen, "#backup-message").startswith("Not enough free space")

            show(_review_details(available=(False, "admission_timeout")))
            assert create.disabled
            assert _text(screen, "#backup-message").startswith(
                "Create backup is unavailable"
            )

            show(_review_details())
            assert not create.disabled
            assert _text(screen, "#backup-message").startswith("Review the displayed")

            # Any form change voids the review: Create disables and says why.
            box = screen.query_one("#backup-temporary", Checkbox)
            box.value = not box.value
            await pilot.pause()
            assert create.disabled
            assert _text(screen, "#backup-message") == CREATE_NEEDS_REVIEW

            # Review round 1: pressing Create voids the review too; through the
            # run and after it ends, the line says how Create comes back.
            show(_review_details())
            assert not create.disabled
            started = []
            service.start_backup = lambda *args, **kwargs: started.append(args) or "op-1"
            await pilot.click("#backup-create")
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert len(started) == 1
            assert create.disabled
            assert _text(screen, "#backup-message") == CREATE_STARTED

            # Other modes never carry it.
            await pilot.click("#backup-open-inspect")
            assert _text(screen, "#backup-message") == ""
    finally:
        service.close()



# --- review round 1 -------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.skipif(
    not hasattr(os, "geteuid") or os.geteuid() == 0,
    reason="needs POSIX permissions that apply to this user",
)
async def test_an_archive_under_an_unreadable_folder_never_takes_the_app_down(
    tmp_path,
):
    """The pre-check raised PermissionError out of the button handler.

    That exited the whole app -- and the setup wizard under it. The archive
    goes on to the inspection, which owns that failure and reports it.
    """
    from textual.widgets import Input

    from tldw_chatbook.Backup_Recovery.recovery_service import RecoveryService

    locked = tmp_path / "locked"
    locked.mkdir()
    service = RecoveryService(tmp_path / "control")
    app = _harness(service, config_paths=(), initial_mode="inspect")
    locked.chmod(0)
    try:
        async with app.run_test(size=(100, 36)) as pilot:
            screen = app.screen
            screen.query_one("#backup-source", Input).value = str(
                locked / "home.tldw-backup.zip"
            )
            await pilot.click("#backup-inspect")
            await pilot.pause()
            assert app.is_running
            assert app.screen is screen
            assert (
                service.current() is not None
                or _text(screen, "#backup-message") != ""
            )
    finally:
        locked.chmod(0o700)
        service.close()


def _shown_and_needed(screen, selector):
    """Rows the line shows, and rows its text needs at the current width."""
    from textual.widgets import Static

    line = screen.query_one(selector, Static)
    width = line.content_region.width
    needed = line.get_content_height(screen.size, screen.size, width)
    return line.content_region.height, needed


@pytest.mark.asyncio
async def test_the_new_messages_fit_their_line_at_the_narrowest_width(tmp_path):
    """Review round 1: at 54x22 the line shows 4 rows, and both clipped.

    The settings-file message lost its ``tldw-cli --config`` command and the
    Partial reason lost "press Review again". The line's max-height is what
    keeps the form usable at that size, so the copy has to fit it.
    """
    from textual.widgets import Input

    from tldw_chatbook.Backup_Recovery.recovery_service import RecoveryService

    settings = tmp_path / "Downloads" / "a-rather-long-folder-name" / "config.toml"
    settings.parent.mkdir(parents=True)
    settings.write_text("[general]\n")
    service = RecoveryService(tmp_path / "control")
    app = _harness(service, config_paths=(tmp_path / "config.toml",))
    try:
        async with app.run_test(size=(54, 22)) as pilot:
            screen = app.screen
            await pilot.click("#backup-open-inspect")
            screen.query_one("#backup-source", Input).value = str(settings)
            await pilot.pause()
            screen._inspect()
            await pilot.pause()
            assert _text(screen, "#backup-message").startswith("That's a settings")
            shown, needed = _shown_and_needed(screen, "#backup-message")
            assert needed <= shown, (shown, needed)

            await pilot.click("#backup-open-create")
            screen._show_preview(
                screen._revision,
                _review_details(complete=False),
                ((), tmp_path / "b.tldw-backup.zip", {"allow_partial": False}),
            )
            await pilot.pause()
            assert "Review again" in _text(screen, "#backup-message")
            shown, needed = _shown_and_needed(screen, "#backup-message")
            assert needed <= shown, (shown, needed)

            # A short staging volume: macOS's temporary folder alone is wider
            # than the line, so the reason must not need its path.
            details = _review_details()
            details["capacity"] = _capacity(
                sufficient=False,
                path="/private/var/folders/p_/x47tgtn57cv43r7yxxn40tyh0000gn/T",
            )
            screen._show_preview(
                screen._revision,
                details,
                ((), tmp_path / "b.tldw-backup.zip", {"allow_partial": False}),
            )
            await pilot.pause()
            assert "temporary folder" in _text(screen, "#backup-message")
            shown, needed = _shown_and_needed(screen, "#backup-message")
            assert needed <= shown, (shown, needed)
    finally:
        service.close()


@pytest.mark.asyncio
async def test_changing_the_archive_path_clears_the_previous_problem(tmp_path):
    """Review round 1: "not a folder" stayed above a field now holding a file."""
    from textual.widgets import Input

    from tldw_chatbook.Backup_Recovery.recovery_service import RecoveryService

    service = RecoveryService(tmp_path / "control")
    app = _harness(service, config_paths=(), initial_mode="inspect")
    try:
        async with app.run_test(size=(100, 36)) as pilot:
            screen = app.screen
            field = screen.query_one("#backup-source", Input)
            field.value = str(tmp_path)
            await pilot.click("#backup-inspect")
            await pilot.pause()
            assert _text(screen, "#backup-message").startswith("Choose the archive file")
            field.value = str(tmp_path / "config.toml")
            await pilot.pause()
            assert _text(screen, "#backup-message") == ""
    finally:
        service.close()


@pytest.mark.asyncio
async def test_enter_in_the_archive_field_inspects_it(tmp_path):
    """Review round 1: setup focuses the field, so Enter is the next keystroke."""
    from textual.widgets import Input

    from tldw_chatbook.Backup_Recovery.recovery_service import RecoveryService

    settings = tmp_path / "config.toml"
    settings.write_text("[general]\n")
    service = RecoveryService(tmp_path / "control")
    app = _harness(service, config_paths=(), initial_mode="inspect")
    try:
        async with app.run_test(size=(100, 36)) as pilot:
            screen = app.screen
            field = screen.query_one("#backup-source", Input)
            assert screen.focused is field
            field.value = str(settings)
            await pilot.press("enter")
            await pilot.pause()
            assert _text(screen, "#backup-message").startswith(
                "That's a settings file, not a backup archive."
            )
            assert service.current() is None
    finally:
        service.close()


@pytest.mark.asyncio
async def test_setup_entry_leads_with_inspect_and_its_title_follows_the_pane(tmp_path):
    """Review round 1: Create backup was still the first action from setup.

    And after switching to Create, the title still said "Restore from a
    backup" above the Create form.
    """
    from textual.widgets import Button

    from tldw_chatbook.Backup_Recovery.recovery_service import RecoveryService

    service = RecoveryService(tmp_path / "control")
    app = _harness(service, config_paths=(), initial_mode="inspect")
    try:
        async with app.run_test(size=(100, 36)) as pilot:
            screen = app.screen
            first = screen.query_one("#backup-actions").query(Button).first()
            assert first.id == "backup-open-inspect"
            await pilot.click("#backup-open-create")
            assert _text(screen, "#backup-title") == "Backup & Restore"
            await pilot.click("#backup-open-inspect")
            assert _text(screen, "#backup-title") == "Restore from a backup"
    finally:
        service.close()


@pytest.mark.asyncio
async def test_the_ordinary_entry_keeps_create_first_and_its_title(tmp_path):
    """The paired control: Settings' entry is unchanged by the setup entry."""
    from textual.widgets import Button

    from tldw_chatbook.Backup_Recovery.recovery_service import RecoveryService

    service = RecoveryService(tmp_path / "control")
    app = _harness(service, config_paths=())
    try:
        async with app.run_test(size=(100, 36)) as pilot:
            screen = app.screen
            first = screen.query_one("#backup-actions").query(Button).first()
            assert first.id == "backup-open-create"
            await pilot.click("#backup-open-inspect")
            assert _text(screen, "#backup-title") == "Backup & Restore"
    finally:
        service.close()
