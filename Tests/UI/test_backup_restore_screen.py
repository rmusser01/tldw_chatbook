"""Canonical backup UI reports only the evidence supplied by its service."""

import pytest

from Tests.Backup_Recovery.conftest import (
    helper_resource_root as helper_resource_root,  # noqa: PLC0414 - pytest fixture re-export
)
from Tests.Backup_Recovery.test_launcher import (
    pre_safety_recovery as pre_safety_recovery,  # noqa: PLC0414 - shared actual native fixture
)


@pytest.fixture(autouse=True)
def _disable_model_catalog_refresh():
    """This screen harness has no application startup or catalog refresh.

    Production composition runs in a fresh, explicitly selected child profile.
    Avoid importing the full application solely to patch its startup here.
    """


def test_archive_verification_does_not_claim_profile_opened():
    from tldw_chatbook.UI.Screens.backup_restore_state import result_label

    assert (
        result_label(
            archive_verified=True,
            restoration_validated=False,
            opened=False,
            needs_setup=False,
        )
        == "Archive verified"
    )


@pytest.mark.parametrize(
    ("verified", "validated", "opened", "setup", "expected"),
    [
        (False, False, False, False, "Not verified"),
        (True, True, False, False, "Restoration validated"),
        (True, True, True, False, "Opened successfully"),
        (True, True, True, True, "Needs setup"),
    ],
)
def test_result_labels_distinguish_validation_opening_and_setup(
    verified, validated, opened, setup, expected
):
    from tldw_chatbook.UI.Screens.backup_restore_state import result_label

    assert (
        result_label(
            archive_verified=verified,
            restoration_validated=validated,
            opened=opened,
            needs_setup=setup,
        )
        == expected
    )


@pytest.mark.asyncio
async def test_actual_archive_inspection_survives_navigation(tmp_path, monkeypatch):
    import asyncio
    from threading import Event

    from textual.app import App
    from textual.widgets import Input, Static

    from Tests.Backup_Recovery.test_archive_reader import archive
    from tldw_chatbook.Backup_Recovery import archive_reader
    from tldw_chatbook.Backup_Recovery.recovery_service import RecoveryService
    from tldw_chatbook.UI.Screens.backup_restore_screen import BackupRestoreScreen

    source = archive(tmp_path)
    service = RecoveryService(tmp_path / "control")
    entered, release = Event(), Event()
    acquire = archive_reader.acquire

    def held_acquire(*args, **kwargs):
        entered.set()
        assert release.wait(10)
        return acquire(*args, **kwargs)

    monkeypatch.setattr(archive_reader, "acquire", held_acquire)

    class Harness(App):
        def on_mount(self):
            self.push_screen(BackupRestoreScreen(service, config_paths=()))

    app = Harness()
    try:
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.click("#backup-open-inspect")
            screen = app.screen
            screen.query_one("#backup-source", Input).value = str(source)
            await pilot.click("#backup-inspect")
            for _ in range(100):
                if entered.is_set():
                    break
                await asyncio.sleep(0.01)
            assert entered.is_set()
            operation = service.current()["operation_id"]
            await pilot.press("escape")
            assert not isinstance(app.screen, BackupRestoreScreen)
            assert service.status(operation)["state"] == "running"
            release.set()
            result = await asyncio.to_thread(service.wait, operation, timeout=10)
            assert result["result"]["archive_verified"]
            app.push_screen(BackupRestoreScreen(service, config_paths=()))
            await pilot.pause()
            assert "Archive verified" in str(
                app.screen.query_one("#backup-status", Static).render()
            )
            await pilot.click("#backup-open-inspect")
            for _ in range(100):
                summary = str(
                    app.screen.query_one("#backup-inspection-summary", Static).render()
                )
                if "Files:" in summary:
                    break
                await asyncio.sleep(0.01)
            assert "Files:" in summary
    finally:
        release.set()
        await asyncio.to_thread(service.close)


@pytest.mark.asyncio
async def test_password_mismatch_stays_in_view_and_inputs_clear_on_back(tmp_path):
    from textual.app import App
    from textual.widgets import Input, Static

    from tldw_chatbook.Backup_Recovery.recovery_service import RecoveryService
    from tldw_chatbook.UI.Screens.backup_restore_screen import BackupRestoreScreen

    service = RecoveryService(tmp_path / "control")

    class Harness(App):
        def on_mount(self):
            self.push_screen(BackupRestoreScreen(service, config_paths=()))

    app = Harness()
    try:
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.click("#backup-open-create")
            screen = app.screen
            password = screen.query_one("#backup-password", Input)
            confirmation = screen.query_one("#backup-password-confirm", Input)
            password.value = "first-private-password"
            confirmation.value = "different"
            await pilot.click("#backup-review")
            assert service.current() is None
            assert "match" in str(screen.query_one("#backup-message", Static).render())
            await pilot.press("escape")
            assert password.value == ""
            assert confirmation.value == ""
    finally:
        service.close()


@pytest.mark.asyncio
async def test_actual_recovery_copy_requires_confirmation_and_displays_tombstone(
    tmp_path, monkeypatch, helper_resource_root
):
    import asyncio
    from dataclasses import replace
    from threading import Event

    from textual.app import App
    from textual.widgets import Button, Checkbox, Static

    from Tests.Backup_Recovery.test_held_sqlite_rollback import replacement_case
    from tldw_chatbook.Backup_Recovery import crypto, replacement
    from tldw_chatbook.Backup_Recovery.recovery_service import RecoveryService
    from tldw_chatbook.UI.Screens.backup_restore_screen import BackupRestoreScreen

    monkeypatch.setattr(crypto, "_package_resource_root", lambda: helper_resource_root)
    with replacement_case(tmp_path, monkeypatch, prepared=False) as case:
        candidate, plan, _, _, _, _ = case
        control = tmp_path / "control"
        operation = replacement.replace(
            replace(
                plan, acknowledged_credential_issues=("credential_format_unreadable",)
            ),
            candidate,
            control_root=control,
            rollback_password=b"private rollback",
            cancel=Event(),
        )
        service = RecoveryService(control)

        class Harness(App):
            def on_mount(self):
                self.push_screen(BackupRestoreScreen(service))

        app = Harness()
        try:
            async with app.run_test(size=(80, 24)) as pilot:
                await pilot.click("#backup-open-copies")
                async with asyncio.timeout(10):
                    while not app.screen.query(".backup-review-delete"):
                        await asyncio.sleep(0.01)
                app.screen.query_one(".backup-review-delete", Button).focus()
                await pilot.press("enter")
                assert operation in str(
                    app.screen.query_one("#backup-delete-preview", Static).render()
                )
                app.screen.query_one("#backup-delete-copy", Button).focus()
                await pilot.press("enter")
                assert service.current() is None
                app.screen.query_one("#backup-delete-confirm", Checkbox).value = True
                await pilot.pause()
                await pilot.press("enter")
                current = service.current()
                assert current is not None
                terminal = await asyncio.to_thread(
                    service.wait, current["operation_id"], timeout=10
                )
                assert terminal["state"] == "succeeded", terminal
                async with asyncio.timeout(10):
                    while any(
                        not button.disabled
                        for button in app.screen.query(".backup-review-delete")
                    ):
                        await asyncio.sleep(0.01)
                copy = service.recovery_copies()[0]
                assert copy.status == "missing" and not copy.path.exists()
        finally:
            await asyncio.to_thread(service.close)


@pytest.mark.asyncio
async def test_narrow_local_lists_do_not_start_an_operation(tmp_path):
    import asyncio

    from textual.app import App
    from textual.widgets import Static

    from tldw_chatbook.Backup_Recovery.recovery_service import RecoveryService
    from tldw_chatbook.UI.Screens.backup_restore_screen import BackupRestoreScreen

    service = RecoveryService(tmp_path / "control")

    class Harness(App):
        def on_mount(self):
            self.push_screen(BackupRestoreScreen(service))

    app = Harness()
    try:
        async with app.run_test(size=(54, 20)) as pilot:
            for action, title in (
                ("copies", "Recovery copies"),
                ("profiles", "Restored profiles"),
            ):
                assert await pilot.click("#backup-open-" + action)
                async with asyncio.timeout(5):
                    while title not in str(
                        app.screen.query_one("#backup-list-title", Static).render()
                    ):
                        await asyncio.sleep(0.01)
                assert service.current() is None
            await pilot.press("escape")
            assert not isinstance(app.screen, BackupRestoreScreen)
    finally:
        service.close()


@pytest.mark.asyncio
async def test_credential_omission_acknowledgement_does_not_transfer_to_new_source(
    tmp_path,
):
    from textual.app import App
    from textual.widgets import Checkbox, Input

    from tldw_chatbook.Backup_Recovery.recovery_service import RecoveryService
    from tldw_chatbook.UI.Screens.backup_restore_screen import BackupRestoreScreen

    service = RecoveryService(tmp_path / "control")

    class Harness(App):
        def on_mount(self):
            self.push_screen(BackupRestoreScreen(service))

    app = Harness()
    try:
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.click("#backup-open-create")
            screen = app.screen
            screen.query_one("#backup-credentials", Checkbox).value = True
            await pilot.pause()
            code = "credential_format_unreadable"
            screen._review_codes_seen = (code,)
            await screen._show_credential_review((code,))
            screen.query_one(".backup-acknowledge-credential", Checkbox).value = True
            await pilot.pause()
            assert screen._options()["acknowledged_credential_issues"] == (code,)
            screen.query_one("#backup-destination", Input).value = str(
                tmp_path / "other.zip"
            )
            await pilot.pause()
            assert screen._options()["acknowledged_credential_issues"] == ()
    finally:
        service.close()


@pytest.mark.asyncio
async def test_later_rollback_reviews_native_plan_then_preserves_new_safety_copy(
    tmp_path, monkeypatch, helper_resource_root
):
    import asyncio

    from textual.app import App
    from textual.widgets import Button, Checkbox, Input, Static

    from Tests.Backup_Recovery.test_later_rollback import _completed
    from tldw_chatbook.Backup_Recovery.recovery_service import RecoveryService
    from tldw_chatbook.UI.Screens.backup_restore_screen import BackupRestoreScreen

    with _completed(tmp_path, monkeypatch, helper_resource_root) as (
        case,
        _original,
        prior,
    ):
        service = RecoveryService(tmp_path / "control")
        selector = case[-1]

        def reviewed_target(paths, *, options):
            # This installed fixture declares the current Research DB directly;
            # its deliberately minimal config does not select that DB. Keep this
            # target-review seam explicit, not a claim of full profile discovery.
            assert paths == (selector,)
            return case[1].target

        monkeypatch.setattr(service, "preview_backup", reviewed_target)

        class Harness(App):
            def on_mount(self):
                self.push_screen(BackupRestoreScreen(service, config_paths=(selector,)))

        app = Harness()
        try:
            async with app.run_test(size=(100, 36)) as pilot:
                await pilot.click("#backup-open-copies")
                async with asyncio.timeout(10):
                    while not app.screen.query(".backup-review-rollback"):
                        await asyncio.sleep(0.01)
                screen = app.screen
                screen.query_one(".backup-review-rollback", Button).focus()
                await pilot.press("enter")
                screen.query_one("#backup-copy-password", Input).value = "original"
                screen.query_one("#backup-later-review", Button).focus()
                await pilot.press("enter")
                async with asyncio.timeout(15):
                    while screen.query_one("#backup-later-start", Button).disabled:
                        await asyncio.sleep(0.02)
                assert "Restore:" in str(
                    screen.query_one("#backup-later-preview", Static).render()
                )
                assert screen.query_one("#backup-copy-password", Input).value == ""
                screen.query_one("#backup-later-start", Button).focus()
                await pilot.press("enter")
                assert service.current()["kind"] == "preview_rollback"
                screen.query_one("#backup-copy-password", Input).value = "original"
                screen.query_one("#backup-safety-password", Input).value = "new safety"
                screen.query_one("#backup-safety-confirm", Input).value = "new safety"
                screen.query_one("#backup-later-confirm", Checkbox).value = True
                await pilot.pause()
                screen.query_one("#backup-later-start", Button).focus()
                await pilot.press("enter")
                operation = service.current()["operation_id"]
                assert service.current()["kind"] == "later_rollback"
                assert all(
                    screen.query_one("#" + field, Input).value == ""
                    for field in (
                        "backup-copy-password",
                        "backup-safety-password",
                        "backup-safety-confirm",
                    )
                )
                result = await asyncio.to_thread(service.wait, operation, timeout=40)
                assert result["state"] == "succeeded", dict(result)
                assert selector.read_bytes() == prior
                copies = service.recovery_copies()
                assert len(copies) == 2 and all(
                    row.status == "verified" for row in copies
                )
        finally:
            await asyncio.to_thread(service.close)


@pytest.mark.asyncio
async def test_pending_abort_preserves_actual_pre_safety_originals(
    pre_safety_recovery, monkeypatch, tmp_path
):
    import asyncio

    from textual.app import App
    from textual.widgets import Button, Input

    from tldw_chatbook.Backup_Recovery import bootstrap
    from tldw_chatbook.UI.Screens.backup_restore_screen import BackupRestoreScreen

    service, pending, before, selector = pre_safety_recovery
    forwarded = []
    start = service.start_recovery

    def observed(*args, **kwargs):
        forwarded.append((kwargs["action"], kwargs.get("rollback_password")))
        return start(*args, **kwargs)

    monkeypatch.setattr(service, "start_recovery", observed)

    class Harness(App):
        def on_mount(self):
            self.push_screen(BackupRestoreScreen(service))

    app = Harness()
    async with app.run_test(size=(90, 30)) as pilot:
        await pilot.click("#backup-open-copies")
        async with asyncio.timeout(15):
            while not app.screen.query(".backup-recover-abort"):
                await asyncio.sleep(0.05)
        screen = app.screen
        button = screen.query_one(".backup-recover-abort", Button)
        assert str(button.label) == "Abort untouched replacement"
        screen.query_one(
            "#backup-copy-password", Input
        ).value = "unrelated-private-password"
        button.focus()
        await pilot.press("enter")
        async with asyncio.timeout(40):
            while service.current() is None or service.current()["state"] == "running":
                await asyncio.sleep(0.05)
        assert service.current()["result"]["aborted"]
        assert forwarded == [("abort", None)]
        assert screen.query_one("#backup-copy-password", Input).value == ""
        assert not service.pending_operations()
        assert all(path.read_bytes() == content for path, content in before.items())
        assert bootstrap.startup_permission(selector, tmp_path / "bootstrap")[0]
        assert service.status(pending)["result"]["aborted"]


@pytest.mark.asyncio
async def test_editing_source_requires_new_actual_archive_inspection(tmp_path):
    import asyncio

    from textual.app import App
    from textual.widgets import Button, Input, Static

    from Tests.Backup_Recovery.test_archive_reader import archive
    from tldw_chatbook.Backup_Recovery.recovery_service import RecoveryService
    from tldw_chatbook.UI.Screens.backup_restore_screen import BackupRestoreScreen

    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    source_a = archive(first, data=b"archive A")
    source_b = archive(second, data=b"different archive B bytes")
    service = RecoveryService(tmp_path / "control")

    class Harness(App):
        def on_mount(self):
            self.push_screen(BackupRestoreScreen(service))

    app = Harness()
    try:
        async with app.run_test(size=(90, 30)) as pilot:
            await pilot.click("#backup-open-inspect")
            screen = app.screen
            screen.query_one("#backup-source", Input).value = str(source_a)
            await pilot.click("#backup-inspect")
            async with asyncio.timeout(15):
                while screen._inspection_id is None:
                    await asyncio.sleep(0.05)
            first_id = screen._inspection_id
            screen.query_one("#backup-source", Input).value = str(source_b)
            await pilot.pause(0.5)
            assert screen._inspection_id is None
            assert screen._inspection_summary is None
            assert not screen.query_one("#backup-restore-form").display
            assert screen.query_one("#backup-start-restore", Button).disabled
            assert "Files:" not in str(
                screen.query_one("#backup-inspection-summary", Static).render()
            )
            assert service.inspection(
                first_id
            ).digest  # Accepted acquired bytes remain owned.
            await pilot.click("#backup-inspect")
            async with asyncio.timeout(15):
                while screen._inspection_id is None:
                    await asyncio.sleep(0.05)
            assert screen._inspection_id != first_id
            assert (
                service.inspection(screen._inspection_id).digest
                != service.inspection(first_id).digest
            )
            assert screen._inspection_summary["payload_bytes"] == len(
                b"different archive B bytes"
            )
    finally:
        await asyncio.to_thread(service.close)
