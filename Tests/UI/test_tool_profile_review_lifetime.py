"""Delayed Tool Profile review preparation belongs to its initiating visit."""

import asyncio
import threading

import pytest
from textual.widgets import Button

from Tests.private_profile import private_profile_test
from Tests.UI.test_settings_configuration_hub import StyledSettingsDestinationHarness
from Tests.UI.test_settings_tool_profiles import (
    ToolProfileListing,
    _build_test_app,
    _open_settings_category,
    _profile,
    _WorkflowService,
)
from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog
from tldw_chatbook.Widgets.enhanced_file_picker import EnhancedFileOpen
from tldw_chatbook.Widgets.Settings_Widgets.tool_pack_import_review import (
    ToolPackExportReviewModal,
    ToolPackImportOptionsModal,
    ToolPackImportReviewModal,
)


async def _wait(pilot, predicate):
    async with asyncio.timeout(5):
        while not predicate():
            await pilot.pause(0.025)
    await pilot.pause()


async def _activate(pilot, selector):
    button = pilot.app.screen.query_one(selector, Button)
    button.focus()
    button.scroll_visible(animate=False, immediate=True)
    await pilot.pause()
    await pilot.press("enter")


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["import", "export"])
@pytest.mark.parametrize(
    "departure", ["category", "return", "modal", "remove", "newer", "same"]
)
@private_profile_test
async def test_delayed_review_cannot_outlive_its_initiating_visit(
    request, tmp_path, monkeypatch, operation, departure
):
    entered, release = threading.Event(), threading.Event()

    class Service(_WorkflowService):
        def inspect_import(self, *args, **kwargs):
            if operation == "import" and not entered.is_set():
                entered.set()
                assert release.wait(20)
            return super().inspect_import(*args, **kwargs)

        def capture_export(self, *args, **kwargs):
            if operation == "export" and not entered.is_set():
                entered.set()
                assert release.wait(20)
            return super().capture_export(*args, **kwargs)

    service = Service(ToolProfileListing(profiles=(_profile("research"),)))
    app = _build_test_app()
    app.tool_pack_service = service
    host = StyledSettingsDestinationHarness(app, "settings")
    push = host.push_screen_wait

    async def choose_archive(screen):
        # Only file selection is supplied; options/review/navigation are real.
        if isinstance(screen, EnhancedFileOpen):
            return tmp_path / "audit.tldw-tool-pack"
        return await push(screen)

    monkeypatch.setattr(host, "push_screen_wait", choose_archive)
    async with host.run_test(size=(80, 24)) as pilot:
        await _open_settings_category(pilot, "#settings-category-tool-profiles")
        await host.workers.wait_for_complete()
        settings = host.screen
        try:
            await _activate(
                pilot,
                "#tool-profiles-import"
                if operation == "import"
                else "#tool-profile-export-0",
            )
            if operation == "import":
                await _wait(
                    pilot, lambda: isinstance(host.screen, ToolPackImportOptionsModal)
                )
                await _activate(pilot, "#tool-pack-import-options-review")
            await _wait(pilot, entered.is_set)
            worker = next(
                w for w in host.workers if w.group == f"settings-tool-pack-{operation}"
            )
            assert host.screen is settings
            if departure in {"category", "return"}:
                await _activate(pilot, "#settings-category-theme")
                await _wait(
                    pilot,
                    lambda: (
                        settings.active_category == "theme"
                        and not settings._category_pane_swap_pending
                    ),
                )
                if departure == "return":
                    await _activate(pilot, "#settings-category-tool-profiles")
                    await _wait(
                        pilot,
                        lambda: (
                            settings.active_category == "tool-profiles"
                            and not settings._category_pane_swap_pending
                        ),
                    )
            elif departure == "modal":
                await host.push_screen(
                    ConfirmationDialog(title="Unrelated", message="Another review")
                )
                await pilot.pause()
                await pilot.press("escape")
                await _wait(pilot, lambda: host.screen is settings)
            elif departure == "remove":
                await host.pop_screen()
                await pilot.pause()
            else:
                replacement = (
                    operation
                    if departure == "same"
                    else ("export" if operation == "import" else "import")
                )
                await _activate(
                    pilot,
                    "#tool-profile-export-0"
                    if replacement == "export"
                    else "#tool-profiles-import",
                )
                expected = (
                    ToolPackExportReviewModal
                    if replacement == "export"
                    else ToolPackImportOptionsModal
                )
                await _wait(pilot, lambda: isinstance(host.screen, expected))
                await pilot.press("escape")
                await _wait(pilot, lambda: host.screen is settings)
            expected_screen, expected_focus = host.screen, host.focused
            release.set()
            await _wait(
                pilot,
                lambda: (
                    worker.is_finished
                    or isinstance(
                        host.screen,
                        (ToolPackImportReviewModal, ToolPackExportReviewModal),
                    )
                ),
            )
            assert host.screen is expected_screen
            assert host.focused is expected_focus
            assert not any(call[0] in {"import", "publish"} for call in service.calls)
        finally:
            release.set()
            if isinstance(
                host.screen,
                (
                    ToolPackImportReviewModal,
                    ToolPackExportReviewModal,
                    ToolPackImportOptionsModal,
                ),
            ):
                await pilot.press("escape")
            await host.workers.wait_for_complete()


@pytest.mark.asyncio
@pytest.mark.parametrize("decision", ["cancel", "accept", "revise"])
@private_profile_test
async def test_owned_import_dialogs_preserve_current_review_flow(
    request, tmp_path, monkeypatch, decision
):
    service = _WorkflowService(ToolProfileListing(profiles=(_profile("research"),)))
    app = _build_test_app()
    app.tool_pack_service = service
    host = StyledSettingsDestinationHarness(app, "settings")
    push = host.push_screen_wait

    async def choose_archive(screen):
        if isinstance(screen, EnhancedFileOpen):
            return tmp_path / "audit.tldw-tool-pack"
        return await push(screen)

    monkeypatch.setattr(host, "push_screen_wait", choose_archive)
    async with host.run_test(size=(80, 24)) as pilot:
        await _open_settings_category(pilot, "#settings-category-tool-profiles")
        await host.workers.wait_for_complete()
        settings = host.screen
        await _activate(pilot, "#tool-profiles-import")
        await _wait(pilot, lambda: isinstance(host.screen, ToolPackImportOptionsModal))
        await _activate(pilot, "#tool-pack-import-options-review")
        await _wait(pilot, lambda: isinstance(host.screen, ToolPackImportReviewModal))
        if decision == "revise":
            await _activate(pilot, "#tool-pack-import-revise")
            await _wait(
                pilot, lambda: isinstance(host.screen, ToolPackImportOptionsModal)
            )
            await _activate(pilot, "#tool-pack-import-options-review")
            await _wait(
                pilot, lambda: isinstance(host.screen, ToolPackImportReviewModal)
            )
        await _activate(
            pilot,
            "#tool-pack-import-cancel"
            if decision == "cancel"
            else "#tool-pack-import-unbound",
        )
        await host.workers.wait_for_complete()
        assert host.screen is settings
        calls = [call[0] for call in service.calls]
        if decision == "cancel":
            assert calls == ["inspect"]
            assert settings._tool_profiles_result == "Import cancelled"
        else:
            assert calls == (
                ["inspect", "inspect", "import"]
                if decision == "revise"
                else ["inspect", "import"]
            )
            assert (
                settings._tool_profiles_result == "Imported audit unbound · revision 1"
            )


@pytest.mark.asyncio
@pytest.mark.parametrize("valid", [False, True])
@private_profile_test
async def test_delayed_destination_capture_cannot_publish_after_departure(
    request, tmp_path, monkeypatch, valid
):
    from tldw_chatbook.Tool_Packs.publication import CapturedToolPackDestination
    from tldw_chatbook.Widgets.enhanced_file_picker import EnhancedFileSave

    entered, release = threading.Event(), threading.Event()
    capture = CapturedToolPackDestination.capture

    def held_capture(path):
        entered.set()
        assert release.wait(20)
        return capture(path)

    monkeypatch.setattr(CapturedToolPackDestination, "capture", held_capture)
    service = _WorkflowService(ToolProfileListing(profiles=(_profile("research"),)))
    app = _build_test_app()
    app.tool_pack_service = service
    host = StyledSettingsDestinationHarness(app, "settings")
    push = host.push_screen_wait

    async def choose_destination(screen):
        if isinstance(screen, EnhancedFileSave):
            return tmp_path / ("safe.tldw-tool-pack" if valid else "invalid.txt")
        return await push(screen)

    monkeypatch.setattr(host, "push_screen_wait", choose_destination)
    async with host.run_test(size=(80, 24)) as pilot:
        await _open_settings_category(pilot, "#settings-category-tool-profiles")
        await host.workers.wait_for_complete()
        settings = host.screen
        try:
            await _activate(pilot, "#tool-profile-export-0")
            await _wait(
                pilot, lambda: isinstance(host.screen, ToolPackExportReviewModal)
            )
            await _activate(pilot, "#tool-pack-export-continue")
            await _wait(pilot, entered.is_set)
            await _activate(pilot, "#settings-category-theme")
            await _wait(
                pilot,
                lambda: (
                    settings.active_category == "theme"
                    and not settings._category_pane_swap_pending
                ),
            )
            release.set()
            await host.workers.wait_for_complete()
            assert host.screen is settings
            assert [call[0] for call in service.calls] == ["capture"]
            assert settings._tool_profiles_result == ""
            assert list(tmp_path.glob("*.tldw-tool-pack")) == []
        finally:
            release.set()


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["import", "export"])
@pytest.mark.parametrize("phase", ["preparation", "mutation", "uncertain"])
@private_profile_test
async def test_departure_suppresses_preparation_errors_but_keeps_admitted_outcomes(
    request, tmp_path, monkeypatch, operation, phase
):
    from tldw_chatbook.Tool_Packs.contracts import ToolPackError
    from tldw_chatbook.Widgets.enhanced_file_picker import EnhancedFileSave

    entered, release = threading.Event(), threading.Event()
    category = (
        ("archive_invalid" if operation == "import" else "profile_invalid")
        if phase == "preparation"
        else ("activation_failed" if operation == "import" else "publication_failed")
        if phase == "mutation"
        else (
            "activation_uncertain" if operation == "import" else "durability_uncertain"
        )
    )

    def fail_after_release(*args, **kwargs):
        entered.set()
        assert release.wait(20)
        raise ToolPackError(operation, category)

    service = _WorkflowService(ToolProfileListing(profiles=(_profile("research"),)))
    method = (
        ("inspect_import" if operation == "import" else "capture_export")
        if phase == "preparation"
        else ("import_unbound" if operation == "import" else "publish_export")
    )
    monkeypatch.setattr(service, method, fail_after_release)
    app = _build_test_app()
    app.tool_pack_service = service
    host = StyledSettingsDestinationHarness(app, "settings")
    push = host.push_screen_wait

    async def choose_file(screen):
        if isinstance(screen, (EnhancedFileOpen, EnhancedFileSave)):
            return tmp_path / "audit.tldw-tool-pack"
        return await push(screen)

    monkeypatch.setattr(host, "push_screen_wait", choose_file)
    async with host.run_test(size=(80, 24)) as pilot:
        await _open_settings_category(pilot, "#settings-category-tool-profiles")
        await host.workers.wait_for_complete()
        settings = host.screen
        settings._set_tool_profiles_result("Previous outcome")
        try:
            await _activate(
                pilot,
                "#tool-profiles-import"
                if operation == "import"
                else "#tool-profile-export-0",
            )
            if operation == "import":
                await _wait(
                    pilot, lambda: isinstance(host.screen, ToolPackImportOptionsModal)
                )
                await _activate(pilot, "#tool-pack-import-options-review")
            if phase != "preparation":
                review_type = (
                    ToolPackImportReviewModal
                    if operation == "import"
                    else ToolPackExportReviewModal
                )
                await _wait(pilot, lambda: isinstance(host.screen, review_type))
                await _activate(
                    pilot,
                    "#tool-pack-import-unbound"
                    if operation == "import"
                    else "#tool-pack-export-continue",
                )
            await _wait(pilot, entered.is_set)
            await _activate(pilot, "#settings-category-theme")
            await _wait(
                pilot,
                lambda: (
                    settings.active_category == "theme"
                    and not settings._category_pane_swap_pending
                ),
            )
            await _activate(pilot, "#settings-category-tool-profiles")
            await _wait(
                pilot,
                lambda: (
                    settings.active_category == "tool-profiles"
                    and not settings._category_pane_swap_pending
                ),
            )
            release.set()
            await host.workers.wait_for_complete()
            assert host.screen is settings
            assert settings._tool_profiles_result == (
                "Previous outcome"
                if phase == "preparation"
                else (
                    "Import outcome uncertain. Check the current profile state before retrying."
                    if operation == "import"
                    else "Export outcome uncertain. Check the chosen destination before retrying."
                )
                if phase == "uncertain"
                else f"{operation.title()} failed · {category}"
            )
        finally:
            release.set()
