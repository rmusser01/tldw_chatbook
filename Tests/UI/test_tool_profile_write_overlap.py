"""New management actions cannot cancel observation of an admitted write."""

import threading

import pytest
from textual.screen import ModalScreen
from textual.widgets import Static

from Tests.private_profile import private_profile_test
from Tests.UI.test_settings_configuration_hub import StyledSettingsDestinationHarness
from Tests.UI.test_settings_tool_profiles import (
    ToolProfileListing,
    _build_test_app,
    _open_settings_category,
    _profile,
    _WorkflowService,
)
from Tests.UI.test_tool_profile_review_lifetime import _activate, _wait
from tldw_chatbook.Tool_Packs.contracts import ToolPackError
from tldw_chatbook.Tool_Packs.publication import ToolPackPublicationResult
from tldw_chatbook.UI.Screens.settings_screen import SettingsScreen
from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog
from tldw_chatbook.Widgets.enhanced_file_picker import (
    EnhancedFileOpen,
    EnhancedFileSave,
)
from tldw_chatbook.Widgets.Settings_Widgets.tool_pack_import_review import (
    ToolPackExportReviewModal,
    ToolPackImportOptionsModal,
    ToolPackImportReviewModal,
)
from tldw_chatbook.Widgets.Settings_Widgets.tool_profiles_panel import ToolProfilesPanel


@pytest.mark.parametrize(
    "operation,outcome,followup",
    [
        (operation, outcome, "repeat")
        for operation in ("import", "export", "remove")
        for outcome in ("success", "refused", "uncertain", "malformed")
    ]
    + [
        (operation, "success", followup)
        for operation in ("import", "export", "remove")
        for followup in ("other", "cancel_worker")
    ]
    + [
        (operation, outcome, "recreate")
        for operation in ("import", "export", "remove")
        for outcome in ("success", "refused", "uncertain", "malformed")
    ]
    + [
        (operation, "success", "recreate_after")
        for operation in ("import", "export", "remove")
    ],
)
@private_profile_test
async def test_repeat_action_keeps_admitted_write_and_truthful_outcome(
    request, tmp_path, monkeypatch, operation, outcome, followup
):
    entered, release, completed = (
        threading.Event(),
        threading.Event(),
        threading.Event(),
    )

    class Service(_WorkflowService):
        def finish(self, result):
            entered.set()
            assert release.wait(20)
            try:
                if outcome == "refused":
                    raise ToolPackError(
                        operation,
                        {
                            "import": "activation_failed",
                            "export": "publication_failed",
                            "remove": "in_use",
                        }[operation],
                    )
                if operation == "import":
                    self.listing = ToolProfileListing(
                        profiles=self.listing.profiles
                        + (
                            _profile(
                                "audit", origin="imported", binding_state="unbound"
                            ),
                        )
                    )
                elif operation == "remove":
                    self.listing = ToolProfileListing(profiles=())
                if outcome == "malformed":
                    from dataclasses import replace

                    return replace(
                        result,
                        **{
                            "import": {"installed": None},
                            "export": {"archive_sha256": None},
                            "remove": {"tombstone": None},
                        }[operation],
                    )
                if outcome == "uncertain":
                    if operation == "export":
                        return ToolPackPublicationResult("9" * 64, True, True)
                    raise ToolPackError(
                        operation,
                        "activation_uncertain"
                        if operation == "import"
                        else "outcome_uncertain",
                    )
                return result
            finally:
                completed.set()

        def import_unbound(self, review):
            return self.finish(super().import_unbound(review))

        def publish_export(self, review, destination, **context):
            return self.finish(super().publish_export(review, destination, **context))

        def remove_profile(self, profile_id, *, expected_revision):
            return self.finish(
                super().remove_profile(profile_id, expected_revision=expected_revision)
            )

    service = Service(
        ToolProfileListing(
            profiles=(_profile("research", origin="imported", binding_state="unbound"),)
        )
    )
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
        selector = (
            "#tool-profiles-import"
            if operation == "import"
            else f"#tool-profile-{operation}-0"
        )
        try:
            await _activate(pilot, selector)
            if operation == "import":
                await _wait(
                    pilot, lambda: isinstance(host.screen, ToolPackImportOptionsModal)
                )
                await _activate(pilot, "#tool-pack-import-options-review")
                await _wait(
                    pilot, lambda: isinstance(host.screen, ToolPackImportReviewModal)
                )
                await _activate(pilot, "#tool-pack-import-unbound")
            elif operation == "export":
                await _wait(
                    pilot, lambda: isinstance(host.screen, ToolPackExportReviewModal)
                )
                await _activate(pilot, "#tool-pack-export-continue")
            else:
                await _wait(pilot, lambda: isinstance(host.screen, ConfirmationDialog))
                await _activate(pilot, "#confirm-button")
            await _wait(pilot, entered.is_set)
            first = next(
                w for w in host.workers if w.group == f"settings-tool-pack-{operation}"
            )
            assert host.screen is settings
            if followup == "cancel_worker":
                first.cancel()
                await _wait(pilot, lambda: first.is_finished)
                assert not completed.is_set()
                assert app._tool_profile_operations.pending(operation) is not None
                release.set()
                await _wait(pilot, completed.is_set)
                await _wait(
                    pilot,
                    lambda: app._tool_profile_operations.pending(operation) is None,
                )
                return
            if followup == "recreate_after":
                release.set()
                await _wait(pilot, completed.is_set)
                await _wait(pilot, lambda: first.is_finished)
            if followup in {"recreate", "recreate_after"}:
                await host.pop_screen()
                await pilot.pause()
                assert not settings.is_attached
                assert first.is_cancelled == (followup == "recreate")
                settings = SettingsScreen(app)
                await host.push_screen(settings)
                await _open_settings_category(pilot, "#settings-category-tool-profiles")
                await host.workers.wait_for_complete()
                assert ("in progress" in settings._tool_profiles_result) == (
                    followup == "recreate"
                )
            if followup == "recreate_after":
                newer_focus = settings.query_one("#settings-category-search")
                newer_focus.focus()
                await pilot.pause()
            elif followup in {"repeat", "recreate"}:
                await _activate(pilot, selector)
                assert first.is_cancelled == (followup == "recreate")
                assert host.screen is settings
                assert "in progress" in settings._tool_profiles_result
                receipt = next(
                    w
                    for w in settings.query(Static)
                    if w.display and str(w.renderable) == settings._tool_profiles_result
                )
                region, clip = host.screen._compositor.visible_widgets[receipt]
                assert region.intersection(clip) == region
                newer_focus = settings.query_one("#settings-category-search")
                newer_focus.focus()
                await pilot.pause()
            else:
                replacement = {
                    "import": "export",
                    "export": "remove",
                    "remove": "import",
                }[operation]
                await _activate(
                    pilot,
                    "#tool-profiles-import"
                    if replacement == "import"
                    else f"#tool-profile-{replacement}-0",
                )
                await _wait(pilot, lambda: isinstance(host.screen, ModalScreen))
                newer_focus = host.focused
            release.set()
            await _wait(pilot, completed.is_set)
            await _wait(pilot, lambda: first.is_finished)
            await _wait(
                pilot,
                lambda: (
                    settings.query_one(ToolProfilesPanel).profile_ids
                    == tuple(p.profile_id for p in service.listing.profiles)
                ),
            )
            await _wait(
                pilot, lambda: "in progress" not in settings._tool_profiles_result
            )
            await pilot.pause()
            assert host.focused is newer_focus
            writes = [
                call
                for call in service.calls
                if call[0]
                == {"import": "import", "export": "publish", "remove": "remove"}[
                    operation
                ]
            ]
            assert len(writes) == 1
            result = settings._tool_profiles_result
            if outcome == "success":
                assert result.startswith(
                    {
                        "import": "Imported audit",
                        "export": "Exported Tool Pack",
                        "remove": "Removed research",
                    }[operation]
                )
            elif outcome in {"uncertain", "malformed"}:
                assert "uncertain" in result and "failed" not in result.casefold()
            else:
                assert {
                    "import": "activation_failed",
                    "export": "publication_failed",
                    "remove": "active run",
                }[operation] in result
            if operation != "export":
                assert settings.query_one(ToolProfilesPanel).profile_ids == tuple(
                    p.profile_id for p in service.listing.profiles
                )
            assert app._tool_profile_operations.pending(operation) is None
            if outcome == "refused":
                await _activate(pilot, selector)
                await _wait(pilot, lambda: isinstance(host.screen, ModalScreen))
        finally:
            release.set()
            if isinstance(host.screen, ModalScreen):
                await pilot.press("escape")
            await host.workers.wait_for_complete()
            owner = getattr(app, "_tool_profile_operations", None)
            if owner is not None:
                await owner.close_and_drain()


@private_profile_test
async def test_newer_progress_does_not_hide_a_completion_from_listing_refresh(request):
    from Tests.Tool_Packs.test_operations import result_for

    release = threading.Event()
    service = _WorkflowService(ToolProfileListing(profiles=(_profile("research"),)))
    app = _build_test_app()
    app.tool_pack_service = service
    host = StyledSettingsDestinationHarness(app, "settings")
    async with host.run_test(size=(80, 24)) as pilot:
        await _open_settings_category(pilot, "#settings-category-tool-profiles")
        await host.workers.wait_for_complete()
        settings = host.screen
        settings._subscription_readiness_timer.pause()
        owner = app._get_tool_profile_operations()
        newer_focus = settings.query_one("#settings-category-search")
        newer_focus.focus()
        await pilot.pause()

        def remove(cancelled):
            service.listing = ToolProfileListing(profiles=())
            return result_for("remove")

        def export(cancelled):
            assert release.wait(10)
            return result_for("export")

        try:
            await owner.start("remove", "research", remove)
            task = owner.start("export", "research", export)
            settings._poll_subscription_readiness()
            await host.workers.wait_for_complete()
            await pilot.pause()
            assert settings._tool_profiles_result.startswith("Export in progress")
            assert settings.query_one(ToolProfilesPanel).profile_ids == ()
            assert host.focused is newer_focus
            release.set()
            await task
            settings._poll_subscription_readiness()
            assert settings._tool_profiles_result.startswith("Exported Tool Pack")
        finally:
            release.set()
            await owner.close_and_drain()
