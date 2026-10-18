"""Management review uses mounted actions and the actual plugin owner."""

import pytest
from textual.widgets import Button, OptionList

from Tests.Plugins.conftest import native_console as _native_console
from Tests.Plugins.conftest import native_package as _native_package
from Tests.private_profile import private_profile_test
from Tests.UI.test_plugins_browse import (
    BrowseApp,
    DelayedPluginCall,
    HeldPluginUnmount,
    settled,
    wait_for_plugin_prune,
)

native_console = _native_console
native_package = _native_package


@private_profile_test
@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(80, 24), (120, 35)])
async def test_review_growth_keeps_focused_action_painted(
    request, native_console, native_package, size
):
    from Tests.UI.test_plugins_browse import paint
    from tldw_chatbook.Widgets.plugin_review_modal import PluginReviewModal

    rig = native_console
    package = native_package()
    original = (package / "skills/review/SKILL.md").read_text()
    for index in range(12):
        directory = package / "skills" / f"review-{index}"
        directory.mkdir()
        (directory / "SKILL.md").write_text(
            original.replace("name: review", f"name: review-{index}")
        )
    from tldw_chatbook.Plugins.git_source import select_local_source

    operation = "focus-growth-acquisition"
    destination = await rig.service.acquisition_destination(operation)
    producer = await rig.service.acquire_snapshot(
        select_local_source(package), operation, destination
    )
    app = BrowseApp(rig)
    async with app.run_test(size=size) as pilot:
        await settled(pilot, app.screen)
        await app.push_screen(
            PluginReviewModal(
                app.plugin_source_service, producer=producer, workspace_id="workspace-a"
            )
        )
        modal = app.screen
        await modal.workers.wait_for_complete()
        await pilot.pause()
        button = modal.query_one("#plugin-review-prepare", Button)
        button.focus()
        await pilot.pause()
        assert app.focused is button and "Review action" in paint(modal)
        await pilot.press("enter")
        await pilot.pause()
        assert modal.draft.review is not None
        assert app.focused is button
        assert button in modal._compositor.visible_widgets
        assert (
            button.region.intersection(
                modal.query_one("#plugin-review-shell").content_region
            )
            == button.region
        )
        assert "Review action" in paint(modal)


@private_profile_test
@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(80, 24), (120, 35)])
async def test_refresh_growth_keeps_focused_action_and_cleaned_status_painted(
    request, native_console, size
):
    from Tests.UI.test_plugins_browse import paint
    from tldw_chatbook.Widgets.plugin_review_modal import PluginReviewModal

    rig = native_console
    installed = await rig.install(enable=False)
    for _ in range(12):
        creation = await rig.service.review_data_creation(installed.installation_id)
        await rig.service.create_data(creation, creation.operation_id)
    rows = await rig.service.data_status(installation_id=installed.installation_id)
    roots = tuple(row["root"] for row in rows)
    deletion = await rig.service.review_data_deletion(roots)
    assert (await rig.service.delete_data(roots, deletion.operation_id)).committed
    cleaned = await rig.service.data_status(installation_id=installed.installation_id)
    assert len(cleaned) == 12 and all(row["cleaned_absent"] for row in cleaned)
    app = BrowseApp(rig)
    async with app.run_test(size=size) as pilot:
        await settled(pilot, app.screen)
        await app.push_screen(
            PluginReviewModal(
                app.plugin_source_service,
                installation_id=installed.installation_id,
                workspace_id="workspace-a",
            )
        )
        modal = app.screen
        await modal.workers.wait_for_complete()
        await pilot.pause()
        button = modal.query_one("#plugin-review-refresh", Button)
        button.focus()
        await pilot.pause()
        assert app.focused is button and "Refresh status" in paint(modal)
        await pilot.press("enter")
        await pilot.pause()
        assert app.focused is button
        assert button in modal._compositor.visible_widgets
        assert (
            button.region.intersection(
                modal.query_one("#plugin-review-shell").content_region
            )
            == button.region
        )
        painted = paint(modal)
        assert "Refresh status" in painted
        assert cleaned[-1]["root"].root_id in painted
        assert "Cleaned — saved data removed; users: ()" in painted


@private_profile_test
@pytest.mark.asyncio
async def test_mounted_management_opens_exact_review_without_committing(
    request, native_console
):
    rig = native_console
    installed = await rig.install(enable=False)
    rig.registry.set_active_workspace("workspace-a")
    app = BrowseApp(rig)
    async with app.run_test(size=(80, 24)) as pilot:
        screen = app.screen
        await settled(pilot, screen)
        listing = screen.query_one("#plugins-list", OptionList)
        assert listing.option_count == 1
        listing.focus()
        await pilot.press("down", "enter")
        await pilot.pause()
        assert screen.browse.selected_installation_id == installed.installation_id
        button = screen.query_one("#plugins-manage", Button)
        button.focus()
        await pilot.press("enter")
        await pilot.pause()
        assert app.screen is not screen
        assert app.screen.installation_id == installed.installation_id
        await pilot.press("escape")
        await pilot.pause()
        assert app.screen is screen
        rows = await rig.service.management_snapshot("workspace-a")
        assert not rows[0].effective_active


@pytest.mark.asyncio
async def test_install_intent_is_workspace_scoped_and_does_not_grant_trust(
    native_console, native_package
):
    service = native_console.service
    review = await service.review_install(
        native_package(),
        selection=("skill:review",),
        workspace_id="workspace-a",
        enable_in_workspace=True,
    )
    assert review.intent == "enabled"
    assert (await service.commit(review, review.operation_id)).committed
    a = (await service.management_snapshot("workspace-a"))[0]
    b = (await service.management_snapshot("workspace-b"))[0]
    assert a.effective_active and not a.trusted
    assert not b.effective_active and not a.activation_default


def test_persistence_failure_is_not_a_durable_disable():
    from tldw_chatbook.UI.Plugins_Modules.operations import operation_label

    assert (
        "not saved"
        in operation_label(committed=False, stopped=True, cleanup_pending=False).lower()
    )
    assert (
        "pending"
        in operation_label(committed=True, stopped=False, cleanup_pending=True).lower()
    )


@private_profile_test
@pytest.mark.asyncio
async def test_mounted_review_is_invalidated_by_changed_selection(
    request, native_console
):
    from textual.widgets import Select, SelectionList

    from tldw_chatbook.Widgets.plugin_review_modal import PluginReviewModal

    rig = native_console
    installed = await rig.install(enable=False)
    app = BrowseApp(rig)
    async with app.run_test(size=(120, 35)) as pilot:
        await settled(pilot, app.screen)
        await app.push_screen(
            PluginReviewModal(
                app.plugin_source_service,
                installation_id=installed.installation_id,
                workspace_id="workspace-a",
            )
        )
        await pilot.pause()
        modal = app.screen
        await modal.workers.wait_for_complete()
        await pilot.pause()
        modal.query_one("#plugin-review-action", Select).value = "select"
        await pilot.pause()
        modal.query_one("#plugin-review-prepare", Button).focus()
        await pilot.press("enter")
        await pilot.pause()
        assert modal.draft.review is not None
        assert modal.draft.review.kind == "select"
        original = modal.draft.review
        modal.query_one(SelectionList).deselect_all()
        await pilot.pause()
        assert modal.draft.review is None
        assert modal.query_one("#plugin-review-confirm", Button).disabled
        assert not (await rig.service.lookup_operation(original.operation_id)).committed


@private_profile_test
@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(80, 24), (120, 35)])
async def test_actual_first_use_setup_through_masked_dialog(request, tmp_path, size):
    from types import SimpleNamespace

    from textual.widgets import Input

    from tldw_chatbook.Plugins.authority_store import FilePluginMarkerStore
    from tldw_chatbook.Plugins.service import PluginService
    from tldw_chatbook.Widgets.plugin_review_modal import PluginAuthorityDialog

    class IsolatedQualifiedMarkerDouble(FilePluginMarkerStore):
        # Qualification seam only: this does NOT exercise or qualify OS keychain.
        reduced_protection = False

    service = PluginService(
        tmp_path / "fresh-profile",
        workspace_lookup=lambda _: None,
        marker_store_factory=lambda _: IsolatedQualifiedMarkerDouble(
            tmp_path / "protected-marker-double"
        ),
    )
    app = BrowseApp(SimpleNamespace(service=service, registry=None))
    try:
        assert await service.authority_status() == "needs_setup"
        async with app.run_test(size=size) as pilot:
            await settled(pilot, app.screen)
            app.screen.query_one("#plugins-authority", Button).focus()
            await pilot.press("enter")
            await pilot.pause()
            assert isinstance(app.screen, PluginAuthorityDialog)
            dialog = app.screen
            password = dialog.query_one("#password-input", Input)
            confirm = dialog.query_one("#confirm-input", Input)
            assert password.password and confirm.password
            password.value = confirm.value = "private test passphrase"
            dialog.query_one("#submit-button", Button).focus()
            await pilot.press("enter")
            for _ in range(100):
                await pilot.pause(0.02)
                if app.screen is not dialog:
                    break
            assert app.screen is not dialog
            assert await service.authority_status() == "ready"
            assert password.value == confirm.value == ""
            await settled(pilot, app.screen)
            assert (
                "unavailable"
                not in str(app.screen.query_one("#plugins-status").render()).lower()
            )
    finally:
        await service.aclose()


@private_profile_test
@pytest.mark.asyncio
async def test_close_reopen_preserves_synthetic_data_and_focus(request, native_console):
    from textual.widgets import Select, TextArea

    from tldw_chatbook.Widgets.plugin_review_modal import PluginReviewModal

    rig = native_console
    installed = await rig.install(enable=False)
    app = BrowseApp(rig)
    async with app.run_test(size=(80, 24)) as pilot:
        await settled(pilot, app.screen)
        original = app.screen
        await app.push_screen(
            PluginReviewModal(
                app.plugin_source_service,
                workspace_id="workspace-a",
                installation_id=installed.installation_id,
            )
        )
        await pilot.pause()
        modal = app.screen
        await modal.workers.wait_for_complete()
        await pilot.pause()
        modal.query_one("#plugin-review-action", Select).value = "test-hook"
        await pilot.pause()
        field = modal.query_one("#plugin-review-test-data", TextArea)
        field.text = '{"prompt":"retained synthetic input"}'
        field.focus()
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
        assert app.screen is original
        await app.push_screen(
            PluginReviewModal(
                app.plugin_source_service,
                workspace_id="workspace-a",
                installation_id=installed.installation_id,
            )
        )
        await pilot.pause()
        field = app.screen.query_one("#plugin-review-test-data", TextArea)
        assert field.text == '{"prompt":"retained synthetic input"}'
        assert app.focused is field


@private_profile_test
@pytest.mark.asyncio
async def test_unlock_wrong_password_stays_in_masked_flow_then_correct_unlock(
    request, tmp_path
):
    from types import SimpleNamespace

    from textual.widgets import Input

    from tldw_chatbook.Plugins.authority_store import FilePluginMarkerStore
    from tldw_chatbook.Plugins.service import PluginService
    from tldw_chatbook.Widgets.plugin_review_modal import PluginAuthorityDialog

    def service():
        return PluginService(
            tmp_path / "authority-profile",
            workspace_lookup=lambda _: None,
            marker_store_factory=lambda _: FilePluginMarkerStore(tmp_path / "marker"),
            accept_reduced_protection=True,
        )

    first = service()
    await first.bootstrap("correct private passphrase")
    await first.aclose()
    owner = service()
    try:
        assert await owner.authority_status() == "locked"
        app = BrowseApp(SimpleNamespace(service=owner, registry=None))
        async with app.run_test(size=(80, 24)) as pilot:
            await settled(pilot, app.screen)
            await app.push_screen(PluginAuthorityDialog(owner, "unlock"))
            await pilot.pause()
            dialog = app.screen
            field = dialog.query_one("#password-input", Input)
            assert app.focused is field
            field.value = "incorrect passphrase"
            dialog.query_one("#submit-button", Button).focus()
            await pilot.press("enter")
            await pilot.pause()
            assert app.screen is dialog and field.value == "" and field.password
            assert await owner.authority_status() == "locked"
            assert dialog.query_one("#error-message").has_class("visible")
            field.value = "correct private passphrase"
            retry = dialog.query_one("#submit-button", Button)
            # Textual ignores repeated Enter during the previous press effect.
            for _ in range(100):
                await pilot.pause(0.02)
                if not retry.has_class("-active") and not dialog._submitting:
                    break
            assert not retry.has_class("-active") and not dialog._submitting
            retry.focus()
            await pilot.press("enter")
            for _ in range(100):
                await pilot.pause(0.02)
                if app.screen is not dialog:
                    break
            assert app.screen is not dialog and field.value == ""
            assert await owner.authority_status() == "ready_reduced"
    finally:
        await owner.aclose()


@pytest.mark.asyncio
async def test_authority_reset_requires_exact_current_review(native_console):
    from dataclasses import replace

    service = native_console.service
    installed = await native_console.install(enable=False)
    review = await service.review_authority_reset()
    with pytest.raises(ValueError, match="stale"):
        await service.reset_authority(replace(review, namespace_id="0" * 64))
    activate = await service.review_activation(
        installed.installation_id, workspace_id="workspace-a", intent="enabled"
    )
    await service.commit(activate, activate.operation_id)
    with pytest.raises(ValueError, match="stale"):
        await service.reset_authority(review)
    current = await service.review_authority_reset()
    assert await service.reset_authority(current)
    assert await service.authority_status() == "needs_setup"


@private_profile_test
@pytest.mark.asyncio
async def test_status_refresh_sees_new_retained_root(request, native_console):
    from textual.widgets import Static

    from tldw_chatbook.Widgets.plugin_review_modal import PluginReviewModal

    rig = native_console
    installed = await rig.install(enable=False)
    app = BrowseApp(rig)
    async with app.run_test(size=(80, 24)) as pilot:
        await settled(pilot, app.screen)
        await app.push_screen(
            PluginReviewModal(
                app.plugin_source_service,
                workspace_id="workspace-a",
                installation_id=installed.installation_id,
            )
        )
        await app.screen.workers.wait_for_complete()
        modal = app.screen
        assert modal.details["roots"] == ()
        review = await rig.service.review_data_creation(installed.installation_id)
        root = await rig.service.create_data(review, review.operation_id)
        await modal._refresh()
        assert root.root_id in str(
            modal.query_one("#plugin-review-body", Static).render()
        )


@private_profile_test
@pytest.mark.asyncio
async def test_mounted_exact_reviews_cover_installed_lifecycle(
    request, native_console, native_package
):
    import asyncio

    from textual.widgets import Input, Select, SelectionList

    from tldw_chatbook.Plugins.git_source import select_local_source
    from tldw_chatbook.Widgets.plugin_review_modal import PluginReviewModal

    rig = native_console
    package = native_package()
    operation = "mounted-lifecycle-acquisition"
    destination = await rig.service.acquisition_destination(operation)
    acquired = await rig.service.acquire_snapshot(
        select_local_source(package), operation, destination
    )
    app = BrowseApp(rig)
    async with app.run_test(size=(120, 35)) as pilot:
        await settled(pilot, app.screen)
        await app.push_screen(
            PluginReviewModal(
                app.plugin_source_service, producer=acquired, workspace_id="workspace-a"
            )
        )
        modal = app.screen
        for _ in range(100):
            await pilot.pause(0.02)
            if not modal._loading:
                break
        assert not modal._loading
        modal.query_one("#plugin-review-selection", SelectionList).select(
            "skill:review"
        )

        async def confirm(action, reference=""):
            nonlocal modal
            modal.query_one("#plugin-review-action", Select).value = action
            modal.query_one("#plugin-review-reference", Input).value = reference
            await pilot.pause()
            modal.query_one("#plugin-review-prepare", Button).focus()
            await pilot.press("enter")
            await pilot.pause()
            assert modal.draft.review is not None, str(
                modal.query_one("#plugin-review-status").render()
            )
            review = modal.draft.review
            if action == "install":
                dialect = modal.query_one("#plugin-review-dialect", Select)
                dialect.value = "portable"
                await pilot.pause()
                assert modal.draft.review is None
                assert modal.query_one("#plugin-review-confirm", Button).disabled
                for _ in range(100):
                    await pilot.pause(0.02)
                    if not modal._loading:
                        break
                modal.query_one("#plugin-review-prepare", Button).focus()
                await pilot.press("enter")
                await pilot.pause()
                assert modal.draft.review is not None
                assert modal.draft.review is not review
                review = modal.draft.review
                # An obsolete queued event cannot reinterpret this newer review.
                dialect.post_message(Select.Changed(dialect, "auto"))
                await pilot.pause()
                assert modal.draft.review is review
                assert not modal.query_one("#plugin-review-reference", Input).display
                modal.query_one("#plugin-review-action", Select).focus()
                await pilot.press("escape")
                await app.push_screen(
                    PluginReviewModal(
                        app.plugin_source_service,
                        producer=acquired,
                        workspace_id="workspace-a",
                    )
                )
                modal = app.screen
                for _ in range(100):
                    await pilot.pause(0.02)
                    if not modal._loading:
                        break
                assert modal.draft.review is review
                assert modal.draft.selection == ("skill:review",)
                assert app.focused.id == "plugin-review-action"
            modal.query_one("#plugin-review-confirm", Button).focus()
            await pilot.press("enter")
            await pilot.pause()
            job = modal.operations.jobs[modal._job_id]
            assert job.review is review
            await asyncio.wait_for(asyncio.shield(job.task), 10)
            assert job.receipt.committed, job.receipt
            return review

        installed = await confirm("install")
        modal.query_one("#plugin-review-refresh", Button).focus()
        await pilot.press("enter")
        await pilot.pause()
        assert "install: Saved" in str(modal.query_one("#plugin-review-body").render())
        identity = installed.installation_id
        original_revision = installed.inspection.effective_digest
        assert not (await rig.service.management_snapshot("workspace-a"))[0].trusted
        await pilot.press("escape")
        await app.push_screen(
            PluginReviewModal(
                app.plugin_source_service,
                installation_id=identity,
                workspace_id="workspace-a",
            )
        )
        modal = app.screen
        for _ in range(100):
            await pilot.pause(0.02)
            if not modal._loading:
                break
        assert not modal._loading
        for action in (
            "trust",
            "enable",
            "global-enable",
            "global-disable",
            "inherit",
            "disable",
            "select",
        ):
            await confirm(action)
        skill = package / "skills/review/SKILL.md"
        skill.write_text(skill.read_text() + "\nUpdated local fixture.\n")
        updated = await confirm("update")
        assert updated.inspection.effective_digest != original_revision
        await confirm("rollback", original_revision)
        await confirm("create-data")
        await confirm("delete-data")
        await confirm("disable-all")
        await confirm("uninstall")
        rows = await rig.service.management_snapshot("workspace-a")
        assert not any(row.installation_id == identity for row in rows)


@private_profile_test
@pytest.mark.asyncio
@pytest.mark.parametrize("reload_action", ["refresh", "next", "failed-refresh"])
async def test_candidate_reload_requires_fresh_discard_review(
    request, native_console, native_package, monkeypatch, reload_action
):
    from pathlib import Path

    from tldw_chatbook.Plugins.git_source import select_local_source
    from tldw_chatbook.Widgets.plugin_review_modal import PluginStatusModal

    rig = native_console
    package = native_package()
    snapshots = {}
    for index in range(51 if reload_action == "next" else 2):
        identity = f"discard-reload-{index:02d}"
        snapshots[identity] = await rig.service.acquire_snapshot(
            select_local_source(package),
            identity,
            await rig.service.acquisition_destination(identity),
        )

    app = BrowseApp(rig)
    async with app.run_test(size=(120, 35)) as pilot:
        await settled(pilot, app.screen)
        sources = app.plugin_source_service
        await app.push_screen(PluginStatusModal(sources, candidates=True))
        modal = app.screen
        await pilot.pause()

        async def press(identity):
            button = modal.query_one(identity, Button)
            for _ in range(100):
                await pilot.pause(0.02)
                if not button.disabled and not button.has_class("-active"):
                    break
            assert not button.disabled
            button.focus()
            button.scroll_visible(animate=False, immediate=True)
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()

        modal.query_one(OptionList).highlighted = 0
        original_id = modal.rows[0]["operation_id"]
        await press("#plugin-status-review")
        assert modal.review is not None
        confirm = modal.query_one("#plugin-status-confirm", Button)
        assert not confirm.disabled

        original_listing = sources.list_persisted_candidates
        if reload_action == "failed-refresh":

            async def unavailable(**kwargs):
                raise OSError("candidate inventory temporarily unavailable")

            monkeypatch.setattr(sources, "list_persisted_candidates", unavailable)

        await press(
            "#plugin-status-next"
            if reload_action == "next"
            else "#plugin-status-refresh"
        )
        assert confirm.disabled, "reloading must invalidate the visible confirmation"
        assert modal.review is None

        if reload_action == "failed-refresh":
            monkeypatch.setattr(sources, "list_persisted_candidates", original_listing)
            await press("#plugin-status-refresh")

        before = await original_listing(limit=50)
        assert (
            next(row for row in before if row["operation_id"] == original_id)["reason"]
            is None
        )
        selected = next(
            index
            for index, row in enumerate(modal.rows)
            if row["operation_id"] != original_id
        )
        modal.query_one(OptionList).highlighted = selected
        fresh_id = modal.rows[selected]["operation_id"]
        await press("#plugin-status-review")
        await press("#plugin-status-confirm")
        after = await original_listing(offset=modal.page * 50, limit=50)
        assert (
            next(row for row in after if row["operation_id"] == fresh_id)["reason"]
            == "reference_already_released"
        )
        assert all(
            Path(row.materialized_identity).is_dir() for row in snapshots.values()
        )


@private_profile_test
@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(80, 24), (120, 35)])
async def test_uninstall_retained_data_status_and_history_cleanup(
    request, native_console, size
):
    import asyncio

    from textual.widgets import Input, Select

    from tldw_chatbook.Widgets.plugin_review_modal import (
        PluginReviewModal,
        PluginStatusModal,
    )

    rig = native_console
    installed = await rig.install(enable=False)
    creation = await rig.service.review_data_creation(installed.installation_id)
    root = await rig.service.create_data(creation, creation.operation_id)
    app = BrowseApp(rig)
    async with app.run_test(size=size) as pilot:
        await settled(pilot, app.screen)

        async def ready():
            for _ in range(100):
                await pilot.pause(0.02)
                if not app.screen._loading:
                    return app.screen
            raise AssertionError("manager did not load")

        async def press(modal, identity):
            button = modal.query_one(identity, Button)
            for _ in range(100):
                await pilot.pause(0.02)
                if not button.disabled and not button.has_class("-active"):
                    break
            assert not button.disabled
            button.focus()
            await pilot.press("enter")
            await pilot.pause()

        await app.push_screen(
            PluginReviewModal(
                app.plugin_source_service,
                installation_id=installed.installation_id,
                workspace_id="workspace-a",
            )
        )
        modal = await ready()
        modal.query_one("#plugin-review-action", Select).value = "uninstall"
        await pilot.pause()
        await press(modal, "#plugin-review-prepare")
        original = modal.draft.review
        await press(modal, "#plugin-review-confirm")
        job = modal.operations.jobs[modal._job_id]
        await asyncio.wait_for(asyncio.shield(job.task), 10)
        assert job.receipt.committed and root.path.exists()
        await press(modal, "#plugin-review-refresh")
        assert "uninstall: Saved" in str(
            modal.query_one("#plugin-review-body").render()
        )
        assert "installation_unavailable" not in str(
            modal.query_one("#plugin-review-status").render()
        )
        assert job.review is original
        await pilot.press("escape")
        await app.push_screen(PluginStatusModal(app.plugin_source_service))
        await pilot.pause()
        history = app.screen
        index = next(
            i for i, item in enumerate(history.rows) if item.action == "uninstall"
        )
        history.query_one(OptionList).highlighted = index
        await press(history, "#plugin-status-data")
        modal = await ready()
        assert modal.installation_id == installed.installation_id
        modal.query_one("#plugin-review-action", Select).value = "delete-data"
        modal.query_one("#plugin-review-reference", Input).value = root.root_id
        await pilot.pause()
        await press(modal, "#plugin-review-prepare")
        assert modal.draft.review.targets[0]["root_id"] == root.root_id
        await press(modal, "#plugin-review-confirm")
        cleanup = modal.operations.jobs[modal._job_id]
        await asyncio.wait_for(asyncio.shield(cleanup.task), 10)
        assert cleanup.receipt.committed
        assert not root.path.exists()
        await press(modal, "#plugin-review-refresh")
        body = str(modal.query_one("#plugin-review-body").render())
        assert "Cleaned — saved data removed" in body
        assert "retained; users" not in body
        assert not (await rig.service.management_snapshot("workspace-a"))


@private_profile_test
@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["added", "settled", "generation"])
async def test_root_review_discloses_two_workspaces_and_rejects_changed_users(
    request,
    native_console,
    change,
):
    from textual.widgets import Select

    from tldw_chatbook.Widgets.plugin_review_modal import PluginReviewModal

    rig = native_console
    installed = await rig.install(enable=False)
    creation = await rig.service.review_data_creation(installed.installation_id)
    root = await rig.service.create_data(creation, creation.operation_id)
    tokens = []

    async def reserve(workspace):
        token = await rig.service._call(
            lambda: rig.service._coordinator.owner.reserve_launch(
                "test-root-user",
                installed.installation_id,
                workspace,
                installed.inspection.effective_digest,
                roots=(root,),
            )
        )
        tokens.append(token)
        return token

    await reserve("workspace-a")
    await reserve("workspace-b")
    app = BrowseApp(rig)
    try:
        async with app.run_test(size=(120, 35)) as pilot:
            await settled(pilot, app.screen)
            await app.push_screen(
                PluginReviewModal(
                    app.plugin_source_service,
                    installation_id=installed.installation_id,
                    workspace_id="workspace-a",
                )
            )
            modal = app.screen
            await modal.workers.wait_for_complete()
            await pilot.pause()
            modal.query_one("#plugin-review-action", Select).value = "delete-data"
            await pilot.pause()
            modal.query_one("#plugin-review-prepare", Button).focus()
            await pilot.press("enter")
            await pilot.pause()
            body = str(modal.query_one("#plugin-review-body").render())
            assert root.root_id in body and str(root.generation) in body
            assert "workspace-a" in body and "workspace-b" in body
            assert all(token in body for token in tokens)
            assert not modal.query_one("#plugin-review-confirm", Button).disabled
            review = modal.draft.review
            if change == "added":
                await reserve("workspace-b")
            elif change == "settled":
                token = tokens.pop()
                await rig.service._call(
                    lambda: rig.service._coordinator.owner.settle_process(token, True)
                )
            else:
                for token in tokens:
                    await rig.service._call(
                        lambda token=token: rig.service._coordinator.owner.settle_process(
                            token, True
                        )
                    )
                tokens.clear()
                attachment = await rig.service.review_data_attachment((root,), None)
                await rig.service.delete_data((root,), attachment.operation_id)
            modal.query_one("#plugin-review-confirm", Button).focus()
            await pilot.press("enter")
            await pilot.pause()
            assert not modal.operations.jobs
            assert modal.draft.review is None
            assert modal.query_one("#plugin-review-confirm", Button).disabled
            assert root.path.exists()
            assert not (
                await rig.service.lookup_operation(review.operation_id)
            ).committed
    finally:
        for token in tokens:
            await rig.service._call(
                lambda token=token: rig.service._coordinator.owner.settle_process(
                    token, True
                )
            )


@private_profile_test
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault", ["missing-completeness", "incomplete", "unknown-coverage", "oversized"]
)
async def test_root_impact_never_confirms_incomplete_or_unknown_projection(
    request, native_console, monkeypatch, fault
):
    from textual.widgets import Select

    from tldw_chatbook.Widgets.plugin_review_modal import PluginReviewModal

    rig = native_console
    installed = await rig.install(enable=False)
    creation = await rig.service.review_data_creation(installed.installation_id)
    root = await rig.service.create_data(creation, creation.operation_id)
    actual = rig.service.data_status

    async def observed(*args, **kwargs):
        rows = await actual(*args, **kwargs)
        changed = []
        for row in rows:
            row = dict(row)
            if fault == "missing-completeness":
                row.pop("blockers_complete", None)
            elif fault == "incomplete":
                row["blockers_complete"] = False
            elif fault == "unknown-coverage":
                row["blockers"] = (
                    {
                        "owner_token": "unknown-owner",
                        "reason": "root_owner_coverage_incomplete",
                    },
                )
            else:
                row["blockers"] = (
                    {
                        "owner_token": "x" * 65536,
                        "workspace_id": "workspace-a",
                        "reason": "root_user",
                    },
                )
            changed.append(row)
        return tuple(changed)

    monkeypatch.setattr(rig.service, "data_status", observed)
    app = BrowseApp(rig)
    async with app.run_test(size=(80, 24)) as pilot:
        await settled(pilot, app.screen)
        await app.push_screen(
            PluginReviewModal(
                app.plugin_source_service,
                installation_id=installed.installation_id,
                workspace_id="workspace-a",
            )
        )
        modal = app.screen
        await modal.workers.wait_for_complete()
        await pilot.pause()
        modal.query_one("#plugin-review-action", Select).value = "delete-data"
        await pilot.pause()
        modal.query_one("#plugin-review-prepare", Button).focus()
        await pilot.press("enter")
        await pilot.pause()
        assert modal.draft.review is not None
        assert modal.query_one("#plugin-review-confirm", Button).disabled
        assert root.path.exists() and not modal.operations.jobs


@pytest.mark.asyncio
async def test_retained_root_inventory_bound_counts_only_exact_target(
    native_console, monkeypatch
):
    from dataclasses import replace

    from tldw_chatbook.Widgets.plugin_review_modal import PluginReviewModal

    rig = native_console
    installed = await rig.install(enable=False)
    creation = await rig.service.review_data_creation(installed.installation_id)
    root = await rig.service.create_data(creation, creation.operation_id)
    actual = rig.service.data_status
    target_rows = [
        dict((await actual((root,)))[0], root=replace(root, root_id=f"target-{i}"))
        for i in range(257)
    ]
    calls = []

    async def paged(roots=None, *, installation_id=None, limit=50, offset=0):
        assert installation_id == installed.installation_id
        assert 1 <= limit <= 50
        calls.append((limit, offset))
        return tuple(target_rows[offset : offset + limit])

    monkeypatch.setattr(rig.service, "data_status", paged)
    modal = PluginReviewModal(
        type("Sources", (), {"plugins": rig.service})(),
        installation_id=installed.installation_id,
        workspace_id="workspace-a",
    )
    with pytest.raises(ValueError, match="retained_root_inventory_incomplete"):
        await modal._data_rows()
    assert sum(limit for limit, _ in calls) == 257
    assert len(calls) == 6


@pytest.mark.asyncio
async def test_installed_manager_collects_all_51_exact_root_refs(
    native_console, monkeypatch
):
    from tldw_chatbook.Plugins.source_service import SourceService
    from tldw_chatbook.Widgets.plugin_review_modal import PluginReviewModal

    rig = native_console
    installed = await rig.install(enable=False)
    refs = []
    for _ in range(51):
        review = await rig.service.review_data_creation(installed.installation_id)
        refs.append(await rig.service.create_data(review, review.operation_id))
    modal = PluginReviewModal(
        SourceService(rig.service),
        installation_id=installed.installation_id,
        workspace_id="workspace-a",
    )
    modal.details = await rig.service.management_details(
        installed.installation_id, workspace_id="workspace-a"
    )
    actual = rig.service.data_status
    calls = []

    async def observed(roots=None, **kwargs):
        calls.append((len(roots), kwargs["limit"], kwargs["offset"]))
        return await actual(roots, **kwargs)

    monkeypatch.setattr(rig.service, "data_status", observed)
    rows = await modal._data_rows()
    assert {row["root"] for row in rows} == set(refs)
    assert calls == [(51, 50, 0), (51, 50, 50)]


@private_profile_test
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "size,retained,change",
    [
        ((80, 24), False, "none"),
        ((80, 24), True, "none"),
        ((120, 35), False, "none"),
        ((120, 35), True, "none"),
        ((80, 24), False, "authority"),
        ((80, 24), True, "dirty"),
        ((80, 24), True, "unknown"),
    ],
)
async def test_reboot_reconciliation_uses_current_refs_before_proposed_targets(
    request, native_console, monkeypatch, retained, size, change
):
    """Actual qualified-owner proof simulation, not an OS reboot/keychain claim."""
    import asyncio

    from textual.widgets import Select

    from tldw_chatbook.Plugins import data_cleanup
    from tldw_chatbook.Widgets.plugin_review_modal import (
        PluginReviewModal,
        PluginStatusModal,
    )

    rig = native_console
    installed = await rig.install(enable=False)
    creation = await rig.service.review_data_creation(installed.installation_id)
    before = await rig.service.create_data(creation, creation.operation_id)
    if retained:
        assert (await rig.service.uninstall(installed.installation_id)).committed
    coordinator = rig.service._coordinator
    monkeypatch.setattr(coordinator.authority.marker_store, "reduced_protection", False)
    await rig.service._call(
        lambda: coordinator.authority.save_runtime_checkpoint("a" * 32, "clean")
    )
    await rig.service._call(lambda: setattr(coordinator.root_usage, "dirty", False))
    monkeypatch.setattr(
        data_cleanup, "boot_identity", lambda: "11111111-2222-3333-4444-555555555555"
    )
    app = BrowseApp(rig)
    async with app.run_test(size=size) as pilot:
        await settled(pilot, app.screen)

        async def press(identity):
            button = app.screen.query_one(identity, Button)
            for _ in range(100):
                await pilot.pause(0.02)
                if not button.disabled and not button.has_class("-active"):
                    break
            assert not button.disabled
            button.focus()
            await pilot.press("enter")
            await pilot.pause()

        if retained:
            await app.push_screen(PluginStatusModal(app.plugin_source_service))
            await app.screen.workers.wait_for_complete()
            await pilot.pause()
            history = app.screen
            history.query_one(OptionList).highlighted = next(
                i for i, item in enumerate(history.rows) if item.action == "uninstall"
            )
            await press("#plugin-status-data")
        else:
            await app.push_screen(
                PluginReviewModal(
                    app.plugin_source_service,
                    installation_id=installed.installation_id,
                    workspace_id="workspace-a",
                )
            )
        modal = app.screen
        await modal.workers.wait_for_complete()
        await pilot.pause()
        modal.query_one("#plugin-review-action", Select).value = "reconcile-data"
        await pilot.pause()
        prior_jobs = set(modal.operations.jobs)
        if change == "unknown":
            await rig.service._call(
                lambda: coordinator.root_usage.recovery.update(
                    {before.root_id: "root_owner_coverage_incomplete"}
                )
            )
        await press("#plugin-review-prepare")
        if change == "unknown":
            assert modal.draft.review is None
            assert "root_owner_coverage_incomplete" in str(
                modal.query_one("#plugin-review-status").render()
            )
            assert set(modal.operations.jobs) == prior_jobs
            await rig.service._call(
                lambda: coordinator.root_usage.recovery.pop(before.root_id)
            )
            return
        assert "Review ready" in str(modal.query_one("#plugin-review-status").render())
        review = modal.draft.review
        assert review.action == "reconcile"
        assert review.targets[0]["generation"] > before.generation
        assert modal.draft.root_impact[0]["root"] == before
        body = str(modal.query_one("#plugin-review-body").render())
        assert f"Current generation {before.generation}" in body
        assert f"proposed generation {review.targets[0]['generation']}" in body
        before_refs = modal.draft.root_before_refs
        captured_workspace = modal.workspace_id
        await pilot.press("escape")
        await app.push_screen(
            PluginReviewModal(
                app.plugin_source_service,
                installation_id=installed.installation_id,
                workspace_id=captured_workspace,
            )
        )
        modal = app.screen
        await modal.workers.wait_for_complete()
        await pilot.pause()
        assert modal.draft.review is review
        assert modal.draft.root_before_refs is before_refs
        if change == "authority":
            activation = await rig.service.review_activation(
                installed.installation_id, workspace_id="workspace-a", intent="enabled"
            )
            await rig.service.commit(activation, activation.operation_id)
        elif change == "dirty":
            await rig.service._call(
                lambda: coordinator.authority.save_runtime_checkpoint("b" * 32, "dirty")
            )
        await press("#plugin-review-confirm")
        job = modal.operations.jobs[modal._job_id]
        assert job.review is review
        await asyncio.wait_for(asyncio.shield(job.task), 10)
        if change != "none":
            assert not job.receipt.committed
            assert job.receipt.persistence_error == "PermissionError"
            current = (
                await rig.service.data_status(installation_id=installed.installation_id)
            )[0]["root"]
            assert current == before and before.path.exists()
            return
        assert job.receipt.committed, job.receipt
        await press("#plugin-review-refresh")
        assert "reconcile-data: Saved" in str(
            modal.query_one("#plugin-review-body").render()
        )
        current = (
            await rig.service.data_status(installation_id=installed.installation_id)
        )[0]["root"]
        assert current.root_id == before.root_id
        assert current.generation == review.targets[0]["generation"]
        assert before.path.exists()
        with pytest.raises(ValueError, match="root_review_stale"):
            await rig.service.data_status((before,))


async def _current_mapping_case(rig, native_package, tmp_path, aliases):
    """Use actual catalog/credential owners; profile and discovery writes do not connect."""
    import json

    from Tests.MCP.test_credential_bindings import MemoryCredentialBackend
    from tldw_chatbook.MCP.credential_bindings import CredentialBindingService
    from tldw_chatbook.MCP.local_control_service import LocalMCPControlService
    from tldw_chatbook.MCP.local_store import LocalMCPStore

    credentials = CredentialBindingService(MemoryCredentialBackend())
    binding = credentials.create_opaque(
        endpoint_origin="https://mcp.example",
        headers={"Authorization": "Bearer fixture-only"},
        method="bearer",
    )
    local = LocalMCPControlService(
        store=LocalMCPStore(tmp_path / "management-mcp.json"),
        credential_service=credentials,
    )
    local.save_external_profile(
        {
            "profile_id": "remote",
            "transport": "streamable_http",
            "url": "https://mcp.example/rpc",
            "credential_reference": binding.reference_id,
            "credential_generation": binding.authority_generation,
        }
    )
    local.store.save_discovery_snapshot(
        "remote",
        {
            "tools": [
                {
                    "name": "echo",
                    "description": "Echo",
                    "inputSchema": {"type": "object"},
                }
            ],
        },
    )
    rig.service._mcp_mapping_owner = local
    await rig.service._call(
        lambda: setattr(rig.service._coordinator, "mcp_mapping_owner", local)
    )
    root = native_package(
        extension={
            "version": 1,
            "agents": ["agent.md"],
            "requires": {"agent:reader": ["mcp:remote"]},
        }
    )
    names = ["search", "arithmetic"]
    (root / "agent.md").write_text(
        "---\nname: reader\ndescription: Read\ntools: "
        + json.dumps(names)
        + "\n---\nRead."
    )
    (root / "mcp.json").write_text(
        json.dumps(
            {
                "$schema": "https://agent-plugins.org/schemas/1.0.0/mcp.schema.json",
                "mcpServers": {
                    "remote": {
                        "type": "streamable-http",
                        "url": "https://mcp.example/rpc",
                    }
                },
            }
        )
    )
    installed = await rig.service.review_install(
        root,
        selection=("skill:review", "agent:reader", "mcp:remote"),
        workspace_id="workspace-a",
    )
    await rig.service.commit(installed, installed.operation_id)
    values = {
        "connections": {"mcp:remote": "remote"},
        "tools": {"mcp:remote": ["echo"]},
    }
    if aliases in {"builtin", "mixed"}:
        values["tool_references"] = {
            "agent:reader": {"arithmetic": "builtin:calculator"}
        }
    if aliases in {"legacy", "canonical", "mixed"}:
        key = "tool_references" if aliases == "legacy" else "component_tools"
        values.setdefault(key, {}).setdefault("agent:reader", {})[
            "search"
        ] = "mcp:remote::echo"
    return installed, local, binding, values


@private_profile_test
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "aliases", ["legacy", "canonical", "builtin", "mixed", "competing", "unknown-key"]
)
async def test_management_configuration_keeps_closed_complete_reference_inputs(
    request, native_console, native_package, tmp_path, aliases
):
    import json

    from textual.widgets import Input, Select

    from tldw_chatbook.Plugins.host_references import mapping_id, tool_constraint_id
    from tldw_chatbook.Widgets.plugin_review_modal import PluginReviewModal

    rig = native_console
    installed, local, _, values = await _current_mapping_case(
        rig,
        native_package,
        tmp_path,
        "mixed" if aliases in {"competing", "unknown-key"} else aliases,
    )
    if aliases == "competing":
        values["component_tools"]["agent:reader"]["arithmetic"] = "builtin:calculator"
    if aliases == "unknown-key":
        values["guessed_rows"] = []
    app = BrowseApp(rig)
    app.local_mcp_control_service = local
    async with app.run_test(size=(120, 35)) as pilot:
        await settled(pilot, app.screen)
        await app.push_screen(
            PluginReviewModal(
                app.plugin_source_service,
                installation_id=installed.installation_id,
                workspace_id="workspace-a",
            )
        )
        modal = app.screen
        await modal.workers.wait_for_complete()
        modal.query_one("#plugin-review-action", Select).value = "configure"
        modal.query_one("#plugin-review-reference", Input).value = json.dumps(values)
        await pilot.pause()
        if aliases in {"competing", "unknown-key"}:
            with pytest.raises(ValueError):
                await modal._prepare()
            assert modal.draft.review is None
            assert not (
                await rig.service.management_details(
                    installed.installation_id, workspace_id="workspace-a"
                )
            )["mappings"]
            return
        await modal._prepare()
        review = modal.draft.review
        rows = json.loads(review.mappings_json)
        identities = {row["mapping_id"] for row in rows}
        assert {row["kind"] for row in rows} >= {"connection", "tool"}
        if aliases in {"legacy", "canonical", "mixed"}:
            assert tool_constraint_id("agent:reader", "search") in identities
        if aliases in {"builtin", "mixed"}:
            assert mapping_id("agent:reader", "arithmetic") in identities
        await rig.service.commit(review, review.operation_id)
        current = await rig.service.management_details(
            installed.installation_id, workspace_id="workspace-a"
        )
        assert sorted(current["mappings"], key=lambda row: row["mapping_id"]) == sorted(
            rows, key=lambda row: row["mapping_id"]
        )
        assert not rig.gateway.payloads


@private_profile_test
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "state", ["empty", "builtin", "mixed", "refresh-failed", "newer-draft"]
)
async def test_profile_save_prefills_only_fresh_empty_unchanged_draft(
    request, native_console, native_package, tmp_path, monkeypatch, state
):
    import asyncio
    import json
    from uuid import uuid4

    from textual.widgets import Input, Select, Static

    from tldw_chatbook.UI.Plugins_Modules.review import PluginConnectionDraft
    from tldw_chatbook.Widgets.plugin_review_modal import PluginReviewModal

    rig = native_console
    installed, local, binding, values = await _current_mapping_case(
        rig, native_package, tmp_path, state
    )
    if state in {"builtin", "mixed"}:
        configured = await rig.service.review_configuration(
            installed.installation_id, **values
        )
        await rig.service.commit(configured, configured.operation_id)
    details = await rig.service.management_details(
        installed.installation_id, workspace_id="workspace-a"
    )
    app = BrowseApp(rig)
    app.local_mcp_control_service = local
    async with app.run_test(size=(120, 35)) as pilot:
        await settled(pilot, app.screen)
        await app.push_screen(
            PluginReviewModal(
                app.plugin_source_service,
                installation_id=installed.installation_id,
                workspace_id="workspace-a",
            )
        )
        modal = app.screen
        await modal.workers.wait_for_complete()
        modal.query_one("#plugin-review-action", Select).value = "save-profile"
        field = modal.query_one("#plugin-review-reference", Input)
        field.value = "original exact draft"
        await pilot.pause()
        draft = PluginConnectionDraft(
            uuid4().hex,
            installed.installation_id,
            "workspace-a",
            "mcp:remote",
            details["inspection"],
            credential_reference=binding.reference_id,
            credential_generation=binding.authority_generation,
        )
        modal.draft.review = draft
        original = rig.service.management_details
        reached, release = asyncio.Event(), asyncio.Event()
        calls = 0

        async def refresh(*args, **kwargs):
            nonlocal calls
            calls += 1
            result = await original(*args, **kwargs)
            if calls == 2:  # real save's precheck first, post-save refresh second
                reached.set()
                if state == "refresh-failed":
                    raise PermissionError("refresh_unavailable")
                if state == "newer-draft":
                    await release.wait()
            return result

        monkeypatch.setattr(rig.service, "management_details", refresh)
        task = asyncio.create_task(modal._save_profile(draft))
        try:
            if state == "newer-draft":
                await asyncio.wait_for(reached.wait(), 5)
                field.value = "newer exact user draft"
                await pilot.pause()
                release.set()
            await asyncio.wait_for(task, 5)
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)
        saved = modal.operations.profile_results[draft.token]
        status = str(modal.query_one("#plugin-review-status", Static).render())
        assert saved["profile_id"] in status and "saved without connecting" in status
        if state == "empty":
            assert json.loads(field.value) == {
                "connections": {"mcp:remote": saved["profile_id"]}
            }
        else:
            assert field.value == (
                "newer exact user draft"
                if state == "newer-draft"
                else "original exact draft"
            )
            assert "Draft preserved" in status
        current = await original(installed.installation_id, workspace_id="workspace-a")
        assert current["mappings"] == details["mappings"]
        assert not rig.gateway.payloads


@private_profile_test
@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(80, 24), (120, 35)])
async def test_action_inputs_hide_without_losing_drafts_or_keyboard_focus(
    request, native_console, size
):
    from textual.widgets import Input, Select, SelectionList, TextArea

    from tldw_chatbook.Widgets.plugin_review_modal import PluginReviewModal

    rig = native_console
    installed = await rig.install(enable=False)
    app = BrowseApp(rig)
    async with app.run_test(size=size) as pilot:
        await settled(pilot, app.screen)
        await app.push_screen(
            PluginReviewModal(
                app.plugin_source_service,
                installation_id=installed.installation_id,
                workspace_id="workspace-a",
            )
        )
        modal = app.screen
        await modal.workers.wait_for_complete()
        await pilot.pause()
        action = modal.query_one("#plugin-review-action", Select)
        reference = modal.query_one("#plugin-review-reference", Input)
        synthetic = modal.query_one("#plugin-review-test-data", TextArea)
        selection = modal.query_one("#plugin-review-selection", SelectionList)
        assert not reference.display and not synthetic.display
        assert not selection.display
        action.value = "configure"
        await pilot.pause()
        assert reference.display and not synthetic.display and not selection.display
        reference.value = '{"models":{}}'
        action.value = "test-hook"
        await pilot.pause()
        assert not reference.display and synthetic.display and selection.display
        synthetic.text = '{"prompt":"retained"}'
        synthetic.focus()
        action.value = "disable"
        await pilot.pause()
        assert not reference.display and not synthetic.display
        assert app.focused is action
        await pilot.press("tab")
        assert app.focused is modal.query_one("#plugin-review-prepare", Button)
        await pilot.press("enter")
        await pilot.pause()
        assert modal.draft.review is not None
        action.value = "configure"
        await pilot.pause()
        assert modal.draft.review is None
        assert modal.query_one("#plugin-review-confirm", Button).disabled
        assert reference.value == '{"models":{}}'
        action.value = "test-hook"
        await pilot.pause()
        assert synthetic.text == '{"prompt":"retained"}'
        synthetic.focus()
        await pilot.press("escape")
        await app.push_screen(
            PluginReviewModal(
                app.plugin_source_service,
                installation_id=installed.installation_id,
                workspace_id="workspace-a",
            )
        )
        reopened = app.screen
        await reopened.workers.wait_for_complete()
        await pilot.pause()
        assert reopened.query_one("#plugin-review-action", Select).value == "test-hook"
        assert (
            reopened.query_one("#plugin-review-test-data", TextArea).text
            == '{"prompt":"retained"}'
        )
        assert app.focused is reopened.query_one("#plugin-review-test-data", TextArea)


@private_profile_test
@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(80, 24), (120, 35)])
async def test_fresh_install_inputs_are_visible_before_action_changes(
    request, native_console, native_package, size
):
    from textual.widgets import Input, Select, SelectionList, TextArea

    from tldw_chatbook.Widgets.plugin_review_modal import PluginReviewModal

    rig = native_console
    producer = await rig.service.review_install(
        native_package(), selection=("skill:review",), workspace_id="workspace-a"
    )
    app = BrowseApp(rig)
    async with app.run_test(size=size) as pilot:
        await settled(pilot, app.screen)
        await app.push_screen(
            PluginReviewModal(
                app.plugin_source_service, producer=producer, workspace_id="workspace-a"
            )
        )
        modal = app.screen
        await modal.workers.wait_for_complete()
        await pilot.pause()
        action = modal.query_one("#plugin-review-action", Select)
        selection = modal.query_one("#plugin-review-selection", SelectionList)
        dialect = modal.query_one("#plugin-review-dialect", Select)
        assert action.value == "install"
        assert selection.display and dialect.display
        assert not modal.query_one("#plugin-review-reference", Input).display
        assert not modal.query_one("#plugin-review-test-data", TextArea).display
        reached = set()
        for _ in range(len(modal.focus_chain) + 1):
            reached.add(getattr(app.focused, "id", None))
            await pilot.press("tab")
        assert {selection.id, dialect.id} <= reached
        assert modal.draft.review is None


@pytest.mark.asyncio
async def test_console_ui_capture_uses_owning_plugin_workspace(native_console):
    """The UI builder must capture the owning session's plugin ceiling."""
    from tldw_chatbook.Chat.console_chat_models import (
        ConsoleProviderSelection,
        ConsoleWorkspaceContext,
    )
    from tldw_chatbook.UI.Console_Modules.session import ConsoleSessionController
    from tldw_chatbook.Workspaces.models import DEFAULT_WORKSPACE_ID

    rig = native_console
    installed = await rig.install()
    expected = rig.service.capture_maximum("workspace-a")["available_skills"]
    assert len(expected) == 1
    assert expected[0]["plugin_installation_id"] == installed.installation_id
    assert not rig.service.capture_maximum("workspace-b")["available_skills"]
    assert not rig.service.capture_maximum(None)["available_skills"]
    other = rig.store.create_session(workspace_id="workspace-b")
    global_session = rig.store.create_session(workspace_id=DEFAULT_WORKSPACE_ID)
    rig.store.set_workspace_context(
        ConsoleWorkspaceContext(active_workspace_id="workspace-b")
    )
    builder = ConsoleSessionController.__new__(ConsoleSessionController)
    builder.app_instance = rig.controller.app
    builder._provider_readiness_app_config_fn = dict
    builder._build_provider_selection_fn = lambda _: ConsoleProviderSelection(
        provider="llama_cpp", explicit_model="m"
    )
    builder._current_chat_store_accessor = lambda: rig.store
    builder._chat_store_accessor = lambda: rig.store
    builder._rag_source_types_accessor = list
    builder._rag_top_k_accessor = lambda: 4
    builder._scratch_snapshot_provider = lambda _: None

    owner = builder._build_console_turn_execution_context(rig.session.id)
    assert len(owner.skill_context_maximum["available_skills"]) == 1
    captured = owner.skill_context_maximum["available_skills"][0]
    assert captured["plugin_installation_id"] == installed.installation_id
    assert captured["plugin_workspace_id"] == "workspace-a"
    assert captured["name"] == expected[0]["name"]
    for session in (other, global_session):
        context = builder._build_console_turn_execution_context(session.id)
        assert not context.skill_context_maximum["available_skills"]
        assert not any(
            row.get("plugin_owned")
            for row in context.skill_context_maximum["available_skills"]
        )


from Tests.Plugins.test_declared_variables import variable_case as _variable_case

variable_case = _variable_case


@private_profile_test
@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(80, 24), (120, 35)])
async def test_declared_value_keyboard_mask_review_and_revoke(
    request, variable_case, size
):
    import asyncio
    import json

    from textual.widgets import Input, Select

    from Tests.Plugins.test_native_components import install_components
    from Tests.UI.consolidated_css import BUNDLED_STYLESHEET, ConsolidatedCSSApp
    from Tests.UI.test_plugins_browse import paint
    from tldw_chatbook.Plugins.source_service import SourceService
    from tldw_chatbook.Widgets.plugin_review_modal import (
        PluginReviewModal,
        PluginVariableDialog,
    )

    case = variable_case
    installed = await install_components(
        case.service, case.package, ("skill:review", "hook:guard")
    )

    class App(ConsolidatedCSSApp):
        CSS_PATH = BUNDLED_STYLESHEET

        async def on_mount(self):
            await self.push_screen(
                PluginReviewModal(
                    SourceService(case.service),
                    installation_id=installed.installation_id,
                    workspace_id="workspace-a",
                )
            )

    app = App()
    async with app.run_test(size=size) as pilot:
        modal = app.screen
        await modal.workers.wait_for_complete()
        await pilot.pause()

        async def press(selector):
            button = app.screen.query_one(selector, Button)
            button.focus()
            await pilot.pause()
            button.scroll_visible(animate=False, immediate=True)
            await pilot.pause()
            assert button in app.screen._compositor.visible_widgets
            await pilot.press("enter")
            await pilot.pause()

        modal.query_one("#plugin-review-action", Select).value = "save-variable"
        await pilot.pause()
        await press("#plugin-review-prepare")
        dialog = app.screen
        assert isinstance(dialog, PluginVariableDialog)
        dialog.query_one("#plugin-variable-name", Select).focus()
        await pilot.press("enter", "down", "enter", "tab")
        await pilot.pause()
        field = dialog.query_one("#plugin-variable-value", Input)
        assert app.focused is field and field.password
        await pilot.press(*"secret-value")
        assert field.value == "secret-value" and "secret-value" not in paint(dialog)
        await press("#plugin-variable-save")
        for _ in range(100):
            await pilot.pause(0.02)
            if app.screen is modal:
                break
        assert app.screen is modal and field.value == ""
        reference = modal.query_one("#plugin-review-reference", Input)
        assert app.focused is reference
        configuration = json.loads(reference.value)
        assert configuration["variables"]["TOKEN"].startswith("var1.")
        assert "secret-value" not in repr(modal.draft) + paint(modal) + repr(
            modal.operations.variable_results
        )
        assert (
            await case.service.capture_mcp_snapshot(
                installed.installation_id, "workspace-a", "before"
            )
        ).selection != ("hook:guard", "skill:review")
        await press("#plugin-review-prepare")
        assert modal.draft.review is not None
        await press("#plugin-review-confirm")
        await asyncio.wait_for(modal.operations.jobs[modal._job_id].task, 10)
        snapshot = await case.service.capture_mcp_snapshot(
            installed.installation_id, "workspace-a", "after"
        )
        assert "hook:guard" in snapshot.selection and not snapshot.unavailable_hooks
        modal.query_one("#plugin-review-action", Select).value = "save-variable"
        await pilot.pause()
        await press("#plugin-review-prepare")
        dialog = app.screen
        dialog.query_one("#plugin-variable-name", Select).focus()
        await pilot.press("enter", "down", "enter")
        await pilot.pause()
        await press("#plugin-variable-revoke")
        assert app.focused.id == "plugin-variable-confirm"
        await press("#plugin-variable-confirm")
        await pilot.pause()
        assert app.screen is modal and modal.draft.review is None
        assert modal.query_one("#plugin-review-confirm", Button).disabled
        assert (
            await case.service.capture_mcp_snapshot(
                installed.installation_id, "workspace-a", "revoked"
            )
        ).unavailable_hooks == ("hook:guard",)
        modal.query_one("#plugin-review-action", Select).value = "save-variable"
        await pilot.pause()
        await press("#plugin-review-prepare")
        dialog = app.screen
        field = dialog.query_one("#plugin-variable-value", Input)
        field.value = "discard-this-secret"
        await pilot.press("escape")
        assert app.screen is modal and field.value == ""


@private_profile_test
@pytest.mark.asyncio
async def test_declared_value_invalid_type_and_navigation_clear_transient_input(
    request, variable_case, monkeypatch
):
    import asyncio

    from textual.widgets import Input, Select

    from Tests.Plugins.test_native_components import install_components
    from Tests.UI.consolidated_css import BUNDLED_STYLESHEET, ConsolidatedCSSApp
    from Tests.UI.test_plugins_browse import paint
    from tldw_chatbook.UI.Plugins_Modules.operations import PluginOperations
    from tldw_chatbook.Widgets.plugin_review_modal import PluginVariableDialog

    case = variable_case
    installed = await install_components(
        case.service, case.package, ("skill:review", "hook:guard")
    )
    details = await case.service.management_details(
        installed.installation_id, workspace_id="workspace-a"
    )
    operations = PluginOperations(case.service)

    class App(ConsolidatedCSSApp):
        CSS_PATH = BUNDLED_STYLESHEET

    app = App()
    entered, release = asyncio.Event(), asyncio.Event()
    real_save = case.service.create_variable_reference

    async def paused_save(*args, **kwargs):
        entered.set()
        await release.wait()
        return await real_save(*args, **kwargs)

    monkeypatch.setattr(case.service, "create_variable_reference", paused_save)
    async with app.run_test(size=(80, 24)) as pilot:
        dialog = PluginVariableDialog(operations, installed.installation_id, details)
        await app.push_screen(dialog)
        await pilot.pause()
        field = dialog.query_one("#plugin-variable-value", Input)
        assert not field.password  # COUNT is integer, not a secret.
        await pilot.press(*"invalid-integer")
        dialog.query_one("#plugin-variable-save", Button).focus()
        await pilot.press("enter")
        await pilot.pause()
        assert field.value == "" and "invalid-integer" not in paint(dialog)
        assert not operations.variable_tasks and not entered.is_set()
        dialog.query_one("#plugin-variable-name", Select).focus()
        await pilot.press("enter", "down", "enter", "tab")
        await pilot.press(*"navigation-secret")
        dialog.query_one("#plugin-variable-save", Button).focus()
        pressing = asyncio.create_task(pilot.press("enter"))
        navigation = None
        try:
            await asyncio.wait_for(entered.wait(), 2)
            assert field.value == "" and "navigation-secret" not in paint(dialog)
            navigation = app.pop_screen()  # Removal may await the message handler.
        finally:
            release.set()
            await asyncio.wait_for(pressing, 5)
            if navigation is not None:
                await navigation
            await asyncio.gather(*operations.variable_tasks.values())
        assert field.value == ""
        assert "navigation-secret" not in repr(operations.variable_results)
        reopened = PluginVariableDialog(operations, installed.installation_id, details)
        await app.push_screen(reopened)
        await pilot.pause()
        assert reopened.references["TOKEN"].startswith("var1.")
        assert reopened.query_one("#plugin-variable-value", Input).value == ""
        await pilot.press("escape")


@private_profile_test
@pytest.mark.asyncio
async def test_value_configuration_preserves_existing_connection_and_model(
    request,
    variable_case,
):
    import json

    from Tests.Plugins.test_native_components import install_components
    from tldw_chatbook.LLM_Provider_Catalog.local_llm_provider_catalog_service import (
        LocalLLMProviderCatalogService,
    )
    from tldw_chatbook.MCP.local_store import LocalExternalMCPProfile
    from tldw_chatbook.Plugins.variables import handle
    from tldw_chatbook.Widgets.plugin_review_modal import _configuration_references

    case = variable_case
    owner = case.service._coordinator.mcp_mapping_owner
    owner.store.save_profile(
        LocalExternalMCPProfile(
            profile_id="existing",
            transport="streamable_http",
            url="https://example.invalid/mcp",
        )
    )
    mcp = {"$schema": "https://agent-plugins.org/schemas/1.0.0/mcp.schema.json"}
    mcp["mcpServers"] = {
        "remote": {"type": "streamable-http", "url": "https://example.invalid/mcp"}
    }
    (case.package / "mcp.json").write_text(json.dumps(mcp))
    manifest = json.loads((case.package / "plugin.json").read_text())
    manifest["extensions"]["io.github.rmusser01.chatbook"]["agents"] = ["agent.md"]
    (case.package / "plugin.json").write_text(json.dumps(manifest))
    (case.package / "agent.md").write_text(
        "---\nname: reviewer\ndescription: Review\ntools: inherit\nmodel: preferred\n---\nReview."
    )
    models = LocalLLMProviderCatalogService(
        provider_catalog_loader=lambda: {"llama_cpp": ["m"]},
        settings_loader=lambda: {
            "api_settings": {
                "llama_cpp": {"api_url": "http://127.0.0.1:8080/v1/chat/completions"}
            }
        },
        environ={},
    )
    await case.service._call(
        lambda: setattr(case.service._coordinator, "model_mapping_owner", models)
    )
    installed = await install_components(
        case.service, case.package, ("hook:guard", "mcp:remote", "agent:reviewer")
    )
    secret = await case.service.create_variable_reference(
        installed.installation_id, "TOKEN", "protected"
    )
    initial = {
        "connections": {"mcp:remote": "existing"},
        "models": {"agent:reviewer": "llama_cpp/m"},
        "variables": {"TOKEN": handle(secret)},
    }
    review = await case.service.review_configuration(
        installed.installation_id, **initial
    )
    assert (await case.service.commit(review, review.operation_id)).committed
    details = await case.service.management_details(
        installed.installation_id, workspace_id="workspace-a"
    )
    reconstructed = _configuration_references(details)
    for key, value in initial.items():
        assert reconstructed[key] == value
    count = await case.service.create_variable_reference(
        installed.installation_id, "COUNT", 8
    )
    reconstructed["variables"]["COUNT"] = handle(count)
    updated = await case.service.review_configuration(
        installed.installation_id, **reconstructed
    )
    assert (await case.service.commit(updated, updated.operation_id)).committed
    current = await case.service.management_details(
        installed.installation_id, workspace_id="workspace-a"
    )
    assert _configuration_references(current) == reconstructed
    assert owner.client is None  # Configuration is metadata, never a connection.


@pytest.mark.asyncio
@pytest.mark.parametrize("aliases", ["legacy", "canonical", "builtin", "mixed"])
async def test_complete_prefill_roundtrips_actual_reviewed_reference_forms(
    native_console, native_package, tmp_path, aliases
):
    from tldw_chatbook.Widgets.plugin_review_modal import _configuration_references

    rig = native_console
    installed, _local, _binding, values = await _current_mapping_case(
        rig, native_package, tmp_path, aliases
    )
    review = await rig.service.review_configuration(installed.installation_id, **values)
    assert (await rig.service.commit(review, review.operation_id)).committed
    details = await rig.service.management_details(
        installed.installation_id, workspace_id="workspace-a"
    )
    projected = _configuration_references(details)
    expected = {
        key: values.get(key, {})
        for key in (
            "connections",
            "tools",
            "models",
            "component_tools",
            "tool_references",
            "variables",
        )
    }
    if aliases == "legacy":
        # Fresh MCP aliases use canonical rows even through the old input spelling.
        expected["component_tools"], expected["tool_references"] = (
            expected["tool_references"],
            {},
        )
        from tldw_chatbook.Plugins.host_references import (
            capture_reference,
            tool_constraint_id,
        )

        rows = list(details["mappings"])
        source = next(
            row
            for row in rows
            if row["component_id"] == "mcp:remote" and row["kind"] == "tool"
        )
        historical = capture_reference(
            installed.installation_id,
            details["inspection"],
            "agent:reader",
            "tool",
            source["target_reference"],
            label="search",
            mappings=rows,
        )
        old_rows = [
            row
            for row in rows
            if row["mapping_id"] != tool_constraint_id("agent:reader", "search")
        ]
        old_rows.append(historical)
        old_projection = _configuration_references(dict(details, mappings=old_rows))
        assert old_projection["tool_references"] == values["tool_references"]
        assert old_projection["component_tools"] == {}
        # Projection has no authority write: protected current rows remain unchanged.
        assert (
            details["mappings"]
            == (
                await rig.service.management_details(
                    installed.installation_id, workspace_id="workspace-a"
                )
            )["mappings"]
        )
    assert projected == expected
    repeated = await rig.service.review_configuration(
        installed.installation_id, **projected
    )
    assert repeated.mappings_json == review.mappings_json
    assert not rig.gateway.payloads


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "corruption", ["unknown", "duplicate", "revision", "connection", "competing"]
)
async def test_complete_prefill_refuses_unrepresentable_whole_set(
    native_console, native_package, tmp_path, corruption
):
    from copy import deepcopy

    from tldw_chatbook.Plugins.host_references import tool_constraint_id
    from tldw_chatbook.Widgets.plugin_review_modal import _configuration_references

    rig = native_console
    installed, _local, _binding, values = await _current_mapping_case(
        rig, native_package, tmp_path, "mixed"
    )
    review = await rig.service.review_configuration(installed.installation_id, **values)
    assert (await rig.service.commit(review, review.operation_id)).committed
    details = await rig.service.management_details(
        installed.installation_id, workspace_id="workspace-a"
    )
    rows = deepcopy(list(details["mappings"]))
    if corruption == "unknown":
        rows.append(dict(rows[0], mapping_id="unrecognized-reference"))
    elif corruption == "duplicate":
        rows.append(dict(rows[0]))
    elif corruption == "revision":
        rows[0]["revision_digest"] = "0" * 64
    elif corruption == "connection":
        rows = [row for row in rows if row["kind"] != "connection"]
    else:
        builtin = next(
            row for row in rows if row["target_reference"] == "builtin:calculator"
        )
        source = next(
            row
            for row in rows
            if row["component_id"] == "mcp:remote" and row["kind"] == "tool"
        )
        rows.append(
            dict(
                builtin,
                kind="configuration",
                mapping_id=tool_constraint_id("agent:reader", "arithmetic"),
                target_reference=source["mapping_id"],
            )
        )
    with pytest.raises((ValueError, PermissionError)):
        _configuration_references(dict(details, mappings=rows))
    assert not rig.gateway.payloads


@private_profile_test
@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["saved", "refresh-failed", "newer-draft"])
async def test_variable_save_prefill_keeps_result_honest_and_newer_draft(
    request, variable_case, monkeypatch, state
):
    import asyncio
    import json

    from textual.widgets import Input, Static

    from Tests.Plugins.test_native_components import install_components
    from Tests.UI.consolidated_css import BUNDLED_STYLESHEET, ConsolidatedCSSApp
    from tldw_chatbook.Plugins.source_service import SourceService
    from tldw_chatbook.Plugins.variables import handle
    from tldw_chatbook.Widgets.plugin_review_modal import PluginReviewModal

    case = variable_case
    installed = await install_components(
        case.service, case.package, ("skill:review", "hook:guard")
    )
    metadata = await case.service.create_variable_reference(
        installed.installation_id, "TOKEN", "protected-fixture"
    )

    class App(ConsolidatedCSSApp):
        CSS_PATH = BUNDLED_STYLESHEET

    app = App()
    async with app.run_test(size=(120, 35)) as pilot:
        await app.push_screen(
            PluginReviewModal(
                SourceService(case.service),
                installation_id=installed.installation_id,
                workspace_id="workspace-a",
            )
        )
        modal = app.screen
        await modal.workers.wait_for_complete()
        field = modal.query_one("#plugin-review-reference", Input)
        field.value = "original exact draft"
        await pilot.pause()
        reached, release = asyncio.Event(), asyncio.Event()
        original = case.service.management_details

        async def refresh(*args, **kwargs):
            result = await original(*args, **kwargs)
            reached.set()
            if state == "refresh-failed":
                raise PermissionError("refresh_unavailable")
            if state == "newer-draft":
                await release.wait()
            return result

        monkeypatch.setattr(case.service, "management_details", refresh)
        task = asyncio.create_task(modal._variable_result(("saved", "TOKEN", metadata)))
        try:
            if state == "newer-draft":
                await asyncio.wait_for(reached.wait(), 5)
                field.value = "newer exact user draft"
                await pilot.pause()
                release.set()
            await asyncio.wait_for(task, 5)
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)
        status = str(modal.query_one("#plugin-review-status", Static).render())
        assert "Reference saved." in status
        if state == "saved":
            assert json.loads(field.value)["variables"] == {"TOKEN": handle(metadata)}
        else:
            assert field.value == (
                "newer exact user draft"
                if state == "newer-draft"
                else "original exact draft"
            )
            assert "preserved" in status.lower()
        assert "protected-fixture" not in field.value + status
        assert not (
            await original(installed.installation_id, workspace_id="workspace-a")
        )["mappings"]


@private_profile_test
@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["load-success", "load-error", "retained-success"])
async def test_close_during_child_prune_settles_original_review_load(
    request, native_console, monkeypatch, boundary
):
    import asyncio

    from textual.widgets import Static

    from tldw_chatbook.Widgets.plugin_review_modal import PluginReviewModal

    rig = native_console
    installed = await rig.install(enable=False)
    retained = None
    if boundary == "retained-success":
        creation = await rig.service.review_data_creation(installed.installation_id)
        retained = await rig.service.create_data(creation, creation.operation_id)
        assert (await rig.service.uninstall(installed.installation_id)).committed
    app = BrowseApp(rig)
    async with app.run_test(size=(120, 35)) as pilot:
        await settled(pilot, app.screen)
        owner = app.screen
        method = "data_status" if retained is not None else "management_details"
        gate = DelayedPluginCall(getattr(rig.service, method))
        monkeypatch.setattr(rig.service, method, gate)
        hold = HeldPluginUnmount()
        try:
            await app.push_screen(
                PluginReviewModal(
                    app.plugin_source_service,
                    installation_id=installed.installation_id,
                    workspace_id="workspace-a",
                )
            )
            modal = app.screen
            await asyncio.wait_for(gate.entered.wait(), 3)
            await modal.mount(hold)
            body = modal.query_one("#plugin-review-body", Static)
            before = str(body.render())
            worker = next(w for w in modal.workers if w.group == "plugin-review-load")
            modal.query_one("#plugin-review-close", Button).press()
            await wait_for_plugin_prune(modal, body, hold)
            assert app.screen is owner
            if boundary == "load-error":
                await rig.service.aclose()
            gate.release.set()
            await asyncio.wait_for(worker.wait(), 10)
            assert not worker.is_cancelled
            assert str(body.render()) == before
            assert modal.installation_id == installed.installation_id
            if boundary == "load-error":
                assert isinstance(gate.error, PermissionError)
                assert str(gate.error) == "plugin_service_closed"
            elif retained is not None:
                assert tuple(row["root"] for row in gate.result) == (retained,)
                assert gate.calls[0][1]["installation_id"] == installed.installation_id
                assert retained.path.exists()
            else:
                assert (
                    gate.result["installation"]["installation_id"]
                    == installed.installation_id
                )
                assert gate.calls == [
                    ((installed.installation_id,), {"workspace_id": "workspace-a"})
                ]
        finally:
            gate.release.set()
            hold.release.set()


@private_profile_test
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "boundary", ["save-success", "save-error", "refresh-error", "queued-save"]
)
async def test_confirmed_profile_save_survives_close_during_child_prune(
    request, native_console, native_package, tmp_path, monkeypatch, boundary
):
    import asyncio
    import json

    from textual.widgets import Input, Select, Static

    from tldw_chatbook.UI.Plugins_Modules.review import PluginConnectionDraft
    from tldw_chatbook.Widgets.plugin_review_modal import PluginReviewModal

    rig = native_console
    installed, local, binding, _ = await _current_mapping_case(
        rig, native_package, tmp_path, "empty"
    )
    app = BrowseApp(rig)
    app.local_mcp_control_service = local
    original_details = rig.service.management_details
    async with app.run_test(size=(120, 35)) as pilot:
        await settled(pilot, app.screen)
        owner = app.screen
        await app.push_screen(
            PluginReviewModal(
                app.plugin_source_service,
                installation_id=installed.installation_id,
                workspace_id="workspace-a",
            )
        )
        modal = app.screen
        await modal.workers.wait_for_complete()
        modal.query_one("#plugin-review-action", Select).value = "save-profile"
        configuration = json.dumps(
            {
                "component_id": "mcp:remote",
                "credential_reference": binding.reference_id,
                "credential_generation": binding.authority_generation,
            }
        )
        reference = modal.query_one("#plugin-review-reference", Input)
        reference.value = configuration
        await pilot.pause()
        modal.query_one("#plugin-review-prepare", Button).focus()
        await pilot.press("enter")
        await pilot.pause()
        review = modal.draft.review
        assert isinstance(review, PluginConnectionDraft)
        assert review.installation_id == installed.installation_id
        assert review.component_id == "mcp:remote"
        assert modal.draft.configuration == configuration
        operations = modal.operations
        before_profiles = local.store.list_profiles()
        if boundary == "queued-save":
            gate = DelayedPluginCall(modal._save_profile)
            monkeypatch.setattr(modal, "_save_profile", gate)
        elif boundary == "refresh-error":
            gate = DelayedPluginCall(original_details)
            calls = 0

            async def held_refresh(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 2:  # Actual save precheck, then its independent refresh.
                    return await gate(*args, **kwargs)
                return await original_details(*args, **kwargs)

            monkeypatch.setattr(rig.service, "management_details", held_refresh)
        else:
            gate = DelayedPluginCall(operations.save_profile)
            monkeypatch.setattr(operations, "save_profile", gate)
        hold = HeldPluginUnmount()
        try:
            await modal.mount(hold)
            status = modal.query_one("#plugin-review-status", Static)
            before = str(status.render())
            modal.query_one("#plugin-review-confirm", Button).focus()
            await pilot.press("enter")
            await asyncio.wait_for(gate.entered.wait(), 3)
            worker = next(w for w in modal.workers if w.group == "plugin-profile-view")
            assert modal.draft.review is review
            modal.query_one("#plugin-review-close", Button).press()
            await wait_for_plugin_prune(modal, reference, hold)
            assert status.is_detached
            assert app.screen is owner
            assert modal.draft.configuration == configuration
            if boundary in {"save-error", "refresh-error"}:
                await rig.service.aclose()
            gate.release.set()
            await asyncio.wait_for(worker.wait(), 10)
            assert not worker.is_cancelled
            assert str(status.render()) == before
            assert reference.value == configuration
            assert modal.draft.review is review
            assert operations.profile_tasks[review.token].done()
            if boundary == "save-error":
                assert isinstance(gate.error, PermissionError)
                assert str(gate.error) == "plugin_service_closed"
                assert operations.profile_tasks[review.token].exception() is gate.error
                assert review.token not in operations.profile_results
                assert local.store.list_profiles() == before_profiles
            else:
                result = operations.profile_results[review.token]
                assert result["installation_id"] == installed.installation_id
                assert result["component_id"] == review.component_id
                record = local.store.get_profile(result["profile_id"])
                assert record.plugin_owner["installation_id"] == review.installation_id
                assert record.plugin_owner["component_id"] == review.component_id
                assert len(local.store.list_profiles()) == len(before_profiles) + 1
                if boundary == "refresh-error":
                    assert isinstance(gate.error, PermissionError)
                    assert str(gate.error) == "plugin_service_closed"
                else:
                    details = await original_details(
                        installed.installation_id, workspace_id="workspace-a"
                    )
                    assert not details["mappings"]
            if boundary == "queued-save":
                assert gate.calls == [((review,), {})]
            elif boundary != "refresh-error":
                assert gate.calls == [((review, local), {})]
            assert local.client is None
            assert not rig.gateway.payloads
        finally:
            gate.release.set()
            hold.release.set()


@private_profile_test
@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["success", "error"])
async def test_confirmed_explicit_test_survives_close_during_child_prune(
    request, native_console, native_package, tmp_path, monkeypatch, outcome
):
    import asyncio
    import json

    from textual.widgets import Select, SelectionList, Static, TextArea

    from Tests.hooks_v2_process_support import child_argv
    from tldw_chatbook.Chat.console_runtime import ConsoleRuntime
    from tldw_chatbook.Widgets.plugin_review_modal import PluginReviewModal

    rig = native_console
    namespace = "io.github.rmusser01.chatbook"
    package = native_package(
        extension={"version": 1, "hooks": f"./{namespace}/hooks.json"}
    )
    directory = package / namespace
    directory.mkdir()
    marker = tmp_path / "reviewed-hook-ran"
    code = (
        "import json,sys;from pathlib import Path;event=json.load(sys.stdin);"
        f"Path({str(marker)!r}).write_text(event['origin']);"
        "print(json.dumps({'version':2,'decision':'pass'}))"
    )
    (directory / "hooks.json").write_text(
        json.dumps(
            {
                "version": 2,
                "hooks": [
                    {
                        "id": "check",
                        "event": "UserPromptSubmit",
                        "type": "command",
                        "effects": [],
                        "argv": child_argv(code),
                    }
                ],
            }
        )
    )
    installed = await rig.service.review_install(
        package,
        selection=("hook:check",),
        workspace_id="workspace-a",
        enable_in_workspace=True,
    )
    assert (await rig.service.commit(installed, installed.operation_id)).committed
    trust = await rig.service.review_trust(installed.installation_id)
    assert (await rig.service.commit(trust, trust.operation_id)).committed
    runtime = ConsoleRuntime(app=rig.controller.app)
    rig.controller.app.console_runtime = runtime
    runtime.set_chat_store(rig.store)
    runtime.set_chat_controller(rig.controller)
    app = BrowseApp(rig)
    app.console_runtime = runtime
    messages = tuple(rig.store.messages_for_session(rig.session.id))
    try:
        async with app.run_test(size=(120, 35)) as pilot:
            await settled(pilot, app.screen)
            owner = app.screen
            await app.push_screen(
                PluginReviewModal(
                    app.plugin_source_service,
                    installation_id=installed.installation_id,
                    workspace_id="workspace-a",
                    component_id="hook:check",
                )
            )
            modal = app.screen
            await modal.workers.wait_for_complete()
            modal.query_one("#plugin-review-action", Select).value = "test-hook"
            modal.query_one("#plugin-review-selection", SelectionList).select(
                "hook:check"
            )
            modal.query_one("#plugin-review-test-data", TextArea).text = (
                '{"prompt":"synthetic teardown test"}'
            )
            await pilot.pause()
            modal.query_one("#plugin-review-prepare", Button).focus()
            await pilot.press("enter")
            await pilot.pause()
            review = modal.draft.review
            assert runtime._plugin_tests.reviews[review.token] is review
            assert review.installation_id == installed.installation_id
            assert review.component_id == "hook:check"
            assert review.session_id == rig.session.id
            assert not marker.exists()
            gate = DelayedPluginCall(runtime.execute_plugin_test)
            monkeypatch.setattr(runtime, "execute_plugin_test", gate)
            hold = HeldPluginUnmount()
            try:
                await modal.mount(hold)
                body = modal.query_one("#plugin-review-body", Static)
                status = modal.query_one("#plugin-review-status", Static)
                before = str(body.render()), str(status.render())
                modal.query_one("#plugin-review-confirm", Button).focus()
                await pilot.press("enter")
                await asyncio.wait_for(gate.entered.wait(), 3)
                worker = next(w for w in modal.workers if w.group == "plugin-test-view")
                modal.query_one("#plugin-review-close", Button).press()
                await wait_for_plugin_prune(modal, body, hold)
                assert status.is_detached
                assert app.screen is owner
                if outcome == "error":
                    runtime.discard_plugin_test(review)
                gate.release.set()
                await asyncio.wait_for(worker.wait(), 10)
                assert not worker.is_cancelled
                assert (str(body.render()), str(status.render())) == before
                assert gate.calls == [((review,), {})]
                assert modal.draft.review is review
                if outcome == "success":
                    assert gate.result.succeeded
                    assert marker.read_text() == "host_plugin_test"
                    task = runtime._plugin_tests.tasks[review.token]
                    assert task.done() and task.result() is gate.result
                    assert await runtime.execute_plugin_test(review) is gate.result
                else:
                    assert isinstance(gate.error, ValueError)
                    assert str(gate.error) == "plugin_test_review_unavailable"
                    assert review.token not in runtime._plugin_tests.tasks
                    assert not marker.exists()
                assert not rig.gateway.payloads
                assert tuple(rig.store.messages_for_session(rig.session.id)) == messages
                assert rig.controller.in_flight_run_count() == 0
            finally:
                gate.release.set()
                hold.release.set()
    finally:
        await runtime.close_hooks_v2()
