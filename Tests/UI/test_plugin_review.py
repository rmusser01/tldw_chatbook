"""Management review uses mounted actions and the actual plugin owner."""

import pytest
from textual.widgets import Button, OptionList

from Tests.Plugins.conftest import native_console as _native_console
from Tests.Plugins.conftest import native_package as _native_package
from Tests.UI.test_plugins_browse import BrowseApp, settled

native_console = _native_console
native_package = _native_package


@pytest.mark.asyncio
async def test_mounted_management_opens_exact_review_without_committing(native_console):
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


@pytest.mark.asyncio
async def test_mounted_review_is_invalidated_by_changed_selection(native_console):
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


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(80, 24), (120, 35)])
async def test_actual_first_use_setup_through_masked_dialog(tmp_path, size):
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


@pytest.mark.asyncio
async def test_close_reopen_preserves_synthetic_data_and_focus(native_console):
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


@pytest.mark.asyncio
async def test_unlock_wrong_password_stays_in_masked_flow_then_correct_unlock(tmp_path):
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


@pytest.mark.asyncio
async def test_status_refresh_sees_new_retained_root(native_console):
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


@pytest.mark.asyncio
async def test_mounted_exact_reviews_cover_installed_lifecycle(
    native_console, native_package
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


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(80, 24), (120, 35)])
async def test_uninstall_retained_data_status_and_history_cleanup(native_console, size):
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


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["added", "settled", "generation"])
async def test_root_review_discloses_two_workspaces_and_rejects_changed_users(
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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault", ["missing-completeness", "incomplete", "unknown-coverage", "oversized"]
)
async def test_root_impact_never_confirms_incomplete_or_unknown_projection(
    native_console, monkeypatch, fault
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
    native_console, monkeypatch, retained, size, change
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


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(80, 24), (120, 35)])
async def test_action_inputs_hide_without_losing_drafts_or_keyboard_focus(
    native_console, size
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


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(80, 24), (120, 35)])
async def test_fresh_install_inputs_are_visible_before_action_changes(
    native_console, native_package, size
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
        assert not context.skill_context_maximum["native_components"]
