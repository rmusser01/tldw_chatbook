"""Selected-profile recovered media uses actual owner review and worker lifetime."""

import asyncio
import json
from threading import Event

import pytest

from Tests.Backup_Recovery.conftest import (
    helper_resource_root as helper_resource_root,  # noqa: PLC0414 - pytest fixture re-export
)


@pytest.fixture(autouse=True)
def _disable_model_catalog_refresh():
    """The minimal recovery host does not construct model services."""


@pytest.fixture
def selected_media(tmp_path, monkeypatch):
    import keyring
    from keyring.backends.null import Keyring

    from tldw_chatbook.Backup_Recovery import bootstrap
    from tldw_chatbook.Backup_Recovery.control_records import (
        admission_authority,
        bind_profile,
    )
    from tldw_chatbook.Backup_Recovery.recovered_media import (
        RecoveredMedia,
        current_profile_id,
    )
    from tldw_chatbook.Backup_Recovery.recovery_service import RecoveryService

    monkeypatch.setattr(keyring, "get_keyring", lambda: Keyring())
    bootstrap_root = tmp_path / "bootstrap"
    monkeypatch.setattr(bootstrap, "default_bootstrap_root", lambda: bootstrap_root)
    parent = tmp_path / "profile"
    parent.mkdir(mode=0o700)
    selector = parent / "config.toml"
    selector.write_text(
        '[general]\nusers_name="reader"\n[paths]\ndata_dir='
        + json.dumps(str(parent / "data"))
        + "\n"
    )
    selector.chmod(0o600)
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(selector))
    authority = admission_authority(bootstrap_root)
    authority.register("profile", (parent,))
    bind_profile(bootstrap_root, selector, ("profile",), bootstrap_root / "admission")
    store = RecoveredMedia(parent / "data/reader/recovered_media")
    source = parent / "opaque"
    source.write_bytes(b"retained bytes")
    asset = store.retain(
        source,
        profile=current_profile_id(),
        message="message",
        slug="clip",
        media_type="video/webm",
    )
    store.add_reference(
        asset,
        profile="historical",
        message="old-message",
        slug="old-clip",
        media_type="video/webm",
    )
    service = RecoveryService(tmp_path / "control")
    try:
        yield service, store, asset, selector
    finally:
        service.close()


@pytest.mark.parametrize("change", ["bytes", "selection"])
def test_service_reviews_selected_owner_and_rechecks_config(
    selected_media, change, monkeypatch, tmp_path
):
    service, store, asset, selector = selected_media
    page = service.recovered_media_details(limit=1)
    assert page["details"].assets[0].asset_id == asset
    review = service.review_recovered_media(asset)
    assert len(review.asset.references) == 2
    if change == "bytes":
        selector.write_text(selector.read_text() + "\n# changed after review\n")
    else:
        other = tmp_path / "other.toml"
        other.write_text('[general]\nusers_name="other"\n')
        monkeypatch.setenv("TLDW_CONFIG_PATH", str(other))
    operation = service.start_recovered_media_action(
        review, action="delete", user_selected=True
    )
    state = service.wait(operation, timeout=10)
    assert state["state"] == "failed"
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(selector))
    assert store.resolve(asset)[0] == "ready"


def test_recovered_action_refuses_changed_actual_paired_generation(
    tmp_path, monkeypatch, helper_resource_root
):
    import tomllib

    from Tests.Backup_Recovery.test_current_generation_requirements import (
        _healthy_replacement,
    )
    from tldw_chatbook.Backup_Recovery import bootstrap
    from tldw_chatbook.Backup_Recovery.profile_paths import user_data_dir
    from tldw_chatbook.Backup_Recovery.recovered_media import (
        RecoveredMedia,
        current_profile_id,
    )
    from tldw_chatbook.Backup_Recovery.recovery_service import RecoveryService

    with _healthy_replacement(tmp_path, monkeypatch, helper_resource_root) as (
        case,
        _,
        _,
    ):
        selector = case[-1]
        store = RecoveredMedia(
            user_data_dir(tomllib.loads(selector.read_text())) / "recovered_media"
        )
        source = selector.parent / "opaque"
        source.write_bytes(b"reviewed generation bytes")
        asset = store.retain(
            source,
            profile=current_profile_id(),
            message="m",
            slug="clip",
            media_type="image/png",
        )
        service = RecoveryService(tmp_path / "control")
        try:
            review = service.review_recovered_media(asset)
            assert review.source.witnesses != b"[]"
            payload = store.resolve(asset)[1]
            original = payload.read_bytes()
            pair = (
                tmp_path
                / "bootstrap"
                / ("activation-" + bootstrap._key(str(selector)) + ".json")
            )
            generation = json.loads(review.source.witnesses)[0]["generation"]
            pair.write_text(pair.read_text().replace(generation, "changed-generation"))
            operation = service.start_recovered_media_action(
                review, action="delete", user_selected=True
            )
            assert service.wait(operation, timeout=10)["state"] == "failed"
            assert payload.read_bytes() == original
        finally:
            service.close()


def test_completed_owner_delete_is_not_relabelled_by_later_config_edit(
    selected_media, monkeypatch
):
    from tldw_chatbook.Backup_Recovery import recovered_media as owner

    service, store, asset, selector = selected_media
    review = service.review_recovered_media(asset)
    original = owner.delete_reviewed_asset

    def completed_then_edit(review):
        original(review)
        selector.write_text(
            selector.read_text() + "\n# changed after owner completion\n"
        )

    monkeypatch.setattr(owner, "delete_reviewed_asset", completed_then_edit)
    operation = service.start_recovered_media_action(
        review, action="delete", user_selected=True
    )
    state = service.wait(operation, timeout=10)
    assert store.resolve(asset) == ("deleted", None)
    assert state["state"] == "succeeded"


def test_selected_catalog_pagination_and_explicit_orphan_cleanup(selected_media):
    from tldw_chatbook.Backup_Recovery.recovered_media import current_profile_id

    service, store, asset, selector = selected_media
    second = store.retain(
        selector.parent / "opaque",
        profile=current_profile_id(),
        message="second",
        slug="clip",
        media_type="video/webm",
    )
    first_page = service.recovered_media_details(limit=1)["details"]
    second_page = service.recovered_media_details(limit=1, offset=1)["details"]
    assert first_page.total == 2 and first_page.has_more and not second_page.has_more
    assert {first_page.assets[0].asset_id, second_page.assets[0].asset_id} == {
        asset,
        second,
    }
    store.release_reference(
        profile=current_profile_id(),
        message="second",
        slug="clip",
        media_type="video/webm",
    )
    review = service.review_recovered_media(second)
    assert review.asset.asset.orphan_eligible
    with pytest.raises(ValueError, match="preview_required"):
        service.start_recovered_media_action(
            review, action="cleanup", user_selected=False
        )
    assert store.resolve(second)[0] == "ready"
    operation = service.start_recovered_media_action(
        review, action="cleanup", user_selected=True
    )
    assert service.wait(operation, timeout=10)["state"] == "succeeded"
    assert store.resolve(second) == ("deleted", None)
    assert store.resolve(asset)[0] == "ready"


def test_owner_reference_change_refuses_reviewed_delete(selected_media):
    service, store, asset, _ = selected_media
    review = service.review_recovered_media(asset)
    store.add_reference(
        asset, profile="new-alias", message="new", slug="clip", media_type="video/webm"
    )
    operation = service.start_recovered_media_action(
        review, action="delete", user_selected=True
    )
    assert service.wait(operation, timeout=10)["state"] == "failed"
    assert store.resolve(asset)[0] == "ready"


@pytest.mark.parametrize("parent_exists", [True, False])
def test_selected_absent_catalog_does_not_create_root(selected_media, parent_exists):
    service, _, _, selector = selected_media
    destination = selector.parent / "empty-data"
    selector.write_text(
        '[general]\nusers_name="reader"\n[paths]\ndata_dir='
        + json.dumps(str(destination))
        + "\n"
    )
    if parent_exists:
        (destination / "reader").mkdir(parents=True, mode=0o700)
        page = service.recovered_media_details()
        assert page["status"] == "absent" and page["details"] is None
    else:
        from tldw_chatbook.Utils.private_paths import PrivatePathError

        with pytest.raises(PrivatePathError):
            service.recovered_media_details()
    assert not (destination / "reader/recovered_media").exists()


@pytest.mark.asyncio
async def test_mounted_review_discloses_aliases_and_retains_accepted_delete(
    selected_media, monkeypatch
):
    from textual.app import App
    from textual.widgets import Button, Checkbox, Static

    from tldw_chatbook.Backup_Recovery import recovered_media as owner
    from tldw_chatbook.UI.Screens.backup_restore_screen import BackupRestoreScreen

    service, store, asset, _ = selected_media

    class Host(App):
        def on_mount(self):
            self.push_screen(BackupRestoreScreen(service))

    entered, release = Event(), Event()
    original = owner.delete_reviewed_asset

    def retained(review):
        entered.set()
        assert release.wait(10)
        return original(review)

    monkeypatch.setattr(owner, "delete_reviewed_asset", retained)
    app = Host()
    try:
        async with app.run_test(size=(120, 48)) as pilot:
            await pilot.click("#backup-open-profiles")
            async with asyncio.timeout(10):
                while not list(app.screen.query("#backup-open-media")):
                    await asyncio.sleep(0.02)
            await pilot.click("#backup-open-media")
            async with asyncio.timeout(10):
                while not list(app.screen.query(".backup-review-media")):
                    await asyncio.sleep(0.02)
            button = next(
                row
                for row in app.screen.query(".backup-review-media")
                if row.name == asset
            )
            button.focus()
            await pilot.press("enter")
            async with asyncio.timeout(10):
                while app.screen._media_review is None:
                    await asyncio.sleep(0.02)
            shown = str(app.screen.query_one("#backup-media-review", Static).render())
            assert (
                "historical" in shown and "old-message" in shown and "old-clip" in shown
            )
            assert "recorded" in shown.lower()
            assert app.screen.query_one("#backup-media-delete", Button).disabled
            app.screen.query_one("#backup-media-confirm", Checkbox).value = True
            await pilot.pause()
            app.screen.query_one("#backup-media-delete", Button).focus()
            await pilot.press("enter")
            assert await asyncio.to_thread(entered.wait, 5)
            await pilot.press("escape")
            assert not isinstance(app.screen, BackupRestoreScreen)
            release.set()
            state = await asyncio.to_thread(
                service.wait, service.current()["operation_id"], timeout=10
            )
            assert state["state"] == "succeeded"
            assert store.resolve(asset) == ("deleted", None)
    finally:
        release.set()


@pytest.mark.asyncio
@pytest.mark.parametrize("dismiss", ["cancel", "navigate"])
async def test_late_owner_review_is_discarded_after_dismissal(
    selected_media, monkeypatch, dismiss
):
    from textual.app import App
    from textual.widgets import Button, Static

    from tldw_chatbook.Backup_Recovery import recovered_media as owner
    from tldw_chatbook.UI.Screens.backup_restore_screen import BackupRestoreScreen

    service, store, asset, _ = selected_media

    class Host(App):
        def on_mount(self):
            self.push_screen(BackupRestoreScreen(service))

    entered, release = Event(), Event()
    original = owner.review_recovered_asset

    def delayed(*args):
        result = original(*args)
        entered.set()
        assert release.wait(10)
        return result

    monkeypatch.setattr(owner, "review_recovered_asset", delayed)
    app = Host()
    try:
        async with app.run_test(size=(120, 48)) as pilot:
            await pilot.click("#backup-open-profiles")
            async with asyncio.timeout(10):
                while not list(app.screen.query("#backup-open-media")):
                    await asyncio.sleep(0.02)
            await pilot.click("#backup-open-media")
            async with asyncio.timeout(10):
                while not list(app.screen.query(".backup-review-media")):
                    await asyncio.sleep(0.02)
            screen = app.screen
            screen.query_one(".backup-review-media", Button).focus()
            await pilot.press("enter")
            assert await asyncio.to_thread(entered.wait, 5)
            target = (
                "#backup-media-cancel"
                if dismiss == "cancel"
                else "#backup-open-profiles"
            )
            screen.query_one(target, Button).focus()
            await pilot.press("enter")
            release.set()
            await screen.workers.wait_for_complete()
            assert screen._media_review is None
            assert str(screen.query_one("#backup-media-review", Static).render()) == ""
            assert service.current() is None
            assert store.resolve(asset)[0] == "ready"
    finally:
        release.set()


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["held", "deleted"])
async def test_mounted_non_actionable_rows_keep_recorded_details(selected_media, state):
    from textual.app import App
    from textual.widgets import Button, Static

    from tldw_chatbook.UI.Screens.backup_restore_screen import BackupRestoreScreen

    service, store, asset, _ = selected_media
    if state == "held":
        store.hold(asset, "retained-recovery-copy")
    else:
        store.delete(asset)

    class Host(App):
        def on_mount(self):
            self.push_screen(BackupRestoreScreen(service))

    app = Host()
    async with app.run_test(size=(120, 48)) as pilot:
        await pilot.click("#backup-open-profiles")
        async with asyncio.timeout(10):
            while not list(app.screen.query("#backup-open-media")):
                await asyncio.sleep(0.02)
        await pilot.click("#backup-open-media")
        async with asyncio.timeout(10):
            while not list(app.screen.query(".backup-review-media")):
                await asyncio.sleep(0.02)
        screen = app.screen
        screen.query_one(".backup-review-media", Button).focus()
        await pilot.press("enter")
        async with asyncio.timeout(10):
            while screen._media_review is None:
                await asyncio.sleep(0.02)
        shown = str(screen.query_one("#backup-media-review", Static).render())
        assert "14 recorded bytes" in shown and "historical" in shown
        assert (
            "retained-recovery-copy" if state == "held" else "Catalog state: deleted"
        ) in shown
        assert not screen.query_one("#backup-media-delete", Button).display
        assert not screen.query_one("#backup-media-cleanup", Button).display
