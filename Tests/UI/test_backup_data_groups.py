"""Mounted group controls preserve the scope the user actually reviewed."""

import asyncio
import hashlib
import json
import zipfile
from contextlib import asynccontextmanager

import pytest


@pytest.fixture(autouse=True)
def _disable_model_catalog_refresh():
    """This recovery-only harness never starts the full application."""


@asynccontextmanager
async def mounted(tmp_path, *, size=(90, 32)):
    from Tests.UI.consolidated_css import ConsolidatedCSSApp as App
    from tldw_chatbook.Backup_Recovery.recovery_service import RecoveryService
    from tldw_chatbook.UI.Screens.backup_restore_screen import BackupRestoreScreen

    config = tmp_path / "config.toml"
    config.write_text('[general]\nusers_name="Ada"\n')
    service = RecoveryService(tmp_path / "control")

    class Harness(App):
        def on_mount(self):
            self.push_screen(BackupRestoreScreen(service, config_paths=(config,)))

    try:
        async with Harness().run_test(size=size) as pilot:
            yield pilot, pilot.app.screen, service
    finally:
        await asyncio.to_thread(service.close)


async def press(pilot, identifier):
    from textual.widgets import Button

    pilot.app.screen.query_one(identifier, Button).focus()
    await pilot.press("enter")
    await pilot.pause()


async def inspect_profile(
    pilot, screen, service, tmp_path, *, external=False, evaluations=False
):
    from Tests.Backup_Recovery.test_restore_destinations import profile_archive

    source = profile_archive(tmp_path, external=external)
    # The older destination-only fixture puts all files into one coarse group.
    # Selection needs ordinary owner dependencies and shared-store groups, as
    # emitted by capture, including config support for the selected database.
    with zipfile.ZipFile(source) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    document = json.loads(members["manifest.json"])
    config_id = "profile:source:config"
    if evaluations:
        from tldw_chatbook.Backup_Recovery.owner_registry import install_adapters

        owner = next(
            owner for owner in install_adapters()
            if owner.owner_id == "eval.definitions"
        )
        key, root, payload = "profile:source:eval.definitions", "eval-root", "payload/eval"
        content = b"[]"
        document["owners"].append({
            "owner_id": owner.owner_id,
            "schema_version": owner.schema_policy().versions[-1],
            "capabilities": [],
        })
        document["directories"].append({
            "logical_id": root, "root_id": root, "parent_id": None,
            "relative_path": "", "synthetic": True,
            "metadata": {"version": 1, "mode": 448, "mtime_ns": 0},
        })
        document["files"].append({
            "logical_id": key, "root_id": root, "parent_id": root,
            "relative_path": "saved.json", "owner_id": owner.owner_id,
            "payload": payload, "size": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        })
        document["producer_inventory"].extend((
            {"logical_id": root, "owner_id": owner.owner_id,
             "status": "included_directory", "dependencies": []},
            {"logical_id": key, "owner_id": owner.owner_id,
             "status": "included", "dependencies": [config_id]},
        ))
        members[payload] = content
    for item in document["producer_inventory"]:
        if item["status"] == "included" and item["owner_id"] != "config":
            item["dependencies"] = [config_id]
    document["dependency_groups"] = []
    represented = {
        row["logical_id"] for row in (*document["files"], *document["directories"])
    }
    for item in document["producer_inventory"]:
        group_members = {item["logical_id"], *item["dependencies"]}
        if item.get("shared_group"):
            group_members.update(
                alias["logical_id"]
                for alias in document["producer_inventory"]
                if alias.get("shared_group") == item["shared_group"]
            )
        document["dependency_groups"].append(
            {
                "group_id": "group:"
                + hashlib.sha256(item["logical_id"].encode()).hexdigest(),
                "members": sorted(group_members & represented),
                "complete": True,
            }
        )
    members["manifest.json"] = json.dumps(document).encode()
    with zipfile.ZipFile(source, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    operation = service.start_inspection(source, password=None)
    result = await asyncio.to_thread(service.wait, operation)
    assert result["state"] == "succeeded"
    await press(pilot, "#backup-open-inspect")
    async with asyncio.timeout(10):
        while screen._inspection_summary is None:
            await asyncio.sleep(0.02)


@pytest.mark.asyncio
async def test_backup_everything_default_and_keyboard_scope_at_narrow_width(tmp_path):
    from textual.containers import VerticalScroll
    from textual.widgets import Checkbox, Select

    async with mounted(tmp_path, size=(54, 22)) as (pilot, screen, _service):
        await press(pilot, "#backup-open-create")
        assert screen._options().get("data_groups", "missing") is None
        screen.query_one("#backup-data-mode", Select).value = "choose"
        await pilot.pause()
        boxes = list(screen.query("#backup-data-choices Checkbox"))
        assert boxes and not any(box.value for box in boxes)
        box = screen.query_one("#backup-data-group-prompts", Checkbox)
        box.focus()
        await pilot.pause()
        await pilot.wait_for_scheduled_animations()
        assert (
            box.region.intersection(
                screen.query_one("#backup-body", VerticalScroll).content_region
            )
            == box.region
        ), {
            "regions": {
                identifier: screen.query_one(f"#{identifier}").region
                for identifier in (
                    "backup-title",
                    "backup-actions",
                    "backup-status",
                    "backup-body",
                    "backup-message",
                    "backup-footer-actions",
                )
            },
            "rendered": "\n".join(
                strip.text for strip in screen._compositor.render_strips()
            ),
        }
        await pilot.press("space")
        assert screen._options()["data_groups"] == ("prompts",)
        screen.query_one("#backup-data-mode", Select).value = "all"
        await pilot.pause()
        assert screen._options()["data_groups"] is None
        assert not screen.query_one("#backup-data-choices").display


@pytest.mark.asyncio
async def test_empty_backup_group_selection_refuses_before_discovery(tmp_path):
    from textual.widgets import Button, Input, Select, Static

    async with mounted(tmp_path) as (pilot, screen, service):
        await press(pilot, "#backup-open-create")
        assert screen.query("#backup-data-mode"), "Backup scope controls are missing"
        screen.query_one("#backup-destination", Input).value = str(
            tmp_path / "selected.tldw-backup.zip"
        )
        screen.query_one("#backup-data-mode", Select).value = "choose"
        await pilot.pause()
        await press(pilot, "#backup-review")
        assert "Choose at least one data group" in str(
            screen.query_one("#backup-message", Static).render()
        )
        assert screen.query_one("#backup-create", Button).disabled
        assert service.current() is None


@pytest.mark.asyncio
async def test_backup_review_names_required_groups_and_invalidates_on_scope_change(
    tmp_path,
):
    from textual.widgets import Button, Checkbox, Select, Static

    from tldw_chatbook.Backup_Recovery.models import Inventory

    async with mounted(tmp_path) as (pilot, screen, _service):
        await press(pilot, "#backup-open-create")
        assert screen.query("#backup-data-mode"), "Backup scope controls are missing"
        screen.query_one("#backup-data-mode", Select).value = "choose"
        screen.query_one("#backup-data-group-prompts", Checkbox).value = True
        await pilot.pause()
        inventory = Inventory((), True, "selected-scope", ())
        details = {
            "inventory": inventory,
            "complete": True,
            "maintenance": "Writers resume after copying.",
            "availability": (True, ""),
            "capacity": (),
            "credential_mode": "exclude",
            "requested_groups": ("prompts",),
            "effective_groups": ("prompts", "conversations"),
            "required_groups": ("conversations",),
            "whole_profile": False,
        }
        screen._show_preview(
            screen._revision,
            details,
            ((tmp_path / "config.toml",), tmp_path / "backup.zip", screen._options()),
        )
        coverage = str(screen.query_one("#backup-coverage", Static).render())
        assert "Complete selected coverage" in coverage
        assert "Required linked groups: Conversations, notes, and personas" in coverage
        assert "whole profile" in coverage
        assert not screen.query_one("#backup-create", Button).disabled
        screen.query_one("#backup-data-group-writing", Checkbox).focus()
        await pilot.press("space")
        assert screen._preview is None
        assert screen.query_one("#backup-create", Button).disabled


@pytest.mark.asyncio
async def test_verified_archive_offers_only_available_restore_data_groups(tmp_path):
    from textual.widgets import Select

    async with mounted(tmp_path) as (pilot, screen, service):
        await inspect_profile(pilot, screen, service, tmp_path)
        assert screen.query("#backup-restore-data-mode"), (
            "Restore scope controls are missing"
        )
        assert screen.query_one("#backup-restore-data-mode", Select).value == "all"
        screen.query_one("#backup-restore-data-mode", Select).value = "choose"
        await pilot.pause()
        boxes = list(screen.query("#backup-restore-data-choices Checkbox"))
        assert {box.name for box in boxes} == {
            "settings",
            "conversations",
            "workspaces",
        }
        assert not any(box.value for box in boxes)
        assert all(not box.has_class("backup-safety-member") for box in boxes)
        assert not any(
            (box.id or "").startswith("backup-inert-group-") for box in boxes
        )


@pytest.mark.asyncio
async def test_empty_restore_selection_refuses_before_destination_review(tmp_path):
    from textual.widgets import Button, Select, Static

    async with mounted(tmp_path) as (pilot, screen, service):
        await inspect_profile(pilot, screen, service, tmp_path)
        assert screen.query("#backup-restore-data-mode"), (
            "Restore scope controls are missing"
        )
        screen.query_one("#backup-restore-data-mode", Select).value = "choose"
        await pilot.pause()
        await press(pilot, "#backup-review-restore")
        assert "Choose at least one data group" in str(
            screen.query_one("#backup-message", Static).render()
        )
        assert screen._restore_plan is None
        assert screen.query_one("#backup-start-restore", Button).disabled


def dependent_evaluation_target(tmp_path):
    """A controller input with the installed Evals-to-ChaChaNotes dependency."""
    from tldw_chatbook.Backup_Recovery.models import Inventory, StorageItem

    config = StorageItem(
        "config", "profile:local:config", tmp_path / "config.toml", "included", ()
    )
    conversations = StorageItem(
        "db.chachanotes.primary", "profile:local:db.chachanotes.primary",
        tmp_path / "ChaChaNotes.db", "included", (config.logical_id,),
    )
    evaluations = StorageItem(
        "db.evals", "profile:local:db.evals", tmp_path / "evals.db", "included",
        (config.logical_id, conversations.logical_id),
    )
    return Inventory((config, conversations, evaluations), True, "local", ())


@pytest.mark.asyncio
async def test_target_required_group_exposes_setup_before_destination_review(
    tmp_path, monkeypatch
):
    from textual.widgets import Button, Checkbox, Input, Select, Static

    async with mounted(tmp_path) as (pilot, screen, service):
        await inspect_profile(pilot, screen, service, tmp_path, evaluations=True)
        screen.query_one("#backup-restore-mode", Select).value = "replace"
        screen.query_one("#backup-restore-data-mode", Select).value = "choose"
        screen.query_one("#backup-restore-data-group-conversations", Checkbox).value = True
        screen.query_one("#backup-target-config", Input).value = str(tmp_path / "config.toml")
        await pilot.pause()
        assert not screen.query_one("#backup-setup-destination").display
        target = dependent_evaluation_target(tmp_path)
        monkeypatch.setattr(service, "preview_backup", lambda *args, **kwargs: target)
        preview = service.preview_restore
        reviewed = []

        def observe_preview(*args, **kwargs):
            reviewed.append(kwargs)
            return preview(*args, **kwargs)

        monkeypatch.setattr(service, "preview_restore", observe_preview)
        await press(pilot, "#backup-review-restore")
        await screen.workers.wait_for_complete()
        await pilot.pause()
        assert screen.query_one("#backup-setup-destination").display
        review = str(screen.query_one("#backup-restore-preview", Static).render())
        assert "Required linked groups: Evaluations" in review
        assert "Selected data groups: Conversations, notes, and personas" in review
        assert not reviewed, "Destination validation must wait for the newly visible fields"
        assert screen._restore_plan is None
        assert screen.query_one("#backup-start-restore", Button).disabled

        setup = tmp_path / "setup"
        setup.mkdir(mode=0o700)
        screen.query_one("#backup-setup-parent", Input).value = str(setup)
        await pilot.pause()
        assert screen.query_one("#backup-setup-destination").display
        await press(pilot, "#backup-review-restore")
        await screen.workers.wait_for_complete()
        await pilot.pause()
        assert len(reviewed) == 1
        assert reviewed[0]["setup_parent"] == setup
        assert reviewed[0]["data_groups"] == ("conversations",)
        assert reviewed[0]["target"] is target
        # This UI fixture supplies no native filesystem authority; the real
        # planner still refuses it after receiving the complete user choices.
        assert screen._restore_plan is None
        assert screen.query_one("#backup-start-restore", Button).disabled

        screen.query_one("#backup-target-config", Input).value = str(tmp_path / "other.toml")
        await pilot.pause()
        assert not screen.query_one("#backup-setup-destination").display


@pytest.mark.asyncio
async def test_changed_selection_discards_inflight_target_requirements(tmp_path, monkeypatch):
    from threading import Event

    from textual.widgets import Button, Checkbox, Input, Select

    entered, release = Event(), Event()
    async with mounted(tmp_path) as (pilot, screen, service):
        await inspect_profile(pilot, screen, service, tmp_path, evaluations=True)
        screen.query_one("#backup-restore-mode", Select).value = "replace"
        screen.query_one("#backup-restore-data-mode", Select).value = "choose"
        screen.query_one("#backup-restore-data-group-conversations", Checkbox).value = True
        screen.query_one("#backup-target-config", Input).value = str(tmp_path / "config.toml")
        await pilot.pause()
        target = dependent_evaluation_target(tmp_path)

        def held_target(*args, **kwargs):
            entered.set()
            assert release.wait(10)
            return target

        monkeypatch.setattr(service, "preview_backup", held_target)
        try:
            await press(pilot, "#backup-review-restore")
            async with asyncio.timeout(5):
                while not entered.is_set():
                    await asyncio.sleep(0.02)
            screen.query_one("#backup-restore-data-group-conversations", Checkbox).value = False
            screen.query_one("#backup-restore-data-group-workspaces", Checkbox).value = True
            await pilot.pause()
        finally:
            release.set()
        await screen.workers.wait_for_complete()
        await pilot.pause()
        assert not screen.query_one("#backup-setup-destination").display
        assert screen._effective_restore_groups() == {"workspaces"}
        assert screen._restore_plan is None
        assert screen.query_one("#backup-start-restore", Button).disabled


@pytest.mark.asyncio
async def test_legacy_everything_review_names_all_available_groups(tmp_path):
    from textual.widgets import Input, Static

    (tmp_path / "destinations").mkdir(mode=0o700)
    async with mounted(tmp_path) as (pilot, screen, service):
        await inspect_profile(pilot, screen, service, tmp_path)
        screen.query_one("#backup-root-0", Input).value = str(
            tmp_path / "destinations" / "restored"
        )
        screen.query_one("#backup-profile-name-0", Input).value = "Restored profile"
        await pilot.pause()
        await press(pilot, "#backup-review-restore")
        await screen.workers.wait_for_complete()
        await pilot.pause()
        assert screen._restore_plan is not None
        review = str(screen.query_one("#backup-restore-preview", Static).render())
        assert (
            "Included data groups: Settings, Conversations, notes, and personas, Workspaces and agents"
            in review
        )


@pytest.mark.asyncio
async def test_restore_selection_filters_external_destination_and_reaches_real_plan(
    tmp_path,
):
    from textual.widgets import Button, Checkbox, Input, Select, Static

    (tmp_path / "destinations").mkdir(mode=0o700)
    async with mounted(tmp_path) as (pilot, screen, service):
        await inspect_profile(pilot, screen, service, tmp_path, external=True)
        assert screen.query("#backup-restore-data-mode"), (
            "Restore scope controls are missing"
        )
        screen.query_one("#backup-restore-data-mode", Select).value = "choose"
        screen.query_one("#backup-restore-data-group-workspaces", Checkbox).value = True
        screen.query_one("#backup-root-0", Input).value = str(
            tmp_path / "destinations" / "restored"
        )
        screen.query_one("#backup-profile-name-0", Input).value = "Restored workspaces"
        await pilot.pause()
        assert not screen.query_one("#backup-root-1", Input).display
        await press(pilot, "#backup-review-restore")
        await screen.workers.wait_for_complete()
        await pilot.pause()
        assert screen._restore_plan is not None, str(
            screen.query_one("#backup-restore-preview", Static).render()
        )
        assert screen._restore_plan.requested_groups == ("workspaces",)
        assert not screen.query_one("#backup-start-restore", Button).disabled
        review = str(screen.query_one("#backup-restore-preview", Static).render())
        assert "Workspaces and agents" in review
        assert "external" not in dict(screen._restore_plan.restore)


@pytest.mark.asyncio
async def test_changed_restore_selection_discards_inflight_review(
    tmp_path, monkeypatch
):
    from threading import Event

    from textual.widgets import Button, Checkbox, Input, Select

    (tmp_path / "destinations").mkdir(mode=0o700)
    entered, release = Event(), Event()
    async with mounted(tmp_path) as (pilot, screen, service):
        await inspect_profile(pilot, screen, service, tmp_path)
        assert screen.query("#backup-restore-data-mode"), (
            "Restore scope controls are missing"
        )
        screen.query_one("#backup-restore-data-mode", Select).value = "choose"
        screen.query_one("#backup-restore-data-group-workspaces", Checkbox).value = True
        screen.query_one("#backup-root-0", Input).value = str(
            tmp_path / "destinations" / "restored"
        )
        screen.query_one("#backup-profile-name-0", Input).value = "Restored workspaces"
        await pilot.pause()
        preview = service.preview_restore
        delivered = []

        def held_preview(*args, **kwargs):
            entered.set()
            assert release.wait(10)
            try:
                plan = preview(*args, **kwargs)
            except (OSError, ValueError, RuntimeError) as error:
                delivered.append(error)
                raise
            delivered.append(plan)
            return plan

        monkeypatch.setattr(service, "preview_restore", held_preview)
        try:
            await press(pilot, "#backup-review-restore")
            async with asyncio.timeout(5):
                while not entered.is_set():
                    await asyncio.sleep(0.02)
            screen.query_one(
                "#backup-restore-data-group-settings", Checkbox
            ).value = True
            await pilot.pause()
        finally:
            release.set()
        await screen.workers.wait_for_complete()
        await pilot.pause()
        assert delivered and not isinstance(delivered[0], Exception), delivered
        assert delivered and delivered[0].requested_groups == ("workspaces",)
        assert screen._restore_plan is None
        assert screen.query_one("#backup-start-restore", Button).disabled
