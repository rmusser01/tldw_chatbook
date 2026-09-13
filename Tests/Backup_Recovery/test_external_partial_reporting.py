"""External payload backups require and report the existing Partial review."""

import json
from threading import Event

import pytest


@pytest.fixture
def complete_local_capture(tmp_path, monkeypatch):
    """Provide actual covered config/external files behind installed adapters."""
    from tldw_chatbook.Backup_Recovery import (
        bootstrap,
        capture_service,
        owner_registry,
    )
    from tldw_chatbook.Backup_Recovery import (
        capture as capture_module,
    )
    from tldw_chatbook.Backup_Recovery import (
        inventory as inventory_module,
    )
    from tldw_chatbook.Backup_Recovery.inventory import classify_entries
    from tldw_chatbook.Backup_Recovery.models import StorageItem

    source = tmp_path / "source"
    source.mkdir(mode=0o700)
    config = source / "config.toml"
    config.write_text('[general]\nusers_name="fixture"\n')
    config.chmod(0o600)
    external = source / "external.txt"
    external.write_text("selected external evidence")
    external.chmod(0o600)
    config_item = StorageItem(
        "config", "profile:fixture:config", config, "included", ()
    )
    external_item = StorageItem(
        "external.files",
        "profile:fixture:external.files:selected",
        external,
        "included",
        (),
    )
    inventories = {
        "coherent": classify_entries((config_item,)),
        "external": classify_entries((config_item, external_item)),
    }
    assert inventories["coherent"].complete
    assert inventories["external"].complete

    selected = [inventories["external"]]

    def discover(*_args, **_kwargs):
        return selected[0]

    monkeypatch.setattr(capture_service, "discover", discover)
    monkeypatch.setattr(capture_module, "discover", discover)
    monkeypatch.setattr(inventory_module, "discover", discover)
    monkeypatch.setattr(owner_registry, "_adapters", {})
    monkeypatch.setattr(
        bootstrap, "default_bootstrap_root", lambda: tmp_path / "bootstrap"
    )
    return config, external, inventories, selected


def test_external_files_require_partial_acknowledgement_before_native_capture(
    tmp_path, complete_local_capture
):
    """Removing either capture guard would let covered external bytes proceed."""
    from tldw_chatbook.Backup_Recovery.capture import (
        CaptureReviewRequired,
        _capture_under_maintenance,
    )
    from tldw_chatbook.Backup_Recovery.capture_service import capture, preview_capture
    from tldw_chatbook.Backup_Recovery.control_records import admission_authority

    config, external, inventories, _ = complete_local_capture
    options = {"external_roots": (external.parent,), "staging_parent": tmp_path}
    preview = preview_capture((config,), options=options)
    destination = tmp_path / "refused.tldw-backup.zip"
    original = {config: config.read_bytes(), external: external.read_bytes()}

    with pytest.raises(CaptureReviewRequired) as error:
        capture(
            (config,),
            preview.scope_digest,
            destination,
            options=options,
            cancel=Event(),
        )
    assert error.value.issues == ("partial_archive",)
    assert not destination.exists()
    assert not (tmp_path / "bootstrap").exists()

    authority = admission_authority(tmp_path / "bootstrap")
    authority.register("fixture.sources", (config.parent,))
    with (
        authority.maintenance(("bootstrap.unbound", "fixture.sources"), 1) as session,
        pytest.raises(CaptureReviewRequired) as error,
    ):
        _capture_under_maintenance(
            session,
            (config,),
            inventories["external"].scope_digest,
            destination,
            options=options,
            cancel=Event(),
        )
    assert error.value.issues == ("partial_archive",)
    assert not destination.exists()
    assert not tuple(tmp_path.glob("capture-*"))
    assert {path: path.read_bytes() for path in original} == original


@pytest.mark.parametrize(
    ("inventory_name", "allow_partial", "expected_consistency", "expected_complete"),
    (
        ("external", True, "partial", False),
        ("coherent", False, "coherent", True),
    ),
)
def test_service_reports_verified_manifest_consistency(
    tmp_path,
    monkeypatch,
    complete_local_capture,
    inventory_name,
    allow_partial,
    expected_consistency,
    expected_complete,
):
    """Publishing from Inventory.complete would misreport a valid external archive."""
    from tldw_chatbook.Backup_Recovery.archive_reader import acquire, verify_sealed
    from tldw_chatbook.Backup_Recovery.limits import ArchiveLimits
    from tldw_chatbook.Backup_Recovery.recovery_service import RecoveryService

    config, external, inventories, selected = complete_local_capture
    selected[0] = inventories[inventory_name]
    options = {
        "allow_partial": allow_partial,
        "staging_parent": tmp_path,
        "external_roots": (external.parent,) if inventory_name == "external" else (),
    }
    destination = tmp_path / (inventory_name + ".tldw-backup.zip")
    original = {config: config.read_bytes(), external: external.read_bytes()}
    service = RecoveryService(tmp_path / "service-control")
    monkeypatch.setattr(
        service,
        "backup_capability",
        lambda *_args, **_kwargs: (True, "focused_component_qualification"),
    )
    try:
        details = service.preview_backup_details(
            (config,), options=options, destination=destination
        )
        assert details["inventory"].complete
        assert details["complete"] is expected_complete
        operation = service.start_backup(
            (config,),
            details["inventory"].scope_digest,
            destination,
            options=options,
            password=None,
        )
        state = service.wait(operation, timeout=15)
    finally:
        service.close()

    assert state["state"] == "succeeded", dict(state)
    assert state["result"]["complete"] is expected_complete
    sealed = acquire(
        destination, tmp_path / "inspection", ArchiveLimits(), None, Event()
    )
    manifest = verify_sealed(sealed)
    assert manifest.consistency == expected_consistency
    assert json.loads(sealed.manifest_bytes)["consistency"] == expected_consistency
    assert {path: path.read_bytes() for path in original} == original


@pytest.mark.asyncio
async def test_ui_labels_external_archive_partial_and_requires_existing_checkbox():
    """This focused component test exercises the real screen with inert service IO."""
    from pathlib import Path

    from textual.app import App
    from textual.widgets import Button, Checkbox, Static

    from tldw_chatbook.Backup_Recovery.models import Inventory, StorageItem
    from tldw_chatbook.UI.Screens.backup_restore_screen import BackupRestoreScreen

    class Service:
        issue_code = staticmethod(str)

        def current(self):
            return None

    inventory = Inventory(
        (
            StorageItem(
                "external.files",
                "profile:fixture:external.files:selected",
                Path("/selected/external.txt"),
                "included",
                (),
            ),
        ),
        True,
        "reviewed-scope",
        (),
    )
    details = {
        "inventory": inventory,
        "complete": False,
        "maintenance": "Writers pause during capture.",
        "capacity": (
            {
                "path": "/volume",
                "required_bytes": 1,
                "available_bytes": 2,
                "sufficient": True,
            },
        ),
        "credential_mode": "exclude",
        "availability": (True, "focused_component_qualification"),
    }
    screen = BackupRestoreScreen(
        Service(), config_paths=(Path("/profile/config.toml"),)
    )

    class Harness(App):
        def on_mount(self):
            self.push_screen(screen)

    async with Harness().run_test(size=(90, 30)):
        screen._show_mode("create")
        reviewed = (
            (Path("/profile/config.toml"),),
            Path("/output.zip"),
            {"allow_partial": False},
        )
        screen._show_preview(screen._revision, details, reviewed)
        coverage = str(screen.query_one("#backup-coverage", Static).render())
        assert "Archive classification: Partial" in coverage
        assert screen.query_one("#backup-create", Button).disabled
        assert "Partial archive" in str(
            screen.query_one("#backup-partial", Checkbox).label
        )

        reviewed = (*reviewed[:2], {"allow_partial": True})
        screen._show_preview(screen._revision, details, reviewed)
        assert not screen.query_one("#backup-create", Button).disabled
