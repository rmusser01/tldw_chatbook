"""Directory-only rollback coverage preserves excluded children and fences drift."""

import json
import os
from threading import Event

import pytest

from Tests.Backup_Recovery.test_archive_writer import captured
from Tests.Backup_Recovery.test_restore_plan import producer, sealed
from tldw_chatbook.Backup_Recovery import crypto
from tldw_chatbook.Backup_Recovery.archive_writer import write_archive
from tldw_chatbook.Backup_Recovery.capture import CaptureResult
from tldw_chatbook.Backup_Recovery.control_records import register_pending
from tldw_chatbook.Backup_Recovery.journal import Journal
from tldw_chatbook.Backup_Recovery.models import Inventory, StorageItem
from tldw_chatbook.Backup_Recovery.publication import publish_candidate
from tldw_chatbook.Backup_Recovery.restore_plan import plan_restore
from tldw_chatbook.Backup_Recovery.staging import stage_restore


def node(path):
    info = path.stat()
    return info.st_dev, info.st_ino, info.st_mode & 0o777, info.st_mtime_ns


def directory_case(
    tmp_path,
    helper_resource_root,
    monkeypatch,
    *,
    damage=None,
    original_mode=0o700,
    synthetic=False,
):
    def excluded(doc):
        producer(doc)
        doc["directories"][0]["synthetic"] = synthetic
        doc["producer_inventory"].append(
            {
                "logical_id": "optional",
                "owner_id": "ui.state",
                "status": "intentionally_excluded",
                "dependencies": [],
                "shared_group": None,
            }
        )
        doc["exclusions"].append(
            {"logical_id": "optional", "reason": "intentionally_excluded"}
        )

    archive = sealed(tmp_path, mutate=excluded)
    target = tmp_path / "live"
    target.mkdir(mode=original_mode)
    optional = target / "optional"
    optional.write_bytes(b"preserved child")
    optional.chmod(0o640)
    os.utime(optional, ns=(22000000000, 22000000000))
    os.utime(target, ns=(11000000000, 11000000000))
    inventory = Inventory(
        (
            StorageItem("ui.state", "root", target, "included_directory", ()),
            StorageItem("ui.state", "optional", optional, "included", ()),
        ),
        True,
        "local",
        (),
    )
    plan = plan_restore(
        archive, mode="replace", destinations={"root": target}, target=inventory
    )
    (tmp_path / "control").mkdir(mode=0o700)
    journal = Journal(tmp_path / "control", "directory-op")
    candidate = stage_restore(
        archive, plan, tmp_path / "work", Event(), journal=journal
    )
    selector = target / "config.toml"
    bootstrap = tmp_path / "bootstrap"
    register_pending(
        bootstrap, journal.operation_id, ("profile",), journal.root.parent, (selector,)
    )
    journal.prepare_publication(
        candidate,
        plan,
        bootstrap_root=bootstrap,
        namespaces=("profile",),
        selectors=(selector,),
        generation="g",
    )
    capture = captured(tmp_path, optional.read_bytes())
    doc = json.loads(capture.manifest_bytes)
    doc["owners"][0]["owner_id"] = "ui.state"
    doc["files"][0]["owner_id"] = "ui.state"
    producer(doc)
    doc["credential_policy"] = "rollback"
    doc["directories"][0]["metadata"] = {
        "version": 1,
        "mode": node(target)[2],
        "mtime_ns": node(target)[3],
    }
    if damage == "mode":
        doc["directories"][0]["metadata"]["mode"] = 0o600
    elif damage == "time":
        doc["directories"][0]["metadata"]["mtime_ns"] += 1
    elif damage == "owner":
        doc["owners"].append(
            {"owner_id": "external.files", "schema_version": 0, "capabilities": []}
        )
        doc["producer_inventory"][0]["owner_id"] = "external.files"
    monkeypatch.setattr(crypto, "_package_resource_root", lambda: helper_resource_root)
    rollback = tmp_path / "rollback.tldw-backup.zip.age"
    write_archive(
        CaptureResult(capture.root, capture.inventory, json.dumps(doc).encode()),
        rollback,
        password=b"test-only-password",
        cancel=Event(),
    )
    return candidate, plan, journal, rollback, optional


def verify(journal, rollback, tmp_path, *, coverage=None):
    journal.verify_rollback(
        rollback,
        password=b"test-only-password",
        work_root=tmp_path / "verify",
        cancel=Event(),
        coverage={"root": "root"} if coverage is None else coverage,
    )


def test_exact_directory_rollback_restores_node_after_child_rename(
    tmp_path, helper_resource_root, monkeypatch
):
    candidate, plan, journal, rollback, optional = directory_case(
        tmp_path, helper_resource_root, monkeypatch
    )
    original_directory, original_child = node(optional.parent), node(optional)
    verify(journal, rollback, tmp_path)
    publish_candidate(candidate, plan, journal, rollback)
    assert node(optional.parent)[3] != original_directory[3]
    journal.validate_installed(candidate, plan)
    applied = {key: value for key, _, value in plan.metadata}["root"]
    assert node(optional.parent)[2:] == (applied.mode, applied.mtime_ns)
    assert node(optional) == original_child
    assert optional.read_bytes() == b"preserved child"
    assert journal.recover() == "recovery_required"


@pytest.mark.parametrize("damage", ["mode", "time", "owner"])
def test_directory_rollback_rejects_wrong_node_coverage(
    tmp_path, helper_resource_root, monkeypatch, damage
):
    candidate, plan, journal, rollback, optional = directory_case(
        tmp_path, helper_resource_root, monkeypatch, damage=damage
    )
    before = node(optional.parent)
    with pytest.raises(ValueError, match="rollback_coverage_mismatch"):
        verify(journal, rollback, tmp_path)
    with pytest.raises(ValueError, match="rollback_required"):
        publish_candidate(candidate, plan, journal, rollback)
    assert node(optional.parent) == before


def test_directory_rollback_requires_explicit_coverage_before_publication(
    tmp_path, helper_resource_root, monkeypatch
):
    candidate, plan, journal, rollback, optional = directory_case(
        tmp_path, helper_resource_root, monkeypatch
    )
    before = node(optional.parent)
    with pytest.raises(ValueError, match="rollback_coverage_mismatch"):
        verify(journal, rollback, tmp_path, coverage={})
    with pytest.raises(ValueError, match="rollback_required"):
        publish_candidate(candidate, plan, journal, rollback)
    assert node(optional.parent) == before


@pytest.mark.parametrize("boundary", ["before", "partial", "after"])
def test_directory_metadata_interruption_reopens_without_touching_preserved_child(
    tmp_path, helper_resource_root, monkeypatch, boundary
):
    from tldw_chatbook.Backup_Recovery import publication

    candidate, plan, journal, rollback, optional = directory_case(
        tmp_path, helper_resource_root, monkeypatch
    )
    preserved = node(optional)
    verify(journal, rollback, tmp_path)
    publish_candidate(candidate, plan, journal, rollback)
    original = publication._installed_metadata

    def interrupted(path, expected, metadata, **kwargs):
        if path != optional.parent:
            return original(path, expected, metadata, **kwargs)
        if boundary == "partial":
            os.chmod(path, metadata["mode"])
        elif boundary == "after":
            original(path, expected, metadata, **kwargs)
        raise OSError("interrupted owned directory metadata")

    monkeypatch.setattr(publication, "_installed_metadata", interrupted)
    with pytest.raises(OSError):
        journal.validate_installed(candidate, plan)
    fresh = Journal(journal.root.parent, journal.operation_id)
    assert fresh.recover() == "recovery_required"
    monkeypatch.setattr(publication, "_installed_metadata", original)
    fresh.validate_installed(candidate, plan)
    assert node(optional) == preserved
    assert optional.read_bytes() == b"preserved child"


@pytest.mark.parametrize("phase", ["before_intent", "after_intent", "after_validation"])
def test_directory_metadata_drift_is_not_overwritten(
    tmp_path, helper_resource_root, monkeypatch, phase
):
    from tldw_chatbook.Backup_Recovery import publication

    candidate, plan, journal, rollback, optional = directory_case(
        tmp_path, helper_resource_root, monkeypatch
    )
    verify(journal, rollback, tmp_path)
    publish_candidate(candidate, plan, journal, rollback)
    original = publication._installed_metadata
    if phase == "after_intent":

        def stop(path, expected, metadata, **kwargs):
            if path == optional.parent:
                raise OSError("pause after durable intent")
            return original(path, expected, metadata, **kwargs)

        monkeypatch.setattr(publication, "_installed_metadata", stop)
        with pytest.raises(OSError):
            journal.validate_installed(candidate, plan)
        monkeypatch.setattr(publication, "_installed_metadata", original)
    elif phase == "after_validation":
        journal.validate_installed(candidate, plan)
    os.utime(optional.parent, ns=(77000000000, 77000000000))
    with pytest.raises(ValueError, match="directory_metadata_(changed|unproven)"):
        Journal(journal.root.parent, journal.operation_id).validate_installed(
            candidate, plan
        )
    assert node(optional.parent)[3] == 77000000000
    assert optional.read_bytes() == b"preserved child"


def test_child_rename_without_recorded_directory_timestamp_remains_fenced(
    tmp_path, helper_resource_root, monkeypatch
):
    candidate, plan, journal, rollback, optional = directory_case(
        tmp_path, helper_resource_root, monkeypatch
    )
    verify(journal, rollback, tmp_path)
    original = journal._append

    def interrupted(parent, event, evidence):
        if event == "artifact_published":
            raise OSError("rename completed before record")
        return original(parent, event, evidence)

    monkeypatch.setattr(journal, "_append", interrupted)
    with pytest.raises(OSError):
        publish_candidate(candidate, plan, journal, rollback)
    assert (optional.parent / "note.txt").read_bytes() == b"durable"
    with pytest.raises(ValueError, match="directory_metadata_unproven"):
        Journal(journal.root.parent, journal.operation_id).validate_installed(
            candidate, plan
        )
    assert journal.recover() == "recovery_required"


def test_directory_drift_between_preflight_and_native_open_is_not_overwritten(
    tmp_path, helper_resource_root, monkeypatch
):
    from tldw_chatbook.Backup_Recovery import publication

    candidate, plan, journal, rollback, optional = directory_case(
        tmp_path, helper_resource_root, monkeypatch
    )
    verify(journal, rollback, tmp_path)
    publish_candidate(candidate, plan, journal, rollback)
    original = publication._installed_metadata

    def raced(path, expected, metadata, **kwargs):
        if path == optional.parent:
            os.utime(path, ns=(88000000000, 88000000000))
        return original(path, expected, metadata, **kwargs)

    monkeypatch.setattr(publication, "_installed_metadata", raced)
    with pytest.raises(ValueError, match="directory_metadata_changed"):
        journal.validate_installed(candidate, plan)
    assert node(optional.parent)[3] == 88000000000


def test_retry_rejects_mtime_only_state_impossible_for_chmod_then_utime(
    tmp_path, helper_resource_root, monkeypatch
):
    from tldw_chatbook.Backup_Recovery import publication

    candidate, plan, journal, rollback, optional = directory_case(
        tmp_path, helper_resource_root, monkeypatch, original_mode=0o750
    )
    verify(journal, rollback, tmp_path)
    publish_candidate(candidate, plan, journal, rollback)
    original = publication._installed_metadata

    def stop(path, expected, metadata, **kwargs):
        if path == optional.parent:
            raise OSError("stop after intent")
        return original(path, expected, metadata, **kwargs)

    monkeypatch.setattr(publication, "_installed_metadata", stop)
    with pytest.raises(OSError):
        journal.validate_installed(candidate, plan)
    monkeypatch.setattr(publication, "_installed_metadata", original)
    applied = {key: value for key, _, value in plan.metadata}["root"]
    os.utime(optional.parent, ns=(applied.mtime_ns, applied.mtime_ns))
    assert node(optional.parent)[2:] == (0o750, applied.mtime_ns)
    with pytest.raises(ValueError, match="directory_metadata_changed"):
        Journal(journal.root.parent, journal.operation_id).validate_installed(
            candidate, plan
        )
    assert node(optional.parent)[2:] == (0o750, applied.mtime_ns)


def test_existing_synthetic_container_metadata_is_never_applied(
    tmp_path, helper_resource_root, monkeypatch
):
    candidate, plan, journal, rollback, optional = directory_case(
        tmp_path, helper_resource_root, monkeypatch, original_mode=0o750, synthetic=True
    )
    assert "root" not in dict(plan.restore)
    verify(journal, rollback, tmp_path, coverage={})
    publish_candidate(candidate, plan, journal, rollback)
    before = node(optional.parent)
    child = node(optional)
    journal.validate_installed(candidate, plan)
    assert node(optional.parent) == before
    assert node(optional) == child
