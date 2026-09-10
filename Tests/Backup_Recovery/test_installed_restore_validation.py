"""Installed validation preserves the fence and proves bytes on every retry."""

import json
from pathlib import Path

import pytest

from Tests.Backup_Recovery.test_publication_crashes import _publication
from tldw_chatbook.Backup_Recovery.journal import Journal
from tldw_chatbook.Backup_Recovery.publication import publish_candidate


def published(tmp_path):
    candidate, plan, journal, bootstrap, selector = _publication(tmp_path)
    journal.prepare_publication(
        candidate,
        plan,
        bootstrap_root=bootstrap,
        namespaces=("profile",),
        selectors=(selector,),
        generation="g",
    )
    publish_candidate(candidate, plan, journal, None)
    return candidate, plan, journal, bootstrap, selector


def test_installed_validation_applies_reviewed_metadata_and_retains_fence(tmp_path):
    from tldw_chatbook.Backup_Recovery.bootstrap import startup_permission

    candidate, plan, journal, bootstrap, selector = published(tmp_path)
    journal.validate_installed(candidate, plan)
    for key, _desired, applied in plan.metadata:
        info = dict(plan.restore)[key].stat()
        assert info.st_mode & 0o777 == applied.mode
        assert info.st_mtime_ns == applied.mtime_ns
    fresh = Journal(journal.root.parent, journal.operation_id)
    assert fresh.recover() == "recovery_required"
    assert startup_permission(selector, bootstrap) == (False, "recovery_pending")
    assert any(
        json.loads(path.read_bytes())["event"] == "installed_validated"
        for path in journal.root.glob("[0-9]*.json")
    )


def test_success_record_does_not_hide_later_installed_drift(tmp_path):
    candidate, plan, journal, _, _ = published(tmp_path)
    journal.validate_installed(candidate, plan)
    dict(plan.restore)["file"].write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        Journal(journal.root.parent, journal.operation_id).validate_installed(
            candidate, plan
        )


def test_validation_refuses_incomplete_publication(tmp_path):
    candidate, plan, journal, bootstrap, selector = _publication(tmp_path)
    journal.prepare_publication(
        candidate,
        plan,
        bootstrap_root=bootstrap,
        namespaces=("profile",),
        selectors=(selector,),
        generation="g",
    )
    with pytest.raises(ValueError, match="publication_incomplete"):
        journal.validate_installed(candidate, plan)


@pytest.mark.parametrize("damage", ["missing", "tampered"])
def test_validation_requires_exact_verified_manifest(tmp_path, damage):
    candidate, plan, journal, _, _ = published(tmp_path)
    manifest = journal.root / "verified-manifest.json"
    if damage == "missing":
        manifest.unlink()
    else:
        manifest.write_bytes(b"{}")
    with pytest.raises((ValueError, FileNotFoundError)):
        journal.validate_installed(candidate, plan)
    assert journal.recover() == "recovery_required"


def test_metadata_interruption_can_resume_from_actual_installed_bytes(
    tmp_path, monkeypatch
):
    from tldw_chatbook.Backup_Recovery import publication

    candidate, plan, journal, _, _ = published(tmp_path)
    original = publication._installed_metadata
    calls = []

    def interrupted(path, expected, metadata):
        original(path, expected, metadata)
        calls.append(path)
        if len(calls) == 1:
            raise OSError("simulated interrupted metadata barrier")

    monkeypatch.setattr(publication, "_installed_metadata", interrupted)
    with pytest.raises(OSError):
        journal.validate_installed(candidate, plan)
    monkeypatch.setattr(publication, "_installed_metadata", original)
    Journal(journal.root.parent, journal.operation_id).validate_installed(
        candidate, plan
    )
    assert journal.recover() == "recovery_required"


def test_metadata_drift_after_validation_is_not_silently_repaired(tmp_path):
    candidate, plan, journal, _, _ = published(tmp_path)
    journal.validate_installed(candidate, plan)
    path = dict(plan.restore)["file"]
    path.chmod(0o700)
    with pytest.raises(ValueError, match="installed_metadata_changed"):
        journal.validate_installed(candidate, plan)
    assert path.stat().st_mode & 0o777 == 0o700


def sqlite_publication(tmp_path):
    import sqlite3
    from contextlib import closing
    from threading import Event

    from Tests.Backup_Recovery.test_restore_plan import sealed
    from tldw_chatbook.Backup_Recovery.control_records import register_pending
    from tldw_chatbook.Backup_Recovery.restore_plan import plan_restore
    from tldw_chatbook.Backup_Recovery.staging import stage_restore
    from tldw_chatbook.Research_Interop.recovery import recovery_adapters

    owner = recovery_adapters()[0]
    source = tmp_path / "older.db"
    with closing(sqlite3.connect(source)) as db:
        for sql in owner.schema_policy().schema_sql[0][1]:
            if not sql.startswith("CREATE TABLE sqlite_sequence"):
                db.execute(sql)
        db.execute(
            "INSERT INTO research_runs(id,query,created_at,updated_at) VALUES ('kept','nebula','now','now')"
        )
        db.commit()

    def research(doc):
        doc["owners"][0].update(owner_id=owner.owner_id, schema_version=0)
        doc["files"][0]["owner_id"] = owner.owner_id

    archive = sealed(tmp_path, mutate=research, data=source.read_bytes())
    plan = plan_restore(
        archive, mode="isolated", destinations={"root": tmp_path / "new"}, target=None
    )
    (tmp_path / "control").mkdir(mode=0o700)
    journal = Journal(tmp_path / "control", "sqlite-op")
    candidate = stage_restore(
        archive, plan, tmp_path / "work", Event(), journal=journal
    )
    selector = tmp_path / "new" / "config.toml"
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
        generation="sqlite",
    )
    publish_candidate(candidate, plan, journal, None)
    return candidate, plan, journal


def test_actual_migrated_sqlite_is_revalidated_only_on_disposable_copy(
    tmp_path, monkeypatch
):
    from tldw_chatbook.DB import private_sqlite

    candidate, plan, journal = sqlite_publication(tmp_path)
    original = private_sqlite.open_recovery_validation
    seen = []

    def checked(owner, path, *, writable):
        seen.append(path)
        assert Path(path) not in dict(plan.restore).values()
        assert writable is False
        return original(owner, path, writable=writable)

    monkeypatch.setattr(private_sqlite, "open_recovery_validation", checked)
    journal.validate_installed(candidate, plan)
    assert seen
    assert not list(dict(plan.destinations)["root"].glob("*-wal"))


def test_installed_sqlite_sidecar_refuses_before_disposable_validation(tmp_path):
    candidate, plan, journal = sqlite_publication(tmp_path)
    Path(str(dict(plan.restore)["file"]) + "-wal").write_bytes(b"unsettled")
    with pytest.raises(ValueError, match="installed_sqlite_sidecar_present"):
        journal.validate_installed(candidate, plan)


from Tests.Backup_Recovery.test_file_inventory import (
    installed_model as installed_model,  # noqa: PLC0414
)


def test_installed_model_bundle_rechecks_actual_owner_dependencies(
    tmp_path, installed_model, monkeypatch
):
    from threading import Event

    from Tests.Backup_Recovery.test_restore_plan import archive_entries
    from tldw_chatbook.Backup_Recovery.control_records import register_pending
    from tldw_chatbook.Backup_Recovery.restore_plan import plan_restore
    from tldw_chatbook.Backup_Recovery.staging import stage_restore

    _store, _desc, config, adapter = installed_model
    archive, doc = archive_entries(tmp_path, config, adapter.discover(config))
    destinations = {
        row["logical_id"]: tmp_path / "new-config"
        if row.get("synthetic")
        else tmp_path / "new-data" / "Local" / "models"
        for row in doc["directories"]
        if row["parent_id"] is None
    }
    destinations["profile:p:paths.data_dir"] = tmp_path / "new-data"
    plan = plan_restore(
        archive,
        mode="isolated",
        destinations=destinations,
        target=None,
        profile_names={"p": "Local"},
    )
    (tmp_path / "control").mkdir(mode=0o700)
    journal = Journal(tmp_path / "control", "model-op")
    candidate = stage_restore(
        archive, plan, tmp_path / "work", Event(), journal=journal
    )
    selector = tmp_path / "new-config" / "config.toml"
    bootstrap = tmp_path / "bootstrap"
    selectors = (selector, *(path for _, path in plan.selectors))
    register_pending(
        bootstrap, journal.operation_id, ("profile",), journal.root.parent, selectors
    )
    journal.prepare_publication(
        candidate,
        plan,
        bootstrap_root=bootstrap,
        namespaces=("profile",),
        selectors=selectors,
        generation="models",
    )
    publish_candidate(candidate, plan, journal, None)
    original = type(adapter).validate_restore_dependencies
    checked = []

    def check(self, item, path, candidates, *, topology):
        checked.append(item.logical_id)
        return original(self, item, path, candidates, topology=topology)

    monkeypatch.setattr(type(adapter), "validate_restore_dependencies", check)
    journal.validate_installed(candidate, plan)
    assert checked
    assert any(
        path.read_bytes() == b"actual model payload"
        for key, path in plan.restore
        if path.is_file()
    )


@pytest.mark.parametrize("drift", [False, True])
def test_validation_checks_retained_original_evidence(
    tmp_path, helper_resource_root, monkeypatch, drift
):
    from threading import Event

    from Tests.Backup_Recovery.test_publication_crashes import (
        _encrypted_rollback,
        _replacement,
    )

    candidate, plan, journal, previous = _replacement(tmp_path)
    archive = _encrypted_rollback(tmp_path, previous, helper_resource_root, monkeypatch)
    journal.verify_rollback(
        archive,
        password=b"test-only-password",
        work_root=tmp_path / "verify",
        cancel=Event(),
        coverage={"file": "file"},
    )
    publish_candidate(candidate, plan, journal, archive)
    with journal._locked(exclusive=False) as fd:
        records = journal._records(fd)
    prepared = next(row for row in records if row.event == "prepared")
    retained = Path(
        next(
            row["retained"]
            for row in prepared.evidence["artifacts"]
            if row["previous"] is not None
        )
    )
    if drift:
        retained.write_bytes(b"lost original")
        with pytest.raises(ValueError, match="installed_objects_changed"):
            journal.validate_installed(candidate, plan)
    else:
        with pytest.raises(ValueError, match="installed_directory_rollback_required"):
            journal.validate_installed(candidate, plan)
        assert retained.read_bytes() == b"original bytes"
    assert journal.recover() == "recovery_required"


def test_installed_validation_does_not_accept_empty_success_evidence(tmp_path):
    _candidate, _plan, journal, _, _ = published(tmp_path)
    with journal._locked(exclusive=False) as fd:
        receipt = journal._records(fd)[0].evidence
    with pytest.raises(ValueError, match="journal_evidence_invalid"):
        journal.record(
            "installed_validated",
            {
                "plan_digest": receipt["plan_digest"],
                "descriptor_digest": receipt["descriptor"]["sha256"],
                "manifest_digest": receipt["manifest_digest"],
                "artifacts": [],
            },
        )


def test_existing_owned_directory_metadata_requires_rollback_before_mutation(
    tmp_path, helper_resource_root, monkeypatch
):
    from threading import Event

    from Tests.Backup_Recovery.test_restore_plan import producer, sealed
    from tldw_chatbook.Backup_Recovery.control_records import register_pending
    from tldw_chatbook.Backup_Recovery.models import Inventory, StorageItem
    from tldw_chatbook.Backup_Recovery.restore_plan import plan_restore
    from tldw_chatbook.Backup_Recovery.staging import stage_restore

    def excluded(doc):
        producer(doc)
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
    target.mkdir(mode=0o700)
    optional = target / "optional"
    optional.write_bytes(b"preserved")
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
    from Tests.Backup_Recovery.test_publication_crashes import _encrypted_rollback

    rollback = _encrypted_rollback(
        tmp_path, optional, helper_resource_root, monkeypatch
    )
    journal.verify_rollback(
        rollback,
        password=b"test-only-password",
        work_root=tmp_path / "verify",
        cancel=Event(),
        coverage={},
    )
    publish_candidate(candidate, plan, journal, rollback)
    before = target.stat()
    with pytest.raises(ValueError, match="installed_directory_rollback_required"):
        journal.validate_installed(candidate, plan)
    assert (target.stat().st_mode, target.stat().st_mtime_ns) == (
        before.st_mode,
        before.st_mtime_ns,
    )
    assert optional.read_bytes() == b"preserved"


def test_native_process_exit_after_metadata_reopens_and_revalidates(tmp_path):
    import dataclasses
    import os
    import subprocess
    import sys

    candidate, plan, journal, _, _ = published(tmp_path)
    serialized = dataclasses.asdict(plan)
    script = """
import json, os, sys
from pathlib import Path
from tldw_chatbook.Backup_Recovery.archive_models import Metadata
from tldw_chatbook.Backup_Recovery.restore_plan import RestorePlan
from tldw_chatbook.Backup_Recovery.journal import Journal
from tldw_chatbook.Backup_Recovery import publication
value = json.loads(sys.argv[4])
for field in ("restore", "retire", "preserve", "destinations", "selectors", "containers"):
    value[field] = tuple((key, Path(path)) for key, path in value[field])
value["profile_names"] = tuple(tuple(row) for row in value["profile_names"])
value["issues"] = tuple(value["issues"])
value["metadata"] = tuple((key, Metadata.model_validate(desired) if desired else None, Metadata.model_validate(applied)) for key, desired, applied in value["metadata"])
plan = RestorePlan(**value)
original = publication._installed_metadata
def stop_after_native_barrier(*args):
    original(*args)
    os._exit(73)
publication._installed_metadata = stop_after_native_barrier
Journal(Path(sys.argv[2]), sys.argv[3]).validate_installed(Path(sys.argv[1]), plan)
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(candidate),
            str(journal.root.parent),
            journal.operation_id,
            json.dumps(
                serialized,
                default=lambda value: (
                    value.model_dump() if hasattr(value, "model_dump") else str(value)
                ),
            ),
        ],
        cwd=Path(__file__).resolve().parents[2],
        env=dict(os.environ),
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 73, result.stderr
    fresh = Journal(journal.root.parent, journal.operation_id)
    assert fresh.recover() == "recovery_required"
    fresh.validate_installed(candidate, plan)
    assert dict(plan.restore)["file"].read_bytes() == b"durable"


def test_reviewed_executable_bit_and_directory_time_survive_publication(
    tmp_path, monkeypatch
):
    from Tests.Backup_Recovery import test_restore_plan

    original = test_restore_plan.sealed

    def metadata(document):
        document["files"][0]["metadata"] = {
            "version": 1,
            "mode": 0o700,
            "mtime_ns": 12000000000,
        }
        document["directories"][0]["metadata"]["mtime_ns"] = 11000000000

    monkeypatch.setattr(
        test_restore_plan, "sealed", lambda path: original(path, mutate=metadata)
    )
    candidate, plan, journal, _, _ = published(tmp_path)
    assert dict(plan.restore)["file"].stat().st_mode & 0o777 == 0o600
    journal.validate_installed(candidate, plan)
    assert dict(plan.restore)["file"].stat().st_mode & 0o777 == 0o700
    assert dict(plan.restore)["root"].stat().st_mtime_ns == 11000000000
