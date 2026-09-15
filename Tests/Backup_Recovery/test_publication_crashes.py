"""Durable recovery records never infer successful publication from intent."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "damage", ["truncated", "gap", "duplicate", "unknown", "linked"]
)
def test_damaged_journal_is_retained_for_recovery(tmp_path, damage):
    from tldw_chatbook.Backup_Recovery.journal import Journal

    journal = Journal(tmp_path, "op")
    journal.record("prepared", {"generation": "g1", "mode": "isolated"})
    record = journal.root / "000000.json"
    if damage == "truncated":
        record.write_bytes(b'{"version":1,')
    elif damage == "gap":
        record.rename(journal.root / "000001.json")
    elif damage == "duplicate":
        record.write_bytes(b'{"version":1,"version":1}')
    elif damage == "unknown":
        value = json.loads(record.read_bytes())
        value["event"] = "invented_success"
        record.write_text(json.dumps(value))
    else:
        os.link(record, journal.root / "linked")
    before = {p.name: p.read_bytes() for p in journal.root.iterdir()}
    assert Journal(tmp_path, "op").recover() == "recovery_required"
    assert before == {p.name: p.read_bytes() for p in journal.root.iterdir()}


@pytest.mark.parametrize("boundary", ["prepared", "publication_started"])
def test_process_exit_after_durable_record_retains_startup_fence(tmp_path, boundary):
    from tldw_chatbook.Backup_Recovery.bootstrap import startup_permission
    from tldw_chatbook.Backup_Recovery.control_records import register_pending
    from tldw_chatbook.Backup_Recovery.journal import Journal

    config = tmp_path / "broken.toml"
    config.write_bytes(b"broken [")
    register_pending(
        tmp_path / "bootstrap", "op", ("scope",), tmp_path / "control", (config,)
    )
    script = """
import os, sys
from pathlib import Path
from tldw_chatbook.Backup_Recovery.journal import Journal
j = Journal(Path(sys.argv[1]), "op")
j.record("prepared", {"generation":"g1", "mode":"isolated"})
if sys.argv[2] == "publication_started":
    j.record("publication_started", {})
os._exit(73)
"""
    child = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path), boundary],
        timeout=20,
        capture_output=True,
        check=False,
    )
    assert child.returncode == 73, child.stderr.decode()
    expected = "prepared" if boundary == "prepared" else "recovery_required"
    assert Journal(tmp_path, "op").recover() == expected
    assert startup_permission(config, tmp_path / "bootstrap") == (
        False,
        "recovery_pending",
    )
    assert config.read_bytes() == b"broken ["


def test_replacement_cannot_start_before_verified_rollback(tmp_path):
    from tldw_chatbook.Backup_Recovery.journal import Journal

    journal = Journal(tmp_path, "op")
    journal.record("prepared", {"generation": "g1", "mode": "replace"})
    with pytest.raises(ValueError, match="rollback_required"):
        journal.record("publication_started", {})


def test_artifact_identity_recognizes_rename_without_completion_record(tmp_path):
    from tldw_chatbook.Backup_Recovery.journal import Journal, observe_artifact
    from tldw_chatbook.Backup_Recovery.native_files import publish_new

    candidate = tmp_path / "candidate"
    candidate.mkdir(mode=0o700)
    (candidate / "data").write_bytes(b"durable new bytes")
    (candidate / "empty").mkdir()
    target = tmp_path / "target"
    journal = Journal(tmp_path, "op")
    journal.record(
        "prepared",
        {
            "generation": "g1",
            "mode": "isolated",
            "artifacts": [
                {
                    "logical_id": "data",
                    "candidate": observe_artifact(candidate),
                    "target": str(target),
                    "previous": None,
                    "retained": None,
                }
            ],
        },
    )
    journal.record("publication_started", {})
    publish_new(candidate, target)
    assert journal.artifact_states() == {"data": "published"}
    assert Journal(tmp_path, "op").recover() == "recovery_required"
    (target / "data").write_bytes(b"unreviewed change")
    assert journal.artifact_states() == {"data": "uncertain"}


def test_unrecorded_directory_child_invalidates_artifact(tmp_path):
    from tldw_chatbook.Backup_Recovery.journal import Journal, observe_artifact

    source = tmp_path / "source"
    source.mkdir(mode=0o700)
    (source / "one").write_bytes(b"one")
    journal = Journal(tmp_path, "op")
    journal.record(
        "prepared",
        {
            "generation": "g",
            "mode": "isolated",
            "artifacts": [
                {
                    "logical_id": "tree",
                    "candidate": observe_artifact(source),
                    "target": str(tmp_path / "target"),
                    "previous": None,
                    "retained": None,
                }
            ],
        },
    )
    (source / "two").write_bytes(b"two")
    assert journal.artifact_states() == {"tree": "uncertain"}


def test_incomplete_publication_remains_recovery_required(tmp_path):
    from tldw_chatbook.Backup_Recovery.journal import Journal

    journal = Journal(tmp_path, "operation-1")
    journal.record("prepared", {"generation": "g1", "mode": "isolated"})
    journal.record("publication_started", {})
    assert Journal(tmp_path, "operation-1").recover() == "recovery_required"


def test_prepared_record_survives_fresh_reader(tmp_path):
    from tldw_chatbook.Backup_Recovery.journal import Journal

    Journal(tmp_path, "operation-1").record(
        "prepared", {"generation": "g1", "mode": "isolated"}
    )
    assert Journal(tmp_path, "operation-1").recover() == "prepared"


@pytest.mark.parametrize("event", ["committed", "artifact_published", "unknown"])
def test_event_cannot_skip_preparation(tmp_path, event):
    from tldw_chatbook.Backup_Recovery.journal import Journal

    journal = Journal(tmp_path, "operation-1")
    with pytest.raises(ValueError, match="journal_transition_invalid"):
        journal.record(event, {})


def test_journal_refuses_secret_fields(tmp_path):
    from tldw_chatbook.Backup_Recovery.journal import Journal

    with pytest.raises(ValueError, match="journal_evidence_invalid"):
        Journal(tmp_path, "operation-1").record(
            "prepared",
            {"generation": "g1", "mode": "isolated", "password": "never-persist"},
        )
    assert all(
        b"never-persist" not in path.read_bytes() for path in tmp_path.rglob("*.json")
    )


def test_preparation_refuses_retained_path_without_previous_object(tmp_path):
    from tldw_chatbook.Backup_Recovery.journal import Journal, observe_artifact

    candidate = tmp_path / "candidate"
    candidate.write_bytes(b"candidate bytes")
    journal = Journal(tmp_path, "op")
    with pytest.raises(ValueError, match="journal_evidence_invalid"):
        journal.record(
            "prepared",
            {
                "generation": "g",
                "mode": "isolated",
                "artifacts": [
                    {
                        "logical_id": "data",
                        "candidate": observe_artifact(candidate),
                        "target": str(tmp_path / "target"),
                        "previous": None,
                        "retained": str(tmp_path / "unexpected-retained"),
                    }
                ],
            },
        )
    assert candidate.read_bytes() == b"candidate bytes"
    assert not (journal.root / "000000.json").exists()


def test_tree_observation_refuses_earlier_child_changed_during_later_read(
    tmp_path, monkeypatch
):
    from tldw_chatbook.Backup_Recovery import journal

    candidate = tmp_path / "candidate"
    candidate.mkdir(mode=0o700)
    first = candidate / "a"
    second = candidate / "b"
    first.write_bytes(b"original")
    second.write_bytes(b"later child")
    later_identity = second.stat().st_dev, second.stat().st_ino
    original_read = os.read
    changed = False

    def read(fd, size):
        nonlocal changed
        info = os.fstat(fd)
        if not changed and (info.st_dev, info.st_ino) == later_identity:
            # Sorted traversal already read/closed a. In-place mutation does
            # not change the directory names or its mtime, and keeps a's size.
            first.write_bytes(b"modified")
            changed = True
        return original_read(fd, size)

    monkeypatch.setattr(journal.os, "read", read)
    with pytest.raises(ValueError, match="artifact_changed"):
        journal.observe_artifact(candidate)
    assert changed
    assert first.read_bytes() == b"modified"
    assert second.read_bytes() == b"later child"


def _publication(tmp_path, *, pending=True, nested=False):
    from threading import Event

    from Tests.Backup_Recovery.test_restore_plan import sealed
    from tldw_chatbook.Backup_Recovery.control_records import register_pending
    from tldw_chatbook.Backup_Recovery.journal import Journal
    from tldw_chatbook.Backup_Recovery.restore_plan import plan_restore
    from tldw_chatbook.Backup_Recovery.staging import stage_restore

    archive = sealed(tmp_path)
    target = tmp_path / ("parents/nested/installed" if nested else "installed")
    plan = plan_restore(
        archive, mode="isolated", destinations={"root": target}, target=None
    )
    control = tmp_path / "control"
    control.mkdir(mode=0o700)
    journal = Journal(control, "publish-op")
    candidate = stage_restore(
        archive, plan, tmp_path / "stage", Event(), journal=journal
    )
    bootstrap = tmp_path / "bootstrap"
    selector = target / "config.toml"
    if pending:
        register_pending(bootstrap, "publish-op", ("profile",), control, (selector,))
    return candidate, plan, journal, bootstrap, selector


@pytest.mark.parametrize("boundary", ["prepared", "publication_started", "pending"])
def test_publication_retries_failed_durable_record_barrier(
    tmp_path, monkeypatch, boundary
):
    from tldw_chatbook.Backup_Recovery import publication
    from tldw_chatbook.Backup_Recovery.bootstrap import _key
    from tldw_chatbook.Backup_Recovery.control_records import register_pending

    candidate, plan, journal, bootstrap, selector = _publication(
        tmp_path, pending=boundary != "pending"
    )
    record_path = (
        bootstrap / f"pending-{_key('publish-op')}.json"
        if boundary == "pending"
        else journal.root / ("000001.json" if boundary == "prepared" else "000002.json")
    )
    native_fsync = os.fsync
    failing = True

    def fail_barrier(fd):
        if failing and record_path.exists():
            actual, expected = os.fstat(fd), record_path.stat()
            if (actual.st_dev, actual.st_ino) == (expected.st_dev, expected.st_ino):
                raise OSError("injected_record_barrier")
        native_fsync(fd)

    monkeypatch.setattr(os, "fsync", fail_barrier)
    if boundary == "pending":
        with pytest.raises(OSError, match="injected_record_barrier"):
            register_pending(
                bootstrap, "publish-op", ("profile",), journal.root.parent, (selector,)
            )
    preparation = {
        "bootstrap_root": bootstrap,
        "namespaces": ("profile",),
        "selectors": (selector,),
        "generation": "new",
    }
    if boundary == "prepared":
        with pytest.raises(OSError, match="injected_record_barrier"):
            journal.prepare_publication(candidate, plan, **preparation)
    else:
        journal.prepare_publication(candidate, plan, **preparation)
    if boundary == "publication_started":
        with pytest.raises(OSError, match="injected_record_barrier"):
            publication.publish_candidate(candidate, plan, journal, None)
    assert record_path.is_file()
    with pytest.raises(OSError, match="injected_record_barrier"):
        publication.publish_candidate(candidate, plan, journal, None)
    assert not dict(plan.destinations)["root"].exists()
    failing = False
    publication.publish_candidate(candidate, plan, journal, None)
    assert all(value == "published" for value in journal.artifact_states().values())
    assert (dict(plan.destinations)["root"] / "note.txt").read_bytes() == b"durable"


def test_started_publication_refuses_replaced_reviewed_parent(tmp_path):
    from tldw_chatbook.Backup_Recovery.publication import publish_candidate

    parent = tmp_path / "parents" / "nested"
    parent.mkdir(parents=True)
    candidate, plan, journal, bootstrap, selector = _publication(tmp_path, nested=True)
    journal.prepare_publication(
        candidate,
        plan,
        bootstrap_root=bootstrap,
        namespaces=("profile",),
        selectors=(selector,),
        generation="new",
    )
    journal.record("publication_started", {})
    parent.rename(parent.with_name("original-parent"))
    parent.mkdir()
    with pytest.raises(ValueError, match="publication_parent_changed"):
        publish_candidate(candidate, plan, journal, None)
    assert not dict(plan.destinations)["root"].exists()


def test_native_publication_rechecks_reviewed_parent_after_preflight(
    tmp_path, monkeypatch
):
    from tldw_chatbook.Backup_Recovery import publication

    parent = tmp_path / "parents" / "nested"
    parent.mkdir(parents=True)
    candidate, plan, journal, bootstrap, selector = _publication(tmp_path, nested=True)
    journal.prepare_publication(
        candidate,
        plan,
        bootstrap_root=bootstrap,
        namespaces=("profile",),
        selectors=(selector,),
        generation="new",
    )
    native = publication.publish_new

    def replaced(source, target, **kwargs):
        parent.rename(parent.with_name("original-parent"))
        parent.mkdir()
        native(source, target, **kwargs)

    monkeypatch.setattr(publication, "publish_new", replaced)
    with pytest.raises(OSError, match="publication_parent_changed"):
        publication.publish_candidate(candidate, plan, journal, None)
    assert not dict(plan.destinations)["root"].exists()


def test_publication_requires_fixed_pending_association(tmp_path):
    candidate, plan, journal, bootstrap, selector = _publication(
        tmp_path, pending=False
    )
    with pytest.raises(ValueError, match="publication_pending_mismatch"):
        journal.prepare_publication(
            candidate,
            plan,
            bootstrap_root=bootstrap,
            namespaces=("profile",),
            selectors=(selector,),
            generation="g",
        )
    assert not dict(plan.destinations)["root"].exists()


def test_real_isolated_publication_retains_recovery_fence(tmp_path):
    from tldw_chatbook.Backup_Recovery.bootstrap import startup_permission
    from tldw_chatbook.Backup_Recovery.publication import publish_candidate

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
    assert (dict(plan.destinations)["root"] / "note.txt").read_bytes() == b"durable"
    assert set(journal.artifact_states().values()) == {"published"}
    assert journal.recover() == "recovery_required"
    assert startup_permission(selector, bootstrap) == (False, "recovery_pending")


@pytest.mark.parametrize("damage", ["descriptor", "candidate", "target", "pointer"])
def test_prepared_publication_rechecks_local_evidence(tmp_path, damage):
    from tldw_chatbook.Backup_Recovery.publication import publish_candidate

    candidate, plan, journal, bootstrap, selector = _publication(tmp_path)
    journal.prepare_publication(
        candidate,
        plan,
        bootstrap_root=bootstrap,
        namespaces=("profile",),
        selectors=(selector,),
        generation="g",
    )
    target = dict(plan.destinations)["root"]
    if damage == "descriptor":
        (candidate / "candidate.json").write_bytes(b"{}")
    elif damage == "candidate":
        descriptor = json.loads((candidate / "candidate.json").read_bytes())
        row = next(row for row in descriptor["artifacts"] if row["kind"] == "file")
        Path(row["candidate"]).write_bytes(b"changed")
    elif damage == "target":
        target.mkdir()
        (target / "keep").write_bytes(b"unreviewed")
    else:
        next(bootstrap.glob("pending-*.json")).unlink()
    with pytest.raises((ValueError, OSError)):
        publish_candidate(candidate, plan, journal, None)
    if damage == "target":
        assert (target / "keep").read_bytes() == b"unreviewed"
    else:
        assert not target.exists()


def test_stage_receipt_refuses_self_consistent_descriptor_rewrite(tmp_path):
    import hashlib

    candidate, plan, journal, bootstrap, selector = _publication(tmp_path)
    descriptor_path = candidate / "candidate.json"
    document = json.loads(descriptor_path.read_bytes())
    row = next(row for row in document["artifacts"] if row["kind"] == "file")
    path = Path(row["candidate"])
    path.write_bytes(b"forged but consistent")
    info = path.stat()
    row.update(
        size=info.st_size,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        identity=[info.st_dev, info.st_ino, info.st_mode, info.st_mtime_ns],
    )
    descriptor_path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="candidate_receipt_changed"):
        journal.prepare_publication(
            candidate,
            plan,
            bootstrap_root=bootstrap,
            namespaces=("profile",),
            selectors=(selector,),
            generation="g",
        )
    assert not dict(plan.destinations)["root"].exists()


@pytest.mark.parametrize(
    "boundary", ["publication_started", "rename", "artifact_published"]
)
def test_real_publication_process_exit_keeps_bytes_and_fence(tmp_path, boundary):
    from tldw_chatbook.Backup_Recovery.bootstrap import startup_permission
    from tldw_chatbook.Backup_Recovery.journal import Journal

    script = """
import os, sys
from pathlib import Path
from Tests import network_guard
network_guard.install()
from Tests.Backup_Recovery.test_publication_crashes import _publication
from tldw_chatbook.Backup_Recovery import publication
root = Path(sys.argv[1])
boundary = sys.argv[2]
candidate, plan, journal, bootstrap, selector = _publication(root)
journal.prepare_publication(candidate, plan, bootstrap_root=bootstrap,
    namespaces=("profile",), selectors=(selector,), generation="g")
append = journal._append
def record(parent, event, evidence):
    append(parent, event, evidence)
    if event == boundary: os._exit(73)
journal._append = record
rename = publication.publish_new
def publish(source, target, **kwargs):
    rename(source, target, **kwargs)
    if boundary == "rename": os._exit(73)
publication.publish_new = publish
publication.publish_candidate(candidate, plan, journal, None)
raise AssertionError("boundary not reached")
"""
    child = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path), boundary],
        timeout=30,
        check=False,
        capture_output=True,
    )
    assert child.returncode == 73, child.stderr.decode()[-4000:]
    journal = Journal(tmp_path / "control", "publish-op")
    states = journal.artifact_states()
    assert set(states.values()) == (
        {"staged"} if boundary == "publication_started" else {"published"}
    )
    assert journal.recover() == "recovery_required"
    target = tmp_path / "installed"
    if boundary != "publication_started":
        assert (target / "note.txt").read_bytes() == b"durable"
    assert startup_permission(target / "config.toml", tmp_path / "bootstrap") == (
        False,
        "recovery_pending",
    )


def _replacement(tmp_path, *, tree=False):
    from threading import Event

    from Tests.Backup_Recovery.test_restore_plan import producer, sealed
    from tldw_chatbook.Backup_Recovery.control_records import register_pending
    from tldw_chatbook.Backup_Recovery.journal import Journal
    from tldw_chatbook.Backup_Recovery.models import Inventory, StorageItem
    from tldw_chatbook.Backup_Recovery.restore_plan import plan_restore
    from tldw_chatbook.Backup_Recovery.staging import stage_restore

    archive = sealed(tmp_path, mutate=producer)
    target = tmp_path / "live"
    target.mkdir(mode=0o700)
    previous = target / "note.txt"
    previous.write_bytes(b"original bytes")
    previous.chmod(0o640)
    os.utime(previous, ns=(123000000000, 123000000000))
    items = [
        StorageItem("ui.state", "root", target, "included_directory", ()),
        StorageItem("ui.state", "file", previous, "included", ("root",)),
    ]
    if tree:
        old = target / "obsolete"
        old.mkdir(mode=0o750)
        (old / "empty").mkdir(mode=0o700)
        (old / "data").write_bytes(b"old subtree")
        (old / "data").chmod(0o640)
        items.extend(
            [
                StorageItem("ui.state", "old", old, "included_directory", ("root",)),
                StorageItem(
                    "ui.state",
                    "old-empty",
                    old / "empty",
                    "included_directory",
                    ("old",),
                ),
                StorageItem("ui.state", "old-data", old / "data", "included", ("old",)),
            ]
        )
    inventory = Inventory(tuple(items), True, "local", ())
    plan = plan_restore(
        archive, mode="replace", destinations={"root": target}, target=inventory
    )
    control = tmp_path / "control"
    control.mkdir(mode=0o700)
    journal = Journal(control, "replace-op")
    candidate = stage_restore(
        archive, plan, tmp_path / "stage", Event(), journal=journal
    )
    selector = target / "config.toml"
    bootstrap = tmp_path / "bootstrap"
    register_pending(bootstrap, "replace-op", ("profile",), control, (selector,))
    journal.prepare_publication(
        candidate,
        plan,
        bootstrap_root=bootstrap,
        namespaces=("profile",),
        selectors=(selector,),
        generation="g",
    )
    return candidate, plan, journal, previous


def test_replacement_refuses_unverified_archive_filename(tmp_path):
    from tldw_chatbook.Backup_Recovery.publication import publish_candidate

    candidate, plan, journal, previous = _replacement(tmp_path)
    fake = tmp_path / "rollback.age"
    fake.write_bytes(b"not verified")
    with pytest.raises(ValueError, match="rollback_required"):
        publish_candidate(candidate, plan, journal, fake)
    assert previous.read_bytes() == b"original bytes"


def test_publication_uses_only_explicit_parent_containers(tmp_path):
    from tldw_chatbook.Backup_Recovery.publication import publish_candidate

    candidate, plan, journal, bootstrap, selector = _publication(tmp_path, nested=True)
    journal.prepare_publication(
        candidate,
        plan,
        bootstrap_root=bootstrap,
        namespaces=("profile",),
        selectors=(selector,),
        generation="g",
    )
    assert not (tmp_path / "parents").exists()
    publish_candidate(candidate, plan, journal, None)
    assert (dict(plan.destinations)["root"] / "note.txt").read_bytes() == b"durable"
    assert set(journal.artifact_states().values()) == {"published"}
    (tmp_path / "parents" / "unexpected").write_bytes(b"unreviewed")
    assert "uncertain" in journal.artifact_states().values()


def _encrypted_rollback(
    tmp_path, previous, helper_resource_root, monkeypatch, *, data=None
):
    from threading import Event

    from Tests.Backup_Recovery.test_archive_writer import captured
    from tldw_chatbook.Backup_Recovery import crypto
    from tldw_chatbook.Backup_Recovery.archive_writer import write_archive
    from tldw_chatbook.Backup_Recovery.capture import CaptureResult

    monkeypatch.setattr(crypto, "_package_resource_root", lambda: helper_resource_root)
    capture = captured(tmp_path, previous.read_bytes() if data is None else data)
    document = json.loads(capture.manifest_bytes)
    document["credential_policy"] = "rollback"
    info = previous.stat()
    document["files"][0]["metadata"] = {
        "version": 1,
        "mode": info.st_mode & 0o777,
        "mtime_ns": info.st_mtime_ns,
    }
    tree = previous.parent / "obsolete"
    if tree.exists():
        import hashlib

        for key, source, relative, parent_id in [
            ("old", tree, "obsolete", "root"),
            ("old-empty", tree / "empty", "obsolete/empty", "old"),
        ]:
            info = source.stat()
            document["directories"].append(
                {
                    "logical_id": key,
                    "root_id": "root",
                    "parent_id": parent_id,
                    "relative_path": relative,
                    "metadata": {
                        "version": 1,
                        "mode": info.st_mode & 0o777,
                        "mtime_ns": info.st_mtime_ns,
                    },
                }
            )
        source = tree / "data"
        payload = source.read_bytes()
        (capture.root / "payload/2").write_bytes(payload)
        (capture.root / "payload/2").chmod(0o600)
        info = source.stat()
        document["files"].append(
            {
                "logical_id": "old-data",
                "root_id": "root",
                "parent_id": "old",
                "relative_path": "obsolete/data",
                "owner_id": "notes",
                "payload": "payload/2",
                "size": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "metadata": {
                    "version": 1,
                    "mode": info.st_mode & 0o777,
                    "mtime_ns": info.st_mtime_ns,
                },
            }
        )
        document["dependency_groups"][0]["members"].append("old-data")
    root_info = previous.parent.stat()
    document["directories"][0]["metadata"] = {
        "version": 1,
        "mode": root_info.st_mode & 0o777,
        "mtime_ns": root_info.st_mtime_ns,
    }
    document["owners"].append(
        {"owner_id": "ui.state", "schema_version": 0, "capabilities": []}
    )
    document["producer_inventory"] = [
        {
            "logical_id": row["logical_id"],
            "owner_id": "ui.state" if row["logical_id"] == "root" else "notes",
            "status": "included_directory",
            "dependencies": [row["parent_id"]] if row["parent_id"] else [],
            "shared_group": None,
        }
        for row in document["directories"]
    ] + [
        {
            "logical_id": row["logical_id"],
            "owner_id": row["owner_id"],
            "status": "included",
            "dependencies": [row["parent_id"]],
            "shared_group": None,
        }
        for row in document["files"]
    ]
    capture = CaptureResult(
        capture.root, capture.inventory, json.dumps(document).encode()
    )
    path = tmp_path / "rollback.tldw-backup.zip.age"
    write_archive(capture, path, password=b"test-only-password", cancel=Event())
    return path


@pytest.mark.parametrize("case", ["exact", "unrelated", "changed", "wrong_password"])
def test_authenticated_raw_rollback_binds_actual_coverage(
    tmp_path, helper_resource_root, monkeypatch, case
):
    from threading import Event

    from tldw_chatbook.Backup_Recovery import crypto
    from tldw_chatbook.Backup_Recovery.publication import publish_candidate

    candidate, plan, journal, previous = _replacement(tmp_path)
    archive = _encrypted_rollback(
        tmp_path,
        previous,
        helper_resource_root,
        monkeypatch,
        data=b"unrelated original" if case == "unrelated" else None,
    )
    if case in {"unrelated", "wrong_password"}:
        with pytest.raises((ValueError, crypto.CryptoError)):
            journal.verify_rollback(
                archive,
                password=b"wrong"
                if case == "wrong_password"
                else b"test-only-password",
                work_root=tmp_path / "verify",
                cancel=Event(),
                coverage={"file": "file", "root": "root"},
            )
        assert previous.read_bytes() == b"original bytes"
        with pytest.raises(ValueError, match="rollback_required"):
            publish_candidate(candidate, plan, journal, archive)
        return
    journal.verify_rollback(
        archive,
        password=b"test-only-password",
        work_root=tmp_path / "verify",
        cancel=Event(),
        coverage={"file": "file", "root": "root"},
    )
    if case == "changed":
        raw = archive.read_bytes()
        archive.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
        with pytest.raises(ValueError, match="rollback_ciphertext_changed"):
            publish_candidate(candidate, plan, journal, archive)
        assert previous.read_bytes() == b"original bytes"
        return
    publish_candidate(candidate, plan, journal, archive)
    with journal._locked(exclusive=False) as fd:
        records = journal._records(fd)
    prepared = next(row for row in records if row.event == "prepared")
    item = next(
        row for row in prepared.evidence["artifacts"] if row["logical_id"] == "file"
    )
    retained = Path(item["retained"])
    assert retained.read_bytes() == b"original bytes"
    assert retained.stat().st_mode & 0o777 == 0o640
    assert retained.stat().st_mtime_ns == 123000000000
    assert previous.read_bytes() == b"durable"
    assert journal.recover() == "recovery_required"


def test_authenticated_tree_retirement_keeps_original_metadata(
    tmp_path, helper_resource_root, monkeypatch
):
    from threading import Event

    from tldw_chatbook.Backup_Recovery.journal import observe_artifact
    from tldw_chatbook.Backup_Recovery.publication import publish_candidate

    candidate, plan, journal, previous = _replacement(tmp_path, tree=True)
    old = previous.parent / "obsolete"
    expected = observe_artifact(old, metadata=True)
    archive = _encrypted_rollback(tmp_path, previous, helper_resource_root, monkeypatch)
    journal.verify_rollback(
        archive,
        password=b"test-only-password",
        work_root=tmp_path / "verify",
        cancel=Event(),
        coverage={"file": "file", "old": "old", "root": "root"},
    )
    publish_candidate(candidate, plan, journal, archive)
    with journal._locked(exclusive=False) as fd:
        records = journal._records(fd)
    prepared = next(row for row in records if row.event == "prepared")
    retained = Path(
        next(
            row["retained"]
            for row in prepared.evidence["artifacts"]
            if row["logical_id"] == "old"
        )
    )
    actual = observe_artifact(retained, metadata=True)
    assert {k: v for k, v in actual.items() if k != "path"} == {
        k: v for k, v in expected.items() if k != "path"
    }
    assert not old.exists()
    assert journal.artifact_states()["old"] == "retired"


def test_process_exit_after_original_retirement_retains_both_generations(
    tmp_path, helper_resource_root
):
    from tldw_chatbook.Backup_Recovery.bootstrap import startup_permission
    from tldw_chatbook.Backup_Recovery.journal import Journal

    script = """
import os, sys
from pathlib import Path
from threading import Event
from Tests import network_guard
network_guard.install()
import pytest
from Tests.Backup_Recovery.test_publication_crashes import _replacement, _encrypted_rollback
from tldw_chatbook.Backup_Recovery import publication
root = Path(sys.argv[1])
candidate, plan, journal, previous = _replacement(root)
archive = _encrypted_rollback(root, previous, Path(sys.argv[2]), pytest.MonkeyPatch())
journal.verify_rollback(archive, password=b"test-only-password", work_root=root/"verify",
    cancel=Event(), coverage={"file":"file", "root":"root"})
retire = publication._retire
def crash(item):
    retire(item)
    os._exit(73)
publication._retire = crash
publication.publish_candidate(candidate, plan, journal, archive)
"""
    child = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path), str(helper_resource_root)],
        timeout=40,
        capture_output=True,
        check=False,
    )
    assert child.returncode == 73, child.stderr.decode()[-4000:]
    journal = Journal(tmp_path / "control", "replace-op")
    assert journal.artifact_states() == {"file": "retired"}
    assert journal.recover() == "recovery_required"
    with journal._locked(exclusive=False) as fd:
        records = journal._records(fd)
    item = next(row for row in records if row.event == "prepared").evidence[
        "artifacts"
    ][0]
    assert Path(item["retained"]).read_bytes() == b"original bytes"
    assert Path(item["candidate"]["path"]).read_bytes() == b"durable"
    assert not (tmp_path / "live/note.txt").exists()
    assert startup_permission(
        tmp_path / "live/config.toml", tmp_path / "bootstrap"
    ) == (False, "recovery_pending")


@pytest.mark.parametrize("selected", ["other", "affected"])
def test_publication_pending_must_cover_actual_enrolled_profile(tmp_path, selected):
    from threading import Event

    from Tests.Backup_Recovery.test_restore_plan import sealed
    from tldw_chatbook.Backup_Recovery.bootstrap import startup_permission
    from tldw_chatbook.Backup_Recovery.control_records import (
        admission_authority,
        bind_profile,
        register_pending,
    )
    from tldw_chatbook.Backup_Recovery.journal import Journal
    from tldw_chatbook.Backup_Recovery.restore_plan import plan_restore
    from tldw_chatbook.Backup_Recovery.staging import stage_restore

    bootstrap = tmp_path / "bootstrap"
    authority = admission_authority(bootstrap)
    selectors = {}
    for name in ("affected", "other"):
        data = tmp_path / name
        data.mkdir(mode=0o700)
        selector = tmp_path / (name + ".toml")
        selector.write_text("profile = " + repr(name))
        authority.register(name, (data, selector))
        bind_profile(bootstrap, selector, (name,), bootstrap / "admission")
        selectors[name] = selector
    archive = sealed(tmp_path)
    destination = tmp_path / "affected" / "restored"
    plan = plan_restore(
        archive, mode="isolated", destinations={"root": destination}, target=None
    )
    control = tmp_path / "control"
    control.mkdir(mode=0o700)
    journal = Journal(control, "unrelated-scope")
    stage = stage_restore(archive, plan, tmp_path / "stage", Event(), journal=journal)
    register_pending(
        bootstrap, journal.operation_id, (selected,), control, (selectors[selected],)
    )
    # This is an actual admitted independent profile, unlike the conservative
    # no-registry case where any pending operation blocks every unbound startup.
    assert startup_permission(selectors["affected"], bootstrap) == (
        (True, "startup_allowed")
        if selected == "other"
        else (False, "recovery_pending")
    )

    def prepare():
        journal.prepare_publication(
            stage,
            plan,
            bootstrap_root=bootstrap,
            namespaces=(selected,),
            selectors=(selectors[selected],),
            generation="generation",
        )

    if selected == "other":
        with pytest.raises(ValueError, match="publication_scope_uncovered"):
            prepare()
        assert not destination.exists()
    else:
        from tldw_chatbook.Backup_Recovery.publication import publish_candidate

        prepare()
        publish_candidate(stage, plan, journal, None)
        assert (destination / "note.txt").read_bytes() == b"durable"
