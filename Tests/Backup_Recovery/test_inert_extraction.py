"""Manual extraction copies selected untrusted bytes without opening their schema."""

import hashlib
import json
import sqlite3
import zipfile
from dataclasses import FrozenInstanceError
from threading import Event

import pytest

from Tests.Backup_Recovery.test_archive_reader import manifest
from tldw_chatbook.Backup_Recovery import archive_reader
from tldw_chatbook.Backup_Recovery.limits import ArchiveLimits


def _archive(tmp_path, *, encrypted=False):
    db_path = tmp_path / "unexpected.sqlite"
    with sqlite3.connect(db_path) as db:
        db.executescript(
            "CREATE TABLE recovered(value); CREATE TRIGGER unexpected AFTER INSERT ON recovered BEGIN SELECT load_extension('never'); END;"
        )
    payloads = {
        "db": db_path.read_bytes(),
        "config": b"broken = [secret-value",
        "other": b"not selected",
    }
    doc = manifest()
    doc.update(
        consistency="partial",
        owners=[
            {"owner_id": "unknown.newer", "schema_version": 999, "capabilities": []}
        ],
    )
    doc["files"] = [
        {
            "logical_id": key,
            "root_id": "root",
            "parent_id": "root",
            "relative_path": key + ".sqlite",
            "owner_id": "unknown.newer",
            "payload": "payload/" + key,
            "size": len(value),
            "sha256": hashlib.sha256(value).hexdigest(),
        }
        for key, value in payloads.items()
    ]
    doc["dependency_groups"] = [
        {"group_id": "unsupported", "members": ["db", "config"], "complete": False},
        {"group_id": "other", "members": ["other"], "complete": True},
    ]
    if encrypted:
        doc["credential_policy"] = "include"
        doc["relocations"] = [
            {"logical_id": "config", "locator": "/private/secret-original-locator"}
        ]
    source = tmp_path / "archive.zip"
    with zipfile.ZipFile(source, "w") as output:
        output.writestr("manifest.json", json.dumps(doc))
        for key, value in payloads.items():
            output.writestr("payload/" + key, value)
    if encrypted:
        from tldw_chatbook.Backup_Recovery import crypto

        protected = tmp_path / "archive.zip.age"
        crypto.transform(
            source,
            protected,
            password=b"private-password",
            decrypt=False,
            cancel=Event(),
        )
        source = protected
    archive = archive_reader.acquire(
        source,
        tmp_path / "acquisition",
        ArchiveLimits(),
        b"private-password" if encrypted else None,
        Event(),
    )
    return archive, payloads


def test_extracts_only_selected_unsupported_bytes_without_database_execution(
    tmp_path, monkeypatch
):
    archive, payloads = _archive(tmp_path)
    from tldw_chatbook.Backup_Recovery.inert_extraction import (
        extract_inert,
        preview_inert_extraction,
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("inert extraction opened a database")

    monkeypatch.setattr(sqlite3, "connect", forbidden)
    destination = tmp_path / "manual"
    plan = preview_inert_extraction(
        archive,
        group_ids=("unsupported",),
        destination=destination,
        limits=ArchiveLimits(),
        cancel=Event(),
    )
    with pytest.raises(FrozenInstanceError):
        plan.destination = tmp_path / "changed"
    assert not destination.exists()
    result = extract_inert(archive, plan, cancel=Event())
    assert result.state == "inert_extracted"
    report = json.loads(result.report_path.read_text())
    assert report["groups"] == [{"group_id": "unsupported", "complete": False}]
    assert {row["logical_id"] for row in report["files"]} == {"db", "config"}
    for row in report["files"]:
        path = destination / row["output"]
        assert (
            path.suffix == ".bin" and path.read_bytes() == payloads[row["logical_id"]]
        )
        assert path.stat().st_mode & 0o777 == 0o600
    assert "secret-value" not in result.report_path.read_text()
    assert not report.get("restoration_validated", False)


@pytest.mark.parametrize(
    "change", ["source", "parent", "destination", "cancel", "capacity"]
)
def test_execution_refuses_changed_review_without_publishing(
    tmp_path, monkeypatch, change
):
    from collections import namedtuple

    from tldw_chatbook.Backup_Recovery import inert_extraction as inert
    from tldw_chatbook.Backup_Recovery import space

    archive, _ = _archive(tmp_path)
    parent = tmp_path / "output-parent"
    parent.mkdir(mode=0o700)
    destination = parent / "manual"
    cancel = Event()
    plan = inert.preview_inert_extraction(
        archive,
        group_ids=("unsupported",),
        destination=destination,
        limits=ArchiveLimits(),
        cancel=cancel,
    )
    if change == "source":
        archive.path.write_bytes(b"changed")
    elif change == "parent":
        parent.rename(tmp_path / "old-parent")
        parent.mkdir(mode=0o700)
    elif change == "destination":
        destination.write_bytes(b"foreign")
    elif change == "cancel":
        cancel.set()
    else:
        usage = namedtuple("usage", "total used free")
        monkeypatch.setattr(space.shutil, "disk_usage", lambda path: usage(1, 1, 0))
    with pytest.raises((ValueError, OSError, InterruptedError)):
        inert.extract_inert(archive, plan, cancel=cancel)
    assert (
        destination.read_bytes() == b"foreign"
        if change == "destination"
        else not destination.exists()
    )


@pytest.mark.parametrize(
    "kind", ["source", "bootstrap", "default_service", "custom", "link"]
)
def test_refuses_protected_or_linked_output_without_registry_reads(
    tmp_path, monkeypatch, kind
):
    from tldw_chatbook.Backup_Recovery import (
        bootstrap,
        service_storage,
    )
    from tldw_chatbook.Backup_Recovery import (
        inert_extraction as inert,
    )

    archive, _ = _archive(tmp_path)
    fixed = tmp_path / "fixed"
    monkeypatch.setattr(bootstrap, "default_bootstrap_root", lambda: fixed)
    monkeypatch.setattr(
        service_storage,
        "default_control_root",
        lambda: tmp_path / "service" / "control",
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("manual extraction consulted managed registry")

    monkeypatch.setattr(bootstrap, "_registry", forbidden)
    protected = ()
    if kind == "source":
        destination = archive.path.parent / "manual"
    elif kind == "bootstrap":
        destination = fixed / "manual"
    elif kind == "default_service":
        destination = tmp_path / "service" / "manual"
    elif kind == "custom":
        destination = tmp_path / "custom" / "manual"
        protected = (tmp_path / "custom",)
    else:
        (tmp_path / "linked").symlink_to(tmp_path, target_is_directory=True)
        destination = tmp_path / "linked" / "manual"
    with pytest.raises((ValueError, OSError)):
        inert.preview_inert_extraction(
            archive,
            group_ids=("unsupported",),
            destination=destination,
            protected_roots=protected,
            limits=ArchiveLimits(),
            cancel=Event(),
        )


def test_native_destination_race_never_replaces_foreign_name(tmp_path, monkeypatch):
    from tldw_chatbook.Backup_Recovery import inert_extraction as inert

    archive, _ = _archive(tmp_path)
    destination = tmp_path / "manual"
    plan = inert.preview_inert_extraction(
        archive,
        group_ids=("unsupported",),
        destination=destination,
        limits=ArchiveLimits(),
        cancel=Event(),
    )
    publish = inert.publish_new

    def race(*args, **kwargs):
        destination.write_bytes(b"foreign")
        return publish(*args, **kwargs)

    monkeypatch.setattr(inert, "publish_new", race)
    with pytest.raises(FileExistsError):
        inert.extract_inert(archive, plan, cancel=Event())
    assert destination.read_bytes() == b"foreign"
    assert list(tmp_path.glob(".inert-extraction-*/publication-intent.json"))


def test_failure_after_actual_native_publish_retains_intent_and_published_bytes(
    tmp_path, monkeypatch
):
    from tldw_chatbook.Backup_Recovery import inert_extraction as inert

    archive, payloads = _archive(tmp_path)
    destination = tmp_path / "manual"
    plan = inert.preview_inert_extraction(
        archive,
        group_ids=("unsupported",),
        destination=destination,
        limits=ArchiveLimits(),
        cancel=Event(),
    )
    publish = inert.publish_new

    def interrupted(*args, **kwargs):
        publish(*args, **kwargs)
        raise OSError("interrupted native completion")

    monkeypatch.setattr(inert, "publish_new", interrupted)
    with pytest.raises(OSError):
        inert.extract_inert(archive, plan, cancel=Event())
    report = json.loads((destination / "inert-mapping.json").read_text())
    assert all(
        (destination / row["output"]).read_bytes() == payloads[row["logical_id"]]
        for row in report["files"]
    )
    assert list(tmp_path.glob(".inert-extraction-*/publication-intent.json"))


def test_renewed_limits_refuse_oversized_input_before_output(tmp_path):
    from tldw_chatbook.Backup_Recovery import inert_extraction as inert

    archive, _ = _archive(tmp_path)
    with pytest.raises(ValueError, match="input_limit|decrypted_limit"):
        inert.preview_inert_extraction(
            archive,
            group_ids=("unsupported",),
            destination=tmp_path / "manual",
            limits=ArchiveLimits(input_bytes=1, decrypted_bytes=1),
            cancel=Event(),
        )
    assert not (tmp_path / "manual").exists()


def test_actual_encrypted_manual_copy_keeps_credentials_inert_and_report_nonsecret(
    tmp_path, monkeypatch, helper_resource_root
):
    import keyring

    from tldw_chatbook.Backup_Recovery import crypto
    from tldw_chatbook.Backup_Recovery import inert_extraction as inert

    monkeypatch.setattr(crypto, "_package_resource_root", lambda: helper_resource_root)
    archive, payloads = _archive(tmp_path, encrypted=True)

    def forbidden(*args, **kwargs):
        raise AssertionError("manual extraction touched keyring")

    for name in ("get_password", "set_password", "delete_password"):
        monkeypatch.setattr(keyring, name, forbidden)
    plan = inert.preview_inert_extraction(
        archive,
        group_ids=("unsupported",),
        destination=tmp_path / "manual",
        limits=ArchiveLimits(),
        cancel=Event(),
    )
    result = inert.extract_inert(archive, plan, cancel=Event())
    encoded = result.report_path.read_text()
    assert (
        "secret-original-locator" not in encoded
        and "secret-value" not in encoded
        and "private-password" not in encoded
    )
    report = json.loads(encoded)
    assert all(
        (result.destination / row["output"]).read_bytes() == payloads[row["logical_id"]]
        for row in report["files"]
    )


def test_fresh_manual_extraction_survives_damaged_registry_without_runtime_imports(
    tmp_path,
):
    import subprocess
    import sys

    _archive(tmp_path)
    script = r"""
import builtins,sys,sqlite3
from pathlib import Path
from threading import Event
from Tests.network_guard import install,blocked_attempts
install()
original=builtins.__import__
forbidden={'tldw_chatbook.app','tldw_chatbook.config','tldw_chatbook.Backup_Recovery.staging','tldw_chatbook.Backup_Recovery.owner_registry','tldw_chatbook.Backup_Recovery.activation'}
def guarded(name,*args,**kwargs):
 assert name not in forbidden,name
 return original(name,*args,**kwargs)
builtins.__import__=guarded
def no_database(*args,**kwargs):raise AssertionError('database opened')
sqlite3.connect=no_database
from tldw_chatbook.Backup_Recovery import bootstrap
from tldw_chatbook.Backup_Recovery.inert_extraction import preview_inert_extraction,extract_inert
from tldw_chatbook.Backup_Recovery.archive_reader import acquire
from tldw_chatbook.Backup_Recovery.limits import ArchiveLimits
root=Path(sys.argv[1]);fixed=root/'damaged-bootstrap';fixed.mkdir(mode=0o700)
(fixed/'registry.json').write_bytes(b'broken registry')
bootstrap.default_bootstrap_root=lambda:fixed
def no_registry(*args,**kwargs):raise AssertionError('registry consulted')
bootstrap._registry=no_registry
archive=acquire(root/'archive.zip',root/'child-acquisition',ArchiveLimits(),None,Event())
plan=preview_inert_extraction(archive,group_ids=('unsupported',),destination=root/'child-manual',limits=ArchiveLimits(),cancel=Event())
result=extract_inert(archive,plan,cancel=Event())
assert result.state=='inert_extracted'
assert (fixed/'registry.json').read_bytes()==b'broken registry'
assert not forbidden.intersection(sys.modules),forbidden.intersection(sys.modules)
assert not blocked_attempts(),blocked_attempts()
"""
    with (tmp_path / "manual-child.log").open("w+") as log:
        result = subprocess.run(
            [sys.executable, "-c", script, str(tmp_path)],
            stdout=log,
            stderr=log,
            timeout=20,
            check=False,
        )
        log.seek(0)
        assert result.returncode == 0, log.read()[-7000:]


@pytest.mark.parametrize("groups", [(), ("missing",), ("unsupported", "unsupported")])
def test_group_selection_is_explicit_and_nonexpanding(tmp_path, groups):
    from tldw_chatbook.Backup_Recovery import inert_extraction as inert

    archive, _ = _archive(tmp_path)
    with pytest.raises(ValueError):
        inert.preview_inert_extraction(
            archive,
            group_ids=groups,
            destination=tmp_path / "manual",
            limits=ArchiveLimits(),
            cancel=Event(),
        )
    assert not (tmp_path / "manual").exists()


def test_cancellation_after_real_payload_write_removes_only_owned_stage(
    tmp_path, monkeypatch
):
    from contextlib import contextmanager

    from tldw_chatbook.Backup_Recovery import inert_extraction as inert

    archive, _ = _archive(tmp_path)
    cancel = Event()
    destination = tmp_path / "manual"
    plan = inert.preview_inert_extraction(
        archive,
        group_ids=("unsupported",),
        destination=destination,
        limits=ArchiveLimits(),
        cancel=cancel,
    )
    create = inert.create_private_file
    written = []

    @contextmanager
    def cancel_after_payload(path):
        with create(path) as fd:
            yield fd
        if path.suffix == ".bin":
            written.append(path.stat().st_size)
            cancel.set()

    monkeypatch.setattr(inert, "create_private_file", cancel_after_payload)
    with pytest.raises(InterruptedError):
        inert.extract_inert(archive, plan, cancel=cancel)
    assert written and written[0] > 0
    assert not destination.exists() and not list(tmp_path.glob(".inert-extraction-*"))
    assert archive.path.is_file()


def test_overlapping_groups_copy_each_member_once_and_report_both(tmp_path):
    from tldw_chatbook.Backup_Recovery import inert_extraction as inert

    archive, payloads = _archive(tmp_path)
    doc = archive_reader.verify_sealed(archive).model_dump(mode="json")
    doc["dependency_groups"].append(
        {"group_id": "overlap", "members": ["db"], "complete": True}
    )
    source = tmp_path / "overlap.zip"
    with zipfile.ZipFile(source, "w") as output:
        output.writestr("manifest.json", json.dumps(doc))
        for key, value in payloads.items():
            output.writestr("payload/" + key, value)
    archive = archive_reader.acquire(
        source, tmp_path / "overlap-acquired", ArchiveLimits(), None, Event()
    )
    plan = inert.preview_inert_extraction(
        archive,
        group_ids=("unsupported", "overlap"),
        destination=tmp_path / "manual",
        limits=ArchiveLimits(),
        cancel=Event(),
    )
    result = inert.extract_inert(archive, plan, cancel=Event())
    report = json.loads(result.report_path.read_text())
    assert len(report["files"]) == 2 and len(report["groups"]) == 2
    assert len(list((result.destination / "payload").iterdir())) == 2
