"""Replacement captures semantic SQLite originals under uninterrupted admission."""

import hashlib
import json
import sqlite3
import subprocess
import sys
import zipfile
from contextlib import closing, contextmanager
from pathlib import Path
from threading import Event

import pytest

from Tests.Backup_Recovery.test_archive_reader import manifest
from Tests.Backup_Recovery.test_capture_sqlite_materialization import state
from tldw_chatbook.Backup_Recovery import bootstrap, publication
from tldw_chatbook.Backup_Recovery.archive_reader import acquire, verify_sealed
from tldw_chatbook.Backup_Recovery.control_records import (
    admission_authority,
    bind_profile,
    register_pending,
)
from tldw_chatbook.Backup_Recovery.journal import Journal
from tldw_chatbook.Backup_Recovery.limits import ArchiveLimits
from tldw_chatbook.Backup_Recovery.models import Inventory, StorageItem
from tldw_chatbook.Backup_Recovery.restore_plan import plan_restore
from tldw_chatbook.Backup_Recovery.staging import stage_restore
from tldw_chatbook.Research_Interop.recovery import recovery_adapters


@contextmanager
def replacement_case(
    tmp_path,
    monkeypatch,
    *,
    guard=True,
    tree=False,
    omit_shm=False,
    extras=None,
    missing_selector=False,
    prepared=True,
):
    owner = recovery_adapters()[0]
    live = tmp_path / "live"
    live.mkdir(mode=0o700)
    source = live / "research.db"
    selector = live / "config.toml"
    selector.write_text('[general]\nusers_name="old"\n')
    selector.chmod(0o600)
    with closing(sqlite3.connect(source)) as db:
        for sql in owner.schema_policy().schema_sql[0][1]:
            if not sql.startswith("CREATE TABLE sqlite_sequence"):
                db.execute(sql)
        db.execute(
            "INSERT INTO research_runs(id,query,created_at,updated_at) VALUES ('kept','old','now','now')"
        )
        db.commit()
    source.chmod(0o600)
    incoming = source.read_bytes()
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import os,sqlite3,sys; os.umask(0o077); c=sqlite3.connect(sys.argv[1]); c.execute('PRAGMA journal_mode=WAL'); c.execute('PRAGMA wal_autocheckpoint=0'); c.execute(\"UPDATE research_runs SET query='WAL-only'\"); c.commit(); os._exit(0)",
            str(source),
        ],
        check=True,
        timeout=15,
    )
    extra_items = extras(live) if extras is not None else ()
    root = tmp_path / "bootstrap"
    authority = admission_authority(root)
    authority.register("profile", (live,))
    bind_profile(root, selector, ("profile",), root / "admission")
    monkeypatch.setattr(bootstrap, "default_bootstrap_root", lambda: root)
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(selector))
    selector.write_bytes(b'api_key="test-only-original-secret"\nbroken = [')
    doc = manifest()
    doc["directories"][0]["synthetic"] = not tree
    doc["owners"] = [
        {"owner_id": "config", "schema_version": 1, "capabilities": []},
        {"owner_id": owner.owner_id, "schema_version": 0, "capabilities": []},
    ]
    payloads = {"config": b'[general]\nusers_name="new"\n', "db": incoming}
    doc["files"] = [
        {
            "logical_id": "profile:profile:"
            + ("config" if key == "config" else "research.local"),
            "root_id": "root",
            "parent_id": "root",
            "relative_path": "config.toml" if key == "config" else "research.db",
            "owner_id": "config" if key == "config" else owner.owner_id,
            "payload": "payload/" + key,
            "size": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
        for key, data in payloads.items()
    ]
    doc["producer_inventory"] = [
        {
            "logical_id": f["logical_id"],
            "owner_id": f["owner_id"],
            "status": "included",
            "dependencies": [],
            "shared_group": None,
        }
        for f in doc["files"]
    ]
    doc["producer_inventory"].append(
        {
            "logical_id": "root",
            "owner_id": "ui.state" if tree else "config",
            "status": "included_directory",
            "dependencies": [],
            "shared_group": None,
        }
    )
    if tree:
        doc["owners"].append(
            {"owner_id": "ui.state", "schema_version": 1, "capabilities": []}
        )
    doc["dependency_groups"][0]["members"] = [
        "profile:profile:config",
        "profile:profile:research.local",
    ]
    if extra_items:
        from tldw_chatbook.Backup_Recovery.owner_registry import install_adapters

        declarations = {adapter.owner_id: adapter for adapter in install_adapters()}
        for item in extra_items:
            if item.status != "included":
                continue
            extra_root = "root"
            if item.shared_group:
                extra_root = "root-" + item.logical_id
                doc["directories"].append(
                    {
                        "logical_id": extra_root,
                        "root_id": extra_root,
                        "parent_id": None,
                        "relative_path": "",
                        "synthetic": True,
                        "metadata": {"version": 1, "mode": 0o700, "mtime_ns": 0},
                    }
                )
                doc["producer_inventory"].append(
                    {
                        "logical_id": extra_root,
                        "owner_id": item.owner,
                        "status": "included_directory",
                        "dependencies": [],
                        "shared_group": None,
                    }
                )
            data = item.path.read_bytes()
            payload_key = item.logical_id.rsplit(":", 1)[-1]
            payloads[payload_key] = data
            doc["files"].append(
                {
                    "logical_id": item.logical_id,
                    "root_id": extra_root,
                    "parent_id": extra_root,
                    "relative_path": item.path.name,
                    "owner_id": item.owner,
                    "payload": "payload/" + payload_key,
                    "size": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
            )
            doc["producer_inventory"].append(
                {
                    "logical_id": item.logical_id,
                    "owner_id": item.owner,
                    "status": "included",
                    "dependencies": list(item.dependencies),
                    "shared_group": item.shared_group,
                }
            )
            if item.owner not in {row["owner_id"] for row in doc["owners"]}:
                doc["owners"].append(
                    {
                        "owner_id": item.owner,
                        "schema_version": max(
                            declarations[item.owner].schema_policy().versions
                        ),
                        "capabilities": [],
                    }
                )
            doc["dependency_groups"][0]["members"].append(item.logical_id)
    archive_path = tmp_path / "replacement.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("manifest.json", json.dumps(doc))
        for key, data in payloads.items():
            archive.writestr("payload/" + key, data)
    archive = acquire(archive_path, tmp_path / "input", ArchiveLimits(), None, Event())
    items = [
        StorageItem("config", "profile:profile:config", selector, "included", ()),
        StorageItem(
            owner.owner_id, "profile:profile:research.local", source, "included", ()
        ),
    ]
    items += [
        StorageItem(
            "sqlite.transient",
            "db" + suffix,
            Path(str(source) + suffix),
            "intentionally_excluded",
            ("profile:profile:research.local",),
        )
        for suffix in ("-wal", "-shm")
    ]
    if tree:
        items.append(StorageItem("ui.state", "root", live, "included_directory", ()))
        raw = live / "old-state.toml"
        raw.write_bytes(b"[state]\nselected=17\n")
        raw.chmod(0o600)
        items.append(StorageItem("ui.state", "old-state", raw, "included", ("root",)))
    items.extend(extra_items)
    if omit_shm:
        items = [item for item in items if not item.logical_id.endswith("-shm")]
    target = Inventory(tuple(items), True, "reviewed-local", ())
    plan = plan_restore(
        archive,
        mode="replace",
        destinations={
            **{row["root_id"]: live for row in doc["directories"]},
            "profile:profile:paths.data_dir": live / "data",
        },
        target=target,
        profile_names={"profile": "Local"},
    )
    control = tmp_path / "control"
    control.mkdir(mode=0o700)
    journal = Journal(control, "held-sqlite")
    candidate = stage_restore(
        archive, plan, tmp_path / "candidate", Event(), journal=journal
    )
    if not prepared:
        yield candidate, plan, journal, None, source, selector
        return
    selectors = (live / "unrelated.toml",) if missing_selector else (selector,)
    register_pending(root, journal.operation_id, ("profile",), control, selectors)
    with authority.maintenance(
        ("profile", "bootstrap.unbound") if guard else ("profile",), 3
    ) as session:
        journal.prepare_publication(
            candidate,
            plan,
            bootstrap_root=root,
            namespaces=("profile",),
            selectors=selectors,
            generation="new",
        )
        yield candidate, plan, journal, session, source, selector


def run_capture(case, tmp_path, **kwargs):
    from tldw_chatbook.Backup_Recovery import replacement

    candidate, plan, journal, session, _, _ = case
    return replacement.capture_verify_rollback(
        candidate,
        plan,
        journal,
        tmp_path / "rollback.tldw-backup.zip.age",
        session=kwargs.get("session", session),
        password=b"test-only-password",
        work_root=tmp_path / "rollback-work",
        cancel=kwargs.get("cancel", Event()),
        acknowledged_credential_issues=kwargs.get(
            "issues", ("credential_format_unreadable",)
        ),
    )


@pytest.mark.parametrize("tree", [False, True])
def test_real_wal_and_corrupt_config_round_trip_under_one_session(
    tmp_path, monkeypatch, helper_resource_root, tree
):
    from tldw_chatbook.Backup_Recovery import crypto

    monkeypatch.setattr(crypto, "_package_resource_root", lambda: helper_resource_root)
    with replacement_case(tmp_path, monkeypatch, tree=tree) as case:
        with case[2]._locked(exclusive=False) as fd:
            prepared = publication._Prepared.model_validate(
                case[2]._records(fd)[-1].evidence
            )
        coverage = {
            row.logical_id: row.logical_id
            for row in prepared.artifacts
            if row.previous is not None
        }
        coverage.update(
            {row.logical_id: row.logical_id for row in prepared.directory_metadata}
        )
        with pytest.raises(ValueError, match="rollback_sqlite_owner_receipt_required"):
            case[2].verify_rollback(
                case[4],
                password=b"test-only-password",
                work_root=tmp_path / "forbidden-verifier",
                cancel=Event(),
                coverage=coverage,
            )
        before = state(case[4])
        original_config = case[5].read_bytes()
        path = run_capture(case, tmp_path)
        assert path.is_file()
        assert state(case[4]) == before
        assert case[5].read_bytes() == original_config
        case[3]._check()
        archived = acquire(
            path,
            tmp_path / "roundtrip",
            ArchiveLimits(),
            b"test-only-password",
            Event(),
        )
        doc = verify_sealed(archived)
        assert doc.consistency == "coherent"
        assert "sqlite.transient" not in {owner.owner_id for owner in doc.owners}
        with zipfile.ZipFile(archived.path) as archive:
            by_id = {f.logical_id: f for f in doc.files}
            assert (
                archive.read(by_id["profile:profile:config"].payload) == original_config
            )
            if tree:
                assert (
                    archive.read(by_id["old-state"].payload)
                    == b"[state]\nselected=17\n"
                )
            restored = tmp_path / "checked.db"
            restored.write_bytes(
                archive.read(by_id["profile:profile:research.local"].payload)
            )
        with closing(sqlite3.connect(restored)) as db:
            assert db.execute("SELECT query FROM research_runs").fetchall() == [
                ("WAL-only",)
            ]
            assert db.execute("PRAGMA journal_mode").fetchone() == ("delete",)
        with case[2]._locked(exclusive=False) as fd:
            records = case[2]._records(fd)
        assert records[-1].event == "rollback_verified"
        assert records[-1].evidence["sqlite_groups"]
        assert not bootstrap.startup_permission(case[5], tmp_path / "bootstrap")[0]
        publication.publish_candidate(case[0], case[1], case[2], path)
        assert not Path(str(case[4]) + "-wal").exists()
        assert not Path(str(case[4]) + "-shm").exists()
        assert all(
            value == "published" or value == "retired"
            for value in case[2].artifact_states().values()
        )


def test_structurally_valid_sqlite_receipt_is_not_public_caller_authority(
    tmp_path, monkeypatch
):
    from tldw_chatbook.Backup_Recovery.journal import observe_artifact

    with replacement_case(tmp_path, monkeypatch) as case:
        with case[2]._locked(exclusive=False) as fd:
            prepared = case[2]._records(fd)[-1].evidence
        groups = [
            {**row, "schema_version": 0, "payload_size": 1, "payload_digest": "0" * 64}
            for row in prepared["rollback_sources"]
        ]
        proof = {
            "ciphertext": observe_artifact(case[4]),
            "sealed_digest": "0" * 64,
            "manifest_digest": "0" * 64,
            "coverage": {
                row["logical_id"]: row["logical_id"]
                for row in prepared["artifacts"]
                if row["previous"] is not None
            },
            "sqlite_groups": groups,
            "credential_issues": [],
        }
        with pytest.raises(ValueError, match="held_capture_required"):
            case[2].record("rollback_verified", proof)
        assert case[2].artifact_states()


@pytest.mark.parametrize(
    "kind",
    [
        "missing",
        "fake",
        "guard",
        "mapping",
        "owner",
        "cancelled",
        "credential_issues",
        "workspace",
    ],
)
def test_unqualified_capture_retains_originals_and_no_receipt(
    tmp_path, monkeypatch, kind
):
    with replacement_case(tmp_path, monkeypatch, guard=kind != "guard") as case:
        from dataclasses import replace

        from tldw_chatbook.Backup_Recovery import replacement

        before = state(case[4])
        kwargs = {}
        if kind in {"missing", "fake"}:
            kwargs["session"] = None if kind == "missing" else object()
        if kind == "cancelled":
            cancel = Event()
            cancel.set()
            kwargs["cancel"] = cancel
        if kind == "credential_issues":
            kwargs["issues"] = ()
        if kind == "owner":
            candidate, plan, *rest = case
            changed = replace(
                plan.target,
                items=tuple(
                    replace(item, owner="ui.state")
                    if item.owner == "research.local"
                    else item
                    for item in plan.target.items
                ),
            )
            case = (candidate, replace(plan, target=changed), *rest)
        if kind == "mapping":
            from tldw_chatbook.Backup_Recovery.bootstrap import _key

            path = tmp_path / "bootstrap" / ("profile-" + _key(str(case[5])) + ".json")
            record = json.loads(path.read_bytes())
            record["roots"] = [str(tmp_path / "other")]
            path.write_text(json.dumps(record))
        with pytest.raises((ValueError, RuntimeError, InterruptedError)):
            if kind == "workspace":
                replacement.capture_verify_rollback(
                    case[0],
                    case[1],
                    case[2],
                    tmp_path / "out.tldw-backup.zip.age",
                    session=case[3],
                    password=b"test-only",
                    work_root=case[4].parent / "forbidden",
                    cancel=Event(),
                    acknowledged_credential_issues=("credential_format_unreadable",),
                )
            else:
                run_capture(case, tmp_path, **kwargs)
        assert state(case[4]) == before
        assert not (case[4].parent / "forbidden").exists()
        with case[2]._locked(exclusive=False) as fd:
            assert case[2]._records(fd)[-1].event == "prepared"


@pytest.mark.parametrize(
    "fault",
    ["source_changed", "ciphertext_changed", "encryption_failure", "journal_barrier"],
)
def test_failure_during_held_encryption_never_authorizes_publication(
    tmp_path, monkeypatch, helper_resource_root, fault
):
    from tldw_chatbook.Backup_Recovery import crypto, replacement

    monkeypatch.setattr(crypto, "_package_resource_root", lambda: helper_resource_root)
    with replacement_case(tmp_path, monkeypatch) as case:
        original = replacement.write_archive
        before = state(case[4])

        def injected(*args, **kwargs):
            if fault == "encryption_failure":
                raise OSError("encryption failed")
            result = original(*args, **kwargs)
            if fault == "source_changed":
                case[5].write_bytes(b"later user edit")
            if fault == "ciphertext_changed":
                Path(args[1]).write_bytes(b"invalid encrypted archive")
            return result

        monkeypatch.setattr(replacement, "write_archive", injected)
        if fault == "journal_barrier":
            flush = Journal._flush_records

            def fail_terminal(self, parent):
                if self._records(parent)[-1].event == "rollback_verified":
                    raise OSError("journal barrier failed")
                return flush(self, parent)

            monkeypatch.setattr(Journal, "_flush_records", fail_terminal)
        with pytest.raises((ValueError, OSError)):
            run_capture(case, tmp_path)
        assert state(case[4]) == before
        with case[2]._locked(exclusive=False) as fd:
            events = [row.event for row in case[2]._records(fd)]
        assert "publication_started" not in events
        assert not bootstrap.startup_permission(case[5], tmp_path / "bootstrap")[0]


def test_bound_and_unbound_native_writers_wait_through_real_encryption(
    tmp_path, monkeypatch, helper_resource_root
):
    import select

    from tldw_chatbook.Backup_Recovery import crypto, replacement

    monkeypatch.setattr(crypto, "_package_resource_root", lambda: helper_resource_root)
    children = []
    writer = r"""
import sys
from pathlib import Path
from tldw_chatbook.Backup_Recovery.admission import Admission
print('ready',flush=True)
with Admission(Path(sys.argv[1])).normal((sys.argv[2],)):
    Path(sys.argv[3]).write_text('finished')
print('finished',flush=True)
"""
    try:
        with replacement_case(tmp_path, monkeypatch) as case:
            original = replacement.write_archive

            def checked(*args, **kwargs):
                case[3]._check()
                for name in ("profile", "bootstrap.unbound"):
                    child = subprocess.Popen(
                        [
                            sys.executable,
                            "-c",
                            writer,
                            str(tmp_path / "bootstrap/admission"),
                            name,
                            str(tmp_path / (name + ".done")),
                        ],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                    )
                    children.append(child)
                    assert select.select([child.stdout], [], [], 5)[0]
                    assert child.stdout.readline().strip() == "ready"
                result = original(*args, **kwargs)
                case[3]._check()
                assert all(child.poll() is None for child in children)
                return result

            monkeypatch.setattr(replacement, "write_archive", checked)
            run_capture(case, tmp_path)
            case[3]._check()
            assert all(
                not select.select([child.stdout], [], [], 0.1)[0] for child in children
            )
        for child in children:
            stdout, stderr = child.communicate(timeout=5)
            assert child.returncode == 0, stderr
            assert "finished" in stdout
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
                child.wait()


@pytest.mark.parametrize("owner", ["config", "research.local"])
def test_credential_processing_cannot_change_retained_snapshot_expectations(
    tmp_path, monkeypatch, owner
):
    from tldw_chatbook.Backup_Recovery import replacement

    with replacement_case(tmp_path, monkeypatch) as case:
        before = state(case[4])
        original = replacement.process_credentials

        def changed(stage, inventory, **kwargs):
            result = original(stage, inventory, **kwargs)
            next(
                item.path for item in inventory.items if item.owner == owner
            ).write_bytes(b"changed private snapshot")
            return result

        monkeypatch.setattr(replacement, "process_credentials", changed)
        with pytest.raises(ValueError, match="rollback_snapshot_changed"):
            run_capture(case, tmp_path)
        assert state(case[4]) == before
        assert not (tmp_path / "rollback.tldw-backup.zip.age").exists()


def test_actual_unclassified_shm_cannot_receive_snapshot_coverage(
    tmp_path, monkeypatch
):
    with (
        pytest.raises(ValueError, match="rollback_sidecar_unclassified"),
        replacement_case(tmp_path, monkeypatch, omit_shm=True),
    ):
        pass


def test_rollback_output_cannot_be_a_fresh_sibling_inside_owned_root(
    tmp_path, monkeypatch
):
    from tldw_chatbook.Backup_Recovery import replacement

    with replacement_case(tmp_path, monkeypatch) as case:
        before = state(case[4])
        called = []

        def encrypt(*args, **kwargs):
            called.append(True)
            raise RuntimeError("encryption must not start")

        monkeypatch.setattr(replacement, "write_archive", encrypt)
        with pytest.raises(ValueError, match="rollback_output_overlaps_source"):
            replacement.capture_verify_rollback(
                case[0],
                case[1],
                case[2],
                case[4].parent / "sibling.tldw-backup.zip.age",
                session=case[3],
                password=b"test-only",
                work_root=tmp_path / "untouched-work",
                cancel=Event(),
                acknowledged_credential_issues=("credential_format_unreadable",),
            )
        assert not called
        assert not (tmp_path / "untouched-work").exists()
        assert state(case[4]) == before


def recovered_originals(live):
    from tldw_chatbook.Backup_Recovery.recovered_media_schema import migrate

    catalog = live / "catalog.sqlite3"
    payload = live / ("a" * 32 + ".payload")
    data = b"\x00test-only-original-image\xff"
    payload.write_bytes(data)
    payload.chmod(0o600)
    with closing(sqlite3.connect(catalog)) as connection:
        migrate(connection)
        connection.execute(
            "INSERT INTO assets VALUES (?, ?, ?, 'image/png', 'ready')",
            ("a" * 32, hashlib.sha256(data).hexdigest(), len(data)),
        )
        connection.commit()
    catalog.chmod(0o600)
    return (
        StorageItem("recovered.media", "catalog", catalog, "included", ("image",)),
        StorageItem("recovered.media", "image", payload, "included", ()),
    )


def shared_core_originals(live):
    from Tests.Backup_Recovery.test_capture_sqlite_materialization import _WAL
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    source = live / "shared.db"
    store = CharactersRAGDB(source, "held-rollback-alias-fixture")
    store.add_note("original", "body")
    store.close()
    from tldw_chatbook.runtime_policy.server_credentials import RECOVERY_SETUP_REQUIRED

    # This owner has no usable citation credential; do not query any keyring.
    with closing(sqlite3.connect(source)) as db:
        db.execute(
            "UPDATE rag_identity_context SET fingerprint_key_id=?",
            (RECOVERY_SETUP_REQUIRED,),
        )
        db.commit()
    subprocess.run(
        [sys.executable, "-c", _WAL, str(source), "notes"], check=True, timeout=15
    )
    for suffix in ("", "-wal", "-shm"):
        Path(str(source) + suffix).chmod(0o600)
    return (
        StorageItem(
            "db.chachanotes.primary",
            "profile:profile:db.chachanotes.primary",
            source,
            "included",
            (),
            shared_group="shared",
        ),
        StorageItem(
            "notes.sync_bindings",
            "profile:profile:notes.sync_bindings",
            source,
            "included",
            ("profile:profile:db.chachanotes.primary",),
            shared_group="shared",
        ),
        *(
            StorageItem(
                "sqlite.transient",
                "core" + suffix,
                Path(str(source) + suffix),
                "intentionally_excluded",
                ("profile:profile:db.chachanotes.primary",),
            )
            for suffix in ("-wal", "-shm")
        ),
    )


@pytest.mark.parametrize("kind", ["recovered", "aliases"])
def test_installed_mixed_and_shared_owners_round_trip(
    tmp_path, monkeypatch, helper_resource_root, kind
):
    from tldw_chatbook.Backup_Recovery import crypto

    monkeypatch.setattr(crypto, "_package_resource_root", lambda: helper_resource_root)
    extras = recovered_originals if kind == "recovered" else shared_core_originals
    with replacement_case(
        tmp_path, monkeypatch, tree=kind == "recovered", extras=extras
    ) as case:
        source = case[4].parent / (
            "catalog.sqlite3" if kind == "recovered" else "shared.db"
        )
        before = state(source)
        output = run_capture(case, tmp_path)
        assert state(source) == before
        archived = acquire(
            output,
            tmp_path / "role-roundtrip",
            ArchiveLimits(),
            b"test-only-password",
            Event(),
        )
        doc = verify_sealed(archived)
        with case[2]._locked(exclusive=False) as fd:
            receipt = case[2]._records(fd)[-1].evidence
        groups = {row["logical_id"]: row for row in receipt["sqlite_groups"]}
        files = {row.logical_id: row for row in doc.files}
        if kind == "aliases":
            for short, owner in (
                ("core", "db.chachanotes.primary"),
                ("bindings", "notes.sync_bindings"),
            ):
                groups[short] = groups["profile:profile:" + owner]
                files[short] = files["profile:profile:" + owner]
        with zipfile.ZipFile(archived.path) as archive:
            if kind == "recovered":
                assert "image" not in groups
                assert groups["catalog"]["owner_id"] == "recovered.media"
                assert (
                    archive.read(files["image"].payload)
                    == b"\x00test-only-original-image\xff"
                )
                assert (
                    files["image"].sha256
                    == hashlib.sha256(
                        (source.parent / ("a" * 32 + ".payload")).read_bytes()
                    ).hexdigest()
                )
            else:
                assert groups["core"]["owner_id"] == "db.chachanotes.primary"
                assert groups["bindings"]["owner_id"] == "notes.sync_bindings"
                assert groups["core"]["sidecars"] == groups["bindings"]["sidecars"]
                assert set(groups["core"]["sidecars"]) == {"core-wal", "core-shm"}
                assert groups["core"]["artifacts"] == groups["bindings"]["artifacts"]
                assert (
                    groups["core"]["payload_digest"]
                    == groups["bindings"]["payload_digest"]
                )
                for key in ("core", "bindings"):
                    checked = tmp_path / (key + ".db")
                    checked.write_bytes(archive.read(files[key].payload))
                    with closing(sqlite3.connect(checked)) as db:
                        assert db.execute("SELECT title FROM notes").fetchall() == [
                            ("WAL-only",)
                        ]
        assert receipt["coverage"]
        publication.publish_candidate(case[0], case[1], case[2], output)
        assert not Path(str(source) + "-wal").exists()
        assert not Path(str(source) + "-shm").exists()


@pytest.mark.parametrize("suffix", ["-wal", "-shm"])
def test_raw_recovered_payload_never_classifies_as_sqlite_sidecar_main(
    tmp_path, suffix
):
    from tldw_chatbook.Backup_Recovery.owner_registry import install_adapters

    items = recovered_originals(tmp_path)
    payload = items[1]
    lookalike = StorageItem(
        "sqlite.transient",
        "lookalike",
        Path(str(payload.path) + suffix),
        "intentionally_excluded",
        (payload.logical_id,),
    )
    owners = {owner.owner_id: owner for owner in install_adapters()}
    with pytest.raises(ValueError, match="rollback_sidecar_unclassified"):
        publication._sidecar_main(lookalike, (*items, lookalike), owners)


def test_preparation_requires_actual_restored_config_selector(tmp_path, monkeypatch):
    with (
        pytest.raises(ValueError, match="publication_selector_mismatch"),
        replacement_case(tmp_path, monkeypatch, missing_selector=True),
    ):
        pass


@pytest.mark.parametrize("relation", ["different_group", "different_path", "hardlink"])
def test_sidecar_evidence_requires_same_path_and_declared_group(tmp_path, relation):
    from dataclasses import replace
    from types import SimpleNamespace

    from tldw_chatbook.Backup_Recovery.owner_registry import install_adapters

    source = tmp_path / "main.db"
    source.write_bytes(b"local-original-observation")
    main = StorageItem(
        "research.local", "main", source, "included", (), shared_group="shared"
    )
    if relation != "different_group":
        other = tmp_path / "other.db"
        if relation == "hardlink":
            other.hardlink_to(source)
        else:
            other.write_bytes(source.read_bytes())
        alias = replace(main, logical_id="alias", path=other)
    else:
        alias = replace(main, logical_id="alias", shared_group="unrelated")
    sidecar = Path(str(source) + "-wal")
    sidecar.write_bytes(b"local-sidecar-observation")
    items = (
        main,
        alias,
        StorageItem(
            "sqlite.transient", "wal", sidecar, "intentionally_excluded", ("main",)
        ),
    )
    plan = SimpleNamespace(target=Inventory(items, True, "local", ()))
    artifacts = [
        {
            "logical_id": item.logical_id,
            "target": str(item.path),
            "previous": {"kind": "file"},
        }
        for item in (main, alias)
    ]
    owners = {owner.owner_id: owner for owner in install_adapters()}
    if relation in {"different_group", "hardlink"}:
        expected = (
            "artifact_unverified"
            if relation == "hardlink"
            else "rollback_sidecar_unclassified"
        )
        with pytest.raises(ValueError, match=expected):
            publication._rollback_sources(plan, artifacts, owners)
    else:
        groups = {
            row["logical_id"]: row
            for row in publication._rollback_sources(plan, artifacts, owners)
        }
        assert groups["main"]["sidecars"]
        assert groups["alias"]["sidecars"] == {}
