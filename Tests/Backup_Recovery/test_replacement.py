"""Actual outer replacement preserves held safety-copy and credential ordering."""

from dataclasses import replace
from pathlib import Path
from threading import Event

import pytest

from Tests.Backup_Recovery.test_held_sqlite_rollback import replacement_case
from tldw_chatbook.Backup_Recovery import crypto, replacement


def test_replace_cannot_start_without_rollback_password():
    with pytest.raises(ValueError, match="rollback_password_required"):
        replacement.require_rollback_password(b"")


def test_outer_replace_publishes_only_after_verified_exact_originals(
    tmp_path, monkeypatch, helper_resource_root
):
    monkeypatch.setattr(crypto, "_package_resource_root", lambda: helper_resource_root)

    def existing_data(live):
        (live / "data").mkdir(mode=0o700)
        return ()

    with replacement_case(
        tmp_path, monkeypatch, prepared=False, extras=existing_data
    ) as case:
        candidate, plan, _, _, source, selector = case
        original_config = selector.read_bytes()
        plan = replace(
            plan, acknowledged_credential_issues=("credential_format_unreadable",)
        )
        operation = replacement.replace(
            plan,
            candidate,
            control_root=tmp_path / "control",
            rollback_password=b"test-only-password",
            cancel=Event(),
        )
        import json
        import os
        import sqlite3
        import subprocess
        import sys
        import zipfile
        from contextlib import closing

        from tldw_chatbook.Backup_Recovery import archive_reader
        from tldw_chatbook.Backup_Recovery.journal import Journal
        from tldw_chatbook.Backup_Recovery.limits import ArchiveLimits

        journal = Journal(tmp_path / "control", operation)
        with journal._locked(exclusive=False) as parent:
            rows = journal._records(parent)
        assert rows[-1].event == "committed"
        proof = next(row.evidence for row in rows if row.event == "rollback_verified")
        archived = archive_reader.acquire(
            Path(proof["ciphertext"]["path"]),
            tmp_path / "original-readback",
            ArchiveLimits(),
            b"test-only-password",
            Event(),
        )
        doc = archive_reader.verify_sealed(archived)
        with zipfile.ZipFile(archived.path) as packed:
            assert (
                packed.read(
                    next(row.payload for row in doc.files if row.owner_id == "config")
                )
                == original_config
            )
            restored = tmp_path / "original.db"
            restored.write_bytes(
                packed.read(
                    next(
                        row.payload
                        for row in doc.files
                        if row.owner_id == "research.local"
                    )
                )
            )
        with closing(sqlite3.connect(restored)) as db:
            assert db.execute("SELECT query FROM research_runs").fetchall() == [
                ("WAL-only",)
            ]
        with closing(sqlite3.connect(source)) as db:
            assert db.execute("SELECT query FROM research_runs").fetchall() == [
                ("old",)
            ]
        assert not Path(str(source) + "-wal").exists()
        prepared = next(row.evidence for row in rows if row.event == "prepared")
        identity = prepared["replacement_profiles"][0]["installation_id"]
        script = r"""
import sys
from pathlib import Path
from Tests.network_guard import install, blocked_attempts
install()
from tldw_chatbook.Backup_Recovery import bootstrap
bootstrap.default_bootstrap_root=lambda:Path(sys.argv[1])
from tldw_chatbook.config import CLI_APP_CLIENT_ID
assert CLI_APP_CLIENT_ID==sys.argv[2],CLI_APP_CLIENT_ID
assert not blocked_attempts(),blocked_attempts()
print('replacement identity consumed by ordinary config')
"""
        environment = dict(
            os.environ, HOME=str(tmp_path / "home"), TLDW_CONFIG_PATH=str(selector)
        )
        child = subprocess.run(
            [sys.executable, "-c", script, str(tmp_path / "bootstrap"), identity],
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert child.returncode == 0, child.stderr[-5000:] + child.stdout[-1000:]
        assert "replacement identity consumed" in child.stdout
        assert len(identity) == 32 and identity != "tldw_cli_local_instance_v1"
        association = next((tmp_path / "bootstrap").glob("activation-*.json"), None)
        assert association is not None
        association.write_text(json.dumps({"corrupt": True}))
        child = subprocess.run(
            [sys.executable, "-c", script, str(tmp_path / "bootstrap"), identity],
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert child.returncode != 0 and "record_version" in child.stderr


def _credential_candidate(case, tmp_path, monkeypatch, *, retain=False):
    """Capture actual owned keyring material, encrypt/acquire it, and stage it."""
    import hashlib
    import json
    import zipfile

    from Tests.RuntimePolicy.test_server_credentials import FakeKeyring
    from tldw_chatbook.Backup_Recovery import archive_reader, credentials
    from tldw_chatbook.Backup_Recovery.limits import ArchiveLimits
    from tldw_chatbook.Backup_Recovery.models import Inventory, StorageItem
    from tldw_chatbook.Backup_Recovery.restore_plan import plan_restore
    from tldw_chatbook.Backup_Recovery.staging import stage_restore
    from tldw_chatbook.runtime_policy.server_credentials import (
        KeyringServerCredentialStore,
    )

    candidate, plan, _, _, source, _ = case
    archive = replacement._acquired_source(candidate, plan, Event())
    doc = archive_reader.verify_sealed(archive).model_dump(mode="json")
    with zipfile.ZipFile(archive.path) as packed:
        payloads = {row["payload"]: packed.read(row["payload"]) for row in doc["files"]}
    backend = FakeKeyring()
    store = KeyringServerCredentialStore(keyring_backend=backend)
    monkeypatch.setattr(credentials, "_credential_store", lambda: store)
    exported = tmp_path / "credential-export"
    exported.mkdir(mode=0o700)
    (exported / "payload").mkdir(mode=0o700)
    target_bytes = json.dumps(
        {
            "targets": [
                {
                    "server_id": "peer",
                    "base_url": "https://example.invalid",
                    "auth_reference": "keyring:api_key",
                }
            ]
        }
    ).encode()
    path = exported / "payload" / "targets"
    path.write_bytes(target_bytes)
    path.chmod(0o600)
    store.set_secret("peer", "api_key", "captured-new-secret")
    item = StorageItem(
        "mcp.targets", "profile:profile:mcp.targets", path, "included", ()
    )
    captured = [item]
    if retain:
        config = exported / "payload" / "config"
        content = (
            payloads["payload/config"]
            + b'\n[tldw_api]\nbase_url="https://example.invalid/"\n'
        )
        store.set_secret("https://example.invalid", "api_key", "retained-config-secret")
        config.write_bytes(content)
        config.chmod(0o600)
        captured.append(
            StorageItem("config", "profile:profile:config", config, "included", ())
        )
        payloads["payload/config"] = content
        row = next(row for row in doc["files"] if row["owner_id"] == "config")
        row.update(size=len(content), sha256=hashlib.sha256(content).hexdigest())
    issues = credentials.process_credentials(
        exported,
        Inventory(tuple(captured), True, "source", ()),
        mode="include",
        encrypted=True,
    )
    assert any("manual_recovery" in issue for issue in issues) if retain else not issues
    material = (exported / "credential-recovery.json").read_bytes()
    payloads.update(
        {"payload/targets": target_bytes, "payload/credential-recovery.json": material}
    )
    doc["credential_policy"] = "include"
    doc["owners"].extend(
        [
            {"owner_id": "mcp.targets", "schema_version": 1, "capabilities": []},
            {
                "owner_id": "recovery.credentials",
                "schema_version": 1,
                "capabilities": [],
            },
        ]
    )
    doc["directories"].append(
        {**doc["directories"][0], "logical_id": "secrets", "root_id": "secrets"}
    )
    for key, owner, root, relative, payload in [
        (
            "profile:profile:mcp.targets",
            "mcp.targets",
            "root",
            "data/Local/mcp_server_targets.json",
            "payload/targets",
        ),
        (
            "credentials",
            "recovery.credentials",
            "secrets",
            "credential-recovery.json",
            "payload/credential-recovery.json",
        ),
    ]:
        data = payloads[payload]
        doc["files"].append(
            {
                "logical_id": key,
                "root_id": root,
                "parent_id": root,
                "relative_path": relative,
                "owner_id": owner,
                "payload": payload,
                "size": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
        doc["producer_inventory"].append(
            {
                "logical_id": key,
                "owner_id": owner,
                "status": "included",
                "dependencies": [],
                "shared_group": None,
            }
        )
    for key, parent, relative in [
        ("mcp-data", "root", "data"),
        ("mcp-user", "mcp-data", "data/Local"),
    ]:
        doc["directories"].append(
            {
                **doc["directories"][0],
                "logical_id": key,
                "root_id": "root",
                "parent_id": parent,
                "relative_path": relative,
                "synthetic": False,
            }
        )
        doc["producer_inventory"].append(
            {
                "logical_id": key,
                "owner_id": "mcp.targets",
                "status": "included_directory",
                "dependencies": [],
                "shared_group": None,
            }
        )
    next(row for row in doc["files"] if row["owner_id"] == "mcp.targets")[
        "parent_id"
    ] = "mcp-user"
    doc["producer_inventory"].append(
        {
            "logical_id": "secrets",
            "owner_id": "recovery.credentials",
            "status": "included_directory",
            "dependencies": [],
            "shared_group": None,
        }
    )
    doc["dependency_groups"][0]["members"].append("profile:profile:mcp.targets")
    doc["dependency_groups"].append(
        {"group_id": "secrets", "members": ["credentials"], "complete": True}
    )
    plain = tmp_path / "incoming-credentials.zip"
    with zipfile.ZipFile(plain, "w") as packed:
        packed.writestr("manifest.json", json.dumps(doc))
        for name, data in payloads.items():
            packed.writestr(name, data)
    cipher = tmp_path / "incoming-credentials.age"
    crypto.transform(plain, cipher, password=b"incoming", decrypt=False, cancel=Event())
    acquired = archive_reader.acquire(
        cipher, tmp_path / "credential-acquired", ArchiveLimits(), b"incoming", Event()
    )
    current = source.parent / "data" / "Local" / "mcp_server_targets.json"
    current.parent.mkdir(mode=0o700, parents=True)
    current.write_bytes(target_bytes)
    current.chmod(0o600)
    store.set_secret("peer", "api_key", "current-shared-secret")
    target = replace(
        plan.target,
        items=(
            *plan.target.items,
            replace(item, path=current),
            StorageItem(
                "mcp.targets",
                "mcp-data",
                current.parent.parent,
                "included_directory",
                (),
            ),
            StorageItem(
                "mcp.targets", "mcp-user", current.parent, "included_directory", ()
            ),
        ),
    )
    reviewed = plan_restore(
        acquired,
        mode="replace",
        destinations=dict((*plan.destinations, *plan.selectors)),
        target=target,
        profile_names=dict(plan.profile_names),
        acknowledged_credential_issues=("credential_format_unreadable",),
    )
    prepared = stage_restore(
        acquired, reviewed, tmp_path / "credential-candidate", Event()
    )
    return prepared, reviewed, store, backend, current


@pytest.mark.parametrize("route", ["complete", "interrupted"])
def test_outer_credentials_are_remapped_before_hashes_and_journaled_before_values(
    tmp_path, monkeypatch, helper_resource_root, route
):
    import json

    from tldw_chatbook.Backup_Recovery.journal import Journal

    monkeypatch.setattr(crypto, "_package_resource_root", lambda: helper_resource_root)
    with replacement_case(tmp_path, monkeypatch, prepared=False) as case:
        candidate, plan, store, _backend, current = _credential_candidate(
            case, tmp_path, monkeypatch
        )
        original = (candidate / "payload" / "targets").read_bytes()
        acquisition = replacement._acquired_source(candidate, plan, Event())
        calls = []
        write_scope = store.set_recovery_secret_if_absent
        flushed = []
        flush = Journal._flush_records

        def checked_flush(self, parent):
            result = flush(self, parent)
            flushed[:] = [row.event for row in self._records(parent)]
            return result

        monkeypatch.setattr(Journal, "_flush_records", checked_flush)

        def checked_write(*args):
            assert "rollback_verified" in flushed and "credential_intended" in flushed
            calls.append(args[0])
            return write_scope(*args)

        monkeypatch.setattr(store, "set_recovery_secret_if_absent", checked_write)
        if route == "interrupted":
            from tldw_chatbook.Backup_Recovery import bootstrap, publication
            from tldw_chatbook.Backup_Recovery.control_records import (
                admission_authority,
            )

            append = Journal._append

            def failed_append(self, parent, event, evidence):
                if event == "credential_applied":
                    raise OSError("after actual keyring write")
                return append(self, parent, event, evidence)

            monkeypatch.setattr(Journal, "_append", failed_append)
            with pytest.raises(OSError, match="after actual keyring write"):
                replacement.replace(
                    plan,
                    candidate,
                    control_root=tmp_path / "control",
                    rollback_password=b"rollback",
                    cancel=Event(),
                )
            monkeypatch.setattr(Journal, "_append", append)
            pending, _ = bootstrap._records(tmp_path / "bootstrap")
            assert len(pending) == 1
            operation = pending[0]["operation_id"]
            journal = Journal(tmp_path / "control", operation)
            with journal._locked(exclusive=False) as parent:
                prior = journal._records(parent)
            assert prior[-1].event == "credential_intended" and len(calls) == 1
            assert not bootstrap.startup_permission(case[5], tmp_path / "bootstrap")[0]
            assert b"broken" in case[5].read_bytes()
            rollback = Path(
                next(
                    row.evidence["ciphertext"]["path"]
                    for row in prior
                    if row.event == "rollback_verified"
                )
            )
            with admission_authority(tmp_path / "bootstrap").maintenance(
                (*pending[0]["namespaces"], "bootstrap.unbound"), 3
            ) as session:
                replacement._apply_replacement_credentials(
                    candidate, journal, session=session, cancel=Event()
                )
                publication.publish_candidate(
                    candidate, plan, journal, rollback, session=session
                )
                publication.finalize_candidate(
                    candidate, plan, journal, session=session
                )
        else:
            operation = replacement.replace(
                plan,
                candidate,
                control_root=tmp_path / "control",
                rollback_password=b"rollback",
                cancel=Event(),
            )
        assert len(calls) == 1
        journal = Journal(tmp_path / "control", operation)
        with journal._locked(exclusive=False) as parent:
            rows = journal._records(parent)
        events = [row.event for row in rows]
        assert (
            events.index("rollback_verified")
            < events.index("credential_intended")
            < events.index("credential_applied")
            < events.index("publication_started")
        )
        purpose = json.loads(current.read_text())["targets"][0][
            "auth_reference"
        ].removeprefix("keyring:")
        assert purpose.startswith("recovery_")
        assert store.get_secret("peer", purpose) == "captured-new-secret"
        assert store.get_secret("peer", "api_key") == "current-shared-secret"
        assert (candidate / "payload" / "targets").read_bytes() == original
        assert "captured-new-secret" not in repr([row.evidence for row in rows])
        assert "current-shared-secret" not in repr([row.evidence for row in rows])
        import shutil
        import zipfile

        from tldw_chatbook.Backup_Recovery import archive_reader
        from tldw_chatbook.Backup_Recovery.limits import ArchiveLimits

        prepared = next(row.evidence for row in rows if row.event == "prepared")
        retained = Path(prepared["incoming_credentials"]["ciphertext"]["path"])
        shutil.rmtree(acquisition.path.parent)
        (tmp_path / "incoming-credentials.age").unlink()
        recovered = archive_reader.acquire(
            retained,
            tmp_path / "retained-readback",
            ArchiveLimits(),
            b"incoming",
            Event(),
        )
        with zipfile.ZipFile(recovered.path) as packed:
            assert b"captured-new-secret" in packed.read(
                "payload/credential-recovery.json"
            )
        for event in (
            "credential_intended",
            "credential_applied",
            "credentials_completed",
        ):
            with pytest.raises(
                ValueError, match="credential_held_application_required"
            ):
                journal.record(
                    event, next(row.evidence for row in rows if row.event == event)
                )


def test_explicit_safety_file_is_captured_and_preserved_without_replacement(
    tmp_path, monkeypatch, helper_resource_root
):
    import zipfile

    from tldw_chatbook.Backup_Recovery import archive_reader
    from tldw_chatbook.Backup_Recovery.journal import Journal
    from tldw_chatbook.Backup_Recovery.limits import ArchiveLimits
    from tldw_chatbook.Backup_Recovery.models import StorageItem
    from tldw_chatbook.Backup_Recovery.restore_plan import plan_restore
    from tldw_chatbook.Backup_Recovery.staging import stage_restore

    monkeypatch.setattr(crypto, "_package_resource_root", lambda: helper_resource_root)
    with replacement_case(tmp_path, monkeypatch, prepared=False) as case:
        candidate, original_plan, _, _, source, _ = case
        archive = replacement._acquired_source(candidate, original_plan, Event())
        safety = source.parent / "preserved-definition.toml"
        safety.write_bytes(b"[state]\nselection=17\n")
        safety.chmod(0o600)
        before = (safety.stat().st_ino, safety.read_bytes(), safety.stat().st_mtime_ns)
        item = StorageItem("ui.state", "safety-definition", safety, "included", ())
        target = replace(
            original_plan.target, items=(*original_plan.target.items, item)
        )
        plan = plan_restore(
            archive,
            mode="replace",
            destinations=dict((*original_plan.destinations, *original_plan.selectors)),
            target=target,
            profile_names=dict(original_plan.profile_names),
            safety_scope=(item.logical_id,),
            acknowledged_credential_issues=("credential_format_unreadable",),
        )
        assert (item.logical_id, safety) in plan.preserve
        assert (
            safety not in dict(plan.restore).values()
            and safety not in dict(plan.retire).values()
        )
        candidate = stage_restore(archive, plan, tmp_path / "safety-candidate", Event())
        operation = replacement.replace(
            plan,
            candidate,
            control_root=tmp_path / "control",
            rollback_password=b"rollback",
            cancel=Event(),
        )
        assert before == (
            safety.stat().st_ino,
            safety.read_bytes(),
            safety.stat().st_mtime_ns,
        )
        journal = Journal(tmp_path / "control", operation)
        with journal._locked(exclusive=False) as parent:
            rows = journal._records(parent)
        proof = next(row.evidence for row in rows if row.event == "rollback_verified")
        assert [row["logical_id"] for row in proof["safety_sources"]] == [
            item.logical_id
        ]
        from tldw_chatbook.Backup_Recovery.journal import _validate

        prior = rows[
            : next(
                index
                for index, row in enumerate(rows)
                if row.event == "rollback_verified"
            )
        ]
        for altered in ([], [*proof["safety_sources"], *proof["safety_sources"]]):
            with pytest.raises(ValueError, match="journal_evidence_invalid"):
                _validate(
                    "rollback_verified", {**proof, "safety_sources": altered}, prior
                )

        copied = archive_reader.acquire(
            Path(proof["ciphertext"]["path"]),
            tmp_path / "safety-readback",
            ArchiveLimits(),
            b"rollback",
            Event(),
        )
        doc = archive_reader.verify_sealed(copied)
        payload = next(
            row.payload for row in doc.files if row.logical_id == item.logical_id
        )
        with zipfile.ZipFile(copied.path) as packed:
            assert packed.read(payload) == before[1]


@pytest.mark.parametrize("change", ["source", "destination"])
def test_credential_drift_leaves_original_files_and_foreign_scopes_unchanged(
    tmp_path, monkeypatch, helper_resource_root, change
):
    import json

    from tldw_chatbook.Backup_Recovery import bootstrap
    from tldw_chatbook.Backup_Recovery.journal import Journal

    monkeypatch.setattr(crypto, "_package_resource_root", lambda: helper_resource_root)
    with replacement_case(tmp_path, monkeypatch, prepared=False) as case:
        candidate, plan, store, _backend, current = _credential_candidate(
            case, tmp_path, monkeypatch
        )
        document = json.loads((candidate / "candidate.json").read_text())
        purpose = json.loads(next(iter(document["credential_scopes"].values())))[
            "purpose"
        ]
        changed = "api_key" if change == "source" else purpose
        store.set_secret("peer", changed, "foreign-edit")
        before = {path: path.read_bytes() for path in (case[4], case[5], current)}
        with pytest.raises(ValueError, match="credential_scope_changed"):
            replacement.replace(
                plan,
                candidate,
                control_root=tmp_path / "control",
                rollback_password=b"rollback",
                cancel=Event(),
            )
        assert all(path.read_bytes() == data for path, data in before.items())
        assert store.get_secret("peer", changed) == "foreign-edit"
        pending, _ = bootstrap._records(tmp_path / "bootstrap")
        journal = Journal(tmp_path / "control", pending[0]["operation_id"])
        with journal._locked(exclusive=False) as parent:
            events = [row.event for row in journal._records(parent)]
        assert events[-1] == "rollback_verified" and "credential_intended" not in events


@pytest.mark.parametrize("cause", ["source", "cancel", "capacity", "acknowledgement"])
def test_refusal_before_publication_preserves_current_bytes(
    tmp_path, monkeypatch, helper_resource_root, cause
):
    from tldw_chatbook.Backup_Recovery import space

    monkeypatch.setattr(crypto, "_package_resource_root", lambda: helper_resource_root)
    with replacement_case(tmp_path, monkeypatch, prepared=False) as case:
        candidate, plan, _, _, source, selector = case
        before = (source.read_bytes(), selector.read_bytes())
        cancelled = Event()
        if cause == "source":
            acquired = replacement._acquired_source(candidate, plan, cancelled)
            acquired.path.write_bytes(b"changed sealed bytes")
        elif cause == "cancel":
            cancelled.set()
        elif cause == "capacity":

            def insufficient(_):
                raise ValueError("insufficient_space")

            monkeypatch.setattr(space, "require_capacity", insufficient)
        with pytest.raises((ValueError, InterruptedError)):
            replacement.replace(
                plan,
                candidate,
                control_root=tmp_path / "control",
                rollback_password=b"rollback",
                cancel=cancelled,
            )
        assert before == (source.read_bytes(), selector.read_bytes())


def test_bound_and_unbound_writers_wait_through_credentials_and_finalization(
    tmp_path, monkeypatch, helper_resource_root
):
    import select
    import subprocess
    import sys

    from tldw_chatbook.Backup_Recovery import publication

    monkeypatch.setattr(crypto, "_package_resource_root", lambda: helper_resource_root)
    children, logs = [], []
    script = r"""
import sys
from pathlib import Path
from tldw_chatbook.Backup_Recovery.admission import Admission
print('ready',flush=True)
with Admission(Path(sys.argv[1])).normal((sys.argv[2],)):
    Path(sys.argv[3]).write_text('finished')
print('finished',flush=True)
"""

    def blocked():
        assert len(children) == 2
        assert all(child.poll() is None for child in children)
        assert all(
            not (tmp_path / (name + ".done")).exists()
            for name in ("profile", "bootstrap.unbound")
        )

    try:
        with replacement_case(tmp_path, monkeypatch, prepared=False) as case:
            candidate, plan, store, _backend, _current = _credential_candidate(
                case, tmp_path, monkeypatch
            )
            write_archive = replacement.write_archive
            set_secret = store.set_recovery_secret_if_absent
            finalize = publication.finalize_candidate

            def checked_archive(*args, **kwargs):
                for name in ("profile", "bootstrap.unbound"):
                    log = (tmp_path / (name + ".stderr")).open("w+")
                    logs.append(log)
                    child = subprocess.Popen(
                        [
                            sys.executable,
                            "-c",
                            script,
                            str(tmp_path / "bootstrap/admission"),
                            name,
                            str(tmp_path / (name + ".done")),
                        ],
                        stdout=subprocess.PIPE,
                        stderr=log,
                        text=True,
                    )
                    children.append(child)
                    assert select.select([child.stdout], [], [], 5)[0]
                    assert child.stdout.readline().strip() == "ready"
                result = write_archive(*args, **kwargs)
                blocked()
                return result

            def checked_secret(*args):
                blocked()
                return set_secret(*args)

            def checked_finalize(*args, **kwargs):
                blocked()
                result = finalize(*args, **kwargs)
                kwargs["session"]._check()
                blocked()
                return result

            monkeypatch.setattr(replacement, "write_archive", checked_archive)
            monkeypatch.setattr(store, "set_recovery_secret_if_absent", checked_secret)
            monkeypatch.setattr(publication, "finalize_candidate", checked_finalize)
            replacement.replace(
                plan,
                candidate,
                control_root=tmp_path / "control",
                rollback_password=b"rollback",
                cancel=Event(),
            )
        for child, log in zip(children, logs, strict=True):
            output, _ = child.communicate(timeout=5)
            log.seek(0)
            assert child.returncode == 0, log.read()
            assert "finished" in output
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
                child.wait()
        for log in logs:
            log.close()


def test_projection_dependencies_accept_explicit_preserved_safety_scope(tmp_path):
    from Tests.Backup_Recovery.test_projection_publication import omitted_projection
    from tldw_chatbook.Backup_Recovery.models import Inventory, StorageItem
    from tldw_chatbook.Backup_Recovery.projection_publication import (
        dependent_retirements,
    )

    live = tmp_path / "live"
    live.mkdir(mode=0o700)
    source = StorageItem(
        "research.local",
        "profile:profile:research.local",
        live / "research.db",
        "included",
        (),
    )
    config = StorageItem(
        "config", "profile:profile:config", live / "config.toml", "included", ()
    )
    definitions = StorageItem(
        "ui.state", "definitions", live / "definitions.toml", "included", ()
    )
    projections = tuple(
        replace(item, dependencies=(*item.dependencies, definitions.logical_id))
        for item in omitted_projection(live)
    )
    items = (source, config, definitions, *projections)
    target = Inventory(items, True, "current", ())
    restored = [(source.logical_id, source.path), (config.logical_id, config.path)]
    preserved = [(item.logical_id, item.path) for item in (definitions, *projections)]
    with pytest.raises(ValueError, match="shared_scope_expansion_required"):
        dependent_retirements(target, restored, [], preserved)
    retired, retained, issues = dependent_retirements(
        target, restored, [], preserved, safety_scope=(definitions.logical_id,)
    )
    assert (definitions.logical_id, definitions.path) in retained
    assert (definitions.logical_id, definitions.path) not in retired
    assert set(retired) == {(item.logical_id, item.path) for item in projections}
    assert issues and all(
        issue.startswith("projection_reconciliation_required:") for issue in issues
    )


def test_unsupported_secret_is_explicitly_retained_encrypted_and_inactive(
    tmp_path, monkeypatch, helper_resource_root
):
    import json
    import shutil
    import tomllib
    import zipfile

    from tldw_chatbook.Backup_Recovery import archive_reader
    from tldw_chatbook.Backup_Recovery.journal import Journal
    from tldw_chatbook.Backup_Recovery.limits import ArchiveLimits

    monkeypatch.setattr(crypto, "_package_resource_root", lambda: helper_resource_root)
    with replacement_case(tmp_path, monkeypatch, prepared=False) as case:
        candidate, plan, store, _backend, _current = _credential_candidate(
            case, tmp_path, monkeypatch, retain=True
        )
        acquisition = replacement._acquired_source(candidate, plan, Event())
        descriptor = json.loads((candidate / "candidate.json").read_text())
        assert descriptor["credential_issues"] == [
            "credential_isolated_retention_required"
        ]
        before = case[5].read_bytes()
        control_before = tuple(sorted((tmp_path / "control").iterdir()))
        with pytest.raises(
            ValueError, match="credential_omission_acknowledgement_required"
        ):
            replacement.replace(
                plan,
                candidate,
                control_root=tmp_path / "control",
                rollback_password=b"rollback",
                cancel=Event(),
            )
        assert case[5].read_bytes() == before
        assert tuple(sorted((tmp_path / "control").iterdir())) == control_before
        plan = replace(
            plan,
            acknowledged_credential_issues=(
                *plan.acknowledged_credential_issues,
                "credential_isolated_retention_required",
            ),
        )
        operation = replacement.replace(
            plan,
            candidate,
            control_root=tmp_path / "control",
            rollback_password=b"rollback",
            cancel=Event(),
        )
        config = tomllib.loads(case[5].read_text())
        from tldw_chatbook.Backup_Recovery.credentials import RECOVERY_SETUP_REQUIRED

        assert config["tldw_api"]["auth_reference"] == RECOVERY_SETUP_REQUIRED
        assert (
            store.get_secret("https://example.invalid", "api_key")
            == "retained-config-secret"
        )
        assert b"retained-config-secret" not in case[5].read_bytes()
        assert store.get_secret("peer", "api_key") == "current-shared-secret"
        journal = Journal(tmp_path / "control", operation)
        with journal._locked(exclusive=False) as parent:
            rows = journal._records(parent)
        prepared = next(row.evidence for row in rows if row.event == "prepared")
        assert any(
            row.event == "credential_applied" and row.evidence["action"] == "retain"
            for row in rows
        )
        assert "retained-config-secret" not in repr([row.evidence for row in rows])
        shutil.rmtree(acquisition.path.parent)
        (tmp_path / "incoming-credentials.age").unlink()
        archived = archive_reader.acquire(
            Path(prepared["incoming_credentials"]["ciphertext"]["path"]),
            tmp_path / "retained-unsupported",
            ArchiveLimits(),
            b"incoming",
            Event(),
        )
        with zipfile.ZipFile(archived.path) as packed:
            assert b"retained-config-secret" in packed.read(
                "payload/credential-recovery.json"
            )


def test_existing_finalized_isolated_primitive_keeps_legacy_identity(
    tmp_path, monkeypatch
):
    from Tests.Backup_Recovery.test_publication_finalization import finalize, installed
    from tldw_chatbook.Backup_Recovery import bootstrap
    from tldw_chatbook.Backup_Recovery.isolated_restore import installation_client_id

    with installed(tmp_path) as case:
        finalize(case)
    monkeypatch.setattr(bootstrap, "default_bootstrap_root", lambda: case[4])
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(case[5]))
    assert installation_client_id() == "tldw_cli_local_instance_v1"


def test_outer_retires_target_only_file_after_exact_rollback_copy(
    tmp_path, monkeypatch, helper_resource_root
):
    import zipfile

    from tldw_chatbook.Backup_Recovery import archive_reader
    from tldw_chatbook.Backup_Recovery.journal import Journal
    from tldw_chatbook.Backup_Recovery.limits import ArchiveLimits

    monkeypatch.setattr(crypto, "_package_resource_root", lambda: helper_resource_root)
    with replacement_case(tmp_path, monkeypatch, prepared=False, tree=True) as case:
        candidate, plan, _, _, source, _ = case
        retired = source.parent / "old-state.toml"
        before = retired.read_bytes()
        assert ("old-state", retired) in plan.retire
        plan = replace(
            plan, acknowledged_credential_issues=("credential_format_unreadable",)
        )
        operation = replacement.replace(
            plan,
            candidate,
            control_root=tmp_path / "control",
            rollback_password=b"rollback",
            cancel=Event(),
        )
        assert not retired.exists()
        journal = Journal(tmp_path / "control", operation)
        with journal._locked(exclusive=False) as parent:
            rows = journal._records(parent)
        events = [row.event for row in rows]
        assert events.index("rollback_verified") < events.index("artifact_retired")
        rollback = next(
            row.evidence for row in rows if row.event == "rollback_verified"
        )
        archive = archive_reader.acquire(
            Path(rollback["ciphertext"]["path"]),
            tmp_path / "retired-readback",
            ArchiveLimits(),
            b"rollback",
            Event(),
        )
        doc = archive_reader.verify_sealed(archive)
        with zipfile.ZipFile(archive.path) as packed:
            assert (
                packed.read(
                    next(
                        row.payload
                        for row in doc.files
                        if row.logical_id == "old-state"
                    )
                )
                == before
            )


@pytest.mark.parametrize("change", ["source", "destination"])
@pytest.mark.parametrize("resume", [False, True])
def test_completed_credential_drift_refuses_before_publication(
    tmp_path, monkeypatch, helper_resource_root, change, resume
):
    import json

    from tldw_chatbook.Backup_Recovery import bootstrap, publication
    from tldw_chatbook.Backup_Recovery.control_records import admission_authority
    from tldw_chatbook.Backup_Recovery.journal import Journal

    monkeypatch.setattr(crypto, "_package_resource_root", lambda: helper_resource_root)
    with replacement_case(tmp_path, monkeypatch, prepared=False) as case:
        candidate, plan, store, _backend, current = _credential_candidate(
            case, tmp_path, monkeypatch
        )
        before = {path: path.read_bytes() for path in (case[4], case[5], current)}
        descriptor = json.loads((candidate / "candidate.json").read_text())
        purpose = json.loads(next(iter(descriptor["credential_scopes"].values())))[
            "purpose"
        ]
        apply = replacement._apply_replacement_credentials

        def drift():
            if change == "source":
                store.set_secret("peer", "api_key", "changed-after-completion")
            else:
                store.delete_secret("peer", purpose)

        def completed(*args, **kwargs):
            apply(*args, **kwargs)
            if resume:
                raise InterruptedError("after durable credential completion")
            drift()

        monkeypatch.setattr(replacement, "_apply_replacement_credentials", completed)
        if resume:
            with pytest.raises(
                InterruptedError, match="after durable credential completion"
            ):
                replacement.replace(
                    plan,
                    candidate,
                    control_root=tmp_path / "control",
                    rollback_password=b"rollback",
                    cancel=Event(),
                )
        else:
            with pytest.raises(ValueError, match="credential_scope_changed"):
                replacement.replace(
                    plan,
                    candidate,
                    control_root=tmp_path / "control",
                    rollback_password=b"rollback",
                    cancel=Event(),
                )
        pending, _ = bootstrap._records(tmp_path / "bootstrap")
        journal = Journal(tmp_path / "control", pending[0]["operation_id"])
        with journal._locked(exclusive=False) as parent:
            rows = journal._records(parent)
        proof = next(row.evidence for row in rows if row.event == "rollback_verified")
        assert Path(proof["ciphertext"]["path"]).is_file() and proof["sqlite_groups"]
        if resume:
            with admission_authority(tmp_path / "bootstrap").maintenance(
                (*pending[0]["namespaces"], "bootstrap.unbound"), 3
            ) as session:
                with pytest.raises(ValueError, match="maintenance_session_required"):
                    publication.publish_candidate(
                        candidate, plan, journal, Path(proof["ciphertext"]["path"])
                    )
                drift()
                with pytest.raises(ValueError, match="credential_scope_changed"):
                    publication.publish_candidate(
                        candidate,
                        plan,
                        journal,
                        Path(proof["ciphertext"]["path"]),
                        session=session,
                    )
        with journal._locked(exclusive=False) as parent:
            rows = journal._records(parent)
        assert rows[-1].event == "credentials_completed"
        assert not any(row.event == "publication_started" for row in rows)
        assert all(path.read_bytes() == data for path, data in before.items())
        assert not bootstrap.startup_permission(case[5], tmp_path / "bootstrap")[0]
