"""Later rollback restores authenticated snapshots after preserving current edits."""

import shutil
import sqlite3
import subprocess
import sys
import zipfile
from contextlib import closing, contextmanager
from dataclasses import replace
from pathlib import Path
from threading import Event

import pytest

from Tests.Backup_Recovery.test_held_sqlite_rollback import replacement_case
from tldw_chatbook.Backup_Recovery import (
    archive_reader,
    bootstrap,
    crypto,
    recovery_copies,
    replacement,
)
from tldw_chatbook.Backup_Recovery.journal import Journal
from tldw_chatbook.Backup_Recovery.limits import ArchiveLimits


@contextmanager
def _completed(tmp_path, monkeypatch, helper, *, control_root=None):
    from Tests.RuntimePolicy.test_server_credentials import FakeKeyring
    from tldw_chatbook.Backup_Recovery import credentials
    from tldw_chatbook.runtime_policy.server_credentials import (
        KeyringServerCredentialStore,
    )

    store = KeyringServerCredentialStore(keyring_backend=FakeKeyring())
    monkeypatch.setattr(credentials, "_credential_store", lambda: store)
    monkeypatch.setattr(crypto, "_package_resource_root", lambda: helper)
    with replacement_case(tmp_path, monkeypatch, prepared=False) as case:
        candidate, plan, _, _, source, selector = case
        original_config = selector.read_bytes()
        operation = replacement.replace(
            replace(
                plan, acknowledged_credential_issues=("credential_format_unreadable",)
            ),
            candidate,
            control_root=control_root or tmp_path / "control",
            rollback_password=b"original",
            cancel=Event(),
        )
        subprocess.run(
            [
                sys.executable,
                "-c",
                "import os,sqlite3,sys; os.umask(0o077); c=sqlite3.connect(sys.argv[1]); c.execute('PRAGMA journal_mode=WAL'); c.execute('PRAGMA wal_autocheckpoint=0'); c.execute(\"UPDATE research_runs SET query='post-restore edit'\"); c.commit(); os._exit(0)",
                str(source),
            ],
            check=True,
            timeout=15,
        )
        selector.write_bytes(
            selector.read_bytes().replace(
                b"[general]\n", b"[general]\npost_restore=true\n"
            )
        )
        yield case, operation, original_config


def test_default_control_stays_excluded_during_actual_later_capture(
    tmp_path, monkeypatch, helper_resource_root
):
    from tldw_chatbook.Backup_Recovery import inventory, service_storage
    from tldw_chatbook.Backup_Recovery.later_rollback import preview_rollback

    selected_default = tmp_path / "app-config" / "config.toml"
    monkeypatch.setattr(
        service_storage, "default_config_path", lambda: selected_default
    )
    control = service_storage.default_control_root()
    workspace = service_storage.ensure_storage(control)
    with _completed(
        tmp_path, monkeypatch, helper_resource_root, control_root=control
    ) as (case, operation, original_config):
        approved = preview_rollback(
            operation,
            control_root=control,
            old_password=b"original",
            target=case[1].target,
            cancel=Event(),
        )
        capture = replacement.capture_verify_rollback
        observed = []

        def checked_capture(*args, **kwargs):
            item = inventory._service_control_exclusion()[0]
            assert item.status == "intentionally_excluded"
            assert args[0].is_relative_to(workspace)
            result = capture(*args, **kwargs)
            service_storage.verify_default_storage()
            observed.append(result)
            return result

        monkeypatch.setattr(replacement, "capture_verify_rollback", checked_capture)
        result = recovery_copies.rollback(
            operation,
            control_root=control,
            old_password=b"original",
            new_password=b"new",
            cancel=Event(),
            approved_plan=approved,
        )
        assert observed and result != operation
        assert case[-1].read_bytes() == original_config
        service_storage.verify_default_storage()
        assert {path.name for path in control.parent.iterdir()} == {
            "control",
            "control-work",
        }


def test_later_rollback_preserves_post_edits_without_old_raw_objects(
    tmp_path, monkeypatch, helper_resource_root
):
    from tldw_chatbook.Backup_Recovery.later_rollback import preview_rollback

    with _completed(tmp_path, monkeypatch, helper_resource_root) as (
        case,
        operation,
        original_config,
    ):
        candidate, original_plan, _, _, source, selector = case
        control = tmp_path / "control"
        approved = preview_rollback(
            operation,
            control_root=control,
            old_password=b"original",
            target=original_plan.target,
            cancel=Event(),
        )
        assert approved.local_snapshot.operation_id == operation
        before = selector.read_bytes()
        with pytest.raises(ValueError, match="preview_required"):
            recovery_copies.rollback(
                operation,
                control_root=control,
                old_password=b"original",
                new_password=b"new safety",
                cancel=Event(),
            )
        assert selector.read_bytes() == before
        journal = Journal(control, operation)
        with journal._locked(exclusive=False) as parent:
            prepared = next(
                row.evidence
                for row in journal._records(parent)
                if row.event == "prepared"
            )
        for row in prepared["artifacts"]:
            if row["retained"]:
                retained = Path(row["retained"])
                if retained.is_dir():
                    shutil.rmtree(retained)
                else:
                    retained.unlink()
        shutil.rmtree(candidate)
        result = recovery_copies.rollback(
            operation,
            control_root=control,
            old_password=b"original",
            new_password=b"new safety",
            cancel=Event(),
            approved_plan=approved,
        )
        assert result != operation
        assert selector.read_bytes() == original_config
        with closing(sqlite3.connect(source)) as db:
            assert db.execute(
                "SELECT query FROM research_runs WHERE id='kept'"
            ).fetchone() == ("WAL-only",)
            assert db.execute("PRAGMA user_version").fetchone() == (0,)
        copies = recovery_copies.list_recovery_copies(control)
        safety = next(copy for copy in copies if copy.operation_id == result)
        archive = archive_reader.acquire(
            safety.path,
            tmp_path / "new-safety-readback",
            ArchiveLimits(),
            b"new safety",
            Event(),
        )
        doc = archive_reader.verify_sealed(archive)
        with zipfile.ZipFile(archive.path) as packed:
            config = next(row for row in doc.files if row.owner_id == "config")
            assert packed.read(config.payload) == before
            row = next(row for row in doc.files if row.owner_id == "research.local")
            snapshot = tmp_path / "post-edit-snapshot.db"
            snapshot.write_bytes(packed.read(row.payload))
        with closing(sqlite3.connect(snapshot)) as db:
            assert db.execute(
                "SELECT query FROM research_runs WHERE id='kept'"
            ).fetchone() == ("post-restore edit",)
        assert bootstrap.startup_permission(selector, tmp_path / "bootstrap")[0]


@pytest.mark.parametrize("change", ["target", "password", "source", "capacity"])
def test_later_rollback_refuses_before_new_pending(
    tmp_path, monkeypatch, helper_resource_root, change
):
    from tldw_chatbook.Backup_Recovery.later_rollback import preview_rollback

    with _completed(tmp_path, monkeypatch, helper_resource_root) as (
        case,
        operation,
        _,
    ):
        _, original, _, _, _, selector = case
        control = tmp_path / "control"
        plan = preview_rollback(
            operation,
            control_root=control,
            old_password=b"original",
            target=original.target,
            cancel=Event(),
        )
        password = b"original"
        if change == "target":
            selector.write_bytes(selector.read_bytes() + b"changed=true\n")
        elif change == "password":
            password = b"wrong"
        elif change == "source":
            plan = replace(
                plan,
                local_snapshot=replace(plan.local_snapshot, rollback_digest="0" * 64),
            )
        else:
            from tldw_chatbook.Backup_Recovery import archive_reader

            def insufficient(*_args):
                raise ValueError("insufficient_space")

            monkeypatch.setattr(archive_reader, "_space", insufficient)
        before = selector.read_bytes()
        with pytest.raises((ValueError, crypto.CryptoError)):
            recovery_copies.rollback(
                operation,
                control_root=control,
                old_password=password,
                new_password=b"new",
                cancel=Event(),
                approved_plan=plan,
            )
        assert selector.read_bytes() == before
        assert bootstrap._records(tmp_path / "bootstrap")[0] == []


def test_later_installed_validation_failure_restores_post_edits(
    tmp_path, monkeypatch, helper_resource_root
):
    from tldw_chatbook.Backup_Recovery import publication
    from tldw_chatbook.Backup_Recovery.later_rollback import preview_rollback

    with _completed(tmp_path, monkeypatch, helper_resource_root) as (
        case,
        operation,
        _,
    ):
        _, original, _, _, source, selector = case
        control = tmp_path / "control"
        plan = preview_rollback(
            operation,
            control_root=control,
            old_password=b"original",
            target=original.target,
            cancel=Event(),
        )
        before = {
            path: path.read_bytes()
            for path in (
                source,
                selector,
                Path(str(source) + "-wal"),
                Path(str(source) + "-shm"),
            )
        }
        validate = publication._validate_installed

        def rejected(*args, **kwargs):
            validate(*args, **kwargs)
            raise ValueError("actual installed validation rejected")

        monkeypatch.setattr(publication, "_validate_installed", rejected)
        with pytest.raises(ValueError, match="replacement_rolled_back"):
            recovery_copies.rollback(
                operation,
                control_root=control,
                old_password=b"original",
                new_password=b"new",
                cancel=Event(),
                approved_plan=plan,
            )
        assert all(path.read_bytes() == data for path, data in before.items())
        assert bootstrap.startup_permission(selector, tmp_path / "bootstrap")[0]


_KILLED_LATER = r"""
import os,sys
from pathlib import Path
from threading import Event
from pydantic import TypeAdapter
from Tests.network_guard import install,blocked_attempts
install()
from Tests.RuntimePolicy.test_server_credentials import FakeKeyring
from tldw_chatbook.runtime_policy.server_credentials import KeyringServerCredentialStore
from tldw_chatbook.Backup_Recovery import bootstrap,crypto,credentials,publication,recovery_copies
from tldw_chatbook.Backup_Recovery.restore_plan import RestorePlan
root,helper,operation=sys.argv[1:]
root=Path(root)
bootstrap.default_bootstrap_root=lambda:root/'bootstrap'
crypto._package_resource_root=lambda:Path(helper)
store=KeyringServerCredentialStore(keyring_backend=FakeKeyring())
credentials._credential_store=lambda:store
plan=TypeAdapter(RestorePlan).validate_json((root/'approved.json').read_bytes())
original=publication._retire
def retire(*args,**kwargs):
 original(*args,**kwargs)
 assert not blocked_attempts(),blocked_attempts()
 os._exit(91)
publication._retire=retire
recovery_copies.rollback(operation,control_root=root/'control',old_password=b'original',new_password=b'rollback',cancel=Event(),approved_plan=plan)
"""


@pytest.mark.parametrize("action", ["finish", "rollback"])
def test_fresh_process_recovers_killed_later_rollback(
    tmp_path, monkeypatch, helper_resource_root, action
):
    from pydantic import TypeAdapter

    from Tests.Backup_Recovery.test_replacement_recovery import _child
    from tldw_chatbook.Backup_Recovery.later_rollback import preview_rollback
    from tldw_chatbook.Backup_Recovery.restore_plan import RestorePlan

    with _completed(tmp_path, monkeypatch, helper_resource_root) as (
        case,
        operation,
        old_config,
    ):
        _, original, _, _, source, selector = case
        control = tmp_path / "control"
        plan = preview_rollback(
            operation,
            control_root=control,
            old_password=b"original",
            target=original.target,
            cancel=Event(),
        )
        before = selector.read_bytes()
        (tmp_path / "approved.json").write_bytes(
            TypeAdapter(RestorePlan).dump_json(plan)
        )
        with (tmp_path / "killed-later.log").open("w+") as log:
            child = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    _KILLED_LATER,
                    str(tmp_path),
                    str(helper_resource_root),
                    operation,
                ],
                stdout=log,
                stderr=log,
                timeout=35,
                check=False,
            )
            log.seek(0)
            assert child.returncode == 91, log.read()[-5000:]
        pending = bootstrap._records(tmp_path / "bootstrap")[0]
        assert len(pending) == 1
        resumed = pending[0]["operation_id"]
        assert resumed != operation
        assert not bootstrap.startup_permission(selector, tmp_path / "bootstrap")[0]
        _child(tmp_path, helper_resource_root, "recover", action, "none", resumed)
        assert selector.read_bytes() == (old_config if action == "finish" else before)
        with closing(sqlite3.connect(source)) as db:
            assert db.execute(
                "SELECT query FROM research_runs WHERE id='kept'"
            ).fetchone() == (
                ("WAL-only" if action == "finish" else "post-restore edit"),
            )
        assert bootstrap.startup_permission(selector, tmp_path / "bootstrap")[0]


def test_later_snapshot_credentials_use_fresh_scope_without_shared_overwrite(
    tmp_path, monkeypatch, helper_resource_root
):
    import hashlib
    import json
    import tomllib

    from Tests.Backup_Recovery.test_replacement import _credential_candidate
    from tldw_chatbook.Backup_Recovery import credentials
    from tldw_chatbook.Backup_Recovery.credential_policies import GENERATION_KEYRINGS
    from tldw_chatbook.Backup_Recovery.later_rollback import preview_rollback

    monkeypatch.setattr(crypto, "_package_resource_root", lambda: helper_resource_root)
    with replacement_case(tmp_path, monkeypatch, prepared=False) as case:
        candidate, plan, store, _backend, targets = _credential_candidate(
            case, tmp_path, monkeypatch
        )
        control = tmp_path / "control"
        original_config = case[-1].read_bytes()
        operation = replacement.replace(
            plan,
            candidate,
            control_root=control,
            rollback_password=b"original",
            cancel=Event(),
        )
        installed_purpose = json.loads(targets.read_bytes())["targets"][0][
            "auth_reference"
        ].removeprefix("keyring:")
        store.set_secret("peer", "api_key", "foreign-shared-value")
        # Review the actual empty keyring slots named by installed setup markers.
        # The held safety capture below must reproduce these exact issue codes.
        configured = tomllib.loads(case[-1].read_text())
        acknowledged = []
        for section, service, names in GENERATION_KEYRINGS:
            for name in names:
                if name in configured.get(section, {}):
                    credentials._capture_record(
                        {
                            "kind": "generation",
                            "file": "payload/"
                            + hashlib.sha256(b"profile:profile:config").hexdigest(),
                            "service": service,
                            "username": name,
                            "remappable": False,
                        },
                        [],
                        acknowledged,
                    )
        approved = preview_rollback(
            operation,
            control_root=control,
            old_password=b"original",
            target=plan.target,
            cancel=Event(),
            acknowledged_credential_issues=tuple(acknowledged),
        )
        process = replacement.process_credentials

        def checked_capture(*args, **kwargs):
            issues = process(*args, **kwargs)
            assert set(issues) == set(acknowledged), issues
            return issues

        monkeypatch.setattr(replacement, "process_credentials", checked_capture)
        recovery_copies.rollback(
            operation,
            control_root=control,
            old_password=b"original",
            new_password=b"new",
            cancel=Event(),
            approved_plan=approved,
        )
        restored = json.loads(targets.read_bytes())["targets"][0][
            "auth_reference"
        ].removeprefix("keyring:")
        assert restored.startswith("recovery_") and restored not in {
            "api_key",
            installed_purpose,
        }
        assert store.get_secret("peer", restored) == "current-shared-secret"
        assert store.get_secret("peer", "api_key") == "foreign-shared-value"
        assert store.get_secret("peer", installed_purpose) == "captured-new-secret"
        assert case[-1].read_bytes() == original_config


def test_later_preview_refuses_changed_installed_locator(
    tmp_path, monkeypatch, helper_resource_root
):
    from tldw_chatbook.Backup_Recovery.later_rollback import preview_rollback

    with _completed(tmp_path, monkeypatch, helper_resource_root) as (
        case,
        operation,
        _,
    ):
        selector = case[-1]
        selector.write_bytes(b'[general]\nusers_name="Local"\n')
        with pytest.raises(
            ValueError, match="owner_relocation_unverified:research.local"
        ):
            preview_rollback(
                operation,
                control_root=tmp_path / "control",
                old_password=b"original",
                target=case[1].target,
                cancel=Event(),
            )
        assert bootstrap._records(tmp_path / "bootstrap")[0] == []


def test_later_rollback_preserves_explicit_safety_only_file(
    tmp_path, monkeypatch, helper_resource_root
):
    from Tests.RuntimePolicy.test_server_credentials import FakeKeyring
    from tldw_chatbook.Backup_Recovery import credentials
    from tldw_chatbook.Backup_Recovery.later_rollback import preview_rollback
    from tldw_chatbook.Backup_Recovery.models import StorageItem
    from tldw_chatbook.Backup_Recovery.restore_plan import plan_restore
    from tldw_chatbook.Backup_Recovery.staging import stage_restore
    from tldw_chatbook.runtime_policy.server_credentials import (
        KeyringServerCredentialStore,
    )

    store = KeyringServerCredentialStore(keyring_backend=FakeKeyring())
    monkeypatch.setattr(credentials, "_credential_store", lambda: store)
    monkeypatch.setattr(crypto, "_package_resource_root", lambda: helper_resource_root)
    with replacement_case(tmp_path, monkeypatch, prepared=False) as case:
        candidate, original, _, _, source, _ = case
        archive = replacement._acquired_source(candidate, original, Event())
        safety = source.parent / "preserved.toml"
        safety.write_bytes(b"[state]\nselection=17\n")
        safety.chmod(0o600)
        item = StorageItem("ui.state", "safety-definition", safety, "included", ())
        target = replace(original.target, items=(*original.target.items, item))
        plan = plan_restore(
            archive,
            mode="replace",
            destinations=dict((*original.destinations, *original.selectors)),
            target=target,
            profile_names=dict(original.profile_names),
            safety_scope=(item.logical_id,),
            acknowledged_credential_issues=("credential_format_unreadable",),
        )
        candidate = stage_restore(archive, plan, tmp_path / "safety-candidate", Event())
        control = tmp_path / "control"
        operation = replacement.replace(
            plan,
            candidate,
            control_root=control,
            rollback_password=b"original",
            cancel=Event(),
        )
        safety.write_bytes(b"[state]\nselection=99\n")
        before = (safety.read_bytes(), safety.stat().st_ino, safety.stat().st_mtime_ns)
        approved = preview_rollback(
            operation,
            control_root=control,
            old_password=b"original",
            target=target,
            cancel=Event(),
        )
        assert approved.safety_scope == (item.logical_id,)
        assert safety not in dict((*approved.restore, *approved.retire)).values()
        recovery_copies.rollback(
            operation,
            control_root=control,
            old_password=b"original",
            new_password=b"new",
            cancel=Event(),
            approved_plan=approved,
        )
        assert before == (
            safety.read_bytes(),
            safety.stat().st_ino,
            safety.stat().st_mtime_ns,
        )


@pytest.mark.parametrize("damage", [None, "owner", "kind"])
def test_later_rollback_restores_receipt_bound_file_absence(
    tmp_path, monkeypatch, helper_resource_root, damage
):
    from Tests.RuntimePolicy.test_server_credentials import FakeKeyring
    from tldw_chatbook.Backup_Recovery import credentials
    from tldw_chatbook.Backup_Recovery.later_rollback import preview_rollback
    from tldw_chatbook.Backup_Recovery.restore_plan import plan_restore
    from tldw_chatbook.Backup_Recovery.staging import stage_restore
    from tldw_chatbook.runtime_policy.server_credentials import (
        KeyringServerCredentialStore,
    )

    store = KeyringServerCredentialStore(keyring_backend=FakeKeyring())
    monkeypatch.setattr(credentials, "_credential_store", lambda: store)
    monkeypatch.setattr(crypto, "_package_resource_root", lambda: helper_resource_root)
    with replacement_case(tmp_path, monkeypatch, prepared=False) as case:
        candidate, original, _, _, source, _ = case
        archive = replacement._acquired_source(candidate, original, Event())
        for path in (source, Path(str(source) + "-wal"), Path(str(source) + "-shm")):
            path.unlink()
        target = replace(
            original.target,
            items=tuple(
                replace(row, status="unused") if row.path == source else row
                for row in original.target.items
            ),
        )
        plan = plan_restore(
            archive,
            mode="replace",
            destinations=dict((*original.destinations, *original.selectors)),
            target=target,
            profile_names=dict(original.profile_names),
            acknowledged_credential_issues=("credential_format_unreadable",),
        )
        candidate = stage_restore(archive, plan, tmp_path / "absent-candidate", Event())
        control = tmp_path / "control"
        operation = replacement.replace(
            plan,
            candidate,
            control_root=control,
            rollback_password=b"original",
            cancel=Event(),
        )
        assert source.is_file()
        current = original.target
        if damage:
            if damage == "kind":
                source.unlink()
                source.mkdir(mode=0o700)
            current = replace(
                current,
                items=tuple(
                    replace(
                        row,
                        **(
                            {"owner": "ui.state"}
                            if damage == "owner"
                            else {"status": "included_directory"}
                        ),
                    )
                    if row.path == source
                    else row
                    for row in current.items
                ),
            )
            expected = (
                "target_owner_unclassified"
                if damage == "owner"
                else "local_snapshot_absence_unclassified"
            )
            with pytest.raises(ValueError, match=expected):
                preview_rollback(
                    operation,
                    control_root=control,
                    old_password=b"original",
                    target=current,
                    cancel=Event(),
                )
            assert bootstrap._records(tmp_path / "bootstrap")[0] == []
            return
        approved = preview_rollback(
            operation,
            control_root=control,
            old_password=b"original",
            target=current,
            cancel=Event(),
        )
        assert ("profile:profile:research.local", source) in approved.retire
        result = recovery_copies.rollback(
            operation,
            control_root=control,
            old_password=b"original",
            new_password=b"new",
            cancel=Event(),
            approved_plan=approved,
        )
        assert not source.exists()
        saved = next(
            row
            for row in recovery_copies.list_recovery_copies(control)
            if row.operation_id == result
        )
        acquired = archive_reader.acquire(
            saved.path, tmp_path / "absence-readback", ArchiveLimits(), b"new", Event()
        )
        assert any(
            row.owner_id == "research.local"
            for row in archive_reader.verify_sealed(acquired).files
        )
