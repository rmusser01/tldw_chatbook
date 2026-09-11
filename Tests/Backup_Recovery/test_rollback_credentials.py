"""Authenticated captured values restore into new scopes without shared writes."""

import json
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from threading import Event

import pytest

from Tests.Backup_Recovery.test_held_sqlite_rollback import replacement_case
from Tests.Backup_Recovery.test_replacement import _credential_candidate
from tldw_chatbook.Backup_Recovery import bootstrap, crypto, publication, replacement
from tldw_chatbook.Backup_Recovery.journal import Journal


@pytest.mark.parametrize(
    "event", ["rollback_credentials_planned", "rollback_credential_applied"]
)
def test_public_journal_refuses_caller_supplied_credential_progress(
    tmp_path, monkeypatch, event
):
    with (
        replacement_case(tmp_path, monkeypatch, prepared=False) as case,
        pytest.raises(ValueError, match="recovery_execution_required"),
    ):
        case[2].record(event, {})


@pytest.mark.parametrize(
    ("changed", "tree"), [("changed", False), ("deleted", False), ("changed", True)]
)
def test_authenticated_old_secret_restores_to_new_owner_reference(
    tmp_path, monkeypatch, helper_resource_root, changed, tree
):
    monkeypatch.setattr(crypto, "_package_resource_root", lambda: helper_resource_root)
    with replacement_case(tmp_path, monkeypatch, prepared=False, tree=tree) as case:
        candidate, plan, store, backend, targets = _credential_candidate(
            case, tmp_path, monkeypatch
        )
        originals = {
            path: path.read_bytes()
            for path in (
                case[4],
                case[5],
                targets,
                Path(str(case[4]) + "-wal"),
                Path(str(case[4]) + "-shm"),
            )
        }
        finalize = publication.finalize_candidate

        def stopped(*args, **kwargs):
            raise KeyboardInterrupt("actual publication completed")

        monkeypatch.setattr(publication, "finalize_candidate", stopped)
        with pytest.raises(KeyboardInterrupt):
            replacement.replace(
                plan,
                candidate,
                control_root=tmp_path / "control",
                rollback_password=b"rollback",
                cancel=Event(),
            )
        monkeypatch.setattr(publication, "finalize_candidate", finalize)
        if changed == "changed":
            store.set_secret("peer", "api_key", "other-profile-new-value")
        else:
            store.delete_secret("peer", "api_key")
        deleted = list(backend.deleted)
        pending, _ = bootstrap._records(tmp_path / "bootstrap")
        operation = pending[0]["operation_id"]
        assert (
            replacement.recover_replacement(
                operation,
                control_root=tmp_path / "control",
                action="rollback",
                rollback_password=b"rollback",
                cancel=Event(),
            )
            == "rolled_back"
        )
        row = json.loads(targets.read_text())["targets"][0]
        purpose = row["auth_reference"].removeprefix("keyring:")
        assert purpose.startswith("recovery_")
        assert store.get_secret("peer", purpose) == "current-shared-secret"
        assert store.get_secret("peer", "api_key") == (
            "other-profile-new-value" if changed == "changed" else None
        )
        assert backend.deleted == deleted
        previous = json.loads(originals[targets])
        previous["targets"][0]["auth_reference"] = "keyring:" + purpose
        assert json.loads(targets.read_text()) == previous
        assert all(
            path.read_bytes() == value
            for path, value in originals.items()
            if path != targets
        )
        journal = Journal(tmp_path / "control", operation)
        with journal._locked(exclusive=False) as parent:
            records = journal._records(parent)
        assert records[-1].event == "rolled_back"
        assert bootstrap.startup_permission(case[5], tmp_path / "bootstrap")[0]


@contextmanager
def _interrupted(tmp_path, monkeypatch, helper, *, second=False):
    monkeypatch.setattr(crypto, "_package_resource_root", lambda: helper)
    with replacement_case(tmp_path, monkeypatch, prepared=False) as case:
        candidate, plan, store, backend, targets = _credential_candidate(
            case, tmp_path, monkeypatch
        )
        if second:
            from tldw_chatbook.Backup_Recovery.restore_plan import plan_restore
            from tldw_chatbook.Backup_Recovery.staging import stage_restore

            archive = replacement._acquired_source(candidate, plan, Event())
            document = json.loads(targets.read_bytes())
            document["targets"].append(
                {
                    "server_id": "second",
                    "base_url": "https://second.invalid",
                    "auth_reference": "keyring:api_key",
                }
            )
            targets.write_text(json.dumps(document))
            store.set_secret("second", "api_key", "second-captured-secret")
            plan = plan_restore(
                archive,
                mode="replace",
                destinations=dict((*plan.destinations, *plan.selectors)),
                target=plan.target,
                profile_names=dict(plan.profile_names),
                acknowledged_credential_issues=plan.acknowledged_credential_issues,
            )
            candidate = stage_restore(
                archive, plan, tmp_path / "second-candidate", Event()
            )
        original = targets.read_bytes()
        finalize = publication.finalize_candidate

        def stopped(*args, **kwargs):
            raise KeyboardInterrupt("actual forward publication")

        monkeypatch.setattr(publication, "finalize_candidate", stopped)
        with pytest.raises(KeyboardInterrupt):
            replacement.replace(
                plan,
                candidate,
                control_root=tmp_path / "control",
                rollback_password=b"rollback",
                cancel=Event(),
            )
        monkeypatch.setattr(publication, "finalize_candidate", finalize)
        pending, _ = bootstrap._records(tmp_path / "bootstrap")
        yield case, store, backend, targets, original, pending[0]["operation_id"]


def _recover(tmp_path, operation):
    return replacement.recover_replacement(
        operation,
        control_root=tmp_path / "control",
        action="rollback",
        rollback_password=b"rollback",
        cancel=Event(),
    )


def _records(tmp_path, operation):
    journal = Journal(tmp_path / "control", operation)
    with journal._locked(exclusive=False) as parent:
        return journal._records(parent)


@pytest.mark.parametrize(
    "boundary", ["originals_validated", "rollback_activation_recorded"]
)
def test_late_old_scope_drift_requires_new_terminal_proof(
    tmp_path, monkeypatch, helper_resource_root, boundary
):
    with _interrupted(tmp_path, monkeypatch, helper_resource_root) as (
        _case,
        store,
        _backend,
        targets,
        original,
        operation,
    ):
        append = Journal._append

        def interrupted(self, parent, event, evidence):
            append(self, parent, event, evidence)
            if event == boundary:
                raise KeyboardInterrupt("actual reverse proof written")

        monkeypatch.setattr(Journal, "_append", interrupted)
        with pytest.raises(KeyboardInterrupt):
            _recover(tmp_path, operation)
        monkeypatch.setattr(Journal, "_append", append)
        assert targets.read_bytes() == original
        store.set_secret("peer", "api_key", "late-foreign-value")
        assert _recover(tmp_path, operation) == "rolled_back"
        rows = _records(tmp_path, operation)
        last_plan = next(
            row for row in rows if row.event == "rollback_credentials_planned"
        )
        phase = rows[rows.index(last_plan) + 1 :]
        assert [row.event for row in phase][-3:] == [
            "originals_validated",
            "rollback_activation_recorded",
            "rolled_back",
        ]
        purpose = json.loads(targets.read_bytes())["targets"][0][
            "auth_reference"
        ].removeprefix("keyring:")
        assert store.get_secret("peer", purpose) == "current-shared-secret"
        assert store.get_secret("peer", "api_key") == "late-foreign-value"
        from tldw_chatbook.Backup_Recovery.isolated_restore import (
            installation_client_id,
        )

        assert len(installation_client_id()) == 32


@pytest.mark.parametrize("unavailable", ["old_only", "backend"])
def test_backend_unavailability_never_authorizes_shared_write(
    tmp_path, monkeypatch, helper_resource_root, unavailable
):
    from tldw_chatbook.Backup_Recovery import credentials

    with _interrupted(tmp_path, monkeypatch, helper_resource_root) as (
        case,
        store,
        _backend,
        targets,
        _original,
        operation,
    ):
        read = credentials._read_scope

        def failing(record, selected):
            if unavailable == "backend" or record["purpose"] == "api_key":
                raise RuntimeError("test backend failure")
            return read(record, selected)

        monkeypatch.setattr(credentials, "_read_scope", failing)
        before = {path: path.read_bytes() for path in (case[4], case[5], targets)}
        if unavailable == "backend":
            with pytest.raises(
                ValueError, match="rollback_credential_store_unavailable"
            ):
                _recover(tmp_path, operation)
            assert all(path.read_bytes() == value for path, value in before.items())
            assert not any(
                row.event == "rollback_credentials_planned"
                for row in _records(tmp_path, operation)
            )
            assert not bootstrap.startup_permission(case[5], tmp_path / "bootstrap")[0]
        else:
            assert _recover(tmp_path, operation) == "rolled_back"
            purpose = json.loads(targets.read_bytes())["targets"][0][
                "auth_reference"
            ].removeprefix("keyring:")
            assert store.get_secret("peer", purpose) == "current-shared-secret"
        assert store.get_secret("peer", "api_key") == "current-shared-secret"


class _PersistentKeyring:
    """Owned test backend; actual store operations survive abrupt child exit."""

    def __init__(self, path):
        self.path = Path(path)

    def get_password(self, service, username):
        return next(
            (
                value
                for owner, user, value in json.loads(self.path.read_text())
                if (owner, user) == (service, username)
            ),
            None,
        )

    def set_password(self, service, username, value):
        rows = [
            row
            for row in json.loads(self.path.read_text())
            if row[:2] != [service, username]
        ]
        rows.append([service, username, value])
        with self.path.open("w") as stream:
            json.dump(rows, stream)
            stream.flush()
            os.fsync(stream.fileno())

    def delete_password(self, service, username):
        raise AssertionError("recovery must not delete credential scopes")


_CHILD = r"""
import os,sys,json
from pathlib import Path
from threading import Event
from Tests.network_guard import install,blocked_attempts
install()
from Tests.Backup_Recovery.test_rollback_credentials import _PersistentKeyring
from tldw_chatbook.Backup_Recovery import bootstrap,crypto,credentials,publication,replacement
from tldw_chatbook.Backup_Recovery.journal import Journal
from tldw_chatbook.runtime_policy.server_credentials import KeyringServerCredentialStore
root,helper,operation,boundary=sys.argv[1:]
root=Path(root)
bootstrap.default_bootstrap_root=lambda:root/'bootstrap'
crypto._package_resource_root=lambda:Path(helper)
store=KeyringServerCredentialStore(keyring_backend=_PersistentKeyring(root/'backend.json'))
credentials._credential_store=lambda:store
if boundary=='value_write':
 original=store._keyring.set_password
 def write(*args):
  original(*args);os._exit(91)
 store._keyring.set_password=write
if boundary in ('rollback_credentials_planned','rollback_credential_applied','originals_validated','rollback_activation_recorded'):
 original=Journal._append
 def append(self,parent,event,evidence):
  original(self,parent,event,evidence)
  if event==boundary:os._exit(91)
 Journal._append=append
if boundary.startswith('credential_'):
 original=publication._reverse_native_move
 def move(intent):
  original(intent)
  if intent.step==boundary:os._exit(91)
 publication._reverse_native_move=move
result=replacement.recover_replacement(operation,control_root=root/'control',action='rollback',rollback_password=b'rollback',cancel=Event())
assert result=='rolled_back',result
from tldw_chatbook.Backup_Recovery.isolated_restore import installation_client_id
assert len(installation_client_id())==32
assert not blocked_attempts(),blocked_attempts()
print(result)
"""


def _child(tmp_path, helper, operation, boundary, expected):
    log = tmp_path / ("recovery-" + (boundary or "finished") + ".log")
    with log.open("w+") as stream:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                _CHILD,
                str(tmp_path),
                str(helper),
                operation,
                boundary,
            ],
            stdout=stream,
            stderr=stream,
            timeout=45,
            check=False,
        )
        stream.seek(0)
        assert result.returncode == expected, stream.read()[-8000:]


@pytest.mark.parametrize(
    "boundary",
    [
        "rollback_credentials_planned",
        "value_write",
        "rollback_credential_applied",
        "credential_publish",
        "credential_retire",
    ],
)
def test_fresh_process_kill_reuses_exact_planned_scope(
    tmp_path, monkeypatch, helper_resource_root, boundary
):
    from tldw_chatbook.runtime_policy.server_credentials import (
        KeyringServerCredentialStore,
    )

    with _interrupted(tmp_path, monkeypatch, helper_resource_root) as (
        case,
        store,
        backend,
        targets,
        _original,
        operation,
    ):
        if boundary == "credential_retire":
            append = Journal._append

            def interrupted(self, parent, event, evidence):
                append(self, parent, event, evidence)
                if event == "originals_validated":
                    raise KeyboardInterrupt("originals are live before late amendment")

            monkeypatch.setattr(Journal, "_append", interrupted)
            with pytest.raises(KeyboardInterrupt):
                _recover(tmp_path, operation)
            monkeypatch.setattr(Journal, "_append", append)
        store.set_secret("peer", "api_key", "foreign-shared-value")
        saved = tmp_path / "backend.json"
        saved.write_text(
            json.dumps([[*key, value] for key, value in backend.values.items()])
        )
        saved.chmod(0o600)
        _child(tmp_path, helper_resource_root, operation, boundary, 91)
        rows = _records(tmp_path, operation)
        intended = next(
            row.evidence for row in rows if row.event == "rollback_credentials_planned"
        )
        purpose = next(iter(intended["scopes"].values()))["purpose"]
        assert not bootstrap.startup_permission(case[5], tmp_path / "bootstrap")[0]
        _child(tmp_path, helper_resource_root, operation, "", 0)
        assert (
            json.loads(targets.read_bytes())["targets"][0]["auth_reference"]
            == "keyring:" + purpose
        )
        actual = KeyringServerCredentialStore(keyring_backend=_PersistentKeyring(saved))
        assert actual.get_secret("peer", purpose) == "current-shared-secret"
        assert actual.get_secret("peer", "api_key") == "foreign-shared-value"
        rows = _records(tmp_path, operation)
        assert sum(row.event == "rollback_credentials_planned" for row in rows) == 1
        assert rows[-1].event == "rolled_back"


@pytest.mark.parametrize(
    ("boundary", "damage"),
    [
        ("rollback_credentials_planned", "foreign"),
        ("rollback_credential_applied", "missing"),
        ("rollback_credentials_planned", "candidate"),
    ],
)
def test_interrupted_plan_rejects_changed_value_or_candidate(
    tmp_path, monkeypatch, helper_resource_root, boundary, damage
):
    with _interrupted(tmp_path, monkeypatch, helper_resource_root) as (
        case,
        store,
        _backend,
        targets,
        _original,
        operation,
    ):
        store.set_secret("peer", "api_key", "foreign-shared-value")
        before = {path: path.read_bytes() for path in (case[4], case[5], targets)}
        append = Journal._append

        def interrupted(self, parent, event, evidence):
            append(self, parent, event, evidence)
            if event == boundary:
                raise KeyboardInterrupt("durable credential progress")

        monkeypatch.setattr(Journal, "_append", interrupted)
        with pytest.raises(KeyboardInterrupt):
            _recover(tmp_path, operation)
        monkeypatch.setattr(Journal, "_append", append)
        rows = _records(tmp_path, operation)
        plan = next(
            row.evidence for row in rows if row.event == "rollback_credentials_planned"
        )
        purpose = next(iter(plan["scopes"].values()))["purpose"]
        if damage == "foreign":
            store.set_secret("peer", purpose, "other-owner-value")
        elif damage == "missing":
            store.delete_secret("peer", purpose)
        else:
            Path(plan["artifacts"][0]["candidate"]["path"]).write_text('{"targets":[]}')
        with pytest.raises(
            ValueError, match="rollback_(credential_value_changed|originals_unverified)"
        ):
            _recover(tmp_path, operation)
        assert all(path.read_bytes() == value for path, value in before.items())
        assert not bootstrap.startup_permission(case[5], tmp_path / "bootstrap")[0]
        assert store.get_secret("peer", "api_key") == "foreign-shared-value"
        with pytest.raises(ValueError, match="rollback_direction_selected"):
            replacement.recover_replacement(
                operation,
                control_root=tmp_path / "control",
                action="finish",
                rollback_password=b"rollback",
                cancel=Event(),
            )


def test_private_candidate_capacity_refuses_before_values_or_reverse_effects(
    tmp_path, monkeypatch, helper_resource_root
):
    from tldw_chatbook.Backup_Recovery import space

    with _interrupted(tmp_path, monkeypatch, helper_resource_root) as (
        case,
        store,
        backend,
        targets,
        _original,
        operation,
    ):
        store.set_secret("peer", "api_key", "foreign-shared-value")
        before = {path: path.read_bytes() for path in (case[4], case[5], targets)}
        values = dict(backend.values)
        check = space.require_capacity

        def full(requirements):
            if any("retained" in str(path) for path in requirements):
                raise ValueError("insufficient_space")
            check(requirements)

        monkeypatch.setattr(space, "require_capacity", full)
        with pytest.raises(ValueError, match="insufficient_space"):
            _recover(tmp_path, operation)
        assert backend.values == values
        assert all(path.read_bytes() == value for path, value in before.items())
        assert not any(
            row.event == "rollback_credentials_planned"
            for row in _records(tmp_path, operation)
        )


def test_actual_installed_rejection_uses_authenticated_value_in_same_session(
    tmp_path, monkeypatch, helper_resource_root
):
    monkeypatch.setattr(crypto, "_package_resource_root", lambda: helper_resource_root)
    with replacement_case(tmp_path, monkeypatch, prepared=False) as case:
        candidate, plan, store, _backend, targets = _credential_candidate(
            case, tmp_path, monkeypatch
        )
        validate = publication._validate_installed

        def reject(*args):
            validate(*args)
            store.set_secret("peer", "api_key", "foreign-shared-value")
            raise ValueError("actual installed validation rejection")

        monkeypatch.setattr(publication, "_validate_installed", reject)
        sessions = []
        reverse = replacement._rollback_replacement

        def checked(journal, prepared, session, material, cancel):
            assert session is material.session
            session._check()
            assert not bootstrap.startup_permission(case[5], tmp_path / "bootstrap")[0]
            sessions.append(session)
            return reverse(journal, prepared, session, material, cancel)

        monkeypatch.setattr(replacement, "_rollback_replacement", checked)
        with pytest.raises(ValueError, match="replacement_rolled_back"):
            replacement.replace(
                plan,
                candidate,
                control_root=tmp_path / "control",
                rollback_password=b"rollback",
                cancel=Event(),
            )
        assert len(sessions) == 1
        purpose = json.loads(targets.read_bytes())["targets"][0][
            "auth_reference"
        ].removeprefix("keyring:")
        assert store.get_secret("peer", purpose) == "current-shared-secret"
        assert store.get_secret("peer", "api_key") == "foreign-shared-value"
        assert bootstrap.startup_permission(case[5], tmp_path / "bootstrap")[0]


def test_second_amendment_preserves_prior_scope_and_raw_original(
    tmp_path, monkeypatch, helper_resource_root
):
    with _interrupted(tmp_path, monkeypatch, helper_resource_root, second=True) as (
        _case,
        store,
        _backend,
        targets,
        original,
        operation,
    ):
        store.set_secret("peer", "api_key", "first-foreign")
        append = Journal._append

        def stopped(self, parent, event, evidence):
            append(self, parent, event, evidence)
            if event == "rollback_activation_recorded":
                raise KeyboardInterrupt("first credential-aware rollback validation")

        monkeypatch.setattr(Journal, "_append", stopped)
        with pytest.raises(KeyboardInterrupt):
            _recover(tmp_path, operation)
        monkeypatch.setattr(Journal, "_append", append)
        first = json.loads(targets.read_bytes())["targets"][0]["auth_reference"]
        store.set_secret("second", "api_key", "second-foreign")
        assert _recover(tmp_path, operation) == "rolled_back"
        active = json.loads(targets.read_bytes())["targets"]
        assert active[0]["auth_reference"] == first
        assert [
            store.get_secret(
                row["server_id"], row["auth_reference"].removeprefix("keyring:")
            )
            for row in active
        ] == ["current-shared-secret", "second-captured-secret"]
        rows = _records(tmp_path, operation)
        assert sum(row.event == "rollback_credentials_planned" for row in rows) == 2
        prepared = next(row.evidence for row in rows if row.event == "prepared")
        artifact = next(
            row for row in prepared["artifacts"] if row["target"] == str(targets)
        )
        assert Path(artifact["retained"]).read_bytes() == original
        assert any(
            row.event == "move_intended"
            and row.evidence["step"] == "credential_unpublish"
            for row in rows
        )


def test_missing_previously_applied_carried_scope_refuses(
    tmp_path, monkeypatch, helper_resource_root
):
    with _interrupted(tmp_path, monkeypatch, helper_resource_root, second=True) as (
        case,
        store,
        backend,
        targets,
        _original,
        operation,
    ):
        store.set_secret("peer", "api_key", "first-foreign")
        append = Journal._append

        def first_stop(self, parent, event, evidence):
            append(self, parent, event, evidence)
            if event == "rollback_activation_recorded":
                raise KeyboardInterrupt("first activation durable")

        monkeypatch.setattr(Journal, "_append", first_stop)
        with pytest.raises(KeyboardInterrupt):
            _recover(tmp_path, operation)
        monkeypatch.setattr(Journal, "_append", append)
        purpose = json.loads(targets.read_bytes())["targets"][0][
            "auth_reference"
        ].removeprefix("keyring:")
        assert store.get_secret("peer", purpose) == "current-shared-secret"
        store.set_secret("second", "api_key", "second-foreign")

        def second_stop(self, parent, event, evidence):
            append(self, parent, event, evidence)
            if event == "rollback_credentials_planned":
                raise KeyboardInterrupt("second amendment durable before new receipts")

        monkeypatch.setattr(Journal, "_append", second_stop)
        with pytest.raises(KeyboardInterrupt):
            _recover(tmp_path, operation)
        monkeypatch.setattr(Journal, "_append", append)
        rows = _records(tmp_path, operation)
        assert sum(row.event == "rollback_credentials_planned" for row in rows) == 2
        assert rows[-1].event == "rollback_credentials_planned"
        store.delete_secret("peer", purpose)
        before = targets.read_bytes()
        values = dict(backend.values)
        try:
            outcome = _recover(tmp_path, operation)
        except ValueError as error:
            assert str(error) == "rollback_credential_value_changed"
            assert store.get_secret("peer", purpose) is None
            assert targets.read_bytes() == before
            assert backend.values == values
            assert not bootstrap.startup_permission(case[5], tmp_path / "bootstrap")[0]
            assert (
                _records(tmp_path, operation)[-1].event
                == "rollback_credentials_planned"
            )
        else:
            assert False, (
                "unexpected recovery",
                outcome,
                "deleted applied scope recreated",
                store.get_secret("peer", purpose),
                "terminal",
                _records(tmp_path, operation)[-1].event,
            )
