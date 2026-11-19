"""Copied recovery records prove completion without becoming local authority."""

from __future__ import annotations

import hashlib
import json

import pytest


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.write_text(json.dumps(value), encoding="utf-8")
    path.chmod(0o600)
    return path


def _base(root):
    bootstrap = root / "home/.config/tldw_cli/recovery-bootstrap"
    old = str(root.parent / "original unavailable" / "Data")
    _write(
        bootstrap / "ancestry-settled.json",
        {"version": 1, "root_inode": 123, "root_device": 456, "root_path": old},
    )
    _write(
        bootstrap / "admission/registry.json",
        {
            "version": 1,
            "entries": {
                "bootstrap.unbound": {
                    "roots": [old],
                    "historical": ["path:" + old, "inode:123"],
                    "pending": None,
                    "proposed": [],
                }
            },
        },
    )
    for path in (bootstrap / "unbound-owner", bootstrap / "admission/registry.lock"):
        path.touch(mode=0o600)
    return bootstrap


def _object(old, name, *, directory=False):
    return {
        "path": str(old / name),
        "device": 1,
        "inode": 2,
        "kind": "directory" if directory else "file",
        "size": 0,
        "sha256": "a" * 64,
    }


def _journal(root, terminal="committed", *, prepared_abort=False):
    old = root.parent / "original unavailable"
    selector = str(old / "config.toml")
    descriptor = _object(old, "candidate.json")
    publication = {
        "bootstrap_root": str(old / "bootstrap"),
        "namespaces": ["profile.test"],
        "selectors": [selector],
        "archive_digest": "b" * 64,
        "plan_digest": "c" * 64,
        "descriptor": descriptor,
    }
    candidate = {
        "local_plan": None,
        "stage": _object(old, "stage", directory=True),
        "descriptor": descriptor,
        "archive_digest": "b" * 64,
        "manifest_digest": "d" * 64,
        "plan_digest": "c" * 64,
    }
    prepared = {
        "generation": "g1",
        "mode": "replace"
        if terminal == "rolled_back" or prepared_abort
        else "isolated",
        "publication": publication,
        "replacement_profiles": [{"config": selector, "installation_id": "e" * 32}],
        "installed_paths": [{"path": str(old / "installed"), "device": 1, "inode": 2}],
    }
    events = [("candidate_staged", candidate)]
    if terminal == "prepublication_aborted":
        if prepared_abort:
            events.append(("prepared", prepared))
        events.append(
            (
                terminal,
                {
                    "candidate_digest": _digest(candidate),
                    "prepared_digest": _digest(prepared) if prepared_abort else None,
                    "target_fingerprint": "f" * 64,
                    "publication": publication,
                },
            )
        )
    else:
        events.append(("prepared", prepared))
        if terminal == "rolled_back":
            rollback = {
                "ciphertext": _object(old, "rollback.age"),
                "sealed_digest": "a" * 64,
                "manifest_digest": "d" * 64,
                "coverage": {},
            }
            started = {
                "prepared_digest": _digest(prepared),
                "rollback_digest": _digest(rollback),
                "generation": "f" * 32,
                "profiles": [{"config": selector, "installation_id": "1" * 32}],
                "retained_credential_scopes": [],
            }
            originals = {
                "rollback_started_digest": _digest(started),
                "artifacts": [],
                "directories": [],
            }
            activation = {
                "generation": "f" * 32,
                "selectors": [selector],
                "owners": ["hooks"],
                "originals_digest": _digest(originals),
                "records": [_object(old, "activation.json")],
            }
            events.extend(
                [
                    ("rollback_verified", rollback),
                    ("publication_started", {}),
                    ("rollback_started", started),
                    ("originals_validated", originals),
                    ("rollback_activation_recorded", activation),
                    (
                        "rolled_back",
                        {
                            "generation": "f" * 32,
                            "activation_digest": _digest(activation),
                        },
                    ),
                ]
            )
        else:
            installed = {
                "plan_digest": "c" * 64,
                "manifest_digest": "d" * 64,
                "descriptor_digest": "a" * 64,
                "artifacts": [_object(old, "installed")],
            }
            activation = {
                "generation": "g1",
                "plan_digest": "c" * 64,
                "installed_digest": _digest(installed),
                "selectors": [selector],
                "owners": ["hooks"],
                "records": [
                    _object(old, name) for name in ("required", "profile", "activation")
                ],
            }
            events.extend(
                [
                    ("publication_started", {}),
                    ("installed_validated", installed),
                    ("activation_recorded", activation),
                    (
                        "committed",
                        {
                            "generation": "g1",
                            "activation_digest": _digest(activation),
                            "catalog_digest": None,
                            "credential_digest": None,
                        },
                    ),
                ]
            )
    operation_id = "copied-operation"
    directory = (
        root
        / "home/.config/tldw_cli/recovery/control"
        / ("operation-" + hashlib.sha256(operation_id.encode()).hexdigest())
    )
    previous = ""
    for index, (event, evidence) in enumerate(events):
        record = {
            "version": 1,
            "operation_id": operation_id,
            "sequence": index,
            "previous": previous,
            "event": event,
            "evidence": evidence,
        }
        _write(directory / f"{index:06d}.json", record)
        previous = _digest(record)
    (directory / "journal.lock").touch(mode=0o600)
    return directory


def _validate(root):
    from tldw_chatbook.Backup_Recovery.detached_evidence import validate_copied_evidence

    return validate_copied_evidence(root)


def test_native_ancestry_is_inert_and_needs_no_journal(tmp_path, monkeypatch):
    from tldw_chatbook.Backup_Recovery import bootstrap, control_records, journal

    _base(tmp_path)
    before = {
        p.relative_to(tmp_path): p.read_bytes()
        for p in tmp_path.rglob("*")
        if p.is_file()
    }

    def forbidden(*args, **kwargs):
        raise AssertionError("copied evidence tried to establish local authority")

    monkeypatch.setattr(journal.Journal, "__init__", forbidden)
    monkeypatch.setattr(bootstrap, "_control_records", forbidden)
    monkeypatch.setattr(control_records, "_control_records", forbidden)
    assert _validate(tmp_path) == ()
    assert {
        p.relative_to(tmp_path): p.read_bytes()
        for p in tmp_path.rglob("*")
        if p.is_file()
    } == before


@pytest.mark.parametrize(
    "terminal", ["committed", "rolled_back", "prepublication_aborted"]
)
def test_recognized_terminal_sequences_never_open_original_paths(
    tmp_path, monkeypatch, terminal
):
    from tldw_chatbook.Backup_Recovery import journal, owner_registry

    _base(tmp_path)
    _journal(tmp_path, terminal)

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "historical proof performed live filesystem/owner discovery"
        )

    monkeypatch.setattr(journal, "observe_artifact", forbidden)
    monkeypatch.setattr(owner_registry, "install_adapters", forbidden)
    original_stat, original_open = journal.os.stat, journal.os.open
    old = str(tmp_path.parent / "original unavailable")

    def local_stat(path, *args, **kwargs):
        assert not str(path).startswith(old), "recorded original path was stat-ed"
        return original_stat(path, *args, **kwargs)

    def local_open(path, *args, **kwargs):
        assert not str(path).startswith(old), "recorded original path was opened"
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(journal.os, "stat", local_stat)
    monkeypatch.setattr(journal.os, "open", local_open)
    assert _validate(tmp_path) == ()


def test_prepared_prepublication_abort_is_recognized(tmp_path):
    _base(tmp_path)
    _journal(tmp_path, "prepublication_aborted", prepared_abort=True)
    assert _validate(tmp_path) == ()


@pytest.mark.parametrize(
    "name",
    [
        "pending-any.json",
        "activation-update-any.json",
    ],
)
def test_surviving_intents_block_even_when_contents_are_empty(tmp_path, name):
    bootstrap = _base(tmp_path)
    (bootstrap / name).touch(mode=0o600)
    assert "detached_evidence_pending" in _validate(tmp_path)


def test_registry_publication_intent_blocks(tmp_path):
    bootstrap = _base(tmp_path)
    (bootstrap / "admission/registry.pending.json").touch(mode=0o600)
    assert "detached_evidence_pending" in _validate(tmp_path)


@pytest.mark.parametrize(
    "field,value", [("pending", "operation"), ("proposed", ["C:/unavailable"])]
)
def test_unsettled_registry_entries_block(tmp_path, field, value):
    bootstrap = _base(tmp_path)
    path = bootstrap / "admission/registry.json"
    record = json.loads(path.read_text())
    record["entries"]["bootstrap.unbound"][field] = value
    _write(path, record)
    assert "detached_evidence_pending" in _validate(tmp_path)


@pytest.mark.parametrize(
    "raw",
    [
        '{"version":1,"version":1}',
        '{"version":2}',
        '{"version":true}',
        '{"version":1,"root_path":NaN}',
        "{broken",
    ],
)
def test_invalid_ancestry_syntax_blocks(tmp_path, raw):
    bootstrap = _base(tmp_path)
    (bootstrap / "ancestry-settled.json").write_text(raw)
    assert "detached_evidence_invalid" in _validate(tmp_path)


@pytest.mark.parametrize(
    "damage",
    [
        "hash",
        "sequence",
        "version",
        "duplicate",
        "terminal",
        "identity",
        "gap",
        "extra",
    ],
)
def test_corrupt_or_ambiguous_journals_block(tmp_path, damage):
    _base(tmp_path)
    directory = _journal(tmp_path)
    path = directory / "000005.json"
    record = json.loads(path.read_text())
    if damage == "hash":
        record["previous"] = "0" * 64
    elif damage == "sequence":
        record["sequence"] = 99
    elif damage == "version":
        record["version"] = 2
    elif damage == "identity":
        record["operation_id"] = "different-operation"
    elif damage == "duplicate":
        path.write_text(
            path.read_text().replace('"version": 1', '"version": 1, "version": 1')
        )
    elif damage == "terminal":
        path.unlink()
    elif damage == "gap":
        (directory / "000002.json").unlink()
    elif damage == "extra":
        (directory / "temporary.json").write_text("{}")
    if damage in {"hash", "sequence", "version", "identity"}:
        _write(path, record)
    assert _validate(tmp_path)


def test_well_formed_terminal_label_without_proof_blocks(tmp_path):
    _base(tmp_path)
    directory = _journal(tmp_path)
    for path in directory.glob("*.json"):
        if path.name != "000005.json":
            path.unlink()
    record = json.loads((directory / "000005.json").read_text())
    (directory / "000005.json").unlink()
    record.update(sequence=0, previous="")
    _write(directory / "000000.json", record)
    assert _validate(tmp_path)


def test_empty_journal_remains_unsettled(tmp_path):
    directory = _journal(tmp_path)
    for path in directory.glob("*.json"):
        path.unlink()
    assert "detached_evidence_unsettled" in _validate(tmp_path)


def test_acquired_evidence_record_budget_is_bounded(tmp_path, monkeypatch):
    from tldw_chatbook.Backup_Recovery import detached_evidence

    _base(tmp_path)
    monkeypatch.setattr(detached_evidence, "MAX_EVIDENCE_BYTES", 32)
    assert "detached_evidence_limit" in _validate(tmp_path)


def test_copied_bootstrap_unknown_records_are_not_silently_skipped(tmp_path):
    bootstrap = _base(tmp_path)
    _write(bootstrap / "future-authority.json", {"version": 1})
    assert "detached_evidence_unsupported" in _validate(tmp_path)


def _activation_store(root):
    control = root / "home/.config/tldw_cli/recovery/control"
    generation = "g1"
    directory = control / "activation" / ("generation-" + _key(generation))
    _write(
        directory / "required.json",
        {"version": 1, "generation": generation, "owners": ["hooks"]},
    )
    approval = _write(
        directory / ("approved-" + _key("hooks") + ".json"),
        {"version": 1, "generation": generation, "owner": "hooks"},
    )
    return directory, approval


def _key(value):
    return hashlib.sha256(value.encode()).hexdigest()


def test_valid_copied_approval_stays_inert(tmp_path):
    _activation_store(tmp_path)
    assert _validate(tmp_path) == ()


@pytest.mark.parametrize(
    "damage",
    [
        "version",
        "duplicate",
        "generation",
        "unknown_owner",
        "required_missing",
        "unknown",
    ],
)
def test_malformed_copied_activation_history_blocks(tmp_path, damage):
    directory, approval = _activation_store(tmp_path)
    if damage == "version":
        _write(approval, {"version": 2, "generation": "g1", "owner": "hooks"})
    elif damage == "duplicate":
        approval.write_text(
            '{"version":1,"generation":"g1","owner":"hooks","owner":"hooks"}'
        )
    elif damage == "generation":
        _write(approval, {"version": 1, "generation": "other", "owner": "hooks"})
    elif damage == "unknown_owner":
        _write(
            directory / ("approved-" + _key("unknown") + ".json"),
            {"version": 1, "generation": "g1", "owner": "unknown"},
        )
    elif damage == "required_missing":
        (directory / "required.json").unlink()
    else:
        _write(directory / "future.json", {"version": 1})
    assert _validate(tmp_path)


@pytest.mark.parametrize(
    "raw",
    [
        '{"version":2}',
        '{"version":1,"version":1}',
        '{"version":1,"profile_id":"p","config":"relative","data":"relative"}',
    ],
)
def test_copied_catalog_syntax_is_checked_without_opening_targets(tmp_path, raw):
    directory = tmp_path / "home/.config/tldw_cli/recovery/control/profile-catalog"
    path = _write(directory / (_key("p") + ".json"), {"version": 1})
    path.write_text(raw)
    assert _validate(tmp_path)


def test_valid_copied_catalog_never_checks_target_existence(tmp_path):
    directory = tmp_path / "home/.config/tldw_cli/recovery/control/profile-catalog"
    missing = tmp_path.parent / "original unavailable"
    _write(
        directory / (_key("p") + ".json"),
        {
            "version": 1,
            "profile_id": "p",
            "config": str(missing / "config.toml"),
            "data": str(missing / "data"),
        },
    )
    assert _validate(tmp_path) == ()


def test_activation_witness_requires_the_copied_publication_journal(tmp_path):
    bootstrap = _base(tmp_path)
    missing = tmp_path.parent / "original unavailable"
    selector = str(missing / "config.toml")
    witness = {
        "operation_id": "missing-operation",
        "generation": "g1",
        "owners": ["hooks"],
        "namespaces": ["profile.test"],
        "store_root": str(missing / "activation"),
    }
    _write(
        bootstrap / ("profile-" + _key(selector) + ".json"),
        {
            "version": 1,
            "selector": selector,
            "fingerprint": "a" * 64,
            "namespaces": ["profile.test"],
            "roots": [str(missing)],
            "activation": witness,
        },
    )
    _write(
        bootstrap / ("activation-" + _key(selector) + ".json"),
        {"version": 1, "selector": selector, "activation": witness},
    )
    assert "detached_evidence_unsettled" in _validate(tmp_path)


@pytest.mark.parametrize(
    "name",
    [
        "unbound-owner",
        "admission/registry.lock",
        "admission/registry.json",
        "ancestry-settled.json",
    ],
)
def test_incomplete_native_enrollment_blocks_without_repair(tmp_path, name):
    bootstrap = _base(tmp_path)
    (bootstrap / name).unlink()
    assert "detached_evidence_unsettled" in _validate(tmp_path)


def test_control_root_unknown_record_blocks(tmp_path):
    control = tmp_path / "home/.config/tldw_cli/recovery/control"
    _write(control / "service.json", {"version": 1, "kind": "backup_recovery_service"})
    _write(control / "future-authority.json", {"version": 1})
    assert "detached_evidence_unsupported" in _validate(tmp_path)


@pytest.mark.parametrize("name", ["verified-manifest.json", "restore-plan.json"])
@pytest.mark.parametrize(
    "raw", ['{"format_version":999}', '{"version":1,"version":1}', "{broken"]
)
def test_malformed_journal_companion_is_not_silently_skipped(tmp_path, name, raw):
    directory = _journal(tmp_path)
    (directory / name).write_text(raw)
    assert _validate(tmp_path)


@pytest.mark.parametrize("name", ["unbound-owner", "admission/registry.lock"])
def test_enrollment_files_cannot_be_directories(tmp_path, name):
    bootstrap = _base(tmp_path)
    path = bootstrap / name
    path.unlink()
    path.mkdir(mode=0o700)
    assert _validate(tmp_path)


def test_journal_lock_cannot_be_a_directory(tmp_path):
    directory = _journal(tmp_path)
    (directory / "journal.lock").unlink()
    (directory / "journal.lock").mkdir(mode=0o700)
    assert _validate(tmp_path)


def test_journal_validation_work_has_a_record_budget(tmp_path, monkeypatch):
    from tldw_chatbook.Backup_Recovery import detached_evidence

    _journal(tmp_path)
    monkeypatch.setattr(detached_evidence, "MAX_JOURNAL_EVENTS", 2, raising=False)
    assert "detached_evidence_limit" in _validate(tmp_path)


def test_actual_completed_native_publication_remains_parseable_after_copy(
    tmp_path, monkeypatch
):
    import shutil

    from Tests.Backup_Recovery.test_publication_finalization import finalize, installed
    from tldw_chatbook.Backup_Recovery import journal, owner_registry

    original = tmp_path / "original unavailable"
    original.mkdir(mode=0o700)
    with installed(original) as case:
        finalize(case)
    copied = tmp_path / "acquired"
    shutil.copytree(original, copied)
    config = copied / "home/.config/tldw_cli"
    config.mkdir(parents=True, mode=0o700)
    (copied / "bootstrap").rename(config / "recovery-bootstrap")
    (config / "recovery").mkdir(mode=0o700)
    control = config / "recovery/control"
    (copied / "control").rename(control)
    original.rename(tmp_path / "retained original")
    original_stat, original_open = journal.os.stat, journal.os.open

    def local_stat(path, *args, **kwargs):
        assert not str(path).startswith(str(original)), "original location was stat-ed"
        return original_stat(path, *args, **kwargs)

    def local_open(path, *args, **kwargs):
        assert not str(path).startswith(str(original)), "original location was opened"
        return original_open(path, *args, **kwargs)

    def no_discovery():
        raise AssertionError("copied records cannot discover live owner state")

    monkeypatch.setattr(journal.os, "stat", local_stat)
    monkeypatch.setattr(journal.os, "open", local_open)
    monkeypatch.setattr(owner_registry, "install_adapters", no_discovery)
    assert _validate(copied) == ()
    manifest = next(control.glob("operation-*/verified-manifest.json"))
    saved = manifest.read_bytes()
    manifest.unlink()
    assert "detached_evidence_unsettled" in _validate(copied)
    manifest.write_bytes(saved)
    manifest.chmod(0o600)
    plan = next(control.glob("operation-*/restore-plan.json"))
    plan.unlink()
    assert "detached_evidence_unsettled" in _validate(copied)


def test_isolated_work_without_its_publication_journal_is_unsettled(tmp_path):
    control = tmp_path / "home/.config/tldw_cli/recovery/control"
    _write(control / "service.json", {"version": 1, "kind": "backup_recovery_service"})
    (control / "isolated-missing-operation").mkdir(mode=0o700)
    assert "detached_evidence_unsettled" in _validate(tmp_path)


def test_missing_journal_lock_is_not_recreated(tmp_path):
    directory = _journal(tmp_path)
    (directory / "journal.lock").unlink()
    assert "detached_evidence_unsettled" in _validate(tmp_path)
    assert not (directory / "journal.lock").exists()


def test_copied_controls_ignore_separate_payload_copies(tmp_path):
    from tldw_chatbook.Backup_Recovery.detached_evidence import validate_copied_controls

    bootstrap = _base(tmp_path)
    control = _journal(tmp_path).parent
    shadow = tmp_path / "work/source"
    _base(shadow)
    _journal(shadow)
    assert validate_copied_controls(bootstrap, control) == ()
    assert _validate(tmp_path) == ()


@pytest.mark.parametrize(
    "damage", ["absent_journal", "unfinished", "missing_bootstrap", "missing_control"]
)
def test_copied_controls_need_complete_publication_evidence(tmp_path, damage):
    from tldw_chatbook.Backup_Recovery.detached_evidence import validate_copied_controls

    bootstrap = _base(tmp_path)
    journal = _journal(tmp_path)
    control = journal.parent
    if damage == "absent_journal":
        for member in journal.iterdir():
            member.unlink()
        journal.rmdir()
    elif damage == "unfinished":
        (journal / "000005.json").unlink()
    elif damage == "missing_bootstrap":
        bootstrap = tmp_path / "missing-bootstrap"
    else:
        control = tmp_path / "missing-control"
    assert "detached_evidence_unsettled" in validate_copied_controls(bootstrap, control)


def test_copied_controls_inspect_explicit_roots_without_trusting_labels(tmp_path):
    from tldw_chatbook.Backup_Recovery.detached_evidence import validate_copied_controls

    bootstrap = _base(tmp_path)
    control = _journal(tmp_path).parent
    renamed_bootstrap = bootstrap.parent / "copied-enrollment"
    control.rename(control.parent / "copied-control")
    bootstrap.rename(renamed_bootstrap)
    control = renamed_bootstrap.parent / "recovery/copied-control"
    _write(control / "future-authority.json", {"version": 1})
    assert "detached_evidence_unsupported" in validate_copied_controls(
        renamed_bootstrap, control
    )


@pytest.mark.parametrize(
    "relative",
    [
        "skills/foo/pending-request.json",
        "skills/foo/registry.json",
        "skills/foo/journal.lock",
        "skills/foo/control/service.json",
        "skills/foo/recovery-bootstrap/ancestry-settled.json",
        "skills/foo/.tldw-bootstrap-directory-create.pending",
        "control/registry.json",
        "control/operation-" + "a" * 64 + "/000000.json",
        "operation-" + "b" * 64 + "/000000.json",
    ],
)
def test_payload_names_do_not_select_recovery_evidence_grammar(tmp_path, relative):
    _base(tmp_path)
    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.write_bytes(b"ordinary user payload, not a recovery record")
    assert _validate(tmp_path) == ()


@pytest.mark.parametrize(
    "relative",
    [
        ".",
        "home",
        "home/.config",
        "home/.config/tldw_cli",
        "home/.config/tldw_cli/recovery",
    ],
)
def test_creation_intents_at_installed_ancestor_placements_block(tmp_path, relative):
    _base(tmp_path)
    parent = tmp_path / relative
    parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    (parent / ".tldw-bootstrap-directory-create.pending").touch(mode=0o600)
    assert "detached_evidence_pending" in _validate(tmp_path)


def test_completed_operation_staging_payload_is_not_recovery_evidence(tmp_path):
    _base(tmp_path)
    control = _journal(tmp_path).parent
    work = control / "isolated-copied-operation/skills/foo"
    work.mkdir(parents=True, mode=0o700)
    for name in ("pending-request.json", "registry.json", "journal.lock"):
        (work / name).write_bytes(b"ordinary preserved staging payload")
    assert _validate(tmp_path) == ()


def test_explicit_copied_controls_check_their_installed_creation_parent(tmp_path):
    from tldw_chatbook.Backup_Recovery.detached_evidence import validate_copied_controls

    bootstrap = _base(tmp_path)
    control = _journal(tmp_path).parent
    (bootstrap.parent / ".tldw-bootstrap-directory-create.pending").touch(mode=0o600)
    assert "detached_evidence_pending" in validate_copied_controls(bootstrap, control)
