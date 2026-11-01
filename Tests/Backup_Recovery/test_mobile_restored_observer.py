"""Local observer guard mechanics; not a positive committed-chain substitute."""

import hashlib
from types import SimpleNamespace

import pytest

from tldw_chatbook.Backup_Recovery import mobile_restored_observer as restored
from tldw_chatbook.Backup_Recovery.storage_admission import _CaptureFileDescriptors
from tldw_chatbook.Chat.console_trace_read_budget import (
    _TraceReadBudget,
    _TraceReadLimit,
)


def _leaf(tmp_path):
    path = tmp_path / "current.db"
    path.write_bytes(b"original")
    path.chmod(0o600)
    info = path.stat()
    previous = SimpleNamespace(device=info.st_dev, inode=info.st_ino + 1)
    candidate = SimpleNamespace(device=info.st_dev, inode=info.st_ino)
    target = SimpleNamespace(
        witnesses=[
            SimpleNamespace(
                path=str(path), device=previous.device, inode=previous.inode
            )
        ]
    )
    artifacts = [
        SimpleNamespace(
            target=str(path), action="publish", previous=previous, candidate=candidate
        )
    ]
    installed = {
        "artifacts": [
            {
                "path": str(path),
                "kind": "file",
                "device": info.st_dev,
                "inode": info.st_ino,
            }
        ]
    }
    return path, target, installed, artifacts


def test_main_generation_survives_data_edit_and_sidecar_rotation(tmp_path):
    path, target, installed, artifacts = _leaf(tmp_path)
    custody = _CaptureFileDescriptors(None)
    budget = _TraceReadBudget()
    try:
        before = restored._physical([target], installed, artifacts, budget, custody)
        path.write_bytes(b"legitimate later data, different size")
        wal = path.with_name(path.name + "-wal")
        wal.write_bytes(b"ephemeral")
        wal.chmod(0o600)
        assert (
            restored._physical([target], installed, artifacts, budget, custody)
            == before
        )
        wal.unlink()
        assert (
            restored._physical([target], installed, artifacts, budget, custody)
            == before
        )
    finally:
        custody.retire()
    assert not custody.fds and not custody.close_failed


@pytest.mark.parametrize("changed", ["inode", "mode", "sidecar_link"])
def test_leaf_generation_and_sidecar_safety_refuse(tmp_path, changed):
    path, target, installed, artifacts = _leaf(tmp_path)
    if changed == "inode":
        path.rename(tmp_path / "retained.db")
        path.write_bytes(b"original")
        path.chmod(0o600)
    elif changed == "mode":
        path.chmod(0o644)
    else:
        path.with_name(path.name + "-wal").symlink_to(path)
    custody = _CaptureFileDescriptors(None)
    try:
        with pytest.raises(ValueError, match="mobile_restored_target_changed"):
            restored._physical(
                [target], installed, artifacts, _TraceReadBudget(), custody
            )
    finally:
        custody.retire()
    assert not custody.fds and not custody.close_failed


def test_manifest_private_mode_precedes_decode(tmp_path, monkeypatch):
    path = tmp_path / "verified-manifest.json"
    raw = b"{}"
    path.write_bytes(raw)
    path.chmod(0o644)
    monkeypatch.setattr(
        restored, "_manifest", lambda *args: pytest.fail("unsafe manifest decoded")
    )
    custody = _CaptureFileDescriptors(None)
    try:
        with pytest.raises(ValueError, match="verified_manifest_not_private"):
            restored._manifest_bytes(
                SimpleNamespace(root=tmp_path),
                hashlib.sha256(raw).hexdigest(),
                _TraceReadBudget(),
                custody,
            )
    finally:
        custody.retire()
    assert not custody.fds and not custody.close_failed


def test_zero_allowance_refuses_before_manifest_open(tmp_path, monkeypatch):
    monkeypatch.setattr(
        restored,
        "_regular",
        lambda *args, **kwargs: pytest.fail("opened before pricing"),
    )
    budget = _TraceReadBudget()
    budget.reserve(budget._remaining)
    custody = _CaptureFileDescriptors(None)
    try:
        with pytest.raises(_TraceReadLimit):
            restored._manifest_bytes(
                SimpleNamespace(root=tmp_path), "f" * 64, budget, custody
            )
    finally:
        custody.retire()


def assert_actual_restored_gate_controls(
    tmp_path, monkeypatch, preview, stores, mobile_journal, restored_journal, budgets
):
    """Probe the actual integration's restored chain; never manufacture a positive."""
    import json
    import os
    from pathlib import Path

    from Tests.Backup_Recovery.config_test_support import select_config_source
    from tldw_chatbook.Backup_Recovery import bootstrap
    from tldw_chatbook.Backup_Recovery.activation import ActivationStore
    from tldw_chatbook.Backup_Recovery.mobile_consolidation import (
        _source_claim_locations,
        require_mobile_cutover,
    )
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    outcomes = []
    attempts = []
    report = {"scope": "genuine restored target; exact public gate and native writes"}

    def gate(label, expected=None, error_type=ValueError):
        start = len(budgets)
        failure = None
        try:
            if expected is None:
                assert require_mobile_cutover(preview.source_dir, stores) is None
                outcome = "accepted"
            else:
                with pytest.raises(error_type, match="^" + expected + "$") as refused:
                    require_mobile_cutover(preview.source_dir, stores)
                outcome = {
                    "type": type(refused.value).__name__,
                    "reason": str(refused.value),
                }
            actual = budgets[start:]
            assert len(actual) == 1 and actual[0]._parent is None
            outcomes.append(
                {
                    "label": label,
                    "outcome": outcome,
                    "budget_identity": id(actual[0]),
                    "remaining": actual[0]._remaining,
                }
            )
        except BaseException as error:
            failure = {"type": type(error).__name__, "reason": str(error)}
            raise
        finally:
            attempts.append(
                {
                    "label": label,
                    "failure": failure,
                    "observation_point": "gate_return_or_exception_after_native_finally",
                    "budgets": [
                        {
                            "identity": id(row),
                            "parent_identity": id(row._parent)
                            if row._parent is not None
                            else None,
                            "remaining": row._remaining,
                        }
                        for row in budgets[start:]
                    ],
                }
            )
            report["attempts"] = attempts
            report["outcomes"] = outcomes
            (tmp_path / "restored-native-controls.json").write_text(
                json.dumps(report, indent=2)
            )

    def changed_bytes(path, code):
        raw = path.read_bytes()
        info = path.stat()
        try:
            path.write_bytes(raw + b"\n")
            gate(path.name, code)
        finally:
            path.write_bytes(raw)
            assert path.read_bytes() == raw
            assert (path.stat().st_dev, path.stat().st_ino) == (
                info.st_dev,
                info.st_ino,
            )
        gate("restored-after-" + path.name)

    gate("direct-native-restore")
    physical = [(p.stat().st_dev, p.stat().st_ino) for p in stores.paths]
    edited = CharactersRAGDB(stores.chat_path, "post-restoration-native-edit")
    try:
        note = edited.add_note(
            "After direct restore", "Readable current canonical edit"
        )
        assert (
            edited.get_note_by_id(note)["content"] == "Readable current canonical edit"
        )
    finally:
        edited.close_connection()
    gate("ordinary-data-edit")
    selector = Path(preview.target_selector)
    config = select_config_source(monkeypatch, selector)
    root = bootstrap.default_bootstrap_root()
    before = bootstrap._binding(
        selector, bootstrap._records(root)[1], bootstrap._registry(root)
    )
    assert before is not None and before.get("activation")
    assert config.save_setting_to_cli_config(
        "general", "default_theme", "textual-light"
    )
    after = bootstrap._binding(
        selector, bootstrap._records(root)[1], bootstrap._registry(root)
    )
    assert after == dict(
        before, fingerprint=hashlib.sha256(selector.read_bytes()).hexdigest()
    )
    assert before["fingerprint"] != after["fingerprint"]
    from tldw_chatbook.Backup_Recovery.inventory import read_profile_config

    assert read_profile_config(selector)["general"]["default_theme"] == "textual-light"
    assert physical == [(p.stat().st_dev, p.stat().st_ino) for p in stores.paths]
    report.update(
        selector=str(selector),
        binding_before=before,
        binding_after=after,
        same_db_inodes=physical,
    )
    gate("native-theme-save")
    claim = [
        p for _lock, p in _source_claim_locations(preview.source_dir) if p.exists()
    ]
    assert len(claim) == 1
    changed_bytes(claim[0], "mobile_source_claim_changed")
    activation = after["activation"]
    required = (
        ActivationStore(Path(activation["store_root"]))._generation(
            activation["generation"]
        )
        / "required.json"
    )
    changed_bytes(required, "mobile_restored_activation_changed")
    changed_bytes(
        restored_journal.root / "verified-manifest.json", "verified_manifest_changed"
    )
    changed_bytes(restored_journal.root / "restore-plan.json", "recovery_plan_changed")
    target = stores.chat_path
    retained = tmp_path / "retained-restored-chat.db"
    raw = target.read_bytes()
    target.rename(retained)
    try:
        target.write_bytes(raw)
        target.chmod(0o600)
        gate("changed-main-inode", "mobile_restored_target_changed")
    finally:
        target.unlink()
        retained.rename(target)
    gate("restored-main-inode")
    schema_owner = CharactersRAGDB(target, "native-schema-refusal-probe")
    try:
        with schema_owner.transaction() as cursor:
            version = cursor.execute(
                "SELECT version FROM db_schema_version WHERE schema_name=?",
                ("rag_char_chat_schema",),
            ).fetchone()[0]
            cursor.execute(
                "UPDATE db_schema_version SET version=? WHERE schema_name=?",
                (version + 1, "rag_char_chat_schema"),
            )
        try:
            gate("changed-native-schema", "mobile_restored_schema_changed")
        finally:
            with schema_owner.transaction() as cursor:
                cursor.execute(
                    "UPDATE db_schema_version SET version=? WHERE schema_name=?",
                    (version, "rag_char_chat_schema"),
                )
    finally:
        schema_owner.close_connection()
    gate("restored-native-schema")
    operation = "isolated-restored-pending-refusal"
    pending = root / ("pending-" + bootstrap._key(operation) + ".json")
    value = {
        "version": 1,
        "operation_id": operation,
        "namespaces": after["namespaces"],
        "control_root": str(mobile_journal.root.parent),
        "selectors": [str(selector)],
    }
    fd = os.open(pending, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(json.dumps(value).encode())
        gate("pending-overlap", "recovery_pending", bootstrap.RecoveryRequired)
    finally:
        pending.unlink()
    gate("restored-after-all-refusals")
    assert physical == [(p.stat().st_dev, p.stat().st_ino) for p in stores.paths]
    report["completed"] = True
    (tmp_path / "restored-native-controls.json").write_text(
        json.dumps(report, indent=2)
    )
