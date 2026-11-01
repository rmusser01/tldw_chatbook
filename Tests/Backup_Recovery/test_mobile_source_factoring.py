"""Fixed-source factoring controls; branch doubles grant no operation authority."""

from types import SimpleNamespace

import pytest

from tldw_chatbook.Backup_Recovery import archive_reader, later_rollback
from tldw_chatbook.Backup_Recovery.mobile_canonical_copy import _CanonicalCopy


@pytest.mark.parametrize("mobile,expected", [(True, 1), (False, 2)])
def test_adjacent_planner_source_verification_counts(monkeypatch, mobile, expected):
    # Only dispatch/count behavior is isolated here. Actual journal authority
    # and pending refusal are exercised by the native capture control below.
    proof = _CanonicalCopy(None, "digest", False, None) if mobile else object()
    archive = SimpleNamespace(digest="archive")
    source, target, budget, cancel = object(), object(), object(), object()
    calls = []
    hashes = []
    result = {"member": object()}

    def verify(actual_source, mode, digest, actual_archive, **kwargs):
        assert (actual_source, mode, digest, actual_archive) == (
            source,
            "replace",
            "archive",
            archive,
        )
        assert kwargs == {"_read_budget": budget, "cancel": cancel}
        calls.append("verify")
        archive_reader.verify_sealed(actual_archive)
        return object(), proof

    def generic(actual_source, actual_archive, actual_target, **kwargs):
        assert actual_target is target
        verify(
            actual_source, "replace", actual_archive.digest, actual_archive, **kwargs
        )
        archive_reader.verify_sealed(actual_archive)
        return result

    from tldw_chatbook.Backup_Recovery import mobile_canonical_copy

    monkeypatch.setattr(later_rollback, "_verify_snapshot_source", verify)
    monkeypatch.setattr(later_rollback, "_preserved_snapshot_members", generic)
    monkeypatch.setattr(
        archive_reader, "verify_sealed", lambda actual: hashes.append(actual) or actual
    )
    monkeypatch.setattr(
        mobile_canonical_copy,
        "_mobile_preserved_members",
        lambda actual_proof, actual_archive, actual_target, **kwargs: result,
    )
    assert (
        later_rollback._plan_snapshot_members(
            source, "replace", archive, target, _read_budget=budget, cancel=cancel
        )
        is result
    )
    assert calls == ["verify"] * expected
    assert hashes == [archive] * (expected + 1)


def test_planner_keeps_real_mode_validation(monkeypatch):
    source = object()
    archive = SimpleNamespace(digest="archive")
    with pytest.raises(ValueError, match="local_snapshot_source_required"):
        later_rollback._plan_snapshot_members(source, "merge", archive, object())


def test_owned_reader_requires_existing_budget_and_actual_journal(tmp_path):
    from tldw_chatbook.Backup_Recovery.journal import Journal
    from tldw_chatbook.Chat.console_trace_read_budget import _TraceReadBudget

    with pytest.raises(TypeError, match="invalid_snapshot_journal"):
        later_rollback._accounted_snapshot_evidence(object(), _TraceReadBudget())
    control = tmp_path / "control"
    control.mkdir(mode=0o700)
    journal = Journal(control, "fixed")
    with pytest.raises(TypeError, match="existing_snapshot_budget_required"):
        later_rollback._accounted_snapshot_evidence(journal, None)
    with pytest.raises(TypeError):
        later_rollback._accounted_snapshot_evidence(
            journal, _TraceReadBudget(), rows=[]
        )


def test_native_capture_pending_default_counts_and_fresh_census(tmp_path, monkeypatch):
    import hashlib
    import json
    from threading import Event

    from Tests.Backup_Recovery import mobile_consolidation_fixture
    from Tests.Backup_Recovery.test_mobile_consolidation_lifecycle import (
        _capture_fixture,
    )
    from Tests.Backup_Recovery.test_mobile_owning_transactions import _small_native_seed
    from Tests.network_guard import blocked_attempts
    from Tests.real_profile_guard import take_violations
    from tldw_chatbook.Backup_Recovery.journal import Journal
    from tldw_chatbook.Backup_Recovery.mobile_canonical_copy import _canonical_copy
    from tldw_chatbook.Backup_Recovery.mobile_consolidation_apply import (
        _captured_mobile,
    )
    from tldw_chatbook.Backup_Recovery.restore_plan import LocalSnapshotSource
    from tldw_chatbook.Backup_Recovery.storage_admission import _CaptureFileDescriptors
    from tldw_chatbook.Chat.console_trace_read_budget import _TraceReadBudget

    monkeypatch.setattr(mobile_consolidation_fixture, "_seed", _small_native_seed)
    before = {}
    reads, retired = [], []
    report = {}
    try:
        with _capture_fixture(tmp_path, monkeypatch) as values:
            session, journal, _, stores, preview, inventory, _ = values
            paths = (*preview.capture_scope, *stores.paths)
            before = {
                str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in paths
                if p.is_file()
            }
            destination = tmp_path / "captures"
            destination.mkdir(mode=0o700)
            with _captured_mobile(
                session,
                journal,
                preview,
                inventory.scope_digest,
                destination=destination,
                password=b"isolated-source-factoring",
                cancel=Event(),
            ):
                pass
        proof = _canonical_copy(journal)
        assert proof is not None and proof.pending
        source = LocalSnapshotSource(
            journal.root.parent, journal.operation_id, proof.evidence_digest
        )
        native_records, native_retire = Journal._records, _CaptureFileDescriptors.retire

        def records(actual, *args, **kwargs):
            if actual.root == journal.root:
                reads.append(kwargs.get("_budget"))
            return native_records(actual, *args, **kwargs)

        def retire(actual):
            result = native_retire(actual)
            assert (
                not actual.fds
                and not actual.directory_iterators
                and not actual.close_failed
            )
            retired.append(actual)
            return result

        monkeypatch.setattr(Journal, "_records", records)
        monkeypatch.setattr(_CaptureFileDescriptors, "retire", retire)
        budget = _TraceReadBudget()
        with pytest.raises(ValueError, match="recovery_copy_pending"):
            later_rollback._verify_snapshot_source(
                source, "replace", "unused", _read_budget=budget
            )
        assert reads == [budget]
        report["explicit_reads"] = len(reads)
        report["explicit_remaining"] = budget._remaining
        assert retired
        reads.clear()
        with pytest.raises(ValueError, match="recovery_copy_pending"):
            later_rollback._verify_snapshot_source(source, "replace", "unused")
        assert (
            len(reads) == 2 and reads[0] is None and type(reads[1]) is _TraceReadBudget
        )
        report["default_reads"] = len(reads)
        reads.clear()
        exhausted = _TraceReadBudget()
        exhausted.reserve(exhausted._remaining)
        with pytest.raises(Exception) as refusal:
            later_rollback._verify_snapshot_source(
                source, "replace", "unused", _read_budget=exhausted
            )
        from tldw_chatbook.Chat.console_trace_read_budget import _TraceReadLimit

        assert type(refusal.value) is _TraceReadLimit and not reads
        unexpected = journal.root / "unexpected-control-artifact"
        unexpected.write_bytes(b"retained negative evidence")
        unexpected.chmod(0o600)
        with pytest.raises(ValueError, match="journal_sequence_invalid"):
            later_rollback._verify_snapshot_source(
                source, "replace", "unused", _read_budget=_TraceReadBudget()
            )
        assert (
            len(reads) == 1 and unexpected.read_bytes() == b"retained negative evidence"
        )
        report.update(retired_custodies=len(retired), unknown_artifact_retained=True)
    finally:
        from pathlib import Path

        report.update(
            before=before,
            after={p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in before},
            profile_guard=take_violations(),
            network_guard=blocked_attempts(),
        )
        (tmp_path / "source-factoring-native.json").write_text(
            json.dumps(report, indent=2)
        )
    assert report["before"] == report["after"]
    assert not report["profile_guard"] and not report["network_guard"]


@pytest.mark.parametrize("mismatch", [False, True])
def test_genuine_current_preview_budget_identity_and_no_nested_issue(
    tmp_path, monkeypatch, mismatch
):
    from tldw_chatbook.Backup_Recovery.journal import Journal
    from tldw_chatbook.Backup_Recovery.mobile_preview_budget import current
    from tldw_chatbook.Backup_Recovery.restore_plan import LocalSnapshotSource
    from tldw_chatbook.Backup_Recovery.storage_admission import _preview_reads
    from tldw_chatbook.Chat.console_trace_read_budget import _TraceReadBudget

    control = tmp_path / "control"
    control.mkdir(mode=0o700)
    journal = Journal(control, "current-budget")
    source = LocalSnapshotSource(control, journal.operation_id, "0" * 64)
    budget, other = _TraceReadBudget(), _TraceReadBudget()
    native_records = Journal._records
    observed = []

    class ObservedNativeRead(Exception):
        pass

    def records(actual, parent, **kwargs):
        result = native_records(actual, parent, **kwargs)
        assert result == [] and kwargs["_budget"] is budget
        observed.append(actual.root)
        raise ObservedNativeRead

    monkeypatch.setattr(Journal, "_records", records)
    with _preview_reads(_read_budget=budget):
        scope = current()
        assert scope is not None and scope.read_budget is budget
        monkeypatch.setattr(
            _TraceReadBudget,
            "__init__",
            lambda _: pytest.fail("nested allowance issued"),
        )
        if mismatch:
            with pytest.raises(ValueError, match="mobile_preview_budget_changed"):
                later_rollback._verify_snapshot_source(
                    source, "replace", "unused", _read_budget=other
                )
            assert not observed
        else:
            # Genuine native empty-journal parsing reaches the observer once;
            # it supplies no fabricated source/terminal model or authority.
            with pytest.raises(ObservedNativeRead):
                later_rollback._verify_snapshot_source(source, "replace", "unused")
            assert observed == [journal.root]
    assert not scope.active
    assert all(
        not resource.fds
        and not resource.directory_iterators
        and not resource.close_failed
        for resource in scope.resources
    )
