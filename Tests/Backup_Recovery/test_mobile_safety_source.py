"""R142 native custody controls; no synthetic positive source authority."""

import pytest

from tldw_chatbook.Backup_Recovery import later_rollback
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.Backup_Recovery.journal import Journal
from tldw_chatbook.Chat.console_trace_read_budget import _TraceReadBudget


def test_native_containment_borrows_custody_without_retiring_owner(tmp_path):
    root = tmp_path / "owned"
    root.mkdir(mode=0o700)
    child = root / "leaf"
    child.write_bytes(b"current")
    child.chmod(0o600)
    custody = storage._CaptureFileDescriptors(None)
    assert storage._contains_owned_path(root, child, _custody=custody)
    assert not storage._contains_owned_path(child, root, _custody=custody)
    assert not custody.fds and not custody.close_failed
    custody.retire()


def test_failed_borrowed_custody_refuses_before_containment_open(tmp_path, monkeypatch):
    custody = storage._CaptureFileDescriptors(None)
    custody.close_failed = True

    def unexpected(*args, **kwargs):
        pytest.fail("failed custody opened a directory")

    monkeypatch.setattr(storage._CaptureFileDescriptors, "pinned_directory", unexpected)
    with pytest.raises(ValueError, match="invalid_mobile_preservation_custody"):
        storage._contains_owned_path(tmp_path, tmp_path, _custody=custody)


def test_failed_snapshot_custody_refuses_before_journal_open(tmp_path, monkeypatch):
    control = tmp_path / "control"
    control.mkdir(mode=0o700)
    journal = Journal(control, "native-custody")
    custody = storage._CaptureFileDescriptors(None)
    custody.close_failed = True

    def unexpected(*args, **kwargs):
        pytest.fail("failed custody opened journal")

    monkeypatch.setattr(Journal, "_locked", unexpected)
    with pytest.raises(ValueError, match="invalid_mobile_preservation_custody"):
        later_rollback._accounted_snapshot_evidence(
            journal, _TraceReadBudget(), _custody=custody
        )


def test_current_preview_requires_its_actual_custody_before_journal_open(
    tmp_path, monkeypatch
):
    control = tmp_path / "control"
    control.mkdir(mode=0o700)
    journal = Journal(control, "native-custody")
    budget = _TraceReadBudget()
    foreign = storage._CaptureFileDescriptors(None)

    def unexpected(*args, **kwargs):
        pytest.fail("foreign custody opened journal")

    with storage._preview_reads(_read_budget=budget):
        monkeypatch.setattr(Journal, "_locked", unexpected)
        with pytest.raises(ValueError, match="mobile_canonical_custody_changed"):
            later_rollback._accounted_snapshot_evidence(
                journal, budget, _custody=foreign
            )
    assert not foreign.fds and not foreign.close_failed
    foreign.retire()


@pytest.mark.parametrize("kind", ["failed", "foreign"])
def test_canonical_archive_refuses_bad_borrower_before_archive_io(
    tmp_path, monkeypatch, kind
):
    from tldw_chatbook.Backup_Recovery import archive_reader
    from tldw_chatbook.Backup_Recovery.archive_models import SealedArchive
    from tldw_chatbook.Backup_Recovery.mobile_canonical_copy import (
        _CanonicalCopy,
        _verify_canonical_archive,
    )

    # Negative input only: no fabricated object can become positive evidence.
    proof = _CanonicalCopy(None, "", False, None)
    archive = SealedArchive(tmp_path / "absent", "", b"")
    budget = _TraceReadBudget()
    custody = storage._CaptureFileDescriptors(None)
    custody.close_failed = kind == "failed"

    def unexpected(*args, **kwargs):
        pytest.fail("bad borrower reached archive IO")

    monkeypatch.setattr(archive_reader, "verify_sealed", unexpected)
    with (
        storage._preview_reads(_read_budget=budget),
        pytest.raises(ValueError, match="mobile_canonical_custody_changed"),
    ):
        _verify_canonical_archive(
            proof, archive, "", _read_budget=budget, _custody=custody
        )
    assert not custody.fds


def test_native_default_prepared_case_never_enters_mobile_alternative(
    tmp_path, monkeypatch
):
    from Tests.Backup_Recovery.test_held_sqlite_rollback import replacement_case
    from tldw_chatbook.Backup_Recovery import mobile_builtin_preservation, replacement

    def unexpected(*args, **kwargs):
        pytest.fail("profile-owned originals invoked mobile source proof")

    monkeypatch.setattr(
        mobile_builtin_preservation, "_mobile_safety_members", unexpected
    )
    with replacement_case(tmp_path, monkeypatch) as case:
        _, plan, journal, session, _, _ = case
        expected = replacement._checked_originals(plan, journal, session)
        budget = _TraceReadBudget()
        assert (
            replacement._checked_originals(
                plan, journal, session, _source_read_budget=budget
            )
            == expected
        )
        assert budget._remaining == 16 * 1024**2


def test_profile_owned_builtin_retains_original_path(
    tmp_path, monkeypatch, helper_resource_root
):
    from Tests.Backup_Recovery.test_held_sqlite_rollback import (
        test_shared_directory_safety_copy_keeps_each_actual_alias_tree,
    )
    from tldw_chatbook.Backup_Recovery import mobile_builtin_preservation, replacement

    native = replacement._checked_originals
    calls = []

    def checked(*args, **kwargs):
        # Genuine generic prepared case, with an unused optional allowance.
        budget = _TraceReadBudget()
        result = native(*args, **kwargs, _source_read_budget=budget)
        assert budget._remaining == 16 * 1024**2
        calls.append(result)
        return result

    def unexpected(*args, **kwargs):
        pytest.fail("profile-owned builtin invoked alternate source read")

    monkeypatch.setattr(replacement, "_checked_originals", checked)
    monkeypatch.setattr(
        mobile_builtin_preservation, "_mobile_safety_members", unexpected
    )
    test_shared_directory_safety_copy_keeps_each_actual_alias_tree(
        tmp_path, monkeypatch, helper_resource_root, "valid"
    )
    assert len(calls) >= 3


def test_genuine_committed_source_three_prepared_safety_boundaries(
    tmp_path, monkeypatch
):
    """One fresh native producer; stop after three checks, before SafetyCopy IO."""
    import hashlib
    import json
    from pathlib import Path

    from Tests.Backup_Recovery.test_mobile_builtin_preservation import (
        test_existing_bound_public_builtin_start,
    )
    from Tests.network_guard import blocked_attempts
    from Tests.real_profile_guard import take_violations
    from tldw_chatbook.Backup_Recovery import mobile_builtin_preservation, replacement
    from tldw_chatbook.Backup_Recovery.limits import ArchiveLimits
    from tldw_chatbook.Backup_Recovery.native_files import create_private_directory

    class BeforeSafetyCopy(Exception):
        pass

    calls, buckets, scopes, phases, nested = [], [], [], [], []
    facts_before = facts_after = None
    original_session = None
    native_members = mobile_builtin_preservation._mobile_safety_members
    native_retire = storage._CaptureFileDescriptors.retire

    def facts(paths):
        result = {}
        for path in paths:
            for suffix in ("", "-wal", "-shm"):
                current = Path(str(path) + suffix)
                if not current.exists():
                    result[str(current)] = None
                    continue
                info = current.stat()
                result[str(current)] = (
                    info.st_dev,
                    info.st_ino,
                    info.st_mode,
                    info.st_size,
                    info.st_mtime_ns,
                    info.st_ctime_ns,
                    hashlib.sha256(current.read_bytes()).hexdigest(),
                )
        return result

    def members(*args, **kwargs):
        budget = args[-2]
        phases.append({"point": "before", "remaining": budget._remaining})
        try:
            result = native_members(*args, **kwargs)
            calls.append(
                {
                    "budget_id": id(budget),
                    "source": args[-3].operation_id,
                    "journal": args[1].operation_id,
                    "members": len(result),
                    "session_names": args[5]._names,
                }
            )
            return result
        finally:
            phases.append({"point": "after", "remaining": budget._remaining})

    def retire(bucket):
        result = native_retire(bucket)
        if original_session is not None and bucket in original_session._scopes:
            assert (
                not bucket.fds
                and not bucket.directory_iterators
                and not bucket.close_failed
            )
            buckets.append(bucket)
        return result

    def component(
        candidate,
        plan,
        journal,
        destination,
        *,
        session,
        password,
        work_root,
        cancel,
        acknowledged_credential_issues,
        _source_archive=None,
        _source_read_budget=None,
        _source_journal=None,
    ):
        nonlocal facts_before, facts_after, original_session
        original_session = session
        assert type(_source_read_budget) is _TraceReadBudget
        assert _source_read_budget._parent is None
        assert type(_source_journal) is Journal and _source_journal is not journal
        options = {
            "_source_archive": _source_archive,
            "_source_read_budget": _source_read_budget,
            "_source_journal": _source_journal,
            "cancel": cancel,
        }
        target_paths = [
            item.path
            for item in plan.target.items
            if item.owner
            in {"db.chachanotes.primary", "db.workspaces", "db.agent_runs"}
        ]
        facts_before = facts(target_paths)
        native_init = _TraceReadBudget.__init__

        def issued(actual):
            native_init(actual)
            nested.append(actual)

        with monkeypatch.context() as observer:
            observer.setattr(_TraceReadBudget, "__init__", issued)
            try:
                first, _ = replacement._checked_originals(
                    plan, journal, session, **options
                )
                create_private_directory(work_root)
                limits = ArchiveLimits()
                with session._replacement_capture_scope(
                    plan,
                    journal,
                    work_root,
                    limits=limits,
                    byte_budget=limits.expanded_bytes,
                    **options,
                ):
                    scope = storage._local.capture_scope
                    scopes.append(scope)
                    assert scope.active and scope.session is session
                assert not scope.active and not scope.resources
                assert getattr(storage._local, "capture_scope", None) is None
                third, _ = replacement._checked_originals(
                    plan, journal, session, **options
                )
                assert third == first
                assert len(calls) == 3 and len({row["budget_id"] for row in calls}) == 1
                assert not nested and all(
                    bucket not in session._scopes for bucket in buckets
                )
                assert not destination.exists()
            finally:
                facts_after = facts(target_paths)
                assert facts_after == facts_before
        raise BeforeSafetyCopy("component stopped before native SafetyCopy capture")

    monkeypatch.setattr(mobile_builtin_preservation, "_mobile_safety_members", members)
    monkeypatch.setattr(storage._CaptureFileDescriptors, "retire", retire)
    monkeypatch.setattr(replacement, "capture_verify_rollback", component)
    try:
        with pytest.raises(AssertionError):
            test_existing_bound_public_builtin_start(tmp_path, monkeypatch)
        report = json.loads((tmp_path / "builtin-existing-binding.json").read_text())
        assert [error["type"] for error in report["errors"]] == ["BeforeSafetyCopy"], (
            report["errors"]
        )
        assert len(calls) == 3 and len(scopes) == 1 and len(buckets) == 3
        assert original_session is not None and not original_session._active
        assert facts_before is not None and facts_before == facts_after
    finally:
        evidence = {
            "scope": "genuine three pre-SafetyCopy source checks, no capture/publication",
            "calls": calls,
            "phases": phases,
            "nested_allowances": len(nested),
            "buckets_retired": len(buckets),
            "scope_retired": [
                not scope.active and not scope.resources for scope in scopes
            ],
            "session_retired": original_session is not None
            and not original_session._active,
            "target_before": facts_before,
            "target_after": facts_after,
            "profile_violations": take_violations(),
            "network_violations": blocked_attempts(),
        }
        (tmp_path / "mobile-safety-component.json").write_text(
            json.dumps(evidence, indent=2, default=str)
        )
        assert not evidence["profile_violations"] and not evidence["network_violations"]
