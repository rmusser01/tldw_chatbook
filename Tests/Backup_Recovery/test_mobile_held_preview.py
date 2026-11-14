"""Fixed explicit mobile preservation allowance; no fabricated positive authority."""

from threading import Event

import pytest

from tldw_chatbook.Backup_Recovery import journal, storage_admission
from tldw_chatbook.Backup_Recovery.mobile_builtin_preservation import (
    _mobile_bound_sources,
    _preservation_budget,
)
from tldw_chatbook.Chat.console_trace_read_budget import (
    _TraceReadBudget,
    _TraceReadLimit,
)


@pytest.mark.parametrize("mode", ["normal", "cancel", "changed", "exhausted"])
def test_explicit_match_keeps_one_allowance_and_retires_real_resources(tmp_path, mode):
    path = tmp_path / "cipher-😀"
    path.write_bytes(b"native ciphertext representation")
    path.chmod(0o600)
    expected = journal._Object.model_validate(journal.observe_artifact(path))
    budget = _TraceReadBudget()
    custody = storage_admission._CaptureFileDescriptors(None)
    cancel = Event()
    if mode == "cancel":
        cancel.set()
    elif mode == "changed":
        path.write_bytes(b"changed native bytes")
    elif mode == "exhausted":
        budget.reserve(budget._remaining)
    try:
        if mode == "exhausted":
            with pytest.raises(_TraceReadLimit, match="mobile_trace_input_limit"):
                journal._matches(
                    expected,
                    str(path),
                    cancel=cancel,
                    _read_budget=budget,
                    _custody=custody,
                )
        else:
            assert journal._matches(
                expected,
                str(path),
                cancel=cancel,
                _read_budget=budget,
                _custody=custody,
            ) is (mode == "normal")
        assert (
            not custody.fds
            and not custody.directory_iterators
            and not custody.close_failed
        )
        assert budget._remaining < 16 * 1024**2
    finally:
        custody.retire()
    assert path.exists()


def test_explicit_match_owns_and_closes_its_native_custody(tmp_path, monkeypatch):
    path = tmp_path / "cipher"
    path.write_bytes(b"native bytes")
    expected = journal._Object.model_validate(journal.observe_artifact(path))
    native = storage_admission._CaptureFileDescriptors.retire
    retired = []

    def retire(self):
        native(self)
        assert not self.fds and not self.directory_iterators and not self.close_failed
        retired.append(self)

    monkeypatch.setattr(storage_admission._CaptureFileDescriptors, "retire", retire)
    assert journal._matches(expected, str(path), _read_budget=_TraceReadBudget())
    assert len(retired) == 1


def test_explicit_allowance_cannot_replace_current_native_allowance():
    first, foreign = _TraceReadBudget(), _TraceReadBudget()
    assert _preservation_budget(first) is first
    with storage_admission._preview_reads(_read_budget=first):
        assert _preservation_budget() is first
        assert _preservation_budget(first) is first
        with pytest.raises(ValueError, match="mobile_preview_budget_changed"):
            _preservation_budget(foreign)
    with pytest.raises(TypeError, match="invalid_mobile_preservation_budget"):
        _preservation_budget(object())


def test_non_native_candidate_objects_never_issue_capture_scope(tmp_path):
    assert getattr(storage_admission._local, "capture_scope", None) is None
    with (
        pytest.raises(
            ValueError, match="local_snapshot_builtin_native_session_required"
        ),
        _mobile_bound_sources(
            object(),
            object(),
            object(),
            "0" * 64,
            tmp_path,
            object(),
            Event(),
            _read_budget=_TraceReadBudget(),
        ),
    ):
        pytest.fail("caller objects acquired native source scope")
    assert getattr(storage_admission._local, "capture_scope", None) is None


def test_genuine_mobile_held_preview_stops_before_staging(tmp_path, monkeypatch):
    """Authenticate the real produced copy; deliberately perform no staging."""
    import hashlib
    import json
    from contextlib import contextmanager
    from pathlib import Path

    from Tests.Backup_Recovery.test_mobile_builtin_preservation import (
        test_existing_bound_public_builtin_start,
    )
    from tldw_chatbook.Backup_Recovery import (
        later_rollback,
        mobile_builtin_preservation,
        staging,
    )

    scopes, previews, stages, nested_budgets = [], [], [], []
    bound_sources = mobile_builtin_preservation._mobile_bound_sources
    preview = later_rollback._preview
    verify_source = later_rollback._verify_snapshot_source
    phases = []
    source_calls = 0
    held_budget = None

    def observed_source(*args, **kwargs):
        nonlocal source_calls
        budget = kwargs.get("_read_budget")
        if budget is not held_budget or budget is None:
            return verify_source(*args, **kwargs)
        index = source_calls
        source_calls += 1
        phase = ("entry", "planner", "exit")[index] if index < 3 else None
        if phase is not None:
            phases.append(
                {"phase": phase, "point": "before", "remaining": budget._remaining}
            )
        try:
            return verify_source(*args, **kwargs)
        finally:
            if phase is not None:
                phases.append(
                    {"phase": phase, "point": "after", "remaining": budget._remaining}
                )

    @contextmanager
    def observed_sources(*args, **kwargs):
        nonlocal held_budget
        held_budget = kwargs["_read_budget"]
        prefix_remaining = held_budget._remaining
        actual = None
        native_init = _TraceReadBudget.__init__

        def observed_init(value):
            native_init(value)
            nested_budgets.append(value)

        # The caller has already issued kwargs['_read_budget']; cover the
        # native authenticated prefix, its body and its complete retirement.
        with monkeypatch.context() as observer:
            observer.setattr(_TraceReadBudget, "__init__", observed_init)
            try:
                with bound_sources(*args, **kwargs) as issued:
                    _members, _records, budget, scope = issued
                    assert budget is kwargs["_read_budget"]
                    assert storage_admission._local.capture_scope is scope
                    actual = {
                        "scope": scope,
                        "budget": budget,
                        "session": scope.session,
                        "remaining_entry": budget._remaining,
                        "retired": False,
                    }
                    scopes.append(actual)
                    yield issued
            finally:
                if actual is not None:
                    scope = actual["scope"]
                    actual.update(
                        retired=not scope.active,
                        resources_remaining=len(scope.resources),
                        sqlite_directory_removed=scope.sqlite_directory is None,
                        current_scope_cleared=getattr(
                            storage_admission._local, "capture_scope", None
                        )
                        is not scope,
                        remaining_exit=actual["budget"]._remaining,
                    )
                evidence = {
                    "observation_point": "source_context_exit_before_staging",
                    "scope_issued": actual is not None,
                    "prefix_remaining_before": prefix_remaining,
                    "source_verification_calls": source_calls,
                    "source_phases": phases,
                    "nested_allowances": len(nested_budgets),
                    "held_preview_calls": len(previews),
                    "staging_calls": len(stages),
                    "remaining_after_source_context": kwargs["_read_budget"]._remaining,
                    "scope": None
                    if actual is None
                    else {
                        key: actual[key]
                        for key in (
                            "remaining_entry",
                            "remaining_exit",
                            "retired",
                            "resources_remaining",
                            "sqlite_directory_removed",
                            "current_scope_cleared",
                        )
                    },
                }
                (tmp_path / "builtin-held-preview-retirement.json").write_text(
                    json.dumps(evidence, indent=2)
                )

    def observed_preview(*args, **kwargs):
        if kwargs.get("session") is not None:
            assert len(scopes) == 1
            actual = scopes[0]
            assert kwargs["_read_budget"] is actual["budget"]
            assert kwargs["session"] is actual["session"]
            assert storage_admission._local.capture_scope is actual["scope"]
            previews.append(actual["budget"])
        return preview(*args, **kwargs)

    def target_facts(paths):
        facts = {}
        for raw in paths:
            for suffix in ("", "-wal", "-shm"):
                path = Path(raw + suffix)
                if not path.exists():
                    facts[str(path)] = None
                    continue
                info = path.stat()
                facts[str(path)] = (
                    info.st_dev,
                    info.st_ino,
                    info.st_mode,
                    info.st_size,
                    info.st_mtime_ns,
                    info.st_ctime_ns,
                    hashlib.sha256(path.read_bytes()).hexdigest(),
                )
        return facts

    class BeforeStageSentinel(Exception):
        pass

    def before_stage(archive, plan, destination, cancel, *, session):
        assert len(scopes) == 1 and len(previews) == 1 and not stages
        actual = scopes[0]
        assert actual["retired"] and actual["current_scope_cleared"]
        assert actual["resources_remaining"] == 0 and actual["sqlite_directory_removed"]
        assert session is actual["session"]
        session._check()
        report = json.loads((tmp_path / "builtin-existing-binding.json").read_text())
        paths = tuple(report["target_after"])
        stages.append(
            {
                "target_before": target_facts(paths),
                "paths": paths,
                "remaining": actual["budget"]._remaining,
                "archive_digest": archive.digest,
                "plan_archive_digest": plan.archive_digest,
                "candidate_absent": not destination.exists(),
            }
        )
        assert archive.digest == plan.archive_digest and not destination.exists()
        raise BeforeStageSentinel("test_stop_before_native_staging")

    monkeypatch.setattr(
        mobile_builtin_preservation, "_mobile_bound_sources", observed_sources
    )
    monkeypatch.setattr(later_rollback, "_preview", observed_preview)
    monkeypatch.setattr(later_rollback, "_verify_snapshot_source", observed_source)
    monkeypatch.setattr(staging, "stage_restore", before_stage)
    with pytest.raises(AssertionError):
        test_existing_bound_public_builtin_start(tmp_path, monkeypatch)
    report = json.loads((tmp_path / "builtin-existing-binding.json").read_text())
    assert len(stages) == len(scopes) == len(previews) == 1
    assert nested_budgets == []
    assert report["later_preview"] is True
    assert report["replacement"]["state"] != "succeeded"
    assert [(row["type"], row["code"]) for row in report["errors"]] == [
        ("BeforeStageSentinel", "test_stop_before_native_staging")
    ]
    assert stages[0]["target_before"] == target_facts(stages[0]["paths"])
    assert report["source_before"] == report["source_after_replacement"]
    assert report["package_before"] == report["package_after_replacement"]
    assert not report["profile_violations"] and not report["network_violations"]
    assert len(report["pairs_after_all_services_close"]) == 1
    assert report["pairs_after_all_services_close"][0]["coupled"]
    report["held_preview_component"] = {
        "reached_once": True,
        "native_source_scope_retired": scopes[0]["retired"],
        "resources_remaining": scopes[0]["resources_remaining"],
        "sqlite_directory_removed": scopes[0]["sqlite_directory_removed"],
        "same_allowance": previews[0] is scopes[0]["budget"],
        "nested_allowances": len(nested_budgets),
        "remaining_entry": scopes[0]["remaining_entry"],
        "remaining_exit": scopes[0]["remaining_exit"],
        "target_unchanged_after_sentinel": True,
        "candidate_absent": stages[0]["candidate_absent"],
        "staging_or_publication_qualified": False,
    }
    (tmp_path / "builtin-held-preview-component.json").write_text(
        json.dumps(report, indent=2)
    )
