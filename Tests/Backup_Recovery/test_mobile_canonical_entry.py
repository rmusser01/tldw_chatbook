"""Caller-only doubles; these controls issue no source or replacement authority."""

from contextlib import contextmanager
from threading import Event
from types import SimpleNamespace

import pytest

from tldw_chatbook.Backup_Recovery import archive_reader, later_rollback, restore_plan
from tldw_chatbook.Backup_Recovery import mobile_canonical_copy as mobile
from tldw_chatbook.Backup_Recovery.restore_plan import LocalSnapshotSource
from tldw_chatbook.Chat.console_trace_read_budget import _TraceReadBudget


@pytest.fixture
def entry_case(tmp_path, monkeypatch):
    calls, source_archives, supplied_budgets = [], [], []
    actual = mobile._CanonicalCopy(None, "fresh", False, None)
    stale = mobile._CanonicalCopy(None, "stale", False, None)
    archive = SimpleNamespace(digest="archive")
    document = SimpleNamespace(producer_inventory=(), files=(), directories=())
    target = SimpleNamespace(items=())
    cancel = Event()
    result = SimpleNamespace(restore=("member",), retire=(), archive_digest="archive")
    state = SimpleNamespace(fail=None, actual=actual)
    native_archive = mobile._verify_canonical_archive

    def verify_archive(proof, selected, digest, **kwargs):
        assert proof is state.actual and proof is not stale
        assert selected is archive and digest == archive.digest
        assert kwargs == {"_read_budget": supplied_budgets[-1], "cancel": cancel}
        calls.append("archive")
        if state.fail == "archive":
            raise ValueError("unit_archive_failure")
        return document

    def verify_source(source, mode, digest, selected=None, **kwargs):
        assert isinstance(source, LocalSnapshotSource)
        assert mode == "replace" and digest == archive.digest
        assert kwargs["cancel"] is cancel
        supplied_budgets.append(kwargs["_read_budget"])
        source_archives.append(selected)
        calls.append("source")
        if state.fail == "source":
            raise ValueError("unit_source_failure")
        if state.fail == "cancel":
            assert cancel.is_set()
            raise InterruptedError("unit_cancelled")
        if selected is not None:
            verify_archive(state.actual, selected, digest, **kwargs)
        return object(), state.actual

    def members(proof, selected, current, **kwargs):
        assert proof is state.actual and selected is document and current is target
        assert kwargs == {"_read_budget": supplied_budgets[-1], "cancel": cancel}
        calls.append("members")
        return {}

    def plan(selected, **kwargs):
        assert selected is archive
        calls.append("plan")
        assert (
            later_rollback._plan_snapshot_members(
                kwargs["local_snapshot"],
                kwargs["mode"],
                selected,
                kwargs["target"],
                _read_budget=kwargs["_mobile_read_budget"],
                cancel=kwargs["_mobile_cancel"],
            )
            == {}
        )
        return result

    monkeypatch.setattr(later_rollback, "_verify_snapshot_source", verify_source)
    monkeypatch.setattr(mobile, "_verify_canonical_archive", verify_archive)
    monkeypatch.setattr(mobile, "_mobile_preserved_members", members)
    monkeypatch.setattr(archive_reader, "verify_sealed", lambda selected: document)
    monkeypatch.setattr(restore_plan, "plan_restore", plan)
    monkeypatch.setattr(
        later_rollback, "_current_config_scope", lambda *args: calls.append("config")
    )
    monkeypatch.setattr(
        restore_plan, "recheck_targets", lambda selected: calls.append("targets")
    )
    return SimpleNamespace(
        state=state,
        calls=calls,
        source_archives=source_archives,
        supplied_budgets=supplied_budgets,
        actual=actual,
        stale=stale,
        archive=archive,
        document=document,
        target=target,
        cancel=cancel,
        result=result,
        native_archive=native_archive,
        journal=SimpleNamespace(root=tmp_path / "operation", operation_id="operation"),
    )


def invoke(case, budget=None):
    return mobile._preview_canonical(
        case.journal,
        case.stale,
        case.archive,
        case.target,
        (),
        _read_budget=budget,
        cancel=case.cancel,
    )


@pytest.mark.parametrize("explicit", [False, True])
def test_entry_retains_fresh_source_then_one_archive_and_full_plan_exit(
    entry_case, explicit
):
    case = entry_case
    budget = _TraceReadBudget() if explicit else None
    assert invoke(case, budget) is case.result
    assert case.source_archives == [None, case.archive, case.archive]
    assert case.calls == [
        "source",
        "archive",
        "members",
        "plan",
        "source",
        "archive",
        "members",
        "source",
        "archive",
        "config",
        "targets",
    ]
    actual_budget = case.supplied_budgets[0]
    assert type(actual_budget) is _TraceReadBudget
    assert all(value is actual_budget for value in case.supplied_budgets)
    if explicit:
        assert actual_budget is budget
    assert 0 < actual_budget._remaining < 16 * 1024**2


@pytest.mark.parametrize("failure", ["source", "archive", "cancel"])
def test_entry_refusal_stops_before_mapping(entry_case, failure):
    case = entry_case
    case.state.fail = failure
    if failure == "cancel":
        case.cancel.set()
    error = InterruptedError if failure == "cancel" else ValueError
    with pytest.raises(error, match="unit_"):
        invoke(case, _TraceReadBudget())
    assert case.source_archives == [None]
    assert case.calls == (["source", "archive"] if failure == "archive" else ["source"])


def test_actual_source_family_change_hits_native_type_refusal_before_mapping(
    entry_case, monkeypatch
):
    case = entry_case
    case.state.actual = object()
    monkeypatch.setattr(mobile, "_verify_canonical_archive", case.native_archive)
    with pytest.raises(ValueError, match="mobile_canonical_archive_required"):
        invoke(case, _TraceReadBudget())
    assert case.calls == ["source"]
    assert case.source_archives == [None]


@pytest.mark.parametrize("is_mobile", [False, True])
def test_outside_fixed_entry_default_source_dispatch_keeps_original_reads(
    tmp_path, monkeypatch, is_mobile
):
    # Native journal/crypto behavior is covered separately. These doubles assert
    # unchanged default dispatch composition, not a valid source or permission.
    from tldw_chatbook.Backup_Recovery import journal as journal_module

    calls = []
    rows = [
        SimpleNamespace(
            event="mobile_prepared" if is_mobile else "rollback_verified", evidence={}
        ),
        SimpleNamespace(
            event="mobile_committed" if is_mobile else "committed", evidence={}
        ),
    ]

    @contextmanager
    def locked(*, exclusive):
        assert exclusive is False
        calls.append("lock")
        yield 12

    def records(parent):
        assert parent == 12
        calls.append("default_read")
        return rows

    native_journal = SimpleNamespace(_locked=locked, _records=records)
    monkeypatch.setattr(later_rollback, "_journal", lambda *args: native_journal)
    monkeypatch.setattr(
        later_rollback, "_evidence_digest", lambda evidence: "source-digest"
    )
    if is_mobile:
        proof = mobile._CanonicalCopy(
            SimpleNamespace(archive=SimpleNamespace(path="cipher")),
            "source-digest",
            False,
            None,
        )

        def canonical(selected, *, _read_budget):
            assert selected is native_journal and _read_budget is None
            calls.append("default_mobile_reader")
            return proof

        monkeypatch.setattr(mobile, "_canonical_copy", canonical)
        monkeypatch.setattr(
            journal_module,
            "_matches",
            lambda *args, **kwargs: calls.append("original_cipher") or True,
        )
    else:
        proof = SimpleNamespace(sealed_digest="archive")
        monkeypatch.setattr(
            later_rollback._Rollback, "model_validate", lambda evidence: proof
        )
    source = LocalSnapshotSource(tmp_path, "operation", "source-digest")
    assert later_rollback._verify_snapshot_source(source, "replace", "archive") == (
        native_journal,
        proof,
    )
    assert calls == (
        ["lock", "default_read", "default_mobile_reader", "original_cipher"]
        if is_mobile
        else ["lock", "default_read"]
    )
