"""Local pricing mechanics; upstream doubles confer no recovery authority."""

from contextlib import contextmanager
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pytest

from tldw_chatbook.Backup_Recovery import mobile_native_operation as operation
from tldw_chatbook.Backup_Recovery import mobile_terminal_pricing as pricing
from tldw_chatbook.Chat.console_trace_read_budget import (
    _TraceReadBudget,
    _TraceReadLimit,
)


@pytest.mark.parametrize("collect", [False, True])
@pytest.mark.parametrize("failure", [None, "pending", "witness", "retirement"])
def test_original_preopen_checks_and_default_return(monkeypatch, collect, failure):
    from tldw_chatbook.Backup_Recovery import (
        journal,
        mobile_consolidation,
        storage_admission,
    )

    events = []
    rows = [object()]
    prepared = SimpleNamespace(source_fingerprint="source")

    def step(name):
        events.append(name)
        if name == failure:
            raise ValueError("injected_" + name)

    @contextmanager
    def locked(*, exclusive):
        assert exclusive is False
        step("lock")
        try:
            yield 42
        finally:
            step("unlock")

    def records(fd):
        assert fd == 42
        step("records")
        return rows

    def state(actual):
        assert actual is rows
        step("state")
        return prepared, {}

    captured = SimpleNamespace(
        check=lambda: step("capture"),
        claim=SimpleNamespace(custody=SimpleNamespace(close_failed=False)),
        preview=SimpleNamespace(
            source_dir=Path("/unit-only-source"),
            source_witnesses=("same",),
            source_fingerprint="source",
        ),
        journal=SimpleNamespace(_locked=locked, _records=records),
        session=object(),
    )
    monkeypatch.setattr(journal, "_mobile_state", state)
    monkeypatch.setattr(
        mobile_consolidation, "_path_witnesses", lambda paths: [(None, ("same",))]
    )
    monkeypatch.setattr(
        operation.control_records, "_mobile_pending", lambda *args: step("pending")
    )
    monkeypatch.setattr(
        storage_admission,
        "_mobile_check_witnesses",
        lambda *args, **kwargs: step("witness"),
    )
    monkeypatch.setattr(
        operation, "_require_readers_retired", lambda session: step("retirement")
    )
    collector = []
    consumed = []

    def call():
        kwargs = {"_pricing_inputs": collector} if collect else {}
        result = operation._native_preopen(captured, Event(), **kwargs)
        consumed.append(True)  # Pricing is reachable only after normal return.
        return result

    if failure:
        with pytest.raises(ValueError, match="^injected_" + failure + "$"):
            call()
        assert consumed == []
    else:
        assert call() is None
        assert events == [
            "capture",
            "lock",
            "records",
            "unlock",
            "state",
            "pending",
            "witness",
            "retirement",
        ]
        if collect:
            assert collector[0] is rows and collector[1] is prepared
        else:
            assert collector == []


def test_default_preopen_keeps_original_nested_call_shape(monkeypatch):
    calls = []

    def capture_check(captured, cancel):
        calls.append((captured, cancel))

    monkeypatch.setattr(operation, "_native_capture_check", capture_check)
    monkeypatch.setattr(operation, "_require_readers_retired", lambda session: None)
    captured, cancel = SimpleNamespace(session=object()), Event()
    assert operation._native_preopen(captured, cancel) is None
    assert calls == [(captured, cancel)]


class PendingBoundary(Exception):
    pass


def _sizing_inputs(monkeypatch):
    from tldw_chatbook.Backup_Recovery.journal import _Event

    row = _Event(
        operation_id="local-sizing",
        sequence=0,
        previous="",
        event="mobile_prepared",
        evidence={"path": '/unit/😀/\\"'},
    )
    expected = row.model_dump()
    targets = [
        SimpleNamespace(owner_id=owner, witnesses=[], sidecars=[])
        for owner in ("db.chachanotes.primary", "db.workspaces", "db.agent_runs")
    ]
    prepared = SimpleNamespace(targets=targets, bootstrap_root="/unit")
    budget = _TraceReadBudget()
    captured = SimpleNamespace(
        session=object(),
        journal=SimpleNamespace(operation_id="local-sizing"),
        claim=SimpleNamespace(custody=object()),
        cancel=Event(),
        check=lambda: pytest.fail("new R130 validation invocation"),
    )
    monkeypatch.setattr(pricing, "_mobile_coordinator_budget", lambda session: budget)

    def pending(*args):
        raise PendingBoundary

    monkeypatch.setattr(pricing, "observation", pending)
    monkeypatch.setattr(
        _Event, "model_dump", lambda *args, **kwargs: pytest.fail("discarded dump")
    )
    return captured, ([row], prepared), budget, expected


def test_future_ref_codec_uses_native_fields_without_discarded_dump(monkeypatch):
    captured, inputs, budget, expected = _sizing_inputs(monkeypatch)
    native = pricing._serialization
    observed = []

    def serialization(value, actual, *, encoded=False):
        assert actual is budget and value == expected and encoded
        before = actual._remaining
        native(value, actual, encoded=encoded)
        observed.append(before - actual._remaining)

    monkeypatch.setattr(pricing, "_serialization", serialization)
    with pytest.raises(PendingBoundary):
        pricing._preprice_terminal(
            captured, native_inputs=inputs, receipts=None, history_location=None
        )
    assert len(observed) == 1 and observed[0] > 0


def test_ref_view_capacity_refuses_before_field_reads_or_codec(monkeypatch):
    captured, inputs, budget, _ = _sizing_inputs(monkeypatch)

    class NoFields:
        event = "mobile_prepared"

        @property
        def version(self):
            pytest.fail("view fields read before reservation")

    inputs = ([NoFields()], inputs[1])
    # Initial512 + capture-list192 fit; the next native container(6)=256 does not.
    budget.reserve(budget._remaining - 959)
    monkeypatch.setattr(
        pricing, "_serialization", lambda *args, **kwargs: pytest.fail("codec entered")
    )
    with pytest.raises(_TraceReadLimit, match="mobile_trace_input_limit"):
        pricing._preprice_terminal(
            captured, native_inputs=inputs, receipts=None, history_location=None
        )
    assert budget._remaining == 255
