"""Real terminal codec; prefix doubles prove pricing mechanics, not authority."""

import json
import sys
from threading import Event
from types import SimpleNamespace

import pytest

from tldw_chatbook.Backup_Recovery import mobile_terminal_pricing as pricing
from tldw_chatbook.Backup_Recovery.journal import _Event, _MobileCommitted
from tldw_chatbook.Chat.console_trace_read_budget import (
    _TraceReadBudget,
    _TraceReadLimit,
)


def _local_inputs(monkeypatch, suffix, imported):
    budget = _TraceReadBudget()
    root = "/tmp/terminal-local/" + suffix
    targets = [
        SimpleNamespace(
            owner_id=owner,
            witnesses=[
                SimpleNamespace(path=root + "/parent"),
                SimpleNamespace(path=root + "/" + family),
            ],
            sidecars=[
                SimpleNamespace(path=root + "/" + family + tail)
                for tail in ("", "-wal", "-shm")
            ],
        )
        for owner, family in (
            ("db.chachanotes.primary", "chat"),
            ("db.workspaces", "workspaces"),
            ("db.agent_runs", "runs"),
        )
    ]
    prepared = SimpleNamespace(bootstrap_root=root, targets=targets)
    captured = SimpleNamespace(
        session=object(),
        journal=SimpleNamespace(operation_id="local-terminal"),
        preview=SimpleNamespace(
            source_fingerprint="a" * 64, destination=SimpleNamespace(identity="b" * 64)
        ),
        claim=SimpleNamespace(custody=object()),
        cancel=Event(),
    )
    row = _Event(
        operation_id="local-terminal",
        sequence=0,
        previous="0" * 64,
        event="mobile_capture_verified",
        evidence={"purpose": "canonical"},
    )
    monkeypatch.setattr(pricing, "_mobile_coordinator_budget", lambda session: budget)
    monkeypatch.setattr(
        pricing,
        "observation",
        lambda path, *args: {
            "path": str(path),
            "device": 1,
            "inode": 2,
            "kind": "file",
            "size": 1,
            "sha256": "c" * 64,
        },
    )
    receipts = (
        {
            family: [
                (
                    family,
                    json.dumps(
                        {
                            "entry_count": 1,
                            "entries_sha256": "d" * 64,
                            "command_history_count": None,
                        }
                    ),
                )
            ]
            for family in ("chat", "workspaces", "runs")
        }
        if imported
        else None
    )
    return (
        captured,
        {
            "native_inputs": ([row], prepared),
            "receipts": receipts,
            "history_location": None,
        },
        budget,
    )


def _observe(monkeypatch, budget, *, fail=None):
    seen = {
        "trees": [],
        "mobile_dumps": 0,
        "event_dumps": [],
        "roundtrips": [],
        "encoded": [],
    }
    native_value = _TraceReadBudget.value
    native_container = _TraceReadBudget.container
    mobile_dump = _MobileCommitted.model_dump
    event_dump = _Event.model_dump
    decode = _Event.model_validate_json
    encode = pricing._encoded

    def value(actual, data, *, text_copies=1):
        caller = sys._getframe(1)
        stage = None
        if (
            actual is budget
            and caller.f_code.co_name == "_preprice_terminal"
            and isinstance(data, dict)
        ):
            if data.get("kind") in ("imported", "no_data") and "targets" in data:
                stage = "terminal"
            elif data.get("event") == "mobile_committed":
                stage = "event"
        if stage:
            occurrence = sum(row[0] == stage for row in seen["trees"]) + 1
            if fail == (stage, occurrence):
                actual.reserve(actual._remaining)
            before = actual._remaining
            result = native_value(actual, data, text_copies=text_copies)
            seen["trees"].append((stage, text_copies, before - actual._remaining, data))
            return result
        return native_value(actual, data, text_copies=text_copies)

    def container(actual, count=0):
        caller = sys._getframe(1)
        if (
            actual is budget
            and fail == "view"
            and count == 6
            and caller.f_code.co_name == "_preprice_terminal"
            and "event" in caller.f_locals
            and "event_data" not in caller.f_locals
        ):
            actual.reserve(actual._remaining)
        return native_container(actual, count)

    def dumped_mobile(model, *args, **kwargs):
        seen["mobile_dumps"] += 1
        assert [row[1] for row in seen["trees"] if row[0] == "terminal"] == [8, 1, 1]
        return mobile_dump(model, *args, **kwargs)

    def dumped_event(model, *args, **kwargs):
        if model.event == "mobile_committed":
            assert [row[1] for row in seen["trees"] if row[0] == "event"] == [1, 1]
            result = event_dump(model, *args, **kwargs)
            seen["event_dumps"].append(result)
            return result
        return event_dump(model, *args, **kwargs)

    def encoded(model):
        result = encode(model)
        seen["encoded"].append(result)
        return result

    def decoded(cls, raw, *args, **kwargs):
        result = decode(raw, *args, **kwargs)
        seen["roundtrips"].append((raw, event_dump(result)))
        return result

    monkeypatch.setattr(_TraceReadBudget, "value", value)
    monkeypatch.setattr(_TraceReadBudget, "container", container)
    monkeypatch.setattr(_MobileCommitted, "model_dump", dumped_mobile)
    monkeypatch.setattr(_Event, "model_dump", dumped_event)
    monkeypatch.setattr(_Event, "model_validate_json", classmethod(decoded))
    monkeypatch.setattr(pricing, "_encoded", encoded)
    return seen


@pytest.mark.parametrize("imported", [False, True], ids=["no-data", "imported"])
@pytest.mark.parametrize("suffix", ["ascii", '😀/escaped-"-\\'])
def test_each_actual_dump_has_independent_full_tree_reservation(
    monkeypatch, imported, suffix
):
    captured, kwargs, budget = _local_inputs(monkeypatch, suffix, imported)
    seen = _observe(monkeypatch, budget)
    assert pricing._preprice_terminal(captured, **kwargs) is None
    assert seen["mobile_dumps"] == 1
    first, second = seen["event_dumps"]
    assert first == second and first is not second
    assert first["evidence"] is not second["evidence"]
    raw, decoded = seen["roundtrips"][0]
    assert (
        raw
        == seen["encoded"][0]
        == json.dumps(first, sort_keys=True, separators=(",", ":")).encode()
    )
    assert decoded == first
    terminal = [row for row in seen["trees"] if row[0] == "terminal"]
    assert [row[1] for row in terminal] == [8, 1, 1, 4]
    assert terminal[1][2] == terminal[2][2] > 0
    event = [row for row in seen["trees"] if row[0] == "event"]
    assert event[0][2] == event[1][2] > terminal[1][2]
    # Two complete reservations include two container/scalar sets, not only text.
    check = _TraceReadBudget()
    before = check._remaining
    check.value(terminal[1][3], text_copies=2)
    assert terminal[1][2] + terminal[2][2] > before - check._remaining
    assert budget._parent is None and 0 < budget._remaining < 16 * 1024**2


@pytest.mark.parametrize(
    "stage", [("terminal", 2), ("terminal", 3), "view", ("event", 1), ("event", 2)]
)
def test_capacity_refuses_before_unpaid_tree_or_dump(monkeypatch, stage):
    captured, kwargs, budget = _local_inputs(monkeypatch, "capacity", True)
    seen = _observe(monkeypatch, budget, fail=stage)
    with pytest.raises(_TraceReadLimit, match="mobile_trace_input_limit"):
        pricing._preprice_terminal(captured, **kwargs)
    assert seen["mobile_dumps"] == (
        0 if stage in (("terminal", 2), ("terminal", 3)) else 1
    )
    assert seen["event_dumps"] == seen["encoded"] == seen["roundtrips"] == []
    assert budget._remaining == 0


def test_native_path_validation_and_record_limit_remain(monkeypatch):
    captured, kwargs, budget = _local_inputs(monkeypatch, "../bad", False)
    with pytest.raises(ValueError, match="artifact_path_invalid"):
        pricing._preprice_terminal(captured, **kwargs)
    captured, kwargs, budget = _local_inputs(monkeypatch, "record-limit", False)
    seen = _observe(monkeypatch, budget)
    monkeypatch.setattr(pricing, "RECOVERY_RECORD_BYTES", 1)
    with pytest.raises(ValueError, match="journal_record_limit"):
        pricing._preprice_terminal(captured, **kwargs)
    assert len(seen["event_dumps"]) == 2 and len(seen["encoded"]) == 1
    assert seen["roundtrips"] == []
