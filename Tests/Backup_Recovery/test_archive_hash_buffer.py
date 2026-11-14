"""Actual archive hash buffers retain native byte and allocation boundaries."""

import errno
import hashlib
import io
from contextlib import contextmanager
from threading import Event
from types import SimpleNamespace

import pytest

from tldw_chatbook.Backup_Recovery import archive_reader
from tldw_chatbook.Backup_Recovery.limits import _TraceReadLimit
from tldw_chatbook.Backup_Recovery.models import (
    _mobile_budget_pair,
    _mobile_operation_budgets,
    _TraceReadBudget,
)


def test_budgeted_seal_hash_reuses_actual_full_view_and_slices_only_tail(
    tmp_path, monkeypatch
):
    tail = b"native tail"
    content = b"a" * archive_reader._BUFFER + b"b" * archive_reader._BUFFER + tail
    path = tmp_path / "sealed.zip"
    path.write_bytes(content)
    expected = hashlib.sha256(content).hexdigest()
    native_sha256 = hashlib.sha256
    native_regular = archive_reader._regular
    views, counts, streams = [], [], []

    class Digest:
        def __init__(self):
            self.actual = native_sha256()

        def update(self, view):
            views.append(view)
            self.actual.update(view)

        def hexdigest(self):
            return self.actual.hexdigest()

    @contextmanager
    def regular(*args, **kwargs):
        with native_regular(*args, **kwargs) as stream:
            streams.append(stream)

            class Stream:
                fileno = stream.fileno
                read = stream.read

                def readinto(self, buffer):
                    count = stream.readinto(buffer)
                    counts.append(count)
                    return count

            yield Stream()

    monkeypatch.setattr(archive_reader, "hashlib", SimpleNamespace(sha256=Digest))
    monkeypatch.setattr(archive_reader, "_regular", regular)
    aggregate, budget = _mobile_operation_budgets()
    before = budget._remaining, aggregate._remaining
    assert archive_reader._hash(path, Event(), _read_budget=budget) == expected
    assert counts == [archive_reader._BUFFER, archive_reader._BUFFER, len(tail), 0]
    assert len(views) == 3
    assert all(type(view) is memoryview for view in views)
    assert views[0] is views[1]
    assert views[2] is not views[0]
    assert all(view.obj is views[0].obj for view in views)
    assert len(views[0]) == archive_reader._BUFFER
    assert len(views[2]) == len(tail)
    assert 0 < budget._remaining < before[0] == 16 * 1024**2
    assert before[0] - budget._remaining == before[1] - aggregate._remaining
    assert _mobile_budget_pair(aggregate, budget)
    assert type(streams[0]) is io.FileIO
    counts.clear()
    assert archive_reader._hash(path, Event()) == expected
    assert not counts
    assert type(streams[1]) is io.BufferedReader
    assert all(stream.closed for stream in streams)


@pytest.mark.parametrize("boundary", ["read", "partial-view"])
def test_budgeted_seal_hash_refuses_at_read_and_view_boundaries(
    tmp_path, monkeypatch, boundary
):
    path = tmp_path / "sealed.zip"
    path.write_bytes(b"native tail")
    budget = _TraceReadBudget()
    native_regular, native_view = archive_reader._regular, memoryview
    state = {"armed": False, "allocated": False, "read": False}

    class PartialView:
        def __getitem__(self, key):
            state["allocated"] = True
            raise AssertionError("unpriced native partial view allocated")

    def view(buffer):
        # Admit exactly the read fee, so a new tail view cannot be allocated.
        state["view"] = native_view(buffer)
        assert not state["armed"]
        state["armed"] = True
        budget.reserve(budget._remaining - (512 if boundary == "partial-view" else 0))
        return PartialView() if boundary == "partial-view" else state["view"]

    @contextmanager
    def regular(*args, **kwargs):
        with native_regular(*args, **kwargs) as stream:
            state.update(stream=stream, fd=stream.fileno())

            class Stream:
                fileno = stream.fileno

                def readinto(self, buffer):
                    if boundary == "read":
                        state["allocated"] = True
                        raise AssertionError("unpriced native read reached")
                    state["read"] = True
                    count = stream.readinto(buffer)
                    assert 0 < count < len(buffer)
                    return count

            yield Stream()

    monkeypatch.setattr(archive_reader, "_regular", regular)
    monkeypatch.setattr(archive_reader, "memoryview", view, raising=False)
    with pytest.raises(_TraceReadLimit, match="mobile_trace_input_limit"):
        archive_reader._hash(path, Event(), _read_budget=budget)
    assert state["armed"]
    assert not state["allocated"]
    # The original loop safely refuses its prepaid768 before this read;
    # the new512 read reaches native bytes and refuses the separate256 view.
    expected_remaining = 512 if boundary == "partial-view" and not state["read"] else 0
    assert budget._remaining == expected_remaining
    assert type(state["view"]) is native_view
    assert state["stream"].closed
    with pytest.raises(OSError) as closed:
        archive_reader.os.fstat(state["fd"])
    assert closed.value.errno == errno.EBADF


def test_budgeted_seal_hash_keeps_original_eof_fee_and_coupled_debit(
    tmp_path, monkeypatch
):
    path = tmp_path / "sealed.zip"
    path.write_bytes(b"")
    aggregate, budget = _mobile_operation_budgets()
    before = budget._remaining, aggregate._remaining
    native_regular = archive_reader._regular
    state, counts = {}, []

    @contextmanager
    def regular(*args, **kwargs):
        with native_regular(*args, **kwargs) as stream:
            state.update(stream=stream, fd=stream.fileno())

            class Stream:
                fileno = stream.fileno

                def readinto(self, buffer):
                    count = stream.readinto(buffer)
                    counts.append(count)
                    return count

            yield Stream()

    monkeypatch.setattr(archive_reader, "_regular", regular)
    assert (
        archive_reader._hash(path, Event(), _read_budget=budget)
        == hashlib.sha256(b"").hexdigest()
    )
    assert counts == [0]
    assert before[0] - budget._remaining == 1024 + 1 + 768
    assert before[0] - budget._remaining == before[1] - aggregate._remaining
    assert _mobile_budget_pair(aggregate, budget)
    assert state["stream"].closed
    with pytest.raises(OSError) as closed:
        archive_reader.os.fstat(state["fd"])
    assert closed.value.errno == errno.EBADF
