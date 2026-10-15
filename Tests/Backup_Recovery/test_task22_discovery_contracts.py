"""Unexecuted Task22 discovery regression controls on actual installed owners."""

import errno
import json
import stat
import sys
import threading

import pytest

from Tests.Backup_Recovery import test_raw_native_retirement as original
from tldw_chatbook.Backup_Recovery import bootstrap
from tldw_chatbook.Backup_Recovery import config_participants as config_sources
from tldw_chatbook.Backup_Recovery import mcp_source_participants as mcp_sources
from tldw_chatbook.Backup_Recovery import raw_participants as raw
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.MCP.execution_log import MCPExecutionLog, build_record
from tldw_chatbook.Utils import private_paths as private

configured_source = original.configured_source
local_root = original.local_root


class _NativeLedger:
    """Observe real returns; callbacks never assert or interrupt native cleanup."""

    def __init__(self):
        self.state = None
        self.operation = None
        self.thread = threading.get_ident()
        self.rows = []
        self.live = {}
        self.closing = {}
        self.native_closing = {}
        self.errors = []
        self.components = 0
        self.after_component = None
        self.previous = sys.getprofile()
        self.allocate = private._allocate_native_descriptor.__code__
        self.component = private._open_directory_component.__code__
        self.raw_close = raw._close_descriptor.__code__
        self.native_close = private._native_close.__code__

    def identity(self, fd):
        value = raw.os.fstat(fd)
        return value.st_dev, value.st_ino, stat.S_IFMT(value.st_mode)

    def opened(self, fd, kind):
        if type(fd) is not int or fd in self.live:
            raise AssertionError("invalid or still-live descriptor incarnation")
        row = dict(fd=fd, identity=self.identity(fd), kind=kind,
                   native_return=False, close_attempts=0, closed=False)
        self.rows.append(row)
        self.live[fd] = row

    def attach(self, operation):
        self.operation = operation
        self.state = raw._states[operation]
        for fd in self.state.pins.values():
            self.opened(fd, "retained-pin")

    def observe(self, frame, event, result):
        if self.previous is not None:
            self.previous(frame, event, result)
        if self.state is None or threading.get_ident() != self.thread:
            return
        try:
            active = getattr(raw._local, "operation", None) is self.operation
            if (frame.f_code in (self.raw_close, self.native_close)
                    and result is raw.os.close and event.startswith("c_")):
                owned = (frame.f_locals.get("state") is self.state
                         if frame.f_code is self.raw_close else active)
                if event == "c_call" and owned:
                    row = self.live.get(frame.f_locals["fd"])
                    if row is not None:
                        if self.closing.get(id(frame)) is not row:
                            raise AssertionError("native close has no exact creator frame")
                        if self.identity(row["fd"]) != row["identity"]:
                            raise AssertionError("native close changed descriptor incarnation")
                        row["close_attempts"] += 1
                        self.native_closing[id(frame)] = row
                elif id(frame) in self.native_closing:
                    row = self.native_closing.pop(id(frame))
                    if event != "c_return":
                        raise AssertionError("actual native close did not return normally")
                    try:
                        raw.os.fstat(row["fd"])
                    except OSError as error:
                        if error.errno != errno.EBADF:
                            raise
                    else:
                        raise AssertionError("descriptor live at native C-return")
                    row["native_return"] = True
                return
            if frame.f_code is self.allocate and event == "return" and active:
                if type(result) is int:
                    self.opened(result, "actual-allocation")
            elif frame.f_code in (self.raw_close, self.native_close):
                owned = (frame.f_locals.get("state") is self.state
                         if frame.f_code is self.raw_close else active)
                if owned and event == "call":
                    fd = frame.f_locals["fd"]
                    row = self.live.get(fd)
                    if row is not None:
                        if self.identity(fd) != row["identity"]:
                            raise AssertionError("close changed descriptor incarnation")
                        self.closing[id(frame)] = row
                elif event == "return" and id(frame) in self.closing:
                    row = self.closing.pop(id(frame))
                    # The actual raw close may retire an outer native-close row.
                    if not row["closed"]:
                        if not row["native_return"]:
                            raise AssertionError("missing exact creator native C-return")
                        try:
                            raw.os.fstat(row["fd"])
                        except OSError as error:
                            if error.errno != errno.EBADF:
                                raise
                        else:
                            raise AssertionError("descriptor live at close return")
                        if row["fd"] in self.state.descriptors:
                            raise AssertionError("closed descriptor remains registered")
                        row["closed"] = True
                        del self.live[row["fd"]]
            elif (frame.f_code is self.component and event == "return"
                  and active and type(result) is int):
                self.components += 1
                if self.after_component is not None and self.components == 1:
                    self.after_component()
        except BaseException as error:
            self.errors.append((frame.f_code.co_qualname, event, type(error).__name__))


def _owner(configured_source, kind):
    if kind == "config":
        return configured_source, "config", configured_source._get_effective_config_path()
    source = MCPExecutionLog(configured_source.get_user_data_dir() / "mcp_execution_log.jsonl")
    source.append(build_record(server_key="task22", tool_name="probe",
                               initiator="test", ok=True, duration_ms=0))
    assert source.read_recent()[0]["tool_name"] == "probe"
    assert mcp_sources.binding(source)[0] == "mcp.history"
    return source, mcp_sources.ROUTE, source.path


def _attempt(source, route, selected, monkeypatch, *, ordinary, kind, change):
    """One real scope and walk; restore fault injections before scope retirement."""
    expected = selected.read_bytes()
    ledger = _NativeLedger()
    outcome = dict(error=None, returned=False, discovery=[], custody=0,
                   changed=False, allocations_at_change=None)
    module, slot = ((mcp_sources, "history_operation") if kind == "mcp"
                    else (config_sources, "binding"))
    callback = vars(module)[slot]
    runtime_code = raw._runtime_operation.__code__
    phase = "setup"

    def delegated(value):
        discovery = sys._getframe(1).f_code is runtime_code
        if phase == "walk":
            if discovery:
                outcome["discovery"].append("runtime")
            else:
                outcome["custody"] += 1
        if discovery and change == "decline":
            return False
        return callback(value)

    sys.setprofile(ledger.observe)
    try:
        with raw._scope(source, route, writing=True) as operation:
            ledger.attach(operation)
            state = ledger.state
            leases = tuple(state.leases)
            assert state.active and state.participant is not None and state.pins
            assert state.source is source and state.pinned
            assert raw._participant_identity(state.participant).owner == (
                "mcp.history" if kind == "mcp" else "config")
            assert all(lease in storage._live_leases for lease in leases)
            baseline = set(state.descriptors)
            parent = leaf_fd = None
            with monkeypatch.context() as patch:
                if ordinary:
                    # Original dynamic public route: a documented optional-record
                    # absence, not a product alias or fabricated source owner.
                    patch.delattr(private, "_PARENT_WALK_SOURCE")
                if change in ("decline", "delegate"):
                    patch.setattr(module, slot, delegated)
                else:
                    def change_after_first_component():
                        outcome["changed"] = True
                        outcome["allocations_at_change"] = len(ledger.rows)
                        if change == "binding":
                            patch.setattr(module, slot, delegated)
                        elif change == "code":
                            patch.setattr(callback, "__code__", callback.__code__.replace(
                                co_consts=callback.__code__.co_consts + ("task22-held-discovery",)))
                        else:
                            assert change == "defaults" and callback.__defaults__ is None
                            patch.setattr(callback, "__defaults__", (None,))
                    ledger.after_component = change_after_first_component
                phase = "walk"
                try:
                    parent, leaf = private._open_verified_parent(selected, missing_leaf_allowed=False)
                    outcome["returned"] = True
                    assert leaf == selected.name
                    assert stat.S_ISDIR(raw.os.fstat(parent).st_mode)
                    phase = "leaf"
                    leaf_fd = private._native_open(leaf, private._PRIVATE_FILE_OPEN_FLAGS, dir_fd=parent)
                    assert raw.os.read(leaf_fd, len(expected) + 1) == expected
                except BaseException as error:
                    outcome["error"] = error
                finally:
                    phase = "retirement"
                    if leaf_fd is not None:
                        private._native_close(leaf_fd)
                    if parent is not None:
                        private._native_close(parent)
            assert vars(module)[slot] is callback
            assert state.descriptors == baseline
            assert all(lease in storage._live_leases for lease in leases)
    finally:
        sys.setprofile(ledger.previous)
    assert not ledger.errors, ledger.errors
    assert not ledger.live and not ledger.closing and not ledger.native_closing
    assert ledger.rows and all(row["closed"] and row["native_return"]
                               and row["close_attempts"] == 1 for row in ledger.rows)
    assert any(row["kind"] == "retained-pin" for row in ledger.rows)
    assert not state.active and not state.uncertain
    assert not state.pins and not state.files and not state.descriptors
    assert operation not in raw._states and operation not in storage._raw_operations
    assert all(lease not in storage._live_leases for lease in leases)
    assert selected.read_bytes() == expected
    outcome["physical_closes"] = len(ledger.rows)
    outcome["retained_pin_closes"] = sum(row["kind"] == "retained-pin" for row in ledger.rows)
    outcome["allocations"] = sum(row["kind"] == "actual-allocation" for row in ledger.rows)
    outcome["all_rows"] = len(ledger.rows)
    outcome["components"] = ledger.components
    return outcome


def _receipt(request, label, outcome):
    value = {key: item for key, item in outcome.items() if key != "error"}
    value["error"] = None if outcome["error"] is None else type(outcome["error"]).__name__
    request.node.user_properties.append((label, json.dumps(value, sort_keys=True)))


@pytest.mark.parametrize("kind,change", [("mcp", "decline"), ("mcp", "delegate"), ("config", "delegate")])
def test_changed_discovery_preserves_actual_ordinary_route(configured_source, monkeypatch, request, kind, change):
    source, route, selected = _owner(configured_source, kind)
    ordinary = _attempt(source, route, selected, monkeypatch, ordinary=True, kind=kind, change=change)
    _receipt(request, "ordinary", ordinary)
    assert ordinary["error"] is None and ordinary["returned"]
    assert ordinary["allocations"] > 1 and ordinary["components"] > 1
    assert len(ordinary["discovery"]) > 1
    if kind == "config":
        assert ordinary["custody"] > 0
    candidate = _attempt(source, route, selected, monkeypatch, ordinary=False, kind=kind, change=change)
    _receipt(request, "candidate", candidate)
    assert candidate["error"] is None, candidate["error"]
    assert candidate["returned"]
    assert candidate["discovery"] == ordinary["discovery"]
    assert candidate["allocations"] == ordinary["allocations"]
    if kind == "config":
        assert candidate["custody"] > 0


@pytest.mark.parametrize("kind", ["mcp", "config"])
@pytest.mark.parametrize("change", ["binding", "code", "defaults"])
def test_held_discovery_change_refuses_before_next_native_allocation(configured_source, monkeypatch, request, kind, change):
    source, route, selected = _owner(configured_source, kind)
    outcome = _attempt(source, route, selected, monkeypatch, ordinary=False, kind=kind, change=change)
    _receipt(request, "held", outcome)
    assert outcome["changed"] and outcome["allocations_at_change"] is not None
    assert outcome["components"] == 1
    assert outcome["all_rows"] == outcome["allocations_at_change"]
    assert not outcome["returned"]
    assert type(outcome["error"]) is bootstrap.RecoveryRequired
    assert str(outcome["error"]) == "raw_operation_provenance_invalid"
