"""Fixed cutover accounting controls; no positive recovery authority is mocked."""

import inspect

import pytest

from tldw_chatbook.Chat import console_trace_read_budget as accounting


def test_cutover_issuer_is_fixed_and_cannot_change_existing_defaults():
    ordinary = accounting._TraceReadBudget()
    aggregate, native = accounting._mobile_operation_budgets()
    before = ordinary._remaining, aggregate._remaining, native._remaining
    budget = accounting._mobile_cutover_budget()
    assert type(budget) is accounting._TraceReadBudget
    assert budget._parent is None
    assert budget._remaining == 64 * 1024**2 - 512
    assert before == (16 * 1024**2, 4 * 1024**3 - 512, 16 * 1024**2)
    assert (ordinary._remaining, aggregate._remaining, native._remaining) == before
    assert accounting._mobile_budget_pair(aggregate, native)
    assert not inspect.signature(accounting._mobile_cutover_budget).parameters
    assert accounting._mobile_cutover_budget() is not budget


def test_cutover_exhaustion_refuses_without_partial_debit():
    budget = accounting._mobile_cutover_budget()
    budget.reserve(budget._remaining - 3)
    with pytest.raises(accounting._TraceReadLimit, match="mobile_trace_input_limit"):
        budget.reserve(4)
    assert budget._remaining == 3
    budget.reserve(3)
    with pytest.raises(accounting._TraceReadLimit):
        budget.reserve(1)
    assert budget._remaining == 0


@pytest.mark.parametrize("argument", [16, 128 * 1024**2])
def test_cutover_issuer_accepts_no_caller_capacity(argument):
    with pytest.raises(TypeError):
        accounting._mobile_cutover_budget(argument)


def test_depleted_cutover_counter_refuses_before_source_discovery(
    tmp_path, monkeypatch
):
    from tldw_chatbook.Backup_Recovery import mobile_consolidation as mobile
    from tldw_chatbook.Backup_Recovery.mobile_consolidation_recovery import (
        _require_mobile_cutover,
    )

    budget = accounting._mobile_cutover_budget()
    budget.reserve(budget._remaining)
    issued = []

    def exhausted():
        issued.append(budget)
        return budget

    monkeypatch.setattr(accounting, "_mobile_cutover_budget", exhausted)
    monkeypatch.setattr(
        mobile,
        "_source_claim_locations",
        lambda *args: pytest.fail("source discovery preceded reservation"),
    )
    stores = mobile.CanonicalConsoleStores(
        *(tmp_path / name for name in ("chat.db", "workspaces.db", "runs.db"))
    )
    before_names = sorted(path.name for path in tmp_path.iterdir())
    with pytest.raises(accounting._TraceReadLimit, match="mobile_trace_input_limit"):
        _require_mobile_cutover(tmp_path / "source", stores)
    assert issued == [budget]
    assert budget._remaining == 0
    assert sorted(path.name for path in tmp_path.iterdir()) == before_names


def assert_original_observer_accounting(tmp_path, monkeypatch, preview, stores):
    """Exercise real copies on a genuine public target, then inject scarcity only."""
    import errno
    import hashlib
    import json
    from pathlib import Path

    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.Backup_Recovery.mobile_consolidation import (
        require_mobile_cutover,
    )

    def target_facts():
        result = {}
        for main in stores.paths:
            for suffix in ("", "-wal", "-shm"):
                path = Path(str(main) + suffix)
                if not path.exists():
                    result[str(path)] = None
                    continue
                info = path.stat()
                result[str(path)] = (
                    info.st_dev,
                    info.st_ino,
                    info.st_mode,
                    info.st_size,
                    info.st_mtime_ns,
                    info.st_ctime_ns,
                    hashlib.sha256(path.read_bytes()).hexdigest(),
                )
        return result

    native_issue = accounting._mobile_cutover_budget
    native_init = storage._PreviewScope.__init__
    native_state = storage._PreviewScope._source_state
    native_buffer = storage._PreviewScope._buffer
    native_readv = storage.os.readv
    outcomes = []

    def check_phase(phase):
        issued, scopes, descriptors, state_calls, read_calls = [], [], set(), [], []
        injected = []
        before = target_facts()

        def issue():
            value = native_issue()
            issued.append(value)
            return value

        def init(scope, *args, **kwargs):
            native_init(scope, *args, **kwargs)
            scopes.append(scope)

        def source_state(scope, source):
            state_calls.append(str(source))
            if phase == "source-scarcity" and not injected:
                issued[0].reserve(issued[0]._remaining - 1)
                injected.append({"point": "before_native_source_state", "remaining": 1})
            return native_state(scope, source)

        def buffer(scope):
            value = native_buffer(scope)
            descriptors.update(scope.custody.fds)
            if phase == "copy-scarcity" and not injected:
                issued[0].reserve(issued[0]._remaining - 767)
                injected.append({"point": "before_native_copy_read", "remaining": 767})
            return value

        def readv(fd, buffers):
            scope = getattr(storage._local, "preview_scope", None)
            if scope is not None and any(scope is row for row in scopes):
                read_calls.append(fd)
                descriptors.add(fd)
            return native_readv(fd, buffers)

        failure = None
        try:
            with monkeypatch.context() as patch:
                patch.setattr(accounting, "_mobile_cutover_budget", issue)
                patch.setattr(storage._PreviewScope, "__init__", init)
                patch.setattr(storage._PreviewScope, "_source_state", source_state)
                patch.setattr(storage._PreviewScope, "_buffer", buffer)
                patch.setattr(storage.os, "readv", readv)
                if phase == "positive":
                    assert require_mobile_cutover(preview.source_dir, stores) is None
                else:
                    with pytest.raises(
                        accounting._TraceReadLimit, match="mobile_trace_input_limit"
                    ):
                        require_mobile_cutover(preview.source_dir, stores)
            assert len(issued) == 1
            assert len(scopes) == (3 if phase == "positive" else 1)
            assert all(scope.read_budget is issued[0] for scope in scopes)
            assert all(
                not scope.active and not scope.resources and scope.directory is None
                for scope in scopes
            )
            assert all(
                not scope.custody.fds and not scope.custody.close_failed
                for scope in scopes
            )
            for fd in descriptors:
                with pytest.raises(OSError) as closed:
                    storage.os.fstat(fd)
                assert closed.value.errno == errno.EBADF
            if phase == "positive":
                assert len(state_calls) == 6 and read_calls
                assert all(
                    scope.copied_bytes > 0 and len(scope.snapshots) == 1
                    for scope in scopes
                )
            else:
                assert len(injected) == 1 and not read_calls
                assert not scopes[0].snapshots and scopes[0].copied_bytes == 0
                assert issued[0]._remaining == (
                    1 if phase == "source-scarcity" else 767
                )
                assert len(state_calls) == 1
            assert target_facts() == before
        except BaseException as error:
            failure = {"type": type(error).__name__, "code": str(error)}
            raise
        finally:
            outcomes.append(
                {
                    "phase": phase,
                    "failure": failure,
                    "injected": injected,
                    "coordinators": [
                        {"identity": id(row), "remaining": row._remaining}
                        for row in issued
                    ],
                    "scopes": [
                        {
                            "same_coordinator": bool(issued)
                            and row.read_budget is issued[0],
                            "active": row.active,
                            "resources": len(row.resources),
                            "directory": str(row.directory) if row.directory else None,
                            "copied_bytes": row.copied_bytes,
                            "custody_fds": list(row.custody.fds)
                            if row.custody
                            else None,
                            "close_failed": row.custody.close_failed
                            if row.custody
                            else None,
                        }
                        for row in scopes
                    ],
                    "source_state_calls": state_calls,
                    "native_copy_read_calls": len(read_calls),
                    "observed_descriptors": sorted(descriptors),
                    "target_before": before,
                    "target_after": target_facts(),
                }
            )
            (tmp_path / "original-cutover-accounting.json").write_text(
                json.dumps(outcomes, indent=2)
            )

    for phase in ("positive", "source-scarcity", "copy-scarcity"):
        check_phase(phase)
