"""Native operation components never stand in for populated import readiness."""

import sqlite3
from contextlib import contextmanager
from copy import copy
from threading import Event

import pytest

from Tests.Backup_Recovery.test_mobile_captured_sqlite import (
    _fingerprints,
    _observe_physical_retirement,
)
from Tests.Backup_Recovery.test_mobile_consolidation_lifecycle import _capture_fixture


@contextmanager
def captured_operation(tmp_path, monkeypatch):
    from tldw_chatbook.Backup_Recovery.mobile_consolidation_apply import (
        _captured_mobile,
    )

    with _capture_fixture(tmp_path, monkeypatch) as values:
        session, journal, _, _, preview, inventory, _ = values
        output = tmp_path / "archives"
        output.mkdir(mode=0o700)
        with _captured_mobile(
            session,
            journal,
            preview,
            inventory.scope_digest,
            destination=output,
            password=b"test-private-mobile-rollback",
            cancel=Event(),
        ) as captured:
            yield captured


def test_component_retires_actual_readers_before_native_owner_and_refuses_copy(
    tmp_path, monkeypatch
):
    with captured_operation(tmp_path, monkeypatch) as captured:
        with (
            _observe_physical_retirement(monkeypatch),
            captured.sqlite_reads(
                password=b"test-private-mobile-rollback", cancel=Event()
            ) as reads,
        ):
            reads.page("legacy", "chat", "messages")
            with (
                pytest.raises((ValueError, RuntimeError), match="preparation|retired"),
                captured.owner("db.agent_runs"),
            ):
                pytest.fail("native owner overlapped captured readers")
        assert reads.resources == []
        with (
            pytest.raises(ValueError, match="native|cohort"),
            copy(captured).owner("db.agent_runs"),
        ):
            pytest.fail("copied captured lifecycle admitted native owner")
        # Existing native component access does not run an import writer.
        with (
            captured.owner("db.agent_runs") as owner,
            owner.connection() as connection,
        ):
            assert (
                connection.execute(
                    "SELECT count(*) FROM mobile_consolidation_batches"
                ).fetchone()[0]
                == 0
            )
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            connection.execute("SELECT 1")
        captured.claim.check()


def test_production_operation_derives_all_inputs_and_refuses_before_first_effect(
    tmp_path, monkeypatch
):
    import json

    from tldw_chatbook.Backup_Recovery import mobile_runs_preflight, sqlite_validation

    expired = []
    native_expired = sqlite_validation._Restrictions.expired

    def observe_expired(restrictions):
        result = native_expired(restrictions)
        if result and len(expired) < 8:
            expired.append(
                {
                    "steps": restrictions.steps,
                    "step_budget": sqlite_validation._STEP_BUDGET,
                    "cancelled": restrictions.cancel is not None
                    and restrictions.cancel.is_set(),
                    "deadline": restrictions.deadline,
                    "observed_after_native_call": sqlite_validation.monotonic(),
                }
            )
        return result

    monkeypatch.setattr(sqlite_validation._Restrictions, "expired", observe_expired)
    observed = []
    original_plan = mobile_runs_preflight._plan_runs

    def observe(reads, *, _read_budget):
        before = _read_budget._remaining
        plan = original_plan(reads, _read_budget=_read_budget)
        assert _read_budget._remaining < before < 16 * 1024 * 1024
        observed.append(plan)
        return plan

    monkeypatch.setattr(mobile_runs_preflight, "_plan_runs", observe)
    with captured_operation(tmp_path, monkeypatch) as captured:
        before = _fingerprints(tmp_path / "profile")
        with captured.journal._locked(exclusive=False) as fd:
            records = tuple(captured.journal._records(fd))
        try:
            with pytest.raises(
                ValueError, match="mobile_native_operation_blocked"
            ) as error:
                captured.consolidate(
                    password=b"test-private-mobile-rollback", cancel=Event()
                )
        finally:
            (tmp_path / "native-expiry.json").write_text(json.dumps(expired))
        blockers = error.value.blockers
        assert len(observed) == 1
        plan = observed[0]
        assert plan.source_rows
        assert all(blocker in blockers for blocker in plan.blockers)
        assert {
            (row.table, row.original, row.dependency_domain, row.dependency_id)
            for row in blockers
            if row.reason == "native_consumer_required"
        } == {
            (d.table, d.original, d.owner, d.requirement) for d in plan.pending_duties
        }
        assert {
            row.original
            for row in blockers
            if row.reason == "native_runlog_receipt_membership_required"
        } == {(h.logical_id,) for h in plan.history}
        for diagnostic in (plan, plan.integrated, plan.integrated.sync.trace.content):
            with pytest.raises(ValueError):
                diagnostic.require_complete()
        assert {
            row.dependency_id
            for row in blockers
            if row.reason == "native_owner_writer_required"
        } == {"db.chachanotes.primary", "db.workspaces", "db.agent_runs"}
        assert _fingerprints(tmp_path / "profile") == before
        with captured.journal._locked(exclusive=False) as fd:
            assert tuple(captured.journal._records(fd)) == records
        assert not tuple((tmp_path / "archives").glob(".mobile-read-*"))
        captured.claim.check()
        from tldw_chatbook.Backup_Recovery.journal import _mobile_state
        from tldw_chatbook.Backup_Recovery.mobile_consolidation_apply import (
            _captured_mobile,
        )
        from tldw_chatbook.Chat.console_trace_read_budget import _TraceReadBudget

        prepared, _ = _mobile_state(records)

        def renewed_budget(_self):
            pytest.fail("same native operation constructed a second allowance")

        with monkeypatch.context() as retry:
            retry.setattr(_TraceReadBudget, "__init__", renewed_budget)
            with pytest.raises(ValueError, match="attempted"):
                captured.consolidate(
                    password=b"test-private-mobile-rollback", cancel=Event()
                )
            with pytest.raises(ValueError, match="cohort"):
                copy(captured).consolidate(
                    password=b"test-private-mobile-rollback", cancel=Event()
                )
            with (
                pytest.raises(ValueError, match="attempted"),
                _captured_mobile(
                    captured.session,
                    captured.journal,
                    captured.preview,
                    prepared.inventory_digest,
                    destination=tmp_path / "archives",
                    password=b"test-private-mobile-rollback",
                    cancel=Event(),
                ),
            ):
                pytest.fail("same live operation reissued capture lifecycle")


def test_native_owner_revalidates_capture_before_subsequent_transaction(
    tmp_path, monkeypatch
):
    with (
        captured_operation(tmp_path, monkeypatch) as captured,
        captured.owner("db.agent_runs") as owner,
    ):
        original = captured.preview
        captured.preview = copy(original)
        try:
            with (
                pytest.raises(ValueError, match="cohort"),
                owner.transaction(),
            ):
                pytest.fail("replaced capture reached a native transaction")
        finally:
            captured.preview = original


def test_native_preopen_refuses_replaced_cohorts_selectors_capture_and_cancel(
    tmp_path, monkeypatch
):
    with captured_operation(tmp_path, monkeypatch) as captured:
        before = _fingerprints(tmp_path / "profile")
        with captured.journal._locked(exclusive=False) as fd:
            records = tuple(captured.journal._records(fd))
        for field in (
            "session",
            "journal",
            "claim",
            "preview",
            "source",
            "canonical",
            "cancel",
        ):
            original = getattr(captured, field)
            setattr(captured, field, copy(original))
            try:
                with (
                    pytest.raises(ValueError, match="cohort"),
                    captured.owner("db.agent_runs"),
                ):
                    pytest.fail("replaced native cohort admitted owner")
            finally:
                setattr(captured, field, original)
        cancel = Event()
        cancel.set()
        with pytest.raises(InterruptedError, match="cancelled"):
            captured.consolidate(
                password=b"test-private-mobile-rollback", cancel=cancel
            )
        selector = captured.preview.source_selector
        original = selector.read_bytes()
        try:
            selector.write_bytes(original + b"\n# changed native selector\n")
            with (
                pytest.raises(RuntimeError, match="selector_review_changed"),
                captured.owner("db.agent_runs"),
            ):
                pytest.fail("changed selector admitted owner")
        finally:
            selector.write_bytes(original)
        source = captured.preview.source_dir
        moved = source.with_name(source.name + "-moved")
        source.rename(moved)
        try:
            with (
                pytest.raises((ValueError, RuntimeError), match="source|changed"),
                captured.owner("db.agent_runs"),
            ):
                pytest.fail("changed source location admitted owner")
        finally:
            moved.rename(source)
        target = captured.preview.destination.run_path
        moved = target.with_name(target.name + "-original")
        target.rename(moved)
        try:
            target.write_bytes(moved.read_bytes())
            with (
                pytest.raises(RuntimeError, match="object_changed"),
                captured.owner("db.agent_runs"),
            ):
                pytest.fail("same-byte replaced target admitted owner")
        finally:
            target.unlink()
            moved.rename(target)
        assert _fingerprints(tmp_path / "profile") == before
        with captured.source.path.open("ab") as stream:
            stream.write(b"changed captured source")
        with (
            pytest.raises(ValueError, match="capture_changed"),
            captured.owner("db.agent_runs"),
        ):
            pytest.fail("changed authenticated capture admitted owner")
        with captured.journal._locked(exclusive=False) as fd:
            assert tuple(captured.journal._records(fd)) == records
    with (
        pytest.raises(ValueError, match="retired"),
        captured.owner("db.agent_runs"),
    ):
        pytest.fail("stale lifecycle admitted owner")


def test_uncertain_reader_retirement_prevents_native_owner_until_process_exit(tmp_path):
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path

    proof_path = tmp_path / "proof.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "--tb=short",
            "--show-capture=no",
            "--basetemp",
            str(tmp_path / "child"),
            "Tests/Backup_Recovery/test_mobile_native_operation.py::test_uncertain_retirement_child",
        ],
        env=dict(os.environ, TLDW_NATIVE_RETIREMENT_PROOF=str(proof_path)),
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=240,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    proof = json.loads(proof_path.read_text())
    assert proof["owner_refusals"] == 2
    assert proof["contender_returncode"] == 23
    from tldw_chatbook.Backup_Recovery.control_records import admission_authority

    with admission_authority(Path(proof["control_root"])).maintenance(
        ("bootstrap.unbound", "mobile.profile"), 2
    ):
        pass


def test_uncertain_retirement_child(tmp_path, monkeypatch):
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path

    proof_path = os.environ.get("TLDW_NATIVE_RETIREMENT_PROOF")
    if proof_path is None:
        pytest.skip("private physical-retirement child only")
    from tldw_chatbook.Backup_Recovery import storage_admission
    from tldw_chatbook.Backup_Recovery.admission import AdmissionError

    reached = {}
    native_close = storage_admission.os.close

    def fail_exact_pin(fd):
        if fd == reached.get("fd"):
            reached["close_reached"] = True
            raise OSError("injected captured SQLite pin close")
        return native_close(fd)

    with (
        pytest.raises(AdmissionError, match="resources_not_retired"),
        captured_operation(tmp_path, monkeypatch) as captured,
    ):
        with captured.journal._locked(exclusive=False) as parent:
            before = tuple(captured.journal._records(parent))
        with (
            pytest.raises(RuntimeError, match="resources_not_retired"),
            captured.sqlite_reads(
                password=b"test-private-mobile-rollback", cancel=Event()
            ) as reads,
        ):
            reads.page("legacy", "chat", "messages")
            reached["fd"] = reads._connections["legacy", "chat"][
                1
            ].capture_stream.fileno()
            reached["identity"] = os.fstat(reached["fd"])
            monkeypatch.setattr(storage_admission.os, "close", fail_exact_pin)
        assert reached["close_reached"]
        assert not reads.active and reads.resources
        assert storage_admission._local.mobile_sqlite_scope is None
        actual = os.fstat(reached["fd"])
        assert (actual.st_dev, actual.st_ino) == (
            reached["identity"].st_dev,
            reached["identity"].st_ino,
        )
        for scope in (
            captured.owner("db.agent_runs"),
            captured.session.mobile_consolidation_owner(
                captured.journal, "db.agent_runs"
            ),
        ):
            with (
                pytest.raises(ValueError, match="readers_not_retired"),
                scope,
            ):
                pytest.fail("unretired native pin admitted owner")
        captured.claim.check()
        with captured.journal._locked(exclusive=False) as parent:
            assert tuple(captured.journal._records(parent)) == before
    assert storage_admission._failed_capture_holds
    probe = """from Tests.real_profile_guard import install
install()
from Tests.network_guard import install
install()
from pathlib import Path
from tldw_chatbook.Backup_Recovery.control_records import admission_authority
from tldw_chatbook.Backup_Recovery.admission import AdmissionError
import sys
try:
 with admission_authority(Path(sys.argv[1])).maintenance(('bootstrap.unbound','mobile.profile'),0.1): pass
except AdmissionError:
 sys.exit(23)
raise AssertionError('uncertain retirement released maintenance')
"""
    control_root = str(captured.session._control.parent)
    result = subprocess.run(
        [sys.executable, "-c", probe, control_root],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 23, result.stdout + result.stderr
    Path(proof_path).write_text(
        json.dumps(
            {
                "control_root": control_root,
                "owner_refusals": 2,
                "contender_returncode": result.returncode,
                "pin_fd": reached["fd"],
                "pid": os.getpid(),
            }
        )
    )
    # Process exit, never retries or cleared flags, retires the quarantined pin.
    os._exit(0)
