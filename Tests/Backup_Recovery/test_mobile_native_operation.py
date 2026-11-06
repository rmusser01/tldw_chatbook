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
    from functools import partial

    from Tests.Backup_Recovery import mobile_consolidation_fixture
    from Tests.Backup_Recovery.test_mobile_owning_transactions import _small_native_seed

    monkeypatch.setattr(
        mobile_consolidation_fixture,
        "_seed",
        partial(_small_native_seed, source_case="physical_snapshot"),
    )

    from tldw_chatbook.Backup_Recovery import mobile_runs_preflight

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
        with pytest.raises(
            ValueError, match="mobile_native_operation_blocked"
        ) as error:
            captured.consolidate(
                password=b"test-private-mobile-rollback", cancel=Event()
            )
        blockers = error.value.blockers
        assert len(observed) == 1
        plan = observed[0]
        assert plan.source_rows
        assert all(blocker in blockers for blocker in plan.blockers)
        assert {
            "workspace_records",
            "conversations",
            "messages",
            "agent_runs",
            "change_snapshots",
        } <= {row.table for row in plan.source_rows}
        assert plan.pending_duties
        snapshots = {
            row.original for row in plan.rows if row.table == "change_snapshots"
        }
        assert snapshots
        assert {(row.table, row.original, row.reason) for row in blockers} == {
            ("change_snapshots", original, "native_physical_dependency_unproved")
            for original in snapshots
        }
        for diagnostic in (plan, plan.integrated, plan.integrated.sync.trace.content):
            with pytest.raises(ValueError):
                diagnostic.require_complete()
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


def test_real_writer_row_data_prices_once_and_refuses_marker_loss(
    tmp_path, monkeypatch
):
    from functools import partial

    from Tests.Backup_Recovery import mobile_consolidation_fixture
    from Tests.Backup_Recovery.test_mobile_owning_transactions import _small_native_seed
    from tldw_chatbook.Backup_Recovery import (
        mobile_captured_sqlite,
        mobile_identity_plan,
        mobile_runs_preflight,
        mobile_sync_plan,
    )
    from tldw_chatbook.Backup_Recovery.limits import _TraceReadLimit
    from tldw_chatbook.Backup_Recovery.mobile_identity_plan import _rows
    from tldw_chatbook.Backup_Recovery.models import _TraceReadBudget
    from tldw_chatbook.Backup_Recovery.storage_admission import (
        _local,
        _mobile_admission,
    )

    monkeypatch.setattr(
        mobile_consolidation_fixture,
        "_seed",
        partial(_small_native_seed, source_case="physical_snapshot"),
    )
    native_plan = mobile_runs_preflight._plan_runs
    native_page = mobile_captured_sqlite._CapturedSQLite.page
    physical, observed = [], []
    foreign_pages = {"foreign": "must remain untouched"}
    foreign_marker = ("foreign registration",)

    def page(self, purpose, family, table, *, offset=0, limit=100):
        result = native_page(self, purpose, family, table, offset=offset, limit=limit)
        physical.append((purpose, family, table, offset, limit))
        return result

    def observe(reads, *, _read_budget):
        budget = _read_budget
        admission = _mobile_admission(reads.session)
        assert budget is reads.read_budget is admission[2]
        assert budget._parent is admission[4] is None
        owner = mobile_captured_sqlite._row_data(reads, budget)
        registration = reads._row_data_owner
        origin, resources, pages = owner.origin, reads.resources, owner.pages
        assert registration is owner.registration
        assert registration is mobile_captured_sqlite._row_data_marker.get()
        assert registration[0] is owner and registration[1] is origin
        fee_ids = []
        native_value = _TraceReadBudget.value

        def value(self, actual, **options):
            if self is budget:
                fee_ids.append(id(actual))
            return native_value(self, actual, **options)

        with monkeypatch.context() as fee:
            fee.setattr(_TraceReadBudget, "value", value)
            first = _rows(reads, "legacy", "chat", "console_trace_artifacts")
            columns, native_rows = pages[
                "legacy", "chat", "console_trace_artifacts", 0, 1000
            ]
            assert first and all(fee_ids.count(id(row)) == 1 for row in native_rows)
            paid = len(fee_ids)
            second = _rows(
                reads, "legacy", "chat", "console_trace_artifacts", _read_budget=budget
            )
            assert len(fee_ids) == paid
            sync_first = _rows(reads, "legacy", "chat", "sync_log", _read_budget=budget)
            paid = len(fee_ids)
            sync_second = mobile_sync_plan._sync_rows(reads, budget)
            assert sync_first and sync_first == sync_second
            assert len(fee_ids) == paid
            assert all(
                left is not right
                for left, right in zip(sync_first, sync_second, strict=True)
            )
            assert all(
                left[key] is right[key]
                for left, right in zip(sync_first, sync_second, strict=True)
                for key in left
            )
        assert first == second and first is not second
        assert all(left is not right for left, right in zip(first, second, strict=True))
        blob = next(name for name in columns if type(first[0][name]) is bytes)
        original_blob = native_rows[0][columns.index(blob)]
        assert first[0][blob] is second[0][blob] is original_blob
        first[0][blob] = b"caller-local replacement"
        assert second[0][blob] is original_blob
        assert (
            physical.count(("legacy", "chat", "console_trace_artifacts", 0, 1000)) == 1
        )
        owner.page("legacy", "chat", "console_trace_artifacts", offset=0, limit=1)
        assert physical.count(("legacy", "chat", "console_trace_artifacts", 0, 1)) == 1
        empty = owner.page(
            "legacy", "chat", "console_trace_artifacts", offset=2**31 - 1, limit=1000
        )
        assert empty[1] == ()
        assert (
            owner.page(
                "legacy",
                "chat",
                "console_trace_artifacts",
                offset=2**31 - 1,
                limit=1000,
            )
            is empty
        )

        absent = []
        native_absence = mobile_captured_sqlite._runs24_receipts_absent

        def absence(*args):
            result = native_absence(*args)
            absent.append(result)
            return result

        with monkeypatch.context() as receipt:
            receipt.setattr(mobile_captured_sqlite, "_runs24_receipts_absent", absence)
            for _ in range(2):
                assert _rows(reads, "legacy", "runs", "console_command_receipts") == ()
        assert absent == [True, True]
        assert ("legacy", "runs", "console_command_receipts", 0, 1000) not in pages

        class Hostile(str):
            def __eq__(self, other):
                pytest.fail("selector equality ran before exact type guard")

            def __hash__(self) -> int:
                raise AssertionError("selector hashing ran before exact type guard")

        for offset, width, purpose in (
            (False, 1000, "legacy"),
            (0, 1000.0, "legacy"),
            (0, 1000, Hostile("legacy")),
        ):
            before = budget._remaining, len(physical)
            with pytest.raises(ValueError, match="^mobile_sqlite_page_limit$"):
                owner.page(
                    purpose,
                    "chat",
                    "console_trace_artifacts",
                    offset=offset,
                    limit=width,
                )
            assert (budget._remaining, len(physical)) == before

        def no_lookup(*args, **kwargs):
            pytest.fail("invalid row-data reached page lookup")

        cases = (
            "slot",
            "marker",
            "resource",
            "slot-marker",
            "slot-resource",
            "marker-resource",
            "missing-slot",
            "origin",
            "resources-origin",
            "marker-origin",
            "registration-pair",
            "copied-owner",
            "counter",
            "parent",
            "tls",
            "member",
        )
        for case in cases:
            before = budget._remaining, len(physical)
            token = None
            removed_at = swapped_at = None
            with monkeypatch.context() as fault:
                fault.setattr(mobile_captured_sqlite._RowData, "page", no_lookup)
                if case in {"slot", "slot-marker", "slot-resource"}:
                    fault.setattr(reads, "_row_data_owner", None)
                if case == "missing-slot":
                    fault.delattr(reads, "_row_data_owner")
                if case in {"resource", "slot-resource", "marker-resource"}:
                    removed_at = resources.index(owner)
                    resources.pop(removed_at)
                if case in {"origin", "resources-origin", "marker-origin"}:
                    replacement = list(resources)
                    altered = origin[:2] + (replacement,) + origin[3:]
                    fault.setattr(owner, "origin", altered)
                    if case == "resources-origin":
                        fault.setattr(reads, "resources", replacement)
                if case in {
                    "marker",
                    "slot-marker",
                    "marker-resource",
                    "marker-origin",
                    "registration-pair",
                }:
                    marker = (
                        (owner, owner.origin)
                        if case in {"marker-origin", "registration-pair"}
                        else None
                    )
                    token = mobile_captured_sqlite._row_data_marker.set(marker)
                    if case == "registration-pair":
                        fault.setattr(reads, "_row_data_owner", marker)
                if case == "copied-owner":
                    swapped_at = resources.index(owner)
                    resources[swapped_at] = copy(owner)
                if case == "counter":
                    fault.setattr(reads, "read_budget", copy(budget))
                if case == "parent":
                    fault.setattr(budget, "_parent", object())
                if case == "tls":
                    fault.setattr(_local, "mobile_sqlite_scope", None)
                if case == "member":
                    fault.setitem(
                        reads._connections,
                        ("legacy", "chat"),
                        reads._connections["canonical", "chat"],
                    )
                try:
                    with pytest.raises(ValueError, match="^mobile_row_data_changed$"):
                        _rows(reads, "legacy", "chat", "console_trace_artifacts")
                    assert (budget._remaining, len(physical)) == before, case
                finally:
                    if token is not None:
                        mobile_captured_sqlite._row_data_marker.reset(token)
                    if removed_at is not None:
                        resources.insert(removed_at, owner)
                    if swapped_at is not None:
                        resources[swapped_at] = owner

        plan = native_plan(reads, _read_budget=budget)
        assert plan.source_rows and plan.blockers == ()

        # Spend the SAME allowance; admit list/key, refuse before a fresh dict.
        def no_dict(*args, **kwargs):
            pytest.fail("fresh dict constructed before its native reservation")

        assert len(columns) >= 3
        key_fee, list_fee = 64 + 32 * 5, 64
        row_fee = 64 + 32 * (len(columns) * 2)
        budget.reserve(budget._remaining - (list_fee + key_fee + row_fee - 1))
        before = len(physical)
        with monkeypatch.context() as clone:
            clone.setattr(mobile_identity_plan, "dict", no_dict, raising=False)
            with pytest.raises(_TraceReadLimit, match="mobile_trace_input_limit"):
                _rows(reads, "legacy", "chat", "console_trace_artifacts")
        assert len(physical) == before
        # A fresh physical page fits the key fee but cannot be retained unpriced.
        before = len(physical)
        with pytest.raises(_TraceReadLimit, match="mobile_trace_input_limit"):
            owner.page("legacy", "chat", "console_trace_artifacts", offset=0, limit=999)
        assert len(physical) == before + 1
        assert ("legacy", "chat", "console_trace_artifacts", 0, 999) not in pages
        budget.reserve(budget._remaining)
        assert budget._remaining == 0
        before = len(physical), len(pages)
        with pytest.raises(_TraceReadLimit, match="mobile_trace_input_limit"):
            owner.page(
                "legacy", "chat", "console_trace_artifacts", offset=0, limit=1000
            )
        assert (len(physical), len(pages)) == before
        owner.pages = foreign_pages
        owner.retire()
        assert owner.closed and pages == {}
        assert foreign_pages == {"foreign": "must remain untouched"}
        assert reads._row_data_owner is registration
        assert any(resource is owner for resource in resources)
        assert owner.origin is origin
        for lose_marker in (False, True):
            before = budget._remaining, len(physical)
            token = (
                mobile_captured_sqlite._row_data_marker.set(None)
                if lose_marker
                else None
            )
            try:
                with pytest.raises(ValueError, match="^mobile_row_data_changed$"):
                    _rows(reads, "legacy", "chat", "console_trace_artifacts")
                assert (budget._remaining, len(physical)) == before
            finally:
                if token is not None:
                    mobile_captured_sqlite._row_data_marker.reset(token)
        # Leave these corruptions in place: lexical cleanup must use issuer locals.
        owner.origin = ()
        owner.pages = foreign_pages
        mobile_captured_sqlite._row_data_marker.set(foreign_marker)
        observed.append((reads, owner, resources, pages, origin[16][0][1][0]))
        return plan

    monkeypatch.setattr(mobile_captured_sqlite._CapturedSQLite, "page", page)
    monkeypatch.setattr(mobile_runs_preflight, "_plan_runs", observe)
    with captured_operation(tmp_path, monkeypatch) as captured:
        before = _fingerprints(tmp_path / "profile")
        with captured.journal._locked(exclusive=False) as fd:
            records = tuple(captured.journal._records(fd))
        restore = mobile_captured_sqlite._row_data_marker.set(
            mobile_captured_sqlite._row_data_marker.get()
        )
        try:
            with pytest.raises(ValueError, match="^mobile_row_data_changed$"):
                captured.consolidate(
                    password=b"test-private-mobile-rollback", cancel=Event()
                )
            assert len(observed) == 1
            reads, owner, resources, pages, connection = observed[0]
            assert owner.closed and pages == {} and owner.origin is None
            assert reads._row_data_owner is None
            assert not any(resource is owner for resource in resources)
            assert reads.resources == []
            owner.retire()
            assert foreign_pages == {"foreign": "must remain untouched"}
            with pytest.raises(sqlite3.ProgrammingError, match="closed"):
                sqlite3.Connection.in_transaction.__get__(connection)
            assert mobile_captured_sqlite._row_data_marker.get() is foreign_marker
            assert _fingerprints(tmp_path / "profile") == before
            with captured.journal._locked(exclusive=False) as fd:
                assert tuple(captured.journal._records(fd)) == records
            captured.claim.check()
        finally:
            mobile_captured_sqlite._row_data_marker.reset(restore)


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


@pytest.mark.parametrize("resource", ("reader", "source_claim"))
def test_uncertain_reader_retirement_prevents_native_owner_until_process_exit(
    tmp_path, resource
):
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path

    proof_path = tmp_path / "proof.json"
    child_test = (
        "test_uncertain_retirement_child"
        if resource == "reader"
        else "test_uncertain_source_claim_retirement_child"
    )
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
            "Tests/Backup_Recovery/test_mobile_native_operation.py::" + child_test,
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
    control_root = str(captured.session._control.parent)
    contender_returncode = _assert_retirement_quarantine(control_root)
    Path(proof_path).write_text(
        json.dumps(
            {
                "control_root": control_root,
                "owner_refusals": 2,
                "contender_returncode": contender_returncode,
                "pin_fd": reached["fd"],
                "pid": os.getpid(),
            }
        )
    )
    # Process exit, never retries or cleared flags, retires the quarantined pin.
    os._exit(0)


def _assert_retirement_quarantine(control_root):
    import subprocess
    import sys

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
    result = subprocess.run(
        [sys.executable, "-c", probe, control_root],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 23, result.stdout + result.stderr
    return result.returncode


def test_uncertain_source_claim_retirement_child(tmp_path, monkeypatch):
    import json
    import os
    from pathlib import Path

    proof_path = os.environ.get("TLDW_NATIVE_RETIREMENT_PROOF")
    if proof_path is None:
        pytest.skip("private source-claim retirement child only")
    from tldw_chatbook.Backup_Recovery import storage_admission
    from tldw_chatbook.Backup_Recovery.admission import AdmissionError
    from tldw_chatbook.Backup_Recovery.mobile_consolidation_apply import (
        _captured_mobile,
    )

    reached = {}
    native_close = storage_admission.os.close

    def fail_exact_claim(fd):
        if fd == reached.get("fd"):
            reached["close_reached"] = True
            raise OSError("injected source-claim descriptor close")
        return native_close(fd)

    admitted, refusals = [], []
    with (
        pytest.raises(AdmissionError, match="resources_not_retired"),
        _capture_fixture(tmp_path, monkeypatch) as values,
    ):
        session, journal, _, _, preview, inventory, _ = values
        output = tmp_path / "archives"
        output.mkdir(mode=0o700)
        with (
            pytest.raises(RuntimeError, match="resources_not_retired"),
            _captured_mobile(
                session,
                journal,
                preview,
                inventory.scope_digest,
                destination=output,
                password=b"test-private-mobile-rollback",
                cancel=Event(),
            ) as captured,
        ):
            reached["fd"] = captured.claim.descriptor
            reached["identity"] = os.fstat(reached["fd"])
            monkeypatch.setattr(storage_admission.os, "close", fail_exact_claim)
        assert reached["close_reached"]
        assert not captured.active and captured.claim.custody.close_failed
        actual = os.fstat(reached["fd"])
        assert (actual.st_dev, actual.st_ino) == (
            reached["identity"].st_dev,
            reached["identity"].st_ino,
        )
        with journal._locked(exclusive=False) as parent:
            before = tuple(journal._records(parent))
            before_bytes = _fingerprints(journal.root)
        before_profile = _fingerprints(tmp_path / "profile")
        for name, owner_scope in (
            ("captured", captured.owner("db.agent_runs")),
            ("direct", session.mobile_consolidation_owner(journal, "db.agent_runs")),
        ):
            try:
                with owner_scope:
                    admitted.append(name)
            except (ValueError, RuntimeError) as error:
                refusals.append((name, str(error)))
        with journal._locked(exclusive=False) as parent:
            after = tuple(journal._records(parent))
            after_bytes = _fingerprints(journal.root)
        after_profile = _fingerprints(tmp_path / "profile")
    assert storage_admission._failed_capture_holds
    assert not admitted
    assert refusals == [
        ("captured", "mobile_lifecycle_retired"),
        ("direct", "mobile_native_readers_not_retired"),
    ]
    assert before == after and before_bytes == after_bytes
    assert before_profile == after_profile
    actual = os.fstat(reached["fd"])
    assert (actual.st_dev, actual.st_ino) == (
        reached["identity"].st_dev,
        reached["identity"].st_ino,
    )
    control_root = str(session._control.parent)
    contender_returncode = _assert_retirement_quarantine(control_root)
    Path(proof_path).write_text(
        json.dumps(
            {
                "control_root": control_root,
                "owner_refusals": len(refusals),
                "refusals": refusals,
                "contender_returncode": contender_returncode,
                "claim_fd": reached["fd"],
                "claim_close_failed": captured.claim.custody.close_failed,
                "captured_active": captured.active,
                "journal_records_before": len(before),
                "journal_records_after": len(after),
                "journal_bytes_unchanged": before_bytes == after_bytes,
                "profile_unchanged": before_profile == after_profile,
                "pid": os.getpid(),
            }
        )
    )
    # Actual process exit is the only retirement of this uncertain native claim.
    os._exit(0)


def test_real_writer_reproves_candidate_after_data_hit_before_effects(
    tmp_path, monkeypatch
):
    from functools import partial

    from Tests.Backup_Recovery import mobile_consolidation_fixture
    from Tests.Backup_Recovery.test_mobile_owning_transactions import _small_native_seed
    from tldw_chatbook.Backup_Recovery import (
        mobile_captured_sqlite,
        mobile_runs_preflight,
    )
    from tldw_chatbook.Backup_Recovery.storage_admission import _mobile_admission

    monkeypatch.setattr(
        mobile_consolidation_fixture,
        "_seed",
        partial(_small_native_seed, source_case="physical_snapshot"),
    )
    native_plan, native_page = (
        mobile_runs_preflight._plan_runs,
        mobile_captured_sqlite._CapturedSQLite.page,
    )
    physical, observed = [], []

    def page(self, purpose, family, table, *, offset=0, limit=100):
        result = native_page(self, purpose, family, table, offset=offset, limit=limit)
        physical.append((purpose, family, table, offset, limit))
        return result

    def observe(reads, *, _read_budget):
        admission = _mobile_admission(reads.session)
        assert _read_budget is reads.read_budget is admission[2]
        plan = native_plan(reads, _read_budget=_read_budget)
        owner = mobile_captured_sqlite._row_data(reads, _read_budget)
        key = ("legacy", "chat", "messages", 0, 1000)
        assert key in owner.pages
        before = len(physical)
        assert owner.page(*key[:3], offset=key[3], limit=key[4]) is owner.pages[key]
        assert len(physical) == before
        candidate = reads._connections["legacy", "chat"][3]
        with candidate.open("r+b") as stream:
            first = stream.read(1)
            assert len(first) == 1
            stream.seek(0)
            stream.write(bytes([first[0] ^ 1]))
            stream.flush()
        observed.append((reads, owner))
        return plan

    monkeypatch.setattr(mobile_captured_sqlite._CapturedSQLite, "page", page)
    monkeypatch.setattr(mobile_runs_preflight, "_plan_runs", observe)
    with captured_operation(tmp_path, monkeypatch) as captured:
        before = _fingerprints(tmp_path / "profile")
        with captured.journal._locked(exclusive=False) as fd:
            records = tuple(captured.journal._records(fd))
        with pytest.raises(ValueError, match="^mobile_sqlite_candidate_changed$"):
            captured.consolidate(
                password=b"test-private-mobile-rollback", cancel=Event()
            )
        assert len(observed) == 1
        reads, owner = observed[0]
        assert owner.closed and owner.pages == {} and owner.origin is None
        assert reads.resources == [] and reads._row_data_owner is None
        assert mobile_captured_sqlite._row_data_marker.get() is None
        assert _fingerprints(tmp_path / "profile") == before
        with captured.journal._locked(exclusive=False) as fd:
            assert tuple(captured.journal._records(fd)) == records
        captured.claim.check()
