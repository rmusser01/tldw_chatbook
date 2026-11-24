"""Read-only planning must use the real native primary and required settlement."""

from pathlib import Path

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run

pytestmark = pytest.mark.bootstrap_profile

# A hang bound for every child, not a speed bound: these children take 35-62 s
# at load 70-90 (lessons-testing-evidence: "A fixed child budget kills a child
# that is only slow").
_HANG_BUDGET_SECONDS = 180


def test_accepted_planning_uses_original_native_primary(tmp_path):
    _run(
        tmp_path,
        "primary",
        "success",
        script=Path(__file__).with_name("workflow_planning_probe.py").read_text(),
        timeout=_HANG_BUDGET_SECONDS,
    )


@pytest.mark.parametrize("mode", ["new-saved", "new-unsaved"])
def test_original_initial_planning_persistence_and_acceptance(tmp_path, mode):
    _run(
        tmp_path,
        mode,
        "success",
        script=Path(__file__).with_name("workflow_planning_probe.py").read_text(),
        timeout=_HANG_BUDGET_SECONDS,
    )


def test_planning_never_constructs_optional_tool_services(tmp_path):
    _run(
        tmp_path,
        "factory",
        "success",
        script=Path(__file__).with_name("workflow_planning_probe.py").read_text(),
        timeout=_HANG_BUDGET_SECONDS,
    )


@pytest.mark.parametrize(
    "mode", ["invalid-json", "budget-zero", "cancel-cleanup", "tools"]
)
def test_native_planning_receiving_boundaries(tmp_path, mode):
    _run(
        tmp_path,
        mode,
        "success",
        script=Path(__file__).with_name("workflow_planning_probe.py").read_text(),
        timeout=_HANG_BUDGET_SECONDS,
    )


@pytest.mark.parametrize("mode", ["wrapper", "wrapper-changed", "wrapper-unavailable"])
def test_native_planning_future_verifier_metadata(tmp_path, mode):
    _run(
        tmp_path,
        mode,
        "success",
        script=Path(__file__).with_name("workflow_planning_probe.py").read_text(),
        timeout=_HANG_BUDGET_SECONDS,
    )


@pytest.mark.parametrize(
    "mode", ["instructions", "unused", "repair", "direct-guard", "cleanup-source"]
)
def test_native_planning_original_input_and_receivers(tmp_path, mode):
    _run(
        tmp_path,
        mode,
        "success",
        script=Path(__file__).with_name("workflow_planning_probe.py").read_text(),
        timeout=_HANG_BUDGET_SECONDS,
    )


@pytest.mark.parametrize("mode", ["worker-failure", "response-close-failure"])
def test_native_planning_required_failure_retains_original_custody(tmp_path, mode):
    # A failed native scope retains its real owner (ADR-356 decision 6).
    # This is an uncertainty-retention control, never a successful drain receipt.
    with pytest.raises(AssertionError, match="F_REQUIRED_CUSTODY_RETAINED_" + mode):
        _run(
            tmp_path,
            mode,
            "retained",
            script=Path(__file__).with_name("workflow_planning_probe.py").read_text(),
            timeout=_HANG_BUDGET_SECONDS,
        )


@pytest.mark.parametrize(
    "mode", ["required-write", "required-return", "late-source-return"]
)
def test_native_planning_returned_required_failure_hands_custody_to_recovery(
    tmp_path, mode
):
    # After positive physical return, ADR-356 decision 6 records the failure in a
    # recovery receipt and retires the owner once; never a settled drain.
    _run(
        tmp_path,
        mode,
        "success",
        script=Path(__file__).with_name("workflow_planning_probe.py").read_text(),
        timeout=_HANG_BUDGET_SECONDS,
    )
    output = (tmp_path / "restart-output.log").read_text(encoding="utf-8")
    assert "F_RETURNED_REQUIRED_FAILURE_RECOVERED_" + mode in output


def test_planning_retains_root_until_original_provider_worker_returns(tmp_path):
    _run(
        tmp_path,
        "worker-tail",
        "success",
        script=Path(__file__).with_name("workflow_planning_probe.py").read_text(),
        timeout=_HANG_BUDGET_SECONDS,
    )


def test_shutdown_waits_for_same_native_planning_worker_return(tmp_path):
    _run(
        tmp_path,
        "shutdown-worker-tail",
        "success",
        script=Path(__file__).with_name("workflow_planning_probe.py").read_text(),
        timeout=_HANG_BUDGET_SECONDS,
    )
    output = (tmp_path / "restart-output.log").read_text(encoding="utf-8")
    assert "F_SHUTDOWN_WAITED_FOR_PLANNING_WORKER" in output
    # TASK-41636: usage received before shutdown is kept; a settled no-plan drain.
    assert "F_NATIVE_PRIMARY_REQUIRED_PLAN_RETURN" in output


@pytest.mark.parametrize(
    "mode", ["system-context", "system-oversize", "repair-instruction"]
)
def test_native_planning_retains_required_selected_and_repair_inputs(tmp_path, mode):
    _run(
        tmp_path,
        mode,
        "success",
        script=Path(__file__).with_name("workflow_planning_probe.py").read_text(),
        timeout=_HANG_BUDGET_SECONDS,
    )


def test_native_runtime_waits_for_original_maintenance_operation_return(tmp_path):
    _run(
        tmp_path,
        "maintenance-tail",
        "success",
        script=Path(__file__).with_name("workflow_planning_probe.py").read_text(),
        timeout=_HANG_BUDGET_SECONDS,
    )


@pytest.mark.parametrize("control", ["admitted", "kill-switch", "guard-refused"])
def test_planning_stop_poll_uses_its_own_native_admission(tmp_path, control):
    script = Path(__file__).with_name("workflow_planning_probe.py").read_text()
    setup = r"""
    from tldw_chatbook.Agents import activation as poll_activation
    from tldw_chatbook.Agents.automatic_work_runtime import AutomaticWorkContext
    from tldw_chatbook.Agents.run_context import current_run_actor
    from tldw_chatbook.Backup_Recovery.admission_runtime import execution_identity
    from tldw_chatbook.Chat import stream_stall_watchdog as poll_watchdog

    poll_control = POLL_CONTROL
    poll_seen = threading.Event()
    poll_results, poll_owners = [], []
    poll_callback = threading.local()
    poll_actor = []
    provider_entered = threading.Event()
    provider_threads, unarmed_poll_results, armed_pauses = [], [], []
    original_watch = poll_watchdog.watch_content_stalls
    predicate_code = AutomaticWorkContext.should_cancel.__code__

    def observe_native_predicate(frame, event, argument):
        if frame.f_code is not predicate_code or event != "call" or not getattr(poll_callback, "active", False):
            return
        try:
            task = asyncio.current_task()
        except RuntimeError:
            return
        if task is None:
            state = poll_activation._guard.context.get()
            poll_owners.append((
                state is not None and state.identity == execution_identity(),
                current_run_actor() is poll_actor[0],
            ))

    def observe_watch(source, window, **kwargs):
        original_stop = kwargs["should_stop"]
        poll_actor[:] = [current_run_actor()]

        def observe_stop():
            assert asyncio.current_task() is None
            if poll_control == "guard-refused" and not provider_entered.is_set():
                # Let the real worker finish admission before faulting a fresh
                # callback; no synthetic continuation or model-loop wait.
                value = original_stop()
                unarmed_poll_results.append(value)
                return value
            pause = None
            before = os.environ.get("TLDW_AGENTS_WORKFLOW_RUNS_ENABLED")
            if poll_control == "guard-refused":
                from tldw_chatbook.Backup_Recovery import storage_admission
                assert provider_entered.is_set() and provider_threads
                with storage_admission._lock:
                    assert not any(
                        acquisition.thread.ident == provider_threads[0]
                        and acquisition.operation is None
                        for acquisition in storage_admission._pending_acquisitions
                    ), "original provider admission is still pending"
                armed_pauses.append(threading.get_ident())
                pause = storage_admission._begin_local_pause()
            elif poll_control == "kill-switch":
                os.environ["TLDW_AGENTS_WORKFLOW_RUNS_ENABLED"] = "false"
            poll_callback.active = True
            try:
                value = original_stop()
                poll_results.append(value)
                return value
            finally:
                poll_callback.active = False
                if pause is not None:
                    pause.resume()
                if before is None:
                    os.environ.pop("TLDW_AGENTS_WORKFLOW_RUNS_ENABLED", None)
                else:
                    os.environ["TLDW_AGENTS_WORKFLOW_RUNS_ENABLED"] = before
                poll_seen.set()

        kwargs["should_stop"] = observe_stop
        return original_watch(source, window, **kwargs)

    poll_watchdog.watch_content_stalls = observe_watch
    threading.setprofile_all_threads(observe_native_predicate)
""".replace("POLL_CONTROL", repr(control))
    provider = "    def provider(**kwargs):\n"
    assert script.count(provider) == 1
    script = script.replace(
        provider,
        setup
        + provider
        + "        provider_threads.append(threading.get_ident())\n        provider_entered.set()\n"
        + "        assert poll_seen.wait(HANG_WAIT_SECONDS), 'original stop poll did not run'\n",
        1,
    )
    dispatch = '        else:\n            result = await prepared.run_planning()\n        if mode in {"repair", "repair-instruction"}:'
    assert script.count(dispatch) == 1
    script = script.replace(
        dispatch,
        r"""
        else:
            from tldw_chatbook.Agents.automatic_work_budget import AutomaticWorkRefused
            try:
                result = await prepared.run_planning()
            except AutomaticWorkRefused as error:
                assert poll_control != "admitted" and str(error) == "workflow_not_settled"
                with owned_goal_storage(service), service.db.connection() as conn:
                    tokens = list(conn.execute("SELECT state,actual_amount FROM automatic_work_reservations WHERE kind='tokens'"))
                    saved = service.get(attempt.workflow_id)
                assert [tuple(row) for row in tokens] == [("uncertain", None)]
                assert saved.status == "recovery_required" and saved.accounting.uncertain
                result = saved
            assert poll_results and poll_results[0] is (poll_control != "admitted")
            if poll_control != "guard-refused":
                assert poll_owners and all(owned and same_actor for owned, same_actor in poll_owners)
            else:
                assert not poll_owners, "refused callback entered the ledger predicate"
                assert provider_entered.is_set() and armed_pauses
                assert all(value is False for value in unarmed_poll_results)
            print("F_NATIVE_STOP_POLL_" + poll_control, flush=True)
            if poll_control != "admitted":
                assert not result.plans
                return
        if mode in {"repair", "repair-instruction"}:""",
        1,
    )
    cleanup = "        sys.setprofile(None)\n"
    assert script.count(cleanup) == 1
    script = script.replace(
        cleanup, "        threading.setprofile_all_threads(None)\n" + cleanup, 1
    )
    _run(tmp_path, "primary", "success", script=script, timeout=_HANG_BUDGET_SECONDS)
    output = (tmp_path / "restart-output.log").read_text(encoding="utf-8")
    assert "F_NATIVE_STOP_POLL_" + control in output


@pytest.mark.parametrize("control", ["fresh-handle", "close-refused"])
def test_planning_stop_poll_proves_fresh_handle_physical_return(tmp_path, control):
    script = Path(__file__).with_name("workflow_planning_probe.py").read_text()
    setup = r"""
    import sqlite3
    from tldw_chatbook.Agents import activation as poll_activation
    from tldw_chatbook.Agents.automatic_work_runtime import AutomaticWorkContext, current_automatic_work
    from tldw_chatbook.Agents.run_context import current_run_actor
    from tldw_chatbook.Backup_Recovery.admission_runtime import execution_identity
    from tldw_chatbook.DB.private_sqlite import _ordinary_connections
    from tldw_chatbook.Chat import stream_stall_watchdog as poll_watchdog

    poll_control = POLL_CONTROL
    poll_seen = threading.Event()
    poll_results, poll_owners, physical_rows = [], [], []
    poll_callback = threading.local()
    poll_actor = []
    original_watch = poll_watchdog.watch_content_stalls
    predicate_code = AutomaticWorkContext.should_cancel.__code__

    def physically_closed(connection):
        if connection is None:
            return True
        try:
            sqlite3.Connection.in_transaction.__get__(connection)
        except sqlite3.ProgrammingError:
            return True
        return False

    def observe_native_predicate(frame, event, argument):
        if frame.f_code is not predicate_code or event != "return" or not getattr(poll_callback, "active", False):
            return
        row = poll_callback.row
        context = frame.f_locals["self"]
        database = poll_callback.database
        state = poll_activation._guard.context.get()
        poll_owners.append((
            state is not None and state.identity == execution_identity(),
            current_run_actor() is poll_actor[0],
        ))
        row["predicate_false"] = argument is False
        row["same_database"] = context.ledger._db is database
        connection = getattr(database._thread_local, "conn", None)
        row["connection"] = connection
        row["lease"] = _ordinary_connections.get(connection) if connection is not None else None
        if argument is False and connection is not None and poll_control == "close-refused":
            # Fault only AFTER the genuine admitted predicate returned False.
            # The DB/guard callable identities remain untouched.
            row["had_binding"] = "close" in vars(connection)
            row["binding"] = vars(connection).get("close")

            def refuse_physical_close():
                row["refusal_calls"] += 1
                raise RuntimeError("original callback physical close refused")

            connection.close = refuse_physical_close
            row["injected"] = True

    def observe_watch(source, window, **kwargs):
        original_stop = kwargs["should_stop"]
        poll_actor[:] = [current_run_actor()]

        def observe_stop():
            assert asyncio.current_task() is None
            context = current_automatic_work()
            database = context.ledger._db
            previous = getattr(database._thread_local, "conn", None)
            # Native stock close is setup, not the fault: the actual predicate
            # must acquire a fresh handle on this original model-call loop.
            database.close()
            row = {
                "setup_closed": physically_closed(previous) and getattr(database._thread_local, "conn", None) is None,
                "predicate_false": False, "same_database": False,
                "refusal_calls": 0, "injected": False,
            }
            physical_rows.append(row)
            poll_callback.database, poll_callback.row = database, row
            poll_callback.active = True
            try:
                value = original_stop()
                poll_results.append(value)
                connection = row.get("connection")
                if connection is not None:
                    row["retired_before"] = physically_closed(connection) and connection not in _ordinary_connections and not database._connection_quiescence.is_registered(connection)
                    row["retained_before"] = not physically_closed(connection) and getattr(database._thread_local, "conn", None) is connection and _ordinary_connections.get(connection) is row["lease"] and database._connection_quiescence.is_registered(connection)
                return value
            finally:
                poll_callback.active = False
                connection = row.get("connection")
                if connection is not None:
                    if row["injected"]:
                        if row["had_binding"]:
                            connection.close = row["binding"]
                        else:
                            del connection.close
                        row["binding_restored"] = ("close" in vars(connection)) is row["had_binding"] and (not row["had_binding"] or vars(connection)["close"] is row["binding"])
                    # Only the original public/native closer may retire it.
                    # No cache, lease, registry or unknown marker is reset.
                    database.close()
                    row["retired_after_native_close"] = physically_closed(connection) and connection not in _ordinary_connections and not database._connection_quiescence.is_registered(connection)
                poll_seen.set()

        kwargs["should_stop"] = observe_stop
        return original_watch(source, window, **kwargs)

    poll_watchdog.watch_content_stalls = observe_watch
    threading.setprofile_all_threads(observe_native_predicate)
""".replace("POLL_CONTROL", repr(control))
    provider = "    def provider(**kwargs):\n"
    assert script.count(provider) == 1
    script = script.replace(
        provider,
        setup
        + provider
        + "        assert poll_seen.wait(HANG_WAIT_SECONDS), 'original stop poll did not run'\n",
        1,
    )
    dispatch = '        else:\n            result = await prepared.run_planning()\n        if mode in {"repair", "repair-instruction"}:'
    assert script.count(dispatch) == 1
    script = script.replace(
        dispatch,
        r"""
        else:
            from tldw_chatbook.Agents.automatic_work_budget import AutomaticWorkRefused
            try:
                result = await prepared.run_planning()
            except AutomaticWorkRefused as error:
                assert poll_control == "close-refused" and str(error) == "workflow_not_settled"
                with owned_goal_storage(service), service.db.connection() as conn:
                    tokens = list(conn.execute("SELECT state,actual_amount FROM automatic_work_reservations WHERE kind='tokens'"))
                    saved = service.get(attempt.workflow_id)
                assert [tuple(row) for row in tokens] == [("uncertain", None)]
                assert saved.status == "recovery_required" and saved.accounting.uncertain
                result = saved
            assert poll_owners and all(owned and same_actor for owned, same_actor in poll_owners)
            assert physical_rows
            for row in physical_rows:
                assert row["setup_closed"] and row["predicate_false"] and row["same_database"]
                assert row["connection"] is not None and row["lease"] is not None
                assert row["retired_after_native_close"], "actual native handle did not physically return"
                if poll_control == "close-refused":
                    assert row["injected"] and row["refusal_calls"] > 0
                    assert row["retained_before"] and not row["retired_before"]
                    assert row["binding_restored"]
                else:
                    assert row["retired_before"] and row["refusal_calls"] == 0
            assert poll_results and poll_results[0] is (poll_control == "close-refused")
            print("F_NATIVE_STOP_POLL_" + poll_control, flush=True)
            if poll_control == "close-refused":
                assert not result.plans
                return
        if mode in {"repair", "repair-instruction"}:""",
        1,
    )
    cleanup = "        sys.setprofile(None)\n"
    assert script.count(cleanup) == 1
    script = script.replace(
        cleanup, "        threading.setprofile_all_threads(None)\n" + cleanup, 1
    )
    _run(tmp_path, "primary", "success", script=script, timeout=_HANG_BUDGET_SECONDS)
    output = (tmp_path / "restart-output.log").read_text(encoding="utf-8")
    assert "F_NATIVE_STOP_POLL_" + control in output
