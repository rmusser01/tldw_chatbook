"""Original Console child admission must decide even before worker body entry."""

import asyncio
import inspect
import json
import threading
import time
from contextlib import nullcontext
from pathlib import Path

import pytest

from tldw_chatbook.Agents import activation, agent_service
from tldw_chatbook.Agents.automatic_work_runtime import AutomaticWorkContext

from Tests.Chat.test_console_ordinary_tree import ordinary_host as ordinary_host  # noqa: PLC0414
from Tests.Chat.test_console_reference_custody import reference_host as reference_host  # noqa: PLC0414
from Tests.Chat.test_console_reference_custody import until

pytestmark = pytest.mark.bootstrap_profile

# _run supplies a private HOME/config/data layout. Nested pytest must select
# that same profile before its conftest imports, rather than the outer runner's.
_NESTED_PROFILE_SCRIPT = (
    "import os\nfrom pathlib import Path\n"
    "os.environ['TLDW_TEST_CONFIG_ROOT']=str(Path(os.environ['TLDW_CONFIG_PATH']).parent.parent)\n"
    "import pytest\n"
)


def _assert_nested_profile():
    import os
    from tldw_chatbook.Backup_Recovery import bootstrap

    root = Path(os.environ["TLDW_TEST_CONFIG_ROOT"]).resolve()
    assert Path.home().resolve() == root / "home"
    assert Path(os.environ["XDG_DATA_HOME"]).resolve() == root / "data"
    assert bootstrap.effective_config_path().resolve() == root / "config/config.toml"
    assert bootstrap.default_bootstrap_root().resolve().is_relative_to(root / "home")


@pytest.mark.asyncio
@pytest.mark.parametrize("setup_fault", ["passive", "opaque"])
async def test_original_child_db_entry_failure_decides_and_settles(
    ordinary_host, monkeypatch, setup_fault
):
    from Tests.Agents.test_agent_service import fence
    from Tests.Chat.test_console_skill_substitution import _RecordingGateway
    from tldw_chatbook.Backup_Recovery import participants

    import os
    import sys
    import traceback

    if os.environ.get("TASK8_CHILD_REFUSAL_PEER") != setup_fault:
        from Tests.Backup_Recovery.test_home_citation_retirement import _run

        script = (
            _NESTED_PROFILE_SCRIPT
            + "os.environ['TASK8_CHILD_REFUSAL_PEER']="
            + repr(setup_fault)
            + "\n"
            "result=pytest.main(['-p','no:cacheprovider','-q','-s','-W','error::pytest.PytestUnraisableExceptionWarning','-W','error::pytest.PytestUnraisableExceptionWarning',"
            + repr(
                __file__
                + "::test_original_child_db_entry_failure_decides_and_settles["
                + setup_fault
                + "]"
            )
            + "])\nassert result == 0\nprint('retired and reopened')\n"
        )
        if setup_fault == "opaque":
            with pytest.raises(
                AssertionError, match="TASK8_OPAQUE_CHILD_CUSTODY_RETAINED"
            ):
                await asyncio.to_thread(
                    _run,
                    ordinary_host.tmp_path,
                    "child",
                    "refused",
                    script=script,
                    timeout=30,
                )
        else:
            await asyncio.to_thread(
                _run,
                ordinary_host.tmp_path,
                "child",
                "refused",
                script=script,
                timeout=30,
            )
        return
    _assert_nested_profile()

    h = ordinary_host
    monkeypatch.setenv("TLDW_AGENTS_MAX_LIVE_SUBAGENTS", "2")
    monkeypatch.setenv("TLDW_AGENTS_AUTOWAKE_ENABLED", "false")
    h.controller._confirm_project_instruction_dispatch = lambda notice: "proceed"
    original_cached = participants.core_cached_connection
    original_wait = threading.Event.wait
    original_hook = threading.excepthook
    failures, handshakes, notifications, services = [], [], [], []
    failure = OSError("original child DB entry refused")

    def cached(database, connection):
        value = original_cached(database, connection)
        if threading.current_thread().name.startswith("fleet-") and not failures:
            failures.append(threading.current_thread())
            raise failure
        return value

    def wait(event, timeout=None):
        caller = inspect.currentframe().f_back
        if caller.f_code.co_name == "_launch_prepared_child_result":
            services.append(caller.f_locals["self"])
            # Bound the RED broken handshake without leaving its parent parked.
            value = original_wait(event, 2)
            handshakes.append(value)
            return value
        return original_wait(event, timeout)

    def hook(args):
        if not args.thread.name.startswith("fleet-"):
            original_hook(args)

    from tldw_chatbook.DB import base_db

    original_trace = threading.gettrace()
    passive_code = base_db._operation_owned_connection.__wrapped__.__code__

    def trace(frame, event, arg):
        if frame.f_code is not passive_code:
            return None
        entry = frame.f_locals.get("_entry_return")
        if (
            event == "line"
            and entry is not None
            and entry.stage == "passive_returned"
            and threading.current_thread().name.startswith("fleet-")
            and not failures
        ):
            failures.append(threading.current_thread())
            raise failure
        return trace

    if setup_fault == "opaque":
        monkeypatch.setattr(participants, "core_cached_connection", cached)
    else:
        threading.settrace(trace)
    monkeypatch.setattr(threading.Event, "wait", wait)
    monkeypatch.setattr(threading, "excepthook", hook)

    class Gateway(_RecordingGateway):
        async def stream_chat(self, resolution, messages, **kwargs):
            self.payloads.append(messages)
            if len(self.payloads) == 1:
                yield fence("spawn_subagent", {"task": "must never enter child MODEL"})
            else:
                yield "parent completed"

    gateway = Gateway()
    h.controller.provider_gateway = h.bridge._gateway = gateway
    # Observe the installed service notification without replacing the launcher.
    from tldw_chatbook.Agents.agent_service import AgentService

    init = AgentService.__init__

    def initialized(self, *args, **kwargs):
        init(self, *args, **kwargs)
        original = self._on_child_settled

        def settled(run_id, status):
            notifications.append((run_id, status, self.db.get_run(run_id)["status"]))
            if original is not None:
                original(run_id, status)

        self._on_child_settled = settled

    monkeypatch.setattr(AgentService, "__init__", initialized)
    caught_error = None
    try:
        send = asyncio.create_task(
            h.controller.submit_draft("spawn a child", session_id=h.session.id)
        )
        h.tasks.append(send)
        if setup_fault == "passive":
            result = await send
        else:
            await until(lambda: bool(notifications) and len(gateway.payloads) == 2)
            result = None
        assert failures, "test missed original DB entry boundary"
        assert handshakes == [True], "original parent never learned failed admission"
        assert result is None or result.accepted
        assert len(gateway.payloads) == 2, "failed setup entered child provider"
        rows = h.database.list_runs(h.session.persisted_conversation_id)
        children = [row for row in rows if row["agent_kind"] == "subagent"]
        assert len(children) == 1 and children[0]["status"] == "error"
        assert notifications == [(children[0]["id"], "error", "error")]
        primary = next(row for row in rows if row["agent_kind"] == "primary")
        steps = h.database.get_run(primary["id"])["steps"]
        spawn_result = next(
            step
            for step in steps
            if step["kind"] == "tool_result" and step["tool_name"] == "spawn_subagent"
        )
        assert "running now" not in spawn_result["result"]
        assert "could not start" in spawn_result["result"]
        assert "original child DB entry refused" in str(
            services[0]._injected_fleet_coordinator.snapshot()
        )
        tree = services[0]._ordinary_tree
        child_owner, member = next(
            (owner, member) for owner, member in tree.members.items() if owner.child
        )
        if setup_fault == "passive":
            await until(lambda: not h.runtime.execution_capacity.snapshot().executions)
            assert member.host_returned and member.drained and member.durable
            assert tree.retired.done()
        else:
            assert child_owner._cleanup_unproven and not child_owner._root_finished
            assert not member.host_returned and not tree.retired.done()
    except BaseException as error:
        caught_error = error
    finally:
        threading.settrace(original_trace)
        for worker in failures:
            worker.join(5)
            assert not worker.is_alive()

    if caught_error is not None:
        traceback.print_exception(caught_error, file=sys.__stderr__)
        sys.__stderr__.flush()
        os._exit(1)

    if setup_fault == "opaque":
        print("TASK8_OPAQUE_CHILD_CUSTODY_RETAINED", file=sys.__stderr__, flush=True)
        os._exit(1)  # Retained custody is a negative oracle, never physical return.


def exercise_tool_refusal(boundary, service, owner):
    """Use the original wrapper, service, storage and finite physical workers."""
    from tldw_chatbook.Agents.agent_models import ToolResult
    from tldw_chatbook.Agents.automatic_work_runtime import current_automatic_work
    from tldw_chatbook.Backup_Recovery import storage_admission

    original_operations = frozenset(owner._operations)
    calls, workers, parks, pending = [], [], [], {}
    active = [True]
    check_code = AutomaticWorkContext.check.__wrapped__.__code__
    admission_code = activation._guard.execution.__wrapped__.__code__
    old_profile = threading.getprofile()
    automatic = current_automatic_work()
    if boundary == "automatic" and automatic is None:
        ledger = service.db.automatic_work
        chain = ledger.create_chain("refusal-probe", root_submission_id="refusal-probe")
        with service.db.connection() as conn:
            row = conn.execute(
                "SELECT owner_id FROM automatic_work_runtime_owner WHERE singleton=1"
            ).fetchone()
        issuer = row[0] if row else "probe"
        parent = service.db.create_run(
            conversation_id="refusal-probe", agent_kind="primary", work_chain_id=chain
        )
        service.db.set_status(parent, "done", "parent")
        source = service.db.create_run(
            conversation_id="refusal-probe", agent_kind="subagent", parent_run_id=parent
        )
        service.db.set_status(source, "done", "saved result")
        ledger.claim_wake(
            chain,
            attempt_id="probe",
            owner_id=issuer,
            session_id="probe",
            run_ids=[source],
        )
        assert ledger.accept_wake("probe", owner_id=issuer)
        automatic = AutomaticWorkContext(ledger, chain, issuer, "probe")
        automatic.mark_accepted()

    def observe(frame, event, arg):
        if event == "return" and id(frame) in pending:
            pending.pop(id(frame))()
        if event != "call" or not threading.current_thread().name.startswith("tool-"):
            return
        if frame.f_code.co_name == "_runner" and frame.f_globals is vars(agent_service):
            workers.append(threading.current_thread())
        if (
            frame.f_code is threading.Event.wait.__code__
            and frame.f_back.f_code.co_name == "retain_unknown_return"
        ):
            parks.append(frame.f_locals["self"])
        if not active[0]:
            return
        if (
            boundary == "activation"
            and frame.f_code is admission_code
            and frame.f_locals["self"] is activation._guard
        ):
            pause = storage_admission._begin_local_pause()
            pending[id(frame)] = pause.resume
        if boundary == "automatic" and frame.f_code is check_code:
            context = frame.f_locals["self"]
            context._accepted.clear()
            pending[id(frame)] = context.mark_accepted
        if boundary == "goal_cancelled" and frame.f_code is check_code:
            coordinator = frame.f_locals["self"].goal._coordinator
            original_stop = coordinator._stop_requested
            coordinator._stop_requested = True
            pending[id(frame)] = lambda: setattr(
                coordinator, "_stop_requested", original_stop
            )

    def invoke():
        calls.append(True)
        return ToolResult(ok=True, content="original admitted callback")

    failure = None
    try:
        threading.setprofile(observe)
        with automatic.scope() if automatic is not None else nullcontext():
            refused = agent_service._call_with_timeout(
                invoke, 0.2, "native-refusal", owner=owner, execution_owner=service
            )
            active[0] = False
            retained = owner.capacity.snapshot().tool_workers
            unused = owner.reserve_tool()
            unused.finish()

        assert not refused.ok
        assert refused.dispatch_state == "not_started", refused
        assert refused.error == (
            "agent_activation_required"
            if boundary == "activation"
            else "cancelled"
            if boundary == "goal_cancelled"
            else "acceptance_required"
        ), refused
        if boundary in {"automatic", "goal_cancelled"}:
            from tldw_chatbook.Agents.agent_models import RunTerminationReason

            assert refused.termination_reason is (
                RunTerminationReason.CANCELLED
                if boundary == "goal_cancelled"
                else RunTerminationReason.AUTHORITY_CHANGED
            )
        assert retained == 0, "unused original operation remained counted"
        assert calls == [], "refused callback entered"

    except BaseException as error:
        failure = error
    finally:
        active[0] = False
        deadline = time.monotonic() + 5
        while (
            any(worker.is_alive() for worker in workers) and time.monotonic() < deadline
        ):
            for park in tuple(parks):
                park.set()
            for worker in workers:
                worker.join(0.01)
        assert all(not worker.is_alive() for worker in workers)
        threading.setprofile(old_profile)
        for release in pending.values():
            release()
        if failure is None:
            assert frozenset(owner._operations) == original_operations, (
                "probe changed pre-existing provider operations"
            )
    if failure is not None:
        import os
        import sys
        import traceback

        traceback.print_exception(failure, file=sys.__stderr__)
        sys.__stderr__.flush()
        os._exit(1)  # Private peer, after finite synthetic workers returned.


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["activation", "automatic"])
async def test_ordinary_original_tool_refusal_returns_unused_operation(
    ordinary_host, monkeypatch, boundary
):
    import os
    import sys
    import traceback
    from Tests.Chat.test_console_skill_substitution import _RecordingGateway
    from Tests.Agents.test_agent_service import fence

    if os.environ.get("TASK8_TOOL_REFUSAL_PEER") != boundary:
        from Tests.Backup_Recovery.test_home_citation_retirement import _run

        script = (
            _NESTED_PROFILE_SCRIPT
            + "os.environ['TASK8_TOOL_REFUSAL_PEER']="
            + repr(boundary)
            + "\n"
            "result=pytest.main(['-p','no:cacheprovider','-q','-s','-W','error::pytest.PytestUnraisableExceptionWarning','-W','error::pytest.PytestUnraisableExceptionWarning',"
            + repr(
                __file__
                + "::test_ordinary_original_tool_refusal_returns_unused_operation["
                + boundary
                + "]"
            )
            + "])\nassert result == 0\nprint('retired and reopened')\n"
        )
        await asyncio.to_thread(
            _run,
            ordinary_host.tmp_path,
            boundary,
            "ordinary",
            script=script,
            timeout=30,
        )
        return
    _assert_nested_profile()

    h = ordinary_host
    monkeypatch.setenv("TLDW_AGENTS_AUTOWAKE_ENABLED", "true")
    h.controller._confirm_project_instruction_dispatch = lambda notice: "proceed"
    checked, services = [], []

    class Gateway(_RecordingGateway):
        async def stream_chat(self, resolution, messages, **kwargs):
            self.payloads.append(messages)
            if len(self.payloads) == 1:
                service = h.bridge._fleet_services[h.session.persisted_conversation_id]
                services.append(service)
                exercise_tool_refusal(boundary, service, service._ordinary_tree.owner)
                checked.append(True)
                yield fence("calculator", {"expression": "2+2"})
            else:
                yield "parent finished"

    h.controller.provider_gateway = h.bridge._gateway = Gateway()
    try:
        result = await h.controller.submit_draft(
            "ordinary native tool", session_id=h.session.id
        )
        assert result.accepted and checked == [True]
        row = h.database.get_run(services[0]._ordinary_tree.owner.run_id)
        tools = [step for step in row["steps"] if step["kind"] == "tool_result"]
        assert len(tools) == 1 and tools[0]["tool_name"] == "calculator"
        assert json.loads(tools[0]["result"]) == {
            "expression": "2+2",
            "result": 4,
            "result_type": "int",
        }
        await until(lambda: not h.runtime.execution_capacity.snapshot().executions)
    except BaseException:
        traceback.print_exc(file=sys.__stderr__)
        sys.__stderr__.flush()
        os._exit(1)  # Isolated uncertainty; finite workers joined by helper.


@pytest.mark.parametrize("boundary", ["activation", "automatic"])
def test_planning_original_tool_refusal_returns_unused_operation(tmp_path, boundary):
    from Tests.Backup_Recovery.test_home_citation_retirement import _run

    probe = Path(__file__).with_name("workflow_planning_probe.py").read_text()
    original = "        calls.append(owner)"
    injected = (
        original
        + "\n        if len(calls) == 1:\n            exercise_tool_refusal(boundary, prepared.native.service, owner)"
    )
    assert probe.count(original) == 1
    script = (
        "import os,sys,traceback\n"
        "from Tests.Chat.test_native_admission_return import exercise_tool_refusal\n"
        "boundary=sys.argv[1]\nsys.argv[1]='tools'\n"
        "try:\n    exec("
        + repr(probe.replace(original, injected))
        + ", {'__name__':'__main__','exercise_tool_refusal':exercise_tool_refusal,'boundary':boundary})\n"
        "except BaseException:\n    traceback.print_exc(file=sys.__stderr__)\n    sys.__stderr__.flush()\n    os._exit(1)\n"
    )
    _run(tmp_path, boundary, "native", script=script, timeout=30)


def test_failed_predispatch_native_lease_close_still_blocks_recovery(tmp_path):
    from Tests.Backup_Recovery.test_activation_agents import _SCRIPT
    from Tests.Backup_Recovery.test_home_citation_retirement import _run

    script = (
        _SCRIPT.split("effects=[]")[0]
        + r"""
import threading
from types import SimpleNamespace
from contextlib import nullcontext
from tldw_chatbook.Agents.agent_service import _call_with_timeout
from tldw_chatbook.Agents.agent_models import ToolResult
from tldw_chatbook.Agents.execution_capacity import RuntimeCapacity, WorkOrigin
from tldw_chatbook.Backup_Recovery.admission import AdmissionTimeout
capacity = RuntimeCapacity()
owner = capacity.begin_execution(origin=WorkOrigin.MANUAL, conversation_id="no-effect")
service = SimpleNamespace(db=db, _ordinary_tree=SimpleNamespace(
    capacity=capacity, members={owner}, retain_source_scope=nullcontext,
    source=SimpleNamespace(activate=nullcontext), check_local_sources=lambda: None))
original_close = storage.StorageLease.close
failed, effects = [], []
def close(lease):
    if threading.current_thread().name.startswith("tool-") and not failed:
        failed.append(lease)
        raise OSError("failed native admission lease close")
    original_close(lease)
storage.StorageLease.close = close
try:
    result = _call_with_timeout(lambda: effects.append(True) or ToolResult(ok=True), .2, "lease-close", owner=owner, execution_owner=service)
    assert result.dispatch_state == "not_started" and not result.ok, result
    assert result.error == "failed native admission lease close", result
    assert effects == [] and len(failed) == 1
    assert capacity.snapshot().tool_workers == 0
    with storage._lock:
        assert failed[0] in storage._live_leases and failed[0]._key in storage._holds
    db.close()
    startup = storage._startups.pop((os.getpid(), str(root)), None)
    if startup is not None: startup.close()
    try:
        with authority.maintenance(('profile',), .04):
            raise AssertionError('failed admission close lost original Recovery hold')
    except AdmissionTimeout: pass
finally:
    storage.StorageLease.close = original_close
    for lease in failed: original_close(lease)
    owner.finish_root()
    db.close()
with authority.maintenance(('profile',), 2): pass
assert capacity.snapshot().executions == ()
assert not blocked_attempts()
print('retired and reopened')
"""
    )
    _run(tmp_path, "tool", "inactive", script=script, timeout=20)


@pytest.mark.parametrize(
    "failure",
    [
        "provider",
        "entry",
        "exit",
        "identical_exit",
        "finish",
        "retained",
        "retained_after",
        "unmarked",
        "guard_marked",
    ],
)
def test_typed_errors_never_certify_unreturned_tool_work(monkeypatch, failure):
    from contextlib import contextmanager
    from types import SimpleNamespace

    from tldw_chatbook.Agents.agent_models import ToolResult
    from tldw_chatbook.Agents.automatic_work_budget import _policy_refusal
    from tldw_chatbook.Agents.execution_capacity import (
        OwnedOperation,
        RuntimeCapacity,
        WorkOrigin,
    )

    capacity = RuntimeCapacity()
    owner = capacity.begin_execution(
        origin=WorkOrigin.MANUAL, conversation_id="negative"
    )
    error = _policy_refusal("model_call_budget")
    parks, workers, calls, entered = [], [], [], []

    @contextmanager
    def source():
        entered.append(True)
        if failure == "entry":
            raise error
        try:
            yield
        finally:
            if failure in {"exit", "identical_exit"}:
                raise error

    @contextmanager
    def retained():
        yield
        if failure in {"retained", "retained_after"}:
            raise error

    def event():
        value = threading.Event()
        parks.append(value)
        return value

    import sys

    previous_profile = sys.getprofile()
    assert previous_profile is None
    constructor_code = threading.Thread.__init__.__code__
    runner_code = next(
        code
        for code in agent_service._call_with_timeout.__code__.co_consts
        if inspect.iscode(code) and code.co_name == "_runner"
    )

    def observe_constructor(frame, event, arg):
        if event == "return" and frame.f_code is constructor_code:
            worker, target = frame.f_locals["self"], frame.f_locals["target"]
            if (
                worker.name == "tool-negative"
                and inspect.isfunction(target)
                and target.__code__ is runner_code
            ):
                workers.append(worker)

    sys.setprofile(observe_constructor)

    def invoke():
        calls.append(True)
        if failure in {"provider", "identical_exit"}:
            raise error
        return ToolResult(ok=True)

    original_finish = OwnedOperation.finish

    def finish(operation):
        if failure == "finish" and threading.current_thread().name.startswith("tool-"):
            raise error
        return original_finish(operation)

    if failure in {"retained", "unmarked", "guard_marked"}:

        @contextmanager
        def refused(*args, **kwargs):
            from tldw_chatbook.Agents.automatic_work_budget import AutomaticWorkRefused

            raise (
                error
                if failure == "guard_marked"
                else AutomaticWorkRefused("model_call_budget")
            )
            yield

        monkeypatch.setattr(activation, "execution", refused)
    monkeypatch.setattr(OwnedOperation, "finish", finish)
    monkeypatch.setattr(
        agent_service,
        "threading",
        SimpleNamespace(**(vars(threading) | {"Event": event})),
    )
    service = SimpleNamespace(
        _ordinary_tree=SimpleNamespace(
            capacity=capacity,
            members={owner},
            owner=owner,
            lock=threading.RLock(),
            check=lambda: None,
            source=SimpleNamespace(activate=source),
            retain_source_scope=retained,
            check_local_sources=lambda: None,
        )
    )
    try:
        result = agent_service._call_with_timeout(
            invoke, 0.2, "negative", owner=owner, execution_owner=service
        )
        if failure == "provider":
            assert (
                result.dispatch_state == "settled"
                and result.error == "model_call_budget"
            )
            assert result.termination_reason is None
            assert calls == entered == [True]
            assert capacity.snapshot().tool_workers == 0
        elif failure in {"unmarked", "guard_marked"}:
            assert (
                result.dispatch_state == "not_started"
                and result.error == "model_call_budget"
            )
            assert result.termination_reason is None and not entered and not calls
            assert capacity.snapshot().tool_workers == 0
        else:
            assert entered == ([] if failure == "retained" else [True])
            assert calls == (
                [True]
                if failure in {"exit", "identical_exit", "finish", "retained_after"}
                else []
            )
            assert result.dispatch_state == "uncertain", result
            assert (
                capacity.snapshot().tool_workers
                == capacity.snapshot().stopping_tool_workers
                == 1
            )
            again = agent_service._call_with_timeout(
                invoke, 0.2, "next", owner=owner, execution_owner=service
            )
            assert (
                again.dispatch_state == "not_started"
                and "previous_tool_still_running" in again.error
            )
            assert owner._cleanup_unproven
    finally:
        sys.setprofile(previous_profile)
        deadline = time.monotonic() + 5
        while (
            any(worker.is_alive() for worker in workers) and time.monotonic() < deadline
        ):
            for park in tuple(parks):
                park.set()
            for worker in workers:
                worker.join(0.01)
        assert all(not worker.is_alive() for worker in workers)
        for operation in tuple(owner._operations):
            original_finish(operation)
        owner.finish_root()


from Tests.Chat.test_goal_conversation_provisioning import stores as stores  # noqa: E402, PLC0414


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reason", ["cancelled", "iteration_wall_budget", "binding_changed"]
)
async def test_original_goal_policy_reasons_keep_typed_metadata(
    stores, monkeypatch, reason
):
    from Tests.Chat.test_console_goal_dispatch import build_goal_rig
    from Tests.Chat.test_automatic_provider_budget import response
    from tldw_chatbook.Agents.automatic_work_budget import AutomaticWorkRefused
    from tldw_chatbook.Agents.automatic_work_runtime import current_automatic_work
    from tldw_chatbook.Agents.agent_models import RunTerminationReason

    observed = []

    def provider(**kwargs):
        context = current_automatic_work()
        authority = context.goal
        coordinator = authority._coordinator
        original = (
            coordinator._stop_requested,
            authority.started,
            authority._source_root_identities,
        )
        try:
            if reason == "cancelled":
                coordinator._stop_requested = True
            elif reason == "iteration_wall_budget":
                authority.started = (
                    time.monotonic() - authority.goal.policy.iteration_wall_seconds - 1
                )
            else:
                authority._source_root_identities = (
                    (Path(authority.goal.request.binding.locator), None),
                )
            with pytest.raises(
                AutomaticWorkRefused, match="^" + reason + "$"
            ) as caught:
                context.check()
            error = caught.value
            assert vars(error).get("_policy_refusal") is True
            assert (
                error.termination_reason
                is {
                    "cancelled": RunTerminationReason.CANCELLED,
                    "iteration_wall_budget": RunTerminationReason.AUTOMATIC_LIMIT,
                    "binding_changed": RunTerminationReason.AUTHORITY_CHANGED,
                }[reason]
            )
            observed.append(reason)
        finally:
            (
                coordinator._stop_requested,
                authority.started,
                authority._source_root_identities,
            ) = original
        return response()

    goal, _store, _session, _controller, coordinator, gateway, _calls = build_goal_rig(
        stores, monkeypatch, provider
    )
    try:
        result = await coordinator.dispatch_once(goal.id)
        assert result.outcome.status == "done"
        assert observed == [reason]
    finally:
        await gateway.aclose()


@pytest.mark.parametrize("source", ["zero_limit", "task_source"])
def test_original_workflow_policy_issuers_mark_only_explicit_refusal(source):
    from tldw_chatbook.Agents.automatic_work_budget import (
        AutomaticWorkLimits,
        AutomaticWorkRefused,
    )
    from tldw_chatbook.Agents.workflow_task_scope import _refuse
    from tldw_chatbook.DB.automatic_work import AutomaticWorkLedger

    with pytest.raises(AutomaticWorkRefused) as caught:
        if source == "zero_limit":
            AutomaticWorkLedger._check_workflow_zero_limits(
                AutomaticWorkLimits(generations=0)
            )
        else:
            _refuse("source_changed")
    assert vars(caught.value).get("_policy_refusal") is True
    assert caught.value.reason == (
        "generation_budget"
        if source == "zero_limit"
        else "workflow_task_source_changed"
    )
    assert "_policy_refusal" not in vars(AutomaticWorkRefused(caught.value.reason))


@pytest.mark.parametrize("silent_close", [False, True])
def test_second_check_ledger_handle_returns_before_unused_finish(
    tmp_path, monkeypatch, silent_close
):
    from types import SimpleNamespace
    from Tests.Agents.test_automatic_child_scope import accepted_context
    from tldw_chatbook.Agents.execution_capacity import RuntimeCapacity, WorkOrigin
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
    import sqlite3

    monkeypatch.setenv("TLDW_AGENTS_AUTOWAKE_ENABLED", "true")
    db = AgentRunsDB(tmp_path / "ledger.sqlite", client_id="returned-check")
    automatic = accepted_context(db)
    capacity = RuntimeCapacity()
    owner = capacity.begin_execution(
        origin=WorkOrigin.AUTOMATIC, conversation_id="conversation"
    )
    original_close = db.close
    handles, workers, parks = [], [], []
    active = [True]
    check_code = AutomaticWorkContext.check.__wrapped__.__code__
    old_profile = threading.getprofile()

    def close():
        if silent_close and threading.current_thread().name.startswith("tool-"):
            return  # Real suppressed/failed close leaves its original slot/lease.
        original_close()

    def observe(frame, event, arg):
        if not threading.current_thread().name.startswith("tool-"):
            return
        if frame.f_code.co_name == "_runner" and frame.f_globals is vars(agent_service):
            if event == "call":
                workers.append(threading.current_thread())
            elif event == "return" and silent_close:
                original_close()  # Fixture-only native-handle cleanup after retention assertions.
        if event == "call" and frame.f_code is check_code and active[0]:
            active[0] = False
            automatic.ledger.pause(automatic.chain_id, "model_call_budget")
            handles.append(db._thread_local.conn)
        if (
            event == "call"
            and frame.f_code is threading.Event.wait.__code__
            and frame.f_back.f_code.co_name == "retain_unknown_return"
        ):
            parks.append(frame.f_locals["self"])

    monkeypatch.setattr(db, "close", close)
    try:
        threading.setprofile(observe)
        with automatic.scope():
            result = agent_service._call_with_timeout(
                lambda: pytest.fail("policy-refused provider entered"),
                0.2,
                "ledger-check",
                owner=owner,
                execution_owner=SimpleNamespace(db=db),
            )
        assert len(handles) == 1
        if silent_close:
            assert result.dispatch_state == "uncertain", result
            assert handles[0] in db._maintenance_participant.connections
            assert capacity.snapshot().tool_workers == 1
            assert owner._cleanup_unproven
        else:
            assert (
                result.dispatch_state == "not_started"
                and result.error == "model_call_budget"
            ), result
            with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
                sqlite3.Connection.in_transaction.__get__(handles[0])
            assert handles[0] not in db._maintenance_participant.connections
            assert capacity.snapshot().tool_workers == 0
    finally:
        deadline = time.monotonic() + 5
        while (
            any(worker.is_alive() for worker in workers) and time.monotonic() < deadline
        ):
            for park in tuple(parks):
                park.set()
            for worker in workers:
                worker.join(0.01)
        assert all(not worker.is_alive() for worker in workers)
        threading.setprofile(old_profile)
        for operation in tuple(owner._operations):
            operation.finish()  # Exact synthetic test owner, never a provider's existing operation.
        owner.finish_root()
        original_close()


def test_cancelled_refusal_waits_for_covering_return(monkeypatch):
    from contextlib import contextmanager
    from types import SimpleNamespace
    from tldw_chatbook.Agents import automatic_work_runtime
    from tldw_chatbook.Agents.automatic_work_budget import AutomaticWorkRefused
    from tldw_chatbook.Agents.execution_capacity import (
        RuntimeCapacity,
        WorkOrigin,
        OwnedOperation,
    )

    entered, release = threading.Event(), threading.Event()
    workers, finishes = [], []
    capacity = RuntimeCapacity()
    owner = capacity.begin_execution(
        origin=WorkOrigin.MANUAL, conversation_id="cover-return"
    )

    class Automatic:
        goal = None
        ledger = SimpleNamespace(_db=None)
        checks = 0

        def check(self):
            self.checks += 1
            if self.checks == 2:
                raise AutomaticWorkRefused("acceptance_required")

        def should_cancel(self):
            return False

        @contextmanager
        def scope(self):
            workers.append(threading.current_thread())
            yield
            entered.set()
            assert release.wait(5)

    original_finish = OwnedOperation.finish

    def finish(operation):
        finishes.append(operation)
        original_finish(operation)

    monkeypatch.setattr(OwnedOperation, "finish", finish)
    monkeypatch.setattr(
        automatic_work_runtime, "current_automatic_work", lambda: Automatic()
    )
    try:
        result = agent_service._call_with_timeout(
            lambda: pytest.fail("refused callback entered"),
            0.2,
            "held-cover",
            should_cancel=entered.is_set,
            owner=owner,
        )
        assert (
            entered.is_set()
            and result.dispatch_state == "uncertain"
            and "cancelled" in result.error
        )
        assert not finishes and capacity.snapshot().tool_workers == 1
    finally:
        release.set()
        for worker in workers:
            worker.join(5)
            assert not worker.is_alive()
        owner.finish_root()
    assert len(finishes) == 1 and capacity.snapshot().executions == ()


@pytest.mark.parametrize(
    "fault",
    [
        "none",
        "refund",
        "finish",
        "non_refundable",
        "terminal",
        "readback",
        "already_released",
    ],
)
def test_public_automatic_child_refusal_settles_once_and_refunds_charge(
    tmp_path, monkeypatch, fault
):
    from Tests.Agents.test_agent_service import FleetChat, fence
    from Tests.Agents.test_automatic_child_scope import accepted_context
    from tldw_chatbook.Agents.agent_models import AgentConfig, RunBudget
    from tldw_chatbook.Agents.agent_service import AgentService
    from tldw_chatbook.Agents.execution_capacity import RuntimeCapacity, WorkOrigin
    from tldw_chatbook.Agents.fleet_coordinator import FleetCoordinator
    from tldw_chatbook.Agents.tool_catalog import ToolCatalogRegistry
    from tldw_chatbook.Backup_Recovery import storage_admission
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    monkeypatch.setenv("TLDW_AGENTS_AUTOWAKE_ENABLED", "true")
    monkeypatch.setenv("TLDW_AGENTS_MAX_LIVE_SUBAGENTS", "2")
    db = AgentRunsDB(tmp_path / "charged.sqlite", client_id="charged-refusal")
    automatic = accepted_context(db)
    scripts = [fence("spawn_subagent", {"task": "first refused"})]
    if fault == "none":
        scripts.append(fence("spawn_subagent", {"task": "retry succeeds"}))
    scripts.append("parent completed")
    chat = FleetChat(
        scripts,
        {"retry succeeds": ["child complete"]} if fault == "none" else {},
        allow_unconsumed=False,
    )
    capacity = RuntimeCapacity()
    notifications, releases, finishes, children = [], [], [], []
    with automatic.scope():
        service = AgentService(
            db,
            ToolCatalogRegistry(),
            chat_call=chat,
            runtime_capacity=capacity,
            work_origin=WorkOrigin.AUTOMATIC,
            fleet_coordinator=FleetCoordinator(max_live=2, clock=time.monotonic),
            on_child_settled=lambda *args: notifications.append(args),
        )
    original_release = automatic.ledger.release
    original_finish = service._finish_execution
    terminal = service._set_terminal_status

    def terminal_write(run_id, *args, **kwargs):
        if fault == "terminal" and db.get_run(run_id)["agent_kind"] == "subagent":
            return False
        return terminal(run_id, *args, **kwargs)

    monkeypatch.setattr(service, "_set_terminal_status", terminal_write)

    def release(*args, **kwargs):
        releases.append(args[0])
        if fault == "refund":
            raise OSError("controlled refund return failed")
        if fault == "non_refundable":
            assert automatic.ledger.commit(args[0], owner_id=automatic.owner_id)
        if fault == "already_released":
            assert original_release(*args, **kwargs)
        return original_release(*args, **kwargs)

    def finish(owner):
        if owner.child:
            finishes.append(owner)
            if fault == "finish":
                raise OSError("controlled finish return failed")
        return original_finish(owner)

    monkeypatch.setattr(automatic.ledger, "release", release)
    monkeypatch.setattr(service, "_finish_execution", finish)
    admission_code = activation._guard.execution.__wrapped__.__code__
    pending, refused = {}, []
    import sys

    old_profile = threading.getprofile()
    old_main_profile = sys.getprofile()
    read_code = db.get_run.__func__.__code__

    def observe(frame, event, arg):
        if (
            fault == "readback"
            and event == "return"
            and frame.f_code is read_code
            and frame.f_back.f_code.co_name == "run_child"
        ):
            raise OSError("original child terminal readback failed")
        if event == "return" and id(frame) in pending:
            pending.pop(id(frame)).resume()
        if event == "call" and frame.f_code.co_name == "run_child_owned":
            children.append(threading.current_thread())
        if (
            event == "call"
            and frame.f_code is admission_code
            and frame.f_locals["self"] is activation._guard
            and threading.current_thread().name.startswith("fleet-")
            and not refused
        ):
            refused.append(True)
            pending[id(frame)] = storage_admission._begin_local_pause()

    try:
        threading.setprofile(observe)
        if fault == "readback":
            sys.setprofile(observe)
        run_id, outcome = service.run_turn(
            conversation_id="conversation",
            messages=[{"role": "user", "content": "go"}],
            config=AgentConfig(
                model="test",
                system_prompt="Test",
                allowed_tools=("spawn_subagent",),
                native_tools=False,
                budget=RunBudget(max_subagents=1),
            ),
            api_endpoint="llama_cpp",
        )
        for child in children:
            child.join(5)
            assert not child.is_alive()
        assert releases and releases.count(releases[0]) == 1, (
            "refused reservation refund was retried"
        )
        assert len(releases) == (2 if fault == "none" else 1)
        assert len(notifications) == (
            2
            if fault == "none"
            else 0
            if fault in {"refund", "non_refundable", "terminal", "readback"}
            else 1
        )
        assert outcome.subagents_spawned == (1 if fault == "none" else 0)
        results = [
            row
            for row in db.get_run(run_id)["steps"]
            if row["kind"] == "tool_result" and row["tool_name"] == "spawn_subagent"
        ]
        assert (
            "could not start" in results[0]["result"]
            and "running now" not in results[0]["result"]
        )
        snapshot = db.automatic_work.snapshot(automatic.chain_id)
        assert snapshot.used["child_launch"] == (
            1 if fault in {"none", "non_refundable"} else 0
        )
        assert snapshot.reserved["child_launch"] == (1 if fault == "refund" else 0)
        if fault == "none":
            assert len(results) == 2 and "running now" in results[1]["result"]
            assert len(finishes) == 2
            assert not capacity.snapshot().executions
        elif fault == "already_released":
            assert len(finishes) == 1 and not capacity.snapshot().executions
        else:
            assert "settlement failed" in results[0]["result"]
            child = next(owner for owner in capacity._owners.values() if owner.child)
            assert child._cleanup_unproven and not child._root_finished
            assert len(finishes) == (1 if fault == "finish" else 0)
    finally:
        for pause in pending.values():
            pause.resume()
        for child in children:
            child.join(5)
            assert not child.is_alive()
        threading.setprofile(old_profile)
        sys.setprofile(old_main_profile)
        db.close()


from Tests.Agents.test_native_legacy_hooks import (
    native_hooks as native_hooks,
    hook_file as hook_file,
)  # noqa: E402, PLC0414


def test_real_hook_capture_error_stays_generic_in_tool_result(
    native_hooks, monkeypatch
):
    from tldw_chatbook.Agents import automatic_work_runtime
    from tldw_chatbook.Agents.automatic_work_budget import AutomaticWorkRefused
    from tldw_chatbook.Agents.execution_capacity import RuntimeCapacity, WorkOrigin
    from tldw_chatbook.Agents.workflow_task_scope import _check_hooks
    from tldw_chatbook.Agents.legacy_hook_ownership import _NativeLegacyHookSource
    from types import SimpleNamespace

    h = native_hooks
    import toml

    raw = toml.loads(h.hook_file.read_text())
    raw["hooks"] = {"enabled": False, "hook": []}
    h.hook_file.write_text(toml.dumps(raw))
    code = _NativeLegacyHookSource.capture.__wrapped__.__code__
    captures = []

    def observe(frame, event, arg):
        if (
            event == "return"
            and frame.f_code is code
            and frame.f_locals["self"]._admissions == 0
        ):
            captures.append(True)
            raise RuntimeError("original capture return failed")

    import sys

    old_current_profile = sys.getprofile()
    try:
        sys.setprofile(observe)
        with pytest.raises(AutomaticWorkRefused) as caught:
            _check_hooks(h.runtime, h.session.id)
    finally:
        sys.setprofile(old_current_profile)
    translated = caught.value
    assert captures == [True]
    assert vars(translated).get("_policy_refusal") is not True

    class Automatic:
        goal = None
        ledger = SimpleNamespace(_db=None)
        checks = 0

        def check(self):
            self.checks += 1
            if self.checks == 2:
                raise translated

        def scope(self):
            return nullcontext()

        def should_cancel(self):
            return False

    capacity = RuntimeCapacity()
    owner = capacity.begin_execution(
        origin=WorkOrigin.AUTOMATIC, conversation_id="capture"
    )
    old_profile = threading.getprofile()
    try:
        monkeypatch.setattr(
            automatic_work_runtime, "current_automatic_work", lambda: Automatic()
        )
        result = agent_service._call_with_timeout(
            lambda: pytest.fail("capture-refused callback entered"),
            0.2,
            "capture",
            owner=owner,
        )
        assert captures == [True], result
        assert (
            result.error == "workflow_task_effectful_hooks_unsupported"
            and result.dispatch_state == "not_started"
        )
        assert result.termination_reason is None
        assert capacity.snapshot().tool_workers == 0
    finally:
        threading.setprofile(old_profile)
        owner.finish_root()


def test_child_failed_native_admission_lease_keeps_child_and_recovery_owners(tmp_path):
    from Tests.Backup_Recovery.test_activation_agents import _SCRIPT
    from Tests.Backup_Recovery.test_home_citation_retirement import _run

    script = (
        _SCRIPT.split("effects=[]")[0]
        + r"""
import threading,time
from Tests.Agents.test_agent_service import FleetChat,fence
from tldw_chatbook.Agents import activation as agents_activation
from tldw_chatbook.Agents.execution_capacity import RuntimeCapacity
from tldw_chatbook.Agents.fleet_coordinator import FleetCoordinator
from tldw_chatbook.Backup_Recovery.activation import _key
from tldw_chatbook.Backup_Recovery.admission import AdmissionTimeout
from tldw_chatbook.Tools import workspace_file_roots
approval = activation._generation('generation') / ('approved-'+_key('config')+'.json')
saved = approval.read_bytes()
failed,notifications,workers=[],[],[]
original_close=storage.StorageLease.close
admission_code=agents_activation._guard.execution.__wrapped__.__code__
pending=[]
def observe(frame,event,arg):
    if event=='return' and id(frame) in pending:
        pending.remove(id(frame)); approval.write_bytes(saved); approval.chmod(0o600)
    if event=='call' and frame.f_code.co_name=='run_child_owned': workers.append(threading.current_thread())
    if event=='call' and frame.f_code is original_close.__code__ and threading.current_thread().name.startswith('fleet-') and not failed:
        failed.append(frame.f_locals['self'])
        approval.write_bytes(saved); approval.chmod(0o600)
        raise OSError('child native admission lease close failed')
    if event=='call' and frame.f_code is admission_code and threading.current_thread().name.startswith('fleet-') and not pending and not failed:
        approval.unlink(); pending.append(id(frame))
capacity=RuntimeCapacity()
chat=FleetChat([fence('spawn_subagent',{'task':'refuse before MODEL'}),'parent completed'],{},allow_unconsumed=False)
service=AgentService(db,ToolCatalogRegistry(),chat_call=chat,runtime_capacity=capacity,fleet_coordinator=FleetCoordinator(max_live=2,clock=time.monotonic),on_child_settled=lambda *args:notifications.append(args))
threading.setprofile(observe)
try:
    run_id,outcome=service.run_turn(conversation_id='child-lease',messages=[{'role':'user','content':'spawn'}],config=AgentConfig(model='test',system_prompt='',allowed_tools=('spawn_subagent',),native_tools=False),api_endpoint='llama_cpp')
    for worker in workers: worker.join(5); assert not worker.is_alive()
    assert len(failed)==1 and outcome.subagents_spawned==0
    child_row=next(row for row in db.list_runs('child-lease') if row['agent_kind']=='subagent')
    assert child_row['status']=='error'
    assert service.registry.resolve_owner_for_name('fs_read') is None
    assert not service._agent_worktrees
    assert notifications==[(child_row['id'],'error')]
    assert failed[0] in storage._live_leases
    db.close()
    registry=workspace_file_roots._default_registry_instance
    if registry is not None: registry.db.close()
    startup=storage._startups.pop((os.getpid(),str(root)),None)
    if startup is not None: startup.close()
    try:
        with authority.maintenance(('profile',),.04): raise AssertionError('child failed admission lost Recovery gate')
    except AdmissionTimeout: pass
    children=[owner for owner in capacity._owners.values() if owner.child]
    assert len(children)==1 and children[0]._cleanup_unproven and not children[0]._root_finished, 'failed raw child lease falsely drained child owner'
finally:
    approval.write_bytes(saved); approval.chmod(0o600)
    threading.setprofile(None)
    storage.StorageLease.close=original_close
    for lease in failed: original_close(lease)
    db.close()
with authority.maintenance(('profile',),2): pass
print('retired and reopened')
"""
    )
    _run(tmp_path, "child-lease", "approved", script=script, timeout=20)


@pytest.mark.asyncio
async def test_genuine_goal_second_worker_check_preserves_exact_cancelled_reason(
    stores, monkeypatch
):
    from Tests.Chat.test_console_goal_dispatch import build_goal_rig
    from Tests.Chat.test_automatic_provider_budget import response
    from tldw_chatbook.Agents.automatic_work_runtime import current_automatic_work
    from tldw_chatbook.Agents.execution_capacity import current_execution_owner

    checked = []

    def provider(**kwargs):
        authority = current_automatic_work().goal
        bridge = authority._coordinator.controller._agent_bridge
        service = bridge._fleet_services[authority.conversation_id]
        exercise_tool_refusal("goal_cancelled", service, current_execution_owner())
        checked.append(True)
        return response()

    goal, _store, _session, _controller, coordinator, gateway, _calls = build_goal_rig(
        stores, monkeypatch, provider
    )
    try:
        result = await coordinator.dispatch_once(goal.id)
        assert result.outcome.status == "done" and checked == [True]
    finally:
        await gateway.aclose()


@pytest.mark.parametrize("marked", [False, True])
def test_first_goal_check_does_not_upgrade_unmarked_collaborator_error(
    monkeypatch, marked
):
    from types import SimpleNamespace
    from tldw_chatbook.Agents import automatic_work_runtime
    from tldw_chatbook.Agents.automatic_work_budget import (
        AutomaticWorkRefused,
        _policy_refusal,
    )
    from tldw_chatbook.Agents.agent_models import RunTerminationReason

    error = (
        _policy_refusal("cancelled") if marked else AutomaticWorkRefused("cancelled")
    )

    def check():
        raise error

    monkeypatch.setattr(
        automatic_work_runtime,
        "current_automatic_work",
        lambda: SimpleNamespace(goal=object(), check=check),
    )
    result = agent_service._call_with_timeout(
        lambda: pytest.fail("first check refused"), 0.2, "first-check"
    )
    assert result.dispatch_state == "not_started"
    assert result.termination_reason is (
        RunTerminationReason.CANCELLED
        if marked
        else RunTerminationReason.UNKNOWN_EFFECT
    )


@pytest.mark.parametrize(
    "decision",
    [
        "pause",
        "false",
        "replay",
        "opaque",
        "isolation",
        "borrowed",
        "changed_db",
        "changed_guard",
        "changed_isolation",
        "forwarding_isolation",
        "forwarding_execution",
    ],
)
def test_original_child_refusal_record_is_invocation_bound(tmp_path, decision):
    from Tests.Backup_Recovery.test_activation_agents import _SCRIPT
    from Tests.Backup_Recovery.test_home_citation_retirement import _run

    script = (
        _SCRIPT.split("effects=[]")[0]
        + r"""
import threading
from tldw_chatbook.Agents import activation as agents
from tldw_chatbook.Backup_Recovery.activation import _key
from tldw_chatbook.Backup_Recovery.admission_runtime import _AdmissionRefusalReturn, RecoveryAdmissionGuard
from tldw_chatbook.DB import base_db
service=AgentService(db,ToolCatalogRegistry(),chat_call=lambda **kw:None)
decision=route
approval=activation._generation('generation')/('approved-'+_key('config')+'.json')
saved=approval.read_bytes()
held,release=threading.Event(),threading.Event()
foreign=[]
def unrelated():
    lease=storage.acquire_storage(selector)
    foreign.append(lease);held.set()
    assert release.wait(10)
    lease.close()
worker=threading.Thread(target=unrelated)
worker.start();assert held.wait(5)
record=_AdmissionRefusalReturn()
body=[]
old_error=None
pause=None
profile=None
borrowed=None
original_inner=base_db._operation_owned_connection
original_guard=agents._guard.execution
original_isolation=agents._guard.worker_isolation
try:
    if decision in ('pause','replay'):
        pause=storage._begin_local_pause()
    if decision=='replay':
        try: storage.acquire_storage(selector,_refusal_nonce=object())
        except Exception as error: old_error=error
        assert old_error is not None
        pause.resume();pause=None
    if decision in ('false','isolation','borrowed'):
        approval.unlink()
    if decision=='borrowed':
        borrowed=db._held_connection()
    if decision in ('replay','opaque','isolation'):
        target=RecoveryAdmissionGuard.worker_isolation.__wrapped__.__code__ if decision=='isolation' else storage._Acquisition.__init__.__code__
        def profile(frame,event,arg):
            if frame.f_code is target and event==('return' if decision=='isolation' else 'call') and (decision!='isolation' or record.guard_returned):
                if decision=='replay': raise old_error
                raise OSError('original invocation opaque boundary')
        sys.setprofile(profile)
    forwarded=[]
    if decision in ('forwarding_isolation','forwarding_execution'):
        from contextlib import nullcontext
        approval.unlink()
        class Forwarder:
            def __init__(self,original): self.original=original
            def __getattr__(self,name): return getattr(self.original,name)
            def __call__(self,*args,**kwargs):
                forwarded.append(True)
                return nullcontext()
        if decision=='forwarding_isolation': agents._guard.worker_isolation=Forwarder(original_isolation)
        else: agents._guard.execution=Forwarder(original_guard)
    if decision=='changed_isolation':
        from contextlib import nullcontext
        approval.unlink()
        agents._guard.worker_isolation=lambda:nullcontext()
    if decision=='changed_guard':
        agents._guard.execution=lambda *args,**kwargs:None
    if decision=='changed_db':
        from contextlib import contextmanager
        @contextmanager
        def changed(database,**kwargs):
            kwargs['_entry_return'].stage='passive_returned'
            raise OSError('changed inner cannot publish')
            yield
        base_db._operation_owned_connection=changed
        entry=base_db._OperationEntryReturn(db)
        try:
            with base_db.operation_owned_connection(db,_entry_return=entry): pass
        except RuntimeError as error: assert 'original database entry record' in str(error)
        else: raise AssertionError('changed DB manager accepted')
        assert entry.error is None and not entry.bookkeeping_returned
    else:
        @agents.worker_guard(service,_refusal_return=record)
        def invoke(): body.append(True)
        try: invoke()
        except agents.AgentActivationRequired as error:
            if decision in ('pause','false','borrowed'):
                assert record.returned and record.guard_returned and record.error is error
                assert record.thread is threading.current_thread() and record.used
            else:
                assert not record.returned
        except OSError:
            assert decision=='isolation' and record.guard_returned and not record.returned
        else: raise AssertionError('refusal entered worker body')
        assert not body and agents._guard.context.get() is None
        assert not forwarded
        if borrowed is not None:
            assert db._thread_local.conn is borrowed and borrowed.execute('SELECT 1').fetchone()[0]==1
        if decision=='replay': assert not old_error._local_pause_decision[1]._pause_consumed
        if record.returned:
            try: invoke()
            except agents.AgentActivationRequired: pass
            else: raise AssertionError('consumed invocation reused')
    # Another worker's same-profile native hold does not veto this return.
    assert worker.is_alive() and foreign[0] in storage._live_leases
finally:
    sys.setprofile(None)
    base_db._operation_owned_connection=original_inner
    agents._guard.execution=original_guard
    agents._guard.worker_isolation=original_isolation
    if pause is not None: pause.resume()
    approval.write_bytes(saved);approval.chmod(0o600)
    db.close()
    release.set();worker.join(5);assert not worker.is_alive()
print('retired and reopened')
"""
    )
    _run(tmp_path, decision, "approved", script=script, timeout=20)


@pytest.mark.parametrize(
    "fault", ["success", "entry", "covering", "reset", "finish", "never_activated"]
)
def test_deferred_original_ssh_finish_follows_all_original_returns(monkeypatch, fault):
    from contextlib import contextmanager
    from contextvars import ContextVar
    from types import SimpleNamespace
    from tldw_chatbook.Agents.agent_models import ToolResult
    from tldw_chatbook.Agents.execution_capacity import (
        OwnedOperation,
        RuntimeCapacity,
        WorkOrigin,
    )
    from tldw_chatbook.Chat import console_ssh_custody as ssh

    capacity = RuntimeCapacity()
    owner = capacity.begin_execution(
        origin=WorkOrigin.MANUAL, conversation_id="ssh-return"
    )
    call = object.__new__(ssh._ModelSshCall)
    call.uncertain, call.finished, call.operation = False, False, None
    call.requests, call.master_launches, call.session_starts = [], [], []
    events, workers, attempts, calls = [], [], [], []
    context = ContextVar("test_original_ssh_return", default=None)

    class Current:
        get = context.get
        set = context.set

        def reset(self, token):
            context.reset(token)
            if fault == "reset":
                raise OSError("original SSH reset return failed")

    monkeypatch.setattr(ssh, "_current", Current())

    def check():
        if fault == "entry":
            raise OSError("original SSH check refused")

    call.check = check

    @contextmanager
    def source():
        yield
        if fault == "covering":
            raise OSError("covering source exit failed")

    @contextmanager
    def retained():
        yield

    def check_source():
        if fault == "never_activated":
            raise RuntimeError("original pre-source refusal")

    service = SimpleNamespace(
        _ordinary_tree=SimpleNamespace(
            capacity=capacity,
            members={owner},
            owner=owner,
            lock=threading.RLock(),
            check=lambda: None,
            source=SimpleNamespace(activate=source),
            retain_source_scope=retained,
            check_local_sources=check_source,
        )
    )
    original_finish = OwnedOperation.finish

    def finish(operation):
        attempts.append(operation)
        assert ssh.current_ssh_call() is None
        assert activation._guard.context.get() is None
        assert not call.finished and not call.uncertain
        assert all(
            r.proven
            for r in (*call.requests, *call.master_launches, *call.session_starts)
        )
        if fault == "finish":
            raise OSError("original operation finish return failed")
        original_finish(operation)

    def event():
        value = threading.Event()
        events.append(value)
        return value

    import sys

    previous_profile = sys.getprofile()
    assert previous_profile is None
    constructor_code = threading.Thread.__init__.__code__
    runner_code = next(
        code
        for code in agent_service._call_with_timeout.__code__.co_consts
        if inspect.iscode(code) and code.co_name == "_runner"
    )

    def observe_constructor(frame, event, arg):
        if event == "return" and frame.f_code is constructor_code:
            worker, target = frame.f_locals["self"], frame.f_locals["target"]
            if (
                worker.name == "tool-ssh-control"
                and inspect.isfunction(target)
                and target.__code__ is runner_code
            ):
                workers.append(worker)

    sys.setprofile(observe_constructor)

    monkeypatch.setattr(OwnedOperation, "finish", finish)
    monkeypatch.setattr(
        agent_service,
        "threading",
        SimpleNamespace(**(vars(threading) | {"Event": event})),
    )

    def invoke():
        calls.append(True)
        assert ssh.current_ssh_call() is call
        return ToolResult(ok=True)

    try:
        result = agent_service._call_with_timeout(
            invoke,
            0.2,
            "ssh-control",
            owner=owner,
            execution_owner=service,
            ssh_call=call,
        )
        assert calls == ([] if fault in {"entry", "never_activated"} else [True])
        if fault == "never_activated":
            assert not result.ok and not call.finished and call.operation is None
            assert len(attempts) == 1 and capacity.snapshot().tool_workers == 0
        elif fault == "success":
            assert result.ok and call.finished and len(attempts) == 1
            assert capacity.snapshot().tool_workers == 0
        else:
            assert result.dispatch_state == "uncertain" and not call.finished
            assert owner._cleanup_unproven and capacity.snapshot().tool_workers == 1
            assert len(attempts) == (1 if fault == "finish" else 0)
    finally:
        sys.setprofile(previous_profile)
        for event in events:
            event.set()
        for worker in workers:
            worker.join(5)
            assert not worker.is_alive()
        for operation in tuple(owner._operations):
            original_finish(operation)
        owner.finish_root()


def test_original_ssh_activate_default_finishes_under_same_current_call():
    from tldw_chatbook.Agents.execution_capacity import RuntimeCapacity, WorkOrigin
    from tldw_chatbook.Chat.console_ssh_custody import _ModelSshCall, current_ssh_call

    capacity = RuntimeCapacity()
    owner = capacity.begin_execution(
        origin=WorkOrigin.MANUAL, conversation_id="ssh-default"
    )
    operation = owner.reserve_tool()
    call = object.__new__(_ModelSshCall)
    call.uncertain, call.finished = False, False
    call.requests, call.master_launches, call.session_starts = [], [], []
    call.check = lambda: None
    finish, observed = operation.finish, []

    def same_finish():
        assert current_ssh_call() is call and not call.finished
        observed.append(operation)
        finish()

    operation.finish = same_finish
    try:
        with call.activate(operation):
            assert not observed and current_ssh_call() is call
        assert call.finished and observed == [operation]
        assert current_ssh_call() is None and capacity.snapshot().tool_workers == 0
    finally:
        owner.finish_root()


from Tests.Agents.test_workflow_child_launch import workflow_host as workflow_host  # noqa: E402, PLC0414
from Tests.Agents.test_workflow_task_scope import task_host as task_host  # noqa: E402, PLC0414


def test_actual_workflow_child_guard_refusal_returns_failed_receipt(workflow_host):
    from Tests.Agents.test_workflow_child_launch import (
        _prepared_service,
        _launch,
        _join,
    )
    from tldw_chatbook.Agents.agent_service import _WorkflowChildLaunchFailed
    from tldw_chatbook.Backup_Recovery import storage_admission as storage

    host = workflow_host
    notifications, pending = [], {}
    service, _ = _prepared_service(
        host,
        lambda **_: pytest.fail("refused child entered provider"),
        on_settled=lambda *args: notifications.append(args),
    )
    code = activation._guard.execution.__wrapped__.__code__
    previous = threading.getprofile()

    def observe(frame, event, arg):
        if event == "return" and id(frame) in pending:
            pending.pop(id(frame)).resume()
        if (
            event == "call"
            and frame.f_code is code
            and threading.current_thread().name.startswith("fleet-")
            and not notifications
            and not pending
        ):
            pending[id(frame)] = storage._begin_local_pause()

    try:
        threading.setprofile(observe)
        with pytest.raises(_WorkflowChildLaunchFailed) as refused:
            _launch(service, host)
        _join(host)
        receipt = refused.value.receipt
        assert receipt.failure == "setup", receipt
        assert host.db.get_run(receipt.run_id)["status"] == "error"
        assert notifications == [(receipt.run_id, "error")]
        assert service.runtime_capacity.snapshot().child_executions == 0
        snapshot = host.db.automatic_work.snapshot(host.authorization.claim.chain_id)
        assert snapshot.used["child_launch"] == snapshot.reserved["child_launch"] == 0
        assert receipt.owner._root_finished and not receipt.owner._cleanup_unproven
    finally:
        for pause in pending.values():
            pause.resume()
        threading.setprofile(previous)
        _join(host)


def test_actual_finished_child_resume_refusal_is_public_failure_without_spawn_charge(
    tmp_path,
):
    from Tests.Agents.test_fleet_runtime import make_fleet_service
    from Tests.Agents.test_fleet_continuation import (
        _run,
        _finished_child,
        _await_retained,
    )
    from Tests.Agents.test_agent_service import fence
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
    from tldw_chatbook.Backup_Recovery import storage_admission as storage

    db = AgentRunsDB(tmp_path / "resume.sqlite", client_id="resume-refusal")
    holder = {}
    service, chat, fleet = make_fleet_service(
        db,
        [
            fence("spawn_subagent", {"task": "original child"}),
            fence("wait_agents", {}),
            "first complete",
            lambda: fence(
                "send_to_agent", {"id": holder["handle"], "message": "resume refused"}
            ),
            "second complete",
        ],
        {"original child": ["child complete"]},
    )
    pending = {}
    previous = threading.getprofile()
    code = activation._guard.execution.__wrapped__.__code__

    def observe(frame, event, arg):
        if event == "return" and id(frame) in pending:
            pending.pop(id(frame)).resume()
        if (
            event == "call"
            and frame.f_code is code
            and threading.current_thread().name.startswith("fleet-")
            and not pending
        ):
            pending[id(frame)] = storage._begin_local_pause()

    try:
        _run(service)
        child = _finished_child(fleet)
        holder["handle"] = child.handle_id
        _await_retained(fleet, child.handle_id)
        threading.setprofile(observe)
        run_id, outcome = _run(service)
        assert outcome.subagents_spawned == 0
        rows = [
            row
            for row in db.list_runs("c", include_superseded=True)
            if row.get("resumed_from_run_id") == child.run_id
        ]
        assert len(rows) == 1 and rows[0]["status"] == "error"
        result = next(
            step
            for step in db.get_run(run_id)["steps"]
            if step["kind"] == "tool_result" and step["tool_name"] == "send_to_agent"
        )
        assert (
            "could not start" in result["result"]
            and "running now" not in result["result"]
        )
    finally:
        for pause in pending.values():
            pause.resume()
        threading.setprofile(previous)
        for thread in service._fleet_threads.values():
            thread.join(5)
            assert not thread.is_alive()
        db.close()


def test_tool_start_exception_after_target_entry_keeps_original_operation(monkeypatch):
    from tldw_chatbook.Agents.agent_models import ToolResult
    from tldw_chatbook.Agents.execution_capacity import RuntimeCapacity, WorkOrigin

    capacity = RuntimeCapacity()
    owner = capacity.begin_execution(
        origin=WorkOrigin.MANUAL, conversation_id="handoff"
    )
    entered, release = threading.Event(), threading.Event()
    workers, finishes = [], []
    original_start = threading.Thread.start

    def start(thread):
        original_start(thread)
        if thread.name == "tool-handoff":
            workers.append(thread)
            assert entered.wait(5)
            raise OSError("start returned error after target entered")

    def work():
        entered.set()
        assert release.wait(5)
        return ToolResult(ok=True)

    owner.on_drained(lambda result: finishes.append(result))
    monkeypatch.setattr(threading.Thread, "start", start)
    try:
        result = agent_service._call_with_timeout(work, 0.2, "handoff", owner=owner)
        assert result.dispatch_state == "uncertain"
        assert capacity.snapshot().tool_workers == 1 and not owner._cleanup_unproven
        assert not finishes
    finally:
        release.set()
        for worker in workers:
            worker.join(5)
            assert not worker.is_alive()
        owner.finish_root()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault",
    ["optional_db", "early_lookup", "late_child_finish", "anchor", "callback_db"],
)
async def test_original_native_child_and_finalization_boundaries(
    ordinary_host, monkeypatch, fault
):
    import os, sys, traceback
    from Tests.Agents.test_agent_service import fence
    from Tests.Chat.test_console_skill_substitution import _RecordingGateway
    from Tests.Backup_Recovery.test_home_citation_retirement import _run
    from tldw_chatbook.Agents.execution_capacity import (
        current_execution_owner,
        ExecutionOwner,
    )
    from tldw_chatbook.Agents.agent_models import ToolResult
    from tldw_chatbook.Chat.console_ordinary_tree import (
        current_ordinary_tree,
        _OrdinaryTurn,
    )
    from tldw_chatbook.Backup_Recovery import storage_admission

    if os.environ.get("TASK8_R2_NATIVE_PEER") != fault:
        script = (
            _NESTED_PROFILE_SCRIPT
            + "os.environ['TASK8_R2_NATIVE_PEER']="
            + repr(fault)
            + "\nresult=pytest.main(['-p','no:cacheprovider','-q','-s','-W','error::pytest.PytestUnraisableExceptionWarning',"
            + repr(
                __file__
                + "::test_original_native_child_and_finalization_boundaries["
                + fault
                + "]"
            )
            + "])\nassert result==0\nprint('retired and reopened')\n"
        )
        if fault == "optional_db":
            await asyncio.to_thread(
                _run,
                ordinary_host.tmp_path,
                fault,
                "success",
                script=script,
                timeout=25,
            )
        else:
            with pytest.raises(AssertionError, match="TASK8_R2_RETAINED_" + fault):
                await asyncio.to_thread(
                    _run,
                    ordinary_host.tmp_path,
                    fault,
                    "unknown",
                    script=script,
                    timeout=25,
                )
        return
    _assert_nested_profile()
    h = ordinary_host
    h.controller._confirm_project_instruction_dispatch = lambda notice: "proceed"
    monkeypatch.setenv("TLDW_AGENTS_MAX_LIVE_SUBAGENTS", "2")
    monkeypatch.setenv("TLDW_AGENTS_AUTOWAKE_ENABLED", "false")
    if fault == "optional_db":
        h.app.workspace_registry_service = None
    trees, lookups, host_calls, finishes, workers, pending = [], [], [], [], [], {}
    pause_returned = threading.Event()
    callback_db_opened = threading.Event()
    old_profile = threading.getprofile()
    current_code = threading.current_thread.__code__
    guard_code = activation._guard.execution.__wrapped__.__code__

    def observe(frame, event, arg):
        if event == "call" and frame.f_code.co_name == "run_child_owned":
            workers.append(threading.current_thread())
        if (
            fault == "early_lookup"
            and event == "call"
            and frame.f_code is current_code
            and frame.f_back.f_code.co_name == "run_child_owned"
            and not lookups
        ):
            lookups.append(True)
            raise OSError("original child early thread lookup failed")
        if fault == "late_child_finish":
            if event == "return" and id(frame) in pending:
                pending.pop(id(frame)).resume()
                pause_returned.set()
            if (
                event == "call"
                and frame.f_code is guard_code
                and threading.current_thread().name.startswith("fleet-")
                and not lookups
            ):
                lookups.append(True)
                pending[id(frame)] = storage_admission._begin_local_pause()
            if (
                event == "call"
                and frame.f_code.co_name == "_finish_execution"
                and frame.f_locals["owner"].child
            ):
                finishes.append(frame.f_locals["owner"])
            if (
                event == "call"
                and frame.f_code is _OrdinaryTurn.host_returned.__code__
                and frame.f_locals["owner"].child
            ):
                host_calls.append(frame.f_locals["owner"])
                raise OSError(
                    "late original child source returned before host publication"
                )

    reserve = ExecutionOwner.reserve_tool

    def observed_reserve(owner):
        operation = reserve(owner)
        finish = operation.finish

        def after_return():
            finish()
            if (
                fault == "callback_db"
                and threading.current_thread().name == "tool-r2-native"
            ):
                h.database.get_run(
                    operation.owner.run_id
                )  # Real new handle after work return.
                callback_db_opened.set()

        operation.finish = after_return
        return operation

    if fault == "callback_db":
        monkeypatch.setattr(ExecutionOwner, "reserve_tool", observed_reserve)

    class Gateway(_RecordingGateway):
        async def stream_chat(self, resolution, messages, **kwargs):
            if fault == "late_child_finish" and not trees:
                # Arm the original-site observer only once the real provider
                # is current, before its child dispatch, on existing threads too.
                assert threading.getprofile() is None and sys.getprofile() is None
                threading.setprofile_all_threads(observe)
            self.payloads.append(messages)
            tree, owner = current_ordinary_tree(), current_execution_owner()
            trees.append(tree)
            if len(self.payloads) == 1 and fault in {"anchor", "callback_db"}:
                try:
                    service = h.bridge._fleet_services[
                        h.session.persisted_conversation_id
                    ]
                    original_hosts = set(tree.primary_hosts)
                    store = h.runtime._chat_store
                    calls = []
                    if fault == "anchor":
                        h.runtime._chat_store = object()
                    try:
                        result = agent_service._call_with_timeout(
                            lambda: calls.append(True) or ToolResult(ok=True),
                            0.2,
                            "r2-native",
                            owner=owner,
                            execution_owner=service,
                        )
                    finally:
                        h.runtime._chat_store = store
                    assert result.dispatch_state == "uncertain"
                    if fault == "callback_db":
                        await until(callback_db_opened.is_set)
                    await until(lambda: owner._cleanup_unproven)
                    if fault == "anchor":
                        assert not calls and tree.primary_hosts == original_hosts
                        assert h.runtime.execution_capacity.snapshot().tool_workers == 1
                    else:
                        assert (
                            calls == [True]
                            and len(tree.primary_hosts - original_hosts) == 1
                        )
                        assert not tree.retired.done()
                    print("TASK8_R2_RETAINED_" + fault, file=sys.__stderr__, flush=True)
                    os._exit(1)
                except BaseException as error:
                    traceback.print_exception(error, file=sys.__stderr__)
                    sys.__stderr__.flush()
                    os._exit(1)
            if len(self.payloads) == 1:
                yield fence("spawn_subagent", {"task": "original native child"})
            else:
                yield "completed"

    h.controller.provider_gateway = h.bridge._gateway = Gateway()
    try:
        if fault == "late_child_finish":
            assert old_profile is None and sys.getprofile() is None
        else:
            threading.setprofile(observe)
        send = asyncio.create_task(
            h.controller.submit_draft("native boundary", session_id=h.session.id)
        )
        h.tasks.append(send)
        if fault == "optional_db":
            assert (await send).accepted
            assert any(None in tree.source.databases for tree in trees)
            await until(lambda: not h.runtime.execution_capacity.snapshot().executions)
            assert (
                len(h.gateway.payloads) == 0
            )  # The replacement owns actual provider calls.
            assert any(owner.child for owner in trees[0].members)
            assert trees[0].retired.done()
        else:

            def persisted_failure():
                for row in h.database.list_runs(h.session.persisted_conversation_id):
                    if row["agent_kind"] == "primary":
                        for step in h.database.get_run(row["id"])["steps"]:
                            if (
                                step["kind"] == "tool_result"
                                and step["tool_name"] == "spawn_subagent"
                            ):
                                assert (
                                    "could not start" in step["result"]
                                    and "running now" not in step["result"]
                                )
                                return True
                return False

            if fault == "late_child_finish":
                await until(lambda: bool(trees))
                await until(pause_returned.is_set)
            await until(persisted_failure)
            for worker in workers:
                worker.join(5)
                assert not worker.is_alive()
            tree = trees[0]
            owner, member = next((o, m) for o, m in tree.members.items() if o.child)
            assert (
                owner._cleanup_unproven
                and not owner._root_finished
                and not member.host_returned
            )
            if fault == "early_lookup":
                assert lookups == [True] and not host_calls
            else:
                assert finishes == host_calls == [owner]
            if fault == "late_child_finish":
                threading.setprofile_all_threads(old_profile)
            print("TASK8_R2_RETAINED_" + fault, file=sys.__stderr__, flush=True)
            os._exit(1)
    except BaseException as error:
        print(
            "TASK8_R2_STAGE",
            fault,
            len(trees),
            len(workers),
            len(lookups),
            len(finishes),
            len(host_calls),
            pause_returned.is_set(),
            file=sys.__stderr__,
            flush=True,
        )
        traceback.print_exception(error, file=sys.__stderr__)
        sys.__stderr__.flush()
        os._exit(1)
    finally:
        if fault == "late_child_finish":
            threading.setprofile_all_threads(old_profile)
        else:
            threading.setprofile(old_profile)
        for pause in pending.values():
            pause.resume()


from Tests.Agents.test_fleet_runtime import git_repo as git_repo  # noqa: E402, PLC0414


@pytest.mark.parametrize(
    "route_change",
    [
        "error",
        "none",
        "local",
        "family",
        "authority",
        "cache",
        "receiver",
        "partial",
        "partial_absent",
        "second_admit",
    ],
)
def test_refused_isolated_checkout_unroute_failure_retains_original_owner(
    git_repo, tmp_path, route_change
):
    import sys
    from Tests.Agents.test_agent_service import FleetChat, fence
    from Tests.Agents.test_fleet_runtime import _fs_local_provider, ISO_CFG
    from tldw_chatbook.Agents import agent_worktree
    from tldw_chatbook.Agents.local_tool_provider import RunAdmittedWorkspaceRoot
    from tldw_chatbook.Agents.execution_capacity import RuntimeCapacity
    from tldw_chatbook.Agents.fleet_coordinator import FleetCoordinator
    from tldw_chatbook.Agents.tool_catalog import ToolCatalogRegistry
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
    from tldw_chatbook.Backup_Recovery import storage_admission as storage

    db = AgentRunsDB(tmp_path / "route.sqlite", client_id="route-return")
    provider = _fs_local_provider(git_repo)
    authority = RunAdmittedWorkspaceRoot(
        workspace_id="workspace",
        binding_id="selected",
        alias="selected",
        root=git_repo,
        locator_fingerprint="f" * 64,
        root_identity=agent_worktree._worktree_root_identity(git_repo),
        allow_write=True,
        guard=lambda _: True,
    )
    registry = ToolCatalogRegistry()
    registry.register_provider(provider)
    notifications, calls, pending, workers, finishes = [], [], {}, [], []
    other = _fs_local_provider(git_repo)

    class OtherFamily:
        def list_catalog(self):
            return other.list_catalog()

    replacement = OtherFamily() if route_change == "family" else other
    service = agent_service.AgentService(
        db,
        registry,
        chat_call=FleetChat(
            [
                fence("spawn_subagent", {"task": "isolated", "isolation": "worktree"}),
                "parent done",
            ],
            {},
            allow_unconsumed=False,
        ),
        runtime_capacity=RuntimeCapacity(),
        fleet_coordinator=FleetCoordinator(max_live=2, clock=time.monotonic),
        worktree_repo_authority=authority,
        on_child_settled=lambda *args: notifications.append(args),
    )
    guard_code = activation._guard.execution.__wrapped__.__code__
    retire_code = provider.retire_run_workspace_root.__func__.__code__
    previous, previous_main = threading.getprofile(), sys.getprofile()

    def observe(frame, event, arg):
        if event == "return" and id(frame) in pending:
            pending.pop(id(frame)).resume()
        if event == "call" and frame.f_code.co_name == "run_child_owned":
            workers.append(threading.current_thread())
        if (
            event == "call"
            and frame.f_code is guard_code
            and threading.current_thread().name.startswith("fleet-")
            and not pending
        ):
            pending[id(frame)] = storage._begin_local_pause()
            assert provider._agent_roots
            if route_change in {"none", "local", "family"}:
                with registry._catalog_lock:
                    registry._providers[:] = (
                        [] if route_change == "none" else [replacement]
                    )
                registry.reset_catalog_cache()
                selected = registry.resolve_owner_for_name("fs_read")
                assert (
                    selected is None
                    if route_change == "none"
                    else selected[1] is replacement
                )
            elif route_change in {"authority", "cache"}:
                import dataclasses

                run_id, root = next(iter(provider._agent_roots.items()))
                with provider._agent_roots_lock:
                    if route_change == "authority":
                        provider._agent_roots[run_id] = dataclasses.replace(root)
                    else:
                        provider._path_specs_by_alias[root.alias] = dict(
                            provider._path_specs_by_alias[root.alias]
                        )
            elif route_change == "receiver":
                provider.retire_run_workspace_root = lambda *_args, **_kwargs: (
                    pytest.fail("changed receiver called")
                )
        if (
            event == "call"
            and frame.f_code.co_name == "_finish_execution"
            and frame.f_locals["owner"].child
        ):
            finishes.append(frame.f_locals["owner"])
        if (
            event == "return"
            and route_change == "second_admit"
            and frame.f_code is provider.admit_run_workspace_root.__func__.__code__
            and frame.f_locals["self"] is provider
        ):
            first = frame.f_locals["authority_specs"]
            assert (
                provider.admit_run_workspace_root(
                    frame.f_locals["run_id"], frame.f_locals["authority"]
                )
                is None
            )
            assert (
                provider._path_specs_by_alias[frame.f_locals["authority"].alias]
                is not first
            )
        if (
            event == "return"
            and frame.f_code is provider.admit_run_workspace_root.__func__.__code__
            and route_change in {"partial", "partial_absent"}
            and frame.f_locals["self"] is provider
        ):
            if route_change == "partial_absent":
                provider.retire_run_workspace_root(frame.f_locals["run_id"])
                assert frame.f_locals["run_id"] not in provider._agent_roots
            raise OSError(
                "original admission returned installed route before receipt capture"
            )
        if (
            event == "call"
            and frame.f_code is retire_code
            and frame.f_locals["self"] is provider
        ):
            calls.append(frame.f_locals["run_id"])
            if route_change == "error":
                raise OSError("original isolated checkout unroute failed")

    try:
        threading.setprofile(observe)
        sys.setprofile(observe)
        run_id, outcome = service.run_turn(
            conversation_id="route",
            messages=[{"role": "user", "content": "go"}],
            config=ISO_CFG,
            api_endpoint="llama_cpp",
        )
        assert outcome.subagents_spawned == 0
        rows = [row for row in db.list_runs("route") if row["agent_kind"] != "primary"]
        assert len(rows) == 1 and db.get_run(rows[0]["id"])["status"] == "error"
        child_id = rows[0]["id"]
        clean = route_change in {"none", "local", "family"}
        assert not other._agent_roots
        if clean:
            assert calls == [child_id] and child_id not in provider._agent_roots
            assert len(notifications) == len(finishes) == 1
        else:
            assert not notifications and not finishes
            child = next(
                owner
                for owner in service.runtime_capacity._owners.values()
                if owner.child
            )
            assert child._cleanup_unproven and not child._root_finished
            assert (child_id in provider._agent_roots) is (
                route_change != "partial_absent"
            )
        created = next(iter(service._agent_worktrees.values()))
        assert (
            created.worktree_path.is_dir()
            and (created.worktree_path / "seed.txt").read_text() == "base\n"
        )
        result = next(
            step
            for step in db.get_run(run_id)["steps"]
            if step["kind"] == "tool_result" and step["tool_name"] == "spawn_subagent"
        )
        assert (
            "worktree isolation refused [admit_failed]"
            if route_change in {"partial", "partial_absent", "second_admit"}
            else "could not start"
        ) in result["result"] and "running now" not in result["result"]
    finally:
        sys.setprofile(previous_main)
        threading.setprofile(previous)
        for pause in pending.values():
            pause.resume()
        for worker in workers:
            worker.join(5)
            assert not worker.is_alive()
        from tldw_chatbook.Agents.local_tool_provider import LocalToolProvider

        for run_id in tuple(provider._agent_roots):
            LocalToolProvider.retire_run_workspace_root(provider, run_id)
        for created in service._agent_worktrees.values():
            agent_worktree.discard_agent_worktree(git_repo, created)
        db.close()


def exercise_original_planning_registration_failure(prepared, owner):
    import os, sys
    from tldw_chatbook.Agents.agent_models import ToolResult
    from tldw_chatbook.Agents.automatic_work_runtime import current_automatic_work

    automatic = current_automatic_work()
    entered, release = threading.Event(), threading.Event()
    workers, attempts = [], []
    original = prepared._tool_workers

    class Registration(list):
        def append(self, worker):
            workers.append(worker)
            attempts.append(True)
            # Adversarial registration queues the same target then fails;
            # ordinary registration now precedes the original start call.
            worker.start()
            assert entered.wait(5)
            raise OSError("original planning worker registration failed")

    prepared._tool_workers = Registration(original)
    code = AutomaticWorkContext.check.__wrapped__.__code__
    previous = threading.getprofile()

    def observe(frame, event, arg):
        if (
            event == "call"
            and frame.f_code is code
            and threading.current_thread().name == "tool-r2-registration"
        ):
            entered.set()
            assert release.wait(5)
            automatic._accepted.clear()

    baseline = frozenset(owner._operations)
    try:
        threading.setprofile(observe)
        result = agent_service._call_with_timeout(
            lambda: ToolResult(ok=True),
            0.2,
            "r2-registration",
            owner=owner,
            execution_owner=prepared.native.service,
        )
        assert result.dispatch_state == "uncertain" and attempts == [True]
        assert owner._cleanup_unproven and len(owner._operations - baseline) == 1
    finally:
        release.set()
        for worker in workers:
            worker.join(5)
            assert not worker.is_alive()
        automatic._accepted.set()
        threading.setprofile(previous)
        prepared._tool_workers = original
    assert frozenset(owner._operations) == baseline
    print("TASK8_R2_REGISTRATION_CUSTODY_RETAINED", file=sys.__stderr__, flush=True)
    os._exit(1)  # Negative owner-custody oracle, never original root drain.


def test_original_planning_poststart_registration_failure_preserves_worker(tmp_path):
    from Tests.Backup_Recovery.test_home_citation_retirement import _run

    probe = Path(__file__).with_name("workflow_planning_probe.py").read_text()
    original = "        calls.append(owner)"
    injected = (
        original
        + "\n        if len(calls) == 1:\n            exercise_original_planning_registration_failure(prepared, owner)"
    )
    assert probe.count(original) == 1
    script = (
        "from Tests.Chat.test_native_admission_return import exercise_original_planning_registration_failure\n"
        + probe.replace(original, injected)
    )
    with pytest.raises(AssertionError, match="TASK8_R2_REGISTRATION_CUSTODY_RETAINED"):
        _run(tmp_path, "tools", "unknown", script=script, timeout=25)


def test_ambiguous_start_before_ident_retains_same_target_until_positive_return(
    monkeypatch,
):
    from tldw_chatbook.Agents.agent_models import ToolResult
    from tldw_chatbook.Agents.execution_capacity import RuntimeCapacity, WorkOrigin

    capacity = RuntimeCapacity()
    owner = capacity.begin_execution(
        origin=WorkOrigin.MANUAL, conversation_id="pre-entry"
    )
    release, started = threading.Event(), threading.Event()
    workers, calls, drains = [], [], []
    original_start = threading.Thread.start

    def queued_then_error(worker):
        if worker.name != "tool-pre-entry":
            return original_start(worker)
        workers.append(worker)
        # Adversarial startup: the original native Thread bootstrap is queued,
        # but it cannot publish ident or enter the target before caller return.
        with threading._active_limbo_lock:
            threading._limbo[worker] = worker

        def delayed_original_bootstrap():
            started.set()
            assert release.wait(5)
            worker._bootstrap()

        threading._start_new_thread(delayed_original_bootstrap, ())
        assert started.wait(5)
        assert worker.ident is None
        raise OSError("original target queued before startup result failed")

    monkeypatch.setattr(threading.Thread, "start", queued_then_error)
    owner.on_drained(lambda proof: drains.append(proof))
    try:
        result = agent_service._call_with_timeout(
            lambda: calls.append(True) or ToolResult(ok=True),
            0.2,
            "pre-entry",
            owner=owner,
        )
        assert result.dispatch_state == "uncertain" and not calls and not drains
        assert capacity.snapshot().tool_workers == 1
        (operation,) = owner._operations
        assert operation._unreturned_start_worker is workers[0]
        owner.finish_root()
        assert not drains and capacity.snapshot().tool_workers == 1
    finally:
        release.set()
        for worker in workers:
            assert worker._started.wait(5)
            worker.join(5)
            assert not worker.is_alive()
    assert calls == [True] and drains == [True]
    assert capacity.snapshot().executions == ()


def test_unqualified_tool_factory_never_receives_original_target(monkeypatch):
    from tldw_chatbook.Agents.agent_models import ToolResult
    from tldw_chatbook.Agents.execution_capacity import RuntimeCapacity, WorkOrigin

    capacity = RuntimeCapacity()
    owner = capacity.begin_execution(
        origin=WorkOrigin.MANUAL, conversation_id="factory"
    )
    original = threading.Thread
    factories, calls, workers = [], [], []
    release = threading.Event()

    def factory(*args, **kwargs):
        factories.append(True)
        worker = original(*args, **kwargs)
        workers.append(worker)
        worker.start()
        raise OSError("factory queued target before returning thread")

    def effect():
        calls.append(True)
        assert release.wait(5)
        return ToolResult(ok=True)

    monkeypatch.setattr(threading, "Thread", factory)
    try:
        result = agent_service._call_with_timeout(effect, 0.2, "factory", owner=owner)
        assert not factories and not workers and not calls
        assert result.dispatch_state == "not_started" and not result.ok
        assert capacity.snapshot().tool_workers == 0
    finally:
        release.set()
        for worker in workers:
            worker.join(5)
        owner.finish_root()


_PLANNING_HANDOFF = {}


def exercise_original_planning_ambiguous_handoff(prepared, owner, mode):
    import sys
    from tldw_chatbook.Agents.agent_models import ToolResult
    from tldw_chatbook.Agents.execution_capacity import ExecutionOwner

    release, finishing, returned, join_entered = (threading.Event() for _ in range(4))
    original_start = threading.Thread.start
    original_reserve = ExecutionOwner.reserve_tool
    baseline = frozenset(owner._operations)
    workers, finishes = [], []
    previous = threading.getprofile()
    assert previous is None and sys.getprofile() is None

    def reserve(receiver):
        operation = original_reserve(receiver)
        finish = operation.finish

        def held_finish():
            finish()
            if threading.current_thread().name == "tool-r3-handoff":
                assert operation not in receiver._operations
                finishes.append(operation)
                finishing.set()
                assert release.wait(15)
                returned.set()

        operation.finish = held_finish
        return operation

    def start(worker):
        if worker.name != "tool-r3-handoff":
            return original_start(worker)
        workers.append(worker)
        assert worker in prepared._tool_workers
        if mode == "preident":
            with threading._active_limbo_lock:
                threading._limbo[worker] = worker

            def bootstrap():
                assert release.wait(15)
                worker._bootstrap()

            threading._start_new_thread(bootstrap, ())
            assert worker.ident is None
        else:
            original_start(worker)
            assert finishing.wait(5)
        raise OSError("same original planning target handoff ambiguous")

    def observe(frame, event, arg):
        if (
            event == "call"
            and frame.f_code is threading.Thread.join.__code__
            and frame.f_locals.get("self") in workers
        ):
            join_entered.set()

    def release_after_original_join():
        assert join_entered.wait(15)
        assert finishing.is_set() and not returned.is_set()
        assert not prepared._native_closed and not prepared._planning_settled
        assert frozenset(owner._operations) == baseline
        assert workers[0].is_alive()
        release.set()

    ExecutionOwner.reserve_tool = reserve
    threading.Thread.start = start
    try:
        result = agent_service._call_with_timeout(
            lambda: ToolResult(ok=True),
            0.2,
            "r3-handoff",
            owner=owner,
            execution_owner=prepared.native.service,
        )
        assert result.dispatch_state == "uncertain" and len(workers) == 1
        assert workers[0] in prepared._tool_workers
        if mode == "held_finish":
            assert finishing.is_set() and not returned.is_set()
            assert frozenset(owner._operations) == baseline
        else:
            assert workers[0].ident is None and not finishing.is_set()
            assert len(owner._operations - baseline) == 1
    finally:
        ExecutionOwner.reserve_tool = original_reserve
        threading.Thread.start = original_start
    state = dict(
        prepared=prepared,
        owner=owner,
        workers=workers,
        release=release,
        returned=returned,
        finishes=finishes,
        baseline=baseline,
    )
    _PLANNING_HANDOFF.update(state)
    if mode == "held_finish":
        threading.setprofile_all_threads(observe)
        monitor = threading.Thread(target=release_after_original_join, daemon=True)
        monitor.start()
        _PLANNING_HANDOFF["monitor"] = monitor


def assert_original_planning_ambiguous_return(mode):
    import os, sys

    state = _PLANNING_HANDOFF
    (worker,) = state["workers"]
    if mode == "preident":
        assert worker.ident is None and not state["prepared"]._native_closed
        assert not state["prepared"]._planning_settled
        assert len(state["owner"]._operations - state["baseline"]) == 1
        state["release"].set()
        assert worker._started.wait(5)
        worker.join(5)
        assert not worker.is_alive()
        print("TASK8_R3_PREIDENT_CLOSE_RETAINED", file=sys.__stderr__, flush=True)
        os._exit(1)  # A refused original close, not a positive preparation drain.
    threading.setprofile_all_threads(None)
    state["monitor"].join(5)
    assert not state["monitor"].is_alive()
    assert not worker.is_alive() and state["returned"].is_set()
    assert len(state["finishes"]) == 1


@pytest.mark.parametrize("mode", ["held_finish", "preident"])
def test_original_planning_ambiguous_start_uses_existing_worker_witness(tmp_path, mode):
    from Tests.Backup_Recovery.test_home_citation_retirement import _run

    probe = Path(__file__).with_name("workflow_planning_probe.py").read_text()
    needle = "        calls.append(owner)"
    injected = (
        needle
        + "\n        if len(calls) == 1:\n            exercise_original_planning_ambiguous_handoff(prepared,owner,"
        + repr(mode)
        + ")"
    )
    assert probe.count(needle) == 1
    probe = probe.replace(needle, injected)
    needle = "        else:\n            result = await prepared.run_planning()\n        if mode in"
    replacement = (
        "        else:\n            try:\n                result = await prepared.run_planning()\n            except RuntimeError:\n                assert_original_planning_ambiguous_return("
        + repr(mode)
        + ")\n                raise\n            assert_original_planning_ambiguous_return("
        + repr(mode)
        + ")\n        if mode in"
    )
    assert probe.count(needle) == 1
    script = (
        "from Tests.Chat.test_native_admission_return import exercise_original_planning_ambiguous_handoff,assert_original_planning_ambiguous_return\n"
        + probe.replace(needle, replacement)
    )
    if mode == "preident":
        with pytest.raises(AssertionError, match="TASK8_R3_PREIDENT_CLOSE_RETAINED"):
            _run(tmp_path, "tools", "unknown", script=script, timeout=30)
    else:
        _run(tmp_path, "tools", "success", script=script, timeout=30)


@pytest.mark.parametrize(
    "state",
    [
        "default",
        "already_absent",
        "other_alias",
        "missing_root",
        "missing_cache",
        "root_none",
        "cache_none",
        "static_alias",
        "same_authority",
        "after_remove_readmit",
        "malformed",
    ],
)
def test_original_provider_route_return_compares_both_owned_maps(git_repo, state):
    from Tests.Agents.test_fleet_runtime import _fs_local_provider
    from tldw_chatbook.Agents.local_tool_provider import RunAdmittedWorkspaceRoot
    from tldw_chatbook.Agents.agent_worktree import _worktree_root_identity

    provider = _fs_local_provider(git_repo)
    authority = RunAdmittedWorkspaceRoot(
        workspace_id="workspace",
        binding_id="binding",
        alias="original",
        root=git_repo,
        locator_fingerprint="f" * 64,
        root_identity=_worktree_root_identity(git_repo),
        allow_write=True,
        guard=lambda _: True,
    )
    if state == "default":
        assert provider.admit_run_workspace_root("run", authority) is None
        with pytest.raises(ValueError):
            provider.admit_run_workspace_root("other", authority)
        provider.retire_run_workspace_root("run")
        assert (
            "run" not in provider._agent_roots
            and authority.alias not in provider._path_specs_by_alias
        )
        assert provider.retire_run_workspace_root("run") is None
        return
    specs = provider.admit_run_workspace_root("run", authority, _return_specs=True)
    assert (
        type(specs) is dict and provider._path_specs_by_alias[authority.alias] is specs
    )

    def retire():
        provider.retire_run_workspace_root(
            "run", _expected_authority=authority, _expected_specs=specs
        )

    if state in {"already_absent", "other_alias", "after_remove_readmit"}:
        retire()
        if state == "already_absent":
            retire()
            return
        provider.admit_run_workspace_root(
            "other" if state == "other_alias" else "run", authority
        )
    elif state == "same_authority":
        provider.admit_run_workspace_root("run", authority)
    elif state == "missing_root":
        del provider._agent_roots["run"]
    elif state == "missing_cache":
        del provider._path_specs_by_alias[authority.alias]
    elif state == "root_none":
        provider._agent_roots["run"] = None
    elif state == "cache_none":
        provider._path_specs_by_alias[authority.alias] = None
    elif state == "static_alias":
        provider._admitted_roots = {authority.alias: authority}
    roots, cache = dict(provider._agent_roots), dict(provider._path_specs_by_alias)
    if state == "malformed":
        with pytest.raises(RuntimeError):
            provider.retire_run_workspace_root("run", _expected_specs=specs)
    else:
        with pytest.raises(RuntimeError):
            retire()
    assert provider._agent_roots.keys() == roots.keys()
    assert provider._path_specs_by_alias.keys() == cache.keys()
    assert all(provider._agent_roots[key] is value for key, value in roots.items())
    assert all(
        provider._path_specs_by_alias[key] is value for key, value in cache.items()
    )


@pytest.mark.parametrize("fault", ["cancel_admit", "entered_replaced"])
def test_original_provider_route_physical_failure_retains_child(
    git_repo, tmp_path, fault
):
    import sys
    from Tests.Agents.conftest import join_fleet_children
    from Tests.Agents.test_agent_service import fence
    from Tests.Agents.test_automatic_child_scope import accepted_context, make_service
    from Tests.Agents.test_fleet_runtime import _fs_local_provider, ISO_CFG
    from tldw_chatbook.Agents import agent_worktree
    from tldw_chatbook.Agents.execution_capacity import RuntimeCapacity
    from tldw_chatbook.Agents.fleet_coordinator import FleetCoordinator
    from tldw_chatbook.Agents.local_tool_provider import RunAdmittedWorkspaceRoot
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    db = AgentRunsDB(tmp_path / "route.sqlite", client_id="route")
    context = accepted_context(db, child_launches=1)
    provider = _fs_local_provider(git_repo)
    authority = RunAdmittedWorkspaceRoot(
        workspace_id="workspace",
        binding_id="selected",
        alias="selected",
        root=git_repo,
        locator_fingerprint="f" * 64,
        root_identity=agent_worktree._worktree_root_identity(git_repo),
        allow_write=True,
        guard=lambda _: True,
    )
    fleet = FleetCoordinator(max_live=2, clock=time.monotonic)
    installed, attempts, replacements = [], [], []
    finishes, notices, launches = [], [], []
    cancelled = asyncio.CancelledError("original route admission interrupted")

    def child():
        (run_id, root, specs) = installed[0]
        assert provider._agent_roots[run_id] is root
        assert provider._path_specs_by_alias[root.alias] is specs
        provider.admit_run_workspace_root(run_id, root)
        replacements.append(provider._path_specs_by_alias[root.alias])
        return "completed in original checkout"

    service, chat = make_service(
        db,
        context,
        [
            fence("spawn_subagent", {"task": "isolated", "isolation": "worktree"}),
            fence("wait_agents", {}),
            "done",
        ],
        {"isolated": [child]},
        provider=provider,
        runtime_capacity=RuntimeCapacity(),
        fleet_coordinator=fleet,
        worktree_repo_authority=authority,
        on_child_settled=lambda *args: notices.append(args),
    )
    admit_code = provider.admit_run_workspace_root.__func__.__code__
    retire_code = provider.retire_run_workspace_root.__func__.__code__
    finish_code = service._finish_execution.__func__.__code__
    launch_code = service._launch_prepared_child_result.__func__.__code__

    # Keep the call observer active after the independent cancellation injection.
    monitoring = sys.monitoring
    tool_id = next(i for i in range(1, 6) if monitoring.get_tool(i) is None)
    returned = monitoring.events.PY_RETURN

    def admit_return(_code, _offset, result):
        locals_ = sys._getframe(1).f_locals
        if locals_["_return_specs"]:
            installed.append((locals_["run_id"], locals_["authority"], result))
            if fault == "cancel_admit":
                raise cancelled

    def observe(frame, event, arg):
        if event == "call":
            if frame.f_code is retire_code:
                attempts.append(frame.f_locals["_expected_specs"])
            elif frame.f_code is finish_code and frame.f_locals["owner"].child:
                finishes.append(frame.f_locals["owner"])
            elif frame.f_code is launch_code:
                launches.append(
                    (frame.f_locals["child_owner"], frame.f_locals["child_reservation"])
                )

    previous, main = threading.getprofile(), sys.getprofile()
    monitoring.use_tool_id(tool_id, "isolated-route-return")
    try:
        monitoring.register_callback(tool_id, returned, admit_return)
        monitoring.set_local_events(tool_id, admit_code, returned)
        threading.setprofile_all_threads(observe)

        def turn():
            return service.run_turn(
                conversation_id="conversation",
                messages=[{"role": "user", "content": "go"}],
                config=ISO_CFG,
                api_endpoint="llama_cpp",
            )

        if fault == "cancel_admit":
            with pytest.raises(asyncio.CancelledError) as interrupted:
                turn()
            assert interrupted.value is cancelled
        else:
            _, outcome = turn()
            assert outcome.status == "done"
        join_fleet_children(service)
        ((run_id, root, specs),) = installed
        ((owner, reservation),) = launches
        (handle,) = fleet.snapshot()
        assert root.root.is_dir() and (root.root / "seed.txt").read_text() == "base\n"
        assert provider._agent_roots[run_id] is root
        with context.ledger.transaction() as conn:
            reservation_state = context.ledger._reservation(
                conn, reservation, context.owner_id
            )["state"]
        if fault == "cancel_admit":
            assert handle.status == "running" and not chat.child_calls and not attempts
            assert provider._path_specs_by_alias[root.alias] is specs
            assert reservation_state == "reserved"
        else:
            assert handle.status == db.get_run(run_id)["status"] == "done"
            assert (
                handle.result
                == db.get_run(run_id)["result"]
                == "completed in original checkout"
            )
            assert reservation_state == "committed"
            assert len(attempts) == 1 and attempts[0] is specs
            assert (
                replacements[0] is not specs
                and provider._path_specs_by_alias[root.alias] is replacements[0]
            )
        assert not notices and not finishes
        assert service.runtime_capacity._owners[owner.execution_id] is owner
        assert owner._cleanup_unproven and not owner._root_finished
    finally:
        threading.setprofile_all_threads(previous)
        sys.setprofile(main)
        monitoring.set_local_events(tool_id, admit_code, 0)
        monitoring.register_callback(tool_id, returned, None)
        monitoring.free_tool_id(tool_id)
        join_fleet_children(service)
        for run_id in tuple(provider._agent_roots):
            provider.retire_run_workspace_root(run_id)
        for created in service._agent_worktrees.values():
            agent_worktree.discard_agent_worktree(git_repo, created)
        db.close()


def test_postcheck_thread_rebinding_never_exposes_original_tool_target(monkeypatch):
    import ast
    import sys
    from tldw_chatbook.Agents.agent_models import ToolResult
    from tldw_chatbook.Agents.execution_capacity import RuntimeCapacity, WorkOrigin

    capacity = RuntimeCapacity()
    owner = capacity.begin_execution(
        origin=WorkOrigin.MANUAL, conversation_id="postcheck"
    )
    caller = agent_service._call_with_timeout
    lines, start = inspect.getsourcelines(caller)
    (call_line,) = (
        start + call.lineno - 1
        for call in ast.walk(ast.parse("".join(lines)))
        if isinstance(call, ast.Call)
        and any(
            k.arg == "target"
            and isinstance(k.value, ast.Name)
            and k.value.id == "_runner"
            for k in call.keywords
        )
    )
    substitutions, targets, slots = [], [], []

    def factory(*args, **kwargs):
        targets.append(kwargs["target"])
        raise OSError("foreign factory retained target")

    def observe(frame, event, arg):
        if (
            frame.f_code is caller.__code__
            and event == "line"
            and frame.f_lineno == call_line
        ):
            monkeypatch.setattr(threading, "Thread", factory)
            substitutions.append(True)
        return observe

    def effect():
        slots.append(capacity.snapshot().tool_workers)
        return ToolResult(ok=True)

    previous = sys.gettrace()
    try:
        sys.settrace(observe)
        result = caller(effect, 1, "postcheck", owner=owner)
        assert substitutions and not targets
        assert result.ok and slots == [1]
        assert capacity.snapshot().tool_workers == 0
    finally:
        sys.settrace(previous)
        owner.finish_root()
