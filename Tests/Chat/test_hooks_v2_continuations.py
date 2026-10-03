"""Bounded Stop admission through the existing queue owner."""


def test_user_work_and_chain_cap_prevent_continuation():
    from tldw_chatbook.Agents.hooks_v2.continuations import ContinuationPolicy

    values = {
        "admitted_turns": 0,
        "elapsed_seconds": 0.0,
        "foreground_waiting": False,
        "revoked": False,
        "draining": False,
        "closed": False,
        "vetoed": False,
    }
    assert ContinuationPolicy.permits(**values)
    assert not ContinuationPolicy.permits(**{**values, "foreground_waiting": True})
    assert not ContinuationPolicy.permits(**{**values, "admitted_turns": 3})


import json
from uuid import uuid4

import pytest

pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.requires_cleanup]

from Tests.Agents.test_hooks_v2_execution import command
from Tests.Chat.test_console_fleet_wake import _controller_rig
from Tests.conftest import _close_database_instance
from tldw_chatbook.Agents.activation import worker_guard
from tldw_chatbook.Chat.console_runtime import ConsoleRuntime
from tldw_chatbook.Chat.console_turn_context import ConsoleTurnCustodyRequest


@pytest.mark.asyncio
@pytest.mark.parametrize("propose, expected", [(False, 1), (True, 4)])
@pytest.mark.parametrize("ephemeral", [False, True])
async def test_real_queue_stop_continuations_are_bounded(
    tmp_path, monkeypatch, propose, expected, ephemeral
):
    rig = _controller_rig(tmp_path)
    db, app, runs_db, store, session, gateway, _, controller = rig
    runtime = ConsoleRuntime(app=app)
    app.console_runtime = runtime
    runtime.set_chat_store(store)
    runtime.set_chat_controller(controller)
    if ephemeral:
        session = store.create_session(ephemeral=True)
    from tldw_chatbook.Agents.hooks_v2.continuations import ContinuationAdmission

    consumed = []
    original_consume = ContinuationAdmission.consume

    def consume_once(gate):
        consumed.append(gate.durable_acceptance_fingerprint()["gate_id"])
        original_consume(gate)

    monkeypatch.setattr(ContinuationAdmission, "consume", consume_once)
    result = (
        {"version": 2, "decision": "pass", "continuation": {"message": "follow up"}}
        if propose
        else {"version": 2, "decision": "pass"}
    )
    handler = command(
        f"print({json.dumps(result)!r})", name="Stop", effects=["continuation"]
    )
    runtime.ensure_hooks_v2(session.id, (handler,), lambda *_: True)
    request = ConsoleTurnCustodyRequest(
        turn_id=str(uuid4()),
        session_id=session.id,
        draft="hello",
        configuration=controller.resolve_turn_configuration_snapshot(session.id),
    )
    try:
        turn_id = runtime.accept_turn(request)
        result = await runtime.wait_for_turn(turn_id)
        assert result.accepted
        assert len(gateway.payloads) == expected
        assert len(consumed) == len(set(consumed)) == expected - 1
        assert controller.prompt_queue_registry.snapshot(session.id).total_count == 0
        assert db._connection_quiescence.connection_count() == 1
        import threading

        from tldw_chatbook.Backup_Recovery.participants import _repository_participant

        assert all(
            lease.resource_thread is threading.current_thread()
            for lease in _repository_participant(runs_db).connections.values()
        )
        if propose:
            assert "untrusted" in str(gateway.payloads[1]).lower()
    finally:
        await runtime.close_hooks_v2()
        await runtime.dispose()
        runs_db.close()
        _close_database_instance(db)


@pytest.mark.asyncio
@pytest.mark.parametrize("model_cap, exhausted", [(1, True), (3, False)])
async def test_actual_settled_agent_budget_is_retained(
    tmp_path, monkeypatch, model_cap, exhausted
):
    import asyncio
    from dataclasses import replace

    from Tests.Agents.test_fleet_runtime import FLEET_CFG, make_inline_service
    from tldw_chatbook.Agents.hooks_v2.budgets import HookBudgetOwner
    from tldw_chatbook.Agents.hooks_v2.engine import HookEngine
    from tldw_chatbook.Agents.hooks_v2.lifecycle import HookSessionLifecycle
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    db = AgentRunsDB(tmp_path / "budget.db", client_id="h5")
    engine = HookEngine((), lambda *_: True, HookBudgetOwner())
    lifecycle = HookSessionLifecycle(engine, "session")
    engine.lifecycle_owner = lifecycle
    service, _ = make_inline_service(db, ["done"], monkeypatch)
    service._hooks_v2_engine = engine
    service._hooks_v2_lifecycle = lifecycle
    service._hooks_v2_session_id = "session"
    service._hooks_v2_turn_id = "accepted-parent"
    try:
        _, outcome = await asyncio.to_thread(
            worker_guard(service)(service.run_turn),
            conversation_id="c",
            messages=[{"role": "user", "content": "go"}],
            config=replace(
                FLEET_CFG, budget=replace(FLEET_CFG.budget, max_model_turns=model_cap)
            ),
            api_endpoint="llama_cpp",
        )
        assert outcome.final_text == "done"
        assert "accepted-parent" in lifecycle.terminal_budgets
        remaining = lifecycle.terminal_budgets["accepted-parent"]
        assert (remaining is False) is exhausted
        if not exhausted:
            assert remaining.max_model_turns == 2
    finally:
        await engine.close()
        _close_database_instance(db)


@pytest.fixture
async def stop_case(tmp_path):
    import asyncio
    from types import SimpleNamespace

    db, app, runs_db, store, session, gateway, _, controller = _controller_rig(tmp_path)
    runtime = ConsoleRuntime(app=app)
    app.console_runtime = runtime
    runtime.set_chat_store(store)
    runtime.set_chat_controller(controller)
    entered = tmp_path / "entered"
    release = tmp_path / "release"
    events = tmp_path / "stops.jsonl"
    authority = [True]

    def install(results, *, hold=False, observe_prompt=False):
        handlers = []
        for index, result in enumerate(results):
            code = (
                "import json,sys,time;from pathlib import Path;event=json.load(sys.stdin);"
                f"p=Path({str(events)!r});p.write_text((p.read_text() if p.exists() else '')+json.dumps(event)+'\\n');"
                f"Path({str(entered)!r}).touch();"
            )
            if hold:
                code += f"exec({f'while not Path({str(release)!r}).exists(): time.sleep(0.005)'!r});"
            code += f"print({json.dumps(dict(version=2, decision='pass', **result))!r})"
            handlers.append(
                command(
                    code,
                    name="Stop",
                    id=f"stop-{index}",
                    effects=["continuation", "stop_continuations"],
                )
            )
        if observe_prompt:
            handlers.append(
                command(
                    f"from pathlib import Path;p=Path({str(tmp_path / 'prompts')!r});p.write_text((p.read_text() if p.exists() else '')+'prompt\\n');print('{{\"version\":2,\"decision\":\"pass\"}}')",
                    name="UserPromptSubmit",
                    id="human-prompt",
                )
            )
        return runtime.ensure_hooks_v2(
            session.id, tuple(handlers), lambda *_: authority[0]
        )

    def submit(text="hello"):
        request = ConsoleTurnCustodyRequest(
            turn_id=str(uuid4()),
            session_id=session.id,
            draft=text,
            configuration=controller.resolve_turn_configuration_snapshot(session.id),
        )
        turn = runtime.accept_turn(request)
        return request, asyncio.create_task(runtime.wait_for_turn(turn))

    async def wait_entered():
        for _ in range(2000):
            if entered.exists():
                return
            await asyncio.sleep(0.005)
        raise AssertionError("controlled Stop command did not start")

    case = SimpleNamespace(
        db=db,
        app=app,
        store=store,
        session=session,
        gateway=gateway,
        runtime=runtime,
        controller=controller,
        coordinator=controller.prompt_queue_coordinator,
        entered=entered,
        release=release,
        events=events,
        authority=authority,
        install=install,
        submit=submit,
        wait_entered=wait_entered,
    )
    yield case
    release.touch()
    await runtime.close_hooks_v2()
    await runtime.dispose()
    runs_db.close()
    _close_database_instance(db)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind",
    [
        "foreground",
        "deadline",
        "veto",
        "revoked",
        "closed",
        "oversize",
        "maintenance",
        "valid",
    ],
)
async def test_real_scheduler_failure_matrix_with_success_control(stop_case, kind):
    case = stop_case
    results = [
        {"continuation": {"message": "first"}},
        {"continuation": {"message": "second"}},
    ]
    if kind == "veto":
        results.append({"stop_continuations": True})
    if kind == "oversize":
        results = [{"continuation": {"message": "x" * 4096}}] * 2
    case.install(results, hold=True)
    _, pending = case.submit()
    await case.wait_entered()
    if kind == "foreground":
        snapshot = case.coordinator.registry.snapshot(case.session.id)
        assert (
            await case.controller.queue_prompt(
                case.session.id, text="human next", expected_revision=snapshot.revision
            )
        ).applied
    elif kind == "deadline":
        import time

        case.coordinator._chains[case.session.id].continuation_started = (
            time.monotonic() - 120
        )
    elif kind == "revoked":
        case.authority[0] = False
    elif kind == "closed":
        case.coordinator.pause_for_stop(case.session.id)
    elif kind == "maintenance":
        case.coordinator.maintenance_close_admission()
    case.release.touch()
    await pending
    expected = 4 if kind == "valid" else 5 if kind == "foreground" else 1
    # Foreground discards the original proposal; its own later settlement may
    # start a new bounded chain, which has fresh human priority and identity.
    assert len(case.gateway.payloads) == expected
    if kind == "foreground":
        assert case.gateway.payloads[1][-1]["content"] == "human next"
    if kind == "valid":
        assert "first\n\nsecond" in case.gateway.payloads[1][-1]["content"]
    assert case.coordinator.registry.snapshot(case.session.id).total_count == 0


@pytest.mark.asyncio
async def test_actual_stop_seals_a_still_running_stop_proposal(stop_case):
    import asyncio

    case = stop_case
    engine = case.install(
        [{"continuation": {"message": "delayed machine proposal"}}], hold=True
    )
    request, pending = case.submit()
    await case.wait_entered()
    assert len(case.gateway.payloads) == 1
    assert engine.processes.records
    assert not case.coordinator.stop_pending_continuation("unrelated-session")
    assert case.controller.stop_active_run()
    assert case.session.id in case.coordinator._sealed_continuations
    assert not case.controller.is_stop_allowed
    case.controller.stop_active_run()
    # Sealing does not abandon the still-running command's bounded custody.
    assert engine.processes.records
    case.release.touch()
    await pending
    assert len(case.gateway.payloads) == 1
    assert (
        case.db.get_connection()
        .execute("SELECT COUNT(*) FROM console_hook_continuation_receipts")
        .fetchone()[0]
        == 0
    )
    assert not case.coordinator._stop_parents
    assert not case.coordinator._stop_outcomes
    for _ in range(200):
        if not engine.cleanup_pending:
            break
        await asyncio.sleep(0.01)
    assert not engine.cleanup_pending
    assert not engine.processes.records
    assert not case.controller.stop_active_run()
    fresh, followup = case.submit("fresh ordinary turn")
    assert fresh.turn_id != request.turn_id
    await followup
    assert len(case.gateway.payloads) == 5
    assert (
        case.db.get_connection()
        .execute("SELECT COUNT(*) FROM console_hook_continuation_receipts")
        .fetchone()[0]
        == 3
    )


@pytest.mark.asyncio
async def test_simultaneous_callbacks_reserve_one_scheduler_identity(stop_case):
    import asyncio

    case = stop_case
    case.install([{"continuation": {"message": "one"}}])
    original = case.coordinator.schedule_continuation
    duplicate_results = []

    async def simultaneous(*args):
        results = await asyncio.gather(original(*args), original(*args))
        duplicate_results.append(results)
        return results[0]

    case.coordinator.schedule_continuation = simultaneous
    _, pending = case.submit()
    await pending
    assert len(case.gateway.payloads) == 4
    assert all(
        sum(result is not None for result in pair) <= 1 for pair in duplicate_results
    )
    assert (
        case.db.get_connection()
        .execute("SELECT COUNT(*) FROM console_hook_continuation_receipts")
        .fetchone()[0]
        == 3
    )


@pytest.mark.asyncio
async def test_lost_scheduler_response_never_returns_machine_work_to_queue(stop_case):
    case = stop_case
    case.install([{"continuation": {"message": "once"}}])
    submit = case.coordinator._submit_queued

    async def lost(*args, **kwargs):
        await submit(*args, **kwargs)
        raise RuntimeError("lost scheduler response")

    case.coordinator.bind_runtime_submitter(lost)
    _, pending = case.submit()
    with pytest.raises(RuntimeError, match="lost scheduler response"):
        await pending
    assert len(case.gateway.payloads) == 2
    assert not case.coordinator._machine_entries
    assert case.coordinator.registry.snapshot(case.session.id).total_count == 0
    assert (
        case.db.get_connection()
        .execute("SELECT COUNT(*) FROM console_hook_continuation_receipts")
        .fetchone()[0]
        == 1
    )


from Tests.Plugins.conftest import (
    native_console as native_console,  # noqa: PLC0414 -- pytest fixture re-export
)
from Tests.Plugins.conftest import (
    native_package as native_package,  # noqa: PLC0414 -- pytest fixture re-export
)


@pytest.mark.asyncio
@pytest.mark.parametrize("draining", [False, True])
async def test_reviewed_plugin_drain_blocks_actual_stop_scheduler(
    native_console, native_package, tmp_path, draining
):
    import asyncio

    case = native_console
    installed = await case.install()
    from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService

    case.controller.app.local_chat_conversation_service = ChatConversationService(
        case.store.persistence.db
    )
    runtime = ConsoleRuntime(app=case.controller.app)
    runtime.set_chat_store(case.store)
    runtime.set_chat_controller(case.controller)
    marker, release = tmp_path / "stop-ready", tmp_path / "stop-release"
    hook = command(
        "from pathlib import Path;import time;"
        f"Path({str(marker)!r}).touch();"
        f"exec({f'while not Path({str(release)!r}).exists(): time.sleep(.005)'!r});"
        'print(\'{"version":2,"decision":"pass","continuation":{"message":"follow"}}\')',
        name="Stop",
        effects=["continuation"],
    )
    runtime.ensure_hooks_v2(case.session.id, (hook,), lambda *_: True)
    maximum = case.service.capture_maximum("workspace-a")
    assert maximum["available_skills"]
    hold = await case.service.admit(maximum, "held-pending")
    await asyncio.to_thread(
        case.service.bind_run, hold["available_skills"], "held-root", lambda: None
    )
    request = ConsoleTurnCustodyRequest(
        turn_id=str(uuid4()),
        session_id=case.session.id,
        draft="hello",
        configuration=case.controller.resolve_turn_configuration_snapshot(
            case.session.id
        ),
    )
    turn = runtime.accept_turn(request)
    pending = asyncio.create_task(runtime.wait_for_turn(turn))
    apply = None
    try:
        for _ in range(400):
            if marker.exists():
                break
            await asyncio.sleep(0.005)
        assert marker.exists()
        if draining:
            root = native_package()
            skill = root / "skills/review/SKILL.md"
            skill.write_text(skill.read_text() + "\nReviewed update.\n")
            review = await case.service.review_revision(installed.installation_id, root)
            apply = asyncio.create_task(
                case.service.apply_revision(review, review.operation_id)
            )
            for _ in range(200):
                if case.service.revision_drain.blockers(review.drain_token):
                    break
                await asyncio.sleep(0.005)
            assert case.service.revision_drain.blockers(review.drain_token)
            assert not apply.done()
        release.touch()
        await pending
        assert len(case.gateway.payloads) == (1 if draining else 4)
        connection = case.store.persistence.db.get_connection()
        assert connection.execute(
            "SELECT COUNT(*) FROM console_hook_continuation_receipts"
        ).fetchone()[0] == (0 if draining else 3)
    finally:
        release.touch()
        await asyncio.to_thread(case.service.complete_run, "held-root")
        await case.service.retire_pending("held-pending")
        if apply is not None:
            await apply
        await runtime.close_hooks_v2()


@pytest.mark.asyncio
async def test_machine_input_keeps_host_carrier_through_actual_preparation(
    stop_case, monkeypatch
):
    from tldw_chatbook.Agents import agent_models

    observed = []
    original = agent_models.check_host_context

    def check(messages, **kwargs):
        observed.extend(
            row["content"]
            for row in messages
            if isinstance(row.get("content"), agent_models.PluginContextText)
            and row["content"].checked_hook_origins()
        )
        return original(messages, **kwargs)

    monkeypatch.setattr(agent_models, "check_host_context", check)
    stop_case.install([{"continuation": {"message": "whole contributed block"}}])
    _, pending = stop_case.submit()
    await pending
    assert len(stop_case.gateway.payloads) == 4
    assert observed
    assert all(value.checked_origins() == () for value in observed)
    assert all(
        value.checked_hook_origins()[0].byte_count == len("whole contributed block")
        for value in observed
    )


@pytest.mark.asyncio
async def test_human_arrival_after_claim_discards_machine_and_drains_human(stop_case):
    case = stop_case
    case.install([{"continuation": {"message": "follow"}}])
    submit = case.coordinator._submit_queued
    raced = False

    async def racing(prompt, **kwargs):
        nonlocal raced
        if not raced:
            raced = True
            snapshot = case.coordinator.registry.snapshot(case.session.id)
            assert (
                await case.controller.queue_prompt(
                    case.session.id,
                    text="human wins",
                    expected_revision=snapshot.revision,
                )
            ).applied
        return await submit(prompt, **kwargs)

    case.coordinator.bind_runtime_submitter(racing)
    _, pending = case.submit()
    await pending
    assert len(case.gateway.payloads) == 5
    assert "human wins" in str(case.gateway.payloads[1])
    assert case.coordinator.registry.snapshot(case.session.id).total_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "interruption", ["human", "stop", "revoked", "context", "maintenance"]
)
async def test_human_arrival_while_acceptance_worker_waits(stop_case, interruption):
    import asyncio
    import threading

    case = stop_case
    case.install([{"continuation": {"message": "superseded machine"}}])
    entered, release = threading.Event(), threading.Event()
    commit = case.store.commit_durable_turn
    once = False

    def held(acceptance):
        nonlocal once
        if acceptance.continuation_receipt is not None and not once:
            once = True
            entered.set()
            assert release.wait(30)
        return commit(acceptance)

    case.store.commit_durable_turn = held
    _, pending = case.submit()
    try:
        assert await asyncio.to_thread(entered.wait, 30)
        if interruption in {"human", "context"}:
            snapshot = case.coordinator.registry.snapshot(case.session.id)
            assert (
                await case.controller.queue_prompt(
                    case.session.id,
                    text="worker race human",
                    expected_revision=snapshot.revision,
                )
            ).applied
            if interruption == "context":
                from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole

                preparation = case.store.preparation_for_session(case.session.id)
                case.store.create_sibling(
                    preparation.transient_user_message_id,
                    role=ConsoleMessageRole.USER,
                    content="unrelated branch",
                    persist=False,
                )
        elif interruption == "stop":
            assert case.controller.stop_active_run()
        elif interruption == "maintenance":
            case.coordinator.maintenance_close_admission()
        else:
            case.authority[0] = False
        release.set()
        await pending
        assert len(case.gateway.payloads) == (5 if interruption == "human" else 1)
        if interruption == "human":
            assert "worker race human" in str(case.gateway.payloads[1])
        assert case.db.get_connection().execute(
            "SELECT COUNT(*) FROM console_hook_continuation_receipts"
        ).fetchone()[0] == (3 if interruption == "human" else 0)
        assert case.db.get_connection().execute(
            "SELECT COUNT(*) FROM messages WHERE role='user'"
        ).fetchone()[0] == (5 if interruption == "human" else 1)
        if interruption == "context":
            from tldw_chatbook.Chat.console_prompt_queue import PromptQueuePauseReason

            assert (
                case.coordinator.registry.snapshot(case.session.id).pause_reason
                is PromptQueuePauseReason.CONTEXT_CHANGED
            )
    finally:
        release.set()
        if not pending.done():
            await pending


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_cap, expected, wall_hold, duplicate",
    [
        (1, 1, False, False),
        (3, 3, False, False),
        (3, 1, True, False),
        (1, 1, False, True),
        (3, 3, False, True),
    ],
)
async def test_actual_agent_remaining_budget_controls_real_scheduler(
    stop_case, tmp_path, monkeypatch, model_cap, expected, wall_hold, duplicate
):
    import asyncio
    from dataclasses import replace

    from Tests.Agents.test_fleet_runtime import FLEET_CFG, make_inline_service
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    case = stop_case
    engine = case.install(
        [{"continuation": {"message": "continue budget"}}], hold=wall_hold
    )
    duplicate_results = []
    if duplicate:
        schedule = case.coordinator.schedule_continuation

        async def simultaneous(*args):
            results = await asyncio.gather(schedule(*args), schedule(*args))
            duplicate_results.append(results)
            return results[0]

        case.coordinator.schedule_continuation = simultaneous
    db = AgentRunsDB(tmp_path / "real-agent-budget.db", client_id="h5")
    service, _ = make_inline_service(db, ["done"] * 5, monkeypatch)
    calls = []

    async def agent_stream(resolution, messages, **kwargs):
        lifecycle = engine.lifecycle_owner
        service._hooks_v2_engine = engine
        service._hooks_v2_lifecycle = lifecycle
        service._hooks_v2_session_id = case.session.id
        service._hooks_v2_turn_id = case.coordinator._chains[
            case.session.id
        ].request.turn_id
        calls.append(service._hooks_v2_turn_id)
        _, outcome = await asyncio.to_thread(
            worker_guard(service)(service.run_turn),
            conversation_id=case.session.id,
            messages=messages,
            config=replace(
                FLEET_CFG,
                budget=replace(
                    FLEET_CFG.budget,
                    max_model_turns=model_cap,
                    max_wall_seconds=(
                        5.0 if wall_hold else FLEET_CFG.budget.max_wall_seconds
                    ),
                ),
            ),
            api_endpoint="llama_cpp",
        )
        yield outcome.final_text

    case.gateway.stream_chat = agent_stream
    try:
        _, pending = case.submit()
        if wall_hold:
            await case.wait_entered()
            import time

            lifecycle = engine.lifecycle_owner
            remaining = lifecycle.terminal_budgets[calls[0]]
            assert remaining is not False, "the real parent must complete before expiry"
            deadline = (
                lifecycle.terminal_budget_times[calls[0]] + remaining.max_wall_seconds
            )
            await asyncio.sleep(max(0.0, deadline - time.monotonic()) + 0.05)
            case.release.touch()
        await pending
        assert len(calls) == expected
        if duplicate:
            assert len(duplicate_results) == expected
            assert (
                sum(value is not None for pair in duplicate_results for value in pair)
                == expected - 1
            )
        assert not engine.lifecycle_owner.terminal_budgets
        assert not engine.lifecycle_owner.terminal_budget_times
        assert not engine.lifecycle_owner.inherited_budgets
        assert not case.coordinator._stop_parents
        assert not case.coordinator._stop_outcomes
        assert case.coordinator.registry.snapshot(case.session.id).total_count == 0
        assert (
            case.db.get_connection()
            .execute("SELECT COUNT(*) FROM console_hook_continuation_receipts")
            .fetchone()[0]
            == expected - 1
        )
    finally:
        _close_database_instance(db)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [False, True, "stop"])
async def test_consumption_before_foreground_or_uncertain_failure_is_not_reopened(
    stop_case, monkeypatch, failure
):
    import asyncio
    import threading

    from tldw_chatbook.Agents.hooks_v2.continuations import ContinuationAdmission

    case = stop_case
    case.install([{"continuation": {"message": "accepted machine"}}])
    entered, release = threading.Event(), threading.Event()
    write = ContinuationAdmission.write
    captured = []

    def held(gate, **kwargs):
        write(gate, **kwargs)
        if not captured:
            captured.append(gate)
            entered.set()
            assert release.wait(5)
            if failure is True:
                raise RuntimeError("controlled uncertain acceptance failure")

    monkeypatch.setattr(ContinuationAdmission, "write", held)
    _, pending = case.submit()
    try:
        for _ in range(600):
            if entered.is_set():
                break
            await asyncio.sleep(0.005)
        assert entered.is_set()
        if not failure:
            snapshot = case.coordinator.registry.snapshot(case.session.id)
            assert (
                await case.controller.queue_prompt(
                    case.session.id,
                    text="later human",
                    expected_revision=snapshot.revision,
                )
            ).applied
        if failure == "stop":
            assert case.controller.stop_active_run()
        release.set()
        await pending
        assert len(case.gateway.payloads) == (1 if failure else 6)
        if not failure:
            assert "accepted machine" in str(case.gateway.payloads[1])
            assert "later human" in str(case.gateway.payloads[2])
        assert case.db.get_connection().execute(
            "SELECT COUNT(*) FROM console_hook_continuation_receipts"
        ).fetchone()[0] == (0 if failure is True else 1 if failure == "stop" else 4)
        with pytest.raises(PermissionError):
            captured[0].consume()
        assert not case.coordinator._machine_entries
        assert case.coordinator.registry.snapshot(case.session.id).total_count == 0
    finally:
        release.set()
        if not pending.done():
            await pending


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["values", "revision", "source"])
async def test_first_save_handoff_refuses_changed_authoritative_identity(
    native_console, tmp_path, change
):
    from dataclasses import replace

    from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService
    from tldw_chatbook.Chat.console_library_policy import ConsoleAssistantLibraryAccess

    case = native_console
    await case.install()
    case.controller.app.local_chat_conversation_service = ChatConversationService(
        case.store.persistence.db
    )
    runtime = ConsoleRuntime(app=case.controller.app)
    runtime.set_chat_store(case.store)
    runtime.set_chat_controller(case.controller)
    hook = command(
        "print("
        + repr(
            json.dumps(
                {
                    "version": 2,
                    "decision": "pass",
                    "continuation": {"message": "follow"},
                }
            )
        )
        + ")",
        name="Stop",
        effects=["continuation"],
    )
    runtime.ensure_hooks_v2(case.session.id, (hook,), lambda *_: True)
    coordinator = case.controller.prompt_queue_coordinator
    capture = coordinator.capture_hook_parent
    seen = []

    def changed(*args, accepted_policy=None, **kwargs):
        assert (
            accepted_policy.source == "durable" and accepted_policy.policy_revision == 1
        )
        seen.append(accepted_policy)
        holder = case.session.library_policy_holder
        original = holder.snapshot
        altered = replace(
            original,
            **(
                {
                    "assistant_access": (
                        ConsoleAssistantLibraryAccess.BLOCKED
                        if original.assistant_access
                        is ConsoleAssistantLibraryAccess.ALLOWED
                        else ConsoleAssistantLibraryAccess.ALLOWED
                    )
                }
                if change == "values"
                else (
                    {"policy_revision": 2}
                    if change == "revision"
                    else {"source": "unavailable"}
                )
            ),
        )
        holder.snapshot = altered
        try:
            capture(*args, accepted_policy=holder.snapshot, **kwargs)
        finally:
            holder.snapshot = original

    coordinator.capture_hook_parent = changed
    try:
        request = ConsoleTurnCustodyRequest(
            turn_id=str(uuid4()),
            session_id=case.session.id,
            draft="first save",
            configuration=case.controller.resolve_turn_configuration_snapshot(
                case.session.id
            ),
        )
        assert request.configuration.library_policy_maximum.source == "new_session"
        assert (await runtime.wait_for_turn(runtime.accept_turn(request))).accepted
        assert seen
        assert len(case.gateway.payloads) == 1
        assert (
            case.store.persistence.db.get_connection()
            .execute("SELECT COUNT(*) FROM console_hook_continuation_receipts")
            .fetchone()[0]
            == 0
        )
    finally:
        await runtime.close_hooks_v2()


@pytest.mark.asyncio
async def test_machine_lineage_stays_out_of_human_hook_and_history(stop_case):
    from Tests.Chat.test_console_prompt_queue_coordinator import RecordingPromptHistory

    case = stop_case
    history = RecordingPromptHistory()
    case.controller.prompt_history = history
    case.install([{"continuation": {"message": "x" * 4096}}], observe_prompt=True)
    _, pending = case.submit()
    await pending
    assert len(case.gateway.payloads) == 4
    assert (case.entered.parent / "prompts").read_text().splitlines() == ["prompt"]
    assert history.items == ["hello"]
    events = [json.loads(row) for row in case.events.read_text().splitlines()]
    assert [event["initiator"] for event in events] == [
        "manual",
        "continuation",
        "continuation",
        "continuation",
    ]
    rows = (
        case.db.get_connection()
        .execute(
            "SELECT metadata_json FROM messages WHERE role='user' AND metadata_json IS NOT NULL"
        )
        .fetchall()
    )
    assert len(rows) == 3
    assert all(json.loads(row[0])["initiator"] == "hook_continuation" for row in rows)
    assert not case.coordinator._machine_entries


@pytest.mark.asyncio
async def test_slot_reuse_requires_the_exact_current_machine_claim(stop_case):
    case = stop_case
    case.install([{"continuation": {"message": "follow"}}])
    submit = case.coordinator._submit_queued
    previous = []

    async def checked(prompt, **kwargs):
        token = kwargs["authorization"]
        assert case.coordinator.reuses_claimed_slot(token, case.session.id)
        assert not case.coordinator.reuses_claimed_slot(None, case.session.id)
        if previous:
            assert not case.coordinator.reuses_claimed_slot(
                previous[-1], case.session.id
            )
        previous.append(token)
        return await submit(prompt, **kwargs)

    case.coordinator.bind_runtime_submitter(checked)
    _, pending = case.submit()
    await pending
    assert len(previous) == 3
    assert not case.coordinator.reuses_claimed_slot(previous[-1], case.session.id)


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["preparing", "streaming"])
async def test_other_actual_root_still_occupies_capacity(stop_case, monkeypatch, phase):
    import asyncio

    case = stop_case
    monkeypatch.setattr(
        type(case.controller), "max_parallel_runs", property(lambda _: 2)
    )
    case.install([{"continuation": {"message": "follow"}}], hold=True)
    _, parent = case.submit()
    await case.wait_entered()
    entered, release = asyncio.Event(), asyncio.Event()
    resolve, stream = case.gateway.resolve_for_send, case.gateway.stream_chat

    async def held_resolve(selection):
        entered.set()
        await release.wait()
        return await resolve(selection)

    async def held_stream(*args, **kwargs):
        entered.set()
        await release.wait()
        async for chunk in stream(*args, **kwargs):
            yield chunk

    if phase == "preparing":
        case.gateway.resolve_for_send = held_resolve
    else:
        case.gateway.stream_chat = held_stream
    other = case.store.create_session(title="other root", ephemeral=True)
    request = ConsoleTurnCustodyRequest(
        turn_id=str(uuid4()),
        session_id=other.id,
        draft="other work",
        configuration=case.controller.resolve_turn_configuration_snapshot(other.id),
    )
    other_turn = case.runtime.accept_turn(request)
    try:
        await asyncio.wait_for(entered.wait(), 3)
        monkeypatch.setattr(
            type(case.controller), "max_parallel_runs", property(lambda _: 1)
        )
        case.release.touch()
        await parent
        assert len(case.gateway.payloads) == 1
        assert (
            case.db.get_connection()
            .execute("SELECT COUNT(*) FROM console_hook_continuation_receipts")
            .fetchone()[0]
            == 0
        )
    finally:
        release.set()
        await case.runtime.wait_for_turn(other_turn)


@pytest.mark.asyncio
async def test_unrelated_stop_task_cancellation_propagates(stop_case):
    import asyncio

    case = stop_case
    engine = case.install([{"continuation": {"message": "never replay"}}], hold=True)
    _, pending = case.submit()
    await case.wait_entered()
    settlement = case.coordinator._chains[case.session.id]
    cancellation = settlement.hook_cancel_event
    assert cancellation is not None and not cancellation.is_set()
    settlement.pending_stop_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    for _ in range(200):
        if not engine.cleanup_pending:
            break
        await asyncio.sleep(0.01)
    assert not cancellation.is_set()
    assert not engine.cleanup_pending
    assert settlement.hook_cancel_event is None
    assert settlement.pending_stop_task is None
    assert settlement.pending_stop_key is None
    assert len(case.gateway.payloads) == 1
