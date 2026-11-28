"""Child hooks at actual inline/fleet admission and durable settlement."""

import asyncio
import json

import pytest

from Tests.agent_runs_owner import owned_agent_runs_db
from Tests.Agents.test_fleet_runtime import (
    FLEET_CFG,
    fence,
    make_fleet_service,
    make_inline_service,
)
from Tests.Agents.test_hooks_v2_execution import command
from tldw_chatbook.Agents.activation import worker_guard
from tldw_chatbook.Agents.agent_models import SPAWN_TOOL_NAME, WAIT_AGENTS_TOOL_NAME
from tldw_chatbook.Agents.hooks_v2.budgets import HookBudgetOwner
from tldw_chatbook.Agents.hooks_v2.engine import HookEngine

pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.requires_cleanup]


@pytest.mark.asyncio
@pytest.mark.parametrize("fleet", [False, True])
async def test_child_narrowing_and_post_context_use_real_owner(
    tmp_path, monkeypatch, fleet
):
    db = owned_agent_runs_db(tmp_path / "runs.db", client_id="h4")
    observed = tmp_path / "start.json"
    stopped = tmp_path / "stop.txt"
    start = command(
        "import json,sys;from pathlib import Path;event=json.load(sys.stdin);"
        f"Path({str(observed)!r}).open('a').write(json.dumps(event) + '\\n');"
        'print(json.dumps({"version":2,"decision":"pass","child_limits":{"tool_ids":[],"budget_caps":{"max_model_turns":3,"max_total_tokens":5000}}}))',
        name="SubagentStart",
        effects=["child_limits"],
        id="child-start",
    )
    post = command(
        f"from pathlib import Path;Path({str(stopped)!r}).open('a').write('stop\\n');"
        'print(\'{"version":2,"decision":"pass","context":[{"text":"child settled instruction","lifetime":"turn"}]}\')',
        name="SubagentStop",
        effects=["context"],
        required=True,
        id="child-stop",
    )
    engine = HookEngine((start, post), lambda *_: True, HookBudgetOwner())
    child = [fence("find_tools", {"query": "calculator"}), "child result"]
    parent = [fence(SPAWN_TOOL_NAME, {"task": "child"})]
    if fleet:
        service, chat, _ = make_fleet_service(
            db, parent + [fence(WAIT_AGENTS_TOOL_NAME, {}), "done"], {"child": child}
        )
    else:
        service, chat = make_inline_service(db, parent + child + ["done"], monkeypatch)
    service._hooks_v2_engine = engine
    service._hooks_v2_session_id = "session"
    service._hooks_v2_turn_id = "turn"
    try:
        _, outcome = await asyncio.to_thread(
            worker_guard(service)(service.run_turn),
            conversation_id="c",
            messages=[{"role": "user", "content": "go"}],
            config=FLEET_CFG,
            api_endpoint="llama_cpp",
        )
        assert outcome.final_text == "done"
        assert db.count_subagent_runs("c") == 1
        starts = observed.read_text().splitlines()
        assert len(starts) == 1
        assert stopped.read_text().splitlines() == ["stop"]
        event = json.loads(starts[0])
        assert event["data"]["model"] == FLEET_CFG.model
        assert event["data"]["budget_caps"]["max_subagents"] == 0
        assert event["data"]["tool_ids"]
        assert "child settled instruction" in str(chat.calls[-1]["messages_payload"])
        child_row = next(
            row for row in db.list_runs("c") if row["agent_kind"] == "subagent"
        )
        assert child_row["status"] == "done"
        assert child_row["budget"]["max_model_turns"] == 3
        assert child_row["budget"]["max_total_tokens"] == 5000
        assert child_row["budget"]["max_subagents"] == 0
        tool_steps = [
            step for step in child_row["steps"] if step.get("kind") == "tool_result"
        ]
        assert tool_steps and tool_steps[0]["tool_outcome"] != "success", child_row[
            "steps"
        ]
    finally:
        await engine.close()
        db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("fleet", [False, True])
async def test_child_refusal_creates_no_child_but_spawn_tool_settles(
    tmp_path, monkeypatch, fleet
):
    db = owned_agent_runs_db(tmp_path / "runs.db", client_id="h4-deny")
    hook = command("raise SystemExit(1)", name="SubagentStart", effects=["deny"])
    engine = HookEngine((hook,), lambda *_: True, HookBudgetOwner())
    replies = [fence(SPAWN_TOOL_NAME, {"task": "child"}), "done"]
    if fleet:
        service, chat, _ = make_fleet_service(db, replies)
    else:
        service, chat = make_inline_service(db, replies, monkeypatch)
    service._hooks_v2_engine = engine
    service._hooks_v2_session_id = "session"
    service._hooks_v2_turn_id = "turn"
    try:
        _, outcome = await asyncio.to_thread(
            worker_guard(service)(service.run_turn),
            conversation_id="c",
            messages=[{"role": "user", "content": "go"}],
            config=FLEET_CFG,
            api_endpoint="llama_cpp",
        )
        assert outcome.final_text == "done"
        assert db.count_subagent_runs("c") == 0
        assert "initialization refused" in str(chat.calls)
    finally:
        await engine.close()
        db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("fleet", [False, True])
async def test_required_child_stop_holds_actual_parent_after_child_is_durable(
    tmp_path, monkeypatch, fleet
):
    db = owned_agent_runs_db(tmp_path / "race.db", client_id="h4-race")
    entered = tmp_path / "child-post-entered"
    release = tmp_path / "child-post-release"
    hook = command(
        "from pathlib import Path;import time;"
        f"Path({str(entered)!r}).write_text('entered');"
        f"exec({f'while not Path({str(release)!r}).exists(): time.sleep(0.01)'!r});"
        'print(\'{"version":2,"decision":"pass","context":[{"text":"parent release","lifetime":"turn"}]}\')',
        name="SubagentStop",
        required=True,
        effects=["context"],
    )
    engine = HookEngine((hook,), lambda *_: True, HookBudgetOwner())
    replies = [fence(SPAWN_TOOL_NAME, {"task": "child"})]
    if fleet:
        service, chat, _ = make_fleet_service(
            db,
            replies + [fence(WAIT_AGENTS_TOOL_NAME, {}), "done"],
            {"child": ["child result"]},
        )
    else:
        service, chat = make_inline_service(
            db, replies + ["child result", "done"], monkeypatch
        )
    service._hooks_v2_engine = engine
    service._hooks_v2_session_id = "session"
    service._hooks_v2_turn_id = "turn"
    pending = asyncio.create_task(
        asyncio.to_thread(
            worker_guard(service)(service.run_turn),
            conversation_id="c",
            messages=[{"role": "user", "content": "go"}],
            config=FLEET_CFG,
            api_endpoint="llama_cpp",
        )
    )
    try:
        for _ in range(2000):
            if entered.exists():
                break
            await asyncio.sleep(0.005)
        assert entered.exists(), (
            pending.result() if pending.done() else None,
            chat.calls,
        )
        child = next(
            row for row in db.list_runs("c") if row["agent_kind"] == "subagent"
        )
        assert child["status"] == "done"
        assert not pending.done()
        release.write_text("release")
        _, outcome = await pending
        assert outcome.final_text == "done"
        assert "parent release" in str(chat.calls[-1]["messages_payload"])
    finally:
        release.write_text("release")
        await asyncio.gather(pending, return_exceptions=True)
        await engine.close()
        db.close()


@pytest.mark.asyncio
async def test_late_child_settlement_cannot_attach_to_a_new_parent(tmp_path):
    from tldw_chatbook.Agents.hooks_v2.lifecycle import HookSessionLifecycle

    db = owned_agent_runs_db(tmp_path / "late.db", client_id="h4-late")
    parent = db.create_run(conversation_id="c", agent_kind="primary")
    child = db.create_run(
        conversation_id="c", agent_kind="subagent", parent_run_id=parent
    )
    db.set_status(parent, "done", "parent done")
    db.set_status(child, "done", "child done")
    hook = command(
        'print(\'{"version":2,"decision":"pass","context":[{"text":"late child context","lifetime":"turn"}]}\')',
        name="SubagentStop",
        effects=["context"],
    )
    engine = HookEngine((hook,), lambda *_: True, HookBudgetOwner())
    lifecycle = HookSessionLifecycle(engine, "session")
    lifecycle.checkpoints.bind_owner(parent, lifecycle.scope_id)
    lifecycle.close_scope(parent)
    current = lifecycle.open_scope()
    try:
        future = lifecycle.install(
            lifecycle.event(
                "SubagentStop",
                run_id=parent,
                data={"child_run_id": child, "status": "done"},
            ),
            parent,
        )
        assert future is None
        assert lifecycle.diagnostics["late_context"] == 1
        await lifecycle.wait(current)
        assert not lifecycle.context.blocks(current, "model")
        assert not engine.processes.records
    finally:
        lifecycle.seal()
        await engine.close()
        db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("deny_resume", [False, True])
async def test_resume_cannot_skip_actual_subagent_start(tmp_path, deny_resume):
    from Tests.Agents.test_fleet_continuation import _run, _subagent_rows

    marker, deny = tmp_path / "starts.jsonl", tmp_path / "deny"
    code = (
        "import json,sys;from pathlib import Path;e=json.load(sys.stdin);"
        f"p=Path({str(marker)!r});p.write_text((p.read_text() if p.exists() else '')+json.dumps(e)+'\\n');"
        f"blocked=Path({str(deny)!r}).exists();"
        "print(json.dumps(dict(version=2,decision='deny' if blocked else 'pass',"
        "**({} if blocked else {'child_limits':{'tool_ids':[], 'budget_caps':{'max_model_turns':2}}}))))"
    )
    engine = HookEngine(
        (
            command(
                code,
                name="SubagentStart",
                required=True,
                effects=["deny", "child_limits"],
            ),
        ),
        lambda *_: True,
        HookBudgetOwner(),
    )
    db = owned_agent_runs_db(tmp_path / "runs.db", client_id="review")
    holder = {}

    def resume():
        return fence(
            "send_to_agent",
            {"id": holder["handle"], "message": "private-supervisor-rider"},
        )

    service, chat, fleet = make_fleet_service(
        db,
        [
            fence("spawn_subagent", {"task": "study"}),
            fence("wait_agents", {}),
            "first done",
            resume,
            fence("wait_agents", {}),
            "second done",
        ],
        {"study": ["original child done", "resumed child done"]},
        # The denied-control path intentionally leaves the denied child's reply.
        allow_unconsumed=deny_resume,
    )
    service._hooks_v2_engine = engine
    service._hooks_v2_session_id = "session"
    service._hooks_v2_turn_id = "turn-one"
    try:
        old_parent, first = await asyncio.to_thread(
            worker_guard(service)(_run), service
        )
        assert first.status == "done" and first.final_text == "first done"
        original = _subagent_rows(db)
        assert len(original) == 1 and original[0]["status"] == "done"
        old = original[0]["id"]
        handle = next(item for item in fleet.snapshot() if item.run_id == old)
        assert fleet.get_retained(handle.handle_id) is not None
        holder["handle"] = handle.handle_id
        assert len(chat.child_calls["study"]) == 1
        assert len(marker.read_text().splitlines()) == 1
        assert original[0]["parent_run_id"] == old_parent
        if deny_resume:
            deny.touch()
        service._hooks_v2_turn_id = "turn-two"
        new_parent, second = await asyncio.to_thread(
            worker_guard(service)(_run), service
        )
        assert second.status == "done" and second.final_text == "second done"
        rows = _subagent_rows(db)
        assert not engine.processes.records
        assert engine.budget_owner.snapshot()["tickets"] == 0
        if deny_resume:
            assert len(rows) == 1, (
                "required SubagentStart denial was bypassed by retained resume"
            )
            assert len(chat.child_calls["study"]) == 1
        else:
            assert len(rows) == 2 and len(chat.child_calls["study"]) == 2
            resumed = next(row for row in rows if row["id"] != old)
            assert resumed["resumed_from_run_id"] == old
            assert resumed["parent_run_id"] == new_parent
            assert resumed["budget"]["max_model_turns"] == 2
            payload = chat.child_calls["study"][1]["messages_payload"]
            for name in (
                "list_peer_agents",
                "send_to_peer",
                "report_progress",
                "read_agent_messages",
            ):
                assert name not in payload[0]["content"]
            assert (
                sum("original child done" in row.get("content", "") for row in payload)
                == 1
            )
            assert (
                sum(
                    "private-supervisor-rider" in row.get("content", "")
                    for row in payload
                )
                == 1
            )
            assert "private-supervisor-rider" not in json.dumps(
                fleet.snapshot(), default=str
            )
        assert len(marker.read_text().splitlines()) == 2, (
            "resumed child admission never emitted SubagentStart"
        )
    finally:
        await engine.close()
        db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["before-start", "after-start"])
async def test_child_stop_publication_cancel_keeps_actual_installation_join(
    tmp_path, phase, monkeypatch
):
    """Standalone protocol control; native source lifetime has separate journeys."""
    import sys
    import threading
    import time

    from tldw_chatbook.Agents.hooks_v2.lifecycle import HookSessionLifecycle

    db = owned_agent_runs_db(tmp_path / "publication-cancel.db", client_id="h4-cancel")
    parent = db.create_run(conversation_id="c", agent_kind="primary")
    child = db.create_run(
        conversation_id="c", agent_kind="subagent", parent_run_id=parent
    )
    db.set_status(parent, "done", "parent done")
    db.set_status(child, "done", "child done")
    entered, release = tmp_path / "entry", tmp_path / "release"
    hook = command(
        "from pathlib import Path;import time;"
        f"Path({str(entered)!r}).touch();"
        f"exec({f'while not Path({str(release)!r}).exists(): time.sleep(0.01)'!r});"
        'print(\'{"version":2,"decision":"pass"}\')',
        name="SubagentStop",
        required=phase == "after-start",
    )
    engine = HookEngine((hook,), lambda *_: True, HookBudgetOwner())
    lifecycle = HookSessionLifecycle(engine, "session")
    lifecycle.checkpoints.bind_owner(parent, lifecycle.scope_id)
    event = lifecycle.event(
        "SubagentStop", run_id=parent, data={"child_run_id": child, "status": "done"}
    )
    publication = waiter = None
    release_finalization = None
    try:
        if phase == "before-start":
            finished = threading.Event()
            facts, errors = {}, []

            def producer():
                previous = sys.getprofile()

                def cancel_before_loop_dispatch(frame, observed, value):
                    if (
                        frame.f_code is asyncio.run_coroutine_threadsafe.__code__
                        and observed == "return"
                    ):
                        coroutine = frame.f_locals["coro"]
                        if (
                            coroutine.cr_code.co_name == "complete"
                            and coroutine.cr_code.co_filename.endswith(
                                "/hooks_v2/lifecycle.py"
                            )
                        ):
                            facts["cancelled"] = value.cancel()
                            facts["publication"] = value

                sys.setprofile(cancel_before_loop_dispatch)
                try:
                    lifecycle.install(event, parent, _wait_native_admission=True)
                except BaseException as error:
                    errors.append(error)
                finally:
                    sys.setprofile(previous)
                    finished.set()

            thread = threading.Thread(target=producer, name="pre-dispatch-native-hook")
            thread.start()
            # Deliberate finite test barrier: the owner loop cannot dispatch the
            # queued coroutine until the actual producer cancelled publication.
            assert finished.wait(10) and not errors, [
                type(error).__name__ for error in errors
            ]
            thread.join(10)
            assert not thread.is_alive() and facts["cancelled"]
            publication = facts["publication"]
            assert publication.cancelled()
            assert not lifecycle._executions and not lifecycle._installation_admissions
            for _ in range(3):
                await asyncio.sleep(0)
            assert not entered.exists() and not engine.processes.records
        else:
            original_execute = lifecycle._execute
            finalization_entered = asyncio.Event()
            release_finalization = asyncio.Event()

            async def hold_real_finalization(token):
                try:
                    # The real lifecycle body/handler always runs and its exact
                    # result or exception is preserved by this finally.
                    return await original_execute(token)
                finally:
                    finalization_entered.set()
                    await release_finalization.wait()

            monkeypatch.setattr(lifecycle, "_execute", hold_real_finalization)

            def install_until_command_started():
                deadline = time.monotonic() + 10
                installed = lifecycle.install(
                    event, parent, _wait_native_admission=True
                )
                pause = threading.Event()
                while not entered.exists():
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    pause.wait(min(0.005, remaining))
                return installed

            # Reuse the producer thread; an external process owns this marker.
            publication = await asyncio.to_thread(install_until_command_started)
            assert entered.exists(), (
                "actual command did not enter within original bound"
            )
            admitted = lifecycle._installation_admissions[publication]
            assert not admitted.done()
            joined_actual_ack = asyncio.get_running_loop().create_future()
            original_add_callback = admitted.add_done_callback

            def observe_actual_join(callback):
                original_add_callback(callback)
                if not joined_actual_ack.done():
                    joined_actual_ack.set_result(None)

            # Observe registration on the real independent ACK; do not replace
            # its value, create a receiver receipt or claim native admission.
            monkeypatch.setattr(admitted, "add_done_callback", observe_actual_join)
            waiter_began = asyncio.Event()

            async def wait_actual_installation():
                waiter_began.set()
                return await lifecycle.wait_installation(publication)

            waiter = asyncio.create_task(wait_actual_installation())
            await waiter_began.wait()
            assert publication.cancel()
            assert publication.cancelled() and not admitted.done()
            await asyncio.wait_for(finalization_entered.wait(), 10)
            done, _ = await asyncio.wait(
                {waiter, joined_actual_ack},
                timeout=10,
                return_when=asyncio.FIRST_COMPLETED,
            )
            assert joined_actual_ack in done, (
                "waiter returned without joining real finalization"
            )
            assert not waiter.done() and not admitted.done()
            release_finalization.set()
            with pytest.raises(asyncio.CancelledError):
                await waiter
            assert admitted.done() and not admitted.cancelled()
            assert admitted.result() is None
    finally:
        if release_finalization is not None:
            release_finalization.set()
        if waiter is not None:
            await asyncio.gather(waiter, return_exceptions=True)
        release.touch()
        if publication is not None and not publication.done():
            publication.cancel()
        lifecycle.seal()
        await engine.close()
        assert not engine.processes.records and not engine.cleanup_pending
        db.close()
