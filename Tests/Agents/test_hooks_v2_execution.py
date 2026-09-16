"""Real bounded command execution through the application-owned hook entry."""

import asyncio
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tldw_chatbook.Agents.hooks_v2 import parse_event, parse_handlers


def event(name="PreToolUse", runtime="runtime-a"):
    return parse_event(
        {
            "protocol_version": 2,
            "event_id": "event-1",
            "event": name,
            "timestamp": "2026-09-16T00:00:00Z",
            "runtime_session_id": runtime,
            "initiator": "manual",
            "origin": "test",
            "causal_chain_id": "chain-1",
            "causal_depth": 0,
            "data": (
                {"tool_id": "local:test", "provider": "local"}
                if name == "PreToolUse"
                else {}
            ),
        }
    )


def command(code="pass", name="PreToolUse", **kwargs):
    from tldw_chatbook.config import get_user_data_dir

    data_root = get_user_data_dir()
    assert str(data_root).startswith("/private/tmp/")
    assert data_root.is_dir()
    # Children inherit the isolated runner and independently prove its provenance.
    prelude = (
        "import pathlib,os,keyring,socket,tldw_chatbook;"
        f"assert str(pathlib.Path(tldw_chatbook.__file__).resolve())=={str(Path(__file__).resolve().parents[2] / 'tldw_chatbook/__init__.py')!r};"
        "assert type(keyring.get_keyring()).__module__=='keyring.backends.null';"
        "assert socket.getaddrinfo.__module__=='sitecustomize';"
        "assert str(pathlib.Path(os.environ['XDG_DATA_HOME']).resolve()).startswith('/private/tmp/');"
        "from tldw_chatbook.config import get_user_data_dir;"
        f"assert str(get_user_data_dir())=={str(data_root)!r};"
    )
    return parse_handlers(
        [
            {
                "id": "command",
                "event": name,
                "type": "command",
                "effects": [],
                "argv": [sys.executable, "-c", prelude + code],
                **kwargs,
            }
        ]
    )[0]


@pytest.fixture
async def command_hook_case():
    from tldw_chatbook.Agents.hooks_v2.budgets import HookBudgetOwner
    from tldw_chatbook.Agents.hooks_v2.engine import HookEngine

    owner = HookBudgetOwner()
    engine = HookEngine((command(),), lambda *_: True, owner)
    yield SimpleNamespace(
        engine=engine,
        event=event(),
        owner=owner,
        live_children=lambda: len(engine.processes.records),
    )
    await engine.close()


@pytest.mark.asyncio
async def test_command_success_and_disposal_use_same_entry(command_hook_case):
    case = command_hook_case
    assert (await case.engine.fire_async(case.event)).succeeded
    await case.engine.close()
    assert not (await case.engine.fire_async(case.event)).succeeded
    assert case.live_children() == 0


def test_normalized_command_control():
    assert command().argv[0] == sys.executable
    assert event().event == "PreToolUse"


@pytest.mark.asyncio
async def test_sync_worker_and_same_loop_refusal(command_hook_case):
    case = command_hook_case
    with pytest.raises(RuntimeError, match="owner loop"):
        case.engine.fire(case.event)
    assert (await asyncio.to_thread(case.engine.fire, case.event)).succeeded


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("code", "failure"),
    [
        ("import os;os.write(1,b'x'*16385)", "stdout_overflow"),
        ("import os;os.write(1,b'\\xff')", "invalid_output"),
        (
            'print(\'{"version":2,"decision":"pass","decision":"deny"}\')',
            "invalid_output",
        ),
        ("raise SystemExit(2)", "nonzero_exit"),
    ],
)
async def test_complete_stdout_gate(code, failure):
    from tldw_chatbook.Agents.hooks_v2.budgets import HookBudgetOwner
    from tldw_chatbook.Agents.hooks_v2.engine import HookEngine

    engine = HookEngine(
        (command(code, effects=["deny"]),), lambda *_: True, HookBudgetOwner()
    )
    result = await engine.fire_async(event())
    assert result.failures[0].code == failure
    assert not result.allowed and not result.accepted
    assert not engine.processes.records
    await engine.close()


@pytest.mark.asyncio
async def test_exact_stdout_limit_and_both_streams_are_drained():
    from tldw_chatbook.Agents.hooks_v2.budgets import HookBudgetOwner
    from tldw_chatbook.Agents.hooks_v2.engine import HookEngine

    body = b'{"version":2,"decision":"pass"}'
    code = (
        f"import os;os.write(2,b'e'*200000);os.write(1,{body!r}+b' '*{16384-len(body)})"
    )
    engine = HookEngine((command(code),), lambda *_: True, HookBudgetOwner())
    result = await engine.fire_async(event())
    assert result.succeeded
    await engine.close()


@pytest.mark.asyncio
async def test_timeout_and_cancellation_reap_real_child(command_hook_case):
    from tldw_chatbook.Agents.hooks_v2.engine import HookEngine

    case = command_hook_case
    engine = HookEngine(
        (command("import time;time.sleep(30)", timeout_seconds=0.2, effects=["deny"]),),
        lambda *_: True,
        case.owner,
    )
    outcome = await engine.fire_async(case.event)
    assert not outcome.allowed
    assert not engine.processes.records, [
        (outcome, j.task.result(), j.capture.transport, j.capture.exited.done())
        for j in engine.processes.records.values()
    ]
    assert case.owner.snapshot()["tickets"] == 0
    await engine.close()


@pytest.mark.asyncio
async def test_cancelled_launch_keeps_process_and_ticket_until_publication(monkeypatch):
    from tldw_chatbook.Agents.hooks_v2.budgets import HookBudgetOwner
    from tldw_chatbook.Agents.hooks_v2.engine import HookEngine

    owner = HookBudgetOwner()
    engine = HookEngine(
        (command("import time;time.sleep(30)", effects=["deny"]),),
        lambda *_: True,
        owner,
    )
    original = owner.loop.subprocess_exec
    spawned = asyncio.Event()
    publish = asyncio.Event()
    transports = []

    async def held(*args, **kwargs):
        result = await original(*args, **kwargs)
        transports.append(result[0])
        spawned.set()
        await publish.wait()
        return result

    monkeypatch.setattr(owner.loop, "subprocess_exec", held)
    caller = asyncio.create_task(engine.fire_async(event()))
    await asyncio.wait_for(spawned.wait(), 2)
    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller
    assert owner.snapshot()["tickets"] == 1
    assert len(engine.processes.records) == 1
    close = asyncio.create_task(engine.close())
    await asyncio.sleep(0)
    close.cancel()
    with pytest.raises(asyncio.CancelledError):
        await close
    publish.set()
    await engine.close()
    assert transports[0].get_returncode() is not None
    assert not engine.processes.records
    assert owner.snapshot()["tickets"] == 0


@pytest.mark.asyncio
async def test_scoped_handler_chain_identity_and_deadline():
    from tldw_chatbook.Agents.hooks_v2.budgets import HookBudgetOwner
    from tldw_chatbook.Agents.hooks_v2.engine import HookEngine

    owner = HookBudgetOwner()
    a = command(
        'print(\'{"version":2,"decision":"pass","updated_input":{"n":2}}\')',
        effects=["updated_input"],
    )
    b = command(
        'import json,sys; e=json.load(sys.stdin);assert e["data"]["tool_args"]=={"n":2}',
        id="second",
        effects=["deny"],
    )
    engine = HookEngine((a, b), lambda *_: True, owner)
    scope = engine.begin_event(event())
    first = await engine.fire_handler_async(scope, a.id, event())
    assert dict(first.accepted[0][1].updated_input) == {"n": 2}
    changed = event().model_dump()
    changed = {k: v for k, v in changed.items() if v is not None}
    changed["data"]["tool_args"] = {"n": 2}
    assert (
        await asyncio.to_thread(engine.fire_handler, scope, b.id, parse_event(changed))
    ).succeeded
    deadline = scope.deadline
    assert scope.active_seconds > 0
    changed["event_id"] = "other"
    assert not (
        await engine.fire_handler_async(scope, b.id, parse_event(changed))
    ).succeeded
    other = HookEngine((a, b), lambda *_: True, owner)
    assert not (await other.fire_handler_async(scope, a.id, event())).succeeded
    assert scope.deadline == deadline
    scope.close()
    assert not (await engine.fire_handler_async(scope, a.id, event())).succeeded
    await engine.close()
    await other.close()


@pytest.mark.asyncio
async def test_sealed_engine_authorized_teardown_and_other_runtime():
    from tldw_chatbook.Agents.hooks_v2.budgets import HookBudgetOwner
    from tldw_chatbook.Agents.hooks_v2.engine import HookEngine

    owner = HookBudgetOwner()
    permitted = True
    engine = HookEngine(
        (command(), command(name="SessionEnd", id="end")), lambda *_: permitted, owner
    )
    other = HookEngine((command(),), lambda *_: True, owner)
    engine.begin_close()
    deadline = engine._sealed_at
    assert not (await engine.fire_async(event())).succeeded
    assert (await engine.fire_teardown_async(event("SessionEnd"))).succeeded
    engine.begin_close()
    assert engine._sealed_at == deadline
    permitted = False
    assert not (await engine.fire_teardown_async(event("SessionEnd"))).succeeded
    assert (await other.fire_async(event(runtime="b"))).succeeded
    await engine.close()
    assert not engine.notify_teardown(event("SessionEnd"))
    await other.close()


@pytest.mark.asyncio
async def test_configuration_refusal_is_scoped_and_never_activates_batch():
    from tldw_chatbook.Agents.hooks_v2.budgets import HookBudgetOwner
    from tldw_chatbook.Agents.hooks_v2.engine import HookEngine
    from tldw_chatbook.Agents.run_hooks import load_hooks_config

    config = load_hooks_config(
        {
            "hooks": {
                "enabled": True,
                "handler": [
                    {
                        "id": "bad",
                        "type": "command",
                        "event": "PreToolUse",
                        "effects": ["deny"],
                        "argv": [],
                    },
                    {
                        "id": "optional",
                        "type": "command",
                        "event": "UserPromptSubmit",
                        "effects": [None],
                        "argv": [],
                    },
                ],
            }
        }
    )
    engine = HookEngine.from_config(config, lambda *_: True, HookBudgetOwner())
    assert engine.definitions == ()
    assert not (await engine.fire_async(event())).allowed
    other = await engine.fire_async(event("SessionEnd"))
    assert other.allowed and not other.succeeded
    await engine.close()


@pytest.mark.asyncio
async def test_seal_kills_child_while_owner_publication_is_pending():
    import threading

    from tldw_chatbook.Agents.hooks_v2.budgets import HookBudgetOwner
    from tldw_chatbook.Agents.hooks_v2.engine import HookEngine
    from tldw_chatbook.Agents.hooks_v2.ownership import HostProcessOwner

    published = threading.Event()
    release = threading.Event()

    class HeldOwner(HostProcessOwner):
        def publish_process(self, token, provenance):
            super().publish_process(token, provenance)
            published.set()
            release.wait(5)

    budgets = HookBudgetOwner()
    engine = HookEngine(
        (command("import time;time.sleep(30)", effects=["deny"]),),
        lambda *_: True,
        budgets,
        process_owner=HeldOwner(),
    )
    caller = asyncio.create_task(engine.fire_async(event()))
    assert await asyncio.to_thread(published.wait, 2)
    job = next(iter(engine.processes.records.values()))
    try:
        engine.begin_close()
        await asyncio.wait_for(asyncio.shield(job.capture.exited), 0.5)
        assert budgets.snapshot()["tickets"] == 1
    finally:
        release.set()
        await engine.close()
        await caller


@pytest.mark.asyncio
async def test_failed_terminal_settlement_retains_exact_resources():
    from tldw_chatbook.Agents.hooks_v2.budgets import HookBudgetOwner
    from tldw_chatbook.Agents.hooks_v2.engine import HookEngine
    from tldw_chatbook.Agents.hooks_v2.ownership import HostProcessOwner

    class RefusingOwner(HostProcessOwner):
        refuse = True

        def settle_process(self, token, confirmed):
            if self.refuse:
                raise RuntimeError("secret-bearing diagnostic")
            super().settle_process(token, confirmed)

    owner = RefusingOwner()
    budget = HookBudgetOwner()
    engine = HookEngine(
        (command(effects=["deny"]),), lambda *_: True, budget, process_owner=owner
    )
    outcome = await engine.fire_async(event())
    assert outcome.outstanding_cleanup and not outcome.accepted
    assert budget.snapshot()["execution"] == budget.snapshot()["tickets"] == 1
    assert len(owner.records) == len(engine.processes.records) == 1
    assert "secret" not in repr(outcome)
    owner.refuse = False
    await engine.processes.reap_pending()
    assert budget.snapshot()["tickets"] == 0
    await engine.close()


@pytest.mark.asyncio
async def test_refused_kill_retains_live_child_until_later_terminal_proof(monkeypatch):
    import signal

    from tldw_chatbook.Agents.hooks_v2 import command_executor
    from tldw_chatbook.Agents.hooks_v2.budgets import HookBudgetOwner
    from tldw_chatbook.Agents.hooks_v2.engine import HookEngine

    real_kill = os.killpg

    def refuse(pid, sig):
        if sig == signal.SIGKILL:
            raise PermissionError("injected host refusal")
        return real_kill(pid, sig)

    budget = HookBudgetOwner()
    engine = HookEngine(
        (command("import time;time.sleep(30)", timeout_seconds=0.2, effects=["deny"]),),
        lambda *_: True,
        budget,
    )
    with monkeypatch.context() as m:
        m.setattr(os, "killpg", refuse)
        m.setattr(command_executor, "REAP_SECONDS", 0.1)
        result = await engine.fire_async(event())
    job = next(iter(engine.processes.records.values()))
    try:
        assert result.outstanding_cleanup
        assert job.capture.transport.get_returncode() is None
        assert budget.snapshot()["tickets"] == budget.snapshot()["execution"] == 1
    finally:
        real_kill(job.capture.transport.get_pid(), signal.SIGKILL)
        await asyncio.wait_for(asyncio.shield(job.capture.exited), 2)
        await engine.processes.reap_pending()
        await engine.close()
    assert not engine.cleanup_pending and budget.snapshot()["tickets"] == 0


@pytest.mark.asyncio
async def test_environment_refs_reserved_roots_and_authority_acceptance(tmp_path):
    from tldw_chatbook.Agents.hooks_v2.budgets import HookBudgetOwner
    from tldw_chatbook.Agents.hooks_v2.engine import HookEngine

    # The native parser requires explicit declaration of reference names.
    raw = command(
        'import os;assert os.environ["PLAIN"]=="literal";assert os.environ["PLUGIN_ROOT"]=="host-root"'
    ).model_dump()
    raw = {k: v for k, v in raw.items() if v is not None}
    # model_dump includes command-inapplicable defaults only when non-None.
    raw["env"] = {"PLAIN": "literal", "TOKEN": {"variable": "TOKEN_REF"}}
    handler = parse_handlers([raw], declared_variable_names={"TOKEN_REF"})[0]
    stages = []

    def authority(_handler, _event, stage):
        stages.append(stage)
        return stage != "accept"

    engine = HookEngine(
        (handler,),
        authority,
        HookBudgetOwner(),
        environment=lambda *_: {"TOKEN_REF": "private"},
        host_environment=lambda *_: {"PLUGIN_ROOT": "host-root"},
    )
    refused = await engine.fire_async(event())
    assert refused.failures[0].code == "authority_refused"
    assert "launch" in stages and "accept" in stages
    engine.authority_check = lambda *_: True
    assert (await engine.fire_async(event())).succeeded
    await engine.close()


@pytest.mark.asyncio
async def test_disabled_config_keeps_required_constraint():
    from tldw_chatbook.Agents.hooks_v2.budgets import HookBudgetOwner
    from tldw_chatbook.Agents.hooks_v2.engine import HookEngine
    from tldw_chatbook.Agents.run_hooks import load_hooks_config

    config = load_hooks_config(
        {
            "hooks": {
                "enabled": False,
                "handler": [
                    {
                        "id": "required",
                        "event": "PreToolUse",
                        "type": "command",
                        "effects": [],
                        "argv": [sys.executable, "-c", "raise SystemExit(99)"],
                        "required": True,
                    }
                ],
            }
        }
    )
    engine = HookEngine.from_config(config, lambda *_: True, HookBudgetOwner())
    result = await engine.fire_async(event())
    assert not result.allowed
    assert result.failures[0].code == "disabled"
    assert not engine.processes.records
    await engine.close()


@pytest.mark.asyncio
async def test_notification_queue_overflow_is_synchronous_and_controlling_stays_available():
    from tldw_chatbook.Agents.hooks_v2.budgets import HookBudgetOwner
    from tldw_chatbook.Agents.hooks_v2.engine import HookEngine

    owner = HookBudgetOwner()
    engine = HookEngine(
        (command(name="SessionEnd"), command(id="guard", effects=["deny"])),
        lambda *_: True,
        owner,
    )
    assert all(engine.notify(event("SessionEnd")) for _ in range(64))
    assert not engine.notify(event("SessionEnd"))
    assert engine.notification_omissions == 1
    assert owner.snapshot()["observations"] == 64
    assert (await engine.fire_async(event())).allowed
    engine.begin_close()
    await engine.close()
    assert not engine.cleanup_pending
    assert all(value == 0 for value in owner.snapshot().values())


@pytest.mark.asyncio
async def test_interrupt_wall_deadline_includes_queue_and_is_not_reset():
    import time

    from tldw_chatbook.Agents.hooks_v2.budgets import HookBudgetOwner
    from tldw_chatbook.Agents.hooks_v2.engine import HookEngine

    owner = HookBudgetOwner()
    blockers = [owner.reserve("runtime-a", True) for _ in range(4)]
    await asyncio.gather(*(t.acquire() for t in blockers))
    engine = HookEngine((command(name="Interrupt"),), lambda *_: True, owner)
    started = time.monotonic()
    result = await engine.fire_teardown_async(event("Interrupt"))
    elapsed = time.monotonic() - started
    assert not result.succeeded and 0.9 <= elapsed < 1.5
    assert not engine.processes.records
    for t in blockers:
        t.release()
    engine.begin_close()
    deadline = engine.teardown_deadline
    engine.begin_close()
    assert engine.teardown_deadline == deadline
    await engine.close()


@pytest.mark.asyncio
async def test_exact_stdin_limit_is_delivered_and_oversize_refuses_whole():
    import json

    from tldw_chatbook.Agents.hooks_v2.budgets import HookBudgetOwner
    from tldw_chatbook.Agents.hooks_v2.engine import HookEngine
    from tldw_chatbook.Agents.hooks_v2.validation import INPUT_BYTES

    payload = {
        k: v for k, v in event("UserPromptSubmit").model_dump().items() if v is not None
    }
    payload["data"] = {"prompt": ""}
    size = len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode())
    payload["data"]["prompt"] = "a" * (INPUT_BYTES - size)
    accepted = parse_event(payload)
    engine = HookEngine(
        (
            command(
                f"import sys;assert len(sys.stdin.buffer.read())=={INPUT_BYTES}",
                name="UserPromptSubmit",
                required=True,
            ),
        ),
        lambda *_: True,
        HookBudgetOwner(),
    )
    assert (await engine.fire_async(accepted)).succeeded
    oversized = accepted.model_copy(
        update={"data": {"prompt": payload["data"]["prompt"] + "a"}}
    )
    result = await engine.fire_async(oversized)
    assert not result.allowed and result.failures[0].code == "invalid_event"
    assert not engine.processes.records
    await engine.close()


@pytest.mark.asyncio
async def test_spawn_failure_and_required_context_have_no_empty_success():
    from tldw_chatbook.Agents.hooks_v2.budgets import HookBudgetOwner
    from tldw_chatbook.Agents.hooks_v2.engine import HookEngine

    owner = HookBudgetOwner()
    missing = parse_handlers(
        [
            {
                "id": "missing",
                "event": "PreToolUse",
                "type": "command",
                "effects": ["deny"],
                "argv": ["/definitely-absent-h2-command"],
            }
        ]
    )[0]
    engine = HookEngine((missing,), lambda *_: True, owner)
    outcome = await engine.fire_async(event())
    assert outcome.failures[0].code == "launch_failed"
    assert not engine.processes.records and owner.snapshot()["tickets"] == 0
    await engine.close()
    context = HookEngine(
        (command(effects=["context"], required=True, require_context=True),),
        lambda *_: True,
        owner,
    )
    outcome = await context.fire_async(event())
    assert not outcome.allowed and not outcome.accepted
    await context.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("leader_exits", [False, True])
async def test_posix_owned_group_descendants_are_reaped(tmp_path, leader_exits):
    import os

    from tldw_chatbook.Agents.hooks_v2.budgets import HookBudgetOwner
    from tldw_chatbook.Agents.hooks_v2.engine import HookEngine

    if os.name != "posix":
        pytest.fail("This real process-group qualification requires a POSIX worker")
    pidfile = tmp_path / "grandchild"
    child = command("import time;time.sleep(30)").argv[2]
    code = (
        "import subprocess,sys,pathlib,time;"
        f"p=subprocess.Popen([sys.executable,'-c',{child!r}]);"
        f"pathlib.Path({str(pidfile)!r}).write_text(str(p.pid));"
        + ("pass" if leader_exits else "time.sleep(30)")
    )
    engine = HookEngine(
        (command(code, timeout_seconds=0.6, effects=["deny"]),),
        lambda *_: True,
        HookBudgetOwner(),
    )
    outcome = await engine.fire_async(event())
    assert pidfile.exists()
    assert not outcome.allowed
    assert not engine.processes.records
    await engine.close()


@pytest.mark.asyncio
async def test_notification_errors_remain_metadata_only():
    from tldw_chatbook.Agents.hooks_v2.budgets import HookBudgetOwner
    from tldw_chatbook.Agents.hooks_v2.engine import HookEngine

    engine = HookEngine(
        (command('print("private output")', name="SessionEnd"),),
        lambda *_: True,
        HookBudgetOwner(),
    )
    assert engine.notify_teardown(event("SessionEnd"))
    await engine.close()
    assert engine.notification_failures == {"invalid_output": 1}
    assert "private" not in repr(engine.notification_failures)
