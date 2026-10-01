"""Lifecycle effects at real Console admission and runtime custody boundaries."""

from types import SimpleNamespace

import pytest

from Tests.Agents.test_hook_permissions import _approve, _edit
from Tests.Agents.test_hook_permissions import hook_file as _hook_file
from Tests.Agents.test_hooks_v2_execution import command
from Tests.Chat.test_console_fleet_wake import _controller_rig
from tldw_chatbook.Agents.run_hooks import HookOutcome
from tldw_chatbook.Chat.console_runtime import ConsoleRuntime

pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.requires_cleanup]
hook_file = _hook_file


@pytest.fixture
async def session_case(tmp_path):
    notifications = []

    class LegacyRecorder:
        async def fire_async(self, event, **kwargs):
            notifications.append((event, kwargs))
            return HookOutcome()

        def notify(self, event, **kwargs):
            notifications.append((event, kwargs))

    rig = _controller_rig(tmp_path, ensure_run_hooks=lambda: LegacyRecorder())
    chacha, app, _runs, store, session, gateway, _bridge, controller = rig
    runtime = ConsoleRuntime(app=app)
    app.console_runtime = runtime
    runtime.set_chat_store(store)
    runtime.set_chat_controller(controller)
    accepted = []
    controller.on_submission_accepted = lambda: accepted.append(session.id)
    permit = tmp_path / "permit"
    marker = tmp_path / "initialized"
    hook = command(
        "from pathlib import Path;"
        f"Path({str(marker)!r}).write_text('started');"
        f"raise SystemExit(0 if Path({str(permit)!r}).exists() else 1)",
        name="SessionStart",
        required=True,
    )
    runtime.ensure_hooks_v2(session.id, (hook,), lambda *_: True)
    case = SimpleNamespace(
        runtime=runtime,
        controller=controller,
        session=session,
        gateway=gateway,
        permit=permit,
        marker=marker,
        refuse_required_initialization=lambda: permit.unlink(missing_ok=True),
        allow_required_initialization=lambda: permit.write_text("allow"),
        submit=lambda: controller.submit_draft("hello", session_id=session.id),
        admitted_turn_count=lambda: len(accepted),
        root_stop_count=lambda: sum(event == "Stop" for event, _ in notifications),
    )
    yield case
    await runtime.close_hooks_v2()
    await runtime.dispose()
    _runs.close()
    chacha.close()


@pytest.mark.asyncio
async def test_failed_initialization_never_publishes_a_root_turn(session_case):
    case = session_case
    case.refuse_required_initialization()
    await case.submit()
    assert case.admitted_turn_count() == 0
    assert case.root_stop_count() == 0
    assert case.marker.exists(), "the controlled command did not execute"
    case.allow_required_initialization()
    await case.submit()
    assert case.admitted_turn_count() == 1
    assert case.root_stop_count() == 1


@pytest.mark.asyncio
async def test_valid_console_submission_control(session_case):
    session_case.allow_required_initialization()
    result = await session_case.submit()
    assert result.accepted
    assert session_case.admitted_turn_count() == 1
    assert session_case.root_stop_count() == 1
    assert session_case.gateway.payloads


@pytest.mark.asyncio
async def test_initialization_cancel_releases_slot_and_retains_child_cleanup(
    session_case, tmp_path
):
    import asyncio

    case = session_case
    await case.runtime.close_hooks_v2(case.session.id)
    entered = tmp_path / "entered"
    release = tmp_path / "release"
    hook = command(
        "from pathlib import Path;import time;"
        f"Path({str(entered)!r}).write_text('entered');"
        f"exec({f'while not Path({str(release)!r}).exists(): time.sleep(0.01)'!r})",
        name="SessionStart",
        required=True,
    )
    engine = case.runtime.ensure_hooks_v2(case.session.id, (hook,), lambda *_: True)
    pending = asyncio.create_task(case.submit())
    try:
        for _ in range(400):
            if entered.exists():
                break
            await asyncio.sleep(0.005)
        assert entered.exists()
        concurrent = await case.submit()
        assert not concurrent.accepted
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert case.admitted_turn_count() == 0
        assert case.root_stop_count() == 0
        assert case.controller.in_flight_run_count() == 0
        release.write_text("allow")
        result = await case.submit()
        assert result.accepted
        assert case.admitted_turn_count() == 1
        assert case.root_stop_count() == 1
    finally:
        release.write_text("release")
        if not pending.done():
            pending.cancel()
        await engine.close()


@pytest.mark.asyncio
async def test_master_off_preserves_required_session_refusal(session_case):
    case = session_case
    case.allow_required_initialization()
    case.runtime.get_hooks_v2(case.session.id).enabled = False
    result = await case.submit()
    assert not result.accepted
    assert case.admitted_turn_count() == 0
    assert case.root_stop_count() == 0
    assert not case.marker.exists()


@pytest.mark.asyncio
async def test_runtime_context_is_live_attributed_and_once_per_send(session_case):
    from tldw_chatbook.Agents.agent_models import PluginContextText

    case = session_case
    await case.runtime.close_hooks_v2(case.session.id)
    hook = command(
        'print(\'{"version":2,"decision":"pass","context":[{"text":"session material","lifetime":"runtime"}]}\')',
        name="SessionStart",
        required=True,
        effects=["context"],
    )
    case.runtime.ensure_hooks_v2(case.session.id, (hook,), lambda *_: True)
    assert (await case.submit()).accepted
    assert (await case.submit()).accepted
    for payload in case.gateway.payloads:
        blocks = [
            row["content"]
            for row in payload
            if "session material" in str(row.get("content"))
        ]
        assert len(blocks) == 1
        assert isinstance(blocks[0], PluginContextText)
        assert blocks[0].checked_hook_origins()


@pytest.mark.asyncio
async def test_live_viewless_session_end_is_observed_once(session_case, tmp_path):
    case = session_case
    await case.runtime.close_hooks_v2(case.session.id)
    marker = tmp_path / "end"
    hook = command(
        f"from pathlib import Path;Path({str(marker)!r}).write_text('end')",
        name="SessionEnd",
    )
    case.runtime.ensure_hooks_v2(case.session.id, (hook,), lambda *_: True)
    assert (await case.submit()).accepted
    await case.runtime.close_hooks_v2(case.session.id)
    assert marker.read_text() == "end"
    await case.runtime.close_hooks_v2(case.session.id)


@pytest.mark.asyncio
async def test_scheduled_wake_initializes_without_user_prompt_submit(
    session_case, tmp_path
):
    from Tests.Chat.test_console_fleet_wake import (
        _drain,
        _settle,
        _survivor,
        _terminal_subagent_run,
    )

    case = session_case
    await case.runtime.close_hooks_v2(case.session.id)
    prompt = tmp_path / "prompt"
    start = command(
        f"from pathlib import Path;Path({str(case.marker)!r}).write_text('start')",
        name="SessionStart",
        required=True,
    )
    user = command(
        f"from pathlib import Path;Path({str(prompt)!r}).write_text('prompt')",
        name="UserPromptSubmit",
        id="user",
    )
    case.runtime.ensure_hooks_v2(case.session.id, (start, user), lambda *_: True)
    _, run_id = _terminal_subagent_run(
        case.controller._agent_bridge.runs_db, case.session.id
    )
    case.controller.fleet_wake.on_fleet_drained(
        _drain(case.session.id, _survivor(run_id, session_id=case.session.id))
    )
    assert await _settle(lambda: case.gateway.payloads)
    assert case.marker.exists()
    assert not prompt.exists()
    assert case.admitted_turn_count() == 0  # Composer publication is manual-only.
    assert case.root_stop_count() == 1


@pytest.mark.asyncio
async def test_revoked_runtime_contribution_blocks_next_actual_send(session_case):
    case = session_case
    await case.runtime.close_hooks_v2(case.session.id)
    authorized = True
    hook = command(
        'print(\'{"version":2,"decision":"pass","context":[{"text":"revocable runtime","lifetime":"runtime"}]}\')',
        name="SessionStart",
        required=True,
        effects=["context"],
    )
    case.runtime.ensure_hooks_v2(case.session.id, (hook,), lambda *_: authorized)
    assert (await case.submit()).accepted
    first_count = len(case.gateway.payloads)
    authorized = False
    result = await case.submit()
    assert not result.accepted
    assert len(case.gateway.payloads) == first_count
    assert case.root_stop_count() == 1


def test_shared_owner_scope_rejects_stale_claims_and_keeps_dependency_scope():
    from Tests.Agents.test_hooks_v2_post_checkpoints import post_event
    from tldw_chatbook.Agents.hooks_v2.checkpoints import (
        HookCheckpointError,
        HookCheckpointStore,
    )
    from tldw_chatbook.Agents.hooks_v2.engine import HookEventOutcome, HookFailure

    store = HookCheckpointStore()
    store.bind_owner("session")
    store.bind_owner("turn", "session")
    store.bind_owner("run", "turn")
    store.bind_owner("sibling", "session")
    with pytest.raises(HookCheckpointError):
        store.bind_owner("unknown-child", "unknown")
    with pytest.raises(HookCheckpointError):
        store.bind_owner("run", "session")
    with pytest.raises(HookCheckpointError):
        store.bind_owner("session", "run")
    event = post_event()
    token = store.begin(event, (), owner_id="turn", dependency_requirements=("dep",))
    store.accept(
        token, HookEventOutcome(failures=(HookFailure("dep", "failed", False, True),))
    )
    store.assert_next_input_allowed("run", required_handler_ids=())
    store.assert_next_input_allowed("sibling", required_handler_ids=("dep",))
    with pytest.raises(HookCheckpointError, match="failed"):
        store.assert_next_input_allowed("run", required_handler_ids=("dep",))
    with pytest.raises(HookCheckpointError, match="unknown"):
        store.assert_next_input_allowed("run", required_handler_ids=None)
    store.retire_owner("run")
    with pytest.raises(HookCheckpointError):
        store.bind_owner("run", "turn")
    store.retire_owner("turn")
    assert not store._entries
    store.assert_next_input_allowed("sibling")


@pytest.mark.asyncio
async def test_idle_config_replacement_creates_a_new_hook_session(
    session_case, tmp_path, hook_file
):
    from Tests.hooks_v2_process_support import child_argv

    case = session_case
    await case.runtime.close_hooks_v2(case.session.id)
    marker = tmp_path / "generations"

    def configuration(label):
        code = f"from pathlib import Path;p=Path({str(marker)!r});p.write_text((p.read_text() if p.exists() else '')+{label!r})"
        return {
            "hooks": {
                "handler": [
                    {
                        "id": "start",
                        "event": "SessionStart",
                        "type": "command",
                        "required": True,
                        "effects": [],
                        "argv": child_argv(code),
                    }
                ]
            }
        }

    permissions = case.runtime.ensure_hook_permissions()
    case.controller._hook_permissions_accessor = lambda: permissions
    _edit(hook_file, lambda section: section.update(configuration("one")["hooks"]))
    assert not (await case.submit()).accepted
    assert not marker.exists()
    assert _approve(permissions).ready
    assert (await case.submit()).accepted
    first = case.runtime._hooks_v2_lifecycles[case.session.id]
    assert (await case.submit()).accepted
    assert marker.read_text() == "one"
    # A stale app dictionary cannot approve or activate changed saved hooks.
    case.runtime._app.app_config = configuration("one")
    _edit(hook_file, lambda section: section.update(configuration("two")["hooks"]))
    assert not permissions.snapshot().ready
    assert not first.current()
    assert not (await case.submit()).accepted
    assert marker.read_text() == "one"
    assert _approve(permissions).ready
    assert (await case.submit()).accepted
    second = case.runtime._hooks_v2_lifecycles[case.session.id]
    assert first is not second
    assert not first.live and second.live
    assert not first.context._rows
    assert marker.read_text() == "onetwo"


@pytest.mark.asyncio
async def test_session_seal_and_checkpoint_currentness_have_one_lock_order(
    session_case, monkeypatch
):
    import asyncio
    import threading

    case = session_case
    case.allow_required_initialization()
    assert (await case.submit()).accepted
    runtime = case.runtime
    lifecycle = runtime._hooks_v2_lifecycles[case.session.id]
    scope = lifecycle.open_scope()
    checkpoint_entered = threading.Event()
    sealing_entered = threading.Event()
    original_current = lifecycle.current
    original_lock = runtime._run_hooks_lock
    original_seal = lifecycle.seal
    lock_timeouts = []
    admission_sealed = []

    class BoundedMapLock:
        def __enter__(self):
            # Bound the actual get/seal lock, without acquiring it a second time.
            if not original_lock.acquire(timeout=0.5):
                lock_timeouts.append(True)
                raise RuntimeError("runtime/checkpoint lock inversion")
            return self

        def __exit__(self, *_args):
            original_lock.release()

    def current_under_checkpoint():
        checkpoint_entered.set()
        assert sealing_entered.wait(2), "sealing never reached the owner"
        return original_current()

    def seal_under_runtime():
        admission_sealed.append(lifecycle.engine._sealed_at is not None)
        sealing_entered.set()
        original_seal()

    monkeypatch.setattr(runtime, "_run_hooks_lock", BoundedMapLock())
    monkeypatch.setattr(lifecycle, "current", current_under_checkpoint)
    monkeypatch.setattr(lifecycle, "seal", seal_under_runtime)
    waiter = asyncio.create_task(lifecycle.wait(scope))
    assert await asyncio.to_thread(checkpoint_entered.wait, 2)
    closer = asyncio.create_task(
        asyncio.to_thread(runtime._seal_hooks_v2, case.session.id)
    )
    results = await asyncio.wait_for(
        asyncio.gather(waiter, closer, return_exceptions=True), 4
    )
    assert not lock_timeouts, results
    assert admission_sealed == [True]
    assert closer.exception() is None
    assert not lifecycle.checkpoints.is_current(scope)
    assert (
        not runtime._disposed
    )  # Exercise live session closure, not shutdown short-circuit.
    outcome = await lifecycle.engine.fire_async(
        lifecycle.event("SessionStart", data={"reason": "resume"})
    )
    assert not outcome.allowed


@pytest.mark.asyncio
async def test_absent_v2_control_does_not_treat_invalid_config_as_absence(
    session_case, hook_file
):
    import toml

    case = session_case
    await case.runtime.close_hooks_v2(case.session.id)
    raw = toml.loads(hook_file.read_text())
    raw["hooks"] = {}
    hook_file.write_text(toml.dumps(raw))
    assert await case.runtime.prepare_hooks_v2(case.session.id) is None
    raw["hooks"] = "invalid saved section"
    hook_file.write_text(toml.dumps(raw))
    with pytest.raises(RuntimeError, match="Review enabled hooks"):
        await case.runtime.prepare_hooks_v2(case.session.id)
