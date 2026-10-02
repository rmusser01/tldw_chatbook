"""Actual auxiliary candidates, committed memory, and scoped post fences."""

import asyncio
from types import SimpleNamespace

import pytest

pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.requires_cleanup]

from Tests.Agents.test_hooks_v2_execution import command
from Tests.Chat.test_console_context_compaction import (
    _Gateway,
    _manual_transaction_inputs,
    _prepare,
    _Repository,
    _resolution,
    _transaction_inputs,
)
from tldw_chatbook.Agents.hooks_v2.budgets import HookBudgetOwner
from tldw_chatbook.Agents.hooks_v2.checkpoints import HookCheckpointError
from tldw_chatbook.Agents.hooks_v2.engine import HookEngine
from tldw_chatbook.Agents.hooks_v2.lifecycle import (
    CompactionHooks,
    HookSessionLifecycle,
)
from tldw_chatbook.Chat.console_context_compaction import (
    CompactionTerminal,
    ConsoleCompactionService,
)
from tldw_chatbook.Chat.console_prepared_request import PreparedConsoleRequest


@pytest.mark.asyncio
@pytest.mark.parametrize("manual", [False, True])
@pytest.mark.parametrize("failure", ["none", "pre", "post"])
async def test_actual_compaction_commit_and_required_input_fence(manual, failure):
    pre = command(
        (
            "raise SystemExit(1)"
            if failure == "pre"
            else 'print(\'{"version":2,"decision":"pass","context":[{"text":"candidate only","lifetime":"turn"}]}\')'
        ),
        name="PreCompact",
        effects=["context"],
        required=True,
        id="pre",
    )
    post = command(
        (
            "raise SystemExit(1)"
            if failure == "post"
            else 'print(\'{"version":2,"decision":"pass","context":[{"text":"next input only","lifetime":"turn"}]}\')'
        ),
        name="PostCompact",
        effects=["context"],
        required=True,
        id="post",
    )
    engine = HookEngine((pre, post), lambda *_: True, HookBudgetOwner())
    lifecycle = HookSessionLifecycle(engine, "session")
    scope = lifecycle.open_scope()
    repository = _Repository()
    gateway = _Gateway()
    candidates = []
    original = gateway.complete_auxiliary

    async def record(request, **kwargs):
        candidates.append(request.messages)
        return await original(request, **kwargs)

    gateway.complete_auxiliary = record
    hooks = CompactionHooks(
        lifecycle,
        scope,
        reason="manual" if manual else "automatic",
        prepare=lambda rows, cap: _prepare(
            PreparedConsoleRequest(active_request=rows), response_tokens=cap
        ),
        current=lambda: True,
        memory_current=lambda memory: memory in repository.memories,
    )
    service = ConsoleCompactionService(repository, gateway)
    try:
        if manual:
            plan, prompt, admission = _manual_transaction_inputs()
            result = await service.summarize_manual(
                plan=plan,
                admission=admission,
                resolution=_resolution(),
                prompt=prompt,
                current_admission=lambda: admission,
                prepare_projection=_prepare,
                hooks=hooks,
            )
        else:
            plan, prompt, prefix, admission, commit = _transaction_inputs()
            result = await service.compact(
                plan=plan,
                admission=admission,
                branch_commit=commit,
                resolution=_resolution(),
                prompt=prompt,
                current_admission=lambda: admission,
                prepare_main=_prepare,
                prefix_messages=prefix,
                hooks=hooks,
            )
        await hooks.finish()
        if failure == "pre":
            assert result.terminal is CompactionTerminal.FAILED
            assert gateway.calls == 0
            assert not result.attempted
            assert (
                repository.finishes[-1][1]["failure_reason"]
                == "required_pre_compact_failed"
            )
            assert not repository.memories
            return
        assert result.terminal is CompactionTerminal.SUCCEEDED
        assert len(repository.memories) == 1
        assert "candidate only" in str(candidates)
        assert "next input only" not in str(candidates)
        assert "candidate only" not in repository.memories[0].summary_text
        receiver = lifecycle.open_scope() if manual else scope
        if failure == "post":
            with pytest.raises(HookCheckpointError):
                await lifecycle.wait(receiver)
            assert len(repository.memories) == 1
        else:
            await lifecycle.wait(receiver)
            rows = (
                lifecycle.claim_handoff(receiver)
                if manual
                else lifecycle.context.blocks(receiver, "model")
            )
            assert "next input only" in str(rows)
            assert "candidate only" not in str(rows)
            assert not lifecycle.claim_handoff(receiver)
            assert not lifecycle.context.blocks(receiver, "model")
    finally:
        lifecycle.seal()
        await engine.close()


@pytest.mark.asyncio
async def test_focused_fallback_gets_its_actual_required_candidate():
    engine = HookEngine(
        (
            command(
                'print(\'{"version":2,"decision":"pass","context":[{"text":"mandatory candidate","lifetime":"turn"}]}\')',
                name="PreCompact",
                effects=["context"],
                required=True,
            ),
        ),
        lambda *_: True,
        HookBudgetOwner(),
    )
    lifecycle = HookSessionLifecycle(engine, "session")
    repository = _Repository()
    gateway = _Gateway()
    gateway.texts = ["", "Compact facts."]
    seen = []
    original = gateway.complete_auxiliary

    async def record(request, **kwargs):
        seen.append(request.messages)
        return await original(request, **kwargs)

    gateway.complete_auxiliary = record
    plan, prompt, admission = _manual_transaction_inputs(focus="decisions")
    hooks = CompactionHooks(
        lifecycle,
        lifecycle.open_scope(),
        reason="manual",
        prepare=lambda rows, cap: _prepare(
            PreparedConsoleRequest(active_request=rows), response_tokens=cap
        ),
        current=lambda: True,
        memory_current=lambda _: True,
    )
    try:
        result = await ConsoleCompactionService(repository, gateway).summarize_manual(
            plan=plan,
            admission=admission,
            resolution=_resolution(),
            prompt=prompt,
            current_admission=lambda: admission,
            prepare_projection=_prepare,
            hooks=hooks,
        )
        await hooks.finish()
        assert result.terminal is CompactionTerminal.SUCCEEDED
        assert len(seen) == 2
        assert all("mandatory candidate" in str(rows) for rows in seen)
        assert seen[0][0] != seen[1][0]
    finally:
        lifecycle.seal()
        await engine.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "manual,post_failure,memory_changed,pre_failure",
    [
        (False, False, False, False),
        (False, True, False, False),
        (True, False, False, False),
        (True, True, False, False),
        (True, False, True, False),
        (False, False, False, True),
        (True, False, False, True),
    ],
)
async def test_console_compaction_real_sqlite_commit_and_provider_fence(
    tmp_path, manual, post_failure, memory_changed, pre_failure
):
    from Tests.Chat.test_console_context_compaction import _real_selection_controller
    from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
    from tldw_chatbook.Chat.console_context_policy import (
        ConsoleContextPolicyOverrides,
        ContextBudgetMode,
        ContextCompactionMode,
    )
    from tldw_chatbook.Chat.console_runtime import ConsoleRuntime

    db, repository, controller, store, session, conversation_id, _ = (
        _real_selection_controller(tmp_path, "h4-real-commit")
    )
    runtime = ConsoleRuntime(app=None)
    runtime.set_chat_store(store)
    runtime.set_chat_controller(controller)
    controller._agent_runtime_enabled = False
    gateway = controller.provider_gateway
    from Tests.console_provider_doubles import with_destination

    async def resolve(_selection):
        return with_destination(_resolution())

    gateway.resolve_for_send = resolve
    payloads = []
    auxiliary = []
    complete = gateway.complete_auxiliary

    async def record_aux(request, **kwargs):
        auxiliary.append(request.messages)
        return await complete(request, **kwargs)

    async def stream(_resolution, messages, **_kwargs):
        payloads.append(getattr(messages, "messages_payload", messages))
        yield "final answer"

    gateway.complete_auxiliary = record_aux
    gateway.stream_chat = stream
    for index in range(2):
        store.append_message(
            session.id,
            role=ConsoleMessageRole.USER,
            content="old question " + "x " * 450,
            persist=True,
        )
        store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="old answer " + "y " * 450,
            persist=True,
        )
    target = store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="current request",
        persist=True,
    )
    store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="current answer",
        persist=True,
    )
    store.set_session_context_policy_overrides(
        session.id,
        ConsoleContextPolicyOverrides(
            budget_mode=ContextBudgetMode.CUSTOM,
            custom_budget_tokens=1800,
            compaction_mode=(
                ContextCompactionMode.OFF if manual else ContextCompactionMode.AUTOMATIC
            ),
            summary_max_tokens=100,
        ),
    )
    hooks = (
        command(
            'print(\'{"version":2,"decision":"pass","context":[{"text":"runtime retained","lifetime":"runtime"}]}\')',
            name="SessionStart",
            required=True,
            effects=["context"],
            id="start",
        ),
        command(
            (
                "raise SystemExit(1)"
                if pre_failure
                else 'print(\'{"version":2,"decision":"pass","context":[{"text":"pre only","lifetime":"turn"}]}\')'
            ),
            name="PreCompact",
            required=True,
            effects=["context"],
            id="pre",
        ),
        command(
            (
                "raise SystemExit(1)"
                if post_failure
                else 'print(\'{"version":2,"decision":"pass","context":[{"text":"post once","lifetime":"turn"}]}\')'
            ),
            name="PostCompact",
            required=True,
            effects=["context"],
            id="post",
        ),
    )
    runtime.ensure_hooks_v2(session.id, hooks, lambda *_: True)
    commit_errors = []
    original_commit = repository.commit_memory_selection_if_current

    def inspect_commit(commit):
        try:
            return original_commit(commit)
        except Exception as error:
            commit_errors.append(
                (
                    type(error).__name__,
                    str(error),
                    [
                        (previous.message_id, row.parent_message_id, row.message_id)
                        for previous, row in zip(
                            commit.durable_lineage, commit.durable_lineage[1:]
                        )
                        if previous.message_id != row.parent_message_id
                    ],
                )
            )
            raise

    repository.commit_memory_selection_if_current = inspect_commit
    transactions = []
    compact = controller._compaction_service.compact

    async def record_transaction(**kwargs):
        value = await compact(**kwargs)
        transactions.append(value)
        return value

    controller._compaction_service.compact = record_transaction
    try:
        if manual:
            result = await controller.summarize_up_to(target.id)
            assert result.accepted is (not pre_failure), result
            assert not payloads
        else:
            result = await controller.submit_draft("next", session_id=session.id)
        snapshots = controller._durable_context_snapshots(session.id)
        state = repository.load_applicable_branch_memory(
            conversation_id, frozenset(row.message_id for row in snapshots)
        )
        if pre_failure:
            assert gateway.calls == 0
            assert not payloads and state.memory is None
            attempts = repository.list_auxiliary_attempts(conversation_id)
            assert [row["failure_reason"] for row in attempts] == [
                "required_pre_compact_failed"
            ]
            return
        assert state.memory is not None, (result, transactions, commit_errors)
        assert gateway.calls == 1
        assert "pre only" in str(auxiliary)
        assert "runtime retained" not in str(auxiliary)
        assert "post once" not in state.memory.summary_text
        handoff_origin = None
        if manual:
            life = runtime._hooks_v2_lifecycles[session.id]
            if not post_failure:
                handoff_origin = next(
                    row["content"].checked_hook_origins()
                    for rows in life.context._rows.values()
                    for row in rows
                    if "post once" in row["content"]
                )
            if memory_changed:
                assert repository.deactivate_memory(
                    state.memory.memory_id,
                    expected_revision=state.memory.revision,
                    reset_at="2026-09-16T00:00:00Z",
                )
            result = await controller.submit_draft("next", session_id=session.id)
        if post_failure:
            assert not payloads
        else:
            assert result.accepted, result
            assert len(payloads) == 1
            assert str(payloads[0]).count("post once") == (0 if memory_changed else 1)
            if manual and not memory_changed:
                delivered = next(
                    row["content"]
                    for row in payloads[0]
                    if "post once" in str(row.get("content"))
                )
                assert delivered.checked_hook_origins() == handoff_origin
            if memory_changed:
                assert life.diagnostics["late_context"] == 1
            assert str(payloads[0]).count("runtime retained") == 1
            assert "pre only" not in str(payloads[0])
            if manual:
                assert (
                    await controller.submit_draft("again", session_id=session.id)
                ).accepted
                assert "post once" not in str(payloads[-1])
    finally:
        await runtime.close_hooks_v2()
        await runtime.dispose()
        db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancelled", [False, True])
async def test_manual_post_requirement_precedes_next_input_and_late_effects_are_discarded(
    tmp_path, cancelled
):
    entered = tmp_path / "post-entered"
    release = tmp_path / "post-release"
    post = command(
        "from pathlib import Path;import time;"
        f"Path({str(entered)!r}).write_text('entered');"
        f"exec({f'while not Path({str(release)!r}).exists(): time.sleep(0.01)'!r});"
        'print(\'{"version":2,"decision":"pass","context":[{"text":"owned post","lifetime":"turn"}]}\')',
        name="PostCompact",
        effects=["context"],
        required=True,
    )
    engine = HookEngine((post,), lambda *_: True, HookBudgetOwner())
    lifecycle = HookSessionLifecycle(engine, "session")
    operation = lifecycle.open_scope()
    receiver = lifecycle.open_scope()
    hooks = CompactionHooks(
        lifecycle,
        operation,
        reason="manual",
        prepare=lambda *_: None,
        current=lambda: True,
        memory_current=lambda _: True,
    )
    memory = SimpleNamespace(memory_id="committed", summarized_prefix_digest="digest")
    hooks.committed(memory)
    pending = asyncio.create_task(lifecycle.wait(receiver))
    try:
        for _ in range(400):
            if entered.exists():
                break
            await asyncio.sleep(0.005)
        assert entered.exists()
        assert not pending.done(), "next input escaped the preinstalled requirement"
        if cancelled:
            lifecycle.close_scope(operation)
        release.write_text("release")
        await hooks.finish()
        if cancelled:
            with pytest.raises(HookCheckpointError):
                await pending
            assert lifecycle.diagnostics["late_context"] == 1
            assert not lifecycle.claim_handoff(receiver)
        else:
            await pending
            rows = lifecycle.claim_handoff(receiver)
            assert "owned post" in str(rows)
            assert rows[0]["content"].checked_hook_origins()
            assert not lifecycle.claim_handoff(lifecycle.open_scope())
    finally:
        release.write_text("release")
        lifecycle.seal()
        await engine.close()
        await asyncio.gather(pending, return_exceptions=True)


@pytest.mark.asyncio
async def test_successful_manual_post_checkpoints_retire_with_consumed_handoffs():
    import gc
    import weakref

    engine = HookEngine((), lambda *_: True, HookBudgetOwner())
    lifecycle = HookSessionLifecycle(engine, "session")
    retired = []
    try:
        for index in range(3):
            operation = lifecycle.open_scope()
            hooks = CompactionHooks(
                lifecycle,
                operation,
                reason="manual",
                prepare=lambda *_: None,
                current=lambda: True,
                memory_current=lambda _: True,
            )
            hooks.committed(
                SimpleNamespace(
                    memory_id=f"memory-{index}",
                    summarized_prefix_digest="digest",
                    summary_text="private memory",
                )
            )
            await hooks.finish()
            receiver = lifecycle.open_scope()
            await lifecycle.wait(receiver)
            lifecycle.claim_handoff(receiver)
            lifecycle.close_scope(receiver)
            retired.append(weakref.ref(hooks))
            del hooks
            await asyncio.sleep(0)
            gc.collect()
            assert not lifecycle.checkpoints._entries
            assert all(reference() is None for reference in retired)
            assert tuple(lifecycle.checkpoints._parents) == (lifecycle.scope_id,)
    finally:
        lifecycle.seal()
        await engine.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancelled", [False, True])
async def test_previous_manual_handoff_does_not_keep_failed_operation_live(
    tmp_path, cancelled
):
    from Tests.Chat.test_console_context_compaction import _real_selection_controller
    from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
    from tldw_chatbook.Chat.console_context_policy import (
        ConsoleContextPolicyOverrides,
        ContextBudgetMode,
        ContextCompactionMode,
    )
    from tldw_chatbook.Chat.console_runtime import ConsoleRuntime

    db, _repository, controller, store, session, _conversation_id, _ = (
        _real_selection_controller(tmp_path, "h4-fix1-manual")
    )
    runtime = ConsoleRuntime(app=None)
    runtime.set_chat_store(store)
    runtime.set_chat_controller(controller)
    controller._agent_runtime_enabled = False
    from Tests.console_provider_doubles import with_destination

    async def resolve(_selection):
        return with_destination(_resolution())

    controller.provider_gateway.resolve_for_send = resolve
    payloads = []

    async def stream(_resolution, messages, **_kwargs):
        payloads.append(getattr(messages, "messages_payload", messages))
        yield "answer"

    controller.provider_gateway.stream_chat = stream
    for index in range(2):
        store.append_message(
            session.id,
            role=ConsoleMessageRole.USER,
            content="question " + "x " * 450,
            persist=True,
        )
        store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="answer " + "y " * 450,
            persist=True,
        )
    target = store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="current request",
        persist=True,
    )
    store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="current answer",
        persist=True,
    )
    store.set_session_context_policy_overrides(
        session.id,
        ConsoleContextPolicyOverrides(
            budget_mode=ContextBudgetMode.CUSTOM,
            custom_budget_tokens=1800,
            compaction_mode=ContextCompactionMode.OFF,
            summary_max_tokens=100,
        ),
    )
    post = command(
        'print(\'{"version":2,"decision":"pass","context":[{"text":"operation A handoff","lifetime":"turn"}]}\')',
        name="PostCompact",
        effects=["context"],
        required=True,
    )
    runtime.ensure_hooks_v2(session.id, (post,), lambda *_: True)
    try:
        assert (await controller.summarize_up_to(target.id)).accepted
        lifecycle = runtime._hooks_v2_lifecycles[session.id]
        first_owner = lifecycle._handoff[0]
        assert lifecycle.checkpoints.is_current(first_owner)
        seen = []
        entered = asyncio.Event()

        async def failing_summary(**kwargs):
            hooks = kwargs["hooks"]
            seen.append(hooks.owner)
            # Keep a real candidate checkpoint/event mapping so retirement has
            # more to clean than an otherwise empty operation scope.
            await hooks.before(
                kwargs["plan"].auxiliary_messages, kwargs["plan"].requested_output_cap
            )
            entered.set()
            if cancelled:
                await asyncio.Event().wait()
            raise ValueError("controlled second-operation failure")

        controller._compaction_service.summarize_manual = failing_summary
        pending = asyncio.create_task(controller.summarize_up_to(target.id))
        await asyncio.wait_for(entered.wait(), 3)
        if cancelled:
            pending.cancel()
        result = await pending
        assert not result.accepted
        assert len(seen) == 1
        second_owner = seen[0]
        assert lifecycle._handoff[0] == first_owner
        assert not lifecycle.checkpoints.is_current(second_owner)
        assert all(
            owner != second_owner for owner, _ in lifecycle.context._events.values()
        )
        assert all(
            entry.owner_id != second_owner
            for entry in lifecycle.checkpoints._entries.values()
        )
        accepted = await controller.submit_draft("claim A", session_id=session.id)
        assert accepted.accepted, accepted
        assert len(payloads) == 1
        assert str(payloads[0]).count("operation A handoff") == 1
        assert not lifecycle.checkpoints.is_current(first_owner)
        assert not lifecycle.checkpoints.is_current(second_owner)
        assert lifecycle._handoff is None
        assert all(
            owner != second_owner for owner, _ in lifecycle.context._events.values()
        )
    finally:
        await runtime.close_hooks_v2()
        await runtime.dispose()
        db.close()
