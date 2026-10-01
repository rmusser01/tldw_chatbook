# Expanded shared hook runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver all approved v2 hook events, effects, command/MCP execution and bounded continuations while preserving legacy hooks.

**Architecture:** Add focused hooks_v2 modules beside the legacy engine. Actual tool, scheduler, child and compaction owners supply event boundaries; required effects use admission checkpoints and ordinary authorization.

**Tech Stack:** Python 3.12+, Textual 8.2.8, Pydantic 2, private SQLite, httpx, portalocker and existing trust/credential primitives.

**Spec:** [Plugin spec](../specs/2026-09-15-managed-plugins-design.md) and [hook spec](../specs/2026-09-15-expanded-hook-runtime-design.md). Read both; this plan covers its assigned subsystem within the complete [delivery plan](2026-09-15-managed-plugins-delivery.md).

## Global Constraints

- Python >=3.12; current checkout pins Textual 8.2.8, Pydantic >=2.4,<3 and portalocker 3.2.0. Preserve these pins; use the existing SQLite/httpx/crypto/keyring seams.
- One installation and selected revision exist per user-data directory.
- Global default activation starts disabled. Importing a catalog does not install or enable its entries.
- All approved hook additions are in scope. Delivery stages are ordering, not deferral.
- No parallel agent/permission runtime, package build/install execution, vendor grants or per-workspace package versions.
- Required constraints never disappear because parsing, configuration, hooks or persistence fail. Native/foreign instructions remain attributed untrusted context.
- Package activation grants no filesystem binding, tool permission, network credential or trusted project status. Console local tools retain scratch/explicit-binding authority.
- Apply current authority before injection/launch/dispatch/result acceptance. Workspace disable preserves other authorized scopes; namespace marker changes alone do not cancel all work.
- Full-suite runs require explicit user opt-in. Every task runs its exact feature/regression files and a successful control on the same production entry.
- Implementation uses an isolated execution worktree and profile; do not repoint a shared editable environment. Verify child interpreter package provenance as well as pytest cwd.
- Every To Do task must move In Progress and receive its Implementation Plan via Backlog CLI before code changes. Add Implementation Notes and mark Done only after its acceptance criteria, review, targeted tests and static checks pass.

ADR required: yes
ADR paths: [ADR-162](../../../backlog/decisions/162-managed-agent-plugins.md); [ADR-163](../../../backlog/decisions/163-expanded-console-hook-runtime.md)
Reason: Implements the accepted storage, trust, runtime and UI contracts. No additional ADR is needed unless implementation changes one of those decisions.

---

## Execution and evidence

This is an implementation plan, not implemented code or passing runtime evidence.
The code blocks below are small invariant/RED-test sketches, not a complete
implementation to paste blindly. Each task must also exercise its named production
entry and failure/control matrix. Preserve the stated interfaces across tasks;
when current library behavior contradicts a sketch, establish the real RED failure
and correct the sketch/test before implementation, as the repository's
[testing-evidence lesson](../../../backlog/docs/lessons-testing-evidence.md) requires.

Read [live verification](../../../backlog/docs/lessons-live-verification.md) before
running the app. Isolate config, data, credential and child-process roots before
importing runtime code, and verify the isolation. Use sys.executable for controlled
children. A disabled path failing from an unrelated event-loop error is not evidence.

Each task is one independently reviewable deliverable. Its checklist is the
sequence of small test/implementation increments; repeat the RED/GREEN cycle for
each listed failure/control case. Do not implement the entire subsystem before
running its first integration test. File roles and public contracts below define
the decomposition; no unrelated broad refactoring is part of these plans.

## File ownership map

| Task | New implementation units | Existing integration boundaries |
| --- | --- | --- |
| H1 | `tldw_chatbook/Agents/hooks_v2/__init__.py`, `tldw_chatbook/Agents/hooks_v2/models.py`, `tldw_chatbook/Agents/hooks_v2/validation.py`, `tldw_chatbook/Agents/hooks_v2/matching.py` | `tldw_chatbook/Agents/run_hooks.py`, `tldw_chatbook/config.py` |
| H2 | `tldw_chatbook/Agents/hooks_v2/command_executor.py`, `tldw_chatbook/Agents/hooks_v2/budgets.py`, `tldw_chatbook/Agents/hooks_v2/engine.py`, `tldw_chatbook/Agents/hooks_v2/ownership.py` | `tldw_chatbook/Chat/console_runtime.py` |
| H3 | `tldw_chatbook/Agents/hooks_v2/tool_pipeline.py`, `tldw_chatbook/Agents/hooks_v2/checkpoints.py` | `tldw_chatbook/Agents/agent_runtime.py`, `tldw_chatbook/Agents/agent_service.py`, `tldw_chatbook/Chat/console_agent_bridge.py`, `tldw_chatbook/Chat/console_chat_controller.py` |
| H4 | `tldw_chatbook/Agents/hooks_v2/lifecycle.py`, `tldw_chatbook/Agents/hooks_v2/context.py` | `tldw_chatbook/Chat/console_runtime.py`, `tldw_chatbook/Chat/console_chat_controller.py`, `tldw_chatbook/Chat/console_agent_bridge.py`, `tldw_chatbook/Chat/console_context_compaction.py`, `tldw_chatbook/Agents/agent_service.py` |
| H5 | `tldw_chatbook/Agents/hooks_v2/continuations.py` | `tldw_chatbook/Chat/console_prompt_queue.py`, `tldw_chatbook/Chat/console_prompt_queue_coordinator.py`, `tldw_chatbook/Chat/console_runtime.py`, `tldw_chatbook/Chat/console_interrupt_rounds.py` |
| H6 | `tldw_chatbook/Agents/hooks_v2/mcp_executor.py`, `tldw_chatbook/Agents/hooks_v2/mcp_results.py`, `tldw_chatbook/Agents/hooks_v2/causality.py` | `tldw_chatbook/Agents/hooks_v2/engine.py`, `tldw_chatbook/Agents/hooks_v2/lifecycle.py` |

## H1: Validate explicit v2 hook definitions and effects

**Backlog:** [TASK-32676](../../../backlog/tasks/task-32676%20-%20Validate-explicit-v2-hook-definitions-and-effects.md). **Requires:** [TASK-32645](../../../backlog/tasks/task-32645%20-%20Design-managed-plugins-and-expanded-hook-runtime.md).

**Deliverable:** Make richer hook declarations inspectable and deterministic while preserving the legacy six-event configuration.

**Files:**

- Create: `tldw_chatbook/Agents/hooks_v2/__init__.py`
- Create: `tldw_chatbook/Agents/hooks_v2/models.py`
- Create: `tldw_chatbook/Agents/hooks_v2/validation.py`
- Create: `tldw_chatbook/Agents/hooks_v2/matching.py`
- Modify: `tldw_chatbook/Agents/run_hooks.py`
- Modify: `tldw_chatbook/config.py`
- Test: `Tests/Agents/test_hooks_v2_validation.py`
- Test: `Tests/Agents/test_run_hooks.py`

**Interfaces**

- Consumes: Legacy RunHooksConfig/load_hooks_config remain their own schema. The companion hook spec defines the complete event/effect matrix.
- Produces: parse_handlers(value: object) -> tuple[HookHandler, ...]; parse_result(value: object, handler: HookHandler) -> HookResult; handler_phase(handler: HookHandler) -> str. Frozen Pydantic HookHandler carries the exact section 2.1 fields. HookEvent carries section 2.2 host-owned identities and event data; HookResult carries only section 2.3 fields. validation never imports Plugins. User config and owned plugin definitions normalize into these same types.

- [x] **1. Create the first behavioral test and its local fixtures.** Use the fixture contract in the implementation increments below; initial import/behavior must fail for the missing feature.

```python
def test_mixed_transformer_runs_in_transformation_phase():
    from tldw_chatbook.Agents.hooks_v2.validation import parse_handlers, handler_phase
    handlers = parse_handlers([{"id": "rewrite", "event": "PreToolUse", "type": "command", "argv": ["review"], "effects": ["updated_input", "deny"]}])
    assert handler_phase(handlers[0]) == "transform"
    guards = parse_handlers([{"id": "guard", "event": "PreToolUse", "type": "command", "argv": ["review"], "effects": ["deny"]}])
    assert handler_phase(guards[0]) == "validate"
```

- [x] **2. Establish RED through the intended entry.** Run `python -m pytest Tests/Agents/test_hooks_v2_validation.py -q`. Expected: the named new behavior fails, while any positive precondition/control succeeds. Resolve test harness/API errors before changing production code.

- [x] **3. Implement the smallest invariant, then integrate the real owner.** This kernel states the ordering/data rule; the following increments supply the complete behavior and limits.

```python
def classify_effects(effects: frozenset[str], required: bool) -> str:
    if "updated_input" in effects:
        return "transform"
    if "deny" in effects or required:
        return "validate"
    return "context" if effects else "observe"
```

  - [x] 3.1. Add the new namespace without moving the legacy engine. Parse hooks.handler separately and preserve legacy hooks.hook logging/output contracts exactly.
  - [x] 3.2. Implement closed event/type/effect validation, bounded JSON parsing, structured matchers and one-pass typed MCP templates. Reject type-inappropriate fields, duplicate keys and ownership fields supplied by output.
  - [x] 3.3. Implement required/require_context/dependency classification and phase ordering. The small classification kernel applies only after event validation; dependency-controlled effect-free handlers also use the controlling path.
  - [x] 3.4. Add positive and negative table cases for all 13 events, unsupported fields, required teardown/approval/Stop and invalid context lifetimes; check the master switch suppresses execution without removing requirements.

**Failure and successful-control matrix:** Empty-success versus required context, malformed result, every mixed PreToolUse effect set, oversized payload, missing typed template path and legacy unknown/non-JSON stdout controls.

- [x] **4. Establish GREEN and preserve the neighboring path.** Run the exact files below. Expected: all execute and pass, with no silent coroutine/platform skips used as qualification. Inspect real resources/output, not source-string matches.

```bash
python -m pytest Tests/Agents/test_hooks_v2_validation.py Tests/Agents/test_run_hooks.py -q
```

- [x] **5. Review, record evidence and commit the task.** Update its ACs/notes and the relevant authoring/operation documentation; record platform limits. Run `git diff --check`, Python syntax checks and the verification-environment formatter/linter on the changed Python files as specified in the delivery plan. Stage the exact task-owned files, including any test fixtures and generated CSS, and commit; never stage unrelated work.

```bash
backlog task task-32676 --plain
git diff --check
```

## H2: Execute v2 command hooks with bounded resource ownership

**Backlog:** [TASK-32677](../../../backlog/tasks/task-32677%20-%20Execute-v2-command-hooks-with-bounded-resource-ownership.md). **Requires:** [TASK-32676](../../../backlog/tasks/task-32676%20-%20Validate-explicit-v2-hook-definitions-and-effects.md).

**Deliverable:** Run reviewed hook commands with fair application-wide limits and cancellation that retains ownership until actual cleanup.

**Files:**

- Create: `tldw_chatbook/Agents/hooks_v2/command_executor.py`
- Create: `tldw_chatbook/Agents/hooks_v2/budgets.py`
- Create: `tldw_chatbook/Agents/hooks_v2/engine.py`
- Create: `tldw_chatbook/Agents/hooks_v2/ownership.py`
- Modify: `tldw_chatbook/Chat/console_runtime.py`
- Test: `Tests/Agents/test_hooks_v2_execution.py`
- Test: `Tests/Agents/test_hooks_v2_budgets.py`
- Test: `Tests/Chat/test_console_runtime_shutdown.py`

**Interfaces**

- Consumes: H1 normalized handlers/events/results; platform-qualified process termination already used by Agents/run_hooks.py.
- Produces: HookEngine(definitions: tuple[HookHandler, ...], authority_check: Callable, budget_owner: HookBudgetOwner).fire(event: HookEvent) -> HookEventOutcome; async fire_async(event: HookEvent) -> HookEventOutcome; notify(event: HookEvent) -> bool; async close() -> None. HookEventOutcome records accepted effects, omissions, failures and outstanding cleanup. One application HookBudgetOwner supplies reserve(runtime_id: str, observation: bool) and exact execution/ticket/queue counters; context-manager tickets survive suspension. Inject HookProcessOwner with reserve_launch(event: HookEvent) -> str, publish_process(token: str, provenance: dict) -> None and settle_process(token: str, confirmed: bool) -> None. This protocol is defined in hooks_v2/ownership.py without importing Plugins; plugin composition supplies F2 runtime ownership, while standalone handlers retain their existing host owner.

- [x] **1. Create the first behavioral test and its local fixtures.** Use the fixture contract in the implementation increments below; initial import/behavior must fail for the missing feature.

```python
import pytest

@pytest.mark.asyncio
async def test_command_success_and_disposal_use_same_entry(command_hook_case):
    case = command_hook_case
    assert (await case.engine.fire_async(case.event)).succeeded
    await case.engine.close()
    assert not (await case.engine.fire_async(case.event)).succeeded
    assert case.live_children() == 0
```

- [x] **2. Establish RED through the intended entry.** Run `python -m pytest Tests/Agents/test_hooks_v2_execution.py -q`. Expected: the named new behavior fails, while any positive precondition/control succeeds. Resolve test harness/API errors before changing production code.

- [x] **3. Implement the smallest invariant, then integrate the real owner.** This kernel states the ordering/data rule; the following increments supply the complete behavior and limits.

```python
from collections import deque

class ReadyRuntimes:
    def __init__(self):
        self.ready = deque()
    def add(self, runtime_id: str):
        if runtime_id not in self.ready:
            self.ready.append(runtime_id)
    def next(self) -> str:
        return self.ready.popleft()
```

  - [x] 3.1. Build command_hook_case locally from a v2 command using sys.executable, a complete HookEvent and the real HookEngine; live_children reads retained owned process handles, not a fake idle counter.
  - [x] 3.2. Capture bounded stdin/stdout/stderr during I/O; run sync calls only on agent threads and use async entry from the event loop. Keep optional observation workers separate from controlling effects.
  - [x] 3.3. Implement app-owned reservation tickets, round-robin runtimes and FIFO per-runtime delivery. Release scarce execution slots during approval/nested waits while retaining lifetime tickets; refuse overflow under the event policy.
  - [x] 3.4. Retain launch and process ownership through cancel/timeout/reap. Close queue admission before draining; notification is at most 3 seconds and post-kill reap at most 5 seconds, with unresolved children still counted.

**Failure and successful-control matrix:** Eight app/four runtime execution slots, 64/16 lifetime tickets, 128/64 observation deliveries with 8/4 workers, multiple runtimes, fairness, cancelled launch, caller cancellation during close, bounded output capture and positive real subprocess controls on each supported platform.

- [x] **4. Establish GREEN and preserve the neighboring path.** Run the exact files below. Expected: all execute and pass, with no silent coroutine/platform skips used as qualification. Inspect real resources/output, not source-string matches.

```bash
python -m pytest Tests/Agents/test_hooks_v2_execution.py Tests/Agents/test_hooks_v2_budgets.py Tests/Chat/test_console_runtime_shutdown.py -q
```

- [x] **5. Review, record evidence and commit the task.** Update its ACs/notes and the relevant authoring/operation documentation; record platform limits. Run `git diff --check`, Python syntax checks and the verification-environment formatter/linter on the changed Python files as specified in the delivery plan. Stage the exact task-owned files, including any test fixtures and generated CSS, and commit; never stage unrelated work.

```bash
backlog task task-32677 --plain
git diff --check
```

## H3: Integrate hook input transformations and post-event barriers

**Backlog:** [TASK-32678](../../../backlog/tasks/task-32678%20-%20Integrate-hook-input-transformations-and-post-event-barriers.md). **Requires:** [TASK-32677](../../../backlog/tasks/task-32677%20-%20Execute-v2-command-hooks-with-bounded-resource-ownership.md).

**Deliverable:** Apply structured hook effects at the real tool boundary without bypassing review or allowing later model steps to overtake required context.

**Files:**

- Create: `tldw_chatbook/Agents/hooks_v2/tool_pipeline.py`
- Create: `tldw_chatbook/Agents/hooks_v2/checkpoints.py`
- Modify: `tldw_chatbook/Agents/agent_runtime.py`
- Modify: `tldw_chatbook/Agents/agent_service.py`
- Modify: `tldw_chatbook/Chat/console_agent_bridge.py`
- Modify: `tldw_chatbook/Chat/console_chat_controller.py`
- Test: `Tests/Agents/test_hooks_v2_tool_pipeline.py`
- Test: `Tests/Agents/test_hooks_v2_post_checkpoints.py`
- Test: `Tests/Agents/test_post_tool_dispatch_hook.py`
- Test: `Tests/Chat/test_console_run_hooks_regressions.py`

**Interfaces**

- Consumes: H2 engine and existing guard_tool_calls/post_tool_call dependency injection before approval exemptions and result truncation.
- Produces: async prepare_tool(event: HookEvent, engine: HookEngine) -> PreparedHookCall returns exact original/final arguments, effects and definition/owner generations. HookCheckpointStore.begin(event: HookEvent, requirements: tuple[str, ...]) -> str; accept(token: str, result: HookEventOutcome) -> None; fail(token: str, reason: str) -> None; assert_next_input_allowed(owner_id: str) -> None. Checkpoints enter pending before completion publication; accepted context and release use one synchronized operation.

- [x] **1. Create the first behavioral test and its local fixtures.** Use the fixture contract in the implementation increments below; initial import/behavior must fail for the missing feature.

```python
import pytest

@pytest.mark.asyncio
async def test_next_model_step_cannot_overtake_required_post_hook(post_hook_case):
    case = post_hook_case
    await case.dispatch_and_hold_required_post_hook()
    assert case.tool_result_visible()
    assert not case.next_input_allowed()
    case.release_post_hook_with_context("reviewed")
    await case.checkpoint_settled.wait()
    assert case.next_input_allowed()
    assert case.next_input_context() == "reviewed"
```

- [x] **2. Establish RED through the intended entry.** Run `python -m pytest Tests/Agents/test_hooks_v2_tool_pipeline.py -q`. Expected: the named new behavior fails, while any positive precondition/control succeeds. Resolve test harness/API errors before changing production code.

- [x] **3. Implement the smallest invariant, then integrate the real owner.** This kernel states the ordering/data rule; the following increments supply the complete behavior and limits.

```python
def freeze_candidate(arguments: dict) -> bytes:
    import json
    return json.dumps(arguments, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
```

  - [x] 3.1. Build post_hook_case around the real agent dispatch callback and next-input guard with deterministic events; define its six scenario methods and checkpoint_settled event in the same test module. Do not simulate next-input readiness with an unrelated helper.
  - [x] 3.2. Run transformers in order and validate every replacement; freeze the candidate, run final validators and context-only handlers, then bind ordinary permission review to that candidate. Revalidate changed arguments before exemptions/approval/dispatch.
  - [x] 3.3. Extend the post-tool consumer seam to await/control required effects while preserving old optional callbacks. Install pending checkpoints before any completion consumer can admit a model call or root settlement.
  - [x] 3.4. Handle PostToolUse then known-error PostToolUseFailure separately. Commit accepted context and release together; cancelled/revoked results cannot satisfy requirements, and already-settled tool effects are never replayed.

**Failure and successful-control matrix:** Transformer A then B then final guard C; stale approval hash; preauthorized Canvas call; durable invocation; context-only handler; required effect-free completion; tool error, not-dispatched denial and uncertain remote cancellation; concurrent child/model settlement; valid success on every refusal entry.

- [x] **4. Establish GREEN and preserve the neighboring path.** Run the exact files below. Expected: all execute and pass, with no silent coroutine/platform skips used as qualification. Inspect real resources/output, not source-string matches.

```bash
python -m pytest Tests/Agents/test_hooks_v2_tool_pipeline.py Tests/Agents/test_hooks_v2_post_checkpoints.py Tests/Agents/test_post_tool_dispatch_hook.py Tests/Chat/test_console_run_hooks_regressions.py -q
```

- [x] **5. Review, record evidence and commit the task.** Update its ACs/notes and the relevant authoring/operation documentation; record platform limits. Run `git diff --check`, Python syntax checks and the verification-environment formatter/linter on the changed Python files as specified in the delivery plan. Stage the exact task-owned files, including any test fixtures and generated CSS, and commit; never stage unrelated work.

```bash
backlog task task-32678 --plain
git diff --check
```

## H4: Wire session child and compaction hook boundaries

**Backlog:** [TASK-32679](../../../backlog/tasks/task-32679%20-%20Wire-session-child-and-compaction-hook-boundaries.md). **Requires:** [TASK-32678](../../../backlog/tasks/task-32678%20-%20Integrate-hook-input-transformations-and-post-event-barriers.md).

**Deliverable:** Expose the approved lifecycle events where their effects can be applied safely to the owning run and context.

**Files:**

- Create: `tldw_chatbook/Agents/hooks_v2/lifecycle.py`
- Create: `tldw_chatbook/Agents/hooks_v2/context.py`
- Modify: `tldw_chatbook/Chat/console_runtime.py`
- Modify: `tldw_chatbook/Chat/console_chat_controller.py`
- Modify: `tldw_chatbook/Chat/console_agent_bridge.py`
- Modify: `tldw_chatbook/Chat/console_context_compaction.py`
- Modify: `tldw_chatbook/Agents/agent_service.py`
- Test: `Tests/Chat/test_hooks_v2_lifecycle.py`
- Test: `Tests/Agents/test_hooks_v2_child_events.py`
- Test: `Tests/Chat/test_hooks_v2_compaction.py`
- Test: `Tests/Chat/test_console_context_compaction.py`

**Interfaces**

- Consumes: H3 checkpoints/effect batches and the actual Console admission, child-draft and ContextCompactionService.compact boundaries.
- Produces: HookSessionLifecycle.reserve(event: HookEvent) -> str; async initialize(token: str) -> HookEventOutcome; publish(token: str) -> None; cancel(token: str) -> None. ContextLedger.accept(event: HookEvent, result: HookEventOutcome) -> None; blocks(owner_id: str, boundary: str) -> tuple[dict, ...]; close(owner_id: str) -> None. Context records use host-assigned origin and runtime/turn/child lifetime, with no promotion to system authority.

- [x] **1. Create the first behavioral test and its local fixtures.** Use the fixture contract in the implementation increments below; initial import/behavior must fail for the missing feature.

```python
import pytest

@pytest.mark.asyncio
async def test_failed_initialization_never_publishes_a_root_turn(session_case):
    case = session_case
    case.refuse_required_initialization()
    await case.submit()
    assert case.admitted_turn_count() == 0
    assert case.root_stop_count() == 0
    case.allow_required_initialization()
    await case.submit()
    assert case.admitted_turn_count() == 1
```

- [x] **2. Establish RED through the intended entry.** Run `python -m pytest Tests/Chat/test_hooks_v2_lifecycle.py -q`. Expected: the named new behavior fails, while any positive precondition/control succeeds. Resolve test harness/API errors before changing production code.

- [x] **3. Implement the smallest invariant, then integrate the real owner.** This kernel states the ordering/data rule; the following increments supply the complete behavior and limits.

```python
def narrow_tools(parent: frozenset[str], proposal: frozenset[str]) -> frozenset[str]:
    if not proposal <= parent:
        raise ValueError("hook_child_tools_widened")
    return proposal
```

  - [x] 3.1. Define session_case around the real Console submit/controller/runtime using controlled command hooks; count actual accepted turns and root Stop events. Use base input validation and a provisional run-capacity reservation before SessionStart.
  - [x] 3.2. Publish a live session only after controlling initialization succeeds; dependency-only failure leaves independent capabilities eligible. Replace immutable hook sets only at an idle boundary; tab focus and archived history never fire initialization.
  - [x] 3.3. Place SubagentStart after inherited restrictions and before child admission, and SubagentStop before the active parent next-input checkpoint can release. Narrow tools/budgets; late parentless context is diagnosed and discarded.
  - [x] 3.4. Add PreCompact to real candidate input, PostCompact after successful commit and before next input. Keep runtime context separately owned through compaction; failed required PreCompact aborts without deleting history, and failed PostCompact never rolls back a committed summary.

**Failure and successful-control matrix:** Manual versus scheduled submission, explicit required versus dependency-only failure, master-off, cancelled initialization, session replacement, child tool/model restrictions, late child settlement, real compaction with positive context and no duplicate summary blocks.

- [x] **4. Establish GREEN and preserve the neighboring path.** Run the exact files below. Expected: all execute and pass, with no silent coroutine/platform skips used as qualification. Inspect real resources/output, not source-string matches.

```bash
python -m pytest Tests/Chat/test_hooks_v2_lifecycle.py Tests/Agents/test_hooks_v2_child_events.py Tests/Chat/test_hooks_v2_compaction.py Tests/Chat/test_console_context_compaction.py -q
```

- [x] **5. Review, record evidence and commit the task.** Update its ACs/notes and the relevant authoring/operation documentation; record platform limits. Run `git diff --check`, Python syntax checks and the verification-environment formatter/linter on the changed Python files as specified in the delivery plan. Stage the exact task-owned files, including any test fixtures and generated CSS, and commit; never stage unrelated work.

```bash
backlog task task-32679 --plain
git diff --check
```

## H5: Schedule bounded Stop continuations and teardown

**Backlog:** [TASK-32680](../../../backlog/tasks/task-32680%20-%20Schedule-bounded-Stop-continuations-and-teardown.md). **Requires:** [TASK-32679](../../../backlog/tasks/task-32679%20-%20Wire-session-child-and-compaction-hook-boundaries.md).

**Deliverable:** Allow useful automatic follow-up while preserving user priority, cancellation and finite scheduler ownership.

**Files:**

- Create: `tldw_chatbook/Agents/hooks_v2/continuations.py`
- Modify: `tldw_chatbook/Chat/console_prompt_queue.py`
- Modify: `tldw_chatbook/Chat/console_prompt_queue_coordinator.py`
- Modify: `tldw_chatbook/Chat/console_runtime.py`
- Modify: `tldw_chatbook/Chat/console_interrupt_rounds.py`
- Test: `Tests/Chat/test_hooks_v2_continuations.py`
- Test: `Tests/Chat/test_hooks_v2_teardown.py`
- Test: `Tests/Chat/test_console_viewless_hooks.py`
- Test: `Tests/Chat/test_console_interrupt_rounds.py`

**Interfaces**

- Consumes: H4 lifecycle and existing queue/settlement/interrupt hosts; use their run budgets and cancellation owner rather than a recursive model call.
- Produces: ContinuationPolicy.permits(*, admitted_turns: int, elapsed_seconds: float, foreground_waiting: bool, revoked: bool, draining: bool, closed: bool, vetoed: bool) -> bool. async schedule_continuation(parent_turn_id: str, event_id: str, proposals: tuple[HookResult, ...]) -> str | None uses one durable scheduler deduplication identity. Reuse existing scheduler checkpoint recovery for uncertain admission.

- [x] **1. Create the first behavioral test and its local fixtures.** Use the fixture contract in the implementation increments below; initial import/behavior must fail for the missing feature.

```python
def test_user_work_and_chain_cap_prevent_continuation():
    from tldw_chatbook.Agents.hooks_v2.continuations import ContinuationPolicy
    values = dict(admitted_turns=0, elapsed_seconds=0.0, foreground_waiting=False, revoked=False, draining=False, closed=False, vetoed=False)
    assert ContinuationPolicy.permits(**values)
    assert not ContinuationPolicy.permits(**{**values, "foreground_waiting": True})
    assert not ContinuationPolicy.permits(**{**values, "admitted_turns": 3})
```

- [x] **2. Establish RED through the intended entry.** Run `python -m pytest Tests/Chat/test_hooks_v2_continuations.py -q`. Expected: the named new behavior fails, while any positive precondition/control succeeds. Resolve test harness/API errors before changing production code.

- [x] **3. Implement the smallest invariant, then integrate the real owner.** This kernel states the ordering/data rule; the following increments supply the complete behavior and limits.

```python
def continuation_key(parent_turn_id: str, event_id: str) -> tuple[str, str]:
    return parent_turn_id, event_id
```

  - [x] 3.1. Use the real queue coordinator for Stop settlement and combine valid proposals in stable order into one next turn. Enforce 4 KiB/message and 8 KiB/combined with a whole-proposal refusal, not truncation.
  - [x] 3.2. Reserve deduplication identity atomically with scheduler admission. Keep inherited parent/workspace budgets, manual-origin distinction and foreground priority; never hold a stale proposal behind queued user work.
  - [x] 3.3. Wire Interrupt only after immediate admission sealing. Still-authorized handlers get their bounded observation window; plugin revocation suppresses affected callbacks and host cleanup cannot be vetoed.
  - [x] 3.4. Exercise mounted and viewless shutdown through the shared interrupt host; settle cancelled waiters independently of retained process reaping and keep metadata-only diagnostics.

**Failure and successful-control matrix:** Two simultaneous Stop callbacks, lost scheduler response, process restart, uncertain dispatch, pending plugin update, user queue arrival, continuation veto, 120-second cap, oversized combined input and repeated cancellation during cleanup.

- [x] **4. Establish GREEN and preserve the neighboring path.** Run the exact files below. Expected: all execute and pass, with no silent coroutine/platform skips used as qualification. Inspect real resources/output, not source-string matches.

```bash
python -m pytest Tests/Chat/test_hooks_v2_continuations.py Tests/Chat/test_hooks_v2_teardown.py Tests/Chat/test_console_viewless_hooks.py Tests/Chat/test_console_interrupt_rounds.py -q
```

- [x] **5. Review, record evidence and commit the task.** Update its ACs/notes and the relevant authoring/operation documentation; record platform limits. Run `git diff --check`, Python syntax checks and the verification-environment formatter/linter on the changed Python files as specified in the delivery plan. Stage the exact task-owned files, including any test fixtures and generated CSS, and commit; never stage unrelated work.

```bash
backlog task task-32680 --plain
git diff --check
```

## H6: Invoke MCP-backed hooks through normal tool authority

**Backlog:** [TASK-32685](../../../backlog/tasks/task-32685%20-%20Invoke-MCP-backed-hooks-through-normal-tool-authority.md). **Requires:** [TASK-32680](../../../backlog/tasks/task-32680%20-%20Schedule-bounded-Stop-continuations-and-teardown.md), [TASK-32684](../../../backlog/tasks/task-32684%20-%20Expose-owned-plugin-MCP-tools-with-scoped-connection-leases.md).

**Deliverable:** Complete hook interoperability with MCP handlers that preserve recursion, initialization and permission boundaries.

**Files:**

- Create: `tldw_chatbook/Agents/hooks_v2/mcp_executor.py`
- Create: `tldw_chatbook/Agents/hooks_v2/mcp_results.py`
- Create: `tldw_chatbook/Agents/hooks_v2/causality.py`
- Modify: `tldw_chatbook/Agents/hooks_v2/engine.py`
- Modify: `tldw_chatbook/Agents/hooks_v2/lifecycle.py`
- Test: `Tests/Agents/test_hooks_v2_mcp_results.py`
- Test: `Tests/Agents/test_hooks_v2_mcp_execution.py`
- Test: `Tests/Plugins/test_mcp_initialization.py`

**Interfaces**

- Consumes: M1 typed MCPToolResult, M4 normal scoped owned invocation and H1-H5 event/effect/budget contracts.
- Produces: normalize_hook_result(result: MCPToolResult, handler: HookHandler) -> HookResult checks errors before effects. MCPHookExecutor.invoke(event: HookEvent, handler: HookHandler) -> Awaitable[HookEventOutcome] passes ordinary tool authorization. CausalChain.enter(event_id: str, handler_id: str, tool_id: str) -> CausalChain refuses a repeated identity or depth above four. Initialization eligibility includes required guards on prospective calls.

- [ ] **1. Create the first behavioral test and its local fixtures.** Use the fixture contract in the implementation increments below; initial import/behavior must fail for the missing feature.

```python
def test_error_payload_cannot_return_a_pass(handler, typed_result):
    import pytest
    from tldw_chatbook.Agents.hooks_v2.mcp_results import normalize_hook_result
    error = typed_result(is_error=True, structured={"version": 2, "decision": "pass"})
    with pytest.raises(ValueError):
        normalize_hook_result(error, handler)
    good = typed_result(is_error=False, structured={"version": 2, "decision": "pass"})
    assert normalize_hook_result(good, handler).decision == "pass"
```

- [ ] **2. Establish RED through the intended entry.** Run `python -m pytest Tests/Agents/test_hooks_v2_mcp_results.py -q`. Expected: the named new behavior fails, while any positive precondition/control succeeds. Resolve test harness/API errors before changing production code.

- [ ] **3. Implement the smallest invariant, then integrate the real owner.** This kernel states the ordering/data rule; the following increments supply the complete behavior and limits.

```python
def reject_tool_error(result):
    if result.transport_error is not None or result.is_error:
        raise ValueError("hook_mcp_execution_failed")
```

  - [ ] 3.1. Define handler and typed_result fixtures in the result test module using real HookHandler and M1 MCPToolResult constructors. Retain complete fields and bound encoded content/structured/meta payload before normalization.
  - [ ] 3.2. Implement the exact accepted result forms: structured-only, one JSON text object, exact mirrored structured/text, or empty success. Reject conflicting mirrors, extra blocks, duplicate keys, invalid structured fallback and non-boolean isError; require_context rejects empty success.
  - [ ] 3.3. Dispatch typed input templates through the ordinary tool invoker with fresh schema/definition/authority checks. Suspend scarce execution slots during approval/nested waits while retaining tickets; approval observations and teardown cannot prompt.
  - [ ] 3.4. Resolve provisional initialization on an independently eligible connected capability view, with static dependency/guard-cycle rejection plus dynamic causal depth checks. Disconnected servers stay unready; initialization never creates temporary rights or bypasses guards.

**Failure and successful-control matrix:** Both real stdio and controlled Streamable HTTP; tool error containing pass; all accepted/invalid result forms; MCP-to-hook-to-MCP cycles; missing connection, stale mapping, revoked A with live B, two pending post-event requirements and known/unknown cancellation outcomes.

- [ ] **4. Establish GREEN and preserve the neighboring path.** Run the exact files below. Expected: all execute and pass, with no silent coroutine/platform skips used as qualification. Inspect real resources/output, not source-string matches.

```bash
python -m pytest Tests/Agents/test_hooks_v2_mcp_results.py Tests/Agents/test_hooks_v2_mcp_execution.py Tests/Plugins/test_mcp_initialization.py -q
```

- [ ] **5. Review, record evidence and commit the task.** Update its ACs/notes and the relevant authoring/operation documentation; record platform limits. Run `git diff --check`, Python syntax checks and the verification-environment formatter/linter on the changed Python files as specified in the delivery plan. Stage the exact task-owned files, including any test fixtures and generated CSS, and commit; never stage unrelated work.

```bash
backlog task task-32685 --plain
git diff --check
```

## Shared hook limits

The application counters belong to one host owner and include cleanup-pending work.

| Resource | Default / ceiling | Behavior |
| --- | --- | --- |
| Command or MCP execution | 10 s / 60 s per handler | Timeout follows controlling event policy. |
| Effectful event execution | 60 s total active execution | Stop launching further handlers; fail controlling required effects. |
| Effectful event wall time | 180 s total including queue and all approval waits | Settle the required event as failed; no repeated wait can extend the deadline. |
| Interactive MCP approval wait | 120 s separate wall-time ceiling | Cancel pending hook call; deny required parent guard, otherwise diagnose omission. |
| Interrupt/SessionEnd notification | 1 s / 3 s wall time per event, including queue and execution | End notification/output acceptance at deadline; no new approval, connection or continuation; initiate scoped host cleanup. |
| Post-kill reap | 5 s maximum host cleanup allowance after notification/execution ends | Outside the handler/event window; preserve cleanup-pending ownership if unresolved. |
| Input envelope | 1 MiB UTF-8, depth 32, 16,384 JSON nodes | Required guard rejects the parent operation whole; optional observation drops with diagnostic. |
| Structured stdout / MCP tool-result payload | 16 KiB UTF-8 | Bound capture; MCP includes content, structuredContent and metadata before v2 normalization. Overflow fails the handler; protocol framing is independently bounded. |
| stderr retained | 4 KiB UTF-8 | Truncate diagnostic capture with marker; never parse as effects. |
| Context | 4 KiB/block; 16 KiB/event; companion 32 KiB/send aggregate | Reject blocks/effect batch whole when required; no silent constraint truncation. |
| Concurrent effectful executions | 4/runtime; 8/application across all v2 sessions | FIFO within a runtime, round-robin admission across eligible runtimes; cancellation releases reservations correctly. |
| Outstanding effectful handlers | 16/runtime; 64/application, including queued, active and suspended nested/approval continuations | Reserve one lifetime ticket per handler; refuse overflow under the event's required/optional failure policy; never dispatch an unguarded parent call. |
| Observation queue | 64 pending handler deliveries/runtime; 128/application; 4 active workers/runtime and 8/application | Each matching handler is one delivery. Drop newest optional delivery with a count; fair round-robin admission across runtimes; required handlers never enter this lossy queue. |
| Definitions | 64 hooks/installation; 256 active/runtime | Refuse activation of the overflowing set; no arbitrary tail omission. |
| Matcher patterns | 32/handler; 256 characters/pattern | Reject invalid definition. |
| Causal depth | 4 nested hook/tool levels | Cycle/limit failure as section 6; no guard bypass. |
| Stop continuations | 3 turns and 120 s wall time per chain | End chain; tighter inherited budgets always win. |
| Continuation message | 4 KiB; 8 KiB combined/settlement | Reject oversized proposal/combined turn. |
| Context/decision reason display | 1,000 characters for reason | Sanitize diagnostic view; decision remains structured. |

### R70 — Actual provisional MCP admission

H6 builds its private prospective MCP invocation view through the real validated Console admission path before SessionStart, using the resolved configuration and existing workspace/parent, registry and reservation owners. Host injection is an extension seam; the normal caller must supply the context. The view remains unadvertised and grants no temporary permission or accepted root turn. Missing native dependency declarations remain unavailable until I1 supplies the graph. Later invocations use the actual AgentService run context and existing ToolHookRun guards and post-event settlement.

Initialization uses only independently eligible, already-connected capabilities. Enforce this narrowing at the actual standalone and owned connection/dispatch boundaries so an intervening disconnect refuses instead of reconnecting. Preserve exact current authority, request provenance and lifetime ownership during approval/nested suspension. A cancelled reacquisition waiter cannot release a lifetime ticket for unresolved work. This adds a narrow preparation interface and may add preparation work; it does not add a second permission, budget or lifecycle runtime.

### R71 — Internal hook operations and outer barriers

An MCP operation performing a hook uses a scoped operation checkpoint in the existing shared store. It retains an exact live parent and queries actual parent/ancestor pending and failed requirements for the tool's declared dependencies, including owning-event requirement IDs. Unknown mappings refuse. It runs ordinary tool guards and joins its own required post events before returning.

The outer event's generic next-input and terminal barriers remain installed on that outer owner; they cannot block the internal work needed to complete the same event. They still prevent normal input/settlement until all required outer work, including sibling post events, settles. A fresh unrelated checkpoint or an empty dependency list cannot manufacture readiness. Parent closure/revocation, static and dynamic cycles, actual resource custody and the original event budgets remain controlling. This narrow continuation query avoids self-wait while preserving dependency authority; it requires direct pending/failed ancestor, independent initializer and two-post-event controls.

### R72 — Nested MCP context and pending turn admission

An MCP call that performs a hook still runs ordinary PreToolUse/PostToolUse hooks. Their accepted context keeps its original event/handler attribution and turn lifetime even when their internal operation scope settles before the next input. Stage those contributions through the existing context/checkpoint owners until the containing event succeeds and its current authority is rechecked. Do not publish them directly into a live parent while an enclosing sibling can still deny that event; preserve the parent and sibling joins and complete per-event/shared-send limits.

During SessionStart, nested turn context belongs only to the exact pending submission that triggered initialization. Use its existing admission/lifecycle reservation and release the contribution only to that submission's accepted turn; do not create a root run early, promote it to runtime context, or transfer it to a later submission. SessionStart's own direct context keeps its documented runtime lifetime. Failed or replaced admission, enclosing denial, cancellation, revocation or closed parent discards the nested batch. Required context that cannot be delivered safely fails its controlling admission; omission cannot count as success.

For an already admitted operation, use the actual receiving input scope and the existing context carrier. Normal retirement of the nested request does not by itself retire accepted parent-bound context; parent currentness, original effect/definition authority and the containing acceptance still govern delivery. There is no second context runtime or generic cross-turn transfer. Qualify actual first-input delivery, no next-turn leak, cancelled/failed/replaced admission, sibling denial, required overflow, normal admitted nested context and unchanged direct SessionStart runtime contributions.

### R73 — Session capability views and bounded teardown

A pending submission's MCP view has the exact input lifetime defined by R72. A successful live session may also retain the same prepared registry/configuration ceiling as a private runtime view, with no input-context destination or accepted root identity. Retiring a pending binding cannot overwrite a newer binding, retain a failed admission, or transfer its turn context to the runtime view. Events with an actual run still require their exact run authority.

Seal first closes ordinary admission and starts the original teardown clock through the immediate engine cancellation fence, before entering lifecycle/checkpoint locks. Before retiring ordinary checkpoint/input scopes, the existing lifecycle/checkpoint owner may issue a narrow teardown scope from its already-prepared capability view and positively checked dependency state; ordinary scopes then close without reopening. It must retain the exact source identity/readiness provenance and refuse unknown, pending, failed or changed requirements; a closed or discarded requirement is never evidence of independence. Capturing this host state must not delay cancellation on external I/O. This is one bounded transfer within the existing owner, not a new runtime, restored closed scope, new permission or standalone empty dependency store.

Only an actual host-issued Interrupt/SessionEnd delivery can carry this scope across the seal. Its nested normal tool preparation, guards and post-event joins execute as continuations of that delivery in the same engine and retain its original notification deadline, causal limits, counters and cleanup allowance. A public event label or boolean cannot create that authority. Current tool/configuration/credential/scope permissions and already-connected checks still apply at the real dispatch and acceptance boundaries. Guards are enforced; no approval prompt, connect/reconnect, continuation turn, replay or deadline reset is introduced. A runtime-only view supplies no destination for nested turn context, so an undeliverable required contribution refuses under its ordinary controlling policy.

Retire this scope/view at the bounded notification boundary without confusing that retirement with actual request/process cleanup. Unresolved resource custody stays with its existing owners. Repeated close must neither issue new authority nor extend deadlines. Qualify actual Console close after a completed turn, preserved ordinary post-seal refusal, a denying nested guard, stale/dependency/context refusal, and real notification/cleanup timing. Best-effort omission remains correct whenever ordinary current authority or the exact original deadline no longer permits dispatch.


### H6 implementation handoff

Task18 binds MCP result normalization/capture and causal checks to the existing normal tool pipeline, checkpoints, approval rounds, connection owners and HookBudgetOwner. Console supplies an unadvertised prospective context from its actual validated submission; AgentService supplies the admitted run context. R70–R73 define the accepted integration, nested dependency/context and bounded teardown ownership. I1 still supplies managed native graph/application composition; unsupported/unknown graph requirements refuse.

Verification selection is the exact Task18 three files, sixteen affected full files and three selected invocation groups recorded in `task-18-covering-tests.json` under the managed-plugins delivery artifacts. Final raw evidence, unchanged source hashes, exact Task18 BASE static parity and operational limits belong in `task-18-report.md`; intermediate kernel successes are not final H6 qualification. No full suite is required or claimed.


H6 Fix1 binds a private read-only currentness callback to the exact captured result and original call identity. The existing normal owner revalidates profile, definition, permission and owned mapping/scope/credential authority after required post-event settlement and after the final asynchronous hook-authority check, before native effects are accepted. The original one-time approval remains specific to that invocation; validation does not prompt, invoke, connect or recreate grants. Blocking owner validation remains off-loop under the same job/ticket and original deadline. A completed MCP request or cancelled waiter cannot settle a still-running validator.

Initial review I1 is repaired in `f4952cafe98ee4ddab4c53d0fd5c67a371146e80` and accepted by independent scoped review. Final amended verification covers seven files (252 passed) plus ordinary exact-three (83 passed), with overlapping counts and five-file static BASE parity. The prior sixteen-file FD growth418 remains deferred to I7/final qualification; no leak-free or cross-platform claim follows.
