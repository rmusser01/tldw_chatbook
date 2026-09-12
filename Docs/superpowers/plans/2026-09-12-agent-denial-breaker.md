# Consecutive Denial Breaker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop repeated authoritative tool denials at a coherent batch boundary with truthful terminal evidence and per-run isolation.

**Architecture:** Add bounded approval provenance to existing review and tool-result seams, then consume it with one ephemeral runtime counter. Preserve string-review compatibility and existing broad blocked display outcomes. Copy the configured limit through existing budget constructors and reuse the existing RUN_STUCK/Console terminal pipeline.

**Tech Stack:** Python 3.11+, dataclasses, existing Textual Console, pytest, existing SQLite and local JSONL run-log fixtures; no dependencies or migration.

**Spec:** `Docs/superpowers/specs/2026-09-12-agent-denial-breaker-design.md`.

## Global Constraints

- `[agents] denial_circuit_breaker_limit` defaults to integer 3; explicit integer 0 disables.
- Count only authoritative explicit user Deny or configured permission Off; never infer a denial from display/result text.
- Timeout, no_callback, root-change, unavailable authority, unresolved approval, cancellation and restored_pending do not count.
- Successful `ToolResult.ok=True`, authoritative approval and any non-denial settled result reset the streak.
- Evaluate the trailing streak only after a complete tool batch; approved/successful tails reset it, and four denials in one batch at limit 3 report count 4.
- Preserve current cancellation and restoration/persistence-error precedence, complete native tool-call/reply pairing and already-approved dispatch semantics.
- Stop without another model call, preserve streamed partial assistant text, and emit an honest System explanation plus a run-local log record.
- Fresh runtime invocation means streak 0; no durable denial counter, sibling state, shared service counter or transcript reconstruction.
- No schema migration, dependencies, broad repeat-guard refactor, full test sweep or live user configuration imports.
- Work only in /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/agent-orchestration-pr, branch codex/agent-orchestration-remaining. Use .superpowers/sdd/2026-09-11-agent-orchestration-pr-integration/venv/bin/python under pytest isolation. Root owns staging, commits and Backlog status. Workers leave edits unstaged and never dispatch subagents.

ADR required: yes (new compatible extension; accepted ADR-078 remains immutable).
ADR path: `backlog/decisions/154-agent-denial-streak-boundary.md`.
Reason: additive bounded provider/review/runtime control provenance; existing accounting and admission interfaces remain unchanged.

## File responsibilities

- `tldw_chatbook/Agents/agent_models.py`: small shared facts, normalization, RunBudget configuration/inheritance, optional terminal count.
- `tldw_chatbook/Agents/agent_runtime.py`: exact review lookup normalization, per-call accounting and completed-batch stop.
- `tldw_chatbook/Agents/agent_service.py`: review observer normalization and existing run-log vocabulary; no fleet state change.
- `tldw_chatbook/Agents/mcp_tool_provider.py`, `local_tool_provider.py`, `virtual_cli_provider.py`, `raw_shell_tool_provider.py`, `builtin_tool_gate.py` and builtin invocation in `tool_catalog.py`: authoritative invocation decision facts and matching ephemeral stamp provenance.
- `tldw_chatbook/Chat/console_chat_controller.py`: authoritative review fact construction and denial-only missing-placeholder System fallback.
- `tldw_chatbook/Chat/console_agent_bridge.py`: callback annotations, fresh agents setting resolution and budget intersection.
- `tldw_chatbook/config.py`, `Docs/User_Guide/console/agent-runs-and-tools.md`, new ADR-154, TASK-18929: setting, behavioral contract and evidence.
- Focused new tests: `Tests/Agents/test_denial_circuit_breaker.py`, `Tests/Agents/test_denial_provenance.py`, `Tests/Chat/test_console_denial_terminal.py`; extend existing budget, review-hook, provider-continuation and fleet tests where their real fixtures are needed.

### Task 1: Typed provenance with legacy compatibility

**Files:** Modify agent_models.py, agent_runtime.py `_effective_review_verdict`/LoopDeps annotations, agent_service.py `_wrap_review_with_observation`/callback annotation, console_agent_bridge.py callback annotation. Create `Tests/Agents/test_denial_provenance.py`.

**Interfaces:** Produces `ApprovalDecision`, `ToolReviewDecision`, `ToolReviewValue`, `normalize_tool_review`, keyword-only `ToolResult.approval_decision`, `_effective_review_decision`; consumes existing ToolCall and ToolResult.

- [x] Before implementation, verify root created ADR-154, linked unchanged ADR-078, moved TASK-18929 In Progress through Backlog CLI, reconciled the setting criterion to default3/explicit0 and added this plan. Accepted ADR-078 stays unchanged.

- [x] Write failing compatibility tests:

```python
from dataclasses import dataclass
from tldw_chatbook.Agents.agent_models import (
    ToolResult, ToolReviewDecision, normalize_tool_review,
)


def test_legacy_strings_keep_verdict_without_authority():
    for text in ('proceed', 'denied by you', 'ERROR: permission Off'):
        decision = normalize_tool_review(text)
        assert decision.verdict == text
        assert decision.approval_decision is None


def test_new_tool_result_field_does_not_shift_subclass_positionals():
    @dataclass(frozen=True)
    class ChildResult(ToolResult):
        extra: str = ''
    value = ChildResult(False, '', 'failed', None, 'child')
    assert value.extra == 'child'
    assert value.approval_decision is None
    denied = ToolResult.blocked('no', approval_decision='denied')
    assert denied.outcome == 'blocked'
    assert denied.approval_decision == 'denied'
```

- [x] Run `python -m pytest Tests/Agents/test_denial_provenance.py -q`; expect missing new symbols/signature failure, not environment/import failure.
- [x] Implement minimal shared types:

```python
ApprovalDecision: TypeAlias = Literal['approved', 'denied']

@dataclass(frozen=True)
class ToolReviewDecision:
    verdict: str
    approval_decision: ApprovalDecision | None = None

ToolReviewValue: TypeAlias = str | ToolReviewDecision

def normalize_tool_review(value: ToolReviewValue) -> ToolReviewDecision:
    if isinstance(value, ToolReviewDecision):
        fact = value.approval_decision
        return ToolReviewDecision(
            value.verdict, fact if fact in ('approved', 'denied') else None
        )
    return ToolReviewDecision(value)

# Add to ToolResult; preserve all existing fields/order:
approval_decision: ApprovalDecision | None = field(default=None, kw_only=True)

# Extend existing factory body/signature:
@classmethod
def blocked(cls, error: str, *, approval_decision: ApprovalDecision | None = None):
    return cls(ok=False, error=error, outcome=TOOL_OUTCOME_BLOCKED,
               approval_decision=approval_decision)
```

- [x] Preserve exact lookup compatibility by adding `_effective_review_decision(call, verdicts, *, call_id=None)` with the existing call-id-first/name fallback and normalize only the selected value; `_effective_review_verdict` delegates and returns `.verdict`. In service observation use `normalize_tool_review(selected).verdict`, so observed error remains a string. Change type annotations only where needed; do not stringify or flatten values while merging hooks.
- [x] Restore the inherited durable trace fixture to exercise real tool results: its placeholder `model` selects progressive discovery, so the scripted immediate calls are refused as undisclosed and no result reaches redaction. Use the existing known `gpt-4o` fixture model for that single test and assert successful execution of all three synthetic tools before checking redacted/omitted states. Do not change production disclosure or permission behavior. Root baseline and isolated known-model control are recorded in this plan workspace.
- [x] Add call-id-over-name, name fallback, absent proceed, malformed approval-fact, mixed string/structured callback and observer error-text tests. Run focused new tests plus `Tests/Agents/test_agent_runtime_review_hook.py` and `test_trace_approval_capture.py`; expect unchanged dispatch/refusal/Trace behavior.
- [x] Review checkpoint: verify the already-created ADR-154 and updated TASK setting criterion are linked. Root records the focused passing evidence and commits only this slice when authorized.

### Task 2: Attach facts at authoritative decisions

**Files:** Create the small pure `Agents/approval_provenance.py` stamp/key helper. Modify `Chat/console_chat_controller.py` review builders, `Agents/mcp_tool_provider.py`, `Agents/local_tool_provider.py`, `Agents/virtual_cli_provider.py`, `Agents/raw_shell_tool_provider.py`, `Agents/builtin_tool_gate.py`, and builtin invocation in `Agents/tool_catalog.py`. Extend new provenance tests, `Tests/Agents/test_mcp_refusal_provenance.py`, `test_local_tool_provider.py`, `test_virtual_cli_provider.py`, `test_raw_shell_tool_provider.py`, `test_builtin_tool_gate.py`, actual builtin provider coverage and the affected Console review-hook tests.

**Interfaces:** Consumes Task 1 shared types. Produces structured values in review maps and ToolResult facts with unchanged verdict strings, stamps and outcome display.

- [ ] Repair the root-qualified inherited fixture failures before judging producer regressions: replace the MCP audit test's source-literal assertion with actual unresolved invocation/audit assertions while retaining audit vocabulary checks; fill the local bare controller's absent question setter and real InterruptRoundHost state; give the virtual and raw-integration fake turn contexts their required tool_policy_profile_id (default), verifying that each required-field access is unchanged at the task baseline. Root baseline is 309 passed/7 failed across six producer modules; a test-only complete-fixture control passed eight selected cases. Preserve current product composition/permission guards. Exact nodes/evidence are in the plan ledger.
- [ ] Read `scratch/producer-gap-review.md` in this plan workspace. Virtual review deliberately returns proceed for both approved and denied stamps, while builtin/raw/virtual configured-Off routes can bypass review. Attach facts at these actual invocation owners. Keep Task 3's result-only dispatched accounting; do not substitute earlier review metadata for a later root/kill-switch/unresolved result. Preserve the raw runtime's opaque refused outcome without a fact when it supplies no authoritative reason.
- [ ] Use frozen ApprovalStamp(raw decision, optional fact) in the existing stamp maps; keep legacy raw-string accessors through detailed counterparts. The pure helper module imports no Chat code and adds no registry/counter. Builtin check_detailed returns immutable refusal+fact from one check; check unwraps it. Use the actually selected call/name key for unanswered metadata, including name fallback when a call ID has no map entry. In strongest-scope aggregation only a selected deny is marked unresolved when any contributing same-name deny was unanswered; preserve existing selected approval scope.
- [ ] Preserve explicit versus unanswered provenance before flattening every affected stamp/direct-callback map. Use a small immutable internal value associated with the existing run/call-keyed stamp, or equivalent metadata with identical clear/pop/nested-scope lifetime; do not add a global last-decision field or independent shared counter. Keep existing raw-string helper APIs for their current callers where necessary. A deny-looking raw fallback is not sufficient: require the actual raw choice to be deny and its selected call/name key to be answered. Missing/malformed/timeout/exception/defaulted-unresolved decisions carry no fact. Qualify exact key fallback and scope restoration with tests. Propose the minimum additive gate/stamp seam to root before source edits if existing APIs leave a design choice.
- [ ] Add failing explicit/implicit decision matrix tests at actual builder/provider entry points. Reuse real `build_tool_review_hook` fixture from `Tests/Agents/test_trace_approval_capture.py`; use `ApprovalDecisions.unresolved_keys` to prove defaulted deny is excluded. Assert this matrix on returned normalized metadata:

```python
import pytest

@pytest.mark.parametrize('raw,unanswered,expected', [
    ('deny', False, 'denied'), ('deny', True, None),
    ('approve_once', False, 'approved'),
    ('approve_session', False, 'approved'),
    ('timeout', False, None), (None, False, None),
])
def test_review_metadata_comes_from_answered_decision(raw, unanswered, expected):
    # Extract this pure helper in console_chat_controller for all builders.
    from tldw_chatbook.Chat.console_chat_controller import _approval_decision_fact
    assert _approval_decision_fact(raw, unanswered=unanswered) == expected
```

- [ ] Run the new matrix and provider tests before implementation; expect missing metadata/fact mismatch.
- [ ] Add this shared Console builder helper and apply before flattening:

```python
def _approval_decision_fact(decision, *, unanswered=False):
    if unanswered:
        return None
    if decision == 'deny':
        return 'denied'
    if decision in {'approve_once', 'approve_session', 'always_allow'}:
        return 'approved'
    return None

# At each existing owner branch, preserve the owner's verdict calculation:
verdicts[key] = ToolReviewDecision(
    existing_verdict,
    _approval_decision_fact(
        decision, unanswered=approval_was_unanswered(row, decisions)
    ),
)
```

`existing_verdict` in this replacement means the exact current expression at that site, not a new wire string: approved branches remain `proceed`, existing refusal branches keep their current constant/formatted copy. Builders must pass only approval choices that their current owner accepts; helper must not widen those choices. Permission-Off branches attach denied directly from resolved state. Kill-switch/root checks attach no fact. Preserve run-id stamps, strongest-scope logic, every-hook clear and unanswered audit behavior.

- [ ] Update all flattening builders: `build_mcp_review_hook`, `build_tool_review_hook`, `build_local_review_hook`, `build_managed_skill_promotion_review_hook`, `build_virtual_cli_review_hook`, `build_raw_shell_review_hook`. In `build_combined_review_hook`, retain structured values intact. Where a builder does not observe a permission-Off decision, rely on invocation facts instead of re-resolving policy.
- [ ] MCP: change only authoritative configured-Off and explicit-deny ToolResult factories to `blocked(..., approval_decision='denied')`. Do not annotate the no-callback branch even though it shares DENY_REFUSAL copy. Preserve approved provenance on `_execute` return values with `dataclasses.replace(result, approval_decision='approved')` when the actual gate decision was approved/session-approved. Local: derive denied from final `PERMISSION_OFF` or authoritative `APPROVAL_REFUSED`, approved from actual approved gate facts, attaching to the inner result before returning the existing envelope. Errors/timeouts/unresolved leave None.
- [ ] Builtin: add a detailed gate decision from the same single check/resolve, preserving `check()` string-or-None compatibility; builtin invocation consumes the detailed fact without re-resolving or parsing refusal copy. Off and authoritative explicit stamp denial carry denied, while kill-switch/ephemeral/authority/no-stamp/check failure do not. Virtual CLI: preserve always-proceed review/stamp enforcement and annotate its actual denied/Off result, carrying unanswered provenance through its direct callback and stamp lifetime. Recognize allow_matching only at owners that already accept it. Raw shell: annotate actual resolved-Off reads (both first and final provider checks) and authoritative direct stamped denial; leave adjacent disabled/resolve-error and opaque runtime admission refusals unannotated. Preserve all existing permission, generation, arm and stamp checks.
- [ ] Add actual builder-to-provider tests for virtual explicit versus defaulted unresolved deny; configured Off without pending review for virtual/raw/builtin; raw first/final Off versus disabled/exception; builtin Off versus kill-switch/no-stamp; malformed/missing/direct callback decisions; stamp clear/pop/run-scope lifetime and approved execution failures. Keep known baseline failures distinct from newly observed regressions and run only the named affected provider/builder tests.
- [ ] Add real invocation tests for Off versus no_callback with identical copy, explicit deny versus unresolved, approval then execution error, timeout/root-change, and ordinary failed dispatch. Ensure `.ok` and `.outcome` remain unchanged. Run only named provider/review files and inspect all failures.
- [ ] Review checkpoint: compare every new denied assignment to a real raw decision/state. No string comparison to refusal prose is permitted. Root commits this slice after targeted green evidence.

### Task 3: Configurable run-local counter and coherent stop

**Files:** Modify agent_models.py RunBudget and child helpers; agent_runtime.py both result paths and completed-batch boundary; console_agent_bridge.py budget resolver/intersection; config.py comment. Create `Tests/Agents/test_denial_circuit_breaker.py`; extend `Tests/Chat/test_console_agent_run_budget.py`.

**Interfaces:** Consumes structured metadata from Tasks 1–2. Produces `RunBudget.denial_circuit_breaker_limit`, `RunOutcome.denial_count` (default0, terminal-only), and `RUN_STUCK` with STEP_ERROR and error JSONL event.

- [ ] Add a focused runtime helper in the new test file; use actual `run_agent_loop`, native calls, and generous budgets so existing step/repeat guards do not mask this guard:

```python
import json
from tldw_chatbook.Agents.agent_models import (
    AgentConfig, ModelTurn, RunBudget, ToolCall, ToolLoadSelection,
    ToolResult, ToolReviewDecision,
)
from tldw_chatbook.Agents.agent_runtime import LoopDeps, run_agent_loop


def run_batch(results, *, limit=3, review=None):
    calls = [ToolCall(name=f'tool_{i}', args={}, call_id=f'c{i}')
             for i in range(len(results))]
    raw = [dict(id=c.call_id, type='function', function=dict(
        name=c.name, arguments=json.dumps(c.args))) for c in calls]
    script = [ModelTurn(tool_calls=tuple(calls), assistant_message=dict(
        role='assistant', content='partial plan', tool_calls=raw)),
        ModelTurn(text='done')]
    invoked, records, model_calls = [], [], []
    def model(messages, schemas):
        model_calls.append(list(messages))
        return script.pop(0)
    def invoke(call):
        invoked.append(call.call_id)
        return results[int(call.call_id[1:])]
    deps = LoopDeps(call_model=model, invoke_tool=invoke,
        spawn=lambda task: ToolResult(ok=True), find_tools=lambda q: [],
        load_schemas=lambda ids, messages, call: ToolLoadSelection(),
        should_cancel=lambda: False, clock=lambda: 0,
        review_tool_calls=review,
        on_record=lambda kind, payload: records.append((kind, payload)))
    cfg = AgentConfig(model='m', system_prompt='s',
        allowed_tools=tuple(c.name for c in calls),
        budget=RunBudget(max_steps=100, denial_circuit_breaker_limit=limit))
    return run_agent_loop(cfg, [{'role':'user','content':'go'}], [], deps), invoked, records, model_calls


def test_completed_batch_reports_observed_count_and_keeps_every_reply():
    denied = ToolResult.blocked('no', approval_decision='denied')
    out, invoked, records, model_calls = run_batch([denied] * 4)
    assert out.status == 'stuck' and out.denial_count == 4
    assert invoked == ['c0', 'c1', 'c2', 'c3']
    assert len(model_calls) == 1
    assert [r['tool_call_id'] for r in out.final_messages if r['role']=='tool'] == invoked
    assert len([r for r in records if r[0]=='error']) == 1


def test_approved_tail_resets_completed_batch_streak():
    denied = ToolResult.blocked('no', approval_decision='denied')
    out, invoked, records, model_calls = run_batch(
        [denied] * 3 + [ToolResult(ok=False, error='execution failed', approval_decision='approved')])
    assert out.status == 'done' and out.denial_count == 0
    assert len(model_calls) == 2
```

- [ ] Run the new runtime tests; expect missing RunBudget field or absence of stuck behavior. Add parameterized disable/reset/legacy-string/successful-denial-copy cases before coding.
- [ ] Add the strict field coercer in models, normalize only this new field in RunBudget post-init, and use it in bridge fresh `[agents]` resolution:

```python
DEFAULT_DENIAL_CIRCUIT_BREAKER_LIMIT = 3

def coerce_denial_circuit_breaker_limit(value: object) -> int:
    if type(value) is int and value >= 0:
        return value
    if isinstance(value, str) and value.strip().isascii() and value.strip().isdigit():
        try:
            return int(value.strip())
        except ValueError:
            pass
    return DEFAULT_DENIAL_CIRCUIT_BREAKER_LIMIT

# RunBudget field and post-init:
denial_circuit_breaker_limit: int = DEFAULT_DENIAL_CIRCUIT_BREAKER_LIMIT
# Preserve existing post-init validation, then:
object.__setattr__(self, 'denial_circuit_breaker_limit',
    coerce_denial_circuit_breaker_limit(self.denial_circuit_breaker_limit))

# Both child RunBudget constructors:
denial_circuit_breaker_limit=child.denial_circuit_breaker_limit,
# Bridge intersection constructor:
denial_circuit_breaker_limit=int(ceiling(
    maximum.denial_circuit_breaker_limit, live.denial_circuit_breaker_limit)),
# Bridge fresh configuration constructor, inside existing local-import boundary:
denial_circuit_breaker_limit=coerce_denial_circuit_breaker_limit(
    get_cli_setting('agents', 'denial_circuit_breaker_limit', 3)),
```

Keep config-read exception fallback at default3 without resetting unrelated budget fields. Extend budget tests for missing,0,1,3,'0',' 4 ',True,False,-1,1.5,float('inf'),'bad'; check both child helpers and (0,3)/(3,0)/(3,2)/(0,0) intersections.

- [ ] Implement the local accounting helper and call it only after each result/history settlement in both branches:

```python
consecutive_denials = 0

def account_denial(*, review: ToolReviewDecision, result: ToolResult | None,
                   synthetic_restore: bool = False) -> None:
    nonlocal consecutive_denials
    denied = False
    if not synthetic_restore and not (result is not None and result.ok):
        if result is not None:
            denied = result.approval_decision == 'denied'
        elif review.verdict != 'proceed':
            denied = review.approval_decision == 'denied'
    consecutive_denials = consecutive_denials + 1 if denied else 0
```

For pre-dispatch refusal pass `result=None` and the selected normalized review value. For invocation pass actual ToolResult; do not treat review's default proceed as approval. Restored_pending passes synthetic_restore=True. Do not put accounting in the up-front Trace loop.

- [ ] At completed batch end, preserve cancellation first, then terminalize without another model call:

```python
if deps.should_cancel():
    return _outcome(RUN_CANCELLED)
if budget.denial_circuit_breaker_limit and consecutive_denials >= budget.denial_circuit_breaker_limit:
    coherent_len = len(messages)  # only after every result/restore expansion succeeded
    summary = (f'Agent stopped: {consecutive_denials} consecutive tool calls were denied. '
               'Review the denial reasons or rephrase, then retry.')
    add(STEP_ERROR, summary=summary)
    _emit_record(deps, 'error', content=summary, status=RUN_STUCK)
    return _outcome(RUN_STUCK, denial_count=consecutive_denials)
```

- [ ] Add one actual virtual CLI builder/provider/runtime integration: three explicit denials stop after a completed batch, while three identical defaulted unresolved choices do not. Assert the review still permits provider dispatch and final invocation facts govern the counter, including a later authority refusal overriding earlier approval/denial metadata.
- [ ] Add count-across-turns, non-denial reset, explicit-approved failure, same-name distinct-ID review, legacy deny string exclusion, no extra model call, and fence-protocol tests. Extend actual `Tests/Agents/test_provider_continuation_runtime.py` fixtures for continuation refusal and restored_pending exclusion, settled history pairing and cancellation/persistence-error precedence. Keep every existing repeat guard unchanged.
- [ ] Run new runtime tests, selected continuation cases and budget tests; expect green with complete output. Review/commit checkpoint after evidence.

### Task 4: Terminal System row, retained history and fleet isolation

**Files:** Modify `Chat/console_chat_controller.py:_finalize_agent_failure` only if fallback needs feature-specific System row; agent_service.py/run_log.py vocabulary docstrings. Create `Tests/Chat/test_console_denial_terminal.py`; extend real fleet/service fixtures in `Tests/Agents/test_fleet_runtime.py` and run-log service tests.

**Interfaces:** Consumes `RunOutcome.denial_count`, existing `_agent_failure_visible_copy`, `_append_failure_system_row`, and `_record_run_assistant_message`. No new Console state, event or DB schema.

- [ ] Add terminal tests using the actual controller and ConsoleChatStore. Start with normal placeholder and missing-placeholder variants, seeded partial assistant content. Produce outcome using actual runtime helper above (import helper only from the focused test module) instead of manually inventing STEP_ERROR. Assert System text, preserved partial content, terminal state and subsequent accepted submit.

Required assertion body after fixture drives finalization:

```python
rows = store.messages_for_session(session.id)
system_rows = [r for r in rows if r.role is ConsoleMessageRole.SYSTEM]
assert len([r for r in system_rows if '3 consecutive tool calls were denied' in r.content]) == 1
assert any(r.role is ConsoleMessageRole.ASSISTANT and 'partial plan' in r.content for r in rows)
assert controller._agent_failure_visible_copy(outcome).startswith('Agent stopped:')
assert outcome.denial_count == 3
```

Construct controller/store/gateway with the real setup in `Tests/Chat/test_console_provider_failure_copy.py:test_agent_failure_row_carries_body_and_image_recovery_hint`; use a queued deterministic gateway for next-submit acceptance and inspect the store, not a mock `_append_failure_system_row` call alone.

- [ ] Run `python -m pytest Tests/Chat/test_console_denial_terminal.py -q`; missing-placeholder System expectation should fail before the fix.
- [ ] Keep normal placeholder branch unchanged. For missing-placeholder denial outcome, select or create failed assistant using existing store helpers, preserve its content and durable anchoring, then append `_append_failure_system_row(session_id, visible_copy)` exactly once. Use `getattr(outcome, 'denial_count', 0) > 0` as discriminator; do not parse reason text. Avoid adding the same explanation to assistant and System. Keep all unrelated error fallback behavior unchanged.
- [ ] Extend actual service/run-log fixture: run through AgentService with local writer and SQLite, load JSONL records, assert one error record contains count, `status='stuck'`, correct child/primary run id, and no denial payload secrets. Assert stored STEP_ERROR independently. Update vocabulary docstrings to include error; do not add tables/types registry.
- [ ] Extend real fleet fixture with Event-gated sibling and denying child. Assert child status stuck, sibling still running then done after gate release, supervisor not cancelled/stuck by shared counter. Continue retained child and prove two fresh denials do not inherit its previous count3. Verify its first model history includes all replies from completed denied batch. Use Events and fixture timeouts; no sleeps or live model requests.
- [ ] Run only terminal file, new run-log test selectors and added fleet selectors, then existing nearby failure/fleet cases touched by the change. Review evidence distinguishes pure-loop, service persistence and Console transcript guarantees. Root review/commit checkpoint.

#### Documentation, acceptance evidence and bounded review

**Files:** Modify `Docs/User_Guide/console/agent-runs-and-tools.md`, `config.py` commented agents block, TASK-18929 and new ADR-154 through root's authorized Backlog workflow.

**Interfaces:** Final behavior from Tasks1–4; no new code interface.

- [ ] Add this user-facing setting text and matching commented TOML:

```toml
# denial_circuit_breaker_limit = 3  # Consecutive denied calls; 0 disables.
```

“`[agents] denial_circuit_breaker_limit` defaults to 3. Explicit 0 disables it. A run stops before its next model request when a completed tool batch ends with that many consecutive user-denied or permission-Off calls. Approved, successful and other non-denial results reset the streak, so an approved tail can keep a mixed batch running. The message reports the observed count, which can exceed the limit in one batch. Each child and each new or resumed run starts its own count. Review the denial reasons or rephrase, then retry.”

- [ ] Verify the previously-created compatible extension decision at the verified ADR-154 path matches implementation; leave accepted ADR-078 unchanged. Link the new decision from plan/task/implementation notes and verify setting and authoritative/batch criteria match the adopted spec before marking items complete.
- [ ] Run `rg -n 'ToolReviewValue|approval_decision|denial_circuit_breaker_limit|denial_count'` over modified source for self-review; inspect every provenance assignment and budget constructor for lost zero or broad blocked counting. Run repository-selected linter/formatter only on changed Python files. Do not run full suite without opt-in.
- [ ] Record actual targeted commands/results in task notes, including no extra model call, completed native history, real System row, JSONL count, child isolation, reset on retry and cancellation precedence. Mark Done only after all repository DoD items are met; root owns CLI status and final commit. Do not claim live-model verification from deterministic fixtures.

## Plan self-review

Coverage: typed legacy compatibility Task1; authority/exclusions Task2; defaults/reset/batch/coherent history Task3; terminal/persistence/isolation/retry Task4; docs/ADR/AC Task4. No independent provider-wide redesign or speculative storage is included. All new shared signatures appear before consumption. Production snippets are insertion/replacement patterns and retain the surrounding existing exception/permission logic; implementer must read referenced exact functions before editing.
