---
id: TASK-34568
title: Acknowledge Console decisions and grant outcomes honestly
status: In Progress
assignee:
  - '@codex'
created_date: '2026-10-06 06:42'
updated_date: '2026-10-07 03:49'
labels: []
dependencies:
  - TASK-34565
  - TASK-34566
documentation:
  - >-
    Docs/superpowers/specs/2026-10-05-console-approval-ux-and-responsiveness-design.md
  - backlog/decisions/221-console-approval-interaction-and-feedback.md
  - Docs/superpowers/plans/2026-10-05-console-approval-ux-and-responsiveness.md
priority: high
type: enhancement
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Show that an answer registered and distinguish it from grant application, final checks, tool execution and model continuation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Applying feedback is immediate and may coalesce into a newer state without delaying worker release or FIFO promotion.
- [x] #2 Controller receipt, host settlement, remembering outcomes and confirmed backend start derive from the matching authoritative owners.
- [x] #3 Grant failures retain existing current-call execution behavior and explain that permission was not remembered without automatic re-prompting.
- [x] #4 Stale or reordered observations affect only the owning chat and call, cannot overwrite newer states, and never authorize execution or enter durable capture.
- [x] #5 Later owner-confirmed grouped grant success corrects prior failure copy without changing actual permission effects.
- [x] #6 Completed individual tool rows cannot retain shared Starting feedback; grant facts remain visible.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
### Task 5 Publish truthful receipt and grant observations

**Backlog:** TASK-34568. **ADR:** ADR-221/067/195/210. **Consumes:** captured owner/round/revision/call identity, Task 3's Applying state and generation fence.

**Files:** Create `tldw_chatbook/Agents/approval_observation.py`, `Chat/console_approval_feedback.py`, `UI/Console_Modules/approval_feedback.py`, `Tests/Agents/test_approval_observation.py`, `Tests/Chat/test_console_approval_feedback.py`, and `Tests/UI/test_approval_feedback.py`. Modify approval provenance's optional observational metadata, controller receipt/outcome callbacks, interrupt host settlement, provider grant wrappers, `MCP/unified_control_plane_service.py` actual grant writers, `Chat/console_raw_cli.py:grant_model_session`, `Chat/console_agent_bridge.py`, `Chat/console_tool_activity.py`, and `UI/Console_Modules/wiring.py`. Add an optional display-only feedback field to `ConsoleActivityPresentation` in `Chat/console_chat_models.py`, rendered by `Widgets/Console/console_transcript.py`; keep it outside durable/model serialization. Existing `Widgets/Console/console_status_chips.py:sync_run_chip` supplies the active run status.

**Interfaces:** Define frozen `ApprovalObservationIdentity(session_id: str, run_id: str, round_id: str, revision: int, call_key: str = '')`. Define frozen `ApprovalObservation(identity: ApprovalObservationIdentity, kind: str, outcome: str, actual_scope: str | None = None, error_code: str = '')`, restricted to code-owned kinds and outcomes: received; settled/accepted/timeout/cancelled/revoked; grant/applied/not_applied/failed/unknown; dispatch_started; backend_started; tool_completed; model_wait. No arbitrary exception/body is carried.

Produce `approval_observation_scope(identity: ApprovalObservationIdentity, sink: Callable[[ApprovalObservation], None]) -> Iterator[None]` and `publish_grant_application(outcome: Literal['applied','not_applied','failed'], *, error_code: str = '') -> None`. These are optional ContextVar-based display observations, never permission inputs. Define frozen `ApprovalFeedback(identity: ApprovalObservationIdentity, sequence: int, decision_state: str, selected_scope: str | None, grant_state: str, applied_scope: str | None, execution_state: str, terminal_outcome: str)`, using the finite states already declared above, and `format_approval_feedback(feedback: ApprovalFeedback) -> str`. It keeps separate facts, not arbitrary output text.

Define `ApprovalFeedbackStore.bind_round(view: ApprovalBatchView) -> None`, `context_for_call(run_id: str, call_key: str, *, fallback_tool_name: str = '') -> ApprovalObservationIdentity | None`, `publish(observation: ApprovalObservation) -> bool`, `snapshot(session_id: str, run_id: str) -> tuple[ApprovalFeedback, ...]`, and `retire_run(run_id: str) -> None`. UI collaborators consume snapshots and format_approval_feedback; producers never use publish's return value to decide access.

- [ ] **Step 1: Write failing ownership and arbitration tests.** Cover receipt without settlement, Stop and deadline between those points, grant failure with otherwise permitted execution, raw grant no-op after Disarm, and missing grant evidence. Add `test_reordered_feedback_cannot_regress_execution`, `test_old_round_feedback_cannot_change_next_card`, `test_observer_failure_does_not_change_verdict_or_release`, `test_feedback_does_not_delay_fifo_promotion`, and `test_fast_completion_does_not_require_applying_paint`. Compare legacy trace counts, refusal strings, stamp values and actual service effects with observation disabled/enabled. Run the new files and verify the intended red results.
- [ ] **Step 2: Add the noncontrolling store and correlation.** The controller registers captured view identity and attaches observation contexts to its ApprovalDecisions metadata, separate from map entries. Hooks/bridge register those contexts for normal invocation lookup without changing ApprovalStamp or provider stamp keys. Use the existing run/call context; name fallback is allowed only for one unambiguous shared-verdict group. Ambiguous or missing context yields no attributed grant claim. Delivery notifications read the latest reduced snapshot rather than replaying an older partial snapshot. Release locks before callbacks and keep them bounded and exception-isolated.
- [ ] **Step 3: Observe receipt and final host outcome.** Preserve resolve_pending_approval's existing None-return compatibility. A matching committing message can emit receipt only after existing guards accept it; it still sets the Event promptly. The host's final outcome callback emits settled acceptance only after deadline/cancellation arbitration, before teardown. Accepting metadata must not make a raw None stamp present or alter exact/name refusal precedence. UI generation is observational; the gate's original round identity remains authoritative.
- [ ] **Step 4: Observe actual grant application at owners.** Open only the display observation scope around existing grant calls, explicitly carrying that display context into worker/coroutine boundaries without copying unrelated policy context. Service cache/durable writer paths publish applied only after actual success, including an already-present equivalent rule. Exception paths publish a controlled failure code; absent writers or raw Disarm/no-op publish not_applied. Keep return contracts and best-effort current-call behavior unchanged. A name-coalesced tool-wide grant reports its actual applied scope, not a false per-row confirmation of another selected scope. It never claims a rule was saved merely because a void call returned.
- [ ] **Step 5: Project states without keeping cards alive.** The UI collaborator receives named late-binding read/paint/refresh services through wiring. Update the matching card, otherwise its owning tool/run status. Old receipts cannot overwrite settled/cancelled/execution facts; grant facts update separately. Emit Starting for early dispatch and Running only after an actual owner start, otherwise retain Starting until a result. Use provider-request facts for model_wait. Before retiring run maps, transfer final derived facts to the existing session-only tool presentation so late UI scheduling cannot lose a grant failure. This transfer does not await UI paint or enter durable capture/history; producer store writes use existing owner-thread routing.
- [ ] **Step 6: Verify grants and observer isolation.** Run the new feedback/observation tests, `Tests/Chat/test_console_interrupt_rounds.py`, `test_console_tool_activity.py`, `test_console_raw_shell_revocation.py`, affected provider/gate tests, `Tests/Agents/test_trace_approval_capture.py`, `test_approval_denial_reasons.py`, and the relevant per-call/None provenance cases. Confirm no trace-step inflation, content-bearing diagnostics, cross-chat notice or automatic retry. Measure input acknowledgment and worker/FIFO release using Task 1's identical boundaries.
- [ ] **Step 7: Close and commit.** Document the actual owner signal for each state, failure/no-op cases, serialization exclusions and exact qualification receipts. Do not mark this task Done on UI text tests alone.

ADR required: yes
ADR path: backlog/decisions/221-console-approval-interaction-and-feedback.md
Reason: Existing ADR221/067/195/210 governs optional noncontrolling observations and session-only feedback; no authority or persistence migration.

Compatible observation clarification: publish_grant_application accepts optional actual_scope, supplied by the actual writer. This supports the required coalesced-grant scope evidence without inferring from the user selection; no authority/return contract changes.

PR3036 review remediation: reproduce the concrete source/CI findings, implement minimal display/correlation fixes, run targeted RED/GREEN and independent scoped review. ADR required: no new ADR; existing ADR221 and ADR220 govern observational ownership and consent. Preserve enforcement, lifetimes, stamps, originals and arbitration. No full sweep without opt-in.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Task5 adds optional owner observations, a monotonic session-only feedback store, and retained tool feedback after cards clear. Controller receipt follows Event release; final settlement uses the existing interrupt-host arbitration seam. Actual service writes report their own applied scope; missing writers, failed writes and raw Disarm remain honest. No permission returns, stamp keys, None presence, refusal precedence or trace counts were changed. Legacy name groups project only to observed invocation members; group completion/exact-input save success remains unknown when not proven.

Compatible display-only additions: publish_grant_application(actual_scope=...), ApprovalDecisions observational attributes outside dict entries, ApprovalFeedback.confirmed_backend_start=False, member projection lookup, and nested provider observation scopes. Confirmed start survives late/reordered failed results. Early dispatch remains Starting; actual tool/raw output is Running evidence. Full feedback wraps below the existing tool header using central tokens; CSS rebuilt.

ADR221/067/195/210 applies. Evidence: Docs/superpowers/qa/2026-10-05-console-approval-ux/task-5/receipts.json. New ownership/store/UI checks:12/24/9 passed. Existing host29,activity12,raw revocation7,builtin74,MCP90,raw provider32,raw progress10,trace26,denial6,activity persistence157; CSS bundle5/token references1 and detachable hook checks pass. All46 local native-worker failing node IDs also fail with exact BASE owner source overlay; no native-worker success claim. Existing screen ratchet and pre-existing lint diagnostics remain documented. Native/browser frame timing, latency distributions, broader Inspect qualification and independent review remain open; retain In Progress and AC1 unchecked.

Independent Task5 fix1 review at d28deef7b91d9ddfd9fa0b1c4c9aa60d9ce794b8 verified all five findings addressed: hook audit writes, exact call attribution, conservative grouped starts, virtual grant writers, completed child retirement. No new Critical/Important issue in the fix range. Owners34/feedback29/MCP90/trace26/host29/UI9 and marker2 receipts verified. Native dispatch and presented-frame qualification remain open. ADR-221; explicit legacy fallback metadata is observational only.

PR3036 remediation: later actual grouped grant success corrects failure and later failures cannot regress an applied fact. Terminal collapsed and expanded notices retain grants without shared Starting while pending siblings preserve it. Reducer31, UIfeedback15, scope14, host12, toolactivity12 and hook inventory selection1 passed; independent review found no remaining actionable issue. Existing ADR221/220. Evidence: Docs/superpowers/qa/2026-10-05-console-approval-ux/pr3036-remediation/README.md. Native latency/full qualification remains open; In Progress.
<!-- SECTION:NOTES:END -->

## Renumbering provenance

Originally TASK-34415 in the reviewed approval checkout. Renumbered to TASK-34568 during PR integration onto current dev because older unrelated TASK-34411/34412 already landed. The six approval records moved together to preserve dependency order; original verification hashes/commit references retain their historical context.
