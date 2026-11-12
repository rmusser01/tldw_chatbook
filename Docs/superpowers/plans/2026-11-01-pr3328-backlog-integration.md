# PR3328 local integration plan

Source cf4607ad63c784b32628cf0a6286c957e56bacb0, base e39df4875d636e37a9b28aeba84c36a7e2948f3d. Rebase retained source commits onto 4a38b3d2dba872e46ca8f9569908d9ccfd7aa2e9 on codex/pr3328-resync. Drop only four pure size commits cf4607ad63,5a58c0bb23,7df9a94b55,c7267a9d32; preserve all current dev ratchet files unchanged.

ADR required: no
ADR path: backlog/decisions/052-console-conversation-memory-and-compaction-policy.md and existing ADR230
Reason: Restore existing command ownership, capacity estimation and outcome visibility; no storage/permission/runtime-policy boundary changes.

1. Rebase and resolve overlapping controller/tool/trace/summary code deliberately. Verify new base ancestry and retained behavior; notify parent of material conflicts.
2. Start TASK33621.4 In Progress through Backlog CLI. Record explicit unresolved follow-ups. No main task Done before independent review.
3. Causal RED/GREEN for actual delayed slash/palette worker ownership; capture explicit current origin and reject stale start without billing.
4. Causal RED/GREEN for Automatic forecast mandatory current-input estimate and shared provider capacity; preserve estimated semantics, cold-tokenizer and current metadata publication. Do not add a future-send assembler or execute forecast hooks/provider effects.
5. .102 only if existing custody owner supports settlement; otherwise retain explicit reason. Keep .100 and other follow-ups open.
6. Targeted mounted/controller/SQLite preservation checks with a separately verified task-private fixture root, qualified Python, content-free receipts, lint/diff/inventory/census reconciliation. No full sweep, ratchet increases or remote actions.

Evidence and ledger remain in the ignored task workspace. Parent owns independent review and all remote actions.

## Selected implementation and qualification

- I1: `action_compact_console_context` calls `capture_compact_origin` synchronously;
  `run_compact_command(..., origin=...)` applies the existing chat/composer guard
  with that explicit origin before `compact_context_now`. Both deferred slash and
  palette paths must refuse after A -> B; started reports retain A.
- I2: the existing display estimator counts current static mandatory input without
  replaying Send. `ConsoleReadinessConfigProjection` publishes only scalar local
  input/output caps for its captured settings/source owner; both controls and
  gateway use the existing capacity solver and shared positive-cap alias reader.
  Do not call preview/planner or permission readers from forecast.
- I3: keyword `turn_skill_bindings` passes through `refuse_at_ask_trigger` from Retry,
  Continue, Regenerate and Edit/resend to the real preparation/accounting decision.
  A bound-skill Automatic-budget/Ask control must refuse before sibling/attempt,
  preserving actual SQLite rows and making no provider call.
- Coordinator-selected partial AC3 scope: all agent-enabled/bridge-present chats
  keep next-send behavior unavailable, including light surfaces. Existing last
  accounting cannot prove current tool overhead. Above the displayed estimated
  trigger, retain an explicitly UNKNOWN high-context advisory; never infer ASK or
  Ready from that bound. Below it, remain unavailable. This is not user acceptance
  of narrowed AC3. AC3 remains unchecked and TASK33621.4 stays In Progress pending
  qualified current tool provenance and independent review. No new planner-input
  retention, cache, invalidation or permission policy is introduced.
- .102 is explicitly deferred: a billed summary can finish without its command
  result report after outer worker cancellation. Accepted-turn custody does not
  own this command. All other filed follow-ups remain explicit.

Focused nodes: `test_deferred_compact_worker_never_targets_a_later_chat` in
Tests/UI/test_console_compaction_hold_flow.py (slash/palette), and
`test_automatic_forecast_matches_real_preflight_with_overhead_and_cap`,
`test_bound_skill_ask_gate_uses_its_overhead_before_regenerate_commits`,
`test_cold_forecast_does_not_read_history_or_count`, and
`test_unavailable_forecast_is_honest_without_hiding_live_or_failed_work` in
Tests/UI/test_console_compaction_forecast.py. The mounted forecast journey checks
both substantial mandatory input and a provider input cap against real preflight,
then low-unknown/high-advisory agent states without calling tool preview. Preserve
pending/bound/actual-skill summary fences, recovery/provider-contact copy,
emergency-stop precedence, held-send rewind/resume and current metadata publication.

## Final integration qualification

Current receipt outcomes, source identities and remaining limits are recorded in
[the integration QA](../qa/2026-11-01-pr3328-backlog-integration.md). I1/I2/I3 causal
controls and narrow currentness/fence/recovery controls pass. Three mounted held
journeys time out before hold assertions in finite native authority capture; one
exact merged-base comparison reproduces that shared precondition timeout, not
hold-feature completion. No timeout or admission guard is weakened. Seven
passing node selectors are added to the UI census without deleting entries or
changing its floor. TASK33621.4 stays In Progress and broad AC3 unchecked; old
task implementation/live/all-green/ratchet paragraphs are historical source
evidence, superseded by this final qualification. Parent owns independent review
continuation and remote actions.

Final review copy correction: remove the unproved precommit claim from the
compact exception report; qualify F1 hold-card choices as prepared composer
behavior and its forecast row as estimated/static versus unavailable agent/tool
prediction. Preserve type-only logging, all ownership/admission guards and the
existing runtime owners. Narrow receipt r-d9e4aa47 has two clean direct-copy
passes and completed F1 body assertions with a native-worker teardown error
(three selected nodes; XML aggregate4/0 failures/1 error includes teardown
accounting); it is not an all-GREEN receipt or an established
inherited teardown issue. No further reruns, new tests, census changes or
unrelated shutdown repair are part of this correction.
