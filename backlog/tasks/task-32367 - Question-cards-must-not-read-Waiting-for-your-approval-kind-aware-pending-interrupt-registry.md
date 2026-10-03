---
id: TASK-32367
title: >-
  Question cards must not read Waiting for your approval (kind-aware
  pending-interrupt registry)
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-11 01:55'
updated_date: '2026-10-03 06:37'
labels:
  - console
  - approvals
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`has_pending_approval_round` covers all five interrupt kinds (including the question card) while the inspector counts mounted approval cards only, so the transcript activity line can say "Waiting for your approval" while a question card is up. Lane B's minimum fix (R15) corrected the false invariant comment in chat_screen.py; the registry itself is still kind-blind. Found by lane B's final review (task-32345 area).

Source: approval-card / MCP-permissions fix wave 2026-09-10/11 (plan `Docs/superpowers/plans/2026-09-10-approval-card-fix-wave.md`, review snapshot `.impeccable/critique/2026-09-10T17-31-53Z__hatbook-widgets-chat-widgets-chat-approval-card-py.md`); rider recorded in the lane ledger, not fixed in the wave.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A pending question card never produces the "Waiting for your approval" activity line; it produces copy that names a question
- [x] #2 The pending-interrupt registry exposes the interrupt kind to the activity classifier and the inspector count agrees with it
- [x] #3 A test pins one question card + one approval card → the line names the approval, and a lone question card → the question copy
- [x] #4 Keyboard review focuses the visible pending decision card when a tool approval is queued behind another confirmation, while retaining visible approval priority.
- [x] #5 A pending agent chat-creation confirmation is classified as a confirmation, contributes zero tool-approval rounds, and stays reachable through keyboard Review and its attention tab; Close declines it without creating a chat.
- [x] #6 A question arriving after its session commits Close returns cancelled without a retained card or wait; existing sibling questions stay answerable.
- [x] #7 A delayed chat-create confirmation from a closed or no-longer-viewed session cannot replace the active session decision; a live sibling confirmation remains answerable.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/094-console-turn-lifetime-and-navigation-boundary.md; backlog/decisions/067-indefinite-human-approval-waits.md; backlog/decisions/195-console-live-tool-call-presentation.md
Reason: Complete the existing kind-aware pending-decision projection into Inspector status/counts; no new owner or registry.
1. Mount actual ChatScreen with real question and approval worker rounds; reproduce Inspector mismatch for a lone question and multiple approval rounds while retaining existing activity precedence.
2. Reuse pending kind registry/shared copy. Add one locked kind-count accessor; derive active-session Inspector approval count and Live work copy from it, retaining compatibility fallback for missing legacy controller seams.
3. Cover lone question, question plus approval, two queued approvals, resolution, sibling session and detach/remount with production-shaped round ownership and visible cards.
4. Run targeted Inspector/activity/pending-lifetime tests and native Console journey; update user guide and QA.
5. Self-review integrated changes, resolve PR feedback and require final-head checks before normal merge.

Qodo chat-create projection follow-up (PR2953):
ADR required: no
ADR path: backlog/decisions/150-agent-chat-fork-and-spawn.md; existing ADR-067/094/195 above.
Reason: Classify an existing standalone confirmation through the existing registry and focus/Close seams without changing its owner, permission or grant policy.
6. Reproduce the real mounted chat-create round incorrectly counting as a tool approval; cover confirmation copy, keyboard/attention-tab focus, visible tool-approval priority and sibling isolation.
7. Extend the existing real background Close matrix and crowded confirmation geometry for chat-create; use the existing confirmation copy and registry, add only its kind, card selector and truthful Close consequence.
8. Run targeted projection/Close and chat-create consent/revocation neighbors, token/artifact/lint guards, and native approval/Close replay; require fresh final-head Qodo and all four GitHub gates before normal merge.

Qodo Close/confirmed-create race follow-up (PR2953 finding ece44de5-bd8f-4ef5-97a1-77cebdc2e262):
ADR required: no
ADR path: backlog/decisions/150-agent-chat-fork-and-spawn.md; existing ADR-067/094 above.
Reason: Enforce the existing committed Close fence at confirmation finalization and the shared executor; no new admission owner or permission policy.
9. Reproduce remembered-grant resurrection after real Close and both confirmed new_chat/fork_chat executions while a committed Close retains its source session.
10. Keep final confirmation decision/grant updates and resolver writes under the existing round lock; refuse shared executor entry on the exact committed Close generation, excluding failed-provisional markers.
11. Run affected consent/execution and mounted Close/projection tests, ordinary census/preflight/lint, independent review and native replay on frozen sources; require fresh final-head Qodo and all four GitHub gates before normal merge.

Qodo in-flight creation follow-up (PR2953 finding59151d44-d74b-48d8-9844-aeeb441d6079):
ADR required: no
ADR path: backlog/decisions/094-console-turn-lifetime-and-navigation-boundary.md; backlog/decisions/150-agent-chat-fork-and-spawn.md.
Reason: Suppress late cancellation output at the existing synchronous UI handoff and reuse the existing orphan soft-delete contract; no new admission owner, transaction policy or runtime boundary.
12. Reproduce both new_chat/fork_chat with real SQLite creation paused after entry and with completion queued across actual Close. Assert refused outcome, no UI completion and no live orphan, explicitly distinguishing soft-delete from physical rollback.
13. Recheck source currentness after worker I/O and at the UI-thread completion gate; on refusal discard the just-created conversation using the existing worker-side orphan helper. Keep UI callbacks/DB work outside pending-round locks and do not wait on a worker from the UI. Resolve the current view sink at handoff, preserving live-source durable results across view detachment. Distinguish pre-admission dispatch failure from an already-admitted completion exception, so cleanup never deletes a chat already placed in the store.
14. Add the test helper Returns docs, run affected regressions/consent/Close/UI and ordinary performance/preflight/lint guards, peer review and fresh native replay; require fresh final-head Qodo/all four CI gates before normal merge.

Qodo4ae8306e final-head maintainability follow-up: name the existing five-second chat-create synchronization deadline per test module and use it for related event waits/worker joins. Preserve exact value and all executable module AST after constant substitution. Run both complete affected Chat test modules against real SQLite/worker seams, zero-new lint/range format and artifact preflight. No production, timeout/cap, budget, interface or authority changes. ADR required: no. ADR path: N/A (mechanical test-only refactor preserving contracts). Require refreshed final-head review/all four gates/latest dev before normal merge.

Qodo d9c05b6a question/Close race: reproduce with the real question host and complete Close, pausing before question registration and after early registration. Include an already-closed owner and a still-pending sibling; ensure workers are released on red failures. Confirm deterministic red for late admission before changing production. Publish the committed close generation under the existing shared interrupt lock and check that generation in the existing question check/register critical section; leave callbacks outside the non-reentrant lock. Preserve run revocation, busy limits, no-session behavior and per-session isolation. Run full question/host/Close/chat-create neighboring contracts and exact Perf Guards; review lock ordering independently and fresh artifact preflight/lint. ADR required: no new ADR. ADR paths: backlog/decisions/094-console-turn-lifetime-and-navigation-boundary.md and backlog/decisions/067-indefinite-human-approval-waits.md. Reason: routine correction of their existing destructive-close cancellation/admission contract using current owners/locks, no new authority, persistence or service boundary.

Qualification precision: initial new regression without bootstrap_profile raised raw_source_selection_changed, not the race; preserve it as invalid setup evidence. Existing marker restores the real collection-owned config participant; valid RED has exactly two5s late-worker waits failing and the already-registered variant passing, then allthree GREEN. The first194-case neighborhood run has186passes and eight setup failures (five attention raw-source errors, three durable-postcommit Hooks-unavailable preconditions). Compare those eight exact nodes using only the two original reviewed methods in an otherwise identical combined-tree process, then rerun all194 neighbors with explicit collection-profile ownership for the three config-admitting test modules. No getter mocking or config admission bypass; preserve conditional mode in QA. Run mounted Console approval/Close files with ordinary wrappers and retain original native receipts without new native replay claims.
Final review chat-create presentation follow-up (Qodo6f41bacd investigation):
ADR required: no new ADR.
ADR paths: backlog/decisions/094-console-turn-lifetime-and-navigation-boundary.md; backlog/decisions/150-agent-chat-fork-and-spawn.md.
Reason: routine correction at existing UI-thread confirmation projection; no new lock, authority, owner or persistence contract.
The alleged register-after-sweep ordering is excluded by the shared standalone registry lock; preserve exact call-through62-pass probe and its queue-owner fixture correction. Independently reproduce the separate captured-payload presentation path using real mounted Console: pause at worker marshal after registration, complete source Close, resume and inspect survivor task state/card. Add late sibling-navigation coverage so an old callback cannot replace a live sibling confirmation. Fix only the existing UI-thread marshal currentness seam after valid RED. Preserve live and legacy/no-session consent, FIFO/remount and cancellation. Run complete consent/execution, pending projection, mounted Close and exact performance neighbors; fresh lint/derived preflight and independent review. Retain earlier native receipts with their source limits, no fresh native replay claim. Require clean final-head Qodo/all four hosted gates/latest dev before verified normal merge.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Inspector now derives pending copy and tool-approval counts from the existing session-owned kind registry through a locked kind-count accessor. Queued approval rounds count; questions and skill/worktree confirmations do not. Existing broad interrupt predicate and shared activity precedence are preserved. Eighteen focused checks cover lone question, mixed precedence, queued count2→1, sibling isolation, navigation and fresh remount using actual workers/cards/bridge snapshots. Console guide updated. Existing ADR-067/094/195 apply; no new ADR. Independent review clear; evidence and fixture limits are in Docs/superpowers/qa/2026-10-01-console-tool-ux-followups.md.

Qodo queued-decision finding fixed in the shared review router: scan painted cards in approval-first order independently of queued-round counts. Three mounted cases verify Alt+A, Inspector and attention-tab entry points, with approval priority retained. Independent review is clear.

Final combined tree rebased onto dev31d4f9b764: 44 targeted Console/close/attribute/latency checks passed; one empty exemption set skipped as expected. Fresh derived-artifact preflight passes, with census122 and unchanged startup cap. Final native approval07 and Close08 hashes match production sources; private profiles unchanged. PR2953 retains final-head review/CI/normal-merge checkpoint.

Clean latest-dev27e718f01d81 rebase preserves pending/focus behavior. All three mounted projection cases and two upstream compaction-copy plus two v74 creation/upgrade checks pass on the combined tree, with exercised Python/test hashes unchanged. Earlier130 status/composer/rail cases all have passing evidence; one inherited startup timeout passed unchanged in a fresh private profile. Final preflight and current native approval/Close receipts pass. Exact-head GitHub gates and fresh Qodo resolution remain tracked in PR2953 before normal merge.

Clean rebase onto devab4df9995954 (PR2903 warm-config settings/snapshot paths) preserves all six patches. Combined verification: 16 passed with zero skips (three mounted pending projections and thirteen config warm-read safety cases). Fresh native approval11 passes nine Ask-gated fs_read journeys; Close12 passes four real worker closes and five-kind maximum-risk geometry. Current UI/config hashes, app/process exit0, no network and unchanged real profiles verified. Sanitized receipt: Docs/superpowers/qa/2026-10-01-console-tool-ux-config-integration.json. Final published-head GitHub gates and Qodo resolution remain required for normal merge in PR2953.

Fresh Qodo fixture-documentation finding corrected for all three new pending-projection tests with summary/Args; executable AST unchanged and Ruff/format pass. The previously qualified combined warm-config/projection behavior is unchanged. PR2953 fresh final-head gates/review remain required.

PR2953 final-head Qodo finding bbb8ec52-4caf-42d4-987b-565ebe92410f exposed the standalone chat-create bridge using the generic approval kind. Reopened for this existing pending-kind projection gap; new AC records the confirmation/count/focus/Close outcomes before implementation.

Standalone chat creation now registers its own confirmation kind, contributes zero tool approvals, and uses the shared keyboard/attention-tab Review route. Close declines exact-session standalone rounds and rejects both remembered grants and requests returning from enrichment after its committed fence. Real mounted regressions reproduce the old count and both Close gaps. Latest-dev fccf70d3b0 combined verification: 90 focused checks plus the exact ordinary storage census pass with zero failures/errors/skips; native approval22 passes nine journeys and Close25 six real worker closes plus six-kind short-terminal geometry. Real profiles unchanged, processes exit0, no egress, and current production/source pins verified. Existing ADR-067/094/150/195 apply; no new owner or permission policy. Fresh final-head CI/Qodo and normal merge remain required in PR2953.

Reopened for fresh final-head Qodo finding ece44de5-bd8f-4ef5-97a1-77cebdc2e262: Close can race the remembered-grant finalization and an already-approved executor while its source remains during drain. Existing AC5 covers the no-created-chat outcome.

Final-head Qodo finding ece44de5-bd8f-4ef5-97a1-77cebdc2e262 is fixed at the existing shared boundary: final verdict/grant and resolver updates use the standalone round lock, and both new_chat/fork_chat executors refuse the exact committed Close generation before DB/UI work. Failed-provisional markers remain excluded. Three regressions reproduced grant resurrection and both durable late creates on the published source, then passed. Latest-dev f80d3e0090 qualification passes64 unique affected consent/execution/Close/projection and ordinary storage-census cases with no failures/errors/skips; final-byte artifact preflight passes(census123),25 changed Python files add zero Ruff diagnostics, and changed ranges format cleanly. Native approval23 passes9 journeys; Close26 passes6 real worker closes and short-terminal geometry, with process/socket exit0, no egress and real profiles unchanged. Native/affected replay byte pins are preserved; a subsequent whitespace-only return-dict format is separately proven whole-module AST-identical, and all3 race regressions pass on final bytes. User guide and incident lesson updated. Existing ADR-067/094/150 apply; no new owner/admission policy. Fresh exact-published-head Qodo/all4 GitHub gates and verified normal merge remain required in PR2953.

Reopened for fresh exact-head Qodo59151d44-d74b-48d8-9844-aeeb441d6079: an executor already past entry can outlive bounded Close and queue a late created-chat completion. Existing orphan cleanup is best-effort soft-delete, not physical rollback; acceptance remains no live created chat/no late UI handoff. Helper Returns rulee7e76a0b-b7b4-476b-9ea5-665267b9629c will be corrected.

In-flight Qodo59151d44 and helper-doc rulee7e76a0 are fixed at the existing shared completion boundary: qualify source ownership after worker I/O and synchronously on UI, resolve the current sink, and reuse worker-side best-effort orphan soft-delete when Close wins. Track local UI admission so later retirement never deletes an already-placed chat or masks its original exception. No lock/transaction spans UI handoff; detached live views preserve durable results. Seven interleaving cases reproduced the prior failures; all nine race/view cases pass in the final combined verification. Rebased onto devbb865f5cfe (Resend preserved, combined census124). 78 unique affected/Resend/ordinary-storage checks pass with zero failures/errors/skips; artifact preflight and edited-range formatting pass;25 changed Python files add zero Ruff diagnostics; independent source/integration reviews clear. Native approval24 passes9 journeys and Close27 six real worker closes plus short-terminal geometry; source pins stable, apps/PTYs/sockets clean, no egress and real profiles unchanged. Terminal AX unqualified; native scope excludes the forced race and broader provider/hooks/plugins/MCP coverage. User guide/incident lesson/QA updated. Existing ADR094/150 apply; no new owner or rollback contract. Fresh published-head Qodo/all four CI gates and verified normal merge remain required in PR2953.

Qodo4ae8306e names the existing5s chat-create synchronization timeout in each affected test module;16 related event waits/joins retain exact limits. Whole executable module ASTs unchanged after private constant substitution/removal;235 other pins exact (237 current). Both complete Chat modules pass56 cases, zero failures/errors/skips, real worker/SQLite seams retained. Range format, zero-new Ruff34/full preflight pass. Production/groupedUI/helper owners unchanged; original42/147/native/grouped17 receipts retain identities with mechanical equivalence. No ADR required: test-only refactor, no boundary/dependency/budget/CI/cap changes. QA chat_create_timeout_review_followup records proof. Final published-head review/all four gates/current dev still required for normal merge.

Qodo d9c05b6a late-question/Close race fixed using the existing shared host lock: committed generation publication and question check/register serialize, so late closed-owner questions return cancelled without retained UI/wait while sibling questions remain answerable. Valid RED has2 late-wait failures/1 existing-registration pass; final ordinary3-case regression passes. All194 neighbors pass with explicit existing collection-profile ownership for three config-admitting modules; initial8 setup failures reproduce exactly using only original reviewed methods, not a pure-dev checkout. All37 exact Perf Guards and4 mounted wrappers/17 journeys pass, limits/counters unchanged. Fresh preflight/range format/zero-new Ruff35 and independent review pass. Only2 controller methods change;236 prior other pins exact,238 current. Existing question-test AST unchanged apart from new regression/private timeout. Existing ADR094/067 apply; no new owner or authority. QA question_close_admission_review_followup and incident lesson preserve source identities, fixture modes and limits. Original native receipts retained; no fresh native claim. Fresh published-head Qodo/all four gates/current dev and verified normal merge remain tracked in PR2953.

Final review identified a separate stale captured chat-create UI payload after Close; reproduced lock-order probe excludes Qodo6f41bacd registry-sweep race. Reopening for mounted presentation qualification under new AC.

Delayed chat-create projection now qualifies the current UI sink, live round, committed Close and scoped active session at callback execution. Teardown derives the active parked head, preserving a live sibling. A trusted internal session_scoped flag preserves legacy unparked initial projection; this extension to the marshal-only plan is required by a reproduced legacy regression. Only request_chat_create_confirm and _marshal_pending_chat_create change; callbacks remain outside locks. Existing ADR-094/150 apply, no new owner/authority/persistence or public payload contract.

Mounted navigation and completed-Close regressions reproduce stale source cards on reviewed 54c; an isolated unconditional-clear branch comparison reproduces sibling erasure in the otherwise corrected controller. Qodo's separate register-after-sweep claim is excluded by six real call-through lock-order probes; its prior-head overview is clean/resolved after the rebuttal. Final local qualification passes all 58 checked-in Chat cases plus those six probes (64 total with explicit existing collection-profile ownership), five ordinary mounted wrappers covering 23 fresh-app journeys, and all 37 exact Perf Guards, without failures/errors/skips. Existing journey AST/public wrapper census and 235 other source pins are exact (238 current). Range formatting, zero-new Ruff across 35 files, full derived-artifact preflight and independent read-only review pass. Guide, incident lesson and QA chat_create_ui_projection_review_followup preserve valid/invalid fixture modes and source evidence. Original native receipts retained; no fresh native replay. Task remains In Progress until fresh published-head review and all four CI gates pass; verified normal merge remains tracked in PR2953.

Latest-dev PR2964 queue-button integration: all 38 Console patches replay unchanged onto dev74bd039d. Only the inherited queue region and two generated-style pins change; 235 other prior pins remain exact, with the inherited queue regression additionally pinned (239 total). Fresh coupled qualification passes all 23 Console projection/Close/approval journeys plus all 17 inherited queue-button cases (22 top-level cases), and all 37 exact Perf Guards, with zero failures/errors/skips and unchanged caps/budgets/counters. Zero-new Ruff35 and diff checks pass. QA queue_button_dev_integration retains exact sources and fixture modes. Original64-case/native receipts retain their identities; no fresh Chat or native replay claim for this rebase. Fresh artifact preflight and published-head Qodo/all four hosted gates/current dev still required; task remains In Progress.
<!-- SECTION:NOTES:END -->
