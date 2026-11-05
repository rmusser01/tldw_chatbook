---
id: TASK-38532
title: Expose workflow setup and phase controls in Console
status: Done
created_date: 2026-11-01 17:31
dependencies:
- TASK-38531
labels:
- console
- agents
- workflows
references:
- Docs/superpowers/plans/2026-11-01-native-phased-agent-workflows-implementation.md#task-9
documentation:
- Docs/superpowers/specs/2026-11-01-native-phased-agent-workflows-design.md
- backlog/decisions/285-native-phased-agent-workflows.md
updated_date: 2026-11-05 02:58
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deliver implementation slice 9 of the accepted native shared-worktree workflow design so expose workflow setup and phase controls in Console. Keep existing execution, permission and persistence ownership; this task does not authorize unrelated changes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Native workflow commands/setup/status/history expose exact revisions, current scope, limits, dirty input and change-review readiness without clearing drafts.
- [x] #2 Approval, revision, Pause, Stop, final acceptance and explicit Resume/recovery use their exact durable owners and remain usable across navigation.
- [x] #3 Existing fleet/Changes UI, finite DB offload/subscription patterns, design tokens, keyboard/focus and narrow/wide states are reused and verified.
- [x] #4 Native planning receipts consumed by workflow controls certify the originally acquired configuration generator; substituting another same-runtime generator refuses without a completed receipt or release notification.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: amend existing ADR; no new ADR.
ADR path: backlog/decisions/285-native-phased-agent-workflows.md
Reason: the existing physical-return and consumer contract requires the original configuration generator, not only its mutable context-manager container.

1. Complete the blocking native prerequisite from the scoped Task 8 review: bind the original configuration generator at acquisition, preserve exact return/error semantics, and add native same-runtime substitution refusal coverage. Amend the existing ADR/spec/plan before behavior changes. Keep Task 38531 open and all UI implementation gated until independent prerequisite review reconciles its signoff.
2. Reuse the existing Console command, Goal modal, finite service, subscription and design-token patterns for workflow setup, history, status and exact-revision review controls. Preserve unrelated draft/attachments and keep accepted work in its native runtime owner.
3. Verify mounted command/click/keyboard, stale and duplicate controls, navigation/detach, recovery and narrow/wide behavior with the native runtime and production CSS. Keep live-provider and restart qualification in the existing final qualification slice.
4. Run the targeted affected tests and style governance, update user guidance, self-review and commit only owned changes. Mark acceptance complete only after the independent UI/spec/quality gate.
5. Address the six independent UI findings together. Key approvals to the exact decision, keep drained failures repairable through the native owner, use safe modal dismissal, name the phase being authorized, bypass Default binding repair during setup, and show actionable storage-read failures. Verify each regression on its actual receiving path and preserve earlier qualifications. Existing ADR-285 governs workflow authority and ADR-031 governs safe modal dismissal. No new ADR is required for this correction.
6. Correct the round-one diagnostic disclosure at the workflow presentation boundary. Use fixed operation, error/cause category, bounded structural field/code and retry guidance. Verify malformed synthetic saved input through mounted History plus original storage refusals and retry. Existing ADR-285 applies; no new ADR is required for this error-reporting correction.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

The blocking runtime prerequisite is independently accepted at `1bdfd5f3c6c6306bf89d4bd8a4403a56d2eb9f56`: planning retains its configuration generator at native acquisition, validates that same body before terminal effects, and requires its original exit at the strict receipt consumer. Ten targeted native cases pass; independent review `b17de773f415e0c49941b8d651197477a27a0f551edafaf9f2cf6b14c4596931` is SPEC PASS / QUALITY APPROVED. ADR-285 applies. The terminal substitution negative deliberately retains its original root without creator return. Existing optional-import and unmounted fleet-attention warnings remain disclosed; mounted UI, live provider and restart are separate qualification. UI acceptance criteria remain in progress.

The Console UI implementation is ready for independent review. `/workflow new` prepares a plan for the selected chat and `/workflow` reopens its bounded history. Setup submits passive choices to the existing native planning owner. Detail shows the exact prepared revision, task scope, budget, saved handoff/checkpoint and phase Change Review evidence; approval, Pause, Resume, revision, Stop and quality acceptance route through the original coordinator. Stable decision positions prevent a repeated approval click from becoming Pause. View teardown waits for its already-issued finite reads, while accepted work remains with the runtime. The final targeted UI and adjusted popup cohort passed 18 cases on the private mounted Console host. ADR-285, the Console user guide, and the Task 9 report document the scope, compatibility baselines and verification limits. Remaining UI acceptance criteria await independent review.

Fix round 1 addresses six independent review findings under ADR-285 and ADR-031. Approval receipts now follow the complete reviewed decision, drained failures expose bounded repair and Stop, and phase headings identify the phase being authorized. Modal completion respects the current screen stack. Default setup reads preserve obsolete binding rows, and storage errors leave actionable Retry/Refresh or reopen paths. A resolved background workflow clears its attention link. The private native/UI and compatibility receipts, including the distinct initial mounted Default cleanup failure, are recorded in the appended Task 9 report. UI acceptance remains pending scoped independent review.

Fix round 2 confines workflow diagnostics and unavailable notices to safe operation, type, field and reason metadata. Malformed saved records and local setup validation no longer copy objective text into warnings or error notices. The native storage and command contracts remain with their existing owners. The appended report records the mounted private-history proof, storage retry and exact displayed-selection refusal. Task 38532 awaits scoped independent review.

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->


<!-- TASK9-ORIGINAL-PLANNING-GENERATOR-BEGIN -->
## Task 9 prerequisite — original planning configuration generator

**WHAT.** At original native planning configuration acquisition, including the pre-phase preparation, retain the actual manager and its originally created generator with the issuing runtime and registry. Before terminal effects require that same manager still selects that live original body on that same runtime. At the existing configuration release receive its original native exit normally, verify the originally entered generator completed, and carry this association through the original F receiving frame before issuing a complete receipt. A different generator of the same code/runtime, a closed generator by itself, absent fields, counts or a copied receipt cannot certify return. On uncertain exit preserve the actual partial disposition. Never close a substitute to recover the original, replay ownership release, or require every other legitimate configuration scope to have returned.

**WHY.** The independent 7dc71ac1 native direct probe deliberately replaced only the planning manager's gen with a second already-entered stock generator on the same runtime. F returned and notified with its original generator still suspended, the second closed and one first-send source active. Only a later fixture exit returned the original generator. This demonstrated a false positive receipt, without evidence of spontaneous substitution, a coordinated run, pending children, extra charges or an actually blocked writer.

**COST AND SCOPE.** One passive planning acquisition association and finite source/state checks can reduce availability on a changed or unproven receiver. Existing worker, root, claim, slot, cancellation, G metadata/pop, post-release error and shutdown-fence obligations still apply. The execution-only configuration transition remains separately owned. Preserve existing wake, accounting, consent and public coordinator contracts. Verify original acquisition and terminal substitution refusals and positive direct/coordinated delivery under the original private bounds. Task 38531 and Task 9 UI stay gated until independent prerequisite review reconciles the runtime signoff.
<!-- TASK9-ORIGINAL-PLANNING-GENERATOR-END -->


## Final implementation notes — Console workflow controls

Console workflow setup, this-chat history, exact-revision phase/quality controls and status are independently accepted through `81ec86d2c854795274557e1bdd4725a317e7ba2a`. Existing Console/Goal forms, native coordinator, finite service readers, required checkpoint evidence, exact phase Changes links, fleet navigation and token-backed layout are reused. Native preparation derives frozen provider references from the passive setup; accepted execution remains runtime-owned. View reads drain before detach, modal completion respects the covering screen, and current native Recovery cannot be overridden by a view. ADR-285 and the existing safe-modal contract apply; no extra execution engine, poller, permission store, logger policy or schema was added.

The original six Important findings and background-status Minor were addressed in the first scoped fix. The second scoped fix keeps malformed private record input out of warnings, structured diagnostics and unavailable notices while retaining safe storage errors and retry guidance. Independent final scoped review `dcc0509039f63cd87524e48b35458bfe057852cc7aab4622c68a410e57ba3d88` is SPEC PASS / QUALITY APPROVED with no open findings; prerequisite AC4 retains review `b17de773f415e0c49941b8d651197477a27a0f551edafaf9f2cf6b14c4596931`.

Targeted native/mounted checks, original/fix capture sets, static checks and exact-path protection are recorded in the full report and review receipts. Original staged 61-path diff, 279 protected files and thirteen full support pairs remain preserved. Separate cohorts and adverse outcomes retain their source-specific qualifications, including the unattributed earlier mounted Default cleanup operation, expected retained shutdown cases, optional-library/unmounted-attention warnings and existing legacy/governance failures. No full-suite or clean standalone-branch claim. Full mounted application, live-provider file correction, process restart and profile backup/recovery remain the final qualification stage. The feature stays disabled by default.
