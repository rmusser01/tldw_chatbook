---
id: TASK-38532
title: Expose workflow setup and phase controls in Console
status: In Progress
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
updated_date: 2026-11-05 00:04
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deliver implementation slice 9 of the accepted native shared-worktree workflow design so expose workflow setup and phase controls in Console. Keep existing execution, permission and persistence ownership; this task does not authorize unrelated changes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Native workflow commands/setup/status/history expose exact revisions, current scope, limits, dirty input and change-review readiness without clearing drafts.
- [ ] #2 Approval, revision, Pause, Stop, final acceptance and explicit Resume/recovery use their exact durable owners and remain usable across navigation.
- [ ] #3 Existing fleet/Changes UI, finite DB offload/subscription patterns, design tokens, keyboard/focus and narrow/wide states are reused and verified.
- [ ] #4 Native planning receipts consumed by workflow controls certify the originally acquired configuration generator; substituting another same-runtime generator refuses without a completed receipt or release notification.
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
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

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
