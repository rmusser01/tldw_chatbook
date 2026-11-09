---
id: TASK-40065
title: Keep captured approval choices on the shared native phone route
status: In Progress
created_date: 2026-11-09 10:29
labels:
- console
- mobile
- approval
- ci
priority: high
references:
- backlog/decisions/235-console-approval-interaction-and-feedback.md
- backlog/decisions/236-served-desktop-approval-relay.md
- tldw_chatbook/Chat/console_remote_rounds.py
- tldw_chatbook/Web_Server/static/mobile/app.js
updated_date: 2026-11-09 13:31
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The latest dev approval work landed while PR #3345 was in CI. The shared phone Console needs to offer and accept the same captured decisions as the owning native round, including narrowed or shared verdicts, without bringing back the retired mobile engine or endpoints.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The active phone snapshot and exact native decision command honor the captured approval row choices. One shared verdict has one phone control, and withheld grants cannot be submitted.
- [x] #2 The rebased branch preserves native receipt ordering, approval feedback and terminal-only review, and passes the original software UAT and relevant local checks without higher runtime or source limits.
- [ ] #3 Independent exact-commit review and the published head’s own hosted gates verify the integration.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/235-console-approval-interaction-and-feedback.md and backlog/decisions/236-served-desktop-approval-relay.md
Reason: Compose the existing captured presentation and feedback with the already-authorized shared native command owner. The authority, persistence, permission scopes, and legacy-route retirement remain governed by the existing decisions.

1. Preserve the exact dev and published refs, replay every commit, and identify current workflow, approval, and decision-reference overlaps.
2. Reproduce the mismatch on the real retained approval worker and native phone command. Bind the projected and accepted options to the exact captured native view, and render one control per verdict key with all reviewed call summaries.
3. Keep exact-target checks and receipt ordering, attach the existing advisory feedback after the worker event release, and retain terminal-only paths. Preserve all original browser and mounted assertions.
4. Reconcile the duplicate decision ID and check the merged test rosters, artifacts, and pins. Run focused native and hosted-lane equivalents and the original UAT at existing limits. Require exact review and own hosted gates before closeout.
5. Keep the real execution-fence test on the admitted bootstrap profile via its existing per-node marker after the connected run showed the fixture redirected config before the fence assertion.
6. After the hosted lint gate exposed the incoming approval call-site debt, make the four existing shortest-zip semantics explicit and render the two exception-type-only teardown warnings before dispatching them. Verify the unchanged Ruff baseline, reviewed diagnostic inventory and connected original tests at the existing limits.
7. Keep exactly the seven failed Architecture import cases on the admitted bootstrap profile. Retain every original assertion and run the original Architecture guard command at the existing cap.
8. Lock the already-enrolled 2,107 UI entries into the checker’s literal coverage floor, preserving their identities and positions. Keep the original deletion assertion, shard map, and runtime caps.
9. Restore ADR-235’s app-lifetime approval label in the canonical option table and the existing phone mirror. Verify the original geometry paint and choice ordering at the existing viewport and caps.
10. Reproduce the native Settings Return stall when saved-default convergence invalidates the first preparation and the existing resume wakeup has already run. Preserve the first refusal. If confirmed, allow one fresh preparation only while the same exact draft, claim, session and mounted visit still own the return. Keep cancellation, supersession, and all original assertions and limits.
11. Diagnose UI28’s pre-holder readiness wait with an actual unrelated mounted message pump held while the real Console refresh settles. If confirmed, narrow only that fixture wait to its existing owner predicates and Console after-refresh signal, preserving the two-second deadline and all held-REBUILD, phase and Stop assertions.
12. Reproduce the independent review’s stale second-discovery edge with an actual A-to-B-to-A session selection and the original route. Recheck the retry’s exact native owner at the modal push boundary, while preserving the first attempt and post-transfer ownership.
13. Retain the exact hosted Perf Guard timeout annotation and all seven passing step logs. Run the original five count-based keystroke census cases on two isolated pytest workers with worker restarts disabled. Keep the job at 15 minutes, every case at its original limit, and the actual latency and compact-setup probes serial. Verify exact commands, same cases and private-profile teardown before independent review and fresh hosted qualification.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
The native phone snapshot and decision command now use the owning round’s captured choices. Shared verdicts display one control with each reviewed summary, and malformed or withheld choices are refused. Receipt recording still precedes worker release and advisory feedback follows it. The terminal-only review path and legacy-route retirement remain governed by ADR-235 and ADR-236.

The original keyboard/native phone cohort passed 41 cases. The connected approval cohort had 174 passes and one fixture refusal, followed by a pass for that exact execution-fence case after preserving its admitted bootstrap profile. All 118 served-native and authentication cases passed. Original software UAT passed all 15 browser cases and the mounted parent and three children. Exact committed-head review and its own hosted qualification are required for AC3.

The b498 hosted follow-up repaired four implicit shortest zip sites and two exception-type-only warning formats without changing the Ruff baseline or runtime behavior. The official diagnostic review and writer changed only the existing warning-owner digest. Seven Architecture import cases now retain the existing admitted bootstrap profile. The original Architecture lane passed 424 cases. The 2,107 already-enrolled UI entries are now the coverage floor, with the census and its original deletion check intact.

The canonical approval label now says Until Chatbook exits in the terminal and phone. The Settings Return restores its exact draft after a real saved-default change invalidates the first preparation. The single retry rechecks its exact native owner before modal push. A real A-to-B-to-A selection during the retry rejects the stale modal and retains the pending draft. UI28 polls the original Console readiness conditions under the existing two-second wait. The held-descendant control passed, and the blocked Console control still expired. No runtime limits or original assertions changed.

The final original UI5 shard passed all 119 existing cases and one new stale-owner case. Chromium and WebKit both passed the phone-option control. The final original software UAT passed 15 browser cases and the mounted parent and three children with ordinary reopen and matching retained artifacts. Optional native return diagnostics retain their mixed results. The passive check recorded zero executions of the added retry branch in those eight optional controls. Exact committed-head review and its own hosted gates remain the AC3 evidence.

GitHub cancelled the cf70156 Perf Guard at its existing 15-minute limit after all seven test steps passed. The five count-based keystroke cases now run on two isolated pytest workers with worker restarts disabled. All five passed in the local candidate invocation. Every selector and timeout is preserved, and the latency and compact-setup probes still run serially. The successor’s hosted wall-clock result remains required.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
