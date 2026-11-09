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
updated_date: 2026-11-09 20:20
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
14. Preserve the hosted pending-layout picker failure and reproduce its zero-region assertion by recomposing the real Notes canvas after the real click opens the picker. Read the same armed button’s shifted geometry at dispatch before the background widget can retire. Keep the single click, ten-second readiness deadline, modal focus, folder, filter and durable mutation assertions. Verify the same forced recompose and all three picker journeys. ADR required: no, because this changes only when the existing fixture observes geometry.
15. Reproduce the review finding by replacing the Move button during Pilot’s first yield. Observe the first dispatch of the actual native Button.Pressed through Textual’s existing test message hook. Compare the originating button and its then-current shifted geometry, deduplicate delivery of the same event, and preserve the original one-click helper and deadline. Require the replacement control to fail and the post-click recompose and all original cases to pass.
16. Reproduce UI31 with a real pending context worker and a competing native FULL sync. Move the existing stricter settle before the synthetic durable-commit window and immediately invoke the same manual timer callback inside it. Preserve every original hold, replay, provider, and timing assertion and the native product policy. Verify the same competitor and the original whole test file.
17. Trace UI19’s actual counted row, compositor clip, tray height and bounded owner. Preserve the original filter and hit-test assertions. Exercise the real legacy action mount after a native fit without that control and change only the measured owner when the causal control identifies it.
18. Trace a same-owner FULL refusal colliding with a previously armed fixed core/control-bar timer. Preserve the fixed callback’s native release opportunity and original maintenance and explicit-ready behavior. Keep the refused FULL’s existing backoff delay when the earlier callback runs, without raising any deadline or changing the original assertions. Verify the actual lock-held CORE control and original replay controls.
19. Trace both Settings Return attempts through actual readiness and context preparation, discovery and exact pre-push ownership. Isolate a native stale-read or read-publication refusal before changing its producer. Keep the one owned convergence reopen, original handoff authority, ABA refusal and waits. ADR required: no. ADR path: N/A. Reason: restore the existing native backoff and exact return contracts. ADR-095 D1 already governs convergence.
20. Reproduce the UI23 Notifications fixture looking up the toolbar while the real two-row table is attached and the toolbar children are still mounting. Use the existing 15-second activation budget for the previous button and the selected replacement. Keep both real overlapping writes, native commits, refresh order, original mutation assertions and runtime limits. ADR required: no. ADR path: N/A. Reason: fixture waits on the existing native mount boundary.
21. Reproduce UI10's empty image-refusal strip by holding the native checked-readiness lock through the original paint attempt. Verify the same staged attachment, blocker and strip assertions after the existing checked-render helper completes. Reuse that helper at the three staged-image call sites and retain every original assertion, Pilot pause, runtime limit and product policy. ADR required: no. ADR path: N/A. Reason: synchronize the fixture on the existing checked display boundary.
22. Reproduce the UI28 Stop visibility failure at the native callback handoff with the real rebuild lock held. Keep the original five-second observation, Pilot cadence, real Stop and phase/custody assertions, and wait on the existing button’s rendered state as well as the queued flag. Verify the same lock control and original file. ADR required: no. ADR path: N/A. Reason: observe completion of the existing checked presentation in the fixture.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
The native phone snapshot and decision command now use the owning round’s captured choices. Shared verdicts display one control with each reviewed summary, and malformed or withheld choices are refused. Receipt recording still precedes worker release and advisory feedback follows it. The terminal-only review path and legacy-route retirement remain governed by ADR-235 and ADR-236.

The original keyboard/native phone cohort passed 41 cases. The connected approval cohort had 174 passes and one fixture refusal, followed by a pass for that exact execution-fence case after preserving its admitted bootstrap profile. All 118 served-native and authentication cases passed. Original software UAT passed all 15 browser cases and the mounted parent and three children. Exact committed-head review and its own hosted qualification are required for AC3.

The b498 hosted follow-up repaired four implicit shortest zip sites and two exception-type-only warning formats without changing the Ruff baseline or runtime behavior. The official diagnostic review and writer changed only the existing warning-owner digest. Seven Architecture import cases now retain the existing admitted bootstrap profile. The original Architecture lane passed 424 cases. The 2,107 already-enrolled UI entries are now the coverage floor, with the census and its original deletion check intact.

The canonical approval label now says Until Chatbook exits in the terminal and phone. The Settings Return restores its exact draft after a real saved-default change invalidates the first preparation. The single retry rechecks its exact native owner before modal push. A real A-to-B-to-A selection during the retry rejects the stale modal and retains the pending draft. UI28 polls the original Console readiness conditions under the existing two-second wait. The held-descendant control passed, and the blocked Console control still expired. No runtime limits or original assertions changed.

The final original UI5 shard passed all 119 existing cases and one new stale-owner case. Chromium and WebKit both passed the phone-option control. The final original software UAT passed 15 browser cases and the mounted parent and three children with ordinary reopen and matching retained artifacts. Optional native return diagnostics retain their mixed results. The passive check recorded zero executions of the added retry branch in those eight optional controls. Exact committed-head review and its own hosted gates remain the AC3 evidence.

GitHub cancelled the cf70156 Perf Guard at its existing 15-minute limit after all seven test steps passed. The five count-based keystroke cases now run on two isolated pytest workers with worker restarts disabled. All five passed in the local candidate invocation. Every selector and timeout is preserved, and the latency and compact-setup probes still run serially. The successor’s hosted wall-clock result remains required.

On 98596fe, Perf Guard passed all seven original steps in 9 minutes 17 seconds with the original limits. UI16 separately opened the Notes picker but later read a zero region from its old Move button. A native post-click recompose reproduced the same assertion after a successful click. The test now records the exact hit-tested button and its shifted geometry at dispatch, keeping one click and the ten-second readiness deadline. The same forced recompose passed, and all three original picker journeys passed. The next exact head still needs its own hosted gates.

Independent review found a second test race during Pilot’s first yield. The retained control replaced the shifted Move button, and the unshifted replacement emitted the actual press and opened the picker. The prior local observation missed that replacement. The final test uses Textual’s existing message hook to record the actual originating Button.Pressed and its first region, once per message, before the modal changes the background. It rejects the replacement control and passes both the real post-click recompose and all three original picker cases. The original click helper, selectors and limits remain as before. The existing testing-evidence lesson records the distinction.

On 1c3c7d0, UI19 reported one match but its row was clipped. A native control measured the partial fit pinning the tray to 10 rows from a 13-row layout. Waiting only for an unlaid mounted list still failed during removal. The tray now uses its owning screen’s layout signal whenever the expected list is absent or unlaid. Independent bounded-Console controls verify both states and the eventual refit. The original filter and paint assertions are preserved.

UI31’s fixture could yield to a real context-owned full sync after opening its synthetic durable-commit window. The existing strict settle now finishes before that window, and the fixture invokes the same manual timer immediately inside it. The same competing-sync control and all four original cases passed with the original assertions and waits. The hosted callback ordering in both failures remains unobserved.

The scoped rail and size run passed 210 cases. An optional Character rail fixture refused profile selection before it reached the UI, and that test is outside the PR gate roster. Diagnostic inventory and all 34 boot ratchets passed. Fresh original software UAT passed all 15 browser cases and the mounted parent and three children with matching retained artifacts, ordinary reopen, and clean guard records. Hosted qualification remains pending on the new source head.

On 4470ef9, the repaired UI19 and UI31 shards passed all 164 and 375 original cases. UI3 still observed seven warm-up refusals in three seconds. A fixed CORE callback could discard FULL’s already-requested longer backoff. A real config lock reproduced eight warm-up starts in the original window. The fix preserves the fixed CORE callback, carries the original absolute FULL deadline, and honors urgent requests before and after promotion and across rearming. The same native control and all 35 replay and coalescing cases passed.

UI5’s native resume overlapped the first attempt, but both attempts returned without restoring the draft. A real old checked read held after native lock release reproduced the same refusal for the current explicit opener. After the old read retires, the opener can issue one ordinary checked read for its still-current key. The original same-key busy admission and exact modal ownership fences remain. Both the real Return control and the native lease retirement regression passed. All 22 original Settings save and Return cases passed, including discovery and stale selection refusal. The hosted callback order is unobserved.

Two supplemental runs retain their mixed results. Eight unmounted Settings doubles lacked a Textual node tree, and a mode-bar double lacked a content field. Those exact unchanged cases are outside the PR gate roster. The native busy open, move, and cancel cases passed. All 207 size and globals cases, the diagnostic inventory, 34 boot checks, all 15 original browser cases, and the original mounted journey passed on this source. Hosted qualification remains pending on the new commit.

On 35fcbaf, UI3 and UI5 passed all original cases with the four replay controls. UI23 found the Notifications table attached but could not find Mark read’s nested toolbar control before the first action. A real held toolbar mount reproduced the same first lookup failure. Both the original and selected replacement buttons now use one existing 15-second activation budget, and the fixture presses the replacement without yielding after its final readiness check. The same native control and all 12 original Notifications cases passed, including both overlapping mutations and their native commit-to-refresh assertions. Runtime, original browser and mounted UAT consumers are byte-identical to 35fcbaf, so those actual software UAT results carry through. The hosted callback ordering remains unobserved, and the new commit must earn its own hosted qualification.

UI10’s hosted staged-image test found an empty strip after both live blocker checks passed. A controlled local run held the real config rebuild lock until the checked display proof expired. The bare sync deferred and left the strip empty. With the same lock control, the existing checked-render helper let all original assertions pass after release. All five original file cases passed after using that helper at the three staged-image sites. The product, the 180-second case limit, and the original software UAT consumers are unchanged. The hosted callback order was not observed.

The separate hosted admission-sensitive 3 step passed the Roleplay body and timed out at its existing three-second native executor join. A local run of the original 213-case step passed, with no live callback at the traced join in the affected case. The observer detected a live native callback in the existing canceled-worker control. This leaves the hosted cause open. Roleplay and shared retirement are unchanged. The new head still requires its own admission and full hosted gates.

UI28’s rebuild case observed the real rule phase idle and the helper retired, while the mounted Stop button still showed the prior state. A native control with the rebuild lock held reproduced the original visibility assertion after the callback cleared its queued flag. A second control established a cold checked read and verified that production replay hid that same button. The fixture now waits on the rendered Stop state and the queued flag within the original five seconds. All 143 original assertions, the real Stop action and the 180-second case limit are preserved.

All 24 original file cases passed. Ruff’s one-line formatting preserved the complete executable AST, and the affected rebuild case passed on those final bytes. Two additional diagnostic attempts stopped at distinct native handoff preconditions and remain separate from the passing receipts. The actual hosted callback order is unknown. The original browser and mounted UAT evidence from the recovery commit carries forward through the same runtime and consumed UAT files.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
