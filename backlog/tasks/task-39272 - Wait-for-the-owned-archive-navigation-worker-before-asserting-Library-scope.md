---
id: TASK-39272
title: Wait for the owned archive navigation worker before asserting Library scope
status: Done
created_date: 2026-11-08 21:17
assignee:
- '@codex'
labels:
- bug
- tests
- console
- ci
priority: high
references:
- Tests/UI/test_console_archive_recovery_flow.py
- tldw_chatbook/UI/Console_Modules/archive_recovery.py
documentation:
- Docs/Architecture/console.md
- Docs/Architecture/library-shell.md
- backlog/decisions/166-agent-assisted-archive-recovery.md
updated_date: 2026-11-08 22:23
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PR #3345's own completion-head UI19 failed its original manual archive recovery/imported-JSON journey at scope active versus all after the Library became visible. Preserve the original behavior and limits while establishing whether the test observes before the product-created archive navigation worker finishes. This is a new genuine CI repair; the three already-completed landing task records and their one completion commit remain untouched.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The original manual-refresh/imported-JSON journey retains every assertion and exactly one click, positively identifies the actual host-owned archive-recovery-library Worker, and observes its completion within the existing 15-second POSIX/60-second Windows wait and original 180-second case cap.
- [x] #2 A controlled native navigation boundary reproduces the original assertion before the correction, passes after it, and demonstrates that real worker failure/cancellation remains failure. Original sources, logs, native phase/closure and unchanged-source evidence are retained without retries-to-green or production changes.
- [x] #3 The complete affected native flow file and original UI19 coverage qualify under unchanged limits, scoped static analysis introduces no findings, and independent correction review accepts the tested candidate. Notes document unchanged runtime/UAT inputs and the separate exact-committed-head review/own hosted-CI requirements before conditional auto-merge.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/166-agent-assisted-archive-recovery.md (existing decision).
Reason: test-only observation of an existing product-created Textual Worker; no runtime, navigation, storage, security, resource or authority boundary changes.

1. Retain own fb0dc68 UI19/run42514930244/job128838099342 raw failure, complete run/job metadata and original Test/adapter/harness sources. The hosted log proves only active-versus-all at line560; its worker state is unobserved. Read the complete existing fixture, click helper, native archive worker/open_library, Library initialization/navigation and related completed tasks38701/38753. Independent read-only source review supports, but does not yet prove, the premature-observation hypothesis.
2. Before permanent changes, run one controlled original-node RED under the unchanged180-second case cap. Hold only the existing Test navigation stub AFTER its real host.push_screen completes but before it returns. Observe the real Textual current Worker, exact host/group, Library-visible/active scope and unfinished worker. Release the hold when the original bounded wait evaluates its genuine predicate, without changing its return value; release again in finally for cleanup. Require the unchanged original scope assertion to fail, then actual physical reap and source/HEAD/index/mode guards. This demonstrates a controlled mechanism, not exact hosted phase attribution.
3. After reaping, make the smallest Test-only correction: retain get_current_worker() inside that existing stub; use the same bounded wait for that exact non-vacuously captured host-owned worker to finish, propagate its real wait() result/failure/cancellation, and keep the Library type and original all-scope/imported-session assertions. No extra click, second allowance, sleep, broad all-worker drain, fake getter, production context change, skip or retry.
4. Run the identical phase control on the corrected original node, plus real worker failure/cancellation controls that must fail honestly. Qualify the whole native flow file together with every original UI19 selector exactly once (expand the two existing flow selectors into the whole file, preserve other selectors/order). Keep original case180 and helper15/60; no hosted job-wall/platform claim from local runs. Freeze all tracked/untracked source membership/SHA/modes, HEAD/index and runner/probe inputs; physically reap before edits.
5. Scoped Ruff/format and original-assertion AST/static checks, exact one-Test-only runtime/Test delta, all product/workflow/package/UAT inputs unchanged, preservation of all122 original artifacts and completed landing task records. Add a concise evidenced testing lesson. Obtain independent tested-candidate review, finish only this atomic repair task after its local AC/DoD, normal commit/hooks, then independent exact NEW-head review/register, fresh actual-ref exact lease/auto-off push and separate remote/GitHub verification.
6. The already-completed tasks and QA README receive no repeated housekeeping changes. Require the correction head's OWN automatic full genuine required/UI/admission/actual Windows/Perf/platform/actionable gates and all-thread/dev/queue checks before conditional auto-merge. No unchanged-head CI rerun or manual required dispatch; preserve every earlier success/failure in its original scope.
Prelaunch preparation review refinement (before any epoch806 launch): explicitly pass -p archive_navigation_phase_probe806 in the actual pytest argv. The first prepared hold could be missed while real push_screen was still mounting. Preserve that unlaunched probe/preparation. Add a Test-Pilot-instance-only click-return interlock: execute the one original native Pilot.click and retain its exact bool, then only for the successful manual-library click await an Event set after the real push_screen returns and the archive Worker is held. Return the genuine bool unchanged. This forces the original bounded predicate to run in the held phase without faking it, adding a sleep, another click or another timeout allowance. Original case180/helper15/60 remain unchanged. Record positive click/held/native owner/state/scope facts; retain finally release and require physical native terminal/task closure. No production/Test source edit is made for this diagnostic.
Second prelaunch refinement, still before epoch806: independent review correctly rejected an unbounded held_ready wait. Preserve v2. Use ONE absolute monotonic deadline set to the existing15s POSIX/60s Windows allowance after the genuine click returns. Within that same deadline, the original helper waits for held_ready OR the actual captured worker to finish; a finished worker is awaited to propagate its native error/cancellation, and a missing hold is rejected. The later original predicate helper is enclosed by that SAME absolute deadline, so no second allowance is added. Normal held-phase observation still returns the original predicate value and releases without yielding. Native finally cleanup/receipt must remain reachable on prerequisite failure.
Actual selection audit before qualification810 corrects the earlier planning shorthand: official scripts/check_ui_pr_gate_census.py --shard18 32 returns66 selectors and contains ONE archive-flow selector (the failing manual/imported-JSON case), not two. Expand that one selector to the complete original flow file in-place; all other65 selectors retain exact order and occur once. The plateau node is covered by the whole flow file, without duplicating another shard. The union remains66 selector arguments, adds only the rest of this directly affected native file, and retains original180-case cap/profiles. This is targeted local coverage, not hosted job-wall or native Windows qualification.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
The archive test now captures the actual native Worker inside its existing navigation stub, waits for that exact host-owned archive-recovery-library worker, and awaits its real result before reading Library scope. All nine original assertion ASTs and the single click remain; four ownership/non-vacuity checks were added. Every other function/class AST, production code, and the original 15-second POSIX/60-second Windows helper and 180-second case cap remain unchanged.

Controlled RED806 reproduced the original scope assertion while the real worker was unfinished after push_screen. The identical probe passed807 after the correction. Native cancellation808 and failure809 still fail at Worker.wait() with WorkerCancelled/WorkerFailed; these are expected negative controls, not passing runs. Uninstrumented810 passed 204 unique cases without skips/errors: all 162 original UI19 cases plus the whole 43-case affected flow file, with the other 65 selector arguments kept in order. All owned processes reaped and source/mode/membership/HEAD/index guards stayed unchanged.

Scoped Ruff introduces no findings (three pre-existing findings outside the changed function); changed-function formatting passes. Independent local correction/evidence review accepts the tested SHA6680c3ddfe9ca8c46b89f426c92c5d7bc1c302e878251b9cc6e571986c72d387 with no actionable findings. The testing-evidence lesson records the visible-screen/unfinished-worker incident. ADR required: no new ADR; this test-only fix preserves ADR-166 and the existing Console/Library boundaries.

Raw original failure, sources, controlled phases, JUnit, selector and physical-closure evidence are retained under completion-metadata-20261108/UI19-failure. The controlled mechanism does not establish the exact unrecorded hosted worker phase. Original browser795/mounted796 UAT is retained only through measured unchanged runtime/package/UAT-module/helper inputs, including its V1/native-child-origin limitation; no later UAT or whole old Test-map identity is claimed. Normal repair commit, independent exact-new-head review and its own full hosted PR gates remain separate landing requirements. The three original Done landing tasks and QA README remain untouched.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Qualified the exact owned archive navigation worker before scope assertions; preserved native failures, original behavior and limits.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
