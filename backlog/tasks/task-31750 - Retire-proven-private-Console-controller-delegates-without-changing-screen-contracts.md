---
id: TASK-31750
title: >-
  Retire proven private Console controller delegates without changing screen
  contracts
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-05 20:06'
updated_date: '2026-09-06 04:50'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Remove redundant private screen forwarding methods so existing controller ownership is the normal call path, preserving public, framework, and documented screen-name contracts.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Exactly the64 approved private delegates are removed and their real callers resolve the existing owners, while public/event/action methods, agent-bridge injection and transcript ephemeral lookup remain intact.
- [x] #2 Callback lookup phases, arguments, await/return behavior and targeted fault-injection semantics are preserved, with explicit late-bound hook regressions.
- [ ] #3 Affected whole-file checks and static ownership tests pass or separately evidenced baseline failures remain tracked, and actual screen size satisfies unchanged ceilings without unrelated compression or boundary moves.
- [x] #4 The residual internals-decomposition UI cases preserve real paste confirmation/reset with unchanged payload, compact ready-state geometry relative to the owning tab row, and visible staged source details; the complete affected file passes.
- [x] #5 The Watchlists follow action remains reachable by a real click after scrolling into view and routes the eligible run exactly once; the complete Console live-work handoff file passes.
- [x] #6 The seven expanded handoff failures retain real restore/supersession, empty-channel, notice, legacy-state, nested card ownership and media-sendability assertions under current owners; navigation evidence proves the actual leave/return phases, including a reused Console instance.
- [x] #7 Ordered-resume release and acquisition-error probes initialize their current prompt owner without changing fault injection, event ordering, logged-error privacy or startup-state cleanup assertions; the complete lifecycle selection passes.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. ADR path: N/A. Follow root-approved Docs/superpowers/plans/2026-09-05-console-private-delegate-cleanup-proposal.md exactly, after TASK31749 purehelper stage. 1. Read task and retain exact64-method caller/patch/phase inventory. 2. Remove delegates by existing-owner cohorts, retarget only actual screen callers and named late-bound wiring lambda bodies, preserving documented excluded screen methods. 3. Migrate receiver-specific test patches and bare-shell wiring, preserving every behavioral assertion and failure/cancellation phase. 4. Verify argument/return fidelity, delayed/replacement-owner callbacks, whole affected83-file census and architecture guards; measure actual unchanged-ceiling counts. 5. Root reviews each cohort before scoped commits; no shared diagnostic fixture edits or STAY extension.
6. Residual UI qualification: reproduce the three internals-decomposition failures before editing. Trace real paste hit testing and current tab-row/Inspector ownership, repair only demonstrated test precondition/geometry drift, preserve confirmation/reset payload and source-detail assertions, and run the complete affected file with static checks and independent review. No new ADR: existing Console topology (ADR-083) remains unchanged. Broader census and size evidence stay open independently.
7. Reproduce the residual Watchlists follow click's OutOfBounds failure, qualify the actual scroll owner with the existing production-CSS harness (the plain harness omits the Inspector scroll rule), then use the existing scroll-to-visible pattern before the real click without replacing it with a direct press. Preserve exact run/route assertions and verify the complete handoff file with review. Track newly exposed handoff failures separately rather than broadening this repair silently. Test-only precondition repair, no ADR required.
8. Reproduce all seven expanded handoff failures. Give direct restore shells an isolated real ConsoleRuntime and current settings wiring, preserving the live app runtime; follow the existing right-rail nested viewport and retrieval helper owners. Trace the real stage/leave/restage path before changing its synchronization, qualify actual navigation phases rather than adding delays, and retain all payload and rendered-source checks. Run all affected complete files, review the diff independently, and record negative controls where new synchronization hides a previously exposed phase. ADR required: no for test-only restoration of established runtime/rail boundaries (ADR-033/083); reassess before any runtime behavior change.
   Diagnosis changed the navigation hypothesis: TASK-31520 intentionally reuses Console. Library navigation completes, but returning resumes the same Console with A still resident and B still pending. No timing-only repair is justified. Proposed bounded runtime repair is to consume and refresh pending live-work handoffs on ordinary resume using the existing claim/acknowledgement and staging paths, preserving suspend/ordered-startup boundaries; hold implementation for design approval under the brainstorming skill. Test-only repairs proceed independently.
9. User approved the bounded resume repair. Extend real navigation coverage across resident/empty launch and new/no-new handoff combinations; assert same-instance resume, no hidden consumption, rendered latest title, sent-notice clearing only on new staging, and a settled claim. Rename/reuse the existing claim-and-refresh helper from ordinary warm resume (not mount replay or ordered saved-chat startup), without adding a timer or changing acknowledgement/failure containment. Verify red before production edits, then complete handoff/reuse/ordered-resume files and static checks with independent review. ADR required: no; routine lifecycle regression repair preserves ADR-033 handoff ownership and TASK-31520 installed-screen reuse, with no new interface or storage policy.
10. Complete lifecycle verification exposed two constructor-bypass fixtures that patch a nonexistent prompt owner before invoking startup. Initialize only that owner with their existing async spy, retain release/error assertions unchanged, rerun the complete lifecycle selection and independently review. ADR required: no; test-only fixture repair preserves existing controller ownership.
<!-- SECTION:PLAN:END -->

11. Remaining exact-node inventory follow-up (2026-09-06): the first 68-file
selection reached 1,091 cases (1,071 passed / 20 failed). Run complete files
containing the 607 unreached cases without another failure-count cutoff. For
reproduced removed-delegate or constructor-bypass fixture failures, wire the
existing real owner or retarget the same spy to that owner, preserving callback
phase, fault injection and every behavioral assertion. Diagnose provider,
per-session stop and dictation failures independently before any correction;
do not infer fixture drift from an exception alone. Preserve and separately
report architecture/size failures unless the exact current ownership contract
can be proved. Run each modified complete file, scoped checks and independent
review. ADR required: no for test-only owner/precondition corrections; assess
separately and obtain design approval for any new runtime behavior change.

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The 64 private delegates and late-bound callbacks are verified; 65 ownership guards pass after rebasing onto 53194eee67. Whole-file census remains incomplete: pre-rebase 1749 passed / 7 failed, two owner fixtures subsequently repaired. New upstream Console work measures 16899 lines / 508 methods against the retained 16818 / 505 ceiling. Further bounded paydown and remaining UI cases are still required. No new ADR; boundaries unchanged.

Residual UI qualification: all three internals-decomposition cases reproduced
(`/private/tmp/tldw-console-residual-ui-baseline.xml`). The paste test now lets
two-row layout settle before the real click and checks confirmation state;
the compact ready-state geometry follows the margin-owning tab row; staged
details are read after opening the actual Inspector. Payload, copy, action-row
and source-detail assertions remain intact. The complete file passes 142 tests
in 325.26s (`/private/tmp/tldw-console-residual-ui-final.xml`).

Watchlists Follow initially reproduced OutOfBounds. Scrolling alone still failed:
the plain destination harness omits the app bundle's Inspector scroll rule.
Reusing the existing production-CSS harness and normal scrolling preserves the
real click and exact eligible run/route/once assertions. The targeted case passes.
The complete handoff file moved from 58 passed/8 failed to 59 passed/7 failed in
90.67s (`/private/tmp/tldw-console-handoff-qualified.xml`). Four constructor-bypass
restore fixtures, a stale launch after navigation, another old tray-index
expectation and a removed private RAG helper reference remain unresolved.

No production/CSS changes, forced scrolling, direct-press substitution, raised
limits or assertion weakening. Both bounded repairs received clear independent
review; full-file Ruff, changed-region formatting and diff checks pass. Three
existing dependency warnings remain. Testing-evidence lessons record the mouse
layout and missing-CSS incidents. No new ADR. AC4 is complete; AC3/AC5 and overall
task status remain open for the unqualified failures and broader census.

Handoff owner follow-up: all seven remaining failures reproduced
(`/private/tmp/tldw-handoff-seven-baseline.xml`). Four restore fixtures now use
an isolated real runtime and current settings controllers; the live app's view
and store are explicitly checked for preservation. Removed the two stubs whose
only purpose was satisfying the unintended runtime attach. Media sendability
uses the existing retrieval helper owner. Card swaps retain the exact
Environment/Tasks/Subagents/tray sequence, outer Live Work order, nested viewport
parentage and stable root/bounded identities in both directions.

Five restore/helper cases and the topology case pass separately. An initial
complete-file run caught a mistyped Tasks selector; after correcting it to the
production ID, final complete handoff verification is **65 passed / 1 failed in
81.59s** (`/private/tmp/tldw-handoff-owner-qualified.xml`), three existing dependency
warnings. Full-file lint, changed-region formatting and diff checks pass;
independent review found no actionable issues in the six test-only repairs.
No production code changed. The runtime-isolation incident is recorded in the
testing-evidence lessons. No new ADR for this fixture/ownership maintenance.

The remaining failure is not waived: diagnostics observed LibraryScreen on leave,
then the SAME ChatScreen on return, with navigation complete, A resident and B
still pending. TASK-31520 intentionally made Console reusable, so compose's
handoff consumption no longer runs on return; the resume hook lacks equivalent
live-work consumption. Keep the existing failing regression and hold the bounded
resume consumption/refresh fix for brainstorming design approval. AC3/AC5/AC6 and
overall task status remain In Progress; broader census/architecture work is open.

Approved warm-resume follow-up: new real-navigation resident/empty cases both
failed before production edits; no-new-handoff controls both passed
(`/private/tmp/tldw-warm-handoff-red.xml`). Renamed the existing supersession
helper to `_stage_pending_console_launch_from_store` and reused it after session
reconciliation on ordinary warm resume only. First-mount/ordered-startup guards,
claim acknowledgement, refresh failure containment and ownership are unchanged;
no timer or queue was added. All four cases then passed in 25.51s
(`/private/tmp/tldw-warm-handoff-green.xml`). Mounted tray text, same-instance
return, no suspended consumption, notice semantics and settled claims are checked.

The first complete five-file lifecycle run reached 145 passed / 2 failed in
262.48s (`/private/tmp/tldw-warm-handoff-lifecycle.xml`), including all 69 handoff
cases passing. Both failures were setup-time missing `_prompts` owners on
constructor-bypass ordered-resume fixtures, before startup executes. Initializing
that owner with their existing async spies restores both cases (2 passed in
3.71s); release/event-order, acquisition-error propagation, log privacy and
startup cleanup assertions remain unchanged. Independent review found no
actionable issues in either bounded change. No new ADR: routine lifecycle and
fixture repair follows ADR-033 and TASK-31520.

Architecture qualification remains red: 10 passed / 2 failed in 2.96s
(`/private/tmp/tldw-warm-handoff-architecture.xml`). Console is 16900 lines
against 16818 (HEAD was already 16899); no method added. The unchanged realtime
wiring routes through `_dictation`, while its unchanged AST test still expects
the retired screen delegate. Neither budget nor guard was relaxed. AC3 and the
broader task remain open; the testing-evidence lessons record the compose/reuse
trap. The prior approval hold is resolved by the user's explicit approval.

Final complete five-file lifecycle selection: 147 passed in 243.84s, six existing warning occurrences (/private/tmp/tldw-warm-handoff-lifecycle-final.xml). Includes all 69 live-work handoff tests, Console/generic reuse, ordered-resume navigation and saved active-path coverage. Full changed-file Ruff, changed-region formatting and diff checks pass; independent review found no actionable issues. AC5/AC6/AC7 are qualified; AC3 and overall status remain In Progress for the broader census and unchanged architecture failures.
<!-- SECTION:NOTES:END -->

Remaining inventory follow-up: six complete repaired files pass together,
127 passed in 71.00s with three dependency warnings
(`/private/tmp/tldw-inventory-six-owner-files-final.xml`). Inspector cost-data,
citation activity/banner setters, permission command output/clear, provider memo
spies and composer visible-session/prompts fixtures now follow current owners.
The impersonate completion double accepts and asserts the exact production route.
Production navigation retains real Settings leave/return and all draft/video/store
assertions while requiring the intended same installed Console instance; prompt
handoff settlement calls the current owner. No behavioral assertions removed,
no production change, no new ADR. Full lint, changed-region format, diff checks
and independent review pass. First inventory pass was 1,071 passed / 20 failed;
607 unreached cases continue in 31 complete files. AC3 and task remain open.
