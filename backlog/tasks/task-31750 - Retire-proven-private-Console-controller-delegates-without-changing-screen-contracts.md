---
id: TASK-31750
title: >-
  Retire proven private Console controller delegates without changing screen
  contracts
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-05 20:06'
updated_date: '2026-09-06 03:57'
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
- [ ] #5 The Watchlists follow action remains reachable by a real click after scrolling into view and routes the eligible run exactly once; the complete Console live-work handoff file passes.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. ADR path: N/A. Follow root-approved Docs/superpowers/plans/2026-09-05-console-private-delegate-cleanup-proposal.md exactly, after TASK31749 purehelper stage. 1. Read task and retain exact64-method caller/patch/phase inventory. 2. Remove delegates by existing-owner cohorts, retarget only actual screen callers and named late-bound wiring lambda bodies, preserving documented excluded screen methods. 3. Migrate receiver-specific test patches and bare-shell wiring, preserving every behavioral assertion and failure/cancellation phase. 4. Verify argument/return fidelity, delayed/replacement-owner callbacks, whole affected83-file census and architecture guards; measure actual unchanged-ceiling counts. 5. Root reviews each cohort before scoped commits; no shared diagnostic fixture edits or STAY extension.
6. Residual UI qualification: reproduce the three internals-decomposition failures before editing. Trace real paste hit testing and current tab-row/Inspector ownership, repair only demonstrated test precondition/geometry drift, preserve confirmation/reset payload and source-detail assertions, and run the complete affected file with static checks and independent review. No new ADR: existing Console topology (ADR-083) remains unchanged. Broader census and size evidence stay open independently.
7. Reproduce the residual Watchlists follow click's OutOfBounds failure, qualify the actual scroll owner with the existing production-CSS harness (the plain harness omits the Inspector scroll rule), then use the existing scroll-to-visible pattern before the real click without replacing it with a direct press. Preserve exact run/route assertions and verify the complete handoff file with review. Track newly exposed handoff failures separately rather than broadening this repair silently. Test-only precondition repair, no ADR required.
<!-- SECTION:PLAN:END -->

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
<!-- SECTION:NOTES:END -->
