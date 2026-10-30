---
id: TASK-37630
title: Repair raw shell output and search outcome handling
status: In Progress
assignee:
  - '@codex'
created_date: 2026-10-29 22:34
updated_date: '2026-10-30T08:20:00Z'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep raw shell error details available to models, let empty searches complete without exhausting retries, and preserve Windows command text.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Shell results retain stderr and all executor-provided output beyond 4000 characters.
- [x] #2 Unambiguous no-match grep, rg and findstr calls do not trigger the ordinary failure streak; genuine and ambiguous command failures still do.
- [x] #3 Windows cmd commands retain quoting and preserve intended output and exit codes.
- [ ] #4 Focused regression tests, lint, documentation and ADR review are complete.
- [x] #5 The new CLI regressions run automatically in the existing Linux Python 3.12 and 3.14 PR workflow.
- [ ] #6 Cancellation settlement does not access a worker-owned synchronization lock after confirmed worker termination, while late launch stays refused.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes. ADR path: backlog/decisions/254-cli-search-outcomes-and-listing-compatibility.md. Reason: refine search outcomes and optional listing worker arguments while retaining ADR-078/094/101 authority. Follow Docs/superpowers/plans/2026-10-29-cli-tool-recovery.md with focused red/green tests, implementation, documentation, targeted checks and fresh review.
PR routine follow-up: add a fail-first CI routing contract, select the new regression modules in bounded separate invocations of the existing Tool Recovery job, retain the two-version matrix and timeout, and verify the contract before publication. ADR required: no new ADR; existing ADR-254 and CI execution structure apply.

Current-head review follow-up: reproduce PowerShell backslash/quote parsing with simple full-path searches and compound failure controls; add failing classifier and real-shell regressions, preserve backslashes for PowerShell without broadening recognized command forms, run scoped verification, and have the reviewer verify the fix. ADR required: no new ADR; implements the existing ADR-254 unambiguous-command rule.

CI ferry follow-up: investigate the Python 3.14 cancellation-test timeout and UI-ready module-census failure independently. Replace unsafe inspection of a multiprocessing event after worker termination with an observation of real parent-side launch signaling; retain the actual worker and cancellation behavior. Attribute the census delta without raising ADR-097 ratchets. Verify scoped controls and obtain review before publication. ADR required: no new ADR for the test observation repair; ADR-254 and ADR-097 apply.

Local cancellation diagnostics also exposed the executor finally block re-signaling an abort event after worker death. Add a deterministic post-cleanup signal guard to the real-worker regression, observe RED, then skip abort signaling for a worker already proven dead and retain all bounded tree cleanup.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Removed the provider-only 4000-character slice while retaining executor/storage bounds. Simple complete grep/rg/findstr no-match outcomes no longer consume the failure streak; diagnostics, compounds, truncated previews and ambiguous commands remain failures. Native cmd command text is preserved. Earlier targeted evidence: 80 agent checks; 142 process/native/import checks with 7 POSIX skips plus one repaired import fixture. Existing unrelated E9/F findings were unchanged.

The PR Tool Recovery workflow now selects raw-agent and real-worker regressions separately on Linux Python 3.12/3.14 with bundle equality and result artifacts. The fail-first CI routing contract passed 5 checks. No dependency or matrix expansion.

The user selected superpowers:requesting-code-review instead of Qodo for PR #3314. Its current-head review found POSIX backslash escaping could reject PowerShell full-path searches and conceal compound failures. Eight classifier/native cases failed before preserving PowerShell backslashes, then 33 recovery and 88 provider/integration checks passed. The reviewer verified both native outcomes and found no remaining material production issue. Its minor stdin concern was addressed with DEVNULL; both native cases passed again. Ruff lint/format and diff checks pass; GitHub CI remains pending.

Modified shell helper/provider/executor, regression fixtures/tests, CI routing, architecture/user docs and testing lessons. ADR-254 governs the outcome and worker contracts; this review fix needs no new ADR because it implements the existing unambiguous-command rule. No full suite was run.

CI follow-up: Python3.14 failed while reading a multiprocessing Event after worker termination. Parent-side real signal observation avoids that unsafe test read. A local retry also exposed executor finalization signaling the abort Event after cleanup; a bounded post-cleanup fault guard failed before skipping signals to a dead worker. All46 native process checks pass with7 explicit POSIX skips; scoped Ruff passes. Auto-merge is disabled during repair. The independent boot-census investigation reproduced the same native module set with base runtime sources; no ratchet has been loosened.
<!-- SECTION:NOTES:END -->
