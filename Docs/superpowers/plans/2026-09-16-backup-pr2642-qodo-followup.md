# Backup PR #2642 rebase and Qodo follow-up

Task: TASK-32628. Follow-up explicitly requested on 2026-09-16: rebase on latest
dev, address Qodo findings, verify, and merge.

ADR required: no
ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md
Reason: these are corrections within the approved backup and recovery contracts.

## Stage 1: Rebase without changing the verified tree
**Goal:** Put the existing PR content directly on latest dev.
**Success Criteria:** Preserve the previous history locally and verify exact tree equality.
**Tests:** Git tree/diff equality and base-parent check.
**Status:** Complete

Latest fetched dev is `657f70ffe7`. An ordinary rebase attempted to replay 9,952
historical commits and was aborted. Saved `193c276340` at
`codex/backup-pr2642-before-rebase-20260916`; consolidated the exact same tree as
`b5251e9a6e`, directly on dev, using normal commit hooks. The file diff is empty.
Published with an exact old-head force-with-lease.

## Stage 2: Verify and address Qodo findings
**Goal:** Correct actionable review findings without broadening backup scope.
**Success Criteria:** Every finding has a tested fix or an evidence-backed disposition.
**Tests:** Regressions for URL credential exclusion, archive parent replacement,
voice relocation, malformed provider config, staging cleanup, and registration
failure recovery; existing wrapper and admission tests for annotation/docs/constant changes.
**Status:** Complete

All nine existing findings have fixes and regression coverage: protocol-relative
URL credentials, eight installed voice selectors, publication parent identities,
strict recovered-provider config validation, wrapper annotations/documentation,
a named polling interval, failed-staging cleanup, and pre-intent registration-lock
cleanup. Independent boundary reviews found no remaining material issues. Two
earlier server-auth findings remain resolved with their protections retained.

Targeted verification also exposed stale fixtures (directory umask, transient TTS
lock ordering, implemented participants, provider transport hooks, selected-profile
history and Console startup ownership). Corrections preserve production safeguards.
The startup import census is restored to its unchanged 1022-module budget by
deferring absent-root proof until needed. Both failing Windows evidence checkouts
now enable Git long paths before checkout, matching the backup workflow.

Scoped static comparison: all 21 changed Python files parse; 42 existing Ruff
findings and zero new; Bandit has zero findings. Broad targeted run passed 320
cases with three stale-fixture failures. The finite review selection and corrected
Console cases then passed 96 cases; the remaining inventory fixture's cross-owner
assertion ran after authority had been narrowed to one file. Moving that unchanged
assertion before per-file binding passed the focused two-case repeat. All observed
failures are resolved; every new regression has passing local evidence.

## Stage 3: Verify the final branch and merge
**Goal:** Merge the corrected PR after Qodo feedback and required checks are addressed.
**Success Criteria:** Targeted native verification and static checks pass, current
review findings are resolved, GitHub merge requirements are met, and dev contains the merge.
**Tests:** Relevant macOS/Linux/Windows backup checks, scoped Ruff/Bandit, required
PR checks, current-head Qodo review and final remote-state verification.
**Status:** In Progress

Prior native results remain tied to their original source revisions. No old result
will be represented as testing changed review fixes. Preserve the requester's
Change summary and update PR evidence before merging.
