---
id: TASK-37269
title: Fix Windows recovery and helper regressions before portable bundle PR
status: In Progress
assignee:
  - '@codex'
created_date: '2026-10-28T22:53:00Z'
updated_date: '2026-10-28T23:54:00Z'
labels:
  - windows
  - recovery
  - testing
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Resolve the remaining Windows recovery and package-helper failures so the portable bundle can be reviewed with reproducible targeted evidence.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Previously failing encrypted-helper, archive and restore cases pass under the Windows private test environment without bypassing runtime safety checks.
- [x] #2 Native alias and private-metadata regressions exercise real Windows filesystem behavior without requiring administrator-only symlink privileges.
- [x] #3 A fresh isolated launch reopens real persisted data with correct profile/bootstrap selection and unchanged original files.
- [ ] #4 Affected recovery and portable checks, static checks and review evidence are recorded before opening the portable bundle PR.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR paths: existing ADR-125, ADR-126, ADR-127 and ADR-256
Reason: correct demonstrated bugs/test assumptions within existing recovery and helper contracts.

1. Reproduce helper qualification and encrypted caller failures in the isolated Windows test runtime.
2. Correct native metadata/alias and installed-schema fixture assumptions without weakening safety assertions.
3. Trace and correct fresh-child home/bootstrap selectors; exercise actual isolated launch and data-alias refusal.
4. Run affected recovery files plus the portable cohort and static checks; requalify the artifact if runtime bytes change.
5. Review, record exact results and open the portable bundle PR on a codex/ branch.

Detailed plan: Docs/superpowers/plans/2026-10-28-windows-recovery-test-fixes.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Resolved the recorded Windows/helper regression failures without changing production recovery or crypto behavior. The isolated helper dependency was missing in the earlier environment: a disposable interpreter reproduced unisolated PYTHONPATH success and isolated -I transform_failed, while the fully provisioned private runtime qualifies the unchanged hash-anchored worker. Full crypto tests now use native Windows junction/named-pipe/sharing-denial fixtures, CRLF diagnostics, native ACL metadata and recursive child RSS. Explicit streaming-budget cases replace a vacuous default-limit test; bypassing the real pipe budgets makes both fail.

Archive/planner tests now inspect installed native metadata, exercise actual unprivileged NTFS junction refusal with unchanged target bytes, and compare the untouched historical schema-76 fixture's migrated candidate to installed schema84. Six Persona cases use the existing native seed helper rather than an intentionally POSIX-only publisher. Recovery tests inherit owner-private temporary storage. Fresh launch binds HOME and USERPROFILE consistently, asserts native reparse refusal, and avoids bytecode writes in its disposable package copy using -B without increasing the original timeouts or bypassing authority.

Verification: frozen combined affected recovery + portable cohort361 passed, one intentional duplicate-ZIP warning,1210.67s (output/windows-portable-verified-final.log). Packaging/native-builder18 passed plus5 subtests. Full-file controls: archive48, planner110, crypto44; fresh CLI/data alias2 passed. Strict lint on new runtime/package and corrected tests, fatal lint on legacy fixtures, format and whitespace checks pass. Legacy archive style diagnostics predate this change; planner import diagnostics were removed. Fresh independent whole-branch review found no Critical,Important or Minor findings.

The first combined attempt was interrupted when a sandbox write grant made a worktree ancestor shared-writable:325 passed then one refusal/35 setup errors. Guards correctly refused; it is not counted green. Subsequent collection also exposed sandbox-owned source files lacking workspace-owner access after narrowing inheritance. Only this task-owned tree was corrected and source readability/native ancestors were checked before the successful frozen rerun; primary checkout and runtime guard policy remained unchanged.

ADR required:no new decision. Existing ADR-125/126/127 and accepted portable ADR-254/255/256 govern the behavior. Provisional portable ADR IDs252/253/254 were renumbered because current dev already uses252/253. Decision content and approval are unchanged; native patch bytes/provenance stay cd9196d6b7df2e620d906d9e748654ddef604b28672f5f72fcb7b4fc5112669b, now protected by LF checkout policy.

Plan/evidence/rulings: Docs/superpowers/plans/2026-10-28-windows-recovery-test-fixes.md. Existing exact-ZIP/native GUI qualification remains historical to798f8fca75+portable source; these fixture fixes require no new binary. Final dev integration, fresh startup recheck and PR publication are in progress. No full sweep was run under repository opt-in policy.
<!-- SECTION:NOTES:END -->
