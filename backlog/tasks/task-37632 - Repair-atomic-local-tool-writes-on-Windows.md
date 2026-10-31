---
id: TASK-37632
title: Repair atomic local tool writes on Windows
status: Done
assignee:
  - '@codex'
created_date: 2026-10-29 22:34
updated_date: '2026-10-30T12:20:00Z'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Allow valid Windows file-tool writes to complete with the existing pinned-directory and stale-write guarantees.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Windows fs_write creates and replaces files through native retained directory handles.
- [x] #2 Stale CAS, root replacement, reparse and sensitive-path controls remain enforced.
- [x] #3 POSIX and remote stdlib worker behavior retain their existing contracts.
- [x] #4 Real worker tests, focused atomic-write tests, lint and documentation are complete.
- [x] #5 Canonical atomic-write sources regenerate a remote worker within the existing size limit while complete multi-chunk reads and write safeguards are retained.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes. ADR path: backlog/decisions/254-cli-search-outcomes-and-listing-compatibility.md. Reason: refine search outcomes and optional listing worker arguments while retaining ADR-078/094/101 authority. Follow Docs/superpowers/plans/2026-10-29-cli-tool-recovery.md with focused red/green tests, implementation, documentation, targeted checks and fresh review.

CI source-size paydown: share repeated descriptor reading, use equivalent optional-flag composition, unify successful temp bookkeeping, and regenerate the bundle without modifying its builder or increasing budgets. Add actual large-file preview/CAS verification and repeat scoped native/bundled controls. ADR required: no new ADR; preserve ADR-254 mutation boundary and existing generation contract.

Bundle import-guard follow-up: reproduce the CI false positive locally by isolating the one fcntl-only test; recognize only the exact Windows-only branch in the existing AST reachability guard, and prove that making the native import POSIX-reachable is still rejected. Execute the real generated selector for both platform branches, repeat source parity/size and native write controls, and request independent review. ADR required: no; test-only correction of the existing ADR-181 POSIX remote-host boundary.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Windows write/read/CAS helpers retain native directory handles, read locked targets through their lock handle, and use FileRenameInformationEx/POSIX replacement to keep the lock through swap. Private-DACL policy, ordinary rename behavior, POSIX mode handling, exclusions and root pins remain intact. Original real worker/stale/absent/cleanup/inode/lock/junction controls passed; six existing file-symlink fixtures require unavailable WinError1314 privileges and remain outside this native evidence.

Fast Lane exposed the regenerated artifact at 5730 lines versus its 5706 limit. A behavior-preserving canonical-source refactor now shares complete descriptor reads, composes the same optional flags with zero defaults, and unifies successful temp bookkeeping. The builder and ratchet are unchanged; regenerated size is exactly 5706. A real multi-chunk preview/CAS case passed before and after the refactor. No flag, CAS check, lock, descriptor cleanup or watchdog protection was removed.

Verification: 43 native atomic/listing/bundled-loopback controls passed, plus 20 write/edit/race/cross-process controls with one POSIX-mode skip; two size-envelope and five workflow-safety checks passed. A Python 3.10.5 bare interpreter imported the artifact and read 1048599 bytes correctly with site packages disabled. The POSIX bundle module cannot collect on Windows because it imports fcntl; its full contracts and early size checks are now selected in the Linux Python 3.12/3.14 workflow. Scoped lint introduces no findings beyond the inherited undefined re. Fresh independent review found no issues, confirmed source/artifact parity, and passed three size/routing checks plus the new multi-chunk case.

Existing ADR-254 governs the retained mutation boundary; no new ADR or dependency is needed. No full local suite was run; current-head GitHub checks remain required before merge.

The full Linux bundle selection exposed a platform-reachability false positive after 194 passing worker checks: the Windows-only native import was counted as reachable on POSIX SSH hosts. Reproduced the exact census failure locally. The guard now recognizes only the true body of the exact os.name == nt condition under ADR-181; its function allowlist is unchanged. Three POSIX-reachable condition mutations and one else-body mutation remain rejected. Real generated-selector execution returns stdlib os on POSIX and retains its loud ImportError on Windows. Moving fcntl into its one POSIX-marked test lets portable bundle contracts collect on Windows. Final targeted verification: 58 passed and one explicit POSIX skip; two size guards and CI routing passed; unchanged builder byte parity and scoped lint passed. Production sources, generated artifact, workflow, limits and permissions are unchanged. Existing ADR-181 applies; no new ADR is required. Independent review and fresh current-head GitHub checks are required before re-arming/merge.
<!-- SECTION:NOTES:END -->
