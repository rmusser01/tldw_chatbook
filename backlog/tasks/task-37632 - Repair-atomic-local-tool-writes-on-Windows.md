---
id: TASK-37632
title: Repair atomic local tool writes on Windows
status: Done
assignee:
  - '@codex'
created_date: 2026-10-29 22:34
updated_date: '2026-10-31T17:56:00Z'
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

Queue integration: reproduce stale bundle parity and real Git reader contracts; mechanically move the host-only review reader to existing Agents/code_review_source.py under ADR-281/181, preserve shared hardened Git execution and all refusals, simplify only one-use public text result assembly, regenerate with unchanged builder and 5706 ceiling, verify targeted source/worker/size/boot controls and obtain exact-head independent review. Correct the byte fixture to write explicit UTF-8/CRLF bytes. ADR required: no new ADR; this restores existing host source ownership without changing authority or worker generation policy.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Windows write/read/CAS helpers retain native directory handles, read locked targets through their lock handle, and use FileRenameInformationEx/POSIX replacement to keep the lock through swap. Private-DACL policy, ordinary rename behavior, POSIX mode handling, exclusions and root pins remain intact. Original real worker/stale/absent/cleanup/inode/lock/junction controls passed; six existing file-symlink fixtures require unavailable WinError1314 privileges and remain outside this native evidence.

Fast Lane exposed the regenerated artifact at 5730 lines versus its 5706 limit. A behavior-preserving canonical-source refactor now shares complete descriptor reads, composes the same optional flags with zero defaults, and unifies successful temp bookkeeping. The builder and ratchet are unchanged; regenerated size is exactly 5706. A real multi-chunk preview/CAS case passed before and after the refactor. No flag, CAS check, lock, descriptor cleanup or watchdog protection was removed.

Verification: 43 native atomic/listing/bundled-loopback controls passed, plus 20 write/edit/race/cross-process controls with one POSIX-mode skip; two size-envelope and five workflow-safety checks passed. A Python 3.10.5 bare interpreter imported the artifact and read 1048599 bytes correctly with site packages disabled. The POSIX bundle module cannot collect on Windows because it imports fcntl; its full contracts and early size checks are now selected in the Linux Python 3.12/3.14 workflow. Scoped lint introduces no findings beyond the inherited undefined re. Fresh independent review found no issues, confirmed source/artifact parity, and passed three size/routing checks plus the new multi-chunk case.

Existing ADR-254 governs the retained mutation boundary; no new ADR or dependency is needed. No full local suite was run; current-head GitHub checks remain required before merge.

The full Linux bundle selection exposed a platform-reachability false positive after 194 passing worker checks: the Windows-only native import was counted as reachable on POSIX SSH hosts. Reproduced the exact census failure locally. The guard now recognizes only the true body of the exact os.name == nt condition under ADR-181; its function allowlist is unchanged. Three POSIX-reachable condition mutations and one else-body mutation remain rejected. Real generated-selector execution returns stdlib os on POSIX and retains its loud ImportError on Windows. Moving fcntl into its one POSIX-marked test lets portable bundle contracts collect on Windows. Final targeted verification: 58 passed and one explicit POSIX skip; two size guards and CI routing passed; unchanged builder byte parity and scoped lint passed. Production sources, generated artifact, workflow, limits and permissions are unchanged. Existing ADR-181 applies; no new ADR is required. Independent review and fresh current-head GitHub checks are required before re-arming/merge.

Queue integration follow-up: upstream host-only immutable Git review helpers made the unchanged builder emit 5934 lines while the committed worker remained stale at 5706. Moved ReviewGitReader/ReviewGitEntry and their private argv chunker into the existing Agents/code_review_source.py host owner; normalized ASTs are identical apart from qualified calls to the same shared Git helpers. The hardened byte runner, refusal rules, immutable-object environment, caches and fresh-index checks remain unchanged. The public text wrapper now assembles decoded streams and conditional markers directly in its result, preserving both truncation predicates. Canonical regeneration produces 5705 lines against the unchanged 5706 ceiling; no builder, closure, protocol, permission or queue-policy change.

Targeted verification: 58 worker/native write/listing controls passed with one POSIX skip; 14 portable real-Git/reader/cap/refusal/exclusion controls passed; two size guards and both boot/preimport budgets passed (680/686 boot modules, 558/558 preimport modules, 425288/425347 LOC); routing guard passed; bare Python3.10.5 loaded the stdlib-only artifact and read1048599 bytes correctly. Scoped lint matches HEAD exactly: six inherited exact-type E721 findings in the host owner, no new findings. The moved definitions reuse an existing module, so no project module was added to eager imports.

The wider targeted native run is retained honestly in cli-fix-evidence/queue-review-git-green.txt despite its provisional filename: 7 failures,45 passed,1 skip,88 fixture errors. Most review-source fixtures require unavailable Windows file-symlink privilege; three worker/process fixtures assume POSIX paths/signals/executables, one glob fixture uses a Windows-invalid filename, and a location-rule fixture fails its ambient-home premise. The SHA256 byte fixture was corrected to write explicit UTF8/CRLF bytes with core.autocrlf=false, and its portable rerun passed. The observed large-diff timeout also reproduced against the unmodified pre-repair Git module (queue-git-cap-baseline.txt). No security assertion or platform fixture was removed or weakened. The complete review-source and Git safety modules are now selected in their own existing Linux recovery-matrix invocation and retain cli-review-results.xml; the new routing requirement failed before selection and passed afterward. Those native limitations require fresh Linux CI before merge. No full local suite was run.

ADR required: no new ADR; existing ADR-281 host evidence ownership and ADR-181 worker boundary apply. Independent exact-head review and current-head GitHub checks gate integration.
<!-- SECTION:NOTES:END -->
