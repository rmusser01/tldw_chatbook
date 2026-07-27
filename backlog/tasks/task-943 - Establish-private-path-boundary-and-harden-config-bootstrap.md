---
id: TASK-943
title: Establish private-path boundary and harden config bootstrap
status: Done
assignee:
  - '@codex'
created_date: '2026-07-23 13:55'
updated_date: '2026-07-23 18:14'
labels:
  - security
  - privacy
  - storage
dependencies: []
references:
  - backlog/decisions/029-local-private-data-boundary.md
documentation:
  - Docs/superpowers/specs/2026-07-23-local-privacy-containment-design.md
  - Docs/superpowers/plans/2026-07-23-private-path-config-bootstrap.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Introduce the dependency-leaf privacy posture used by later storage owners, then contain first-run and existing effective configuration files without following links or trusting attacker-writable namespaces.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The stdlib-only private-path boundary reports distinct created-private, hardened-private, already-private, unsafe-parent, wrong-owner, link/non-regular, operation-failed, and unverified-platform outcomes.
- [x] #2 Lexical path selection is preserved for link detection, and unsafe config targets or parent namespaces fail closed before bootstrap content is read or the first config is created.
- [x] #3 Application-owned private directories are `0700` and first config creation is `0600` on POSIX without a group/world-readable interval.
- [x] #4 Eligible existing effective config files are automatically hardened to `0600`; no unrelated fallback config file is created.
- [x] #5 Windows diagnostics report permission enforcement as unverified rather than owner-only or ACL-secure.
- [x] #6 The repository-root filenames `/openai-api-key.txt` and `/moonshot-api-key.txt` are ignored and covered by a `git check-ignore` regression test.
- [x] #7 Behavioral tests cover target and parent replacement, shared-sticky-parent creation, symlinks, non-regular objects, ownership, POSIX modes, failures, and Windows posture.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/029-local-private-data-boundary.md
Reason: Implements the accepted local private-data boundary without changing it.

1. Define stdlib-only structured private-path outcomes and lexical selection.
2. Add red-green POSIX descriptor traversal, ownership/type checks, application-directory hardening, private creation, sticky-parent rejection, race coverage, and honest Windows posture.
3. Route effective config bootstrap read/create through the pinned boundary and remove alternative fallback creation.
4. Add exact repository-root credential filename ignores with git check-ignore regression coverage.
5. Run focused and broader config/UI regressions, static checks, and security review before completing TASK-943.

Detailed plan: Docs/superpowers/plans/2026-07-23-private-path-config-bootstrap.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the ADR-029 private-path boundary and config bootstrap containment. Added the stdlib-only private-path result model, lexical selection, descriptor-pinned POSIX traversal, 0700 application-directory hardening, exclusive 0600 text creation, honest Windows unverified posture, lexical fail-closed config read/create integration, cache invalidation and success-only publication, and exact repository-root credential ignore guards.

Security review corrections require no-follow/nonblocking/no-controlling-terminal capabilities, reject FIFO/device and multiply-linked leaves, pre-encode before filesystem mutation, bound zero-progress writes, retain owner-only residue after post-create failure instead of unsafe name-based unlink rollback, harden existing default config directories, clear stale normalized/raw caches before retry, and surface unexpected git check-ignore failures. Test harness corrections create trusted explicit config parents, isolate UI collection from caller HOME/XDG paths, canonicalize inherited test roots, and restore caller environment independently of sandbox ownership.

ADR required: yes. ADR path: backlog/decisions/029-local-private-data-boundary.md. No new ADR was needed because the implementation follows ADR-029; the conservative retained-residue policy is reconciled in the detailed plan.

Verification with repository Python 3.12.11: focused private/config/ignore slice 106 passed with 1 existing requests dependency warning; broader config/UI slice 260 passed, 16 expected skips, and the same existing warning; default-temp bootstrap probe 13 passed. compileall, dependency-leaf source assertion, git diff --check, scoped Ruff check, and Ruff format check passed. Final independent security review reported no unresolved findings.

Primary implementation and coverage: tldw_chatbook/Utils/private_paths.py, tldw_chatbook/config.py, .gitignore, Tests/Utils/test_private_paths.py, Tests/Utils/test_repository_credential_ignore.py, Tests/test_config_private_bootstrap.py, adjacent config tests, and authorized test-harness isolation files. Detailed plan reconciled at Docs/superpowers/plans/2026-07-23-private-path-config-bootstrap.md.

Final whole-range review found two unresolved P1 findings. TASK-943 is reopened pending strict-decrypt cache invalidation/success-only publication and mixed root/UI fixture session-finish idempotence independent of hook ordering. No follow-up implementation has begun.

Mixed-fixture P1 follow-up implemented test-only. Shared root/UI autouse setup now creates the trusted per-test config directory idempotently; pytest_sessionfinish ordering is explicit so UI restores the root fixture state first and root restores the original caller state last, deleting only an owned bootstrap root. RED evidence reproduced the mixed-node FileExistsError and missing hook-order declarations. GREEN regression runs one non-UI and one UI node together and verifies exact environment restoration, owned-root deletion, inherited-root preservation, and external sentinel integrity: 3 passed. Current verification: focused private/config/ignore slice 107 passed with 1 existing requests dependency warning; broader config/UI slice 263 passed, 16 expected skips, and the same warning; compileall, dependency-leaf source assertion, git diff --check, scoped Ruff check, and Ruff format check passed. No production files were edited in this fixture follow-up. TASK-943 remains In Progress pending clean strict-decrypt and fixture re-reviews.

Final strict-decryption P1 follow-up is complete and independently re-reviewed clean. Bootstrap decryption is recursive and strict when encryption is enabled and a password is present: any corrupt enc: value makes bootstrap unsuccessful, returns internal defaults, leaves raw and normalized caches empty, and retries a repaired file on the next ordinary load. The public decrypt_config_section helper remains tolerant and the no-password behavior is unchanged. The mixed-fixture P1 follow-up is also independently re-reviewed clean: shared trusted directory setup is idempotent, ordered UI/root session teardown restores exact caller values, only the owning root is deleted, and inherited external sentinels are preserved. Final verification on HEAD 141172fc plus the harness diff: mixed regression 3 passed; encryption/bootstrap suite 52 passed; exact focused Task 6 gate 108 passed with 1 existing requests dependency warning; broader config/UI gate 263 passed, 16 expected skips, and the same warning. compileall, dependency-leaf source assertion, git diff --check, scoped Ruff check, and Ruff format check passed. Both final P1 reviews report no unresolved findings.
<!-- SECTION:NOTES:END -->
