---
id: TASK-488
title: Establish private-path boundary and harden config bootstrap
status: Done
assignee:
  - '@codex'
created_date: '2026-07-23 13:55'
updated_date: '2026-07-23 17:36'
labels:
  - security
  - privacy
  - storage
dependencies: []
references:
  - backlog/decisions/022-local-private-data-boundary.md
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
ADR path: backlog/decisions/022-local-private-data-boundary.md
Reason: Implements the accepted local private-data boundary without changing it.

1. Define stdlib-only structured private-path outcomes and lexical selection.
2. Add red-green POSIX descriptor traversal, ownership/type checks, application-directory hardening, private creation, sticky-parent rejection, race coverage, and honest Windows posture.
3. Route effective config bootstrap read/create through the pinned boundary and remove alternative fallback creation.
4. Add exact repository-root credential filename ignores with git check-ignore regression coverage.
5. Run focused and broader config/UI regressions, static checks, and security review before completing TASK-488.

Detailed plan: Docs/superpowers/plans/2026-07-23-private-path-config-bootstrap.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the ADR-022 private-path boundary and config bootstrap containment. Added the stdlib-only private-path result model, lexical selection, descriptor-pinned POSIX traversal, 0700 application-directory hardening, exclusive 0600 text creation, honest Windows unverified posture, lexical fail-closed config read/create integration, cache invalidation and success-only publication, and exact repository-root credential ignore guards.

Security review corrections require no-follow/nonblocking/no-controlling-terminal capabilities, reject FIFO/device and multiply-linked leaves, pre-encode before filesystem mutation, bound zero-progress writes, retain owner-only residue after post-create failure instead of unsafe name-based unlink rollback, harden existing default config directories, clear stale normalized/raw caches before retry, and surface unexpected git check-ignore failures. Test harness corrections create trusted explicit config parents, isolate UI collection from caller HOME/XDG paths, canonicalize inherited test roots, and restore caller environment independently of sandbox ownership.

ADR required: yes. ADR path: backlog/decisions/022-local-private-data-boundary.md. No new ADR was needed because the implementation follows ADR-022; the conservative retained-residue policy is reconciled in the detailed plan.

Verification with repository Python 3.12.11: focused private/config/ignore slice 106 passed with 1 existing requests dependency warning; broader config/UI slice 260 passed, 16 expected skips, and the same existing warning; default-temp bootstrap probe 13 passed. compileall, dependency-leaf source assertion, git diff --check, scoped Ruff check, and Ruff format check passed. Final independent security review reported no unresolved findings.

Primary implementation and coverage: tldw_chatbook/Utils/private_paths.py, tldw_chatbook/config.py, .gitignore, Tests/Utils/test_private_paths.py, Tests/Utils/test_repository_credential_ignore.py, Tests/test_config_private_bootstrap.py, adjacent config tests, and authorized test-harness isolation files. Detailed plan reconciled at Docs/superpowers/plans/2026-07-23-private-path-config-bootstrap.md.
<!-- SECTION:NOTES:END -->
