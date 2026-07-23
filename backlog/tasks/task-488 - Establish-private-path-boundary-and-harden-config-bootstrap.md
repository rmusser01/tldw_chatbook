---
id: TASK-488
title: Establish private-path boundary and harden config bootstrap
status: In Progress
assignee:
  - '@codex'
created_date: '2026-07-23 13:55'
updated_date: '2026-07-23 14:48'
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
- [ ] #1 The stdlib-only private-path boundary reports distinct created-private, hardened-private, already-private, unsafe-parent, wrong-owner, link/non-regular, operation-failed, and unverified-platform outcomes.
- [ ] #2 Lexical path selection is preserved for link detection, and unsafe config targets or parent namespaces fail closed before bootstrap content is read or the first config is created.
- [ ] #3 Application-owned private directories are `0700` and first config creation is `0600` on POSIX without a group/world-readable interval.
- [ ] #4 Eligible existing effective config files are automatically hardened to `0600`; no unrelated fallback config file is created.
- [ ] #5 Windows diagnostics report permission enforcement as unverified rather than owner-only or ACL-secure.
- [ ] #6 The repository-root filenames `/openai-api-key.txt` and `/moonshot-api-key.txt` are ignored and covered by a `git check-ignore` regression test.
- [ ] #7 Behavioral tests cover target and parent replacement, shared-sticky-parent creation, symlinks, non-regular objects, ownership, POSIX modes, failures, and Windows posture.
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
