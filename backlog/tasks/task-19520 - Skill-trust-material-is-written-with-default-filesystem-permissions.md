---
id: TASK-19520
title: Skill trust material is written with default filesystem permissions
status: In Progress
assignee:
  - '@codex'
created_date: '2026-08-21 17:00'
updated_date: '2026-08-21 23:45'
labels:
  - skills
  - security
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during TASK-17963's review (writer-unique temp files): nothing in
`Skills_Interop/skill_trust_store.py`, `skill_trust_crypto.py`, or
`skill_trust_models.py` sets restrictive permissions on the files it writes —
no `chmod`, no `umask`, no `0o600` (grep-verified twice, by the implementer
and independently by the reviewer). The trust manifest, the encrypted
approved-version snapshots, and the secure-generation marker therefore land
with whatever the process umask gives them, typically world-readable on a
shared machine.

This is pre-existing and was explicitly out of TASK-17963's scope (that task
converted five atomic writes to writer-unique temp names and preserved
existing semantics deliberately). It is filed separately because trust
material is the one thing in the skills subsystem whose confidentiality and
integrity the ADR-009 boundary actually depends on.

Note the encrypted snapshots are encrypted at rest (passphrase-derived), so
the exposure is not equivalent to leaking plaintext skills — but the manifest
and generation marker are metadata an attacker with read access could use,
and defence-in-depth argues for owner-only bits regardless.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Trust-store writes (manifest, snapshots, generation marker) create files with owner-only permissions (`0o600`), and their containing directory is owner-only (`0o700`) where it is created by this code
- [x] #2 Permissions are applied so there is no window where the file exists world-readable before being restricted (set on the temp file BEFORE `replace`, given the atomic-write path from TASK-17963)
- [x] #3 Pre-existing trust files created before this change are tightened on next write (or an explicit decision is recorded that they are not)
- [x] #4 Behavior is verified by a test asserting the resulting mode bits, not by inspection
- [x] #5 Windows/non-POSIX behavior is considered — the change must not break on platforms where `chmod` bits are advisory
<!-- AC:END -->

## Implementation Plan

1. Add RED tests for an explicit owner-only mode on the shared atomic replace
   primitive: `0o600` before replacement, exclusive-create collision ownership,
   descriptor and temp cleanup, and unchanged default behavior.
2. Implement owner-only exclusive temp pre-creation in
   `Skills_Interop/atomic_write.py` while preserving existing callers and error
   propagation.
3. Add RED production-path tests for manifest, snapshot, generation-marker, and
   manifest-rollback bytes; cover snapshot-first directory creation, pre-replace
   modes, and tightening of legacy files/directories.
4. Opt only trust-store JSON/bytes writers into owner-only creation, normalize
   trust-owned directories to `0o700` on POSIX, and secure the trust root before
   snapshot-first creation.
5. Mutation-check both file and directory guards, run the complete Skills and
   full repository suites plus static checks, obtain independent code review,
   then record evidence and close this five-digit task by editing its source file
   directly.
6. Address the post-review stale deterministic-temp finding with a bounded
   owner-only alternate-name retry, regression tests for recovery, exhaustion,
   and cleanup ownership, and only the focused verification requested by the
   user before returning the task to Done.

Detailed plan:
`Docs/superpowers/plans/2026-08-21-task-19520-skill-trust-permissions.md`.

ADR required: no

ADR path: `backlog/decisions/009-local-skill-trust-boundary.md`

Reason: direct hardening of ADR-009's existing persistence boundary; storage
ownership, trust policy, cryptography, authentication, and ACL contracts do not
change.

## Implementation Notes

- Added an opt-in `owner_only` atomic-replace path that exclusively creates the
  writer-owned temp at `0o600`, applies POSIX `fchmod` before the writer runs,
  preserves unexplained collision files, closes descriptors on every path, and
  leaves ordinary atomic-write callers unchanged.
- Routed trust-store JSON and bytes publications through that path, secured the
  trust root before snapshot-first creation, normalized trust-owned directories
  to `0o700` on POSIX, and tightened legacy trust files/directories on their next
  write. Non-POSIX platforms keep the atomic path without relying on advisory
  Unix mode enforcement.
- Added production-path and seam tests for manifests, encrypted snapshots,
  generation markers, rollback bytes, pre-replace modes, descriptor/error
  precedence, legacy tightening, and unchanged default behavior.
- Fresh focused verification after implementation: 53 related tests passed in
  3.46s; Ruff check passed; Ruff format reported four files already formatted;
  `git diff --check` passed. Five targeted mutations each failed at the intended
  permission/cleanup assertion and were restored.
- At the user's direction, the repository-wide run was stopped and was not used
  as TASK-19520's completion gate. Its 272 current failure/error nodes and the 29
  failures from the completed Skills gate are preserved in
  `backlog/docs/task-19520-verification-failure-inventory.md`; all 301 are mapped
  through `TASK-19642` children or four defensible existing-task assignments.
- Independent final production/security review and failure-task accounting
  review reported no remaining findings after the task groups were made atomic
  and allocated above the repository-wide task-ID maximum.
- ADR check: existing ADR-009 applies; no new ADR was required. The change is
  owner-only filesystem defence in depth, not a same-UID, administrator, ACL,
  mount-policy, disk-encryption, Windows-ACL, or multi-tenant isolation boundary.
- Modified implementation/test files: `Skills_Interop/atomic_write.py`,
  `Skills_Interop/skill_trust_store.py`,
  `Tests/Skills/test_atomic_write_concurrency.py`, and
  `Tests/Skills/test_skill_trust_permissions.py`. Verification fallout is
  tracked in `TASK-19642`; the interrupted-cache incident is recorded in
  `backlog/docs/lessons-testing-evidence.md`.



## Notes

Related: TASK-17963 (`Skills_Interop/atomic_write.py` is the natural seam —
its `replace_atomically` already owns the pre-replace window). ADR-009
defines the trust boundary this protects.

This hardening uses existing OS controls rather than superseding them. It can
protect trust material from other unprivileged POSIX users when directory
traversal or a permissive umask would otherwise allow access, but it does not
separate application users sharing one OS identity, restrict same-UID or
root/administrator processes, replace disk encryption, or implement Windows
and storage-specific ACL policy. Effective ACL and mount semantics remain
authoritative; the change is defence in depth, not a new tenant boundary.
