---
id: TASK-19520
title: >-
  Skill trust material is written with default filesystem permissions
status: To Do
assignee: []
created_date: '2026-08-21 17:00'
labels:
  - skills
  - security
priority: medium
dependencies: []
---

## Description

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

## Acceptance Criteria

- [ ] Trust-store writes (manifest, snapshots, generation marker) create files with owner-only permissions (`0o600`), and their containing directory is owner-only (`0o700`) where it is created by this code
- [ ] Permissions are applied so there is no window where the file exists world-readable before being restricted (set on the temp file BEFORE `replace`, given the atomic-write path from TASK-17963)
- [ ] Pre-existing trust files created before this change are tightened on next write (or an explicit decision is recorded that they are not)
- [ ] Behavior is verified by a test asserting the resulting mode bits, not by inspection
- [ ] Windows/non-POSIX behavior is considered — the change must not break on platforms where `chmod` bits are advisory

## Notes

Related: TASK-17963 (`Skills_Interop/atomic_write.py` is the natural seam —
its `replace_atomically` already owns the pre-replace window). ADR-009
defines the trust boundary this protects.
