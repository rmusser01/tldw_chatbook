---
id: TASK-34200
title: Admission registry locks the app out after a reboot renumbers the volume
status: Done
assignee:
  - '@claude'
created_date: '2026-10-03 19:30'
labels:
  - backup-recovery
  - startup
  - bug
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
After the owner's Mac rebooted on 2026-09-28, every start of tldw_chatbook failed with "Recovery required: recovery_scope_uncertain" before any config loaded, on every profile. The admission registry identifies registered files by `inode:<st_dev>:<st_ino>`, and macOS gave the data volume a new device number at boot (16777234 → 16777230). The registry's own bootstrap marker, unchanged, same path and inode, therefore looked replaced, and the admission check failed closed. Any macOS user can hit this after a reboot, and the recovery CLI offers no re-enroll for it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A registry recorded before a volume renumbering still admits the app after it
- [x] #2 A file replaced or copied at a registered path is still refused
- [x] #3 New registrations no longer record the device number
- [x] #4 Registries already on disk keep working without any migration step
- [x] #5 ADR-126 records the identity change and the trade-off accepted
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Owner decision 2026-10-03: drop the device number (not a volume UUID). Identity tokens become path + `inode:<ino>` via one helper in `Backup_Recovery/bootstrap.py`. Every comparison against stored `historical` tokens reads old `inode:<dev>:<ino>` tokens as `inode:<ino>`. Convert all producers and comparers (admission, control_records, activation, config_participants, storage_admission, publication, replacement, later_rollback, effective_roots, bootstrap). Regression tests reproduce the owner's registry with a changed device number, plus a negative control on dev. ADR-126 amendment.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Root cause on the owner's machine (2026-10-03): `~/.config/tldw_cli/recovery-bootstrap/admission/registry.json` recorded the `unbound-owner` marker as `inode:16777234:247591642`; after the 2026-09-28 reboot the same file (inode 247591642, ctime unchanged since 2026-09-20) stat'ed as device 16777230, so `control_records._existing_admission_authority` found the marker's token missing from `historical` and raised RecoveryRequired("recovery_scope_uncertain") at `import tldw_chatbook.config`, on every profile. Unblocked by moving the registry aside to `recovery-bootstrap.bak-20261003` (owner's choice; nothing was pending).
Fix (owner's choice: drop the device): `bootstrap.inode_token` produces `inode:<ino>`; `bootstrap.identity_view` reads a stored `inode:<dev>:<ino>` as `inode:<ino>` for every comparison. All producers and comparers converted: admission (`_tokens`, group overlap, publication root check), control_records (startup marker), activation, config_participants, storage_admission, publication, replacement, later_rollback, effective_roots (set equality), bootstrap (pending-record overlap). The journal/publication `(device, inode)` tuples are untouched: they live within one operation, which never spans a reboot. The PERF-08 posture stamps likewise keep `st_dev` (process-lifetime). ADR-126 amended.
Tests: `Tests/Backup_Recovery/test_admission_device_renumber.py` -- the owner's case (registry in the old format with a different device still admits), the fence (a marker replaced at the same path is still refused), new registrations are device-free, and the helper reads both formats. Negative control on dev: 3 of the 4 fail, and the fence test passes on both. `test_first_user_data_binding.py` feeds an old-format registry and still passes (AC #4). `test_selected_owner_absence.py` pinned the stored format (`inode:{st_dev}:{st_ino}`) and was rewritten on purpose. Comparison on 25 identity/admission files against dev: the only branch-only failure was that pin.
Full Tests/Backup_Recovery (303 files, -n 8): 438 failures/errors on the branch, almost all the local RecoveryRequired baseline. Rerunning exactly those on dev left 80 that passed there; rerunning the 80 on the branch, 79 passed (load flakiness under the 1 h full run); the last passes 3/3 on both trees. No regression remains.
Qodo round (#2994): identity_view now normalizes only a well-formed legacy token (`inode:<digits>:<digits>`), so a malformed stored token can never match (new test; a loose-regex mutant turns 2 tests red); replacement builds its token through `bootstrap.inode_token_for`; tmp_path Args docs. The cross-volume false-overlap point is the accepted trade-off: every comparison a false inode match could flip fails closed (recorded in the ADR-126 amendment).
<!-- SECTION:NOTES:END -->
