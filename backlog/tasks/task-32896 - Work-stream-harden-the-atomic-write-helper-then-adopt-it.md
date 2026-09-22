---
id: TASK-32896
title: "Work stream: harden the atomic-write helper, then adopt it"
status: To Do
assignee: []
created_date: '2026-09-21 23:05'
labels:
  - tier2-review
  - review-helpers
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
**This inverts TASK-32808.5, which is marked Done.** Measured across four slices,
`Utils/atomic_file_ops.py` is the **weakest** of three durability implementations in this repo: it does
`f.flush(); os.fsync(f.fileno()); os.replace(...)` with **no parent-directory fsync anywhere** and no
Darwin `F_FULLFSYNC`, while `Backup_Recovery/native_platform.py::flush_file` does both and
`Notes/sync_paths.PinnedSyncRoot.replace_bytes` fsyncs the temp file *and* the parent directory.

`os.replace` + a file fsync is atomic but **not durable**: the rename itself can be lost on power failure
until the parent directory is fsynced, and on Darwin plain `os.fsync` does not force the drive cache.
So every further "adopt the shared helper" conversion is a **silent durability downgrade** at the sites
that are currently stronger.

Order matters and is the point of this task: harden first, adopt second, in that order, in one PR.

Source: tier-2 code review 2026-09-21 -- `qa/tier2-code-review-2026-09-21/report.md` (26 slices, 890,356 lines: the surface tier 1 never reached). Per-slice evidence in `qa/tier2-code-review-2026-09-21/slices/`, reproductions in `phase4-verification.md`, and per-finding re-validation against `origin/dev d0face3ebe` in `validation/`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 `atomic_file_ops` fsyncs the parent directory and uses `F_FULLFSYNC` on Darwin
- [ ] #2 A test asserts the parent-directory fsync happens, not just the file fsync
- [ ] #3 The private-write path opens with `O_EXCL|O_NOFOLLOW` and mode-at-open
- [ ] #4 Only after the above, the listed hand-rolled sites are converted
- [ ] #5 TASK-32808.5 is re-opened or superseded with a note saying why
<!-- AC:END -->
