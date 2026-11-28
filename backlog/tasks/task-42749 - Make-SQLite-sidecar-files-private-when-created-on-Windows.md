---
id: TASK-42749
title: Make the files SQLite creates beside a private database private when they are created on Windows
status: In Progress
assignee:
  - '@claude'
created_date: '2026-11-27 22:45'
updated_date: '2026-11-28 10:30'
labels:
  - security
  - windows
  - sqlite
dependencies: []
references:
  - tldw_chatbook/DB/private_sqlite.py
  - tldw_chatbook/Utils/private_paths.py
  - tldw_chatbook/Utils/windows_files.py
  - backlog/decisions/126-complete-local-backup-and-recovery.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The private SQLite seam keeps a database's content private even when the database sits in a folder other local users may read: every connect checks the folder chain and makes the database file private. On Windows that promise does not reach the files SQLite creates itself beside the database. The write-ahead log (`-wal`), its index (`-shm`) and the rollback journal (`-journal`) are created by SQLite with whatever permissions the folder hands down, and nothing makes them private until the next connect to that database. The write-ahead log holds recent pages of the database and a rollback journal holds the earlier content of the pages a write changes, so in such a folder other local users can read database content.

Found on 2026-11-27 while TASK-40608 checked whether a connect to an already-open database could skip its checks (it cannot). The gap is on clean `dev` too and is a separate security fix. Performance work keeps connections open longer, which makes "the next connect" later.

The fix must not weaken or skip any check a connect makes today, and must not widen any reuse window.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A regression test that fails on dev shows that in a folder BUILTIN\Users may read, after a write through a connection from connect_private_sqlite, the -wal and -shm files are private without another connect
- [x] #2 What a rollback journal gets is settled either by a test that shows it private during a write, or by a plain statement of what is not covered and why
- [x] #3 Every check a connect makes today still runs and no reuse window is wider
- [ ] #4 POSIX behaviour is verified by a test, and the helper-process path is unchanged unless that test shows the same gap
- [x] #5 Each new guard has a negative control, a mutation that makes a test written for it fail
- [x] #6 The owner approved the design before implementation, and an ADR-126 note records the decision, the probe and what is not covered
- [x] #7 Tests/Architecture and scripts/preflight.sh pass, or each failure is shown to fail the same way on clean dev
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce the gap as a failing test on `origin/dev` (done 2026-11-27).
2. Probe what SQLite does with each of the three files on Windows, and what backup, restore and recovery code assumes about them (done 2026-11-27).
3. Put the options to the owner (done). The owner chose on 2026-11-28: create a missing `-wal`/`-shm` privately before SQLite does, with no pin statement inside connect.
4. Implement test first: writable Windows connects only; after every existing name is checked; skipped for a folder that gives others nothing, a rollback-journal database, an empty `must_exist` file, read-only connects and mode-preserving owners; also after the seam's own switch back to WAL. One mutation per guard.
5. Compare the affected tests with clean dev, run Tests/Architecture and scripts/preflight.sh, record the decision in ADR-126.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**What changed.** A writable connect on Windows creates a missing `-wal` and `-shm` privately, after every existing name beside the database has been checked as before and before `sqlite3.connect`. SQLite treats a zero-length log or index as absent and opens it in place, so the files it writes are the private ones. Owner's choice on 2026-11-28: pre-create, with no pin statement inside connect.

- `Utils/private_paths.py`: `create_private_sqlite_logs` (skips a folder that gives others nothing, a rollback-journal database by its header, and an empty file on a `must_exist` connect); `_create_private_log` (a name that appeared first is opened, classified and made private; a created name is checked afterwards; four attempts, like the existing sidecar revalidation); `create_private_sqlite_wal` for the seam's own switch back to WAL in backup and restore. `PrivatePathResult.directory_mode` (informational, `compare=False`) carries the verified folder's mode, so a private folder costs no extra walk.
- `DB/private_sqlite.py`: the Windows preparation loop records which names are absent and calls the helper on writable, non-mode-preserving connects; `_restore_destination_mode` calls the WAL hook. Line-neutral (4,041 lines).
- `backlog/docs/backup-recovery-owner-inventory.md`: two rows for the new opens.
- ADR-126 2026-11-28 note: the gap, the rule, what is not covered, the cost.

**Not covered** (also in the ADR): the rollback journal (a new file per write transaction); the interleaving where the last connection using the log closes while another is connected but idle (2-12 of 160 writes in the stress probe, against 155-160 before); read-only connects to a WAL database; a database switched to WAL later. A new database that stays in rollback mode keeps two empty private files beside it in a folder others may read (pinned by a test).

**Evidence (Windows 10, Python 3.12.10, SQLite 3.49.1, on `origin/dev` `2df861ec7d`).**
- `Tests/DB/test_private_sqlite_sidecar_privacy.py`: 13 passed, 1 skipped (the POSIX journal test). The reproduction failed on dev before the change.
- Eleven mutations, each failing a test written for it (listed in the ADR note).
- Compared with clean `dev` per test from JUnit reports: `Tests/DB`, `Tests/Utils` and 78 more files naming the log files, 4,938 tests, no outcome differs (one xdist worker crash on both trees dropped ten tests from the changed run; those two files rerun alone: 11 failures each, same ids). `Tests/Architecture` and the helper packaging test, 1,533 tests: only the owner inventory differed, fixed here.
- `scripts/preflight.sh`: all checks pass except Canvas Mermaid generated assets, which needs Python 3.12.11 (this machine has 3.12.10; local-only).
- Cost of the step alone: 0.03 ms in a private folder, 6.6 ms for both names in a readable folder (medians of 30).

**Not verified.** Linux and macOS (no WSL or Docker here); AC #4 waits for CI. Nothing committed or pushed.
<!-- SECTION:NOTES:END -->
