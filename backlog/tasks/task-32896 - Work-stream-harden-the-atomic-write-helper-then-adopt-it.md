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

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented on `fix/tier2-atomic` as `34c0316b01`. Not pushed. **The premise held**: `atomic_file_ops.py`
was the weakest of the three, independently confirmed by the S25 validation slice, which had already run the
same comparison and concluded *"do not consolidate Backup_Recovery's writers down onto atomic_file_ops — the
repo-wide question runs the other direction."*

**Import direction: the barriers moved DOWN into `Utils`, nothing imports upward.** New stdlib-only
`Utils/file_durability.py` owns `flush_file`/`flush_directory`; `native_platform` re-exports them so its
seven importers are untouched. The alternative — `Utils/atomic_file_ops` importing `Backup_Recovery` — would
point a leaf module at a feature package and drag the ADR-126 recovery bootstrap plus loguru/tempfile/shutil
behind all 17 importers of the helper. Note `native_platform.flush_file` already reached *into*
`Utils/windows_files` for its nt branch, so `Utils` was already on the far side of that edge; this makes it
one-directional rather than split.

**Converted (4):** `Audio/voiceprint.py::_atomic_write` (had **no** fsync at all),
`Scheduling/scheduler_heartbeat.py`, `LLM_Provider_Catalog/models_dev_catalog.py`,
`LLM_Provider_Catalog/model_discovery_disk_cache.py`.

**Deliberately NOT converted (6)**, barriers added in place with the reason recorded at each site:
`file_notes_service.py:576` (TOCTOU content-hash recheck must sit between write and rename),
`file_notes_service.py:746` (streams from an open fd, preserves source mode, rechecks source identity),
`raw_participants._replace` (pinned-dir-fd publication — and **one fix did close both**: `template_store.py`
and `mcp_source_participants.write_json` route through it), `mcp_source_participants.backup_corrupt`,
`video_store._atomic_publish` (streams a `BinaryIO`, verifies staged size, commits inside a publication
gate), `Petdex/review.py::write_native_export` (identity/authority recheck between write and rename).

## Three corrections to this task's stated premises

1. **`Petdex/review.py:170` has no `_fsync_parent` to lift.** It has a file fsync and no parent fsync. The
   call was added rather than moved.
2. **`Actor_Packs/publication.py:259` is not a private copy.** It takes an *fd* and returns a
   `"durable"`/`"unsupported"` verdict that is recorded in a publication receipt. Lifting it into the helper
   would change a recovery receipt's semantics. Left alone.
3. **The private-write premise was backwards, and the instruction to match `key_protector`'s hand-rolled
   open was wrong.** `tempfile.mkstemp` already opens `O_CREAT|O_EXCL|O_NOFOLLOW` at `0o600` — verified
   directly: `oct(stat.S_IMODE(os.fstat(fd).st_mode))` is `0o600`. The post-hoc chmod therefore only ever
   *widens* 0o600 -> 0o644; there is no window in which a private file is world-readable. So `private=True`
   **suppresses the widening chmod** instead of re-implementing the open — less code, same property.

**Red before green, precisely:** `test_atomic_write_text_fsyncs_the_parent_directory` failed on the unfixed
helper with exactly one fsync, on a regular file (`st_mode=33152` = `0o100600`, `S_ISREG`), no directory
inode — `AssertionError: only the file was fsynced; the rename itself is still losable`. 7 of the new tests
red, 8 pre-existing green; all 16 green after.

Gates (re-verified independently): `preflight_rc=0`; size ratchet node-id set identical to `origin/dev`;
`Tests/Notes` 89 failed / 3445 passed / 135 errors, **identical name sets at base and after** — zero
regressions; `Tests/Audio` 36 passed; `Tests/LLM_Provider_Catalog` 344 passed with 3 failures proven
pre-existing at base.

**Two pre-existing hangs**, reproduced identically at base with the changes stashed:
`Tests/Backup_Recovery/test_domain_owners.py::test_real_config_save_process_excluded_until_capture_retires`
blocks forever on `proc.stdout.readline()` waiting for `READY`, plus a second subprocess-wait hang in the
same suite. They wedge at 0% CPU indefinitely; run that suite with `--timeout`. Related to TASK-32907.

Follow-up filed as **TASK-32911**: `Skills_Interop/atomic_write.py` is a **fourth** implementation with
**zero** fsync calls (confirmed independently), weaker than the hardened helper on durability but stronger
on `owner_only` precreate — so not a straight conversion. It also carries three one-line Darwin-barrier
upgrades, and records why this class of defect survives: `MCP/permission_store.py:1010` carried a comment
citing TASK-32808.5 asserting it was the only writer fsyncing both file and parent. True when written, false
now, and exactly the kind of note that freezes a weakness in place. De-fossilised here.
<!-- SECTION:NOTES:END -->
