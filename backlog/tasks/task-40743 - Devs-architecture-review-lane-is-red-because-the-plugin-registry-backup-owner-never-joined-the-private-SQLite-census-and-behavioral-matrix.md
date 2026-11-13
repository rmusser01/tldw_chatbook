---
id: TASK-40743
title: >-
  Dev's architecture-review lane is red because the plugin registry backup owner
  never joined the private-SQLite census and behavioral matrix
status: Done
assignee:
  - '@claude'
created_date: '2026-11-13 09:45'
updated_date: '2026-11-13 16:14'
labels:
  - follow-up
  - tests
  - sqlite
  - backup
  - ci
dependencies:
  - TASK-40132
references:
  - Tests/DB/test_private_sqlite.py
  - Tests/DB/test_private_sqlite_inventory.py
  - tldw_chatbook/Plugins/backup.py
  - tldw_chatbook/DB/private_sqlite.py
  - 'https://github.com/rmusser01/tldw_chatbook/pull/3508'
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-up to TASK-40132 (PR #3508, merged 2026-11-12 20:37Z). Checked on dev beab5c1e4e, 2026-11-13.

**Why.** The required "PR Fast Lane (architecture-review tests)" job has been red on dev since #3508 merged, so every code PR's required gate fails and the merge queue cannot land anything. The job fails on exactly two nodes. PR #3415's run of 2026-11-13 (job 131608351750) shows 2 failed, 771 passed, and PR #3430's run (job 131741735076) shows the same two plus one of its own.

**The two nodes.**
- `Tests/DB/test_private_sqlite_inventory.py::test_backup_inventory_matches_current_sqlite_and_settings_operations`. The census finds a `copy_private_sqlite` site it does not expect: `('tldw_chatbook/Plugins/backup', '_Plugins.capture', 'copy_private_sqlite')`.
- `Tests/DB/test_private_sqlite.py::test_every_backup_enabled_owner_has_a_behavioral_operation`. The registry holds a backup-enabled owner that no behavioral operation covers: `recovery.plugins.registry`.

**Cause.** TASK-40132's commit 7de6d1864b added the `recovery.plugins.registry` owner (`centralized_backup_allowed=True`) and the `_Plugins.capture` copy through it. In the same test file it updated the neighbouring pins: the qualified validation dispatch, the C/B id ranges, the opt-in count (22 to 23) and the core recovery factory set. It missed these two. PR #3508's own head (3630632783) is red on both nodes. Dev just before the merge (4ab2be1eeb) is green on both. The precedent for this class of red is TASK-39114.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Both nodes pass, and each is first shown red on dev with the same command
- [x] #2 The plugin registry owner is covered by a behavioral check that runs a real private-SQLite copy under its own owner id. A control shows the check goes red when that owner loses backup authority
- [x] #3 The backup census names the plugin capture's copy site, and no existing expectation is removed or weakened
- [x] #4 The architecture-review lane's own pytest command passes locally on the fix
- [x] #5 Any other red in `Tests/DB/test_private_sqlite*.py` is recorded with its cause and is shown to be identical on dev by test id, rather than hidden
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce both nodes on dev beab5c1e4e, and on the #3508 head, the #3508 merge and the merge before it, to confirm where the red came from.
2. Check production. Does `_Plugins.capture` really copy through `copy_private_sqlite("recovery.plugins.registry", ...)`, and does the owner's policy allow it? If not, the defect is in production and needs a production fix with a red test.
3. If production is right, add the owner to `COPY_BACKUP_OWNER_IDS`. That parametrizes the real-copy behavioral node for it. Also add the census row.
4. Mutation control: turn off the owner's `centralized_backup_allowed` and revert with an Edit.
5. Run the whole private-SQLite class and the lane's command. Compare the failing test ids with dev.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Base.** dev beab5c1e4e. Commands run with an isolated `HOME`, XDG dirs, `TLDW_CONFIG_PATH` and the null keyring.

**Root cause (test-side, both nodes).**
- Production is wired correctly. `_Plugins.capture` (`Plugins/backup.py`) copies `registry.sqlite3` through `copy_private_sqlite("recovery.plugins.registry", ...)`. The owner's policy allows that copy: private_file + read_only_uri, `centralized_backup_allowed`, `recovery_capture_allowed`.
- 7de6d1864b updated its neighbouring pins in `test_private_sqlite_inventory.py` but missed two:
  - the direct backup census row for `_Plugins.capture`;
  - the owner's slot in `COPY_BACKUP_OWNER_IDS`, which is the list that `test_every_backup_enabled_owner_has_a_behavioral_operation` compares with the registry.
- **Provenance.** The two nodes are:
  - red on #3508's own head 3630632783;
  - red on its merge 0b14aac3bf;
  - green on dev just before it, 4ab2be1eeb (the #3507 merge).

  #3508 was merged at 20:37Z, before its architecture-review run finished. That run then failed on these two nodes.

**Change (two test lines, no production edit).**
- `Tests/DB/test_private_sqlite.py`: `recovery.plugins.registry` joins `COPY_BACKUP_OWNER_IDS`, next to `recovery.recovered_media`, whose tree shape the plugins owner copies. That adds `test_copy_backup_owners_execute_real_transactional_copy[recovery.plugins.registry]`. The node runs a real `copy_private_sqlite` under the owner id and reads the row back through the owner's read-only open, the same coverage every other copy owner has.
- `Tests/DB/test_private_sqlite_inventory.py`: the census expects one `("tldw_chatbook/Plugins/backup", "_Plugins.capture", "copy_private_sqlite")` site.
- No expectation was removed or loosened.
- Both lines match the placement in open PRs #3495 (db871e39c5) and #3496 (c42948a4af), which carry the same fix inside larger changes. The resulting blobs are byte-identical (1a10096636..4e4e32cce4, 259ded70ee..8db6c08316), so those PRs merge over this one cleanly.

**RED, then green (AC#1).**
- Dev beab5c1e4e, both nodes: 2 failed. The census shows `Left contains 1 more item: {('tldw_chatbook/Plugins/backup', '_Plugins.capture', 'copy_private_sqlite'): 1}`. The enumerator shows `Extra items in the right set: 'recovery.plugins.registry'`.
- The fix, same command: both pass.
- CI shows the same two nodes red: PR #3415 job 131608351750 (2 failed, 771 passed) and PR #3430 job 131741735076 (the same two, plus its own app-lookup census pin).

**Controls (AC#2).** `centralized_backup_allowed=False` was set on `recovery.plugins.registry` with an Edit, then reverted with an Edit. `git diff -- tldw_chatbook` was empty afterwards. With the mutation, three nodes went red, all on the same refusal, `ValueError: SQLite owner recovery.plugins.registry does not allow centralized backup` (the enumerator on its set mismatch):
- the new copy node;
- the enumerator;
- `Tests/Plugins/test_recovery.py::test_plugin_registry_capture_folds_wal_into_its_exact_catalog`. That test drives `_Plugins.capture` itself under a real capture scope, so the capture path really goes through this owner's private-SQLite copy.

After the revert, all four nodes pass.

**Wider runs.**
- `Tests/DB/test_private_sqlite*.py` plus `Tests/Architecture/test_backup_owner_inventory.py` at `-n 4`:
  - dev: 4 failed, 653 passed, 2 skipped;
  - fix: 2 failed, 656 passed, 2 skipped. The extra passing node is the new copy case.
  - The failing ids, diffed between dev and the fix, differ only by the two target nodes.
- The architecture-review lane's own pytest command (the step from `derived-artifacts.yml`, serial, `--timeout=180`) on the fix: 781 passed, 2 skipped, 1 xfailed (4m23s). (AC#4)
- `Tests/Plugins/test_recovery.py`, `Tests/Backup_Recovery/test_detached_layout.py` and `Tests/UI/test_backup_data_groups.py` with `-p Tests.Plugins.platform_support`, `-n 4`: 108 passed, 3 skipped. The skips are Windows-only.
- `ruff check --select F` on both files: clean.
- `PYTHON=.venv/bin/python ./scripts/preflight.sh`: all derived-artifact checks passed.

**Two other reds in the file set (AC#5): out of scope, recorded only.** Both are identical by test id on dev beab5c1e4e and on 4ab2be1eeb, from before #3508. No CI lane runs either file. Neither involves plugins.
- `Tests/DB/test_private_sqlite_lock_preservation.py::test_copy_borrows_one_two_slot_envelope_for_sequential_preparations` still uses the retired `settings.bulk_backup` owner (`ValueError: Unknown SQLite owner`). bd1a4258ea (TASK-33244) retired it. TASK-39114 re-pointed these uses in `test_private_sqlite.py` (to `tts.profile_migration_backup`) but not in this file.
- `Tests/DB/test_private_sqlite_protocol.py::test_isolated_entry_skips_all_package_initializers` builds a minimal installed tree that lacks `Utils/platform_files.py`. `Utils/private_paths.py` has imported that module at top level since b5251e9a6e (TASK-32628). The isolated helper therefore exits 1 on `ModuleNotFoundError`. The helper swallows the traceback; a scratch copy that prints it showed the error.
<!-- SECTION:NOTES:END -->
