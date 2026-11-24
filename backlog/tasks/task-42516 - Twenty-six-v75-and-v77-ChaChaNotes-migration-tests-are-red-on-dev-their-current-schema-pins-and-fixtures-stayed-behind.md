---
id: TASK-42516
title: "Twenty-six v75 and v77 ChaChaNotes migration tests are red on dev: their current-schema pins and fixtures stayed behind"
status: Done
assignee:
- '@claude'
created_date: 2026-11-24 17:58
updated_date: 2026-11-24 18:17
labels:
- follow-up
- tests
- dev-red
- ci-lanes
- db
priority: medium
references:
- Tests/DB/test_chachanotes_v77_agent_chat_starts_migration.py
- Tests/DB/test_chachanotes_v75_hook_continuation_receipts_migration.py
- .github/workflows/derived-artifacts.yml
- Tests/CI/test_ci_queue_pressure_contract.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the round-25 review of TASK-42514 (2026-11-24) and reproduced on dev 9fa2bf1b1d at `-n 4`: 23 of 50 tests in `Tests/DB/test_chachanotes_v77_agent_chat_starts_migration.py` and 3 of 6 in `Tests/DB/test_chachanotes_v75_hook_continuation_receipts_migration.py` fail. First `E` lines: `assert 87 == 80` and `assert 87 == 81` (current-schema pins), exact-catalog mismatches against the v80 catalogs, `assert 18 == 17` and row mismatches (a new checkpoint column), and `sqlite3.OperationalError: no such column: checkpoint.mixin_snapshot_schema_version` from the native-76 fixture.

The files guard the v75 hook-receipt and v77 agent-chat-start migrations: exact catalogs, lossless legacy checkpoint rows, and refusal of unknown native-76 catalogs. No PR lane runs either file. TASK-32695 mentions the v75 file's version literals in passing but tracks plugin qualification, not these reds. `Tests/ChaChaNotesDB/test_sargable_timestamps.py` is tracked by TASK-41042 and TASK-41954 and is not part of this task.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Each red node's cause is recorded with the commit that moved the schema or fixture past it
- [x] #2 Every node passes on dev; pins of the current schema follow it, checks of a historical step stay literal, and no assertion is dropped
- [x] #3 Both files run in a PR lane step, timed with that step's flags, with their config reads checked, and none of their nodes runs in a second step
<!-- AC:END -->

## Implementation Plan

1. Reproduce through the scratch-profile wrapper and group the 26 nodes by first `E` line.
2. For each group, find the commit that moved the schema or fixture (`git log -S`, landing merge on dev's first-parent line).
3. Read each assert: a current-schema version pin becomes `CharactersRAGDB._CURRENT_SCHEMA_VERSION`; a fresh or upgraded exact catalog names the current lineage; a historical step check stays literal; a fixture that no longer builds the historical file is repaired to build it, keeping its own checks.
4. Check config reads with a temporary `RecoveryRequired` recorder while both files run sandboxed, then lane them in the Dreams riders step (Tests/DB keeps the bootstrap profile, so after `Tests/Tool_Packs`) and in `DREAMS_TRACKING_RIDERS`; time them with the step's flags.
5. Soak 10 times at `-n 4`; run `Tests/CI`, preflight, the Ruff gates and the Dreams step.

## Implementation Notes

**Causes, by node** (all test or fixture drift; red on dev 9fa2bf1b1d, green after):

- **v75 file, 3 nodes: `test_genuine_upgrade_creates_receipts[73]`, `[74]` (`87 == 81` at :53) and `test_v75_failure_rolls_back_ddl_and_reopens_cleanly` (:172).** Pins of the schema the constructor upgrades to. They said 77 on dev, stale since v78 landed via 6feb84c1d2203bc3c6d0eecd2823f4e2409b4b4e (#2918, 2026-10-06). 6b1e2737485dbeafc22ce34d38448f6d5b796ce2 set them to 81 on a branch, but it landed via 14de6845b3365503c75a45c72c8f13ac6538ce65 (#3148, 2026-10-31) when dev was already at v84. The schema had moved past 81 with 01b17108c79d4e5f42332fa492850fcfd795e474 (v82, ADR-239), which landed via 6db2e5b5357f46a638857dd8d7b91bbc58fd343f (#3153, 2026-10-18). Fix: `CharactersRAGDB._CURRENT_SCHEMA_VERSION`. The `version == 74` check after the injected V74→V75 failure is a historical step check and stays literal.
- **v77 file, `test_v77_keeps_every_predecessor_checkpoint_column_and_indexes[False]`, `[True]` (`87 == 80` at :71).** The pin of 80 was set by 09357e62150bd52e2bf7c7c7a043653065bec470 (landed via #3043, 2026-10-08). The schema moved past it with d24c4c04b2fef49e49ab9ac4ac6eae1704ba3fda (v81, ADR-232), which landed via a7bde4e5e5a13a4b8325f094dbd968a4f96c60a7 (#3010, 2026-10-13). Behind that pin, `row == before` also failed on the checkpoint column `mixin_snapshot_schema_version`. That column came with 7970fbcada1e3ed7f75f524636e855f2d1f6c250 (v86, ADR-286), which landed via 18482c1cf7003b4e64939c7e216612bcb628d683 (#3346, 2026-11-02). Fix: `CURRENT`, and the row's new column is popped and asserted `None` (a predecessor row has none) before the unchanged `row == before`.
- **`test_legacy_queue_constructor_preserves_predecessor`, 4 nodes (`row == before` at :586).** The same v86 column, then the same v80 pin. Same fix.
- **`test_populated_legacy_queue_paths_are_lossless`, 6 nodes (`18 == 17` at :733).** The v86 column (popped and asserted `None`; `len(row) == 17` is kept). Behind it were the exact catalog `CHACHANOTES_V80_SCHEMAS[0]` (stale since v81) and the v80 pin. The catalog now names `CHACHANOTES_V87_SCHEMAS[0]`; the dictionary case already used the moving `CHACHANOTES_DICTIONARY_UPDATE_SCHEMA`.
- **`test_v77_constructor_catalog_matches_exact_native76_capture`, 4 nodes (`actual == expected` at :502).** The fresh constructor catalog. The primary catalog moved with v81. The shared one also gained `idx_subscriptions_source` in the Subscriptions constructor (task-34399, e043992f56594e72578d235280083631488a627c, landed via dcdd088f3e #3083, 2026-10-14). They now name `CHACHANOTES_V87_SCHEMAS[0]` and `_SUBSCRIPTIONS_CHILD_SEGMENTS_V87_SCHEMAS[1]` (current primary plus the source-indexed Subscriptions catalog). The pin becomes `CURRENT`.
- **`test_unknown_native76_catalog_refuses_without_rewriting`, 3 nodes, and `test_legacy_native76_upgrade_preserves_exact_receipts[False-False]`, `[True-False]` (`no such column: checkpoint.mixin_snapshot_schema_version`).** These are fixture failures. `_legacy_native76` wrote its machine owner into a v76 file with today's `ConsoleDispatchRepository.insert_with_messages`, which since 7970fbcada reads the v86 column before it writes. The fixture now runs the repository's own statements for that acceptance (two messages with their semantic revisions, the active leaf, the checkpoint row, the graph epoch) through `_seed_native76_machine_owner`, without that read. This follows the file's `_seed_legacy_queue_owner` pattern.
- **`test_legacy_native76_upgrade_preserves_exact_receipts[False-True]`, `[True-True]` (`catalog in (...)` at :337).** This is also a fixture failure. The shared native-76 file was built with today's Subscriptions constructor, which adds `idx_subscriptions_source` (task-34399) and stamps Subscriptions v3. The v3 stamp came with ADR-207, cd65558bc9a714bcb36f3e938ab3abaf5f358efb, committed on dev on 2026-10-12. A native-76 shared file had neither, and the product's V76→V77 shared-file gate requires the v2 stamp. The fixture now drops the index and stamps v2; ADR-207's v3 keeps the DDL unchanged, so that is the native-76 shared catalog exactly. A probe confirmed the dropped index was the only catalog difference. The :337 check is unchanged.
  - Behind these four nodes: the v80 pin; the rows, which now carry the v86 column (popped, asserted `None`, then compared to `before`); and the expected catalog. That was the v78-v80 transform chain of the pre-upgrade catalog; it is now the current lineage's native-76 member, `CHACHANOTES_V87_SCHEMAS[2|3]`, or for a shared file `_SUBSCRIPTIONS_CHILD_SEGMENTS_V87_SCHEMAS[4|6]` (the native catalog with the Subscriptions catalog that has no source index).

**Kept literal (historical steps):** the v75 file's `version == 74` after the injected failure; the v77 file's `== 76` and `== 77` stamps on the native and standalone predecessors.

**Config reads.**
- Run sandboxed: the Tests/DB rule in `Tests/conftest.py` and the v75 file's `bootstrap_profile` mark were temporarily removed, and `RecoveryRequired.__init__` temporarily recorded every construction. All 56 tests passed and the recorder logged none, so neither file reads config in-process.
- The recorder was validated on `Tests/Chat/test_console_trace_sidecar_redaction.py`, a known trap file: 10 lines for its 5 reds. Every probe edit was reverted (`git status` clean).
- Both files still keep the bootstrap profile, by the conftest directory rule and the v75 mark.

**Lane.**
- **Placement:** the Dreams tracking riders step, after `Tests/Tool_Packs` (bootstrap profile) and after the provider failure copy file. Both are also in `DREAMS_TRACKING_RIDERS`.
- **No double runs:** no other PR-lane step lists either file or `Tests/DB`; the contract's one-owner check confirms it.
- **Serial times with the step's flags:**
  - v77 file: 50 tests, 27.2 s, load 12.9→13.5.
  - v75 file: 6 tests, 4.5 s, load 13.5→13.7.
- **Lane-install venv:** both passed (56) in a scratch venv built from the lane's install line, with pytest 9.2.0.

**Checks.**
- **Red/green:** dev, 26 failed and 30 passed; after, 56 passed.
- **Soak:** both files plus the queue-pressure contract, 10 runs at `-n 4`: 10/10 green (212 passed each), 1-min load 14.0-43.0 with an 8-process CPU burner.
- **`Tests/CI` at `-n 4`:** 1947 passed, 2 skipped, 1 failed. The failure, `test_bootstrap_root_cleanup.py::test_owned_root_is_removed[fork-child-terminated]` (`child=None`, at load ~13), is code this branch does not touch; the file passed 5/5 on rerun. It is the fork-SIGTERM class in TASK-42470's lesson.
- **Static gates:** preflight passed, `ruff check tldw_chatbook --select F` is clean, and `Tests/lint/ruff_gate.py` passed.
- **The step:** `r19-step.sh fu-r25-devreds "Run the Dreams tracking tests and unlaned riders" r25-step2 -p no:xdist --timeout=180`: 30 targets, 1442 passed, 6 skipped, 0 failed, 13 min 53 s wall (pytest 822.83 s), 1-min load 35.9 at start, 9.1 at end.

**Seen, not changed (follow-ups for the coordinator).**
- `Tests/Backup_Recovery/test_chachanotes_native76_compatibility.py` imports `_legacy_native76` and is red on dev: 28 failed, 18 passed. With this fixture, 16 still fail on its own stale pins: `== 80` at :117, the v80 catalogs at :555 and :614, and ADR-207's v2→v3 migration at :253. Its one laned node, `test_shipped76_staged_failure_rolls_back_checkpoint_rebuild` (console-p0-gate), passes.
- **Product question:** a native-76 shared file that today's `SubscriptionsDB` opens first is migrated to Subscriptions v3 and gains `idx_subscriptions_source`. `CharactersRAGDB` then refuses it with `V76→V77 unsupported native receipt catalog`; opened by ChaChaNotes first, it upgrades to v87. Both orders were probed.

**Files.** `Tests/DB/test_chachanotes_v77_agent_chat_starts_migration.py`, `Tests/DB/test_chachanotes_v75_hook_continuation_receipts_migration.py`, `.github/workflows/derived-artifacts.yml`, `Tests/CI/test_ci_queue_pressure_contract.py`.
