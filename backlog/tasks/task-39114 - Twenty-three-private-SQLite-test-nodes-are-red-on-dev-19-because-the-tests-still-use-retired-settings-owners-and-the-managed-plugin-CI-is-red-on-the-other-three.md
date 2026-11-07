---
id: TASK-39114
title: "Twenty-three private-SQLite test nodes are red on dev, 19 because the tests still use retired settings owners, and the managed-plugin CI is red on the other three"
status: To Do
created_date: 2026-11-07 02:03
labels:
- follow-up
- tests
- sqlite
- backup
- ci
priority: low
references:
- Tests/DB/test_private_sqlite.py
- Tests/DB/test_private_sqlite_inventory.py
- tldw_chatbook/Backup_Recovery/detached_layout.py
- tldw_chatbook/DB/private_sqlite.py
- .github/workflows/plugin-qualification.yml
dependencies:
- TASK-39053
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 11 of the 2026-10-23 burn-down; checked on dev 2684e5453f, 2026-11-07. TASK-39053's notes (round-11 branch fu/r11-backupinv) list them as pre-existing. Line numbers are dev's.

**The run.** `Tests/DB/test_private_sqlite.py` and `Tests/DB/test_private_sqlite_inventory.py` at `-n 4` on dev: 23 failed, 385 passed, 2 skipped (61.6 s, load 34-47). A rerun of the 23 nodes gave the same 23. None is the RecoveryRequired trap.

| Cause | Nodes | What fails |
|---|---|---|
| Retired settings owners: `settings.integrity`, `settings.bulk_backup` and `settings.vacuum` left the registry in bd1a4258ea (TASK-33244). `test_private_sqlite.py` still uses them as fixture owners (`:76` `COPY_BACKUP_OWNER_IDS`, and from `:1392` on). | 19 | "Unknown SQLite owner" or a KeyError. Includes `test_retired_settings_owner_policies_are_absent`, the `read_only_*` and `copy_private_sqlite_*` nodes, and `copy_backup_owners…[settings.bulk_backup]`. |
| Descriptor-close count drift, not yet explained | 1 | `test_migration_immutable_failure_retains_complete_destination[True]`: `assert 4 == 6` at `:938`. |
| A raw connection added by e350634a8a (portable Windows bundle) is not in the private-SQLite inventory: `Backup_Recovery/detached_layout._normalize_sqlite` | 3 | `test_raw_connection_census_is_qualified_and_transition_aware`, `test_private_sqlite_seam_calls_use_literal_module_owned_ids`, `test_backup_inventory_matches_current_sqlite_and_settings_operations`, all in the inventory file. |

**Lanes.**
- No required lane runs either whole file.
- `plugin-qualification.yml:114` runs the inventory file on PRs that touch Plugins paths. Its latest run (42321038264) is red on ubuntu, macOS and Windows on exactly the three inventory nodes.
- `derived-artifacts.yml:981` runs one inventory node, which is green.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 All 23 nodes pass at `-n 4`, each shown red first with the same command
- [ ] #2 No node is deleted or weakened to go green: every retired-owner node still exercises the same seam with a current registry owner
- [ ] #3 The descriptor-close count is explained (which closes went missing, and why), and the expectation follows production; if it is a production regression, that is fixed instead
- [ ] #4 `_normalize_sqlite`'s connection is qualified in the inventory, and the plugin-qualification job's three nodes pass
- [ ] #5 Both files run on a required PR lane (never the fast PR contract), with three timings of that step's command recorded
<!-- AC:END -->
