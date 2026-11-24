---
id: TASK-42514
title: "Six durable fleet progress tests are red on dev: their schema pins stayed at v80"
status: Done
assignee:
- '@claude'
created_date: 2026-11-24 16:54
updated_date: 2026-11-24 17:19
labels:
- follow-up
- tests
- dev-red
- ci-lanes
priority: medium
references:
- Tests/Agents/test_durable_fleet_messages.py
- .github/workflows/derived-artifacts.yml
- Tests/CI/test_ci_queue_pressure_contract.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by round 24 of the follow-up burn-down (2026-11-24) and reproduced on dev 9fa2bf1b1d. Six nodes of `Tests/Agents/test_durable_fleet_messages.py` fail: `test_existing_chat_migrates_and_progress_metadata_never_exposes_bodies[73]`, `[74]`, `[75]`, `[76]`, `test_chat_v76_installed_backup_and_shared_subscription_schema` and `test_v79_browse_index_migration_rolls_back_partial_ddl_then_retries`. Each stops at a literal schema version of 80 (`assert 87 == 80`, or `(87, ...) == (80, ...)` for the recovery adapter's schema policy).

The file guards durable fleet progress: migration of existing chats, the progress table's privacy, the recovery adapter's view of the current schema, and the V79→V80 rollback and retry. No PR lane runs it, so it has been red without anyone seeing it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The cause of the six reds is recorded, with the commit that made them red
- [x] #2 The six nodes pass on dev, and every other assertion in the file is kept
- [x] #3 The file runs in a PR lane step, timed with that step's flags, and none of its nodes runs in a second step
<!-- AC:END -->

## Implementation Plan

1. Reproduce the six reds through the scratch-profile wrapper and read each first `E` line.
2. Find when the pins went stale (`git log -G` on `_CURRENT_SCHEMA_VERSION`, landing merge on dev's first-parent line).
3. If the schema moved on purpose, replace the literal 80 with `CharactersRAGDB._CURRENT_SCHEMA_VERSION`, the form `Tests/Backup_Recovery/test_core_owners.py` uses; keep the V79→V80 label and error checks, which name a historical step.
4. Lane the file in the Dreams tracking riders step (sandboxed, so before `Tests/Tool_Packs`) and in `DREAMS_TRACKING_RIDERS`; time it serially with the step's flags.
5. Soak the file 20 times at `-n 4` under load; run `Tests/CI`, `Tests/Architecture` serially, preflight and the Ruff gates; run the changed step.

## Implementation Notes

- **Cause: test drift.** The file pins the current ChaChaNotes schema as a literal 80 in four places: the core recovery adapter's current schema (`schema_sql[0] == (80, actual)`), the stamp restore after the wrong-stamp checks, and two post-migration version checks. Its last edit, ebf4c11a3d614d181fb31a31b9a42e0971dd1c84 (2026-10-08), set those pins when v80 was current. d24c4c04b2fef49e49ab9ac4ac6eae1704ba3fda moved the schema to v81 on purpose (ADR-232 device-local response rules); it landed on dev as a7bde4e5e5a13a4b8325f094dbd968a4f96c60a7 (PR #3010, 2026-10-13), where `_CURRENT_SCHEMA_VERSION` goes 80 → 81 and the test still says 80. Later bumps (now 87, the ADR-337 amendment) kept it red. No PR lane ran the file, so nothing reported it.
- **Fix.** `CURRENT = CharactersRAGDB._CURRENT_SCHEMA_VERSION`, the form `Tests/Backup_Recovery/test_core_owners.py` already uses for the same adapter check; the stamp restore binds it as a parameter. Every other assertion is unchanged, including the V79→V80 label and `SchemaError` checks, which name a historical step.
- **Red/green.** Dev 9fa2bf1b1d, `-n 4`: 6 failed (`assert 87 == 80` at lines 505 and 860, `(87, ...) == (80, ...)` at line 255), 10 passed. After: 16 passed.
- **Lane.** Dreams tracking riders step (`dreams-tracking`), before `Tests/Tool_Packs`: every test is `@private_profile_test`, so the file is sandboxed. Added to `DREAMS_TRACKING_RIDERS` in `Tests/CI/test_ci_queue_pressure_contract.py`, whose one-owner check confirms no other PR-lane step lists the file, a test in it, or a parent directory (only `Tests/Agents/test_fleet_messages.py` was laned before, in pr-fast-lane). Serial time with the step's flags (`-p no:xdist --timeout=180 --tb=short`): 16 tests, 44.0 s, 1-min load 13.2 → 11.8. Also 34/34 (with TASK-42515's file) in a scratch venv from the lane's install line (pytest 9.2.0).
- **Checks.** Soak, both files, 20 runs at `-n 4`: 20/20 green (34 passed each), 1-min load 6.9-36.3 with an 8-process CPU burner over runs 5-20. `Tests/CI` `-n 4`: 1948 passed, 2 skipped. `Tests/Architecture` serially: 1507 passed, 5 skipped. Preflight passed; `ruff check tldw_chatbook --select F` clean; `Tests/lint/ruff_gate.py` passed. The step: `bash $FU/r19-step.sh fu-r25-devreds "Run the Dreams tracking tests and unlaned riders" r25-step -p no:xdist --timeout=180`: 28 targets, 1386 passed, 6 skipped, 0 failed, 15 min 15 s wall (pytest 906.98 s), 1-min load 19.0 at start, 16.1 at end.
- **Files.** `Tests/Agents/test_durable_fleet_messages.py`, `.github/workflows/derived-artifacts.yml`, `Tests/CI/test_ci_queue_pressure_contract.py`.
