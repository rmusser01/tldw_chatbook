---
id: TASK-34907
title: Recovery schema-as-data across the recovery.py family
status: To Do
created_date: 2026-10-12 17:54
labels:
- simplification
assignee:
- '@zcode'
updated_date: 2026-10-16 00:32
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
21 recovery.py files (4.5k LOC) embed each domain full DDL as manual snapshots duplicating the runtime schema in the local-store module plus copy-pasted SQLite adapter validate/capture bodies. Schema declared once in a leaf module removes the copy and its drift risk.

Evidence (verified on dev df4985e): Examples: Kanban_Interop/local_kanban_db.py DDL vs Kanban_Interop/recovery.py:11-47 same 30 statements; Writing and Notifications likewise with comments admitting manual capture from installed constructors. Adapter bodies near-identical (Notifications/recovery.py:67-101 vs Kanban_Interop/recovery.py:50-89). Blockers: Tests/Backup_Recovery/test_backup_owner_inventory.py and test_inventory_unused_owners.py pin inventories; ADR-030/021 govern recovery semantics.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 DDL declared once per domain in a leaf module consumed by both the runtime store initializer and the recovery validator,Shared SQLite adapter factory replaces the copy-pasted adapter bodies,Recovery modules still never import runtime engines (documented policy),Backup owner inventory tests green
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Baseline Backup_Recovery suites 2. Map the 21 recovery.py DDL duplicates + adapter bodies 3. Extract schema-as-data leaf modules 4. Migrate consumers 5. Tests vs baseline 6. Review + PR
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
PRECISE CENSUS DONE (supersedes the task's '21 recovery.py files' premise): of 22 */recovery.py files only 9 embed SQLite DDL; 3 more domains live in DB/recovery_operations.py (1,616 LOC 344KB); 4 domains ALREADY single-sourced via leaf modules (guardian_schema.py 36 LOC / recovery_core_schema.py 1008 / recovery_goal_schema.py 148 / recovery_response_rules + shim) - these are the exact target pattern: runtime AND recovery import a leaf that itself imports neither. KEY MECHANICAL FACT: recovery DDL is the SQLite STORED CATALOG (SELECT sql ORDER BY type,name with IF NOT EXISTS stripped + fts5 shadow tables + sqlite_sequence) NOT the runtime script text - any single-source design must model that transformation. The drift detectors (test_operational_owners.py::test_operational_schema_matches_actual_constructor over 9 stores + test_domain_owners.py::test_real_domain_complete_capture_and_current_schema) are GREEN in the 3h14m baseline (385F/5787P known env; /tmp/base-34907.txt) - they are the executable correctness definition. FEASIBILITY TABLE: EASY (kanban notifications.client runtime.event_state runtime.sync_state notes.file_notes) = single clean runtime owner + stored-catalog relation; MEDIUM (scheduled_tasks ALTER-accumulated; tts spread across migrations pkg; research/writing hand-rolled adapters should also rebase onto _SQLiteDeclaration); HARD (evals computed v7 catalog + FTS triggers; agent_runs programmatic delta catalogs; subscriptions ~8 byte-exact alternative catalog families coupled to chachanotes). SEPARABLE SUBTASK: the 6 _SQLiteDeclaration adapter bodies are copy-paste identical modulo (label, version_query, one TypeError) - an adapter dedup is independent of the DDL dedup. PRIVATE-SYMBOL BLAST RADIUS: _SCHEMA pinned by Tests (TTS + Evals), 12+ private adapter classes imported cross-module by Backup_Recovery destinations/restore_plan/later_rollback/staging - renames have a concrete list in the extraction map (recorded in the agent transcript). Implementation: start with the EASY five following guardian_schema.py precedent; verify with the two drift detectors + domain guard files only (fast), NOT the 3h full suite.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
