---
id: TASK-34907
title: Recovery schema-as-data across the recovery.py family
status: To Do
created_date: 2026-10-12 17:54
labels:
- simplification
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

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
