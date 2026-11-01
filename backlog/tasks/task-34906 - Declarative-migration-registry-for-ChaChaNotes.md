---
id: TASK-34906
title: Declarative migration registry for ChaChaNotes
status: To Do
created_date: 2026-10-12 17:54
labels:
- simplification
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
37 of the 76 ChaChaNotes migration step methods (~1.6k LOC) are pure read-sql-file-and-execute boilerplate identical modulo version numbers and filename; Media and Prompts also carry near-verbatim twin registry loops. A migration framework is a version-to-step map plus one guarded runner.

Evidence (verified on dev df4985e): 76 _migrate_from_vX_to_vY methods span ChaChaNotes_DB.py:4285-8222 (~3.9k LOC); ~37 are pure file-backed steps. Client_Media_DB_v2.py:2167-2242 and Prompts_DB.py:1026-1093 implement the same registry-and-loop. ADR-208 (backlog/decisions/208) rejected embedded constants and Python-only migrations - a declarative registry that executes the .sql files is compatible and the migrations README already anticipates it. Recovery catalogs embed further DDL copies (recovery_core_schema.py) - coordinate with the recovery schema-as-data task.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Declarative version-to-filename map consumed by the shared runner replaces the pure file-backed step methods,ADR-208 honored: the on-disk .sql remains the executed source,Python-orchestrated steps remain callables,Media and Prompts twin registries merge,Tests/DB/test_migration_file_sources.py and packaging tests stay green
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
