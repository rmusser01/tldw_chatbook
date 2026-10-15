---
id: TASK-34906
title: Declarative migration registry for ChaChaNotes
status: In Progress
created_date: 2026-10-12 17:54
labels:
- simplification
assignee:
- '@zcode'
updated_date: 2026-10-15 19:33
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

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Baseline ChaCha migration tests 2. Census the 76 step methods - which are pure file-backed 3. Build declarative {version: filename} registry 4. Collapse pure steps + merge Media/Prompts twin registries if cheap 5. ADR-208 compliance + tests vs baseline 6. Review + PR
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
PRECISE CENSUS DONE (supersedes the task's '37 pure steps' estimate): 77 step methods total in ChaChaNotes_DB. 58 are file-backed (read DB/migrations/chachanotes_v*.sql via _migration_file_statements); within those, ~34 are STRICT-PURE (version check + file read + statement loop + version verify + error wrap, nothing else - e.g. v16..v25 run, v32_to_v33 style with inline version SELECT), ~24 are HYBRID (file + genuinely extra DML in the same transaction - e.g. v26_to_v27 seeds rag_identity_context rows, v41+ steps carry python backfills). The remaining 19: 13 driven by INLINE module-level _MIGRATE_Vn_TO_Vm_SQL string constants (v4..v15 - pre-file era) and 6 pure python logic. IMPLEMENTATION SHAPE for the next session: (1) declarative _FILE_MIGRATION_STEPS: Dict[int, str] mapping from_version -> filename for the ~34 strict-pure steps; (2) generic _run_file_migration_step(from_v, to_v, filename) owning the template (require entry version / read+split file / execute per statement / verify to_v / wrap SchemaError); (3) _initialize_schema dispatch consults the registry first then falls back to the method dict; (4) hybrids keep their methods (they can call the generic runner for their file portion); (5) ADR-208 compliance is inherent - the .sql files remain the executed source (test_migration_file_sources + Tests/Packaging pins). Baseline captured: Tests/DB -k 'migration or schema' = 98 failed/643 passed (98 = known env set; /tmp/base-34906.txt). Do NOT blanket-delete method names: the migration dict at _initialize_schema:8243+ and tests reference them; replace method entries with registry lookups incrementally.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
