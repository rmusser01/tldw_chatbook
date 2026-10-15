---
id: TASK-34906
title: Declarative migration registry for ChaChaNotes
status: Done
created_date: 2026-10-12 17:54
labels:
- simplification
assignee:
- '@zcode'
updated_date: 2026-10-15 21:00
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
37 of the 76 ChaChaNotes migration step methods (~1.6k LOC) are pure read-sql-file-and-execute boilerplate identical modulo version numbers and filename; Media and Prompts also carry near-verbatim twin registry loops. A migration framework is a version-to-step map plus one guarded runner.

Evidence (verified on dev df4985e): 76 _migrate_from_vX_to_vY methods span ChaChaNotes_DB.py:4285-8222 (~3.9k LOC); ~37 are pure file-backed steps. Client_Media_DB_v2.py:2167-2242 and Prompts_DB.py:1026-1093 implement the same registry-and-loop. ADR-208 (backlog/decisions/208) rejected embedded constants and Python-only migrations - a declarative registry that executes the .sql files is compatible and the migrations README already anticipates it. Recovery catalogs embed further DDL copies (recovery_core_schema.py) - coordinate with the recovery schema-as-data task.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Declarative version-to-filename map consumed by the shared runner replaces the pure file-backed step methods,ADR-208 honored: the on-disk .sql remains the executed source,Python-orchestrated steps remain callables,Media and Prompts twin registries merge,Tests/DB/test_migration_file_sources.py and packaging tests stay green
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Baseline ChaCha migration tests 2. Census the 76 step methods - which are pure file-backed 3. Build declarative {version: filename} registry 4. Collapse pure steps + merge Media/Prompts twin registries if cheap 5. ADR-208 compliance + tests vs baseline 6. Review + PR
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
REVISED after review round 1 (which caught two REAL bugs): the registry covers 34 steps, not 36 - v27_to_v28 (authority backfill) and v47_to_v48 (seed-validated policy-row seeding) were hybrids MISCLASSIFIED as strict-pure by the AST census (their .sql files carry only DDL; the Python half does data work the schema-fingerprint probe could not see). Both restored to full methods incl dispatch entries. Registry mechanics verified exactly right by the reviewer: all (filename, stamps_in_file) flags match the .sql contents one-to-one (13 file-stamps / 23 runner-stamps pre-removal now 12/22 with 27+47 out... actually 34 entries: 12 file-stamp + 22 runner-stamp); the v15->v81 full-chain probe produced byte-identical sqlite_master fingerprints base-vs-head (685 objects). Runner fidelity fixes from the review: SchemaError wrap now embeds {exc} (poison-test messages matchable again), per-step debug + db_sha256 on the completion line restored, failure logger.error added. Guard updates honest: atomicity guard accepts shims only because the runner calls _require_migration_entry_version (behavioral probes still cover two shims); file-sources test reads the registry class attribute (the runtime source). Actor-pack diagnostics pin repinned to the runner shape (4 diagnostic lines; the load-bearing path-free db_path_str rule kept). VERIFICATION: combined ChaChaNotesDB + migration + actor-pack suites 133 failures vs 132 stashed baseline - the single delta (test_pure_contracts_do_not_cross_runtime_ui_network_or_provider_boundaries) verified failing on pristine dev. Census correction recorded: 34 strict-pure / 28 hybrid (was 36/26).
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
