---
id: TASK-19803
title: 'Chunking template parity PR B: schema v7 rebuild shipped atomically with the CRUD rewrite'
status: Done
assignee: []
created_date: '2026-08-21'
updated_date: '2026-08-21'
labels:
  - chunking
dependencies: [TASK-19802]
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PR B (storage) of the Chunking Template Parity sub-project (ADR-078): migrate the Media DB v6 → v7 as an ADR-030 single-transaction table rebuild of `ChunkingTemplates` (uuid, tags, is_builtin, version, deleted, partial unique index on live names; update-timestamp trigger recreated), with row conversion, quarantine, and seeding of the six server built-ins — shipping **atomically in the same PR** with the rewrite of its only CRUD layer (`ChunkingInteropService`, normalizers), because a schema and its sole reader cannot ship apart (spec §5.2.1).

Spec: `Docs/superpowers/specs/2026-08-21-chunking-template-parity-design.md` (§5, §12 PR-B ACs 16-29). Plan: `Docs/superpowers/plans/2026-08-21-chunking-template-parity.md` (PR B, Tasks 7-8).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 v6→v7 executes as one ADR-030 transaction, statement-by-statement: new columns + partial unique index land, the update-timestamp trigger survives the rebuild, a seeded mid-rebuild failure leaves the DB at v6 with the original table and rows intact, and the historical v6 fixture is produced by bootstrapping at a patched `_CURRENT_SCHEMA_VERSION` (spec ACs 16-19)
- [x] #2 Conversion is honest: `is_system` rows dropped and re-seeded, every other row converted, `general`/`conversational`/`contextual` survive as non-builtin rows, unconvertible rows quarantined (soft-deleted, renamed `<name> (needs review)`, body preserved — never silently re-pointed), dropped operations recorded in `metadata._dropped_operations`; the six built-ins seed and all six execute; a fresh install lands at v7 with none of the five old seeds (spec ACs 20-23, 29)
- [x] #3 Soft delete works end to end (row leaves listings, name re-usable, `version` increments on update); a stored-invalid template is still editable with update validating the new body only — its listed-with-flag and refused-at-apply halves landed with the PR-D surfaces per the Task 8 review condition (spec ACs 24-25)
- [x] #4 CRUD rewrite complete: no `is_system` anywhere in the tree, every read filters `deleted = 0`, every write supplies `uuid` via `transaction()`, `uuid`/`version` sourced from the DB, `ChunkingTemplates` registered in `sql_validation`, and the `test_media_db_schema_v6` version pin updated with subset-delta migration assertions (spec ACs 26-28)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Migration: v7 rebuild (DDL, conversion, quarantine, six built-ins, trigger) in one transactional script (plan Task 7)
2. Same-PR CRUD rewrite for v7 + normalizers + sql_validation + latent `_enforce_policy` fix (plan Task 8)
<!-- SECTION:PLAN:END -->

## Implementation Notes

Approach: schema and its only reader shipped as one PR — the v7 rebuild (conversion, quarantine, pre-proven six server built-ins) plus the `ChunkingInteropService`/normalizer rewrite with validate-on-write and DB-sourced identity.

- Commits `2182d94df..428501457` (PR-B marker `428501457`); SDD tasks 7-8.
- Deviations-with-rulings: spec §13.1 and `.superpowers/sdd/2026-08-21-chunking-template-parity/progress.md` — §5.3 `is_builtin` precedence conflict resolved by migration-side demotion of exactly the 3 seeds; version bump via explicit `version = version + 1` (the trigger only refreshes `updated_at` — the brief's premise was wrong); AC-24 flag/refusal halves deferred to PR D per the reviewer's condition; the deferred `_enforce_policy` 3-arg latent bug killed en route.
- v7 number swept at implementation (216 refs, sole claimant); **merge-time re-sweep owed by the integrator** (final-review I-2).
- Suites: RAG_Admin+DB 1109p (6 pre-existing env failures proven on parent); Chunking-minus-sync 482p.
