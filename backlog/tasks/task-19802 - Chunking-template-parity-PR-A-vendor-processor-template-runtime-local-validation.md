---
id: TASK-19802
title: 'Chunking template parity PR A: vendor the processor, template_runtime, local validation'
status: Done
assignee: []
created_date: '2026-08-21'
updated_date: '2026-08-21'
labels:
  - chunking
dependencies: [TASK-19801]
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PR A (read-side only; no schema, no user-visible change) of the Chunking Template Parity sub-project (ADR-078): vendor the server's `templates.py` processor at the existing engine pin, add the single chatbook seam `Chunking/template_runtime.py` (the one flat→internal mapper, the only name→template resolver, and `apply_template` which synthesizes the flat chunk contract the processor does not supply), and implement local template validation matching the server's validate endpoint check-for-check — warts included.

Spec: `Docs/superpowers/specs/2026-08-21-chunking-template-parity-design.md` (§6, §7, §12 PR-A ACs 6-15). Plan: `Docs/superpowers/plans/2026-08-21-chunking-template-parity.md` (PR A, Tasks 4-6).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 `templates.py` is vendored from the existing pin, **moved** (not appended) from the manifest's `excluded` list to its vendored list, and reproduced byte-faithfully by an idempotent sync (import-rewrite lines excepted) with no new shim module (spec AC 6)
- [x] #2 Exactly one flat→`ChunkingTemplate` mapper exists (clear error on missing `chunking`, not `KeyError`), `template_runtime.resolve_template` is the only name→template resolution, and no production module constructs the fenced `TemplateManager`/`TemplateClassifier`/`TemplateLearner` — each pinned by enumeration guards with their own self-check, and the templates directory proven absent after a boot-and-ingest run with a positive control (spec ACs 7-9)
- [x] #3 `apply_template` runs preprocessing **and** postprocessing (pinned exact output, not "demonstrably different") and synthesizes the full flat chunk contract — offsets, `chunk_index`, `total_chunks`, `word_count`, `metadata.offset_basis` — with offset keys absent rather than present-but-`None` (RAG indexing of a template-chunked item works without `TypeError`) and media navigation returning chunk-sized content (spec ACs 10-13)
- [x] #4 Local validation implements every §7 check, returns `{valid, errors, warnings}` rather than raising, resolves methods against the live engine registry, is pinned by a fixture table generated from the pinned endpoint source (upstream line ranges recorded), and §7.1's three deliberate parity warts are pinned so a later "fix" cannot silently break parity (spec ACs 14-15)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Vendor `templates.py` (manifest move, byte-faithful idempotent sync) (plan Task 4)
2. `template_runtime.py`: mapper with missing-`chunking` guard, v6/v7-stable resolver, `apply_template` with flat-contract synthesis; enumeration guards (plan Task 5)
3. Local validation matching the server endpoint (fixture table from pinned source; warts pinned); scope wiring (plan Task 6)
<!-- SECTION:PLAN:END -->

## Implementation Notes

Approach: vendored the read-side processor at the existing pin (manifest 35→36 files), added the single seam module `Chunking/template_runtime.py`, then local validation generated from the pinned endpoint source so "matches the server" stays provable and undriftable.

- Commits `d628afe6b..2182d94df` (PR-A marker `2182d94df`); SDD tasks 4-6.
- Deviations-with-rulings: spec §13.1 and `.superpowers/sdd/2026-08-21-chunking-template-parity/progress.md` — AC-10's brief pin was wrong vs the engine (sentences fill sequentially; re-measured, plan superseded); the validate endpoint's warnings-typing bug recorded as UPSTREAM_DEFECTS entry #15 (warning-never-flips ruling per spec §7); `resolve_template` deliberately queried only v6/v7-stable columns with the deleted-filter deferred to PR D (pre-flight ruling); resolver-guard legacy allowlist carried a one-entry exception that PR B shrank.
- Byte-faithfulness independently reviewer-proven (exactly one import-rewrite line; post-sync clean tree).
