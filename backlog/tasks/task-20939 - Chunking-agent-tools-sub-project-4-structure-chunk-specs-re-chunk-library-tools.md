---
id: TASK-20939
title: 'Chunking agent tools (sub-project #4): structure/chunk/specs/re-chunk library tools'
status: Done
assignee: []
created_date: '2026-08-22'
updated_date: '2026-08-22'
labels:
  - chunking
dependencies: [TASK-19901]
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Sub-project #4 of the chunking parity program (single PR, stacked on #3/TASK-19901's auto-selection over #2/ADR-078's v7 template store — no new ADR; the spec's §8 rulings are the long-form record): four agent tools over stored chunks — the structure map (node-paginated navigation tree annotated with chunk-unit addresses), chunk fetch (chunk-index primary addressing, family-aware, budget-bounded neighbors), spec list/save (the agent view of the v7 template store), and re-chunk (one item, one transaction, spec override via #3's resolution, opt-in reindex) — plus the two new policy actions (`library.media/rechunk`, `library.templates/save`), one backend read, a behavior-identical one-item rechunk refactor, the student-story end-to-end pin, docs/CHANGELOG, and final review.

Spec: `Docs/superpowers/specs/2026-08-22-chunking-agent-tools-design.md` (§7 ACs; §8's 15 rulings). Plan: `Docs/superpowers/plans/2026-08-22-chunking-agent-tools.md` (six tasks, one PR).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Groundwork: `rechunk_one_item` extracted from the legacy batch with behavior-identity pinned by the 17-test rechunk suite; backend read `get_library_media_chunks` added (family-aware, budget bounds NEIGHBORS only, the requested chunk always returned whole) (spec §4.2, §5)
- [x] #2 Reads: `library_get_media_structure` (node pagination that closes honestly at the 500-node window; revision tokens round-trip and are checked, §8.9) and `library_get_media_chunk` (chunk-index addressing, `chunk_type` families explicit, §8.10; byte budget wins over context count, §8.12; no-chunks degradation keeps the story alive, §8.13) (spec §4.1-§4.2)
- [x] #3 Spec tools: `library_list_chunk_specs`/`library_save_chunk_spec` over the v7 store — specs ARE templates, no second store (§8.3); validity/reserved flags carried, spec-save refusals return the validator's full error array (§8.15); the deadline carry closed (MCP local-control write-action mapping re-mapped off the derived read) (spec §4.3)
- [x] #4 Write: `library_rechunk_media` — one item, one transaction, flat spec override (template XOR plain keys; omitted `overlap` = 0), unresolvable-name named refusal, opt-in `reindex`, outcome vocabulary never a bare "done"; policy enforcer wired at both construction sites; `library.media/rechunk` + `library.templates/save` actions pinned (spec §4.4, §6, §8.4, §8.14)
- [x] #5 Story + docs: student-story end-to-end test pins §7.6 (structure → chunk fetch → spec save → re-chunk → re-read on a real chunked book); dev reference page, mcp.md standalone inventory corrected (18→23), user guide, CHANGELOG; all Task-5/6 carries closed (spec §7)
- [x] #6 Close-out: targeted suites green (close-out run 1 failed / 3296 passed / 28 skipped / 1 xfailed across Library + Media chunk reads + RuntimePolicy + Chunking; the single failure is the documented pre-existing RuntimePolicy migration-audit drift, identical on unmodified HEAD); final review READY-WITH-LISTED-FOLLOWS with both docs-only follows fixed (de-stale docstrings, §8.14 concurrency note)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Extract `rechunk_one_item` from the legacy batch; add the `get_library_media_chunks` backend read (plan Task 1-2)
2. Structure + chunk-fetch tools: node pagination, revision tokens, families, budget, degradation (plan Task 3)
3. Spec list/save tools over the v7 store; close the MCP write-action carry (plan Task 4)
4. Re-chunk tool + policy actions; enforcer at both construction sites (plan Task 5)
5. Student-story end-to-end; docs, mcp.md inventory, CHANGELOG (plan Task 6)
6. Final review + docs-only follows (this task's close-out)
<!-- SECTION:PLAN:END -->

## Implementation Notes

Approach: one PR stacked on TASK-19901 (#3) — a behavior-identical one-item rechunk refactor and one new backend read, then the four tools as descriptor-table siblings in one service (`LocalMediaChunkToolService`), with dispatch, policy, and the MCP manifest fully derived from `LIBRARY_TOOL_DESCRIPTORS` so the two runtimes cannot drift.

- Commits `c90955292..3de048979` (10: spec, plan, groundwork, reads, carry corrections, spec tools, re-chunk, lint, story/docs) plus final-review follow `6f5c9adec` (de-stale docstrings, §8.14 concurrency note, `__all__` sort); SDD tasks 1-6.
- Key rulings: specs-are-templates (the v7 store IS the spec store — no second store); chunk-index primary addressing with the structure tree as the map; the carries closed where the writes went live (save-tool MCP write-action mapping at T4; enforcer at both sites + `library.media/rechunk` at T5); span-bleed = accepted any-overlap semantics, escalated to the maintainer as a possible future contract refinement.
- Final review: READY-WITH-LISTED-FOLLOWS (docs-only) — both follows fixed in `6f5c9adec`. Long-form record: spec §8 (15 rulings) + `.superpowers/sdd/2026-08-22-chunking-agent-tools/progress.md`.
- Follow-up filed: TASK-20940 — FB2 extractor duplicates section titles and nested-section content (pre-existing, surfaced by the T6 story dry run).
