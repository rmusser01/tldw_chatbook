---
id: TASK-15020
title: 'RAG P2ab: instrument renewal + P0-deferred constraints'
status: In Progress
assignee: []
created_date: '2026-08-11 04:37'
updated_date: '2026-08-11 04:38'
labels:
  - rag
  - eval-harness
  - hybrid
  - p2
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
After the weighting arc, hybrid recall/MRR/NDCG sit at 1.000 on every scored eval query, so the harness can only detect regression, not measure improvement. This arc restores that power with fail-first fixture authoring (a candidate is admitted only when today's pipeline measurably fails it) and, inside the same branch, lands the three P0-deferred constraints the renewed harness measures: scoped searches silently dropping to semantic-only, prompts having no keyword-leg coverage at all, and the Library canvas's window ignoring the active profile's default_top_k. Programme: RAG server-port P2, first of two P2 arcs (the second is P2c measured feature admission).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 New candidate fixture categories (compositional/multi-hop, negation-sensitive, acronym-without-context, precision-pressure) are admitted only where measured to fail today's pipeline (target misses top-10 in every vector-bearing mode); a class that cannot be authored to fail is recorded as unfailable in the README rather than force-fit.
- [ ] #2 The scoped category ships with a routing before-pin: scoped golden queries are recorded as failing by routing (the P0 semantic-only-under-scope constraint) prior to B1, giving the scope-aware hybrid fix a documented before-number.
- [ ] #3 The eval corpus scales to ~150 documents while every existing fixture stays byte-identical, and a probe confirms each pre-existing golden query's top-10 is unchanged (or the new doc is reworded) after the corpus addition.
- [ ] #4 Scope-aware hybrid removes the engine's metadata_allowlist raise for hybrid search, retiring the scoped-to-semantic-only disclosure family (ROUTE_NOTE_HYBRID_SCOPED and its route-note/User Guide copy) app-wide.
- [ ] #5 A read-only prompts keyword sub-leg is added to the hybrid engine (Prompts DB FTS5), inventoried and vocabulary-pinned (source_type: prompt) per the chacha private-sqlite pattern, with prompt fixtures' before-state recorded as total absence across all modes.
- [ ] #6 The Library canvas's default (unset/invalid) top_k resolves to the active profile's default_top_k instead of the fixed literal 5, while an explicit user-set top_k value keeps winning unchanged.
- [ ] #7 One deliberate re-stamp closes the sub-arc, replacing the at-ceiling README warning with a per-category headroom table showing the new categories' honest (lower) baselines and the scoped category's post-B1 scores.
- [ ] #8 A live TUI check confirms all three user-visible changes: a scoped hybrid search returns keyword-found in-scope evidence, a prompts hit surfaces in hybrid results, and the Library evidence list remains usable at the profile's default depth (15 rows).
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Spec: Docs/superpowers/specs/2026-08-11-rag-p2ab-instrument-and-deferred-constraints-design.md. Plan: Docs/superpowers/plans/2026-08-11-rag-p2ab-instrument-and-deferred-constraints.md. Sequencing per the plan's 9 tasks: Half A (harness scope machinery -> fail-first authoring + ~150-doc scale-up) lands first; then B1 (scope-aware hybrid; TASK-14752's coverage-copy fix rides inside B1's disclosure-seam work) -> B2 (prompts sub-leg) -> B3 (Library window honors profile); ONE deliberate re-stamp closes the sub-arc with a per-category headroom table.
<!-- SECTION:PLAN:END -->
