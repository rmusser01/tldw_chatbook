---
id: TASK-405
title: >-
  RAG scope narrowing Phase 1: core model, store filtering, retrieval
  enforcement
status: Done
assignee: []
created_date: '2026-07-21 09:48'
updated_date: '2026-07-21 09:49'
labels:
  - rag
  - scope
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Phase 1 of the RAG scope narrowing program (spec: Docs/superpowers/specs/2026-07-21-rag-scope-narrowing-design.md, ADR-005 lineage; plan: Docs/superpowers/plans/2026-07-21-rag-scope-narrowing.md). Delivers the conversation-level hard retrieval filter end-to-end without UI: scope model+codecs+resolver (Chat/rag_scope.py), true store-level allowlist filtering on both vector stores (fixing the verified filter_metadata post-filter starvation), pipeline-leg self-enforcement incl. conversations/prompts exclusion, conversation storage + chat-path wiring + EMPTY short-circuit, and Backend B caller-passed scope for Console Run Library RAG with the D2 guard. Phases 2 (picker modal/Inspector row/chip) and 3 (workspace scope) follow.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Scoped conversations retrieve only in-scope content across semantic, FTS and hybrid paths (both backends)
- [x] #2 Unscoped behavior is byte-identical to pre-branch behavior
- [x] #3 EMPTY scope short-circuits with diagnosed cause; no legs run
- [x] #4 Library screen surfaces remain unscoped (D2)
- [x] #5 All suites green at baselines with ~100 new tests
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Executed via subagent-driven development per Docs/superpowers/plans/2026-07-21-rag-scope-narrowing.md Tasks 1-7: model/codecs, resolver+cache, store-level filtering, leg self-enforcement, storage+chat wiring+EMPTY, Backend B, close-out.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Phase 1 complete on branch claude/rag-scope-spec (17 commits incl. spec+plan). Every task passed independent spec+quality review (4 fix waves, all re-verified; final whole-branch review MERGEABLE with consolidated fix wave landed). Key deviations, all adjudicated and documented in the PR: per-type semantic allowlists replace the plan's flat dict (cross-type id-collision); ScopeCache deferred to Phase 3 (ws stamps make it meaningful); parse_scope(None) warning-free (absence is normal). Follow-ups filed: 406 native-send RAG injection, 407 media dedup collapse, 408 cache-singleton test isolation, 409 retrieve-step latent TypeError. ~100 new tests; suites at baselines.
<!-- SECTION:NOTES:END -->
