---
id: TASK-553.14
title: Capture answer attempts and seal terminal local citation traces
status: Done
assignee: []
created_date: '2026-07-26 18:18'
updated_date: '2026-07-26 20:56'
labels:
  - rag
  - citations
  - provenance
  - local-pipeline
dependencies:
  - TASK-553.13
references:
  - Docs/superpowers/specs/2026-07-23-rag-citation-provenance-design.md
  - >-
    Docs/superpowers/specs/2026-07-26-local-answer-attempt-terminal-sealing-design.md
  - backlog/decisions/024-rag-citation-provenance-and-source-resolution.md
  - TASK-553.13
parent_task_id: TASK-553
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Complete one eligible marker-free local RAG generation by binding its exact final assistant body to a governed answer attempt, sealing the request-scoped citation builder, and atomically persisting the message and canonical trace so retrieval provenance survives restart without overstating citation trust.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The local builder records a bounded governed initial answer attempt whose exact body and secret-scoped integrity fingerprint never enter immutable trace JSON or logs.
- [x] #2 Sealing requires closed, chronologically ordered local retrieval and produces a one-shot validated SealedCitationWrite with repository-owned policy metadata, selected-attempt linkage, and deterministic completeness.
- [x] #3 Eligible marker-free initial Console direct-provider and agent generations use the same repository for capture and persistence, seal from the exact materialized visible body, and atomically persist the message plus trace under stable idempotent identities.
- [x] #4 Disabled, marker-mapping-ineligible, or deterministically unavailable canonical persistence preserves the ordinary answer as ungrounded, while ambiguous transaction failure receives at most one same-identity retry and never leaves partial provenance.
- [x] #5 Failed, stopped, canceled, empty, retry, and regenerate paths do not seal or inherit unfinished builders.
- [x] #6 Focused tests cover builder atomicity, production repository wiring, persistence-capability gating, direct and agent completion, exact-body fidelity, transient-finalizer cleanup, atomic persistence, fallback, idempotent retry, and content-free diagnostics.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/024-rag-citation-provenance-and-source-resolution.md
Reason: Direct implementation of ADR-024’s accepted request-scoped builder, terminal seal, governed answer body, message ownership, and atomic persistence contracts; no new architectural decision.

Detailed plan: Docs/superpowers/plans/2026-07-26-local-answer-attempt-terminal-sealing.md

1. Add repository-owned local seal policy, bounded initial answer attempts, closed-run chronology, and one-shot builder sealing.
2. Return the exact prompt-evidence-set identity from every successful local capture path.
3. Expose fail-closed canonical-write readiness and wire the app’s exact citation repository into Console persistence.
4. Add transient terminal finalization, early-write deferral, stable identity, deterministic fallback, and one ambiguous same-write retry to ConsoleChatStore.
5. Install finalizers only for initial direct-provider and agent sends; clear them on every non-success, empty, retry, regenerate, replacement, and outer-exit path.
6. Prove exact-body atomic persistence and rollback with real SQLite integration tests, then run only the touched-code verification listed in the detailed plan.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Implemented ADR-024's request-scoped local answer-attempt lifecycle: repository-owned seal policy, governed body binding, closed-run chronology, one-shot sealing, and deterministic completeness.
- Carried the exact prompt-evidence-set identity through local capture, gated persistence on the same ready repository/database binding, and wired production Console persistence to that repository.
- Added terminal deferral/finalization with native stable message IDs, exact materialized-body sealing, deterministic ordinary fallback, one bounded ambiguous retry, and cleanup across non-success, retry, and regenerate paths for direct and agent generations.
- Added real CharactersRAGDB controller integration proving atomic message/trace graph ownership, marker-free empty occurrences, governed-body privacy, restart discovery, and deterministic owner-stage rollback with no partial provenance.
- Plan deviation/privacy fix: the real RED test exposed message bodies in CharactersRAGDB SQL DEBUG parameter previews. Added an explicit redact_params keyword to execute_query and enabled it only for add_message's sensitive INSERT; default SQL preview and execution/error semantics remain unchanged. Added focused SQL-log regression coverage.
- ADR required: yes. ADR path: backlog/decisions/024-rag-citation-provenance-and-source-resolution.md. Reason: Direct implementation of the accepted terminal seal and atomic ownership contract; no new decision.
- Scoped verification: SQL-log RED 1 failed/13 deselected, then GREEN 1 passed/13 deselected; complete SQL-log file 14 passed; real integration RED 2 failed/35 deselected solely on privacy, then GREEN 2 passed/35 deselected; final plan gate 254 passed/506 deselected with one existing dependency warning. Ruff lint passed on touched Python files. All task-introduced test formatter drift was corrected: Tests/Chat/test_citation_trace_builder.py, Tests/Chat/test_citation_trace_repository.py, Tests/Chat/test_console_terminal_citation_persistence.py, and Tests/DB/test_sql_debug_logging.py pass Ruff format check. Whole-file Ruff format for ChaChaNotes_DB.py retains identical pre-existing baseline drift before and after this task, while the changed execute_query and add_message hunks are formatter-stable. git diff --check passed.
- Task-6 files: Tests/Chat/test_console_terminal_citation_persistence.py, Tests/DB/test_sql_debug_logging.py, tldw_chatbook/DB/ChaChaNotes_DB.py, and this Backlog task file.
<!-- SECTION:NOTES:END -->
