---
id: TASK-32860
title: One recovery-admission guard for Agents, MCP, and RAG_Search
status: To Do
assignee: []
created_date: '2026-09-19 08:24'
labels:
  - core-review
  - review-cascade
dependencies: []
parent_task_id: TASK-32850
references:
  - qa/cascade-review-2026-09-19/report.md
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Three modules implement the same recovery-admission guard skeleton: `Agents/activation.py` (231 LOC), `MCP/activation.py` (242), `RAG_Search/activation.py` (433 — partially delegating to `Backup_Recovery/activation.py`). Shared verbatim-ish: the `ContextVar`, the per-family `ActivationRequired(PermissionError)`, identical `_identity()` (asyncio task + thread id), `_sources()` reading `bootstrap.effective_config_path()`/`_CONFIG_CACHE`, `execution()` scopes, and `guarded` decorators (~40% divergence is mostly sync-vs-async variants). One `RecoveryAdmissionGuard` parameterized by (exception type, sources, sync/async) deletes ~300-400 of the 906 guard LOC.

Note the name-collider trap for whoever picks this up: `Actor_Packs/activation.py` and `Tool_Packs/activation.py` are unrelated pack-install services — leave them out. ADR required: no — mechanical consolidation of one concept; admission semantics unchanged.

Source: cascade review 2026-09-19 — `qa/cascade-review-2026-09-19/report.md`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 One parameterized guard owns the skeleton; the three modules delegate with their per-family exception types and sources
- [ ] #2 The `Backup_Recovery/activation.py` relationship is reconciled (RAG_Search's partial delegation today) — one delegation story, not two
- [ ] #3 Admission behavior pinned by tests before the refactor: failure modes, identity derivation, config-cache reads
- [ ] #4 Actor_Packs/Tool_Packs activation modules untouched
<!-- AC:END -->
