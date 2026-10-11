---
id: TASK-32860
title: 'One recovery-admission guard for Agents, MCP, and RAG_Search'
status: Done
assignee: []
created_date: '2026-09-19 08:24'
updated_date: '2026-10-02 01:08'
labels:
  - core-review
  - review-cascade
dependencies: []
references:
  - qa/cascade-review-2026-09-19/report.md
parent_task_id: TASK-32850
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
- [x] #1 One parameterized guard owns the skeleton; the three modules delegate with their per-family exception types and sources
- [x] #2 The `Backup_Recovery/activation.py` relationship is reconciled (RAG_Search's partial delegation today) — one delegation story, not two
- [x] #3 Admission behavior pinned by tests before the refactor: failure modes, identity derivation, config-cache reads
- [x] #4 Actor_Packs/Tool_Packs activation modules untouched
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Already implemented by 73c3ce76deb4ec5fb742b9aaab0480b428c48481 and reviewed in merged PR #2785 (3e81b9b198deb4abadef26e4816bcf9483e7b2a8, 2026-09-21). Agents, MCP and RAG delegate to Backup_Recovery.admission_runtime.RecoveryAdmissionGuard, retaining their family policies. RAG uses the same runtime and shared execution identity; Actor_Packs and Tool_Packs activation files were untouched by the consolidation. That commit records 174 matching before/after passes and 54 matching prior failures; this closure does not relabel them all green. Current direct Tests/Backup_Recovery/test_admission_runtime.py coverage passed all five original cases in /private/tmp/backup-followup-targeted-baseline-4g00xhzd. Status was stale; no duplicate refactor was made.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Already implemented by 73c3ce76deb4ec5fb742b9aaab0480b428c48481 and reviewed in merged PR #2785 (3e81b9b198deb4abadef26e4816bcf9483e7b2a8, 2026-09-21). Agents, MCP and RAG delegate to Backup_Recovery.admission_runtime.RecoveryAdmissionGuard, retaining their family policies. RAG uses the same runtime and shared execution identity; Actor_Packs and Tool_Packs activation files were untouched by the consolidation. That commit records 174 matching before/after passes and 54 matching prior failures; this closure does not relabel them all green. Current direct Tests/Backup_Recovery/test_admission_runtime.py coverage passed all five original cases in /private/tmp/backup-followup-targeted-baseline-4g00xhzd. Status was stale; no duplicate refactor was made.
<!-- SECTION:FINAL_SUMMARY:END -->
