---
id: TASK-32862
title: One datetime codec for the MCP and runtime_policy JSON stores
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
Six copies of `_datetime_to_iso` exist: `runtime_policy/source_state.py:133`, `MCP/unified_context_store.py:78`, `MCP/server_target_store.py:372`, `MCP/local_store.py:140`, `MCP/unified_control_models.py:15`, and `UX_Interop/server_parity_contracts.py:565` — the last is a weaker variant that does NOT normalize naive datetimes to UTC, a latent inconsistency (a naive-local datetime would silently emit a naive-local ISO). Plus three `_iso_to_datetime` copies and four now-iso variants (~110 LOC total). One codec module (tz-normalizing both directions) adopted across all sites. While there: `MCP/permission_store._save_locked` (:942-967) hand-rolls a tmp+fsync+`os.replace` path duplicating `mcp_source_participants.write_json` semantics (~26 LOC) — fold it in or document the divergence as spec-pinned. Do NOT attempt a JsonStore base class for these stores: the machinery is already centralized and the recovery framework's identity checks resist it (see the cascade report's retired list). ADR required: no.

Source: cascade review 2026-09-19 — `qa/cascade-review-2026-09-19/report.md`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 One tz-normalizing datetime codec is adopted by all six `_datetime_to_iso` sites and the `_iso_to_datetime`/now-iso variants
- [ ] #2 The `UX_Interop` naive-tz gap is closed (codec normalizes) or the caller is pinned to aware-only with a test
- [ ] #3 `permission_store`'s write path either delegates to the shared one or carries a documented spec-pinned reason
- [ ] #4 Store round-trip tests cover naive and aware inputs at the adopted sites
<!-- AC:END -->
