---
id: TASK-2272
title: MCP Hub live-check residuals — unreadable boolean checkbox in typed forms; audit list does not refresh after a run
status: Done
assignee: []
created_date: '2026-08-04 21:30'
labels:
  - mcp
  - ui
  - ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Two residual defects from the PR-5 live check (2026-08-04, `feat/rag-v2-mcp-guardrails` @ a953e4c1e), neither blocking the PR-5 merge:

1. **Boolean fields in the Test Tool typed form are unreadable.** `search_rag`'s `use_semantic` checkbox renders as an empty ~2-row box — no toggle glyph, no label, state impossible to read. The boolean→Checkbox path in `mcp_schema_form.py` predates PR-5, but PR-5's typed forms for built-in tools made it visible for the first time (previously only remote-server schemas could reach it). The task-2271 live check (2026-08-04) added: mouse CLICKS do not toggle it either — only Tab+Space works — so it is both unreadable and un-clickable. Likely a sizing/label issue in how the Checkbox is composed or styled (compare the working Checkbox usages elsewhere in the app for the idiom). This is a RAG-48 surface — the typed form is the deliverable, and one of its field types ships unreadable.

2. **Audit mode's list does not refresh after a tool run.** A just-completed Test Tool run's entry appears only after a manual refresh (`r`). Either auto-refresh on run completion (the workbench knows when `_run_tool_test` finishes) or make the refresh affordance discoverable in the mode's UI copy.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

- [x] Boolean schema fields render a labeled checkbox whose state is readable, live-verified in the running app (not only in tests — this bug class passed green tests while invisible).
- [x] The CSS class-coverage contract (`Tests/UI/test_css_class_coverage_contract.py`) still passes; any new classes are styled.
- [x] A tool run's audit entry is visible in Audit mode without a manual refresh, or the refresh affordance is visible in the UI.
- [x] Additive tests cover both behaviors.

## Implementation Notes

Both residuals closed by PR-T3 (`feat/rag-truth-mcp-honesty`), the MCP tool-honesty
follow-on to PR-5/PR-T2:

- **Item 1 (unreadable/un-clickable boolean checkbox)** — closed by PR-T3 Task 6,
  commit `ebe141772` ("fix(mcp): boolean fields in the typed form are readable and
  clickable"). Root cause measured under the production CSS bundle: an app-wide
  unscoped `Checkbox { width: 100%; height: 2; }` rule (`_conversations.tcss`) left
  the widget's CONTENT height at literally 0 once its own border consumed both rows,
  so the glyph and label had nowhere to paint. `MCPSchemaForm Checkbox` now gets its
  own `height: auto` escape (the idiom three other screens already carry), plus
  task-1624's colour-only glyph idiom (invisible against the panel when off,
  `$success` when on — there's no On/Off word beside it here, so the glyph IS the
  state carrier). CSS source-edited and the bundle regenerated via
  `python3 -m tldw_chatbook.css.build_css` (never hand-edited);
  `Tests/UI/test_css_class_coverage_contract.py` green. The AC's own "live-verified
  in the running app" bar is folded into PR-T3 Task 9's whole-branch live-check pass
  (scenario 6: "the `use_semantic` checkbox is readable and clickable"), scheduled
  after this closure — the code fix and its additive tests (mounted with the real
  stylesheet, per the commit's own note that a bare test harness never sees the rule
  that broke this) are shipped and green now.
- **Item 2 (audit list not refreshing after a run)** — closed by PR-T3 Task 5,
  commit `92d4d141a` ("fix(mcp): a completed run lands in the audit trail
  immediately"). `_sync_audit_mode()` had exactly one caller (the tail of
  `_sync_children()`); `_run_tool_test()`'s `finally` now also calls a new,
  split-out `_sync_audit_log_entries()` (execution-log read +
  `MCPAuditMode.update_entries()`) after every run — success, blocked, or failed —
  without re-fetching the separate, source-scoped Findings sub-view a Test Tool run
  can never affect.
- Both commits ship additive, TDD red→green tests in `Tests/UI/test_mcp_schema_form.py`
  and `Tests/UI/test_mcp_workbench.py` respectively; no existing test was weakened.
