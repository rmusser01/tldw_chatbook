---
id: TASK-2272
title: MCP Hub live-check residuals — unreadable boolean checkbox in typed forms; audit list does not refresh after a run
status: To Do
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

1. **Boolean fields in the Test Tool typed form are unreadable.** `search_rag`'s `use_semantic` checkbox renders as an empty ~2-row box — no toggle glyph, no label, state impossible to read. The boolean→Checkbox path in `mcp_schema_form.py` predates PR-5, but PR-5's typed forms for built-in tools made it visible for the first time (previously only remote-server schemas could reach it). Likely a sizing/label issue in how the Checkbox is composed or styled (compare the working Checkbox usages elsewhere in the app for the idiom). This is a RAG-48 surface — the typed form is the deliverable, and one of its field types ships unreadable.

2. **Audit mode's list does not refresh after a tool run.** A just-completed Test Tool run's entry appears only after a manual refresh (`r`). Either auto-refresh on run completion (the workbench knows when `_run_tool_test` finishes) or make the refresh affordance discoverable in the mode's UI copy.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

- [ ] Boolean schema fields render a labeled checkbox whose state is readable, live-verified in the running app (not only in tests — this bug class passed green tests while invisible).
- [ ] The CSS class-coverage contract (`Tests/UI/test_css_class_coverage_contract.py`) still passes; any new classes are styled.
- [ ] A tool run's audit entry is visible in Audit mode without a manual refresh, or the refresh affordance is visible in the UI.
- [ ] Additive tests cover both behaviors.
