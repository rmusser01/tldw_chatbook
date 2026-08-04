---
id: TASK-2240
title: 'MCP: pre-select the lone built-in row on fresh installs (R2)'
status: Done
assignee: []
created_date: '2026-08-04 16:18'
updated_date: '2026-08-04 19:37'
labels:
  - ux-review
  - mcp
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
F-054 pre-selection excludes the off built-in, so the inspector stays dead exactly on the state every new user sees; the built-in's detail (what it is, why off, Enable) is informational, not alarmist. Post-fix re-review P1. See Docs/superpowers/qa/2026-08-03-library-roleplay-mcp-ux-review.md.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A lone rail row (incl. the off built-in) is pre-selected on load,Tests updated
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no — UI selection-heuristic change on top of task-2239's display state; no storage/contract change.

1. mcp_workbench.py _preselect_single_problem_on_load(): after the existing single-problem rule, add a lone-row fallback -- when exactly one snapshot renders in the rail (fresh install = the off/opt-in built-in), pre-select it too. Keep _did_initial_preselect one-shot gate (no re-hijack) and restored-view-state precedence.
2. mcp_inspector.py update_readiness(): off/opt-in why-line explains the opt-in ('Why · Off — enable it to let MCP clients use chatbook''s tools.') instead of the alarmist 'Why · Not configured'.
3. Tests: new lone-off-builtin preselect test (workbench + inspector content informational); update zero-problem case in test_no_preselection_with_zero_or_multiple_problems (now preselects the lone row); update destination-shells inspector empty-state assertion for the lone-builtin preselect.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Lone rail row (incl. off built-in) pre-selected on first load; one-shot no-re-hijack gate kept; inspector why-line for off/opt-in now explains the opt-in. Tests updated (see task file).
<!-- SECTION:NOTES:END -->
