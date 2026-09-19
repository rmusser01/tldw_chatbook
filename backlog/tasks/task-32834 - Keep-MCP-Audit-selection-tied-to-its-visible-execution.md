---
id: TASK-32834
title: Keep MCP Audit selection tied to its visible execution
status: Done
assignee:
  - '@codex'
created_date: '2026-09-19 03:45'
updated_date: '2026-09-19 03:58'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep the execution shown in MCP Audit consistent with the selected visible row while filters and newest-first log refreshes change the table.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Filtering or refreshing away a selected execution clears its detail and drill actions without selecting a replacement.
- [x] #2 Newer log entries preserve an unambiguous selected execution and exact tool drilldown; queued stale row activations cannot resolve to a different execution.
- [x] #3 Targeted regressions and private native dark/light checks verify filtering, refresh, and exact drilldown.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce selection loss or stale detail across filters and newest-first refresh.
2. Keep row activation tied to the rendered execution and clear invalid selection without implicit replacement.
3. Verify exact tool drilldown, targeted regressions and private native dark/light states; retain evidence and open a bounded draft PR.
ADR required: no
ADR path: backlog/decisions/170-table-repopulation-selection-boundary.md; backlog/decisions/150-design-token-system-and-design-language.md
Reason: repair selection continuity within the existing Audit flow without changing log storage, permissions or runtime boundaries.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Preserve the selected execution by snapshot row key during filtering and unique metadata match during refresh. Retire old row keys, carry rendered entries in selection messages, and clear absent or ambiguous selections without implicitly choosing another row. Delayed messages cannot select a replacement at an old index.
102 targeted cases and seven preflight guards pass. Three initial regressions and the independently discovered same-snapshot duplicate case failed before repair. New Python files pass Ruff; production adds no lint diagnostics. Independent review has no remaining blocker.
Private native TldwCli/LinuxDriver checks cover dark/light 80x24 and 170x48, with 18 inspected captures, real JSONL storage and two same-name local stdio catalogs. Wide Open tool reaches the exact server/tool; records are synthetic metadata and no tools/call runs. Clean exit, released lock, healthy private DBs and unchanged defaults verified.
Files: MCP Audit/workbench selection, targeted tests, retained QA and review ledgers. Compact Audit filter clipping and stale built-in inspector guidance remain explicitly open for separate bounded reviews; PR2718 owns compact inspector reachability.
ADR required: no; existing backlog/decisions/170-table-repopulation-selection-boundary.md and ADR150/161 apply. No log schema, runtime authority or visual token change.
<!-- SECTION:NOTES:END -->
