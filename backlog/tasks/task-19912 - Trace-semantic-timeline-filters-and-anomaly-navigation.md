---
id: TASK-19912
title: Trace semantic timeline filters and anomaly navigation
status: Done
assignee: []
created_date: '2026-08-22 18:30'
updated_date: '2026-08-23 06:46'
labels: []
dependencies:
  - TASK-19910
  - TASK-19911
references:
  - >-
    Docs/superpowers/specs/2026-08-22-task-19907-trace-v2-exhaustive-collaboration-design.md
  - Docs/superpowers/plans/2026-08-22-task-19911-19912-trace-v2-interface.md
  - >-
    backlog/decisions/080-trace-v2-exhaustive-event-projection-and-collaboration.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Turn the Trace timeline and retrieval controls into an accessible causal debugging instrument for novice and power users.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Timeline lanes are semantically labeled and use non-color event differentiation with turn and agent boundaries
- [x] #2 Every timeline mouse operation has a documented keyboard equivalent
- [x] #3 Users can filter by event kind, status, agent, provider, and time with visible counts and active-filter state
- [x] #4 Users can navigate next and previous match, error, tool, feedback, and child-agent event entirely by keyboard
- [x] #5 Search, filters, timeline selection, ledger selection, and live refresh remain synchronized
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add RED pure-model and Pilot/compositor tests for semantic Input/Model/Tools/Agents lanes, non-color glyph differentiation, turn/agent boundaries, stable event identity, and no-timing collapse.\n2. Add RED keyboard-parity tests for timeline selection, previous/next event, range start/end/clear, zoom, pan, and contextual focus hints matching every mouse operation.\n3. Add a single TraceFilterBar owner for kind/status/agent/provider/time filters, visible/total counts, active state, and narrow/wide presentations without duplicating filter state in the screen.\n4. Add RED navigation tests for previous/next match, error, tool, feedback, and child-agent event, with shared stable event_id selection across search, structured filters, timeline, ledger, pagination, and live insertion.\n5. Implement the smallest pure semantic lane/filter/navigation changes while preserving the responsive TASK-19911 layout, read-only import, privacy, follow, retry, and generation guards.\n6. Verify with the focused timeline/filter/responsive suites, one batched production-CSS compositor pass at the required sizes plus at most one correction pass, the Impeccable detector once after UI changes finish, Ruff/format/diff, and independent spec/quality reviews.\n7. Resolve review findings with focused RED/GREEN evidence and close the task only after final Ready verdicts.\n\nADR required: no.\nADR path: backlog/decisions/080-trace-v2-exhaustive-event-projection-and-collaboration.md\nReason: ADR-080 already approves semantic lanes, structured Trace filters, stable selection, and keyboard parity; this task changes presentation/navigation only and introduces no new storage or service boundary.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented semantic Input/Model/Tools/Agents timeline lanes with monochrome event-family glyphs, explicit turn/agent boundaries, stable event identity, truthful no-timing collapse, and viewport-correct paint/hit behavior.

Added one responsive TraceFilterBar owner for kind, status, agent, provider, and time filters. Wide terminals use contained native selectors with explicit Shown/Matches/Total counts; compact terminals use an actionable summary and fully contained keyboard dialog.

Added bidirectional keyboard navigation for matches, errors, tools, feedback, and child agents. Timeline mouse and keyboard selection now use one atomic screen-owned transaction, reveal paginated/collapsed targets, preserve filters and live-refresh state, and never diverge from the ledger selection.

Updated Docs/Features/Trajectory-View.md and added pure-model, Pilot, production-CSS compositor, responsive, pagination, overlap, live-option, and synchronization regressions. Final verification on latest origin/dev: 182 Trace screen/live/import tests passed; Ruff, format, and diff checks passed; 60/80/100/120-column compositor checks passed; Impeccable detector returned no findings; independent specification and Sr HCI/quality reviews returned Ready Yes.

ADR required: no. ADR-080 already governs the semantic lanes, structured filters, stable selection, and keyboard-parity presentation boundary. Fixed kind colors remain a documented pre-existing secondary treatment; glyphs carry the primary non-color semantics.
<!-- SECTION:NOTES:END -->
