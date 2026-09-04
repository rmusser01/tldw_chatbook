---
id: TASK-31382
title: Make Console context status and automatic compaction truthful
status: Done
assignee: []
created_date: '2026-09-04 18:57'
updated_date: '2026-09-04 19:28'
labels:
  - console
  - ui
  - context
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep inherited compaction behavior intact in the quick Console settings flow and make the existing token/cost status control show how full the next request context is.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Applying untouched quick settings preserves an inherited compaction mode instead of creating a per-conversation override.
- [x] #2 The existing Console token/cost chip shows current safe-input context fullness alongside conversation spend at wide and narrow supported widths.
- [x] #3 The chip explains request usage, safe input capacity, conversation budget, compaction timing, and spend without presenting cumulative billed tokens as context occupancy.
- [x] #4 Targeted unit and mounted Textual tests cover sparse override behavior, known and unknown context capacity, wide and narrow labels, tooltips, keyboard activation, and geometry.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing regressions for untouched quick-popover compaction inheritance and for a combined context/spend chip derived from the existing Console context-control snapshot.
2. Preserve the original sparse compaction override when the mounted selector remains at its initial effective value; persist an explicit mode only after a deliberate change.
3. Extend the existing cost state formatter and status chip wiring with request fullness, conversation budget, and compaction timing while preserving cache alerts and Conversation Inspector activation.
4. Run focused formatter, popover, status-chip, and screen tests; inspect production-styled mounted output at 120x35 and 80x24; run scoped lint, formatting, compilation, and diff checks.
5. Update the task acceptance criteria and implementation notes with exact evidence.

ADR required: no
ADR path: backlog/decisions/052-console-conversation-memory-and-compaction-policy.md and backlog/decisions/095-conversation-owned-console-generation-settings.md
Reason: ADR-052 already defines context capacity and sparse per-conversation policy ownership, while ADR-095 requires quick Apply to preserve that sparse owner; this task is a bug fix and compact UI clarification within those accepted boundaries.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed the quick-settings persistence bug by retaining the original sparse compaction override whenever the selector remains on its initial effective value; deliberate changes still persist explicitly. Extended the existing cost chip with next-request safe-input fullness, conversation budget, effective compaction timing, and clearly separated cumulative spend, reusing the settings-summary estimate so the fast cost refresh does not add tokenization work. Preserved cache alerts and Conversation Inspector activation, including compact and full labels with production-styled geometry coverage. Added focused unit, mounted Textual, live-screen, and automatic-preflight regressions. Verification: 12 focused tests passed; scoped Ruff lint, Python compilation, and git diff checks passed. The repository has no configured formatter; an unconfigured Ruff format check would rewrite pre-existing sections of every touched file, so unrelated dirty-worktree formatting was intentionally preserved. ADR required: no; ADR-052 and ADR-095 already govern the behavior.
<!-- SECTION:NOTES:END -->
