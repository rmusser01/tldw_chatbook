---
id: TASK-18921
title: 'Slash-command popup: rank suggestions by usage'
status: To Do
assignee: []
created_date: '2026-08-19 09:55'
updated_date: '2026-08-19 09:55'
labels:
  - console
  - ux
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Port of hermes-agent's "slash menu leads with your most-used skills" idea (2026-08-19 hermes-release review). The Console slash-command popup (ConsoleCommandPopup, fed by Chat/console_command_suggestions.py) orders suggestions statically. Rank suggestions within the current prefix-filtered set by observed usage — command dispatches and `$skill` invocations counted from existing local records — so frequently used commands surface first. Display-only; matching and filtering semantics unchanged.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Suggestion order within a filtered set is usage-ranked (most-used first) with deterministic tie-breaking (alphabetical) so tests are stable
- [ ] #2 Usage counts derive from existing local data (prompt history JSONL / run records); no new network calls, no telemetry, nothing leaves the machine
- [ ] #3 First-run behavior with zero usage data is identical to today's ordering
- [ ] #4 Unit + UI tests pin ordering, tie-breaking, and that prefix filtering is unchanged
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no.
ADR path: N/A.
Reason: display-order-only change over an existing local data source; no schema or boundary change.

1. Add a small usage-count source over the existing prompt history / run records
2. Feed ranked suggestions into ConsoleCommandPopup (keep the idempotent same-suggestions guard intact)
3. Tests + brief docs note in the Console chat-basics page
<!-- SECTION:PLAN:END -->
