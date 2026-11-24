---
id: TASK-41802
title: Reuse Console transcript structure during plain streaming updates
status: In Progress
created_date: 2026-11-18 17:57
assignee:
- '@codex'
labels:
- performance
- console
references:
- https://github.com/rmusser01/tldw_chatbook/pull/3610
documentation:
- Docs/QA/console-library-responsiveness/README.md
updated_date: 2026-11-18 18:28
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep the measured active transcript update bounded to its changing assistant row.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 An eligible streaming-only update preserves the current session, ordered history, scrollback, links and selection while avoiding a whole transcript grouping pass.
- [x] #2 Structural, cross-session, thinking, tool and terminal changes still project exactly, including focus and Stop.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Trace widget and store projection identity, static stream metadata, render keys, and fallback callers.
2. Add measured real-widget controls and implement the narrowly proven text-only update at existing ownership boundaries.
3. Exercise resync, selection, scrollback, Stop, and streaming latency on the affected paths.
ADR required: no
ADR path: backlog/decisions/090-console-thinking-block-ownership-and-replay.md and backlog/decisions/231-console-polling-and-full-state-reconciliation.md
Reason: Existing poll routing, thinking and effect authority are preserved. Reuse is bounded to presentation state.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Plain same-session streaming keeps causal spans and the current window after comparing ordered IDs and roles. The ordinary path handles thinking, tools, persisted window changes and stream termination. The mounted streaming check preserves the row and focus, and 71 transcript, thinking and Stop cases passed. The link test now waits for its browser worker on both accepted and rejected inputs. The 10,000-message ingest median improved from 18.17 to 1.77 ms CPU. The existing ADR-090 and ADR-231 contracts apply. Local acceptance is verified. Hosted and full Terminal qualification are pending.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
