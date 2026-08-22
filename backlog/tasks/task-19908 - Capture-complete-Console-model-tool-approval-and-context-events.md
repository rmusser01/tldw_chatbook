---
id: TASK-19908
title: Capture complete Console model tool approval and context events
status: To Do
assignee: []
created_date: '2026-08-22 18:28'
labels: []
dependencies:
  - TASK-19907
references:
  - Docs/superpowers/specs/2026-08-22-task-19907-trace-v2-exhaustive-collaboration-design.md
  - Docs/superpowers/plans/2026-08-22-task-19907-19910-trace-v2-event-foundation.md
  - backlog/decisions/080-trace-v2-exhaustive-event-projection-and-collaboration.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Instrument the Console runtime so every observable user, system/context, model, streaming, tool, approval, retrieval, compaction, retry, cancellation, and failure transition is durably recorded by its existing owner and normalized by the Trace v2 projection.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every documented Console event family emits start, outcome, and failure or cancellation records where applicable
- [ ] #2 Tool approval decisions, provider retries/errors, RAG/context injection, and compaction ancestry are captured with causal links
- [ ] #3 Sensitive payload fields are classified at capture time and hidden reasoning content is never persisted
- [ ] #4 Trace capture failures never fail the user run and are surfaced through diagnostics
- [ ] #5 Real-seam integration tests prove ordered capture across a representative Console run
<!-- AC:END -->
