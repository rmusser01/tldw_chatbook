---
id: TASK-553.15
title: Add provisional citation checking and one visible repair transition
status: In Progress
assignee: []
created_date: '2026-07-26 23:03'
updated_date: '2026-07-26 23:04'
labels:
  - rag
  - citations
  - provenance
  - console
dependencies: []
references:
  - Docs/superpowers/specs/2026-07-26-local-citation-repair-transition-design.md
  - Docs/superpowers/specs/2026-07-23-rag-citation-provenance-design.md
  - backlog/decisions/024-rag-citation-provenance-and-source-resolution.md
parent_task_id: TASK-553
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep local RAG answers visibly provisional until citation markers are structurally checked, and make one bounded repair attempt without changing claims or overstating grounded trust.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A local RAG answer remains one provisional assistant message until structural citation checking and any repair select the visible body
- [ ] #2 Valid markers complete without repair while missing or invalid markers trigger at most one direct tool-free repair using the same resolved provider and model
- [ ] #3 A repaired body is selected only when its non-marker text is unchanged and its markers validate; otherwise the original body is selected with honest failure or cancellation copy
- [ ] #4 Successful repair visibly replaces the same message and offers a keyboard-accessible current-session original-attempt preview without mutating message content persistence or provider history
- [ ] #5 Citation checking remains available independently of canonical-write readiness and all repair prompts, outputs, buffers, and diagnostics are bounded and privacy-safe
- [ ] #6 Direct-provider and agent-generated local answers preserve existing stop and session-close compatibility and pass scoped regression coverage
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Publish and independently review the approved design contract.
2. Define pure bounded repair contracts, structural validation, and unchanged-claim selection rules.
3. Carry repair eligibility through local capture and wire one cancellable controller-owned repair session across direct and agent replies.
4. Add safe transient Console presentation and current-session original-attempt reveal without changing durable message content.
5. Add focused contract, controller, store, UI, privacy, and compatibility tests and record scoped verification.

ADR required: no
ADR path: backlog/decisions/024-rag-citation-provenance-and-source-resolution.md
Reason: This directly implements ADR-024 streaming and repair behavior and introduces no new architecture decision.
<!-- SECTION:PLAN:END -->
