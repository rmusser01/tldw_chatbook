---
id: TASK-21509
title: Add persistent grounded Research Workspace Chat
status: To Do
assignee: []
created_date: '2026-08-24 05:54'
updated_date: '2026-08-24 05:54'
labels:
  - research
  - workspace
  - chat
  - rag
dependencies:
  - TASK-21507
  - TASK-21508
references:
  - Docs/superpowers/specs/2026-08-23-research-workspace-design.md
  - Docs/superpowers/plans/2026-08-23-research-workspace-grounded-chat.md
  - backlog/decisions/078-research-workspace-authority-and-screen-boundaries.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add a persistent, evidence-focused workspace conversation that can answer generally or from selected ready sources while remaining separate from Console's agent tools, approvals, and autonomous execution.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Local Chat persists a workspace-scoped canonical conversation and membership; Server Chat persists and reloads a server workspace-scoped conversation through existing chat APIs, without browser-local or parallel Research transcript ownership.
- [ ] #2 `general`, `rag`, and `auto` modes enforce the approved retrieval contract; Grounded blocks without selected ready sources, Auto reports whether retrieval occurred, and requested versus effective retrieval modes remain visible.
- [ ] #3 Local retrieval is restricted to the captured selected-source scope and records source identity/version, citations, retrieval settings, processing route, provider/model, and generation state with each answer.
- [ ] #4 Before local source bodies or excerpts are sent to a remote provider, exact processing-route consent is required and bounded by workspace, provider, endpoint class, redaction policy, and source-body mode; diagnostics remain payload-free.
- [ ] #5 Server mode performs retrieval/inference only through server-owned contracts, never downloads server sources for a client-selected provider, and never falls back to Local when the server or capability is unavailable.
- [ ] #6 Conversation/session selection, streaming stop, drafts, clear, message copy/edit/regenerate/delete/undo/branch, citation inspection, save-to-Quick-Notes, and capability-gated read-aloud behave through existing canonical owners.
- [ ] #7 Workspace or authority switching fences stale reads/streams, persists drafts before switching, and cannot retarget an in-flight request or repaint a newer context.
- [ ] #8 The Research Chat implementation imports no ToolCatalog, MCP/ACP, approval, or agent-loop owner and advertises only implemented keybindings/actions.
- [ ] #9 Targeted local/server contract, retrieval-scope, egress-consent, persistence, citation, stale-result, mounted Textual, and stop/recovery tests pass without a full-suite claim.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no new ADR

ADR path: `backlog/decisions/078-research-workspace-authority-and-screen-boundaries.md`

Reason: ADR-078 already fixes chat ownership, server-side processing, local egress disclosure, no-tools scope, and async fencing. This phase adds no new provider or persistence boundary.

Follow `Docs/superpowers/plans/2026-08-23-research-workspace-grounded-chat.md` task-by-task with test-first checkpoints and one scoped commit per completed plan task.
<!-- SECTION:PLAN:END -->
