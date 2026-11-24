---
id: TASK-21509
title: Add persistent grounded Research Workspace Chat
status: Done
assignee:
- '@codex'
created_date: 2026-08-24 05:54
updated_date: 2026-11-24 17:48
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
- backlog/decisions/322-research-workspace-authority-and-screen-boundaries.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add a persistent, evidence-focused workspace conversation that can answer generally or from selected ready sources while remaining separate from Console's agent tools, approvals, and autonomous execution.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Local Chat persists a workspace-scoped canonical conversation and membership; Server Chat persists and reloads a server workspace-scoped conversation through existing chat APIs, without browser-local or parallel Research transcript ownership.
- [x] #2 `general`, `rag`, and `auto` modes enforce the approved retrieval contract; Grounded blocks without selected ready sources, Auto reports whether retrieval occurred, and requested versus effective retrieval modes remain visible.
- [x] #3 Local retrieval is restricted to the captured selected-source scope and records source identity/version, citations, retrieval settings, processing route, provider/model, and generation state with each answer.
- [x] #4 Before local source bodies or excerpts are sent to a remote provider, exact processing-route consent is required and bounded by workspace, provider, endpoint class, redaction policy, and source-body mode; diagnostics remain payload-free.
- [x] #5 Server mode performs retrieval/inference only through server-owned contracts, never downloads server sources for a client-selected provider, and never falls back to Local when the server or capability is unavailable.
- [x] #6 Conversation/session selection, streaming stop, drafts, clear, message copy/edit/regenerate/delete/undo/branch, citation inspection, save-to-Quick-Notes, and capability-gated read-aloud behave through existing canonical owners.
- [x] #7 Workspace or authority switching fences stale reads/streams, persists drafts before switching, and cannot retarget an in-flight request or repaint a newer context.
- [x] #8 The Research Chat implementation imports no ToolCatalog, MCP/ACP, approval, or agent-loop owner and advertises only implemented keybindings/actions.
- [x] #9 Targeted local/server contract, retrieval-scope, egress-consent, persistence, citation, stale-result, mounted Textual, and stop/recovery tests pass without a full-suite claim.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no new ADR

ADR path: `backlog/decisions/322-research-workspace-authority-and-screen-boundaries.md`

Reason: ADR-322 already fixes chat ownership, server-side processing, local egress disclosure, no-tools scope, and async fencing. This phase adds no new provider or persistence boundary.

Follow `Docs/superpowers/plans/2026-08-23-research-workspace-grounded-chat.md` task-by-task with test-first checkpoints and one scoped commit per completed plan task.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Grounded Chat uses workspace-scoped canonical Chat messages, Local Media FTS/vector retrieval with captured source versions, and server-owned retrieval and inference. Local remote excerpt consent is tied to the exact qualified workspace and route. Device storage retains drafts and bounded Server append receipts.

Local message actions include edit, delete/undo, branching and regeneration. Server actions use versioned question edits and message deletion, with interrupted-answer recovery through the append receipt. Quick Note capture persists the exact message and workspace-source references. The speech action checks the resolved Local destination before reading.

Verified the Local journey through the mounted app with actual Media, Chat and Notes SQLite stores, an injected provider, and a fresh app restart. Verified the Server journey over loopback using tldw_server 204a13724d, its canonical routers and stores, a temporary profile, and its deterministic test provider. The selected Media answer and citation reloaded after opening a fresh client.

The 34-test mounted plan check, 274-test UI and governance check and 156-test closeout passed. The aggregate closeout emitted a file-descriptor growth warning. Ruff diagnostics matched the baseline and CSS reproduced from source. ADR-322 governs the boundaries.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
