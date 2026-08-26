---
id: TASK-18926
title: "Console composer: `!command` shell passthrough (permission-gated)"
status: To Do
assignee: []
created_date: '2026-08-19 09:55'
updated_date: '2026-08-19 09:55'
labels:
  - console
  - tools
  - ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Port of hermes-agent's `!` shell mode (2026-08-19 hermes-release review), adapted to chatbook's security model: typing `! <command>` in the Console composer runs a shell command directly — no model turn, no token spend — and shows bounded output in the transcript. Unlike hermes, execution must route through the existing local-tools architecture: the local-tools master switch and per-tool Allow/Ask/Off permissions (MCP → Tools → Local workspace tools), the `[console] workspace_root` confinement, and the same scrubbed-env execution path the fs_* tools use (governed by ADR-030 local-library-agent-tool-boundary, ADR-032 local-agent-tool-permission-boundary, ADR-033 local-agent-process-execution-boundary). Ask raises the standard approval card; Off refuses with an inline reason. Full output follows the existing spill/display-cap discipline.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A `!`-prefixed draft executes locally with zero provider calls and zero token spend (pinned by a test that asserts no LLM request is made)
- [ ] #2 Output renders as a bounded Tool-style transcript row under the tool result display cap, with the full output reachable via the run-log path
- [ ] #3 Execution is gated by the local-tools master switch and permission model: Ask raises the standard approval card; Off/disabled refuses with an inline reason; Allow runs without asking
- [ ] #4 Confinement and env handling match local tool execution: cwd confined to workspace_root, scrubbed environment, bounded runtime
- [ ] #5 Parsing rules pinned and tested: `!` interacts deterministically with slash-command parsing, collapsed paste tokens (paste tokens suppress it like slash parsing), and the prompt queue (`!` runs immediately, never queues)
- [ ] #6 Tests cover the gate matrix, confinement, output bounding, and parse interactions; user guide documents the command
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no.
ADR path: N/A.
Reason: reuses the existing local-tool permission/confinement architecture (ADRs 030/032/033 govern); this adds a composer entry point to that seam, not a new execution boundary. Link those ADRs from the final implementation notes.

1. Define the `!` parse rule in the composer draft pipeline (after paste-token check, before slash parsing)
2. Route execution through the local-tools executor with a shell tool identity in the permission store
3. Transcript row + run-log spill for bounded output
4. Permission-matrix tests, confinement tests, zero-LLM-call pin, docs
<!-- SECTION:PLAN:END -->
