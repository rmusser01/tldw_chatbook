---
id: TASK-19426
title: Group Console tool activity inside assistant turns
status: In Progress
assignee: []
created_date: '2026-08-21 15:56'
labels: []
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make Console transcripts clearly attribute reasoning and tool activity to the assistant response that produced them, reducing ambiguity and transcript clutter.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each user query is followed by one visually coherent Assistant turn container.
- [ ] #2 Tool and reasoning activity is rendered inside its owning Assistant turn.
- [ ] #3 Tool and reasoning details are collapsed by default and can be expanded independently.
- [ ] #4 The final assistant answer remains visible in the same Assistant turn after its activity rows.
- [ ] #5 Existing tool-output expansion and message actions remain usable.
- [ ] #6 Focused transcript tests and live visual verification cover completed, streaming, failed, and resumed turn shapes.
- [ ] #7 Thinking rows never expose hidden chain-of-thought; absent or unsafe summaries render without a dead disclosure control.
- [ ] #8 Keyboard selection and transcript pruning follow the rendered turn hierarchy without splitting or reversing a turn.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add session-only structured activity presentation plus the ADR-078 optional `ToolResult.outcome`/`AgentStep.tool_outcome` provenance contract. New runtime steps classify from review verdict and `ToolResult.ok` before result flattening; only legacy/malformed persisted steps fall back to direct-controller and `ERROR:`-wrapped provider-result parsing.
2. Derive privacy-safe intermediate Thinking markers with identical live/resume ordering for every primary step shape that proves tool work.
3. Add pure contiguous-message Assistant-turn grouping and visual selection order.
4. Build focused collapsed activity-disclosure and Assistant-turn widgets.
5. Integrate composite turns while preserving container/answer identity as the activity stack changes.
6. Make navigation, windowing, pruning, and plain export operate on whole rendered turns.
7. Add source/bundled TCSS and verify supported wide/narrow layouts.
8. Run focused, integration, baseline-aware lint/format, full-suite, live Console, self-review, and Backlog completion checks.

ADR required: yes
ADR path: backlog/decisions/078-structured-agent-tool-outcome-provenance.md
Reason: collision-safe status adds an optional internal provider/runtime fact that is serialized by the existing `dataclasses.asdict` -> schemaless steps-JSON path. ADR-078 records the status precedence, safe fallback for old/malformed step dictionaries, and why no SQLite or external provider-wire migration is required. Conversation marker persistence stays unchanged; ADR-031 still applies to keybinding/footer-hint truthfulness.
<!-- SECTION:PLAN:END -->
