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
1. Add session-only structured activity presentation and classify success/blocked/failed from direct controller verdicts and ERROR:-wrapped provider results.
2. Derive privacy-safe intermediate Thinking markers with identical live/resume ordering for every primary step shape that proves tool work.
3. Add pure contiguous-message Assistant-turn grouping and visual selection order.
4. Build focused collapsed activity-disclosure and Assistant-turn widgets.
5. Integrate composite turns while preserving container/answer identity as the activity stack changes.
6. Make navigation, windowing, pruning, and plain export operate on whole rendered turns.
7. Add source/bundled TCSS and verify supported wide/narrow layouts.
8. Run focused, integration, baseline-aware lint/format, full-suite, live Console, self-review, and Backlog completion checks.

ADR required: no
ADR path: N/A
Reason: presentation-only change preserving storage, runtime/provider, and run-log contracts; ADR-031 applies.
<!-- SECTION:PLAN:END -->
