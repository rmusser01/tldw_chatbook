---
id: TASK-34414
title: Library sync delete restore test waits for its exact folder
status: Done
assignee:
  - '@codex'
created_date: '2026-10-06 15:27'
updated_date: '2026-10-06 15:40'
labels:
  - testing
  - library
  - ci
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_chatbook/pull/3029'
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make the real Library sync Delete and Undo regression reliable when root folders arrive after the Unfiled row, without changing production Notes behavior or weakening its assertions.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A deterministic delayed root-folder load exposes the prior generic-row settlement failure while preserving real SQLite and filesystem operations.
- [x] #2 The existing regression waits for the current exact VSync row within its unchanged 30 second budget and completes all Delete, Undo, byte and status assertions.
- [x] #3 Targeted verification, static checks and retained RED/GREEN evidence pass without warning suppression, retries or production Library changes.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Retain failed current-head CI and the unchanged isolated passing control.
2. Gate only the real root-folder service response and seed a real Unfiled control; prove that generic-row readiness releases the test before VSync exists.
3. Replace the test navigation readiness with the existing bounded predicate wait for the exact folder identity, then re-query the live row.
4. Run both normal and delayed real Delete/Undo scenarios, static/artifact guards and independent review; retain RED/GREEN and publish only this test repair to PR3029.
ADR required: no
ADR path: N/A
Reason: test-only settlement correction; production Notes, sync, resource ownership and time budgets remain unchanged.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented test-only exact folder readiness correction for PR3029. The helper retains the installed 30s predicate wait and presses freshly queried folder-1 without yielding. Controlled real Unfiled/root-page handoff reproduced original StopIteration: valid RED1 failed5.75s; normal/delayed GREEN2 passed17.66s with no warnings. Initial concurrent control2 passes retained as non-RED. Real SQLite/vault bytes/all Delete/Undo/status assertions and ownership unchanged. Changed test Ruff/format and whitespace clean; independent read-only review no findings. Ten artifact guards pass, missing-cache Mermaid invocation retained then normal pinned-input rerun passes; final task-ID/readability guards pass. QA receipt Docs/QA/task-34414/README.md and exact-row lesson updated. ADR required:no, N/A test-only; no production behavior/threshold change. Only our unpublished colliding ID34413 renumbered after sweep. New-head CI/review/base and resource/native/Windows/participant/full latency qualification are separate and unclaimed.
<!-- SECTION:NOTES:END -->
