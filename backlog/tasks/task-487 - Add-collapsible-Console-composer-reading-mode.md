---
id: TASK-487
title: Add collapsible Console composer reading mode
status: Done
assignee:
  - '@codex'
created_date: '2026-07-23 04:00'
updated_date: '2026-07-23 13:46'
labels:
  - console
  - ui
  - accessibility
dependencies: []
references:
  - Docs/superpowers/specs/2026-07-22-console-collapsible-composer-design.md
  - Docs/superpowers/plans/2026-07-22-console-collapsible-composer.md
  - backlog/decisions/011-chatbook-workbench-ui-system.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let Console users reclaim transcript height while reading long responses by manually collapsing the composer to a safe one-row restore bar without losing unsent work or run control.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Composer collapses manually from the expanded Composer control to an exact one-row status and Expand bar while the separate runtime status strip remains visible when present.
- [x] #2 Collapse and expansion preserve the active mounted composer's draft, paste segments, caret selection, and pending attachments; transient Unfurl and unknown-command send confirmations reset safely.
- [x] #3 Collapsed mode keeps active Stop control visible, blocks hidden draft input and paste, supports F6 and one-press Escape restoration, and preserves transcript selection plus anchored or manually scrolled reading position.
- [x] #4 Collapse state is Console-wide and retained across tab switches and in-app navigation, but resets to expanded for a new app instance and never changes automatically.
- [x] #5 Mounted tests cover expanded and collapsed geometry, focus and keyboard behavior, run and setup states, rapid toggles, state retention, and 140x42 plus 100x32 layouts; focused Console regressions pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/011-chatbook-workbench-ui-system.md
Reason: Focused Console presentation behavior within ADR-011 stable-compose-tree and screen-orchestration boundaries; no storage, schema, service, security, dependency, or cross-module decision.

1. Add failing mounted tests for the stable expanded/collapsed composer presentations, preserved editor state, status derivation, exact one-row geometry, cursor gating, and generated CSS.
2. Implement the dual presentation in ConsoleComposerBar and regenerate the modular stylesheet.
3. Add failing interaction tests for toggle controls, dynamic Escape, F6 focus, hidden-input safety, transient confirmation reset, and stale deferred focus.
4. Implement ChatScreen-owned transient state, state-aware focus resolution, transcript reading-state restoration, setup blocking, and shared Stop routing.
5. Harden with mounted tests for anchored/manual scroll, selection, rapid toggles, active/stale Stop, tab/workspace/navigation/recompose retention, setup, and 140x42 plus 100x32 layouts.
6. Run focused and broader Console/Chat regressions, ruff, compile checks, and git diff checks.
7. Capture and inspect live Textual-web evidence, obtain user approval, add implementation notes, check all acceptance criteria, and mark TASK-487 Done.

Detailed executable plan: Docs/superpowers/plans/2026-07-22-console-collapsible-composer.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Added stable, mounted expanded and collapsed composer presentations. The one-row collapsed bar retains the same editor state in memory while exposing privacy-safe draft, attachment, and generation status plus Expand and contextual Stop controls.
- Kept collapse as transient, manual-only `ChatScreen` state across Console tab and in-app navigation changes. Dynamic priority Escape, F6, setup restoration, and forced focus all use state-aware targets so hidden composer input cannot regain focus.
- Preserved anchored and manually scrolled transcript reading positions plus selection through semantic restoration, with revision and expected-state guards preventing stale rapid-toggle callbacks.
- Reset transient Unfurl and unknown-command confirmations on collapse, blocked hidden input/paste and hidden paste-token geometry, and delegated collapsed Stop through the existing setup-aware run-control path.
- Updated the CSS source and generated bundle, added a fail-closed CSS builder guard, and applied the live-QA toggle-legibility fix so full `Composer ▾`, `Stop`, and `Expand ▴` labels remain visible at compact geometry.
- Added mounted behavior, lifecycle, race, focus, setup, run, and exact 140x42/100x32 geometry coverage. Focused gates passed; the mandatory broad evidence recorded 2023 passes with pre-existing failures reproduced at the pre-feature revision and loopback-denied setup nodes passing with permission.
- Captured and individually inspected six synthetic-only live screenshots under `Docs/superpowers/qa/console-collapsible-composer-2026-07/`; the user explicitly approved the wide and compact expanded, retained-draft, and generating states on 2026-07-23 for commit `20aba85893a4bd06e8091389b2f57161877b154c`.
- ADR required: no. Existing `backlog/decisions/011-chatbook-workbench-ui-system.md` applies because this work stays within its stable compose-tree and screen-orchestration boundaries.
<!-- SECTION:NOTES:END -->
