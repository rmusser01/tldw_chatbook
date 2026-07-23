---
id: TASK-398
title: Add collapsible Console composer reading mode
status: In Progress
assignee:
  - '@codex'
created_date: '2026-07-23 04:00'
updated_date: '2026-07-23 04:14'
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
- [ ] #1 Composer collapses manually from the expanded Composer control to an exact one-row status and Expand bar while the separate runtime status strip remains visible when present.
- [ ] #2 Collapse and expansion preserve the active mounted composer's draft, paste segments, caret selection, and pending attachments; transient Unfurl and unknown-command send confirmations reset safely.
- [ ] #3 Collapsed mode keeps active Stop control visible, blocks hidden draft input and paste, supports F6 and one-press Escape restoration, and preserves transcript selection plus anchored or manually scrolled reading position.
- [ ] #4 Collapse state is Console-wide and retained across tab switches and in-app navigation, but resets to expanded for a new app instance and never changes automatically.
- [ ] #5 Mounted tests cover expanded and collapsed geometry, focus and keyboard behavior, run and setup states, rapid toggles, state retention, and 140x42 plus 100x32 layouts; focused Console regressions pass.
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
7. Capture and inspect live Textual-web evidence, obtain user approval, add implementation notes, check all acceptance criteria, and mark TASK-398 Done.

Detailed executable plan: Docs/superpowers/plans/2026-07-22-console-collapsible-composer.md
<!-- SECTION:PLAN:END -->
