---
id: TASK-32602
title: Preserve Prompt editor focus across terminal resize
status: Done
assignee:
  - '@codex'
created_date: '2026-09-15 06:26'
updated_date: '2026-09-15 07:27'
labels:
  - library
  - prompts
  - focus
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Prompt continuity audit found that resizing a focused Advanced message field from 170x48 to 80x24 leaves it below the viewport. A later wide transition can restore the Create rail row instead of the live editor focus. Stable-size Basic and Advanced layout fixes do not address this shared resize restoration path. Evidence and the removed local-scroll experiment are recorded in Docs/superpowers/qa/2026-09-14-prompt-continuity/README.md. Reproduce with the new continuity journey before its Basic return: focus the Advanced message field, resize to 80x24 and back to 170x48, and assert the same focused widget plus its compositor-painted text.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A focused Prompt text field stays visibly focused across 170x48 to 80x24 and back without changing its content or identity.
- [x] #2 A newer explicit focus move wins over deferred resize restoration, including a move outside the Prompt editor.
- [x] #3 Resize remains free of source reloads and preference writes, with production-CSS and native evidence.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/086-library-adaptive-reader-shell.md (existing); backlog/decisions/150-design-token-system-and-design-language.md (existing)
Reason: Repair resize focus and visibility within the existing destination-owned editor and adaptive shell contracts; no new persistence, interface, or layout policy.

1. Reproduce Basic/Advanced resize visibility and trace competing focus restorations with production CSS.
2. Keep Notes restoration from overriding the Prompt shell and reveal the current editor focus after settled geometry, without overriding newer focus.
3. Verify resize crossings, newer focus moves, retained identity/content, and no source or preference work using targeted tests.
4. Run a disposable-profile native resize journey with fresh shutdown receipts; document evidence and review the bounded diff.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Prompt routes now skip the Notes semantic-focus fallback after shared resize presentation updates. The permanent Prompt work pane reveals only its currently focused descendant after layout; immediate scrolling avoids a second stale callback. Basic and Advanced widgets retain content and identity across compact, wide, and height-only changes.

Added eight real-storage / production-CSS resize and newer-focus regressions. Tightened a neighboring Prompt read-only-preview mount wait and changed the existing Notes stale-restore test to use keyboard Enter, after reproducing its synthetic-input failure with HEAD's unchanged resize method. No other production behavior changed.

Validation: 58 Prompt/resize tests, 6 neighboring Notes focus tests, and 31 governance tests pass (95 disjoint targeted cases; no full sweep). New/small Python files pass Ruff and formatting; changed legacy methods pass formatting with no added full-file diagnostics. Self-review and diff whitespace checks completed.

Native evidence: 16 actual tmux resize cases through TldwCli/LinuxDriver, both modes and themes, isolated ten-database profile, unchanged browse tokens and config hashes. Saved Prompt version 1 verified through SQLite. Normal Ctrl+Q from 80x24 returned from app.run and produced a fresh exit 0; shell return was observed before closing the owned session. Existing startup notices remain recorded.

Evidence: Docs/superpowers/qa/2026-09-15-prompt-resize/README.md. Updated the Prompt user guide, workflow audit, and input-boundary testing lesson.

ADR required: no new ADR. Existing backlog/decisions/086-library-adaptive-reader-shell.md and backlog/decisions/150-design-token-system-and-design-language.md apply. The change repairs existing destination and focus boundaries without changing layout or persistence policy. Integration into dev and the broader feature audit remain pending.
<!-- SECTION:NOTES:END -->
