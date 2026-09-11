---
id: TASK-32334
title: >-
  Inspector section rows carry a status glyph not color alone
status: To Do
assignee:
  - '@zcode'
created_date: '2026-09-10 12:00'
labels:
  - console
  - ux
  - rail-ux-review
priority: medium
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
UX review C5. ConsoleInspectorSection conveys row status only via a CSS class (console-inspector-section-row-<status>); nothing textual or glyphal distinguishes running/done/error/blocked in low-color terminals (console_inspector_section.py ~76-78, 599-605). Add a leading status glyph using the app's existing glyph conventions.

Filed from the 2026-09-10 Console rail UX review (review item C5).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each inspector section row prefixes a status glyph from the app's established glyph set, consistent across statuses
- [ ] #2 Glyph choice does not collide with markers that already have meaning in rail rows (marker tooltip suffix conventions)
- [ ] #3 Color styling remains as the secondary signal
- [ ] #4 Tests assert glyph presence per status
<!-- AC:END -->

## Implementation Notes
<!-- SECTION:NOTES:BEGIN -->
### Walkthrough verification (2026-09-10, dev tip a0b8f96416)

PARTIAL on dev: fleet rows already carry status glyphs (agent.py _AGENT_STATUS_GLYPHS); Environment/Tasks rows do not (console_environment_state.py:639-668). Scope: bring the glyph convention to Environment/Tasks projections.
Live evidence: headless tmux run, scratch profile, 160x45 and 124x45 captures; code evidence re-verified in the implementation worktree.
<!-- SECTION:NOTES:END -->
