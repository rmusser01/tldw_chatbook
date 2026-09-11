---
id: TASK-32334
title: >-
  Inspector section rows carry a status glyph not color alone
status: Done
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

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. RED: update env/tasks pinned strings to glyph-prefixed + add a dedicated glyph test. 2. Add shared STATUS_GLYPHS/status_glyph in console_glyphs; ⚠ ASCII fallback; derive agent.py map from it. 3. Prefix glyphs at every status-bearing env/tasks row site. 4. Run env + glyph + fleet + inspector suites.
<!-- SECTION:PLAN:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Each inspector section row prefixes a status glyph from the app's established glyph set, consistent across statuses
- [x] #2 Glyph choice does not collide with markers that already have meaning in rail rows (marker tooltip suffix conventions)
- [x] #3 Color styling remains as the secondary signal
- [x] #4 Tests assert glyph presence per status
<!-- AC:END -->

## Implementation Notes
<!-- SECTION:NOTES:BEGIN -->
### Close-out (2026-09-10)

**Approach.** Section-row status was colour-class only. Added ONE shared
vocabulary — `Chat/console_glyphs.STATUS_GLYPHS` + `status_glyph()` —
covering done/running/error/cancelled/stuck/blocked; the fleet panel's
private map (`agent.py _AGENT_STATUS_GLYPHS`) now derives from it (same
keys, same marks, plus "blocked"). `Chat/console_environment_state.py`
prefixes the glyph on every status-bearing Environment/Tasks row via
`_with_status_glyph` (unavailable tier, stale Changes/Branch/PR rows,
checks summary, failing check children, tasks head + entries); fresh
(status-less) rows get no glyph. Glyphs ride in `primary_text`, so the
existing render-seam `resolve_glyph_text` ASCII fallback applies; added
the missing "⚠"→"[!]" fallback (same attention mark as ◆). One
self-caught mistake: first draft mapped error to GLYPH_CLOSE (✕, the
close-button mark) instead of the fleet's ✗ — would have changed fleet
rendering; fixed before commit.

**ADR check.** Not required — display vocabulary inside existing widgets.
Linked: glyph sharing follows ADR-034's one-vocabulary precedent.

**Modified.** `Chat/console_glyphs.py`, `Widgets/glyph_fallback.py`,
`UI/Console_Modules/agent.py`, `Chat/console_environment_state.py`,
`Tests/Chat/test_console_environment_state.py` (pins + new glyph test).
Verified: env state + glyphs + env section UI + fleet panel + inspector
section suites — 114 passed. Pre-existing dev-tip failures in
test_console_agent_controller / test_console_rail_color_grammar /
test_console_fleet_survivor_tick are UNRELATED (they pass kwargs dev's
ConsoleLeftRail never accepted; verified those files untouched by this
arc — same failures exist on origin/dev).

### Walkthrough verification (2026-09-10, dev tip a0b8f96416)

PARTIAL on dev: fleet rows already carry status glyphs (agent.py _AGENT_STATUS_GLYPHS); Environment/Tasks rows do not (console_environment_state.py:639-668). Scope: bring the glyph convention to Environment/Tasks projections.
Live evidence: headless tmux run, scratch profile, 160x45 and 124x45 captures; code evidence re-verified in the implementation worktree.
<!-- SECTION:NOTES:END -->
