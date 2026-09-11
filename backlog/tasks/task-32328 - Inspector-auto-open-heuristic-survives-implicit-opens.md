---
id: TASK-32328
title: >-
  Inspector auto-open heuristic survives implicit opens
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
UX review B4. The 118-128 column Inspector auto-open band (_should_open_standard_width_inspector, chat_screen.py ~8298-8329) is permanently suppressed once ANY right_open preference is stored. Use the existing explicit-marker pattern (ADR-043 left_open_explicit) so only explicit Inspector toggles disable the heuristic.

Filed from the 2026-09-10 Console rail UX review (review item B4).
<!-- SECTION:DESCRIPTION:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. RED: marker semantics test (explicit writes/preserves; implicit doesn't). 2. Add key+helper+serializer arm. 3. Heuristic reads marker instead of key presence. 4. Run all five rail-state suites.
<!-- SECTION:PLAN:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Storing an implicit right_open preference (e.g. via reveal logic) no longer permanently disables the auto-open heuristic
- [x] #2 An explicit user toggle of the Inspector rail still disables auto-open
- [x] #3 Existing rail preference serialization round-trips unchanged for existing configs (no migration break)
- [x] #4 Unit tests cover both paths (implicit store keeps heuristic; explicit toggle kills it)
<!-- AC:END -->

## Implementation Notes
<!-- SECTION:NOTES:BEGIN -->
### Close-out (2026-09-10)

**Approach.** Mirrored ADR-043's left_open_explicit marker for the
Inspector: new `CONSOLE_RAIL_RIGHT_OPEN_EXPLICIT_KEY` +
`console_rail_right_open_explicit()` in `Chat/console_rail_state.py`;
`serialize_console_rail_updated_preferences` writes the marker on an
explicit right-rail toggle and preserves it across later unrelated
writes. `_should_open_standard_width_inspector` now returns False on the
MARKER instead of `right_open` key presence — an implicit writer (e.g. a
first-ever section toggle, which serializes `right_open: false` via the
base shape) no longer permanently kills the 118-128-column auto-open
band; only an explicit user toggle does. Round-trips for existing configs
are additive (new key only when explicitly toggled), so no migration.

**ADR check.** Follows existing ADR-043 (explicit-marker pattern) — no
new ADR needed; decision recorded here.

**Modified.** `Chat/console_rail_state.py` (key, helper, serializer arm),
`UI/Screens/chat_screen.py` (heuristic reads marker; import),
`Tests/Chat/test_console_rail_state.py` (+1 test: explicit toggle writes
marker and survives later writes; implicit writes don't; seeded Mapping
omits the key). Verified: rail-state + agent + prune + priority +
narrow-layout suites — 137 passed (1 narrow-layout failure pre-existing
on the clean tree, verified by stash).

### Walkthrough verification (2026-09-10, dev tip a0b8f96416)

VERIFIED (code): chat_screen.py:13077-13078 returns False if 'right_open' key exists in stored prefs at all - any explicit toggle permanently kills the 120-col auto-open. left_open_explicit marker pattern already exists to copy.
Live evidence: headless tmux run, scratch profile, 160x45 and 124x45 captures; code evidence re-verified in the implementation worktree.
<!-- SECTION:NOTES:END -->
