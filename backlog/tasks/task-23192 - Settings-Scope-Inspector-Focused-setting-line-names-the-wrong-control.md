---
id: TASK-23192
title: Settings Scope Inspector 'Focused setting' line names the wrong control
status: Done
assignee: []
created_date: '2026-08-29 02:25'
labels:
  - ux
  - settings
  - a11y
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Scope Inspector's 'Focused setting' line read 'Appearance defaults' while the Reduce motion control demonstrably held focus (verified by a style-diff showing the focused background on that control). The line exists to tell keyboard users what their focus is currently on, which is exactly the guarantee TASK-23109's setting-level landing depends on -- a wrong name is worse than no line, because it contradicts what the user can see. Observed during the TASK-23109 verification pass.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The 'Focused setting' line names the control that actually holds focus, including controls reached by search landing and by plain Tab traversal
- [x] #2 When focus is on a container or a non-setting control, the line says so rather than naming an unrelated setting
- [x] #3 A mounted test focuses at least two distinct settings and asserts the inspector line matches each
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Find how the line derives its label and reproduce the wrong name in a mounted test.
2. Remove the cause rather than patching the one reported control.
3. Mutation-check, then verify live via both search landing and Tab traversal.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Root cause.** The same set of guided fields was written down twice. The
`DescendantFocus` handler kept a hand-maintained set of widget ids it would
record in `_active_settings_field_id`, and the `_*_field_guidance_rows_base`
methods kept their own `if field_id == ...` branches. They drifted:
`settings-appearance-reduce-motion`, `settings-appearance-ascii-glyphs` and
`settings-model-context-window` each had full guidance rows written for
them but were absent from the focus set, so focusing them recorded `None`
and the inspector fell through to the category-level fallback.

**Fix.** The second list is deleted. `_guided_field_id()` asks the
category's own guidance whether it has rows for the focused widget -- a
field is guided exactly when its rows differ from the no-focus fallback --
so the guidance branches are the single definition and the two cannot drift
apart again. Both routes to a control (search landing and plain Tab
traversal) arrive through the same `DescendantFocus`, so both are covered
by one mechanism.

**AC2.** The fallback used to name the CATEGORY ("Appearance defaults",
"Storage defaults", "Provider setup"), which a keyboard user cannot tell
apart from a setting's name. All three now render the shared
`NO_FOCUSED_SETTING_COPY` ("None - Tab to a setting"), which says focus is
not on a setting and names the way back to one.

**Ownership decision.** `settings_search_index.py` was NOT reused as the
naming path. The guidance branches already carry curated labels, and the
index's labels differ where both exist ("Model context window tokens" vs
"Model context window"), so borrowing them would have made the search
result and the inspector disagree. For Appearance the two coverages are
identical, so nothing is lost.

**Trade-off.** `_guided_field_id` probes the guidance by setting and
restoring `_active_settings_field_id`, which is less direct than a
parameter would be, but it avoids re-threading three ~100-line branch
methods and their existing test callers. It runs only on focus change.

**Verification.** `Tests/UI/test_settings_scope_inspector_focus.py` mounts
the real screen and pins the line against two distinct settings reached by
two different routes, plus the non-setting case and the recorded-id
contract. Mutation-checked both ways: making nothing guided fails all
three names; making everything guided fails the non-setting assertion.

Live, isolated fresh profile:

    "/" -> "Reduce motion" -> Enter   Focused setting: Reduce motion
    Tab                              Focused setting: ASCII glyphs
    Tab                              Focused setting: Smooth scrolling
    focus the category rail button   Focused setting: None - Tab to a setting
    "/" -> "Model context window"    Focused setting: Model context window

**Observed, deliberately not actioned.** Console Behavior's guidance has a
`settings-console-context-*` branch its own hand-list never admits, so that
branch is dead today. Its inspector renders no "Focused setting" row, so
nothing displays a wrong name there, and admitting those ids would change
guidance for a category outside these ACs.

**Files.** `tldw_chatbook/UI/Screens/settings_screen.py`,
`Tests/UI/test_settings_scope_inspector_focus.py`.
<!-- SECTION:NOTES:END -->
