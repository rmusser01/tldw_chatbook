---
id: TASK-1371
title: Align theme editor with Settings screen rules (generic organ)
status: Done
assignee: []
created_date: '2026-08-05 23:38'
updated_date: '2026-08-05 23:58'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Re-critique found the Theme editor is the one 'generic organ': a stock Textual theme playground whose commit model, confirmation behavior, and copy don't follow the Settings screen's own rules (staged-vs-instant labeling, honest affordances, product voice). Delete confirmation and keyboard presets/apply hint were fixed in tasks 1367/1369; what remains is aligning the editor's commit-model labeling, confirmation behavior, and copy with the screen's conventions.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Theme editor Apply is explicitly labeled as instant-apply consistent with the screen's commit-model labels,Theme editor copy matches the Settings screen voice (no stock-playground phrasing),Theme editor destructive/mutating actions follow the same confirmation rules as the rest of the screen,Regression tests cover the labeling and any behavior changes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Audit theme editor vs screen conventions (ADR-031, INSTANT_APPLY_BEHAVIOR_COPY, revert confirmation). 2. Align Apply hint copy to screen phrasing. 3. Add confirmation guard to Reset/New when they would discard unsaved edits (mirrors screen revert). 4. De-playground copy (drop exclamation). 5. Update/add regression tests. ADR required: no
ADR path: N/A
Reason: routine UX fix, no architectural decision
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Audit of what remained after tasks 1367/1369, comparing against the screen's conventions (`INSTANT_APPLY_BEHAVIOR_COPY`/`STAGED_SAVE_BEHAVIOR_COPY`, the revert confirmation in `action_settings_revert_category`, ADR-031 rule 3). Three gaps found and fixed in `tldw_chatbook/Widgets/settings_theme_editor.py`:

1. Commit-model labeling: the Apply hint said "takes effect immediately"; reworded to the screen's exact instant-apply phrasing "applies immediately - no Save needed" (the widget still must not import the screen, so the string is mirrored with a comment pointer).
2. Confirmation behavior: the screen confirms before discarding unsaved edits (revert -> ConfirmationDialog "Discard changes"/"Keep editing"), but the editor's Reset and New discarded the working palette with no guard. Both now push the same ConfirmationDialog shape when `is_modified` is set, and run directly when unmodified (lossless path). Bodies moved to `_reset_theme()`/`_new_theme()` post-confirmation helpers, matching the task-1367 delete pattern. Clone needs no guard (it keeps the palette; only renames). Delete was already guarded (task-1367); Apply/Save/Export are commit/non-destructive actions.
3. Copy voice: dropped the playground exclamation in "Theme generated from primary color!" -> "...color." Other notifications already matched the screen's plain-sentence style.

Modified/added files:
- `tldw_chatbook/Widgets/settings_theme_editor.py`
- `Tests/UI/test_settings_theme_editor.py` (updated apply-hint assertion; new tests: reset skips dialog when unmodified, reset confirm/cancel flow, new confirm/cancel flow)

Tests: `Tests/UI/test_settings_theme_editor.py` 15 passed.
<!-- SECTION:NOTES:END -->
