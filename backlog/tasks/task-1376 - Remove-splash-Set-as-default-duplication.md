---
id: TASK-1376
title: Remove splash Set-as-default duplication
status: Done
assignee: []
created_date: '2026-08-05 23:38'
updated_date: '2026-08-05 23:58'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Re-critique minor: the splash viewer's 'Set as default' button duplicates the 'Default card' Select two rows above it — two controls for one setting. Converge on a single control (keep the Select as the source of truth, or make the button reflect/select consistently).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 One control sets the default splash card; no duplicate affordance,Remaining control stays honest about its commit model per ADR-031,Regression test updated/added
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Remove the Set-as-default button + handler; keep the Default card Select as single source of truth (instant-apply, already labeled). 2. Add regression test asserting single control. 3. Report any orphaned CSS selector (none found). ADR required: no
ADR path: N/A
Reason: routine UX fix, no architectural decision
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Kept the `Default card` Select as the single source of truth and removed the duplicate `Set as default` button plus its `handle_set_default_pressed` handler from `tldw_chatbook/Widgets/settings_splash_screen_viewer.py`. The Select was the better control to keep: it already persists `card_selection` on change (instant-apply) and its commit model is labeled inline by the existing "applies immediately - no Save needed; text fields apply on Enter" hint, so ADR-031 copy honesty is preserved with no new copy. The button also had a mild dishonesty (it silently overwrote the Select's value without reflecting the current setting), which the removal eliminates.

Verified no other references to the button id existed (no CSS selector, no screen or test references), so no orphaned CSS cleanup is needed.

Modified/added files:
- `tldw_chatbook/Widgets/settings_splash_screen_viewer.py` (removed button from compose + handler)
- `Tests/UI/test_settings_splash_screen_viewer.py` (new: single-control regression test; instant-apply persistence test with mocked config save)

Tests: `Tests/UI/test_settings_splash_screen_viewer.py` 7 passed.
<!-- SECTION:NOTES:END -->
