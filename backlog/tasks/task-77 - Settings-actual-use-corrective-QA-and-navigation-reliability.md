---
id: TASK-77
title: Settings actual-use corrective QA and navigation reliability
status: In Progress
labels:
- settings
- ux
- qa
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Verify the Settings configuration hub through actual rendered app use and correct any confirmed usability blockers in category navigation, field editing, dropdown selection, save/revert/test feedback, and keyboard operation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Actual rendered Settings walkthrough verifies category navigation works with mouse and keyboard.
- [ ] #2 Provider, model, endpoint, credential, numeric, and toggle controls remain readable while focused and while text is entered.
- [ ] #3 Dropdown selection works without clipping, hidden selection state, or invalid blank provider persistence.
- [ ] #4 Save, revert, validation, and test actions show clear status/recovery feedback and do not block the UI for long work.
- [ ] #5 Any confirmed blocker is covered by a failing regression before the production fix.
- [ ] #6 Final CDP screenshot evidence is captured and explicitly approved before PR creation.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Run the Settings screen in textual-web from an isolated config/data home so actual behavior is observable without contaminating local user state.
2. Use CDP/browser automation to verify rendered category navigation, provider dropdown selection, input focus visibility, save/revert/test feedback, and keyboard traversal.
3. Record any confirmed blocker with exact reproduction steps and root-cause evidence before editing production code.
4. Add a focused failing regression for the first confirmed blocker, then implement the smallest safe fix.
5. Rerun focused Settings tests, run diff hygiene, capture actual final screenshots, and wait for user approval before PR creation.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Slice 1 addressed the confirmed endpoint-field blocker found during CDP QA: clicking a URL-valued provider endpoint in textual-web opened the browser URL instead of behaving like an editable Settings input. Added a URL-safe endpoint input renderer that breaks browser autolinking in display text while preserving the raw value for validation and saving. Added focused regressions for the display transform and endpoint widget composition, then verified with Settings tests and CDP screenshot evidence approved by the user.

Slice 2 addressed the confirmed provider dropdown blocker found during CDP QA: the compact Settings Select kept `max-height: 3`, which constrained its rendered dropdown overlay and made provider choices appear clipped/hidden in textual-web. Updated the focus contract regression to forbid the clipping rule, removed the source CSS cap, rebuilt generated CSS, and captured approved CDP screenshots showing the dropdown open and a provider selection applied with visible state/inspector feedback.

Slice 3 addressed the confirmed footer shortcut leakage found during CDP QA: typing `s`, `r`, or `t` into focused Settings text inputs could invoke Save, Revert, or Test actions through global Settings bindings. Added a focused regression that verifies real `s`/`r`/`t` keypresses remain literal input and save/test/revert actions are ignored while a provider text input owns focus, guarded those action handlers for `Input` and `TextArea` focus, reran focused Settings/provider tests, and captured approved CDP screenshot evidence showing visible typed provider inputs without shortcut-triggered toasts.

Slice 4 addressed the confirmed Console Behavior feedback contradiction found during CDP QA: the clean state showed `No unsaved changes` while the detail pane still reported `Console behavior settings staged.` Added a mounted regression for the clean-state copy, normalized the Console Behavior result helper so staged feedback only appears while the category is dirty, routed widget refresh through that helper, reran focused Console Behavior tests, and captured approved CDP screenshot evidence showing coherent disabled Save/Revert state and clean-state recovery copy.

Review follow-up tightened the Console Behavior result state machine after PR review found additional clean-state paths: undoing changed Console defaults back to loaded values could still render staged copy, and stale staged copy could mask the workbench-scope-unavailable warning. Added regressions for both cases, routed all Console default handlers through the common post-stage updater, prioritized the workbench warning before stale-staged normalization, and preserved successful-save copy when an edit is undone back to the saved value.

Slice 5 addressed the confirmed Console Behavior action-feedback bug found during actual textual-web QA: entering an out-of-range background frame rate could save successfully after silently clamping the value. Added a regression that keeps invalid frame-rate edits dirty, blocks persistence, and shows the `1-12` constraint; added explicit Settings save validation for `background_effects.fps`; reran focused Console Behavior tests; and captured approved CDP screenshot evidence showing the rejected save with visible invalid input and preserved unsaved state.

Slice 6 addressed the merged Settings action regression found before further CDP QA: `action_settings_save_category()` could crash with `NoActiveAppError` when called on a detached Settings screen because the text-entry focus guard assumed an active Textual app context. Reused the existing failing regression as the red case, added a detached-screen-safe focus helper shared by Settings focus guards, reran the full focused Settings suite, and captured approved actual textual-web screenshot evidence at `Docs/superpowers/qa/product-maturity/screen-qa/settings/settings-action-focus-detached-baseline-fixed-2026-06-05.png`.

Review follow-up extended the same detached-screen focus guard to `SettingsScreen.on_key()` after PR review found it still read `self.app.focused` directly. Added a focused red regression for detached slash-key handling, routed `on_key()` through the shared focus helper, and reran focused Settings verification after rebasing on latest `dev`.

Slice 7 addressed the confirmed Advanced Config reachability issue found during actual textual-web/CDP QA: `Validate Raw TOML`, `Load Backup`, and `Save Raw TOML` rendered below the raw editor, making the safety actions hard to discover and unreachable at common viewport heights. Added a mounted regression that keeps the safety action row before the raw editor, moved the action/result block above the editor, narrowed the Settings inspector from the prior equal-width layout to a `3fr / 6fr / 2fr` sections/detail/inspector contract, regenerated the modular CSS bundle, and captured actual textual-web screenshot evidence at `Docs/superpowers/qa/product-maturity/screen-qa/settings/settings-advanced-actions-narrow-inspector-2026-06-06.png`.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
