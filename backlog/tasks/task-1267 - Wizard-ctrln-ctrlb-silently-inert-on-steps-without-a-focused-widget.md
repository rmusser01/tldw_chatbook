---
id: TASK-1267
title: Wizard ctrl+n/ctrl+b silently inert on steps without a focused widget
status: Done
assignee: []
created_date: '2026-07-29 22:28'
updated_date: '2026-07-31 02:37'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during the first-run wizard final-review fix wave (see Docs/superpowers/plans/2026-07-28-first-run-setup-wizard.md). Textual key-binding resolution walks ancestors of the focused widget; on steps like Provider (RadioSet with no default-pressed button) nothing is focused after on_show, so the container's ctrl+n/ctrl+b bindings never fire. Pre-existing focus-management gap, orthogonal to the crash fixed in the wave; needs a focus strategy in each step's on_show (or a screen-level binding), without modifying BaseWizard.py.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 ctrl+n advances from every wizard step without requiring a prior click
- [x] #2 ctrl+b goes back from every step past Welcome
- [x] #3 Pilot regression test covers at least Provider and Model steps
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Resolved by the PR-1116 focus work (merged 98483dbc3): SetupWizardContainer.show_step focuses each incoming step's first displayed focusable (preferred_focus override for Provider), so bindings always resolve without a prior click. Verified on merged dev: test_ctrl_n_on_summary_dismisses_and_completes, test_ctrl_n_still_works_after_focus_was_on_a_now_hidden_widget, test_provider_reentry_with_visible_discovery_button_focuses_list (covers Provider+Model, ctrl+n and ctrl+b), test_ctrl_n_ctrl_b_do_not_crash_and_move_one_step — 4/4 pass. No new code needed.
<!-- SECTION:NOTES:END -->
