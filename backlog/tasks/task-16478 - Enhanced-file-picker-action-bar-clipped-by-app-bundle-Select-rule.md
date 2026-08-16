---
id: TASK-16478
title: Enhanced file picker action bar clipped by app bundle Select rule
status: Done
assignee:
  - '@robert'
created_date: '2026-08-15 23:04'
updated_date: '2026-08-15 23:05'
labels: []
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Under the real app CSS bundle, every filtered EnhancedFileDialog (Roleplay character import, avatar/expression uploads, Console attachments) rendered its file-type Select at 100% width: the filename input was crushed to ~6 columns and the Select/Cancel buttons were laid out past the dialog's right edge and clipped out of view, so the dialog appeared buttonless. The bundle's bare Select rule beats any DEFAULT_CSS; the existing _dialogs.tcss fix was scoped to the vendored FileOpen/FileSave only.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Filtered EnhancedFileDialog renders Import/Cancel inside the dialog bounds under the app CSS bundle
- [x] #2 Filename input keeps flex width (>=20 cols at 160x50) with a filter Select pinned to 24 cols like the vendored dialogs
- [x] #3 Vendored FileOpen/FileSave rows unchanged
- [x] #4 Regression test loads app.py's CSS tiers and fails on the clipped layout
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce under the production CSS tiers and measure (regions of #filename-input/#file-filter/#select/#cancel vs dialog bounds)
2. Extend css/components/_dialogs.tcss Select pin to EnhancedFileDialog; regenerate the bundle via build_css
3. Add a bundle-tier regression test (Tests/UI/test_enhanced_file_dialog_bundle_css.py) and verify it fails red on the unfixed bundle
4. Run picker suites + bundle sync guard; document the recurrence in lessons-testing-evidence.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Root cause: `css/features/_conversations.tcss` has a bare `Select { width: 100%; margin-bottom: 1; }`. CSS_PATH-sourced rules always outrank widget `DEFAULT_CSS` regardless of specificity (the documented "Defect 1" cascade quirk), so `EnhancedFileDialog`'s own input-bar styling could not defend itself. task-1479 fixed this for the vendored dialogs (`FileSave InputBar Select, FileOpen InputBar Select { width: 24 }`) and explicitly left the enhanced picker out of scope.
- Measured breakage (160x50, production CSS stack): filename input width 6, filter Select width 149, `#select` at x=161 and `#cancel` at x=178 -- both past the dialog's right edge (x+width=156), i.e. clipped out of view. InputBar virtual width 189 inside a 150-column body. Unfiltered enhanced dialogs (no Select) were unaffected, which is why the vendored Library dialogs (called without filters) looked fine.
- Fix: added `EnhancedFileDialog InputBar Select` to the existing width-24 pin in `css/components/_dialogs.tcss` (comment updated with the incident); the type selector matches EnhancedFileOpen/EnhancedFileSave/EnhancedSelectDirectory via Textual's CSS type-name MRO. Regenerated `tldw_cli_modular.tcss` with `python -m tldw_chatbook.css.build_css`. Post-fix: filename input 91 cols, Select 24, buttons at x=121/138 -- inside the dialog.
- Regression guard: `Tests/UI/test_enhanced_file_dialog_bundle_css.py` mounts the dialog under the exact `TldwCli.CSS_PATH` stack and asserts button containment + input width + Select pin. Verified red (fails with "#select is clipped past the dialog's right edge (x=161)" on the unfixed bundle via git stash) and green after the fix.
- Verification: 108 passed across the picker suites (bundle-css, enhanced select directory, mount, start dir, bookmarks lazy, filters callable, action tooltips, fspicker keyboard save/entry display, css bundle sync guard); personas bulk-actions tests pass; ruff clean. `Tests/UI/test_design_token_governance.py::test_all_referenced_ds_tokens_are_defined` fails identically with and without this change -- pre-existing on this branch from the separate design-token work, not addressed here.
- Lesson recorded as a recurrence of the TASK-16221 entry in `backlog/docs/lessons-testing-evidence.md`: a bare-host render is not evidence the app-bundle render works.
<!-- SECTION:NOTES:END -->

ADR required: no
ADR path: N/A
Reason: CSS-scoped defect fix within the existing dialog styling convention (the _dialogs.tcss Defect 1 pattern); no architectural boundary touched.
