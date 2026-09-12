---
id: TASK-23196
title: 'Console Context rail: stop repeating provider and model'
status: Done
assignee:
  - '@claude'
created_date: '2026-08-29 21:56'
updated_date: '2026-08-30 02:43'
labels:
  - console
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Provider and model are displayed three times simultaneously: the Context Model section, the status bar, and the Inspector run recipe. The rail's copy is the one that costs scarce vertical space.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Provider and model appear once in the Console chrome
- [x] #2 The Model section retains parameters not shown elsewhere plus its Configure action
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Removed the Provider and Model rows from the Context rail's Model section. Both values were rendering in three places in one screenshot: this section, the persistent status bar, and the Inspector's run recipe.

The rail's copy is the one that went because it is the one costing scarce vertical space. Before removing it I verified the survivor actually survives at every width where the rail is visible: the status bar carries 'Provider: llama_cpp  Model: local-model' in full at 200, 160, 140, 110 and 100 columns, and below 100 the rail force-collapses anyway, so no width loses the information. The section keeps what is shown nowhere else -- Temperature, Max tokens, the system-prompt row and Configure -- so 'Model' now reads as model settings, which is a normal reading of the title. Saves 2 rail rows.

Also removed the now-dead provider_value/model_value derivation and the _summary_row_value import that only fed them.

Test fallout, two causes:
- Mine here: test_console_model_section.py pinned four rows and read the provider value; test_console_rail_reconciliation.py clicked the provider value to exercise activation. Retargeted at the temperature row, which exercises the same path.
- MINE FROM TASK-23195, missed in that task's sweep: test_console_rail_reconciliation.py pinned the exact old hint string in twelve places. That file was not in the batch-3 test run, which is why it was not caught then. Those assertions are about WHEN the hint shows, not what it says, and the text is now a function of which sections are hidden and how wide the rail is -- so they now assert the hint marker via a _shows_outer_hint helper rather than exact copy.

Not mine, verified by re-running with the whole working tree stashed: Tests/Architecture/test_timer_path_static_update_inventory.py has 2 failures naming schedules_workbench.py, and test_console_session_settings.py has 3, all pre-existing on dev.

preflight green. Files: UI/Console_Modules/left_rail.py; Tests/UI/test_console_model_section_dedup.py (new); test_console_model_section.py, test_console_rail_reconciliation.py updated.
<!-- SECTION:NOTES:END -->
