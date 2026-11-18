---
id: TASK-34330
title: 'First-run live-contract suite runs green again'
status: To Do
assignee: []
created_date: '2026-10-03 16:30'
labels:
  - first-run-wizard
  - tests
  - ux-review-2026-10-02
dependencies:
  - TASK-34100.1
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Tests/UI/test_first_run_wizard_live_contract.py drives the real app through first-run setup: keyboard walks, focus after async work, Esc and Exit, kill-and-resume, and re-runs. It is the suite that should catch regressions of the focus and lifecycle contract that TASK-34100.1 put in place. It is 63 of 95 red, both on base 1d8fe87659 and on TASK-34100.1's branch, so it guards nothing.

Two failure families were seen in TASK-34100.1's reviews:
- The file is not marked `bootstrap_profile`. The 41 `-k provider` reds in Tests/Wizards had the same cause (`RecoveryRequired('raw_source_selection_changed')`, fixture drift; see TASK-34100.1 AC#1). Under a temporary `bootstrap_profile` copy, the retargeted resume-marker test passes.
- Most remaining nodes never reach the wizard screen. The picker wait times out at 30 s.

Two Tests/Wizards nodes in the same family are also red on base and on the branch: `test_summary_footer_shows_the_effective_config_path` (RecoveryRequired in `SummaryStep._render_rows`), and the recovery subset of this file (9 failed, 8 passed on both arms).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Tests/UI/test_first_run_wizard_live_contract.py passes under the plain pytest command, serially and with -n 4, with each fixed node's failure cause recorded
- [ ] #2 No node in the file waits out a 30 s timeout before the wizard screen appears
- [ ] #3 Tests/Wizards/test_first_run_setup_wizard.py::test_summary_footer_shows_the_effective_config_path passes
- [ ] #4 Any node that is deleted or skipped instead of fixed is named, with the reason, in the Implementation Notes
<!-- AC:END -->
