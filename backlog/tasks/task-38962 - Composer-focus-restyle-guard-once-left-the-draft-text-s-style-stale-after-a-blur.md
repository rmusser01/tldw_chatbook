---
id: TASK-38962
title: "Composer focus-restyle guard once left the draft text's style stale after a blur"
status: To Do
created_date: 2026-11-05 21:35
labels:
- follow-up
- tests
- flaky
- console
priority: low
references:
- Tests/UI/test_console_composer_state_restyle.py
- tldw_chatbook/Widgets/Console/console_composer_bar.py
- tldw_chatbook/css/features/_console.tcss
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 8 of the 2026-10-23 burn-down; checked on dev 29592ecc9b, 2026-11-05.

**The one failure.** `Tests/UI/test_console_composer_state_restyle.py::test_a_focus_change_restyles_the_bar_and_its_draft_text_only` (`:355`) was added on 2026-11-05 by fff461cab5 (TASK-36511.2), with review fixes c3e807f856 and 40ba1338a8. The UI Fast Lane runs the whole file.
- It failed once, in a 7-file `-n 4` run on the fu-r7-w7perf branch (2026-11-05 07:46, test line `:405` as on dev), with `AssertionError: focused=False`.
- After a blur, the bar's scoped restyle left `Static#console-command-visible-text` with a different computed `background` than a full `update_styles(composer)` gives. In the product that would show as a stale focus colour.

**Not reproduced on dev.** It passed 3 of 3 alone, 10 of 10 whole-file runs at load 14-40, and 3 of 3 with the failing run's file set at load 34-53. The PR author recorded 16 of 16 at load 50.

**Suspected window, not confirmed.**
- The draft text's style depends on the bar's `:focus-within` and on its state classes (`tldw_chatbook/css/features/_console.tcss:1055-1057`).
- `watch_has_focus` (`tldw_chatbook/Widgets/Console/console_composer_bar.py:2235-2251`) restyles the bar and `INTERACTION_STYLED_DESCENDANTS` as soon as `has_focus` flips.
- The class change comes later, in `on_blur` / `on_descendant_blur` (`:4433-4447`), through `_sync_interaction_classes` (`:2218-2233`) with `update=False`.
- The test's `moved()` loop (`:375-383`) can call `set_focus` again inside that gap.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Either the failure is reproduced (for example by forcing a focus bounce or delaying the blur handler) and its cause recorded, or 50 whole-file -n 4 runs at a recorded load of 30 or more stay green and the task closes on that evidence
- [ ] #2 If it reproduces, the fix makes the draft text's computed style after any focus change equal a full restyle's, and a test with the bounce forced shows it, red before the fix
<!-- AC:END -->
