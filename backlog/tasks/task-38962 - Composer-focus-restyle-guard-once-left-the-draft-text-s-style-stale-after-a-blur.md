---
id: TASK-38962
title: "Composer focus-restyle guard once left the draft text's style stale after a blur"
status: Done
assignee:
- '@claude'
created_date: 2026-11-05 21:35
updated_date: 2026-11-07 07:28
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
- [x] #1 Either the failure is reproduced (for example by forcing a focus bounce or delaying the blur handler) and its cause recorded, or 50 whole-file -n 4 runs at a recorded load of 30 or more stay green and the task closes on that evidence
- [x] #2 If it reproduces, the fix makes the draft text's computed style after any focus change equal a full restyle's, and a test with the bounce forced shows it, red before the fix
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Try to force the stale draft-text style with focus bounces queued inside one tick (blur/focus/blur, focus-blur chains, a descendant hop), comparing targeted and full restyles. Use a scratch test that is not committed.
2. If it does not reproduce, run the whole file 50x at -n 4 at load 30 or more and close on that evidence (AC#1's second branch).
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Closed on AC#1's second branch: it did not reproduce, and 50 whole-file runs stayed green. No code change. Base 8b762b52fa, 2026-11-07.

**Attempt to reproduce.** A scratch test (not committed) used the file's own helpers. It queued focus bounces inside one tick with `screen.set_focus`: bar→outside→bar→outside; a longer alternation that ends outside; outside→bar; bar→outside→a focusable descendant→outside; descendant→outside. After focus settled, it compared each composer node's computed rules with a full `update_styles(composer)` 30 times, 10 ms apart. 20 of 20 runs (5 patterns ×4) matched at load 44-53.

What the reading showed: Textual sets `screen.focused` synchronously in `set_focus` and flips `has_focus` only when the widget handles its `Focus`/`Blur` message. The bar's `watch_has_focus` restyle and `on_blur`'s class sync both read `:focus-within` live from `screen.focused`, so a queued Blur/Focus/Blur restyles against the final focus every time. No window was found where the draft text keeps a stale style.

**50 whole-file runs at `-n 4` (AC#1).** `Tests/UI/test_console_composer_state_restyle.py`, 7 tests each: 50 of 50 green (350 test runs), load 45-190 (one-minute average at each start), 24-61 s per run. The file stays where it is (whole file in the UI census).

**AC#2.** Not applicable: it did not reproduce, so there was nothing to fix.
<!-- SECTION:NOTES:END -->
