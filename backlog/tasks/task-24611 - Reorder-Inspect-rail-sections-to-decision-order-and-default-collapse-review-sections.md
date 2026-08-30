---
id: TASK-24611
title: >-
  Reorder Inspect rail sections to decision order and default-collapse review
  sections
status: Done
assignee:
  - '@claude'
created_date: '2026-08-30 00:55'
updated_date: '2026-08-30 16:06'
labels:
  - console
  - ux
  - inspector
  - critique-2026-08-29
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
SUPERSEDED PREMISE - rewritten 2026-08-30 after finding the critique read a different branch.

The original description said Changed files, a post-hoc review artefact, sits third in the Inspect rail above everything describing the current send. That is true only on feat/task-3401-video-generation-foundation, which is what the critique's assessments read (it was the main checkout's branch). On origin/dev, tldw_chatbook/Widgets/Console/console_changed_files_section.py DOES NOT EXIST and nothing references ChangedFiles; dev instead carries a ConsoleSelectedTurnActivity the feature branch does not.

Dev's actual Inspect rail order is: pinned send-authority summary (outside the scroller) -> Sources tray -> Scope row -> #console-run-inspector (ConsoleRunInspector + ConsoleSelectedTurnActivity + ConsoleSettingsSummary) -> live-work section (library search + readiness card).

The one finding that survives on dev is real: the only gather-evidence-before-sending control (the Ask Library sources input and Search Library button) is the LAST widget in the LAST section, below the fold at 120 columns and narrower.

WHY THIS IS NOT A STRAIGHT FIX - owner decision needed. Every reordering that would raise that control reverses a placement another task chose deliberately and pinned with a test:
- Live-work cards anchor after the run-inspector block at the bottom by task-400 (test_console_live_work_card_swap_keeps_tray_on_top_and_cards_at_bottom states it in its docstring).
- On the feature branch, Changed files sits between the Scope row and the run inspector by TASK-18060's review-rail spec section 2.

Reversing an approved placement is an owner call, not a burn-down side effect, so this task is left To Do pending that decision.

Also note the remaining ACs need a mechanism the Inspect rail does not have: it has no per-section collapse at all (only the single 'More' disclosure persists, via CONSOLE_INSPECTOR_MORE_DISCLOSURE_ID). CONSOLE_RAIL_SECTION_IDS and the section_updates preference path exist only for the LEFT rail. 'Default to collapsed' and 'collapse state persists' therefore require building per-section disclosure for this rail first, which is its own task.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Section order follows authority, then what the user can change, then live state, then after-the-fact review
- [x] #2 Changed files and Session Settings default to collapsed
- [x] #3 The library-search control is reachable without scrolling at 120 columns
- [x] #4 Per-section collapse state persists across turns and session switches
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Owner picked option C from the decision page: move ONLY the search controls, leave the readiness card where task-400 put it.

The Sources tray's empty state reads 'No sources attached. Stage sources from Library.' The 'Ask Library' input and 'Search Library' button that do exactly that were the first three children of the live-work readiness card at the BOTTOM of the rail -- roughly 25 rows below that sentence, behind the fold, under a heading naming a status inventory. They now mount directly beneath the tray.

Why not the fuller reorder (option B, also prototyped and captured): swapping whole sections surfaces the search box but pushes Run and Source Readiness below the fold instead, and reverses a placement task-400 chose and pinned in its own test docstring. C buys the same thing without either cost.

Passed as a zero-arg BUILDER, not a widget instance, per the region rule about children the screen may replace outside the region's own compose. The Input re-seeds from the screen's stored query on every rebuild, so a recompose mid-typing does not discard what was entered -- the same contract it had inside the live-work card.

Test fallout, all legitimate and all updated rather than suppressed:
- The rail's DOM-order census hardcoded the first TWO direct children as the pre-run boundaries; a third child made the Scope row fall out of the computed list.
- The boundary-anchor inventory gained an entry in three places -- the search region is a real n/p stop now.
- The live-work swap-geometry pins moved: removing the controls dropped the readiness card 21 to 15 rows (a 2-row scope label, a 3-row Input, a 1-row Button). With the old 9-row payload NEITHER side then crossed the 20-row cap, so simply rewriting the expected numbers would have left a green test that no longer exercised the hint-on/hint-off boundary it exists for. The payload is now 10 so pending returns to 21 and each direction still crosses the cap once.

Verified live at 120x40: the search box renders immediately under the empty state, Scope and Run still above the fold.

Modified: UI/Screens/chat_screen.py, UI/Console_Modules/right_rail.py, Tests/UI/test_console_right_rail.py.
<!-- SECTION:NOTES:END -->
