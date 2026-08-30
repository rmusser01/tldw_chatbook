---
id: TASK-24611
title: >-
  Reorder Inspect rail sections to decision order and default-collapse review
  sections
status: To Do
assignee: []
created_date: '2026-08-30 00:55'
updated_date: '2026-08-30 02:58'
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
- [ ] #1 Section order follows authority, then what the user can change, then live state, then after-the-fact review
- [ ] #2 Changed files and Session Settings default to collapsed
- [ ] #3 The library-search control is reachable without scrolling at 120 columns
- [ ] #4 Per-section collapse state persists across turns and session switches
<!-- AC:END -->
