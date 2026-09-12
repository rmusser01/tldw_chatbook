---
id: TASK-32072
title: >-
  First-run hand-off to Library: the wizard never mentions Library; Get started
  names steps with no controls
status: Done
assignee: []
created_date: '2026-09-08 18:26'
updated_date: '2026-09-08 20:04'
labels:
  - library
  - onboarding
  - ux
  - critique-8
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The setup Summary offers 'Explore Home' and never says where content lives; Get started reads '1 Add · 2 Find · 3 Use' but only the Add step has controls, so a first-timer has no path from 'I imported something' to 'find it' and 'use it in Console'. Evidence: critique #8 snapshot `.impeccable/critique/2026-09-08T16-05-41Z__tldw-chatbook-ui-screens-library-screen-py.md` (dev 1c022378cb, 2026-09-08, dual-agent live review at 235x52/100x30/60x24 on a fresh and a seeded profile). Register row 23.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The wizard Summary offers an action that lands in Library's Import (for example 'Add your first document')
- [x] #2 Each Get started step is a live control that unlocks in sequence (Import a file, Find it, Use it in Console)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Failing tests: the wizard Summary never mentions Library; Get started names three steps with one control.
2. Add 'Add your first document' to the Summary, exiting on TAB_LIBRARY; app.py maps that route to Library's Import canvas.
3. Replace the '1 Add - 2 Find - 3 Use' line with three sequenced controls plus one reason line.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
AC#1: the Summary's exits were 'Review provider setup / Explore Home / Review settings' -- the wizard never named Library at all. It now offers 'Add your first document', which finishes setup on exit_route TAB_LIBRARY; app.py maps that route to {LIBRARY_NAV_CONTEXT_INGEST: True}, so the destination is fixed in the app rather than trusted from the wizard's payload. Verified live on the fresh profile: pressed it on step 6 of 6 and landed on the Import media canvas with the compact Get started rail (caps/16, caps/17).

AC#2: '1 Add . 2 Find . 3 Use' is now three controls -- Import a file (routes to Import), Find it (routes to Search/RAG once any source has content), Use it in Console (stages the selected evidence through the existing _stage_library_rag_result_in_console once a search has results). A step that cannot run yet stays PRESSABLE (TASK-716: a disabled Button never emits Pressed, so its explanation would be unreachable), carries the blocked class, and states its reason plus the step that unlocks it -- both on the hint line under the strip and again as a toast if pressed. LibraryLandingCanvasState gained has_any_content/search_has_results; sync_state patches the strip in place, so the retained-widget promise the landing canvas exists for still holds. Verified live (caps/18).

Files: tldw_chatbook/UI/Wizards/FirstRunSetupWizard.py, tldw_chatbook/app.py, tldw_chatbook/Widgets/Library/library_entry_canvases.py, tldw_chatbook/UI/Screens/library_screen.py, Tests/UI/{test_library_crit8_polish_shell,test_library_shell,test_library_entry_compose_once}.py, Docs/User_Guide/library.md, Docs/User_Guide/First_Run_Setup.md.
<!-- SECTION:NOTES:END -->
