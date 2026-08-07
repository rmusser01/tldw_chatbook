---
id: TASK-2951
title: >-
  task-1266 AC#4 is false on dev: TTSPlaygroundWidget was restored and never
  re-deleted
status: To Do
assignee: []
created_date: '2026-08-07 02:09'
labels:
  - ui
  - speech
  - tech-debt
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
task-1266 (Retire TTSPlaygroundWidget by porting its tests to the rebuilt pane) is marked Done on dev, with AC#4 stating 'TTSPlaygroundWidget is deleted'. It is not deleted: reconciliation commit f9d7e6269 ('refactor(speech): land the rebuild alongside dev's profile work, unmerged') explicitly restored dev's TTSPlaygroundWidget so it could keep owning the playground view while the two designs were reconciled, and no later commit re-deleted it. The class still exists (tldw_chatbook/UI/STTS_Window.py, class TTSPlaygroundWidget(Widget)) with its full duplicate copy of the profile-preview/adoption copy paths, and 14 of its own 94 tests (Tests/UI/test_stts_playground_audio_cpp.py) are red at the current dev-equivalent state -- confirmed independently during voice-profiles slice 2 task 4's gate run (734 passed, 14 failed, same 14 test IDs task 3 already isolated as pre-existing). A task whose acceptance criteria are false on dev misrepresents the board.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The second deletion pass is scoped: re-port anything from TTSPlaygroundWidget not already covered by SpeechPlaygroundPane, then delete TTSPlaygroundWidget and its dedicated test file
- [ ] #2 Every production mount site and event-lookup that still names TTSPlaygroundWidget (the playground branch in STTSWindow.watch_current_view / STTSScreen._redesign_view, and the stts_events.py delivery/invalidation lookups) is repointed at SpeechPlaygroundPane
- [ ] #3 No test coverage is lost: any assertion in test_stts_playground_audio_cpp.py not already duplicated against SpeechPlaygroundPane is ported before the file is deleted
- [ ] #4 The Speech screen is driven on a live run after deletion, not only under pytest
- [ ] #5 task-1266's own AC#4 is corrected (or this task is cross-referenced from it) so the board no longer claims a deletion that did not happen
<!-- AC:END -->
