---
id: TASK-2951
title: >-
  task-1266 AC#4 is false on dev: TTSPlaygroundWidget was restored and never
  re-deleted
status: To Do
assignee: []
created_date: '2026-08-07 02:09'
updated_date: '2026-08-07 02:27'
labels:
  - ui
  - speech
  - tech-debt
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
task-1266 (Retire TTSPlaygroundWidget by porting its tests to the rebuilt pane) is marked Done on dev, with AC#4 stating 'TTSPlaygroundWidget is deleted'. It is not deleted: reconciliation commit f9d7e6269 ('refactor(speech): land the rebuild alongside dev's profile work, unmerged') explicitly restored dev's TTSPlaygroundWidget, and no later commit re-deleted it. The class still exists (tldw_chatbook/UI/STTS_Window.py, class TTSPlaygroundWidget(Widget)) with its full duplicate copy of the profile-preview/adoption copy paths, and 14 of its own 94 tests (Tests/UI/test_stts_playground_audio_cpp.py) are red at the current dev-equivalent state -- confirmed independently during voice-profiles slice 2 task 4's gate run (734 passed, 14 failed, same 14 test IDs task 3 already isolated as pre-existing).

Correction (slice 2 final whole-branch review): nothing in production instantiates TTSPlaygroundWidget any more -- a repo-wide grep for 'TTSPlaygroundWidget(' matches only the class definition itself. STTSWindow.watch_current_view -> _mount_view only ever mounts SpeechPlaygroundPane for the 'playground' view (STTS_Window.py ~:4661-4706); the widget class is unreachable through the live mount path. What remains wired to it are three TOLERANT lookups that name the type but never find it, because it is never mounted: stts_screen.py:163-170's _playground() (self.query_one(TTSPlaygroundWidget) inside a try/except, always returns None) -- which makes the Speech screen's g/r/x/p/s binding mirrors (action_generate_tts / action_random_text / action_clear_text / action_play_audio / action_stop_audio) permanent no-ops -- and two lookups in stts_events.py that fall back to the dead type after SpeechPlaygroundPane (~:1296-1319, audio-delivery routing, still carrying a now-stale "both playgrounds can be mounted" comment; ~:2099, provider-configuration-changed invalidation via self.app.query("SpeechPlaygroundPane, TTSPlaygroundWidget")). AC#2's own reference to "STTSScreen._redesign_view" is also stale -- no method by that name exists on STTSScreen today; the real mount path is STTSWindow._mount_view.

A task whose acceptance criteria describe a mount site that does not mount, alongside a class the board claims is deleted but isn't, still misrepresents the board -- this description is corrected so the wiring it names matches what the current tree actually does. The underlying problem AC#1-#5 address (a duplicate, partially-dead widget with red tests, still reachable by name from three call sites) is unchanged.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The second deletion pass is scoped: re-port anything from TTSPlaygroundWidget not already covered by SpeechPlaygroundPane, then delete TTSPlaygroundWidget and its dedicated test file
- [ ] #2 Every production mount site and event-lookup that still names TTSPlaygroundWidget (the playground branch in STTSWindow.watch_current_view / STTSScreen._redesign_view, and the stts_events.py delivery/invalidation lookups) is repointed at SpeechPlaygroundPane
- [ ] #3 No test coverage is lost: any assertion in test_stts_playground_audio_cpp.py not already duplicated against SpeechPlaygroundPane is ported before the file is deleted
- [ ] #4 The Speech screen is driven on a live run after deletion, not only under pytest
- [ ] #5 task-1266's own AC#4 is corrected (or this task is cross-referenced from it) so the board no longer claims a deletion that did not happen
<!-- AC:END -->
