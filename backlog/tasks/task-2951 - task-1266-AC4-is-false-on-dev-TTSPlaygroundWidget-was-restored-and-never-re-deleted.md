---
id: TASK-2951
title: >-
  task-1266 AC#4 is false on dev: TTSPlaygroundWidget was restored and never
  re-deleted
status: Done
assignee:
  - '@claude'
created_date: '2026-08-07 02:09'
updated_date: '2026-08-07 04:27'
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
- [x] #1 The second deletion pass is scoped: re-port anything from TTSPlaygroundWidget not already covered by SpeechPlaygroundPane, then delete TTSPlaygroundWidget and its dedicated test file
- [x] #2 Every production mount site and event-lookup that still names TTSPlaygroundWidget (the playground branch in STTSWindow.watch_current_view / STTSScreen._redesign_view, and the stts_events.py delivery/invalidation lookups) is repointed at SpeechPlaygroundPane
- [x] #3 No test coverage is lost: any assertion in test_stts_playground_audio_cpp.py not already duplicated against SpeechPlaygroundPane is ported before the file is deleted
- [x] #4 The Speech screen is driven on a live run after deletion, not only under pytest
- [x] #5 task-1266's own AC#4 is corrected (or this task is cross-referenced from it) so the board no longer claims a deletion that did not happen
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read task-1266 history + current STTS_Window.py/stts_screen.py/stts_events.py to confirm the corrected wiring facts in the description.
2. Diff Tests/UI/test_stts_playground_audio_cpp.py's 94 tests (14 red/dead, 80 green) against SpeechPlaygroundPane's existing test suite via a research subagent to find genuine coverage gaps vs already-duplicated coverage.
3. Fix stts_screen.py's _playground() to query SpeechPlaygroundPane instead of the dead TTSPlaygroundWidget, TDD'd (RED test proving the g/r/x/p/s mirror is currently a no-op, then the fix, then GREEN).
4. Repoint stts_events.py's two tolerant TTSPlaygroundWidget lookups (_mounted_playground delivery routing, on_stts_provider_configuration_changed invalidation) at SpeechPlaygroundPane only.
5. Extract shared test fixtures (FakeTTSService, _resolved, _wait_until, _profile_preset, _native_profile_artifact) used by 5 other pane-test files out of test_stts_playground_audio_cpp.py into a new Tests/UI/speech_playground_fixtures.py so those files survive the widget test file's deletion.
6. Port Tests/TTS/test_stts_audio_cpp_generation.py's 6 widget-mounting tests onto SpeechPlaygroundPane directly (mechanical recipe: provider= constructor kwarg, _tts_service_factory/_check_higgs_installation monkeypatches instead of STTS_Window module patches).
7. Dispatch a second subagent to port the remaining ~77 GAP tests from test_stts_playground_audio_cpp.py onto SpeechPlaygroundPane (pure-function tests into test_stts_playground_catalog.py, App-mounted tests into a new test_speech_playground_pane_lifecycle.py), classifying any RED result as either a mechanical fixture difference (fix the test) or a genuine pane defect (do NOT silently fix production code -- flag for review).
8. Delete TTSPlaygroundWidget (and its now-dead module-level _PROFILE_RESULT_STALE_COPY duplicate) from STTS_Window.py; delete Tests/UI/test_stts_playground_audio_cpp.py once its coverage is confirmed ported; clean up unused imports via ruff.
9. Blast-radius grep every remaining TTSPlaygroundWidget reference (code/tests/comments) and rewrite to describe the retired widget without naming the deleted class, except in backlog/ task files and lessons docs.
10. Drive the Speech screen live (tmux) to confirm the pane mounts, the g/r/x/p/s mirror bindings work from the landed state, and nothing regressed.
11. Run the full gate list, ruff on touched files, and a repo-wide --collect-only sweep; update task-1266 cross-reference if needed; mark task-2951 Done.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Second deletion pass complete. TTSPlaygroundWidget (3225 lines) and its dead
_PROFILE_RESULT_STALE_COPY duplicate deleted from STTS_Window.py; nothing
restores it this time since the two tolerant lookups that named it
(stts_screen.py _playground(), stts_events.py delivery + invalidation) are
repointed at SpeechPlaygroundPane only. Its 80 non-dead tests were diffed
against the pane's own test suite before deletion (research subagent, full
read + cross-reference): 3 already duplicated, 77 genuine gaps, all ported
(2 pure-function tests into test_stts_playground_catalog.py, 75 into a new
test_speech_playground_pane_lifecycle.py). The 14 red tests documented
dead-widget-only failures and were dropped with no port (they proved
nothing about live code). Zero silent coverage loss.

Porting surfaced one real, unrelated pane defect (SpeechCatalogMixin marks
a provider "stale" on ANY non-fresh catalog load, not only a load following
a real config change -- absent from the retired widget's own success path,
confirmed via git show HEAD diff) -- left as 2 xfail(strict=True), filed as
task-2970, not fixed here (out of this task's scope).

Full report: .task-2951-report.md (gitignored, worktree-local).

The binding-mirror fix (AC#2) was TDD'd: two new tests in
`Tests/UI/test_speech_shortcuts.py` land focus on the nav rail (matching
the screen's "landed state") and press 'r'/'x', RED against the pre-fix
`_playground()` (confirmed the mirror was a genuine no-op), GREEN after.
Live-verified in tmux with a scratch config afterward: pressed 'x' from the
landed state, the pane's text area cleared; pressed 'r' after re-landing
focus, the text area was seeded with a random sample line -- the mirror
reaches the real mounted pane in the running app, not only under pytest
(AC#4).

`Tests/UI/test_stts_playground_audio_cpp.py` doubled as a shared-fixtures
module for 5 other test files (`FakeTTSService`/`_resolved`/`_wait_until`/
`_profile_preset`/`_native_profile_artifact`); extracted into a new
`Tests/UI/speech_playground_fixtures.py` before deletion so those 5 files
survive it. `Tests/TTS/test_stts_audio_cpp_generation.py`'s 6 tests that
mounted the widget directly were ported onto the pane (provider= kwarg,
`_tts_service_factory`/`_check_higgs_installation` monkeypatches instead of
`STTS_Window` module patches); one needed a real fix -- its expected
`#audio-player-status` copy was the dead widget's wording
("WAV audio ready to play"); updated to the pane's actual, already-tested
copy ("Ready · WAV").

Files touched: `tldw_chatbook/UI/STTS_Window.py` (-3273/+3, ruff-cleaned 38
now-dead imports), `tldw_chatbook/Event_Handlers/STTS_Events/stts_events.py`,
`tldw_chatbook/UI/Screens/stts_screen.py`,
`tldw_chatbook/UI/Speech/{speech_catalog_mixin,speech_playground_model,
speech_profile_mixin}.py` (comment cleanup only, blast-radius pass);
`Tests/UI/test_stts_playground_audio_cpp.py` deleted (-3121);
`Tests/UI/speech_playground_fixtures.py` (new, 277 lines),
`Tests/UI/test_speech_playground_pane_lifecycle.py` (new, 2501 lines),
`Tests/UI/test_stts_playground_catalog.py` (+68); six other Speech test
files updated for the fixture-import repoint or blast-radius comment
cleanup. Dated historical docs under `Docs/superpowers/{plans,specs,qa}/`
and `Docs/Development/TTS/` that name the retired widget were deliberately
left alone (point-in-time records, same treatment as backlog/lessons docs).

Gates: 297 passed / 2 xfail (expected) / 0 failed across the full targeted
Speech/TTS test surface; repo-wide `--collect-only` clean (31717 collected,
0 errors); ruff check + format clean on all touched files. One PRE-EXISTING
failure found and left alone --
`test_speech_tts_settings_ownership_closeout.py::
test_first_time_audio_cpp_setup_lab_generation_and_console_handoff` --
confirmed by running it against a throwaway worktree at the unmodified base
commit (517e1b200), where it also fails; unrelated to this task.
<!-- SECTION:NOTES:END -->
