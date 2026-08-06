---
id: TASK-2453
title: >-
  Roleplay Voice & Speech select silently refuses to assign any unverified
  profile
status: Done
assignee:
  - '@claude'
created_date: '2026-08-05 04:50'
updated_date: '2026-08-05 05:43'
labels: []
dependencies:
  - TASK-2450
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Live verification of task-2450 (voice profiles slice 1) found that the Roleplay character editor's Voice & Speech profile Select (Widgets/Persona_Widgets/personas_character_tts_widget.py, _profile_changed) only accepts an assignment when the chosen profile's availability is exactly 'available'; anything else (including the slice's own interim 'unverified' classification for every legacy-provider profile) is silently restored to the prior selection with no error shown. Since slice 1 always classifies legacy-provider profiles as unverified (never available, pending a later no-catalog state), this widget currently makes it impossible to assign any of the six newly supported providers to a character through the real UI -- the only affordance for creating an assignment in the live app. The backend (TTSProfileService.set_assignment, the character resolver) accepts the assignment correctly when called directly; only this client-side widget gate blocks it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Selecting an unverified legacy-provider profile in the Roleplay Voice & Speech Select persists the assignment (posts CharacterTTSActionRequested with the chosen profile_id) instead of silently reverting to the prior value
- [x] #2 The refusal that remains appropriate -- assigning a profile the service has structurally rejected, e.g. a genuinely unavailable/broken one -- still refuses (unchanged: the pre-existing silent revert, since no refusal message existed to narrow -- confirmed by grep before implementing; adding a new toast was not requested by the controller ruling and was left out of scope)
- [x] #3 A regression test drives the real Select.Changed path (not the service layer directly) for an unverified profile and asserts the resulting CharacterTTSActionRequested carries the chosen profile_id
- [x] #4 Existing audio_cpp assignment behavior for 'available' profiles is unchanged
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed in-slice (task-6b, controller ruling). Found TWO independent gates, both stale for the same reason (written when available/unavailable was the whole vocabulary, never taught 'unverified'): (1) the widget's own client-side gate in personas_character_tts_widget.py::_profile_changed ('if option.availability != "available": revert'), and (2) a SEPARATE, independent screen-side gate in personas_screen.py::_character_tts_assignment_worker ('if tokens[1].state != "available": return', silent, no log, no error) that fixing only the widget does not reach -- live verification proved this the hard way: after fixing (1) alone, a real assignment click updated the Select's own value locally but never persisted (confirmed by leaving and returning to the screen), because the worker below the widget silently dropped it. TDD for both: test_character_tts_widget_accepts_unverified_profile_assignment_without_laundering_it (widget layer: unverified now posts the assign action and stays visibly marked '· unverified' in the option row; unavailable still refused; assigned-but-unverified keeps the ordinary 'Edit' label, not 'Repair', which is reserved for genuinely unavailable) and test_character_tts_assignment_worker_accepts_unverified_profile + test_character_tts_assignment_worker_still_refuses_unavailable_profile (worker layer: set_assignment is actually called for unverified, still skipped for unavailable). All three RED against pre-fix code, confirmed. Fix: both gates changed from '!= available' to '== unavailable'; widget's Edit/Repair label logic changed from 'available-driven' to 'broken (unavailable)-driven' so an unverified assignment is never presented as needing repair. Mutation-verified: reverting either gate independently fails its own test; reverting the label logic is caught by the PRE-EXISTING unavailable-Repair test. Full Tests/UI/test_personas_workbench.py (309 passed, was 307) stayed green. Live re-verified end to end against a real OpenAI account: opened Roleplay > Default Assistant > Voice & Speech, selected the real unverified profile from task-2452's live run, watched the status line update to 'Live-verify openai (task 6b) . Unverified . Used by 1 character. Refresh or repair the profile; the assignment is preserved.' with the Edit button (not Repair), confirmed the assignment survived navigating away and back (real persistence, not an optimistic local render), then clicked Console's speak action on the character's message and heard/observed a real OpenAI TTS call complete and real audio play (afplay) through the assigned profile. See task-2450's task-6b report for the full trace.
<!-- SECTION:NOTES:END -->
