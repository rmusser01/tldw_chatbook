---
id: TASK-2453
title: >-
  Roleplay Voice & Speech select silently refuses to assign any unverified
  profile
status: To Do
assignee: []
created_date: '2026-08-05 04:50'
updated_date: '2026-08-05 04:50'
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
- [ ] #1 Selecting an unverified legacy-provider profile in the Roleplay Voice & Speech Select persists the assignment (posts CharacterTTSActionRequested with the chosen profile_id) instead of silently reverting to the prior value
- [ ] #2 The refusal that remains appropriate -- assigning a profile the service has structurally rejected, e.g. a genuinely unavailable/broken one -- still refuses, with a visible reason rather than a silent revert
- [ ] #3 A regression test drives the real Select.Changed path (not the service layer directly) for an unverified profile and asserts the resulting CharacterTTSActionRequested carries the chosen profile_id
- [ ] #4 Existing audio_cpp assignment behavior for 'available' profiles is unchanged
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Found live, 2026-08-04, during task-2450's Task 6 live verification: opening Roleplay > Characters > Default Assistant > Voice & Speech and selecting the (real, live-created) 'Live-verify OpenAI (task 6) . unverified' option via the Select overlay reverted to 'Use global default' every time, with no toast or error. Traced to Widgets/Persona_Widgets/personas_character_tts_widget.py::_profile_changed: 'if option is None or option.availability != "available": self._restore_selected_value(); return'. This file is untouched by the voice-profiles slice-1 branch (confirmed via git diff ab9105c9d..HEAD --name-only), so the gate is pre-existing, written when audio.cpp's 'available'/'unavailable' pair was the only vocabulary -- it was never taught the slice's new 'unverified' state. Worked around for task-2450's own live verification by calling the real TTSProfileService.set_assignment directly (bypassing only this widget gate); reloading the live app then showed the real assignment correctly, with the exact 'Unverified . Used by 1 character. Refresh or repair the profile; the assignment is preserved.' copy this slice's design intended -- confirming the SERVICE layer is correct and only this client-side gate is stale. See task-2450's Task 6 report for the full trace.
<!-- SECTION:NOTES:END -->
