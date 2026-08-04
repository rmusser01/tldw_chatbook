---
id: TASK-2233
title: 'Roleplay: hide the voice-profile checkbox when no profile is assigned (R2)'
status: Done
assignee: []
created_date: '2026-08-04 16:18'
updated_date: '2026-08-04 18:51'
labels:
  - ux-review
  - roleplay
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The disabled 'Include assigned voice profile' checkbox renders as an unreadable dark smear exactly where the eye lands after the primary CTA. Disabled-with-reason is right for applicable actions; this one is not applicable when nothing is assigned. Post-fix re-review P2. See Docs/superpowers/qa/2026-08-03-library-roleplay-mcp-ux-review.md.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Checkbox is hidden (display False) when no voice profile is assigned
- [x] #2 It reappears (enabled or disabled-with-reason) when a profile is assigned
- [x] #3 Tests updated
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no - visibility change for one inspector checkbox; no behavior, boundary, or schema change. ADR path: N/A.
1. In PersonasInspectorPane._apply_action_state, gate the voice-profile checkbox's display on _tts_export_available (in addition to the existing character-kind gate): hidden when no profile is assigned, shown when one is (the existing enabled/disabled-with-reason logic and the F-041 legibility CSS keep covering the shown case).
2. Update the F-041 legibility test to make the checkbox visible first (set_tts_export_available(True)); add tests: hidden for a character with no assignment, reappears when the assignment resolves, hidden again after it clears; strengthen the workbench export test with display assertions.
3. Run inspector-pane + workbench TTS tests + ruff; commit code + task file.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The 'Include assigned voice profile' checkbox now renders only when the selected character HAS a voice profile assigned. ADR: none (visibility change only, stated in the plan).

Approach: PersonasInspectorPane._apply_action_state gates the checkbox's display on _tts_export_available in addition to the existing character-kind gate (display = kind-is-character AND available), and compose initializes it hidden so the flag is deterministic before the first state application. The shown case is unchanged: enabled when exportable, otherwise disabled-with-reason, and the F-041 legibility CSS (full opacity, dimmed label, surfaced glyph box) still applies - now only to states where the control is actually applicable.

Tests (TDD red first): the F-041 legibility test assigns a profile before asserting; new test_tts_checkbox_hidden_until_a_profile_is_assigned covers hidden-without-assignment, reappear-on-assignment, re-hide-on-clear, and the non-character kind gate; the workbench export test gained display assertions. test_personas_inspector_pane.py: 30 passed; test_personas_workbench.py: 306 passed; ruff clean.

Files: tldw_chatbook/Widgets/Persona_Widgets/personas_inspector_pane.py, Tests/UI/test_personas_inspector_pane.py, Tests/UI/test_personas_workbench.py. The user-guide checkbox visibility line (Docs/User_Guide/roleplay-chat-dictionaries/characters-and-personas.md) ships with task-2234's commit - same file as its toggle-rename docs.
<!-- SECTION:NOTES:END -->
