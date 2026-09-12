---
id: TASK-32015
title: 'Voice docs, tooltips, and hands-free settings rows'
status: Done
assignee:
  - '@robert'
created_date: '2026-09-11 23:51'
updated_date: '2026-09-12 01:18'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The hands-free loop and Speak replies are documented as one table row each; wake-word grammar, degraded mode, Linux setup (extras, portaudio-devel for pyaudio, player packages vs WAV), and the loop's tuning keys (dictation.handsfree_send_delay_seconds, dictation.acoustic_barge_in, dictation.command_prefix) are invisible at the point of use and config-only. Control-bar tooltips do not state the Speak replies/hands-free relationship or how to interrupt a spoken reply.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] New user-guide page for voice/hands-free covering loop mechanics, wake-word grammar, degraded mode, Linux setup matrix, and tuning keys, linked from console.md — `Docs/User_Guide/console/voice-and-hands-free.md`; linked from console.md's child-page list and both control-bar table rows; settings.md's Realtime engine row cross-links it
- [x] Speak-replies and hands-free tooltips state the per-conversation nature, the relationship between the two features, and how to interrupt a spoken reply — module constants `AUTO_SPEAK_SWITCH_TOOLTIP` / `HANDS_FREE_SWITCH_TOOLTIP` in `Widgets/Console/console_speech_controls.py` (the switches' actual home on main), pinned by `test_voice_switch_tooltips_state_relationship_and_interrupts`
- [x] Settings Speech panel exposes handsfree_send_delay_seconds and acoustic_barge_in with validation, round-trip verified by tests — two rows in the Realtime engine section; `test_handsfree_pipeline_tuning_fields_render_and_save` (render + save round-trip) and `test_handsfree_blank_send_delay_deletes_key_and_invalid_refuses_save` (blank deletes the key; 0/invalid refuses the whole Save with an inline error)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Tooltips: Speak replies + Hands-free switches state relationship, per-conversation scope, interrupt/exit paths
2. Settings: handsfree_send_delay_seconds + acoustic_barge_in rows in Speech panel hands-free section (TDD via existing panel test patterns)
3. Docs: new User Guide voice/hands-free page (loop mechanics, wake-word grammar, degraded mode, Linux setup matrix, tuning keys) + links
4. Targeted test runs; task notes
<!-- SECTION:PLAN:END -->

## Implementation Notes

- **Tooltips.** Hoisted to module constants in `console_speech_controls.py` (NOT the control bar -- on main the switches live in `ConsoleSpeechControls` beside the Workbench status, a fact worth knowing before editing them). Auto-speak's tooltip names the per-conversation scope and that hands-free owns reply speech while running; hands-free's names Ctrl+Shift+H, type-to-interrupt, and Esc-to-exit. The paused-state tooltip path reuses the same constant.
- **Settings rows.** Followed the section's established contracts exactly: the send delay is an *optional number* (blank = unset -> Save deletes the key so the readers' 1.5s default keeps winning; a positive-only bound via a new `exclusive_low` flag on `_validated_optional_number`, because a written 0 would make the readers warn-and-fallback every boot), and barge-in is an explicit Switch, always written. Draft fields live in `speech_tts_panel_types.py`'s `_RealtimeSettingsDraft` (snapshot + validator extended); whole-number delays persist as ints (the idle-minutes M5 rule). Two existing exact-equality pins updated for the new dictation keys.
- **Docs.** New page covers the loop phase table, the full wake-word grammar, degraded-mode truthfulness (nothing auto-sends; commands cannot fire mid-capture), consent/re-ask behavior, the Retry/Resume buttons, and a platform setup matrix (extras, `dnf install portaudio-devel gcc` for pyaudio, `dnf install mpv` vs WAV output, pw-play/paplay WAV/FLAC coverage) plus a troubleshooting table. It documents TASK-32013/32014's new behaviors (adaptive WAV fallback, silence warnings) as user-visible behavior.
- **Tests.** Two exact-equality payload pins updated; two new panel tests + one tooltip pin, all watched RED. Note for reviewers running locally: `Tests/UI/test_settings_speech_tts_panel.py` has 5 pre-existing failures on pristine `origin/main` in this environment (layout/machine-dependent: `test_credential_editor_is_a_bounded_modal_with_real_styles`, `test_production_settings_actions_...`, `test_normal_panel_actions_...`, `test_production_bundle_...`, `test_speech_save_and_revert_...`); after this change the suite's failure set is IDENTICAL to that baseline (222 passed). Bumping those tests' fixed viewports for the taller panel did not fix them on this machine, so the size bumps were reverted to keep the diff honest.
- **ADR required: no** -- settings rows follow the existing realtime-section draft/payload architecture; docs and copy only.
