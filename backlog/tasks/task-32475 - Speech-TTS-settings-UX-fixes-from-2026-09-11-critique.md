---
id: TASK-32475
title: Speech & TTS settings UX fixes from 2026-09-11 critique
status: Done
assignee: []
created_date: '2026-09-11 23:53'
updated_date: '2026-09-12 01:02'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the P1+P2 fixes from the impeccable critique of the Speech & TTS settings sub-screen: Kokoro first-run onboarding rows, known-voice/model pickers with custom escape hatch and Speech Lab bridge, governance copy compression, leave-banner contradiction fix, id-leak and casing cleanups, Chatterbox/Higgs form grouping, restore preview.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Kokoro/Chatterbox/Higgs forms surface dependency status and model guidance inline
- [x] #2 Model and Voice value for legacy providers are Selects of known IDs with Custom escape hatch
- [x] #3 First viewport shows editable controls (banner compressed to 2 lines, single Open Speech Lab button)
- [x] #4 Speech & TTS leave banner no longer claims drafts persist across category switches
- [x] #5 Chatterbox/Higgs forms grouped into collapsible sections
- [x] #6 Restore states its scope before/after acting
- [x] #7 Targeted tests updated and passing
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Panel: compress banner, drop top Lab button, remove task static\n2. Panel: legacy model/voice Selects + custom modal + browse-voices bridge\n3. Panel: local dependency + model guidance rows; casing; speed label; save payoff; restore tooltip/summary\n4. Panel: Collapsible grouping for Chatterbox/Higgs\n5. settings_screen: banner special-case, alias and description id-leak fixes\n6. Update affected tests; run targeted suite\n7. Update docs; live verify
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Approach: five critique-driven fix areas landed in one pass over
  `speech_tts_settings_panel.py` plus three small `settings_screen.py` edits,
  honoring the user's "preserve, compress" tone decision for governance copy.
- Onboarding: `_local_dependency_row()` surfaces the same
  `speech_local_dependency_availability` fact the Scope inspector reports,
  inline at the top of the Kokoro/Chatterbox/Higgs forms, plus a Kokoro
  model-file guidance row naming `kokoro-v0_19.onnx` and the docs path; path
  placeholders now carry the real filenames.
- Pickers: legacy Model/Voice values moved from free-text Inputs to Selects
  built from `LEGACY_MODELS`/`LEGACY_MODEL_LABELS`/`LEGACY_VOICE_OPTIONS`
  (labeled, e.g. "Heart (US Female)"), with a `Custom…` sentinel that opens a
  `_CustomIdModal` free-text editor, an unknown saved value kept selectable as
  "(custom)" (the never-drop-a-saved-value precedent from the realtime
  provider select and audio.cpp exact choices), and a `Browse in Speech Lab`
  button beside Voice value that stages the default provider. Collection
  guards ensure the sentinel can never persist.
- Distill: banner compressed to two lines (fuller text preserved as tooltip),
  the banner's `#settings-speech-open-lab` button removed (single Lab action
  with Save at the bottom; handler keeps serving the bottom + audio.cpp
  handoff buttons), and the redundant "Task: set up …" status static dropped
  (kept "Current status: …").
- Clarify: SPEECH_TTS leave-banner special case resolves task-2708's
  contradiction; `audio_cpp` raw id removed from the category summary and
  inspector text but deliberately kept in the search vocabulary (tests assert
  power users grep by config id); row labels sentence-cased; Speed range
  moved into the label; save payoff now names the next step; Restore got a
  scope tooltip and a summary result line (no modal — draft-only action).
- Layout: Chatterbox (Compute and generation / Voice and processing /
  Streaming) and Higgs (Model and voice / Compute / Generation) grouped into
  Collapsibles with unchanged field ids, so collection and tests are
  unaffected.
- Tests: updated 8 sites where tests drove the old Inputs, added 6 new tests
  (picker options + draft dirty, unknown-value "(custom)" option, custom
  modal flow, browse-voices navigation, dependency/guidance rows, SPEECH_TTS
  banner). Targeted suite green: 137 + 191 + 128 + 104 passed across the
  speech panel/policy/scoped/ownership/contracts/preferences/category-sweep
  files; one pre-existing timing flake in `test_audio_cpp_model_library_
  handoff.py` passes solo and is untouched by this change.
- Live-verified on an isolated scratch profile (TLDW_CONFIG_PATH + scratch
  data_dir) at 235x52 and 130-row captures: banner two lines with controls in
  the first viewport, "Kokoro 82M" model dropdown, dependency and
  model-guidance rows, filename-bearing placeholders, no `audio_cpp` leak.
- Environment note: the working tree could not import `tldw_chatbook.Chat`
  (pre-existing uncommitted WIP imports `Chat.console_appearance`, which was
  missing from this checkout). Restored the file byte-identical from the
  sibling worktrees (same md5 in all of them) purely to make verification
  possible; it is the user's in-flight work, not part of this task's changes.
- Docs: `Docs/User_Guide/settings.md` Speech & TTS section rewritten against
  the new UI; the task-2708 Quirks entry removed (fixed);
  `TTS_MODULE_GUIDE.md` "S/TT/S tab" drift renamed to Speech Lab.
- Modified files: `Widgets/Settings_Widgets/speech_tts_settings_panel.py`,
  `UI/Screens/settings_screen.py`, `Tests/UI/test_settings_speech_tts_
  panel.py`, `Tests/UI/test_settings_configuration_hub.py`,
  `Tests/UI/test_legacy_tts_voice_policy.py`,
  `Tests/UI/test_speech_settings_panel_scoped_updates.py`,
  `Docs/User_Guide/settings.md`, `Docs/Development/TTS/TTS_MODULE_GUIDE.md`.
- ADR required: no — UI affordance and copy changes within existing
  boundaries; no schema, ownership, or service-contract change.
<!-- SECTION:NOTES:END -->
