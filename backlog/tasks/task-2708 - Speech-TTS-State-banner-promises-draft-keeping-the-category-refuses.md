---
id: TASK-2708
title: 'Speech & TTS: State banner promises "switching categories keeps this draft" but leaving forces save/discard'
status: Done
assignee: []
created_date: '2026-08-06'
labels: [settings, speech, ui, honesty]
dependencies: []
---

## Description (the why)

Speech & TTS is in `GUIDED_SETTINGS_MUTATION_CATEGORIES`, so when its panel
reports a dirty draft (`handle_speech_tts_draft_modified` seeds
`_settings_drafts[SPEECH_TTS]`), the shared State banner renders the standard
dirty text: **"State: Unsaved changes | Save (s) or Revert (r) — switching
categories keeps this draft."** (`settings_screen.py`,
`_category_state_banner`).

But Speech & TTS is the one guided category where that promise is false:
`_select_category` intercepts any category switch away from SPEECH_TTS with a
dirty panel and runs `_confirm_speech_tts_category_leave` →
`SpeechTTSSettingsPanel.confirm_leave()`, which raises the "Unsaved global
Speech & TTS settings — Save these application-wide changes before
continuing, or discard them?" modal (**Cancel** / **Discard and continue** /
**Save and continue**) and then **clears the draft cache**
(`_clear_speech_tts_draft_cache`). The draft is resolved, never kept; the
category also never carries the rail's `*` marker across a switch the way
the banner's sibling categories do.

Found during the G4 user-guide refresh (dev @ e7b9ebabd, 2026-08-06). The
guide documents the mismatch as a quirk citing this task
(`Docs/User_Guide/settings.md` ▸ Quirks).

## Acceptance Criteria (the what)

- [x] While the Speech & TTS draft is dirty, the State banner's dirty text
      describes what actually happens on leave (the save/discard prompt),
      instead of claiming the draft is kept.
- [x] The other five guided categories keep their existing
      "switching categories keeps this draft." wording.
- [x] A test pins the Speech & TTS dirty-banner wording so the shared text
      cannot silently return.

## Closeout audit 2026-10-11

Verdict: DONE (verified against origin/dev @ a3c47c6c72).

Evidence:
- `tldw_chatbook/UI/Screens/settings_screen.py:9986-9991` — `_category_state_banner_text` has an explicit SPEECH_TTS branch (comment cites task-2708): "State: {unsaved} | Leaving Speech & TTS resolves this draft: save or discard first." The false draft-kept promise is gone; other guided categories keep the shared wording (the special case is scoped to SPEECH_TTS and ADVANCED_CONFIG).
- Wording pinned by `Tests/UI/test_settings_configuration_hub.py:13145-13162` (docstring cites task-2708).
- Landed in `4ef3ef8186` ("feat(settings): speech & TTS UX fixes from the 2026-09-11 critique"). All three ACs ticked.
