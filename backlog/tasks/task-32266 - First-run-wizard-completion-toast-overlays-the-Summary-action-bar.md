---
id: TASK-32266
title: First-run wizard completion toast overlays the Summary action bar
status: Done
assignee: []
created_date: '2026-09-10 18:05'
updated_date: '2026-09-11 16:07'
labels:
  - library
  - notes
  - ux
  - critique-notes-2026-09
  - wizard
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
On wizard completion the toast paints over the Summary screen's action bar, covering the buttons the Summary exists to offer -- including the "Write your first note" hand-off (task-32245), which is the button the wizard was changed to add.

Evidence: Library ▸ Notes critique snapshot `.impeccable/critique/2026-09-10T17-32-16Z__w-chatbook-widgets-library-library-notes-canvas-py.md` (dev e6cb464239, 2026-09-10, dual-assessor live review at 235x52 and 100x30/60x24 on two fresh profiles and one seeded profile with a git-backed 71-file Obsidian vault fixture; the reconciling parent re-tested every disagreement on a third pair of profiles and traced each surviving cause to code). Captures under the session scratchpad `notes-crit2/{C,D,R}/caps/`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The completion toast does not cover the Summary action bar
- [x] #2 Covered by a test or a capture at 235x52
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce live on a fresh scratch profile at 235x52 and 100x30 (walk Quick track to Summary).
2. Trace the toast to its emitter.
3. RED test on the real route.
4. Fix at the emitter.
5. GREEN test + live captures at both sizes.
6. Guide stamp, task hygiene.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Cause (PROVEN, reproduced live on two fresh scratch profiles at 235x52 and 100x30): the toast is "Settings saved successfully!", raised by the app-level STTS handler (`stts_events.py` `_notify_settings_publication`), not by the wizard. `VoiceSetupStep.commit()` posts an `STTSSettingsSaveEvent` while the user advances past the Voice step; the handler always announces the publication. Textual docks toasts bottom-right of the CURRENT screen, so by the time the write settles the wizard has advanced and the toast lands over the Protect step's buttons (capture 01/02) or the Summary's docked exit actions -- "Write your first note" among them. task-32140 (PR #2538) is what made it matter: it moved the Summary's exits into two docked rows reaching further down the pane.

Fix: the Voice step awaits that save's result and renders every outcome in place (it refuses to advance on failure), so the toast was pure duplication. `STTSSettingsSaveEvent` gains a keyword-only `notify_outcome: bool = True`; `build_voice_setup_save_event` -- the wizard's own builder -- passes False, and the handler skips `_notify_settings_publication` for it. Every other requester (Settings ▸ Speech & TTS, which pins the toast by name in three tests) is untouched. Nothing moved in CSS, so the boot-CSS byte budget is unchanged.

Rejected alternatives: repositioning `ToastRack` in `_wizards.tcss` (would have to displace a rule for the ratchet, and only relocates a message the wizard should not be showing at all); keying suppression off `reply_to` (the Settings panel passes it too and legitimately toasts).

Tests: `Tests/TTS/test_stts_settings_reconfiguration.py::test_first_run_voice_save_reports_to_the_wizard_without_a_toast` -- builds the event through the real wizard builder, runs the real handler. RED: `assert app.notifications == []` -> `assert [('Settings saved successfully!', 'information')] == []`. GREEN after the fix, with `recorder.results` still reporting `persisted=True` so the wizard's own feedback path is pinned alongside.

Files: `tldw_chatbook/Event_Handlers/STTS_Events/stts_events.py`, `tldw_chatbook/UI/Wizards/first_run_voice_step_state.py`, `Tests/TTS/test_stts_settings_reconfiguration.py`, `Docs/User_Guide/First_Run_Setup.md`.
<!-- SECTION:NOTES:END -->
