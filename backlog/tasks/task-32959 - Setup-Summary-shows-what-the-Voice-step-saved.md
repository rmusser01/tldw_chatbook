---
id: TASK-32959
title: Setup Summary shows what the Voice step saved
status: Done
assignee:
  - '@Robert'
created_date: '2026-09-26 01:30'
updated_date: '2026-09-27 10:00'
labels:
  - wizard
  - tts
dependencies:
  - TASK-32958
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The first-run setup Summary reads back a ✓/✗/– line per area from saved config, but has no Voice line for any service (PocketTTS, OpenAI, Custom or OmniVoice), so a user gets no confirmation of what the Voice step saved or whether it became the default. Found during TASK-32958 live UAT; pre-existing on dev. (The other two follow-ups first filed here — a broken configured OmniVoice path offering a useless download, and a stale sample overwriting a newer status — were fixed in TASK-32958's PR after Qodo review.)
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The setup Summary shows a Voice line naming the saved service and whether it is the default TTS provider, read back from saved config
- [x] #2 A skipped Voice step shows as not set up (optional), like the other optional areas
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- `build_summary_rows` gains a Voice row right after Speech transcription, built by `_voice_summary_row` in `UI/Wizards/first_run_setup_state.py`.
- It reads only the raw `[app_tts]` table: `load_settings()` back-fills `APP_TTS_CONFIG.default_provider = "openai"` when nothing was saved, so the loaded view would claim a choice nobody made (pinned by `test_loaded_config_backfill_is_not_a_choice`).
- Naming: `omnivoice` → OmniVoice; `openai` → PocketTTS / OpenAI / "Custom endpoint host:port" by comparing the normalized saved endpoint with the Voice step's own endpoint constants (host and port only, so URL userinfo never reaches the screen); any other provider shows its id. An endpoint saved without Use as default reads "(saved, not the default voice)"; nothing saved reads "– not set up (optional)".
- Tests: `TestVoiceSummaryRow` (10 cases) in Tests/Wizards/test_first_run_setup_state.py; `test_summary_step_reads_back_the_saved_voice` render test (hits the local RecoveryRequired storage gate like its existing sibling render test; runs in CI). Wizard suite failure names match pristine dev apart from that gated test.
- Live check (scratch profile, 200×60): PocketTTS + Use as default → Summary "✓ Voice — PocketTTS (default voice)", config `[app_tts] default_provider = "openai"`, `OPENAI_BASE_URL = http://127.0.0.1:8765/v1/audio/speech`.
- User guide: Summary paragraph + Verified-against stamp.
<!-- SECTION:NOTES:END -->
