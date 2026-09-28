---
id: TASK-32958
title: Add OmniVoice to the first-run Voice step
status: Done
assignee:
  - '@Robert'
created_date: '2026-09-25 10:00'
updated_date: '2026-09-26 02:30'
labels:
  - wizard
  - tts
  - omnivoice
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The first-run wizard's Voice step only offered HTTP speech services (PocketTTS, Official OpenAI, a custom compatible endpoint). OmniVoice (PR #2825) runs fully offline on this computer with no server or account, but a new user could only find it later in Lab ▸ Speech. This adds it as a fourth Voice service so setup can install the model, test a sample locally and save it as the default.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A user can select OmniVoice in the Voice step; the endpoint fields give way to a local panel that says whether the engine or model is missing or ready
- [x] #2 The 1.1 GB model installs from the wizard through the shared consent dialog (with its Terms line) and progress bar
- [x] #3 Test and Hear synthesizes and plays a sample on this computer
- [x] #4 Saving with Use as default makes OmniVoice the default TTS provider and writes a fixed voice seed that the sample also used
- [x] #5 Next is never blocked by install state, a declined consent, or an install/sample failure
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Spec: Docs/superpowers/specs/2026-09-25-wizard-omnivoice-voice-option-design.md; plan: Docs/superpowers/plans/2026-09-25-wizard-omnivoice-voice-option.md (executed subagent-driven, per-task review).

- Engine: per-request `extra_params["seed"]` (int in [0, 2**31)) accepted by admission and honoured by the sampler; `OMNIVOICE_SEED` Settings binding writes `[OmniVoiceSettings] seed`.
- `TTS/omnivoice_artifact_catalog.py`: `omnivoice_setup_state()` (engine_missing / model_missing / ready, env `OMNIVOICE_MODEL_ROOT` first) and thin preflight/provision wrappers over `ArtifactAcquisitionService`.
- `first_run_voice_step_state.py`: copy constants, `choose_omnivoice_seed` (keeps a valid existing seed), `build_omnivoice_save_event` (settings carry only the seed), `run_omnivoice_sample` through the app's shared TTS service (RIFF/WAVE + size bound).
- `VoiceSetupStep`: fourth radio, local panel, off-thread state read, install worker (preflight → consent → provision → re-read), OmniVoice branches in Test and Hear and commit, resume restore.
- Seed measurement (real model, 4 sentences): seeded median-F0 spread 50.3% vs 62.8% unseeded — the seed does NOT stabilise the default voice. The seed is still saved (reproducible output, less drift) and the ready copy says replies may vary until a voice profile is created.
- Found in live UAT and fixed: the OmniVoice panel collapsed to one row (bare Vertical = 1fr) hiding Install; Test and Hear never played on a real first run for ANY service because `app.audio_player` is created lazily only by the Speech screens — the step now creates it (wizard tests get a silent player via Tests/Wizards/conftest.py); service labels shortened to OpenAI / Custom to fit 80 columns.
- Live UAT (scratch profile, 80×24 then 200×60): model_missing → consent → 1.1 GB download → ready; sample transcribed by faster-whisper base as exactly "The meeting has been moved to Thursday afternoon."; config gained `[OmniVoiceSettings] seed` and `[app_tts] default_provider = "omnivoice"`; Lab ▸ Speech opened on OmniVoice (Local) / int8hq / Default, "ready". Not verified: a concurrent Parakeet + OmniVoice install.
- Test evidence: Tests/Wizards/test_first_run_voice_omnivoice.py (new), Tests/Wizards/test_first_run_voice_step_state.py, Tests/TTS/test_omnivoice_artifact_catalog.py; the 115 failures in test_first_run_setup_wizard.py and 31 setup errors in test_first_run_setup_integration.py (RecoveryRequired) reproduce by name on pristine origin/dev.
- Diagnostic inventory re-pinned after reading the added statements: three static `logger.opt(exception=True)` messages in the OmniVoice install/state paths, same shape as the Speech step's.
<!-- SECTION:NOTES:END -->
