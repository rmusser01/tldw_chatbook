---
id: TASK-26990
title: Clean Ruff formatter debt for ruff-speech-audio
status: To Do
assignee: []
created_date: '2026-08-31 18:31'
updated_date: '2026-08-31 18:31'
labels:
  - maintenance
  - formatting
  - quality
dependencies:
  - TASK-26000
references:
  - Docs/superpowers/specs/2026-08-30-task-26000-ruff-formatter-debt-design.md
  - Docs/superpowers/reviews/evidence/task-26000/ruff-formatter-debt.json
priority: medium
---

<!-- TASK-26000-BATCH: ruff-speech-audio -->
<!-- TASK-26000-PATHS-SHA256: e238f2e21ddc273dcb9e14d7d4c524660fb67cfaa4e297ac365cdc109fe4517a -->
<!-- TASK-26000-FINAL: false -->

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Clean the `ruff-speech-audio` Ruff formatter batch at the owner boundary recorded as: Audio, STT, and TTS runtime surfaces with direct tests.. The focused test surface recorded by TASK-26000 is `["Tests/Audio", "Tests/STT", "Tests/TTS"]`.
<!-- SECTION:DESCRIPTION:END -->

## Assigned Paths

```json
[
  "Tests/Audio/test_audio_init_lazy_import_safety.py",
  "Tests/Audio/test_dictation_capture_release.py",
  "Tests/Audio/test_dictation_segment_finalization.py",
  "Tests/Audio/test_dictation_segment_transcribing_callback.py",
  "Tests/Audio/test_realtime_mic_tap.py",
  "Tests/Audio/test_streaming_sink.py",
  "Tests/Audio/test_streaming_sink_pump.py",
  "Tests/STT/test_dispatch_coordinator.py",
  "Tests/STT/test_parakeet_dispatch.py",
  "Tests/STT/test_parakeet_onnx.py",
  "Tests/STT/test_routing.py",
  "Tests/TTS/test_audio_player.py",
  "Tests/TTS/test_audio_stitch.py",
  "Tests/TTS/test_character_request_resolver.py",
  "Tests/TTS/test_console_speak_autoplay.py",
  "Tests/TTS/test_kokoro_download_hardening.py",
  "Tests/TTS/test_legacy_request_builder.py",
  "Tests/TTS/test_pcm_stream_plan.py",
  "Tests/TTS/test_profile_reference_materialization.py",
  "Tests/TTS/test_profile_types.py",
  "Tests/TTS/test_tts_app_ownership.py",
  "Tests/TTS/test_tts_connection_error_copy.py",
  "Tests/TTS/test_tts_profile_capabilities.py",
  "Tests/TTS/test_tts_request_admission.py",
  "Tests/TTS_Events/test_spoken_feedback_streaming.py",
  "Tests/TTS_Events/test_utterance_speech_entry.py",
  "tldw_chatbook/Audio/realtime_mic_tap.py",
  "tldw_chatbook/Audio/streaming_sink.py",
  "tldw_chatbook/Event_Handlers/TTS_Events/tts_events.py",
  "tldw_chatbook/STT/dispatch_coordinator.py",
  "tldw_chatbook/STT/executor_worker.py",
  "tldw_chatbook/STT/parakeet_onnx.py",
  "tldw_chatbook/STT/persistence.py",
  "tldw_chatbook/STT/routing.py",
  "tldw_chatbook/TTS/__init__.py",
  "tldw_chatbook/TTS/backends/alltalk.py",
  "tldw_chatbook/TTS/playground_types.py",
  "tldw_chatbook/TTS/profile_types.py"
]
```

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] After rebasing onto current `origin/dev`, reproduce and reconcile every TASK-26000 assigned path; if upstream deleted, renamed, modified, or already formatted it, record that lineage and amend ownership mechanically without silently dropping it or absorbing an unassigned path. <!-- TASK-26000-CONTRACT: rebase-reconcile --><!-- TASK-26000-CONTRACT: drift-reconciliation -->
- [ ] Run Ruff 0.15.22 formatting on only the assigned paths, with no unassigned Python path changed. <!-- TASK-26000-CONTRACT: assigned-paths-only -->
- [ ] Before and after formatting, parse each assigned file on Python 3.12.11 with `ast.parse(..., type_comments=True)`, normalize only `TypeIgnore.lineno`, and require equal `ast.dump(..., include_attributes=False)`. <!-- TASK-26000-CONTRACT: ast-type-comments -->
- [ ] Preserve ordered comment-token text; anchor inline `# noqa`, `# type: ignore`, and single-target Ruff directives to the same deepest AST-node path and significant-token position, preserve standalone file directives between the same adjacent statement paths, and require each `# fmt: off` / `# fmt: on` range to enclose the same ordered AST-node interval. <!-- TASK-26000-CONTRACT: comment-directives -->
- [ ] Ruff lint and `ruff format --check` pass on every touched Python path. <!-- TASK-26000-CONTRACT: ruff-checks -->
- [ ] Implementation Notes record the focused-test rationale and every exact test command/result. <!-- TASK-26000-CONTRACT: focused-tests -->
- [ ] `git diff --check` and `Tests/CI/test_backlog_task_id_uniqueness.py` pass. <!-- TASK-26000-CONTRACT: governance -->
- [ ] The diff contains no hand-written production behavior change. <!-- TASK-26000-CONTRACT: no-handwritten-behavior -->
<!-- AC:END -->
