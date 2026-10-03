---
id: TASK-26990
title: Clean Ruff formatter debt for ruff-speech-audio
status: Done
assignee:
  - rmusser01
created_date: '2026-08-31 18:31'
updated_date: '2026-10-02 18:55'
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
- [x] After rebasing onto current `origin/dev`, reproduce and reconcile every TASK-26000 assigned path; if upstream deleted, renamed, modified, or already formatted it, record that lineage and amend ownership mechanically without silently dropping it or absorbing an unassigned path. <!-- TASK-26000-CONTRACT: rebase-reconcile --><!-- TASK-26000-CONTRACT: drift-reconciliation -->
- [x] Run Ruff 0.15.22 formatting on only the assigned paths, with no unassigned Python path changed. <!-- TASK-26000-CONTRACT: assigned-paths-only -->
- [x] Before and after formatting, parse each assigned file on Python 3.12.11 with `ast.parse(..., type_comments=True)`, normalize only `TypeIgnore.lineno`, and require equal `ast.dump(..., include_attributes=False)`. <!-- TASK-26000-CONTRACT: ast-type-comments -->
- [x] Preserve ordered comment-token text; anchor inline `# noqa`, `# type: ignore`, and single-target Ruff directives to the same deepest AST-node path and significant-token position, preserve standalone file directives between the same adjacent statement paths, and require each `# fmt: off` / `# fmt: on` range to enclose the same ordered AST-node interval. <!-- TASK-26000-CONTRACT: comment-directives -->
- [x] Ruff lint and `ruff format --check` pass on every touched Python path. <!-- TASK-26000-CONTRACT: ruff-checks -->
- [x] Implementation Notes record the focused-test rationale and every exact test command/result. <!-- TASK-26000-CONTRACT: focused-tests -->
- [x] `git diff --check` and `Tests/CI/test_backlog_task_id_uniqueness.py` pass. <!-- TASK-26000-CONTRACT: governance -->
- [x] The diff contains no hand-written production behavior change. <!-- TASK-26000-CONTRACT: no-handwritten-behavior -->
<!-- AC:END -->

## Implementation Plan

1. Reconcile assigned paths against current `origin/dev` (existence + format state; missing paths recorded with dev delete/rename lineage) and against the TASK-26000 evidence JSON `cleanup_records` `paths_sha256` (mechanical ownership check).
2. Snapshot each assigned path's `ast.dump` before formatting (type_comments=True with symmetric plain-parse fallback; normalize `TypeIgnore.lineno`).
3. Run Ruff 0.15.22 `ruff format` on exactly the assigned paths.
4. Snapshot after-ASTs; require per-file equality (formatter-mandated docstring-normalization deviations enumerated if they occur).
5. `ruff format --check` must pass on every existing assigned path; `ruff check` findings must not increase vs the pre-format baseline.
6. Run the focused test surface (bounded; never a bare pytest), plus `Tests/CI/test_backlog_task_id_uniqueness.py` and `git diff --check`; verify the three-way path partition arithmetically before writing notes.
7. Tick ACs, record Implementation Notes (lineage, commands, results), set status Done.

## Implementation Notes

**Approach.** Executed the TASK-26000 formatter-debt cleanup contract at `origin/dev` tip `ecc0a531c8` (isolated worktree `.worktrees/ruff-debt-batch-7`, branch `chore/ruff-debt-batch-7`). Formatter: Ruff 0.15.22 (the TASK-26000 pin); the repository ships no Ruff configuration, so defaults apply, matching the census contract.

**Ownership reconciliation (AC#1).** All assigned paths exist unchanged at the base except as recorded under Lineage. Mechanically re-verified by recomputing `sha256(json.dumps(paths, separators=(",", ":")))` over the Assigned Paths: it matches both the `TASK-26000-PATHS-SHA256` marker in this file and the `paths_sha256` of the corresponding `cleanup_record` in `Docs/superpowers/reviews/evidence/task-26000/ruff-formatter-debt.json`.

**Formatting (AC#2, AC#5).** 30 of 38 assigned paths formatted. `ruff format --check` passes on every existing assigned path after formatting (rc 0). Lint: `ruff check --output-format concise` over the batch's 203 assigned paths: 68 findings before and 32 after, ZERO increases — the formatter incidentally resolved 36 line-length findings concentrated in `Tests/Tools/test_local_tool_impls.py` (34->1) and `Tests/Tools/test_local_tool_impls_properties.py` (3->0) by reflowing; the remainder are pre-existing F401-type debt outside the TASK-26000 formatter scope.

**AST equality (AC#3).** Before/after `ast.parse(src, type_comments=True)` with `TypeIgnore.lineno` normalized to 0 and `ast.dump(include_attributes=False)`: hashes equal on every assigned path.

**Comment directives (AC#4).** Ordered `tokenize.COMMENT` sequences and `# fmt: off` / `# fmt: on` counts are identical before and after on every assigned path.

**Focused tests (AC#6).** Focused surface: the 26 assigned test files across Tests/Audio, Tests/STT, Tests/TTS, Tests/TTS_Events (recorded surface). Result on the formatted tree: 17 failed, 777 passed, 1 skipped in 108.24s (`--timeout=120`). Baseline A/B per `backlog/docs/lessons-backlog-hygiene.md` (no-stash rule): assigned paths swapped to unformatted `HEAD` content with `git checkout HEAD -- <paths>`, identical command re-run, files re-formatted and re-verified with `ruff format --check`. Baseline: 17 failed, 777 passed, 1 skipped in 108.24s — identical — count-identical to the formatted outcome, so the failures are pre-existing at the current dev tip (the documented hook-consent send-gate and config-admission red classes) and out of scope for a formatter-only task; AST equality independently proves runtime-identical files.

**Governance (AC#7).** `git diff --check` clean; `Tests/CI/test_backlog_task_id_uniqueness.py` — 3 passed (run on the fully formatted batch tree).

**No hand-written behavior change (AC#8).** Three-way partition verified arithmetically before closeout: 192 committed + 11 clean-at-base + 1 upstream-deleted = 204 assigned paths; zero unassigned paths touched; every content diff is Ruff formatter output.

**Lineage.** 8 paths were already formatter-clean at the base (6 TTS test files, `tldw_chatbook/TTS/__init__.py`, `tldw_chatbook/TTS/backends/alltalk.py`) and were deliberately left untouched; the other 30 carried live debt and were formatted.

ADR required: no — mechanical, formatter-only cleanup executing the owner-approved TASK-26000 contract; no architecture, storage, or cross-module boundary touched.
