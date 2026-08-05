# TASK-604 Direct-Local transcribe.cpp Batch STT Provider Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a user select an existing compatible GGUF and complete a real Library audio/video transcription with the pinned optional `transcribe.cpp` runtime.

**Architecture:** Keep direct-local execution inside the existing spawned Library parse worker. Revalidate the GGUF immediately before one native model load, derive one exact model declaration from the loaded runtime, seal it into the existing provider registry, run the existing coordinator, and return only picklable normalized transcript/provenance or a bounded path-safe failure envelope to the parent writer/UI.

**Tech Stack:** Python 3.11+, Textual, existing spawn parse pool and STT contracts, `transcribe-cpp==0.1.3`, ffmpeg/WAV normalization, pytest.

---

## Preconditions and scope

- Work only in `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/task-604-transcribe-cpp` on `codex/task-604-transcribe-cpp`.
- Governing designs: `Docs/superpowers/specs/2026-07-23-stt-parakeet-onnx-transcribe-cpp-design.md` and `Docs/superpowers/specs/2026-08-01-task-597-local-gguf-import-design.md`.
- ADR required: no.
- ADR path: `backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md` and `backlog/decisions/041-direct-local-gguf-before-managed-acquisition.md`.
- Reason: the accepted ADRs already govern the provider/runtime boundary, direct-local config ownership, worker-side revalidation, provenance, and the explicit no-managed-store/no-resident-executor scope.
- Run only tests and static checks for files and behavior changed by this task. Do not run or wait for the repository-wide suite or unrelated CI.
- Do not add downloads, copying, hashing, artifact activation, a resident executor, semantic-default participation, or automatic fallback. The deferred managed-store code remains untouched for TASK-1915.

## File map

- Create `tldw_chatbook/STT/transcribe_cpp.py` for the lazy runtime adapter, capability conversion, single-load lifecycle, normalized output, and safe failure translation.
- Modify `tldw_chatbook/Local_Ingestion/transcription_service.py`, `audio_processing.py`, `video_processing.py`, `local_file_ingestion.py`, and `ingest_parse_worker.py` only as needed to carry the direct-local request/result/failure through the existing production worker path.
- Modify `tldw_chatbook/Local_Ingestion/stt_batch_routing.py` and `tldw_chatbook/app.py` for exact manual routing, configured path injection, request lineage, and explicit faster-whisper retry overrides.
- Modify `tldw_chatbook/Library/ingest_capabilities.py`, `tldw_chatbook/Widgets/Library/library_ingest_canvas.py`, `tldw_chatbook/UI/Screens/library_screen.py`, and the first-run speech step only for the provider choice, GGUF picker/admission, key-only config persistence, and bounded recovery buttons.
- Modify `pyproject.toml` for the exact optional extra.
- Add focused tests beside the existing STT, Local_Ingestion, Library, and first-run tests; reuse TASK-597 GGUF fixtures and wheel-target declarations.

### Task 1: Add the pinned, lazy, single-job provider adapter

- [ ] Write failing adapter tests with a fake `transcribe_cpp` module/model covering: no import at module import time; worker-call import/ABI failure; TASK-597 revalidation immediately before one model construction; capabilities read after load and used identically by declaration/probe; one session run; close on success/failure; segment/timing/device/language normalization; and path/raw-exception redaction.
- [ ] Add the exact marker-free `transcription_transcribe_cpp = ["transcribe-cpp==0.1.3"]` optional extra.
- [ ] Implement the smallest adapter/run helper that imports `transcribe_cpp` only inside the execution call, silences native logging, loads one model, builds `model_id=local-gguf:<architecture>`, seals built-ins plus this one non-default model, invokes `TranscriptionCoordinator`, and closes the model in `finally`.
- [ ] Normalize every input to 16 kHz mono signed-16-bit WAV before converting samples to float32 for `Session.run`; request only supported timestamp/language/task combinations.
- [ ] Run only the new adapter/package tests and lint/format checks for these files.

### Task 2: Carry normalized success and safe failure through production ingestion

- [ ] Write failing audio and video ingestion tests proving manual `transcribe-cpp` reaches the provider helper and that `transcription_model` plus the validated provenance document survive parse payload and parent-side persistence.
- [ ] Add request identity/lineage fields to the existing worker options and pass the configured GGUF path separately from the Library form snapshot.
- [ ] Propagate normalized result metadata without changing other providers' legacy result shapes.
- [ ] Add one picklable, bounded direct-local failure type/envelope. Convert admission/runtime/coordinator failures into stable codes, a sanitized failed-attempt document, and only eligible `choose_another_gguf`/`retry_faster_whisper` action strings; pass these through `run_parse_job` and `mark_failed` without raw paths or native exception text.
- [ ] Run only the touched Local_Ingestion and Library queue tests, including a spawned-worker crash/shutdown regression.

### Task 3: Add exact manual routing and explicit retry lineage

- [ ] Write failing routing/app tests proving `transcribe-cpp` is accepted only when explicitly selected, never chosen by `default`, and never silently falls back.
- [ ] Extend the batch route with the manual provider while preserving `en` as the UI/default omitted language and leaving all semantic-default behavior unchanged.
- [ ] Add an app/registry retry seam that clones a failed job with only the audio/video provider changed to `faster-whisper`, retaining `retry_of_job_id` and the source failure snapshot already supported by the queue.
- [ ] Write and run focused tests proving the explicit retry uses faster-whisper and its successful provenance links to the failed direct-local attempt.

### Task 4: Add the provider picker and bounded recovery UI

- [ ] Write failing Library canvas/screen and first-run speech-step tests for: exact provider option; `Choose GGUF…`; `.gguf` filtering; off-loop admission; success persistence only to `[transcription.transcribe_cpp].model_path` via one `save_settings_to_cli_config` call; restart prefill without generic path rendering; and no persistence on cancel/failure.
- [ ] Reuse the existing Textual file picker. Keep the model path out of `library.ingest_options.*`; read it from the dedicated config section only when dispatching the spawned job.
- [ ] Render only a path-free configured/not-configured status. Add `Choose another GGUF…` and `Retry with faster-whisper` buttons only when the worker's allowlisted action names are present.
- [ ] Wire the recovery buttons to the same picker and explicit retry seam, then run only the touched UI/state tests.

### Task 5: Focused vertical verification and closeout

- [ ] Add one focused production-path integration test: Library form snapshot -> app dispatch -> spawn-safe parse entry -> fake native model -> parent writer -> stored transcript/provenance.
- [ ] Add static package/provider smoke covering the exact optional pin and all five TASK-597 wheel target pairs. Preserve Linux ABI/wheel resolution as a gate for a Linux-capable environment; do not claim local execution for unavailable Windows/Linux hosts.
- [ ] Run only the focused TASK-604 test files, `py_compile`/Ruff for modified Python files, the relevant TOML parse check, and `git diff --check`.
- [ ] Self-review the complete branch diff for path leakage, startup imports, silent fallback, shutdown cleanup, and scope creep.
- [ ] Update TASK-604 acceptance criteria and concise Implementation Notes through Backlog CLI only after focused verification passes.
