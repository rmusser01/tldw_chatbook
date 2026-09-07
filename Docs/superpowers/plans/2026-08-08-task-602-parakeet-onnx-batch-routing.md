# TASK-602 Parakeet ONNX Batch Routing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Complete the gated Parakeet ONNX batch path by adding verified managed v2/v3/VAD artifacts, explicit INT8/F32 selection, cancellable long-form execution, normalized provenance, and CPU-only package profiles without opening the semantic-default promotion gate.

**Architecture:** Extend the existing `ModelArtifactService` curated registry and `LocalSTTExecutor`; do not introduce another downloader, queue, worker, writer, or retry system. A small executor-native Parakeet runtime loads only verified local paths, keeps the root plus VAD dependency leased for resident lifetime, and returns the existing normalized STT contract. Existing manual local Parakeet folders remain usable for short-form compatibility, while managed long-form requires the exact pinned VAD dependency. TASK-605 remains the sole owner of switching semantic defaults and removing legacy providers.

**Tech Stack:** Python 3.11+, Textual 8, `onnx-asr[cpu]==0.12.0`, ONNX Runtime CPU, existing `ModelArtifactService`, existing `LocalSTTExecutor`, pytest.

## Global Constraints

- ADR required: no.
- ADR path: `backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md`.
- Reason: ADR-025 already governs artifact identity/dependencies, offline provider loading, resident leases, routing, provenance, package profiles, retry, and promotion gates.
- Keep `parakeet_defaults_enabled=False` in production until TASK-605. Exact/configured `parakeet-onnx` remains usable.
- Default Parakeet precision is INT8; F32 is admitted only when explicitly selected.
- Never download from the executor worker. All remote acquisition stays in the existing parent/UI artifact flow.
- Do not delete legacy NeMo/MLX provider code in this task; TASK-605 owns removal after release gates pass.
- Run only tests covering files or behavior changed by this task. Do not run the repository-wide suite or wait on unrelated CI.
- Preserve AC7 as an open release gate for unavailable Windows/Linux native hosts. Collect macOS evidence now; do not weaken or falsely mark unavailable platform gates complete.
- Exact managed artifacts are limited to the TASK-593-qualified v2/v3 revisions and language allowlist. The pinned VAD descriptor uses the reviewed `istupakov/silero-vad-onnx` immutable revision and SHA-256.

## Task 1: Pin the CPU-only optional-dependency profiles

**Files:**
- Modify: `pyproject.toml`
- Modify: `tldw_chatbook/Utils/optional_deps.py`
- Modify: `tldw_chatbook/Library/ingest_capabilities.py`
- Create: `Tests/STT/test_parakeet_package_profiles.py`
- Modify: `Tests/Utils/test_optional_deps.py`
- Modify: `Tests/Library/test_ingest_capabilities.py`

1. Write failing tests that parse `pyproject.toml` and require `onnx-asr[cpu]==0.12.0` in `audio`, `video`, `media_processing`, `transcription_parakeet`, `transcription_parakeet_onnx`, and `all-tools`, while rejecting accelerator ONNX Runtime distributions in those profiles.
2. Add failing tests that the legacy `transcription_parakeet` feature/probe and Library recovery hint now resolve to the ONNX package/profile.
3. Run only the new package-profile test plus the two changed optional-dependency test files and confirm the intended failures.
4. Repurpose the documented Parakeet extra and add the same CPU baseline to the media extras. Keep the separate ONNX alias for compatibility. Remove only redundant direct `onnxruntime` entries where `onnx-asr[cpu]` already supplies the compatible CPU runtime; do not alter unrelated TTS extras.
5. Update lazy probes and recovery mappings without importing native libraries at module scope.
6. Re-run the focused tests and commit: `build(stt): pin Parakeet ONNX CPU profiles`.

## Task 2: Add exact Parakeet v2/v3, INT8/F32, and VAD artifact closures

**Files:**
- Modify: `tldw_chatbook/Local_Ingestion/parakeet_v2_artifact.py`
- Modify: `tldw_chatbook/Model_Artifacts/curated_registry.py`
- Modify: `tldw_chatbook/UI/Screens/model_curated_view.py`
- Modify: `Tests/Local_Ingestion/test_parakeet_v2_artifact.py`
- Modify: `Tests/Model_Artifacts/test_curated_registry.py`
- Modify: `Tests/UI/test_model_curated_view.py`

1. Write failing descriptor tests for four root selections `(v2|v3) x (int8|f32)` plus one Silero VAD dependency. Assert exact revisions, files, sizes, SHA-256 values, licenses, precision, CPU platform metadata, and root dependency identity.
2. Write a failing closure test proving preflight resolves root plus VAD and source maps cover every declared file without credentials or network imports at worker-module scope.
3. Write a failing registry/view test proving dependency descriptors are catalog-resolvable but only root models render as standalone curated choices.
4. Run the three focused test files and confirm failures.
5. Extend the existing compatibility module with generic `parakeet_reference`, `parakeet_descriptor`, active-resolution, preflight, and provision helpers. Keep existing v2 INT8 helper names as wrappers so current Library/wizard callers do not break.
6. Use a new closure-bearing root revision identity so previously installed dependency-free v2 artifacts remain immutable and visible instead of being silently reinterpreted. Do not delete or mutate old payloads.
7. Register the four root descriptors and the VAD dependency in the existing curated registry; filter the curated screen to root roles.
8. Re-run focused tests and commit: `feat(stt): add managed Parakeet artifact closures`.

## Task 3: Carry explicit precision through batch routing and ingestion options

**Files:**
- Modify: `tldw_chatbook/Local_Ingestion/stt_batch_routing.py`
- Modify: `tldw_chatbook/Library/ingest_capabilities.py`
- Modify: `tldw_chatbook/app.py`
- Modify: `Tests/Transcription/test_stt_batch_routing.py`
- Modify: `Tests/App/test_submit_library_ingest_job.py`
- Modify: `Tests/Library/test_ingest_capabilities.py`

1. Write failing route tests for omitted precision -> INT8, explicit F32 on exact Parakeet routes, rejection of unknown precision, and unchanged faster-whisper/default-gate behavior.
2. Write failing app/capability tests proving the Library exposes a Parakeet-only precision selector and preserves the normalized selection through `_ingest_job_options`.
3. Run only these three test files and confirm failures.
4. Add a precision input to the dependency-free route contract. Keep INT8 as the default and do not let precision change engines or open semantic defaults.
5. Carry the exact precision through audio/video options and select the correct required local filenames for direct folders.
6. Re-run focused tests and commit: `feat(stt): route explicit Parakeet precision`.

## Task 4: Implement the executor-native offline Parakeet runtime

**Files:**
- Create: `tldw_chatbook/STT/parakeet_onnx.py`
- Create: `Tests/STT/test_parakeet_onnx.py`

1. Write failing dependency-free tests using fake ASR/VAD objects for v2/v3 language provenance, INT8/F32 load arguments, no decoder language argument for v3, short-form normalization, long-form segment timestamps, VAD batch size one, and cancellation checked before every segment inference batch.
2. Include a mutation-style assertion: a fake cancellation token flips before the second segment and proves the second ASR call never occurs.
3. Run the new test file and confirm failures.
4. Implement one small runtime that lazily imports `onnx_asr`, loads model and VAD from explicit directories with CPU providers, and never calls a hub/download API.
5. Wrap the pinned 0.12.0 VAD segment loop so it checks cancellation immediately before each one-segment ASR batch. Use VAD for long input (over 30 seconds) or explicit VAD requests; preserve the short-file path.
6. Return `TranscriptionResult` with exact root/dependency lease keys, requested/effective/detected language semantics, capabilities, timings, and stable warnings.
7. Re-run the new tests and commit: `feat(stt): add cancellable Parakeet ONNX runtime`.

## Task 5: Wire managed closures and normalized results through LocalSTTExecutor

**Files:**
- Modify: `tldw_chatbook/STT/executor_worker.py`
- Modify: `tldw_chatbook/app.py`
- Modify: `Tests/STT/test_local_stt_executor.py`
- Modify: `Tests/Library/test_library_ingest_runner.py`

1. Write failing worker tests proving the provider receives the verified root and VAD paths from one acquired closure, keeps both leases for resident lifetime, reuses the same runtime for an identical identity, and returns normalized persisted provenance.
2. Write failing cancellation coverage proving a provider exception after the event is set produces `cancelled`, not `inference_failed`.
3. Write failing app dispatch tests for managed v2/v3 INT8/F32 selection, exact closure identity, missing/corrupt dependency failure, and no acquisition/download import in the worker path.
4. Run only these two test files and confirm failures.
5. Pass the acquired `ArtifactHandle` paths to provider construction, replace the legacy `TranscriptionService` executor adapter with `ParakeetOnnxRuntime`, and serialize its normalized result through the existing provenance builder.
6. Resolve active managed roots by exact model/precision in `_build_local_stt_dispatch`; keep direct local folders as snapshotted compatibility input and fail long-form clearly when no managed VAD is available.
7. Preserve the existing app queue, one-heavy-job lane, parent writer, retry action, and generation fences.
8. Re-run focused tests and commit: `feat(stt): execute managed Parakeet batch jobs`.

## Task 6: Make first-run selection install/configure the exact artifact

**Files:**
- Modify: `tldw_chatbook/UI/Wizards/first_run_speech_step_state.py`
- Modify: `tldw_chatbook/UI/Wizards/FirstRunSetupWizard.py`
- Modify: `Tests/Wizards/test_first_run_speech_step_state.py`
- Modify: `Tests/Wizards/test_first_run_speech_step.py`

1. Write failing pure-state tests mapping English/non-English plus INT8/F32 to exact model/precision selections.
2. Write failing wizard tests proving selection changes update the inspected/install target, preflight/provision use the existing generic artifact service, commit occurs only when that exact artifact is active, and no unavailable selection is persisted.
3. Run only the two speech-step test files and confirm failures.
4. Replace the step's fixed v2 INT8 reference with a reference derived from the pressed language/precision. Reuse the generic preflight/provision helpers; do not create wizard-specific download logic.
5. Update model labels and confirmation copy from the selected descriptor. Keep the skip-safe/no-clobber behavior and direct-local GGUF picker unchanged.
6. Re-run focused tests and commit: `feat(stt): configure exact Parakeet model in setup`.

## Task 7: Focused verification, native macOS evidence, and open-gate documentation

**Files:**
- Modify: `backlog/tasks/task-602 - Integrate-Parakeet-ONNX-batch-routing.md`
- Create: `Docs/STT_Evaluation/task-602/README.md`
- Create: `Docs/STT_Evaluation/task-602/macos-evidence.json`
- Update only directly affected documentation if focused verification exposes stale package/install copy.

1. Run the union of the focused test files changed above, not the full suite.
2. Run Ruff only on changed Python files, `python -m compileall` only on changed package modules, TOML parse/profile assertions, JSON parse, and `git diff --check`.
3. With the existing TASK-593 local v2/v3 model snapshots and a hash-verified pinned VAD file, run a small macOS CPU smoke for v2 INT8, v3 INT8, one F32 case, long-form segmentation/cancellation, resident reuse, and explicit retry wiring. Record exact interpreter/package/model identities and label this native-host evidence accurately.
4. Self-review the complete diff for implicit downloads, fake v3 language enforcement/detection, path leakage, stale worker imports, and accidental semantic-default promotion.
5. Check current PR/branch state again, rebase on latest `origin/dev`, rerun only the focused verification affected by the rebase, and request code review.
6. Update TASK-602 implementation notes and check only acceptance criteria actually evidenced. Keep AC7 and status `In Progress` while Windows/Linux native gates are unavailable; do not claim the task globally complete.
7. Commit documentation/task hygiene: `docs(stt): record TASK-602 focused evidence`.
