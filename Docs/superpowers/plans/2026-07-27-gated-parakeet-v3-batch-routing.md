# Gated Parakeet v3 Batch Routing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add explicit Parakeet v3 INT8 batch transcription and deterministic STT request routing without enabling unqualified Parakeet semantic defaults.

**Architecture:** A small dependency-free routing module resolves a batch request before inference. `provider=default` remains on faster-whisper while the promotion gate is closed; exact `parakeet-onnx` requests select v2 for English or v3 for supported non-English and fail clearly for incompatible requests. The existing `TranscriptionService` remains the narrow execution seam and gains only the v3 model/result differences needed by the working batch path.

**Tech Stack:** Python 3.11+, pytest, Textual ingestion option contracts, `onnx-asr[cpu]==0.12.0`.

**ADR required:** yes

**ADR path:** `backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md`

**Reason:** ADR-025 already governs semantic STT routing, v3's routing-only language behavior, INT8 selection, explicit faster-whisper recovery, and the promotion gate. This child task implements a bounded subset without changing those decisions.

---

### Task 1: Add the gated batch routing policy

**Files:**
- Create: `tldw_chatbook/Local_Ingestion/stt_batch_routing.py`
- Create: `Tests/Transcription/test_stt_batch_routing.py`

- [ ] **Step 1: Write failing routing tests**

Cover these independent outcomes:

```python
def test_default_stays_on_faster_whisper_while_promotion_gate_is_closed():
    route = resolve_batch_stt_route(provider="default", language="en")
    assert route.provider == "faster-whisper"
    assert route.reason == "parakeet_promotion_gate_closed"
    assert route.precision == "int8"
    assert route.local_files_only is True


@pytest.mark.parametrize(
    ("language", "target_language", "provider", "model"),
    [
        ("en", None, "parakeet-onnx", PARAKEET_V2_MODEL),
        ("de", None, "parakeet-onnx", PARAKEET_V3_MODEL),
        ("auto", None, "faster-whisper", None),
        ("ja", None, "faster-whisper", None),
        ("de", "en", "faster-whisper", None),
    ],
)
def test_enabled_default_policy_covers_every_routing_row(
    language, target_language, provider, model
):
    route = resolve_batch_stt_route(
        provider="default",
        language=language,
        target_language=target_language,
        parakeet_defaults_enabled=True,
    )
    assert (route.provider, route.model) == (provider, model)


@pytest.mark.parametrize("language", [None, "", "en", "EN"])
def test_explicit_parakeet_english_selects_v2(language):
    route = resolve_batch_stt_route(provider="parakeet-onnx", language=language)
    assert route.model == PARAKEET_V2_MODEL
    assert route.requested_language == "en"


@pytest.mark.parametrize("language", ["de", "es", "fr", "uk"])
def test_explicit_supported_non_english_selects_v3(language):
    route = resolve_batch_stt_route(provider="parakeet-onnx", language=language)
    assert route.model == PARAKEET_V3_MODEL
    assert route.requested_language == language


@pytest.mark.parametrize(
    ("language", "target_language"),
    [("auto", None), ("ja", None), ("de", "en")],
)
def test_explicit_parakeet_rejects_incompatible_requests_with_retry_guidance(
    language, target_language
):
    with pytest.raises(BatchSTTRoutingError, match="Retry with faster-whisper"):
        resolve_batch_stt_route(
            provider="parakeet-onnx",
            language=language,
            target_language=target_language,
        )
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
../../.venv/bin/python -m pytest Tests/Transcription/test_stt_batch_routing.py -q
```

Expected: collection fails because `stt_batch_routing` does not exist.

- [ ] **Step 3: Implement the minimal routing module**

Add immutable constants for the v2/v3 model IDs and upstream v3 language set, a frozen `BatchSTTRoute`, a `BatchSTTRoutingError`, and:

```python
def resolve_batch_stt_route(
    *,
    provider: str | None,
    language: str | None,
    target_language: str | None = None,
    parakeet_defaults_enabled: bool = False,
) -> BatchSTTRoute:
    ...
```

Rules:

- Normalize missing language to `en` and lowercase explicit codes.
- `default` resolves to faster-whisper while `parakeet_defaults_enabled` is false.
- With the gate enabled, `default` follows ADR-025: English/v2, supported non-English/v3, and auto/unsupported/translation/faster-whisper.
- Exact `parakeet-onnx` selects v2/v3 only for compatible explicit languages and raises with the literal recovery action for auto, unsupported languages, or translation.
- Exact `faster-whisper` remains exact and retains the requested language/task.
- Every batch route records `precision="int8"` and `local_files_only=True`; routing never authorizes a worker download.
- Reject unknown providers; do not prefix-match or silently substitute.

- [ ] **Step 4: Run the routing tests and verify GREEN**

Run:

```bash
../../.venv/bin/python -m pytest Tests/Transcription/test_stt_batch_routing.py -q
```

Expected: all routing tests pass.

- [ ] **Step 5: Commit the routing policy**

```bash
git add tldw_chatbook/Local_Ingestion/stt_batch_routing.py Tests/Transcription/test_stt_batch_routing.py
git commit -m "feat(stt): add gated batch routing policy"
```

### Task 2: Add Parakeet v3 execution and transparent language semantics

**Files:**
- Modify: `tldw_chatbook/Local_Ingestion/transcription_service.py`
- Modify: `Tests/Transcription/test_parakeet_onnx_vertical_slice.py`

- [ ] **Step 1: Write failing service tests**

Add tests proving:

- Explicit `de` plus the v3 model and local INT8 bundle loads `nemo-parakeet-tdt-0.6b-v3`.
- `load_model()` receives no language/decoder constraint.
- The normalized v3 result contains:

```python
{
    "language": None,
    "requested_language": "de",
    "effective_language": "auto",
    "detected_language": None,
    "warnings": ["requested_language_not_enforced"],
}
```

- English v2 additionally reports `requested_language=en`, `effective_language=en`, and no warning while preserving existing legacy result keys.
- Auto, unsupported explicit languages, translation, and mismatched model/language pairs fail before `onnx_asr.load_model()` and contain “Retry with faster-whisper”.

- [ ] **Step 2: Run the focused service tests and verify RED**

Run:

```bash
../../.venv/bin/python -m pytest Tests/Transcription/test_parakeet_onnx_vertical_slice.py -q
```

Expected: v3/result-contract assertions fail because the service is v2-only.

- [ ] **Step 3: Implement minimal v2/v3 model validation and result shaping**

Use the routing constants rather than duplicating model strings. Change `_load_parakeet_onnx_model()` to validate the exact model/language pairing, keep `quantization="int8"` and `CPUExecutionProvider`, and continue requiring a user-selected existing local directory with the required filenames. Do not pass language to `onnx_asr.load_model()`.

A narrow, non-symlink verification-receipt metadata read of at most 64 KiB is
allowed only to identify repository and revision metadata as v2 and reject that
directory when v3 is selected. The receipt is not authenticated and does not
verify file contents or v3 eligibility; malformed, oversized, or otherwise
untrusted receipts are ignored. Do not download artifacts or parse ONNX graphs.

Change `_parakeet_onnx_result()` to receive `requested_language` and distinguish:

- v2: effective English, no detected language, no warnings.
- v3: nullable legacy language, effective auto, no fabricated detection, stable warning.

Keep the existing text and segment keys so current ingestion callers remain compatible.

- [ ] **Step 4: Run the focused service tests and verify GREEN**

Run:

```bash
../../.venv/bin/python -m pytest Tests/Transcription/test_parakeet_onnx_vertical_slice.py -q
```

Expected: all focused Parakeet ONNX tests pass.

- [ ] **Step 5: Commit v3 execution support**

```bash
git add tldw_chatbook/Local_Ingestion/transcription_service.py Tests/Transcription/test_parakeet_onnx_vertical_slice.py
git commit -m "feat(stt): add explicit Parakeet v3 batch inference"
```

### Task 3: Resolve and preserve routes through audio and video ingestion

**Files:**
- Modify: `tldw_chatbook/app.py`
- Modify: `tldw_chatbook/Local_Ingestion/audio_processing.py`
- Modify: `tldw_chatbook/Local_Ingestion/transcription_service.py`
- Modify: `tldw_chatbook/Local_Ingestion/local_file_ingestion.py`
- Modify: `Tests/App/test_submit_library_ingest_job.py`
- Modify: `Tests/Local_Ingestion/test_local_file_ingestion.py`
- Modify: `Tests/Local_Ingestion/test_audio_model_dir_routing.py`
- Modify: `Tests/Transcription/test_faster_whisper_transcription.py`

- [ ] **Step 1: Write failing app and ingestion seam tests**

Add cases proving:

- A Library audio/video job with exact Parakeet plus `de` stores the v3 model, normalized language, and selected local model directory.
- An exact Parakeet English job stores v2.
- A semantic `default` job resolves to faster-whisper while the gate is closed and drops a stale Parakeet directory.
- Audio and video processor calls receive the same resolved provider/model/language/model directory, `precision="int8"`, and `local_files_only=True`.
- Faster-whisper batch model construction receives `local_files_only=True` and `compute_type="int8"` even when service config says `float16`; direct non-batch calls without routed overrides retain the configured compute type.
- Incompatible exact Parakeet requests are caught after the job is claimed and marked failed with sanitized “Retry with faster-whisper” guidance before pool creation or submission. A queue regression test places a valid job behind the invalid job and proves the invalid job never reaches the pool while the valid job is still dispatched in the same top-up pass.

- [ ] **Step 2: Run seam tests and verify RED**

Run:

```bash
../../.venv/bin/python -m pytest \
  Tests/App/test_submit_library_ingest_job.py \
  Tests/Local_Ingestion/test_local_file_ingestion.py \
  Tests/Local_Ingestion/test_audio_model_dir_routing.py \
  Tests/Transcription/test_faster_whisper_transcription.py -q
```

Expected: v3/default-routing assertions fail against the hard-coded v2 app seam.

- [ ] **Step 3: Resolve once at the app option boundary**

In `_ingest_job_options()` call `resolve_batch_stt_route()` for audio/video options, then store the resolved provider, model, normalized requested language, precision, and local-only policy. Retain a model directory only for resolved Parakeet.

In `_top_up_ingest_parse_pool()`, wrap `_ingest_job_options(claimed)` before pool creation. On `BatchSTTRoutingError`, sanitize the message, mark the claimed job failed and retryable, decrement the local `parsing_count` and (for audio/video) `heavy_parsing_count` that were incremented for the claim, then `continue` scanning the queue. This prevents a stuck `PARSING` row without stranding valid jobs behind it.

Keep `local_file_ingestion.py` mechanical: forward the already-resolved fields unchanged to audio and video processors. In `audio_processing.py`, pass precision and the local-only flag into `TranscriptionService`. In the faster-whisper loader, select `effective_compute_type = kwargs.get("compute_type") or self.config["compute_type"]`, use it in both the cache key and `WhisperModel(compute_type=...)`, and pass `local_files_only=True` when requested. Direct callers that omit the override retain configured behavior. Do not download, parse graphs, or reroute in a worker; the service's bounded receipt read described in Task 2 is allowed only to reject repository/revision metadata that identify v2 when v3 is selected.

- [ ] **Step 4: Run seam tests and verify GREEN**

Run:

```bash
../../.venv/bin/python -m pytest \
  Tests/App/test_submit_library_ingest_job.py \
  Tests/Local_Ingestion/test_local_file_ingestion.py \
  Tests/Local_Ingestion/test_audio_model_dir_routing.py \
  Tests/Transcription/test_faster_whisper_transcription.py -q
```

Expected: all focused seam tests pass.

- [ ] **Step 5: Commit batch integration**

```bash
git add \
  tldw_chatbook/app.py \
  tldw_chatbook/Local_Ingestion/audio_processing.py \
  tldw_chatbook/Local_Ingestion/transcription_service.py \
  tldw_chatbook/Local_Ingestion/local_file_ingestion.py \
  Tests/App/test_submit_library_ingest_job.py \
  Tests/Local_Ingestion/test_local_file_ingestion.py \
  Tests/Local_Ingestion/test_audio_model_dir_routing.py \
  Tests/Transcription/test_faster_whisper_transcription.py
git commit -m "feat(ingestion): resolve gated Parakeet batch routes"
```

### Task 4: Verify, document, and close only the child task

**Files:**
- Modify: `Docs/Features/TRANSCRIPTION.md`
- Modify: `Docs/Features/TRANSCRIPTION_PROVIDERS.md`
- Modify: `backlog/tasks/task-602.1 - Stage-gated-Parakeet-v3-batch-routing.md`

- [ ] **Step 1: Update user-facing routing documentation**

Document:

- `en` remains the default requested language.
- Exact Parakeet English uses v2 INT8.
- Exact supported non-English uses v3 INT8 and does not enforce the selected language in the decoder.
- Semantic Parakeet defaults remain gated; auto, unsupported languages, and translation use faster-whisper under the approved policy.
- Batch transcription uses installed/local models only; Parakeet requires a user-selected existing local directory with the required filenames, and faster-whisper uses `local_files_only=True`, so a missing model fails clearly instead of downloading in a worker. The bounded receipt metadata check can reject a v2/v3 mismatch but does not authenticate or verify model contents.

- [ ] **Step 2: Run fresh focused verification**

Run outside the macOS sandbox because the installed optional MLX probe requires Metal:

```bash
../../.venv/bin/python -m pytest \
  Tests/Transcription/test_stt_batch_routing.py \
  Tests/Transcription/test_parakeet_onnx_vertical_slice.py \
  Tests/App/test_submit_library_ingest_job.py \
  Tests/Local_Ingestion/test_local_file_ingestion.py \
  Tests/Local_Ingestion/test_audio_model_dir_routing.py \
  Tests/Transcription/test_faster_whisper_transcription.py \
  Tests/Library/test_ingest_capabilities.py \
  Tests/UI/test_library_ingest_canvas.py \
  Tests/Audio/test_console_dictation.py \
  Tests/UI/test_console_dictation.py -q
```

Run static checks:

```bash
../../.venv/bin/python -m ruff check \
  tldw_chatbook/Local_Ingestion/stt_batch_routing.py \
  tldw_chatbook/Local_Ingestion/transcription_service.py \
  tldw_chatbook/Local_Ingestion/audio_processing.py \
  tldw_chatbook/app.py \
  tldw_chatbook/Local_Ingestion/local_file_ingestion.py \
  Tests/Transcription/test_stt_batch_routing.py \
  Tests/Transcription/test_parakeet_onnx_vertical_slice.py \
  Tests/App/test_submit_library_ingest_job.py \
  Tests/Local_Ingestion/test_local_file_ingestion.py \
  Tests/Local_Ingestion/test_audio_model_dir_routing.py \
  Tests/Transcription/test_faster_whisper_transcription.py
git diff --check
```

Expected: tests and static checks exit zero.

- [ ] **Step 3: Review acceptance criteria and record limits**

Check every TASK-602.1 criterion. Explicitly record that parent TASK-602 remains open for:

- promoted/managed artifact eligibility,
- app-owned `LocalSTTExecutor`,
- durable normalized provenance/retry lineage,
- managed long-form VAD and cancellation,
- the interactive retry action,
- Windows/Linux/platform matrix evidence.

- [ ] **Step 4: Complete TASK-602.1 only after evidence**

Add concise implementation notes, check every child criterion, and set TASK-602.1 to Done. Do not mark TASK-602 Done.

- [ ] **Step 5: Commit documentation and task completion**

```bash
git add \
  Docs/Features/TRANSCRIPTION.md \
  Docs/Features/TRANSCRIPTION_PROVIDERS.md \
  backlog/tasks/task-602.1\ -\ Stage-gated-Parakeet-v3-batch-routing.md
git commit -m "docs(stt): record gated v3 batch routing"
```
