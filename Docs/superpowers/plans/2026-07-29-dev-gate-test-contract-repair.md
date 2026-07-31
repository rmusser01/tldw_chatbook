# Dev-Gate Test Contract Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the confirmed obsolete or nondeterministic test failures from the current `dev` gate and safely refresh the reviewed diagnostic inventory.

**Architecture:** Delete the obsolete worker-event assertion, preserve the shell-test repair already present on current `dev`, and make both PyAudio loop tests synchronous and independent of optional VAD behavior. Review changed diagnostic owners against ADR-029 before updating the generated inventory.

**Tech Stack:** Python 3.11+, pytest, unittest.mock, Ruff.

**ADR required:** no

**ADR path:** backlog/decisions/029-local-private-data-boundary.md

**Reason:** This reconciles tests with accepted runtime contracts and applies ADR-029's existing metadata-only inventory boundary without introducing a new architectural decision.

---

## Scope guard

Do not restore `StreamDone` or `TabState`. Do not replace the latest `dev`
chat-shell fixture or duplicate the streaming-rejection test already present in
`Tests/Event_Handlers/test_retained_worker_adapter.py`. The only non-diagnostic
production edit allowed is adding the three missing registered run-log names to
the existing Library skill shadow set. Never regenerate the inventory before
reviewing its changed owners and sink topology.

## File map

- Modify `Tests/Event_Handlers/test_worker_events_contract.py`: retain only the
  unique non-streaming exception regression.
- Delete `Tests/Event_Handlers/test_worker_local_citation_capture.py`: all cases
  exercise retired worker streaming, sentinel, logging, or builder ownership.
- Verify `Tests/UI/test_chat_shell_bar.py`: preserve the current `dev` repair
  without a branch edit.
- Modify `Tests/Audio/test_audio_integration.py`: make stream-error behavior
  synchronous, VAD-independent, and exact.
- Modify `Tests/Audio/test_recording_service.py`: make PyAudio happy-flow
  behavior synchronous, VAD-independent, and exact.
- Modify `Tests/Chat/test_chat_functions.py`: patch the live runtime-config
  snapshot seam instead of deleted module-level settings.
- Modify `Tests/LLM/test_local_llm_provider_config.py`: patch the same live
  runtime-config snapshot seam for local-LLM provider configuration.
- Modify `Tests/LLM_Provider_Catalog/test_local_openai_compatible_provider_name.py`:
  place its local-LLM fixture under the live `api_settings` snapshot shape.
- Modify `Tests/Chat/test_scope_picker_listers.py`: create its fixture-owned
  trusted Notes base before service construction.
- Modify `Tests/Library/test_library_rag_scope.py`: create both fixture-owned
  trusted Notes bases before service construction.
- Modify `Tests/DB/test_rag_indexing_db.py`: retain large-batch correctness
  coverage without host-dependent wall-clock assertions.
- Modify `Tests/Event_Handlers/test_eval_db_operations_path.py`: create its
  fixture-owned retargeted profile directory before config selection.
- Modify `tldw_chatbook/Library/library_skills_state.py`: synchronize the fixed
  skill collision set with the three registered run-log runtime tools.
- Modify `Tests/Local_Ingestion/test_quick_ingest_db_path.py`: expect the
  canonical profile-aware media database fallback filename.
- Modify `Helper_Scripts/Benchmarks/rag_citation_provenance_benchmark.py`:
  create its isolated trusted config profile before application imports.
- Modify `Docs/security/production-diagnostic-inventory.json`: only after
  reviewing every generated owner/topology change against ADR-029.
- Modify TASK-1333 and this plan only for closeout evidence.

### Task 1: Remove the retired worker-event contract

**Files:**
- Modify: `Tests/Event_Handlers/test_worker_events_contract.py`
- Verify: `Tests/Event_Handlers/test_retained_worker_adapter.py`

- [x] **Step 1: Reproduce RED**

Run:

```bash
../../.venv/bin/python -m pytest \
  Tests/Event_Handlers/test_worker_events_contract.py -q
```

Expected: collection fails because `StreamDone` was deliberately removed.

- [x] **Step 2: Make the smallest test correction**

Update the module description so it claims only the retained non-streaming
failure contract. Import only `chat_wrapper_function`. Delete
`test_chat_wrapper_function_streaming_failure_keeps_sentinel_contract`; do not
replace it because
`test_retained_worker_adapter_rejects_legacy_streaming_bridge` already covers
the live streaming rejection.

- [x] **Step 3: Verify GREEN**

Run:

```bash
../../.venv/bin/python -m pytest \
  Tests/Event_Handlers/test_worker_events_contract.py \
  Tests/Event_Handlers/test_retained_worker_adapter.py -q
```

Expected: both modules pass.

- [x] **Step 4: Commit**

```bash
git add Tests/Event_Handlers/test_worker_events_contract.py
git commit -m "test(chat): remove retired worker-event contract"
```

### Task 1b: Remove obsolete worker-local citation ownership tests

**Files:**
- Delete: `Tests/Event_Handlers/test_worker_local_citation_capture.py`
- Verify: `Tests/Event_Handlers/test_retained_worker_adapter.py`
- Verify: `Tests/Event_Handlers/test_worker_events_contract.py`
- Verify: `Tests/Chat/test_console_local_citation_boundary.py`

- [ ] **Step 1: Reproduce RED**

Run the obsolete module. Expected: all four tests fail because they require the
retired streaming bridge, removed sentinel/error swallowing, or worker-owned
citation-builder stripping.

- [ ] **Step 2: Remove only dead coverage**

Delete the obsolete module. Do not add a compatibility shim: the retained
adapter contract already covers live delegation, streaming rejection, and
non-streaming failure propagation, while native Console owns citation lifetime
and privacy coverage.

- [ ] **Step 3: Verify and commit**

```bash
../../.venv/bin/python -m pytest \
  Tests/Event_Handlers/test_retained_worker_adapter.py \
  Tests/Event_Handlers/test_worker_events_contract.py \
  Tests/Chat/test_console_local_citation_boundary.py -q
git add Tests/Event_Handlers/test_worker_local_citation_capture.py
git commit -m "test(chat): remove retired worker citation tests"
```

### Task 2: Preserve the upstream chat-shell repair

**Files:**
- Verify: `Tests/UI/test_chat_shell_bar.py`

- [x] **Step 1: Rebase onto current `dev`**

Current `dev` independently removed `TabState` and added current persona-label
coverage. Drop the superseded TASK-1333 edit and retain upstream unchanged.

- [x] **Step 2: Verify GREEN**

Run:

```bash
../../.venv/bin/python -m pytest Tests/UI/test_chat_shell_bar.py -q
```

Expected: the module collects and passes.

### Task 3: Make the stream-error test deterministic

**Files:**
- Modify: `Tests/Audio/test_audio_integration.py`

- [x] **Step 1: Reproduce RED**

Run:

```bash
../../.venv/bin/python -m pytest \
  Tests/Audio/test_audio_integration.py::TestErrorRecovery::test_recording_recovery_from_stream_error \
  -q
```

Expected: the assertion sees zero callbacks when installed VAD rejects the
synthetic chunks; the test also races the recorder thread it starts.

- [x] **Step 2: Replace the nondeterministic test body**

Rename the test to
`test_recording_stops_after_stream_error_and_preserves_prior_chunks`. Keep the
existing PyAudio patches, then use this contract:

```python
chunk = b"\x00\x01" * 512
mock_stream.read.side_effect = [chunk, chunk, Exception("Stream error")]

service = AudioRecordingService(backend="pyaudio", use_vad=False)
chunks = []
service.callback = chunks.append
service.is_recording = True

service._pyaudio_recording_loop()

assert chunks == [chunk, chunk]
assert service.is_recording is False
mock_stream.stop_stream.assert_called_once_with()
mock_stream.close.assert_called_once_with()
assert service.stream is None
```

Do not call `start_recording()`, do not spawn a thread, and do not retain the
fourth post-error “recovery” chunk because production stops on the first stream
error.

- [x] **Step 3: Verify GREEN**

Run:

```bash
../../.venv/bin/python -m pytest \
  Tests/Audio/test_audio_integration.py::TestErrorRecovery::test_recording_stops_after_stream_error_and_preserves_prior_chunks \
  -q
```

Expected: the test passes deterministically.

- [x] **Step 4: Commit**

```bash
git add Tests/Audio/test_audio_integration.py
git commit -m "test(audio): make stream-error coverage deterministic"
```

### Task 4: Make the PyAudio flow test deterministic

**Files:**
- Modify: `Tests/Audio/test_recording_service.py`

- [ ] **Step 1: Reproduce RED**

Run:

```bash
../../.venv/bin/python -m pytest \
  Tests/Audio/test_recording_service.py::TestAudioRecordingIntegration::test_pyaudio_recording_flow \
  -q
```

Expected: the test can hang because two recording loops race while optional VAD
may reject the synthetic bytes.

- [ ] **Step 2: Replace the nondeterministic test body**

Construct `AudioRecordingService(backend="pyaudio", use_vad=False)`. Assign the
callback and `is_recording = True` directly, invoke
`_pyaudio_recording_loop()` once, and stop from the callback after exactly
three chunks. Assert `[test_audio] * 3`, stopped state, one `stop_stream()`, one
`close()`, and `service.stream is None`.

- [ ] **Step 3: Verify GREEN**

Run the focused test twice, then the full module:

```bash
../../.venv/bin/python -m pytest \
  Tests/Audio/test_recording_service.py::TestAudioRecordingIntegration::test_pyaudio_recording_flow \
  -q
../../.venv/bin/python -m pytest \
  Tests/Audio/test_recording_service.py::TestAudioRecordingIntegration::test_pyaudio_recording_flow \
  -q
../../.venv/bin/python -m pytest Tests/Audio/test_recording_service.py -q
```

Expected: all runs pass without threads or VAD-dependent behavior.

- [ ] **Step 4: Commit**

```bash
git add Tests/Audio/test_recording_service.py
git commit -m "test(audio): make pyaudio flow deterministic"
```

### Task 4b: Keep the SoundDevice fixture independent of VAD

**Files:**
- Modify: `Tests/Audio/test_recording_service.py`

- [ ] **Step 1: Reproduce RED**

Run the full recording-service module after Task 4. Expected: only
`test_sounddevice_recording_flow` fails because VAD discards its four-sample
synthetic callback.

- [ ] **Step 2: Make the fixture exact**

Construct the SoundDevice service with `use_vad=False`, retain its public
`start_recording()` path, and configure the `InputStream` mock with a side
effect that captures its callback and sets a `threading.Event`. Assert that
event becomes ready within a bounded timeout before invoking the callback.
Always call `stop_recording()` in `finally` so an assertion failure cannot leak
the background thread, then assert that the audio queue is non-empty.

- [ ] **Step 3: Verify and commit**

```bash
../../.venv/bin/python -m pytest \
  Tests/Audio/test_recording_service.py::TestAudioRecordingIntegration::test_sounddevice_recording_flow \
  -q
../../.venv/bin/python -m pytest Tests/Audio/test_recording_service.py -q
git add Tests/Audio/test_recording_service.py
git commit -m "test(audio): make sounddevice flow vad-independent"
```

### Task 4c: Patch the live provider-config seam

**Files:**
- Modify: `Tests/Chat/test_chat_functions.py`

- [ ] **Step 1: Reproduce RED**

Run the three parametrized Llama.cpp endpoint cases and
`test_deepseek_uses_refreshed_handler_fallback_model`. Expected: each fails
because the test tries to patch a deleted module-level `settings` object.

- [ ] **Step 2: Use the request-boundary contract**

Import `RuntimeConfigSnapshot`. For each test, construct a snapshot containing
the existing test's `api_settings` values and monkeypatch that adapter module's
`get_runtime_config_snapshot` function to return it. Remove comments claiming
the adapter reads module-level settings. Do not change production.

- [ ] **Step 3: Verify and commit**

```bash
../../.venv/bin/python -m pytest \
  Tests/Chat/test_chat_functions.py::test_chat_with_llama_posts_to_v1_chat_completions_regardless_of_suffix \
  Tests/Chat/test_chat_functions.py::TestProviderRequestPayloads::test_deepseek_uses_refreshed_handler_fallback_model \
  -q
../../.venv/bin/python -m pytest Tests/Chat/test_chat_functions.py -q
../../.venv/bin/python -m ruff check Tests/Chat/test_chat_functions.py
../../.venv/bin/python -m ruff format --check Tests/Chat/test_chat_functions.py
git add Tests/Chat/test_chat_functions.py
git commit -m "test(chat): patch runtime config snapshots"
```

### Task 4c2: Patch the local-LLM runtime-config seam

**Files:**
- Modify: `Tests/LLM/test_local_llm_provider_config.py`
- Modify: `Tests/LLM_Provider_Catalog/test_local_openai_compatible_provider_name.py`

- [ ] **Step 1: Reproduce RED**

Run the module. Expected: all four cases fail because they try to monkeypatch
the deleted `LLM_API_Calls_Local.settings` object.

- [ ] **Step 2: Use the request-boundary contract**

Import `RuntimeConfigSnapshot`. Replace the settings helper with a snapshot
helper and monkeypatch `local_calls.get_runtime_config_snapshot` in each case.
Preserve the documented `api_url`, legacy `api_ip`, precedence, and missing-URL
assertions. In the provider-name regression, move the existing local-LLM
fixture beneath `api_settings` while preserving its string provider-name
assertion. Do not change production.

- [ ] **Step 3: Verify and commit**

```bash
../../.venv/bin/python -m pytest \
  Tests/LLM/test_local_llm_provider_config.py \
  Tests/LLM_Provider_Catalog/test_local_openai_compatible_provider_name.py -q
../../.venv/bin/python -m ruff check \
  Tests/LLM/test_local_llm_provider_config.py \
  Tests/LLM_Provider_Catalog/test_local_openai_compatible_provider_name.py
../../.venv/bin/python -m ruff format --check \
  Tests/LLM/test_local_llm_provider_config.py \
  Tests/LLM_Provider_Catalog/test_local_openai_compatible_provider_name.py
git add \
  Tests/LLM/test_local_llm_provider_config.py \
  Tests/LLM_Provider_Catalog/test_local_openai_compatible_provider_name.py
git commit -m "test(llm): patch local runtime config snapshot"
```

### Task 4d: Create fixture-owned trusted Notes roots

**Files:**
- Modify: `Tests/Chat/test_scope_picker_listers.py`
- Modify: `Tests/Library/test_library_rag_scope.py`

- [ ] **Step 1: Reproduce RED**

Run the first lister test plus one test using each Library Notes fixture.
Expected: all three fail during setup because `NotesInteropService` correctly
rejects a missing `notes_base` directory.

- [ ] **Step 2: Satisfy the security contract in the fixtures**

In each of the three fixtures, assign `notes_base = tmp_path / "notes_base"`,
call `notes_base.mkdir(mode=0o700)`, and pass that existing path to
`NotesInteropService`. In each fixture teardown, call
`close_all_user_connections()` before closing the template
`CharactersRAGDB`. Do not change production directory verification.

- [ ] **Step 3: Verify and commit**

```bash
../../.venv/bin/python -m pytest \
  Tests/Chat/test_scope_picker_listers.py \
  Tests/Library/test_library_rag_scope.py -q
../../.venv/bin/python -m ruff check \
  Tests/Chat/test_scope_picker_listers.py \
  Tests/Library/test_library_rag_scope.py
../../.venv/bin/python -m ruff format --check \
  Tests/Chat/test_scope_picker_listers.py \
  Tests/Library/test_library_rag_scope.py
git add \
  Tests/Chat/test_scope_picker_listers.py \
  Tests/Library/test_library_rag_scope.py
git commit -m "test(notes): create trusted fixture roots"
```

### Task 4e: Remove host-dependent timing from large-batch correctness coverage

**Files:**
- Modify: `Tests/DB/test_rag_indexing_db.py`

- [ ] **Step 1: Record RED and contention evidence**

Use the repository-wide fail-fast run as RED: the unchanged test indexed all
1,000 items correctly but failed because elapsed wall time was 24.98 seconds
under concurrent test load. Run the exact test alone and record that it passes
without a code change, demonstrating a load-sensitive assertion.

- [ ] **Step 2: Keep behavior coverage and remove the benchmark**

Rename the test to describe large-batch persistence, retain its 1,000 writes,
full-count assertion, and retrieval assertion, and delete only the elapsed-time
measurements and thresholds. Do not change `RAGIndexingDB` production code or
replace the limits with a different arbitrary timeout.

- [ ] **Step 3: Verify and commit**

```bash
../../.venv/bin/python -m pytest \
  Tests/DB/test_rag_indexing_db.py -q
../../.venv/bin/python -m ruff check \
  Tests/DB/test_rag_indexing_db.py
../../.venv/bin/python -m ruff format --check \
  Tests/DB/test_rag_indexing_db.py
git add Tests/DB/test_rag_indexing_db.py
git commit -m "test(db): remove load-sensitive timing gate"
```

### Task 4f: Create the retargeted Evals profile fixture root

**Files:**
- Modify: `Tests/Event_Handlers/test_eval_db_operations_path.py`

- [ ] **Step 1: Reproduce RED**

Run `test_default_db_path_tracks_a_retargeted_profile`.
Expected: setup fails with `PrivatePathError` and reason `missing_parent`
because the fixture selects `profile-two/config.toml` without creating
`profile-two`.

- [ ] **Step 2: Satisfy the trusted-parent contract in the fixture**

Assign the profile directory separately, create it with `mode=0o700`, then
derive `config.toml` beneath it before setting `TLDW_CONFIG_PATH`. Wrap the
assertions in `try/finally` and close `ops.db` so Windows can remove the
temporary SQLite/WAL files deterministically. Do not change production config
or private-path behavior.

- [ ] **Step 3: Verify and commit**

```bash
../../.venv/bin/python -m pytest \
  Tests/Event_Handlers/test_eval_db_operations_path.py -q
../../.venv/bin/python -m ruff check \
  Tests/Event_Handlers/test_eval_db_operations_path.py
../../.venv/bin/python -m ruff format --check \
  Tests/Event_Handlers/test_eval_db_operations_path.py
git add Tests/Event_Handlers/test_eval_db_operations_path.py
git commit -m "test(evals): create retargeted profile root"
```

### Task 4g: Synchronize Library skill shadow names

**Files:**
- Modify: `tldw_chatbook/Library/library_skills_state.py`
- Verify: `Tests/Library/test_library_skills_state.py`

- [ ] **Step 1: Reproduce RED**

Run `test_shadow_name_set_stays_in_sync_with_real_sources`. Expected: the
subset assertion reports `search_run_log`, `run_log_stats`, and
`run_log_slice` missing from `_SHADOWED_BUILTIN_NAMES`.

- [ ] **Step 2: Add only registered collision names**

Add those three runtime tool names to the fixed literal shadow set with one
brief comment tying them to the drift guard. Do not import agent runtime
modules or refactor the collision boundary.

- [ ] **Step 3: Verify and commit**

```bash
../../.venv/bin/python -m pytest \
  Tests/Library/test_library_skills_state.py -q
../../.venv/bin/python -m ruff check \
  tldw_chatbook/Library/library_skills_state.py \
  Tests/Library/test_library_skills_state.py
../../.venv/bin/python -m ruff format --check \
  tldw_chatbook/Library/library_skills_state.py \
  Tests/Library/test_library_skills_state.py
git add tldw_chatbook/Library/library_skills_state.py
git commit -m "fix(library): reserve run-log tool names"
```

### Task 4h: Align quick-ingest fallback filename coverage

**Files:**
- Modify: `Tests/Local_Ingestion/test_quick_ingest_db_path.py`

- [ ] **Step 1: Reproduce RED**

Run the module. Expected: only
`test_fallback_applies_only_when_the_key_is_absent` fails because it expects the
retired `tldw_cli_media_v2.db` basename.

- [ ] **Step 2: Update only the canonical expectation**

Assert the fallback path uses `tldw_chatbook_media_v2.db`. Keep the configured
custom path, expanded-home, and traversal-rejection assertions unchanged. Do
not change production.

- [ ] **Step 3: Verify and commit**

```bash
../../.venv/bin/python -m pytest \
  Tests/Local_Ingestion/test_quick_ingest_db_path.py \
  Tests/Local_Ingestion/test_local_file_ingestion.py -q
../../.venv/bin/python -m ruff check \
  Tests/Local_Ingestion/test_quick_ingest_db_path.py
../../.venv/bin/python -m ruff format --check \
  Tests/Local_Ingestion/test_quick_ingest_db_path.py
git add Tests/Local_Ingestion/test_quick_ingest_db_path.py
git commit -m "test(ingest): expect canonical media DB fallback"
```

### Task 4i: Create the benchmark's isolated config profile

**Files:**
- Modify: `Helper_Scripts/Benchmarks/rag_citation_provenance_benchmark.py`
- Modify: `Tests/Performance/test_rag_citation_provenance_benchmark.py`

- [ ] **Step 1: Reproduce RED**

Run `test_cli_never_reads_or_writes_host_config_data_or_secrets`. Expected: the
subprocess exits with `unsafe_parent: missing_parent` because
`isolated_benchmark_host_state()` does not create `config/tldw_cli`.

- [ ] **Step 2: Satisfy the trusted-profile contract**

Create `config_root / "tldw_cli"` with `mode=0o700, exist_ok=True` before
building the environment overrides, then derive `TLDW_CONFIG_PATH` from that
directory. The context must remain reusable with the same scratch root. Do not
change application private-path behavior or the benchmark's environment
redaction. Exercise the existing isolation/restoration assertions twice against
the same root so reentrancy stays covered.

- [ ] **Step 3: Verify and commit**

```bash
../../.venv/bin/python -m pytest \
  Tests/Performance/test_rag_citation_provenance_benchmark.py -q
../../.venv/bin/python -m ruff check \
  Helper_Scripts/Benchmarks/rag_citation_provenance_benchmark.py \
  Tests/Performance/test_rag_citation_provenance_benchmark.py
../../.venv/bin/python -m ruff format --check \
  Helper_Scripts/Benchmarks/rag_citation_provenance_benchmark.py \
  Tests/Performance/test_rag_citation_provenance_benchmark.py
git add Helper_Scripts/Benchmarks/rag_citation_provenance_benchmark.py
git commit -m "test(benchmark): create isolated config profile"
```

### Task 4j: Render the visible Console Stop control before clicking

**Files:**
- Modify: `Tests/ProductionApp/test_chat_root_state_removal.py`

- [ ] **Step 1: Reproduce RED**

Run
`test_visible_console_stop_cancels_native_run_without_root_worker_state`.
Expected: the provider stream starts, but the pointer click never reaches the
screen handler and the stream is only cancelled during teardown. The test
observes `stop_button.display` using `asyncio.sleep`, which does not guarantee
that Textual has completed the corresponding layout and hit-test refresh.

- [ ] **Step 2: Render before the user action**

Advance the Textual pilot once after the Stop control becomes visible and
before `pilot.click()`. Keep the visible pointer action, provider cancellation,
stopped controller state, and exact partial-response assertions. Do not call
the controller directly, change production cancellation, or mask the failure
with a larger timeout.

- [ ] **Step 3: Verify and commit**

```bash
../../.venv/bin/python -m pytest \
  Tests/ProductionApp/test_chat_root_state_removal.py -q
../../.venv/bin/python -m ruff check \
  Tests/ProductionApp/test_chat_root_state_removal.py
../../.venv/bin/python -m ruff format --check \
  Tests/ProductionApp/test_chat_root_state_removal.py
git add Tests/ProductionApp/test_chat_root_state_removal.py
git commit -m "test(console): render stop control before click"
```

### Task 4k: Wait for replaced Media owners to finish teardown

**Files:**
- Modify: `Tests/ProductionApp/test_media_state_ownership.py`

- [ ] **Step 1: Reproduce RED**

Run
`test_real_metadata_ordering_survives_media_window_replacement` and
`test_real_metadata_mutation_survives_media_screen_teardown`. Expected: each
reaches the incoming Settings screen before the outgoing `MediaWindow` has
finished Textual's asynchronous close/detach lifecycle, so the immediate
`_closed` assertion fails.

- [ ] **Step 2: Await the lifecycle contract**

Replace each immediate close assertion with the existing bounded
`_wait_until(pilot, ...)` helper, waiting until the outgoing window is both
closed and detached. Retain the fresh replacement-instance assertion, the
blocked-old/new-write ordering, and the durable last-edit-wins result. Do not
edit production, add an arbitrary sleep, or remove stale-owner checks.

- [ ] **Step 3: Verify and commit**

```bash
../../.venv/bin/python -m pytest \
  Tests/ProductionApp/test_media_state_ownership.py -q
../../.venv/bin/python -m ruff check \
  Tests/ProductionApp/test_media_state_ownership.py
../../.venv/bin/python -m ruff format --check \
  Tests/ProductionApp/test_media_state_ownership.py
git add Tests/ProductionApp/test_media_state_ownership.py
git commit -m "test(media): await replaced owner teardown"
```

### Task 4l: Settle provider-selection Settings state

**Files:**
- Modify: `Tests/ProductionApp/test_provider_selection_ownership.py`

- [ ] **Step 1: Reproduce RED**

Run
`test_settings_save_preserves_user_session_then_away_command_hands_off`.
Expected: querying the provider control immediately after the category becomes
active can race its recompose and raise `NoMatches`. Waiting only for the
control then exposes a second ordering failure when the full module runs:
programmatic `.value` assignments have not reliably reached the Settings
staging handlers before save, leaving the old OpenAI defaults.

- [ ] **Step 2: Await and observe the live Settings seams**

Use the existing `_wait_for_widget` helper for the provider and model controls.
After each programmatic assignment, invoke the corresponding live `Changed`
handler and use one small bounded pilot-driven predicate helper to observe the
staged provider/model values. After Save, wait for both app defaults to update.
Keep the existing user-session preservation and later provider-handoff
assertions unchanged. Do not edit production or add arbitrary pauses.

- [ ] **Step 3: Verify and commit**

```bash
../../.venv/bin/python -m pytest \
  Tests/ProductionApp/test_provider_selection_ownership.py -q
../../.venv/bin/python -m ruff check \
  Tests/ProductionApp/test_provider_selection_ownership.py
../../.venv/bin/python -m ruff format --check \
  Tests/ProductionApp/test_provider_selection_ownership.py
git add Tests/ProductionApp/test_provider_selection_ownership.py
git commit -m "test(settings): await provider selection state"
```

### Task 4m: Remove obsolete fail-open RAG UI coverage

**Files:**
- Modify: `Tests/RAG/test_rag_ui_integration.py`
- Modify: `tldw_chatbook/Event_Handlers/Chat_Events/chat_rag_events.py`

- [ ] **Step 1: Reproduce RED and confirm the live owner**

Run
`test_capture_unavailable_keeps_ui_pipeline_context_and_legacy_string`.
Expected: production returns `LocalRagContextResult(None, None)` with
`reason=prompt_authority_failure`, while the stale UI integration test expects
raw context for a recognized media candidate. Confirm
`Tests/RAG/test_local_citation_capture.py` already covers prompt-authority
failure, current-authority exclusion without a builder, and the narrow
unsupported-source legacy fallback.

- [ ] **Step 2: Remove only obsolete coverage**

Delete `CaptureUnavailableApp` and its sole test. Clarify the public capture
function's stale docstring: recognized canonical candidates require completed
current prompt authority even when no builder exists; only unsupported results
may retain raw legacy pipeline context. Do not change production behavior,
restore raw recognized candidates to prompt context, or duplicate the focused
citation-capture contracts in this legacy UI integration module.

- [ ] **Step 3: Verify and commit**

```bash
../../.venv/bin/python -m pytest \
  Tests/RAG/test_rag_ui_integration.py \
  Tests/RAG/test_local_citation_capture.py -q
../../.venv/bin/python -m ruff check \
  Tests/RAG/test_rag_ui_integration.py \
  tldw_chatbook/Event_Handlers/Chat_Events/chat_rag_events.py
../../.venv/bin/python -m ruff format --check \
  Tests/RAG/test_rag_ui_integration.py \
  tldw_chatbook/Event_Handlers/Chat_Events/chat_rag_events.py
git add \
  Tests/RAG/test_rag_ui_integration.py \
  tldw_chatbook/Event_Handlers/Chat_Events/chat_rag_events.py
git commit -m "test(rag): remove fail-open UI fallback"
```

### Task 4n: Publish the lazy RAG-admin fixture's runtime state

**Files:**
- Modify: `Tests/RAG_Admin/test_app_lazy_rag_admin_wiring.py`

- [ ] **Step 1: Reproduce RED**

Run `Tests/RAG_Admin/test_app_lazy_rag_admin_wiring.py`. Expected:
`fake_runtime_policy` raises `AttributeError` while assigning the read-only
`TldwCli.current_runtime_backend` property before the lazy-service assertions
can run.

- [ ] **Step 2: Use the live runtime-policy owner**

Replace the two direct compatibility-field assignments with
`app._publish_runtime_policy_projection(context.state)`. Keep the fake state
and every lazy construction, cache, fallback, and wiring assertion unchanged.
Do not edit production or restore a writable compatibility projection.

- [ ] **Step 3: Verify and commit**

```bash
../../.venv/bin/python -m pytest \
  Tests/RAG_Admin/test_app_lazy_rag_admin_wiring.py -q
../../.venv/bin/python -m ruff check \
  Tests/RAG_Admin/test_app_lazy_rag_admin_wiring.py
../../.venv/bin/python -m ruff format --check \
  Tests/RAG_Admin/test_app_lazy_rag_admin_wiring.py
git add Tests/RAG_Admin/test_app_lazy_rag_admin_wiring.py
git commit -m "test(rag): publish fixture runtime state"
```

### Task 4o: Accept complete legacy-backup cleanup

**Files:**
- Modify: `Tests/TTS/test_profile_backup_integration.py`

- [ ] **Step 1: Reproduce RED and confirm the live contract**

Run
`test_real_worker_cancellation_before_legacy_publication_leaves_no_artifact`.
Expected: the `.db` and `.tmp` no-artifact assertions pass, while an
unconditional `backup_root.iterdir()` raises `FileNotFoundError` because
production successfully removed the empty directory. Confirm the later legacy
worker-failure regression has the same stale directory-presence assumption.

- [ ] **Step 2: Assert absence or emptiness**

Change only those two empty-root assertions to accept either a missing backup
root or an existing root with no children. Retain the artifact, manifest,
notification, privacy, cancellation, and in-progress-state assertions. Do not
edit production backup cleanup or create a directory solely for the tests.

- [ ] **Step 3: Verify and commit**

```bash
../../.venv/bin/python -m pytest \
  Tests/TTS/test_profile_backup_integration.py -q
../../.venv/bin/python -m ruff check \
  Tests/TTS/test_profile_backup_integration.py
../../.venv/bin/python -m ruff format --check \
  Tests/TTS/test_profile_backup_integration.py
git add Tests/TTS/test_profile_backup_integration.py
git commit -m "test(backup): accept removed empty root"
```

### Task 4p: Follow the profile-aware backup root

**Files:**
- Modify: `Tests/TTS/test_profile_backup_integration.py`

- [ ] **Step 1: Reproduce RED and identify vacuous checks**

Run the full profile-backup integration module. Expected: successful and
partial backups are written below the live profile-aware user-data root, while
tests inspect the retired unscoped path. Confirm all seven hard-coded
`tmp_path/.local/share/tldw_cli/backups` assignments and the one equivalent
destination-parent comparison, including the two Task 4o assertions.

- [ ] **Step 2: Derive expectations from the live owner**

Replace those retired path expectations with
`tools_settings_module.get_user_data_dir() / "backups"`. Keep each test's
existing distinct-directory, manifest-content, partial-failure, publication,
cancellation, notification, and cleanup assertions. Do not duplicate the
profile-path algorithm, patch production path selection, or create a test-only
compatibility root.

- [ ] **Step 3: Verify and commit**

```bash
../../.venv/bin/python -m pytest \
  Tests/TTS/test_profile_backup_integration.py -q
../../.venv/bin/python -m ruff check \
  Tests/TTS/test_profile_backup_integration.py
../../.venv/bin/python -m ruff format --check \
  Tests/TTS/test_profile_backup_integration.py
git add Tests/TTS/test_profile_backup_integration.py
git commit -m "test(backup): follow profile-aware root"
```

### Task 4q: Retarget secure manifest staging regressions

**Files:**
- Modify: `Tests/TTS/test_profile_backup_integration.py`

- [ ] **Step 1: Reproduce RED and confirm the secure seam**

Confirm real stage creation calls the imported `create_private_text` helper
inside the manifest worker and no longer calls `Path.open` on guarded
platforms. Confirm JSON serialization failures occur before any stage is
created, so the two cleanup-precedence tests currently never reach unlink.
Confirm the replace-failure privacy test rejects unrelated isolated config
bootstrap paths rather than an exposed injected backup value.

- [ ] **Step 2: Exercise the live boundaries**

Wrap `tools_settings_module.create_private_text` to record its stage path,
thread, and active task while delegating to the real helper. Assert one
`.backup_info.json.tmp` creation off the owner thread with no asyncio task.
For cleanup precedence, allow serialization and secure creation to complete,
then raise the intended ordinary or control-flow exception from the second
`_raise_if_textual_worker_cancelled` call so cleanup receives an existing
stage. Narrow the replace-failure privacy check to the injected private
manifest path. Do not edit production, emulate `Path.open`, or weaken the
exclusive-create, cleanup, notification, and value-free diagnostic contracts.

- [ ] **Step 3: Verify and commit**

```bash
../../.venv/bin/python -m pytest \
  Tests/TTS/test_profile_backup_integration.py -q
../../.venv/bin/python -m ruff check \
  Tests/TTS/test_profile_backup_integration.py
../../.venv/bin/python -m ruff format --check \
  Tests/TTS/test_profile_backup_integration.py
git add Tests/TTS/test_profile_backup_integration.py
git commit -m "test(backup): target secure manifest staging"
```

### Task 4r: Inject backup sources through canonical resolvers

**Files:**
- Modify: `Tests/TTS/test_profile_backup_integration.py`

- [ ] **Step 1: Reproduce RED and confirm resolver ownership**

Run the full profile-backup integration module. Expected: the success manifest
contains only Prompts plus TTS Profiles and both partial-failure manifests
contain only Prompts. Confirm `_get_database_path()` intentionally ignores
`db_config` and reads ChaChaNotes and Media resolvers from the window's
`_DB_PATH_RESOLVERS` map. Confirm `_backup_worker` resolves Prompts directly
through the imported `get_prompts_db_path` symbol, so the existing module patch
is effective and must remain.

- [ ] **Step 2: Inject isolated sources through the live seam**

In both backup setup helpers, install instance-level resolver maps that retain
the production map but replace ChaChaNotes and Media with callables returning
the fixture's temporary databases. Retain the direct module-level Prompts
resolver patch. Preserve real copying, profile backup, manifest contents,
partial-failure behavior, and path/privacy assertions. Do not make production
consult `db_config` again, route Prompts through a different production seam,
or patch canonical config globally.

- [ ] **Step 3: Verify and commit**

```bash
../../.venv/bin/python -m pytest \
  Tests/TTS/test_profile_backup_integration.py -q
../../.venv/bin/python -m ruff check \
  Tests/TTS/test_profile_backup_integration.py
../../.venv/bin/python -m ruff format --check \
  Tests/TTS/test_profile_backup_integration.py
git add Tests/TTS/test_profile_backup_integration.py
git commit -m "test(backup): inject canonical source resolvers"
```

### Task 4s: Drop the deleted TTS preference write guard

**Files:**
- Modify: `Tests/TTS/test_tts_preferences.py`

- [ ] **Step 1: Reproduce RED**

Run
`test_reading_legacy_blanks_does_not_mutate_input_or_write_disk`. Expected:
setup raises `AttributeError` while monkeypatching deleted
`config.atomic_write_text`, before the pure parser or its assertions run.
Confirm the test still patches the four live public config mutation helpers and
that `TTSPreferencesSnapshot.from_settings()` does not own persistence.

- [ ] **Step 2: Remove only the dead symbol**

Delete `atomic_write_text` from the helper-name tuple. Retain the unexpected
persistence callback, the zero-call assertion, and the deep input-equality
assertion. Do not restore a config alias or patch private atomic-write
internals.

- [ ] **Step 3: Verify and commit**

```bash
../../.venv/bin/python -m pytest \
  Tests/TTS/test_tts_preferences.py -q
../../.venv/bin/python -m ruff check \
  Tests/TTS/test_tts_preferences.py
../../.venv/bin/python -m ruff format --check \
  Tests/TTS/test_tts_preferences.py
git add Tests/TTS/test_tts_preferences.py
git commit -m "test(tts): guard live preference writers"
```

### Task 4t: Do not synthesize empty Parakeet MLX segments

**Files:**
- Modify: `tldw_chatbook/Local_Ingestion/transcription_service.py`
- Modify: `Tests/Transcription/test_mlx_parakeet_edge_cases.py`
- Verify: `Tests/Transcription/test_mlx_parakeet_integration.py`

- [ ] **Step 1: Preserve the existing RED regression**

Run
`TestMLXParakeetIntegration::test_empty_audio`. Expected: the model returns
empty text and no sentences, but normalization creates one zero-duration empty
segment because `audio_duration` is `0.0` rather than `None`. Confirm
faster-whisper and MLX Whisper already return zero segments for empty text and
that non-empty text without timestamps still receives a single fallback.
Confirm `test_extreme_audio_lengths` has the same stale expectation for its
mocked 10 ms empty transcription.

- [ ] **Step 2: Correct the fallback condition**

In the no-sentences branch, create a fallback segment only when `text` is
non-empty. Otherwise return no segments. Update the adjacent stale comment and
the very-short empty-result assertion from one segment to zero. Do not change
sentence timestamp conversion, chunking, model invocation, metadata, routing,
or error handling.

- [ ] **Step 3: Verify and commit**

```bash
../../.venv/bin/python -m pytest \
  Tests/Transcription/test_mlx_parakeet_integration.py \
  Tests/Transcription/test_mlx_parakeet_edge_cases.py -q
../../.venv/bin/python -m ruff check \
  tldw_chatbook/Local_Ingestion/transcription_service.py \
  Tests/Transcription/test_mlx_parakeet_integration.py \
  Tests/Transcription/test_mlx_parakeet_edge_cases.py
../../.venv/bin/python -m ruff format --check \
  tldw_chatbook/Local_Ingestion/transcription_service.py \
  Tests/Transcription/test_mlx_parakeet_integration.py \
  Tests/Transcription/test_mlx_parakeet_edge_cases.py
git add \
  tldw_chatbook/Local_Ingestion/transcription_service.py \
  Tests/Transcription/test_mlx_parakeet_edge_cases.py
git commit -m "fix(stt): omit empty Parakeet MLX segment"
```

### Task 4u: Short-circuit zero-frame Parakeet MLX input

**Files:**
- Modify: `tldw_chatbook/Local_Ingestion/transcription_service.py`
- Modify: `Tests/Transcription/test_mlx_parakeet_integration.py`
- Modify: `Tests/Transcription/test_mlx_parakeet_transcription.py`

- [ ] **Step 1: Preserve real-input RED**

Run
`TestMLXParakeetIntegration::test_real_transcription_empty_file` from
`test_mlx_parakeet_transcription.py`. On a Mac with Parakeet MLX installed,
expected RED is a Metal allocation error caused by inference on a zero-frame
tensor. Confirm the mocked zero-frame integration currently invokes its loader,
while the 10 ms edge case remains available to test empty post-model output.

- [ ] **Step 2: Return an empty result before model load**

After resolving model/precision/attention settings but before acquiring the
model-load lock, inspect `sf.info()` when available. If the valid container
reports zero frames or exactly zero duration, return an empty Parakeet MLX
result with standard provider metadata, normalized default/request chunk
settings, sample rate, duration `0.0`, and a final progress update. Do not load
or invoke the model. If probing raises, continue through the existing path so
invalid files still fail normally. Simplify the mocked zero-frame test to
use explicit model/precision/attention and invalid chunk/overlap overrides,
install a cached-model spy, and assert the exact normalized result,
initial/final progress, zero loader calls, and zero cached-model inference;
strengthen the real integration to assert no segments. Do not alter non-empty
decoding, chunking, routing, or fallback behavior.

- [ ] **Step 3: Verify and commit**

```bash
../../.venv/bin/python -m pytest \
  Tests/Transcription/test_mlx_parakeet_integration.py \
  Tests/Transcription/test_mlx_parakeet_edge_cases.py \
  Tests/Transcription/test_mlx_parakeet_transcription.py -q
../../.venv/bin/python -m ruff check \
  tldw_chatbook/Local_Ingestion/transcription_service.py \
  Tests/Transcription/test_mlx_parakeet_integration.py \
  Tests/Transcription/test_mlx_parakeet_transcription.py
../../.venv/bin/python -m ruff format --check \
  tldw_chatbook/Local_Ingestion/transcription_service.py \
  Tests/Transcription/test_mlx_parakeet_integration.py \
  Tests/Transcription/test_mlx_parakeet_transcription.py
git add \
  tldw_chatbook/Local_Ingestion/transcription_service.py \
  Tests/Transcription/test_mlx_parakeet_integration.py \
  Tests/Transcription/test_mlx_parakeet_transcription.py
git commit -m "fix(stt): short-circuit empty Parakeet MLX input"
```

### Task 4v: Isolate the no-SoundFile Parakeet regression

**Files:**
- Modify: `Tests/Transcription/test_mlx_parakeet_transcription.py`

- [ ] **Step 1: Reproduce the host-dependent path**

Run `TestMLXParakeetUnit::test_soundfile_not_available` with SoundFile and
Parakeet MLX installed. Expected: patching only `SOUNDFILE_AVAILABLE` leaves
the imported `sf` module live, so the nonexistent `dummy.wav` path bypasses the
intended guard and attempts a real Hugging Face model load.

- [ ] **Step 2: Patch the complete live dependency seam**

Patch `tldw_chatbook.Local_Ingestion.transcription_service.sf` to `None` in
the existing test alongside the false availability flag. Retain the
`TranscriptionError` assertion. Do not modify production or contact the
network.

- [ ] **Step 3: Verify and commit**

```bash
../../.venv/bin/python -m pytest \
  Tests/Transcription/test_mlx_parakeet_transcription.py::TestMLXParakeetUnit::test_soundfile_not_available -q
../../.venv/bin/python -m ruff check \
  Tests/Transcription/test_mlx_parakeet_transcription.py
../../.venv/bin/python -m ruff format --check \
  Tests/Transcription/test_mlx_parakeet_transcription.py
git add Tests/Transcription/test_mlx_parakeet_transcription.py
git commit -m "test(stt): isolate missing SoundFile path"
```

### Task 4w: Retarget command-palette provider ownership tests

**Files:**
- Modify: `Tests/UI/test_command_palette_providers.py`

- [ ] **Step 1: Reproduce the retired-root failure**

Run the command-palette provider module. Expected: the show-current regression
assigns `chat_api_provider_value`, but production ignores that retired root and
returns `Unknown`; the switch regression also asserts the deleted ownership
path instead of the pending-handoff flow.

- [ ] **Step 2: Exercise the current ownership seams**

Give the provider a mounted Console test double. Have its
`current_console_provider_for_command()` method return `OpenAI` for the
show-current case. For the switch case, assert the exact
`ConsoleProviderIntent` is staged on `HandoffChannel.CONSOLE_PROVIDER` and the
mounted Console consumes it. Do not modify production or add compatibility
state.

- [ ] **Step 3: Verify and commit**

```bash
../../.venv/bin/python -m pytest Tests/UI/test_command_palette_providers.py -q
../../.venv/bin/python -m ruff check Tests/UI/test_command_palette_providers.py
../../.venv/bin/python -m ruff format --check Tests/UI/test_command_palette_providers.py
git add Tests/UI/test_command_palette_providers.py
git commit -m "test(ui): follow console provider ownership"
```

### Task 4x: Observe batched Library ingest option persistence

**Files:**
- Modify: `Tests/integration/test_library_ingest_flow.py`

- [ ] **Step 1: Reproduce the retired per-key mock**

Run `test_options_persist_to_config`. Expected: the test patches
`save_setting_to_cli_config`, but submission uses the live batched
`save_settings_to_cli_config`, so the obsolete mock records no calls.

- [ ] **Step 2: Patch and assert the live batch seam**

Capture the mapping passed to `save_settings_to_cli_config`. Assert it is
called once and that the same batch contains the PDF engine plus generic
`chunk=True` and normalized `chunk_size=1024` values. Do not change production
or restore key-by-key config writes.

- [ ] **Step 3: Verify and commit**

```bash
../../.venv/bin/python -m pytest Tests/integration/test_library_ingest_flow.py -q
../../.venv/bin/python -m ruff check Tests/integration/test_library_ingest_flow.py
../../.venv/bin/python -m ruff format --check Tests/integration/test_library_ingest_flow.py
git add Tests/integration/test_library_ingest_flow.py
git commit -m "test(library): observe batched ingest settings"
```

### Task 4y: Follow the private config replacement owner

**Files:**
- Modify: `Tests/test_config_delete_settings.py`
- Modify: `Tests/UI/test_product_maturity_phase6_packaging_data_safety.py`

- [ ] **Step 1: Reproduce the deleted writer seam**

Run the module. Expected: the first structured-mutation regression fails while
reading deleted `config.atomic_write_text`; later write-count, failure, lock,
batch-save, and delete-wrapper tests reference the same retired symbol. The
Phase 6.6 source-seam regression separately requires that deleted generic
writer against the hard-coded default path.

- [ ] **Step 2: Retarget all replacement instrumentation**

Wrap or replace `config.atomic_private_write_text` in each affected regression.
Keep all existing call-count, no-write, failure-phase, lock-order, file-content,
and permission assertions. In the Phase 6.6 source-seam regression, require the
private writer and its application-owned-directory argument within the
`_write_raw_cli_config_unlocked` function block while retaining the
effective-path checks; do not use independent whole-file substring assertions
that unrelated snapshot/bootstrap calls could satisfy. Do not restore
`atomic_write_text`, hard-code `DEFAULT_CONFIG_PATH`, or modify production.

- [ ] **Step 3: Verify and commit**

```bash
../../.venv/bin/python -m pytest \
  Tests/test_config_delete_settings.py \
  Tests/UI/test_product_maturity_phase6_packaging_data_safety.py -q
../../.venv/bin/python -m ruff check \
  Tests/test_config_delete_settings.py \
  Tests/UI/test_product_maturity_phase6_packaging_data_safety.py
../../.venv/bin/python -m ruff format --check \
  Tests/test_config_delete_settings.py \
  Tests/UI/test_product_maturity_phase6_packaging_data_safety.py
git add \
  Tests/test_config_delete_settings.py \
  Tests/UI/test_product_maturity_phase6_packaging_data_safety.py
git commit -m "test(config): follow private atomic writer"
```

### Task 4z: Align Console unknown-command hint copy

**Files:**
- Modify: `Tests/UI/test_console_command_composer.py`

- [ ] **Step 1: Reproduce the stale expected hint**

Run the module. Expected: the first unknown-command case renders the live
registry list with `/generate-image` and `/rewind`, while both expected
constants stop after `/prefill`.

- [ ] **Step 2: Update the independent expected copy**

Append `/generate-image, /rewind` to both curated expected strings in registry
order. Keep every interaction and message-count assertion. Do not call the
production formatter to manufacture the expected value.

- [ ] **Step 3: Verify and commit**

```bash
../../.venv/bin/python -m pytest Tests/UI/test_console_command_composer.py -q
../../.venv/bin/python -m ruff check Tests/UI/test_console_command_composer.py
../../.venv/bin/python -m ruff format --check Tests/UI/test_console_command_composer.py
git add Tests/UI/test_console_command_composer.py
git commit -m "test(console): align unknown command hint"
```

### Task 4aa: Follow Console session and prompt-handoff ownership

**Files:**
- Modify: `Tests/UI/test_console_command_composer.py`
- Modify: `Tests/UI/test_library_prompts_canvas.py`

- [ ] **Step 1: Reproduce the remaining composer ownership failures**

Run both modules. Expected: two literal-send assertions omit the dispatch-time
`session_id`; eight composer prompt-insert cases and three Library source cases
assign or inspect deleted `pending_console_prompt_insert`.

- [ ] **Step 2: Retarget the live ownership seams**

Assert literal sends call `submit_draft` with the exact active session id.
Stage all Console prompt text through
`pending_handoffs.stage(HandoffChannel.CONSOLE_PROMPT_INSERT, text)` and use
`has_pending` for terminal or no-op state. In the Library success case, claim
the staged text, assert it exactly, and acknowledge it; assert no pending
channel for dirty and empty cases. Preserve every existing lifecycle, draft,
collapse, notification, navigation, and source-integrity assertion. Do not
change production.

- [ ] **Step 3: Verify and commit**

```bash
../../.venv/bin/python -m pytest \
  Tests/UI/test_console_command_composer.py \
  Tests/UI/test_library_prompts_canvas.py -q
../../.venv/bin/python -m ruff check \
  Tests/UI/test_console_command_composer.py \
  Tests/UI/test_library_prompts_canvas.py
../../.venv/bin/python -m ruff format --check \
  Tests/UI/test_console_command_composer.py \
  Tests/UI/test_library_prompts_canvas.py
git add \
  Tests/UI/test_console_command_composer.py \
  Tests/UI/test_library_prompts_canvas.py
git commit -m "test(console): follow prompt handoff ownership"
```

### Task 4ab: Follow Console live-work handoff ownership

**Files:**
- Modify: `Tests/UI/test_console_internals_decomposition.py`
- Modify: `Tests/UI/test_console_live_work_handoffs.py`
- Modify: `Tests/UI/test_home_screen.py`

- [ ] **Step 1: Reproduce the retired launch field failures**

Run the three modules. Expected: live-work helper, mounted Console,
Save/inspector/staged-context, action-routing, and Home-isolation cases assign
or inspect deleted `app.pending_console_launch`, so the Console cannot claim
their payloads.

- [ ] **Step 2: Stage and inspect the typed launch channel**

Replace every executable app-root launch-field use with
`pending_handoffs.stage(HandoffChannel.CONSOLE_LIVE_WORK, payload)`. Helper
tests must claim the normalized launch, assert it, and settle the claim;
mounted Console tests use `has_pending` after consumption. Preserve direct
screen-context assertions where that accepted context is the subject, and
retain all rendering, action, navigation, staged-context, inspector, and
Home-isolation assertions. The Home case must assert the launch remains
pending. Do not change production.

- [ ] **Step 3: Verify and commit**

```bash
../../.venv/bin/python -m pytest \
  Tests/UI/test_console_internals_decomposition.py \
  Tests/UI/test_console_live_work_handoffs.py \
  Tests/UI/test_home_screen.py -q
../../.venv/bin/python -m ruff check \
  Tests/UI/test_console_internals_decomposition.py \
  Tests/UI/test_console_live_work_handoffs.py \
  Tests/UI/test_home_screen.py
../../.venv/bin/python -m ruff format --check \
  Tests/UI/test_console_internals_decomposition.py \
  Tests/UI/test_console_live_work_handoffs.py \
  Tests/UI/test_home_screen.py
git add \
  Tests/UI/test_console_internals_decomposition.py \
  Tests/UI/test_console_live_work_handoffs.py \
  Tests/UI/test_home_screen.py
git commit -m "test(console): follow live work handoff ownership"
```

### Task 4ac: Settle Library prompt editor and import-harness ownership

**Files:**
- Modify: `tldw_chatbook/UI/Screens/library_screen.py`
- Modify: `Tests/UI/test_library_prompts_canvas.py`

- [ ] **Step 1: Preserve the two distinct red failure groups**

Run the Library prompt/canvas module. Expected: prompt initialization or
post-save recomposition can mark untouched canonical fields dirty, which
rotates among save/conflict/Unsaved/Console-insert assertions; all seven import
status cases remain empty because the nested unrun `TldwCli` owns the worker
while the active `LibraryHarness` owns the screen stack. Confirm the real-app
prompt-import owner and survive-unmount regressions remain green.

- [ ] **Step 2: Ignore only canonical mount echoes**

Add a prompt-field equality guard modeled on the existing Skills editor guard.
Compare the live prompt fields with the canonical state rendered from the
current prompt detail, or the active conflict snapshot when present, before
marking dirty. Matching mount/recompose events are ignored; genuine edits
still mark dirty. Preserve successful save/conflict recovery, Unsaved copy,
and clean/empty/dirty Console-insert behavior.

- [ ] **Step 3: Run nested imports through the active harness manager**

In the shared import test helper, bridge `screen.app_instance` to the active
`LibraryHarness` worker manager before pressing Import. Retain the existing
bounded wait and every exact status/database assertion. Do not change
production worker ownership or replace the button-level UI flow with a direct
private-method call.

- [ ] **Step 4: Verify and commit**

```bash
../../.venv/bin/python -m pytest \
  Tests/UI/test_library_prompts_canvas.py \
  Tests/ProductionApp/test_personas_library_root_state.py -q
../../.venv/bin/python -m ruff check \
  tldw_chatbook/UI/Screens/library_screen.py \
  Tests/UI/test_library_prompts_canvas.py
../../.venv/bin/python -m ruff format --check \
  tldw_chatbook/UI/Screens/library_screen.py \
  Tests/UI/test_library_prompts_canvas.py
git diff --check
git add \
  tldw_chatbook/UI/Screens/library_screen.py \
  Tests/UI/test_library_prompts_canvas.py
git commit -m "fix(library): ignore prompt mount echoes"
```

### Task 4ad: Settle the Console approval geometry fixture

**Files:**
- Modify: `Tests/UI/test_console_mcp_approval.py`

- [ ] **Step 1: Reproduce the zero-size two-row geometry**

Run the bundled-CSS multi-row geometry regression. Expected: both mounted rows
exist, but the first row's header is 0x0 because the fixture calls `set_batch`
before `ChatApprovalCard.on_mount`'s deferred `_hide_batch_body` callback runs.
Confirm the sibling single-row geometry regression passes because it already
settles the mount first.

- [ ] **Step 2: Follow the established fixture ordering**

Pause the pilot once immediately after entering `run_test`, before querying the
card and calling `set_batch`, matching the sibling single-row regression.
Retain every geometry assertion and make no production or CSS change.

- [ ] **Step 3: Verify and commit**

```bash
../../.venv/bin/python -m pytest Tests/UI/test_console_mcp_approval.py -q
../../.venv/bin/python -m ruff check Tests/UI/test_console_mcp_approval.py
../../.venv/bin/python -m ruff format --check Tests/UI/test_console_mcp_approval.py
git diff --check
git add Tests/UI/test_console_mcp_approval.py
git commit -m "test(console): settle approval card mount"
```

### Task 4ae: Follow screen-state and destination-handoff ownership

**Files:**
- Modify: `Tests/UI/test_console_live_work_handoffs.py`
- Modify: `Tests/UI/test_home_screen.py`

- [ ] **Step 1: Reproduce the four retired-owner failures**

Run the two modules' Schedules, Workflows, Artifacts, and Home flashcards
cases. Expected: Schedules/Workflows assign deleted `_screen_states`, Artifacts
stages a deleted target field and lacks the exact lookup seam on its fake, and
Home inspects a deleted Study pending field. Production correctly reads the
screen-state store and typed handoff channels.

- [ ] **Step 2: Seed and settle current owners**

Save the Schedules/Workflows Chat snapshot through `screen_state_store` under
the current `RuntimeIdentity`. Stage the Artifacts target through
`ARTIFACT_CHATBOOK_TARGET`, add the fake's exact async `get_chatbook` seam, and
assert requested-before-latest selection plus terminal consumption. Claim,
verify, and acknowledge the Home-to-Study `STUDY_INITIAL_SECTION` value.
Preserve every existing UI route, payload, launch, and isolation assertion.
Do not change production or add compatibility state.

- [ ] **Step 3: Verify and commit**

```bash
../../.venv/bin/python -m pytest \
  Tests/UI/test_console_live_work_handoffs.py \
  Tests/UI/test_home_screen.py -q
../../.venv/bin/python -m ruff check \
  Tests/UI/test_console_live_work_handoffs.py \
  Tests/UI/test_home_screen.py
../../.venv/bin/python -m ruff format --check \
  Tests/UI/test_console_live_work_handoffs.py \
  Tests/UI/test_home_screen.py
git diff --check
git add \
  Tests/UI/test_console_live_work_handoffs.py \
  Tests/UI/test_home_screen.py
git commit -m "test(navigation): follow typed state owners"
```

### Task 4af: Follow metadata-only MCP cancellation audit records

**Files:**
- Modify: `Tests/UI/test_console_mcp_approval.py`

- [ ] **Step 1: Reproduce the stale free-form error assertion**

Run the full Console MCP approval module. Expected: cancellation still writes a
denied, failed audit record, but the regression expects the retired
`"run stopped while approval pending"` error string instead of the current
bounded `approval_cancelled` category.

- [ ] **Step 2: Assert the metadata-only outcome**

Replace only the free-form error assertion with the exact
`error_category == "approval_cancelled"` contract. Retain record existence,
server/tool identity, denied decision, and failed outcome assertions. Do not
change production persistence or reintroduce an `error` field.

- [ ] **Step 3: Verify and commit**

```bash
../../.venv/bin/python -m pytest \
  Tests/UI/test_console_mcp_approval.py \
  Tests/MCP/test_control_plane_bridge.py -q
../../.venv/bin/python -m ruff check Tests/UI/test_console_mcp_approval.py
../../.venv/bin/python -m ruff format --check Tests/UI/test_console_mcp_approval.py
git diff --check
git add Tests/UI/test_console_mcp_approval.py
git commit -m "test(mcp): follow metadata-only cancellation audit"
```

### Task 4ag: Align the curated Anthropic Console default

**Files:**
- Modify: `Tests/UI/test_console_session_settings.py`

- [ ] **Step 1: Reproduce the stale curated default**

Run the remote-default regression. Expected: production configuration resolves
Anthropic to `claude-sonnet-5`, which is present in the provider catalog, while
the test's independently curated literal still expects
`claude-sonnet-4-20250514`.

- [ ] **Step 2: Update only the stale literal**

Change Anthropic's expected model to `claude-sonnet-5`. Preserve the complete
representative mapping and each exact configuration-plus-catalog assertion.
Do not change production or derive the expected values from production.

- [ ] **Step 3: Verify and commit**

```bash
../../.venv/bin/python -m pytest \
  Tests/UI/test_console_session_settings.py::test_console_remote_defaults_use_smoke_verified_models \
  Tests/test_config_model_catalog_defaults.py -q
../../.venv/bin/python -m ruff check Tests/UI/test_console_session_settings.py
../../.venv/bin/python -m ruff format --check Tests/UI/test_console_session_settings.py
git diff --check
git add Tests/UI/test_console_session_settings.py
git commit -m "test(console): align curated Anthropic default"
```

### Task 4ah: Follow the explicit provider-model resolver boundary

**Files:**
- Modify: `Tests/UI/test_console_session_settings.py`
- Modify: `Tests/UI/test_provider_model_resolution.py`

- [ ] **Step 1: Reproduce the eight stale-signature failures**

Run the Console runtime-discovery regression and the UI selector merge-cap
module. Expected: all eight pass an app-shaped object to
`resolve_provider_model_options`, whose current API requires the saved-model
mapping and catalog scope service separately.

- [ ] **Step 2: Pass the existing fake values explicitly**

At each call site, pass the fake app's `providers_models` and
`llm_provider_catalog_scope_service` as the two positional inputs. Retain all
existing entries, expected order/labels/warnings, cap boundaries, uncapped
picker, transient-current-model, and scope-call assertions. Do not change
production or add a compatibility overload/helper abstraction.

- [ ] **Step 3: Verify and commit**

```bash
../../.venv/bin/python -m pytest \
  Tests/UI/test_console_session_settings.py::test_console_model_resolution_includes_runtime_discovered_models \
  Tests/UI/test_provider_model_resolution.py \
  Tests/Provider/test_provider_model_resolution.py -q
../../.venv/bin/python -m ruff check \
  Tests/UI/test_console_session_settings.py \
  Tests/UI/test_provider_model_resolution.py
../../.venv/bin/python -m ruff format --check \
  Tests/UI/test_console_session_settings.py \
  Tests/UI/test_provider_model_resolution.py
git diff --check
git add \
  Tests/UI/test_console_session_settings.py \
  Tests/UI/test_provider_model_resolution.py
git commit -m "test(console): follow explicit model resolver inputs"
```

### Task 4ai: Preserve branded missing-key recovery copy

**Files:**
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`

- [ ] **Step 1: Reproduce the user-facing casing regression**

Run
`test_console_missing_key_recovery_action_is_provider_specific`. Expected: the
active session correctly stores canonical `openai`, but the blocker renders
`openai` instead of `OpenAI`; the Settings tooltip follows the same raw-key
path. The existing test already pins the intended branded copy and recovery
behavior.

- [ ] **Step 2: Use the existing display-name owner at the copy boundary**

Import the shared `provider_display_name` helper. In the missing-API-key
branches only, render that display value in the blocker and Settings tooltip.
Keep the canonical provider key for settings, readiness, and routing, and
retain the existing recovery target, field, and blocked-send copy. Do not add
a compatibility field, another provider map, or broaden this repair to
endpoint copy that is not failing.

- [ ] **Step 3: Verify and commit**

```bash
../../.venv/bin/python -m pytest \
  Tests/UI/test_console_session_settings.py::test_console_missing_key_recovery_action_is_provider_specific \
  Tests/UI/test_console_internals_decomposition.py::test_console_provider_blocker_exposes_open_settings_action \
  Tests/UI/test_console_workbench_contract.py -q
../../.venv/bin/python -m ruff check \
  tldw_chatbook/UI/Screens/chat_screen.py
../../.venv/bin/python -m ruff format --check \
  tldw_chatbook/UI/Screens/chat_screen.py
git diff --check
git add \
  tldw_chatbook/UI/Screens/chat_screen.py
git commit -m "fix(console): preserve provider display name in recovery"
```

### Task 4aj: Seed the choose-model routing fixture explicitly

**Files:**
- Modify: `Tests/UI/test_console_workbench_contract.py`

- [ ] **Step 1: Reproduce the contradictory fixture**

Run `test_console_empty_transcript_choose_model_opens_settings`. Expected: the
live setup action reaches the settings modal, but the modal mount raises
`InvalidSelectValueError` because the test's app has an empty provider even
though the regression is named and asserted as a missing-model flow.

- [ ] **Step 2: Configure the missing-model state**

Before mounting the harness, seed the same explicit OpenAI provider, empty
model, and empty-key configuration used by adjacent missing-model Console
coverage. Keep the live setup action wait, pointer click, and settings-modal or
Settings-screen destination assertion unchanged. Do not change production,
relax Select validation, or add timing waits.

- [ ] **Step 3: Verify and commit**

```bash
../../.venv/bin/python -m pytest \
  Tests/UI/test_console_workbench_contract.py::test_console_empty_transcript_choose_model_opens_settings \
  Tests/UI/test_console_workbench_contract.py::test_console_composer_keeps_disabled_reason_outside_input_row \
  Tests/UI/test_console_session_settings.py::test_console_settings_modal_save_returns_validated_settings -q
../../.venv/bin/python -m ruff check \
  Tests/UI/test_console_workbench_contract.py
../../.venv/bin/python -m ruff format --check \
  Tests/UI/test_console_workbench_contract.py
git diff --check
git add \
  Tests/UI/test_console_workbench_contract.py
git commit -m "test(console): seed choose-model routing state"
```

### Task 4ak: Publish the live-config fixture runtime state

**Files:**
- Modify: `Tests/UI/test_console_session_settings.py`

- [ ] **Step 1: Reproduce the retired projection write**

Run `test_real_journey_settings_save_unblocks_console_without_restart`.
Expected: `_build_live_config_test_app` reaches its fake runtime-policy loader
and raises `AttributeError` while assigning the read-only
`current_runtime_backend` property before the real journey can boot.

- [ ] **Step 2: Use the live projection owner**

After assigning the fake runtime policy context, call
`app._publish_runtime_policy_projection(context.state)` and remove both direct
compatibility-field assignments. Keep the real sandboxed config boot, Settings
adapter saves, navigation, restored-session, readiness, and modal-unblocking
assertions unchanged. Do not edit production or replace the journey with a
stubbed configuration.

- [ ] **Step 3: Verify and commit**

```bash
../../.venv/bin/python -m pytest \
  Tests/UI/test_console_session_settings.py::test_real_journey_settings_save_unblocks_console_without_restart \
  Tests/RAG_Admin/test_app_lazy_rag_admin_wiring.py -q
../../.venv/bin/python -m ruff check \
  Tests/UI/test_console_session_settings.py
../../.venv/bin/python -m ruff format --check \
  Tests/UI/test_console_session_settings.py
git diff --check
git add \
  Tests/UI/test_console_session_settings.py
git commit -m "test(console): publish fixture runtime state"
```

### Task 4al: Give the live scheduler a file-backed subscriptions fixture

**Files:**
- Modify: `Tests/UI/test_console_session_settings.py`

- [ ] **Step 1: Reproduce the cross-thread in-memory failure**

Run `test_real_journey_settings_save_unblocks_console_without_restart` after
Task 4ak. Expected: the journey boots and reaches navigation, then the real
scheduler worker fails in `PriorityQueue.load` with
`OperationalError: no such table: subscriptions`. The fixture initializes
`SubscriptionsDB(":memory:")` on the construction thread, while the scheduler
queries a distinct thread-local in-memory connection.

- [ ] **Step 2: Use a private file-backed fixture path**

Resolve the temporary user-data directory after `mkdtemp`. Remove
`get_subscriptions_db_path` from the loop that returns `:memory:` and patch it
separately to `user_data_dir / "subscriptions.sqlite"`. Keep every other fake,
the real scheduler worker, real configuration persistence, navigation, and
journey assertion unchanged. Do not disable scheduling or edit production.

- [ ] **Step 3: Verify and commit**

```bash
../../.venv/bin/python -m pytest \
  Tests/UI/test_console_session_settings.py::test_real_journey_settings_save_unblocks_console_without_restart \
  Tests/Scheduling/test_watchlist_projection.py \
  Tests/DB/test_subscriptions_db.py -q
../../.venv/bin/python -m ruff check \
  Tests/UI/test_console_session_settings.py
../../.venv/bin/python -m ruff format --check \
  Tests/UI/test_console_session_settings.py
git diff --check
git add \
  Tests/UI/test_console_session_settings.py
git commit -m "test(console): share scheduler subscriptions fixture"
```

### Task 4am: Stage resolution overrides through Console-owned state

**Files:**
- Modify: `Tests/UI/test_console_session_settings.py`

- [ ] **Step 1: Reproduce the retired reactive expectation**

Run `test_console_resolution_view_suppresses_boot_echo_reactives`. Expected:
the fresh persisted llama.cpp defaults win correctly, but assigning
`app.chat_api_provider_value = "Anthropic"` no longer affects
`_effective_console_provider_model` because Task 648 removed app-reactive
inputs from that resolver.

- [ ] **Step 2: Follow the current Console owner**

Remove the obsolete app-root provider/model setup. Rename the test and clarify
its comments to describe fresh persisted fallback plus Console-control
precedence. Stage `"Anthropic"` on `console._console_control_provider`, which
the live compact-provider handler owns, and retain both exact provider/model
assertions. Do not change production or restore app-reactive compatibility.

- [ ] **Step 3: Verify and commit**

```bash
../../.venv/bin/python -m pytest \
  Tests/UI/test_console_session_settings.py::test_console_resolution_view_prefers_console_control_over_fresh_defaults \
  Tests/Provider/test_provider_model_resolution.py \
  Tests/UI/test_settings_configuration_hub.py::test_effective_provider_model_prefers_console_overrides -q
../../.venv/bin/python -m ruff check \
  Tests/UI/test_console_session_settings.py
../../.venv/bin/python -m ruff format --check \
  Tests/UI/test_console_session_settings.py
git diff --check
git add \
  Tests/UI/test_console_session_settings.py
git commit -m "test(console): follow resolution control owner"
```

### Task 4an: Assert skill sends against the dispatch session

**Files:**
- Modify: `Tests/UI/test_console_skill_commands.py`

- [ ] **Step 1: Reproduce the stale text-only spy assertions**

Run the skill-command module. Expected: the leading-dollar normal-send
regression expects `submit_draft("$code-review fix it")`, while production
correctly passes the active `session_id` keyword. Confirm the picker-driven
regression contains the same stale expectation for both argument and
no-argument drafts.

- [ ] **Step 2: Assert exact text and exact dispatch owner**

In each affected test, capture the active store session id before dispatch.
Include `session_id=<captured id>` in all three
`submit_spy.assert_called_once_with` assertions. Keep the exact raw drafts,
skill execution, stored transcript, no-TOOL-row, and picker behavior
assertions unchanged. Do not edit production or weaken the spy assertions.

- [ ] **Step 3: Verify and commit**

```bash
../../.venv/bin/python -m pytest \
  Tests/UI/test_console_skill_commands.py \
  Tests/UI/test_console_command_composer.py -q
../../.venv/bin/python -m ruff check \
  Tests/UI/test_console_skill_commands.py
../../.venv/bin/python -m ruff format --check \
  Tests/UI/test_console_skill_commands.py
git diff --check
git add \
  Tests/UI/test_console_skill_commands.py
git commit -m "test(console): assert skill dispatch session"
```

### Task 4ao: Complete the skill unknown-command hint

**Files:**
- Modify: `Tests/UI/test_console_skill_commands.py`

- [ ] **Step 1: Reproduce the stale command list**

Run
`test_bare_slash_skill_name_no_longer_auto_runs_shows_unknown_command_hint`.
Expected: the regression's curated four-command hint is absent because the
current registry also advertises `/generate-image` and `/rewind`.

- [ ] **Step 2: Update only the independent expected copy**

Add `/generate-image` and `/rewind` after `/prefill` in the expected hint.
Retain the submit-not-called, skill-not-executed, draft-preserved, and exact
unknown-send-armed assertions. Do not derive the expected value from
production or change command registration.

- [ ] **Step 3: Verify and commit**

```bash
../../.venv/bin/python -m pytest \
  Tests/UI/test_console_skill_commands.py \
  Tests/UI/test_console_command_composer.py -q
../../.venv/bin/python -m ruff check \
  Tests/UI/test_console_skill_commands.py
../../.venv/bin/python -m ruff format --check \
  Tests/UI/test_console_skill_commands.py
git diff --check
git add \
  Tests/UI/test_console_skill_commands.py
git commit -m "test(console): complete skill command hint"
```

### Task 4ap: Retarget the Console image parity reference

**Files:**
- Modify: `Tests/UI/test_console_workbench_parity_matrix.py`

- [ ] **Step 1: Reproduce and inventory missing references**

Run the parity-matrix module. Expected: the existence gate fails on deleted
`Tests/UI/test_chat_image_attachment.py`. Evaluate every matrix entry and
confirm this is the only missing file/test reference.

- [ ] **Step 2: Point to current native Console coverage**

Replace that file reference with
`Tests/UI/test_console_native_chat_flow.py::test_console_attachment_worker_stages_image_and_inlines_text`.
Keep the existing image/RAG reference, parity categories, and exact
file/test-name validation unchanged. Do not recreate the retired test or
weaken the gate.

- [ ] **Step 3: Verify and commit**

```bash
../../.venv/bin/python -m pytest \
  Tests/UI/test_console_workbench_parity_matrix.py \
  Tests/UI/test_console_native_chat_flow.py::test_console_attachment_worker_stages_image_and_inlines_text \
  Tests/Chat/test_chat_functions.py::TestChatFunction::test_chat_with_image_and_rag -q
../../.venv/bin/python -m ruff check \
  Tests/UI/test_console_workbench_parity_matrix.py
../../.venv/bin/python -m ruff format --check \
  Tests/UI/test_console_workbench_parity_matrix.py
git diff --check
git add \
  Tests/UI/test_console_workbench_parity_matrix.py
git commit -m "test(console): retarget image parity coverage"
```

### Task 4aq: Finish the workspace marker sync migration

**Files:**
- Modify: `Tests/UI/test_console_workspace_context_rail.py`

- [ ] **Step 1: Reproduce the stale helper signature**

Run `test_console_workspace_context_syncs_active_conversation_marker`.
Expected: after correctly restoring the persisted native session, the test
passes a legacy `ChatSessionData` object to the now argument-free
`_sync_console_workspace_context` method and raises `TypeError`.

- [ ] **Step 2: Read the restored native owner**

Remove the `ChatSessionData` import and call
`console._sync_console_workspace_context()` with no argument. Keep the
workspace membership, native persisted-session restoration, UI settle, and
single selected “Planning thread” row assertions unchanged. Do not change
production or add a compatibility parameter.

- [ ] **Step 3: Verify and commit**

```bash
../../.venv/bin/python -m pytest \
  Tests/UI/test_console_workspace_context_rail.py::test_console_workspace_context_syncs_active_conversation_marker \
  Tests/UI/test_console_workspace_context_rail.py -q
../../.venv/bin/python -m ruff check \
  Tests/UI/test_console_workspace_context_rail.py
../../.venv/bin/python -m ruff format --check \
  Tests/UI/test_console_workspace_context_rail.py
git diff --check
git add \
  Tests/UI/test_console_workspace_context_rail.py
git commit -m "test(console): follow workspace sync owner"
```

### Task 4ar: Restore Watchlists source nesting after left alignment

**Files:**
- Modify: `tldw_chatbook/UI/Watchlists_Modules/watchlist_tree.py`
- Modify: `tldw_chatbook/css/features/_watchlists.tcss`
- Regenerate: `tldw_chatbook/css/tldw_cli_modular.tcss`
- Modify: `Tests/UI/test_destination_visual_parity_correction.py`

- [ ] **Step 1: Preserve the failing production contract**

Run the existing visual-parity node at both parametrized viewport sizes and
retain its relative compositor assertion. Expected before the fix: the narrow
case reports the left-aligned source name at column 4 and its parent at column
5.

- [ ] **Step 2: Restore the existing textual indent**

Change only the source label prefix from two to four spaces, update the nearby
production, test, and source-stylesheet explanations, then regenerate the
bundled stylesheet with `../../.venv/bin/python tldw_chatbook/css/build_css.py`.
Do not add a second CSS geometry rule or weaken the relative-column assertion.

- [ ] **Step 3: Verify the focused contract**

```bash
../../.venv/bin/python -m pytest \
  Tests/UI/test_destination_visual_parity_correction.py::test_watchlists_tree_chevron_shares_a_row_with_its_watchlist \
  -q
../../.venv/bin/python -m ruff check \
  tldw_chatbook/UI/Watchlists_Modules/watchlist_tree.py \
  Tests/UI/test_destination_visual_parity_correction.py
../../.venv/bin/python -m ruff format --check \
  tldw_chatbook/UI/Watchlists_Modules/watchlist_tree.py \
  Tests/UI/test_destination_visual_parity_correction.py
git diff --check
```

Expected: both viewport parameters pass, the generated bundle matches the
source stylesheet, and static/diff checks are clean.

### Task 4as: Follow current Speech recovery ownership

**Files:**
- Modify: `Tests/UI/test_disabled_action_recovery_tooltips.py`

- [ ] **Step 1: Preserve the stale-owner failure**

Run `test_stts_missing_speech_dependencies_expose_phase_five_recovery`.
Expected before the repair: the bare `STTSWindow` has no
`#speech-capability-status` after Speech adopted the Lab frame, so the query
raises `NoMatches`.

- [ ] **Step 2: Mount and assert the split current owner**

Replace the sole `STTSWindow` import with `STTSScreen`, add `_build_test_app`,
and mount `STTSScreen(_build_test_app())` through the existing `_ScreenHost`.
Patch dependency probes at `lab_speech_status`. Retain every exact recovery
taxonomy assertion against inspector `#speech-capability-status`, then retain
the exact install-tooltip assertion against rail
`#speech-capability-summary`. Do not add a harness or production compatibility
surface.

- [ ] **Step 3: Verify the focused and neighboring contracts**

```bash
../../.venv/bin/python -m pytest \
  Tests/UI/test_disabled_action_recovery_tooltips.py::test_stts_missing_speech_dependencies_expose_phase_five_recovery \
  Tests/UI/test_stts_capability_state.py \
  Tests/UI/test_disabled_action_recovery_tooltips.py \
  -q
../../.venv/bin/python -m ruff check \
  Tests/UI/test_disabled_action_recovery_tooltips.py
../../.venv/bin/python -m ruff format --check \
  Tests/UI/test_disabled_action_recovery_tooltips.py
git diff --check
```

Expected: the exact inspector taxonomy, summary tooltip, ready refresh, and
other disabled-action recovery contracts pass with static and diff checks
clean.

### Task 4at: Scope never-run Evals recovery to target readiness

**Files:**
- Modify: `Tests/UI/test_evals_bench_editor.py`

- [ ] **Step 1: Identify the screen-level match**

Run `test_never_run_bench_renders_unpreflighted_state` and inspect the
screen-wide `.ds-recovery-callout` match. Expected: the target row correctly
renders `Not yet checked`; the sole callout is the unrelated, valid
`#evals-primary-action-reason`, while `#evals-inspector-bench` has no recovery
callout.

- [ ] **Step 2: Assert absence inside the current owner**

Materialize the `.ds-recovery-callout` query beneath
`#evals-inspector-bench` and assert it is empty. Keep every status assertion
unchanged. Do not remove or restyle the primary-action explanation and do not
rewrite unrelated query idioms.

- [ ] **Step 3: Verify the focused Evals contract**

```bash
../../.venv/bin/python -m pytest \
  Tests/UI/test_evals_bench_editor.py::test_never_run_bench_renders_unpreflighted_state \
  Tests/UI/test_evals_bench_editor.py \
  -q
../../.venv/bin/python -m ruff check Tests/UI/test_evals_bench_editor.py
../../.venv/bin/python -m ruff format --check Tests/UI/test_evals_bench_editor.py
git diff --check
```

Expected: never-run, warned, blocked, ready, editor, and inspector coverage pass
with the screen-level action recovery unchanged.

### Task 4au: Scope duplicate-target readiness rows to EvalsInspector

**Files:**
- Modify: `Tests/UI/test_evals_bench_editor.py`

- [ ] **Step 1: Identify the third badge**

Run `test_bench_with_duplicate_target_id_composes_without_raising`. Expected:
the broad `#evals-inspector-pane` query returns the two intended
`#evals-inspector-target-{0,1}` badges plus the unrelated, valid
`#evals-primary-action-status`.

- [ ] **Step 2: Count rows inside their current owner**

Change only the query root to `#evals-inspector-bench` before selecting
`.ds-status-badge`. Retain the exact two-row count, nonzero region checks, and
all four distinct index-derived editor/inspector id assertions. Do not change
production or filter by incidental text.

- [ ] **Step 3: Verify the focused Evals contract**

```bash
../../.venv/bin/python -m pytest \
  Tests/UI/test_evals_bench_editor.py::test_bench_with_duplicate_target_id_composes_without_raising \
  Tests/UI/test_evals_bench_editor.py \
  -q
../../.venv/bin/python -m ruff check Tests/UI/test_evals_bench_editor.py
../../.venv/bin/python -m ruff format --check Tests/UI/test_evals_bench_editor.py
git diff --check
```

Expected: the duplicate-target regression and the complete Evals bench editor
module pass with the sibling action status untouched.

### Task 4av: Make local-model delete visibility lifecycle-safe

**Files:**
- Modify: `tldw_chatbook/Widgets/HuggingFace/local_models_widget.py`
- Modify: `Tests/UI/test_lab_mode_strip.py`

- [ ] **Step 1: Preserve the real-route failure**

Run `test_lab_route_and_mode_strip_navigate_the_real_shell`. Expected before
the fix: mounting the Models route raises `NoMatches` when
`LocalModelsWidget.on_mount()` queries `#delete-confirm-dialog` before its
composed children are queryable.

- [ ] **Step 2: Split initial and reactive visibility ownership**

Set the existing dialog class to `display: none` in component CSS, remove the
eager child lookup from `on_mount()`, and make the reactive watcher apply
visibility after refresh through a small query-materializing helper that
returns safely when the child is not yet present. Do not remove the dialog or
delete flow and do not cherry-pick the unrelated source commit.

- [ ] **Step 3: Retain show/hide behavior in the existing real-shell test**

After Models mounts, wait for `#delete-confirm-dialog`, assert it is hidden,
set the mounted `LocalModelsWidget.show_delete_confirm` true and then false,
and assert both deferred visibility changes after the pilot settles. Keep the
existing Models/Evals/Speech route and active-chip assertions.

- [ ] **Step 4: Verify focused Lab lifecycle**

```bash
../../.venv/bin/python -m pytest \
  Tests/UI/test_lab_mode_strip.py::test_lab_route_and_mode_strip_navigate_the_real_shell \
  Tests/UI/test_lab_mode_strip.py \
  Tests/UI/test_llm_screen_lab_adoption.py \
  Tests/UI/test_lab_frame_mode_keys.py \
  -q
../../.venv/bin/python -m ruff check \
  tldw_chatbook/Widgets/HuggingFace/local_models_widget.py \
  Tests/UI/test_lab_mode_strip.py
../../.venv/bin/python -m ruff format --check \
  tldw_chatbook/Widgets/HuggingFace/local_models_widget.py \
  Tests/UI/test_lab_mode_strip.py
git diff --check
```

Expected: Models mounts without lifecycle errors, delete confirmation is
hidden/showable/hideable, and Lab navigation remains green.

### Task 4aw: Align Library Collections blocked handoff state

**Files:**
- Modify: `Tests/UI/test_library_content_hub.py`

- [ ] **Step 1: Preserve both stale assertion failures**

Run
`test_library_collections_selection_explains_membership_workspace_and_actions`
and
`test_library_collections_empty_state_keeps_global_browse_rule_and_blocks_wip_actions`.
Expected: all copy/selection/geometry checks pass, and only the final
`disabled is True` assertion fails because TASK-716 keeps the blocked button
pressable.

- [ ] **Step 2: Assert the established recovery interaction**

In each test, capture `#library-use-in-console` and assert it is not disabled
and has `library-source-action-blocked`. Preserve every other assertion. Do not
change production or duplicate the dedicated blocked-press handler coverage.

- [ ] **Step 3: Verify focused and neighboring Library contracts**

```bash
../../.venv/bin/python -m pytest \
  Tests/UI/test_library_content_hub.py::test_library_collections_selection_explains_membership_workspace_and_actions \
  Tests/UI/test_library_content_hub.py::test_library_collections_empty_state_keeps_global_browse_rule_and_blocks_wip_actions \
  Tests/UI/test_library_content_hub.py \
  Tests/UI/test_post_release_workspaces_library_depth.py::test_blocked_use_in_console_press_explains_inline \
  -q
../../.venv/bin/python -m ruff check Tests/UI/test_library_content_hub.py
../../.venv/bin/python -m ruff format --check Tests/UI/test_library_content_hub.py
git diff --check
```

Expected: Collections remain visibly blocked, their buttons remain pressable,
and dedicated recovery behavior remains green.

### Task 4ax: Tolerate Library rail remount during ingest completion

**Files:**
- Modify: `Tests/UI/test_library_shell.py`

- [ ] **Step 1: Preserve the transient failure**

Run `test_library_shell_ingest_canvas_different_canvas_isolation`. Expected
under the failing interleaving: ingest completion schedules a rail recompose
and the loop's unconditional Media-row `query_one()` raises during the
temporary teardown frame.

- [ ] **Step 2: Observe the remounted current row**

Add a test-local `current_media_label()` that returns `None` while the row is
absent, then use the existing bounded `_wait_for_condition` until the current
label contains `Media (1)`. Include final label, selected row, and visible text
in the callable timeout message. Keep the final Notes selection and
ingest-widget absence assertions unchanged. Do not change production or add a
wait helper.

- [ ] **Step 3: Verify focused ingest lifecycle**

```bash
../../.venv/bin/python -m pytest \
  Tests/UI/test_library_shell.py::test_library_shell_ingest_canvas_different_canvas_isolation \
  Tests/UI/test_library_shell.py::test_library_shell_ingest_canvas_live_updates_without_manual_recompose \
  Tests/UI/test_library_shell.py::test_library_shell_ingest_canvas_registry_listener_removed_on_unmount \
  -q
../../.venv/bin/python -m ruff check Tests/UI/test_library_shell.py
../../.venv/bin/python -m ruff format --check Tests/UI/test_library_shell.py
git diff --check
```

Expected: completion survives the rail remount, publishes Media count 1, and
does not change the selected Notes canvas or leak ingest controls.

### Task 4ay: Isolate MCP import-file containment fixtures

**Files:**
- Modify: `Tests/UI/test_mcp_workbench.py`

- [ ] **Step 1: Preserve the config-path failure**

Run
`test_file_requested_pushes_picker_and_loads_selected_file_into_panel`.
Expected before repair: replacing
`mcp_workbench_module.os.path.expanduser` also changes process-wide home
expansion, redirects the isolated config lookup to a directory, and makes the
private-file guard fail before `MCPImportPanel` mounts.

- [ ] **Step 2: Patch the narrow import-root seam**

In all four import-file path regressions, patch
`mcp_workbench_module._mcp_import_home` to the intended temporary root instead
of patching `os.path.expanduser`. Update the three affected test docstrings so
they describe that narrow seam. Preserve every existing assertion and do not
change production path validation or config loading.

- [ ] **Step 3: Verify the import-file contracts**

```bash
../../.venv/bin/python -m pytest \
  Tests/UI/test_mcp_workbench.py::test_file_requested_pushes_picker_and_loads_selected_file_into_panel \
  Tests/UI/test_mcp_workbench.py::test_non_utf8_import_file_does_not_crash_app \
  Tests/UI/test_mcp_workbench.py::test_load_import_file_rejects_path_outside_home_directory \
  Tests/UI/test_mcp_workbench.py::test_load_import_file_rejects_oversized_file \
  -q
../../.venv/bin/python -m ruff check Tests/UI/test_mcp_workbench.py
../../.venv/bin/python -m ruff format --check Tests/UI/test_mcp_workbench.py
git diff --check
```

Expected: config isolation remains valid while all four import-path contracts
exercise their intended branch.

### Task 4az: Align MCP audit-detail fixtures with metadata-only records

**Files:**
- Modify: `Tests/UI/test_mcp_workbench.py`

- [ ] **Step 1: Preserve the three stale payload expectations**

Run the three `test_audit_entry_detail*`/selection nodes together. Expected
before repair: all three fail because the inspector renders the current
metadata-only schema and intentionally omits raw argument values and legacy
result excerpts.

- [ ] **Step 2: Seed and assert the public audit schema**

Change the test-local `_audit_record` factory from retired `arguments`,
`result_excerpt`, and `error` fields to the current status/category/type,
argument-name/count, and result-type/size fields. Keep the existing rendered
inspector journey and drill-through button checks. Parse its JSON detail and
assert the current metadata. In the two privacy tests, inject legacy payload
fields after factory construction solely to prove their raw values, excerpts,
and exception text do not render. Do not change production or add a projection
helper.

- [ ] **Step 3: Verify focused audit-detail contracts**

```bash
../../.venv/bin/python -m pytest \
  Tests/UI/test_mcp_workbench.py::test_audit_entry_selection_shows_pretty_printed_detail_in_inspector \
  Tests/UI/test_mcp_workbench.py::test_audit_entry_detail_omits_argument_values \
  Tests/UI/test_mcp_workbench.py::test_audit_entry_detail_omits_legacy_result_excerpt \
  -q
../../.venv/bin/python -m ruff check Tests/UI/test_mcp_workbench.py
../../.venv/bin/python -m ruff format --check Tests/UI/test_mcp_workbench.py
git diff --check
```

Expected: the rendered inspector exposes useful bounded metadata and no
payload-bearing legacy fields. Existing full-file Ruff debt must remain
unchanged; do not broadly reformat this test module under this repair.

### Task 4ba: Restore Media browsing-shell ownership and settle search workers

**Files:**
- Modify: `Tests/UI/test_media_window_v88_textual.py`

- [ ] **Step 1: Preserve the worker races**

Run the result-loading and item-selection nodes independently. Expected before
repair: both fail after one `pilot.pause()` because the background search has
not populated `list_panel.items`. A worker-only repair remains red because the
isolated mock app does not identify the mounted widget as its current
screen-owned Media destination.

- [ ] **Step 2: Publish the existing owner and await its worker manager**

Before each of the four `activate_media_type()` calls, set
`mock_app_instance.screen_stack` to the mounted `window.screen` and set that
screen's `media_window` to `window`, matching `_is_current_media_owner()`'s
live contract. After each activation, await
`pilot.app.workers.wait_for_complete()` before the test reads results, resets
the search mock, or selects a row, then keep the existing pilot pause. In the
search-button and pagination tests, also await worker completion after the
user action and before inspecting the new service call. Keep all assertions.
In the item-selection test, await worker completion again after
`handle_media_item_selected()` schedules the separate detail load and before
the existing pilot pause and viewer assertions. Do not monkeypatch the
ownership guard, add sleeps or a helper, or change production behavior.

- [ ] **Step 3: Verify the current Media shell**

```bash
../../.venv/bin/python -m pytest \
  Tests/UI/test_media_window_v88_textual.py \
  -q
../../.venv/bin/python -m ruff check Tests/UI/test_media_window_v88_textual.py
../../.venv/bin/python -m ruff format --check Tests/UI/test_media_window_v88_textual.py
git diff --check
```

Expected: all seven current-shell nodes pass with deterministic worker
ownership and unchanged user-facing behavior.

### Task 4bb: Reconcile retired focus-contract selectors

**Files:**
- Modify: `Tests/UI/test_non_obscuring_focus_contract.py`

- [ ] **Step 1: Preserve the complete stale cluster**

Run the full module. Expected before repair: 9 failures and 92 passes. Two
failures target Textual's dead `.collapsible--header`; three read/assert the
deleted legacy chat-tabs stylesheet; and four assert preset/resize selectors
retired by TASK-577.

- [ ] **Step 2: Follow live focus owners and delete dead assertions**

Retarget the Collapsible hover test to the Library/RAG settings card's existing
`CollapsibleTitle` base/hover rules in source and bundle, and retarget the
focus test to assert both expanded/collapsed global `CollapsibleTitle` focus
selectors and the matching ID-scoped Library/RAG focus overrides in source and
bundle. Apply the same non-obscuring focus and accent-border checks to the
scoped rules so their higher specificity cannot regress the reviewed focus
state. Delete the unused `_chat_tabs.tcss` and conversations path constants,
the legacy chat-tab tests, the preset active/hover tests, and only the retired
preset/resize parameters from the shared sidebar hover test. Delete the passing
conversation `Collapsible.-active` test because it asserts a nonexistent title
class for a state no live Collapsible owns. Preserve every remaining contract.
Do not restore retired CSS, change production, or activate the dead unscoped
conversation rules.

- [ ] **Step 3: Verify all remaining focus contracts**

```bash
../../.venv/bin/python -m pytest \
  Tests/UI/test_non_obscuring_focus_contract.py \
  -q
../../.venv/bin/python -m ruff check Tests/UI/test_non_obscuring_focus_contract.py
../../.venv/bin/python -m ruff format --check Tests/UI/test_non_obscuring_focus_contract.py
git diff --check
```

Expected: all 93 remaining current-owner contracts pass, and Ruff plus diff
checks remain clean.

### Task 4bc: Dispatch Personas generation wiring through mounted buttons

**Files:**
- Modify: `Tests/UI/test_personas_generation_wiring.py`

- [ ] **Step 1: Preserve the collection-sensitive failure**

Run the three field-generation wiring nodes while also collecting
`Tests/UI/test_settings_library_rag_defaults.py`. Expected before repair: at
least one mounted editor pointer click intermittently misses, leaving the
controller call list or captured live record empty. Confirm the same three
nodes pass when collected alone.

- [ ] **Step 2: Use the direct mounted-button event seam**

Keep `pilot.click("#personas-library-new")` because opening the editor is the
user-navigation setup under test. For controls already inside the returned
mounted editor, query the `Button` and call `press()` instead of asking the
pilot to resolve pointer geometry while programmatic field changes are posting
dirty/validation events. A direct press only queues `Button.Pressed`, so every
worker-producing press must be followed by `await pilot.pause()` before
`await pilot.app.workers.wait_for_complete()` snapshots the current workers;
retain the existing post-worker pause and every behavior assertion. Non-worker
presses still receive their existing pause before dependent reads or actions.
Do not add sleeps, helpers, production changes, or validation-timer changes.

- [ ] **Step 3: Verify isolated and collection-sensitive coverage**

```bash
../../.venv/bin/python -m pytest \
  Tests/UI/test_personas_generation_wiring.py \
  -q
../../.venv/bin/python -m pytest \
  Tests/UI/test_personas_generation_wiring.py \
  Tests/UI/test_settings_library_rag_defaults.py \
  -q
../../.venv/bin/python -m ruff check \
  Tests/UI/test_personas_generation_wiring.py
../../.venv/bin/python -m ruff format --check \
  Tests/UI/test_personas_generation_wiring.py
git diff --check
```

Expected: all nine wiring nodes and all fourteen Library/RAG settings nodes pass
together, and static plus diff checks remain clean.

### Task 4bd: Align Personas import-failure recovery coverage

**Files:**
- Modify: `Tests/UI/test_personas_workbench.py`

- [ ] **Step 1: Preserve the stale recovery assertion**

Run
`TestImportExport::test_import_failure_shows_recovery_copy` alone. Expected
before repair: the production import path catches and safely categorizes the
injected `ValueError`, but the assertion fails because it still expects raw
`"Unsupported card format"` text. The selection setup also logs a
non-awaitable `chat_dictionary_scope_service` `MagicMock` traceback.

- [ ] **Step 2: Follow the fixed recovery-copy contract**

Set `mock_app_instance.chat_dictionary_scope_service = None` before constructing
the test app because dictionary attachment is outside this regression. Keep the
real character-row selection and direct `_import_character_from_path()` call.
Replace the raw-exception expectation with the exact
`("Character import failed; verify the file and retry.", "error")` notification
and assert no captured message contains the injected exception text. Retain the
selected-character assertion. Do not change production or add a fake service.

- [ ] **Step 3: Verify the import/export cluster**

```bash
../../.venv/bin/python -m pytest \
  Tests/UI/test_personas_workbench.py::TestImportExport \
  -q
../../.venv/bin/python -m ruff check \
  Tests/UI/test_personas_workbench.py
../../.venv/bin/python -m ruff format --check \
  Tests/UI/test_personas_workbench.py
git diff --check
```

Expected: the import/export class passes, the failure path renders only the
fixed recovery copy, and the change introduces no new static or diff issue.

### Task 4be: Remove the retired core-loop handoff field wait

**Files:**
- Modify: `Tests/UI/test_product_maturity_phase1_core_loop.py`

**Existing ADR:** `backlog/decisions/033-application-session-state-ownership.md`

- [ ] **Step 1: Preserve the focused RED**

Run
`test_search_rag_result_stages_context_into_console_core_loop` alone. Expected
before repair: the real handoff navigates to Console, but the test raises
`AttributeError` while polling the deleted app-root `pending_chat_handoff`
field. Confirm TASK-645 moved the channel to `PendingHandoffStore` and that the
current smoke test proves the same flow through visible Console outcomes.

- [ ] **Step 2: Delete only the obsolete intermediate wait**

Remove the comment and `_wait_until` block that read
`app.pending_chat_handoff`. Retain the real `open_chat_with_handoff()` call,
route wait, staged-source count, RAG state, live-work title, evidence readiness,
and suggested composer draft assertions. Do not change production, restore a
compatibility field, or substitute a transient store-internal assertion.

- [ ] **Step 3: Verify the focused module and resumed boundary**

```bash
../../.venv/bin/python -m pytest \
  Tests/UI/test_product_maturity_phase1_core_loop.py \
  -q
../../.venv/bin/python -m pytest \
  Tests/UI/test_product_maturity_gate1_core_loop_screen_adaptation.py \
  Tests/UI/test_product_maturity_phase1_core_loop.py \
  -q
../../.venv/bin/python -m ruff check \
  Tests/UI/test_product_maturity_phase1_core_loop.py
../../.venv/bin/python -m ruff format --check \
  Tests/UI/test_product_maturity_phase1_core_loop.py
git diff --check
```

Expected: the core-loop proof passes through current visible Console behavior,
the adjacent screen-adaptation contract remains green, and static plus diff
checks remain clean.

### Task 4bf: Align the service-unavailable Library handoff state

**Files:**
- Modify: `Tests/UI/test_product_maturity_phase1_empty_setup_states.py`

- [ ] **Step 1: Preserve the three-row RED**

Run the parameterized service-unavailable handoff regression. Expected before
repair: the Library row fails because `#library-use-in-console` is enabled and
marked `library-source-action-blocked`; the Watchlists and Skills rows pass
with disabled controls. Confirm TASK-716 and the focused destination test
establish Library's pressable recovery action.

- [ ] **Step 2: Narrow the differing assertion to Library**

Rename the test from “disable” to “block.” When `route == "library"`, assert
the button is enabled and owns `library-source-action-blocked`; otherwise keep
the disabled assertion. Retain the exact service recovery copy and unavailable
tooltip check for all three rows. Do not change production, parameterize
another behavior field, or duplicate the dedicated blocked-press coverage.

- [ ] **Step 3: Verify the matrix and adjacent Library contract**

```bash
../../.venv/bin/python -m pytest \
  Tests/UI/test_product_maturity_phase1_empty_setup_states.py \
  -q
../../.venv/bin/python -m pytest \
  Tests/UI/test_destination_shells.py::test_library_destination_service_failure_uses_recovery_copy \
  Tests/UI/test_product_maturity_phase1_empty_setup_states.py::test_service_unavailable_states_block_false_console_handoffs \
  -q
../../.venv/bin/python -m ruff check \
  Tests/UI/test_product_maturity_phase1_empty_setup_states.py
../../.venv/bin/python -m ruff format --check \
  Tests/UI/test_product_maturity_phase1_empty_setup_states.py
git diff --check
```

Expected: all service-unavailable rows pass with Library pressable and blocked,
Watchlists and Skills disabled, and static plus diff checks clean.

### Task 4bg: Restore the renumbered UAT task identity

**Files:**
- Modify: `backlog/tasks/task-672 - First-run-character-chat-UAT-orientation-markup-crash-approval-card-mount-order.md`

- [ ] **Step 1: Preserve the identity-guard RED**

Run `test_backlog_task_frontmatter_ids_are_unique`. Expected before repair:
the guard rejects task 672 because the file starts with a legacy task-635
heading instead of YAML frontmatter. Confirm repository history renamed this
same file through several collision-free ids and no other file owns
`TASK-672`.

- [ ] **Step 2: Repair only record identity**

Prepend standard frontmatter for unique id `TASK-672`, its existing title,
`Done` status, empty assignee list, bounded labels, no dependencies, high
priority, and dates grounded in the existing commit history. Change the
Markdown heading to task 672. Preserve the completed acceptance criteria,
plan, implementation notes, and historical narrative byte-for-byte otherwise.
Do not weaken the harness or renumber the file again.

- [ ] **Step 3: Verify Backlog parsing and uniqueness**

```bash
../../.venv/bin/python -m pytest \
  Tests/UI/test_product_maturity_phase1_harness.py::test_backlog_task_frontmatter_ids_are_unique \
  -q
backlog task 672 --plain
../../.venv/bin/python -m pytest \
  Tests/UI/test_product_maturity_phase1_harness.py \
  -q
git diff --check
```

Expected: task 672 parses as a completed task with its existing content, every
task identity remains unique, the harness module passes, and the diff is clean.

### Task 4bh: Migrate focused Study harnesses to typed handoffs

**Files:**
- Modify: `Tests/UI/test_study_screen.py`
- Modify: `Tests/UI/test_study_dashboard.py`
- Modify: `Tests/UI/test_study_quizzes_screen.py`
- Modify: `Tests/UI/test_study_flashcards_screen.py`
- Modify: `Tests/UI/test_product_maturity_phase3_knowledge_entry.py`
- Modify: `Tests/UI/test_product_maturity_phase3_library_study_context.py`
- Modify: `Tests/UI/test_product_maturity_phase3_source_study_generation.py`

**Existing ADR:** `backlog/decisions/033-application-session-state-ownership.md`

- [ ] **Step 1: Preserve the focused RED inventory**

Run the seven listed modules together. Expected before repair: 82 collected,
18 pass and 64 fail. All but the independent app-level runtime-backend callback
fixture fail because the real `StudyScreen` now requires
`app_instance.pending_handoffs`, or because tests still stage values through
retired `pending_study_*` fields/methods. Record the callback failure for a
separate cluster.

- [ ] **Step 2: Supply the current owner in shared test composition**

Add an empty `PendingHandoffStore` to `_build_app_instance()` in the dashboard
suite. In the focused quizzes and flashcards `StudyTestApp` constructors,
install an empty store only when the supplied fake lacks one. In the lower-level
Study screen module, use one small test-local builder for empty or
scope-populated stores and give every direct mount/resume fixture the current
owner. Do not translate legacy fields in a harness or change production.

- [ ] **Step 3: Stage real scope and section inputs through typed channels**

Across the seven modules, replace every `pending_study_scope_context` and
`pending_study_initial_section` setup with explicit `stage()` calls on
`HandoffChannel.STUDY_SCOPE` or `HandoffChannel.STUDY_INITIAL_SECTION`.
Update the direct `TldwCli.open_study_screen` unit fixture to own a real store
and inspect its typed claim. Update the restored-section precedence test to
call `_apply_pending_section_handoff()` and assert the visible/current section
plus consumed pending state. In
`test_handle_runtime_backend_changed_recomputes_workspace_scope_state`, remove
only the obsolete assertion that the screen handler mutates
`app_instance.current_runtime_backend`; retain every scope-state and
controller-refresh assertion. Retain all other behavior assertions.

- [ ] **Step 4: Verify the Study boundary**

```bash
../../.venv/bin/python -m pytest \
  Tests/UI/test_study_screen.py \
  Tests/UI/test_study_dashboard.py \
  Tests/UI/test_study_quizzes_screen.py \
  Tests/UI/test_study_flashcards_screen.py \
  Tests/UI/test_product_maturity_phase3_knowledge_entry.py \
  Tests/UI/test_product_maturity_phase3_library_study_context.py \
  Tests/UI/test_product_maturity_phase3_source_study_generation.py \
  -k "not test_app_level_runtime_backend_callback_updates_backend_and_forwards" \
  -q
../../.venv/bin/python -m pytest \
  Tests/UI/test_study_screen.py::test_app_level_runtime_backend_callback_updates_backend_and_forwards \
  -q
../../.venv/bin/python -m ruff check \
  Tests/UI/test_study_screen.py \
  Tests/UI/test_study_dashboard.py \
  Tests/UI/test_study_quizzes_screen.py \
  Tests/UI/test_study_flashcards_screen.py \
  Tests/UI/test_product_maturity_phase3_knowledge_entry.py \
  Tests/UI/test_product_maturity_phase3_library_study_context.py \
  Tests/UI/test_product_maturity_phase3_source_study_generation.py
git diff --check
```

Expected after this cluster: all 81 in-scope tests pass; the separately run
runtime-callback fixture retains its independent RED for the next cluster.
Static and diff checks introduce no new issues.

### Task 4bi: Follow app-level runtime-policy callback ownership

**Files:**
- Modify: `Tests/UI/test_study_screen.py`

**Existing ADR:** `backlog/decisions/033-application-session-state-ownership.md`

- [ ] **Step 1: Preserve the focused RED**

Run
`Tests/UI/test_study_screen.py::test_app_level_runtime_backend_callback_updates_backend_and_forwards`.
Expected before repair: the unbound application handler fails because the
fixture has no `runtime_policy`; its two writable backend fields are retired
composition state.

- [ ] **Step 2: Exercise the live unit boundary**

Give the fixture a real `RuntimePolicyContext` backed by a small recording
store, empty `app_config`, and a mocked `server_context_provider`. Keep the
active-screen callback. Rename the test for policy commit and forwarding,
assert the handler returns `True`, the authoritative source is `local`, the
store saved that state once, the provider invalidated the unchanged
`None`-to-`None` server binding, and the screen received `"local"`. Remove the
retired writable-field setup and assertions. Do not change production or
instantiate the full application.

- [ ] **Step 3: Verify the focused and Study boundaries**

```bash
../../.venv/bin/python -m pytest \
  Tests/UI/test_study_screen.py::test_app_level_runtime_backend_callback_commits_policy_and_forwards \
  -q
../../.venv/bin/python -m pytest \
  Tests/UI/test_study_screen.py \
  Tests/UI/test_study_dashboard.py \
  Tests/UI/test_study_quizzes_screen.py \
  Tests/UI/test_study_flashcards_screen.py \
  Tests/UI/test_product_maturity_phase3_knowledge_entry.py \
  Tests/UI/test_product_maturity_phase3_library_study_context.py \
  Tests/UI/test_product_maturity_phase3_source_study_generation.py \
  -q
../../.venv/bin/python -m ruff check Tests/UI/test_study_screen.py
../../.venv/bin/python -m ruff format --check Tests/UI/test_study_screen.py
git diff --check
```

Expected: the focused test and all 82 Study-boundary tests pass, static checks
pass, and no production file changes.

### Task 4bj: Follow typed Chat handoff ownership in first-run UAT

**Files:**
- Modify: `Tests/UI/test_uat_first_time_character_chat.py`

**Existing ADR:** `backlog/decisions/033-application-session-state-ownership.md`

- [ ] **Step 1: Preserve the end-to-end RED**

Run
`Tests/UI/test_uat_first_time_character_chat.py::test_first_time_user_character_chat_journey`.
Expected before repair: card import, provider setup, conversation creation, and
Chat navigation occur, but the test times out polling the deleted
`app.pending_chat_handoff` field.

- [ ] **Step 2: Observe and settle through the live store**

Before the Start Chat press, wrap the real `app.pending_handoffs.stage` method.
For `HandoffChannel.CHAT`, detach and record the `ChatHandoffPayload`, then
forward the same channel/value to the original method. Use that recorded value
for the existing metadata assertions. Prove consumption only when the Chat
channel has no pending value, the Console consumer is idle, and its store owns
a session whose character id matches the handoff metadata. Update the related
failure diagnostic and remove the unused `asyncio` import. Do not change
production or weaken the remaining UAT assertions.

- [ ] **Step 3: Verify the UAT boundary**

```bash
../../.venv/bin/python -m pytest \
  Tests/UI/test_uat_first_time_character_chat.py::test_first_time_user_character_chat_journey \
  -q
../../.venv/bin/python -m ruff check \
  Tests/UI/test_uat_first_time_character_chat.py
../../.venv/bin/python -m ruff format --check \
  Tests/UI/test_uat_first_time_character_chat.py
git diff --check
```

Expected: the full UAT passes with its import, recovery, handoff, send, reply,
and persistence assertions intact; static checks pass and no production file
changes.

### Task 4bk: Wait for mounted Personas ownership in first-run UAT

**Files:**
- Modify: `Tests/UI/test_uat_first_time_character_chat.py`

**Existing ADR:** `backlog/decisions/033-application-session-state-ownership.md`

- [ ] **Step 1: Record the observed lifecycle race**

Preserve the exact repeat-run evidence: after the typed-handoff repair, one UAT
run imported the database row but failed at the unchanged
`selected_entity_kind == "character"` assertion because the test invoked the
import continuation after screen assignment but before the Personas
destination was mounted. Surrounding exact runs pass, confirming a lifecycle
race rather than a deterministic handoff regression.

- [ ] **Step 2: Wait on the real destination boundary**

Change the Personas navigation predicate to return `app.screen` only when its
type is `PersonasScreen` and `is_mounted` is true, and bind the returned value
as `personas`. Keep the awaited import continuation, selected-character
assertions, and every later UAT step unchanged. Do not add a sleep or change
production.

- [ ] **Step 3: Verify repeat stability**

Run the exact UAT three consecutive times, then run Ruff and diff checks:

```bash
../../.venv/bin/python -m pytest \
  Tests/UI/test_uat_first_time_character_chat.py::test_first_time_user_character_chat_journey \
  --count=3 -q
../../.venv/bin/python -m ruff check \
  Tests/UI/test_uat_first_time_character_chat.py
git diff --check
```

If `pytest-repeat` is unavailable, invoke the same exact node three times
sequentially. Expected: all three journeys pass without selection or handoff
failure, Ruff passes, the file's already-proven parent format drift does not
increase, and no production file changes.

### Task 4bl: Retarget the app-free Console responsiveness fixture

**Files:**
- Modify: `Tests/UI/test_ui_responsiveness.py`

**Existing ADR:** `backlog/decisions/033-application-session-state-ownership.md`

- [ ] **Step 1: Preserve the focused RED**

Run
`Tests/UI/test_ui_responsiveness.py::test_console_sync_records_worker_lifecycle`.
Expected before repair: the app-free fake enters the current effective-scope
warmer and fails on missing `_console_chat_store` before completing the
worker-lifecycle assertion.

- [ ] **Step 2: Stub the current sync collaborators**

Keep the lightweight `ChatScreen.__new__` fixture. Add an
`observed_active_worker` flag to the core-state stub and assert the monitor
reports one active worker there. Stub the current effective-scope,
dictionary/world-book/avatar, native transcript, and conditional rail
visibility collaborators with no-ops; remove retired collaborator stubs and
unused fake state. Invoke the real `_sync_native_console_chat_ui()` wrapper and
assert both that the core seam ran and the final active-worker count is zero.
Do not create a full application or change production.

- [ ] **Step 3: Verify responsiveness boundaries**

```bash
../../.venv/bin/python -m pytest \
  Tests/UI/test_ui_responsiveness.py::test_console_sync_records_worker_lifecycle \
  -q
../../.venv/bin/python -m pytest \
  Tests/UI/test_ui_responsiveness.py \
  -q
../../.venv/bin/python -m ruff check Tests/UI/test_ui_responsiveness.py
../../.venv/bin/python -m ruff format --check Tests/UI/test_ui_responsiveness.py
git diff --check
```

Expected: the focused and full responsiveness module pass, static checks pass,
and no production file changes.

### Task 4bm: Synchronize the Library thread-worker policy sentinel

**Files:**
- Modify: `Tests/UI/test_unified_shell_phase5_recovery_taxonomy.py`

**ADR required:** no

**ADR path:** N/A

**Reason:** This updates an exact static test allowlist for two existing,
reviewed blocking operations; it does not change a runtime or ownership
boundary.

- [ ] **Step 1: Preserve the focused RED and inventory**

Run
`test_service_backed_policy_destinations_use_async_workers_without_asyncio_run`.
Expected before repair: the test inventories six Library
`@work(thread=True)` decorators but expects four. Verify the six functions are
export counts, export execution, search-history persistence, rail-preference
persistence, verified Parakeet installation, and source-ingest preflight;
Personas and Skills remain at zero and the annotated `asyncio.run` count
remains three. Also record the masked annotation failure: one multiline
`asyncio.run` call places its required marker on the closing line, while the
current sentinel inspects only the opening line.

- [ ] **Step 2: Make the exact inventory syntax-aware**

Expand the explanatory comment to include verified model installation and
source-ingest preflight, then change Library's exact allowed thread-worker
count from four to six. Parse each source once. Count function decorators whose
call target is `work` and whose keyword list contains literal `thread=True`.
For each `asyncio.run` node, inspect all source lines from `lineno` through
`end_lineno` for the required annotation. Retain exact equality, every screen
path, the three-call `asyncio.run` inventory, and `_run_maybe_awaitable`
rejection. Do not add a helper or change production.

- [ ] **Step 3: Verify the policy boundary**

```bash
../../.venv/bin/python -m pytest \
  Tests/UI/test_unified_shell_phase5_recovery_taxonomy.py::test_service_backed_policy_destinations_use_async_workers_without_asyncio_run \
  -q
../../.venv/bin/python -m pytest \
  Tests/UI/test_unified_shell_phase5_recovery_taxonomy.py \
  -q
../../.venv/bin/python -m ruff check \
  Tests/UI/test_unified_shell_phase5_recovery_taxonomy.py
../../.venv/bin/python -m ruff format --check \
  Tests/UI/test_unified_shell_phase5_recovery_taxonomy.py
git diff --check
```

Expected: the focused policy sentinel and full recovery-taxonomy module pass,
static checks pass, and no production file changes.

### Task 4bn: Settle File Notes bulk-unstage without global screen idle

**Files:**
- Modify: `Tests/UI/test_library_file_notes_git.py`

**ADR required:** no

**ADR path:** N/A

**Reason:** This corrects one test's scheduler observation boundary without
changing retained-worker ownership or production behavior.

- [ ] **Step 1: Preserve the focused RED**

Run
`test_unstage_all_summary_counts_the_complete_displayed_snapshot` alone.
Expected before repair: after the Unstage All press, `_wait_until()` enters
`pilot.pause(0.02)` and Textual raises `WaitForScreenTimeout` after 30 seconds
while retained Git refresh messages are still settling. The failure reproduces
both alone and after 2,517 passing full-UI tests.

- [ ] **Step 2: Poll only the asserted retained-work boundary**

Replace that one `_wait_until()` call with at most 200 direct predicate checks,
yielding via `asyncio.sleep(0.01)` between checks. Keep the exact
`unstage_calls == [(1, 2)]` and `status_calls == 2` conditions and raise
`AssertionError("Unstage All did not settle and refresh")` on exhaustion.
Leave the shared helper, bulk-stage test, summary text, and production
unchanged.

- [ ] **Step 3: Verify the bulk-unstage repair and preserve the paired RED**

```bash
../../.venv/bin/python -m pytest \
  Tests/UI/test_library_file_notes_git.py::test_unstage_all_summary_counts_the_complete_displayed_snapshot \
  -q
../../.venv/bin/python -m pytest \
  Tests/UI/test_library_file_notes_git.py::test_stage_all_summary_counts_the_complete_displayed_snapshot \
  -q
../../.venv/bin/python -m ruff check Tests/UI/test_library_file_notes_git.py
../../.venv/bin/python -m ruff format --check \
  Tests/UI/test_library_file_notes_git.py
git diff --check
```

Expected at this checkpoint: bulk Unstage passes; the unchanged paired Stage
test reproduces `WaitForScreenTimeout` at the same generic helper and becomes
Task 4bo's focused RED. Static checks pass and there are no production changes.
Defer the full-module green expectation until Task 4bo.

### Task 4bo: Settle the paired File Notes bulk-stage action

**Files:**
- Modify: `Tests/UI/test_library_file_notes_git.py`

**ADR required:** no

**ADR path:** N/A

**Reason:** This applies the same test-only scheduler boundary to the paired
retained action after its required verification exposed the identical timeout.

- [ ] **Step 1: Preserve the focused RED**

Run `test_stage_all_summary_counts_the_complete_displayed_snapshot` alone.
Expected before repair: the exact service predicate completes, but the generic
helper's `pilot.pause(0.02)` requires global Textual idleness and raises
`WaitForScreenTimeout` after 30 seconds.

- [ ] **Step 2: Poll only the existing stage predicate**

Replace that one `_wait_until()` call with at most 200 direct checks of the
existing `stage_calls == [(1, 2)]` and `status_calls == 2` predicate, yielding
with `asyncio.sleep(0.01)`. Raise
`AssertionError("Stage All did not settle and refresh")` on exhaustion. Leave
the shared helper, summary text, and production unchanged.

- [ ] **Step 3: Verify both actions and the module**

```bash
../../.venv/bin/python -m pytest \
  Tests/UI/test_library_file_notes_git.py::test_stage_all_summary_counts_the_complete_displayed_snapshot \
  Tests/UI/test_library_file_notes_git.py::test_unstage_all_summary_counts_the_complete_displayed_snapshot \
  -q
../../.venv/bin/python -m pytest \
  Tests/UI/test_library_file_notes_git.py \
  -q
../../.venv/bin/python -m ruff check Tests/UI/test_library_file_notes_git.py
../../.venv/bin/python -m ruff format --check \
  Tests/UI/test_library_file_notes_git.py
git diff --check
```

Expected: both actions and the full module pass, static checks introduce no new
debt, and there are no production changes.

### Task 4bp: Select the exact Library source-action CSS rule

**Files:**
- Modify: `Tests/UI/test_product_maturity_phase3_library_contract_layout.py`

**ADR required:** no

**ADR path:** N/A

**Reason:** This fixes a static test parser's selector-prefix collision without
changing the accepted Library styling or production behavior.

- [ ] **Step 1: Preserve the focused RED**

Run `test_library_source_actions_use_console_text_control_style` alone.
Expected before repair: `_css_block(..., ".library-source-action")` extracts
the earlier `.library-source-action-blocked` color-only rule and fails the
transparent-background assertion even though the exact base rule is present.

- [ ] **Step 2: Select the exact base rule**

For the two base-rule extractions only, pass `.library-source-action {` to the
existing helper. Leave the helper, stylesheets, modifier rules, and all
assertions unchanged.

- [ ] **Step 3: Verify the static contract module**

```bash
../../.venv/bin/python -m pytest \
  Tests/UI/test_product_maturity_phase3_library_contract_layout.py \
  -q
../../.venv/bin/python -m ruff check \
  Tests/UI/test_product_maturity_phase3_library_contract_layout.py
../../.venv/bin/python -m ruff format --check \
  Tests/UI/test_product_maturity_phase3_library_contract_layout.py
git diff --check
```

Expected: the module and static checks pass with no production changes.

### Task 4bq: Align Library footer ownership with its live shortcut gate

**Files:**
- Modify: `Tests/UI/test_screen_footer_hints.py`

**ADR required:** no

**ADR path:** N/A

**Reason:** This retargets a focused ownership test to the existing
Search/RAG-only shortcut policy without changing navigation or footer behavior.

- [ ] **Step 1: Preserve the focused RED**

Run `test_library_registration_updates_the_screens_own_footer` alone. Expected
before repair: the initial Library row has no live `u` action and therefore
correctly leaves the screen footer at its default, while the stale test expects
the retired screen-wide hint.

- [ ] **Step 2: Exercise the live dynamic registration owner**

Import `LIBRARY_ROW_BROWSE_SEARCH`. Retain the default screen-footer assertion,
set the mounted screen's selected-row owner to Search/RAG, and call
`_register_footer_shortcuts()`. Assert the existing `u` copy reaches the
screen-owned footer and the host app's footer remains at its default. Do not
change production or add a full navigation journey.

- [ ] **Step 3: Verify footer ownership**

```bash
../../.venv/bin/python -m pytest \
  Tests/UI/test_screen_footer_hints.py \
  -q
../../.venv/bin/python -m ruff check Tests/UI/test_screen_footer_hints.py
../../.venv/bin/python -m ruff format --check \
  Tests/UI/test_screen_footer_hints.py
git diff --check
```

Expected: the footer module and static checks pass with no production changes.

### Task 4br: Retarget pending skill-script task state

**Files:**
- Modify: `Tests/UI/test_skill_script_confirm_card.py`

**ADR required:** no

**ADR path:** N/A

**Reason:** This aligns one direct-screen fixture with the existing native
Console task-state owner without changing the skill confirmation bridge.

- [ ] **Step 1: Preserve the focused RED**

Run `test_set_console_pending_skill_script_preserves_other_resume_fields`
alone. Expected before repair: fixture setup raises `AttributeError` because
the retired `ChatScreen.chat_state` wrapper no longer exists.

- [ ] **Step 2: Use the current task-state owner**

Seed the existing state through `screen.set_task_resume_state(...)`. Read
`screen._task_resume_state` for the add and clear assertions, retaining exact
summary, last-step, and pending payload coverage. Do not change production or
mount the screen.

- [ ] **Step 3: Verify the skill-script module**

```bash
../../.venv/bin/python -m pytest \
  Tests/UI/test_skill_script_confirm_card.py \
  -q
../../.venv/bin/python -m ruff check Tests/UI/test_skill_script_confirm_card.py
../../.venv/bin/python -m ruff format --check \
  Tests/UI/test_skill_script_confirm_card.py
git diff --check
```

Expected: the module and static checks pass with no production changes.

### Task 4bs: Keep the Parakeet file-loader test off the empty-audio path

**Files:**
- Modify: `Tests/Local_Ingestion/test_transcription_service_lazy_mlx.py`

**ADR required:** no

**ADR path:** N/A

**Reason:** This restores the loader test's non-empty-audio premise without
changing the approved zero-frame behavior.

- [ ] **Step 1: Preserve the focused RED**

Run `test_parakeet_file_model_construction_uses_loader` alone. Expected before
repair: `soundfile.info()` reports zero duration, production correctly returns
from the empty-audio fast path, and the loader sentinel is never raised.

- [ ] **Step 2: Supply non-empty metadata**

Change only the fake info result to realistic non-empty duration, frames, and
sample rate. Retain the import sentinel, chained error, exact loader call, and
debug assertion. Do not change production.

- [ ] **Step 3: Verify lazy MLX coverage**

```bash
../../.venv/bin/python -m pytest \
  Tests/Local_Ingestion/test_transcription_service_lazy_mlx.py \
  -q
../../.venv/bin/python -m ruff check \
  Tests/Local_Ingestion/test_transcription_service_lazy_mlx.py
../../.venv/bin/python -m ruff format --check \
  Tests/Local_Ingestion/test_transcription_service_lazy_mlx.py
git diff --check
```

Expected: the module and static checks pass with no production changes.

### Task 4bt: Reject missing local audio before Parakeet model loading

**Files:**
- Modify: `tldw_chatbook/Local_Ingestion/transcription_service.py`
- Modify: `Tests/Transcription/test_mlx_parakeet_transcription.py`

**ADR required:** no

**ADR path:** N/A

**Reason:** This is a routine input-validation bug fix at the existing
transcription service boundary and does not change provider or runtime
ownership.

- [ ] **Step 1: Pin the focused RED without network access**

Extend `test_real_transcription_invalid_file` so the Parakeet loader raises if
called, then retain the `TranscriptionError` assertion. Expected before repair:
the loader sentinel proves invalid input reaches model setup.

- [ ] **Step 2: Validate the shared local-file boundary**

Before conversion and provider dispatch, reject an audio path that does not
exist with `TranscriptionError`. Retain the conditional Parakeet-only
missing-file defense for direct helper calls that explicitly exercise the
SoundFile-unavailable path. Do not change existing-file, conversion, empty-file,
routing, or managed-download behavior.

- [ ] **Step 3: Verify the focused and provider-adjacent coverage**

```bash
../../.venv/bin/python -m pytest \
  Tests/Transcription/test_mlx_parakeet_transcription.py::TestMLXParakeetIntegration::test_real_transcription_invalid_file \
  Tests/Transcription/test_mlx_parakeet_integration.py::TestMLXParakeetIntegration::test_error_handling_invalid_file \
  -q
../../.venv/bin/python -m pytest \
  Tests/Transcription/test_mlx_parakeet_transcription.py \
  -q
../../.venv/bin/python -m ruff check \
  tldw_chatbook/Local_Ingestion/transcription_service.py \
  Tests/Transcription/test_mlx_parakeet_transcription.py
../../.venv/bin/python -m ruff format --check \
  tldw_chatbook/Local_Ingestion/transcription_service.py \
  Tests/Transcription/test_mlx_parakeet_transcription.py
git diff --check
```

Expected: the invalid-file regressions finish without model or network access,
the Parakeet transcription module passes, and static checks remain green.

### Task 4bu: Scope Model Artifacts scandir spies to the service call

**Files:**
- Modify: `Tests/Model_Artifacts/test_service.py`
- Modify: `Docs/superpowers/specs/2026-07-29-dev-gate-test-contract-repair-design.md`
- Modify: `backlog/tasks/task-1333 - Reconcile-stale-dev-gate-chat-and-audio-tests.md`

**ADR required:** no

**ADR path:** N/A

**Reason:** This is a test-isolation correction and does not change production
architecture or behavior.

- [x] **Step 1: Confirm the teardown failure**

Run the focused inventory test. Expected before repair: the test body passes,
then pytest teardown calls the process-wide `os.scandir` spy with an integer
directory descriptor and raises `TypeError`.

- [x] **Step 2: Bound both process-wide monkeypatches**

Use `monkeypatch.context()` around only the `list_installed()` and
`disk_usage()` calls in the two tests that replace `service_module.os.scandir`.
Keep their existing assertions and production code unchanged.

- [x] **Step 3: Verify the focused module**

```bash
../../.venv/bin/python -m pytest -q Tests/Model_Artifacts/test_service.py
../../.venv/bin/python -m ruff check Tests/Model_Artifacts/test_service.py
../../.venv/bin/python -m ruff format --check Tests/Model_Artifacts/test_service.py
```

Expected: all checks pass and pytest teardown uses the restored standard-library
function.

### Task 4bv: Isolate full-app runtime-policy notifications from catalog refresh

**Files:**
- Modify: `Tests/RuntimePolicy/test_runtime_policy_full_app.py`
- Modify: `Docs/superpowers/specs/2026-07-29-dev-gate-test-contract-repair-design.md`
- Modify: `backlog/tasks/task-1333 - Reconcile-stale-dev-gate-chat-and-audio-tests.md`

**ADR required:** no

**ADR path:** N/A

**Reason:** This is a test-isolation correction at an existing fixture seam and
does not change production architecture or behavior.

- [x] **Step 1: Confirm the full-suite-only race**

The full gate fails when startup model-catalog refresh appends an informational
notification after the focused test clears earlier startup messages. The exact
node, full test module, and RuntimePolicy package pass alone, confirming
order/timing dependence rather than a coordinator defect.

- [x] **Step 2: Suppress only the unrelated startup coroutine**

In `_configure_full_app_media_startup()`, replace the app instance's
`_refresh_model_catalogs()` coroutine with an async no-op before `run_test()`.
Do not filter notifications, wait on network-backed refresh work, or change
production.

- [x] **Step 3: Verify focused and adjacent coverage**

```bash
../../.venv/bin/python -m pytest -q \
  Tests/RuntimePolicy/test_runtime_policy_full_app.py
../../.venv/bin/python -m pytest -q Tests/RuntimePolicy
../../.venv/bin/python -m ruff check \
  Tests/RuntimePolicy/test_runtime_policy_full_app.py
../../.venv/bin/python -m ruff format --check \
  Tests/RuntimePolicy/test_runtime_policy_full_app.py
```

Expected: notification assertions remain exact, all runtime-policy coverage
passes, and no startup catalog worker can append unrelated messages.

### Task 4bw: Wait for the live provider Select overlay boundary

**Files:**
- Modify: `Tests/ProductionApp/test_provider_selection_ownership.py`
- Modify: `Tests/UI/test_settings_configuration_hub.py`
- Modify: `Docs/superpowers/specs/2026-07-29-dev-gate-test-contract-repair-design.md`
- Modify: `backlog/tasks/task-1333 - Reconcile-stale-dev-gate-chat-and-audio-tests.md`

**ADR required:** no

**ADR path:** N/A

**Reason:** This aligns two tests with the current Textual mount lifecycle and
does not change production architecture or behavior.

- [x] **Step 1: Confirm the stale readiness boundary**

The full gate and focused production-app node show that private `#label`
readiness is insufficient: the label can be absent, or present before the
`Select`'s required overlay exists, so assigning `value` raises `NoMatches`.

- [x] **Step 2: Wait on the required public child**

In both provider Settings regressions that wait for `#label`, wait for the
current screen's `#settings-provider-value OptionList` descendant and then
query the live `Select`. Remove the now-unused generic `Widget` import. Keep
all existing behavior assertions and production code unchanged.

- [x] **Step 3: Verify focused and adjacent coverage**

Run the two exact regressions, both changed modules, Ruff, format checks, and
`git diff --check`.

Expected: value assignment occurs only after the live Select is fully composed,
and all existing save, placeholder, session, and handoff assertions pass.

### Task 4bx: Give the real STT facade regression an existing input

**Files:**
- Modify: `Tests/STT/test_transcription_service_facade.py`
- Modify: `Docs/superpowers/specs/2026-07-29-dev-gate-test-contract-repair-design.md`
- Modify: `backlog/tasks/task-1333 - Reconcile-stale-dev-gate-chat-and-audio-tests.md`

**ADR required:** no

**ADR path:** N/A

**Reason:** This corrects a test fixture so it satisfies an existing public
input contract and does not change production architecture or behavior.

- [x] **Step 1: Confirm the focused failure**

Run the exact real-facade regression. Expected before repair: the production
missing-file guard rejects the nonexistent literal `audio.wav` before the
mocked recognizer is called.

- [x] **Step 2: Supply truthful local input**

Create an empty `audio.wav` under pytest's `tmp_path`, pass its string path to
the real facade, and retain the exact configured model and source-language
forwarding assertions. Do not mock path existence or change production.

- [x] **Step 3: Verify focused and adjacent STT coverage**

Run the exact node, its full module, the complete `Tests/STT/` package, Ruff on
the changed file, and `git diff --check`.

Expected: the exact node and module pass, all STT package tests pass, and the
production missing-file contract remains exercised by its dedicated tests.

### Task 4by: Keep the historical migration fixture at a truthful v17 shape

**Files:**
- Modify: `Tests/ChaChaNotesDB/test_chachanotes_db.py`
- Modify: `Docs/superpowers/specs/2026-07-29-dev-gate-test-contract-repair-design.md`
- Modify: `backlog/tasks/task-1333 - Reconcile-stale-dev-gate-chat-and-audio-tests.md`

**ADR required:** no

**ADR path:** N/A

**Reason:** This updates a synthetic historical-schema fixture for a newer
already-accepted migration and does not change storage architecture or
production migration behavior.

- [x] **Step 1: Confirm the focused failure**

Run the exact v17-to-v18 system-prompt migration regression. Expected before
repair: replay reaches v27-to-v28 and fails because the current-only
`assistant_authority_id` column was not removed by the synthetic rollback.

- [x] **Step 2: Remove the post-v17 authority column**

Drop `assistant_authority_id` from the current conversations table before
dropping `system_prompt` and setting the recorded schema version to 17. Keep
the existing removal of later provenance tables and every migration/trigger
assertion. Do not change production.

- [x] **Step 3: Verify focused and adjacent migration coverage**

Run the exact node, the full ChaChaNotesDB initialization module, the dedicated
v27-to-v28 character-authority migration module, Ruff/format on the changed
test, and `git diff --check`.

Expected: the synthetic v17 database replays through v28, dedicated authority
migration coverage remains green, and no production file changes.

### Task 4bz: Distinguish Console identity from persona presentation

**Files:**
- Modify: `Tests/Chat/test_console_session_settings.py`
- Modify: `Docs/superpowers/specs/2026-07-29-dev-gate-test-contract-repair-design.md`
- Modify: `backlog/tasks/task-1333 - Reconcile-stale-dev-gate-chat-and-audio-tests.md`

**ADR required:** no new ADR

**ADR path:** `backlog/decisions/037-roleplay-assistant-identity-and-persona-user-profile-separation.md`

**Reason:** This aligns a stale schema assertion with ADR-037's accepted
assistant-identity ownership without changing either production schema.

- [x] **Step 1: Confirm the focused failure**

Run the exact Console schema-ownership regression. Expected before repair:
`ConsoleChatSession` correctly contains durable `assistant_kind` and
`assistant_id`, contradicting the stale disjointness assertion.

- [x] **Step 2: Assert both accepted ownership boundaries**

Require runtime backend, assistant kind/id, and assistant authority on the
native session and reject them from session settings. Reject user/persona
labels and `assistant_name` from both schemas. Do not change production.

- [x] **Step 3: Verify focused and adjacent Console coverage**

Run the exact node, the full Console session-settings module, the Console chat
store module, the Console display-state module, Ruff/format on the changed
test, and `git diff --check`.

Expected: identity persistence and presentation separation remain explicit,
and all adjacent Console ownership coverage passes.

### Task 4ca: Keep all synthetic historical fixtures pre-v28

**Files:**
- Modify: `Tests/Chat/test_conversation_local_marks_service.py`
- Modify: `Tests/DB/test_chachanotes_world_book_priority_migration.py`
- Modify: `Tests/DB/test_chachanotes_world_book_regex_migration.py`
- Modify: `Docs/superpowers/specs/2026-07-29-dev-gate-test-contract-repair-design.md`
- Modify: `backlog/tasks/task-1333 - Reconcile-stale-dev-gate-chat-and-audio-tests.md`

**ADR required:** no

**ADR path:** N/A

**Reason:** This corrects synthetic historical-schema fixtures for an
already-accepted migration without changing storage architecture or production
migration behavior.

- [x] **Step 1: Inventory and reproduce the remaining failures**

Search for tests that create a current `CharactersRAGDB` and roll its schema
version back. Confirm the v16 local-marks and v20/v21 world-book paths all fail
at v27-to-v28 because the synthetic fixture retains
`assistant_authority_id`.

- [x] **Step 2: Remove the current-only authority column**

Drop `assistant_authority_id` before recording each historical schema version.
Keep all existing migration targets, later-provenance cleanup, trigger cleanup,
and outcome assertions. Do not change production.

- [x] **Step 3: Verify focused and adjacent migration coverage**

Run all three repaired migration modules, the full ChaChaNotesDB initialization
module, the dedicated v27-to-v28 authority suite, Ruff/format on the changed
tests, and `git diff --check`.

Expected: all historical databases replay through v28, the dedicated authority
migration remains strict and green, and no production file changes.

### Task 4cb: Measure sustained Chatbook import degradation

**Files:**
- Modify: `Tests/Chatbooks/test_chatbook_performance.py`
- Modify: `Docs/superpowers/specs/2026-07-29-dev-gate-test-contract-repair-design.md`
- Modify: `backlog/tasks/task-1333 - Reconcile-stale-dev-gate-chat-and-audio-tests.md`

**ADR required:** no

**ADR path:** N/A

**Reason:** This stabilizes a host-timing test while retaining its existing
performance contract and does not change runtime behavior.

- [x] **Step 1: Confirm host-timing variance**

Record the full-suite failure from one 39 ms deviation, then rerun the exact
test in isolation. Expected: the identical import path passes quickly, proving
the maximum-sample assertion is sensitive to unrelated suite/host load.

- [x] **Step 2: Compare robust early and late samples**

Compare early and late steady-state medians with a relative bound and small
absolute jitter floor. Keep all real Chatbook creation/import operations and
success assertions. Do not add retries or production instrumentation.

- [x] **Step 3: Verify focused and adjacent Chatbook coverage**

Run the exact node repeatedly through normal focused/module coverage, the full
Chatbooks performance module, Ruff/format, and `git diff --check`.

Expected: normal import variability passes, sustained late-import degradation
remains bounded, and no production file changes.

### Task 4cc: Retain thread-local connection identities

**Files:**
- Modify: `Tests/ChaChaNotesDB/test_chachanotes_db_properties.py`
- Modify: `Docs/superpowers/specs/2026-07-29-dev-gate-test-contract-repair-design.md`
- Modify: `backlog/tasks/task-1333 - Reconcile-stale-dev-gate-chat-and-audio-tests.md`

**ADR required:** no

**ADR path:** N/A

**Reason:** This removes object-lifetime ambiguity from an existing
thread-local identity test without changing storage or connection ownership.

- [x] **Step 1: Confirm object-id reuse**

Record the full-suite failure and inspect the captured connection logs.
Expected: five connections are created, but one short-lived worker/thread id
and one freed Python object address are reused before the final assertion.

- [x] **Step 2: Retain returned connection objects**

Append each worker's real connection object to a locked test-owned list and
derive object identities only after all threads join. Do not add scheduling
delays, retries, barriers, or production state.

- [x] **Step 3: Verify focused and adjacent concurrency coverage**

Run the exact node, the full ChaChaNotes property module, Ruff/format, and
`git diff --check`.

Expected: five distinct retained connection objects are observed under normal
thread scheduling and all adjacent property tests remain green.

### Task 4cd: Scope TTS unlink cleanup fakes to candidate validation

**Files:**
- Modify: `Tests/TTS/test_profile_schema.py`
- Modify: `Docs/superpowers/specs/2026-07-29-dev-gate-test-contract-repair-design.md`
- Modify: `backlog/tasks/task-1333 - Reconcile-stale-dev-gate-chat-and-audio-tests.md`

**ADR required:** no

**ADR path:** N/A

**Reason:** This confines test doubles for shared standard-library modules
without changing TTS storage, cleanup precedence, or runtime behavior.

- [x] **Step 1: Confirm teardown-only failure**

Record the full-suite teardown error and run both unlink-cleanup tests in
isolation. Expected: the bodies pass alone, while full-suite teardown calls the
still-patched process-global `os.unlink` with `dir_fd`.

- [x] **Step 2: Scope shared-module replacements**

Apply the existing `tempfile.mkstemp` and `os.unlink` replacements only around
`validate_profile_candidate()` through `monkeypatch.context()`. Keep private
body-failure injection in the same owned scope. Do not change production or
broaden the fakes to observe pytest cleanup.

- [x] **Step 3: Verify focused and adjacent TTS profile coverage**

Run both parametrized nodes, the complete profile-schema module, Ruff/format,
and `git diff --check`.

Expected: exact cleanup signal/error precedence remains green and pytest
temporary-directory teardown sees the real standard library.

### Task 4ce: Gate real Parakeet MLX tests on the loader API

**Files:**
- Modify: `Tests/Transcription/test_mlx_parakeet_transcription.py`
- Modify: `Tests/Transcription/test_mlx_parakeet_integration.py`
- Modify: `Docs/superpowers/specs/2026-07-29-dev-gate-test-contract-repair-design.md`
- Modify: `backlog/tasks/task-1333 - Reconcile-stale-dev-gate-chat-and-audio-tests.md`

**ADR required:** no

**ADR path:** N/A

**Reason:** This aligns optional real-integration test selection with the
existing production loader contract without changing provider behavior or
dependency architecture.

- [x] **Step 1: Confirm the capability mismatch**

Record the full-suite real-integration failure and inspect the dependency
cache. Expected: package discovery reports Parakeet MLX installed, but the
cached module lacks the callable `from_pretrained` API used by production.

- [x] **Step 2: Gate only real-model integration**

Require a callable `from_pretrained` on the already-cached optional module in
both real-integration entry points. Keep unit/mock tests active. Do not import
the package again, initialize MLX, or download a model during skip selection.

- [x] **Step 3: Verify Parakeet MLX coverage**

Run the failing exact node with skip reasons, both complete Parakeet MLX test
modules, the full Transcription package, Ruff/format, and `git diff --check`.

Expected: real-model cases skip with the precise missing-API reason on this
host, while mocked provider behavior remains green.

### Task 4cf: Classify all faster-whisper real-model tests as slow

**Files:**
- Modify: `Tests/Transcription/test_faster_whisper_transcription.py`
- Modify: `Docs/superpowers/specs/2026-07-29-dev-gate-test-contract-repair-design.md`
- Modify: `backlog/tasks/task-1333 - Reconcile-stale-dev-gate-chat-and-audio-tests.md`

**ADR required:** no

**ADR path:** N/A

**Reason:** This corrects test classification within the existing opt-in
real-inference boundary; it does not change provider behavior, artifact
ownership, or dependency architecture.

- [x] **Step 1: Reproduce the offline artifact download**

Run the complete Transcription package without `--run-slow`. Expected: the
unmarked empty-audio and progress tests instantiate `faster-whisper-tiny` and
attempt a Hugging Face download, while neighboring real-model tests skip.

- [x] **Step 2: Apply the existing slow-test contract**

Add `pytest.mark.slow` only to those two real-model cases. Leave the
invalid-file real integration active because it exits before model loading.

- [x] **Step 3: Verify faster-whisper and Transcription coverage**

Run the two exact nodes with skip reasons, the complete faster-whisper module,
the full Transcription package, Ruff/format, and `git diff --check`.

Expected: the ordinary offline gate performs no model download, unit/mock
coverage remains active, and real inference stays available through
`--run-slow`.

### Task 4cg: Isolate shared-RAG construction races

**Files:**
- Modify: `Tests/RAG/test_ingestion_indexing.py`
- Modify: `Docs/superpowers/specs/2026-07-29-dev-gate-test-contract-repair-design.md`
- Modify: `backlog/tasks/task-1333 - Reconcile-stale-dev-gate-chat-and-audio-tests.md`

**ADR required:** no

**ADR path:** N/A

**Reason:** This isolates controlled concurrency tests from unrelated
full-suite background work without changing the accepted production lock
design.

- [x] **Step 1: Reproduce lock pollution**

The full gate shows a real application embedding build holding the global
construction lock. A deterministic held-lock reproducer makes the exact test
fail before its patched constructor starts.

- [x] **Step 2: Isolate the controlled race**

Install a fresh build lock for every method through a class autouse fixture.
Do not change production code or the fast reset/set lock.

- [x] **Step 3: Verify RAG concurrency coverage**

Run the held-lock reproducer, the complete shared-lock class, the full
ingestion-indexing module, Ruff/format, and `git diff --check`.

### Task 4ch: Wait for the Evals results grid to mount

**Files:**
- Modify: `Tests/UI/test_evals_results_grid.py`
- Modify: `Docs/superpowers/specs/2026-07-29-dev-gate-test-contract-repair-design.md`
- Modify: `backlog/tasks/task-1333 - Reconcile-stale-dev-gate-chat-and-audio-tests.md`

**ADR required:** no

**ADR path:** N/A

**Reason:** This makes an existing test helper wait for an asynchronous
recompose; it does not change the Evals runtime contract.

- [x] **Step 1: Reproduce and isolate the mount race**

The full gate fails after 17,271 passes because one event-loop pause returns
before `#evals-results-grid` mounts. The exact test passes alone.

- [x] **Step 2: Bound the selector wait**

Make the shared run-group selection helper poll briefly for the results-grid
selector, then retain one settling pause before returning the typed widget.

- [x] **Step 3: Verify Evals results-grid coverage**

Run the exact regression repeatedly, the complete results-grid module,
Ruff/format, and `git diff --check`.

The exact regression passed three consecutive runs and the complete module
passed 42/42. Ruff lint and `git diff --check` pass. Ruff format reports the
same inherited whole-file drift on both the changed file and its untouched
`HEAD` version, so this correction does not rewrite unrelated test formatting.

### Task 4ci: Make file-notes lifecycle guards contention-tolerant

**Files:**
- Modify: `Tests/ProductionApp/test_file_notes_session_owner_lifecycle.py`
- Modify: `Docs/superpowers/specs/2026-07-29-dev-gate-test-contract-repair-design.md`
- Modify: `backlog/tasks/task-1333 - Reconcile-stale-dev-gate-chat-and-audio-tests.md`

**ADR required:** no

**ADR path:** N/A

**Reason:** These test-only timeouts guard against deadlocks; they do not define
an application performance contract.

- [x] **Step 1: Confirm contention-sensitive timeout behavior**

The full gate and exact test time out waiting one second for a controlled Git
commit signal while three other repository pytest runs are active. The same
contention stretches application startup to almost three seconds and pushes the
subsequent Git preflight beyond the one-second guard.

- [x] **Step 2: Share a bounded settlement timeout**

Use one 10-second module constant for all five controlled `asyncio.wait_for`
settlement guards. Successful waits still return immediately.

- [x] **Step 3: Verify file-notes owner lifecycle coverage**

Run the exact regression repeatedly under the current contention, the complete
owner-lifecycle module, Ruff/format, and `git diff --check`.

The exact regression passed three consecutive runs under the same concurrent
repository load and the complete module passed 9/9. Ruff lint and
`git diff --check` pass. Ruff format reports the same inherited whole-file
drift on both the changed file and its untouched `HEAD` version.

### Task 4cj: Wait for final Console recovery copy

**Files:**
- Modify: `Tests/UI/test_console_live_work_handoffs.py`
- Modify: `Docs/superpowers/specs/2026-07-29-dev-gate-test-contract-repair-design.md`
- Modify: `backlog/tasks/task-1333 - Reconcile-stale-dev-gate-chat-and-audio-tests.md`

**ADR required:** no

**ADR path:** N/A

**Reason:** This aligns a UI test wait with the existing Workflows background
load and recompose; it does not change application behavior.

- [x] **Step 1: Isolate the intermediate loading state**

The full gate fails after 15,741 passes because the Workflows loading button is
mounted and disabled while the final recovery `Static` is not yet mounted. The
exact parameter passes alone.

- [x] **Step 2: Wait for the asserted terminal copy**

Replace the fixed 0.1-second precondition with a bounded two-second loop for the
exact recovery copy. Keep the disabled-button and no-dispatch assertions.

- [x] **Step 3: Verify Console live-work handoff coverage**

Run both skeletal-destination parameters repeatedly, the complete live-work
handoff module, Ruff/format, and `git diff --check`.

Both parameters passed three consecutive runs and the complete module passed
48/48. Ruff lint, Ruff format, and `git diff --check` pass.

### Task 4ck: Bound File Notes Git hook-cleanup settlement

**Files:**
- Modify: `Tests/Notes/test_file_notes_git_commit_integration.py`
- Modify: `Docs/superpowers/specs/2026-07-29-dev-gate-test-contract-repair-design.md`
- Modify: `backlog/tasks/task-1333 - Reconcile-stale-dev-gate-chat-and-audio-tests.md`

**ADR required:** no

**ADR path:** N/A

**Reason:** This widens a test-only deadlock guard after a controlled commit is
released; it does not change Git behavior or define an application performance
contract.

- [x] **Step 1: Capture the contention-sensitive failure**

Full-gate attempt 23 fails after 9,012 passes when the released commit cycle
does not settle within one second. Pytest then remains in teardown until the
hung run is interrupted, while the production path reports no functional
failure.

- [x] **Step 2: Widen only the completion guard**

Give the released commit waiter ten seconds to settle. Preserve the one-second
controlled-start signal and the exact hook-directory lifetime, `Path.rmdir`,
and removal assertions.

- [x] **Step 3: Verify hook-cleanup coverage**

Run the exact regression repeatedly under the current contention, the complete
commit-integration module, Ruff/format, and `git diff --check`.

The exact regression passed three consecutive runs, including two concurrent
runs, and the complete commit-integration module passed 124/124 under the same
repository load. Ruff lint and `git diff --check` pass. Ruff format reports the
same inherited whole-file drift on both the changed file and its untouched
`HEAD` version.

### Task 4cl: Share File Notes Git integration settle bounds

**Files:**
- Modify: `Tests/Notes/test_file_notes_git_commit_integration.py`
- Modify: `Docs/superpowers/specs/2026-07-29-dev-gate-test-contract-repair-design.md`
- Modify: `backlog/tasks/task-1333 - Reconcile-stale-dev-gate-chat-and-audio-tests.md`

**ADR required:** no

**ADR path:** N/A

**Reason:** This consolidates test-only deadlock guards; it does not change Git
behavior or define an application performance contract.

- [x] **Step 1: Confirm the repeated root cause**

Full-gate attempt 25 fails after 9,011 passes at the sibling
`test_commit_confirmation_cancel_refuses_after_child_begins` post-release
waiter. Like attempt 23's hook-cleanup failure, a one-second timeout cancels
the retained commit waiter and leaves pytest stuck in teardown.

- [x] **Step 2: Use one bounded module timeout**

Replace the module's explicit one- and two-second `asyncio.wait_for` literals
with one ten-second constant. These waits cover controlled runner signals,
retained task settlement, and shutdown—not performance.

- [x] **Step 3: Verify File Notes Git integration coverage**

Run both full-gate failure regressions repeatedly under current contention, the
complete commit-integration module, Ruff/format, and `git diff --check`.

Both regressions passed 6/6 across three runs, including two concurrent runs
where their commit cycles took 1.12–1.26 seconds. The complete integration
module passed 124/124. Ruff lint and `git diff --check` pass. Ruff format
reports only the same inherited whole-file drift present in `HEAD`; changed
wait expressions match Ruff's proposed formatting.

### Task 4cm: Accept the visible Git focus fallback

**Files:**
- Modify: `Tests/UI/test_library_file_notes_git.py`
- Modify: `Docs/superpowers/specs/2026-07-29-dev-gate-test-contract-repair-design.md`
- Modify: `backlog/tasks/task-1333 - Reconcile-stale-dev-gate-chat-and-audio-tests.md`

**ADR required:** no

**ADR path:** N/A

**Reason:** This aligns one UI assertion with the production focus fallback
already covered elsewhere; it does not change application focus behavior.

- [x] **Step 1: Identify the bounded fallback**

Full-gate attempt 28 fails after 17,635 passes because current Git rows are
mounted but do not own focus. Production `_focus_session_git_panel()` explicitly
falls back to the visible Back control if rows remain undisplayed across its
bounded refresh retries, and another workspace focus test already accepts
either visible owner.

- [x] **Step 2: Assert the visible Git surface**

After the current row projection settles, require focus on either the Back
control or the row list. Keep the hidden-entry transition, current-status,
retained-action, and row-projection assertions unchanged.

- [x] **Step 3: Verify File Notes Git coverage**

Run the exact hidden-action journey repeatedly under current contention, the
complete File Notes Git module, Ruff/format, and `git diff --check`.

The exact journey passed three consecutive runs, including two concurrent
runs, and the complete File Notes Git module passed 138/138. Ruff lint and
`git diff --check` pass. Ruff format reports inherited whole-file drift only;
the changed focus assertion is already formatted as proposed.

### Task 4cn: Share native Console chat-flow settle bounds

**Files:**
- Modify: `Tests/UI/test_console_native_chat_flow.py`
- Modify: `Docs/superpowers/specs/2026-07-29-dev-gate-test-contract-repair-design.md`
- Modify: `backlog/tasks/task-1333 - Reconcile-stale-dev-gate-chat-and-audio-tests.md`

**ADR required:** no

**ADR path:** N/A

**Reason:** This adjusts test-only synchronization bounds for controlled
signals; it does not change Console runtime behavior or interfaces.

- [x] **Step 1: Identify the accidental performance gate**

Full-gate attempt 29 records
`test_console_composer_stop_is_subdued_when_idle` as the new failure after
roughly two-thirds of the repository passed. The isolated test passes, while
its fake gateway start signal and eight sibling controlled signals use one- or
two-second `asyncio.wait_for` literals despite the full gate running under
concurrent repository load.

- [x] **Step 2: Share one bounded settlement guard**

Replace the module's one- and two-second controlled-signal waits with one
ten-second constant. Keep all waits bounded, assertions exact, successful runs
immediate, and production code unchanged.

- [x] **Step 3: Verify native Console chat-flow coverage**

Run the exact failed journey repeatedly under current contention, the complete
native Console chat-flow module, Ruff/format, and `git diff --check`.

The exact journey passed three concurrent repetitions, and the complete native
Console chat-flow module passed 273/273 under repository contention. Ruff lint
and `git diff --check` pass. Ruff format reports inherited whole-file drift
only; none of the changed timeout lines appears in the formatter diff.

### Task 4co: Settle MCP workbench startup before lifecycle cancellation

**Files:**
- Modify: `Tests/UI/test_mcp_workbench.py`
- Modify: `Docs/superpowers/specs/2026-07-29-dev-gate-test-contract-repair-design.md`
- Modify: `backlog/tasks/task-1333 - Reconcile-stale-dev-gate-chat-and-audio-tests.md`

**ADR required:** no

**ADR path:** N/A

**Reason:** This aligns a test-only private-seam call with the mounted
workbench lifecycle; it does not change MCP runtime behavior or interfaces.

- [x] **Step 1: Identify the pre-mount private call**

Full-gate attempt 31 fails in
`test_cancel_requested_cancels_worker` because the test directly starts a
lifecycle sync while `MCPToolsMode` exists in the composed tree but its
`#mcp-tools-table` child has not mounted. The exact test passes in isolation,
and the real lifecycle action is unavailable to users until the workbench
controls mount.

- [x] **Step 2: Settle initial workers before the controlled lifecycle**

Use the same `app.workers.wait_for_complete()` boundary already used by sibling
lifecycle tests before invoking `_start_lifecycle()`. Keep the fake connect gate
unreleased after that point, issue the same cancel request, and retain the exact
in-flight cleanup assertion.

- [x] **Step 3: Verify MCP workbench coverage**

Run the exact cancellation test repeatedly under current contention, the
complete MCP workbench module, Ruff/format, and `git diff --check`.

The exact cancellation test passed three concurrent repetitions, and the
complete MCP workbench module passed 196/196. `git diff --check` passes. Ruff
lint reports four inherited errors elsewhere in the module, and Ruff format
reports inherited whole-file drift; the added worker-settlement line is absent
from both findings.

### Task 4cp: Wait for missing-note conflict controls to leave the DOM

**Files:**
- Modify: `Tests/UI/test_library_shell.py`
- Modify: `Docs/superpowers/specs/2026-07-29-dev-gate-test-contract-repair-design.md`
- Modify: `backlog/tasks/task-1333 - Reconcile-stale-dev-gate-chat-and-audio-tests.md`

**ADR required:** no

**ADR path:** N/A

**Reason:** This aligns two test assertions with Textual's asynchronous
recomposition after an already-correct editor reset; it does not change
Library behavior or interfaces.

- [x] **Step 1: Identify the state-before-DOM race**

Full-gate attempt 32 fails after 18,043 passes because missing-note Reload has
already reset the selected note, detail, autosave state, and notes view, but
the old conflict control remains for one render cycle. The exact test passes
in isolation.

- [x] **Step 2: Wait for the asserted DOM state**

Use the existing bounded condition helper in the Reload and symmetric
Overwrite missing-note regressions to wait until the old conflict control is
absent before retaining the exact reset-state assertions.

- [x] **Step 3: Verify Library conflict coverage**

Run both missing-note cases repeatedly under current contention, the complete
Library shell module, Ruff/format, and `git diff --check`.

Both missing-note cases passed three concurrent repetitions (6/6), and the
complete Library shell module passed 267/267. Ruff lint and `git diff --check`
pass. Ruff format reports four inherited formatting hunks elsewhere in the
module; neither added wait appears in that diff.

### Task 4cq: Settle MCP child mounts after startup workers

**Files:**
- Modify: `Tests/UI/test_mcp_workbench.py`
- Modify: `Docs/superpowers/specs/2026-07-29-dev-gate-test-contract-repair-design.md`
- Modify: `backlog/tasks/task-1333 - Reconcile-stale-dev-gate-chat-and-audio-tests.md`

**ADR required:** no

**ADR path:** N/A

**Reason:** This completes the test-only mounted-workbench boundary before a
private lifecycle call; it does not change MCP runtime behavior or interfaces.

- [x] **Step 1: Identify the worker-before-mount gap**

Full-gate attempt 33 fails after 18,684 passes because the cancellation test's
initial worker drain completes before Textual has mounted
`#mcp-inspector-state`. The private lifecycle's optimistic resync then reaches
the Inspector before that child exists. This is the same unavailable-to-users
pre-mount test seam as attempt 31, not a production lifecycle path.

- [x] **Step 2: Complete the established lifecycle boundary**

After the initial `app.workers.wait_for_complete()`, use the same
`pilot.pause()` post-worker mount settlement already used by sibling MCP
workbench tests. Keep the blocked connect gate, direct lifecycle start, cancel
request, and in-flight cleanup assertion unchanged.

- [x] **Step 3: Verify MCP workbench coverage**

Run the exact cancellation test repeatedly under current contention, the
complete MCP workbench module, Ruff/format, and `git diff --check`.

The exact cancellation test passed three concurrent repetitions, and the
complete MCP workbench module passed 196/196. `git diff --check` passes. Ruff
lint reports the same four inherited errors elsewhere in the module, and Ruff
format reports inherited whole-file drift; the added post-worker pause is
absent from both findings.

### Task 4cr: Wait for the retained File Notes subtree to remount

**Files:**
- Modify: `Tests/UI/test_library_file_notes_workspace.py`
- Modify: `Docs/superpowers/specs/2026-07-29-dev-gate-test-contract-repair-design.md`
- Modify: `backlog/tasks/task-1333 - Reconcile-stale-dev-gate-chat-and-audio-tests.md`

**ADR required:** no

**ADR path:** N/A

**Reason:** This aligns an existing test wait with the child identity it
immediately asserts after asynchronous Textual remount; it does not change
File Notes retention or source-switch behavior.

- [x] **Step 1: Identify the root-before-subtree race**

Full-gate attempt 34 fails after 17,720 passes because the retained File Notes
workspace root has remounted, but `#file-notes-editor` has not yet rejoined its
subtree. The immediate query fails and begins test teardown during the partial
mount. The exact journey passes in isolation.

- [x] **Step 2: Wait for the asserted retained editor**

Tighten the journey's existing final-remount predicate to require the editor
selector under the workspace before asserting the same workspace and editor
object identities. Preserve source switching, dirty/conflict veto, flush,
hidden-file refresh, and replica-lifetime coverage.

- [x] **Step 3: Verify File Notes workspace coverage**

Run the exact retained-workspace journey repeatedly under current contention,
the complete File Notes workspace module, Ruff/format, and `git diff --check`.

The exact retained-workspace journey passed three concurrent repetitions, and
the complete File Notes workspace module passed 27/27. Ruff lint and
`git diff --check` pass. Ruff format reports inherited formatting drift
elsewhere in the module; the tightened remount predicate is absent from those
findings.

### Task 5: Review and refresh the diagnostic inventory

**Files:**
- Modify: `Docs/security/production-diagnostic-inventory.json`
- Conditionally modify: a production diagnostic owner and its focused test only
  if review proves an ADR-029 violation

- [ ] **Step 1: Generate and compare the candidate**

Run:

```bash
../../.venv/bin/python -c \
  'import json; from scripts.check_persistent_diagnostic_inventory import build_inventory; print(json.dumps(build_inventory(), indent=2, sort_keys=True))' \
  > /tmp/task-1333-diagnostic-inventory.json
diff -u \
  Docs/security/production-diagnostic-inventory.json \
  /tmp/task-1333-diagnostic-inventory.json
```

Inspect the exact diff and compare every changed owner and every sink-topology
entry with the checked artifact. The checked repository file remains untouched.

- [ ] **Step 2: Review the changed call sites**

Inspect each added or changed production diagnostic. Confirm it logs only
approved operation identity, counts, lengths, status, duration, retry, posture,
and exception class names—not prompts, messages, request/response bodies,
credential fragments, file content, arbitrary values, or raw exception text.

- [ ] **Step 3: Resolve the review**

If every change is safe and sink topology is unchanged, retain the generated
artifact. If any call violates ADR-029, add a focused failing privacy
regression and make the smallest production correction before regenerating.

- [ ] **Step 4: Verify and commit**

```bash
../../.venv/bin/python scripts/check_persistent_diagnostic_inventory.py --write
../../.venv/bin/python scripts/check_persistent_diagnostic_inventory.py
../../.venv/bin/python -m pytest \
  Tests/Architecture/test_persistent_diagnostic_inventory.py -q
git add Docs/security/production-diagnostic-inventory.json
git commit -m "docs(security): refresh reviewed diagnostic inventory"
```

Include any conditionally required focused test/production files in that commit.

### Task 6: Verify and close TASK-1333

**Files:**
- Modify: `backlog/tasks/task-1333 - Reconcile-stale-dev-gate-chat-and-audio-tests.md`
- Modify: `Docs/superpowers/plans/2026-07-29-dev-gate-test-contract-repair.md`

- [ ] **Step 1: Run the affected suite**

```bash
../../.venv/bin/python -m pytest \
  Tests/Event_Handlers/test_worker_events_contract.py \
  Tests/Event_Handlers/test_retained_worker_adapter.py \
  Tests/Event_Handlers/test_eval_db_operations_path.py \
  Tests/UI/test_chat_shell_bar.py \
  Tests/Chat/test_chat_functions.py \
  Tests/Chat/test_scope_picker_listers.py \
  Tests/Library/test_library_rag_scope.py \
  Tests/Library/test_library_skills_state.py \
  Tests/Local_Ingestion/test_quick_ingest_db_path.py \
  Tests/Local_Ingestion/test_local_file_ingestion.py \
  Tests/MCP/test_control_plane_bridge.py \
  Tests/Performance/test_rag_citation_provenance_benchmark.py \
  Tests/Provider/test_provider_model_resolution.py \
  Tests/ProductionApp/test_chat_root_state_removal.py \
  Tests/ProductionApp/test_media_state_ownership.py \
  Tests/ProductionApp/test_personas_library_root_state.py \
  Tests/ProductionApp/test_provider_selection_ownership.py \
  Tests/UI/test_product_maturity_phase6_packaging_data_safety.py \
  Tests/RAG/test_rag_ui_integration.py \
  Tests/RAG/test_local_citation_capture.py \
  Tests/RAG_Admin/test_app_lazy_rag_admin_wiring.py \
  Tests/TTS/test_profile_backup_integration.py \
  Tests/TTS/test_tts_preferences.py \
  Tests/UI/test_command_palette_providers.py \
  Tests/UI/test_console_command_composer.py \
  Tests/UI/test_console_internals_decomposition.py \
  Tests/UI/test_console_live_work_handoffs.py \
  Tests/UI/test_console_mcp_approval.py \
  Tests/UI/test_console_session_settings.py \
  Tests/UI/test_home_screen.py \
  Tests/UI/test_library_prompts_canvas.py \
  Tests/UI/test_provider_model_resolution.py \
  Tests/integration/test_library_ingest_flow.py \
  Tests/test_config_delete_settings.py \
  Tests/Transcription/test_mlx_parakeet_integration.py \
  Tests/Transcription/test_mlx_parakeet_edge_cases.py \
  Tests/Transcription/test_mlx_parakeet_transcription.py \
  Tests/LLM/test_local_llm_provider_config.py \
  Tests/LLM_Provider_Catalog/test_local_openai_compatible_provider_name.py \
  Tests/DB/test_rag_indexing_db.py \
  Tests/Audio/test_audio_integration.py \
  Tests/Audio/test_recording_service.py \
  Tests/Architecture/test_persistent_diagnostic_inventory.py -q
```

Expected: all affected tests pass.

- [ ] **Step 2: Run static and diff checks**

```bash
../../.venv/bin/python -m ruff check \
  Tests/Event_Handlers/test_worker_events_contract.py \
  Tests/Event_Handlers/test_eval_db_operations_path.py \
  Tests/Chat/test_chat_functions.py \
  Tests/Chat/test_scope_picker_listers.py \
  Tests/Library/test_library_rag_scope.py \
  Tests/Library/test_library_skills_state.py \
  Tests/Local_Ingestion/test_quick_ingest_db_path.py \
  Tests/ProductionApp/test_chat_root_state_removal.py \
  Tests/ProductionApp/test_media_state_ownership.py \
  Tests/ProductionApp/test_personas_library_root_state.py \
  Tests/ProductionApp/test_provider_selection_ownership.py \
  Tests/UI/test_product_maturity_phase6_packaging_data_safety.py \
  Tests/RAG/test_rag_ui_integration.py \
  Tests/RAG_Admin/test_app_lazy_rag_admin_wiring.py \
  Tests/TTS/test_profile_backup_integration.py \
  Tests/TTS/test_tts_preferences.py \
  Tests/UI/test_command_palette_providers.py \
  Tests/UI/test_console_command_composer.py \
  Tests/UI/test_console_internals_decomposition.py \
  Tests/UI/test_console_live_work_handoffs.py \
  Tests/UI/test_console_mcp_approval.py \
  Tests/UI/test_console_session_settings.py \
  Tests/UI/test_home_screen.py \
  Tests/UI/test_library_prompts_canvas.py \
  Tests/UI/test_provider_model_resolution.py \
  Tests/integration/test_library_ingest_flow.py \
  Tests/test_config_delete_settings.py \
  Tests/Transcription/test_mlx_parakeet_integration.py \
  Tests/Transcription/test_mlx_parakeet_edge_cases.py \
  Tests/Transcription/test_mlx_parakeet_transcription.py \
  Tests/LLM/test_local_llm_provider_config.py \
  Tests/LLM_Provider_Catalog/test_local_openai_compatible_provider_name.py \
  Tests/DB/test_rag_indexing_db.py \
  Tests/Audio/test_audio_integration.py \
  Tests/Audio/test_recording_service.py \
  Helper_Scripts/Benchmarks/rag_citation_provenance_benchmark.py \
  tldw_chatbook/Event_Handlers/Chat_Events/chat_rag_events.py \
  tldw_chatbook/Library/library_skills_state.py \
  tldw_chatbook/UI/Screens/library_screen.py
../../.venv/bin/python -m ruff format --check \
  Tests/Event_Handlers/test_worker_events_contract.py \
  Tests/Event_Handlers/test_eval_db_operations_path.py \
  Tests/Chat/test_chat_functions.py \
  Tests/Chat/test_scope_picker_listers.py \
  Tests/Library/test_library_rag_scope.py \
  Tests/Library/test_library_skills_state.py \
  Tests/Local_Ingestion/test_quick_ingest_db_path.py \
  Tests/ProductionApp/test_chat_root_state_removal.py \
  Tests/ProductionApp/test_media_state_ownership.py \
  Tests/ProductionApp/test_provider_selection_ownership.py \
  Tests/UI/test_product_maturity_phase6_packaging_data_safety.py \
  Tests/RAG/test_rag_ui_integration.py \
  Tests/RAG_Admin/test_app_lazy_rag_admin_wiring.py \
  Tests/TTS/test_profile_backup_integration.py \
  Tests/TTS/test_tts_preferences.py \
  Tests/UI/test_command_palette_providers.py \
  Tests/UI/test_console_command_composer.py \
  Tests/UI/test_console_internals_decomposition.py \
  Tests/UI/test_console_live_work_handoffs.py \
  Tests/UI/test_console_mcp_approval.py \
  Tests/UI/test_console_session_settings.py \
  Tests/UI/test_home_screen.py \
  Tests/UI/test_library_prompts_canvas.py \
  Tests/UI/test_provider_model_resolution.py \
  Tests/integration/test_library_ingest_flow.py \
  Tests/test_config_delete_settings.py \
  Tests/Transcription/test_mlx_parakeet_integration.py \
  Tests/Transcription/test_mlx_parakeet_edge_cases.py \
  Tests/Transcription/test_mlx_parakeet_transcription.py \
  Tests/LLM/test_local_llm_provider_config.py \
  Tests/LLM_Provider_Catalog/test_local_openai_compatible_provider_name.py \
  Tests/DB/test_rag_indexing_db.py \
  Tests/Audio/test_audio_integration.py \
  Tests/Audio/test_recording_service.py \
  Helper_Scripts/Benchmarks/rag_citation_provenance_benchmark.py \
  tldw_chatbook/Event_Handlers/Chat_Events/chat_rag_events.py \
  tldw_chatbook/Library/library_skills_state.py \
  tldw_chatbook/Local_Ingestion/transcription_service.py \
  tldw_chatbook/UI/Screens/library_screen.py
../../.venv/bin/python scripts/check_persistent_diagnostic_inventory.py
git diff --check origin/dev...HEAD
git diff --check
```

Expected: all checks pass.

- [ ] **Step 3: Run the repository-wide gate**

```bash
../../.venv/bin/python -m pytest -q
```

Expected: the suite collects past the retired import, both PyAudio tests finish
deterministically, and the diagnostic inventory is current. Do not hide
unrelated or environment-dependent failures. Mark TASK-1333 Done only if the
repository Definition of Done is satisfied; otherwise record exact evidence
and leave it In Progress.

- [ ] **Step 4: Request final review**

Review the diff against TASK-1333 and the approved design. Fix only valid
Critical or Important findings and rerun affected verification. Do not make any
additional production behavior edit beyond a documented, test-first Task 5
correction for an actual ADR-029 violation; the Task 4m docstring clarification
is documentation-only.

- [ ] **Step 5: Complete Backlog hygiene**

Check every satisfied acceptance criterion, add concise Implementation Notes,
record the ADR decision and exact verification results, and use the Backlog CLI
to set TASK-1333 Done only when every gate is green.

- [ ] **Step 6: Commit closeout documentation**

```bash
git add \
  Docs/superpowers/plans/2026-07-29-dev-gate-test-contract-repair.md \
  "backlog/tasks/task-1333 - Reconcile-stale-dev-gate-chat-and-audio-tests.md"
git commit -m "docs(testing): record TASK-1333 verification"
```
