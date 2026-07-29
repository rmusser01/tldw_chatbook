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
  Tests/Performance/test_rag_citation_provenance_benchmark.py \
  Tests/ProductionApp/test_chat_root_state_removal.py \
  Tests/ProductionApp/test_media_state_ownership.py \
  Tests/ProductionApp/test_provider_selection_ownership.py \
  Tests/RAG/test_rag_ui_integration.py \
  Tests/RAG/test_local_citation_capture.py \
  Tests/RAG_Admin/test_app_lazy_rag_admin_wiring.py \
  Tests/TTS/test_profile_backup_integration.py \
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
  Tests/ProductionApp/test_provider_selection_ownership.py \
  Tests/RAG/test_rag_ui_integration.py \
  Tests/RAG_Admin/test_app_lazy_rag_admin_wiring.py \
  Tests/TTS/test_profile_backup_integration.py \
  Tests/LLM/test_local_llm_provider_config.py \
  Tests/LLM_Provider_Catalog/test_local_openai_compatible_provider_name.py \
  Tests/DB/test_rag_indexing_db.py \
  Tests/Audio/test_audio_integration.py \
  Tests/Audio/test_recording_service.py \
  Helper_Scripts/Benchmarks/rag_citation_provenance_benchmark.py \
  tldw_chatbook/Event_Handlers/Chat_Events/chat_rag_events.py \
  tldw_chatbook/Library/library_skills_state.py
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
  Tests/RAG/test_rag_ui_integration.py \
  Tests/RAG_Admin/test_app_lazy_rag_admin_wiring.py \
  Tests/TTS/test_profile_backup_integration.py \
  Tests/LLM/test_local_llm_provider_config.py \
  Tests/LLM_Provider_Catalog/test_local_openai_compatible_provider_name.py \
  Tests/DB/test_rag_indexing_db.py \
  Tests/Audio/test_audio_integration.py \
  Tests/Audio/test_recording_service.py \
  Helper_Scripts/Benchmarks/rag_citation_provenance_benchmark.py \
  tldw_chatbook/Event_Handlers/Chat_Events/chat_rag_events.py \
  tldw_chatbook/Library/library_skills_state.py
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
