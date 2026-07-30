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
