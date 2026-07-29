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
`Tests/Event_Handlers/test_retained_worker_adapter.py`. Do not edit production
files unless diagnostic review finds an actual ADR-029 violation, and never
regenerate the inventory before reviewing its changed owners and sink topology.

## File map

- Modify `Tests/Event_Handlers/test_worker_events_contract.py`: retain only the
  unique non-streaming exception regression.
- Verify `Tests/UI/test_chat_shell_bar.py`: preserve the current `dev` repair
  without a branch edit.
- Modify `Tests/Audio/test_audio_integration.py`: make stream-error behavior
  synchronous, VAD-independent, and exact.
- Modify `Tests/Audio/test_recording_service.py`: make PyAudio happy-flow
  behavior synchronous, VAD-independent, and exact.
- Modify `Tests/Chat/test_chat_functions.py`: patch the live runtime-config
  snapshot seam instead of deleted module-level settings.
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
  Tests/UI/test_chat_shell_bar.py \
  Tests/Chat/test_chat_functions.py \
  Tests/Audio/test_audio_integration.py \
  Tests/Audio/test_recording_service.py \
  Tests/Architecture/test_persistent_diagnostic_inventory.py -q
```

Expected: all affected tests pass.

- [ ] **Step 2: Run static and diff checks**

```bash
../../.venv/bin/python -m ruff check \
  Tests/Event_Handlers/test_worker_events_contract.py \
  Tests/Chat/test_chat_functions.py \
  Tests/Audio/test_audio_integration.py \
  Tests/Audio/test_recording_service.py
../../.venv/bin/python -m ruff format --check \
  Tests/Event_Handlers/test_worker_events_contract.py \
  Tests/Chat/test_chat_functions.py \
  Tests/Audio/test_audio_integration.py \
  Tests/Audio/test_recording_service.py
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
additional production edit beyond a documented, test-first Task 5 correction
for an actual ADR-029 violation.

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
