# Dev-Gate Test Contract Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove three obsolete or nondeterministic test failures from the current `dev` gate without changing production behavior.

**Architecture:** Edit only the three failing test modules. Delete assertions for retired or already-covered Chat contracts, and make the PyAudio error test synchronous and independent of optional VAD behavior.

**Tech Stack:** Python 3.11+, pytest, unittest.mock, Ruff.

**ADR required:** no

**ADR path:** N/A

**Reason:** This reconciles tests with accepted runtime contracts and changes no architecture, storage, dependency, security, or production interface.

---

## Scope guard

Do not edit production files. Do not restore `StreamDone` or `TabState`. Do not
add replacement coverage for the unused `from_tab_state` helper. Do not
duplicate the streaming-rejection test already present in
`Tests/Event_Handlers/test_retained_worker_adapter.py`.

## File map

- Modify `Tests/Event_Handlers/test_worker_events_contract.py`: retain only the
  unique non-streaming exception regression.
- Modify `Tests/UI/test_chat_shell_bar.py`: remove the retired `TabState` half
  of the combined context test while retaining live `ChatSessionData` labels.
- Modify `Tests/Audio/test_audio_integration.py`: make stream-error behavior
  synchronous, VAD-independent, and exact.
- Modify TASK-1333 and this plan only for closeout evidence.

### Task 1: Remove the retired worker-event contract

**Files:**
- Modify: `Tests/Event_Handlers/test_worker_events_contract.py`
- Verify: `Tests/Event_Handlers/test_retained_worker_adapter.py`

- [ ] **Step 1: Reproduce RED**

Run:

```bash
../../.venv/bin/python -m pytest \
  Tests/Event_Handlers/test_worker_events_contract.py -q
```

Expected: collection fails because `StreamDone` was deliberately removed.

- [ ] **Step 2: Make the smallest test correction**

Update the module description so it claims only the retained non-streaming
failure contract. Import only `chat_wrapper_function`. Delete
`test_chat_wrapper_function_streaming_failure_keeps_sentinel_contract`; do not
replace it because
`test_retained_worker_adapter_rejects_legacy_streaming_bridge` already covers
the live streaming rejection.

- [ ] **Step 3: Verify GREEN**

Run:

```bash
../../.venv/bin/python -m pytest \
  Tests/Event_Handlers/test_worker_events_contract.py \
  Tests/Event_Handlers/test_retained_worker_adapter.py -q
```

Expected: both modules pass.

- [ ] **Step 4: Commit**

```bash
git add Tests/Event_Handlers/test_worker_events_contract.py
git commit -m "test(chat): remove retired worker-event contract"
```

### Task 2: Remove the retired tab-state fixture

**Files:**
- Modify: `Tests/UI/test_chat_shell_bar.py`

- [ ] **Step 1: Reproduce RED**

Run:

```bash
../../.venv/bin/python -m pytest Tests/UI/test_chat_shell_bar.py -q
```

Expected: collection fails because `TabState` was deliberately removed.

- [ ] **Step 2: Retain only live session coverage**

Remove the `TabState` import, construction, `from_tab_state` call, and
`tab_context` assertions. Rename the combined test to
`test_chat_shell_context_supports_chat_session_data`. Its resolver needs only
`workspace_name="Research Lab"` and `character_label="Vox"`. Keep the existing
`ChatSessionData` assertions for Server, Workspace, Character, and Session
labels.

- [ ] **Step 3: Verify GREEN**

Run:

```bash
../../.venv/bin/python -m pytest Tests/UI/test_chat_shell_bar.py -q
```

Expected: the module collects and passes.

- [ ] **Step 4: Commit**

```bash
git add Tests/UI/test_chat_shell_bar.py
git commit -m "test(chat): remove retired tab-state fixture"
```

### Task 3: Make the stream-error test deterministic

**Files:**
- Modify: `Tests/Audio/test_audio_integration.py`

- [ ] **Step 1: Reproduce RED**

Run:

```bash
../../.venv/bin/python -m pytest \
  Tests/Audio/test_audio_integration.py::TestErrorRecovery::test_recording_recovery_from_stream_error \
  -q
```

Expected: the assertion sees zero callbacks when installed VAD rejects the
synthetic chunks; the test also races the recorder thread it starts.

- [ ] **Step 2: Replace the nondeterministic test body**

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

- [ ] **Step 3: Verify GREEN**

Run:

```bash
../../.venv/bin/python -m pytest \
  Tests/Audio/test_audio_integration.py::TestErrorRecovery::test_recording_stops_after_stream_error_and_preserves_prior_chunks \
  -q
```

Expected: the test passes deterministically.

- [ ] **Step 4: Commit**

```bash
git add Tests/Audio/test_audio_integration.py
git commit -m "test(audio): make stream-error coverage deterministic"
```

### Task 4: Verify and close TASK-1333

**Files:**
- Modify: `backlog/tasks/task-1333 - Reconcile-stale-dev-gate-chat-and-audio-tests.md`
- Modify: `Docs/superpowers/plans/2026-07-29-dev-gate-test-contract-repair.md`

- [ ] **Step 1: Run the affected suite**

```bash
../../.venv/bin/python -m pytest \
  Tests/Event_Handlers/test_worker_events_contract.py \
  Tests/Event_Handlers/test_retained_worker_adapter.py \
  Tests/UI/test_chat_shell_bar.py \
  Tests/Audio/test_audio_integration.py \
  Tests/Audio/test_recording_service.py -q
```

Expected: all affected tests pass.

- [ ] **Step 2: Run static and diff checks**

```bash
../../.venv/bin/python -m ruff check \
  Tests/Event_Handlers/test_worker_events_contract.py \
  Tests/UI/test_chat_shell_bar.py \
  Tests/Audio/test_audio_integration.py
../../.venv/bin/python -m ruff format --check \
  Tests/Event_Handlers/test_worker_events_contract.py \
  Tests/UI/test_chat_shell_bar.py \
  Tests/Audio/test_audio_integration.py
git diff --check origin/dev...HEAD
git diff --check
```

Expected: all checks pass.

- [ ] **Step 3: Run the repository-wide gate**

```bash
../../.venv/bin/python -m pytest -q
```

Expected: the suite collects past the retired imports and no longer reports the
three TASK-1333 failures. Do not hide unrelated or environment-dependent
failures. Mark TASK-1333 Done only if the repository Definition of Done is
satisfied; otherwise record exact evidence and leave it In Progress.

- [ ] **Step 4: Request final review**

Review the diff against TASK-1333 and the approved design. Fix only valid
Critical or Important findings, rerun affected verification, and keep all
production files untouched.

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
