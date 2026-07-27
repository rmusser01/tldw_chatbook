# Console Microphone Dictation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a user record one bounded English utterance from the native Console composer and insert its Parakeet v2 INT8 transcript at the current caret.

**Architecture:** Add a small audio-domain session that composes the existing `AudioRecordingService` and `TranscriptionService`; it records one in-memory PCM buffer and performs one transcription after capture stops. `ChatScreen` owns the session lifecycle, schedules a strict 60-second wall-clock timer, and runs start/stop/transcription work in Textual thread workers, while `ConsoleComposerBar` owns only the microphone button presentation and draft insertion. Extend the existing Parakeet ONNX adapter to pass normalized NumPy audio directly to `onnx-asr`, avoiding a temporary WAV. This is an immediately usable vertical slice under TASK-603; it deliberately does not claim completion of the parent task's future `LocalSTTExecutor`, batch-priority, or bounded-IPC gates.

**Tech Stack:** Python 3.11+, Textual workers/widgets, existing PyAudio/sounddevice recorder, existing `onnx-asr` Parakeet v2 adapter, pytest.

**ADR required:** yes

**ADR path:** `backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md`

**Reason:** ADR-025 already selects explicit English → Parakeet v2 INT8, forbids implicit downloads, preserves buffer transcription, and rejects false streaming claims. No new ADR is needed.

---

### Task 1: Keep captured audio bounded in memory

**Files:**
- Modify: `tldw_chatbook/Audio/recording_service.py`
- Test: `Tests/Audio/test_recording_service.py`

- [x] Add a failing test proving a configured byte limit retains only complete PCM frames, stops recording, and invokes one visible limit callback.
- [x] Run the focused test and confirm it fails because `AudioRecordingService` has no buffer limit.
- [x] Add optional `max_buffer_bytes` and `on_buffer_limit` constructor arguments; reset the byte count on each recording and stop before the limit can be exceeded.
- [x] Run `Tests/Audio/test_recording_service.py`.

### Task 2: Add memory-only Parakeet ONNX buffer transcription

**Files:**
- Modify: `tldw_chatbook/Local_Ingestion/transcription_service.py`
- Modify: `Tests/Transcription/test_parakeet_onnx_vertical_slice.py`

- [x] Add a failing test that sends PCM bytes through `transcribe_buffer()` and proves Parakeet receives a normalized NumPy waveform plus the original sample rate, without opening or creating an audio file.
- [x] Run the focused test and confirm the Parakeet buffer path is absent and falls into temporary-WAV staging.
- [x] Reuse the existing validated/cached Parakeet model loader for file and buffer entry points, and call `onnx-asr` directly with the in-memory NumPy waveform.
- [x] Run the focused Parakeet ONNX tests.

### Task 3: Add one-shot Parakeet Console dictation session

**Files:**
- Create: `tldw_chatbook/Audio/console_dictation.py`
- Create: `Tests/Audio/test_console_dictation.py`

- [x] Add failing tests for explicit `en`, Parakeet v2 INT8, model-directory forwarding, empty capture, failed recorder start, and discard without transcription.
- [x] Run the focused tests and confirm they fail because the session does not exist.
- [x] Implement `ConsoleDictationSession` by composing `AudioRecordingService` and `TranscriptionService`, loading native dependencies lazily, and returning only stripped transcript text.
- [x] Resolve the model directory from the explicit/configured path or the verified Library installer destination; require the four local bundle files and never download.
- [x] Run the focused audio tests.

### Task 4: Add Console microphone control and lifecycle

**Files:**
- Modify: `tldw_chatbook/Widgets/Console/console_composer_bar.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss`
- Modify: `tldw_chatbook/css/tldw_cli_modular.tcss`
- Create: `Tests/UI/test_console_dictation.py`
- Modify: `Tests/UI/test_console_internals_decomposition.py`

- [x] Add mounted failing tests proving the Mic control exists, start/stop work is delegated, success inserts at the caret without sending, and state labels are clear.
- [x] Add mounted failing tests for missing dependency, missing model files, microphone failure, empty audio, and transcription failure; each must assert the user-visible message, unchanged draft, and recovery to idle.
- [x] Add a mounted test proving start schedules a 60-second wall-clock timer and expiry visibly transitions to transcribing before invoking the stop/transcribe worker.
- [x] Run the focused mounted tests and confirm they fail because the control and handlers do not exist.
- [x] Add an eight-cell `Mic` action and composer state synchronizer for idle, starting, recording, and transcribing.
- [x] Add `ChatScreen` lifecycle methods that run blocking audio/STT calls in exclusive thread workers, guard duplicate requests, retain the originating session identity, cancel the wall-clock timer on manual stop/error, and discard capture on unmount.
- [x] Insert successful text with boundary-aware spacing at the current caret; update an off-screen originating session draft instead of inserting into the wrong chat.
- [x] Expand the fixed action-row width in both source and bundled CSS and update the exact CSS contract assertion.
- [x] Run the focused mounted Console tests.

### Task 5: Verify and close the atomic task

**Files:**
- Modify: `backlog/tasks/task-603.1 - Add-Console-microphone-transcription-vertical-slice.md`

- [x] Run focused audio and Console tests with the MLX import isolated from this environment's known eager-import abort.
- [x] Run Ruff on changed Python files and `git diff --check`.
- [x] Perform a self-review against all TASK-603.1 acceptance criteria.
- [x] Record implementation notes, check every TASK-603.1 acceptance criterion, and mark only TASK-603.1 Done; leave parent TASK-603 open.
