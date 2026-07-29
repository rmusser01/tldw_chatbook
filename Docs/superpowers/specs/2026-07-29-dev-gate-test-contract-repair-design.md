# Dev-Gate Test Contract Repair Design

## Goal

Restore the mandatory `dev` pytest gate by reconciling stale or nondeterministic
tests with the production contracts that already exist, then review and refresh
the checked diagnostic inventory under ADR-029. This is a non-production
repair: it must not restore retired Chat infrastructure, change audio-recording
behavior, or silently admit unsafe persistent diagnostics.

## Evidence

The failures reproduce on an exact `origin/dev` checkout:

- `Tests/Event_Handlers/test_worker_events_contract.py` imports `StreamDone`,
  although TASK-577 deliberately removed that event and made the retained
  adapter reject streaming.
- Latest `dev` independently removed the retired `TabState` fixture from
  `Tests/UI/test_chat_shell_bar.py` while adding current persona-label coverage.
  TASK-1333 preserves that upstream repair rather than carrying a competing
  edit.
- `Tests/Audio/test_audio_integration.py` starts a recorder thread and then
  calls the same recording loop repeatedly on the test thread. It also leaves
  VAD enabled, so the assertion depends on installed optional dependencies and
  whether synthetic bytes are classified as speech.
- `Tests/Audio/test_recording_service.py` repeats the same thread-and-direct-loop
  race with VAD enabled, so synthetic audio may never reach the callback that
  stops the loop.
- The adjacent SoundDevice flow fixture also leaves VAD enabled for a
  four-sample synthetic callback, so no audio reaches its queue assertion.
- Four `Tests/Chat/test_chat_functions.py` cases monkeypatch deleted
  module-level `settings` objects. The live Llama.cpp and DeepSeek adapters
  deliberately resolve `get_runtime_config_snapshot()` at each request
  boundary under ADR-029.
- `Tests/Architecture/test_persistent_diagnostic_inventory.py` reports reviewed
  production-owner drift while the persistent sink topology remains unchanged.
  ADR-029 requires inspecting the changed calls before regenerating the checked
  inventory.

TASK-1333 owns these gate repairs and the reviewed generated-inventory refresh.

## Decision

Update the tests to describe current behavior:

1. Keep the non-streaming worker failure regression and delete its obsolete
   streaming-sentinel case. The existing
   `Tests/Event_Handlers/test_retained_worker_adapter.py` already pins the live
   streaming-rejection contract, so TASK-1333 adds no duplicate.
2. Preserve the latest `dev` chat-shell test unchanged. It already covers the
   live session and persona-label contract without `TabState`.
3. In the stream-error regression, run `_pyaudio_recording_loop()` exactly once
   with `is_recording = True` and
   VAD disabled, without calling `start_recording()`. Assert the exact two-chunk
   callback sequence, `stop_stream()`, `close()`, and final stopped state.
   Rename the test so it no longer claims automatic recovery that production
   does not implement.
4. Apply the same deterministic structure to the PyAudio happy-flow regression:
   disable VAD, set the callback and recording state directly, invoke one loop,
   and assert exactly three chunks plus cleanup. Do not start a background
   recorder.
5. Keep the SoundDevice flow through its public start/stop contract, but disable
   VAD for its tiny synthetic callback. Use a bounded event set by the mocked
   `InputStream` constructor before reading the captured callback, and stop the
   mocked recorder in `finally` before asserting that audio was queued.
6. Update the stale provider request tests to patch
   `get_runtime_config_snapshot()` with a `RuntimeConfigSnapshot` containing
   their test configuration. Do not restore or emulate mutable module-level
   settings.
7. Generate the candidate diagnostic inventory, inspect every changed owner and
   sink-topology entry against ADR-029's metadata-only boundary, and refresh the
   checked artifact only if the changes are safe. If review finds an unsafe log
   value, correct that violation under ADR-029 before regenerating; do not bless
   it as inventory drift.

No planned production file changes. No compatibility shims. No broad test
deletion. A production diagnostic may change only if the required ADR-029
review identifies an actual privacy violation.

## Alternatives

- Restoring `StreamDone` or replacing the upstream `TabState` repair would
  contradict the accepted retirement architecture and revive dead ownership.
- Deleting whole test files would make collection green but discard useful
  retained non-streaming, shell-label, and audio-error coverage.
- Replacing retired models with local fixtures or duplicating the existing
  streaming-rejection test would keep dead or redundant coverage alive.
- Blindly regenerating the diagnostic inventory would defeat its review gate.
- The selected edits remove only obsolete assertions, make the audio contracts
  deterministic, and preserve the existing privacy boundary.

## Verification

Run the repaired modules first, then their nearby affected suites, the
diagnostic-inventory guard, static checks on changed files, and the
repository-wide suite. TASK-1333 is complete only when these failures are
absent; unrelated or environment-dependent failures remain recorded rather
than hidden.

## Architecture Decision Record

ADR required: no

ADR path: backlog/decisions/029-local-private-data-boundary.md

Reason: This reconciles tests with already-accepted production boundaries and
applies ADR-029's existing metadata-only inventory review requirement. It does
not introduce a new runtime, storage, security, dependency, or cross-module
decision.
