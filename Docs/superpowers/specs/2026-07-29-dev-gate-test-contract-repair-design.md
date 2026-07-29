# Dev-Gate Test Contract Repair Design

## Goal

Restore the mandatory `dev` pytest gate by reconciling three tests with the
production contracts that already exist. This is a test-only repair: it must
not restore retired Chat infrastructure or change audio-recording behavior.

## Evidence

The failures reproduce on an exact `origin/dev` checkout:

- `Tests/Event_Handlers/test_worker_events_contract.py` imports `StreamDone`,
  although TASK-577 deliberately removed that event and made the retained
  adapter reject streaming.
- `Tests/UI/test_chat_shell_bar.py` imports `TabState`, although the legacy tab
  model was deliberately retired. Its `from_tab_state` helper has no live
  production caller and should not gain replacement fixture coverage.
- `Tests/Audio/test_audio_integration.py` starts a recorder thread and then
  calls the same recording loop repeatedly on the test thread. It also leaves
  VAD enabled, so the assertion depends on installed optional dependencies and
  whether synthetic bytes are classified as speech.

TASK-1333 owns only these deterministic repairs.

## Decision

Update the tests to describe current behavior:

1. Keep the non-streaming worker failure regression and delete its obsolete
   streaming-sentinel case. The existing
   `Tests/Event_Handlers/test_retained_worker_adapter.py` already pins the live
   streaming-rejection contract, so TASK-1333 adds no duplicate.
2. Remove the `TabState` import and the retired-state half of the combined
   shell-context test. Keep the live `ChatSessionData` label assertions. Do not
   add a replacement fixture for the unused `from_tab_state` helper.
3. Run `_pyaudio_recording_loop()` exactly once with `is_recording = True` and
   VAD disabled, without calling `start_recording()`. Assert the exact two-chunk
   callback sequence, `stop_stream()`, `close()`, and final stopped state.
   Rename the test so it no longer claims automatic recovery that production
   does not implement.

No production file changes. No compatibility shims. No broad test deletion.

## Alternatives

- Restoring `StreamDone` or `TabState` would contradict the accepted retirement
  architecture and revive dead ownership.
- Deleting whole test files would make collection green but discard useful
  retained non-streaming, shell-label, and audio-error coverage.
- Replacing retired models with local fixtures or duplicating the existing
  streaming-rejection test would keep dead or redundant coverage alive.
- The selected edits remove only obsolete assertions and make the remaining
  audio contract deterministic.

## Verification

Run the three repaired modules first, then their nearby affected suites, static
checks on changed tests, and the repository-wide suite. TASK-1333 is complete
only when these three failures are absent; unrelated or environment-dependent
failures remain recorded rather than hidden.

## Architecture Decision Record

ADR required: no

ADR path: N/A

Reason: This reconciles tests with already-accepted production boundaries and
does not change runtime architecture, storage, security, dependencies, or
cross-module interfaces.
