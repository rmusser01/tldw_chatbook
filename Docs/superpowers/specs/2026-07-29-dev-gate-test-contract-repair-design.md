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
  model was deliberately retired. The shell context still accepts state-shaped
  objects through its compatibility-shaped helper.
- `Tests/Audio/test_audio_integration.py` starts a recorder thread and then
  calls the same recording loop repeatedly on the test thread. It also leaves
  VAD enabled, so the assertion depends on installed optional dependencies and
  whether synthetic bytes are classified as speech.

TASK-627 remains the inventory ledger for broader baseline failures. TASK-1333
owns only these deterministic repairs.

## Decision

Update the tests to describe current behavior:

1. Keep the non-streaming worker failure regression. Replace the retired
   streaming-sentinel assertion with the live contract: streaming raises
   `ValueError` and posts no application message.
2. Replace the `TabState` import with a tiny local state-shaped fixture while
   retaining coverage of `ChatShellContext.from_tab_state`.
3. Run the PyAudio stream-error test synchronously with VAD disabled. Assert
   that the two chunks before the error reach the callback, the stream closes,
   and recording stops. Rename the test so it no longer claims automatic
   recovery that production does not implement.

No production file changes. No compatibility shims. No broad test deletion.

## Alternatives

- Restoring `StreamDone` or `TabState` would contradict the accepted retirement
  architecture and revive dead ownership.
- Deleting the tests would make collection green but discard useful retained
  adapter, shell-label, and audio-error coverage.
- The selected rewrite keeps that coverage with the smallest honest contract.

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
