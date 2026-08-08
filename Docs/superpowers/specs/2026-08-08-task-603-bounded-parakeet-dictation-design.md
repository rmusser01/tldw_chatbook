# TASK-603: Bounded Parakeet ONNX Dictation Design

**Date:** 2026-08-08

**Status:** Approved direction; failure-oriented review corrections incorporated

**Task:** TASK-603 — Restore bounded Parakeet ONNX dictation buffers

**Architecture:** [ADR-025](../../../backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md)

## Goal

Route Parakeet ONNX microphone and public in-memory buffer transcription through
the app's existing `LocalSTTExecutor`. Preserve the Console microphone,
hands-free, voice-command, and caret-insertion behavior without creating a
second native model process, staging microphone audio on disk, or claiming true
streaming.

This task also supplies the smallest scheduler needed to let interactive
dictation wait behind the currently running batch inference, run before the
next batch item, and then let the batch continue automatically.

## Confirmed product decisions

- English remains the dictation default and routes to Parakeet v2 with INT8 as
  the default precision; an explicitly persisted F32 selection remains usable.
- Parakeet ONNX is buffer transcription, not true streaming.
- A Mic press reserves the next local-STT slot automatically. There is no extra
  pause prompt or separate batch-pause control.
- An active batch inference is never preempted.
- While waiting, the Console shows **Local transcription busy — dictation will
  run next.**
- When the capture bound is reached, capture stops, every retained sample is
  transcribed, and the Console shows **Limit reached — press Mic to continue.**
  Recording never restarts automatically.
- The existing Console interaction model remains intact: transcription inserts
  into the draft and does not send automatically.

## Non-goals

- Do not implement token/audio streaming for Parakeet ONNX.
- Do not change the global STT default promotion or remove retained providers;
  those remain TASK-605 work after the release gates pass.
- Do not add another process pool, worker process, general-purpose priority
  queue, or persisted audio queue.
- Do not write microphone PCM to a temporary WAV file.
- Do not redesign the Console composer, hands-free loop, or voice commands.
- Do not run the repository-wide test suite for this task.

## Current state

`LocalSTTExecutor` already owns a single spawn-context worker, one resident model
identity, artifact leases, cancellation, generation fencing, descendant-process
containment, and worker recycling. It deliberately accepts one request at a
time and has no internal queue. Library Parakeet/transcribe.cpp jobs use it with
file paths.

The public `TranscriptionService.transcribe_buffer()` still sends Parakeet ONNX
buffers to the retained in-process backend. The Console's non-streaming
dictation regime calls this buffer API for whole speech segments. The recorder
has a wall-clock bound and a PCM byte bound, but batch and dictation do not yet
share admission to the executor.

`BufferAudioSource` already defines the provider-neutral, bounded interleaved
PCM contract. TASK-603 adopts that contract at the executor seam instead of
inventing another buffer representation.

## Design

### 1. Keep process control and scheduling separate

`LocalSTTExecutor` remains the process controller and remains queue-less. A
small app-owned `LocalSTTDispatchCoordinator` sits immediately in front of it.
The coordinator owns only admission state:

- the active executor attempt, if any;
- at most one pending dictation buffer;
- whether that pending dictation has temporarily gated the next heavy Library
  dispatch; and
- the callback/future needed to return the result to the dictation caller.

It is not a general job queue. Library jobs remain persisted and ordered by the
existing Library ingest registry. Microphone frames remain owned by the
dictation session. The coordinator never creates a process and never owns a
model.

Library and Console obtain the same coordinator from `TldwCli`, which lazily
constructs it around the same lazily constructed `LocalSTTExecutor`. Individual
`TranscriptionService` instances receive an explicit buffer-dispatch callable;
they cannot create a private executor or reach the app through a module global.

Parakeet model identity and artifact admission are not reimplemented in the
coordinator. The existing Library-only identity construction is split so one
provider-neutral Parakeet dispatch builder resolves the selected model,
precision, configured/local/managed artifact source, snapshot, closure
fingerprint, and lease reference. Library file requests and dictation buffer
requests both call that builder. `transcribe-cpp` keeps its existing path.

### 2. Use the existing file-or-buffer source contract

`ExecutorRequest` changes from a mandatory `source_path` to one mandatory
`source: FileAudioSource | BufferAudioSource`.

`job_id` becomes optional. Library requests continue supplying their persisted
job ID; dictation requests supply `None` and use their capture generation plus
attempt ID for ownership. A synthetic Library job ID must not appear in
dictation provenance.

- Library callers wrap their current path in `FileAudioSource`; their parse and
  persistence path remains unchanged.
- Dictation callers supply `BufferAudioSource` with bytes, sample rate,
  channels, and sample width.
- The parent validates the buffer and its size before IPC. The worker validates
  the typed request again before native inference.
- Only Parakeet ONNX accepts executor buffer requests in this task. Unsupported
  provider/source combinations fail with the existing typed
  `UNSUPPORTED_CAPABILITY` contract.

An executor buffer request may also carry an optional, validated tuple of
strictly increasing segment-end frame offsets. It is valid only for a
`BufferAudioSource`; its final offset must equal the buffer's total frame
count. No offsets means the whole buffer is one segment. This preserves the
Console's silence-delimited logical segments while still sending one bounded
executor request.

The worker branches only at the source boundary:

- file source: run the existing `parse_local_file_for_ingest` flow;
- buffer source: call the already-resident Parakeet runtime directly and return
  the normalized transcription payload without invoking the Library parser.

`ParakeetOnnxRuntime.transcribe_buffer()` converts validated interleaved PCM to
mono float32 in memory, slices it at validated logical segment boundaries,
recognizes those segments in order with the resident model, computes duration
from frame count, and returns the same normalized result/provenance shape as
file transcription. A one-shot public buffer call is one segment. A coalesced
Console request returns ordered segment text so whole-segment voice-command
classification remains correct. No temporary file or second model load is
permitted.

For buffers longer than 30 seconds, the runtime uses the resident managed VAD
in memory when the artifact closure supplies it. A configured or verified
legacy v2 bundle without VAD remains compatible for the Console's bounded
60-second input and uses direct recognition, matching the existing public
buffer behavior rather than turning an optional managed dependency into a new
dictation requirement. Produced capabilities report whether VAD actually ran.

The executor worker converts that normalized result into the existing bounded
executor payload. The compatibility facade maps it to the historical public
dictionary keys while preserving normalized provenance.

### 3. One bounded dictation mailbox

The coordinator exposes two views of the same admission path:

- the public one-shot `transcribe_buffer()` compatibility call submits one
  buffer and blocks its non-UI caller for the terminal result; and
- live Console dictation submits through an asynchronous completion handle, so
  its audio processing thread never sits blocked behind a long active batch.

The second distinction is load-bearing. The existing live service gives its
processing thread only 30 seconds to join on stop; blocking that thread behind
a longer batch would trigger its documented incomplete-transcription path and
drop audio. The processing thread instead drains captured audio into the
mailbox and exits promptly. The Console's existing off-event-loop stop worker
waits for the capture's ordered completion handles, with cancellation and
force-stop available through the coordinator. The Textual event loop never
blocks.

The mailbox rules are:

1. There may be one active native inference and at most one pending dictation
   inference.
2. If the executor is idle, a sealed dictation buffer dispatches immediately.
3. If a batch inference is active, the dictation buffer becomes the one pending
   item and gates dispatch of later heavy Library jobs.
4. Microphone frames that arrive while that item is waiting continue to
   coalesce in the dictation session's single pending PCM buffer; they do not
   become one queued inference per frame or segment. Silence finalization adds
   a frame offset, not another request. The buffer and its ordered boundaries
   are snapshotted only when handed to the executor.
5. Once a dictation request is active, later frames and finalized segments
   coalesce into the one next pending request. The coordinator never admits a
   second pending request. Segment offsets preserve the boundary between
   ordinary dictated text and whole-segment voice commands.
6. Coalescing is allowed only for the same capture generation, model identity,
   sample rate, channel count, and sample width. A different capture receives a
   visible busy failure rather than having two users' audio combined.
7. Pending PCM is memory-only and is released after success, failure,
   cancellation, or shutdown.

There is one canonical Console capture ceiling: 60 seconds. Its byte ceiling is
derived from sample rate, channel count, and sample width rather than restated
as an independent magic number in the recorder, Console controller, and
mailbox. The mailbox checks the duration and derived byte ceiling so it remains
bounded even when a caller bypasses the Console timer. Appending a frame-aligned
chunk either retains the accepted portion or returns a typed overrun signal; it
never drops audio without notifying the controller.

On overrun, the recorder is stopped, the retained PCM is sealed for
transcription, and the Console enters its existing transcribing state with the
approved limit message. After the transcript is inserted, the control returns
to idle. Only another Mic press starts a new capture.

### 4. Dictation gets the next slot, not the current slot

Starting dictation sets a coordinator reservation before capture can submit a
buffer. `_top_up_ingest_parse_pool()` continues dispatching light work, but it
skips new local-STT heavy jobs while a dictation reservation or pending buffer
exists.

If a heavy batch inference is already active:

1. it continues to completion, failure, or cancellation;
2. the Console reports that local transcription is busy;
3. its terminal callback asks the coordinator to dispatch pending dictation
   before calling the normal heavy-lane top-up; and
4. after dictation reaches a terminal result, the coordinator clears the gate
   and invokes the existing Library top-up.

This ordering is explicit and deterministic. It does not poll `executor.busy`,
retry on a timer, preempt a worker, or race two direct `submit()` calls.

Starting and cancelling capture before any PCM is pending clears the
reservation immediately so it cannot stall later batch work.

### 5. Preserve the public and Console compatibility surfaces

`TranscriptionService.transcribe_buffer()` keeps its public signature.

- `provider="parakeet-onnx"` uses the injected app-owned dispatcher and returns
  the compatibility dictionary built from the normalized executor result.
- Retained providers continue through `LegacyTranscriptionBridge` until
  TASK-605. Their code is left in place and marked as retained/dead-after-gate,
  not deleted here.
- A Parakeet ONNX call without an app-owned dispatcher fails clearly; it does
  not silently instantiate a model in the caller process.

The optional dispatcher is an explicit constructor dependency. Direct callers
that want Parakeet ONNX outside `TldwCli` may supply a dispatcher backed by
their own `LocalSTTExecutor`; production app code always supplies the one
app-owned instance. Absence is a typed unavailable error, never an implicit
global singleton or hidden model load.

`create_streaming_transcriber(provider="parakeet-onnx")` returns `None` through
the existing fallback contract. The existing `LazyLiveDictationService` then
uses bounded whole-segment buffer transcription, exactly as it does for every
other non-streaming provider. No Parakeet object advertises `process_audio()` or
another streaming API.

Console wiring injects the app-owned dispatch adapter into the existing service
factory. It does not replace the Mic button or fork another dictation UI.
Existing partial/final events, voice-command classification, hands-free state,
caret insertion, draft preservation, and no-auto-send behavior remain owned by
their current controllers.

The retained silence-transcription warm-up is skipped for this deferred shared-
executor path. Otherwise an active batch inference would block the Mic press
before the recorder opens, contradicting the approved behavior that capture may
continue while dictation waits for the next executor slot. Model loading occurs
inside the reserved executor attempt and the existing preparing/busy states
remain honest about that work.

On a retryable Parakeet failure, the Console retains the exact bounded PCM and
segment-boundary metadata long enough to offer one confirmation:
**Parakeet failed. Retry this audio with faster-whisper?** Acceptance replays
the preserved segments through the retained faster-whisper buffer path and
then follows the normal insertion/voice-command flow. Declining, cancelling,
closing the Console, completing the retry, or app shutdown clears the retained
audio. The retry never starts automatically and is not offered when
faster-whisper is unavailable. Native exception text is not shown.

### 6. Cancellation, shutdown, and stale results

- Cancelling a pending dictation removes it, clears the Library gate, wakes its
  waiting caller with `CANCELLED`, and resumes heavy dispatch.
- Cancelling an active dictation uses the executor's exact attempt-id
  cancellation path. Force-stop remains the existing explicit escalation.
- App shutdown first rejects new coordinator submissions, then resolves any
  pending dictation as cancelled, then closes the shared executor through the
  existing shutdown path.
- Dictation attempts carry a capture generation as well as the executor
  generation/attempt ID. A result from a discarded capture cannot insert text
  into a later capture or keep the batch gate set.
- Live completion handles carry monotonically increasing segment sequence
  numbers. Results are delivered in capture order even if cancellation and
  terminal callbacks arrive on different threads.
- Retryable PCM retained for the faster-whisper offer is bounded by the same
  capture limit and is cleared on every resolution and teardown path.
- Every terminal path clears the pending buffer and reservation exactly once.

## Error and status behavior

The UI receives bounded categories, not native exception text:

- active batch: **Local transcription busy — dictation will run next.**
- capture bound: **Limit reached — press Mic to continue.**
- unsupported streaming: internal `None` fallback, no false streaming claim;
- missing/corrupt model, unavailable runtime, inference failure, or engine
  crash: existing normalized STT failure copy followed by the explicit
  **Retry this audio with faster-whisper?** confirmation when available;
- cancelled capture: existing cancellation behavior, with no draft mutation.

Failures leave the draft unchanged unless a prior, independently successful
segment was already inserted. No error path starts a download.

The explicit-resume rule also wins over the hands-free loop's historical
limit-triggered reopen behavior: reaching the canonical capture ceiling exits
that loop after preserving the captured transcript. It does not send or reopen
the microphone automatically; another physical Mic press is required.

## Focused verification

Only tests covering modified functionality are in scope.

### Contract and worker tests

- executor accepts file and bounded buffer sources and rejects invalid unions;
- dictation requests preserve `job_id=None` in normalized provenance;
- Library file requests retain their existing parse call shape;
- Parakeet buffer requests stay in memory, reuse the resident model, and return
  normalized text, duration, provenance, and warnings;
- ordered segment boundaries are validated and returned in order;
- unsupported provider/buffer combinations fail with a typed code;
- no facade or dictation session creates another executor/process.

### Coordinator tests

- idle dictation dispatches immediately;
- an active batch is not preempted;
- pending dictation is selected before the next heavy batch item;
- light Library work remains dispatchable;
- same-capture audio coalesces within byte/duration limits;
- silence-delimited segments coalesce into one request without losing their
  boundaries;
- a second pending inference is never admitted;
- overrun stops capture visibly and retains the accepted PCM;
- cancelling before dispatch, cancelling active inference, worker failure,
  force-stop, and shutdown each clear the gate once;
- a stale callback cannot complete a newer capture.

Latency fakes must include a real controllable delay; zero-latency fakes cannot
prove the backpressure path. One focused test sets a short join bound and holds
an active batch behind a real event-controlled delay longer than that bound,
proving the same failure shape without a 30-second test sleep: stop neither
blocks the UI thread nor drops the pending audio.

### Console and compatibility tests

- the mounted Console shows busy and limit states and returns to idle;
- a successful transcript still inserts at the caret without sending;
- pressing Mic again is required after a limit stop;
- a coalesced whole-segment voice command is still classified as a command;
- Parakeet streaming factory returns `None` and the buffer fallback is used;
- public Parakeet buffer calls use the injected shared coordinator;
- retryable Parakeet failure offers an explicit faster-whisper replay and
  clears retained PCM after every answer/teardown path;
- retained-provider calls remain unchanged pending TASK-605.

Runtime-focused tests cover 30–60-second buffers with and without a managed VAD
and assert that no disk staging occurs and produced capabilities report the
actual VAD behavior.

A macOS live smoke should use an isolated profile, the repository virtual
environment's absolute Python path, a real installed Parakeet v2 INT8 bundle,
and the real Console Mic affordance. Windows and Linux evidence remains an
explicit release gate because those platforms are unavailable for immediate
testing; their absence does not authorize legacy removal or default promotion.

## Rollout and completion boundary

This implementation may merge with focused tests and macOS evidence while the
TASK-603/TASK-605 cross-platform release gate remains visibly open. It must not:

- mark Windows/Linux parity as verified without evidence;
- promote Parakeet ONNX as the global default;
- delete the retained Parakeet/MLX paths; or
- weaken the TASK-605 removal gate.

## Alternatives rejected

### Put a priority queue inside `LocalSTTExecutor`

Rejected because it couples UI admission policy to process containment and
turns a currently simple single-slot executor into a general scheduler.

### Stage PCM as a temporary WAV

Rejected because the existing buffer contract is explicitly memory-only for
dictation and the resident Parakeet model already accepts NumPy audio.

### Build a second dictation-specific executor

Rejected because it can load a second multi-gigabyte model, violates ADR-025,
and recreates the lifecycle problem the shared executor was built to solve.

### Replace the current Console dictation UI with the old one-shot session

Rejected because the current Console already owns microphone state, voice
commands, hands-free behavior, generation fencing, and insertion semantics.
The task should replace the execution seam, not regress those user-visible
features.

## ADR check

**ADR required:** no new ADR

**ADR path:** `backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md`

**Reason:** ADR-025 already decides the shared executor boundary, bounded
in-memory dictation, no true streaming, dictation-next priority, non-preemption,
and release gates. This design implements that existing decision without
changing its architecture.
