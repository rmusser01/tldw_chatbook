# Speculative voice: GIL-independent device callback

Status: Approved by the user on 2026-09-04 after the five review corrections.

Task: [TASK-23175](../../../backlog/tasks/task-23175%20-%20Add-low-latency-speculative-duplex-voice-pipeline.md)

ADR required: yes

ADR path: `backlog/decisions/098-low-latency-speculative-duplex-voice-pipeline.md`

Reason: Refines the native/Python audio contract, callback memory ownership,
render cancellation, and device-clock evidence. Amend ADR-098 after written
approval; its acoustic, turn, provider-authority, and persistence rules remain.
The accepted amendment and implementation plan are linked from TASK-23175.

## Evidence and objective

On source commit `5c22aef5b94319e5877ddce33f70d40fcb44aa69`, a hardware-free
native driver calling the installed sounddevice CFFI boundary measured a
183640 us delay before Python callback entry during one forced full collection
of the mounted Console heap. The callback body stayed below 594 us. Without
forced collection, the mounted baseline stayed below 2248 us total. Both used
250 synthetic silent frames and the existing native AEC, without opening audio.
The detailed diagnostic is `/private/tmp/tldw-voice-callback-boundary-findings.md`.

This proves a same-process GIL dependency, not that garbage collection caused
the earlier USB/CoreAudio deadline overruns. Native AEC already releases the
GIL around processing; the device callback still needs it before entering
Python. A separate voice event-loop thread does not protect this boundary.

Remove Python execution and Python-owned locks from the device callback.
Preserve data and truthful timing through a bounded Python pause. This is not
a promise of hard real-time OS scheduling or interruption detection while the
entire Python interpreter is unable to run.

## Choice and scope

Add the callback and its fixed-capacity buffers to the existing
`native/voice_aec` component. Keep PortAudio device opening/closing in the
sounddevice adapter and capture consumption/AEC/VAD on the existing voice loop.
The native callback does no DSP, STT, provider work, logging, or Python calls.
No new service, process, vendored dependency, or general audio framework.

Alternatives rejected for this correction:

- Another Python thread leaves the measured GIL dependency intact.
- Moving the entire voice engine into another process would introduce provider,
  promotion and TTS ownership protocols outside this boundary's needs.
- Disabling GC or increasing buffers does not remove Python callback entry and
  does not bound arbitrary GIL-holding work. Neither is part of this change.

Use sounddevice's documented library-author `_StreamBase` extension seam with
`kind="duplex"`, `wrap_callback=None`, a native function pointer and userdata.
`RawStream` itself always installs a Python wrapper and must not be used for
this path. Isolate that dependency seam in one adapter and validate it before
opening a stream. Keep the loaded extension and native owner strongly held.

Retain 48 kHz mono int16, 480 samples per 10 ms callback, current actual-latency
validation, current 64-frame capture/render defaults, and current AEC/timing
thresholds. Do not run qualification, route switching, repetitions, matrices,
soaks, or a full test suite. Do not modify unrelated trace-ledger/CSS work.

## Native ownership and buffers

Each device generation owns one native state allocated before stream start.
It has one capture-record ring and one render ring, with capacities supplied
from the existing transport limits. Allocation is bounded and capacities are
validated before opening. All callback-touched storage is initialized before
start. No allocation, free, Python/CFFI trampoline, mutex wait, blocking system
call, or unbounded loop is permitted in the callback.

Use fixed slots with single-producer/single-consumer index ownership and
release/acquire publication. The callback produces capture and consumes render;
Python consumes capture and produces render. Existing non-real-time transport
locking serializes Python callers. Required native atomics must be lock-free
on the target architecture; reject unsupported construction rather than use a
hidden locking fallback. Never have cancellation rewrite a consumer's index.

A capture record contains input PCM, device ADC/current/DAC timestamps, a native
monotonic observation, callback ordinal, generation, native status and fault
bits, occupancy evidence, and the exact PCM/render identity sent to the device
in that callback (or an explicit silence/no-render identity). Pairing capture
and actual output avoids a separately overflowing native reference queue.

No second queue silently doubles the capture or render allowance. Python pulls
native records on demand through `pop_capture`, constructs the existing
`AudioFrame`/`AecDelayEvidence` contracts, and exposes their paired render
references to the existing preprocessor. Bound the Python reference backlog by
the existing render capacity and retain the bounded capture history for drains.
Occupancy and overflow reporting cover every remaining storage layer.

Full capture queues never overwrite unread data: latch data loss out of band,
fence native output, and expose the failure even if the faulting record was
dropped. Full render queues reject/backpressure through the existing sequencer
contract. An empty render queue emits silence, not a fabricated PortAudio
underflow. Malformed frames, invalid times, missing buffers and device statuses
remain distinct failures. Native callback failures cannot escape as exceptions.

## Clock, startup and acoustic evidence

Capture the monotonic observation inside the native callback, not at Python
drain time. Map its monotonic domain to the Python transport clock once before
start, with bounded, validated calibration; never re-anchor from delayed
consumer arrival. Device timestamps and callback observations still feed the
existing cadence/drift rules. A delayed Python drain is not device clock drift.

Keep the existing first-50-callback/500 ms startup exception. A status-discarded
startup callback emits silence, consumes no queued speech, resets cadence
comparison and advances no admitted capture sequence. A Python pause cannot
extend this allowance: it is measured by native callback ordinal. Status on
callback 51 or later remains fatal. Overflow is never excused as startup priming.

Keep tolerated startup-status diagnostics separate from the fatal native-status
latch. A status discarded within the startup allowance does not poison the
generation or disqualify its existing one-time idle timestamp recovery. Native
statuses outside that allowance, data loss, malformed frames and processing
failures latch independently of the later drain error string. Later clean-status
callbacks and failed acknowledgements cannot erase or relabel those causes. The
single existing idle timestamp recovery requires all fatal latches to be clear;
all its other idle/unused-turn restrictions remain. Reset fatal latches only
with a new, fully owned device generation. No automatic recovery loop.

AEC/VAD and acoustic policy remain on the voice loop with unchanged thresholds.
Their inputs must describe capture-time playback, not processing-time assistant
state. Retain actual-output identity, output epoch and DAC intervals with native
records. Derive each capture's `assistant_rendering` from that historical evidence,
including output already committed before cancellation. Do not AND this with
the mutable attempt/lifecycle `rendering` flag. Cancellation cannot retroactively
turn buffered playback capture into idle capture. Missing or ambiguous playback
context takes the closed playback-admission path, never the idle raw-PCM fallback.

Only actual output PCM is an AEC reference. Discontinuous, dropped or
unacknowledged capture cannot prove healthy full duplex or a completed turn.
A Python pause within the capture bound preserves ordered frames; a longer pause
fails closed. Speech detection and its cancellation request may wait for Python
to resume; this change removes device callback dependence, not all interpreter
scheduling delay.

## Cancellation and playback completion

Separate submission identity from the AEC reference ordinal. Every queued render
slot carries device generation, output epoch and a monotonically increasing
submission ID, never reused within that generation. Render-completion receipts
identify that submission, not an `AudioFrame.sequence` assigned at enqueue time.

The callback assigns a separate contiguous reference ordinal, starting at zero
per device generation, only when current queued PCM is actually committed to
output. Cancelled/unrendered submissions consume no reference ordinal. Explicit
idle or fence-substituted silence keeps its no-render marker and is not a
successful submission. Render-reference `AudioFrame.sequence` and capture's
`render_reference_sequence` use reference ordinals, preserving the existing AEC
gap checks; capture frames retain their own contiguous capture sequence.
Example: submission 0 renders, queued 1 and 2 are cancelled, then
submission 3 renders; receipts identify 0 and 3, but AEC sees references 0 and 1.

`abort_output` atomically advances the native output epoch; admission fencing
additionally closes native output. Both reach the native owner synchronously
before asynchronous provider/TTS cleanup.

The transport's synchronous `fence_output` seam reaches that epoch from attempt
invalidation, tool abort and direct sequencer cancellation before those callers
schedule cleanup. Ordinary attempt cancellation keeps capture and replacement
render admission open; permanent session shutdown uses `request_close` instead.
The same synchronous fence rejects ready old TTS continuations before they can
publish under the new epoch. Async cancellation still observes TTS resource
cleanup, without a later duplicate output abort that could fence replacement PCM.

The callback discards obsolete epochs with at most one ring-capacity scan and
checks the fence at output commit. It never waits for Python. A callback that
observes the new fence cannot commit old audio. One callback already committed
before the fence, plus samples already in hardware buffers, cannot be recalled;
do not describe those as a zero-latency stop guarantee. Reference records must
describe the bytes actually committed, including silence substituted by a fence.

Retain sequential TTS ownership and cleanup receipts. A queued frame's estimated
`ended_ns` is not evidence of playback completion. Once phrase input is closed and
the last submission is known, arm one terminal target on the voice owner. Its
native completion receipt supplies the actual callback DAC end time. Retain at
most one terminal target/receipt/waiter for the active sequencer, not per-frame
futures or an unbounded receipt history.

Boundary discovery and "playback finished" are distinct. Serialize terminal-target
arming, native completion observation and capture admission through the voice
owner. Before admitting a capture whose start is after the target's actual DAC
end, install the immutable generation/epoch/boundary marker in the coordinator.
This ordering cannot depend on an independently scheduled terminal waiter winning
an event-loop race. Once a pending target has rendered, resolve its marker before
releasing later capture; use existing bounded capture storage while resolving it,
and fail closed on loss or ambiguity. Buffer the single marker if generation
completion notification is still pending rather than dropping it.

The marker routes post-boundary speech into the next turn, but does not itself
announce completed playback or authorize promotion. Emit playback-finished only
after the mapped DAC end has elapsed, then require the existing capture/DSP/VAD
and transcript seals before promotion. Buffered speech at or before the boundary
still interrupts/extends the same turn, even if processed later. Cancellation or
route reset invalidates the pending marker and waiter; a late receipt cannot
complete a cancelled submission or apply to a replacement attempt.

Session classification sealing requires native capture/DSP coverage through the
actual boundary in half duplex too. Acoustic gating still rejects playback-period
speech; queued historical idle/gap speech must reach admission before STT can seal.

Turn routing uses actual detected speech onset. Optional `speech_started_ns` on
`AudioFrame` and `AdmittedSpeechFrame` gives silent VAD preroll the onset of its
triggering positive frame; `None` retains the positive frame's own start. Original
PCM, timestamps and playback context remain intact for STT and acoustic seals.
Silent context at/before the DAC boundary must not turn later speech into an
interruption. The classification barrier includes capture currently being
preprocessed/admitted and its earliest preroll timestamp: a DSP acknowledgement
alone cannot let a newly discovered final target seal before speech admission.

Delivery waiting has an absolute deadline, fixed when the target is armed: the
remaining startup allowance plus the configured maximum render-queue duration,
validated output latency, and the existing 500 ms terminal-seal allowance. Polling
or partial progress cannot extend it. After the DAC boundary is known, retain
the existing seal deadline of that boundary plus 500 ms. Delivery timeout, data
loss or uncertain completion fails/cancels the target rather than leaving the
coordinator indefinitely speaking or declaring a successful turn.

## Lifecycle and compatibility

Deactivate native capture admission and fence output before detaching a route or
closing a session. Subsequent callbacks ignore input and emit silence. Retain the
native state, callback pointer and stream until a checked native close result
proves callbacks cannot continue. Sounddevice defaults `stop`/`close` to ignoring
errors and clears its stream pointer even on a close error: require checked
results (`ignore_errors=False` at that seam), never `stream.closed` or a null
wrapper pointer as closure evidence. Retain the original native owner/handle
metadata independently; a failed close is not permission to retry a possibly
invalid handle. Stop failure may proceed to one checked close only after stop
has returned. Never issue close concurrently with a still-running stop.

Use one retained off-loop native-operation owner per live/closing stream. A
teardown observer has a total two-second budget from the initial fence; caller
cancellation or expiry does not cancel or free the native-operation owner. On
expiry or unresolved close error, finish the observer with a categorical failure,
retain the deactivated state in an app-root quarantine and block all subsequent
speculative audio startup, including through a new view/session. At most one
uncertain stream is retained. Expose "audio shutdown unconfirmed", not a false
microphone-closed claim. Only checked close success can release quarantined
callback ownership; late success cannot automatically reopen voice.

Session shutdown invokes native deactivation and checked-close dispatch from the
synchronous facade/core fence. Its deadline observer runs independently of
transcription/provider cleanup, survives caller cancellation, and preserves
categorical uncertainty even when another cleanup blocks or fails. A permanently
closed transport cannot activate capture through delayed startup.

The operation owner must survive voice-loop closure. A never-returning native
call must not become an unbounded join in session/UI cleanup or asyncio default
executor shutdown; use a dedicated daemon operation thread with process-lifetime
native retention for uncertain callback state. Late completion does not call a
closed event loop. Do not force-kill threads, free uncertain native storage at
interpreter finalization, or call PortAudio termination concurrently as a cleanup
shortcut. Bounded observation is not proof that a hung driver call finished;
process termination remains distinct from successful device cleanup.

Bump the native package capability/version together with the app version and
exact companion pin (0.1.9.0 for ABI 1), and validate its interface before
device open. A stale/missing native component makes speculative device startup
unavailable with a content-free actionable error; it must not silently select
the GIL-dependent callback. Do not change the packaged qualification manifest
or introduce any release bypass. Update source identity coverage for the new
native/adapter sources, tests and design. Old qualification evidence stays old.
The supported exact-HEAD source-only launcher remains the only development
selection mechanism and is still excluded from release artifacts.

Ordinary timeout quarantine remains releasable by late checked close while the
app lives. Irreversible native retention applies only when uncertain ownership
crosses app/interpreter shutdown. Categorical startup and shutdown failures are
notified on the UI loop; an obsolete generation cannot repaint or notify over a
replacement view. Neither failure notification contains native exception text.

## Verification and handoff

Use test-first native and Python checks with no physical devices:

1. A native driver invokes the real callback while Python deliberately holds
   the GIL. Prove multiple callbacks complete during that hold, input/output
   bytes and timing records remain ordered, and no Python callback executes.
   Keep this deterministic progress check separate from wall-time diagnostics.
2. Exercise queue boundaries, epoch fencing at output commit, stale generations,
   reference identity, finite timestamps, startup callbacks 50/51, sticky fatal
   status after dropped records/acknowledgement errors, and capture overflow
   fail-closed. Tolerated priming followed by a genuine idle timestamp fault must
   retain the one allowed recovery; a post-startup native status must prevent it.
   Render submission 0, cancel queued 1/2, render 3 through the real preprocessor:
   AEC references remain 0/1, cancelled receipts fail and no false gap is reported.
3. Exercise cancellation during open/start/stop/close, failed teardown retention,
   late callbacks, and replacement-start refusal across new sessions. Cover a
   close error whose wrapper pointer is already null and a never-returning stop
   or close: the observer fails within two seconds, no concurrent native close or
   termination occurs, late callbacks see retained deactivated state, and only
   checked close success releases ownership. Validate installed callback ABI/
   pointer ownership without opening PortAudio hardware.
4. Prove terminal render receipts use actual DAC timestamps, not enqueue estimates;
   queued-but-unrendered, aborted and reset frames cannot finish a turn. Force
   both scheduling orders between the terminal waiter and a post-boundary capture:
   each produces the same new-turn result. A buffered pre-boundary frame delivered
   after DAC completion must still extend the old turn. Cover generation-complete
   notification arriving after the boundary marker, cancellation before marker
   delivery, late obsolete receipts and callback cessation before final render.
   Advance a fake clock to verify delivery and seal deadlines without wall sleeps.
5. Mount the production Console with isolated config and native simulated audio.
   Activate the visible control and assert `ConsoleSpeculativeHandsFreeSession`.
   During one controlled GC pause, retain capture/reference continuity within the
   existing capacity and show native callback progress without entering Python.
   Capture playback frames, cancel the attempt, then drain that backlog: recorded
   playback context must survive the now-false lifecycle flag. With unhealthy
   AEC, none of those frames may reach the idle/raw admission path or STT. A new
   proven-idle frame remains eligible under the existing half-duplex idle policy.
6. Run only affected native/transport/sequencer/lifecycle/source/launcher tests,
   lint and diff checks. Build the changed native artifact with its existing
   provenance checks. Do not claim live success from these software gates.

Before the one ordinary conversation, bind the launcher to the final committed
source, verify the loaded native artifact/version, exact feature checkout and
commit, visible-control selection, current USB route and dependency/settings
metadata without credentials. Only then open the app and request normal speech.
On failure, inspect runtime evidence before asking the user to repeat anything.
