# Speculative voice: post-smoke timing and STT ownership repairs

Status: Independently reviewed without blockers; written specification approved by the user on 2026-09-04.
Date: 2026-09-04
Base: `7c4cc693f36db3d1ddebecb70d9b1435a502089a`
Task: [TASK-23175](../../../backlog/tasks/task-23175%20-%20Add-low-latency-speculative-duplex-voice-pipeline.md)

ADR required: yes — clarify existing shared DSP evidence and STT lifetime contracts.
ADR path: `backlog/decisions/098-low-latency-speculative-duplex-voice-pipeline.md`
Reason: The approved direction repairs evidence identity and shared-model ownership;
it does not replace the runtime or relax acoustic admission.

## Scope and evidence

One ordinary Logi USB conversation produced a relevant response and user-observed
interruption/revision. Revised playback stopped after a PortAudio input overflow.
The user offered further live feedback if useful; none is currently needed.
No repeat conversation, device open, route change, qualification harness, hardware
matrix, soak, full suite, provider call or model download belongs to this repair.

Three software reproductions at the base commit establish:

1. Four native output submissions followed by two silent callbacks retain complete
   PCM and timing with zero native loss. During the last audible DAC tail, capture
   frames repeat the last submission reference, and the isolation monitor reports
   `render-reference-gap` then `missing-render-reference`.
2. Exactly 400 admitted ten-millisecond frames fill one native STT chunk. The next
   unbounded queue read retains its context beyond the configured quiet interval.
3. Two production native adapters on one serial worker and one spawned Parakeet
   process can collide on context ownership. The second native open and rolling
   fallback both fail with the app-owned `stream_active` reason.

These are reproduced defects, not proof that all live RuntimeErrors had this cause.
The live initial generation phase lasted approximately 12 seconds; playback began
one second later. First-token and phrase timing are missing. The host-overflow
cause remains unknown; the scoped CoreAudio log query supplied no additional event.
Do not attribute it to desktop switching, App Nap, Python GC or application ring
overflow without evidence.

## 1. Complete DSP output timeline, unchanged submission receipts

Keep three identities distinct:

| Identity | Meaning and consumers |
| --- | --- |
| Capture/DSP tick | Every accepted callback record, including actual silent output; orders AEC and isolation evidence |
| Native committed-render ordinal | Existing native metadata for successfully committed queued output; not a complete DSP clock |
| TTS submission tuple | Generation, output epoch and submission ID; owns actual DAC completion and cancellation |

The native ABI already returns a paired capture record with actual `output_pcm16`,
ADC/DAC times, accepted capture sequence and validity/fault metadata even when no
submission played. Reuse those bytes. No new callback allocation, native PCM ring,
backend, process, dependency, or native ABI/version bump is planned.

For every valid accepted callback, transport produces one DSP render-reference
frame from the actual output bytes and DAC interval. Its sequence uses the paired
capture/DSP tick, and the capture's `render_reference_sequence` identifies that
same scheduled-output frame. This includes callback-written zeros, including
silence after a cancelled submission. It never fills a missing record with invented
zeros. Generation resets and tolerated startup discards preserve the existing
continuity/fault policy; actual dropped records remain gaps and fail closed.

Only actual submission commits update `RenderBoundary` and the bounded audible
playback interval history. A silent callback with no committed submission cannot
complete TTS, extend an answer's terminal boundary, or count as audible playback.
A submitted block whose actual samples happen to be zero still retains its real
submission receipt; the monitor's existing energy checks decide audibility.

Keep `assistant_rendering` based on historical actual playback intervals at ADC
capture time. It must not become false just because this callback schedules silence.
In the six-record reproduction, the last two captures remain playback-period
captures although their newly scheduled output is silent.

AEC receives each output block exactly once in DSP-tick order before the paired
capture is processed. The isolation monitor retains the bounded actual-output
history across idle callbacks as well, without accumulating warm-up proof from
silence. Preserve complete timing, sequence, energy, saturation, correlation and
leakage checks. Complete silence is valid timing evidence, not proof of acoustic
isolation and not permission to bypass residual-echo classification. A complete
but acoustically ambiguous or inaudible window can still fail closed; the repair
removes false *missing-data* failures, not real acoustic failures.

No duplicate-reference exception, synthetic sequence renumbering after loss,
raw playback-period STT admission, threshold change or larger history budget.
Use the existing capacity bounds and drain acknowledgements. The injected backend
must project the same evidence semantics; it cannot conceal a production-only
difference. Update reference consumers and tests that previously conflated native
committed-render ordinals with DSP ticks. Do not alter cancellation epochs or
submission identities to satisfy those tests.

## 2. One owner of shared Parakeet model state

Retain the existing session-owned process, one serial executor and bounded RPC.
Serializing individual RPCs is insufficient because a native context mutates shared
model state from successful entry through actual context exit.

Use one session/owner-loop arbitration point shared by Parakeet native adapters,
native prewarm and rolling fallback. Prefer an asyncio lock/context manager on the
existing session-owned STT worker; do not introduce an actor framework or a second
model. Acquire ownership only after there is audio/work, not while an idle adapter
waits for its first frame. Native ownership covers context creation/entry, all
pushes for that burst and checked exit. Rolling ownership covers its complete
model request, after all prior context state has been restored. Other recognizers
keep their existing execution behavior.

All waits between native chunks use the configured response-eagerness quiet
interval, including the read after an exactly full 400-frame chunk. Quiet closes
the burst context while retaining the turn's accepted transcript prefix; later
speech creates a fresh context and extends that transcript. Preserve final/partial
coverage semantics, original PCM order and existing batching limits. Quiet expiry
does not submit fabricated audio or discard a frame arriving at the boundary.

Ownership acquisition is cancellable and bounded to 30 seconds, matching the
existing ordinary Parakeet RPC timeout scale; no new user setting. This wait does
not imply an entire burst is limited to 30 seconds. A waiting turn must neither
steal nor close another turn's context. Waiting expiry reports a fixed category
and fails safely without invalidating the active owner's identity.

Cancelling an asyncio wrapper is not evidence that a blocking RPC finished.
Retain and observe in-flight operations; release ownership only after checked
context exit or confirmed process termination. Cancellation while entering/pushing/
exiting, failure during entry/exit and cancellation of a waiter must not leak a
context, release the next owner prematurely, or strand a lock. An uncertain native
context poisons the shared service until its existing bounded shutdown reaps the
process. Never advertise rolling fallback on a closed/uncertain service.

Session teardown still synchronously deactivates audio and starts checked native
close before model cleanup. The STT ownership guard cannot delay that fence.
Queued recognizer work observes shutdown and fails rather than restarting a model.
Retained transcript objects may coexist, but active model contexts may not.
Do not widen transcript/audio queue limits or add unbounded ownership queues.

## 3. Content-free diagnostics at existing boundaries

Use the existing persistent-metadata validator and bounded voice-to-UI diagnostic
bridge; no new log file, trace framework, callback logging or per-frame events.

At first transport fault in a generation, record a bounded owner-side snapshot:
native callback count, sticky status bits, capture loss/invalid counters, current
native capture/render occupancy, and available callback-versus-consumer timing.
Distinguish host status from application ring loss. Describe a drained-time snapshot
as such; do not label it a fault-instant capture or infer callback gaps solely from
consumer arrival time. No native callback hot-path change is required for this
first diagnostic increment.

Add narrow integer metadata fields only where existing fields do not express these
counts truthfully. Keep fixed event/category names, range validation and privacy
tests. Do not serialize arbitrary mappings, object repr, PCM, transcript text,
exception messages, model responses, credentials or new user-content identifiers.

Add one monotonic first-occurrence timing event per attempt for first non-empty
provider delta, first eligible phrase, first synthesis completion and first actual
render receipt. Retain existing generation start/end events. Distinguish queued
audio from hardware receipt; stale attempts cannot publish success after fencing.
Durations are relative to the relevant attempt/phase, not comparisons of wall clocks
across threads. Missing stages remain unavailable rather than fabricated as zero.
These timings diagnose latency; they do not change the 700 ms silence threshold.

Expose a small enum/allowlist of app-owned STT failure reasons (context busy,
identity mismatch, closed/disconnected, timeout, ownership timeout, unknown native
failure). Map at the owning boundary, preserving only fixed categories. Generic
provider exceptions remain class-only/unknown, never arbitrary message matching
that could copy content into a log. Bound duplicate events during fallback failure.

Host input overflow retains the current fail-closed output behavior. Do not add
automatic reopen, status forgiveness, App Nap overrides or device changes as part
of this work. Further live feedback would require a specific question the new
diagnostics can answer, presented to the user separately.

## Alternatives and trade-offs

- **Selected: repair existing evidence and ownership contracts.** Reuses native
  paired PCM and one model; requires careful migration of reference consumers and
  lifetime tests, but preserves the selected runtime and acoustic safeguards.
- **Diagnostics only.** Lowest mutation scope, but knowingly leaves deterministic
  phrase-tail and recognizer-lifetime defects; useful only if implementation is
  paused.
- **New audio backend/process or relaxed safety.** Not justified by current evidence.
  It increases scope or hides failures and does not repair the demonstrated
  contract mismatches. Not selected.

## Software acceptance before any runtime handoff

Use targeted tests, not a full sweep:

1. Real native callback and production transport/preprocessor with fake PortAudio:
   phrase onset, phrase tail, idle gap, cancellation silence, resumed phrase and
   true loss. DSP references include actual zeros; receipts include only actual
   commits; tail capture remains playback-period. Use real AEC and real isolation
   where applicable, not a monitor that always admits.
2. Quiet silence alone never establishes full duplex. Correlated render-only
   capture, missing timing/reference, stale generation, saturation, native status
   and overflow remain closed. Mixed near-end/correlated-tail cases retain the
   existing 50 ms confirmation and strict evidence requirements.
3. Exactly-full native chunk followed by quiet, resumed same-turn speech, two
   simultaneous transcript owners and native-to-rolling transition. Fake model
   with real spawned RPC asserts at most one context and restored model state.
4. Cancellation while waiting, during entry/push/exit, delayed cleanup, failed
   exit, dead process and session shutdown. Actual closure precedes next admission;
   capture shutdown is immediate and no obsolete result affects replacement work.
5. Existing native submission/DAC terminal, same-turn revision and post-playback
   next-turn regressions, plus mounted visible-control proof.
6. Privacy assertions inspect final persistent records with hostile input; unknown
   exceptions cannot retain content. Fake-clock latency tests distinguish provider
   delay, phrase buffering, synthesis and DAC scheduling without changing eagerness.
7. Scoped lint, version/ABI compatibility, source-identity inventory and diff checks.
   Keep historic evidence, packaged unqualified manifest and source-only launcher
   restrictions unchanged. Listed source edits invalidate the old source digest.

A software pass is not a claim that the host overflow or full ordinary conversation
is fixed. TASK-23175 stays In Progress. The user's one conversation is recorded as
positive interruption/revision evidence with incomplete playback/new-turn coverage.

## Expected ownership

Runtime changes are limited to audio transport/contracts, AEC/isolation evidence
consumers, Parakeet/session STT adapters and existing diagnostic projections.
Tests belong to their corresponding Audio/Chat/integration/mounted UI modules.
Only narrow persistent metadata schema additions and voice source/lint inventories
are in scope outside those paths. Preserve all unrelated trace-ledger/CSS changes.

Written-spec review and user review precede implementation planning. The plan will
link ADR-098's clarification, define exact files and RED/GREEN cases, and retain
the no-hardware restriction throughout implementation.
