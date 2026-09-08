# Speculative Voice Process Isolation

Date: 2026-09-05
Status: Approved by the user after independent round-2 review, 2026-09-05
Task: [TASK-23175](../../../backlog/tasks/task-23175%20-%20Add-low-latency-speculative-duplex-voice-pipeline.md)
Baseline: `fd89a7b0c341a26b4780d1d70cf72c28578fa2d9`

ADR required: yes — accepted amendment to the existing ADR.
ADR path: [ADR-098](../../../backlog/decisions/098-low-latency-speculative-duplex-voice-pipeline.md)
Reason: Audio execution, process lifetime and authority-bearing cross-module
contracts change. This supersedes the shared-interpreter limitation of its
UI-loop/native-callback amendments, not its acoustic or persistence policy.

## 1. Problem, evidence and scope

The latest ordinary USB conversation successfully interrupted and revised replies,
then failed on the follow-up after a completed reply. At that boundary the native
capture ring overflowed during a roughly 674 ms consumer gap. A subsequent
generation continued against a transport that permanently rejected output.

Software evidence separates mechanism from historical cause:

- A real native producer continued during a deliberate parent-interpreter GIL
  hold; its Python consumer could not drain the default 64-frame ring.
- A mounted production conversation with fake PortAudio/provider/TTS completed
  two distinct turns with 1,000 callbacks and no overflow in the positive control.
- Injecting a 720 ms GIL hold after follow-up admission reproduced overflow and
  the enabled-but-unusable session. The historical source of the live stall is
  still unknown; this is not evidence that GC or desktop switching caused it.

Incident records, local and non-release evidence:
`/private/tmp/tldw-voice-conversation-fd89a7b0c341-result.md` and
`/private/tmp/tldw-voice-native-conversation-fault-fd89a7b0c341.md`.

The user approved process isolation as the next direction. Implement one bounded
runtime-boundary change, including the unusable-session failure path. Do not
increase rings, disable GC, change acoustic thresholds, or retest physical
qualification. The ordinary conversation has been consumed; all verification
under this design is software-only. No full suite, hardware, provider requests,
model loading/inference, route changes, matrices, repetitions or soaks.

## 2. Decision and guarantees

Move the complete audio-critical loop into one view-session subprocess. Retain
one separate local STT inference process, and retain provider/persistence/TTS
service authority in the app. Reuse existing reducers, transcript algorithms,
native bridge, AEC, VAD and phrase sequencing; no general actor framework or
new third-party dependency.

For an otherwise healthy worker/device, a **720 ms or 3 s app-interpreter GIL
hold** must not stop capture draining, AEC/VAD, transcript orchestration, speech
admission, local attempt invalidation, or native output fencing. Captured speech
is not dropped merely because app-side UI/projection/authority work is waiting.

This is not hard realtime or isolation from OS suspension, CPU starvation,
inference-process stalls, or a GIL hold inside the audio process. App-owned
provider cancellation, new request preparation, synthesis/normalization and
persistence may wait for the app to resume. Local stale PCM must stop without their acknowledgement;
do not claim that the remote provider has stopped before its actual receipt.
Existing queued TTS may drain or underrun during an app stall. Underrun does not
become capture loss or permission to promote an incompletely played answer.

Unchanged behavior: configurable silence defaults to 700 ms; native streaming
STT is preferred with rolling fallback; speech during generation/playback extends
the same turn; speech after actual playback completion starts a new turn; TTS
is sequential and cancellable; AEC is enabled by default and unhealthy AEC with
a healthy transport safely degrades to half duplex. DeepSeek selection remains
app-owned and is not hard-coded into the worker.

## 3. Ownership and minimal extraction

| Owner | Resources and decisions |
| --- | --- |
| App runtime | Visible control and provider preflight; owning-view/session checks; original prepared requests and frozen contexts; gateway, `VoiceAttempt`, trace/usage records; eligibility classification; accepted-turn handoff and winning promotion; shared TTS service and PCM normalization; app-wide cleanup quarantine; child supervision |
| Audio subprocess | Native device stream and rings; AEC/VAD and historical playback classification; capture/DSP/admission/transcript seals; rolling transcript engines; coordinator/timers/turn IDs/attempt epochs; phrase sequencing, validated normalized-PCM consumption, render receipts and synchronous local output fences |
| Local STT subprocess | Model loading, native inference contexts, prewarm and in-memory rolling inference; no PortAudio, app controller, provider gateway or persistence |

The audio entry must not import `tldw_chatbook.app`, Textual, the `Chat` package's
eager service graph, provider adapters, database services, Torch or MLX. Extract
only the dependency-light audio core/contracts needed to satisfy this boundary;
parent composition/effects remain in `Chat`. Keep existing import facades where
needed by callers/tests. Do not duplicate the coordinator or rewrite its policy.

Known import traps are part of the extraction: package `__init__` currently
eagerly installs Textual compatibility, `Chat/__init__` imports conversation
services, and `TTS/__init__` imports its full service graph. Move compatibility
installation to the UI entry boundary, preserving
normal app and standalone-widget behavior with targeted import tests. Audio
diagnostics use an injected categorical sink, not app logging/configuration.
The child phrase sequencer consumes a lightweight normalized-PCM stream contract;
it must not import `TTSAudioResponse` or the eager TTS package. Parent adapters
retain those existing types and the existing decoder implementation.

`PreparedSpeculativeVoiceAttempt`, `ConsoleTurnExecutionContext`, gateway-issued
trace identities, promotion seeds and `VoiceAttempt` snapshots stay as their
original typed objects in the app. They are not pickled, reconstructed, or
replaced with fabricated contexts in the worker. Replace the coordinator's
typed preparation payload with a lightweight validated decision/handle seam;
run the existing strict eligibility classifier on the original objects in the
app. Existing in-process tests use a local adapter to that same seam.

## 4. Launch, STT and process ownership

Launch the audio child with `subprocess.Popen` using `sys.executable` and
`-m tldw_chatbook.Audio.voice_process_entry`, not fork or a multiprocessing target
that re-imports the app's `__main__`.
Use private inherited stdin/stdout pipes; no listening server, shell command,
temporary content files or user-callable qualification override. The child
duplicates its protocol output before importing libraries and redirects ordinary
stdout/stderr to the null device. Library output cannot become protocol or logs.

The app supervisor is lazy and app-owned; the actual audio process, microphone
and STT process remain view-session-owned. Reuse the current background owner
loop for parent provider/IPC work as needed; it is not the audio-safety loop.
One app-wide device lease prevents another view from opening a replacement
until the old ownership result is known.

Startup order is: existing selected-provider preflight; acquire lease and launch;
bounded protocol/source handshake; prepare STT; revalidate parent activation,
session and settings epochs; explicit start permission; child opens native audio;
child sends `session_ready`. Cancelled or stale preparation cannot open capture.
The child reports resolved module root, source identity, protocol version and
native ABI. Mismatch fails closed. The source-only launcher continues to require
its worktree and exact expected commit; this handshake grants no qualification.

Keep Parakeet's existing separate inference process, serialized requests, context
lease, startup/request deadlines and bounded terminate/reap behavior. It is now
owned directly by the audio child, never relayed through the app interpreter.
The audio child is not a multiprocessing daemon. Its guarded lightweight entry
can safely be re-imported when spawning the model process.

Other resolved local STT providers must not move inference into the audio
interpreter or silently lose native streaming. Extend the existing narrow STT
process service to the operations already consumed by the adapters: prepare,
open identified context, push, close context and transcribe bounded buffer.
Provider-specific native objects stay in the model process. Preserve native
capability/prewarm selection, rolling fallback and current local-provider
resolution/settings; no provider plugin framework or cloud-STT expansion.
STT result records preserve existing final/partial and cumulative/segment
semantics; transport chunking must not change transcript merging or coverage.
The audio child receives validated local STT settings, not the app configuration
or provider credentials. PCM remains private audio-to-STT IPC: at most one
request of 960,000 bytes and one bounded result of 65,536 characters in flight.

Parakeet ONNX uses an explicitly injected buffer-only current-process owner in
the existing transcription facade. The model process reuses the existing
resident source/lease/provider implementation, including source precedence,
`int8`/`f32`, root/VAD dependency validation and provenance; it does not create a
fourth process or bypass artifact custody through direct native loading. One
resident identity is retained, and a changed source/model/closure is rejected
without loading a replacement. The facade refuses simultaneous shared-dispatcher
and local-buffer-owner injection. Ordinary ingestion/dictation retain ADR-025's
shared executor. Failed/hung close retains the actual resident and protecting
leases until model exit; a cancelled observer is not cleanup evidence.
On uncertain model cleanup, attempt the categorical receipt and retire that model
with nonzero `os._exit` while the owning frame still retains the resident and
leases. Returning into Python finalization is not process-exit custody. Receipt
or connection-close failure must not skip retirement; the audio owner still
requires actual reap and retains `FORCE_CLOSED`. Normal clean retirement is
unchanged, and a hung cleanup/send remains subject to the existing parent kill
deadline.

Contain the entire audio/STT process tree before granting startup permission.
On POSIX use a new owned session/process group; on Windows use an owned
kill-on-close Job Object through the native platform API. Failure to establish
containment refuses activation. The parent retains the containment handle/identity
until descendants are confirmed absent, including when the audio child crashes.
The model must not inherit app IPC endpoints; otherwise parent death would not
produce EOF. App EOF makes the audio child fence immediately and close/reap its
STT child. These single-process-failure guarantees are tested; simultaneous OS
termination/suspension of all supervisors is not a claimed recovery guarantee.

## 5. Bounded IPC, identity and ordering

Use a small closed protocol: length-prefixed, versioned records with a bounded
JSON metadata header and optional bytes/text payload. Never deserialize app-side
Python objects or accept arbitrary operation names, paths or callable targets.
Validate type, integer range, payload length and identity before allocation or
dispatch. Unknown/malformed protocol fences the session with a fixed category.
No payload, transcript, PCM, exception message or credential enters diagnostics.

Every operation carries session generation and request identity; turn operations
also carry turn ID, transcript revision and attempt epoch. TTS adds phrase and
data sequence IDs; render IDs and device clocks remain child-local. Reject late,
duplicate or out-of-generation effects without giving them new authority.

Use one framed stream per direction with bounded reader/writer queues and
reserved control capacity. Neither the native callback nor the audio loop can
block on pipe I/O, app acknowledgements or queue capacity. Reader/writer threads
do the blocking I/O. At most one scheduled batch wakeup drains each bounded
receiver; no unbounded `call_soon_threadsafe` callback queue.

| Traffic | End-to-end bound / policy |
| --- | --- |
| Ordinary control | 64 records per direction, each at most 4 KiB including metadata; admission failure is terminal, not silent loss |
| Fence/close/fault | Reserved latest-generation fence and one close/fault slot outside ordinary admission; superseded fences coalesce, never disappear behind an exhausted data credit pool |
| Native/resource close receipts | One reserved bounded slot per receipt kind, each at most 4 KiB; preserve both facts through EOF, never infer resource cleanup from native closure or a normal audio PID exit |
| Preparation | At most two live requests; transcript at most 65,536 characters / 256 KiB UTF-8 per request; no unlimited text-bearing control records |
| Promotion or accepted handoff | One outstanding terminal operation; its handle retains original app objects until actual disposition |
| Provider visible deltas | Eight 4 KiB UTF-8 blocks total, with producer backpressure; preserve existing response/phrase limits |
| Normalized TTS PCM | Eight blocks of at most 64 KiB (512 KiB total), whole 960-byte 48 kHz mono PCM16 frames only, one active phrase; replace, do not stack, the existing encoded-byte bridge budget |
| TTS cleanup receipt | One session-wide 4 KiB credit-controlled record, retaining priority; accepted phrase owners stay bounded until actual receipt consumption or transport retirement |
| Projection/draft | Latest preview plus latest recoverable transcript for each of at most two retained turns, each within the transcript limit; revisions coalesce |
| Diagnostics | 64 content-free events; coalesce/drop diagnostics only, never authoritative faults/cleanup |

Draft records carry mandatory `draft_slot` integer0|1 chosen by their sender.
Receive and credit identity use that wire slot, not a local allocation index;
parent and child retire turns independently. Keep the two-turn metadata bound
and at most two frozen turn-to-slot bindings, allowing prepare/preview first.
Same-turn slot changes and replacement of queued/consumer-owned custody fail.
Sequence/replay state follows exact slot+turn and retires by turn. Coalesced
credits require complete matching stream identity before cumulative replacement;
different turns never merge merely because their lanes match. Keep exact credit
checks and independent cleanup, with no new lane/capacity/authority. Verify both
asymmetric-retirement orders through actual pipes.

Credits cover sender queues, partially written records, pipe contents, receiver
queues and consumer-owned blocks together. Returning credit merely on receipt
is invalid. PCM credit returns as validated frames are copied into the existing
bounded native render ring, or explicitly discarded by the matching fence; the
native ring is its existing separate budget, not another unbounded staging queue.
OS pipe storage is transport, not another data budget. Credit/acknowledgement
frames have reserved bounded slots to avoid a full-queue deadlock.

Keep source-format decoding/normalization with the parent-owned TTS response.
Raw PCM remains incremental. The existing whole-WAV fallback keeps its 32 MiB
source-body limit. That body and its existing bounded validation/PCM copies are
separate parent decoder allocations, not a 32 MiB total-memory promise and not
part of the 512 KiB normalized-PCM IPC budget.
Do not rewrite WAV parsing to make this boundary work. Backpressure applies to
normalized frames; cancellation stops normalization and closes the original
response on its owner. Provider/model IDs and arbitrary response metadata stay
in the app; IPC needs only fixed format, phrase/epoch and bounded sequence fields.

Data exhaustion and actual owner cleanup are distinct records. `pcm_end` names
the last data sequence; `tts_closed` reports actual response/iterator cleanup,
not a cancelled waiter. The child can accept exhausted PCM only after that
sequence is consumed/discarded. It cannot treat an early cleanup record as EOF,
start the next synthesis before true predecessor cleanup, or declare phrase
success before both facts hold. Physical playback terminal still requires native
render receipts and capture/transcript seals, not either IPC record. Provider
terminal records likewise cannot overtake their visible-delta sequence barrier.

The parent provider `cleanup` receipt includes the final published
`last_sequence`, including zero for an attempt without visible data. This
sequence is frozen by cancellation's producer-admission fence, not inferred from
the child's currently empty queue. Priority cleanup may arrive ahead of old
published deltas: the child retains their exact discard-only stream until that
boundary has been consumed/discarded and original cleanup obligations settle.
Accepted handoff waits for this actual data boundary before proposing its
consumed sequence; the original cleanup receipt remains prompt and independent.
A previously received provider terminal boundary must agree; cleanup does not
authorize a second or conflicting data end. Closing producer admission wakes new
pull waiters but does not strand already-published bounded writer custody.

The existing credited `tts_closed` receipt also names the final published PCM
`last_sequence`, including zero for no frames. This does not conflate actual
cleanup with data consumption. Record and validate that boundary independently;
only fenced/cancelled streams may use it to retire discard-only data custody,
including cancellation after cleanup arrives. Normal unfenced playback still
requires its actual `pcm_end`. Early priority cleanup must not start the next
phrase while a previous end remains in the single uncredited end slot. Do not
publish an extra cancellation end, which could overflow that slot for queued
cancellations. Matching late normal ends for cancelled streams are redundant
transport facts; conflicting boundaries fail. Retain exact late-record handling
within existing bounds. Actual response close remains independent of publication
and peer dispatch; neither a completed write nor an empty local queue proves
consumption. Test held end/early cleanup/next synthesis and three queued
cancellations with a stalled peer.

Provider preparation failure before an issued handle uses the existing
`provider_failure(code="provider_failed", last_sequence=0)` record and existing
child provider-failure state. The trusted category stays in the original parent;
provider refusal is not a transport fault or an automatic retry.

A live draft is not a recovery request. One closed child-to-parent
`draft_recovery` ordinary control record carries the existing exact
turn/revision/epoch, `action` enum `request|revoke`, and positive `recovery_id`.
It carries no payload, category, handle or callable: request refers only to the
already retained draft and follows its actual consumption barrier. Child speech,
manual cancellation or other currentness changes invalidate a pending request
once, not once per frame. Retain at most the existing two turns and latest token,
not unbounded request history. Before UI delivery the parent processes
already-received revokes and rechecks original view/session/config and token.
This is observed-message ordering, not a synchronous read of a remote child.
The original bounded recovery task/draft remains owned through actual callback
completion; claimed or accepted text cannot be recovered. No new capacity, lane,
generic RPC or diagnostic/projection side channel is introduced. Test actual
parent/child preparation failure and stale/delayed/hanging recovery, keeping the
original safe explanation, draft once, and suspended automatic retry.

An authority fence blocks new preparation/provider publication, not existing
cleanup, terminal-claim or terminal-result receipts to a live pipe. Failed
delivery records sanitized transport failure and begins containment separately
from the actual resource/claim result. It cannot turn completed actual cleanup
into missing evidence or rewrite a settled original terminal outcome.

Use the existing credit protocol for `tts_closed`, with one stable session-wide
receipt key and monotonically increasing receipt sequence, not a window per
phrase. Every receipt still carries its exact phrase identity. The child retains
the outcome in that phrase's cleanup owner before consuming its delivery and
returning credit. That acknowledgment proves receipt consumption only, not native
resource closure, PCM consumption or playback. Actual local cleanup is independent
of ordered receipt publication and acknowledgment. At most three productions
remain retained, with one published receipt and one publisher waiting for credit;
later publishers use a bounded order barrier, not extra queues/tasks. Cancellation
still closes the original response while receipt credit is occupied. Non-clean
cleanup immediately retains/reports its fatal category and prevents replacement
synthesis; it cannot wait behind receipt admission. Accepted queued cancellations
also receive truthful receipts without requiring multiple active PCM windows.

Session `closed` reports native checked-close evidence; separate child-to-parent
`resources_closed` reports the actual model/resource cleanup disposition. Each
has one reserved terminal slot and a fixed categorical outcome. Normal device
lease release requires explicit clean receipts for both, plus confirmed child/
tree/pipe closure and no forced death. Child-owned forced model termination must
remain visible even when the audio PID exits normally. Missing, late, exceptional
or unrecognized resource results remain uncertain; neither a duplicate receipt
nor a cancelled observer may upgrade failed or forced custody to clean. Retained
parser/transport uncertainty independently prevents device reuse; earlier clean
receipts cannot erase a later rejected record. EOF at an empty next-frame boundary
is normal, but EOF within a prefix, header or body is a fixed truncated-frame
failure, not proof that all final traffic was drained intact.

Priority does not mean a partially written frame can be overtaken. After that
bounded frame, write pending fences before new work; dispatch already-received
revocations before queued starts/claims. Preserve sequence barriers for terminal
data. Local audio invalidation never waits for this transport priority.

Use a coalesced 250 ms parent lease renewal and a 5 s child lease deadline.
The 3 s acceptance hold must retain a margin; tests synchronize a fresh renewal.
Lease expiry or EOF fences/ends the session, not auto-restarts it. Protocol
handshake is bounded to 5 s; STT's existing 120 s readiness deadline is separate
and occurs before capture. Normal provider latency is not a heartbeat failure.

## 6. Provider authority, cancellation and promotion

Preparation sends the transcript/revision and epoch to the parent. The parent
validates current ownership, prepares through the existing controller and returns
an opaque issued handle plus the actual eligibility decision. Its bounded map
retains the original request/context. A frozen turn-context handle survives an
obsolete attempt if the existing stable-turn handoff still needs it; lifetime is
not inferred from provider-task cancellation.

Only the parent creates `VoiceAttempt` and enters the gateway. The audio child
receives visible text, inert tool-pending categories and categorical terminal
results. Raw tool arguments, usage/trace snapshots and promotion objects remain
in the app. Parent start validates the issued handle, epoch and current cleanup
quarantine again. Unknown handles cannot request tools, supply a provider request,
change Capture policy or mint a persistence identity.

On admitted speech during generation/playback the child immediately advances its
attempt fence, stops the local sequencer/native output and continues the same
turn's transcript. It sends provider/TTS cancellation without awaiting it.
Already buffered or late old-epoch deltas/PCM are rejected locally. Parent backend
cancellation and actual TTS iterator closure run on their existing owners when
those owners can execute. A cancelled IPC waiter is never a cleanup receipt.
Preserve the existing two-obsolete-cleanup bound and app-wide quarantine. No
replacement dispatch may escape required cleanup disposition, including across
views or after the audio child is gone.

The child owns actual playback completion and the capture/DSP/VAD/STT seal.
Parent arrival time never substitutes for capture onset or DAC time. A terminal
proposal includes its local boundary, sealed transcript revision and final data
sequence. The parent uses its retained original attempt snapshot/seed to promote
the winner exactly once. It validates the supplied visible answer against its
own completed response; the worker cannot substitute an arbitrary history pair.

Parent claim and completion are separate acknowledgements. Before claim, stale
view/session checks and observed revocations win; after the existing app-owned
claim point, persistence/accepted-turn custody survives navigation or child death.
Claims and cancellations are serialized by the parent; drain already-received
revocations before claiming. IPC cannot promise retroactive cancellation of a
claim that preceded receipt of a new revocation. No microphone work or provisional
provider task gains navigation-surviving authority merely by crossing IPC.

Post-playback speech is buffered as a new turn while the preceding promotion is
pending, not appended to the heard answer's user turn. Keep the existing two-turn
bound and promotion-before-next-dispatch ordering; promotion failure preserves
the separate follow-up draft. Stable effectful turns use the existing accepted
handoff and parent classifier; no tools execute in either child.

## 7. Failure and shutdown

Differentiate capability degradation from broken transport. Unhealthy AEC with
healthy device timing keeps functional half duplex. Native/data-loss faults,
permanent render rejection, IPC failure, or lost worker ownership cannot remain
an enabled, generating "half duplex" session. Fence audio and provisional work,
preserve the latest unaccepted transcript once for its owning session, publish
one categorical failure and show Hands-free off when the app can render again.
Never promise recovery of speech still only in a killed inference process.

UI close first invalidates parent session/activation authority synchronously,
then sends the child close fence. It cannot directly dereference child-native
objects or claim synchronous device silence on IPC enqueue. Child receipt fences
native output/producer admission and starts checked close immediately, independently
of model/provider cleanup. Speech-triggered fences remain entirely local.

Observe native checked close independently with the existing 2 s deadline.
Retain ownership on cancellation or timeout. The child then closes/reaps STT using
its existing bounded procedure; process-tree shutdown has a total 6 s graceful
budget, followed by bounded terminate/kill observation. Parent callbacks retain
the result even if the view/awaiting task has disappeared.

A checked native-close receipt permits normal device-lease release. Forced
process death is containment evidence, **not** a fabricated checked-close receipt:
report shutdown-unconfirmed and keep the app's device lease quarantined for the
remaining app lifetime. Do not reopen automatically. Provider/TTS orphan receipts
and already-claimed promotions remain separately app-owned; killing audio does
not complete them. Containment failure retains quarantine and one safe error.

## 8. Software acceptance and integration

Use targeted regressions at the changed seams, then one joined mounted software
gate. No physical qualification or new live conversation is authorized here.

1. Real child launch/import/source/protocol checks: distinct PID, no app/Textual/
   model imports in audio, and no device open before successful current preflight.
   Test missing dependencies, mismatch, cancelled preparation and stale activation.
2. Native producer and actual AEC in the audio child with fake PortAudio. Hold
   the parent GIL deliberately for 720 ms and 3 s. Independently timestamp child
   capture/DSP/admission progress **during**, not only after, each hold. Require
   zero overflow, discontinuity, drift or invalid native records at default rings.
3. Synthetic admitted near-end speech during the hold must fence already queued
   old output in under 150 ms without a parent acknowledgement, preserve the turn
   and reject late bytes. Separate real-AEC tests from synthetic speech-policy
   injection; neither proves acoustic interruption on the USB route.
4. Complete fake-provider sequential TTS, several revisions in one turn, actual
   playback terminal and a distinct follow-up turn through production composition.
   Delay/reorder IPC and promotion replies; verify boundary seals, no repeated
   transcript, no obsolete output, exactly-once promotion and retained follow-up.
5. Exercise every finite queue/credit boundary, incomplete/oversized frames,
   terminal-before-data, stale handles, stalled parent reader, full TTS cancellation
   and late cleanup. Memory remains bounded and local capture/fencing stays live
   until explicit fail-closed expiry, with no silent authoritative-message loss.
   A synthetic WAV larger than 512 KiB must decode and play with bounded PCM
   credits; hold/reorder data-end and actual cleanup independently, including
   cleanup failure and cancellation while all PCM credits are occupied.
6. Device-free process/adapter fakes cover Parakeet and another native-capable
   local adapter plus batch-only rolling fallback. Verify context exclusivity,
   buffer bounds, startup failure, stuck inference, model-child EOF and reaping;
   preserve partial/final and cumulative/segment results across the boundary.
   Use no actual model load, download or inference.
7. Crash/kill the audio child with a fake model descendant, close the parent pipe,
   and interrupt shutdown observers. Check tree containment, independent native
   close, no orphan resurrection, quarantine, preserved drafts and visible off.
   Platform-specific containment tests run only on the host OS; unavailable
   Windows/POSIX counterparts stay explicit unverified platform gates.
8. Mounted visible Hands-free activation must still create
   `ConsoleSpeculativeHandsFreeSession`, now backed by the process facade, never
   legacy `HandsFreeController`. Assert selected module root/source identity and
   absence of import-time device/model work. Verify package/UI compatibility tests.

Update source/lint inventories, developer guidance and ADR-098 after written
approval alongside the implementation plan. Preserve the packaged all-unqualified
manifest, exact-HEAD source-only launcher and historical evidence. New software
results are not release qualification or proof that the live USB issue is solved.
Keep TASK-23175 In Progress and leave unrelated trace-ledger/CSS edits untouched.

## 9. Alternatives and review checkpoint

- **Recommended: isolate the whole audio-critical loop.** Removes the demonstrated
  app-GIL dependency while leaving existing authority owners intact.
- **Move only capture or enlarge rings.** Rejected: AEC/VAD and interruption would
  still stall, and a larger finite buffer would only move the failure threshold.
- **Move the whole Console/provider runtime into the child.** Rejected: unnecessary
  authority/persistence migration, heavy imports and a larger fault boundary.

Workflow checklist:

- [x] Inspect current code, task, ADRs and software/live evidence.
- [x] Resolve the current constraints and compare the three approaches.
- [x] Obtain user approval of the process-isolation direction.
- [x] Write this bounded runtime design (no visual companion needed).
- [x] Independent spec review; blocking finding fixed and round 2 approved.
- [x] Save the reviewed design in its documentation-only commit.
- [x] Obtain written user approval of this spec.
- [x] Amend ADR-098 and write the implementation plan after that approval.

The design is the binding contract, not evidence that every runtime or platform
has passed. Implementation and exact software evidence are recorded in the
linked plan and developer guide; their limitations remain part of acceptance.

Review round 1 identified a WAV/credit deadlock in the initial draft. The revised
boundary keeps existing decoding in the parent, sends normalized PCM under the
IPC budget, and separates data end from actual TTS cleanup. It also specifies STT
result semantics and excludes arbitrary TTS metadata. Independent review round 2
approved the revised design with no remaining serious planning gaps. The user
approved the written spec on 2026-09-05. Implementation follows the
[process-isolation plan](../plans/2026-09-05-speculative-voice-process-isolation.md);
approval is not implementation or runtime verification.

### Fixed runtime-source closure

The process handshake uses 53 fixed runtime sources, including root/package
initializers, audio/model composition, transport, directly composed voice helpers,
provider voice splitting/backpressure and dedicated voice trace/promotion
authority. Every listed file has byte-mutation sensitivity coverage. This is not
recursive attestation of generic configuration, prepared-request, trace-storage
or provider/installer dependency graphs. The separate source inventory includes
576 actual paths (excluding its two header comments), with 203 Python paths.
Neither inventory nor the app/child handshake changes release qualification.

The finite Task9 scheduling/turn-continuity checkpoint retains an earlier
unclassified startup-only transport failure whose original stage/exit evidence
was unavailable. Later passes do not establish its cause or repair. Test-only
startup diagnostics now retain bounded stage/category/exit facts; startup
reliability, live USB behavior and release/platform qualification remain unproved.

The completed software slice at `c36f89e24a1aa73cc302f30f962a6dea34e1244d` passed
1,432 targeted tests with zero skips and three existing warnings (241.18s), after
the final review repairs and synchronous-claim fixture correction. The plan keeps
the prior failed gate distinct. Minor N1 was deferred at that checkpoint: impossible
zero-data terminal traffic for a retired never-issued proposal could update only
terminal bookkeeping, without granting authority. This finite acceptance checked
AC16 only.

The separately approved N1 follow-up omits late-terminal bookkeeping for requests
that were never admitted. Valid late records for admitted requests and existing
retirement bounds remain unchanged. Three rejection regressions failed before
the fix; the subsequent four-file targeted software gate passed 244 tests with
one existing Requests dependency warning. This closes AC17, not a new scheduling,
startup or release acceptance gate; I1 and live/platform/release limits above
remain unchanged.

The separately approved startup-race follow-up registers the already-read
bootstrap with the existing mailbox/receiver before starting either pipe thread.
A deterministic real-subprocess test reproduced queued heartbeat/close traffic
displacing bootstrap at priority dequeue and terminating the child before session
construction. Both cases fail before the reorder and pass afterward: ordinary
startup retains its control sequence and no early credit, while early close
prevents native start and retains both clean closure receipts. No protocol,
priority, credit, timeout or runtime boundary changes. This closes the reproduced
startup race (AC18), not the missing historical I1 evidence or live/release limits.
