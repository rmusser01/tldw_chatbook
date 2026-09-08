# Speculative Voice Process Isolation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep capture, speech admission and stale-output fencing working during
a bounded app-interpreter stall, without moving provider or persistence authority.

**Architecture:** A lightweight view-session audio subprocess owns the existing
audio core, turn coordinator and phrase sequencer. The app retains original
prepared requests, provider attempts, accepted/promotion custody and TTS decoding;
bounded normalized PCM and typed messages cross private pipes. Local STT inference
remains in a separately contained process directly owned by the audio child.

**Tech Stack:** Python 3.12/asyncio/threading/subprocess/multiprocessing, stdlib
framing/JSON, POSIX process groups or Windows Job Objects, existing native ABI 1
duplex bridge/AEC, WebRTC VAD, existing STT/TTS adapters, Textual test pilot.

---

**Approved spec:** [Process-isolation design](../specs/2026-09-05-speculative-voice-process-isolation-design.md)
**Task:** [TASK-23175](../../../backlog/tasks/task-23175%20-%20Add-low-latency-speculative-duplex-voice-pipeline.md)
**Planning baseline:** `7cd97bc816ee8fc07bf56731f04d836bc50e53b7`;
runtime code baseline remains `fd89a7b0c341a26b4780d1d70cf72c28578fa2d9`.

ADR required: yes
ADR path: [ADR-098](../../../backlog/decisions/098-low-latency-speculative-duplex-voice-pipeline.md)
Reason: User-approved audio-process, bounded-IPC and cleanup-ownership amendment.

## Execution constraints and evidence

- Work only in `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/speculative-duplex-voice-adr097`,
  branch `codex/speculative-duplex-voice-adr097`. Preserve the supplied worktree.
- User approved the written spec on 2026-09-05. No new architecture approval is
  needed to implement this boundary; escalate only a material deviation.
- No microphone, live app, real provider request, model load/download/inference,
  physical harness, route changes, repetitions, matrix, soak or full test suite.
  The ordinary conversation is consumed. Synthetic/fake tests are not live proof.
- Keep native ABI/rings, 500 ms startup pre-roll, 700 ms silence default, AEC/
  acoustic-isolation admission and all original provider/persistence safety gates.
- Do not modify/stage/revert unrelated trace-ledger or CSS changes. In particular,
  `Chat/console_dispatch_checkpoint.py` and `console_dispatch_repository.py` are
  outside this plan. No blanket `git add`, cleanup, branch merge or push.
- Follow @superpowers:test-driven-development and @superpowers:verification-before-completion.
  Read relevant incidents in `backlog/docs/lessons-testing-evidence.md` and
  `backlog/docs/lessons-live-verification.md` before work. Reuse existing code per
  @ponytail: no new actor framework, WAV parser, model cache or dependency.
- Commit each coherent task only after its targeted RED/GREEN checks, lint and
  self-review. Use @superpowers:requesting-code-review at the boundary checkpoints
  below. Workers own their listed files and never revert another worker's edits;
  the root integrates shared files and serializes Git writes.

Existing local evidence is described in the spec. The 720 ms injected hold proves
the shared-GIL failure mechanism, not why the historical USB consumer stalled.
The final gate must exercise a real independent native producer, not a Python
producer frozen by the same GIL or an application loop simulated in one process.

### Test command used throughout

From the feature worktree, define this shell helper in each test shell. It creates
a unique owned temporary output directory and never removes previous evidence:

```sh
voice_test() {
  local voice_test_tmp
  voice_test_tmp=$(mktemp -d /private/tmp/tldw-voice-process-test.XXXXXX) || return
  TLDW_NATIVE_DUPLEX_ROOT=/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/lib/python3.12/site-packages \
    PYTHONPATH="$PWD" ../../.venv/bin/python -m pytest -c pyproject.toml \
    --basetemp "$voice_test_tmp" -q --tb=short "$@"
}
```

Tests live under `Tests/` so normal root/UI fixtures supply test-data and
network/audio isolation. A child does not inherit monkeypatches: explicit test
child modules must install their own fake device/model and fail-on-network guards.
Never add an environment/config/CLI factory override to the release entry.
Use `../../.venv/bin/ruff check <owned Python files>` and
`../../.venv/bin/ruff format --check <owned Python files>` per task.

## File map and dependency order

| File | Responsibility |
| --- | --- |
| `Audio/voice_process_types.py` | Dependency-light policy/cleanup/terminal/PCM value contracts; no app objects |
| `Audio/voice_turn_coordinator.py` | Existing reducer moved, not duplicated; policy decisions and context handles replace app context imports |
| `Audio/voice_transcription.py` | Existing serial STT ownership, native and rolling adapter orchestration moved from session |
| `Audio/voice_phrase_sequencer.py` | Existing phrase/Markdown policy with normalized-PCM stream input |
| `Audio/voice_process_core.py` | Existing transport/AEC/VAD/transcript/pump/seal owner and local child effects |
| `Audio/voice_process_protocol.py` | Closed record schemas, framing, credits, bounded priority/coalescing state |
| `Audio/voice_process_io.py` | Pipe reader/writer threads and one bounded loop wakeup per batch |
| `Audio/voice_process_lifetime.py` | Exact owned process-tree containment and checked exit observation |
| `Audio/voice_process_entry.py` | Private stdio bootstrap, handshake, lease, prepare/start/close state machine |
| `Chat/console_voice_process.py` | App-owned process supervisor, device lease and view facade |
| `Chat/console_voice_process_effects.py` | Retained original authority objects, parent provider operations and terminal acknowledgements |
| Existing `Chat/console_voice_tts_bridge.py` | Actual TTS owner/normalization, bounded PCM producer and independent cleanup receipts |
| Existing `Audio/parakeet_voice_worker.py` | Extend current local STT process operations; retain Parakeet compatibility API |
| Existing Chat coordinator/session/phrase modules | Compatibility exports and parent composition only after extraction |

All paths in the map are relative to `tldw_chatbook/`. New modules are listed
under their creating task below. Avoid unrelated package-wide lazy-import work:
the audio closure goes through `Audio/`, not `Chat` or `TTS` package entry points.

Sequence: Tasks 1–2 establish reusable contracts/core; Tasks 3–4 build bounded
transport/containment; Tasks 5–7 supply STT, TTS and app authority; Task 8 composes
and switches production; Tasks 9–10 verify/integrate. Tasks 5 and 6 have separate
file ownership but remain serialized when using subagent-driven-development's
per-task review gates. Do not parallel-edit shared session/composition files.

## Task 1: Lightweight policy seam and package import boundary

**Files:**
- Create: `tldw_chatbook/Audio/voice_process_types.py`
- Create: `tldw_chatbook/Audio/voice_turn_coordinator.py`
- Modify: `tldw_chatbook/Chat/console_speculative_voice.py`
- Modify: `tldw_chatbook/Chat/console_voice_eligibility.py`
- Modify: `tldw_chatbook/Chat/console_voice_controls.py` (enum alias only)
- Modify: `tldw_chatbook/Chat/console_voice_attempts.py` (cleanup enum alias only)
- Modify: `tldw_chatbook/Chat/console_voice_settings.py` (eagerness constant aliases only)
- Modify: `tldw_chatbook/__init__.py`
- Modify: `tldw_chatbook/UI/__init__.py`
- Modify: `tldw_chatbook/Widgets/__init__.py`
- Create: `Tests/Audio/test_voice_process_imports.py`
- Create: `Tests/Audio/test_voice_process_types.py`
- Test: `Tests/Chat/test_console_speculative_voice.py`
- Test: `Tests/Chat/test_console_speculative_voice_properties.py`
- Test: `Tests/Chat/test_console_voice_eligibility.py`
- Test: `Tests/Chat/test_console_voice_settings.py`

- [x] **1. Write RED import/contract tests.** Fresh interpreters must import the
  lightweight coordinator/types with no `textual`, `tldw_chatbook.Chat`,
  `tldw_chatbook.TTS`, app, provider, database, Torch or MLX modules. Importing
  UI or Widgets still installs the idempotent `Static.renderable` compatibility.
  Reject bool/negative epochs, malformed handles, unknown decisions and fabricated
  parent contexts. Test current constructor/static-policy and deferred-policy paths.
- [x] **2. Run `voice_test Tests/Audio/test_voice_process_imports.py Tests/Audio/test_voice_process_types.py`.
  Record expected RED (missing lightweight modules/eager Textual import).**
- [x] **3. Extract the reducer and typed seam.** Move existing reducer behavior
  into `Audio/voice_turn_coordinator.py`; dependency-light events carry a validated
  parent decision, request handle and separately retained turn-context handle.
  Use exact enum/value types, not a permissive `Any` or serialized
  `ConsoleTurnExecutionContext`. Parent `AttemptDispatchPrepared` compatibility
  converts its original strictly validated context/request using the unchanged
  `classify_voice_speculation`; its wrapper maps accepted-handoff handles back to
  original objects. Local test-only static preparation uses the same mapping.
  Keep original parent tool/promotion objects out of core events: tool-pending
  contains the epoch only; terminal disposition is a typed categorical projection,
  not `VoicePromotionOutcome` with its hidden commit member. Move shared enum
  definitions only where identity matters; retain public import aliases.
  Remove only package-root eager shim invocation, invoking the existing function
  at both UI and Widgets package boundaries instead. Preserve tokenizer/logging
  initialization and the current public package API.
- [x] **4. Run the six listed targeted test files, plus
  `Tests/Audio/test_audio_init_lazy_import_safety.py`; expect PASS.** Test spies
  confirm parent classification still receives the original objects by identity
  and child preparation never receives them. Lint owned files.
- [x] **5. Self-review the move and commit owned files:**
  `refactor: separate voice turn policy from app authority`.

## Task 2: Extract the audio core, transcript and normalized phrase consumer

**Files:**
- Create: `tldw_chatbook/Audio/voice_transcription.py`
- Create: `tldw_chatbook/Audio/voice_phrase_sequencer.py`
- Create: `tldw_chatbook/Audio/voice_process_core.py`
- Modify: `tldw_chatbook/Chat/console_speculative_voice_session.py`
- Modify: `tldw_chatbook/Chat/voice_phrase_sequencer.py`
- Modify: `tldw_chatbook/Audio/voice_process_types.py`
- Modify: `tldw_chatbook/Audio/rolling_transcript.py` (typed capacity failure only)
- Modify: `Tests/Audio/test_voice_process_imports.py`
- Create: `Tests/Audio/test_voice_process_core.py`
- Test: `Tests/Chat/test_console_speculative_voice_session.py`
- Test: `Tests/Chat/test_console_voice_stt_ownership.py`
- Test: `Tests/Chat/test_voice_phrase_sequencer.py`
- Test: `Tests/integration/test_speculative_voice_pipeline.py`
- Test: `Tests/Audio/test_rolling_transcript.py`

- [x] **1. Write RED core tests.** Exercise synthetic admitted-frame progression,
  same-turn fencing, exact native terminal/seal ordering, two retained turns,
  direct normalized PCM consumption and independently delayed stream cleanup.
  Add fresh-interpreter import checks for all three extracted modules.
  A fake permanently faulted transport must fence once, refuse future dispatch,
  preserve the available draft once and publish one terminal failure; AEC-only
  degradation with a healthy transport must remain functional half duplex.
  Temporary native render-ring pressure must still retry, not masquerade as a
  permanent transport fault.
- [x] **2. Run `voice_test Tests/Audio/test_voice_process_core.py Tests/Audio/test_voice_process_imports.py`;
  record RED before moving code.**
- [x] **3. Move existing behavior, with narrow adapters.** Extract
  `_SerialSttWorker`, native/rolling wrappers and their helpers into transcription;
  extract `ConsoleSpeculativeVoiceSession`, `_SessionEffects`, capture pump and
  seals into core. Inject categorical diagnostics rather than importing app log
  services. Split transcription preparation from audio start so a later parent
  permit is required. Keep compatibility exports in the existing Chat module.
  Move phrase/Markdown policy unchanged; its child-facing input is an async
  iterator of exact 960-byte PCM16 frames plus an independent cleanup awaitable.
  The old Chat facade adapts `TTSAudioResponse` through the existing normalizer,
  keeping current in-process callers/tests valid until production changes.
  Model work stays off the core loop. Bound pending native-STT frames to ten
  seconds per retained transcript in addition to its single bounded in-flight
  chunk; exhaustion is a typed transcript failure, never silent PCM loss.
  Do not introduce an unbounded task-per-frame or task-per-IPC-message queue.
  On irrecoverable transport failure synchronously fence native output and
  subsequent logical dispatch before scheduling independent cleanup/draft work.
- [x] **4. Run all six listed test files and import tests; expect PASS.**
  Keep default half-duplex startup and 240 ms VAD preroll in boundary regressions.
  Port tests to moved implementation seams only where their former monkeypatch
  targets no longer own behavior; do not weaken assertions or substitute mocks
  that fence earlier than production. Lint and inspect move-only diffs.
- [x] **5. Commit:** `refactor: extract the isolated voice audio core`.
  Request a scoped code review of Tasks 1–2 before building on their policy seam.

## Task 3: Closed protocol and bounded pipe transport

**Files:**
- Create: `tldw_chatbook/Audio/voice_process_protocol.py`
- Create: `tldw_chatbook/Audio/voice_process_io.py`
- Create: `Tests/Audio/test_voice_process_protocol.py`
- Create: `Tests/Audio/test_voice_process_io.py`

- [x] **1. Write RED codec and queue tests.** Include split/short reads, malformed
  JSON/duplicate keys, bool integers, oversized headers/body, unknown operations,
  invalid Unicode/PCM shape, stale generations, duplicate sequences, interrupted
  partial writes and EOF. Fill each queue; stop/fault/credit must still be
  admitted. Exhaust sender credits and prove no more reads from a fake producer
  until actual receiver consumption. Assert at most one scheduled receiver wakeup.
- [x] **2. Run `voice_test Tests/Audio/test_voice_process_protocol.py Tests/Audio/test_voice_process_io.py`;
  record RED.**
- [x] **3. Implement the closed schema and bounded scheduling.** Use an eight-byte
  network-order `(header_length, payload_length)` prefix, then UTF-8 JSON header,
  then payload. Header at most 4,096 bytes; maximum payload 262,144 bytes, narrowed
  by the operation before reading it. Reject unknown/extra header fields and
  operations; no arbitrary paths, callables or pickle. The fixed operation set is
  bootstrap/hello, STT-ready/start/session-ready, lease, prepare/prepared/start-
  attempt, provider-delta/end/tool-pending/failure, cancel/cleanup, synthesize/PCM/
  PCM-end/TTS-closed, terminal-propose/claim/result, preview/draft/diagnostic,
  close/closed/resources-closed/fault and credit. Each operation has explicit direction, identity,
  field and payload requirements; use one table, not extensible registration.
  Validate scalar bounds at the shared choke point, for example:

  ```python
  def checked_epoch(value: object) -> int:
      if type(value) is not int or not 0 <= value < 2**63:
          raise ValueError("voice_protocol_invalid")
      return value

  def checked_pcm(payload: bytes) -> bytes:
      if type(payload) is not bytes or not payload or len(payload) > 65_536:
          raise ValueError("voice_protocol_invalid")
      if len(payload) % 960:
          raise ValueError("voice_protocol_invalid")
      return payload
  ```

  Enforce the spec's 64 ordinary controls, two preparations, one terminal request,
  eight 4 KiB delta blocks, eight 64 KiB PCM blocks, two draft snapshots, one preview
  and 64 diagnostics. Byte credit spans sender/pipe/receiver ownership; release
  only on sequencer consumption/native-ring transfer or matching fence discard.
  Maintain dedicated coalesced slots for fence, close/fault, lease and cumulative
  credit/ack per bounded stream. Do not preempt a partially written frame; after
  it, fences precede new work. Terminal processing waits for its last sequence.
  Reader/writer threads own blocking I/O; loops use nonblocking bounded enqueue,
  asynchronous credit wait and one batched receiver wakeup. Close/fault must wake
  blocked credit waiters without forging success or dropping cleanup receipts.
- [x] **4. Run both files; expect PASS, including cancellation with every data
  credit occupied.** Inspect retained bytes/counts at sender, receiver and callback
  scheduling, not just `Queue.maxsize`. Verify diagnostics contain no test sentinels
  from payloads or exception messages. Lint.
- [x] **5. Commit:** `feat: add bounded private voice process messaging`.

## Task 4: Process containment and child bootstrap

**Files:**
- Create: `tldw_chatbook/Audio/voice_process_lifetime.py`
- Create: `tldw_chatbook/Audio/voice_process_entry.py`
- Create: `tldw_chatbook/Chat/console_voice_process.py`
- Create: `Tests/Audio/fakes/voice_process_child.py`
- Create: `Tests/Audio/test_voice_process_lifetime.py`
- Create: `Tests/Audio/test_voice_process_entry.py`
- Create: `Tests/Chat/test_console_voice_process.py`
- Modify: `tldw_chatbook/Audio/voice_process_protocol.py` (fixed reserved resource-close receipt and distinct truncated-frame failure)
- Modify: `Tests/Audio/test_voice_process_protocol.py` (matching schema/framing regressions)
- Modify: `Tests/Audio/test_voice_process_io.py` (frame-boundary versus truncated EOF regression)

- [x] **1. Write RED launch/lifetime tests using no-device child fakes.** Cover
  no launch before preflight, wrong root/source/protocol/native ABI, closed stdin,
  no permit before STT readiness/current activation, startup cancellation, 5 s
  handshake/lease expiry using fake clocks, and ordinary library stdout isolation.
  A fake model descendant must die if its audio owner crashes or parent pipe
  reaches EOF. Missing containment must refuse startup, not select legacy audio.
- [x] **2. Run `voice_test Tests/Audio/test_voice_process_lifetime.py Tests/Audio/test_voice_process_entry.py Tests/Chat/test_console_voice_process.py`;
  record RED.**
- [x] **3. Implement explicit entry and owned lifecycle.** Parent uses its current
  executable and `-m tldw_chatbook.Audio.voice_process_entry`, private pipes,
  `close_fds=True` and no shell. POSIX uses `start_new_session=True` and retains
  that exact owned group until absence is confirmed. Windows creates and configures
  a non-inheritable kill-on-close Job Object, assigns the Popen process before
  sending bootstrap permission, and observes active process count through the
  Job API; assignment failure tears down without STT/audio. Reuse the existing
  `_WindowsJobApi` primitive in `STT/executor_process_tree.py` through a parent-only
  lazy import where suitable. Its STT package entry must not enter the audio
  import closure; do not copy the unrelated multiprocessing admission controller
  or broaden this task into a package-wide refactor. Do not request job
  breakaway. Wait for child PID exit separately from descendants and native close.
  The entry duplicates private fd 0/1, makes the copies non-inheritable, redirects
  fd 0/1/2 to the null device, then imports audio dependencies. Its guarded main
  can be re-imported by the model's spawn without starting another audio process.
  Never pass app protocol endpoints into a model child. Child lease uses its own
  monotonic clock; already expired leases cannot revive. Parent renewal occurs
  every 250 ms on its background owner, coalesces and expires at 5 s.
  Retain the process facade's enter/close observers independently of caller
  cancellation. Start source checks from actual imported core paths and the
  launcher's existing source identity where available; packaged identity is
  app/worker source fingerprint plus protocol/ABI, not a fabricated Git checkout.
  Keep qualified selection strictly in the parent; entry has no enable/override
  flag and refuses absent valid private bootstrap. Report typed resource cleanup
  separately via the fixed reserved `resources_closed` receipt: native `closed`
  remains independent, and normal PID exit cannot hide forced model termination
  or an unknown resource result. Both explicit clean receipts are required for
  normal lease release. Retain fixed parser/transport uncertainty separately from
  native and resource facts. Only an empty next-frame boundary is ordinary EOF;
  EOF within a prefix, header or body is `voice_transport_truncated` and prevents
  lease reuse even when earlier cleanup receipts were clean. The test child wrapper injects
  fake factories by direct Python call only, never a release message/env option.
- [x] **4. Run the three lifecycle files plus protocol and pipe regressions; expect PASS.** On macOS test actual POSIX child/
  descendant termination, and test Windows API state transitions with explicit
  fakes. Mark real Windows containment unverified on this host; do not claim it
  was exercised. Require finite test watchdogs and verified cleanup even on assertion
  failure. Lint; verify startup leaves microphone/model factories untouched.
- [x] **5. Commit:** `feat: supervise isolated voice process ownership`.

## Task 5: Preserve local STT adapters behind the model process

**Files:**
- Modify: `tldw_chatbook/Audio/parakeet_voice_worker.py`
- Modify: `tldw_chatbook/Audio/voice_transcription.py`
- Modify: `tldw_chatbook/Audio/voice_process_entry.py`
- Modify: `Tests/Audio/test_parakeet_voice_worker.py`
- Create: `Tests/Audio/test_local_voice_stt_process.py`
- Modify: `tldw_chatbook/STT/executor_worker.py` (public buffer-only resident owner; retain source and lease checks)
- Modify: `tldw_chatbook/Local_Ingestion/transcription_service.py` (explicit mutually exclusive local-buffer-owner injection)
- Modify: `Tests/STT/test_transcription_service_facade.py`
- Create: `Tests/STT/test_resident_buffer_runtime.py`
- Test: `Tests/STT/test_local_stt_executor.py`
- Test: `Tests/Chat/test_console_voice_stt_ownership.py`
- Test: `Tests/Audio/test_rolling_transcript.py`

- [x] **1. Write RED fake-model process cases.** Cover Parakeet native/rolling,
  another local candidate with `process_audio`, batch-only fallback, partial/final
  and segment/cumulative semantics, an exact-full-chunk quiet boundary, stale
  context IDs, interrupted model entry/exit, over-limit PCM/text, model crash,
  never-returning inference and cleanup ownership. Assert all adapter operations
  execute in the model PID, never the parent/audio PID.
  For ONNX, exercise the real facade/current-process owner with fake native
  providers, forbidding any extra process. Cover configured external/managed
  source precedence, omitted settings, `int8`/`f32`, unchanged per-request
  provenance, source/closure change rejection and root/VAD lease retention across
  inference, idle reuse, failed/hung close and cancelled observers.
- [x] **2. Run `voice_test Tests/Audio/test_local_voice_stt_process.py Tests/Audio/test_parakeet_voice_worker.py`;
  record the new generic-adapter RED cases.**
- [x] **3. Extend the existing service rather than add a parallel framework.**
  Retain `ParakeetVoiceProcess` as its compatibility API. Add the narrow local-
  provider service path consumed by the extracted transcription wrappers. A
  resolved allowlisted provider/model/language and necessary non-secret local
  options select the existing `TranscriptionService` adapter only in the model
  process. One serialized request supports prepare, open/push/exit context and
  rolling-buffer operations; fake factories exist only in tests. Keep native
  candidates/context managers in the model process, with typed result metadata
  for final/partial and cumulative/segment rather than flattening every result
  to a string. Reuse existing Parakeet prewarm/context/chunk/quiet logic, 120 s
  startup, 30 s request/ownership limits, 960,000-byte PCM and 65,536-character
  result limits. The audio child owns/reaps the model directly; the app never
  brokers PCM. Validate generic adapter cleanup under the same lease and fence
  a dead service rather than advertising rolling fallback from it. Preserve the
  actual model owner's typed cleanup disposition: graceful/unopened cleanup is
  `CLEAN`, forced or abnormal death is `FORCE_CLOSED`, and unconfirmed cleanup is
  not clean. Keep the existing `close()` compatibility API if needed by exposing
  a retained outcome property; Task 8 passes this exact fact to the independent
  child resource receipt. Confirmed process absence alone is insufficient.
  Parakeet ONNX needs a narrow public buffer-only resident owner injected through
  the facade: its ordinary shared coordinator would spawn a fourth process.
  Reuse existing `_load_resident`, `_validate_reuse`, provider and request-metadata
  paths, preserving `ParakeetSourceService.resolve()` and all source/dependency
  custody. Keep one identity; reject changes without hot-swap. Reject dual owner/
  dispatcher injection; ordinary ingestion/dictation remain unchanged. Do not
  release protecting leases on uncertain native close. After attempting its
  categorical receipt, the model exits
  nonzero without unwinding its owning frame into Python finalization. Receipt
  or connection-close failure cannot skip that exit; actual parent reap remains
  required, and hung cleanup/send uses the existing kill deadline. Clean exit is
  unchanged. Reproduce bootstrap finalization retaining a live model after lease
  destruction, and positively check both leases during a guarded pre-exit hold.
  This is an ADR-098
  implementation refinement preserving ADR-025, not a new executor or provider
  boundary; both the spec and ADR record the voice-only exception.
- [x] **4. Run all seven listed test files; expect PASS with fake runtimes only.**
  Prove parent pipe EOF remains observable after model spawn and no model process
  remains after successful close. Lint and inspect provider-specific result
  semantics against `_transcript_text`/`_merge_streaming_text` behavior.
- [x] **5. Commit:** `feat: retain local STT semantics in isolated voice workers`.

## Task 6: Parent-owned TTS decoding and bounded normalized PCM

**Files:**
- Modify: `tldw_chatbook/Chat/console_voice_tts_bridge.py`
- Modify: `tldw_chatbook/Audio/voice_phrase_sequencer.py`
- Modify: `tldw_chatbook/Audio/voice_process_protocol.py` (one credited, priority TTS-cleanup lane)
- Modify: `Tests/Chat/test_console_voice_tts_bridge.py`
- Create: `Tests/Chat/test_console_voice_process_tts.py`
- Modify: `Tests/Audio/test_voice_process_protocol.py`
- Modify: `Tests/Audio/test_voice_process_io.py` (real-pipe stalled-dispatch and receipt-credit races)
- Test: `Tests/Chat/test_voice_phrase_sequencer.py`
- Test: `Tests/TTS/test_pcm_stream_plan.py`

- [x] **1. Write RED TTS ownership/pressure tests.** Use fake `TTSAudioResponse`
  streams with raw PCM and an in-memory WAV larger than 512 KiB but below the
  existing decoder limit. Assert synthesis/iteration/normalization/close owner
  identity, whole-frame output, bounded credits, no encoded data or arbitrary
  metadata in IPC, cancel while full, queued-middle-phrase cancellation and late
  natural-close completion. Deliver `tts_closed` before the last PCM block and
  `pcm_end` before cleanup; neither may fake completion or overlap synthesis.
  A short initial raw chunk must become renderable while the next source chunk
  or EOF is stalled; packet aggregation must not add an unbounded first-audio wait.
  Flood queued requests while predecessor cleanup is stalled and prove retained
  tasks/text remain bounded, with categorical rejection at capacity.
  Stall real pipe dispatch, cancel both accepted waiters, then receive all three
  cleanup facts without overflowing the single credited receipt slot. Keep real
  child cleanup observers for queued requests without opening extra PCM windows.
- [x] **2. Run `voice_test Tests/Chat/test_console_voice_process_tts.py`;
  record RED.**
- [x] **3. Reuse actual TTS service and decoder.** Add the process producer to
  the existing bridge owner, using `iter_normalized_pcm_frames` there. The original
  response and all adapter metadata stay on their owner loop. Group only whole
  960-byte frames into at most 64 KiB packets; acquire global PCM credit before
  retaining/sending another packet. The packet size is a ceiling, not a fill
  requirement: one-frame packets are acceptable when batching would require
  another pending task/queue or delay already available audio. Queued phrase
  task/text retention must also remain within a fixed bound consistent with
  ordinary control admission; one active PCM window alone is insufficient.
  Keep the existing WAV body/copy limits as
  separate decoder memory, not an encoded transport queue. `pcm_end` identifies
  the last packet; shield original response cleanup and report `tts_closed`
  separately with a categorical outcome. Reuse `CreditWindow`/`ReceiveStream` for
  one session-wide `tts_closed` credit with a stable key and increasing sequence,
  keeping its priority. Validate phrase identity and retain the cleanup outcome
  before consuming the delivery. Keep actual cleanup separate from ordered
  publication/receipt retirement; a three-production bound covers all accepted
  owners until acknowledgment or definitive transport retirement. Only one
  publisher waits for the one credit; later owners await the bounded order
  barrier, not additional queues/tasks. Test duplicate/stale/wrong/premature ack
  and final-kernel-write races. A non-clean cleanup reports its fixed fatal category
  immediately, without waiting for receipt credit or permitting a replacement.
  Failure has no success terminator.
  Starting a subsequent phrase awaits actual predecessor cleanup, including when
  an earlier proxy was cancelled. Child fencing discards late PCM and returns its
  credits, but cannot complete parent cleanup. Retain the existing app supervisor
  and sequencer's delayed-cleanup observation; do not layer another 512 KiB queue
  behind `OwnerLoopTtsSynthesizer`.
- [x] **4. Run all six listed test files; expect PASS.** Compare normalized output
  with the existing decoder for the same synthetic WAV, assert eventual progress
  beyond 512 KiB, and assert the device sink rejects old epochs during delayed
  cleanup. Lint.
- [x] **5. Commit:** `feat: stream bounded normalized PCM to the voice process`.

## Task 7: App-owned provider handles, backpressure and terminal custody

**Files:**
- Create: `tldw_chatbook/Chat/console_voice_process_effects.py`
- Modify: `tldw_chatbook/Chat/console_speculative_voice_session.py`
- Modify: `tldw_chatbook/Chat/console_voice_attempts.py`
- Modify: `tldw_chatbook/Chat/console_provider_gateway.py`
- Modify: `tldw_chatbook/Chat/console_voice_process.py`
- Modify: `tldw_chatbook/Chat/console_voice_promotion.py` (expose exact successful claim without duplicating promotion)
- Modify: `tldw_chatbook/Chat/console_voice_supervisor.py` (shared pending-cleanup admission, not only detached orphans)
- Create: `Tests/Chat/test_console_voice_process_effects.py`
- Create: `Tests/Chat/test_console_voice_process_gateway.py`
- Modify: `Tests/Chat/test_console_voice_attempts.py`
- Modify: `Tests/Chat/test_console_voice_promotion.py`
- Test: `Tests/Chat/test_console_voice_eligibility.py`
- Modify: `Tests/Chat/test_console_voice_supervisor.py`
- Modify: `backlog/docs/lessons-testing-evidence.md` (root-owned incident notes)
- Test: `Tests/Chat/test_console_speculative_voice_promotion_races.py`
- Test: `Tests/Chat/test_console_voice_effect_barrier.py`

- [x] **1. Write RED authority and cancellation races.** Retain original typed
  prepared objects and gateway-issued identity sentinels; assert their identity
  at dispatch/promotion and absence from all serialized records. Exercise at
  most two preparations, stale/unknown handles, frozen-context lifetime beyond
  attempt cancellation, app-wide quarantine across two view facades, cancellation
  before provider entry, tools becoming inert pending signals, and claimed
  promotion surviving child death. Compare proposed answer with retained provider
  response and reject mismatches. Distinct next-turn draft survives failed promotion.
  Use the real `ConsoleProviderGateway._stream_generic_chat` with a controlled
  synchronous fake provider iterator: exhaust downstream delta credits, then assert
  `next()` calls, queued bytes and scheduled callbacks stop at their stated bounds.
  Cancel while full and observe actual iterator/worker closure, including an error
  or terminal record competing with the full data lane. Use a new clean test file;
  do not edit the unrelated dirty `Tests/Chat/test_console_provider_gateway.py`.
- [x] **2. Run `voice_test Tests/Chat/test_console_voice_process_effects.py`;
  also run `voice_test Tests/Chat/test_console_voice_process_gateway.py` and record
  RED for the existing unbounded synchronous producer.**
- [x] **3. Implement parent effects using existing owners.** A bounded map keyed
  by generation/turn/epoch/request handle retains `PreparedSpeculativeVoiceAttempt`,
  original `VoiceAttempt`, snapshot and separate context lease. Parent preparation
  runs through `VoiceUiBridge`/controller and strict eligibility, with owning-view,
  current revision, configuration and activation checks before/after awaits.
  Stable accepted handoff may use a revised transcript with an older retained
  context. Keep attempt-current and context-current validation distinct: resolve
  accepted authority by exact issued request/context handles and original
  generation/request/turn/context epoch, while its terminal header carries the
  current transcript revision. Submit that latest bounded transcript with the
  original context; do not substitute the preparation text or weaken promotion's
  exact key/transcript checks. Test revised accepted text and stale context/view/
  configuration refusal before actual synchronous custody transfer.
  `start_attempt` rechecks issued handle and global cleanup quarantine. Preserve
  max-two orphan policy and actual original gateway admission/promotion paths.
  The app-owned gate retains pending cleanup disposition synchronously from
  cancellation, including across facades and after child closure; admission must
  not remain open until the later orphan-detachment threshold. Preserve actual
  detached-survivor custody and ordinary typed dispatch. Test another facade
  during held pre-detachment cleanup after the first observer/child disappears.
  A wire tool signal contains no arguments; actual tool request remains parent-
  owned for the ordinary effect barrier. Parent `VoiceAttempt` currently uses
  synchronous `_notify(on_delta, ...)` and an unbounded event queue: add an awaited
  visible-delta sink seam (retain synchronous observers for current callers),
  split UTF-8 safely into 4 KiB blocks, and wait for eight-block credit before
  pulling the next provider item. Do not schedule one task per delta or swallow
  a full-channel error in `_notify`. Cancellation must wake the credit wait and
  close the actual provider iterator; recheck epoch after every await.
  Carry this bound through the real synchronous producer: `_stream_generic_chat`
  currently has an unbounded asyncio queue plus unbounded `call_soon_threadsafe`
  enqueues, so an awaited sink alone is insufficient. For `VOICE_PROVISIONAL`,
  use cancellation-aware producer admission before queueing/scheduling, allowing
  only one outstanding bounded delivery and acknowledging it only after the async
  consumer resumes from its yield/awaited sink. Reserve one of the eight 4 KiB
  visible-data slots for this gateway handoff and at most seven for downstream
  IPC, so the combined queued/callback/in-flight visible lane stays within 32 KiB.
  The producer may retain one source item while splitting; bound that voice-only
  normalized source item to 256 KiB UTF-8 and reject larger items categorically.
  This separate single-item adapter allocation is not an IPC queue or an aggregate
  answer-length limit. Stop provider iteration at the admission bound, not merely
  queue insertion after scheduling callbacks. Terminal/error delivery has reserved
  bounded state and sequence ordering; cancellation sets the stop signal and
  wakes blocked admission before awaiting actual response/worker cleanup. Preserve
  ordinary typed-provider behavior, gateway admission and trace/usage semantics.
  Parent reducer drains already-received revocations before starts/claims. Keep
  app acceptance's actual claim separate from completion, retaining original
  operation custody if the child/view disappears after claim. Child terminal
  success maps from a typed parent outcome; never accept a boolean, lookalike
  commit or arbitrary assistant text as promotion authority. Pre-claim staleness
  abandons provisional traces; after claim, existing recovery/CAS policy wins.
  `VoiceWinningPromotion.promote` currently hides its synchronous claim behind
  an awaitable that may also represent refusal. Expose the exact successful
  original claim through a narrow parent-local seam, retaining the existing
  method and owner/trace settlement. Calling it, creating a task or receiving an
  awaitable is not claim evidence. Test no claim on refusal, claim before delayed
  settlement, and owner/trace survival after the observer disappears. The
  accepted-turn path already returns only after synchronous custody transfer;
  reuse that original result rather than adding controller authority.
- [x] **4. Run all eight listed test files; expect PASS.** Additionally hold a
  parent cancellation receipt while the child epoch is fenced: no old audio may
  resume, and a cancelled observer must not release the context/quarantine.
  Confirm ordinary typed dispatch remains unaffected. Lint.
- [x] **5. Commit:** `feat: retain voice provider authority across process IPC`.
  Request scoped review of Tasks 3–7's messaging, authority and cleanup boundaries.

## Task 8: Production composition, visible control and fail-closed lifetime

**Files:**
- Modify: `tldw_chatbook/Audio/voice_process_core.py`
- Modify: `tldw_chatbook/Audio/voice_process_entry.py`
- Modify: `tldw_chatbook/Audio/voice_process_lifetime.py` (extend lifecycle dispatch without prematurely releasing data custody)
- Modify: `tldw_chatbook/Audio/voice_process_protocol.py` (closed non-secret STT option fields in bootstrap)
- Modify: `tldw_chatbook/Chat/console_voice_process.py`
- Modify: `tldw_chatbook/Chat/console_speculative_voice_session.py`
- Modify: `tldw_chatbook/Chat/console_runtime.py`
- Inspect: `tldw_chatbook/Chat/console_voice_worker.py` (existing worker shutdown compatibility; no change required)
- Modify: `tldw_chatbook/Chat/console_voice_process_effects.py` (cancelled-provider final sequence in existing cleanup receipt)
- Modify: `Tests/Chat/test_console_voice_process_effects.py`
- Modify: `tldw_chatbook/Chat/console_voice_tts_bridge.py` (truthful cancelled-PCM final sequence)
- Modify: `Tests/Chat/test_console_voice_process_tts.py`
- Modify: `tldw_chatbook/Audio/voice_phrase_sequencer.py` (cancelled PCM final-boundary retirement)
- Modify: `Tests/Audio/test_voice_process_io.py` (required cleanup boundary and held published-write regression)
- Test: `Tests/Chat/test_voice_phrase_sequencer.py`
- Test: `Tests/Chat/test_console_voice_tts_bridge.py`
- Modify: `tldw_chatbook/Chat/console_voice_supervisor.py` (observe original pending/orphan cleanup at app quit)
- Modify: `Tests/Chat/test_console_voice_supervisor.py`
- Modify: `tldw_chatbook/UI/Console_Modules/hands_free.py`
- Modify: `Tests/UI/test_console_speculative_voice_wiring.py`
- Modify: `Tests/UI/test_console_voice_native_callback.py`
- Modify: `Tests/UI/test_console_voice_provider_preflight.py`
- Modify: `Tests/Chat/test_console_voice_worker_integration.py`
- Create: `Tests/Chat/test_console_voice_process_shutdown.py`
- Modify: `Tests/Audio/test_voice_process_core.py` (fatal notification independent of stalled draft preservation)
- Modify: `Tests/Audio/test_voice_process_protocol.py`
- Modify: `Tests/Chat/test_console_voice_process.py` (bootstrap option round-trip and startup compatibility)
- Modify: `Tests/Chat/test_console_voice_preflight.py` (migrate obsolete mutable-session fixture without losing actual preparation-failure behavior)
- Test: `Tests/Audio/test_voice_process_entry.py`
- Test: `Tests/Audio/test_voice_process_lifetime.py`

- [x] **1. Write RED mounted selection and shutdown tests.** Click the actual
  visible Hands-free switch in an isolated Console with real typed provider-
  readiness fixtures and the fake test child. Assert
  `ConsoleSpeculativeHandsFreeSession` wraps the process facade and distinct PID,
  never `HandsFreeController`, and current module root/source identity match.
  Cover provider refusal before child launch, stale readiness/start permit,
  view navigation, re-entry, unmount, app quit, native close exception/hang,
  vanished child and parent-owned orphan or claimed terminal work.
  Cover legacy dictation already starting/recording at entry or becoming active
  during readiness: speculative capture must not overlap or silently adopt it.
- [x] **2. Run `voice_test Tests/UI/test_console_speculative_voice_wiring.py Tests/Chat/test_console_voice_process_shutdown.py`;
  record RED for process selection/lifetime.**
- [x] **3. Compose and switch the qualified path.** `ConsoleRuntime` owns a lazy
  process supervisor and retains its current parent owner loop for provider/UI
  bridges. The factory resolves validated settings and local STT before passing
  only safe startup values; remove reliance on a parent `dictation_service`
  object in the child. Carry Task 5's typed device/compute-type/precision choices
  as closed validated bootstrap fields, preserving omitted defaults; never add
  an arbitrary options mapping, path or callable selector. Test malformed values
  before child preparation as well as exact valid option round-trips.
  Parent preflight, source-only selection gate and visible
  session wrapper remain unchanged. Child prepares STT then waits for current
  parent start permit before native `enter`. Recheck legacy dictation ownership
  on the UI owner at entry and at that permit; refuse conflicting activation
  without stopping the existing recording or discarding its draft. Passing the
  former capture_live flag cannot transfer a parent stream into another process.
  Cache bounded immutable snapshots
  for UI; do not send lambdas or expose a shared core via `observe(callback)`.
  Reuse the preview lane for cumulative assistant preview and per-turn draft
  lanes for live user text; match turn/revision/epoch when composing the existing
  immutable projection. Live draft receipt is not composer recovery. Reproject
  on either lane, reject stale/coalesced old attempts, and clear from authoritative
  cancellation/terminal lifecycle facts. Parent phases describe only observed
  lifecycle/attempt operations, never inferred native playback or AEC health.
  Keep audio-critical state in the child; no extra wire snapshot schema.
  The prerequisite lifecycle dispatcher releases synchronous control records on
  return; data dispatch must instead retain the real mailbox delivery until
  consumption or matching fenced discard. Do not reuse lifecycle auto-release
  for PCM/provider payloads or mistake startup-only permits for ordinary control
  credit handling.
  The existing parent provider cleanup receipt also carries required
  `last_sequence` from the final published delta (zero if none). Cancellation
  stops new producer admission but cannot strand already-published bounded
  records; late records are consumed/discarded through that exact boundary before
  old child-attempt retirement. Cleanup is not an implicit empty-stream claim.
  Accepted handoff also waits for that actual data boundary before proposing the
  consumed sequence, without delaying the original cleanup receipt. Cover the
  combination of priority cleanup, held old data and revised accepted handoff.
  Test cleanup overtaking delayed data, zero-data cancellation, consistent prior
  provider end and conflicting boundaries; keep actual cleanup/terminal receipts
  independent. No new opcode or fabricated consumption credit is introduced.
  The existing credited `tts_closed` receipt carries required `last_sequence`
  from the final published PCM block (zero if none); do not add a cancellation
  `pcm_end`. Record and validate the boundary independently of actual cleanup.
  Only fenced/cancelled streams may use it for discard-only data retirement,
  including fencing after cleanup arrives. Normal unfenced playback still waits
  its actual `pcm_end`, so early priority cleanup cannot start another phrase
  while the previous end occupies the one uncredited slot. Matching late end
  records for cancelled streams are redundant transport facts, not new authority;
  conflicting boundaries fail. Keep exact late-record retirement bounded. Test
  held end/early cleanup/next synthesis, delayed old PCM, three queued cancellations
  with stalled dispatch, and eventual release without delaying original close.
  Convert former cross-thread internals tests into direct core tests or fixed
  content-free process snapshots through test seams, not callable IPC.
  Preserve production preparation-failure recovery: safe original failure
  categories and explanation, original draft once, no automatic retry, and
  stale view/session/config/speech cancellation checks at actual delivery.
  Do not regain test compatibility by copying removed production callbacks into
  a direct-core fixture. Verify the real parent/child preparation-failure path
  independently of legacy policy fixtures; provider refusal is not an IPC fault.
  Map failed preparation before a handle through existing
  `provider_failure(code="provider_failed", last_sequence=0)`; keep the trusted
  failure category in the original parent only. Use one closed child-to-parent
  `draft_recovery` control record with exact turn/revision/epoch, `action` of
  `request` or `revoke`, and positive `recovery_id`. It has no payload, category,
  handle or callable and references only the existing retained draft. Request
  follows actual draft consumption; the child invalidates pending requests on
  actual speech/manual/currentness change, at most one revoke per request.
  Parent drains already-received revokes before recovery delivery and checks
  original view/session/config plus the exact token at the UI callback. Retain
  only the existing two turns/latest token and original recovery task through
  actual completion; no claimed/accepted text may be recovered. Use ordinary
  control credits/capacity, not a new lane or disposable diagnostics/projection.
  Test actual production failure, resumed speech before delivery, session ABA,
  closed/replaced view, duplicate/malformed tokens, late draft, no automatic retry
  and hanging recovery. Keep coordinator/audio state child-local, not callable IPC.
  An authority fence must still permit existing cleanup/terminal claim/result
  receipts to a live pipe. Definitive delivery failure records a sanitized
  transport failure and starts containment separately; it cannot erase or rewrite
  original cleanup or claimed settlement. Verify actual paired child close.
  Parent close immediately invalidates authority; child-received close starts
  native fencing and its two-second observation independently of STT cleanup.
  Retain close receipts through caller cancellation. After a six-second graceful
  tree budget, use two-second terminate observation then two-second kill
  observation; inability to confirm absence stays quarantined. Only a checked
  native close can release normal device ownership; forced death permanently
  quarantines that app's device lease. Do not stop the parent loop under live
  provider/TTS cleanup or claimed promotion. Child exit retires no parent cleanup
  authority. Add a narrow shared-supervisor
  `wait_for_cleanup()` observation of original pending and detached cleanup across
  their owner loops, with no admission release, cancellation propagation or
  private-state polling. App quit fences new work before awaiting this receipt;
  test pending-to-survivor transfer and cancelled/cross-loop observers. A fatal
  native/data/IPC event retires the unusable session, cancels provisional work,
  preserves only recoverable
  unaccepted draft revisions once, and updates visible Hands-free off on its
  owner loop. Fatal notification and native shutdown must not await parent draft
  preservation or its IPC credit; test a never-returning preservation observer
  as well as exceptions, retaining late recovery delivery without repainting a
  replacement. AEC-only health loss continues functional half duplex. No hidden
  fallback to the prior shared-interpreter speculative path.
- [x] **4. Run all five listed UI/worker/shutdown files and
  `Tests/Chat/test_console_voice_preflight.py`, plus the listed protocol and
  process/entry/lifetime/core regressions; expect PASS.** Include stale late
  failure after replacement, full outbound queue on quit and checked close
  arriving after the original waiter disappeared. Lint owned code.
- [x] **5. Commit:** `feat: select isolated speculative voice from the visible control`.

## Task 9: Independent native-producer parent-GIL acceptance

**Files:**
- Modify: `Tests/Audio/fakes/voice_process_child.py`
- Modify: `Tests/Audio/fakes/native_duplex_helpers.py`
- Modify: `Tests/Audio/fakes/native_duplex_driver.cpp`
- Create: `Tests/UI/test_console_voice_process_continuity.py`
- Create: `Tests/integration/test_speculative_voice_process_pipeline.py`
- Modify: `tldw_chatbook/Audio/voice_process_protocol.py` (explicit sender draft slot and full credit identity)
- Modify: `tldw_chatbook/Audio/voice_process_lifetime.py` (draft sender/receiver slot identity)
- Modify: `Tests/Audio/test_voice_process_protocol.py`
- Modify: `Tests/Audio/test_voice_process_io.py` (required draft-slot fixture migration)
- Modify: `Tests/Chat/test_console_voice_process_effects.py` (required draft-slot fixture migration)
- Test: `Tests/Audio/test_voice_process_lifetime.py`

Runtime repair required by the finite continuation test: independent parent/child
turn retirement made inferred draft slots disagree. Add mandatory `draft_slot`
integer0|1 to draft only, derived from the sender's retained slot and used unchanged
for receive/credit identity. Keep the independent two-turn metadata bound
(prepare/preview may retain first) and at most two frozen turn-to-slot bindings.
Reject same-turn slot changes and replacement of queued/consumer-owned custody.
Track each slot's current owner separately from the at-most-two retained turn
bindings. After an empty slot is reassigned, old retained metadata cannot reclaim
it by replay, and retiring that old metadata cannot clear the new owner's state.
Scope sequence/replay state to exact slot+turn and retire by turn. Credit
coalescing compares complete stream identity before cumulative replacement;
different turns cannot merge merely because their lane matches. StreamKey
matching must validate the explicit slot, not assume its own lane is correct.
Keep full reservation/credit identity, byte and sequence checks and independent
cleanup. No early retirement, receipt relabelling, new opcode/lane/owner/capacity
or release selector. Entry retirement remains unchanged. Require actual paired
pipes with both asymmetric-retirement orders, prepare before new draft and held
consumer custody; malformed/missing slot, replay, same-turn slot change,
occupied-slot replacement and full-identity credit-coalescing negatives. Then
complete the native follow-up acceptance. Existing ADR098 applies; no new ADR.

Implementation evidence boundary (2026-09-06): one earlier `speech_first` test
failed during child startup with `transport_failed`, before native production.
Its original stage/exit facts were not retained, and neither later successful
execution nor source inspection established the cause. Bounded test-only startup
diagnostics now retain future stage/category/exit evidence. Keep this incident
explicitly **unclassified**, not repaired or attributed to the historical USB
stall. The controller accepts only the specified finite software evidence after
concrete review defects are corrected; this is not startup-reliability, live USB
or release qualification. The final whole-change review must see this limitation.
Separately, quality review reproduced an unlocked first-draft mailbox scan racing
with actual writer release; that concrete defect requires existing-condition
synchronization and a deterministic real-pipe regression before Task9 acceptance.

- [x] **1. Port the specific existing reproduction, not its temporary scaffolding.**
  Read `/private/tmp/test_voice_native_conversation_probe.py` and the two evidence
  reports named by the spec. Reuse repository native driver helpers and mounted
  fixtures; do not invoke the physical harness. Parent fixture owns network/config
  isolation. Test child installs fake PortAudio, fake STT and fail-on-network
  guards before bootstrap; it imports the same production core/entry runner.
  No arbitrary test factory can be selected through production IPC or config.
- [x] **2. Write and run `voice_test Tests/UI/test_console_voice_process_continuity.py Tests/integration/test_speculative_voice_process_pipeline.py`.
  Record RED on a controlled pre-isolation/shared-interpreter fixture first,
  then test the production process facade.** The C++ producer emits paced 10 ms
  callbacks independently of both Python interpreters. Start each finite hold
  after a fresh lease acknowledgement. Use a tiny native test-driver hold function
  called through `ctypes.PyDLL` for exactly 720 ms and 3 s, without a hardware API.
  Timestamp callback, DSP, admission and output-fence observations in child/native
  memory while the parent cannot execute; read the bounded results afterward.
- [x] **3. Require the following assertions, with default native rings/AEC:**

  ```python
  assert child_pid != parent_pid
  assert captures_during_hold > 0
  assert dsp_acks_during_hold > 0
  assert capture_overflows == render_overflows == reference_overflows == 0
  assert discontinuities == drift_faults == invalid_native_records == 0
  assert 0 <= local_fence_ns - admitted_speech_ns < 150_000_000
  assert interruption_turn_id == original_turn_id
  assert followup_turn_id != original_turn_id
  assert late_old_epoch_render_count == 0
  ```

  Separate actual AEC processing of synthetic constant PCM from injected admitted
  speech; do not describe this as acoustic speech recognition. Seed enough TTS
  PCM for the interrupted interval, then hold actual parent cleanup until resume;
  no fake parent callback may cancel output on the child's behalf. Complete a
  revised reply after several interruptions, terminal playback and follow-up with
  fake provider/STT/TTS through actual preparation/promotion. Preserve default
  preroll and test post-boundary speech admission before and after
  `AttemptPlaybackTerminal` delivery, plus delayed original parent promotion.
- [x] **4. Run both files to GREEN; record exact commands/counts/limits.**
  Avoid `pilot.pause()` in measured intervals (it schedules per-widget callbacks);
  use explicit readiness events and bounded monotonic deadlines. Test startup/
  teardown are excluded from conversation timing. Keep fake native status and
  forced overflow negatives so the now-isolated path demonstrably fails closed.
  This is one finite targeted acceptance group, not repetitions or a soak.
- [x] **5. Lint/review/commit:** `test: prove voice progress during app interpreter stalls`.

## Task 10: Source closure, joined verification and completion record

**Files:**
- Modify: `tldw_chatbook/Audio/voice_process_lifetime.py` (fixed runtime-source fingerprint closure)
- Modify: `Packaging/speculative_voice_source_paths.txt`
- Modify: `Packaging/speculative_voice_python_paths.txt`
- Modify: `Tests/Packaging/test_voice_source_digest.py`
- Modify: `Tests/Packaging/test_speculative_voice_lint_scope.py`
- Modify: `Tests/integration/test_speculative_voice_pipeline.py` (migrate the three obsolete production-factory fixtures exposed by the joined gate; preserve their behavioral assertions at current process boundaries)
- Test: `Tests/Packaging/test_speculative_voice_development_entry.py`
- Test: `Tests/Packaging/test_installed_distribution.py`
- Modify: `Docs/Development/TTS/speculative-duplex-voice.md`
- Modify: `backlog/docs/lessons-testing-evidence.md` (proven composition incidents only)
- Modify: `Docs/superpowers/specs/2026-09-05-speculative-voice-process-isolation-design.md`
- Modify: `Docs/superpowers/plans/2026-09-05-speculative-voice-process-isolation.md`
- Modify: `backlog/tasks/task-23175 - Add-low-latency-speculative-duplex-voice-pipeline.md`

- [x] **1. Add the new code/tests and approved spec/plan to existing source/lint
  inventories, sorted without duplicates.** Extend guards to include this plan's
  exact Python scope; require every listed source to exist and be a regular file.
  Complete the fixed app/child runtime-source fingerprint with the actual model,
  effects, transport and shared helper closure introduced by composition; do not
  confuse this handshake fingerprint with historical qualification authority.
  Include root and package-ancestor initializers and directly composed voice
  contracts, even where an initializer is currently empty. Keep a reviewed fixed
  list rather than dynamically hashing the whole app; document the boundary of
  any generic infrastructure exclusions without claiming exhaustive attestation.
  Test-only child wrappers remain under `Tests/`, excluded from release artifacts.
  Keep all historical source hashes and packaged qualification manifest unchanged.
- [x] **2. Run source/launcher guards:**
  `voice_test Tests/Packaging/test_voice_source_digest.py Tests/Packaging/test_speculative_voice_lint_scope.py Tests/Chat/test_console_voice_settings.py Tests/Packaging/test_speculative_voice_development_entry.py`.
  Also run `voice_test Tests/Packaging/test_installed_distribution.py::test_built_artifacts_match_distribution_contract`.
  Its fixture copies sources into test scratch and builds wheel/sdist with
  `--no-isolation`; it does not install dependencies or run the broader installed-
  distribution suite. Require existing build tools and fail-on-network guards.
  Expect PASS with valid positive current-version fixtures and stale identities
  rejected; no dependency installation or network build is authorized.
- [x] **3. Run the joined gate below after implementation and review corrections.**
  Do not expand to the full suite or any physical/soak wrapper. Record non-host
  platform skips explicitly. Run Ruff only on changed Python paths from this
  plan and `git diff --check` on the owned changes; verify ABI/version unchanged.
- [x] **4. Request final whole-change review against the approved spec.** Resolve
  actual findings with targeted regressions, rerun their tests and the final
  joined group if code changes affect its assertions. Do not call old evidence
  current or attribute the historical stall to the injected test pause.
- [x] **5. Update developer guide and task notes with exact committed source,
  test counts, safety limits and unverified platforms/live behavior.** Mark only
  AC16 if all its outcomes are proved; overall TASK-23175 remains In Progress
  because release/platform qualification is outside this request. Add an incident
  lesson only if implementation yields new generalizable evidence. Commit only
  owned files and report completion of this software slice without launching
  the app, rebinding/opening a launcher, asking for audio or merging the branch.

### Exact joined software gate

The main joined command is:

```sh
voice_test \
  Tests/Audio/test_voice_process_imports.py \
  Tests/Audio/test_voice_process_types.py \
  Tests/Audio/test_voice_process_core.py \
  Tests/Audio/test_voice_process_protocol.py \
  Tests/Audio/test_voice_process_io.py \
  Tests/Audio/test_voice_process_lifetime.py \
  Tests/Audio/test_voice_process_entry.py \
  Tests/Audio/test_local_voice_stt_process.py \
  Tests/Audio/test_parakeet_voice_worker.py \
  Tests/Audio/test_rolling_transcript.py \
  Tests/Audio/test_duplex_transport.py \
  Tests/Audio/test_native_duplex_stream.py \
  Tests/Audio/test_voice_preprocessor.py \
  Tests/Audio/test_audio_init_lazy_import_safety.py \
  Tests/Chat/test_console_speculative_voice.py \
  Tests/Chat/test_console_speculative_voice_properties.py \
  Tests/Chat/test_console_speculative_voice_session.py \
  Tests/Chat/test_console_speculative_voice_promotion_races.py \
  Tests/Chat/test_console_voice_attempts.py \
  Tests/Chat/test_console_voice_promotion.py \
  Tests/Chat/test_console_voice_effect_barrier.py \
  Tests/Chat/test_console_voice_eligibility.py \
  Tests/Chat/test_console_voice_stt_ownership.py \
  Tests/Chat/test_console_voice_supervisor.py \
  Tests/Chat/test_console_voice_preflight.py \
  Tests/Chat/test_console_voice_worker.py \
  Tests/Chat/test_console_voice_worker_integration.py \
  Tests/Chat/test_console_voice_tts_bridge.py \
  Tests/Chat/test_voice_phrase_sequencer.py \
  Tests/Chat/test_console_voice_process.py \
  Tests/Chat/test_console_voice_process_tts.py \
  Tests/Chat/test_console_voice_process_effects.py \
  Tests/Chat/test_console_voice_process_gateway.py \
  Tests/Chat/test_console_voice_process_shutdown.py \
  Tests/TTS/test_pcm_stream_plan.py \
  Tests/UI/test_console_speculative_voice_wiring.py \
  Tests/UI/test_console_voice_native_callback.py \
  Tests/UI/test_console_voice_provider_preflight.py \
  Tests/UI/test_console_voice_process_continuity.py \
  Tests/integration/test_speculative_voice_pipeline.py \
  Tests/integration/test_speculative_voice_process_pipeline.py \
  Tests/Packaging/test_voice_source_digest.py \
  Tests/Packaging/test_speculative_voice_lint_scope.py \
  Tests/Packaging/test_speculative_voice_development_entry.py \
  Tests/Packaging/test_installed_distribution.py::test_built_artifacts_match_distribution_contract \
  Tests/Chat/test_console_voice_settings.py
```

## Planning/review log

- [x] User approved independently reviewed written spec; amend ADR-098.
- [x] Inspect implementation seams, import traps, native test driver and source guards.
- [x] Write this dependency-ordered, software-only plan.
- [x] Independent whole-plan review and correction (two rounds; approved).
- [x] Commit reviewed plan/ADR/task documentation before implementation.
- [x] Execute Tasks 1–10 with task-scoped reviews and fresh verification evidence.

The original planning checkpoint claimed no implementation or runtime result;
completed software prerequisites are recorded below.

Review round 1 found that the generic provider gateway could bypass downstream
backpressure through its existing unbounded producer queue/callbacks. Task 7 now
owns the voice-specific gateway admission seam and a real-gateway synchronous-
provider regression, included in the joined gate. Round 2 approved the whole plan
with no remaining serious gaps or harmful complexity. Both reviews were read-only.
The source-inventory/lint-scope guards passed 21 tests (one existing requests
dependency warning); document links, file targets and the owned diff check passed.
No runtime, hardware, model or provider verification ran at this checkpoint.

Task 1 seam inspection confirmed three identity-preserving alias edits: the
existing `ControlKind` and `AttemptCleanupOutcome` enums and response-eagerness
constants must share their lightweight definitions without importing `Chat`.
The listed ownership and settings regression were extended narrowly; no values
or runtime policy change is intended.

Task 1 completed with independent specification and code-quality approval (no
findings). The first boundary gate recorded 23 expected failures and two existing
passes before implementation. An unknown-cleanup regression recorded three
assertion failures before requiring exact CLEAN/FORCE_CLOSED enum receipts;
unknown receipts now suspend as required by the approved fail-closed contract.
An unchanged cosmetic-terminal property also failed against the original reducer:
its fake omitted the existing classification drain. Supplying that fake receipt
and an ordering assertion preserves the exact winning-snapshot check. Baseline
probe files remain in `/private/tmp/tldw-voice-baseline.90pNTE`.

The root's final Task 1 gate passed **278 tests, zero skips, in 17.57 seconds**:
the six listed files, lazy Audio imports, effect-barrier and VoiceAttempt tests.
Ruff check and format check passed all 13 changed Python files. Existing warnings
were requests dependency compatibility and webrtcvad/pkg_resources deprecation.
This verifies the policy/import prerequisite only, not isolated-process behavior.

Task 2 inspection confirmed the native-adapter failure handler otherwise allows
rolling fallback for any error except a closed Parakeet service. Its ownership
is extended narrowly to recognize the typed native-queue capacity failure as
terminal: losing admitted PCM cannot be repaired by a later fallback window.
The existing rolling-transcript regression file joins the task-scoped gate.

Preparation for Task 4 located the existing Windows Job Object primitive and its
fake API tests in the STT executor. The plan now calls out parent-only reuse to
avoid duplicating native structure/API definitions while retaining Popen ownership
and the voice-specific shutdown budget. No STT executor code has been changed.

Task 2 extracted the existing core, transcription and phrase algorithms behind
the dependency-light boundary. Its initial import/core gate recorded 14 expected
failures and four passes before the move. Additional regressions reproduced a
close-during-start race and review-found production factory/cleanup defects.
The factory now wires rolling diagnostics and actual serial-worker cleanup;
iterator closure and the first natural TTS response close retain independent
ownership through cancellation. The incident is recorded in
`backlog/docs/lessons-testing-evidence.md`.

Quality review also reproduced duplicate pumps under concurrent startup and an
unbounded new sequence-owner index. Preparation/start now share retained shielded
owners; every entry path observes the same completion, and fences guard queued
preparation and late readiness. The redundant index was removed and its test fake
corrected to reject foreign sequences, preserving the original bounded lookup.

Independent specification and code-quality re-reviews approved the corrected
scope, including the reviewer's original real-response cleanup probe and final
eight/nine-case startup/ownership rechecks. The root's fresh joined gate passed
**473 targeted tests, zero skips, in 27.25 seconds**. Ruff check and format check
passed all 12 owned Python files; the two warnings remain the existing requests
dependency and webrtcvad/pkg_resources warnings. These results cover extraction/
lifecycle behavior, not process isolation, real audio, provider/model execution
or qualification. Source/lint inventory integration remains Task 10.

Task 3 uses the canonical dependency-light local-STT provider catalog and preserves
nullable model selection. Correlation IDs remain distinct from parent-issued
request/context handles. A terminal proposal carries the bounded sealed transcript,
promotion/accepted kind, final sequence, nullable child-local boundary identity,
and answer SHA-256/character count; the parent will compare those answer facts with
its retained original response. This is a bounded representation of the approved
authority contract, not a new promotion or qualification authority.

The first protocol gate passed 131 targeted tests, but independent specification
review reproduced missing draft consumption, unchecked projection replacement
identity, and acknowledgments releasing entirely unwritten custody. Regressions
were added before correction. Drafts now exercise complete data/credit round trips
for both retained turns and successive revisions; cleanup and session-closed
messages follow their actual parent/child owners. Re-review required a further
writer-phase correction: a deferred receipt is still impossible while known
unfinished framing bytes remain. Receipts now defer only around a potentially
complete final write; a short write or interruption invalidates impossible pending
receipts before capacity can be returned. Independent specification re-review
approved all corrections after 118 targeted protocol/pipe tests passed. The root's
intermediate protocol/import/contract gate passed 150 tests with no skips.

Code-quality review then reproduced two additional transport defects: a priority
fence made legitimate already-queued data trigger a protocol fault, and closing a
reader's descriptor during a partial read allowed descriptor reuse to redirect
its subsequent read. Regression-first corrections retain bounded matching-fence
discard and real credit return, including local reverse-direction cancellation,
partial frames, full queues and subsequent streams. A reader exclusively owns
its transferred descriptor until its read thread exits; wrappers must transfer
ownership or pass a duplicate, and peer closure precedes bounded join/release.

Integration must register each inbound stream before data or an overtaking fence.
Ordinary barge-in cancels the sole producer and drains known fenced data through
matching discard/credit; `CreditWindow.close()` is terminal admission closure, not
ordinary speech cancellation. Unknown/replayed/retired stream identities still
fail closed. Native completion and real provider/TTS cleanup remain independent.

Both independent final specification and code-quality reviews approved the scope,
each reproducing **134 passing protocol/pipe tests**. The root's final four-file
gate passed **166 targeted tests, zero skips, in 3.66 seconds**. Ruff check and
format check passed all four owned Python files. Only the existing requests
dependency compatibility warning remains. This completes the private messaging
prerequisite, not subprocess/GIL isolation or live/release qualification. No
hardware, app launch, provider/model execution, installation, soak or full suite
ran. Source/lint inventory integration remains Task 10; Task 4 follows.

Task 4 starts from messaging commit `dc79ee929f`. Its lifecycle test wrapper uses
direct Python session injection, never a release argument, environment variable
or bootstrap factory. Until Tasks 5–8 supply the model/effects/production
composition, the release entry validates the private handshake and reports fixed
`startup_failed` without constructing a model or device. Task 8 replaces this
narrow static fail-closed composition body; this intermediate checkpoint cannot
be represented as a functional production process.

The extracted core's total close currently waits for native closure and then
resource cleanup. Task 4 therefore keeps native checked-close observation separate
from total session/process-tree completion; Task 8 must bind that distinct receipt
to the real transport's retained close owner. Neither killing the process nor
cancelling a total-close observer can manufacture device-closure evidence.

Task 4's initial three-file RED gate recorded **23 failures** against the missing
implementation before production code was added. Subsequent subprocess checks
use dedicated temporary roots and finite cleanup deadlines. Self-review reproduced
late handshake/invalid startup revival, failed-launch lease retention, permission
errors mistaken for absence, and native receipts first completed after their
deadline. The corrected three-file gate passed 39 tests; the root's joined
lifecycle/protocol/import/contract gate passed **205 tests, zero skips, in 19.43
seconds**, with Ruff check/format passing all seven owned Python files and the
one existing requests compatibility warning.

Independent specification review then reproduced forced SIGTERM death after a
clean native receipt releasing the device lease (child return code -15, incomplete
resource cleanup, replacement acquisition allowed). This conflicts with the
approved permanent-quarantine policy; forced-termination evidence must remain
independent of native-close evidence. Subsequent regressions corrected this;
the initial green gate alone was not Task 4 completion.

Root also reproduced final native-receipt loss with a deterministic reader pause:
the child exited normally and had completed native closure, but the parent stopped
the reader after PID/tree absence and before reading the final wire receipt.
The one-case scratch probe failed in 4.30 seconds with native confirmation false
and all other closure evidence true. The regression moves into the repository's
guarded test fixture; teardown must retain bounded EOF/receipt observation before
stopping the reader. This is a software-only race, not a new live-audio observation.

The corrected Task 4 gate passed 42 tests and specification re-review approved
both fixes (42 tests in 16.93 seconds). Root's joined gate passed **208 tests in
20.22 seconds**, zero skips, with Ruff check/format passing seven Python files.
Quality review then reproduced child-owned model termination disappearing from
the parent evidence: native closed cleanly, the model was terminated/reaped by the
audio child, the audio PID exited normally, and replacement lease acquisition
was still allowed. The resource observer's disposition must be typed and retained
independently from native closure, including forced or unknown cleanup. Inferring
clean resources from the absence of a fault is insufficient; Task 4 adds one fixed
reserved child-to-parent `resources_closed` receipt with the existing categorical
outcomes. The exact lightweight cleanup enum is the child adapter contract, and
both explicit clean receipts are required before ordinary lease release. No
extensible protocol/factory mechanism or real-model work is added. This implements
the existing ADR-098 independent-cleanup/quarantine contract; no new ADR is needed.
The spec's bounded-traffic/receipt description is clarified accordingly. This
quality correction entered the next gate below.

The explicit resource-receipt correction passed 164 tests; specification recheck
approved it (164 tests in 21.96 seconds). Root's seven-file gate passed **231
tests, zero skips, in 25.23 seconds**, with Ruff check/format passing nine Python
files. Quality re-review then reproduced a parser-level duplicate bypass: the
pipe reader rejected a repeated receipt, but its fixed error category was erased
before final lease evaluation, leaving earlier clean evidence able to release
the device. Regression coverage must traverse real pipes/parser admission, not
only call the parent's receipt handler. Preserve protocol uncertainty separately
from native/resource facts while allowing an ordinary fully drained EOF. This
last quality correction required the real-pipe gates below.

Real-pipe duplicate/malformed, truncated-frame and peer-fault regressions recorded
RED before the integrity correction. The parent now retains fixed transport
uncertainty independently of native/resource facts and inspects retained reader/
writer failure before the final lease decision. Only an empty next-frame boundary
is normal EOF. The worker's five-file gate passed 210 tests; root's seven-file
gate passed **242 tests, zero skips, in 27.41 seconds**, with Ruff check/format
passing ten Python files. Specification recheck next inspected the child-side
reporting path when POSIX group self-termination is not applicable.

Specification recheck reproduced that child-side gap through real pipes and a
scoped non-POSIX entry proxy: malformed parent traffic led to exit result zero and
two clean receipts, with no fault. POSIX self-SIGKILL had masked the missing
category. The existing five-file gate passed 210 tests in 25.60 seconds, while
the additional guarded regression failed in 2.13 seconds. The child must retain
and report a fixed non-EOF transport fault independently of its cleanup receipts;
truncation maps to `transport_failed` on the existing closed fault schema. This
branch simulation is not real Windows validation.

The child-side correction recorded four RED cases before retaining one fixed
fault-report attempt and inspecting thread-retained failure independently of
callback delivery. An undeliverable writer fault also preserves nonzero exit
evidence; native/resource facts remain truthful, and ordinary EOF remains clean.
The final worker gate passed 215 tests. Independent specification recheck approved
the correction with 18 passing entry tests in 5.68 seconds; final quality review
approved all ten Python files after **215 tests passed in 22.12 seconds**, with
process-group absence confirmed.

Root's fresh final seven-file gate passed **247 targeted tests, zero skips, in
26.88 seconds**. Ruff check/format passed all ten owned Python files and the
scoped diff check passed. Only the existing requests dependency warning remains.
This completes Task 4's supervised-ownership prerequisite under ADR-098. Real
Windows is unverified and production composition remains Task 8; this is not
GIL-continuity, live voice or release-qualification evidence. No hardware, app,
provider/model execution, install, native production rebuild, soak or full suite
ran. Task 5 continues; AC16 and overall TASK-23175 remain incomplete/In Progress.

Task 5 starts from reviewed supervision commit `c137eb5d76`. A fresh implementer
owns only its listed model/transcription files and uses fake model processes for
verification. Preserve the compatibility service and original serial adapter
ownership, adding typed result and actual model-cleanup outcomes. The real child
composition remains deferred to Task 8; no model, microphone or provider execution
is part of this prerequisite.

Task 5's narrow service API extends the existing request/ownership machinery with
`LocalVoiceSttProcess` and typed bounded `LocalSttOptions`, retaining the Parakeet
constructor and direct two-argument test factory compatibility. Fixed native-kind
and transcript-result metadata replace model objects across this existing pipe.
Actual child context/candidate/service cleanup acknowledgement plus normal reap
is required for `CLEAN`; launched unknown or forced cleanup cannot become clean
merely through process absence. Parent option resolution and its closed app/child
schema stay with Task 8 composition. This directly implements ADR-098, not a new
provider/plugin boundary.

Task 5's first two-file RED gate completed before production changes: **22 failed,
10 passed, 10.01 seconds**, with the existing requests warning. Nineteen cases
require the new narrow local-process/options API; three existing Parakeet native,
hung and crashed-worker cases require the retained cleanup outcome. Fake child
factories install model/device/egress guards and replace only the heavyweight
service facade beneath the production runtime adapter.

The initial two-file adapter gate then passed **32 tests in 17.16 seconds**,
including actual fake audio-to-model spawning, parent EOF observation and reaping.
The implemented options remain the typed nonsecret device/compute-type/precision
subset; production entry is unchanged. Adversarial bounds, cancelled-observer
coverage, the four-file gate and independent reviews remain pending.

Root compared the original candidate cleanup contract with the new model helper
and requested a guarded regression for async-only cleanup. Three RED cases showed
that successful `aclose` was skipped and failed/hung cleanup could still report
`CLEAN`. A fourth RED case showed failed runtime preparation being retried through
rolling fallback. The four-case run failed in 2.74 seconds before correction.
Cleanup must actually complete in the model process, and failed construction must
fence that service rather than advertising a usable fallback. Review/final gates
remain pending.

The new model cleanup receipt introduced a distinct RPC admission hazard:
`SystemExit`/`KeyboardInterrupt` during inference made the next record a cleanup
receipt, whose truthy string tag was interpreted as inference success. Two guarded
RED cases failed with an attribute error in 2.29 seconds. Require exact boolean
reply tags and operation-specific success values, retaining terminal uncertainty
before any rolling retry. The earlier four-file gate passed 139 tests in 26.24
seconds, but these new corrections still require fresh verification and review.

Final provider inspection found the real ONNX facade requires the app's shared
coordinator, whose executor spawns a fourth process and cannot run inside the
daemon model child. A direct native load would omit artifact/source custody.
Independent spec/ADR review approved a narrow public buffer-only current-process
resident owner and explicit facade injection as a faithful implementation of the
already approved three-process design. Reuse existing load/reuse/provider paths,
retain ADR-025 source/lease/provenance guarantees, preserve ordinary callers, and
retain protecting leases through uncertain native closure. No new user approval
or duplicate ADR is required for this refinement; new processes, generic dispatch
or weaker custody would exceed it. File ownership/tests and ADR-098/spec are
updated before implementation. The pre-refinement four-file gate passed **144
tests in 30.43 seconds**, with lint/format clean and a READY-before-EOF plus POSIX
private-pipe inode-absence check. ONNX refinement and final reviews remain pending.

The ONNX refinement's two-file RED gate completed before production edits:
**17 failed, 34 passed in 7.03 seconds**, with the existing requests warning.
Failures require the absent public resident buffer owner and facade injection.
Real temporary artifact/source fixtures and fake native providers cover managed
root/VAD leases during inference and idle reuse, external/managed/explicit source
precedence, `int8`/`f32`, changed sources/closure, request provenance/options, and
uncertain close. The new voice-only owner can reuse existing load/reuse/provider
and buffer argument helpers while leaving shared executor close behavior intact.

Four additional RED cases run the real facade inside the spawned model process:
**4 failed, 36 deselected in 4.16 seconds**. `int8` reaches the missing dispatcher,
`f32` is rejected by the old option allowlist, and failed/hung cleanup cannot yet
reach the resident owner. The guarded model child forbids additional process
creation. Failed-close coverage also calls the real facade cleanup with its owned
source service, checking that its `finally` path retains root/VAD protection.

The real-facade cleanup-order regression then reproduced **2 failures, 13
deselected in 6.12 seconds**: the newly nested `finally` performed source cleanup
despite an unconfirmed resident close and could replace the primary failure.
Run injected owner close before the unchanged ordinary legacy/source cleanup
chain, retaining that whole facade until model exit if close is uncertain.
An intermediate cancellation run hung in fixture teardown after its model was
reaped; it is not passing evidence. Replace the fake shared multiprocessing-event
wait with a child-local hang under the existing finite watchdog, then verify that
case separately before the joined gate.

The revised individual cancellation/close-hang cases passed **2 tests, 38
deselected in 4.68 seconds** with a 15-second per-test watchdog. The interrupted
pytest process was confirmed absent, with no owned model left. The subsequent
seven-file gate passed **281 tests in 54.75 seconds**, with a 30-second per-test
watchdog and one existing requests warning. Eight-file lint and scoped diff
checks passed; only changed formatting is being normalized in executor code
with pre-existing unrelated formatter differences. Independent review still
needs to settle whether failed-close custody outlives model-worker return through
Python process finalization, then finish spec/quality review and fresh root gates.

The post-format seven-file repeat passed **281 tests in 51.95 seconds**. Spec
review confirmed a real remaining custody gap in the target CPython **3.12.11**:
returning from `_serve` can drop its runtime/lease graph before multiprocessing
finalizers and thread shutdown finish. A stalled bootstrap can therefore leave
MODEL alive without protecting locks. The accepted minimal correction attempts
the `FORCE_CLOSED` receipt and connection close, then calls model-only nonzero
`os._exit` in an exception-proof `finally` while ownership remains in the frame.
No global registry, new process or ordinary clean-shutdown change is needed.
Actual audio-parent reap is still mandatory; blocked cleanup/send remains bounded
by that parent. Add RED/GREEN lease checks across a guarded finalizer/pre-exit
hold before final review. This completes the existing custody contract.

The guarded retirement RED case reproduced that gap in **1 failed test, 40
deselected, 5.12 seconds**: `FORCE_CLOSED` had arrived and MODEL remained alive in
a bounded bootstrap finalizer with an independent fake native owner, yet both
temporary root/VAD removal probes were `AVAILABLE` instead of `BUSY`. Receipt-send
and connection-close exception variants are added before the terminal-exit fix.

The expanded retirement RED group produced **3 failed tests, 40 deselected in
6.76 seconds**. Send/close `BaseException` variants also reached the late Python
finalizer rather than immediate retirement (their retained traceback happened to
keep the leases busy). A test-only wrapper around the real `os._exit` supplies
the positive pre-exit lease barrier; no production test selector is introduced.

The narrow retirement fix passed the focused **11-test group, 32 deselected in
8.03 seconds**, including all three positive pre-exit root/VAD lease barriers,
failed/hung native close and clean/failed receipts. It uses nested `finally`
around receipt/connection handling and a nonzero model exit only for uncertain
retirement; runtime locals remain live until that exit. Eight-file lint and
scoped diff checks passed. The fresh seven-file gate and final reviews follow.

The corrected seven-file gate passed **284 tests in 48.30 seconds**. A test-only
portability refinement injects receipt/close failures using the actual platform
pipe connection type. That final frozen snapshot passed **284 tests in 47.15
seconds**, one existing requests warning, with all-eight lint/scoped diff clean.
Seven files are formatter-clean; `git show HEAD` piped into Ruff confirms that
the executor's remaining whole-file format differences predate this change and
are intentionally preserved. Independent full Task 5 spec review is now active.

Independent spec review approved the frozen eight-file implementation, with
**284 passed, zero skips in 41.95 seconds**, all-eight lint and scoped diff clean,
and the same formatting/warning limitations. Inspection confirmed no ONNX lease
exists at constructor failure, generic failed context entry/exit retains its
candidate, and failed ONNX close retains the facade/resident/lease through the
exception-proof abnormal exit. A fresh quality reviewer is checking correctness
and maintainability before root's joined regression gate and scoped commit.

While quality review reads the frozen snapshot, root's fresh ten-file joined
gate passed **359 tests, zero skips in 50.91 seconds** with the one existing
requests warning. This adds audio-core/import-boundary and Console-session
compatibility to the seven STT targets. Root independently confirmed all-eight
lint, seven-file formatting, and diff checks; Ruff's remaining executor format
diff touches only unchanged helper lines, not the new owner. These results must
be refreshed if review changes the snapshot. No hardware, model/provider request,
app launch, production-native build, qualification or full suite was used.

Fresh independent quality review approved the same snapshot with no actionable
findings. Its seven selected test nodes expanded to **17 passed, zero skips in
9.90 seconds**, covering metadata/context cancellation, terminal failures,
source/closure validation and pre-exit lease custody; lint/diff checks passed.
No code changed during either review or root's final joined gate. Task 5 is ready
for its scoped commit; Task 6 follows. The actual process composition and GIL
acceptance remain pending, so AC16 and overall TASK-23175 remain incomplete.

Task 5 committed as `f18dbf9426f94ba7334013eeb590fe0a56582f4b`
(`feat: retain local STT semantics in isolated voice workers`), exactly 13
reviewed paths. The original 19 unrelated trace-ledger/CSS changes are preserved.

Task 6 completed parent-owned normalized PCM and a child stream adapter without
stacking the legacy encoded-byte queue. One ready 960-byte frame is sent per
credit; actual predecessor cleanup and old PCM credit retirement precede the
next synthesis. Three accepted owners bound pending task/text retention. Actual
cleanup and transport retirement have independent cancellation-isolated receipts.

The new file first recorded **12 expected failures in 1.19 seconds**. An initial
implementation run stopped at 44.62 seconds when pre-coroutine cancellation
stranded cleanup; the existing started guard was reused and regression-tested.
Real stalled-dispatch tests then exposed that three cleanup completions could
overflow the uncredited single-slot receipt lane. Independent boundary review
recommended reusing one existing session-wide credit with exact phrase validation,
not omitting receipts, enlarging an uncredited queue or adding a new opcode. The
spec/ADR and scope were refined before implementation. Its RED evidence was
**4 protocol/final-write failures in 2.58 seconds** and **6 TTS receipt/identity
failures in 1.35 seconds**. Actual cleanup proceeds despite held receipt credit;
non-clean cleanup retains fatal failure immediately and closes replacement
admission. Production pipe I/O and the original decoder limits are unchanged.

Independent spec review found repeated cancelled-owner fencing of a retired key.
The three `aclose/fail/fence` regressions failed as expected, then passed after
the first-revocation guard; affected targets passed 59 tests. Scoped re-review
approved. Fresh quality review found two further P2 cases: unrequested
`CancelledError` falsely reporting clean without PCM-end, and naturally
exhausted old owners fencing after replacement. Both were reproduced before
minimal existing-state guards; the affected three-file gate passed **74 tests,
zero skips in 7.06 seconds**. Scoped quality re-review approved both fixes and
found no new breakage. No findings remain open for Task 6.

Root's final seven-file joined gate on the corrected frozen code passed
**254 tests, zero skips in 8.18 seconds**. All seven owned Python paths passed
Ruff lint/format and scoped diff checks. The sole warning is the pre-existing
requests dependency compatibility warning. The legacy bridge test needed no
edits; six Python files changed. This is software prerequisite evidence, not
isolated production conversation or hardware qualification. AC16 remains
unchecked and TASK-23175 remains In Progress. The exact scoped commit follows.

Task 7 preparation located a hidden claim boundary: the existing winning
promotion returns an awaitable for either synchronous claim or refusal. Task 7
now includes a narrow exact-claim observation seam and its existing clean tests.
An awaitable alone is not claim evidence. Accepted-turn handoff already transfers
custody synchronously; no new controller authority is authorized. Task 7 code
starts only after Task 6's reviewed commit.

Task 7 completed bounded provider IPC, original parent-owned preparation and
claim authority, retained contexts/drafts and cancellation-isolated terminal
custody. The real synchronous gateway acquires admission before pulling the
next provider item or scheduling delivery; typed chat retains its original
chunk shape. Accepted handoff uses the latest transcript/revision with the
original issued context, while promotion requires the exact prepared attempt.
Same-turn replacement retires superseded contexts only after actual cleanup and
credit retirement; a following turn cannot retire a preceding claimed promotion.

Independent spec review exposed a cross-view admission gap before pending
cleanup became a detached orphan. Meaningful RED tests reproduced it; the
existing app-wide supervisor now retains pending cleanup synchronously and
transfers it atomically to the original survivor, sharing the existing bound.
Scoped spec re-review approved. Fresh quality review then reproduced terminal
completion incorrectly substituting for concurrent cleanup during context
retirement. Two permanent RED regressions cover CLEAN and DETACHED outcomes;
the predicate now requires every present receipt. Scoped quality re-review
approved with no new findings. The incident is recorded in
`backlog/docs/lessons-testing-evidence.md`.

Root's final ten-target joined gate on the reviewed frozen code passed
**252 tests, zero skips in 40.27 seconds**. All twelve owned Python paths passed
Ruff lint/format and scoped diff checks. The sole warning is the existing
requests dependency compatibility warning. Both review gates are approved;
there are no open Task 7 findings. Actual production composition, received-batch
ordering and the finite software-only GIL acceptance remain Tasks 8/9 duties,
not claims made by these prerequisite tests. No app, hardware, provider/model
call, qualification, soak or full suite was used. AC16 remains unchecked and
TASK-23175 remains In Progress; the scoped commit precedes Task 8.

### Task 8 completion record — 2026-09-06

The visible factory now selects the contained process facade beneath the existing
session wrapper. Mounted tests prove a distinct guarded child PID and source root,
typed startup/options and refusal of overlapping legacy dictation. Private paired
pipes retain provider/PCM data credit separately from cleanup, original context,
terminal claims and runtime disposal. Live previews and preparation-failure draft
recovery preserve currentness and safe copy without exposing mutable child state.
Native/fatal shutdown does not wait for parent draft preservation or model cleanup.

Composition required truthful final sequences on existing provider/TTS cleanup,
one closed content-free recovery request/revoke control, and observation of the
existing shared cleanup supervisor. ADR-098 and the approved process spec record
these bounded refinements; no new authority, capacity, native ABI or release
override was introduced. The existing worker and app shutdown seams stay unchanged.

Independent specification review passed. Quality review found accepted handoff
could precede delayed cancelled provider data; a paired-pipe RED reproduced it.
The reviewed four-line proposal barrier fixes that race without delaying cleanup.
Scoped re-review approves with no new blocking issue. One Minor migrated native
test assertion uses an unwritten event list; it remains explicitly deferred for
final whole-change review, not represented as meaningful fault evidence.

Fresh root verification on the frozen reviewed fix: **515 passed, zero skips,
144.06 seconds**, with 26 owned Python paths lint/format clean and diff checks
clean. Two existing warnings remain: requests dependency compatibility and pydub
audioop deprecation. A separate 16-test prerequisite import/local-STT check passed
on unchanged paths. These are software composition results, not Task 9 native
continuity, Task 10 source/artifact closure, acoustic or platform qualification.
No live app/audio, provider/model request, dependency install, production-native
build, repetition, soak or full suite ran. AC16 remains unchecked and the overall
task stays In Progress; Tasks 9–10 and final whole-change review continue.

### Task 9 completion record — 2026-09-06

The finite software acceptance is complete from reviewed base `6e8d6a54c8`.
A separate audio child with the installed ABI1 bridge, fake PortAudio and actual
AEC on synthetic PCM preserved callback/DSP progress and local output fencing
through 720/3000/720 ms parent-GIL holds. Actual preparation/promotion completed
three same-turn interruptions, a revised reply and a distinct post-playback
follow-up; both terminal/preroll orders held original promotion. Native status
and overflow controls switched Hands-free off and recovered one draft. None of
this is acoustic speech recognition, live USB proof or release qualification.

The continuation test reproduced sender/receiver draft-slot disagreement after
independent metadata retirement. Mandatory sender `draft_slot` and bounded slot
ownership now preserve exact custody/credit identity. Paired pipes failed both
retirement orders before the repair. Quality review separately reproduced a
first-draft scan racing with real writer release; internal existing-condition
synchronization fixed it, with a deterministic real-pipe RED/GREEN regression.
No receipt relabelling, early retirement, new lane/capacity or ABI change.

Independent spec review confirmed the amended implementation; scoped quality
re-review approved Q1 after fix1. The root's final unchanged-hash ten-path freeze
passed **357 targeted tests, zero skips, in 84.48s** across eleven targets. All nine
owned Python paths passed Ruff lint/format. Two existing warnings remain:
requests dependency compatibility and audioop deprecation. Earlier root356 is
pre-Q1-fix evidence, not the final gate.

I1 remains explicit: an earlier speech-first test failed during startup before
native production, with insufficient original stage/exit evidence to determine
why. Later passes and the Q1 fix do not diagnose or repair it. Bounded test-only
startup diagnostics preserve future failures. The controller accepts only the
finite software slice and carries this unresolved startup-reliability limitation
to developer guidance and final whole-change review. TASK-23175 remains In
Progress; AC16 awaits Task10 and the final review. No new live conversation,
hardware qualification, routes, soak, model/provider call or full suite ran.
Task10 source/fingerprint/artifact closure follows this scoped checkpoint.

## Task10 implementation checkpoint — 2026-09-06

ADR required: no new ADR.
ADR path: backlog/decisions/098-low-latency-speculative-duplex-voice-pipeline.md.
Reason: complete the approved fixed runtime-source closure and migrate obsolete
factory fixtures; ownership, protocol1, native ABI1 and release authority stay unchanged.

The fixed handshake now includes53 runtime sources and all package ancestors,
including provider voice splitting/backpressure and dedicated voice trace
contracts. Inventories contain576 source paths (two extra header comment lines)
and203 Python paths; guards bind both approved plans, annotated/Inspect entries,
regular non-symlink files, exact Python projection and per-runtime byte sensitivity.
The boundary excludes the named generic infrastructure graphs and is not recursive
application attestation. The three old factory integration fixtures now exercise
the actual child STT/core lifecycle and parent process promotion binding.

Independent spec review found the omitted direct authorities; fix1 mutation
RED4 became GREEN54, and all S1/S2/S3 findings passed scoped re-review. Fresh quality
review approved; only opaque test tuple indexing and existing warnings were
reported as Minor and retained for final triage. Covering guards128 and migrated
composition157 passed. The first joined v1 run had1414 passes and3 obsolete-fixture
failures; it is retained as failure evidence, not accepted as the final result.

The final unchanged-hash six-source freeze passed the exact46-target command
above: **1420 passed, zero skips,3 warnings,244.22s**. Existing300s per-test default,
`PIP_NO_INDEX=1 PIP_DISABLE_PIP_VERSION_CHECK=1`, main3.12 venv and installed native
bridge were used. The artifact node built copied scratch sources with existing
build tools/`--no-isolation`, not an install or production-native build. Exhaustive
checks found all53 current runtime files present and byte-identical in both the
2153-member wheel and2355-member sdist. The test child, source-only development
launcher and both Packaging inventories are absent from both. Root evidence is
under `/private/tmp/tldw-task10-final-joined.SmYdQN`.

Native child41837 recorded72/300/72 captures and73/301/71 DSP acknowledgements
through720/3000/720ms parent-GIL holds; admitted-speech local fence intervals were
116209/103666/53209ns. Overflow, discontinuity, drift, invalid/fatal native status
and late-old-audio observations were zero. These finite synthetic software facts
are not acoustic/live recognition or a cause for the historical desktop/USB stall.

All65 changed Python paths pass full-file lint;64 pass format checks. The only
format exception is pre-existing untouched `STT/executor_worker.py` formatting,
which remains included in lint. Warnings: Requests dependency compatibility,
pydub/audioop and webrtcvad/pkg_resources deprecation. Historical qualification
manifest/build identity hashes and all-unqualified flags, native sources and the
19 unrelated trace-ledger/CSS diffs remain unchanged. Owned diff checks pass.

I1 remains an unclassified earlier startup-only failure; later success does not
diagnose or repair it. Startup reliability, non-host containment platforms, live
USB/acoustic behavior and release qualification remain unproved. Whole-change
review and its findings are the remaining completion checkpoint; AC16 stays
unchecked until that review. TASK-23175 remains In Progress. No live audio/app,
provider/model request, route/qualification matrix, repetition, soak or full suite.

### Final whole-change review repair plan

The complete74-path review of7cd97bc816..f8ed0a53b9 found two Important gaps:
replacement provider admission can invalidate legal delayed cancelled data, and
the child converts recoverable speech-only failure to global fatal shutdown.
The1420-pass checkpoint lacks those combined regressions and is not final
completion. Two Minors are the unwritten native-test event list and opaque
promotion-callback tuple indexing. I1 remains separately unclassified.

ADR required: no new ADR; existing ADR098 bounded custody/recovery policy applies.
One complete fix wave owns Audio/voice_process_entry.py and
Chat/console_voice_process_effects.py plus the existing process-shutdown,
speculative-pipeline integration and native-callback UI test files. Additional
scope requires a concrete cause before edits. First prove actual-pipe/coordinator
late-data replacement and child-local speech-failure RED; repair admission without
blocking audio or delaying real cleanup, retain epoch/currentness and fatal
uncertain-cleanup policy; prove GREEN including stopped/superseded waiters and
newer-speech text-promotion suppression. Address both test Minors without new
production APIs. Then run amended-seam checks, one scoped re-review and the exact
joined group because runtime assertions change. No live/physical/full-suite work.

### Task10 joined-gate fixture correction

ADR required: no new ADR.
ADR path: backlog/decisions/098-low-latency-speculative-duplex-voice-pipeline.md.
Reason: repair the verification helper to preserve the existing synchronous claim
and asynchronous original settlement contract; no production boundary change.

Final root gate1 on the frozen review fix selected1432 cases:1429 passed and3
follow-up tests failed (270.41s,0 skips,3 existing warnings). All share a mounted
fixture that replaces synchronous VoiceWinningPromotion.promote with an async
wrapper. A dependency-light diagnostic distinguishes its missing terminal_claim
from normal pending cancellation cleanup. Do not weaken the new admission barrier
or relabel cleanup to make these tests pass. Keep the failed evidence.

The user's persistence instruction permits this explicit bounded exception to
the final-wave cap: correct only existing Tests/UI/test_console_voice_process_continuity.py,
Tests/Audio/fakes/voice_process_child.py and Tests/integration/test_speculative_voice_process_pipeline.py
as needed. Preserve synchronous real claim creation, gate actual owner-rooted
publication, observe settlement separately, and assert child-received claim/result
and turn retirement. No production edits, extra test matrix, app/live/hardware,
provider/model or full suite. Read-only scoped fixture review and one fresh exact46
joined gate follow; retain all existing authority/native/unrelated-file checks.

## Final software-slice completion — 2026-09-06

Committed runtime/test source: `c36f89e24a1aa73cc302f30f962a6dea34e1244d`
(`fix: preserve voice replacement custody and recovery`), eight scoped paths,
625 insertions and 31 deletions above the Task10 checkpoint `f8ed0a53b9`.
The exact reviewed eight-file SHA256 freeze matched after the final run and at
commit; unrelated trace-ledger/CSS files were never staged.

R1 replacement admission now preserves legal cancelled provider data until actual
child discard/credit and separately waits returned parent credit, rechecking
currentness through supersession and Stop. R2 local speech-input-limit failures
reach text-only winning promotion only after clean/data-complete phrase settlement
and no active synthesis; newer speech suppresses obsolete promotion, and uncertain
or failed TTS/transport stays fatal. Both test Minors were corrected.

The complete 74-path final review was followed by one production repair review:
R1/R2/M1/M2 addressed, no new Critical/Important breakage. Minor N1 is explicitly
deferred: a never-issued retired proposal can accept one impossible matching
zero-data provider terminal record, changing only its terminal-seen bookkeeping.
It cannot recreate an attempt or grant provider/promotion authority. Strict
rejection of that impossible retired traffic remains an improvement, not claimed
coverage. I1 remains a separate unclassified startup-only failure with no causal
repair; later passes do not erase it.

The first repaired-source joined gate had **1,429 passed and 3 failed** (270.41s).
All three failures shared a fixture that converted synchronous winner claiming
into an async wrapper, omitting the child claim receipt. A narrow diagnostic
proved the missing receipt, not normal pending cancellation cleanup, blocked
retirement. Under the user's persistence instruction, the bounded Task10 fixture
correction preserved real synchronous claiming and held actual owner-rooted
publication. Actual child claim/result receipt and completed follow-up/retirement
assertions were added. Scoped fixture review approved with no new findings.
Neither production admission nor cleanup was weakened; the failed run is retained.

The final fresh exact 46-target command above passed **1,432 tests, zero skips,
three existing warnings, 241.18 seconds**, exit 0, in the main Python 3.12 venv.
The existing 300s per-test timeout and no-index/no-version-check build environment
were unchanged. Final basetemp:
`/private/tmp/tldw-final-voice-fixture.5nQR0d`.
All 53 runtime sources are present and byte-identical in both the 2,153-member wheel
and 2,355-member sdist. The source-only launcher, test child and Packaging inventories
are absent. Full-file lint passes 65 Python paths; format passes 64, retaining only
the pre-existing untouched `STT/executor_worker.py` formatting exception. Warnings
remain Requests dependency compatibility, pydub/audioop and webrtcvad/pkg_resources.

Synthetic native child 48292 recorded 72/300/72 captures and 72/300/72 DSP
acknowledgements through 720/3000/720ms app-GIL holds. Local admitted-speech fencing
took 64125/46083/44708ns; overflows, discontinuities, drift, invalid/fatal native
status and late obsolete render were zero. Both delayed-promotion terminal orders
and the distinct completed follow-up passed with actual peer receipt/retirement
assertions. This proves finite software scheduling and ownership only: no live
recognition, historical USB/desktopswap cause, startup reliability, non-host
containment or acoustic/release qualification.

Historical manifest/build-identity hashes remain unchanged, all five packaged
platforms remain unqualified, native source diff is empty, and the protected 19
binary diff remains
`601b06a6481de59adce4fa3a2273291471828f2372d44acc847f31666bb3b867`.
No extra live app/audio, provider/model request, physical harness, route switching,
matrix, soak, dependency install, production-native build or full suite was used.

Task10 and the approved finite software slice are complete. Only AC16 is newly
checked; TASK-23175 remains In Progress because broader live/platform/release
qualification is outside this request. The feature branch/worktree is preserved;
no merge or push. Detailed review/diagnostic/full-command evidence is retained in
`/private/tmp/tldw-voice-process-record.cMaqI4/2026-09-05-speculative-voice-process-isolation` (recoverable scratch archive), including
`final-root-gate-2.md`, both failed/intermediate reports and both scoped reviews.
The native/build scratch basetemps are preserved separately.

### Rulings I made

1. Reused the credited TTS-cleanup lane; wrong choice could stall cleanup delivery.
2. Kept session and phrase identities distinct; wrong routing would fail closed.
3. Refused activation during legacy dictation; risk: conservative refusal.
4. Preserved original context with revised accepted speech; risk: stale-context admission or lost revisions.
5. Shared pending-cleanup admission across views; risk: early reopening or unnecessary blocking.
6. Kept app shutdown waiting on original cleanup ownership; risk: premature exit or a stuck quit.
7. Reused bounded cumulative previews and turn drafts; risk: stale text or less-detailed status.
8. Added truthful final provider sequence to cleanup; risk: late-data faults or stranded attempts.
9. Initially proposed cancellation PCM-end receipts; risk: late PCM or phrase leaks. Superseded below.
10. Used credited TTS-cleanup final sequences instead; risk: premature synthesis or late-PCM faults.
11. Migrated preparation-failure fixtures while retaining real-process recovery checks; risk: masking a recovery regression.
12. Added a closed, identity-bound draft-recovery signal; risk: stale recovery or lost bounded text.
13. Kept cleanup and terminal receipts deliverable after authority fencing; risk: false quarantine or rewritten results.
14. Tested speech before/after playback-terminal delivery; risk: another ordering lacks direct process-level evidence.
15. Made draft slots sender-owned with exact credits; risk: turnover faults or cross-turn acknowledgements.
16. Accepted finite software evidence while retaining unexplained startup failure I1; risk: an intermittent startup defect remains.
17. Used a fixed direct-runtime fingerprint, including initializers; risk: missed dependencies or excessive invalidation.
18. Migrated the three obsolete factory tests to current owners; risk: fixture changes conceal a production regression.
19. Deferred N1: one impossible terminal record can update a retired, never-issued request's bookkeeping, but grants no authority; risk: hidden malformed traffic or future misuse.
20. Corrected the verification fixture beyond the review-wave cap under your persistence instruction; preserve synchronous claims and hold real settlement. Risk: masking behavior or extra review/test churn.
