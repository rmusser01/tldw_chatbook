# Speculative voice UI isolation implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement task-by-task. Steps use checkbox tracking.

**Goal:** Keep capture and interruption operational while Textual's loop is blocked.

**Architecture:** One lazy app-owned voice event loop hosts view-owned audio
cores. UI authority and the shared TTS service retain their owner loop, reached
through bounded, cancellable bridges with separate completion receipts.

**Tech Stack:** Python 3.12, asyncio/threading/concurrent.futures, Textual,
existing duplex transport, provider gateway, TTS service, Parakeet subprocess.

**Spec:** `Docs/superpowers/specs/2026-09-04-speculative-voice-ui-isolation-design.md`

ADR required: yes
ADR path: `backlog/decisions/098-low-latency-speculative-duplex-voice-pipeline.md`
Reason: Approved scheduling and cross-loop service-boundary amendment.

## Global Constraints

- Keep the existing 700 ms eagerness default, rolling fallback, acoustic gates,
  50 ms residual-echo classification, terminal watermarks, and cleanup limits.
- No new dependency, larger transport buffer, relaxed timing/AEC threshold,
  device route change, hardware qualification, repetition, matrix, or soak.
- Do not modify unrelated trace-ledger or CSS edits.
- The thread shares Python's GIL; do not claim process-wide isolation.
- Never use a foreign-loop asyncio future, task, iterator, lock, or HTTP client
  directly. Cancelling an observation is not owner completion.
- Capture/PCM/transcripts remain ephemeral. Diagnostics are content-free.
- Targeted tests only, using the repository's main `.venv/bin/python`, pytest,
  and Ruff. Use unique `/private/tmp` pytest basetemp directories.

### Task 1: Bounded owner-loop TTS bridge

**Files:**
- Create: `tldw_chatbook/Chat/console_voice_tts_bridge.py`
- Create: `Tests/Chat/test_console_voice_tts_bridge.py`

**Interfaces:**
- Consumes the existing `TTSAudioResponse` and a callable
  `synthesize_hands_free(*, text: str) -> Awaitable[TTSAudioResponse]`.
- Produces `OwnerLoopTtsSynthesizer(owner_loop, synthesize)` with
  `async synthesize_hands_free(*, text: str) -> TTSAudioResponse` and
  `async aclose() -> None`. The synthesizer and returned proxy are used on
  the voice loop; underlying synthesis/iteration/closure run on owner_loop.

- [x] Write owner-identity, ordering, full-channel cancellation, delayed-close,
  no-overlap, cancellation-before-start, and error-redaction tests first.
  The behavioral core is:
  ```python
  proxy = await bridge.synthesize_hands_free(text="test")
  async with proxy:
      assert b"".join([chunk async for chunk in proxy.byte_stream]) == b"abc"
  assert observed_owner_threads == {ui_thread_id}
  ```
- [x] Run new test file; record expected RED before implementing.
- [x] Implement one owner coroutine per phrase, a `queue.Queue(maxsize=8)`
  of copied blocks no larger than 65_536 bytes, metadata and terminal state
  outside the data queue, asynchronous backpressure, and cancellation fencing.
  ```python
  submitted = asyncio.run_coroutine_threadsafe(produce(), owner_loop)
  # Producer finally resolves an independent concurrent Future receipt.
  # Shield this receipt from observer cancellation; never equate
  # submitted.cancelled() with response.aclose() having completed.
  ```
  Do not start a new synthesis until the previous owner receipt settles.
  Proxy `aclose()` fences bytes, requests cancellation, and waits for actual
  response cleanup; the phrase sequencer's existing supervision handles delays.
- [x] Run bridge tests plus `Tests/Chat/test_voice_phrase_sequencer.py` and
  targeted existing TTS admission cases that exercise real service ownership.
  Run Ruff on owned files; self-review and commit only owned files.

### Task 2: Voice worker and UI lifetime bridge

**Files:**
- Create: `tldw_chatbook/Chat/console_voice_worker.py`
- Create: `Tests/Chat/test_console_voice_worker.py`
- Modify: `tldw_chatbook/Chat/console_voice_supervisor.py`
- Test: `Tests/Chat/test_console_voice_supervisor.py`

**Interfaces:**
- Produces `ConsoleVoiceWorker`: `async run(factory)` executes an awaitable
  factory on its one lazy loop; `async aclose()` closes registered sessions,
  cleans owned gateway resources, and joins off-loop within two seconds.
- Produces `VoiceUiBridge(owner_loop)` with fenced `async call(factory)`,
  bounded `prepare(factory)` (two requests), coalesced `project(callback, value)`,
  synchronous `fence()`, and one terminal-request slot.
- Produces `WorkerVoiceSession(worker, build_core, ui_bridge)` exposing
  `enter(capture_live=...)`, `fence_and_close(reason)`, cached `state`, and
  worker-side `observe(callback)` for explicit immutable diagnostic snapshots.

- [x] Write failing tests for worker progress during an event-bounded UI wait,
  owner-loop identity, stale callbacks/late startup, coalescing, closed-loop
  refusal, observer cancellation versus cleanup completion, and bounded close.
  ```python
  async def on_worker():
      progressed.set()
  task = asyncio.create_task(worker.run(on_worker))
  await asyncio.sleep(0)
  assert progressed.wait(1)  # Deliberately block UI only inside the test.
  await task
  ```
- [x] Run worker test file, record RED, then implement loop/thread startup with
  `concurrent.futures.Future`, `run_coroutine_threadsafe`, and shielded owner
  completion. Keep generation checks before/after UI calls, no blocking UI waits.
- [x] Lock supervisor bookkeeping with `threading.RLock`; preserve the existing
  max-two orphan policy and register done callbacks on their owning loop.
  No duplicated supervisor or asyncio loop-specific global quarantine.
- [x] Run worker and supervisor targeted tests; lint, self-review and commit
  only owned code. Review this task before final integration is accepted.

### Task 3: Production composition and blocked-UI acceptance

**Files:**
- Modify: `tldw_chatbook/Chat/console_speculative_voice_session.py`
- Modify: `tldw_chatbook/Chat/console_runtime.py`
- Modify: `tldw_chatbook/UI/Console_Modules/hands_free.py`
- Modify: `tldw_chatbook/Audio/duplex_transport.py`
- Modify: `tldw_chatbook/Chat/voice_phrase_sequencer.py`
- Test: `Tests/Chat/test_voice_phrase_sequencer.py`
- Modify: `Tests/UI/test_console_voice_callback_continuity.py`
- Create: `Tests/Chat/test_console_voice_worker_integration.py`
- Test: `Tests/integration/test_speculative_voice_pipeline.py`
- Test: `Tests/UI/test_console_speculative_voice_wiring.py`
- Modify: `Packaging/speculative_voice_python_paths.txt`
- Modify: `Packaging/speculative_voice_source_paths.txt`

**Interfaces:**
- Consumes Task 1's owner-loop synthesizer and Task 2's worker/facade/UI bridge.
- Production default UI factory receives the runtime's lazy worker; direct
  core construction remains usable by hardware-free unit tests. No release gate
  override is added.

- [x] Convert mounted UI-block test from expecting overflow to asserting zero
  overflow while 80 and 300 fake callbacks arrive with UI blocked. Record RED.
  Add a worker integration test where admitted speech fences the epoch, cancels
  an active fake provider, and silences stale output within 150 ms before UI
  resumes; then revised input extends the turn and post-playback input is new.
- [x] Split UI preparation of dependencies/settings from worker construction:
  ```python
  async def prepare_on_ui(**request):
      return await ui_bridge.prepare(lambda: controller.prepare_speculative_voice_attempt(**request))
  async def promote_on_ui(**winner):
      return await ui_bridge.terminal(lambda: promote_winner(**winner))
  synthesizer = OwnerLoopTtsSynthesizer(ui_loop, _LazyHandsFreeTts().synthesize_hands_free)
  ```
  Construct core queues/coordinator/transcript/DSP state only in build_core on
  the worker. Marshal projections, draft recovery, accepted handoff, and errors
  to UI; use the generation/epoch fences, not raw cross-thread widget calls.
- [x] Add synchronous thread-safe transport admission fencing for facade close
  if required; keep actual stream teardown on the worker. Preserve overflow
  latching. Move content-free voice log persistence to a bounded off-pump sink.
- [x] Add lazy ConsoleRuntime ownership and disposal: fence new entries,
  close sessions while loop is alive, retain orphan cleanup and quarantine,
  close owned gateway clients on worker, then stop/join off UI. Do not stop
  the loop underneath unfinished cleanup or an accepted promotion.
- [x] Run the exact affected worker/TTS/core/UI/supervisor/transport tests and
  source/launcher guards with asyncio debug checks for cross-loop tests. Add
  new Python files to both source inventories, sorted. No broad test sweep.
- [x] Review the entire isolation diff against the approved spec, fix concrete
  findings, rerun only covering tests, and commit only voice-related files.
  Keep task In Progress until the ordinary conversation genuinely passes.

### Task 4: Readiness handoff, not qualification

- [ ] Check exact committed worktree/import identity and visible-switch selection.
- [x] Check current route and dependencies read-only without credentials.
- [ ] Only after software and runtime readiness are healthy, prepare the single
  ordinary headset conversation with the visible switch; no physical harness,
  repetitions, switching, matrices or soaks. Any failure requires diagnosis
  before asking the user to repeat anything.

## Execution log

- User approved written design on 2026-09-04. Existing worktree verified:
  `codex/speculative-duplex-voice-adr097`, baseline `eb39436bd8`.
- Prior targeted baseline: 132 tests passed; the subsequent two mounted
  characterization cases passed. They characterize failure, not isolation.
- Implemented and independently reviewed Tasks 1–3. Git writes are serialized
  by root in one voice-only integration commit to preserve unrelated dirty work.
- Core/UI gate: 147 passed (`/private/tmp/voice-isolation-core-final`).
  Runtime/source/launcher guards: 102 passed (`/private/tmp/voice-isolation-debug-guards`).
  Final reviewed cross-loop set: 26 passed (`/private/tmp/voice-worker-review-final`).
  These targeted sets overlap; counts are not a unique test total.
- Review fixes have RED/GREEN regressions: three TTS cancellation races,
  failed callback-registration quarantine, orphan-before-finalizer shutdown,
  and the Python 3.11 executor shutdown signature. No remaining reported findings.
- Task 3 clarification: bounded phrase cancellation has a separate real cleanup
  wait, consumed by attempt supervision. No TTS cleanup receipt is inferred
  merely from the sequencer's half-second cancellation return.
- Read-only current route: Logi USB Headset input/output at 48 kHz; dependencies
  present, Parakeet resolved, 700 ms silence, AEC enabled. No audio stream opened.
- Task 4's ordinary conversation is still outstanding; rollout remains disabled.
