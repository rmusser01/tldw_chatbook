# GIL-independent voice callback implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove Python entry from the device callback without losing acoustic safety, cancellation or causal turn boundaries.

**Architecture:** One native callback owns fixed capture/render rings. Python's existing voice worker consumes timestamped records and owns AEC/VAD and conversation policy. Submission receipts and actual-output reference ordinals are distinct; playback boundaries are serialized with capture admission.

**Tech Stack:** C++17, existing pybind11/WebRTC companion, sounddevice's library-author stream seam, Python 3.11+, pytest.

**Spec:** `Docs/superpowers/specs/2026-09-04-speculative-voice-native-callback-design.md` (approved revised boundary).

ADR required: yes
ADR path: `backlog/decisions/098-low-latency-speculative-duplex-voice-pipeline.md`
Reason: Accepted native/Python runtime, cancellation and lifetime amendment.

## Global Constraints

- Work only in `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/speculative-duplex-voice-adr097`, branch `codex/speculative-duplex-voice-adr097`. Preserve unrelated trace-ledger/CSS changes; stage only owned paths.
- Retain 48 kHz mono int16, 480 samples per 10 ms callback, current actual-latency validation, current 64-frame capture/render defaults, and current AEC/timing thresholds.
- No microphone, speaker, physical qualification harness, route switching, repetitions, matrices, soaks, or full test suite during implementation. Use synthetic in-memory PCM and targeted tests only.
- No release bypass; packaged qualification remains unqualified. The source-only exact-HEAD launcher stays excluded from release artifacts.
- No Python callback entry, allocations, waiting locks or unbounded loops on the native callback. Required atomics must be lock-free. No new vendored dependency or process.
- Native cancellation precedes asynchronous provider/TTS cleanup. Keep actual committed output references and capture-time playback context. Missing context fails closed.
- Teardown observation is bounded to two seconds; uncertain native ownership is retained and blocks speculative restart across sessions.
- Main tools: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python`, `pytest`, `ruff`. Use local build isolation directories; do not replace the main venv's installed native package while iterating.
- Read the relevant task and voice entries in `backlog/docs/lessons-testing-evidence.md` and `lessons-live-verification.md`. User restrictions override historical task instructions to repeat physical evidence.

## Task 1: Native callback, rings and Python binding

**Files:**
- Create `native/voice_aec/src/duplex_bridge.h`, `duplex_bridge.cpp`, `duplex_bindings.cpp`.
- Modify `native/voice_aec/src/bindings.cpp`, `CMakeLists.txt`, `pyproject.toml`, `tldw_voice_aec/__init__.py`.
- Create `Tests/Audio/test_native_duplex_bridge.py`, `Tests/Audio/fakes/native_duplex_driver.cpp` and `Tests/Audio/fakes/native_duplex_helpers.py`.

**Interfaces:** Export `NativeDuplexBridge` and integer `DUPLEX_ABI_VERSION = 1` from the companion. Constructor takes keyword-only `capture_capacity=64`, `render_capacity=64`, `generation=0`; reject capacities outside 1..4096 and negative generation. Export:

```python
bridge.callback_address: int
bridge.userdata_address: int
bridge.output_epoch: int
bridge.monotonic_ns() -> int
bridge.set_render_admission(enabled: bool) -> None
bridge.queue_render(pcm16: bytes, submission_id: int, output_epoch: int) -> bool
bridge.abort_output() -> int  # advances epoch, returns new epoch
bridge.deactivate() -> None  # permanent input/output fence
bridge.pop_capture() -> dict | None
bridge.latest_render() -> dict | None
bridge.snapshot() -> dict
bridge.retain_for_process_lifetime() -> None  # irreversible ownership, not activation
```

`pop_capture` supplies `pcm16`, `output_pcm16`, `capture_sequence`, `callback_ordinal`, `generation`, `observed_ns`, `input_adc_time`, `current_time`, `output_dac_time`, `status_bits`, `fatal_status_bits`, `capture_overflows`, `invalid_frames`, `invalid_timing`, `capture_occupancy`, `render_occupancy`, `submission_id`, `output_epoch`, `reference_sequence`, `startup_discarded_before`. Absent committed render identities are `None`. Capture sequences count accepted non-priming records, not callback attempts. Dropped captures still consume sequence IDs and latch overflow. Snapshot provides occupancy, cumulative loss/invalid counters, fatal status, tolerated startup count, callback count, active/admission state and epoch independently of ring draining. Latest render supplies generation/submission/epoch/reference identities and actual DAC time from one coherent publication; it must not be overwritten by silence. Include enough output-timing history/context per captured record to classify previously committed playback after cancellation. No Python consumer-arrival timestamp replaces callback time.

- [x] Write the focused native tests first. Initial absence of `NativeDuplexBridge` is a named failing capability assertion against the installed old companion. Then build only this companion into a fresh temporary wheel/unpack directory and run tests against that artifact, never the old installed package. The helpers must assert the loaded module path.

```python
def test_cancelled_submissions_do_not_skip_aec_reference_ordinals(native_bridge, emit):
    bridge = native_bridge(capture_capacity=64, render_capacity=64, generation=7)
    bridge.set_render_admission(True)
    assert bridge.queue_render(b"\x01\x00" * 480, 0, 0)
    emit(bridge)
    first = bridge.pop_capture()
    assert (first["submission_id"], first["reference_sequence"]) == (0, 0)
    assert bridge.queue_render(bytes(960), 1, 0)
    assert bridge.queue_render(bytes(960), 2, 0)
    epoch = bridge.abort_output()
    assert bridge.queue_render(b"\x02\x00" * 480, 3, epoch)
    emit(bridge)
    second = bridge.pop_capture()
    assert (second["submission_id"], second["reference_sequence"]) == (3, 1)
    assert second["output_pcm16"] == b"\x02\x00" * 480
```

The test helper `emit` invokes the native callback pointer with actual PortAudio ABI-shaped arguments and literal device times, returning output bytes. Put all driver/test-only functions outside production classes. Also test full/empty rings, FIFO wraparound, frame length/null buffers/nonfinite times, status at 50/51, harmless startup count separate from fatal latch, overflow while the fault record is dropped, output fencing and deactivation. Check callback ownership and native progress during a deliberately held GIL using a native test driver; a Python timer is not evidence for this boundary.

- [x] Implement fixed SPSC rings, producer-owned writes and consumer-owned reads, with acquire/release publication. Control uses atomic epoch/admission/active flags and never rewrites the callback's render read cursor. Skip stale render slots with at most capacity iterations. Select then recheck the epoch at output commit; cancelled candidates emit explicit silence without receiving a reference ordinal.

```cpp
// Core invariant, implemented in the real callback (not a Python wrapper):
// reserve capture space -> observe timing/status -> select current output
// -> final fence check -> assign actual reference ordinal -> publish paired record.
// On capture loss: latch fault and fence output, never overwrite unread PCM.
```

Use exact `unsigned long` PortAudio frames/status ABI and a three-double time-info structure. No PortAudio linkage is needed in the bridge; sounddevice owns device APIs. Use a native monotonic clock exposed to Python for calibration. Shared receipt fields require race-free atomic snapshots, not a seqlock over concurrently accessed non-atomic C++ data. Any bounded snapshot retry exhaustion returns no snapshot for retry off-callback. Storage retained for uncertain close must live outside Python object destruction and remain deactivated until process exit.

- [x] Bind the new class in a dedicated binding file, invoked from the existing module initializer. Keep AEC bindings/vendor closure unchanged. Add the exact source files to CMake; bump companion version to `0.1.9.0`. Type/size validation happens before publishing native slots.
- [x] Run the focused native tests on the freshly built artifact plus relevant existing AEC binding tests. Record RED/GREEN commands, native artifact path, compiler/build results and limitations. Self-review for callback allocation, atomics/ABI, input lifetime and pointer lifetime. Commit only Task 1 paths.

## Task 2: Native transport adapter and checked lifecycle

**Files:**
- Create `tldw_chatbook/Audio/native_duplex_stream.py`, `Tests/Audio/test_native_duplex_stream.py`.
- Modify `tldw_chatbook/Audio/duplex_transport.py`, `duplex_contracts.py`, `Tests/Audio/test_duplex_transport.py`.

**Interfaces:** Consume Task 1's bridge without renaming its record fields. Expose `NativeDuplexStream` with checked `start`, `stop`, `close`, `latency`, bridge ownership and an app-root uncertainty registry. Native opening uses `_StreamBase(kind="duplex", wrap_callback=None, callback=<CData pointer>, userdata=<CData pointer>)` and never `RawStream`. Validate ABI before stream creation. Native imports remain lazy. Preserve injected fake-backend operation for deterministic nonnative tests, using the same normalized ingestion policy.

Add frozen `RenderSubmission(generation, output_epoch, submission_id)` and `RenderBoundary(generation, output_epoch, submission_id, ended_ns)` contracts. `queue_render` returns a submission (no synthetic playback-completion timestamp). `DuplexAudioTransport.render_boundary(submission) -> RenderBoundary | None` is nonblocking and validates generation/epoch/fatal faults. Add `AudioFrame.assistant_rendering: bool | None = None` for historical playback context; native ingestion always supplies a proven value or explicitly fails closed on ambiguity. Capture and reference ordinals remain independent.

- [x] Write failing adapter tests capturing the actual callback/userdata arguments at the PortAudio boundary, not merely asserting a fake class exists. Verify the pointer runs without Python. Inject checked stop/close results and a blocked native operation, and prove reference retention/new-session quarantine.

```python
async def test_cancelled_close_observer_keeps_native_owner(close_fixture):
    owner, observer = close_fixture.block_close()
    observer.cancel()
    await close_fixture.consume_cancel(observer)
    assert close_fixture.native_userdata_alive(owner)
    assert close_fixture.new_session_rejected()
    close_fixture.complete_close_successfully()
    await close_fixture.observe_owner_settled()
    assert not close_fixture.native_userdata_alive(owner)
```

`close_fixture` is a test utility with checked native-result controls, weak references and observable adapter startup behavior; it is not a production helper. Test a close error after the wrapper pointer becomes null, never-returning stop/close, late callbacks, expiry at two seconds, no concurrent stop/close/terminate, and no automatic reopen after late completion. Use fake time where possible; no repeated timing runs.

- [x] Implement one retained daemon native-operation owner, a protected receipt and a two-second observation deadline. On uncertainty, deactivate/retain native state and preserve app-root quarantine across transport instances. Do not use the asyncio default executor for potentially permanent native calls. Native close must be checked, not inferred from wrapper state; late results cannot touch a closed loop.
- [x] Replace the default production callback with native-record draining. Do not add another capture/render staging queue. Calibrate native monotonic samples against Python by up to eight bracketed samples, selecting the minimum-span sample; require span <=1 ms before stream start. Use its midpoint offset thereafter, including actual DAC boundary translation. Keep existing device cadence/drift checks and metadata.

```python
observed_ns = record["observed_ns"] + native_to_python_offset_ns
# _map_timing_locked receives observed_ns explicitly; never sample drain arrival.
```

Derive playback context using recorded actual DAC intervals and native output identity, not the current lifecycle flag. Explicit idle is distinct from missing evidence. Expose all actual ring occupancies/counters; overlay sticky native faults even when the detecting record was dropped. Keep harmless priming separate. Reset evidence only for a new fully owned generation.
- [x] Extend render cancellation/fencing to reach the native owner synchronously, keep submission IDs monotonic, and map actual reference IDs into the unchanged AEC continuity contract. Provide retained single latest-completion state so a final target can resolve after its capture record was already processed. Route reset or aborted target rejects its receipt.
- [x] Run targeted stream/transport/AEC integration tests, including native emitted submission IDs 0/3 becoming reference IDs 0/1 through the real preprocessor. Preserve import isolation and fake-backend tests. Self-review and commit Task 2 paths.

## Task 3: Causal playback completion and historical admission

**Files:**
- Modify `tldw_chatbook/Chat/voice_phrase_sequencer.py`, `console_speculative_voice_session.py`, `console_speculative_voice.py`.
- Modify `Tests/Chat/test_voice_phrase_sequencer.py`, `test_console_speculative_voice.py`, `test_console_speculative_voice_session.py`.

**Interfaces:** Consume `RenderSubmission`/`RenderBoundary` and nonblocking transport `render_boundary`. Add `AttemptPlaybackBoundaryKnown(attempt_epoch, render_boundary_ns)` distinct from `AttemptPlaybackTerminal`. The coordinator retains one known boundary for the active attempt, including when generation completion has not arrived. `PhraseSpeechSequencer` exposes its final submission once input is closed and synthesis/queueing is finished; the session's voice owner observes it before capture admission. Do not create per-frame futures.

- [x] Write failing cancellation/history and scheduling-order tests with the real coordinator/preprocessor. Native playback captured before cancellation remains `assistant_rendering=True` during later drain, even after the lifecycle flag is false. Unhealthy AEC must not admit it via idle/raw STT. New proven idle capture remains eligible.

```python
# Both orders must be covered by controlled scheduling, not wall-clock chance:
# native final render -> post-boundary capture -> terminal waiter scheduled
# native final render -> terminal waiter scheduled -> post-boundary capture
# Either way: post-boundary speech belongs to a new turn, no obsolete restart.
```

Also test pre-boundary backlog arriving after DAC time, generation completion after boundary-known, receipt after cancellation, reset during wait, callback cessation, and output epochs interleaving. Derive expected timestamps/IDs literally.
- [x] Change sequencer receipts from estimated `ended_ns` to final submission identity. Arm terminal once, and derive its fixed delivery deadline from remaining startup allowance + maximum queued render duration + validated output latency + 500 ms. In the voice-owner pump, observe an armed target and install the known boundary before admitting later captures. Recheck arming before each admission; do not let an independent waiter decide ordering.
- [x] Teach `_on_speech` to route against a known actual boundary independently of `_terminal` sealing state. Speech at/before the boundary still cancels the same attempt; later speech buffers for the next turn. A known boundary cannot promote; final DAC elapsed, generation complete and causal capture/classification/STT seals remain required. Preserve the existing DAC-boundary+500 ms seal deadline. Fail delivery uncertainty instead of remaining indefinitely speaking.
- [x] Pass frame-recorded playback context into preprocessing and `AdmittedSpeechFrame`. Remove the current-state AND for native records. Preserve conservative fallback for legacy fake frames without weakening actual production evidence. Generation/epoch fencing invalidates target, known marker, receipt and timer together; late callbacks cannot resurrect them.
- [x] Run affected coordinator/session/sequencer tests and targeted worker/TTS cleanup tests, self-review and commit Task 3 paths. Do not edit unrelated trace files even if nearby tests use them.

## Task 4: Production proof, packaging identity and handoff

**Files:**
- Create `Tests/UI/test_console_voice_native_callback.py`.
- Modify `Tests/UI/test_console_voice_callback_continuity.py` only as needed for receipt interfaces.
- Modify `Packaging/speculative_voice_source_paths.txt`, `speculative_voice_python_paths.txt`, `Tests/Packaging/test_voice_source_digest.py`, `test_voice_aec_installed_wheel.py`, and launcher tests if the native capability check requires it.
- Update this plan, governing spec/task notes and native distribution documentation as needed; never edit the packaged qualification manifest to claim qualification.

**Interfaces:** Consume all prior tasks via the visible production control. Use a freshly built local native wheel, real AEC, simulated native device callbacks and isolated test config. Final native artifact path/hash/version and commit are recorded separately from release qualification.

- [x] Add a mounted regression that clicks `#console-hands-free-switch`, asserts `ConsoleSpeculativeHandsFreeSession`, and verifies actual native callback-pointer selection. Apply one controlled GC pause within capture capacity while a native driver emits frames. Assert native progress during GIL hold, no frame/reference loss or false clock drift, and truthful callback timestamps after drain. Exercise interruption and post-playback turn routing using synthetic audio/events.
- [x] Add the exact new sources/tests/spec/plan to source identity and Python lint scope. Verify built wheel exports the callback ABI and original AEC, source-only launcher remains excluded, and missing/old native ABI cannot silently fall back to a Python device callback.

```python
assert native_module.DUPLEX_ABI_VERSION == 1
assert callable(native_module.NativeDuplexBridge)
assert callable(native_module.AecProcessor)
# Runtime tests must also exercise these capabilities, not just their names.
```

- [x] Run only the affected native, transport, sequencer, session, worker/TTS cleanup, mounted UI and packaging-source tests. Run Ruff on changed Python files and `git diff --check`. Compare failures with recorded baseline; do not launch a full suite.
- [x] Perform independent task review, then whole-change review against the initial native-boundary base `ee2a06c34a0eb8d2adde6fcbb236b3359cdd6654`. The whole-change review returned four findings; its execution is complete, not a clean-review claim.
- [ ] Complete the scoped re-review of the consolidated final integration fixes before final HEAD/runtime preflight.
- [ ] Commit only voice changes, record final HEAD and built artifact provenance. Keep TASK-23175 In Progress: an ordinary conversation is not release qualification and historic matrices remain outstanding/out of scope.
- [ ] Before any physical conversation, verify source-only launcher final-HEAD binding, native module provenance, current route and runtime metadata read-only. Installation of the changed local wheel into the runtime venv is a distinct artifact update, not permission to touch other dependencies. Prove visible-control selection programmatically first. Only then offer the user's one ordinary USB conversation. Stop at a failure and inspect evidence before requesting more speech; no harness/repeats/soaks.

Task 4 integration: root project/companion pin and `tldw_chatbook.__version__`
are synchronized to 0.1.9.0. Narrow UI changes expose categorical native startup
and shutdown failures, suppressing obsolete-generation notifications. Synthetic
mounted turn-routing uses default preroll and actual speech onset; original PCM
and timestamps remain STT context. The real native/AEC proof is a separate
eight-frame paced batch during one 80 ms GIL hold at full-GC entry, followed by
actual collection and drain. It does not establish physical acoustic performance.
Verification: 642 affected tests and one built wheel/sdist contract test passed;
known dependency/deprecation warnings remain. No native rebuild/install, release
qualification, device I/O or full suite was performed. Task 4 report/provenance is
in `.superpowers/sdd/2026-09-04-speculative-voice-native-callback/task-4-report.md`.
ADR required: no new ADR. ADR path: `backlog/decisions/098-low-latency-speculative-duplex-voice-pipeline.md`.
Reason: directly implements the approved amendment and its onset/ownership rulings.

## Consolidated final review fixes

The four findings at `4c11ef5901` are bounded repairs of the approved contract:
synchronous native attempt fencing before provider/TTS cleanup, native capture
coverage before half-duplex terminal sealing, permanent session deactivation and
independent checked-close observation before unrelated cleanup, and public version
tuple consistency. Regressions precede each repair. Existing targeted native,
session, worker, mounted-control and version/source inventory checks cover the
affected seams. No native sources or artifact change is needed.

ADR required: no new ADR.
ADR path: `backlog/decisions/098-low-latency-speculative-duplex-voice-pipeline.md`.
Reason: restores its accepted cancellation, capture-seal and native lifetime rules.
The final-fix report records exact RED/GREEN evidence. Scoped re-review, final
committed HEAD/runtime preflight and the ordinary conversation remain pending;
TASK-23175 stays In Progress and qualification authority remains unchanged.
