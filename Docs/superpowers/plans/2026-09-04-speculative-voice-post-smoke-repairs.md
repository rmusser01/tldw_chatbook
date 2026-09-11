# Speculative Voice Post-smoke Repairs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair reproduced phrase-tail DSP evidence and shared Parakeet ownership defects, then add content-free diagnostics for the unresolved latency and host overflow.

**Architecture:** Reuse actual paired output PCM from native ABI 1 for a complete DSP timeline, independently of TTS submission receipts. Protect the existing model process with one session-owned async lease whose lifetime includes checked context cleanup. Observe first-stage timings and fault snapshots through the existing bounded diagnostic bridge; never relax acoustic admission.

**Tech Stack:** Python 3.12/asyncio, existing Parakeet spawn worker, WebRTC AEC/native callback ABI 1, pytest, simulated PortAudio and deterministic fake models.

---

## Authority, environment and evidence limits

- Worktree: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/speculative-duplex-voice-adr097`.
- Branch: `codex/speculative-duplex-voice-adr097`.
- Approved spec: `Docs/superpowers/specs/2026-09-04-speculative-voice-post-smoke-repairs-design.md` (initial commit `23a6303bd865d97942ede474aca2b0accd2f9f91`; user approved written spec).
- Runtime reproduction base: `7c4cc693f36db3d1ddebecb70d9b1435a502089a`.
- Existing task: `backlog/tasks/task-23175 - Add-low-latency-speculative-duplex-voice-pipeline.md`, In Progress. This plan is its focused repair slice, not a new feature or qualification plan.
- ADR required: **yes**, clarification of existing contracts.
- ADR path: `backlog/decisions/098-low-latency-speculative-duplex-voice-pipeline.md`, accepted post-smoke clarification.
- Reason: Shared DSP identity and model lifetime cross module boundaries. Reuse ADR-098; unrelated ADR-097 must not change.

No physical device open, app launch, live conversation, provider call, actual model
load/download, native rebuild/install, qualification command, route change, matrix,
soak or full suite. The one ordinary conversation has already occurred. User offers
of live feedback are conditional, not permission to repeat it. Tests below replace
PortAudio before opening and substitute a spawned fake model before model startup.
The installed native extension can be loaded and driven by fake PortAudio.

Use @superpowers:test-driven-development during implementation and
@superpowers:verification-before-completion before reporting outcomes. Consult
the incident-backed voice entries at the start of `backlog/docs/lessons-testing-evidence.md`
and `backlog/docs/lessons-live-verification.md`; the user's no-hardware restriction
overrides those documents' generic live-verification advice.

Run from the feature worktree. Set only these task-specific shell variables:

```bash
voice_python=/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python
voice_ruff=/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/ruff
export TLDW_NATIVE_DUPLEX_ROOT=/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/lib/python3.12/site-packages
git branch --show-current
git rev-parse HEAD
git status --short
```

Re-establish variables in each fresh shell. Use a fresh `--basetemp` created by
`mktemp -d /private/tmp/tldw-voice-repairs.XXXXXX` for test runs; never reuse a user
directory as pytest's disposable target. Commands below omit optional basetemp.
Any native prerequisite skip is an unmet gate, not success. If the installed ABI
is unavailable, stop that gate and report it; do not install/rebuild automatically.

Unrelated dirty trace-ledger tests/runtime, ADR-097, task-23113.5 and CSS are owned
by the user/other agents. Preserve them. Explicitly stage only each task's listed
edits; inspect the index before every commit and exclude unexpected staged work.
Do not use `git add .`, amend unrelated commits, prune Git or reset the worktree.

## File map and task order

Tasks execute **sequentially** because they share the session composition root.
Ownership in each task means only the described responsibility, not permission to
rewrite the rest of that file. No general refactor or new framework is needed.

| Task | Runtime ownership | Test ownership |
| --- | --- | --- |
| 1: DSP timeline | `tldw_chatbook/Audio/duplex_transport.py`, `duplex_contracts.py`, `acoustic_isolation.py`; narrow `voice_preprocessor.py` consumer migration if required | Existing Audio transport/native/isolation/preprocessor suites and integration reference tests |
| 2: STT lifetime | `tldw_chatbook/Chat/console_speculative_voice_session.py` STT adapters/worker; `tldw_chatbook/Audio/parakeet_voice_worker.py` uncertain-context shutdown | New focused `Tests/Chat/test_console_voice_stt_ownership.py`, existing worker and pipeline integration tests |
| 3: Fault diagnostics | Same session's existing fault hook; Parakeet typed fixed reasons; `tldw_chatbook/Utils/persistent_diagnostics.py` narrow integer fields | New `Tests/Chat/test_console_voice_diagnostics.py`, existing persistent-boundary and worker tests |
| 4: First-stage timing | Session attempt lifecycle/pump, `tldw_chatbook/Chat/voice_phrase_sequencer.py`, optional paired commit metadata in Audio contracts/transport | Diagnostics, sequencer, session and native transport tests |
| 5: Joined verification/docs | Source/lint inventories, existing task and repair documents | Explicit joined file list below; no broad suite |

For rows listing Audio filenames without prefixes, the full prefix is
`tldw_chatbook/Audio/`. New test modules belong in both existing inventories when
created: `Packaging/speculative_voice_source_paths.txt` and
`Packaging/speculative_voice_python_paths.txt`. Add the approved spec/plan and
newly affected persistent-diagnostic source/tests to source identity as appropriate.
Never regenerate historical reports or the packaged unqualified manifest.

## Task 1: Complete paired DSP evidence across silent callbacks

**Modify:** `tldw_chatbook/Audio/duplex_transport.py::_ingest_record_locked`,
`_ingest_native_locked`, `_audio_callback`; `tldw_chatbook/Audio/duplex_contracts.py::AudioFrame`
documentation; `tldw_chatbook/Audio/acoustic_isolation.py::observe`.
**Inspect/migrate only as required:** `tldw_chatbook/Audio/voice_preprocessor.py`.
**Tests:** `Tests/Audio/test_native_duplex_stream.py`,
`Tests/Audio/test_duplex_transport.py`, `Tests/Audio/test_acoustic_isolation.py`,
`Tests/Audio/test_voice_preprocessor.py`, `Tests/integration/test_speculative_voice_pipeline.py`.

- [x] Write `test_native_phrase_tail_keeps_complete_dsp_references` in the existing
  native suite using its `native_transport`, `driver`, `emit_callback` fixtures.
  Run six callbacks with ADC times `10 + i / 100`, current times `10.01 + i / 100`,
  DAC times `10.02 + i / 100`. Queue four audible blocks, then queue nothing.
  Use the real `AcousticIsolationMonitor`, existing deterministic `pcm16(RENDER)`
  and `pcm16(QUIET)` signals, and drain each capture's references. Core oracle:

  ```python
  assert [frame.render_reference_sequence for frame in captures] == list(range(6))
  assert [frame.sequence for frame in references] == list(range(6))
  assert [frame.pcm16 for frame in references[4:]] == [bytes(960), bytes(960)]
  assert all(frame.assistant_rendering for frame in captures[4:])
  assert not any(reason in {"render-reference-gap", "missing-render-reference"}
                 for reason in reasons)
  assert transport.render_boundary(last_submission) == last_actual_boundary
  assert transport.native_counters["capture_overflows"] == 0
  ```

  Do not import the ignored diagnostic tests: those deliberately assert bugs.
  Copy only the reproduction setup into permanent tests with corrected oracles.
- [x] Run RED:
  `"$voice_python" -m pytest -q Tests/Audio/test_native_duplex_stream.py::test_native_phrase_tail_keeps_complete_dsp_references`.
  Expect failure at missing/repeated reference ticks, not fixture/import failure.
- [x] Separate common ingest into two branches: unconditional paired DSP frame
  from actual output bytes with `sequence=capture_sequence`; submission-only
  playback interval and `RenderBoundary` update. Set capture's required reference
  to the paired DSP tick. Remove the obsolete native committed-render-ordinal
  parameter from this projection if it has no remaining consumer; retain native
  record/ABI fields unchanged. Reference-ring capacity is checked on **every**
  accepted callback, including idle injected-backend callbacks. Preserve fault
  latches and original sequence gaps, not a new synthetic contiguous counter.
- [x] Move valid render-history append/pruning in `AcousticIsolationMonitor.observe`
  ahead of the idle return. Preserve render continuity across idle; reset only
  incomplete acoustic/near-end windows as before. Route/fault reset still clears
  proof. Idle zeros do not increment clean windows or clear a real safety fault.
  Retain existing energy, causal/correlation, saturation and 50 ms admission rules.
- [x] Add onset, idle-gap, cancellation-silence and resumed-phrase cases for both
  native and injected transport. Assert cancelled queued blocks have no actual
  receipts; already emitted DAC tails remain playback-period. Add real loss/status,
  missing reference/timing, stale generation and silent-reference overflow cases;
  they must stay closed. Migrate expectations that equated “no TTS” with “no DSP
  reference.” Do not weaken a production assertion to make a fake pass.
- [x] Add one joined fake-PortAudio test using `VoicePreprocessor.from_native` and
  the real monitor across onset/tail/gap. Direct monitor tests isolate the sequence
  oracle; this joined test verifies actual AEC consumes zeros once and in order.
  Acoustic ambiguity may still demote; it must not become raw playback STT or false
  missing data. Preserve existing render-only leakage and near-end confirmation tests.
- [x] Run GREEN:
  `"$voice_python" -m pytest -q Tests/Audio/test_native_duplex_stream.py Tests/Audio/test_duplex_transport.py Tests/Audio/test_acoustic_isolation.py Tests/Audio/test_voice_preprocessor.py Tests/integration/test_speculative_voice_pipeline.py`.
  Expect all selected tests pass, native coverage executed, no hardware opened.
- [x] Lint/format only changed Python paths, inspect `git diff --check`, and commit
  explicit owned paths: `fix: preserve paired DSP references through render silence`.

## Task 2: Bound native quiet waits and shared-model ownership

**Modify:** `tldw_chatbook/Chat/console_speculative_voice_session.py::_SerialSttWorker`,
`_NativeStreamingStt`, `_RollingWindowStt`, `_prepare_streaming_candidate`,
`build_core`; `tldw_chatbook/Audio/parakeet_voice_worker.py` failed context lifecycle.
**Execution clarification:** The existing failure-to-fallback decision in
`tldw_chatbook/Audio/rolling_transcript.py` may gain a narrow typed terminal-failure
guard so a fenced service is not advertised as rolling fallback. Ordinary
recoverable fallback and the active owner's service after a waiting timeout remain
unchanged. This directly enforces the approved spec; test it in the new module.
**Create:** `Tests/Chat/test_console_voice_stt_ownership.py`.
**Existing tests:** `Tests/Audio/test_parakeet_voice_worker.py`,
`Tests/integration/test_speculative_voice_pipeline.py`,
`Tests/Chat/test_console_speculative_voice_session.py`.

- [x] Add a fake context with entry/push/exit events. Submit exactly 400 ten-ms
  frames to the production native adapter with a short injected quiet interval.
  Wait for push and then **exit**, not an arbitrary sleep. Assert accepted coverage
  reaches the final frame; resumed speech opens a new context and retains the prior
  transcript prefix with each PCM byte pushed once. Include 399 and 401 boundaries.
- [x] Run RED:
  `"$voice_python" -m pytest -q Tests/Chat/test_console_voice_stt_ownership.py -k exact_chunk`.
  Expect exit-event timeout for 400 frames on old code.
- [x] Replace the post-full-chunk unbounded `get()` with the same quiet timeout
  used inside chunk collection. On expiry, finalize the **cached last hypothesis**
  and coverage if needed, exit the context, retain prefix, then await fresh audio
  outside the lease. Do not push empty PCM or reprocess the last chunk to mark final.
  A received sentinel is acknowledged once; a frame won at the timeout boundary
  is owned/processed once, never lost. Test final revision and settlement counts.
- [x] Add two native adapters sharing one real `ParakeetVoiceProcess` and serial
  worker; use `Tests.Audio.test_parakeet_voice_worker.fake_runtime` extended with
  an explicit model-mode invariant. Add a rolling request waiting behind native
  ownership. Assert native/rolling both eventually succeed, at most one active
  context, restored mode before rolling and no `stream_active` failures. Preserve
  direct worker tests that intentionally reject an illicit overlapping RPC.
- [x] Run RED:
  `"$voice_python" -m pytest -q Tests/Chat/test_console_voice_stt_ownership.py -k shared_model`.
  Expect the existing native/rolling context collision before the repair.
- [x] Implement a single `asyncio.Lock` on the existing serial STT worker and a
  small context-manager lease for Parakeet. Acquisition uses a 30-second timeout
  constant, not a user setting. Native obtains it after first audio, then owns
  context creation, entry, all burst pushes and checked exit. Prewarm owns its
  entire context. Rolling receives the same worker from `build_core`, owns its
  whole RPC, and runs on that executor. Other providers retain their old paths.
  Do not serialize only `open` or only `push`.
- [x] Implement these exact ownership transitions with existing bounded cleanup:

  | Edge | Required ownership result |
  | --- | --- |
  | Cancel/timeout while waiting | No acquired lease, no model close, no PCM work |
  | Cancel after lease but before RPC | Release after confirming no RPC/context exists |
  | Cancel while entering | Observe the retained executor operation; successful late entry must receive checked exit |
  | Cancel while pushing | Do not cancel the underlying completion receipt; drain it and exit before release |
  | Cancel while exiting | Retain checked exit completion independently of the cancelled adapter waiter |
  | Entry/exit failure with uncertain model state | Mark service unusable, request bounded reap, admit no fallback; release only after confirmed death |
  | Wait expiry at 30 seconds | Fixed ownership-timeout category; active owner unaffected |
  | Session shutdown | Synchronous audio fence/deadline first; service shutdown can interrupt blocked RPC without acquiring STT lease |

  Retain protected executor futures/cleanup tasks in the worker; `asyncio.shield`
  alone without retaining and observing the owner is insufficient. Repeated caller
  cancellation must not discard cleanup. Reuse current RPC timeouts and process
  close/kill bounds; do not invent an unbounded wait for cleanup. A failed reap
  leaves the service fenced rather than admitting a next owner.
- [x] Ensure child `exit` failure cannot clear context bookkeeping and then allow
  rolling on a possibly still-mutated model. Entry failure can also leave model
  state uncertain. Send a categorical failure and retire the service/process at
  that owner boundary. A successful push failure cleanup may allow safe fallback;
  an uncertain/dead service may not. Parent `.closed` is a fence, not proof of death.
- [x] Add event-controlled cancellation cases for each table row, plus repeated
  cancellation, waiter cancellation followed by a healthy third owner, prewarm
  failure, closed-service fallback and shutdown with blocked model cleanup.
  Use a monkeypatched short acquisition timeout for timeout tests. Use real spawned
  fake RPC for mode restoration/dead process and thread events for exact races.
  Always close/reap fixtures in `finally`; no actual MLX import/model.
- [x] Run GREEN:
  `"$voice_python" -m pytest -q Tests/Chat/test_console_voice_stt_ownership.py Tests/Audio/test_parakeet_voice_worker.py Tests/Audio/test_rolling_transcript.py Tests/integration/test_speculative_voice_pipeline.py Tests/Chat/test_console_speculative_voice_session.py`.
- [x] Add the new test to both inventories; lint/format owned paths and inspect
  diff. Commit `fix: retain Parakeet ownership through checked context cleanup`.

## Task 3: Record bounded, content-free fault evidence

**Modify:** `tldw_chatbook/Audio/parakeet_voice_worker.py` app-owned reasons;
`tldw_chatbook/Chat/console_speculative_voice_session.py` error/fault projections;
`tldw_chatbook/Utils/persistent_diagnostics.py` narrow integer allowlist.
**Create:** `Tests/Chat/test_console_voice_diagnostics.py`.
**Existing tests:** `Tests/test_persistent_diagnostic_boundary.py`,
`Tests/test_persistent_diagnostic_sentinel_matrix.py`,
`Tests/Audio/test_parakeet_voice_worker.py`.

**Execution clarifications:** Preserve the existing STT shutdown contract with
bounded pipe-poll slices when concurrent close does not wake a long poll; retain
the same operation deadlines and checked-reap ownership. Explicitly tag child-owned
failure reasons rather than matching provider exception messages. Observe native
faults that produce no capture record at owner drain too, without inventing lag or
fault-instant timestamps. These are narrow enforcement of the approved contracts,
not new timeout, callback or recovery policies.

- [x] Write a session fault test that drives the real native fake-PortAudio path
  and inspects the final persistent record, not a mocked `_persist_voice_event`.
  Force host status with no ring loss, then ring loss without host status. Assert
  one fault snapshot per generation, sticky original cause after secondary errors,
  truthful separate counters and no snapshot implying a fault-instant observation.
- [x] Run RED:
  `"$voice_python" -m pytest -q Tests/Chat/test_console_voice_diagnostics.py -k fault_snapshot`.
  Expect missing diagnostic fields on old code.
- [x] At the existing first `audio_transport_fault` hook, project only these native
  snapshot keys: `callback_count`, `fatal_status_bits`, `capture_overflows`,
  `invalid_frames`, `invalid_timing`, `capture_occupancy`, `render_occupancy`.
  Use explicit corresponding integer log fields prefixed `native_`; do not expand
  arbitrary snapshot dictionaries into the logger. Missing native data is omitted.
  Keep `phase="capture"` and add fixed `operation="drain_snapshot"`.
- [x] Compute `lag_ms` from owner monotonic now minus the paired evidence's mapped
  native `observed_ns`, when present/valid. This is callback-to-consumer lag for
  that drained record, **not** callback cadence or host scheduling delay. Persist
  available evidence only; do not add a native timer or claim a missing measurement
  is zero. Validate native counters as non-bool integers in `[0, 2**64 - 1]`, with
  occupancy additionally bounded by declared capacities; reject invalid values
  rather than globally changing legacy integer serialization.
- [x] Add a typed app-owned Parakeet failure with fixed categories `context_busy`,
  `identity_mismatch`, `closed`, `disconnected`, `timeout`, `ownership_timeout`,
  `unknown_native`. Map child app-owned branches explicitly; unknown runtime
  exceptions carry only class/unknown information. Never recover category by
  comparing arbitrary provider messages. Retain compatibility with existing
  RuntimeError/TimeoutError catches where callers rely on them.
- [x] Add final-record privacy tests using hostile exception messages, transcript,
  PCM-like strings and secret sentinels. A fake provider exception named like an
  app-owned reason must remain unknown. Fixed reason survives native and rolling
  failure projection. At each adapter/fallback boundary, report a given failure
  category once per turn/adapter lifecycle; keep state bounded by the already
  bounded live transcripts. A new successful lifecycle may report a later failure.
- [x] Verify the existing voice-to-UI diagnostic bridge (64-item discardable
  queue) remains the only owner-loop handoff and no callback logs. New field
  allowlisting must actually reach the durable sink despite the best-effort
  exception suppression in `_persist_voice_event`.
- [x] Run GREEN:
  `"$voice_python" -m pytest -q Tests/Chat/test_console_voice_diagnostics.py Tests/Audio/test_parakeet_voice_worker.py Tests/test_persistent_diagnostic_boundary.py Tests/test_persistent_diagnostic_sentinel_matrix.py`.
- [x] Update inventories for new/affected diagnostic files; lint/format owned paths,
  inspect diff, commit `feat: add bounded voice fault diagnostics`.

## Task 4: Distinguish response stages from actual output commitment

**Modify:** `tldw_chatbook/Chat/console_speculative_voice_session.py::_AttemptLifecycle`,
`SpeculativeVoiceAttemptEffects`, capture pump;
`tldw_chatbook/Chat/voice_phrase_sequencer.py::_enqueue_phrase`, `_drain_phrases`,
`_queue_render`; `tldw_chatbook/Audio/duplex_contracts.py::AudioFrame` and
`tldw_chatbook/Audio/duplex_transport.py::_ingest_record_locked` for optional paired
commit metadata only.
**Tests:** `Tests/Chat/test_console_voice_diagnostics.py`,
`Tests/Chat/test_voice_phrase_sequencer.py`,
`Tests/Chat/test_console_speculative_voice_session.py`,
`Tests/Audio/test_native_duplex_stream.py`.

- [x] Add fake-clock tests for a slow first provider delta, punctuation/phrase
  buffering, synthesis delay and delayed DAC commitment. Before a simulated
  native callback, queue acceptance must not produce an actual-render event.
  Multiple deltas/phrases produce each first-stage event once. Cancellation and
  a replacement attempt must not mix measurements or accept late successes.
- [x] Run RED:
  `"$voice_python" -m pytest -q Tests/Chat/test_console_voice_diagnostics.py -k first_stage`.
  Expect absent timing events, not changed silence behavior.
- [x] Store a monotonic attempt-start timestamp and a four-stage seen set on the
  existing bounded `_AttemptLifecycle`. Use the effects' `_on_delta` boundary
  **before** the async event queue for the first non-empty provider delta; the
  provider already filters invalidated epochs. Add a small fixed-stage sequencer
  callback for first accepted eligible phrase and first completed synthesis stream.
  Call back to effects for epoch validation and the existing diagnostic bridge.
  No need to modify `console_voice_attempts.py` just to duplicate its delta hook.
- [x] Use one fixed event `voice_first_stage` with allowlisted `phase` values
  `provider_delta`, `eligible_phrase`, `synthesis_complete`, `render_receipt` and
  nonnegative `duration_ms` since attempt start. Keep stage clocks on the voice
  owner loop. The synthesis stage means the first phrase response's byte stream
  and resource cleanup completed successfully; for lazy streams this includes
  flow control, and first render may precede synthesis completion. Do not describe
  that interval as pure model inference or enforce a false linear stage order.
- [x] Preserve the existing `playback_started` queue-admission event semantics.
  For actual output, carry an optional `committed_render: RenderBoundary | None`
  on the paired `AudioFrame`, default `None` for existing fakes. Set it only from
  actual submitted output in common ingest. This reuses already returned native
  per-record submission metadata; no native ABI/ring or receipt journal changes.
  Silent unsubmitted records have no commit even though they have DSP references.
- [x] Retain the first queued `RenderSubmission` on each sequencer/lifecycle.
  During capture draining, match that exact tuple against paired commit metadata;
  emit the first render stage only for current, valid, nonfaulted evidence. Include
  `latency_ms` for mapped DAC-start minus attempt start if representable, while
  `duration_ms` remains receipt-observation time. Use the paired reference's start
  or boundary end minus the existing 10 ms frame duration. Never mix wall clocks.
  Draining several callbacks at once still observes the first exact receipt;
  do not poll `render_boundary(first_submission)` alone, since its latest-only
  snapshot can already have advanced to a later submission. Missing/lost receipts
  remain unavailable; do not infer a first receipt from a later one.
- [x] Test delayed consumer drain, many queued blocks, streaming response cleanup,
  zero-byte/failed synthesis, stale epoch/generation, missing first record and
  cancellation immediately before late receipt delivery. Verify no user content
  or new turn/content identifiers appear in any final log record. Durations and
  absence are asserted with controlled clocks, not live timing tolerances.
- [x] Run GREEN:
  `"$voice_python" -m pytest -q Tests/Chat/test_console_voice_diagnostics.py Tests/Chat/test_voice_phrase_sequencer.py Tests/Chat/test_console_speculative_voice_session.py Tests/Audio/test_native_duplex_stream.py`.
- [x] Lint/format explicit paths, inspect diff, commit
  `feat: distinguish voice response stages and native render receipts`.

## Task 5: Joined software acceptance and honest handoff

**Modify:** `Packaging/speculative_voice_source_paths.txt`,
`Packaging/speculative_voice_python_paths.txt`, this plan, approved spec status only
if required, and `backlog/tasks/task-23175 - Add-low-latency-speculative-duplex-voice-pipeline.md`
implementation notes. ADR-098 is already clarified. Add a concise incident-backed
lesson only if implementation surfaces genuinely reusable new evidence.

**Joined-gate correction:** The existing positive fixture in
`Tests/Chat/test_console_voice_settings.py` still used app/companion `0.1.8.0`,
although the earlier native-boundary work already moved the app to `0.1.9.0`.
The joined gate (696 passed, one failure) and isolated positive node reproduced the
stale-fixture rejection. Update synthetic matching identities to the current app
version and explicitly retain stale-version rejection coverage. Do not change
production authority checks or packaged qualification data. No new ADR is required
for this test-only correction.

- [x] Review the full repair diff against the approved spec, especially raw STT
  admission, old/new attempt fences, actual terminal receipts, context cleanup,
  bounded queues, default 700 ms and unchanged AEC thresholds. Verify every new
  source/test path appears in the appropriate source/lint inventory. Do not change
  historical artifact/source hashes or promote the packaged qualification flag.
- [x] Run the explicit joined regression gate (software only):

  ```bash
  "$voice_python" -m pytest -q \
    Tests/Audio/test_duplex_contracts.py \
    Tests/Audio/test_duplex_transport.py \
    Tests/Audio/test_native_duplex_stream.py \
    Tests/Audio/test_acoustic_isolation.py \
    Tests/Audio/test_voice_preprocessor.py \
    Tests/Audio/test_parakeet_voice_worker.py \
    Tests/Audio/test_rolling_transcript.py \
    Tests/Chat/test_console_voice_stt_ownership.py \
    Tests/Chat/test_console_voice_diagnostics.py \
    Tests/Chat/test_console_speculative_voice_session.py \
    Tests/Chat/test_voice_phrase_sequencer.py \
    Tests/Chat/test_console_voice_attempts.py \
    Tests/Chat/test_console_voice_worker_integration.py \
    Tests/Chat/test_console_voice_settings.py \
    Tests/integration/test_speculative_voice_pipeline.py \
    Tests/UI/test_console_voice_native_callback.py \
    Tests/UI/test_console_voice_callback_continuity.py \
    Tests/test_persistent_diagnostic_boundary.py \
    Tests/test_persistent_diagnostic_sentinel_matrix.py \
    Tests/Packaging/test_voice_source_digest.py \
    Tests/Packaging/test_speculative_voice_lint_scope.py \
    Tests/Packaging/test_speculative_voice_development_entry.py
  ```

  Expect all selected tests pass; record actual counts, duration and warnings,
  never pre-fill them. The mounted visible Hands-free control must select
  `ConsoleSpeculativeHandsFreeSession` with real native/AEC and **fake PortAudio**.
  Confirm the existing blocked-UI, same-turn revision, post-completion next-turn,
  native shutdown and default-preroll regression nodes actually execute. No soak
  or physical-runner command is part of this gate.
- [x] Run `"$voice_python" Packaging/check_voice_aec_version_sync.py` (expect exit
  zero and unchanged 0.1.9.0/ABI 1). Run `"$voice_ruff" check` and
  `"$voice_ruff" format --check` with the explicit changed Python file list from
  these tasks, not the dirty worktree's unrelated paths. Run `git diff --check`.
- [x] Obtain scoped final code review using @superpowers:requesting-code-review
  when executing this plan. Review only the repair commits and the relevant
  unchanged callers; no new hardware or broad test delegation. If corrections
  are needed, reproduce them first and rerun only affected gates before the joined
  result is claimed.
- [x] Add implementation notes with exact tested commit/source state, test results,
  remaining host-overflow uncertainty and evidence limits. Existing ignored
  `.superpowers/sdd/2026-09-04-speculative-voice-native-callback/ordinary-conversation-7c4cc693f3.md`
  records user-perceived interruption/revision but incomplete playback/new-turn
  proof. Do not turn that into a completed live acceptance claim.
- [x] Commit only final owned documentation/inventory changes, then read back HEAD,
  branch, index and dirty paths. Keep TASK-23175 In Progress. Report software repair
  outcomes separately from the unresolved physical failure. Do not rebind/open the
  old exact-HEAD launcher or ask the user to speak without a specific, separately
  authorized diagnostic question.

## Execution status

- [x] Written design approved by user; ADR-098 clarification recorded.
- [x] Implementation plan independently reviewed: Approved, no issues or recommendations (2026-09-04, round 1).
- [x] User selects subagent-driven execution.
- [x] Tasks 1–5 implemented and verified as the scoped software repair slice.

Execution completed at code/test HEAD `0bc6868f49f7a6fe30e04d96b342d76b25c5cfcb`.
All four runtime tasks passed separate specification and code-quality reviews,
followed by whole-repair review without actionable findings. Review corrections
preserve trusted STT categories through checked retirement and revoke late render
timing authority on output abort; native delayed-drain tests now use twelve actual
submissions and a lost-first-record case.

The first joined gate found a stale `0.1.8.0` qualification fixture (696 passed,
one failed); the separately reviewed test-only correction preserves production
authority and explicit stale-version rejection. The corrected exact 22-file gate
passed **701 tests, zero skips, three existing warnings in 108.03 seconds**.
Ruff and format checks passed for 19 changed Python files; version-lock and diff
checks passed. Mounted native/AEC and visible-control tests used fake PortAudio,
not hardware. The supplied worktree retained unrelated dirty trace-ledger/CSS
changes and owned documentation/inventory updates; this is not a clean release
evidence digest. Final task notes record the commit chain and evidence limits.

No live conversation was repeated. The earlier host-input overflow remains
unexplained, and complete playback/subsequent-new-turn live evidence is missing.
TASK-23175 remains In Progress, packaged rollout remains unqualified, and the
existing branch/worktree are kept without merge, push or launcher rebind.
