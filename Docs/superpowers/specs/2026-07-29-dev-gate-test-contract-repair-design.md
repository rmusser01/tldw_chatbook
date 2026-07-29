# Dev-Gate Test Contract Repair Design

## Goal

Restore the mandatory `dev` pytest gate by reconciling stale or nondeterministic
tests with the production contracts that already exist, then review and refresh
the checked diagnostic inventory under ADR-029. This is a non-production
repair: it must not restore retired Chat infrastructure, change audio-recording
behavior, or silently admit unsafe persistent diagnostics.

## Evidence

The failures reproduce on an exact `origin/dev` checkout:

- `Tests/Event_Handlers/test_worker_events_contract.py` imports `StreamDone`,
  although TASK-577 deliberately removed that event and made the retained
  adapter reject streaming.
- `Tests/Event_Handlers/test_worker_local_citation_capture.py` exclusively
  exercises the same retired streaming bridge, removed sentinel/error
  swallowing, and worker-owned citation-builder stripping. The sole retained
  caller is explicitly non-streaming and passes no citation builder; native
  Console now owns the live citation and privacy lifecycle.
- Latest `dev` independently removed the retired `TabState` fixture from
  `Tests/UI/test_chat_shell_bar.py` while adding current persona-label coverage.
  TASK-1333 preserves that upstream repair rather than carrying a competing
  edit.
- `Tests/Audio/test_audio_integration.py` starts a recorder thread and then
  calls the same recording loop repeatedly on the test thread. It also leaves
  VAD enabled, so the assertion depends on installed optional dependencies and
  whether synthetic bytes are classified as speech.
- `Tests/Audio/test_recording_service.py` repeats the same thread-and-direct-loop
  race with VAD enabled, so synthetic audio may never reach the callback that
  stops the loop.
- The adjacent SoundDevice flow fixture also leaves VAD enabled for a
  four-sample synthetic callback, so no audio reaches its queue assertion.
- Four `Tests/Chat/test_chat_functions.py` cases monkeypatch deleted
  module-level `settings` objects. The live Llama.cpp and DeepSeek adapters
  deliberately resolve `get_runtime_config_snapshot()` at each request
  boundary under ADR-029.
- Four `Tests/LLM/test_local_llm_provider_config.py` cases repeat the deleted
  `settings` pattern for the local-LLM adapter, which also resolves the
  immutable runtime snapshot at each request boundary.
- `Tests/LLM_Provider_Catalog/test_local_openai_compatible_provider_name.py`
  patches the snapshot seam but supplies local-LLM configuration at the retired
  top level instead of `api_settings.local-llm`.
- Three real-seam Notes fixtures pass a nonexistent `tmp_path/notes_base` to
  `NotesInteropService`. ADR-029's trusted-directory boundary correctly
  rejects missing directories instead of creating them implicitly.
- `Tests/DB/test_rag_indexing_db.py::test_large_batch_operations` combines
  useful 1,000-item persistence coverage with hard wall-clock limits. The
  unchanged `origin/dev` test took 24.98 seconds inside a loaded full-suite run
  but 1.25 seconds in isolation, proving the timing assertion depends on host
  contention rather than database behavior.
- `Tests/Event_Handlers/test_eval_db_operations_path.py` retargets
  `TLDW_CONFIG_PATH` beneath a nonexistent `profile-two` directory. The
  private-path boundary correctly rejects that missing trusted parent before
  config or database creation.
- The Library skill-name collision guard reports that the newer runtime tools
  `search_run_log`, `run_log_stats`, and `run_log_slice` are absent from
  `_SHADOWED_BUILTIN_NAMES`, allowing a local skill to collide with a real
  built-in runtime name.
- `Tests/Architecture/test_persistent_diagnostic_inventory.py` reports reviewed
  production-owner drift while the persistent sink topology remains unchanged.
  ADR-029 requires inspecting the changed calls before regenerating the checked
  inventory.

TASK-1333 owns these gate repairs and the reviewed generated-inventory refresh.

## Decision

Update the tests to describe current behavior:

1. Keep the non-streaming worker failure regression and delete its obsolete
   streaming-sentinel case. Delete the fully obsolete worker-local citation
   capture file rather than recreating retired streaming, sentinel, logging, or
   builder ownership. The existing
   `Tests/Event_Handlers/test_retained_worker_adapter.py` already pins the live
   delegation and streaming-rejection contracts, while native Console tests pin
   citation lifetime and privacy, so TASK-1333 adds no duplicates.
2. Preserve the latest `dev` chat-shell test unchanged. It already covers the
   live session and persona-label contract without `TabState`.
3. In the stream-error regression, run `_pyaudio_recording_loop()` exactly once
   with `is_recording = True` and
   VAD disabled, without calling `start_recording()`. Assert the exact two-chunk
   callback sequence, `stop_stream()`, `close()`, and final stopped state.
   Rename the test so it no longer claims automatic recovery that production
   does not implement.
4. Apply the same deterministic structure to the PyAudio happy-flow regression:
   disable VAD, set the callback and recording state directly, invoke one loop,
   and assert exactly three chunks plus cleanup. Do not start a background
   recorder.
5. Keep the SoundDevice flow through its public start/stop contract, but disable
   VAD for its tiny synthetic callback. Use a bounded event set by the mocked
   `InputStream` constructor before reading the captured callback, and stop the
   mocked recorder in `finally` before asserting that audio was queued.
6. Update the stale Llama.cpp, DeepSeek, and local-LLM request tests to patch
   `get_runtime_config_snapshot()` with a `RuntimeConfigSnapshot` containing
   their test configuration under the live `api_settings` shape. Do not restore
   or emulate mutable module-level settings.
7. Create each fixture-owned temporary Notes base directory with mode `0700`
   before constructing `NotesInteropService`, and close all per-user Notes
   connections before the template DB during fixture teardown. Do not weaken
   production path verification or make the service create a security-sensitive
   root implicitly.
8. Generate the candidate diagnostic inventory, inspect every changed owner and
   sink-topology entry against ADR-029's metadata-only boundary, and refresh the
   checked artifact only if the changes are safe. If review finds an unsafe log
   value, correct that violation under ADR-029 before regenerating; do not bless
   it as inventory drift.
9. Keep the large-batch indexing test's 1,000-item write and retrieval
   assertions, but remove its host-dependent elapsed-time measurements. This
   default functional gate must prove persistence correctness, not benchmark a
   contended workstation.
10. Create the retargeted profile fixture directory with mode `0700` before
    selecting its config file, and explicitly close the test-owned Evals
    database connection during teardown. Do not make production config loading
    create or trust a missing security-sensitive parent.
11. Add exactly the three registered run-log runtime tool names to the existing
    fixed Library skill shadow set. Keep the literal collision boundary; do not
    import the agent runtime registry into the pure display-state module.

The only planned production change outside an ADR-029 diagnostic correction is
the three-name synchronization of the existing Library collision boundary. No
compatibility shims. No broad deletion of live tests.

## Alternatives

- Restoring `StreamDone` or replacing the upstream `TabState` repair would
  contradict the accepted retirement architecture and revive dead ownership.
- Deleting live test files would make collection green but discard useful
  retained non-streaming, shell-label, and audio-error coverage. The one deleted
  file contains no live contract after worker ownership moved to native Console.
- Replacing retired models with local fixtures or duplicating the existing
  streaming-rejection test would keep dead or redundant coverage alive.
- Teaching the retained adapter to strip, log, or consume citation builders
  would revive ownership that moved to native Console.
- Blindly regenerating the diagnostic inventory would defeat its review gate.
- Raising the indexing test's timeout would retain a machine-dependent failure
  with a different arbitrary threshold.
- Deriving the Library shadow set dynamically from the agent registry would
  couple a pure display-state module to runtime ownership and replace the
  intentional fixed collision boundary.
- The selected edits remove only obsolete assertions, make the audio contracts
  deterministic, retain large-batch correctness coverage, and preserve the
  existing privacy boundary.

## Verification

Run the repaired modules first, then their nearby affected suites, the
diagnostic-inventory guard, static checks on changed files, and the
repository-wide suite. TASK-1333 is complete only when these failures are
absent; unrelated or environment-dependent failures remain recorded rather
than hidden.

## Architecture Decision Record

ADR required: no

ADR path: backlog/decisions/029-local-private-data-boundary.md

Reason: This reconciles tests with already-accepted production boundaries and
applies ADR-029's existing metadata-only inventory review requirement. It does
not introduce a new runtime, storage, security, dependency, or cross-module
decision.
