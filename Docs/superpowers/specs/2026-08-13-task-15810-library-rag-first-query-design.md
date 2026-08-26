# TASK-15810: Bounded First Library RAG Query Design

**Date:** 2026-08-13

**Status:** Approved direction; document-review corrections incorporated

**Task:** TASK-15810 — Library RAG Answer's first query on a fresh profile never returns

**Architecture:** [ADR-003](../../../backlog/decisions/003-settings-library-rag-defaults.md), [ADR-005](../../../backlog/decisions/005-invest-in-local-rag-mirroring-tldw-server.md)

## Goal

Make the first Library RAG Answer query on a fresh profile complete through the
real UI path. On the reference isolated profile containing 36 indexed notes,
the first Evidence row must render within 30 seconds. If profiling proves that
a legitimate one-time initialization phase is unavoidable, the Library must
name that phase instead of leaving the user at a generic searching status.

The task is root-cause work, not a speculative RAG rewrite. The implementation
must first identify the hot Python frame and its callers, then correct the
smallest shared owner responsible for the spin.

## Confirmed product decisions

- The acceptance bound is under 30 seconds from activating **Run** until the
  first Evidence row renders on the reference 36-note scratch profile.
- The bound is a verification target, not a blind timeout that abandons valid
  work at 30 seconds.
- Ordinary retrieval keeps the existing `searching · <sources>…` status.
- A new progress label is added only if profiling demonstrates a distinct,
  legitimate initialization phase. Its copy must name the work being done.
- The first-query path must remain cancellable, must discard stale results, and
  must not block Textual's event loop.
- Queries, document content, prompts, credentials, and secrets must not be
  added to diagnostic logs or profiler artifacts.

## Non-goals

- Do not replace the local RAG engine, embedding stack, or vector store.
- Do not introduce a parallel retrieval implementation or a second runtime.
- Do not add a dependency, cache layer, process pool, general scheduler, or
  background warm-up without profiler evidence that it is necessary.
- Do not change ranking, source scoping, answer generation, or the Library's
  existing recoverable error contract unless the measured root cause requires
  the smallest possible correction at one of those seams.
- Do not attribute the problem to model loading or embeddings from the existing
  C-level sample; it did not identify the hot Python frame.

## Current behavior and evidence

Two isolated live runs reproduced the same failure. A fresh profile remained at
`searching · Prompts…` for more than four minutes in TASK-15400. A second fresh
profile with 36 real User Guide notes remained at `searching · Notes…` for more
than eleven minutes in TASK-15700 while the process consumed about 98% CPU. The
embedding model was already present and the second run was offline, so a model
download was impossible.

A macOS process sample showed the work inside the CPython interpreter, including
filtering, set membership, `id()`, and `PySys_Audit`, but it could not name the
Python frame. Direct engine calls were used as a fallback, which means the real
Library retrieval path has not yet passed live verification.

The focused baseline on the isolated TASK-15810 branch is green: 143 Library
RAG service and product-maturity tests pass before any task changes.

## Reference acceptance fixture

The 30-second comparison uses one exact cold-query fixture:

- Corpus: every `*.md` file under `Docs/User_Guide`, sorted by repository path.
  The current fixture contains exactly 36 files. The seed record stores the
  path and SHA-256 manifest so a later content change is visible rather than
  silently changing the benchmark.
- Ingestion: a separate scratch-only seed process writes all 36 files through
  `add_note` and indexes them through `index_entries`, then exits. Every write
  and index result must succeed.
- Index preflight: the production stats API must report a non-empty vector
  collection, the recorded chunk count must equal the post-seed count, and the
  indexed metadata must cover all 36 seeded note IDs. The Library rail must
  visibly report `Notes (36)` before the query.
- Process state: the timed run starts a newly launched TUI process after the
  seed process exits. It has no pre-existing in-process shared RAG runtime and
  no earlier query in that process.
- Configuration: builtin active profile `hybrid_basic`, whose retrieval mode is
  `hybrid`; RAG Answer UI mode; Notes as the only selected source; `top_k=15`;
  citations enabled; the `all-MiniLM-L6-v2` model already present in a
  read-only cache; `HF_HUB_OFFLINE=1`; and no network model download path.
- Query: `how do I schedule a watchlist brief`, the predecessor query that
  returned `watchlists.md` through the same seeded corpus when the engine was
  driven directly.

The profiler run may add profiler overhead and is reported separately. The
final under-30-second acceptance measurement is an unprofiled run of this exact
fixture, from activating **Run** to the first visible Evidence row. A second
newly launched cold TUI process uses the same persisted scratch index for the
live responsiveness/cancellation check, so the timed run remains uninterrupted
and neither process has a previously constructed in-process RAG runtime.

## Investigation design

### 1. Reproduce the real path before changing it

Use a fully isolated scratch profile and follow the production path:

1. Activate Library RAG Answer through its real controls.
2. Let `LibraryScreen._start_library_rag_query()` create the request and launch
   its Textual worker.
3. Follow `_execute_library_rag_search()` into `run_library_rag_search()`.
4. Continue through `LibraryLocalRagSearchService` and the shared RAG engine.
5. Stop only after an Evidence row renders, a recoverable error renders, or a
   captured profile proves the CPU spin.

The scratch environment isolates `TLDW_TEST_MODE`, `HOME`, `XDG_DATA_HOME`,
`XDG_CONFIG_HOME`, `TLDW_CONFIG_PATH`, and the configured application data
directory. Before launch and after every boot, the harness parses the effective
scratch TOML with `tomllib` and asserts that `[paths].data_dir` resolves beneath
the scratch root. It then checks the running PID with `lsof`: zero open handles
may resolve beneath the real profile/data roots, every database/config/vector
handle found must resolve beneath the scratch root, and at least one live data
handle must prove that the process is using that scratch root. The real profile
and data directories are also fingerprinted before and after the run as a
separate proof that no mutation occurred.

### 2. Profile the stuck process

Capture a Python-level profile from the real scratch-profile process with
`py-spy` when available. If attaching is not supported in the environment,
instrument the same production path with `cProfile` and capture a bounded run.
The artifact must identify:

- the hottest Python frame;
- its immediate callers up to the Library service boundary;
- whether the cost is one-time initialization or repeated query work; and
- enough call/count or sample evidence to distinguish an accidental loop from
  legitimate bounded computation.

The profile artifact is produced before production code changes. It records
symbols and timing only; it must not contain query text, document content,
prompts, API keys, or configuration secrets.

### 3. Reduce the profile finding to a deterministic regression

Turn the named hot frame into the narrowest deterministic failing test that
still exercises the faulty production seam. Prefer a service or engine test to
a timing-only UI test. The test must fail for the measured mechanism, not merely
because a short wall-clock timeout expires.

Where practical, mutation-check the regression by temporarily restoring the
faulty behavior and confirming that the test fails for the intended reason.

## Implementation design

Correct one lowest shared owner after the profile and RED test establish it.
The likely owner is somewhere between `LibraryLocalRagSearchService` and the
shared local RAG engine, but this design intentionally does not select a frame
in advance.

The fix must preserve these boundaries:

- Library remains the owner of active query state, source selection, results,
  cancellation, and stale-query fencing under ADR-003.
- The existing shared local RAG runtime remains the product retrieval engine
  under ADR-005.
- Runtime construction and CPU-heavy retrieval remain off the Textual event
  loop. `asyncio.to_thread()` or the existing engine's worker boundary is used
  only where the measured owner requires it.
- UI cancellation and stale-result fencing are distinct from termination of
  underlying CPU work. A cancelled or superseded retrieval must either stop
  cooperatively, or remain serialized so another retrieval cannot run beside
  it and amplify the spin. If cooperative termination is impossible at the
  measured seam, any residual background work must stay bounded by the same
  corrected first-query operation and the next request must wait rather than
  create parallel CPU work.
- The current service request and normalized outcome contracts remain intact
  unless the profile proves that the contract itself is the faulty owner.
- A new status phase is introduced only for measured initialization work that
  remains after optimization. It must be truthful and generation-fenced like
  the existing searching state.

If the finding would require a new runtime boundary, durable cache ownership,
storage migration, or cross-module service contract, implementation pauses for
an ADR decision instead of silently expanding this task.

## Failure, cancellation, and privacy

- Existing recoverable Library search failures remain visible and retryable.
- Cancellation of the Textual search worker must prevent a late result or late
  initialization status from replacing a newer query.
- Starting a new query while the first is running keeps the existing generation
  and stale-result protections, and cannot create overlapping CPU retrievals.
- The UI stays responsive while profiling, initialization, and retrieval run.
- Diagnostic logging may include phase names, elapsed time, source types,
  aggregate counts, and exception classes. It may not include user query text,
  retrieved text, note bodies, answer prompts, credentials, or secrets.

## Verification strategy

### Automated regression

1. Capture the profiler artifact and name the hot Python frame before editing
   production code.
2. Add the smallest RED regression for the measured mechanism.
3. Apply the minimal production fix and make that regression GREEN.
4. Mutation-check the regression against the faulty behavior where practical.
5. If the profiled owner is in or below the engine boundary, exercise the
   concrete runtime returned by the production resolver (currently
   `EnhancedRAGServiceV2`) in at least one focused regression; a fake or base
   `RAGService` alone is insufficient.
6. Add a deterministic heartbeat/Pilot assertion around the changed retrieval
   seam. While a gated CPU retrieval is pending, a scheduled Textual heartbeat
   must advance and cancel/navigation input must be processed. The spin
   regression itself remains mechanism-based rather than timeout-only.
7. Run focused tests for the changed engine/service seam, Library RAG state and
   cancellation/serialization, and the product-maturity Pilot path.
8. Run Ruff on every changed Python file and `git diff --check`.

Tests must verify behavior deterministically. The 30-second wall-clock bound is
reserved for the isolated live acceptance run rather than used as the sole unit
test oracle.

### Live acceptance run

Create the exact reference acceptance fixture above. Start the actual TUI and
drive Library RAG Answer from **Run** to a visible Evidence row.

Record:

- the exact isolated environment and launch command;
- the query start timestamp and first-Evidence timestamp;
- elapsed time, which must be less than 30 seconds;
- any named initialization status that appeared;
- the visible Evidence result and final non-searching status;
- responsive cancellation/stale-query behavior if the changed seam touches it;
- live navigation or cancel input acknowledged within one second while the cold
  query is still pending, proving the event loop remains responsive;
- effective scratch TOML validation and running-PID `lsof` evidence proving the
  process opened no real-profile handles; and
- before/after fingerprints separately proving the real configuration and data
  were not modified.

## ADR check

**ADR required:** no

**ADR path:** N/A; ADR-003 and ADR-005 already govern the relevant ownership and
runtime boundaries.

**Reason:** The intended result is a measured performance bug fix inside the
existing Library-to-shared-RAG path. No schema, storage, sync, provider,
security, dependency, or long-lived interface decision is approved here. If
profiling forces one of those changes, work pauses and this ADR decision is
revisited before implementation.

## Success criteria

- A profiler artifact names the hot Python frame and its Library-path callers.
- A deterministic regression fails for the measured cause and passes after the
  minimal fix.
- The first real Library RAG Answer query on the isolated 36-note profile
  renders Evidence in under 30 seconds.
- A deterministic heartbeat assertion and live input check prove the Textual
  event loop remains responsive, while cancellation or supersession cannot
  create overlapping CPU retrievals.
- The Textual event loop, cancellation, stale-query fencing, error recovery,
  privacy constraints, source behavior, and answer flow remain intact.
