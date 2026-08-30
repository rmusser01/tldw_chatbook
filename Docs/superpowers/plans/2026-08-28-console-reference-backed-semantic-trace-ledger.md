# Console Reference-Backed Semantic Trace Ledger Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Replace quadratic Console exchange-capture blobs with a durable reference-backed semantic trace that remains coherent across calls, edits, forks, privacy projections, legacy normalization, and deletion.

**Architecture:** The canonical conversation continues to own ordinary message bodies. A new append-only trace graph references opaque semantic message revisions, stores sanitized provider-only material once, and records immutable surface/header/call boundaries. One transaction-scoped semantic mutation coordinator protects historical meaning; provider calls reserve before dispatch and settle independently afterward. Feature-gated dual reads, bounded legacy normalization, epoch-safe garbage collection, and later physical compaction provide a reversible rollout.

**Tech Stack:** Python 3.11+, Textual 8.x, SQLite/FTS5, dataclasses/enums, Pydantic at configuration boundaries, pytest, Hypothesis, and stdlib subprocess isolation for deferred custom regex.

---

## Governing records

- Umbrella: TASK-23113
- Core work packages: TASK-23113.1 through TASK-23113.9
- Deferred work packages: TASK-23113.10 custom regex and TASK-23113.11 physical compaction
- Approved design: Docs/superpowers/specs/2026-08-28-console-reference-backed-semantic-trace-ledger-design.md
- ADR required: yes
- ADR path: backlog/decisions/097-console-reference-backed-semantic-trace-ledger.md
- Reason: storage ownership, schema, message mutation integrity, provider dispatch, privacy, deletion, and rollout boundaries are architectural decisions.
- Independent follow-up: TASK-24206 owns lossless token chunk-row encoding and is not a prerequisite.

Before starting each Backlog child, move only that child to In Progress and add its concise Implementation Plan section through Backlog.md. Leave later children To Do. Do not mark the umbrella Done until its acceptance criteria are satisfied.

## Delivery order

| Order | Backlog task | Deliverable | Default state after merge |
|---:|---|---|---|
| 1 | TASK-23113.1 | Schema, typed repository, dual-read foundation | normalized writes off |
| 2 | TASK-23113.2 | Semantic revision coordinator and DB guard | coordinator active |
| 3 | TASK-23113.3 | Request provenance, mandatory credential filter, artifacts, headers, surfaces | shadow-build only |
| 4 | TASK-23113.4 | Durable reservation, lifecycle, settlement | hard-off outside tests until .6 |
| 5 | TASK-23113.5 | Fork and temporary-chat integration | gated writer |
| 6 | TASK-23113.6 | Credential filtering, built-in PII masks, viewer | Safe viewer default |
| 7 | TASK-23113.7 | Bounded legacy snapshot normalization | idle migration gated |
| 8 | TASK-23113.8 | Epoch-safe logical GC and owner purge | idle GC gated |
| 9 | TASK-23113.9 | Release gates, default switch, old writer retirement | normalized writer default |
| 10 | TASK-23113.10 | Bounded custom-regex subprocess | opt-in |
| 11 | TASK-23113.11 | Automatic physical compaction | threshold/idle gated |

## Cross-cutting rules

1. Do not add ordinary message text, a digest of it, or a variable-length history/source list to trace-owned rows.
2. The ADR-097 mandatory credential filter precedes every content-bearing trace write, including shadow/test paths. A sanitizer failure records a content-free omission.
3. SQLite atomicity covers only ChaChaNotes. UI/cache/index updates occur after commit and must be idempotent.
4. All trace reachability-edge mutations increment the global graph epoch in the same transaction.
5. Pre-dispatch reservation and dispatch_started are fail-closed. Post-dispatch settlement is idempotent and best-effort.
6. Historical trace is never editable. UI actions are inspect, filter, search the permitted projection, copy, export, and owner-scoped purge.
7. Register every migration in pyproject.toml, MANIFEST.in, Packaging/check_manifest.py, and Tests/Packaging/test_installed_distribution.py.
8. Use real historical SQLite fixtures. Do not fake migrations by only changing PRAGMA user_version.
9. Run targeted tests first. Ask the owner before any full pytest sweep.
10. Re-run schema-version and task-ID collision checks before each schema/task commit because parallel work can move origin/dev.

## Target module boundaries

| Module | Responsibility |
|---|---|
| tldw_chatbook/Chat/console_trace_models.py | Immutable enums/dataclasses and transition validation |
| tldw_chatbook/Chat/console_trace_repository.py | Typed ChaChaNotes trace reads/writes using caller-owned transactions |
| tldw_chatbook/Chat/console_semantic_revision.py | Provider-visible envelope and mutation coordinator |
| tldw_chatbook/Chat/console_trace_provenance.py | Capture-only source descriptors and final-value binding |
| tldw_chatbook/Chat/console_trace_service.py | Surface/header construction, call reservation, lifecycle, settlement |
| tldw_chatbook/Chat/console_trace_redaction.py | Mandatory credentials, built-in PII spans, mask projection |
| tldw_chatbook/Chat/console_trace_legacy.py | Bounded legacy decode, normalization, equivalence |
| tldw_chatbook/Chat/console_trace_maintenance.py | Scheduling, reachability mark/sweep, metrics, later compaction |
| tldw_chatbook/Chat/console_trace_projection.py | Call reconstruction and Safe/Full viewer/export projections |

Avoid a second event vocabulary in codecs or UI. Every layer consumes the same typed logical records.

## TASK-23113.1 — Schema and dual-read foundation

### Task 1: Define logical trace types and invariants

**Files:**

- Create: tldw_chatbook/Chat/console_trace_models.py
- Create: Tests/Chat/test_console_trace_models.py

- [ ] Write failing tests for opaque IDs, immutable records, permitted call transitions, terminal-state rejection, bounded replacement ranges, and content-free omissions.
- [ ] Run: python -m pytest Tests/Chat/test_console_trace_models.py -q
- [ ] Confirm failure because the module does not exist.
- [ ] Implement TraceCallState, TraceOutcome, TraceContentRef, TraceOmission, SemanticRevisionRef, SurfaceBoundary, SurfaceReplacement, FrozenTracePolicy, and transition validators.
- [ ] Generate opaque IDs independently of content. Expose no helper that hashes canonical message text.
- [ ] Re-run the test and confirm PASS.
- [ ] Commit: git commit -am "feat(console): define semantic trace model"

### Task 2: Install the trace storage schema

**Files:**

- Create: tldw_chatbook/DB/migrations/chachanotes_v55_to_v56_console_semantic_trace.sql
- Modify: tldw_chatbook/DB/ChaChaNotes_DB.py
- Modify: tldw_chatbook/DB/sql_validation.py
- Modify: pyproject.toml
- Modify: MANIFEST.in
- Modify: Packaging/check_manifest.py
- Modify: Tests/Packaging/test_installed_distribution.py
- Create: Tests/DB/test_chachanotes_v56_semantic_trace_migration.py
- Modify: Tests/DB/test_sql_validation.py
- Modify: Tests/DB/test_schema_table_allowlist_guard.py
- Modify: Tests/ChaChaNotesDB/test_index_census.py
- Create/modify: a genuine v53 fixture under Tests/DB/fixtures/

- [ ] Verify _CURRENT_SCHEMA_VERSION is still 53. If not, rename all N-to-N+1 paths consistently.
- [ ] Write a failing historical-fixture migration test for foreign keys, uniqueness, indexes, append-only constraints, graph epoch, checkpoint state, and idempotent reopen.
- [ ] Define storage for segments/lineage, semantic revisions, revision-policy bindings, sanitized artifacts, surface nodes, immutable headers, calls, ordered events, response links, redaction spans, conversation owners, migration checkpoints, maintenance lease/state, and singleton graph epoch.
- [ ] Update the ChaChaNotes SQL table allowlist in the same change and keep both static and live-schema parity guards exact.
- [ ] Register every named v56 index in the generated index census with exact uniqueness and column order.
- [ ] Store predecessor head, start node, end node, and replacement node for replacements; never store a shadowed-source array.
- [ ] Make historical rows immutable. Permit only monotonic call state, live-locator retirement, checkpoints, owner detachment, and maintenance fields to change.
- [ ] Register SQL in all four packaging registries.
- [ ] Run: python -m pytest Tests/DB/test_chachanotes_v56_semantic_trace_migration.py Tests/DB/test_sql_validation.py Tests/DB/test_schema_table_allowlist_guard.py Tests/ChaChaNotesDB/test_index_census.py Tests/Packaging/test_installed_distribution.py -q
- [ ] Build the repository sdist and wheel through the packaging test harness; confirm the migration exists in both.
- [ ] Commit the migration, registry, and fixture files explicitly with message: feat(db): add semantic trace ledger schema

### Task 3: Add a transaction-aware repository

**Files:**

- Create: tldw_chatbook/Chat/console_trace_repository.py
- Modify: tldw_chatbook/Chat/chat_persistence_service.py
- Create: Tests/Chat/test_console_trace_repository.py

- [ ] Write failing in-memory SQLite tests for artifact collision verification, revision-policy reuse, header reuse, call idempotency, event sequence monotonicity, owner attachment, and epoch increments.
- [ ] Make repository mutators accept a caller-owned sqlite3.Cursor; do not open nested transactions.
- [ ] On artifact digest lookup, compare stored sanitized bytes, media type, and normalization version. Allocate a new opaque artifact ID on mismatch.
- [ ] Add a narrow repository factory/property to ChatPersistenceService. UI widgets must not issue trace SQL.
- [ ] Run: python -m pytest Tests/Chat/test_console_trace_repository.py -q
- [ ] Confirm PASS and commit: feat(console): add semantic trace repository

### Task 4: Introduce feature-gated dual reads

**Files:**

- Create: tldw_chatbook/Chat/console_trace_projection.py
- Modify: tldw_chatbook/Chat/console_chat_store.py
- Modify: tldw_chatbook/Chat/console_runtime.py
- Modify: tldw_chatbook/UI/Screens/chat_screen.py
- Create: Tests/Chat/test_console_trace_projection.py
- Modify: Tests/UI/test_chat_screen_console_inspector_loader.py

- [ ] Write failing tests proving verified normalized calls are preferred, missing/unverified calls fall back to message_exchanges, and mixed calls keep stable order.
- [ ] Return one discriminated NormalizedTraceCall or LegacyExchangeCall read model. Do not normalize blobs yet.
- [ ] Inject projection through runtime/store construction rather than opening a DB from the screen.
- [ ] Keep normalized writes off and preserve current Inspector behavior.
- [ ] Run both targeted tests and confirm PASS.
- [ ] Commit: feat(console): add dual-read trace projection

## TASK-23113.2 — Semantic revision coordinator

### Task 5: Inventory every model-visible mutation route

**Files:**

- Create: Docs/Development/console-semantic-mutation-inventory.md
- Create: Tests/Chat/test_console_semantic_mutation_inventory.py
- Inspect: tldw_chatbook/Chat/Chat_Functions.py
- Inspect: tldw_chatbook/Chat/console_chat_store.py
- Inspect: tldw_chatbook/Chat/chat_persistence_service.py
- Inspect: tldw_chatbook/DB/ChaChaNotes_DB.py
- Inspect: import, sync, reasoning, tool, and attachment writers found by rg

- [ ] Enumerate all insert/update/delete operations for messages and provider-visible sidecars.
- [ ] Classify each route as model-visible, visibility/ownership-only, or presentation-only, naming its public owner.
- [ ] Encode expected model-visible entry points in a failing census test so a new unclassified writer is visible.
- [ ] Include generation settlement, edit, regeneration, import, sync, attachment mutation, soft delete, and hard delete.
- [ ] Commit: test(console): inventory semantic message mutations

### Task 6: Implement the transaction-scoped mutation guard

**Files:**

- Create: tldw_chatbook/DB/migrations/chachanotes_v56_to_v57_semantic_mutation_guard.sql
- Create: tldw_chatbook/Chat/console_semantic_revision.py
- Create: tldw_chatbook/Chat/console_trace_redaction.py
- Modify: tldw_chatbook/DB/base_db.py
- Modify: tldw_chatbook/DB/ChaChaNotes_DB.py
- Modify: tldw_chatbook/Chat/chat_persistence_service.py
- Modify: pyproject.toml
- Modify: MANIFEST.in
- Modify: Packaging/check_manifest.py
- Modify: Tests/Packaging/test_installed_distribution.py
- Create: Tests/Chat/test_console_semantic_revision_coordinator.py
- Create: Tests/Chat/test_console_trace_credential_filter.py
- Create: Tests/DB/test_chachanotes_v57_semantic_mutation_guard_migration.py
- Create/modify: a genuine v54 fixture under Tests/DB/fixtures/

- [ ] Write failing tests for initial revisions, lazy revision creation for genuine pre-v56 messages, two-policy copy-on-write materialization, edit replacement, hard delete, rollback, and every unguarded direct-SQL bypass category.
- [ ] Confirm v56 has already shipped or is immutable on the prerequisite branch; never revise its SQL to add enforcement later.
- [ ] Register a connection-local SQLite guard function in base_db.py. It reads Python connection-local authorization only and does not query SQLite recursively.
- [ ] Ship and register the v56-to-v57 migration containing triggers that reject protected envelope changes/deletes without the guard. A raw connection lacking the function must fail closed.
- [ ] Implement the minimal versioned ADR-097 mandatory credential sanitizer and content-free failure marker before materialization exists. Add canary assertions across revision-policy artifacts, exceptions, and diagnostics.
- [ ] Implement SemanticRevisionCoordinator.ensure_current_revision() as a transactional, digest-free metadata operation for a pre-v56 message. It points to the live canonical source and reuses an existing current revision when present.
- [ ] Test a genuine pre-v56 conversation whose unchanged selected messages receive revision references and create no provider-only message artifacts.
- [ ] Implement SemanticRevisionCoordinator.mutate_message() to authorize one scoped mutation, preserve reachable policies through the mandatory sanitizer, append revision/replacement, mutate canonical state, retire locators safely, and advance graph epoch in one transaction.
- [ ] Clear authorization on success, rollback, and exception.
- [ ] Register the second migration in all four packaging registries and test an actual v56-to-v57 upgrade plus fresh install.
- [ ] Run: python -m pytest Tests/Chat/test_console_semantic_revision_coordinator.py Tests/Chat/test_console_trace_credential_filter.py Tests/DB/test_chachanotes_v57_semantic_mutation_guard_migration.py Tests/Packaging/test_installed_distribution.py -q
- [ ] Confirm PASS and commit: feat(console): enforce semantic revision mutations

### Task 7: Route all existing writers through the coordinator

**Files:**

- Modify: tldw_chatbook/Chat/chat_persistence_service.py
- Modify: tldw_chatbook/Chat/console_chat_store.py
- Modify: every model-visible writer named by Task 5
- Modify: tests adjacent to each changed writer

- [ ] Add failing atomicity and bypass tests for every model-visible public route before changing it.
- [ ] Route commit_durable_turn, create_message, update_message_content, replace_assistant_generation_projection, subtree deletion, import, sync, and attachment/variant writes through the coordinator.
- [ ] Change ConsoleChatStore edits so memory mutates only after durable commit or is restored exactly on rollback.
- [ ] Test that presentation-only metadata does not create revisions and soft deletion only changes visibility/ownership/epoch while retained semantic bytes remain unchanged.
- [ ] Re-run the census and every touched writer test.
- [ ] Commit: refactor(console): centralize model-visible persistence

## TASK-23113.3 — Provenance, artifacts, surfaces, and headers

### Task 8: Carry capture-only provenance through preparation

**Files:**

- Create: tldw_chatbook/Chat/console_trace_provenance.py
- Modify: tldw_chatbook/Chat/console_prepared_request.py
- Modify: tldw_chatbook/Chat/console_turn_preparation.py
- Create: Tests/Chat/test_console_trace_provenance.py
- Modify: Tests/Chat/test_console_prepared_request.py
- Modify: Tests/Chat/test_console_turn_preparation.py

- [ ] Write failing tests for system framing, memory, mandatory context, compactable history, active request, tools, provider overlays, omissions, and ensure_current_revision() transaction failure.
- [ ] Add immutable descriptors beside semantic units without putting them in provider messages, token counts, authority, or bindings.
- [ ] At durable run admission, call ensure_current_revision() transactionally for every selected saved message lacking a revision, including unchanged pre-v55 rows.
- [ ] If lazy revision persistence fails, Capture On preparation cannot dispatch: interactive sends enter Retry or explicit Capture Off admission, autonomous runs fail safely, and Capture Off starts a fresh preparation that cannot inherit partial Capture On descriptors/policy.
- [ ] Saved-message descriptors carry the resulting semantic revision IDs. Provider-only descriptors carry source kind and frozen policy, not raw copies.
- [ ] Ensure compaction/reordering returns descriptors aligned one-for-one with transformed values.
- [ ] Run all three targeted test files and confirm PASS.
- [ ] Commit: feat(console): carry trace provenance through preparation

### Task 9: Extend credential filtering and verify final provider-bound values

**Files:**

- Modify: tldw_chatbook/Chat/console_trace_redaction.py
- Modify: tldw_chatbook/Chat/console_provider_gateway.py
- Modify: tldw_chatbook/Chat/console_trace_provenance.py
- Modify: Tests/Chat/test_console_trace_credential_filter.py
- Modify: Tests/Chat/test_console_provider_gateway.py
- Modify: Tests/Chat/test_console_trace_provenance.py

- [ ] Write failing tests for known credential kwargs/nesting, URL userinfo/query/fragment, recognized secret formats, sanitizer failure, Anthropic system separation, OpenAI messages, llama.cpp literal ownership, provider transforms, tool schemas, and deliberate mismatch.
- [ ] Extend the Task 6 mandatory filter to provider kwargs, nested provider structures, URLs, and recognized free-text formats before any artifact identity or content-bearing trace write. Return sanitized values or content-free unavailable markers; never return/store match bodies, value hashes, or exception strings.
- [ ] Bind every final semantic value to its descriptor immediately before dispatch and independently reconstruct the captured kwargs.
- [ ] Compare the complete final semantic kwargs component-by-component: messages, effective generation parameters, adapter-default provenance, response format, reasoning/thinking controls, tool schemas, credential-free endpoint identity, provider route/overlays, and every allowlisted extra.
- [ ] Discard ephemeral canonical-content comparison digests before persistence.
- [ ] On mismatch return a typed unavailable record. Never persist raw kwargs or call build_request_capture() as fallback.
- [ ] Add first-write canaries proving secrets are absent from artifacts, headers, surfaces, events, responses, errors, and diagnostics even in shadow/test mode.
- [ ] Run credential/gateway/provenance tests and confirm PASS.
- [ ] Commit: feat(console): sanitize and verify provider-bound trace provenance

### Task 10: Persist bounded surfaces and changed-only headers

**Files:**

- Create: tldw_chatbook/Chat/console_trace_service.py
- Modify: tldw_chatbook/Chat/console_provider_gateway.py
- Create: Tests/Chat/test_console_trace_service.py

- [ ] Write failing tests for unchanged-header reuse, changed-header creation, complete header reconstruction, bounded appends, and 75-percent replacements.
- [ ] Store provider-only components through sanitized artifacts and ordinary saved content through revision references.
- [ ] Build persistent heads incrementally and validate replacement ranges belong to the predecessor surface.
- [ ] Store credential-free endpoint identity only.
- [ ] Reconstruct and compare headers covering effective generation parameters, adapter-default provenance, response format, reasoning/thinking controls, rendered system references, tool schemas, endpoint identity, and provider overlays.
- [ ] Add schema assertions that no call/replacement field contains serialized prior history.
- [ ] Run the targeted service test and confirm PASS.
- [ ] Commit: feat(console): persist bounded trace surfaces and headers

## TASK-23113.4 — Durable call reservation and settlement

### Task 11: Reserve, bind, and mark dispatch before provider entry

**Files:**

- Modify: tldw_chatbook/Chat/console_trace_service.py
- Modify: tldw_chatbook/Chat/console_provider_gateway.py
- Modify: tldw_chatbook/Chat/console_turn_preparation.py
- Modify: tldw_chatbook/Chat/console_runtime.py
- Modify: Tests/Chat/test_console_provider_gateway.py
- Modify: Tests/Chat/test_console_turn_preparation.py
- Create: Tests/Chat/test_console_trace_call_lifecycle.py

- [ ] Write failing tests proving Capture On commits a content-free reserved row, binds surface/header or a persisted incomplete marker, commits dispatch_started, and only then invokes the adapter.
- [ ] Add call identity as conversation + segment + turn + run + call sequence with an immutable idempotency key.
- [ ] On ambiguous reservation commit, query by idempotency key before allocating another call.
- [ ] Distinguish component transformation from trace persistence: a sanitizer/descriptor component failure that is durably represented by an omission/incomplete marker may proceed; inability to persist the boundary, header, or incomplete state blocks dispatch.
- [ ] Add a capture-preparation pause kind for reservation/boundary-state persistence failures. Initial interactive sends offer Retry or explicit one-shot Send without capture; autonomous runs fail safely.
- [ ] Ensure a capture-off continuation is a new admitted run state and cannot mutate the frozen Capture On policy.
- [ ] Test that reservation, boundary/incomplete-state, or dispatch-start persistence failure prevents adapter entry, while a successfully persisted component omission proceeds with disclosed incomplete fidelity.
- [ ] Run the three targeted lifecycle/gateway/preparation test files.
- [ ] Commit: feat(console): reserve traces before provider dispatch

### Task 12: Cover every provider route and multi-call loop

**Files:**

- Modify: tldw_chatbook/Chat/console_provider_gateway.py
- Modify: tldw_chatbook/Chat/console_trace_service.py
- Modify: Tests/Chat/test_console_provider_gateway.py
- Modify: Tests/Chat/test_console_trace_call_lifecycle.py

- [ ] Census every adapter-entry seam with rg before editing: generic chat_api_call/_chat_api_call dispatch, worker dispatch, direct llama.cpp streaming HTTP, direct llama.cpp completion HTTP, stream-to-completion fallback retry, complete_auxiliary(), and future registry/provider helpers.
- [ ] Write failing route-matrix tests for streaming, non-streaming, generic provider worker, both direct llama.cpp HTTP paths, the internal stream-to-completion retry, retries, fallbacks, tools, cancellations, stops, and exceptions.
- [ ] Move provider-adapter entry behind one small gateway method that requires either a committed Capture On dispatch token or an explicit Capture Off admission token.
- [ ] Delete/disable direct capture construction at branch-specific call sites; branch code reports semantic events to the shared call service.
- [ ] Treat each retry and tool-loop iteration as a distinct call with stable run ordering.
- [ ] During an already-running interactive loop, pause before the next adapter entry when reservation fails. Autonomous loops terminate safely.
- [ ] Keep complete_auxiliary() explicitly classified as an admitted Capture Off/untraced sensitive auxiliary path, or change its policy in ADR-097 before tracing it. Test that it cannot accidentally inherit a Capture On run or bypass a required token.
- [ ] Keep the normalized Capture On writer hard-off in all user-facing configuration until TASK-23113.6 completes the all-owner privacy matrix.
- [ ] Re-run the route matrix and existing Tests/Chat/test_console_exchange_capture.py to prove compatibility while the old writer remains gated.
- [ ] Commit: refactor(console): unify traced provider dispatch

### Task 13: Seal responses independently and recover open calls

**Files:**

- Create: tldw_chatbook/Chat/console_trace_settlement.py
- Modify: tldw_chatbook/Chat/console_trace_service.py
- Modify: tldw_chatbook/Chat/console_provider_gateway.py
- Modify: tldw_chatbook/Chat/chat_persistence_service.py
- Modify: tldw_chatbook/Chat/console_runtime.py
- Create: Tests/Chat/test_console_trace_settlement.py

- [ ] Write failing tests for response_started, complete/stopped/error/interrupted outcomes, usage, duplicate settlement, seal failure, and process-restart recovery.
- [ ] After canonical assistant persistence, re-read its immutable revision and link it only when the sanitized provider-facing envelope matches exactly.
- [ ] Otherwise store one sanitized response artifact and label the relationship. Never link by timing or message position alone.
- [ ] Put failed post-dispatch settlements in a bounded, idempotent in-process queue. Do not roll back a provider result or saved assistant message.
- [ ] On cold start map untouched reserved to not_dispatched, uncertain dispatch_started to dispatch_unknown, and open response_started to interrupted. Never move a state backward.
- [ ] Verify response data can remain trace-owned when canonical assistant persistence fails.
- [ ] Run settlement, persistence, and gateway tests; confirm PASS.
- [ ] Commit: feat(console): settle and recover provider call traces

### Task 14: Prove reservation and settlement latency

**Files:**

- Create: Tests/Benchmarks/test_console_trace_call_latency.py
- Create: Tests/Benchmarks/fixtures/console_trace_reference_machine.json

- [ ] Pin Python/SQLite versions, journal/synchronous/page/cache settings, filesystem, CPU model, sample counts, and reference-machine identity in the fixture.
- [ ] For each operation, discard 100 warm-up calls, record 1,000 measured calls per fresh database, and write raw nanosecond samples plus metadata as deterministic JSON under pytest's tmp_path artifact directory so normal runs do not dirty the worktree.
- [ ] Measure reservation plus dispatch_started and settlement transactions separately across five fresh databases.
- [ ] Compute p95 with statistics.quantiles(samples, n=100, method="inclusive")[94] and document that formula beside the fixture.
- [ ] Fail unless reservation p95 is at most 10 ms, every reservation sample is at most 50 ms, and settlement p95 is at most 25 ms.
- [ ] Assert settlement work runs off the Textual UI thread.
- [ ] Run the benchmark test on the reference environment and save raw measurements in the test artifact, not production logs.
- [ ] Commit: test(console): gate trace call persistence latency

## TASK-23113.5 — Fork lineage and temporary-chat promotion

### Task 15: Persist shared fork boundaries

**Files:**

- Modify: tldw_chatbook/Chat/console_chat_fork.py
- Modify: tldw_chatbook/Chat/console_chat_store.py
- Modify: tldw_chatbook/Chat/chat_persistence_service.py
- Modify: tldw_chatbook/Chat/console_trace_repository.py
- Modify: Tests/Chat/test_console_chat_fork.py
- Modify: Tests/Chat/test_console_chat_fork_persistence.py
- Create: Tests/Chat/test_console_trace_fork_lineage.py

- [ ] Write failing tests for durable-to-durable, durable-to-temporary, temporary-to-temporary, and later-saved forks.
- [ ] Capture the source trace boundary in the same snapshot fence as copied active-lineage messages.
- [ ] Attach the child owner to one immutable parent segment/boundary and append only a child suffix.
- [ ] Include failed/abandoned attempts before the fence and exclude later or branch-excluded calls.
- [ ] Compare trace table row counts and artifact bytes before/after fork to prove inherited prefixes are not copied.
- [ ] Test divergent post-fork edits and calls; each branch must reconstruct independently.
- [ ] Run all fork tests and confirm PASS.
- [ ] Commit: feat(console): share immutable trace prefixes across forks

### Task 16: Enforce Save & Send for temporary Capture On

**Files:**

- Modify: tldw_chatbook/Chat/console_chat_store.py
- Modify: tldw_chatbook/Chat/console_turn_preparation.py
- Modify: tldw_chatbook/Chat/console_provider_gateway.py
- Modify: tldw_chatbook/Widgets/Console/console_capture_policy_dialog.py
- Modify: Tests/Chat/test_console_turn_preparation.py
- Create: Tests/UI/test_console_temporary_capture_admission.py

- [ ] Write failing tests that a temporary Capture On send cannot enter the provider adapter.
- [ ] Present Save & Send and explicit Send without capture. Do not present a silent fallback.
- [ ] Save & Send promotes the conversation/in-memory lineage before call reservation, then resumes the exact prepared send only after durable IDs exist.
- [ ] On promotion or reservation failure keep the temporary source intact and send nothing.
- [ ] Verify a temporary fork with only in-memory prefix materializes that prefix once under the saved child without persisting/mutating the source.
- [ ] Run targeted preparation/UI tests and confirm PASS.
- [ ] Commit: feat(console): require durable chat for captured sends

## TASK-23113.6 — Credential filtering, built-in PII, and viewer profiles

### Task 17: Complete credential coverage across every durable and derived owner

**Files:**

- Modify: tldw_chatbook/Chat/console_trace_redaction.py
- Modify: tldw_chatbook/Chat/console_trace_service.py
- Modify: tldw_chatbook/Chat/console_trace_repository.py
- Modify: tldw_chatbook/Chat/console_exchange_export.py
- Modify: Tests/Chat/test_console_trace_credential_filter.py
- Create: Tests/Chat/test_console_trace_privacy_owners.py

- [ ] Extend the foundation with failing structural tests for every nested/config/provider-specific field, canonical revision projection, copy-on-write materialization, exception path, and derived UI/export owner.
- [ ] Keep the mandatory detector versioned and its findings content-free. Never return/store matches, value hashes, or exception strings.
- [ ] Verify filtering before artifact identity, revision projection binding, header/event/response persistence, export, clipboard copy, preview, and logging.
- [ ] On failure persist only unavailable markers.
- [ ] Inspect every trace-owned table/blob plus trace-derived log/preview/clipboard/copy/export fixture for canary secrets. Explicitly exclude canonical conversation rows from that absence assertion.
- [ ] Add dense matching input and structural work caps; do not rely only on wall-clock timing.
- [ ] Run both privacy test files and confirm PASS.
- [ ] Commit: feat(console): filter credentials from semantic traces

### Task 18: Add built-in PII masks and frozen policies

**Files:**

- Modify: tldw_chatbook/Chat/console_trace_redaction.py
- Modify: tldw_chatbook/Chat/console_capture_policy_repository.py
- Modify: tldw_chatbook/Chat/console_trace_repository.py
- Modify: tldw_chatbook/config.py
- Create: Tests/Chat/test_console_trace_pii_masks.py
- Modify: Tests/Chat/test_console_session_settings.py

- [ ] Write failing tests for Unicode codepoint spans, deterministic sorting, overlap union, mixed category, policy reuse, ruleset changes, and PII default Off.
- [ ] Store source ID, field path, start/end, category, detector/rule IDs, versions, outcome, and opaque ruleset revision only.
- [ ] Keep matched values, hashes, substrings, surrounding text, regex source, and standalone lengths out of persistence.
- [ ] Freeze capture/PII policy at run admission across retries and tool loops; support global, sparse conversation, and eligible next-send precedence.
- [ ] Keep canonical conversation content unchanged.
- [ ] Prove provider-only PII transformation is irreversible while canonical revisions remain unchanged and always render through immutable frozen masks in both Safe and Full.
- [ ] Run PII and session-setting tests and confirm PASS.
- [ ] Commit: feat(console): add frozen built-in PII trace masks

### Task 19: Replace storage detail with Safe/Full viewer profiles

**Files:**

- Modify: tldw_chatbook/Widgets/Console/console_conversation_inspector.py
- Modify: tldw_chatbook/Chat/console_trace_projection.py
- Modify: tldw_chatbook/Chat/console_exchange_export.py
- Modify: tldw_chatbook/UI/Screens/settings_privacy_security.py
- Modify: tldw_chatbook/Widgets/Console/console_capture_policy_dialog.py
- Modify: tldw_chatbook/config.py
- Modify: Tests/UI/test_console_conversation_inspector.py
- Modify: Tests/UI/test_console_exchange_export_dialog.py
- Modify: Tests/UI/test_settings_privacy_security.py
- Modify: Tests/Chat/test_console_session_settings.py

- [ ] Write failing tests that Safe/Full read the same trace, both honor frozen masks, Safe never materializes hidden Full bodies, and Full requires a new explicit choice after upgrade.
- [ ] Migrate old capture enabled/disabled to Capture On/Off. Preserve old capture_detail only as historical provenance. Discard obsolete next-send detail overrides at restart.
- [ ] Default every upgraded viewer to Safe and show the one-time explanation; old Full must not auto-reveal.
- [ ] Require explicit confirmation for Full expansion and Full copy/export. Status text separately reports capture, fidelity, mandatory filtering, PII, and viewer profile.
- [ ] Keep normal transcript rendering unchanged and unredacted.
- [ ] Ensure Safe search/copy/export sees only Safe projection; Full search is bounded in memory and never enters FTS/previews/logs.
- [ ] Run all four UI/settings suites and relevant projection tests.
- [ ] Commit: feat(console): make Safe and Full trace viewer profiles

## TASK-23113.7 — Legacy snapshot normalization

### Task 20: Normalize one legacy exchange without invented chronology

**Files:**

- Create: tldw_chatbook/Chat/console_trace_legacy.py
- Modify: tldw_chatbook/Chat/console_exchange_capture.py
- Modify: tldw_chatbook/Chat/console_trace_projection.py
- Create: Tests/Chat/test_console_trace_legacy_normalizer.py
- Modify: Tests/Chat/test_console_exchange_capture.py
- Modify: Tests/DB/test_chachanotes_full_capture_migration.py
- Modify: Tests/DB/test_chachanotes_v53_safe_capture_trim.py

- [ ] Write failing cases for Full, Safe, aggregate omission, ambiguous rows, unmatched provider context, corruption, binary stubs, and truncated captures.
- [ ] Decode only through the bounded production decoder, then apply the current mandatory credential filter. Do not retroactively apply optional PII.
- [ ] Before matching a genuine pre-v56 canonical message, call ensure_current_revision() transactionally so unchanged historical rows can become revision references.
- [ ] Match ordinary rows to unique historical revisions with role/order/shape plus an ephemeral sanitized fingerprint that is discarded before commit.
- [ ] Store ambiguous/unmatched rows separately as sanitized legacy artifacts.
- [ ] Build each call as isolated legacy_snapshot surface nodes using parent_node + component_ref persistent prefixes. Do not infer cross-call edit/fork/predecessor relationships.
- [ ] Label every ambiguity, omission, and fidelity loss in projection output.
- [ ] Run legacy normalizer and existing capture migration tests.
- [ ] Commit: feat(console): normalize legacy captures as snapshots

### Task 21: Add bounded resumable background normalization

**Files:**

- Create: tldw_chatbook/Chat/console_trace_maintenance.py
- Modify: tldw_chatbook/Chat/console_runtime.py
- Modify: tldw_chatbook/Chat/console_trace_repository.py
- Create: Tests/Chat/test_console_trace_legacy_migration.py
- Create: Tests/Benchmarks/fixtures/console_trace_legacy_200_turn_v53.sqlite3
- Create: Tests/Benchmarks/fixtures/console_trace_legacy_200_turn_v53.sha256

- [ ] Write failing tests for resume, idempotency, rollback, equivalence failure, malformed rows, active-provider-run exclusion, and checkpoint reopen.
- [ ] Admit work only when no provider run or other DB maintenance owns the lease.
- [ ] Bound each batch to at most 100 rows, 4 MiB decoded input, and 100 ms write transaction, then yield.
- [ ] Read back and structurally compare the normalized sanitized legacy projection in the same transaction before deleting the legacy blob.
- [ ] Leave failures/checkpoints retryable without content-bearing diagnostics.
- [ ] Add the named pinned v53 200-turn legacy fixture and checksum above; verify checksum before each run and require completion within five seconds on the recorded reference machine.
- [ ] Run the targeted migration test and dual-read projection tests.
- [ ] Commit: feat(console): migrate legacy traces in bounded idle batches

## TASK-23113.8 — Epoch-safe logical garbage collection

### Task 22: Centralize roots, edges, and epoch advancement

**Files:**

- Create: Docs/Development/console-trace-reachability-inventory.md
- Modify: tldw_chatbook/Chat/console_trace_repository.py
- Modify: tldw_chatbook/Chat/console_semantic_revision.py
- Modify: tldw_chatbook/Chat/console_trace_service.py
- Modify: tldw_chatbook/Chat/console_trace_legacy.py
- Create: Tests/Chat/test_console_trace_graph_epoch.py

- [ ] Derive and document roots and edges from every trace foreign key plus lifecycle/retention semantics before implementing the collector.
- [ ] Write a parameterized failing test for every mutation helper: call creation and lifecycle edges, events, surface heads, segment parent/inherited boundaries, conversation visibility/ownership, retention roots, owner/fork links, bindings, locators, headers, artifacts, response links, and migration state.
- [ ] Make every mutation helper advance the singleton epoch in its transaction.
- [ ] Prevent direct repository edge mutation APIs that omit epoch advancement.
- [ ] Add concurrency tests that interleave a mark snapshot with each inventoried mutation and observe an epoch mismatch.
- [ ] Run the epoch test and all repository/coordinator tests.
- [ ] Commit: feat(console): advance trace graph epoch on reachability changes

### Task 23: Mark/sweep unreachable trace data

**Files:**

- Create: tldw_chatbook/DB/migrations/chachanotes_v57_to_v58_trace_gc_guard.sql
- Modify: tldw_chatbook/Chat/console_trace_maintenance.py
- Modify: tldw_chatbook/Chat/console_trace_repository.py
- Modify: tldw_chatbook/Chat/console_chat_store.py
- Modify: tldw_chatbook/DB/base_db.py
- Modify: tldw_chatbook/DB/ChaChaNotes_DB.py
- Modify: pyproject.toml
- Modify: MANIFEST.in
- Modify: Packaging/check_manifest.py
- Modify: Tests/Packaging/test_installed_distribution.py
- Create: Tests/Chat/test_console_trace_gc.py
- Create: Tests/DB/test_chachanotes_v58_trace_gc_guard_migration.py
- Create: Tests/UI/test_console_trace_purge.py

- [ ] Write failing tests for raw/direct deletion rejection, authorized sweep deletion, one-owner purge, shared-fork retention, soft-delete roots, migration-pending roots, interrupted sweep, and concurrent epoch change.
- [ ] Ship/register a v57-to-v58 migration whose deletion triggers permit trace-row deletion only under a connection-local maintenance sweep grant tied to the current lease and exact marked epoch.
- [ ] Add a genuine v57 historical fixture and package the migration in all four registries; never revise the shipped v56/v57 SQL.
- [ ] Mark from durable conversation owners, inherited boundaries, open calls, explicit retention roots, soft-deleted conversations, and pending legacy rows.
- [ ] Record the exact epoch with the mark, acquire maintenance exclusion, and recheck that exact epoch inside the sweep transaction. Abort/retry on mismatch.
- [ ] Authorize the sweep grant only after the in-transaction epoch recheck; clear it in success/rollback/exception paths. A raw connection or stale lease/epoch remains fail-closed.
- [ ] Detach only the selected conversation owner and report remaining fork owners; do not add lineage-wide purge.
- [ ] Report logical live bytes, reclaimed bytes/pages, freelist, WAL, and allocated DB bytes separately.
- [ ] Verify all durable owners/projections are deleted or intentionally retained after reopen.
- [ ] Run GC, purge, fork, and privacy-owner tests.
- [ ] Commit: feat(console): reclaim unreachable semantic traces

## TASK-23113.9 — Release gates and rollout

### Task 24: Build the production-shaped linear-growth gate

**Files:**

- Create: Tests/Benchmarks/test_console_trace_linear_growth.py
- Create: Tests/Benchmarks/fixtures/console_trace_200_turn_fixture.json
- Create: Tests/Benchmarks/fixtures/console_trace_200_turn_fixture.sha256
- Create: tldw_chatbook/Chat/console_trace_metrics.py

- [ ] Generate a versioned semi-incompressible fixture with fixed seed, provider kwargs shape, per-turn lengths, checksum, Python/SQLite versions, SQLite pragmas, and reference-machine identity.
- [ ] Drive it through the real ConsoleProviderGateway, not a hand-built repository call.
- [ ] At turns 1, 50, 100, 150, and 200 record trace-owned live bytes/rows, artifacts/materialized revisions, legacy bytes, DB/freelist/WAL bytes, and encode/decode/settlement timings.
- [ ] Run five fresh databases, define the median as statistics.median over each raw metric, and retain every per-run measurement in deterministic JSON under pytest's tmp_path artifact directory.
- [ ] Fail unless turns 101-200 add at most 1.25 times the bytes and rows added by turns 1-100 and turn 200 trace-owned live bytes are at most 2.0 MiB.
- [ ] Assert no call/header/replacement row has a list/blob proportional to transcript age.
- [ ] Keep metrics free of message/artifact bodies and secret values.
- [ ] Commit: test(console): add linear trace growth release gate

### Task 25: Add the replacement-heavy gate

**Files:**

- Create: Tests/Benchmarks/test_console_trace_replacement_growth.py
- Create: Tests/Benchmarks/fixtures/console_trace_replacement_fixture.json

- [ ] Add a 200-turn real-gateway fixture replacing the oldest 75 percent of active surface every 20 turns.
- [ ] Require the same at-most-1.25 second-half byte/row ratios.
- [ ] Inspect schema and values to prove replacement rows contain a bounded predecessor/range/replacement shell.
- [ ] Include edits, regeneration, retries, tools, RAG/project context, forks, failures, credential filtering, legacy migration, and logical GC across the two canonical fixtures.
- [ ] Commit: test(console): gate replacement-heavy trace growth

### Task 26: Complete the reversible rollout

**Files:**

- Modify: tldw_chatbook/config.py
- Modify: tldw_chatbook/Chat/console_provider_gateway.py
- Modify: tldw_chatbook/Chat/console_exchange_capture.py
- Modify: tldw_chatbook/Chat/console_trace_projection.py
- Modify: tldw_chatbook/UI/Screens/settings_privacy_security.py
- Modify: Tests/Chat/test_console_provider_gateway.py
- Modify: Tests/Chat/test_console_exchange_capture.py
- Modify: Tests/UI/test_settings_privacy_security.py

- [ ] Add separate read-enabled and write-enabled compatibility switches plus non-content metrics for normalized/legacy/fallback/incomplete paths.
- [ ] Test these phases: dual reads + old writer; shadow normalized build; normalized writer + legacy fallback; normalized default + old writer disabled.
- [ ] Prove rollback from normalized writes leaves all committed normalized and legacy history readable.
- [ ] After Tasks 24-25 and TASK-23113.4 latency gates pass, make normalized writes the default.
- [ ] Disable the old repeated-transcript writer for new calls. Keep bounded decoder/dual-read support for old rows.
- [ ] Remove no legacy schema/data in this task.
- [ ] Run targeted gateway, capture, projection, settings, migration, privacy, fork, and benchmark suites.
- [ ] Commit: feat(console): make semantic trace capture the default

### Task 27: Update user and operator documentation

**Files:**

- Modify: Docs/User_Guide/console/context-and-rag.md
- Create: Docs/User_Guide/console/semantic-trace-capture.md
- Modify: Inspector help/copy source found in tldw_chatbook/Widgets/Console/console_conversation_inspector.py
- Modify: backlog/tasks/task-23113.9 - Gate-and-roll-out-the-semantic-trace-ledger.md

- [ ] Explain reference-not-copy ownership, provider-only artifacts, semantic fidelity/omissions, and why the trace viewer is not editable.
- [ ] Explain Capture On/Off, temporary Save & Send, Safe/Full disclosure, mandatory credential limits, PII masks, and unchanged canonical transcript.
- [ ] Explain edits/forks/shared prefixes, owner-scoped purge, legacy omissions, logical reclamation versus physical compaction, and backup/export limits.
- [ ] Verify every footer/help hint names a real implemented action and follows repository keybinding conventions.
- [ ] Run documentation link checks and targeted Inspector UI tests.
- [ ] Commit: docs(console): explain semantic trace capture

## TASK-23113.10 — Deferred bounded custom PII regex

Do not start until TASK-23113.6 and TASK-23113.9 are Done and the core privacy/growth gates remain green.

### Task 28: Validate and version custom rules

**Files:**

- Create: tldw_chatbook/Chat/console_trace_custom_pii.py
- Modify: tldw_chatbook/config.py
- Modify: tldw_chatbook/UI/Screens/settings_privacy_security.py
- Create: Tests/Chat/test_console_trace_custom_pii_config.py
- Modify: Tests/UI/test_settings_privacy_security.py

- [ ] Write failing tests for stable IDs, labels/categories, enabled state, length limits, allowed flags, unsupported constructs, deterministic priority, and opaque ruleset revisions.
- [ ] Store user-authored pattern text only in settings; never copy it to trace mask rows, logs, or diagnostics.
- [ ] Make invalid rules non-runnable with actionable content-free diagnostics.
- [ ] Run config/settings tests and confirm PASS.
- [ ] Commit: feat(console): validate custom PII trace rules

### Task 29: Execute one bounded disposable worker

**Files:**

- Create: tldw_chatbook/Chat/console_trace_regex_worker.py
- Modify: tldw_chatbook/Chat/console_trace_custom_pii.py
- Modify: tldw_chatbook/Chat/console_trace_redaction.py
- Create: Tests/Chat/test_console_trace_regex_worker.py
- Modify: Tests/Chat/test_console_trace_pii_masks.py
- Modify: Docs/User_Guide/console/semantic-trace-capture.md

- [ ] Write failing tests for input bytes, field count, rule count, match count, wall deadline, memory limit, crash, malformed output, and catastrophic backtracking.
- [ ] Spawn one stdlib-only disposable subprocess per capture batch; never reuse it across unrelated captures.
- [ ] Kill on deadline and treat timeout/crash/malformed/excess output as fail-closed redaction failure for affected components.
- [ ] Return field path, Unicode ranges, category, rule ID, and outcome only—never matched text.
- [ ] Use OS/process resource enforcement where supported plus parent-side byte/count/deadline limits. Tests must assert termination/bounds, not a fragile timing threshold alone.
- [ ] Feed successful spans into the same deterministic overlap-union mask path as built-ins.
- [ ] Document custom rule validation, batch/resource limits, failure omissions, and immutable/irreversible masking semantics.
- [ ] Run worker, mask, privacy-owner, viewer, copy, and export tests.
- [ ] Commit: feat(console): isolate custom PII regex execution

## TASK-23113.11 — Deferred automatic physical compaction

Do not start until TASK-23113.8 and TASK-23113.9 are Done and logical reclamation is proven in production-shaped fixtures.

### Task 30: Admit compaction safely

**Files:**

- Modify: tldw_chatbook/Chat/console_trace_maintenance.py
- Modify: tldw_chatbook/Chat/console_runtime.py
- Modify: tldw_chatbook/config.py
- Modify: tldw_chatbook/DB/ChaChaNotes_DB.py
- Modify: tldw_chatbook/DB/base_db.py
- Create: Tests/Chat/test_console_trace_compaction_admission.py
- Create: Tests/DB/test_chachanotes_connection_quiescence.py

- [ ] Write failing tests for active provider run, other maintenance owner, open connections, WAL checkpoint failure, insufficient disk, lease loss, below-threshold freelist, and retry.
- [ ] Admit only after logical GC and configured free-page/database-size/activity thresholds.
- [ ] Add an app-wide ChaChaNotes connection registry/quiescence barrier that can reject new acquisitions, wait for every thread-owned connection to return, and close all registered connections before maintenance.
- [ ] Obtain a visible maintenance lease, pause new provider dispatch, quiesce all connections, checkpoint WAL with TRUNCATE, and preflight free disk for SQLite's same-file VACUUM temporary-space requirement.
- [ ] Treat admission failure as visible pending work, not permission to force compaction.
- [ ] Ensure no VACUUM/physical rewrite runs inside an active write transaction.
- [ ] Guarantee provider dispatch and connection acquisition resume in a finally block after every admission/failure/cancellation path.
- [ ] Run admission and all-thread connection-quiescence tests and confirm PASS.
- [ ] Commit: feat(console): gate trace database compaction

### Task 31: Compact with same-file VACUUM, report, and recover

**Files:**

- Modify: tldw_chatbook/Chat/console_trace_maintenance.py
- Modify: tldw_chatbook/UI/Screens/settings_privacy_security.py
- Create: Tests/Chat/test_console_trace_compaction.py
- Create: Tests/UI/test_console_trace_maintenance_status.py
- Create: Tests/Benchmarks/fixtures/console_trace_compaction.sqlite3
- Create: Tests/Benchmarks/fixtures/console_trace_compaction.sha256
- Modify: Docs/User_Guide/console/semantic-trace-capture.md

- [ ] Write failing tests for successful shrink, interruption/failure, reopen, integrity-check failure, retained fork history, WAL failure, all-thread quiescence, guaranteed dispatch resumption, UI responsiveness, progress reporting, and deferred retry.
- [ ] Run ADR-097's selected same-file VACUUM from a dedicated maintenance worker after the quiescence/checkpoint/preflight sequence.
- [ ] Install a bounded progress callback/cancellation signal when supported by sqlite3; UI status updates must use the existing thread-safe event path.
- [ ] Report logical live, freelist, WAL, and allocated bytes separately before and after.
- [ ] Keep DB readable/retryable after interruption and never claim forensic erasure.
- [ ] After reopen, run the selected bounded integrity verification before releasing the maintenance lease; on failure retain pending/retry state while the finally path still restores connection acquisition and provider dispatch.
- [ ] Require the pinned reference fixture to finish in at most five seconds; give arbitrary databases an estimate/progress instead of a universal promise.
- [ ] Verify the named compaction fixture checksum before the run and document eligibility, progress, retry, file-size, and non-forensic-erasure behavior.
- [ ] Reopen and reconstruct all reachable fork/call traces after compaction.
- [ ] Run compaction, GC, fork, projection, and UI status tests.
- [ ] Commit: feat(console): compact reclaimed trace storage

## Final verification and Backlog closeout

For each child task:

- [ ] Re-read its acceptance criteria before implementation and update them before code if scope legitimately changes.
- [ ] Run git diff --check.
- [ ] Run the targeted suites listed in that child plus tests for reachable dependents.
- [ ] Run formatter/linter commands scoped to changed files.
- [ ] Perform a self-review for raw-content duplication, bypass routes, secret/PII leakage, unbounded lists, thread blocking, and cross-DB atomicity assumptions.
- [ ] If the task exposed a reusable incident/trap, add the incident and rule to the relevant backlog/docs/lessons-*.md; do not invent a lesson.
- [ ] Add concise Implementation Notes with modified files, decisions/trade-offs, verification evidence, and ADR-097.
- [ ] Check every acceptance criterion, then set only that child Done through Backlog.md.

Before declaring the core rollout gate complete:

- [ ] Confirm TASK-23113.1 through TASK-23113.9 are Done and TASK-24206 remains independent. This permits the normalized default but does not close the umbrella.
- [ ] Run the complete targeted matrix:

  python -m pytest Tests/Chat/test_console_trace_models.py Tests/Chat/test_console_trace_repository.py Tests/Chat/test_console_trace_projection.py Tests/Chat/test_console_semantic_revision_coordinator.py Tests/Chat/test_console_semantic_mutation_inventory.py Tests/Chat/test_chat_persistence_service.py Tests/Chat/test_console_trace_provenance.py Tests/Chat/test_console_prepared_request.py Tests/Chat/test_console_turn_preparation.py Tests/Chat/test_console_trace_service.py Tests/Chat/test_console_provider_gateway.py Tests/Chat/test_console_exchange_capture.py Tests/Chat/test_console_trace_call_lifecycle.py Tests/Chat/test_console_trace_settlement.py Tests/Chat/test_console_chat_fork.py Tests/Chat/test_console_chat_fork_persistence.py Tests/Chat/test_console_trace_fork_lineage.py Tests/Chat/test_console_trace_credential_filter.py Tests/Chat/test_console_trace_privacy_owners.py Tests/Chat/test_console_trace_pii_masks.py Tests/Chat/test_console_session_settings.py Tests/Chat/test_console_trace_legacy_normalizer.py Tests/Chat/test_console_trace_legacy_migration.py Tests/Chat/test_console_trace_graph_epoch.py Tests/Chat/test_console_trace_gc.py Tests/DB/test_chachanotes_v56_semantic_trace_migration.py Tests/DB/test_chachanotes_v57_semantic_mutation_guard_migration.py Tests/DB/test_chachanotes_v56_trace_gc_guard_migration.py Tests/DB/test_chachanotes_full_capture_migration.py Tests/DB/test_chachanotes_v53_safe_capture_trim.py Tests/UI/test_chat_screen_console_inspector_loader.py Tests/UI/test_console_conversation_inspector.py Tests/UI/test_console_exchange_export_dialog.py Tests/UI/test_console_temporary_capture_admission.py Tests/UI/test_console_trace_purge.py Tests/UI/test_settings_privacy_security.py Tests/Packaging/test_installed_distribution.py -q

- [ ] Before treating this matrix as complete, append every adjacent writer suite discovered by the Task 5 mutation inventory (including import, sync, attachment, generation, and deletion owners); the inventory document records the exact additional paths for that branch revision.

- [ ] Run the three release-gate benchmark files separately and preserve raw per-run output.
- [ ] Ask the owner whether to run the full test suite. Do not run it without explicit opt-in.
- [ ] Use superpowers:requesting-code-review before merge.
- [ ] Use superpowers:verification-before-completion before any claim that the task is complete or passing.
- [ ] Keep TASK-23113 open until TASK-23113.10 and TASK-23113.11 are Done. Core rollout completion at .9 is a milestone, not parent-task completion.
- [ ] Before closing TASK-23113, run the custom-regex, compaction-admission, connection-quiescence, compaction, maintenance-status, GC, privacy, and projection tests together.

## Explicit non-goals for implementers

- Do not persist raw token chunks; TASK-24206 owns chunk-row encoding.
- Do not capture literal provider HTTP/auth/TLS details.
- Do not sync or index hidden trace content.
- Do not redact canonical conversation messages.
- Do not infer legacy cross-call lineage.
- Do not add lineage-wide purge.
- Do not silently send without capture after a Capture On reservation failure.
- Do not edit historical trace in the viewer.
