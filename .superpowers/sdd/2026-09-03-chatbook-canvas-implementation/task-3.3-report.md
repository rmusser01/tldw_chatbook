# Task 3.3 report — Commit staged mutations with the originating assistant turn

## Status

Implemented and verified. The run-owned controller stages immutable Canvas revisions by exact `(session_id, run_id, tool_call_id)`, freezes source-private settlement contributions at Agent terminalization, and commits the assistant message, metadata-only Canvas cards, revisions, and dispatch-checkpoint deletion in the existing caller-owned transaction. Persistence failures roll back both sides and retain the READY stage for bounded retry; cancellation and terminal Agent failures discard the source-bearing stage.

ADR required: no new ADR. This delivery directly implements the accepted ownership, privacy, temporary-session, and transaction boundaries in `backlog/decisions/115-local-versioned-canvas-artifacts-and-browser-sandbox.md`.

## Implementation

- Added `ConsoleCanvasController` with exact run registration, incarnation-fenced temporary sessions, idempotent tool-call replay, source-private staged rows, sequential ancestry, bounded `ambiguous_ancestry` conflicts, immutable settlements, duplicate-callback handling, session/runtime cleanup, and temporary-session promotion contributions.
- Added an exact-authority lifecycle binding to `CanvasToolProvider`; `ConsoleAgentBridge` uses its server-issued run ID when creating the primary Agent row, then delivers that exact ID and terminal status to the controller in its existing teardown path. `AgentService.run_turn` exposes only the narrow optional requested-ID seam needed for this authoritative binding.
- Extended `ConsoleAssistantSettlement` and `ConsoleDispatchRepository.settle_with_assistant` with the existing narrow transaction-contribution seam. Contributions execute after the assistant CAS update and before checkpoint deletion, using the same transaction and exact native-to-durable assistant ID mapping.
- Extended `ConsoleChatStore` to merge Canvas card metadata, supply the contribution to terminal settlement, confirm only after transaction success, preserve READY stages after write failure, and discard/retire run state on session close, state replacement, and app shutdown. Temporary session activation and promotion use the same participant lifecycle contract.
- Added strict `MessageMetadata` Canvas-card records containing only Canvas/revision IDs, title, sequence, digest, status, origin, reopenability, and a bounded error code.
- Added metadata-only transcript rows and plain-text output. No source field exists in the transcript presentation type; source reopening remains a Canvas-service responsibility.
- Updated TASK-31228 with the projection inventory, transaction boundary, cancellation/retry semantics, sentinel evidence, ADR disposition, and review status.

## RED evidence

The tests were written before their implementation seams existed:

- `../../.venv/bin/python -m pytest Tests/Chat/test_console_canvas_controller.py -q` initially failed collection with `ModuleNotFoundError: tldw_chatbook.Chat.console_canvas_controller`.
- The new real persistence tests initially failed because `ConsoleAssistantSettlement` had no `contributions` field and terminal settlement had no caller-owned Canvas write phase.
- The restored transcript test initially failed because `MessageMetadata` had no `canvas_cards` projection and `console_transcript` had no `canvas_card_presentations` renderer.
- The temporary lifecycle test initially failed because the controller had no `activate_session`/promotion contract.
- During refinement, the focused controller suite exposed the session-close tombstone bug (``13 passed, 1 failed``); the failure showed `settlement_for_assistant(...)` returned `None` instead of the bounded discarded status.
- The write-failure retry test exposed that store settlement wraps repository failure as `ConsoleDispatchSettlementError`; the assertion was narrowed to that exact public failure type.
- Removing the test's temporary DB monkeypatch produced the intended run-identity RED: the Agent row received a random ID instead of the provider scope's server-issued ID. The new authenticated lifecycle binding plus `AgentService.requested_run_id` made the production path own the exact ID.

## GREEN evidence

Fresh final targeted command:

```text
../../.venv/bin/python -m pytest Tests/Chat/test_console_canvas_controller.py Tests/Chat/test_console_dispatch_recovery.py Tests/Chat/test_message_metadata.py Tests/Agents/test_canvas_tool_provider.py Tests/Chat/test_console_agent_bridge.py Tests/Agents/test_agent_runtime.py Tests/Agents/test_provider_continuation_runtime.py Tests/Chat/test_provider_continuation_privacy.py Tests/Chat/test_console_agent_bridge_cancel_all.py Tests/Chat/test_chat_persistence_service.py Tests/Chat/test_console_generation_card.py -q
722 passed, 1 warning in 76.38s
```

The warning is the environment's existing `requests` dependency-version warning. Per repository policy, no full-suite sweep was run.

Fresh static verification:

```text
python -m compileall -q <modified production modules>
ruff check console_canvas_controller.py test_console_canvas_controller.py
ruff check <modified files> --select E9,F --ignore F401
git diff --check
All checks passed; exit 0. `agent_service.py` retains one pre-existing unused `FIND_TOOLS_SCHEMA` import, so the changed-file fatal-error pass ignores baseline F401 rather than changing unrelated code.
```

## Transaction and settlement model

1. The controller registers the exact assistant message and captured Canvas scope for one Agent run.
2. Each mutation compiles a complete document and records a source-bearing private row. Repeating the same run/tool-call/request returns the same revision; changing the request under the same identity fails closed.
3. Agent teardown calls `finish_assistant_run` with the actual returned run ID. `done` freezes a READY settlement; cancellation, error, stuck, missing/mismatched identity, and lifecycle teardown produce a DISCARDED settlement with no contribution.
4. The store combines the settlement's metadata-only cards with existing local message metadata and passes the frozen contribution into terminal dispatch settlement.
5. In one DB transaction, the repository CAS-updates the existing assistant anchor, inserts Canvas documents/revisions through the scoped writer, and deletes the dispatch checkpoint. Canvas-only turns use that already-created assistant anchor even when assistant content is empty.
6. Only a committed repository result advances the controller to COMMITTED. A message-write or revision-write failure leaves no partial durable row and retains READY state for bounded retry. Duplicate finalizers return the frozen settlement and cannot create another revision.
7. Temporary committed runs retain their in-memory graph, present a `temporary` card status, and freeze an exact multi-message promotion contribution. Promotion refuses while any temporary run is unsettled; abort releases only the lease, while confirm/retire removes the exact incarnation-owned graph.

## Source-leak sentinel evidence

- `test_every_non_model_projection_omits_canvas_source` covers display, log, cycle, and continuation projections for all Canvas tools.
- Adversarial nested-field tests rebuild allowlisted metadata and fail closed if any field tries to carry the unique source sentinel.
- `test_real_review_batch_fails_closed_when_canvas_classification_raises` and the real review-batch tests inspect serialized Agent rows/steps and prove the source sentinel is absent.
- `test_metadata_serialization_never_contains_source` proves settlement JSON and object representations exclude source.
- `test_transcript_restores_metadata_only_canvas_card` reconstructs a persisted message, renders the Canvas row and plain-text transcript, and proves the source sentinel is absent.
- Source-bearing contribution and staged-row fields use `repr=False`; their custom representations and fingerprints contain only identities, counts, ancestry, and digests.

## Files changed

- `tldw_chatbook/Chat/console_canvas_controller.py` (new)
- `tldw_chatbook/Chat/console_chat_store.py`
- `tldw_chatbook/Chat/console_agent_bridge.py`
- `tldw_chatbook/Chat/console_dispatch_checkpoint.py`
- `tldw_chatbook/Chat/console_dispatch_repository.py`
- `tldw_chatbook/Chat/message_metadata.py`
- `tldw_chatbook/Agents/canvas_tool_provider.py`
- `tldw_chatbook/Agents/agent_service.py`
- `tldw_chatbook/Widgets/Console/console_transcript.py`
- `Tests/Chat/test_console_canvas_controller.py` (new)
- `Tests/Chat/test_console_dispatch_recovery.py`
- `Tests/Chat/test_console_agent_bridge.py`
- `Tests/Agents/test_canvas_tool_provider.py`
- `backlog/tasks/task-31228 - Integrate-Canvas-tools-with-atomic-Console-turns.md`

## Self-review

- Confirmed the approval bypass remains owned by the nominal authenticated `CanvasToolProvider`; the new lifecycle accessor returns the coordinator only for the exact live registration-authority object.
- Confirmed no Canvas source is placed in `MessageMetadata`, transcript presentation types, contribution fingerprints, reprs, Agent projections, or logs.
- Confirmed contribution writes cannot open or commit transactions and use the existing scoped insert-only writer.
- Confirmed write order and rollback with real SQLite triggers on both the assistant message and Canvas revision paths; revision failure is retried successfully after the injected trigger is removed.
- Confirmed session reuse gets a new incarnation, shutdown clears source-bearing temporary state, promotion refuses unsettled runs, and failed promotion leases retain retryable state.
- Confirmed sequential updates preserve exact parent links and parallel same-parent attempts do not mutate.

## Concerns

- No independent subagent review was launched because the assigned task explicitly prohibited spawning reviewers. Parent/integration review should focus on approval-authority reach and source leakage as requested by the delivery checkpoint.
- Final production composition must inject the same controller instance into the per-run `CanvasToolProvider` and `ConsoleChatStore` (and as the temporary promotion participant when it owns temporary history). The authenticated provider lifecycle binding now makes its server-issued scope ID the actual Agent row ID; the enclosing Canvas integration still owns constructing and injecting those objects.
- Full-suite coverage was intentionally not run under repository policy; the 722-test focused set covers controller, persistence, the complete Console bridge suite, runtime, continuation, cancellation, transcript, and existing card regressions.

## Round 1/5 independent-review fixes (2026-09-04)

### Implementation

- Production composition now creates one `ConsoleCanvasController(CanvasService(db))` in `ConsoleRuntime` and injects that exact object into the real `ConsoleChatStore` as both turn controller and temporary-promotion participant. Each real Agent turn registers a fresh server-bound run scope and gives a run-fenced coordinator to `CanvasToolProvider`; store teardown closes the same owner.
- Every registered stage now captures both the exact `CanvasSessionOwner` incarnation and an opaque `CanvasRunOwner`. Provider calls and terminal callbacks use the run-fenced coordinator, so a retired same-ID session's late tool or terminal callback is inert and cannot mutate or settle its replacement.
- Temporary promotion now freezes an owner/lease/generation/exact-run-set CAS. The lease blocks registration, mutation replay, finalization, confirmation, and reactivation after snapshot. Confirm/retire remove only contributed run IDs; abort releases only the exact lease, while close paths retain an in-flight contribution until its exact confirmation or abort.
- Durable settlement now invokes `confirm_exact_settlement` immediately after SQLite reports COMMITTED and before mutable message/recovery publication. The confirmation is idempotent for reconciliation. A true DB failure never runs it, so the exact stage remains READY and source-bearing for retry.
- Temporary list/read/update resolution now filters committed origins through `scope.active_message_ids`, honors an exact reachable selected revision, and picks the correct reachable head when switching branches. Owner-global sequence allocation preserves global sequence uniqueness even when a historical durable revision is the chosen parent.
- Conflict outcomes are stored in the bounded idempotency cache. An exact request replay returns the same result object after later mutations. A second open/READY run that sees an uncommitted mutation for the same Canvas returns `ambiguous_ancestry` without appending; sequential same-run parents remain ordered.
- Raw Canvas inserts were removed from turn contributions. `_CursorConsoleTransactionWriter.append_canvas_batch` delegates to the canonical transaction-aware `CanvasRepository.append_batch_in_transaction`, which validates the caller-owned transaction, conversation/message/path ownership, UUIDs, digests and source sizes, document/revision/source quotas including existing rows, parent ancestry, exact global sequence, and identity conflicts before its first write. It never starts or commits a nested transaction.
- `MessageMetadata.remap_canvas_origins` is the typed promotion helper. It changes only matching Canvas-card origin message IDs before temporary messages are inserted. Durable revision origins use the same reserved mapping, and restart hydration reconstructs matching card/revision origins without source in metadata.
- Empty assistant rows carrying typed Canvas-card metadata are accepted during temporary promotion, preserving the required Canvas-only assistant turn anchor.

### Exact RED/GREEN evidence

The first four independent-review regressions were added before their corrections and produced this RED:

```text
../../.venv/bin/pytest -q Tests/Chat/test_console_canvas_controller.py -k 'conflict_replay or promotion_lease or late_bound_handle or temporary_history'
4 failed
```

The failures were exact: conflict replay changed after a later update; a new run registered after a promotion snapshot; registration returned no incarnation-fenced run handle; and a sibling temporary branch could list off-path Canvas state. After the lifecycle/branch/idempotency changes, the same command produced `4 passed`.

The typed promotion helper also had an exact RED (`AttributeError: MessageMetadata has no attribute remap_canvas_origins`) before its implementation. Its focused test is included in the final green run. The production-composition and canonical transaction probes were added as verification probes after the core correction rather than represented as pre-implementation RED; this report does not misstate them as strict TDD evidence.

Fresh final focused baseline plus canonical repository verification:

```text
../../.venv/bin/python -m pytest Tests/Chat/test_console_canvas_controller.py Tests/Chat/test_console_dispatch_recovery.py Tests/Chat/test_message_metadata.py Tests/Agents/test_canvas_tool_provider.py Tests/Chat/test_console_agent_bridge.py Tests/Agents/test_agent_runtime.py Tests/Agents/test_provider_continuation_runtime.py Tests/Chat/test_provider_continuation_privacy.py Tests/Chat/test_console_agent_bridge_cancel_all.py Tests/Chat/test_chat_persistence_service.py Tests/Chat/test_console_generation_card.py Tests/Canvas/test_repository.py -q
762 passed, 1 warning in 88.94s
```

Fresh exact production/postcommit probes:

```text
../../.venv/bin/pytest -q Tests/Chat/test_console_chat_controller.py::test_real_agent_composition_advertises_and_invokes_shared_canvas_owner Tests/Chat/test_console_runtime_lifetime.py::test_runtime_composes_one_shared_canvas_owner_into_real_store Tests/Chat/test_console_dispatch_recovery.py::test_post_commit_dispatch_owner_change_reconciles_terminal_message Tests/Chat/test_console_dispatch_recovery.py::test_store_retains_canvas_stage_when_terminal_transaction_fails
4 passed, 1 warning in 1.58s
```

Fresh controller/service/repository branch and transaction verification after the owner-global sequence correction:

```text
../../.venv/bin/pytest -q Tests/Chat/test_console_canvas_controller.py Tests/Canvas/test_service.py Tests/Canvas/test_repository.py
125 passed, 1 warning in 31.35s
```

Fresh final post-self-review regression command (including the promotion-lease replay fence and exact production/postcommit probes):

```text
../../.venv/bin/pytest -q Tests/Chat/test_console_canvas_controller.py Tests/Agents/test_canvas_tool_provider.py Tests/Canvas/test_repository.py -k 'canvas or transaction_append' Tests/Chat/test_console_chat_controller.py::test_real_agent_composition_advertises_and_invokes_shared_canvas_owner Tests/Chat/test_console_runtime_lifetime.py::test_runtime_composes_one_shared_canvas_owner_into_real_store Tests/Chat/test_console_dispatch_recovery.py::test_post_commit_dispatch_owner_change_reconciles_terminal_message Tests/Chat/test_console_dispatch_recovery.py::test_store_retains_canvas_stage_when_terminal_transaction_fails
156 passed, 1 deselected, 1 warning in 12.68s
```

The warning in each run is the environment's existing `requests` dependency-version warning. No full-suite run was performed under repository policy.

### Transaction and settlement correction

For a durable turn, the existing Console transaction first reserves/updates the exact assistant anchor, then calls the canonical Canvas batch validator/writer, then deletes the dispatch checkpoint. Only after the repository returns COMMITTED does the store synchronously confirm the exact opaque run settlement; UI and recovery publication follows. A postcommit publication/recovery-owner fault therefore reconciles from durable message/cards and finds the Canvas stage already COMMITTED with source retired. A precommit message, validation, quota, revision, or trigger failure rolls back the complete caller transaction and leaves the exact frozen stage READY.

For a temporary turn, source stays only in incarnation-owned committed history. Promotion reserves every durable message ID first, remaps typed card origins with that mapping, and supplies the same mapping to the canonical Canvas contribution. The promotion lease fences the exact owner/run snapshot until confirm, abort, or retire; no same-ID replacement or late callback can delete or settle another incarnation.

### Source-leak sentinel evidence

- The final 762-test run retains all prior AgentStep/log/cycle/continuation/transcript sentinel probes.
- `test_promotion_remaps_card_and_revision_origins_together` promotes and rehydrates a Canvas-only empty assistant anchor, checks the card origin equals the durable revision origin, and proves `private origin source` is absent from stored metadata JSON.
- Canonical batch objects keep complete source only in `repr=False` contribution fields. Cards, settlement JSON, durable acceptance fingerprints, and transaction diagnostics remain identity/digest/count only.

### Files changed in this correction

- `tldw_chatbook/Canvas/repository.py`
- `tldw_chatbook/Canvas/service.py`
- `tldw_chatbook/Chat/console_canvas_controller.py`
- `tldw_chatbook/Chat/console_chat_controller.py`
- `tldw_chatbook/Chat/console_chat_store.py`
- `tldw_chatbook/Chat/console_runtime.py`
- `tldw_chatbook/Chat/console_transaction_contribution.py`
- `tldw_chatbook/Chat/message_metadata.py`
- `tldw_chatbook/DB/ChaChaNotes_DB.py`
- `Tests/Canvas/test_repository.py`
- `Tests/Chat/test_console_canvas_controller.py`
- `Tests/Chat/test_console_chat_controller.py`
- `Tests/Chat/test_console_dispatch_recovery.py`
- `Tests/Chat/test_console_runtime_lifetime.py`
- `Tests/Chat/test_message_metadata.py`

### Self-review

- Confirmed the real runtime, store lifecycle, promotion participant, per-run provider, and bridge terminal callback all share one root controller, while provider operations remain permanently fenced to the opaque run handle.
- Confirmed every promotion callback compares exact session owner, lease, generation, and contributed run tuple; no callback uses only a reusable session/run string to remove another incarnation.
- Confirmed durable postcommit confirmation precedes every mutable publication point and reconciliation repeats the same idempotent exact confirmation; true database failures retain READY.
- Confirmed temporary sibling origins cannot be listed, read, or updated, while descendant and switched branches resolve their reachable heads. Historical durable branching uses the owner-global next sequence while retaining the chosen historical parent.
- Confirmed the canonical transaction API counts pre-existing durable rows and validates the complete batch before insert. Injected trigger rollback includes a caller update, both Canvas rows, and the document; the two-connection duplicate-sequence race commits one writer and boundedly rejects the other.
- Confirmed typed metadata remapping preserves unrelated fields and never serializes source. Card and revision origins agree after promotion/restart.
- Confirmed the existing nominal Canvas approval authority and source-free provider projections were not broadened.

### Concerns

- Repository policy forbids an unrequested full-suite sweep; verification is limited to the original 722-test focused baseline, the new canonical repository probes, branch-aware service/controller tests, and exact production/postcommit tests.
- An exploratory broader atomic-promotion file still contains a pre-existing unrelated failure where `test_promotion_persists_sparse_context_policy_inside_the_bundle` deliberately makes `_flush_context_policy_on_first_persist` raise even though the baseline implementation still calls it postcommit. This correction neither introduced nor modifies that context-policy path.

## Round 2/5 independent-review fixes (2026-09-04)

### Implementation

- A failed Canvas run now releases assistant ownership only when the assistant map still names that exact run ID. The bounded discarded settlement remains queryable, while a retry can register a fresh opaque run owner against the same assistant ID. Pruning, session retirement, promotion confirmation, and shutdown use the same compare-and-pop helper, so an old run can never erase its replacement's ownership.
- Temporary `list`, `read`, and `update` resolve exclusively from the incarnation-owned branch-filtered committed overlay plus the current run. They never ask `CanvasService` to resolve a native ephemeral conversation ID; an absent reachable temporary base returns bounded `canvas_base_unavailable` without mutation.
- `ambiguous_ancestry` is now an explicit member of the provider's bounded conflict-result contract at both initial encoding and every display/log/cycle/continuation projection. Unknown conflict codes still fail closed as `operation_failed`.
- Temporary sequence calculation now visits only matching Canvas rows in each committed stage. An unrelated committed stage with no row for the target Canvas contributes nothing and cannot trigger the former empty-variadic `max` failure.
- Promotion retains the typed `MessageMetadata.remap_canvas_origins(...)` result prepared for each row and publishes that same typed object to the live message in the exact postcommit identity-publication loop. A later feedback/content write therefore reuses durable Canvas-card origins instead of overwriting the committed metadata with native IDs.
- Successful retries of an already-durable failed assistant use a new exact generation-projection contribution seam. The assistant generation, remapped card metadata, and canonical Canvas batch share the caller-owned SQLite transaction. The exact settlement is confirmed only after that transaction returns committed and before mutable generation sidecars/publication; a transaction failure leaves the frozen stage READY.

### Exact RED/GREEN evidence

The five review probes were added before their production fixes. The first aggregate RED was:

```text
../../.venv/bin/pytest -q Tests/Chat/test_console_chat_controller.py::test_real_canvas_controller_allows_exact_failed_assistant_retry Tests/Chat/test_console_canvas_controller.py::test_production_temporary_scope_never_queries_nonexistent_durable_owner Tests/Chat/test_console_canvas_controller.py::test_temporary_sequence_ignores_other_committed_canvas_rows Tests/Chat/test_console_canvas_controller.py::test_promotion_remaps_card_and_revision_origins_together Tests/Agents/test_canvas_tool_provider.py::test_ambiguous_ancestry_conflict_survives_every_bounded_projection
6 failed, 1 warning in 2.60s
```

Four failure classes directly exposed the production gaps: temporary service fallthrough, an empty-variadic sequence `TypeError`, live native-origin metadata overwriting the durable remap, and ambiguity being downgraded to `operation_failed`. The retry probe initially selected the appended failure SYSTEM row rather than the failed ASSISTANT row; after correcting that test harness, its exact RED was:

```text
../../.venv/bin/pytest -q --tb=short --show-capture=no Tests/Chat/test_console_chat_controller.py::test_real_canvas_controller_allows_exact_failed_assistant_retry
2 failed: RuntimeError: canvas_assistant_owner_already_registered
```

The same aggregate command after the production correction:

```text
6 passed, 1 warning in 1.56s
```

The retry proof was strengthened after GREEN to verify that a failed Canvas attempt contributes no durable revision, the successful attempt contributes exactly its own source, its revision origin is the durable assistant ID, and neither attempt source appears in message metadata:

```text
../../.venv/bin/pytest -q --tb=short --show-capture=no Tests/Chat/test_console_chat_controller.py::test_real_canvas_controller_allows_exact_failed_assistant_retry
2 passed, 1 warning in 1.10s
```

Fresh prior broad Task-3.3 gate:

```text
../../.venv/bin/python -m pytest Tests/Chat/test_console_canvas_controller.py Tests/Chat/test_console_dispatch_recovery.py Tests/Chat/test_message_metadata.py Tests/Agents/test_canvas_tool_provider.py Tests/Chat/test_console_agent_bridge.py Tests/Agents/test_agent_runtime.py Tests/Agents/test_provider_continuation_runtime.py Tests/Chat/test_provider_continuation_privacy.py Tests/Chat/test_console_agent_bridge_cancel_all.py Tests/Chat/test_chat_persistence_service.py Tests/Chat/test_console_generation_card.py Tests/Canvas/test_repository.py -q --tb=short --show-capture=no
766 passed, 1 warning in 75.00s
```

Fresh production composition/retry/promotion/projection proof:

```text
../../.venv/bin/pytest -q --tb=short --show-capture=no Tests/Chat/test_console_chat_controller.py::test_real_agent_composition_advertises_and_invokes_shared_canvas_owner Tests/Chat/test_console_chat_controller.py::test_real_canvas_controller_allows_exact_failed_assistant_retry Tests/Chat/test_console_runtime_lifetime.py::test_runtime_composes_one_shared_canvas_owner_into_real_store Tests/Chat/test_console_canvas_controller.py::test_promotion_remaps_card_and_revision_origins_together Tests/Agents/test_canvas_tool_provider.py::test_ambiguous_ancestry_conflict_survives_every_bounded_projection
6 passed, 1 warning in 1.89s
```

The mutation inventory initially reported the existing Canvas hard-purge SQL route and the new generation-contribution DB boundary as unclassified. After adding exact classifications, the two census tests produced `2 passed, 4 warnings in 38.93s`. Fatal Ruff (`E9,F63,F7,F82`), `py_compile`, and `git diff --check` all pass.

The final post-report verification combined production composition, both retry variants, runtime identity, temporary durable-service isolation, A/B/A sequencing, promotion/restart, and ambiguity projection: `8 passed, 1 warning in 2.15s`; fresh fatal Ruff, `py_compile`, and `git diff --check` then returned zero errors.

An additional generation/promotion slice produced `47 passed, 25 deselected` plus the same known unrelated baseline failure `test_promotion_persists_sparse_context_policy_inside_the_bundle`; the failing postcommit context-policy hook and its caller are unchanged from reviewed parent `74c26cc5d0` and were already recorded under Round 1 concerns. No full suite was run.

### Transaction and settlement model

Fresh durable turns continue through dispatch settlement. Retried durable assistants now take the equivalent existing-row path: generation replacement runs inside the DB's semantic-mutation transaction; the transaction callback writes exact contributions through `_scoped_console_transaction_writer`, mapping the native assistant ID to its existing durable ID; card metadata is written in the same message update. Once the DB context returns, the persistence service confirms the exact opaque Canvas settlement before returning to mutable generation reconciliation. Any message, ancestry, quota, sequence, or contribution error unwinds the shared transaction and never runs confirmation, retaining READY for retry.

Discarded runs remain bounded records keyed by run ID, but no longer occupy the assistant-registration map. A newly registered retry owns the assistant key. Every release is an exact compare-and-pop, and every callback supplied by a provider is additionally fenced by its opaque `CanvasRunOwner`; a late callback from the discarded run therefore observes the replacement owner and returns inertly.

Temporary history remains incarnation-owned and branch-authorized. Promotion prepares one immutable native-to-durable ID map, uses it for both revision contributions and typed card metadata, and publishes the typed remap only after commit under the existing reservation identity check.

### Source-leak sentinel evidence

- Provider ambiguity tests project the result through immediate, AgentStep, log, cycle, and continuation paths while retaining the existing source sentinel assertions.
- The production retry proof reads the one committed revision source to show attempt isolation, then asserts neither failed nor successful exact source occurs in durable message metadata.
- The promotion/restart proof performs a later feedback write after promotion, rebuilds the conversation through persisted hydration, verifies card origin equals revision origin, and asserts the source sentinel is absent from stored metadata.

### Files changed in this correction

- `tldw_chatbook/Agents/canvas_tool_provider.py`
- `tldw_chatbook/Chat/chat_persistence_service.py`
- `tldw_chatbook/Chat/console_canvas_controller.py`
- `tldw_chatbook/Chat/console_chat_store.py`
- `tldw_chatbook/DB/ChaChaNotes_DB.py`
- `Tests/Agents/test_canvas_tool_provider.py`
- `Tests/Chat/test_console_canvas_controller.py`
- `Tests/Chat/test_console_chat_controller.py`
- `Tests/Chat/test_console_semantic_mutation_inventory.py`

### Self-review

- Verified assistant ownership is released only for the exact retired run and that an old run-fenced coordinator cannot settle or mutate the replacement.
- Verified temporary service isolation does not relax active-path or selected-revision authority; sibling branch tests remain green.
- Verified the retry contribution writes through the canonical Canvas batch validator inside the existing semantic generation transaction, with no nested Canvas transaction and no ID publication before commit.
- Verified postcommit confirmation precedes metadata/sync sidecars and DB failure paths do not call it.
- Verified promotion assigns only the typed remapped metadata prepared for that exact native message and preserves unrelated metadata fields.
- Verified conflict projection still rejects every code outside the two closed, reviewed conflict values.

### Concerns

- Repository policy forbids an unrequested full-suite sweep; verification stayed on the focused 766-test gate, exact new regressions, generation persistence, production composition, and static mutation inventory.
- The unrelated sparse-context-policy atomic-promotion baseline failure remains unchanged and out of scope, as documented above.

## Round 3/5 independent-review fix (2026-09-04)

### Implementation

- Existing-assistant generation settlement now carries an explicit `update_metadata` decision independently of the serialized metadata value. A typed empty `MessageMetadata` is serialized and atomically replaces a stale Canvas card; omitted metadata remains `None` with `update_metadata=False`, preserving compatibility for callers that do not intend a metadata write.
- The Console store merges Canvas settlement into the assistant's typed metadata before the transaction, so replacing Canvas cards with an empty tuple preserves unrelated fields such as `engine`. When that metadata participates in the generation/contribution transaction, the store records that fact and skips the redundant postcommit metadata sidecar.
- The real retry integration proof now covers six cases: neither attempt uses Canvas, only the successful retry uses Canvas, only the failed attempt uses Canvas, the same stale-card clearing while preserving unrelated metadata, both attempts use Canvas, and an injected true generation-transaction failure.

### Exact RED/GREEN evidence

The initial expanded retry cross-product appeared green because `_persist_metadata_only` accidentally repaired the stale durable card after the generation transaction. The regression was therefore strengthened to make that postcommit sidecar raise. Before the production fix, the exact test produced:

```text
../../.venv/bin/pytest -q --tb=short --show-capture=no Tests/Chat/test_console_chat_controller.py::test_real_canvas_controller_allows_exact_failed_assistant_retry
2 failed, 3 passed, 1 warning
RuntimeError: postcommit metadata sidecar unavailable
```

After introducing the explicit metadata-write contract and atomic empty replacement, then adding the true-DB-failure variant:

```text
../../.venv/bin/pytest -q --tb=short --show-capture=no Tests/Chat/test_console_chat_controller.py::test_real_canvas_controller_allows_exact_failed_assistant_retry
6 passed, 1 warning in 2.37s
```

The direct persistence contract proves that omitted metadata preserves an existing unrelated field while an explicit typed empty value clears it:

```text
../../.venv/bin/pytest -q --tb=short --show-capture=no Tests/Chat/test_chat_persistence_service.py::TestChatPersistenceService::test_generation_contribution_metadata_distinguishes_omitted_from_empty
1 passed, 1 warning in 0.47s
```

Fresh affected aggregate, including retry, transaction-failure retention, promotion-origin, and provider coverage:

```text
../../.venv/bin/pytest -q Tests/Chat/test_chat_persistence_service.py Tests/Chat/test_console_chat_controller.py Tests/Chat/test_console_dispatch_recovery.py Tests/Chat/test_console_canvas_controller.py Tests/Agents/test_canvas_tool_provider.py -k 'generation_contribution_metadata_distinguishes_omitted_from_empty or real_canvas_controller_allows_exact_failed_assistant_retry or store_retains_canvas_stage_when_terminal_transaction_fails or promotion_remaps_card_and_revision_origins_together or ambiguous_ancestry_conflict_survives_every_bounded_projection'
111 passed, 76 deselected, 1 warning in 5.06s
```

Fresh prior broad Task-3.3 gate after the correction:

```text
../../.venv/bin/python -m pytest Tests/Chat/test_console_canvas_controller.py Tests/Chat/test_console_dispatch_recovery.py Tests/Chat/test_message_metadata.py Tests/Agents/test_canvas_tool_provider.py Tests/Chat/test_console_agent_bridge.py Tests/Agents/test_agent_runtime.py Tests/Agents/test_provider_continuation_runtime.py Tests/Chat/test_provider_continuation_privacy.py Tests/Chat/test_console_agent_bridge_cancel_all.py Tests/Chat/test_chat_persistence_service.py Tests/Chat/test_console_generation_card.py Tests/Canvas/test_repository.py -q --tb=short --show-capture=no
767 passed, 1 warning in 76.20s
```

Final post-report verification reran the six-case real retry matrix plus the direct omitted/empty persistence contract: `7 passed, 1 warning in 2.73s`. Fatal Ruff (`E9,F63,F7,F82`), `py_compile` for all four changed Python files, and `git diff --check` all returned zero errors.

### Transaction and settlement model

On a successful retry, the existing assistant row, its complete generation projection, the full typed metadata replacement, and any canonical Canvas contribution are one caller-owned SQLite transaction. The explicit empty replacement is therefore durable before the exact Canvas run confirms COMMITTED, and no mutable postcommit metadata publication is needed to clear the previous discarded card. A true database failure rolls back the message and contributions together, does not confirm the stage, and leaves the exact run READY; the test then retries persistence successfully. The failed attempt's opaque callback remains inert after the new run owns the assistant.

The change does not globally erase metadata. `_persist_terminal_generation` first merges the Canvas settlement field into the existing typed object, retaining unrelated fields; only the Canvas card tuple becomes empty. Callers that omit metadata still request no metadata update.

### Source-leak sentinel evidence

- The six-case real retry test asserts that source from the failed attempt never enters `canvas_revisions`, the successful attempt is the only possible durable source, and neither source appears in message metadata.
- Failed-Canvas followed by zero-Canvas success has no Canvas card in either the live assistant or a freshly hydrated store, and has zero durable revisions.
- The injected database failure retains only the bounded discarded card from the already-persisted failed attempt, zero revisions, and a READY retry settlement; the later successful persistence clears it atomically.

### Files changed in this correction

- `tldw_chatbook/Chat/chat_persistence_service.py`
- `tldw_chatbook/Chat/console_chat_store.py`
- `Tests/Chat/test_chat_persistence_service.py`
- `Tests/Chat/test_console_chat_controller.py`

### Self-review

- Verified `update_metadata=True` is chosen from presence of typed metadata, not JSON truthiness, so an intentional empty replacement is representable.
- Verified the no-metadata path remains `metadata_json=None` and `update_metadata=False`.
- Verified unrelated typed metadata survives the stale-card clearing transaction and restart hydration.
- Verified postcommit Canvas confirmation ordering is unchanged and a precommit DB exception retains READY semantics.
- Verified stale callbacks use their original opaque run binding and cannot affect the replacement run.

### Concerns

- Repository policy forbids an unrequested full-suite sweep; verification remains limited to the focused Task-3.3 gate and exact retry/persistence regressions.
- The existing unrelated sparse-context-policy promotion baseline failure remains unchanged and out of scope, as documented in Round 1.
