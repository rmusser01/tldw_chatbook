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
