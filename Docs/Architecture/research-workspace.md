# Research workspace

This document describes the Research destination: the workspace screen (sources, grounded chat, studio), the LOCAL/SERVER authority split, the three-stage durable source-operation receipts, quick notes with owner proofs, and the separately-routed research **runs** engine.

## Route naming (read this first)

The shell destination id is `research`, but its **primary route is `research_workspace`** (the Sources/Chat/Studio screen). The literal route `research` is a *different* screen — the Runs surface (run/event/checkpoint/bundle observation). Palette aliases ("research workspace", "research runs", "deep research", "notebook") all sit under the one destination.

## Authoritative files

| Concern | File | Key symbols |
| --- | --- | --- |
| Workspace screen | `UI/Screens/research_workspace_screen.py` + `UI/Research_Workspace_Modules/` | `ResearchWorkspaceScreen` — panes `sources / chat / studio`; intake modal, source list/receipts/inspector, conflict modals |
| Runs screen | `UI/Screens/research_screen.py`, `UI/Research_Window.py`, `UI/Research_Modules/research_controller.py` | `ResearchWindow` run lifecycle operator UI; thin async controller over the scope service |
| Contracts | `Research_Workspace/contracts.py` | `WorkspaceDataSource` LOCAL/SERVER (server refs require a profile id; local refs may not carry identity), `SourceReadinessState` vocabulary, `ResearchWorkspacePort` Protocol, `require_capability` (fail-closed with reason codes), `MAX_RESEARCH_SELECTION_IDS = 10_100` |
| Surface state | `Research_Workspace/controller.py` | `ResearchWorkspaceController` — data-source selection, catalog staleness generations, capability gating, source attach/remove/reorder, canonical result validation |
| Authorities | `Research_Workspace/local_adapter.py`, `server_adapter.py` | `LocalResearchWorkspaceAdapter` (workspaces registry + notes scope + media reading scope), `ServerResearchWorkspaceAdapter` (server notes workspace service + context provider) |
| Source operations | `Research_Workspace/source_operations.py`, `source_operation_store.py`, `source_association.py`, `source_readiness.py` | the durable receipt contract (see below), the SQLite store (optimistic revisions, idempotency-key dedup), the association coordinator + scheduler, the readiness coordinator |
| Quick notes | `Research_Workspace/quick_notes.py` | save requests with app-minted operation ids, receipt validation at the authority boundary, provenance-as-keywords encoding, conflict errors |
| Presentation | `Research_Workspace/overlay_store.py`, `paste_staging.py`, `layout_state.py`, `source_urls.py` | per-user overlay JSON (folders/annotations/pane prefs); private paste staging (2 MiB/artifact, 100 artifacts); pane layout; http/https-only URL admission |
| Runs engine | `Research_Interop/` | `LocalResearchService` (SQLite sessions/runs/events/artifacts/checkpoints; lease + fencing per ADR-070), `LocalResearchEngine` (phases planning→collecting→synthesizing→packaging; reuses the deep-search pipeline), `ResearchScopeService` (LOCAL\|SERVER routing on a single-worker executor to preserve read-check-write ordering), budget ledger, chat handoff |
| Persistence | `DB/Workspace_DB.py` (schema v8; `research_source_operations` + quick-note receipts), `DB/ChaChaNotes_DB.py` (`research_quick_note_owner_proofs`, local-only) | see below |
| Wiring | `app_service_wiring.py`, `app.py`, `app_ingest_queue.py` | research service wiring; screen late-binding of both ports; held ingest dispatch/retry/reconcile helpers |

## Source admission (dataflow)

1. The intake modal produces a request (`url | file | paste | existing | catalog`). URLs are http/https only; `existing`/`catalog` attach immediately with an idempotency key.
2. Every other kind mints a durable `ResearchSourceOperation` receipt **before any ingest work** (operation id + idempotency key, both rejected if they contain secret/path shapes). Paste bytes stage privately first.
3. A durable ingest job is created with `research_source_operation_id` and `dispatch_held=True` — the generic Library queue selector never dispatches it (see [library-shell.md](./library-shell.md)).
4. The receipt advances CATALOG → in_progress with the job id (optimistic revision; races re-read and accept exact-authority matches).
5. Dispatch releases the hold: local origin tops up the parse pool; server origin submits to the server ingest API. A retry released from hold is atomically claimed as PARSING so it can never fall back to the generic selector.
6. On ingest completion the association coordinator validates the outcome (local needs a numeric media id; server needs a remote id), links membership (local) or saves the workspace source (server, with a derived stable source id and row-shape verification), and runs selection reconciliation — implicit selection means "all sources", and malformed state starts explicit-empty so repair can never widen retrieval.
7. The readiness coordinator requires exactly one readiness row matching the operation; any mismatch fails the stage with an identity error.
8. `complete` = all three stages (catalog → association → readiness) succeeded.

Startup reconciliation fans out three workers: held-job linking/dispatch, association/readiness resume, and a paste-staging sweep. Quick-note reconciliation claims receipts with lease tokens and reconciles against ChaChaNotes rows plus owner proofs (proof mismatches block permanently). Failures are isolated and logged by exception class only.

## Quick notes (write path)

Local saves hold a note-write lock, reconcile receipts first, then write through the notes scope with provenance encoded as base64 keywords. The receipt machine walks `pending → owner_committed → projection_committed` (+ `blocked`), each transition revision-checked and CHECK-floored. Owner proofs (64-hex) live in a **deliberately local-only** ChaChaNotes table with no sync/FTS/export triggers; legacy `research-receipt-proof:` keywords are filtered from visible tags forever. Stale optimistic updates raise a conflict error surfaced as a modal; a dirty editor on owner switch flushes first, with a recovery modal (retry/discard/cancel — cancel keeps the draft and blocks the switch).

## Runs lifecycle

The Runs screen creates a run through the scope service; the local engine executes in an exclusive worker group per run, persisting progress/events/artifacts and a budget ledger, honoring pause/cancel **between phases**, and on completion optionally inserting an assistant message via the chat handoff (which never raises — failure degrades to a notification). Outputs land in the research DB and as chat messages — not as Library media or workspace quick notes. The server side declares session CRUD and filtered run lists unsupported; see the gotcha below about run deletion.

`ResearchWindow` owns the Runs selection, detail, action target, and loaded
bundle/artifact/event payload. Its native two-second interval refreshes only a
selected nonterminal Local run. A read captures the source and window selection
revision before awaiting the existing scope-service executor; it may publish
only while both remain current. Selecting another row, applying a selected-run
update, or switching sources advances that revision, including an away-and-back
transition to the same source/run. An accepted refresh updates progress without
resetting payload state. Terminal and Server selections keep their skip guards.
Switching sources clears selection and payload and immediately paints the empty
run detail, even when the new catalog is empty or unavailable; a later valid
selection paints its own detail. `ResearchController.get_run` retains its
fetched-result cache contract; that cache does not select the window or supply
the current action ID. Mounted regression coverage lives in
`Tests/UI/test_research_run_refresh_publication.py` (TASK-37267). These rules apply
to the separately routed **Runs** screen, not the Workspace source controller.

## Boundaries

- LOCAL and SERVER catalogs are never silently blended, synchronized, or substituted — identity rules are enforced in the contracts and in the receipts' SQL CHECKs.
- Every adapter action passes fail-closed `require_capability` (typed reason codes, recovery actions).
- Operation-store writes run off the UI loop with optimistic revisions; races raise typed conflicts.
- Research runs are a **different subsystem** from workspace sources (`Research_Interop/` vs `Research_Workspace/`) — and `Research_Interop/research_source_catalog.py` is the *academic discovery* catalog for the runs lane, not workspace sources.

## Failure behaviors

| Case | Behavior |
| --- | --- |
| Paste staging unavailable | Receipt fails at CATALOG with a staging reason; intake continues |
| Association failure | Catalog item persists; stage fails with user copy "Catalog item saved, but workspace association failed." |
| Server context changed mid-operation | Receipt fails `server_context_changed`; nothing silently re-targeted |
| Startup resume exception | Isolated per worker; logged by class; deferred, not fatal |
| Quick-note proof mismatch | Blocked permanently with a reason |
| Retry unavailable (job superseded/permanent) | Honest "retry unavailable" notice |

## Governing decisions

ADR-068 (`068-local-research-execution-engine.md`), ADR-070 (`070-research-run-lease-and-durability.md`), ADR-078 (`078-research-workspace-authority-and-screen-boundaries.md`, amends ADR-015/028). Specs: `Docs/superpowers/specs/2026-08-23-research-workspace-design.md`, `2026-08-18-durable-research-jobs-design.md` (+ the plan series under `Docs/superpowers/plans/`). Config: `research_db_path` for the runs DB.

## Verified gotchas

1. The route naming trap: destination `research` → route `research_workspace`; route `research` is the Runs screen.
2. **Server run-delete contradiction**: the scope contract declares server run deletion unsupported while the server service advertises `supports_run_delete = True` — flag before relying on either.
3. Local quick-note search scans bounded membership pages (no cross-database FTS join exists).
4. `release_dispatch_hold` auto-claims only *retries* as PARSING; first-time intake releases as QUEUED.
5. There is no watchlist tie-in anywhere in the research packages — don't assume one from the product copy.
6. The bulk-reader example (`Docs/Examples/agents/bulk-reader/`) is a Settings▸Agents delegation preset, not wired into this screen.
