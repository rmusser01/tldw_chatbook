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
| Persistence | `DB/Workspace_DB.py` (schema v12; source, Quick Note, and Copy receipts plus output membership metadata), `DB/ChaChaNotes_DB.py` (`research_quick_note_owner_proofs`, local-only) | see below |
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

## Grounded Chat

`ResearchChatRegion` drives workspace-scoped conversations with the `research-chat`
topic. `LocalResearchChat` uses `ChatConversationService` and
`ChatPersistenceService` for messages, membership, usage, and citation traces.
The Local retrieval pipelines receive an allowlist of captured selected Media
IDs. The provider gateway receives only accepted evidence, and remote excerpt
egress requires a matching fingerprint from `ResearchEgressConsentService`.

`ServerResearchChat` uses the workspace startup route, selected-Media RAG, and
server chat completion with an explicit workspace scope. It appends user and
assistant messages and citation context through the canonical APIs. The
device overlay keeps the draft, selected conversation ID, consent fingerprints,
and bounded append receipts so context persistence can be retried without
duplicate messages. Its layout compare retries after a Chat-only write when the
committed pane and source organization fields still match.

## Quick notes (write path)

Local saves hold a note-write lock, reconcile receipts first, then write through the notes scope with provenance encoded as base64 keywords. The receipt machine walks `pending → owner_committed → projection_committed` (+ `blocked`), each transition revision-checked and CHECK-floored. Owner proofs (64-hex) live in a **deliberately local-only** ChaChaNotes table with no sync/FTS/export triggers; legacy `research-receipt-proof:` keywords are filtered from visible tags forever. Stale optimistic updates raise a conflict error surfaced as a modal; a dirty editor on owner switch flushes first, with a recovery modal (retry/discard/cancel — cancel keeps the draft and blocks the switch).

On Ctrl+Q, the app's quit walk attempts the same Quick Note save. If there are
unsaved edits after the attempt, or an editor action is still pending, the
shared confirmation dialog focuses **Keep editing** and requires explicit
approval to discard the draft. Edits typed while the save is in flight remain
in the editor with the saved note's current version.

Local Quick Note deletion runs the canonical Notes write in a finite worker,
including during receipt recovery. It waits for the owner result before
cleaning up workspace memberships.

## Primary Studio

`ResearchStudioRegion` projects availability for Summary, Flashcards, Quiz,
Report, and Compare Sources from the selected workspace's capture of ready
Media sources and its processing route. Compare Sources requires two eligible
sources. `ResearchStudioGenerator` rechecks the captured sources and outbound
processing consent before using excerpts. Studio retains the captured route and
generation settings across a retry.

Local Summary, Report, and Compare Sources save to Chatbooks. Flashcards and
Quiz save to the canonical Study deck/cards and Quiz/questions stores. A
workspace output membership carries bounded provenance and refers to the owner
ID. WorkspaceDB v11 adds the membership metadata column. The saved-output
selector keys each item by its owner kind and ID. It can still inspect an output
after a source version changes and can remove a missing owner's Local link.

Server text generation uses the selected server's chat completion route with
chat persistence disabled. The canonical workspace artifact stores the output,
its captured source versions, and citation markers. Editing and replacement
check the owner version and return the content to draft. Markdown export
requires explicit review acceptance and uses the exact accepted artifact version
returned by the server. Server Flashcards and Quiz use the
`research-study.v1` workspace contract. One transaction commits the native
Study deck or quiz, its entries, and bounded provenance keyed by the captured
operation. A retry can recover the same owner after a lost response. Older
servers leave these output cards unavailable. Native Study owns content edits
and deletion. Studio can create a separate new version.

Studio blocks workspace and authority changes while generation is active. Stop
ends generation, and Retry uses the original operation. Study handoffs select
the exact saved deck or quiz by ID even beyond the first list page. Server
Study also checks the active server profile, account, and workspace before
consuming the handoff. The Artifacts handoff reuses the Library's canonical
Chatbook selection.

## Copy, export, and sharing

The workspace Actions menu opens Copy for the selected Local or Server
workspace. The preview freezes the source, destination, owner revisions,
treatments, size estimate, and conflict choices. A repeat Server-to-Local Copy
can find the previous destination in its qualified receipt and refresh the
current version through its canonical owner. Replacing an eligible Local
artifact requires a second confirmation. The owner checks that version before
writing. Copy currently has canonical transfer adapters for text artifacts in
both directions and Server notes into Local Notes. The preview reports other
owner types as blocked when their import contract is unavailable.

WorkspaceDB records each item checkpoint and the Server migration receipt.
Stopping finishes the current checkpoint so a later attempt can recover
acknowledged owners. Copy history can reopen completed and terminal receipts
alongside retryable operations. Local Export uses the same outbound policy and writes
reviewed text plus a Markdown manifest into a ZIP at the chosen path. Copy and
export omit device-only content. Reference and metadata entries contribute no
document content to the export.

Server sharing uses the captured profile, account, and workspace. Share offers
team and organization access, clone permission, password and limit settings for
links, active access editing and revocation, and recipient preview, verification,
import, and clone. A private receipt retains the clone key and canonical Server
operation ID for the selected account. The dialog reads status from the owner
after reopening, including for an operation admitted before a share was revoked.
Copy to Server and Share opens the separate Share dialog after
the Server migration is finalized and Copy reports completion.

## Runs lifecycle

The Runs screen creates a run through the scope service; the local engine executes in an exclusive worker group per run, persisting progress/events/artifacts and a budget ledger, honoring pause/cancel **between phases**, and on completion optionally inserting an assistant message via the chat handoff (which never raises — failure degrades to a notification). Outputs land in the research DB and as chat messages — not as Library media or workspace quick notes. The server side declares session CRUD and filtered run lists unsupported; see the gotcha below about run deletion.

Studio can launch Deep Research from the current selected sources or a saved
text output. The launch retains the origin, selected source versions, normalized
question, and initiating references. Local Runs stores that context on the
native run. Server uses `research-workspace-runs.v1`, which commits the origin,
native run, and first phase intent together. Device overlay v4 retains recent
launches for navigation. Workspace launches leave Chat completion delivery unset.

The Runs detail can return to the verified Workspace. Chatbook compares the
native backend, run ID, launch identity, profile and account, and the saved
bundle before showing the Report preview. The preview writes a draft through
the same Studio owner after confirmation. Reopening a saved import resolves the
original owner. **Create new version** saves a separate Report with lineage to
the original. Older Server versions leave the launch control unavailable.

**Load Bundle** selects the stored report first when one exists. Both that
read and **Load Artifact** use the scope's keyword-only `mode`, `run_id`, and
`artifact_name` contract. The scope dispatches Local reads to its executor.

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

ADR-315 (`315-local-research-execution-engine.md`), ADR-070 (`070-research-run-lease-and-durability.md`), ADR-322 (`322-research-workspace-authority-and-screen-boundaries.md`, amends ADR-015/028). Specs: `Docs/superpowers/specs/2026-08-23-research-workspace-design.md`, `2026-08-18-durable-research-jobs-design.md` (+ the plan series under `Docs/superpowers/plans/`). Config: `research_db_path` for the runs DB.

## Verified gotchas

1. The route naming trap: destination `research` → route `research_workspace`; route `research` is the Runs screen.
2. **Server run-delete contradiction**: the scope contract declares server run deletion unsupported while the server service advertises `supports_run_delete = True` — flag before relying on either.
3. Local quick-note search scans bounded membership pages (no cross-database FTS join exists).
4. `release_dispatch_hold` auto-claims only *retries* as PARSING; first-time intake releases as QUEUED.
5. There is no watchlist tie-in anywhere in the research packages — don't assume one from the product copy.
6. The bulk-reader example (`Docs/Examples/agents/bulk-reader/`) is a Settings▸Agents delegation preset, not wired into this screen.
