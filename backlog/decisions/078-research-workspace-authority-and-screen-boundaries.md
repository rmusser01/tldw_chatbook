# ADR-078: Research Workspace authority and screen boundaries

- **Status:** Accepted
- **Date:** 2026-08-23
- **Last amended:** 2026-08-24 (private Local Quick Note recovery proof)
- **Task:** [TASK-21505](../tasks/task-21505%20-%20Design-Local-Server-Research-Workspace-and-Research-Runs-navigation.md)
- **Design:** [Research Workspace design](../../Docs/superpowers/specs/2026-08-23-research-workspace-design.md)
- **Amends:** ADR-015 (shell destination taxonomy); ADR-028 (adds Research
  contextual workspace quick actions while Settings remains the full manager)
- **Related:** ADR-005 (Console workspace server-readiness), ADR-027 (Default
  workspace chat grouping), ADR-028 (Settings workspace and folder-root
  ownership), ADR-029 (local private data), ADR-031 (TUI keybindings), ADR-043
  (explicit rail preferences versus responsive collapse), ADR-068 (local
  research engine), ADR-070 (research run durability)

## Context

Chatbook has two adjacent but different research concepts:

1. a durable Research Run system with lifecycle, checkpoints, events, bundles,
   and local/server execution; and
2. a desired NotebookLM-like workspace for selecting sources, asking grounded
   questions, taking notes, and generating durable outputs.

The existing `research` route is a real `ResearchScreen`, but ADR-015 folds it
under Library and gives it no shell destination. The server WebUI's Research
Workspace is much broader than that run screen and combines canonical server
state with browser-local folders, annotations, chat state, and other UI state.

Chatbook also has overlapping authority terms. `WorkspaceAuthority` describes
local/server materialization and sync state. The app-wide runtime source can
silently resolve an unconfigured Server request to Local. Neither is a safe
operator-controlled data-source selector for a screen that promises no silent
blending.

Finally, Chatbook already has canonical owners for workspaces, Library items,
Notes, conversations, Study records, Chatbooks, TTS results, and Research Runs.
Mirroring the server's workspace tables locally would create duplicate content
owners and conflict with the existing workspace registry and global browsing
model.

## Decision

### 1. One Research shell destination owns two separate screens

Add one top-level **Research** destination after Library. Its primary route is
the new `research_workspace`. The existing `research` route remains the real
ResearchScreen and is exposed as the destination's **Runs** mode.

Both screens mount a shared Workspace/Runs mode bar and navigate between real
routes. Neither embeds or replaces the other. Saved and programmatic
`research` routes continue to open ResearchScreen.

This amends ADR-015 from 13 to 14 shell destinations. Existing destination
shortcuts become stable mappings keyed by destination ID rather than positions;
all current shortcuts remain unchanged and Research receives F10. The command
palette still exposes one command per destination.

### 2. Workspace data source is screen-local and fail-closed

Introduce `WorkspaceDataSource.LOCAL | SERVER` for Research Workspace. The UI
labels it `Workspace data: Local | Server`. It
selects the complete catalog, identifiers, persistence, mutation, retrieval,
chat, generation, and capability adapter.

It is distinct from `WorkspaceAuthority` and from the app-wide runtime source.
If the selected adapter is missing, unreachable, unauthorized, or unsupported,
the selected source remains visible with recovery actions. No operation falls
back to the other adapter.

Every async request captures qualified authority, server/profile, principal,
workspace, capability revision, and controller revision. Late results cannot
repaint another context.

### 3. Data authority and processing location are shown separately

Local data may use on-device or cloud inference. The UI therefore shows the
effective processing route rather than implying that Local means on-device.
The first remote use of local source bodies for an exact workspace/provider/
endpoint/redaction combination requires explicit preflight consent. Consent is
invalidated when that route changes.

Server data uses server retrieval, chat, and generation APIs. Chatbook does not
silently pull server content into a client-selected inference path.

### 4. Canonical stores remain content owners

Research Workspace is an orchestration and presentation surface. Local
workspace membership associates canonical records without copying their
payloads:

- Library/Notes/files own source content;
- the conversation store owns local workspace chat;
- Notes owns Quick Notes;
- Study owns flashcard and quiz records;
- local Chatbooks own Summary, Report, and Compare artifacts;
- stable Local TTS history owns Audio Summary when available;
- Local Mind Map, Timeline, Slides, and Data Table remain unavailable until a
  working canonical owner/editor exists;
- Research Interop owns Research Runs.

The UI may normalize these as `WorkspaceOutputRef` values in memory. It does
not add a universal output database. If a future output has no safe canonical
owner, that storage decision requires its own ADR or an explicit amendment.

Server mode uses canonical server workspace, source, note, artifact, chat,
sharing, and operation APIs. Mind Map and Timeline content is owned by server
workspace artifacts. Data Table, Slides, and Audio Summary use their native
server table/presentation/stable-audio owner first and a payload-free workspace
artifact reference second; if no stable inspectable native owner is returned,
the action is unavailable. Client-downloaded bytes are never a persistence
receipt.

Research ingestion first creates or reuses an item in the selected authority's
general catalog and then associates its stable identity with the captured
workspace. Local uses a Library item plus `WorkspaceMembership(role=source)`;
Server uses a server Media item plus a server workspace-source row. The
qualified association intent is durable across navigation and restart. It may
never attach to the other authority or to the workspace visible when a late
completion happens to arrive.

The Library ingest-job store's schema v7 persists a `dispatch_held` eligibility
barrier for this two-owner transaction. Research preparation writes a held,
qualified queue row before returning; ordinary Library submissions remain
unheld. Queue selectors, runner top-up, and restart restore cannot dispatch or
prune a held row. Only after the matching source operation durably records its
exact ingest job in the catalog `in_progress` stage may the app durably release
the hold and dispatch through the selected Local or Server owner. Startup reads
one bounded page of held rows and reconciles each independently. Ambiguous or
transient operation-store answers retain both the hold and managed paste
staging; missing, terminal, or authority-incompatible receipts settle the job
durably before staging cleanup. This is an implementation of the existing
qualified-intent owner decision, not a new content owner or sync boundary.

A name-derived workspace keyword may be projected for search/display parity,
but it is not the association or authority boundary: names and tags are
editable and can drift. Removing a workspace association does not delete the
canonical item. If catalog ingestion succeeds and association or indexing
fails, the item remains in the general catalog and the failed stage is
independently retryable.

Local Quick Note creation is the narrow exception to create-then-associate
ordering because Notes and the workspace registry are independent SQLite
owners with no shared transaction. Before the canonical Notes write, WorkspaceDB
claims a row in the dedicated `research_quick_note_receipts` ledger. The
payload-free row binds every qualified authority axis, the Local workspace ID,
canonical Notes user/client, a strictly validated app-minted UUID-v4 operation
token, operation kind, deterministic owner-qualified Note UUID, random
owner-minted proof, lease/claim token and expiry, a separate durable abandonment
deadline, expected delete version, monotonic revision/timestamps, and bounded
sanitized retry state. Every holder mutation compares the exact claim token,
revision, and receipt identity; reclaim rotates the token before inspecting an
owner, so an expired or recreated same-ID claim cannot be advanced by an older
holder. Identity axes use an unambiguous length-prefixed encoding rather than
delimiter joining. The row never enters `workspace_memberships`, Console
context, or RAG scope, and it stores no title, body, tags, provenance, path, or
URL. Consequently, identical tokens or delimiter-shaped identities cannot
associate one user's or workspace's Note with another owner.

The canonical Notes row, user keywords/provenance, and a private hashed recovery
proof commit in one Notes-owner transaction. Notes schema v43 stores that proof
only in `research_quick_note_owner_proofs`, keyed by canonical Note ID with an
owner-bound foreign-key cascade and no sync, keyword, FTS, export, graph, or RAG
trigger. The narrow recovery seam can only add, verify, or remove an exact
Note/proof pair; it never lists or returns proof payloads. Create retry verifies
the exact canonical title, body, tags, provenance, qualified identity, and
private proof before advancing `pending` to `owner_committed`. One WorkspaceDB
transaction then adds the authoritative `WorkspaceMembership(role="note")` and
records `projection_committed`. Recovery next atomically removes the private
proof from the Notes owner, verifies its absence, and only then consumes the
exact receipt. Crashes at each boundary resume from the durable state.
Deterministic UUID existence or a caller token alone is never proof of owner
commit. Restart promotion requires the exact qualified receipt plus its private
proof and canonical owner invariants, except that a `projection_committed`
receipt may resume after its proof has already been removed.

The genuine Notes v42→v43 migration recognizes only the exact historical marker
shape: the case-sensitive `research-receipt-proof:` prefix followed by exactly
64 lowercase hexadecimal characters. It backfills those linked proofs into the
private table and removes only their keyword and note-keyword sync payloads and
rows in the same version-guarded transaction. Prefix-adjacent user tags,
including uppercase, non-hexadecimal, shorter, and longer values, retain their
rows, links, sync history, and ordinary Library visibility. New writes never
create an internal proof keyword, so Notes bodies, sync/export, list/search,
graphs, RAG, Research tags, and logs cannot observe an owner proof.

A pending create carries a short durable work lease and revision fence.
Reconciliation does not inspect or clear it while the lease is live. Lease
expiry makes the receipt eligible for a newly token-fenced recovery holder; it
does not prove abandonment. A missing owner records sanitized backoff and is
retained until the independently durable seven-day abandonment deadline, when
the current holder may remove it by token/revision CAS. A writer delayed beyond
the work lease may finish its Notes transaction; if another holder reclaimed
the receipt, the stale transition fails and the durable owner remains
recoverable rather than orphaned. Each receipt is isolated. Transient failures
record only a bounded reason code, failure count, and exponential retry time;
missing-owner retries do not become poison-blocked. Blocked/backoff rows are
filtered before the bounded SQL limit so one poison row cannot starve later
work. Startup processes one bounded owner-filtered global page, and workspace
listing also reconciles one bounded page. The proof, lease token, note payload,
tags, and provenance never enter logs, overlay state, or recovery copy.

Local Quick Note deletion records a durable receipt before the optimistic Notes
soft delete and binds its expected owner version. Reconciliation inspects the
canonical owner before projection cleanup: an active row at the exact expected
version is deleted again through the versioned owner; a changed or restored
active row blocks as a conflict and retains every projection; only an absent or
tombstoned owner permits cleanup. One WorkspaceDB transaction then removes
every membership role for that Note across all workspaces, removes matching
Note items from every stored RAG scope, and consumes the exact token-fenced
delete receipt. Other independently claimed receipts settle through their own
token/revision fences. This ABA-safe rule preserves the existing canonical-delete
semantics: a Local Note is one shared general-Library record, so deleting it
invalidates every workspace projection rather than pretending one workspace
owns a private copy. An interrupted cleanup resumes at startup even when the
Note no longer appears in the UI.

WorkspaceDB schema v6 keeps migration decisions fail-closed without inferring
ownership from a Note ID or title. Genuine v3→v4 history removes only the
explicit legacy `role="note_pending"` representation. V4→v5 replaces the
unreleased proof-less receipt ledger but preserves every membership, including
blank `research-note-*` rows, because WorkspaceDB cannot consult the independent
Notes owner to classify them safely. V5→v6 preserves those memberships and all
safe v5 receipt rows while adding abandonment and proof-cleanup recovery state.
Runtime abandonment compares parsed SQLite Julian instants rather than raw
timestamp text, so historical space-separated UTC values and runtime ISO/offset
values share the same exact seven-day boundary. No migration heuristic promotes
or deletes an ordinary membership.

### 5. Server folders and annotations are explicit device-only overlays

The server has no canonical Research Workspace folder or annotation APIs.
Chatbook stores those features, along with pane preferences and server
Deep-Research launch context, one bounded unsent Research Chat draft per
qualified workspace, and payload-free append-stage recovery receipts in one
private atomic device overlay keyed by data source, server/profile, principal,
and workspace ID. A successful canonical chat append clears the draft; sent
transcript bodies are never mirrored into the overlay.

The overlay does not create remote workspace records in the local registry and
is never represented as uploaded, shared, or cross-device state. UI copy says
`Device-only overlay — not uploaded, shared, or available on other devices.`

Filesystem folder roots governed by ADR-028 remain a different concept and use
different identifiers and labels.

### 6. Cross-authority transfer is manual Copy with receipts

V1 implements no background synchronization and no cross-authority Move. Copy
is an explicit preflight and confirmation flow over a frozen, versioned
manifest. It supports Copy, Reference, Metadata only, Omit, and Blocked item
policies; conflict choices are explicit.

Execution is idempotent and resumable under a transfer ID. Durable receipts
record per-stage and per-item outcomes and end as Completed, Partially
completed, Rolled back, Failed-retryable, or Failed-terminal. The selected data
source never changes as a side effect of Copy.

### 7. Research Runs retain lifecycle ownership

Workspace may create and associate a Research Run and persist a launch/return
context. Runs remains the lifecycle, checkpoint, event, lease, artifact, and
bundle screen.

Returning a completed bundle is an explicit, validated import that creates a
draft output/reference in the originating workspace. Nothing is inserted
automatically, and repeated imports are idempotent unless the user explicitly
creates a new version.

### 8. Settings remains the full local workspace manager

Research offers contextual local switch, create, rename, duplicate, archive,
and restore actions. Full local management, destructive deletion, and
filesystem-root binding remain in Settings under ADR-028. Server lifecycle
actions are exposed in Research where the server authorizes them.

The built-in Default workspace remains the everyday-chat context from ADR-027
and is not presented as a research notebook.

### 9. Studio uses progressive disclosure

The primary Studio surface shows Summary, Flashcards, Quiz, Report, and Compare
Sources. Mind Map, Slides, Audio Summary, Timeline, Data Table, and work-product
templates live under `More outputs…` with per-authority capability, owner,
reason, and recovery.

The complete audited server namespace remains mapped, but owner links,
contextual menu actions, capability-gated actions, and planned labels do not
compete with the core source-to-answer-to-output loop.

### 10. Side-pane collapse follows the shared rail contract

Sources and Studio are independently collapsible. The exact visible labels are
`<---` to collapse Sources, `--->` to reveal Sources, `--->` to collapse
Studio, and `<---` to reveal Studio. Full textual accessible names and tooltips
describe the action.

As in ADR-043, stored user preference is distinct from width-driven effective
collapse. Responsive layout may force panes closed but does not overwrite the
preference. Explicit toggles always produce a visible result, focus moves to a
surviving reveal control on collapse and into the revealed pane on expansion,
and hidden panes leave the focus cycle. At medium width an explicit reveal
switches the companion pane without rewriting wide-layout preferences. At
narrow width the single-pane mode strip is the equivalent reveal mechanism.

## Alternatives considered

### Two top-level Research destinations

Rejected. It would expand the shell from 13 to 15 destinations, create two
near-identical labels, worsen compact navigation, and increase shortcut and
palette churn. Separate screens under one destination preserve the requested
separation without duplicating shell ownership.

### Keep both screens folded under Library

Rejected. The combined source-selection, grounded-chat, output-generation, and
durable-run workflow has enough independent purpose and persistent state to
justify one Research destination. Library remains the global content owner.

### Reuse the app-wide Local/Server runtime source

Rejected. It is app-global and may silently resolve an unconfigured server to
Local, violating the workbench's authority promise.

### Reuse `WorkspaceAuthority` as the selector

Rejected. That enum describes materialization and sync states such as conflict,
detached, and remote-only; overloading it with an operator's current data source
would collapse two different contracts.

### Mirror the server workspace schema locally

Rejected. Chatbook already has canonical local content owners and a generic
membership registry. A mirror would create duplicate Notes, conversations,
Study, and artifact state and complicate global Library/Artifacts visibility.

### Persist all Studio outputs in a new generic output store

Rejected. The primary output types already have viable canonical owners or
owner extension seams. A universal store is speculative and would become a
second Artifacts system.

### Treat folders and annotations as server state

Rejected. No canonical server API supports that claim. Honest device-only
overlay labeling preserves the useful WebUI behavior without fabricating sync.

### Use a workspace tag as the source relationship

Rejected. Human-readable tags and workspace names are mutable, may not be
renamed together, and cannot safely carry authority or deletion semantics.
Stable local membership/server workspace-source identities are authoritative;
tags are optional projections.

### Attach from an in-memory screen callback after ingestion

Rejected. Library/Media ingestion can outlive the visible screen or app
session. A durable qualified association target and app-level completion stage
prevent attachment to a newly visible workspace and support retry.

### Roll back the canonical item when association fails

Rejected. The general Library/Media item is independently useful and may be a
reused duplicate or belong to other workspaces. Preserve it and retry the
failed association or indexing stage.

### Automatically import completed Research bundles

Rejected. It crosses screen and authority boundaries, can duplicate artifacts,
and makes source/version mismatch recovery ambiguous. Explicit validated import
preserves control and provenance.

## Consequences

### Benefits

- Users get one recognizable Research destination without losing the existing
  run operator or saved route behavior.
- Local/Server authority is honest and cannot silently fall back or blend.
- Processing-route disclosure closes the privacy gap where Local data uses
  cloud inference.
- Existing stores remain canonical, reducing schema and reconciliation work.
- Ingested sources remain discoverable in the selected authority's general
  catalog while stable workspace associations provide eligibility and
  provenance.
- Complete server parity remains discoverable without overwhelming the primary
  flow.
- Side-pane controls share the app's tested preference, responsive-collapse,
  and focus behavior while using the requested compact ASCII labels.
- Device-only behavior is useful but cannot be mistaken for server sharing.
- Deep Research integrates without weakening its durable execution contract.

### Costs and constraints

- ADR-015's destination count and shortcut implementation must change.
- The server API client requires new Research Workspace endpoints before full
  Server parity is available.
- A private device overlay store is new persistence and needs corruption,
  privacy, bounds, export, and cleanup tests.
- Output history requires owner-specific resolvers rather than one simple
  generic table.
- Every asynchronous operation must carry qualified context and fencing.
- Ingestion receipts/coordinators must distinguish catalog, association, and
  readiness outcomes, and Local needs an explicit unlink operation that never
  deletes the Library item.
- The feature must ship in multiple independently testable tasks; no single PR
  should attempt the whole design.

### Follow-on rule

Implementation plans must name the canonical owner and capability behavior for
every output or action they activate. An advertised control without a real
owner, execution path, recovery path, and targeted test does not ship.
