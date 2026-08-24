# Research Workspace design

- **Status:** Proposed for user review
- **Date:** 2026-08-23
- **Task:** [TASK-21505](../../../backlog/tasks/task-21505%20-%20Design-Local-Server-Research-Workspace-and-Research-Runs-navigation.md)
- **Decision:** [ADR-078](../../../backlog/decisions/078-research-workspace-authority-and-screen-boundaries.md)
- **Classification:** Architectural

## Summary

Chatbook will add a NotebookLM-like Research Workspace for turning a bounded
set of sources into grounded conversations, notes, and durable outputs. It will
also make the existing durable research-run operator clearly reachable as a
separate Runs screen.

The shell gains one destination, **Research**, with two separately routed
screens:

- **Workspace** — new route `research_workspace`; Sources, Grounded Chat, and
  Studio/Quick Notes.
- **Runs** — existing route `research`; the existing `ResearchScreen` and
  `ResearchWindow` remain the run lifecycle, checkpoint, event, bundle, and
  artifact surface.

The workbench has an explicit, fail-closed Local/Server data-source selector.
Local and Server are separate catalogs and persistence authorities. They are
never silently blended, synchronized, or substituted for one another. Moving
content between them is an explicit Copy workflow with preflight, confirmation,
conflict handling, and a durable receipt.

## Approved product decisions

1. Use one top-level Research destination with separate Workspace and Runs
   screen tabs.
2. Keep `research` mapped to the existing real `ResearchScreen`; do not repoint
   saved routes to the new workbench.
3. Show the five primary Studio outputs first: Summary, Flashcards, Quiz,
   Report, and Compare Sources.
4. Put Mind Map, Slides, Audio Summary, Timeline, Data Table, and work-product
   templates behind `More outputs…`, grouped and capability-labeled.
5. Use explicit Local and Server catalogs with no background sync or fallback.
6. Show the effective processing route separately from data ownership so
   `Local` cannot imply that processing stays on-device.
7. Keep canonical content in existing owners. Use workspace memberships and
   presentation adapters rather than a second universal artifact database.

## Goals

- Provide persistent research workspaces with source selection, grounded chat,
  quick notes, and focused Studio generation.
- Preserve the complete audited tldw_server Research Workspace control
  namespace without giving every action equal visual prominence.
- Make data authority, processing location, source readiness, and blocked
  capabilities understandable before the user acts.
- Reuse Chatbook's local Library, Notes, RAG, conversations, Study, Chatbooks,
  TTS, workspace registry, and Research Interop services.
- Use the server's canonical workspace, source, notes, artifact, sharing,
  capability, migration, and diagnostic APIs in Server mode.
- Preserve keyboard-first operation, truthful footer help, deterministic focus,
  and useful layouts down to a 60-column terminal.
- Make Deep Research a durable, explicit launch-and-return relationship rather
  than merging the run engine into the workbench.

## Non-goals

- No background Local/Server synchronization or automatic hydration.
- No silent Local fallback when the selected server is unavailable.
- No cross-authority Move operation; v1 transfers are Copy only.
- No second Console: the embedded chat has no tools, approvals, or autonomous
  agent loop.
- No duplicate MCP, ACP, sandbox, provider, or Settings management surfaces.
- No new universal local artifact database.
- No claim that device-only folders or annotations are server-shared state.
- No activation of the server WebUI's planned Research Dossier, Competitive
  Market Memo, or Technical Project Spec templates.
- No change to Research Run execution, checkpoint, lease, or durability
  ownership established by ADR-068 and ADR-070.

## Terminology

### Research destination

The shell owner for Workspace and Runs. Its primary route is
`research_workspace`; its existing `research` route remains the Runs screen.

### Workspace data source

`WorkspaceDataSource.LOCAL | SERVER`, a screen-local selection that chooses the
catalog, identifiers, persistence owner, mutation path, and capabilities for
the workbench. It is not the app-wide runtime source and is not
`WorkspaceAuthority`, which already describes materialization/sync state.

### Processing route

The effective movement of selected content for one request, rendered in plain
language, for example:

```text
Sources: This device -> Inference: llama.cpp on this device
Sources: This device -> Inference: Anthropic cloud
Sources: Server example.org -> Retrieval + inference: Server example.org
```

### Device-only overlay

Folders, annotations, pane preferences, and server-workspace launch context
stored by this Chatbook installation. These values are not uploaded, shared,
or available on another device unless explicitly exported.

### Workspace output reference

A normalized presentation record resolving an output held by its canonical
owner. It is not a persistence owner.

## Information architecture and navigation

### Shell destination

Add a fourteenth `ShellDestination`:

| Field | Value |
| --- | --- |
| Destination ID | `research` |
| Label | `Research` |
| Primary route | `research_workspace` |
| Related route | `research` |
| Position | After Library, before Artifacts |
| New direct key | F10 |

Adding the destination amends ADR-015. Existing destination shortcuts must not
shift. Replace positional zip-based shortcut ownership with a mapping keyed by
destination ID. Home through Settings retain their current Ctrl+digit and
F7/F8/F9 bindings; Research receives F10.

The command palette continues to show one command per shell destination. The
Research command opens Workspace and indexes aliases including `research
workspace`, `research runs`, `research sessions`, `deep research`, and
`notebook`. Direct callers of `research` still mount `ResearchScreen`.

### Shared Research mode bar

Both screens mount the same compact mode bar:

```text
Research  [Workspace] [Runs]
```

The buttons navigate between real screen routes; neither screen embeds the
other. The shell remains active on either route. Route changes preserve each
screen's own state through its existing state-store contract.

### Relationship to neighboring destinations

- **Library** owns global browsing, ingestion, and editing of sources.
  Research Workspace attaches eligible Library items to one workspace.
- **Console** owns agent tools, approvals, autonomous work, and general live
  execution. Research Chat is evidence-focused workspace Q&A.
- **Artifacts** owns global browsing of supported generated artifacts.
  Research Studio is the creation and workspace-specific history surface.
- **Study** owns local flashcard and quiz records and study sessions.
- **Settings** owns full local workspace lifecycle and folder-root management.
- **MCP/ACP** own their server, tool, runtime, task, and session management.
- **Research Runs** owns durable deep-research execution and observation.

## Screen topology

### Destination header

The pinned header contains:

- Research title and Workspace/Runs mode bar.
- `Workspace data: Local | Server` selector.
- Workspace switcher scoped to the selected data source.
- Effective processing-route label.
- Selected-source and readiness summary.
- Primary action for the current empty or working state.
- Workspace and Help overflow menus.

Authority, processing route, and blocking status may compact but never disappear.
At wide widths, `/` opens an inline global workspace search spanning the
current authority's workspace names, attached sources, notes, and output
titles. At medium and narrow widths the same action opens a search modal. It
never searches or displays the other authority unless the user switches
`Workspace data` first.

### Wide layout

At 150 columns or wider:

```text
┌ Research header: mode · data source · workspace · processing · status ┐
├ Sources ──────────────┬ Grounded Chat ───────────────┬ Studio ─────────┤
│ catalog and selection │ transcript, citations, input │ outputs + notes  │
└ operation/readiness/recovery status ───────────────────────────────────┘
```

Sources and Studio have explicit collapse controls. Chat is the dominant pane
and may not resolve below 58 columns. Sources targets at least 32 columns and
Studio at least 42 columns. Borders and gutters consume the remaining budget.

### Medium layout

From 100 through 149 columns, render Chat plus one companion pane. The mode
strip remains visible:

```text
[Sources (12)] [Chat] [Studio (2)]
```

Sources is the default companion until the user explicitly opens Studio. The
preferred companion is persistent and is restored when width permits.

### Narrow layout

Below 100 columns, mount exactly one pane at a time. Do not leave hidden panes
in the focus cycle. The mode strip shows counts and blocked state in text.
At 60 columns the active pane's essential action, status, and recovery must
remain painted and reachable.

### Short-height layout

- At 30 rows or more, use the full destination header and status bar.
- From 24 through 29 rows, use compact header copy and a one-row status bar.
- Below 24 rows, the active pane remains scrollable; authority, processing
  route, and blocking status stay pinned in condensed form.

Width or height changes preserve source selection, chat draft, transcript
reading position, Studio options, pane preference, and semantic focus target.
If reflow hides the focused pane, focus moves to that pane's visible mode
button and the change is announced.

## State and controller architecture

```text
ResearchWorkspaceScreen
  ├── ResearchWorkspaceController       state and orchestration; no DOM
  ├── ResearchHeaderRegion              mode, authority, workspace, status
  ├── ResearchSourcesRegion             sources and selection
  ├── ResearchChatRegion                workspace Q&A
  ├── ResearchStudioRegion              outputs and Quick Notes
  └── ResearchWorkspaceOverlayStore     device-only state

ResearchWorkspacePort
  ├── LocalResearchWorkspaceAdapter
  └── ServerResearchWorkspaceAdapter
```

The screen and regions live under one `UI/Research_Workspace_Modules/` package.
Region widgets own their pixels and pane-local events. The controller owns
state and behavior without composing DOM. Dependencies are explicit and
late-bound following the established screen-decomposition rules.

`ResearchWorkspacePort` is justified because there are two real implementations
at launch. It exposes one normalized workbench contract; it does not erase
authority-specific capabilities or identifiers.

Minimum operations:

- list/get/create/update/archive/restore/delete/duplicate workspaces where
  allowed by the authority;
- list/add/update/remove/reorder/preview sources;
- get readiness and per-output capabilities;
- get/set selected source scope;
- list/create/update/delete notes;
- list/resolve/create/update/delete/export outputs;
- create/list/send/stop workspace chat;
- generate/cancel/retry supported outputs;
- preflight and execute explicit Copy;
- return operation and transfer receipts.

Every normalized identifier includes its authority. Raw local UUIDs and server
integer/string IDs never enter UI cache keys without an authority-qualified
wrapper.

## Immutable request context and async fencing

Every asynchronous read or mutation captures:

- data source;
- local profile or server endpoint/profile identity;
- authenticated principal where available;
- workspace ID;
- capability revision or snapshot timestamp;
- controller context revision.

Results may update only their captured workspace state. A result whose context
revision no longer matches the visible workspace is stored by its owner but
must not repaint the current screen. Switching data source or workspace never
retargets an in-flight request.

V1 switching policy:

- Durable source-ingest and server operations may continue in the background
  and remain visible in the global operation status.
- Active non-durable chat streams and unsaved mutations disable data-source or
  workspace switching with an explicit reason and Stop/Save recovery.
- Note and chat drafts are persisted per qualified workspace before switching.
  If draft persistence fails, switching is blocked with Retry, Discard, and
  Cancel choices.
- Capability or authentication changes increment the context revision and
  require a fresh readiness projection.

## Authority and processing behavior

### Local data

Local mode uses Chatbook-owned workspace, Library, Notes, RAG, conversation,
Study, Chatbook, TTS, and Research services. Model execution may be on-device or
cloud-backed according to the selected provider.

Before the first request that may send local source bodies to a remote
provider for a given workspace/provider/endpoint combination, show a preflight
with:

- selected source count and types;
- destination provider and endpoint class;
- whether full source bodies or retrieved excerpts may be sent;
- configured redaction policy;
- Cancel and Continue actions.

The user's approval is stored as bounded consent for that exact processing
route and invalidated when provider, endpoint, redaction policy, or source-body
mode changes. Persistent diagnostics remain payload-free under ADR-029.

### Server data

Server mode uses server workspace APIs for catalog, retrieval, chat,
generation, notes, artifacts, sharing, and diagnostics. Chatbook does not pull
server source content and silently send it through a client-selected provider.
A future client-inference path requires an explicit Server-to-Local Copy and is
outside this design.

If the selected server is missing, unreachable, unauthenticated, or lacks the
required API, Server remains selected and shows recovery. It never falls back
to Local and never substitutes a local workspace with the same name.

## Canonical ownership

| Entity | Local owner | Server owner | Workbench role |
| --- | --- | --- | --- |
| Workspace record | `WorkspaceDB` / registry service | Workspace API | Select and contextual quick actions |
| Source content | Library media/notes/files and their stores | Server media/workspace source | Attach by reference and preview |
| Membership | `WorkspaceMembership` | Server workspace source membership | Eligibility and provenance |
| Selected RAG scope | Workspace `RagScope` | Server selection API | Grounding scope |
| Chat | Workspace-scoped local conversation | Server conversation/chat API | Persisted evidence conversation |
| Quick Note | Local Notes record + membership | Workspace notes API | Pane-local create/edit/search |
| Summary/Report/Compare | Local Chatbook artifact + membership | Workspace artifact API | Versioned Studio output |
| Flashcards | Local Study deck/cards + membership | Workspace artifact/study API | Generate, reopen in Study |
| Quiz | Local Quiz record + membership | Workspace artifact/study API | Generate, reopen in Study |
| Audio | TTS-owned file/artifact when implemented | Workspace artifact/export API | Future capability-gated output |
| Slides/Mind Map/Timeline/Data Table | Native owner when implemented | Workspace artifact API | Future capability-gated output |
| Research run | Research Interop service | Server Research API | Runs screen lifecycle |
| Copy receipt | Local workspace handoff audit | Server migration/operation receipt plus local audit | Inspectable transfer history |
| Folder/annotation overlay | Device overlay store | Not server-owned | Explicit device-only organization |

The UI builds `WorkspaceOutputRef` values from canonical records and memberships.
It does not persist a parallel output row merely to make a unified list.
Versioning, deletion, and retry follow the canonical owner.

For the primary five, the implementation plan must include a field-level
mapping for owner IDs, versions, provenance, generation configuration,
reopen target, and deletion behavior before code is written.

## Device-only overlay store

Use one private, atomically written JSON store in the Chatbook data directory,
following existing private JSON registry patterns. It contains only
workbench-specific client state:

- source folders and membership within those folders;
- source annotations;
- per-qualified-workspace pane preference and collapsed state;
- recent and pinned server workspaces;
- banner and split/pane presentation preferences;
- durable Deep Research launch/return context for server workspaces.

Keys include data source, server/profile identity, principal identity when
available, and workspace ID. The store never creates a local mirror workspace
record for a remote-only workspace.

Annotations are private user content and receive ADR-029 file permissions.
The store has bounded record and payload limits, rejects unsafe/corrupt data,
and writes atomically. Corruption disables only the affected overlay and offers
export/reset recovery; it does not make the canonical workspace unavailable.

The UI labels these features:

```text
Device-only overlay — not uploaded, shared, or available on other devices.
```

Deleting or losing access to a server workspace leaves its overlay orphaned.
The overlay is retained for a bounded recovery period and can be exported or
deleted from local storage. A newly created workspace reusing the same display
name never inherits the overlay because identity keys include the canonical ID.

## Workspace lifecycle

### Local

Research provides contextual switch, create, rename, duplicate, archive, and
restore actions. `Manage Workspaces…` opens Settings, which remains the full
local lifecycle and filesystem-root owner under ADR-028. Destructive local
delete is completed in Settings. The built-in Default workspace is labeled
`Everyday chats` and is not offered as a research notebook; first use creates
or selects an explicit named workspace.

### Server

Where the server grants permission, Research exposes switch, create, rename,
duplicate, archive, restore, delete, templates, sharing, import, and export.
Unavailable actions remain discoverable through the Workspace menu with owner,
reason, and recovery, not as unexplained disabled buttons.

Recent and pinned status is device-local presentation state. Archive/delete
state is canonical server state.

## Control hierarchy and parity mapping

Every audited server control is classified. Classification determines where it
appears; it does not erase the control from the contract.

### Always-visible core

- Workspace/Runs mode tabs.
- Workspace-data selector.
- Workspace selector.
- Effective processing route.
- Source count and readiness.
- Pane mode strip at medium/narrow widths.
- Current pane's primary action.
- Global workspace search at wide widths, with the same `/` command available
  at every width.

### Workspace menu

- Recent, pinned, archived, and new workspaces.
- Rename, duplicate, archive, restore, and authority-appropriate delete.
- Manage Workspaces.
- Templates: Literature Review, Interview Analysis, Product Brief.
- Customize banner and split workspace.
- Import/export workspace and Export BibTeX.
- Collections.
- Share in Server mode; Local mode offers Export bundle, Copy to Server, and
  Copy to Server and Share.
- Open in Console, corresponding to the server WebUI's Simple Chat handoff.

### Help menu

- Guided first-source tour.
- Keyboard shortcuts.
- Storage and operation status.
- Feature-flagged telemetry status links to its owning Settings/diagnostic
  surface; it is not a standard workspace mutation.

### Owner links

- Create agent task.
- ACP history.
- Sandbox diagnostics.
- MCP, ACP, provider, runtime, and grounded-answer remediation.

These controls show status and navigate to the owning screen or diagnostic
surface. Research does not recreate their management UI.

### Planned labels only

- Research Dossier.
- Competitive Market Memo.
- Technical Project Spec.

They are labeled Planned and are not focusable action buttons.

## Sources pane

### Add Sources

The modal changes vocabulary and behavior by data source:

| Local | Server |
| --- | --- |
| Import Files | Upload |
| Local Library | My Media |
| URL | URL |
| Paste | Paste |
| Search Local | Search Server |

Both authorities support batch URLs, progress, filtering, pagination, and
batch attachment where their adapters report capability. Local operations
reuse Library ingestion; Server operations use server APIs.

### Source list

- Quick URL add.
- Search, advanced filters, sort, pagination.
- Select all/none and persistent selected-source scope.
- Folder tree and explicit `Select folder sources` action.
- Per-row folder membership.
- Preview, annotation, evidence, readiness, and status detail.
- Reorder up/down with keyboard equivalents.
- Batch Move/Copy within the same authority, preview, and remove.
- Per-item conflict preview and receipts for batch Move/Copy.
- Undo where the canonical owner supports a safe reversal.

Source folders are organizational overlays. They are not ADR-028 filesystem
tool roots and must use distinct copy and identifiers.

Readiness distinguishes attached, parsing, indexing, FTS-ready, vector-ready,
failed, unavailable, and stale. Missing embeddings explicitly produce FTS-only
readiness; the UI never claims Hybrid.

## Grounded Chat pane

Chat is workspace Q&A without tools or approvals. The server-compatible mode
values use clearer labels:

| UI label | Contract value | Behavior |
| --- | --- | --- |
| Sources off | `general` | Workspace chat without retrieval; still no agent tools |
| Grounded | `rag` | Retrieval required from selected ready sources |
| Auto retrieval | `auto` | Adapter may retrieve; response reports whether it did |

Controls:

- provider/model or server model selection;
- selected-source chips;
- readiness gate and source-body mode;
- FTS/Vector/Hybrid, top K, threshold, reranking, and citations;
- temporary drag/drop scope;
- conversation/session selection and clear;
- sharing where supported;
- retrieval diagnostics, confidence, token, and cost detail;
- Lorebook activity where supported;
- slash commands that resolve within the same workspace context.

Message actions:

- copy, edit, regenerate, delete, and undo;
- variants and new branch;
- message information and citation inspection;
- save to Quick Notes;
- summarize, translate, shorten, and explain;
- read aloud only when TTS is actually available.

Every answer records source identities and versions, citations, retrieval mode,
processing route, provider/model, and generation state. `Auto retrieval`
explicitly reports `Retrieved` or `Did not retrieve`.

## Studio and Quick Notes

### Primary outputs

The first Studio view shows:

1. Summary
2. Flashcards
3. Quiz
4. Report
5. Compare Sources

Compare Sources requires at least two ready sources. All outputs require at
least one eligible source and an available processing route.

### More outputs

`More outputs…` opens three groups:

- **Learn:** Summary, Flashcards, Quiz.
- **Analyze:** Report, Compare Sources, Mind Map, Timeline, Data Table.
- **Present:** Slides, Audio Summary.

Already visible primary items may appear in their group for orientation but do
not produce duplicate commands. Each unavailable item displays authority,
reason, owner, and recovery. Per-output availability is projected by Chatbook;
the server's coarse text/slides/audio capability categories are insufficient by
themselves.

### Work products

More outputs also lists:

- Executive Brief.
- Literature Matrix.
- Corpus Gap Finder.
- Evidence-Bound Hypotheses.
- Research Proposal Pack.

They become actionable only when an adapter has a real generator and owner.

### Options

- provider/model;
- temperature, top P, and max tokens;
- FTS/Vector/Hybrid, top K, threshold, citations, and reranking;
- slide style when Slides is available;
- TTS provider/model/voice preview/speed/format when Audio Summary is available;
- flashcard deck target.

Options are output-specific and progressively disclosed. The default view does
not show irrelevant fields.

### Output lifecycle

- view and edit;
- download/export;
- retry;
- regenerate and replace;
- regenerate as a new version;
- discuss in the same Grounded Chat;
- save or append to Quick Notes;
- delete and safe undo where supported;
- Data Table CSV/JSON export;
- launch Deep Research where supported.

After creation, every result says where it was saved and offers its canonical
reopen action, for example `Saved to Study > Decks` or `Saved to Artifacts`.
Server export uses the server's accepted, traceable artifact-export contract;
a client-side download is not presented as equivalent server persistence.

### Quick Notes

Quick Notes is a subordinate Studio section, not a second screen owner. It
supports list/load/search, title, Markdown edit/preview, content, tags,
create/update, download, clear, undo, and version-conflict recovery. Capturing
a chat message opens the note with provenance back to the message and sources.

## Explicit cross-authority Copy

Copy never begins from an authority toggle. It is an explicit action on a
workspace or selected items.

### Protocol

1. Resolve exactly one source authority and one destination authority/profile.
2. Freeze a manifest snapshot with identities, versions/hashes, memberships,
   provenance, and redaction classification.
3. Show a preflight row for each item: Copy content, Reference, Metadata only,
   Omit, or Blocked.
4. Show destination, principal, estimated size, secrets omitted, unsupported
   types, and conflicts.
5. Require explicit confirmation.
6. Execute idempotently using a transfer ID and per-item idempotency keys.
7. Persist a receipt after every stage.
8. Finish as Completed, Partially completed, Rolled back, Failed-retryable, or
   Failed-terminal.

V1 does not offer automatic merge or continuous sync. Retrying a partial Copy
reuses its transfer ID and never duplicates already acknowledged items.

Conflict choices are Keep destination, Replace destination, Copy as new,
Reference existing, Omit, or Cancel. Destructive replacement requires a second
confirmation and is unavailable where the destination cannot guarantee safe
version checks.

## Sharing

Server mode exposes supported workspace sharing:

- team or organization target;
- three server-defined permission levels;
- allow clone;
- private link with password, expiry, and use limit;
- active share list and revoke;
- shared-with-me, preview, verification, import, and clone where supported.

Local mode never presents a generic Share action. It offers Export bundle,
Copy to Server, and Copy to Server and Share. The latter is a two-stage flow:
Copy must finish successfully before the Share confirmation appears.

Device-only overlays are excluded from server sharing. The preflight says so.

## Deep Research bridge

Studio may launch Deep Research from a supported output or source selection.
The launch record contains:

- origin workspace reference and data source;
- selected source identities and versions;
- initiating output or message identity;
- normalized query;
- local conversation ID or server chat ID in authority-specific fields;
- return route and creation timestamp.

The run is created and owned by Research Interop or the server Research API.
Workspace associates the run without taking over its status machine, events,
checkpoints, artifacts, lease, or budget.

Runs shows meaningful ADR-068/070 states including Awaiting plan review,
Awaiting source review, Lease held elsewhere, Resume available, and Partial
evidence saved.

When a bundle is available, Runs offers `Return to Workspace`. Workspace calls
the normalized bundle boundary, validates origin and run identity, and previews
the import. The user then creates a draft Report output/reference. Nothing is
inserted automatically. Repeated imports are idempotent and become a new
version only by explicit choice.

## Reliability and concurrent-change behavior

The status bar exposes current authority, storage health, active operations,
retryable failures, and Help. Server quota or policy limits are shown before an
operation when the API reports them.

Canonical server mutations use the server's version/precondition fields. A
stale workspace, note, source, or artifact update never overwrites newer state
silently; recovery is Reload, Fork/Copy as new where supported, or Cancel.

The device overlay store carries a monotonically increasing revision and uses
atomic compare-before-replace behavior. If another Chatbook process changes the
same qualified overlay after it was loaded, the current process offers Reload,
Export this draft, or Fork device overlay. `Keep mine` is allowed only after an
explicit overwrite confirmation and never changes canonical server data.

Workspace import, export, and Local/Server Copy persist operation IDs and
receipts. Restarting Chatbook reconstructs Completed, In progress, Retryable,
and Recovery required states from the owner rather than guessing from UI
state. An interrupted server migration resumes only through the server's
migration status/finalize contract.

Source-version changes invalidate affected citations and Studio source
snapshots. Existing outputs remain inspectable with `Sources changed since
generation`; regenerate uses a fresh snapshot and creates a new version unless
the user explicitly chooses a supported replace operation.

## Capability model

Each action receives a normalized capability result:

```text
available
reason_code
user_message
owner
recovery_action
capability_revision
```

Unknown capabilities fail closed. A control may be:

- active;
- hidden because it is irrelevant to the selected output/state;
- visible and unavailable with reason/recovery;
- owner-link;
- planned label.

The server Research Workspace capability endpoint is a readiness projection,
not a persistence owner. Workspace/source/note/artifact/chat APIs remain the
canonical server stores. Chatbook also does not copy the WebUI's browser-local
chat-session persistence and call it server durability: if canonical server
conversation persistence is unavailable, that behavior is capability-gated.

Unavailable controls never perform a different action and never switch
authority. The footer advertises only working actions in the current context.

## Error and recovery contract

| State | Required behavior |
| --- | --- |
| Server unreachable | Keep Server selected; Retry, Change server, and diagnostics |
| Authentication expired | Preserve drafts; Reauthenticate; no Local fallback |
| Workspace missing/deleted | Preserve overlay; choose another workspace or export/delete overlay |
| Archived workspace | Read-only status with Restore where permitted |
| Source missing | Identify exact source and allow remove/relink |
| Parsing/indexing failed | Status detail, retry/re-add, and unaffected-source continuation |
| Vector runtime unavailable | Explicit FTS-only mode; no Hybrid claim |
| Provider/model unavailable | Preserve prompt/options; change processing route |
| Remote inference not approved | Block send; show egress preflight |
| Generation interrupted | Preserve owner record/draft when possible; Retry or Delete |
| Late stale result | Store under captured context; do not repaint current workspace |
| Note version conflict | Reload, fork, or keep local draft; never overwrite silently |
| Copy partially failed | Resume from receipt, export receipt, or abandon remaining items |
| Overlay corrupt | Disable affected overlay; export/reset; canonical workspace remains usable |
| Concurrent overlay edit | Reload, export draft, fork overlay, or explicitly overwrite device-only state |
| Server version conflict | Reload, fork/copy as new, or cancel; never overwrite silently |
| Storage/quota limit | Preserve draft and show owner, measured limit, cleanup/export recovery |
| Migration interrupted | Resume or inspect the canonical operation receipt; never infer completion |
| Deep Research bundle mismatch | Refuse import; show origin/run mismatch and open Runs |
| Capability changed | Refresh projection and explain which action changed |

Destructive actions use guarded confirmations. Escape requests the safe negative
result. In-progress non-cancelable mutations reject dismissal with a visible
reason and never discard drafts.

## Accessibility and keyboard contract

- Data source, processing route, readiness, error, and capability state are
  text-labeled; color is supplemental.
- Disabled controls remain readable under the project's measured contrast
  rule and include their reason.
- F6/Shift+F6 cycle visible pane roots. Tab/Shift+Tab stay within the active
  pane's ordinary focus order.
- Medium/narrow mode buttons are focusable and announce counts and state.
- Source reorder, pane resizing/collapse, mind maps, slides, audio, and data
  tables have keyboard and textual alternatives.
- Async loading, completion, failure, authority changes, and focus relocation
  are announced without stealing focus unexpectedly.
- Modal Escape and backdrop behavior follows ADR-031's safe-dismissal grammar.
- Screen actions use single-letter htop-style keys outside text inputs and do
  not bind terminal-convention or reserved global keys.
- F1 help and footer hints are generated from the same active binding set.

## Performance contract

- All network, database aggregation, indexing, generation, file parsing, and
  work estimated above 100 ms runs in a worker.
- Workspace, source, note, output, and search lists are paginated or windowed.
- Authority/workspace switches use per-context caches but always revalidate
  capability and readiness revisions.
- Source preview and annotations load on demand.
- Hidden panes do not perform paint-only refresh loops.
- Streaming chat updates only the active message region rather than recomposing
  the screen.
- Cache keys include qualified authority and workspace identity.

## Verification requirements

### Unit and contract tests

- Shell route resolution and saved `research` compatibility.
- Destination-ID shortcut mapping preserves every existing key and assigns F10
  to Research.
- WorkspaceDataSource never calls the other adapter and never uses runtime
  fallback.
- Qualified identity prevents Local/Server and server-profile collisions.
- Effective processing route and egress-consent invalidation.
- Capability projection for all ten outputs and five work products.
- Ownership mapping and `WorkspaceOutputRef` resolution for the primary five.
- RAG selection includes only supported source types and fails closed on empty
  overlap.
- Async context fencing rejects late repaint.
- Copy manifest validation, conflict choices, idempotent retry, and receipt
  terminal states.
- Overlay isolation, bounds, corruption recovery, and private atomic writes.
- Overlay revision conflicts and multi-process recovery choices.
- Server optimistic-conflict, quota, and interrupted-migration recovery.
- Deep Research launch identity and idempotent bundle import.

### Mounted Textual tests

Use the production hierarchy and consolidated stylesheet. Verify at:

- 160x40;
- 120x30;
- 100x30;
- 84x24;
- 80x24;
- 60x20.

Assertions cover essential painted controls, containment, no off-screen pane
blowout, visible-pane focus cycles, reflow focus relocation, preference restore,
draft/selection/scroll preservation, truthful footer hints, readable disabled
reasons, and mode-tab navigation between separate screens.

### Integration tests

- Local workspace/source/note/chat/output round trip through real temporary
  SQLite/JSON stores.
- Server adapter contract tests for workspace CRUD, source status/preview,
  selection/reorder, notes, artifacts, chat, capabilities, export, sharing, and
  diagnostics.
- Server unavailable/auth-expired behavior proves no Local calls occur.
- Local remote-provider preflight proves no request leaves before consent.
- Primary five outputs reopen through their canonical owners.
- Deep Research launches in Runs, survives navigation, returns a validated
  bundle, and imports only after confirmation.

### Live verification

Run targeted real-app checks through the same UI path users take. Local checks
use an isolated profile with realistic sources. Server checks use a configured
test server and prove catalog/source/chat/note/artifact identity. Record the
actual processing route and whether any degradation path ran. Do not treat a
database write alone as evidence that the mounted UI can read the result.

## Delivery decomposition

This is too large for one implementation PR. After spec approval, create
independent Backlog tasks and plans in dependency order:

1. **Research shell and authority foundation** — destination, two real routes,
   stable shortcuts, header/mode bar, qualified identities, adapters, and
   fail-closed read-only catalogs.
2. **Sources and Quick Notes** — attach/search/preview/readiness/selection,
   overlays, Local ingestion, server APIs, and notes CRUD.
3. **Grounded Chat** — local/server conversation persistence, retrieval modes,
   citations, diagnostics, processing-route consent, and message actions.
4. **Primary Studio outputs** — Summary, Flashcards, Quiz, Report, Compare
   Sources, ownership mappings, output history, options, and lifecycle.
5. **Copy, sharing, and Deep Research bridge** — transfer protocol/receipts,
   server sharing, launch context, Runs navigation, and bundle return.
6. **Extended parity** — remaining five outputs, work products, workspace
   import/export/BibTeX, templates, and supported integration diagnostics.

Each task must be independently testable and may expose only capabilities it
actually implements. No phase may add inert advertised controls.

## Acceptance summary

The design is satisfied when:

- Research is one shell destination with separate Workspace and Runs screens.
- Existing `research` routes still open the existing ResearchScreen.
- Local/Server selection is fail-closed and distinct from inference/runtime.
- The effective processing route prevents `Local` from implying on-device
  processing.
- The core source → grounded answer → note/output loop is obvious.
- The primary five outputs lead; all remaining outputs are grouped under More.
- Every audited server control has a documented classification and owner.
- Canonical content owners remain authoritative; no universal duplicate output
  store is introduced.
- Device-only overlays are isolated and honestly labeled.
- Copy is manual, idempotent, resumable, and receipted.
- Deep Research remains separately owned and returns through explicit import.
- Responsive, keyboard, focus, accessibility, failure, and recovery behavior is
  objectively testable at the stated terminal sizes.
