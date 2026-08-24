# ADR-078: Research Workspace authority and screen boundaries

- **Status:** Proposed
- **Date:** 2026-08-23
- **Task:** [TASK-21505](../tasks/task-21505%20-%20Design-Local-Server-Research-Workspace-and-Research-Runs-navigation.md)
- **Design:** [Research Workspace design](../../Docs/superpowers/specs/2026-08-23-research-workspace-design.md)
- **Amends:** ADR-015 (shell destination taxonomy); ADR-028 (adds Research
  contextual workspace quick actions while Settings remains the full manager)
- **Related:** ADR-005 (Console workspace server-readiness), ADR-027 (Default
  workspace chat grouping), ADR-028 (Settings workspace and folder-root
  ownership), ADR-029 (local private data), ADR-031 (TUI keybindings), ADR-068
  (local research engine), ADR-070 (research run durability)

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
- specialist stores own future audio, slides, mind-map, timeline, and data-table
  artifacts when implemented;
- Research Interop owns Research Runs.

The UI may normalize these as `WorkspaceOutputRef` values in memory. It does
not add a universal output database. If a future output has no safe canonical
owner, that storage decision requires its own ADR or an explicit amendment.

Server mode uses canonical server workspace, source, note, artifact, chat,
sharing, and operation APIs.

### 5. Server folders and annotations are explicit device-only overlays

The server has no canonical Research Workspace folder or annotation APIs.
Chatbook stores those features, along with pane preferences and server
Deep-Research launch context, in one private atomic device overlay keyed by
data source, server/profile, principal, and workspace ID.

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
- Complete server parity remains discoverable without overwhelming the primary
  flow.
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
- The feature must ship in multiple independently testable tasks; no single PR
  should attempt the whole design.

### Follow-on rule

Implementation plans must name the canonical owner and capability behavior for
every output or action they activate. An advertised control without a real
owner, execution path, recovery path, and targeted test does not ship.
