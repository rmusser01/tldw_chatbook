# tldw_chatbook Writing Suite Structural Parity Design

**Date:** 2026-04-22  
**Primary Repo:** `tldw_chatbook`  
**Reference Repo:** `tldw_server2`

## Goal

Add the first serious `Writing Suite` vertical to Chatbook as a standalone-first authoring surface that can also operate against `tldw_server` when the user is in server mode.

This first slice is structural authoring parity only. It covers project, manuscript, chapter, scene, outline, working-draft, manual-version, soft-delete, restore, and source-separation behavior. It does **not** include outline generation, revision helpers, export, publishing, collaboration, sync, or generic workflow execution.

The desired outcome is a credible local writing product surface that remains usable offline, plus a source-honest server path that maps to the current server manuscript contract without pretending the server supports structures it does not.

## Context

The parity audit identifies `Writing Suite` as a high-priority Tranche 2 row:

- local parity required
- remote parity required
- explicit local/server source separation required
- no mixed view in the first slice
- no dependency on `Notes` or `Workspaces`

Current Chatbook evidence shows only adjacent writing helpers:

- `tldw_chatbook/UI/ChatbookTemplatesWindow.py`
- `tldw_chatbook/Widgets/TTS/chapter_editor_widget.py`

There is no dedicated local project and manuscript hierarchy.

Current server evidence:

- `tldw_server2/tldw_Server_API/app/api/v1/endpoints/writing.py`
- `tldw_server2/tldw_Server_API/app/api/v1/endpoints/writing_manuscripts.py`
- `tldw_server2/tldw_Server_API/app/api/v1/schemas/writing_manuscript_schemas.py`

The server manuscript contract exposes:

- manuscript projects
- parts
- chapters
- scenes
- structure tree
- reorder endpoint
- soft-delete behavior
- optimistic-locking `expected_version` headers on updates/deletes
- scene content as TipTap JSON plus plain text
- analysis and research routes that are out of scope for this slice

The server does **not** currently expose a first-class `manuscript` entity distinct from project/part, and it does **not** expose parentless scenes directly under a project or part. This design therefore maps server `parts` to Chatbook `manuscripts` and treats direct manuscript-level server scenes as an unsupported server capability until the server contract grows that parent type.

## Fixed Decisions

- The first vertical is structural only.
- `Project` is the required top-level container in both local and server mode.
- A project may contain multiple manuscripts.
- Writing projects are entirely separate from `Notes` and `Workspaces`.
- The UI is one destination with an explicit `Local` / `Server` source switch.
- There is no mixed local/server writing view in v1.
- Authored prose is Markdown.
- `Chapter` is a structural container assembled from scenes.
- `Chapter` does not have independently authored body content.
- `Scene` is the primary authored prose unit.
- A scene may belong to a chapter or directly to a manuscript in the Chatbook target model.
- Direct manuscript-level scenes appear inline in the manuscript outline alongside chapters.
- Workflow/status fields are available but optional so low-friction writing remains the default.
- `Scene` has a mutable autosaved Markdown working draft plus explicit user-created versions.
- `Manuscript` and `Chapter` have autosaved metadata/structure working state plus explicit user-created versions, but no authored body draft.
- Normal autosave does not advance the version number.
- `Create New Version` snapshots the current entity working state into the next numbered immutable version.
- Older versions are read-only until the user explicitly restores or derives a new working state from them.
- Delete is soft-delete with trash and restore.
- Reorder chapters/scenes and move scenes between chapters/manuscript parents are first-class where the active source supports them.

## In Scope

- New writing normalized models for project, manuscript, chapter, scene, version, draft, and outline nodes.
- Local writing store/service for standalone operation.
- Server writing adapter for current server manuscript project, part, chapter, scene, structure, reorder, and soft-delete routes.
- Scope service that routes every action to the selected source and centralizes unsupported server capability checks.
- Source-switched writing destination in the TUI.
- Browse/detail/create/update/delete/restore for local records and for server records only where the current server contract supports the action.
- Markdown editor for scene working drafts.
- Manual version creation for manuscript, chapter, and scene.
- Read-only historical version viewing.
- Restore-from-version behavior.
- Local soft-delete trash view with restore behavior.
- Outline tree/list that shows manuscripts, chapters, and scenes in source order.
- Reorder chapters and scenes.
- Move scenes between chapters and manuscript-level placement where supported by the active source.
- Optional status fields for project, manuscript, chapter, and scene.
- Explicit unavailable states for server-only unsupported actions.

## Out Of Scope

- LLM generation, outline generation, continuation, rewriting, critique, or revision helpers.
- Export, publishing, EPUB/PDF/docx generation, or print layout.
- Collaboration, comments, reviewer workflow, sharing, or permissions beyond existing server auth errors.
- Sync, mirroring, dual-write, or local persistence of server-authored records.
- Mixed local/server writing search or browse.
- Integration with `Notes` or `Workspaces`.
- Server API changes.
- Server trash restore without a verified server restore endpoint.
- Server scene reparenting without a verified server scene-parent update endpoint.
- Server analysis/research routes in `writing_manuscripts.py`.
- Characters, relationships, world info, plot lines, plot events, plot holes, citations, scene links, and research endpoints beyond avoiding schema collisions.
- Full prose IDE features such as split panes, focus mode, grammar checking, or version diffs.

## Approaches Considered

### Option A: Structural-authoring first

Use dedicated entities for `Project -> Manuscript/Unassigned Chapters -> Chapter/Scene`, with source-separated local/server stores, autosaved scene drafts, autosaved container metadata/structure state, manual versions, soft-delete, reorder/move support, and optional status fields.

Why chosen:

- matches the desired product model
- gives clean local/server seams
- keeps v1 serious without pulling in generation/export work
- avoids over-abstracting the hierarchy before the UI exists
- leaves room for later writing helpers to attach to well-defined records

### Option B: Document-first with lightweight outline nodes

Treat manuscript as the primary authored document and model chapters/scenes as lightweight outline nodes.

Why not chosen:

- blurs versioning boundaries
- makes scene movement and restore semantics harder to reason about
- weakens the future parity seam for server chapters and scenes

### Option C: Generic tree model

Use one generic hierarchical node type with a `kind` field for manuscript, chapter, and scene.

Why not chosen:

- flexible but too abstract for the first slice
- makes UI and tests less obvious
- risks hiding contract mismatches behind a generic adapter

## Chosen Model

### Source Model

The writing destination owns a current source:

- `local`
- `server`

Every action includes the active source. The scope service routes the action to the corresponding backend. Local writes never call the server. Server writes never mutate local records.

Server mode does not silently fall back to local mode. If the server is unavailable, unauthorized, or lacks a capability, the action is disabled or fails visibly in server mode while local mode remains usable.

### Entity Model

Chatbook target entities:

- `WritingProject`
- `WritingManuscript`
- `WritingChapter`
- `WritingScene`
- `WritingDraft`
- `WritingVersion`
- `WritingTrashEntry`
- `WritingOutlineNode`

`WritingDraft` is a normalized working-state record. Only scene drafts carry Markdown body content; manuscript and chapter working state carries metadata plus structural membership/order fields.

Hierarchy:

```text
Project
  Manuscript
    Chapter
      Scene
    Scene
  Unassigned Chapters
    Chapter
      Scene
```

Rules:

- Project is required for every manuscript.
- A project can have multiple manuscripts.
- A project can have unassigned chapters outside any manuscript.
- The `Unassigned Chapters` bucket is a presentation node, not a manuscript.
- A manuscript can contain chapters and direct scenes.
- A chapter can contain scenes.
- A scene can exist under a chapter; that chapter may be assigned to a manuscript or remain project-level and unassigned.
- A direct manuscript-level scene cannot exist without a manuscript.
- A direct manuscript-level scene has no chapter parent.
- A chapter body is assembled from its ordered scenes.
- A scene body is Markdown.

### Server Contract Mapping

The current server manuscript contract maps into the target model as follows:

| Chatbook target | Server contract |
| --- | --- |
| Project | `ManuscriptProject` |
| Manuscript | `ManuscriptPart` |
| Chapter | `ManuscriptChapter` |
| Scene | `ManuscriptScene` |
| Unassigned chapters | `ManuscriptStructureResponse.unassigned_chapters` |
| Outline reorder | `/projects/{project_id}/reorder` |
| Soft delete | server delete routes that soft-delete |
| Optimistic lock | `expected_version` header |

Server gaps:

- Direct manuscript-level scenes are not supported by the current server route shape because scenes are created under chapters.
- Markdown is not first-class on the server; server scenes expose TipTap JSON and plain text.
- Manual historical versions for manuscript/chapter/scene are not first-class in the current server manuscript endpoint.
- Trash listing and restore are not verified on the current server manuscript endpoint.
- Scene reparenting between chapters is not verified on the current server manuscript endpoint.

Contract-honest handling:

- Direct manuscript-level scene creation/move is available locally.
- In server mode, direct manuscript-level scene creation/move is disabled centrally unless the server contract adds parentless scene support.
- Server `unassigned_chapters` are rendered in the project-level `Unassigned Chapters` bucket.
- The adapter must not drop unassigned chapters and must not invent a hidden manuscript to contain them.
- Assigning or unassigning a chapter to/from a manuscript is allowed only through a verified chapter parent update path.
- The server adapter may display server chapters without scenes and server scenes under chapters, but it must not invent hidden chapters to fake direct scene support.
- Server scene reorder within the existing chapter may use the reorder endpoint when verified.
- Server scene reparenting between chapters is disabled centrally unless a verified endpoint supports changing the scene parent.
- Server soft-delete is supported where server delete routes soft-delete records.
- Server trash listing and restore are disabled centrally unless verified server endpoints exist.
- Markdown is preserved through a deterministic adapter format rather than silently stripped.
- Server version history support is capability-gated. If the current server contract cannot persist historical versions, `Create New Version` for server records is disabled or implemented only if a verified server endpoint exists. Local manual versioning remains fully available.

### Capability Matrix

| Capability | Local v1 | Server v1 on current contract |
| --- | --- | --- |
| Project CRUD | Supported | Supported through `ManuscriptProject` |
| Multiple manuscripts per project | Supported | Supported by mapping manuscripts to `ManuscriptPart` |
| Project-level unassigned chapters | Supported | Supported through `unassigned_chapters` |
| Chapter CRUD | Supported | Supported through `ManuscriptChapter` |
| Chapter assign/unassign to manuscript | Supported | Supported only through verified chapter `part_id` update/reorder paths |
| Scene CRUD under chapter | Supported | Supported through `ManuscriptScene` |
| Direct manuscript-level scenes | Supported | Disabled until server supports parentless scenes |
| Scene reorder within current parent | Supported | Supported only where current reorder contract verifies cleanly |
| Scene reparent between chapters | Supported | Disabled until server supports scene parent updates |
| Manual versions | Supported for manuscript/chapter/scene | Disabled until server version endpoint exists |
| Historical version restore | Supported for local versions | Disabled until server version/restore endpoint exists |
| Soft-delete | Supported | Supported where server delete routes soft-delete |
| Trash listing and restore | Supported | Disabled until server list-deleted/restore endpoints exist |
| Markdown fidelity | Native Markdown | Best-effort wrapper plus plain-text fallback |

### Markdown Adapter

Local mode stores Markdown as Markdown.

Server mode must preserve user-authored Markdown without pretending the server stores native Markdown. The server adapter should use a deterministic wrapper in scene `content` and derived plain text in `content_plain`, for example:

```json
{
  "type": "chatbook-markdown",
  "schema_version": 1,
  "markdown": "# Scene text"
}
```

When reading server scenes, the adapter should:

- prefer the `chatbook-markdown` wrapper when present
- fall back to a deterministic plain-text-to-Markdown representation when only `content_plain` is available
- never claim rich Markdown fidelity when only plain text is available

This wrapper is an adapter convention, not a server contract change.

## UI Design

The first TUI should be pragmatic and source-explicit:

- top bar with `Local` / `Server` source switch and connection/status label
- project browser
- manuscript browser inside selected project
- outline tree/list for chapters and scenes
- project-level `Unassigned Chapters` bucket when the active source returns or supports unassigned chapters
- detail/editor panel for selected entity
- Markdown editor for scene working draft
- metadata panel for title, summary/synopsis, optional status, and word counts
- versions panel for current working state, historical read-only versions, `Create New Version`, and restore actions
- trash view for soft-deleted writing records

The UI should not be a full prose IDE in v1. It should make structural authoring credible and stable first.

### Outline Behavior

The outline shows:

- manuscripts under a project
- unassigned chapters under a project-level bucket
- chapters under a manuscript
- scenes under chapters
- direct manuscript-level scenes inline alongside chapters

The unassigned-chapter bucket is source-owned structural state:

- it is shown only as a project-level bucket
- it is not treated as a manuscript
- its chapters can be assigned to a manuscript when the active source supports chapter parent updates
- its scenes remain normal chapter-owned scenes

Ordering is source-owned. Reorder and move operations are saved through the current source service. UI state should update after confirmed success unless an explicit optimistic update and rollback path is tested.

### Version Behavior

Working state:

- mutable
- autosaved
- source-owned
- not a historical version by itself
- for scenes, contains Markdown body content plus scene metadata
- for chapters, contains chapter metadata plus ordered child scene membership and assembled read-only preview
- for manuscripts, contains manuscript metadata plus ordered child chapter/direct-scene membership and assembled read-only preview
- never creates authored body content for chapters or manuscripts

Manual version:

- immutable
- numbered sequentially per entity
- created only by explicit user action
- read-only when viewed

Restore:

- copies the historical version payload into the entity working state
- does not mutate the historical version
- does not advance the version number until the user creates another version

### Version Payloads

`WritingVersion` payloads are entity-specific.

Scene version:

- snapshots scene title, synopsis, optional status, metadata, Markdown body, word count, and parent IDs
- restore copies those fields back into the scene working draft
- restore does not move the scene unless the stored parent differs and the active source supports scene reparenting

Chapter version:

- snapshots chapter title, synopsis, optional status, metadata, ordered child scene IDs, and a rendered assembled Markdown snapshot for read-only viewing
- does not duplicate child scene version bodies
- restore copies chapter metadata and, where supported, restores ordering/membership for existing child scenes
- restore does not recreate missing scenes
- restore does not delete extra current scenes unless a later exact-structure restore mode is explicitly designed
- if referenced scenes are missing, deleted, or cannot be reparented in the active source, restore reports a conflict and leaves the chapter working state unchanged

Manuscript version:

- snapshots manuscript title, synopsis, optional status, metadata, ordered outline node IDs, and a rendered assembled Markdown snapshot for read-only viewing
- does not duplicate child chapter or scene version bodies
- restore copies manuscript metadata and, where supported, restores ordering/membership for existing chapters and direct manuscript-level scenes
- restore does not recreate missing child records
- restore does not delete extra current child records unless a later exact-structure restore mode is explicitly designed
- if the active source cannot support a stored direct scene or scene reparent operation, restore reports a capability conflict and leaves the manuscript working state unchanged

This keeps local manual versions useful without turning container restore into implicit destructive child editing.

## Error Handling

### Source Boundary

- Local actions never call the server.
- Server actions never mutate local records.
- Switching source with dirty state must autosave first or block with a clear prompt.
- Failed autosave keeps dirty state and does not advance versions.

### Server Errors

Server mode should surface:

- connection failures
- auth failures
- permission failures
- optimistic-lock conflicts
- unsupported capability states

High-impact failures should emit a notification and keep the relevant inline error visible near the failed action.

### Version Errors

- Failed `Create New Version` leaves the working state unchanged.
- Failed restore leaves the historical version read-only and the working state unchanged.
- Version numbers must not skip on failure.

### Reorder And Move Errors

- Failed reorder/move leaves the visible outline in the last confirmed order unless a tested rollback path exists.
- Deleted-parent restore conflicts should be explicit and resolvable.
- If the original parent is deleted or missing, restore requires the user to choose a valid parent.
- Server trash restore is unavailable unless the server exposes verified list-deleted and restore operations.
- Server soft-deleted records may disappear from normal browse without being restorable from Chatbook in v1.

## Components

Preferred new package names may follow existing Chatbook conventions, but the boundaries should remain explicit:

- `tldw_chatbook/Writing_Interop/writing_models.py`
- `tldw_chatbook/Writing_Interop/local_writing_service.py`
- `tldw_chatbook/Writing_Interop/server_writing_service.py`
- `tldw_chatbook/Writing_Interop/writing_scope_service.py`
- `tldw_chatbook/Writing_Interop/writing_markdown_adapter.py`
- `tldw_chatbook/UI/Screens/writing_screen.py` or `tldw_chatbook/UI/WritingWindow.py`
- `tldw_chatbook/Widgets/Writing/`

Responsibilities:

- Models define normalized records and capability results.
- Local service owns standalone storage and local version history.
- Server service owns API mapping and contract normalization.
- Scope service owns source routing and unsupported-action decisions.
- Markdown adapter owns Markdown-to-server-content and server-content-to-Markdown conversion.
- UI owns presentation, selection state, editor dirty state, and source switching.

## Data Flow

### Local Mode

1. UI sends action to writing scope service with `source=local`.
2. Scope service routes to local writing service.
3. Local service updates local store.
4. Local service returns normalized records.
5. UI updates browse/detail/editor state from normalized records.

Autosave updates only the current entity working state.

`Create New Version` snapshots the current entity working state into the next immutable version.

### Server Mode

1. UI sends action to writing scope service with `source=server`.
2. Scope service checks capability and routes to server writing service.
3. Server service calls the current server API.
4. Server service normalizes server records into Chatbook target records.
5. UI updates browse/detail/editor state from normalized records.

Server records are not persisted as local authored writing records in v1.

## Testing Strategy

### Local Service Tests

Cover:

- project-required manuscript creation
- one project to many manuscripts
- project-level unassigned chapter creation and assignment to a manuscript
- chapter and scene creation
- direct manuscript-level scenes
- scene move between chapters and manuscript parent
- reorder chapters and scenes
- autosave scene working drafts and container metadata/structure working state
- explicit version creation
- read-only historical versions
- restore from version
- soft-delete
- trash restore
- restore conflict when original parent is deleted or missing
- optional status fields

### Server Adapter Tests

Use fake API clients.

Cover:

- project mapping
- part-to-manuscript mapping
- chapter mapping
- scene mapping
- structure normalization
- unassigned chapter normalization without fake manuscripts
- chapter assign/unassign mapping through verified `part_id` update paths
- reorder payloads
- optimistic-lock headers
- soft-delete mapping
- trash/restore capability disabled when server has no restore endpoint
- unsupported direct manuscript-level scenes
- unsupported server scene reparenting
- unsupported server historical versions when no endpoint exists
- Markdown wrapper conversion
- plain-text fallback with fidelity warning
- auth, connection, permission, and conflict failures

### Scope Service Tests

Cover:

- source routing
- local/server hard separation
- no local persistence in server mode
- unsupported action blocking
- capability results for direct scene and versioning support
- capability results for server scene reparenting and trash restore
- consistent normalized records across local/server paths

### UI/State Tests

Cover:

- source switch behavior
- dirty editor source-switch handling
- project and manuscript browser state
- outline display with inline chapters and direct scenes
- outline display for project-level unassigned chapters
- Markdown editing
- autosave behavior
- manual version creation
- historical version read-only state
- restore action
- soft-delete and trash restore
- unsupported server direct-scene affordance disabled
- unsupported server scene-reparent affordance disabled
- unsupported server manual-version affordance disabled when no server endpoint exists
- unsupported server trash-restore affordance disabled when no server endpoint exists

### Regression Tests

Cover:

- writing projects do not appear in `Notes`
- notes and workspaces do not appear in `Writing Suite`
- local writing actions do not call server clients
- server writing actions do not write local records

## Implementation Sequence

1. Normalized models and capability results.
2. Local writing store and service.
3. Markdown adapter.
4. Server writing adapter for supported current server routes.
5. Scope service and capability gates.
6. Source-switched TUI shell.
7. Project and manuscript browse/detail/create/update/delete/restore.
8. Outline tree/list with manuscripts, unassigned chapters, chapters, and scenes.
9. Scene Markdown editor and autosave.
10. Reorder and move operations.
11. Manual versions and read-only historical view.
12. Trash and restore conflict handling.
13. Final parity docs and verification record updates.

## Acceptance Criteria

- Chatbook has a `Writing Suite` destination or screen reachable from the app.
- Local mode works offline for project, manuscript, chapter, scene, scene draft, container working state, manual-version, soft-delete, restore, reorder, and supported move flows.
- Server mode lists and mutates supported server project, part-as-manuscript, chapter, scene, structure, reorder, and soft-delete records without writing local records.
- Server unassigned chapters are visible in a project-level bucket and are never dropped or represented as fake manuscripts.
- Server unsupported actions are disabled centrally and clearly explained.
- Markdown content is preserved locally and through the server adapter convention when possible.
- Historical versions are read-only until explicit restore or draft derivation.
- Local delete is soft-delete with visible trash and restore.
- Server delete maps to the current server soft-delete routes, while server trash restore is disabled unless verified endpoints exist.
- Writing data stays separate from notes and workspaces.
- Tests prove source separation, version behavior, structural editing, and unsupported server capabilities.

## Risks And Mitigations

- **Server model mismatch:** Server parts are not named manuscripts, and scenes cannot be parentless. Mitigate through explicit adapter mapping and capability-gated unsupported actions rather than hidden fake records.
- **Markdown fidelity mismatch:** Server scenes are TipTap/plain rather than Markdown. Mitigate with a deterministic adapter wrapper and visible fallback when only plain text is available.
- **Version-history mismatch:** Local manual versions are required, but server version history may not be present. Mitigate by making server version creation capability-gated and local versioning fully supported.
- **UI scope creep:** Writing could quickly expand into generation/export/revision. Mitigate by keeping v1 structural and deferring writing helpers.
- **Data leakage across domains:** Writing projects must remain separate from notes/workspaces. Mitigate with separate storage, services, screens, and regression tests.

## Follow-On Work

- Server contract extension for direct manuscript-level scenes.
- Server contract extension for first-class Markdown or explicit Chatbook Markdown preservation.
- Server contract extension for manual historical versions if needed.
- Export and publishing.
- LLM outline, revision, critique, and continuation helpers.
- Diffs between versions.
- Sync/mirror design.
- Integration with workflows or research sessions after the structural model is stable.
