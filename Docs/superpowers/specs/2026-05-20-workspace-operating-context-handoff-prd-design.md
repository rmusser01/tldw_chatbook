# Workspace Operating Context And Handoff PRD

Date: 2026-05-20
Status: Draft PRD for user review
Primary Repo: `tldw_chatbook`
Adjacent Repo: `rmusser01/tldw_server`
Scope: Product and architecture contract for Chatbook workspaces as portable operating contexts, including Console UX implications, local/server sync, and handoff boundaries.

## Summary

`tldw_chatbook` needs a canonical workspace model before the Console left rail can safely expose a real workspace switcher.

The requested Console change is visually small: split the left panel into an upper `Staged Context` section and a lower `Convos & Workspaces` section. The underlying product requirement is larger. A workspace is not just a filter. It is the operating context that binds source material, conversations, notes, artifacts, ACP sessions, tool/runtime state, git worktrees, and sandbox/container/VM filesystem bindings.

This PRD defines the target model and the phased path. The current Console implementation should pause on functional workspace switching until this contract is approved. A visual shell can be implemented first only if it stays honest about unavailable workspace switching.

## Inputs

- `Docs/superpowers/specs/2026-05-02-new-user-first-run-shell-ux-design.md`
- `Docs/superpowers/specs/2026-05-02-agentic-terminal-design-system-design.md`
- `Docs/superpowers/specs/2026-05-06-destination-layout-ia-contracts-design.md`
- `Docs/superpowers/specs/2026-04-19-chat-conversations-parity-vertical-design.md`
- `tldw_chatbook/Chat/chat_persistence_service.py`
- `tldw_chatbook/Chat/server_chat_conversation_service.py`
- `tldw_chatbook/UI/Screens/notes_scope_models.py`
- `tldw_chatbook/UX_Interop/server_parity_contracts.py`
- `tldw_server` issue [#1526: Product roadmap: canonical workspace and server-backed workspace record](https://github.com/rmusser01/tldw_server/issues/1526)
- `tldw_server` issue [#1440: Track prototype workspace collaboration productionization](https://github.com/rmusser01/tldw_server/issues/1440)
- `tldw_server` issue [#1528: Product roadmap: connector and MCP pack productization](https://github.com/rmusser01/tldw_server/issues/1528)
- `tldw_server2/apps/packages/ui/src/store/workspace.ts`
- `tldw_server2/apps/packages/ui/src/components/Option/ChatWorkspace/*`
- `tldw_server2/tldw_Server_API/app/core/Agent_Orchestration/models.py`
- `tldw_server2/tools/tldw-agent/internal/workspace/session.go`

## Current Evidence And Gaps

The existing codebase already has partial seams, but not a complete workspace operating-context model.

Current useful anchors:

- `ChatPersistenceService.create_conversation()` accepts `scope_type` and `workspace_id`.
- `ServerChatConversationService.delete_conversation()` can pass `scope_type` and `workspace_id` to the server.
- Notes has explicit `ScopeType.WORKSPACE` state plus selected workspace, workspace note, source, and artifact identifiers.
- `WorkspaceIsolationContract` and `FutureSyncStatusContract` already model workspace isolation and future sync status at the server-parity contract layer.
- The current Console left rail is a single `ConsoleStagedContextTray`, so there is a clear widget seam for a two-section rail.
- `tldw_server` issue #1526 explicitly frames workspace as the product unit for sources, chats, notes, generated artifacts, task-agent runs, decisions, and review history.

Current gaps:

- Chatbook does not yet have a single local workspace registry that owns workspace identity and lifecycle.
- Workspace membership is not consistently enforced across conversations, sources, artifacts, notes, ACP, MCP, schedules, workflows, and Chatbooks.
- Chatbook does not yet distinguish item visibility from active-context eligibility. Workspace switching must not make Notes or Library items disappear.
- Sync/handoff is not implemented. Existing contracts describe future sync state but do not perform local/server migration.
- Runtime bindings are not portable. A local path, git worktree, container, VM, or ACP session cannot currently be moved as a safe, user-visible package.
- Console can show staged context, but it cannot yet truthfully switch the current operating workspace.

## Recommended Initial Product Decisions

These decisions keep the first implementation useful without overreaching.

- Do not add a new top-level `Workspaces` destination in v1. Expose workspace context in Console and deeper workspace management under Library until the model proves it needs a dedicated destination.
- Workspace switching changes active-context eligibility, not Library or Notes visibility. Users can still see, search, open, and edit items from other workspaces, but cannot stage them into the active Console context unless they belong to the current workspace or are explicitly copied/linked into it.
- Start with manual push/pull handoff. Background write sync should remain deferred until conflict handling and replay safety are proven.
- Support both source-content copy and source-reference upload in local-to-server handoff. The default should depend on source type, source location, server policy, and user choice.
- Make the first server-backed handoff target ACP task/run packages, because this is the highest-value portable operating-context use case.
- Treat moving a conversation between workspaces as a fork with provenance, not a silent reassignment.
- Make v1 runtime bindings metadata-first. Local filesystem and git worktree bindings can be described and validated; ACP sessions, containers, and VMs should be inspect-only until recreation safety is designed.
- Expose audit and diagnostic detail to users by default. The UI may collapse verbose details, but it should not hide them behind developer-only logs. Secrets still require explicit redaction/reveal handling.
- Shared or collaborator workspace packages should degrade to a single-user local workspace when offline.
- Ship the Console `Convos & Workspaces` rail as read-only first if the persistence model is not ready. Disabled controls must explain what is missing.

## Workspace Unification Model

Workspace unification should happen through a shared registry and shared context contract, not by turning every screen into a workspace screen.

The unified model has four layers:

1. `Workspace Registry`: the durable list of local and server-backed workspace records, membership tags, authority state, sync state, and runtime bindings.
2. `Active Workspace Context`: the current operating workspace used by Console, ACP, MCP, Workflows, Schedules, and context staging.
3. `Global Item Browser`: Library, Notes, Artifacts, and search surfaces can show all user-owned items with workspace tags and eligibility badges.
4. `Context Eligibility Gate`: actions that involve the active Console context, agent manipulation, RAG staging, ACP runs, or tool use are allowed only when the selected item belongs to the active workspace or is explicitly copied/linked into it.

If a future top-level `Workspaces` destination is added, it should be a management and status surface over the same registry: workspace list, sync health, memberships, handoff actions, audit logs, and runtime bindings. It must not become a second Library or a second Console.

## Problem Statement

The current Chatbook Console is becoming the primary agentic control surface. Users now need a way to understand and switch the current operating context without leaving Console.

The risk is building a narrow UI selector that looks like a workspace switcher but does not actually protect workspace boundaries. If a workspace owns conversations, sources, ACP sessions, git worktrees, and runtime state, then switching the workspace affects:

- which conversations are eligible for the active Console context.
- which Library/RAG sources can be staged, not which sources can be viewed.
- which Chatbooks and artifacts can be used by the active workspace.
- which ACP sessions and tool permissions are valid.
- which git worktree or sandbox filesystem the agent may read or edit.
- whether data is local-only, server-backed, syncing, conflicted, or portable.

Without a product contract, UI work can create false affordances, leak data across workspaces, or make local/server handoff impossible to reason about.

## Goals

- Define `Workspace` as Chatbook's portable operating-context package.
- Keep Console as the primary live work surface while giving it visible workspace context.
- Support future local-to-server and server-to-local handoff of active work.
- Preserve global visibility and search across Library, Notes, Artifacts, and conversations while enforcing strict active-context eligibility by workspace.
- Make local, server, syncing, conflict, detached, and unavailable states visible.
- Align Chatbook with the `tldw_server` canonical workspace roadmap instead of inventing a second model.
- Decompose the work into PR-sized phases with screenshot-based UX approval gates.

## Non-Goals

- Do not implement full real-time collaboration in this PRD.
- Do not wire a functional workspace switcher before workspace persistence and isolation rules exist.
- Do not replace the current Console transcript/composer work.
- Do not move all workspace management into Console. Console displays and uses the active workspace; Library, ACP, MCP, Artifacts, and Settings retain their own ownership boundaries.
- Do not make ACP runtime setup a Settings concern. ACP owns runtime/session setup; Settings owns global defaults.
- Do not collapse MCP tools into generic settings. MCP remains server-first with tool bundles scoped by workspace policy later.
- Do not make server sync mandatory for local-first use.

## Product Model

### Workspace

A workspace is a named operating context containing or referencing:

- Library sources: files, media, notes, conversations, snippets, citations, imports, and search/RAG evidence.
- Conversations: global or workspace-scoped chat sessions, including active drafts and message variants.
- Artifacts: Chatbooks, generated outputs, reports, datasets, and saved deliverables.
- Notes and study state: workspace notes, flashcards, quizzes, and study packs.
- ACP state: agent sessions, task runs, approvals, review history, and runtime readiness.
- MCP/tool state: enabled servers, workspace-scoped tools, trusted paths, and policies.
- Filesystem/runtime bindings: local directory, git repository, git worktree, sandbox/container/VM root, or remote runtime binding.
- Sync metadata: local identity, server identity, version, authority, conflict state, and handoff history.

### Conversation Scope

Conversation scope must remain explicit:

| Scope | Meaning |
| --- | --- |
| `global` | General Console conversation, not bound to a workspace. |
| `workspace` | Conversation belongs to exactly one workspace and must not appear in another workspace's conversation list. |

Existing Chatbook code already has `scope_type` and `workspace_id` seams in chat persistence and server conversation deletion. The PRD treats those as compatibility anchors, not a complete workspace implementation.

### Visibility Versus Context Eligibility

Workspace switching must not hide user-owned content from Library, Notes, Artifacts, or global search.

Items have workspace membership metadata:

- `workspace_ids`: zero, one, or many workspace memberships.
- `workspace_labels`: display labels for each membership.
- `active_context_eligible`: whether the item can be staged, manipulated, or used by the current Console workspace.
- `eligibility_reason`: user-facing explanation when the item is visible but not eligible.

Examples:

- A user in Workspace A can search, open, and edit a Note from Workspace B in Notes.
- The same Note cannot be added to Workspace A's active Console context unless the user explicitly copies or links it into Workspace A.
- Library can show all media and documents with workspace tags.
- Console staging, agent edits, RAG grounding, ACP runs, and tool actions are limited to the active workspace.

Cross-workspace items should show disabled active-context actions with recovery copy such as `Belongs to Workspace B. Copy or link into Workspace A before staging.`

### Workspace Authority

Every workspace has one visible authority state:

| Authority | Meaning |
| --- | --- |
| `local-only` | Workspace exists only in this Chatbook instance. |
| `server-backed` | Workspace has a durable server identity and can be resumed from server. |
| `syncing-to-server` | Local package is uploading or reconciling to server. |
| `syncing-from-server` | Server package is materializing locally. |
| `conflict` | Local and server state disagree and need user choice. |
| `detached` | Local workspace was once server-backed but cannot verify the server identity. |
| `remote-only` | Server workspace is visible but not materialized locally. |
| `runtime-missing` | Metadata exists, but the ACP/runtime/filesystem binding cannot be restored. |

## Console UX Contract

The Console left rail should become a two-section context rail.

```text
+------------------------------------------+
| Staged Context                           |
| No live work item is staged.             |
| Attach Library, runs, Artifacts, or RAG. |
| [Open Library]                           |
+------------------------------------------+
| Convos & Workspaces                      |
| Workspace: Local Research                |
| Authority: local-only | Sync: Not set    |
| Runtime: local fs | ACP: none            |
|                                          |
| Conversations                            |
| > Chat 1                  workspace      |
|   API migration notes     2h ago         |
|   Untitled draft          unsaved        |
|                                          |
| [Change workspace] [New conversation]    |
+------------------------------------------+
```

### Section Behavior

`Staged Context` remains the fast answer to "what will this send/use right now?"

`Convos & Workspaces` answers:

- What workspace am I operating in?
- Is this workspace local, server-backed, syncing, or conflicted?
- Which conversations are available in this workspace?
- Is the current conversation global or workspace-bound?
- What runtime or ACP binding will the agent use?
- Are visible Library/Notes/Artifact items eligible for the active Console context?

### Immediate UI Boundary

A shell-only PR may split the rail and show read-only workspace/conversation metadata. It must not pretend to support true workspace switching.

Until the workspace model exists:

- `Change workspace` should be disabled or open an explanatory unavailable state.
- Conversation rows should come only from currently trustworthy session metadata.
- Any missing workspace service must render as `No workspace selected` or `Workspace service not ready`, not as a fake default workspace.
- Library and Notes should continue to show all items. Workspace tags and eligibility states should be added before any active-context restrictions are enforced.
- Screenshot approval is required before this shell ships.

## Handoff Model

### Local To Server

When a user hands off an active local workspace to a server:

1. Chatbook builds a workspace manifest.
2. Chatbook validates what can be copied, referenced, redacted, or must remain local.
3. The user sees a preflight summary: sources, conversations, artifacts, ACP sessions, runtime bindings, secrets omitted, and conflicts.
4. Chatbook creates or upserts a server workspace record.
5. Transfer proceeds in resumable stages.
6. The active Console context changes to `syncing-to-server`, then `server-backed` or `conflict`.
7. Any untransferable runtime binding becomes an explicit recovery item.

The v1 server-backed handoff target is ACP task/run packages. Source and conversation metadata should be included when required to make the ACP package intelligible, but the first proof should optimize around preserving task/run operating context.

Source transfer supports both modes:

- `copy`: upload source content or a portable bundle to the server when policy allows.
- `reference`: upload a stable source reference, pointer, or metadata-only binding when copying is unnecessary, disallowed, or too expensive.

The handoff preflight must show which mode each source uses.

### Server To Local

When a user pulls a server workspace into Chatbook:

1. Chatbook fetches the workspace manifest and compatibility metadata.
2. The user chooses materialization scope: metadata only, references, copied source content, or full local package where supported.
3. Chatbook maps remote sources to local references or downloads copies according to policy.
4. Runtime bindings are recreated only if safe and explicitly approved.
5. The workspace enters `syncing-from-server`, then `server-backed`, `runtime-missing`, or `conflict`.

### Round Trip

Round-trip handoff is allowed only when identity and version checks pass:

- local workspace id.
- server workspace id.
- manifest version.
- content hashes or stable external references.
- conversation scope keys.
- ACP/session/run lineage.
- source membership lineage.

If checks fail, the user must choose merge, fork, replace local, replace server, or cancel.

## Workspace Manifest

The manifest is the portability contract. It should be versioned and serializable.

Minimum fields:

- `manifest_version`
- `workspace_id`
- `server_workspace_id`
- `display_name`
- `description`
- `authority_state`
- `created_at`
- `updated_at`
- `owner`
- `source_memberships`
- `conversation_memberships`
- `artifact_memberships`
- `note_memberships`
- `study_memberships`
- `acp_session_bindings`
- `mcp_policy_bindings`
- `runtime_bindings`
- `sync_state`
- `conflict_state`
- `redaction_report`
- `handoff_history`

Runtime bindings must not contain raw secrets. Environment variables, keys, tokens, private filesystem paths, and server credentials need explicit redaction or pointer semantics.

Membership fields should not be interpreted as browse filters. They determine active-context eligibility and provenance display. Browsing and search remain global by default unless the user applies an explicit workspace filter.

## Data And Service Boundaries

### Chatbook Local

Chatbook owns:

- local workspace cache and local-only workspace records.
- local conversation association.
- local source and artifact references.
- cross-workspace Library/Notes/Artifacts browsing with workspace membership labels.
- active-context eligibility checks for Console staging and agent manipulation.
- user-visible sync state.
- local runtime binding inspection.
- export/import package creation.

### Server

The server owns:

- canonical server workspace records.
- server-side workspace authorization.
- team ownership, retention, audit, and governance later.
- server-side source/artifact/task membership.
- server-backed ACP task/session state.
- sharing and collaborator flows.

### ACP

ACP owns:

- ACP workspace/session runtime setup.
- agent task/run state.
- branch/session/runtime payload validation.
- execution-specific recovery.

Console may inspect or follow ACP state, but it should not become the ACP setup surface.

### MCP

MCP owns:

- server/tool lifecycle.
- workspace-specific tool bundles.
- trusted path and governance policy.

Console may show tool readiness and use tools in context, but MCP server management stays in MCP.

## Security And Privacy Requirements

- Workspace export/handoff must show a redaction report before transfer.
- All audit and diagnostic detail should be user-accessible by default through expandable UI, logs, or export. Diagnostic detail should not be developer-only.
- Secrets and credentials are never embedded directly in a workspace manifest.
- Filesystem paths must be classified as portable, local-only, or unsafe.
- Server workspace reads and writes must enforce exact workspace scope.
- ACP runtime payloads must be validated before local execution.
- Cross-workspace Console context use and agent manipulation default to denied. Direct user browsing and editing in Library/Notes can remain available when normal item permissions allow it.
- Audit events are required for server handoff, server pull, share, conflict resolution, and runtime binding recreation.

## Error And Recovery States

Every workspace-aware surface needs visible recovery for:

- server unreachable.
- server auth expired.
- workspace not found.
- workspace archived or deleted.
- missing local source file.
- missing local git worktree.
- dirty or conflicted git worktree.
- unavailable sandbox/container/VM runtime.
- ACP session cannot resume.
- MCP tool disabled by policy.
- conversation belongs to another workspace.
- visible item belongs to another workspace and is not active-context eligible.
- source exists but embeddings/RAG index are not ready.
- partial transfer interrupted.
- manifest version unsupported.

## Phased Implementation Plan

### Phase 0: PRD And Decision Record

- Approve this PRD.
- Add a decision record for Chatbook's workspace model and server alignment.
- Record that workspace unification is a shared registry plus active workspace context, with a possible later top-level management surface only if v1 proves it is needed.

### Phase 1: Console Context Rail Shell

- Split Console left rail into `Staged Context` and `Convos & Workspaces`.
- Show current conversation scope and read-only workspace/session metadata.
- Keep workspace switching disabled unless a real service is present.
- Add screenshot approval gate for the rendered Console.

### Phase 2: Local Workspace Registry

- Add local workspace records and service APIs.
- Associate conversations, source refs, notes, and artifacts with workspace ids.
- Support create, rename, archive, delete, switch, and list operations.
- Add workspace membership labels in Library, Notes, Artifacts, and conversation browsing.
- Enforce active-context eligibility without hiding cross-workspace Library or Notes items.

### Phase 3: Workspace Package Manifest

- Define export/import manifest schema.
- Add dry-run validation for missing files, unsupported source refs, and redacted values.
- Add local package export/import with no server requirement.

### Phase 4: Server Bridge

- Add server workspace identity mapping.
- Add local-to-server and server-to-local dry-run flows with both copy and reference transfer modes.
- Make ACP task/run packages the first server-backed handoff target.
- Add resumable handoff status and conflict reporting.
- Keep write sync disabled until conflict semantics are proven.

### Phase 5: ACP, MCP, And Runtime Bindings

- Add ACP session/run bindings to the manifest.
- Add git worktree and sandbox/container/VM binding descriptors.
- Add MCP policy/tool bundle references.
- Require explicit user approval before recreating executable/runtime bindings.

### Phase 6: UX Closeout And Cross-Screen QA

- Verify Home, Console, Library, Artifacts, ACP, MCP, Schedules, Workflows, and Skills against the same active workspace.
- Verify local-only, server-backed, syncing, conflict, and runtime-missing states.
- Verify switching workspaces does not hide Library or Notes items, and that cross-workspace items remain viewable/editable while active-context actions are blocked with recovery copy.
- Capture actual Textual Web/CDP screenshots for every affected screen before approval.

## Testing And QA Requirements

- Unit tests for manifest validation, redaction, membership tagging, active-context eligibility, and conflict classification.
- Service tests for local workspace CRUD and conversation/source/artifact membership.
- Mounted Textual tests for the Console context rail shell.
- Contract tests for local/server authority labels and disabled-action recovery copy.
- Integration tests for local export/import dry runs.
- Later server integration tests for handoff preflight and interrupted-transfer resume.
- Actual CDP screenshots for every UI approval gate.

## Resolved Decisions From User Review

- Workspace switching must not hide Notes, Library items, Artifacts, or conversation records from global browse/search. Workspace tags and active-context eligibility replace hard filtering.
- Unified workspace behavior comes from a shared Workspace Registry, Active Workspace Context, Global Item Browser, and Context Eligibility Gate. A future top-level Workspaces screen would manage the same registry rather than replacing Console or Library.
- Local-to-server handoff supports both copied source content and uploaded source references, depending on source type and policy.
- The first server-backed handoff target is ACP task/run packages.
- Audit and diagnostic details should be visible to users by default, with expandable presentation and export paths.
- Shared or collaborator workspaces map to single-user local workspaces when offline.

## Remaining Deferred Questions

1. Which source types default to copy versus reference in the first server bridge?
2. What is the first UI for copying or linking a visible cross-workspace item into the active workspace?
3. Which ACP task/run fields are required for a minimally useful server-backed handoff package?
4. What exact redaction controls are required for user-visible audit exports?
5. How should multi-workspace membership be displayed in compact terminal widths?

## Approval Gate

Implementation should not resume on workspace switching until this PRD is reviewed and approved.

If only the visual left-rail split is approved first, it must ship as a read-only shell with honest disabled states and actual screenshot approval.
