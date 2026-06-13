# Notes And Workspaces Domain Edge Contract

Date: 2026-04-29

Status: Lane D contract

Related code:

- `tldw_chatbook/Notes/notes_scope_service.py`
- `tldw_chatbook/Notes/server_notes_workspace_service.py`
- `tldw_chatbook/Sync_Interop/sync_scope_service.py`
- `tests/Notes/test_notes_scope_service.py`
- `tests/Notes/test_server_notes_workspace_service.py`
- `tests/Sync_Interop/test_sync_scope_service.py`

## Scope

This contract covers non-UI notes and workspaces parity edges.

In scope:

- Notes graph semantics.
- Workspace-aware sync design boundaries.
- Local/offline graph generation decision.
- Cross-scope move decision.
- Workspace isolation rules.
- Required focused service tests.

Out of scope:

- UI/UX redesign.
- Broad UI tests.
- Write-enabled sync or replay workers.
- Cross-scope moves.
- Workflow integration.
- Local mirroring of server-owned workspace resources.

## Source Authority

| Edge | Source owner | Contract |
|---|---|---|
| Local notes | Local | Local note CRUD/search/keyword state remains local-authoritative. |
| Server user-space notes | Server | Server note CRUD/search/detail remains active-server-owned. |
| Workspace records | Server workspace scope | Workspace CRUD is server-owned and requires explicit workspace identity. |
| Workspace notes | Server workspace scope | Workspace notes remain contained by workspace ID and must not appear in general user-space lists/search. |
| Workspace sources/artifacts | Server workspace scope | Workspace subresources are scoped to the selected workspace. |
| Notes graph | Server user-space notes only | Graph operations are currently server-backed for user-space server notes. |
| Sync/mirror | Deferred dry-run/read-only | No write replay, no local mirror authority, no cross-source mutation dispatch. |

## Graph Semantics

Current graph contract:

- `notes.graph.*.server` actions apply only to server user-space notes.
- Graph nodes and edges returned by the server are authoritative server graph records.
- Manual graph links are server-owned through `create_note_link` and `delete_note_link`.
- Local note graph and workspace note graph operations must hard-stop before backend dispatch.
- Workspace notes are excluded from the global notes graph until workspace graph/sync semantics are explicitly designed.

Required graph record semantics for any shared model:

- Node IDs must be source-qualified when surfaced through a shared seam.
- Edge IDs must preserve server identity and edge type.
- Edge types must not be inferred from label text.
- Graph responses must preserve truncation/cursor metadata when supplied.
- Graph reads must not write local backlinks, local keywords, or workspace links.
- Local generated graphs, if later adopted, must use a different source marker and must not be mixed with server graph mutation APIs.

## Local/Offline Graph Generation Decision

Decision: local/offline graph generation remains deferred as an invokable product edge.

The codebase contains helper logic that can derive local note/tag/manual-link graph shapes, but the public scope service still hard-stops local graph operations. That is the intended Lane D contract.

Rules:

- `get_notes_graph`, `get_note_neighbors`, `create_note_link`, and `delete_note_link` must require `ScopeType.SERVER_NOTE`.
- Local graph calls must not dispatch to local note services.
- Workspace graph calls must not dispatch to server graph APIs.
- Unsupported reports must include `notes.graph.local` and `notes.graph.workspace`.
- A future local graph adoption must first define node/edge identity, keyword-edge provenance, manual-link persistence, deletion semantics, and sync eligibility.

## Workspace Isolation

Workspace content must remain scope-contained.

Rules:

- General local note list/search returns local notes only.
- General server note list/search returns server user-space notes only.
- Workspace note list/search/detail requires `workspace_id`.
- Workspace notes must not be flattened into user-space note list/search results.
- Workspace sources and artifacts require the selected workspace context.
- Workspace content must not leak into general user-space lists/search unless the caller uses an explicitly workspace-scoped operation.
- Deleting or archiving a workspace must not convert its notes to user-space notes in Chatbook.

## Workspace-Aware Sync Boundaries

Sync remains dry-run/read-only for notes and workspaces in this phase.

Allowed:

- Read-only readiness reports.
- Dry-run/mirror reports that describe what would be eligible later.
- Source-aware unsupported reports for unsyncable scopes.
- Per-server identity metadata planning.

Forbidden:

- Write replay workers.
- Remote mutation dispatch from sync.
- Authoritative local mirror copies of server notes/workspaces.
- Sync queues that can replay against a different active server.
- Merging workspace notes into user-space note identity maps.
- Cross-scope moves through sync.

Workspace-aware sync design requirements for future work:

- Identity maps must include source scope: local note, server user-space note, workspace note, workspace source, or workspace artifact.
- Workspace-scoped records must include `workspace_id` in identity keys.
- Server profile ID must be part of any remote identity/cursor key.
- Create/update/delete parity or explicit unsupported-operation handling must exist before write sync.
- Conflict strategy, version/ETag/hash strategy, and redaction rules must be defined per entity kind.

## Cross-Scope Moves

Decision: cross-scope moves remain deferred.

Deferred move classes:

- Local note to server user-space note.
- Server user-space note to local note.
- Server user-space note to workspace note.
- Workspace note to server user-space note.
- Workspace note between workspaces.
- Workspace source/artifact to another workspace.

Required behavior:

- Do not emulate moves by copy/delete.
- Do not expose move as a sync operation.
- Do not silently import workspace notes into local notes.
- Future adoption must define identity continuity, history/audit semantics, permission checks, version handling, conflict behavior, and rollback behavior.

Unsupported reason code for future reporting: `contract_deferred`.

## Unsupported Capabilities

Required unsupported reports:

| Operation ID | Source | Reason code | Contract |
|---|---|---|---|
| `notes.graph.local` | `local` | `local_contract_missing` | Local/offline notes graph generation and manual links are deferred. |
| `notes.graph.workspace` | `workspace` | `scope_not_supported` | Workspace notes are isolated from the global notes graph. |
| `notes.cross_scope_move.local_to_server` | `local` | `contract_deferred` | Local-to-server note moves are deferred. |
| `notes.cross_scope_move.server_to_local` | `server` | `contract_deferred` | Server-to-local note moves are deferred. |
| `notes.cross_scope_move.server_to_workspace` | `server` | `contract_deferred` | Server user-space to workspace note moves are deferred. |
| `notes.cross_scope_move.workspace_to_server` | `workspace` | `contract_deferred` | Workspace to server user-space note moves are deferred. |
| `notes.cross_scope_move.workspace_to_workspace` | `workspace` | `contract_deferred` | Workspace-to-workspace moves are deferred. |
| `notes.sync.write_replay` | `shared` | `contract_deferred` | Write replay workers are forbidden in this phase. |

The graph rows are currently surfaced by `NotesScopeService.list_unsupported_capabilities`. Cross-scope move and write-replay rows are contract rows for future unsupported reporting once callers can attempt those operations.

## Required Service Tests

Existing focused service tests are the required coverage:

- `tests/Notes/test_notes_scope_service.py::test_scope_service_keeps_server_note_search_api_backed`
- `tests/Notes/test_notes_scope_service.py::test_scope_service_keeps_workspace_note_search_client_side`
- `tests/Notes/test_notes_scope_service.py::test_scope_service_routes_server_notes_graph_operations`
- `tests/Notes/test_notes_scope_service.py::test_scope_service_rejects_local_notes_graph_operations_explicitly`
- `tests/Notes/test_notes_scope_service.py::test_scope_service_reports_known_notes_graph_capability_gaps`
- `tests/Notes/test_server_notes_workspace_service.py::test_service_filters_workspace_notes_within_active_workspace_only`
- `tests/Notes/test_server_notes_workspace_service.py::test_service_delegates_notes_graph_operations_to_server_client`
- `tests/Notes/test_server_notes_workspace_service.py::test_service_enforces_notes_graph_policy_actions`
- `tests/Sync_Interop/test_sync_scope_service.py` focused dry-run/read-only sync tests.

No additional tests are required for this contract because the current service tests already cover workspace isolation, graph hard stops, graph server routing, and dry-run sync boundaries. Cross-scope moves remain documentation-only until an invokable move operation exists.
