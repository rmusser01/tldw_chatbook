# Backend Parity UX Handoff Packet

Date: 2026-04-30

Owner: backend parity

Stability: provisional v1

This packet is the UX-facing contract summary for source-honest backend parity. It is not a UI implementation plan and it should not be inferred from current screens.

## Contract Rules

- UX should consume backend-owned services and contract builders, not raw config or direct server clients.
- Every server-backed action must handle `server_unavailable`, `auth_required`, `permission_denied`, and `capability_missing` reports.
- Local, server, workspace, and remote-only data must remain visually distinct; do not blend records without an explicit sync/mirror report.
- Sync is read-only dry-run only in this phase. No UX should imply automatic write sync, queued mutation replay, or a completed local mirror.
- Remote-only domains are server-owned. UX may show local/offline explanatory disabled states, but should not build local CRUD for those domains unless a row is explicitly promoted to `local_parity`.

## Active Server And Auth Status

Contract path:

- `tldw_chatbook/UX_Interop/server_connection_contracts.py`
- `tldw_chatbook/runtime_policy/server_context.py`
- `tldw_chatbook/runtime_policy/server_credentials.py`

Example active server status:

```json
{
  "schema_version": 1,
  "owner": "runtime_policy",
  "stability": "tranche_1",
  "kind": "active_server_status",
  "active_server_id": "local-dev",
  "label": "Local dev server",
  "reachability": "reachable",
  "auth_state": "authenticated",
  "credential_source": "keyring:chatbook:server:local-dev:access"
}
```

Example auth failure:

```json
{
  "schema_version": 1,
  "owner": "runtime_policy",
  "stability": "tranche_1",
  "kind": "auth_failure",
  "reason_code": "auth_required",
  "message": "Sign in to the active server to continue.",
  "recoverable": true,
  "active_server_id": "local-dev"
}
```

UX implications:

- Show the active server label and auth state anywhere server-backed controls appear.
- For recoverable auth failures, route users to sign in rather than showing generic errors.
- Global sign-out clears all stored server credentials and invalidates credential-bound clients, event streams, and sync handles.

## Source Authority And Source Selectors

Contract path:

- `tldw_chatbook/runtime_policy/domain_edge_contracts.py`

Source selector states:

- `local`: local Chatbook storage/service is authoritative for the current view.
- `server`: active tldw server is authoritative for the current view.
- `workspace`: active server plus workspace/resource scope is authoritative.

Example matrix entry:

```json
{
  "domain_id": "chat",
  "label": "Chat",
  "authority": "local_and_server",
  "source_selector_states": ["local", "server", "workspace"],
  "view_model_contract": "chat_source_honest_view_v1",
  "workspace_isolation": "required",
  "uses_event_contract": true,
  "uses_sync_contract": true,
  "unsupported_local_reason_codes": []
}
```

Domain readiness summary:

| Domain | Source Authority | UX State |
| --- | --- | --- |
| Chat | mixed | Ready for source-separated local/server/workspace UX; chat metadata has read-only dry-run mirror reporting only. |
| Media/Reading | mixed | Ready; ingest/job/event records and dry-run mirror reports are available. |
| Notes/Workspaces | mixed/workspace | Ready; workspace notes stay workspace-scoped and graph gaps are unsupported reports. |
| Writing | mixed | Ready; richer server analysis gaps must be displayed as unsupported actions. |
| Research | mixed/workspace | Ready; run/session/delete/filter gaps are unsupported reports. |
| Study/Evaluations | mixed | Ready; unsupported local/server target gaps are explicit reports. |
| RAG/Embeddings | mixed/server-primary | Ready; local per-media embedding admin and server collection export can be unsupported. |
| Audio/Voice | mixed | Ready; websocket streaming/session controls can be unsupported unless adapter capability reports clear them. |
| Translation | mixed/local parity pilot | Ready; server translation remains available, and local translation is available only when `TranslationScopeService.local_service` is configured. |

Remote-only utility rows are individually ready for UX: sharing, web clipper, server tools, Text2SQL, server skills, claims, meetings, outputs, Kanban, and Prompt Studio. Translation is no longer in the remote-only set; it is the first adapter-backed local parity pilot.

## Unsupported Action Reports

Contract path:

- `tldw_chatbook/runtime_policy/domain_edge_contracts.py`
- `tldw_chatbook/runtime_policy/unsupported_capabilities.py`

Example unsupported report:

```json
{
  "operation_id": "sharing.unsupported.local",
  "source": "local",
  "supported": false,
  "reason_code": "server_required",
  "user_message": "Sharing is owned by the active server and is unavailable in local mode.",
  "affected_action_ids": [],
  "domain_id": "sharing",
  "view_model_contract": "sharing_remote_only_view_v1"
}
```

UX implications:

- Disable the affected action IDs when present.
- If `affected_action_ids` is empty, disable the current domain/source action that produced the report.
- Prefer `user_message` for inline explanations, and keep `reason_code` for telemetry/branching.

## Notification And Event Feed Records

Contract path:

- `tldw_chatbook/Notifications/server_notification_events.py`
- `tldw_chatbook/Notifications/event_state_repository.py`
- `tldw_chatbook/runtime_policy/server_parity_models.py`

Example server notification feed item:

```json
{
  "record_id": "server:notification:notification-42",
  "backend": "server",
  "event_key": "server:local-dev:user-1:notifications:global:event-77",
  "source_event_id": "event-77",
  "server_cursor": "event-77",
  "event_kind": "notification.updated",
  "received_at": "2026-04-30T12:00:00Z",
  "id": "notification-42",
  "title": "Ingest finished",
  "severity": "info"
}
```

UX implications:

- Use `record_id` as the presentation key.
- Use `event_key` for dedupe/debug display, not as a user-facing label.
- Presentation and processed cursors are separate. Marking a notification presented should not imply it was processed by every backend workflow.
- Durable replay is bounded by repository retention. Do not promise infinite event history.
- Feed responses include `replay.state`; show `retention_gap` as "older events require server refetch" rather than implying local history is complete.

Example replay metadata:

```json
{
  "replay": {
    "state": "retention_gap",
    "requested_cursor": "1",
    "earliest_retained_cursor": "2",
    "latest_retained_cursor": "3",
    "last_pruned_cursor": "1",
    "pruned_event_count": 1,
    "server_refetch_required": true
  }
}
```

## Sync Dry-Run And Conflict Records

Contract path:

- `tldw_chatbook/Sync_Interop/sync_scope_service.py`
- `tldw_chatbook/Sync_Interop/sync_state_repository.py`
- `tldw_chatbook/Sync_Interop/sync_mirror_report.py`
- `tldw_chatbook/Sync_Interop/sync_readiness.py`

Read-only dry-run domains:

- `notes`
- `workspace_notes`
- `media`
- `research`
- `chat_metadata`

Example mirror report:

```json
{
  "backend": "server",
  "record_id": "server:sync_mirror_report:12",
  "report_id": 12,
  "report": {
    "domain": "chat_metadata",
    "server_profile_id": "local-dev",
    "workspace_id": "workspace-1",
    "source_authority": "server",
    "source_scope": "workspace",
    "dry_run": true,
    "write_enabled": false,
    "mapped_count": 1,
    "actions": [
      {
        "action": "would_compare",
        "mutation_allowed": false,
        "identity": {
          "domain": "chat_metadata",
          "source_authority": "server",
          "source_scope": "workspace",
          "local_entity_id": "local-conv-1",
          "remote_entity_id": "remote-conv-1",
          "server_profile_id": "local-dev",
          "workspace_id": "workspace-1"
        },
        "local_present": true,
        "remote_present": true
      }
    ]
  }
}
```

UX implications:

- Render dry-run reports as readiness/diagnostic information only.
- Never show "sync enabled" or "will sync automatically" from `dry_run: true`.
- Unknown domains must render an unsupported sync-domain report, not a silent fallback.
- Conflict and orphan states should be shown directly; UX should not infer safe sync from a mapping missing either side.

## Workspace Isolation

Workspace-scoped contracts include a workspace/resource scope in the source selector, identity map, event/sync scope, and user-visible record labels.

Rules:

- `workspace_notes` requires `workspace_id` and must not be rendered as global notes.
- Workspace-scoped sync reports include `workspace_scope`/`workspace_id`; UX must preserve that scope in filters and detail routes.
- Cross-workspace moves, cross-scope note moves, and workspace graph semantics remain unsupported unless a capability report explicitly enables them.

Example disabled cross-scope action:

```json
{
  "operation_id": "notes.graph.unsupported.workspace",
  "source": "workspace",
  "supported": false,
  "reason_code": "scope_not_supported",
  "user_message": "Workspace-scoped notes remain isolated from the global notes graph until sync/graph semantics are designed.",
  "affected_action_ids": [
    "notes.graph.list.server",
    "notes.graph.detail.server"
  ]
}
```

## Unified MCP Local/Server Pane

Contract paths:

- `tldw_chatbook/MCP/unified_control_plane_service.py`
- `tldw_chatbook/MCP/server_target_store.py`
- `tldw_chatbook/MCP/server_unified_service.py`

UX implications:

- Keep local MCP registry state separate from server MCP target state.
- The configured target store owns active server target resolution for MCP views.
- Server switching invalidates server MCP target/client state; local MCP inventory should remain local.

## Error Presentation Rules

Server controls should map backend state to UX as follows:

| Backend State | UX Treatment |
| --- | --- |
| `server_unavailable` | Disable server actions, keep local actions available, show retry/connect affordance. |
| `auth_required` | Disable protected server actions and route to sign-in/global server auth. |
| `permission_denied` | Disable the action and show the permission-specific message. |
| `capability_missing` | Hide or disable the action based on density; show the unsupported report message on demand. |
| `not_implemented_locally` | Disable local action or route user to server source if available. |

## Verification Snapshot

- Credential/auth focused regression: `107 passed`.
- Provider migration audit guard: `10 passed`.
- Remote-only utility focused regression: `142 passed`; includes Translation local parity pilot coverage.
- Notifications focused regression: `65 passed`; includes replay-window retention-gap metadata.
- Primary-domain focused regression: `442 passed, 1 warning`.
- Sync/domain mirror regression: `35 passed`.
- Domain-edge plus sync/domain mirror regression: `39 passed`.

Known non-blocking test noise:

- `requests` dependency warning about urllib3/chardet or charset-normalizer versions.
- SWIG deprecation warnings.
- UI rebuild tests are intentionally non-blocking for this backend handoff packet.
