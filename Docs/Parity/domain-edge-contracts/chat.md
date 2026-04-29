# Chat Domain Edge Contract

Date: 2026-04-29

Status: Lane D contract

Related code:

- `tldw_chatbook/Chat/chat_conversation_scope_service.py`
- `tldw_chatbook/Chat/chat_loop_scope_service.py`
- `tldw_chatbook/Chat/server_chat_conversation_service.py`
- `tldw_chatbook/Chat/server_chat_loop_service.py`
- `tests/Chat/test_chat_conversation_scope_service.py`
- `tests/Chat/test_server_chat_loop_service.py`

## Scope

This contract covers non-UI chat parity edges for Chatbook as a standalone local-first client and connected `tldw_server` client.

In scope:

- Local chat-loop execution decision.
- Server chat-loop run/event/approval/cancel boundary.
- Streaming and persistence handoff rules.
- Source-separated local/server conversation history.
- Unsupported-capability IDs and reason codes.
- Required focused service tests.

Out of scope:

- Chat workflows.
- UI/UX redesign.
- Local/server sync or dual-write.
- MCP SDK usage.
- Broad UI tests.

## Source Authority

| Edge | Source owner | Contract |
|---|---|---|
| Main local chat execution | Local | Existing Chatbook chat path remains the local execution path. It is not reimplemented as a local chat-loop run controller in this tranche. |
| Server chat-loop run lifecycle | Server | `ServerChatLoopScopeService` and `ChatConversationScopeService` may launch, observe, approve/reject, and cancel only when the requested mode is `server`. |
| Local persisted conversations | Local | Local conversation list/detail/tree/update/create/delete stay local-authoritative and use local IDs. |
| Server persisted conversations | Server | Server conversation list/detail/tree/update/context/citations stay server-authoritative and use server IDs. |
| Server conversation create/delete | Deferred | The current server conversation contract does not expose first-class create/delete outside launch/persist flows; Chatbook must hard-stop these calls. |
| Server adjunct controls | Server | Commands, knowledge save, share links, and analytics are server-only adjuncts. |

## Local Chat-Loop Execution Decision

Decision: local chat-loop run control is deferred.

Chatbook already has local chat execution through the existing chat send/regenerate/continue path. Lane D must not introduce a second local loop runner or fake local run/event store. Local mode must report `chat.loop.local` as unsupported until a real local run lifecycle contract exists.

Required behavior:

- Local `start_loop`, `list_loop_events`, `approve_loop_call`, `reject_loop_call`, and `cancel_loop` must raise a typed hard stop rather than dispatching to any local or server client.
- The hard stop must not create synthetic server run IDs.
- The hard stop must not persist local messages as if a server loop completed.
- Unsupported reports must name all affected local loop actions.

## Streaming And Persist Handoff

The chat execution seam has two different handoff models:

- Local chat streaming remains owned by the existing Chatbook chat worker and local persistence path.
- Server chat-loop events remain owned by server run/event APIs.

Rules:

- A server chat-loop run ID is not a local conversation ID.
- Server loop events are observation records, not local message history unless a later explicit persist/import operation exists.
- Streaming chunks may update transient UI state only after the stream source is known.
- Final persistence must happen through the owning source only.
- A local conversation must not be appended from server events unless the user triggers a future explicit import/attach flow.
- Server persisted conversation update calls must not be used to emulate server create/delete.
- If the server returns a persisted conversation reference during chat launch, Chatbook may display or fetch it through server conversation detail APIs; it must not write it into the local conversation table as authoritative history.

## Source-Separated History

History views and service results must preserve source separation:

- Local history reads local conversations only.
- Server history reads active-server conversations only.
- Workspace-scoped server conversations must require explicit workspace filters and must not leak into general local history.
- Normalized IDs must retain a source prefix or source field where records are surfaced through shared service seams.
- Server switching invalidates server conversation/run/event assumptions; cached server history must be scoped by active server profile if cached later.
- Sync remains read-only/dry-run in this phase and must not mirror chat records.

## Unsupported Capabilities

Required unsupported reports:

| Operation ID | Source | Reason code | Contract |
|---|---|---|---|
| `chat.loop.local` | `local` | `local_contract_missing` | Local run-control semantics are not implemented. |
| `chat.adjunct_controls.local` | `local` | `local_contract_missing` | Server adjunct controls are unavailable in local mode. |
| `chat.conversation.create.server` | `server` | `server_contract_missing` | Server first-class conversation creation is unavailable outside launch/persist flows. |
| `chat.conversation.delete.server` | `server` | `server_contract_missing` | Server conversation deletion is unavailable. |

Hard stops must dispatch no backend call after deciding the operation is unsupported.

## Required Service Tests

Existing focused service tests are the required coverage:

- `tests/Chat/test_chat_conversation_scope_service.py::test_scope_service_routes_server_chat_loop_and_rejects_local_loop`
- `tests/Chat/test_chat_conversation_scope_service.py::test_scope_service_blocks_unsupported_server_conversation_create_delete_before_dispatch`
- `tests/Chat/test_chat_conversation_scope_service.py::test_scope_service_routes_server_chat_adjunct_controls_and_rejects_local_mode`
- `tests/Chat/test_chat_conversation_scope_service.py::test_scope_service_reports_known_chat_conversation_capability_gaps`
- `tests/Chat/test_server_chat_loop_service.py::test_chat_loop_scope_service_rejects_local_mode_before_policy_dispatch`
- `tests/Chat/test_server_chat_loop_service.py::test_chat_loop_scope_service_enforces_server_mode_policy_and_normalizes_ids`

No additional tests are required for this contract because the current service tests already cover the required non-UI hard stops and server-mode routing.
