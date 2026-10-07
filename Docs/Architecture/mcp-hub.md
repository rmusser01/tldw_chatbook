# MCP hub and permission store

This document describes the MCP package: the unified control-plane service behind the MCP Hub screen, external stdio server management, the permission store and its decision semantics, the standalone built-in MCP server, and the gateway runtime for external tool exposure.

## Authoritative files

| File | Role |
| --- | --- |
| `MCP/permission_store.py` | `MCPPermissionStore` — schema-versioned JSON store; pure resolvers `resolve_effective_state`, `resolve_builtin_state`, `arg_rule_allows`, `definition_hash` |
| `MCP/hub_tool_catalog.py` | `HubTool` normalization (`tool_id = "<server_key>::<name>"`), name dedup |
| `MCP/unified_control_plane_service.py` | `UnifiedMCPControlPlaneService` — the Hub's service: catalogs, `effective_tool_states`, `set_tool_state`, arg rules, session approvals, `execute_hub_tool`, timeouts |
| `MCP/local_control_service.py` | `LocalMCPControlService` — external profile CRUD, connect/disconnect, governance rules, approval requests, runtime activity log |
| `MCP/local_runtime_delegate.py` | `LocalMCPRuntimeDelegate` — status/health/protocol diagnostics |
| `MCP/client.py` | `MCPClient` + `_StdioJSONRPCConnection` — spawn (no shell), initialize/list_tools/call_tool, read and stderr loops, transport-failure cleanup, per-server dispatcher for sampling/elicitation (default deny) |
| `MCP/local_store.py` | `LocalExternalMCPProfile` records in `local_mcp_store.json`; env placeholder sanitization |
| `MCP/spawn_guard.py` | Command-shape screening at save/spawn/import (interpreter `-c`, fetchers, encoded markers, sensitive paths) |
| `MCP/gateway_runtime.py` | `ChatbookGatewayRuntime` — adapter from Chatbook decorators to the strict external `mcp_unified.gateway` (optional extra) |
| `MCP/server.py` | `TldwMCPServer` — the standalone built-in server (`python -m tldw_chatbook.MCP`, stdio) |
| `MCP/execution_log.py` | Execution audit rows and decision tokens (`POLICY_DENIED_DECISION`, `KILL_SWITCH_DENIED_DECISION`, `UNRESOLVED_DENIED_DECISION`) |

## Permission store

The store is a strict-validated JSON file at `<user_data_dir>/mcp_permissions.json`:

```
schema_version: 1
kill_switch: false
profiles:
  default (and named profiles):
    global_default: ask
    servers:
      "<source>:<server_id>":
        default: ask
        tools:
          <tool_name>: { state: allow|ask|deny, definition_hash: …, arg_rules: […] }
```

- **States**: `allow` / `ask` / `deny`; the default global is `ask`.
- **Profiles**: every mutator and resolver takes `profile_id` (default `"default"`). Named workspace profiles exist so an "Always allow" persists into the active workspace profile. The resolver walks `[named, default]` per level: named tool override → named server default → named global → the same three in `default` → `ask`.
- **Atomicity**: `.tmp` sibling + `Path.replace`; per-path locks. Corruption policy: never raise out of `load()` — missing file → fresh default; corrupt or wrong-version file → renamed `.bak` + fresh default.
- `SCHEMA_VERSION` stays 1 even though `profiles` was added — a bump would trigger the corrupt-file reset and destroy every user's permissions (ADR-079).

### Decision semantics (`resolve_effective_state`)

1. Lifecycle block (imported/tombstone profiles) → deny.
2. Precedence per profile in the chain: tool override > server default > global default; absent = inherit.
3. **Rug-pull guard**: a tool-level `allow` is downgraded to `ask` (flagged `config_changed`) when the stored `definition_hash` no longer matches the live tool's description+schema hash. A fresh explicit `set_tool_state` clears the flag.
4. **High-risk floor**: an *inherited* allow (server/global origin) is downgraded to `ask` for tools tagged `mutates` or `process`. Explicit tool-level allows are never floored. Builtins use a wider high-risk set (adds `reads`, `network`) and skip hash comparison; their fallback default is `allow` because they are in-process code.
5. **Arg rules**: an exact canonical `args_json` match (the shape the approval card writes) or a hand-written `field`+`pattern` fnmatch rule. Never applies to high-risk tools; inert on hash mismatch.
6. Resolver failures degrade to "Ask" with a gate-error origin — never a `KeyError`, never a false "set to Off" claim.

## External stdio servers (dataflow)

1. Server configs are `LocalExternalMCPProfile` records in `local_mcp_store.json`. Environment entries use `$VAR` placeholders resolved at spawn; literal secrets are rejected, and `spawn_guard` screens the command line at save/spawn/import.
2. `LocalMCPControlService.connect_profile` launches the process (`create_subprocess_exec`, no shell) under the governance action `mcp.external_profiles.launch.local`; `_StdioJSONRPCConnection` performs initialize / list_tools / call_tool with transport-failure cleanup.
3. The discovery snapshot is persisted; a profile reporting no capabilities is disconnected and refused.
4. `MCPToolProvider.compose_catalog()` (see [tool-catalog.md](./tool-catalog.md)) turns the snapshot into agent-visible tools under one batched `effective_tool_states` resolution.

## Permission decision lifecycle (one tool call)

1. The model emits a call; the Console review hook clears per-run decision stamps at entry (so a raising approval round can never leak a prior turn's stamp), collects pending rows (only resolved `ask` states without a live session approval; `deny` never reaches the card), and makes **one** approval round trip — the `ChatApprovalCard` on the UI thread, bounded by `[mcp] approval_timeout_seconds` (0 = wait indefinitely, ADR-067).
2. The runtime dispatches; `MCPToolProvider.invoke` (serialized on a per-provider lock) checks in order: kill switch (refuses the whole batch without prompting) → per-run approval stamp (peeked, so same-name calls in one turn share the verdict) → fresh gate resolution → persona floor (`require_confirmation` floors allow→ask; origin `persona_policy`; never widens) → `deny` refuses with a pinned copy and audit token → `ask` tries arg rules, then the approval callback, else fails closed.
3. Card decisions apply: `approve_once` executes; `approve_session` approves in-memory for the session; `allow_matching` writes an exact-args rule (degraded to approve-once for high-risk tools); `always_allow` persists `state=allow` with the live definition hash into the active workspace profile. Unanswered keys fail closed to deny/timeout and are carried as `unresolved_keys` so audit rows never claim a human Deny.
4. Execution routes through `execute_hub_tool` (local external profile → on-demand connect; builtin → in-process), bounded by `[mcp] tool_call_timeout_seconds`; results are JSON-redacted, non-text content becomes a placeholder, and sizes are capped; exactly one audit row is recorded per call.

## Standalone server

`MCP/server.py` exposes the app's own capabilities over stdio: the tool catalog is AST-walked from the module (`_register_tools`), and — only when `[mcp] expose_local_tools` is on — the local `fs_*`/`git_*`/`web_*` tools are exposed to external clients, each call still gated by a freshly loaded permission store (so Console "Always allow" grants apply). Ask outcomes fail closed with `EXTERNAL_NO_CALLBACK_REFUSAL` — there is no human to ask. The server's workspace root is `[console] workspace_root` expanded, else the process cwd.

## Gateway runtime

`ChatbookGatewayRuntime` adapts Chatbook tool decorators to the strict external `mcp_unified.gateway` (optional dependency; tests skip when absent). It validates descriptors (name regex, description length, JSON Schema, `additionalProperties: false`), publishes local tool registrations atomically, and exposes resource templates (`conversation://`, `note://`, `character://`, `media://`, `rag-chunk://`), prompts, and continuation tokens (`tldw_continue`, 256 KiB chunks).

## Config keys (`[mcp]`)

`enabled` (default false), `server_name`, `server_version`, `transport` (stdio), `http_port`, `allowed_clients`, `expose_tools` / `expose_resources` / `expose_prompts`, `require_auth`, `rate_limit`, `max_concurrent_requests`, `approval_timeout_seconds` (0 = indefinite), `expose_local_tools` (default false), `tool_call_timeout_seconds`, `hub_lifecycle_timeout_seconds`; plus `[mcp.tools]` / `[mcp.resources]` / `[mcp.prompts]` sub-tables.

## Failure behaviors

| Case | Behavior |
| --- | --- |
| Kill switch on | Catalog empty; every call refuses without prompting; audit token recorded |
| External server crash | Reader marked unavailable; session dropped; next execute reconnects on demand; discovery snapshot flips `is_connected=false` (stale badge, "N servers not connected") |
| Permission resolver raise | `gate_error` verdict → pinned refusal copy + `denied-unresolved` audit |
| Approval timeout/cancel | Fail closed per key; `unresolved_keys` recorded |
| Tool execution timeout | Bounded refuse + execution-log row; best-effort future cancel |
| Governance denial | Recorded `status=blocked`, `error_category=governance_denied`, re-raised |

## Governing decisions

ADR-032 (`032-local-agent-tool-permission-boundary.md` — local tools under the MCP permission model), ADR-053 (`053-mcp-unified-standalone-runtime-boundary.md`), ADR-067 (indefinite human approval waits), ADR-079 (workspace assistant defaults / profiles). Specs: `Docs/superpowers/specs/2026-08-04-local-agent-tools-design.md`, `Docs/superpowers/specs/2026-09-11-mcp-hub-ux/`. User guide: `Docs/User_Guide/mcp.md`.

## Verified gotchas

1. The store's `SCHEMA_VERSION` must stay 1 — a bump would `.bak`-reset every live permission store.
2. Writing `mcp_permissions.json` from inside the workspace would flip every `ask` to `allow`; it is denylisted for both read and write in the fs tool family.
3. Per-run approval stamps are peeked, not popped — same-name calls within one turn share the verdict.
4. `always_allow` is the only persistent card decision; the builtin card deliberately excludes it, and `deny` must stay offered.
5. Quoted `"false"` in TOML is truthy — every `[mcp]` gate must coerce booleans explicitly.
