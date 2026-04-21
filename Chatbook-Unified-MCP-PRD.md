# Chatbook Unified MCP Integration — Product Requirements Document (PRD)

## Summary
Integrate a production‑ready Model Context Protocol (MCP) server and client into Chatbook using the Unified MCP implementation from the sibling tldw_server project. Provide a local, user‑friendly MCP experience with secure defaults, configuration via Chatbook UI, and compatibility with Claude Desktop through a stdio bridge.

- Unified MCP source: `../tldw_server2/tldw_server_API/app/core/MCP_Unified`
- Chatbook current MCP scaffold: `tldw_chatbook/MCP/*`, config toggles at `tldw_chatbook/config.py:2429`
- Outputs:
  - Local WS/HTTP MCP server managed by Chatbook (Unified MCP)
  - Stdio bridge entrypoint for Claude Desktop
  - Chatbook UI to discover tools/resources/prompts and execute tools
  - YAML‑driven module autoload with pointers to Chatbook DBs


## Background & Repo Context
Chatbook already ships basic MCP components intended for stdio clients:
- Built‑in server (FastMCP): `tldw_chatbook/MCP/server.py`
- Client wrapper: `tldw_chatbook/MCP/client.py`
- Config block for MCP: `tldw_chatbook/config.py:2429`

Unified MCP (production‑grade server with WS/HTTP, JWT/RBAC, rate limiting, and module system) lives in the sibling repo:
- Server + lifecycle: `tldw_server_API/app/core/MCP_Unified/server.py`, `__init__.py`
- Protocol/types: `tldw_server_API/app/core/MCP_Unified/protocol.py`
- Endpoints: `tldw_server_API/app/api/v1/endpoints/mcp_unified_endpoint.py`
- Modules system: `tldw_server_API/app/core/MCP_Unified/modules/*`

This PRD unifies Chatbook’s MCP integration around Unified MCP while preserving stdio compatibility via a bridge.


## Goals
- Start/stop a local Unified MCP server from within Chatbook with safe defaults.
- Provide a stdio bridge so Claude Desktop can use the local Unified MCP server.
- Enable discovery and execution of MCP tools/resources/prompts from Chatbook’s UI.
- Allow per‑user customization: module selection, YAML configs, DB paths, and security knobs.
- Keep Chatbook’s current FastMCP server as an optional fallback (phase 1).


## Non‑Goals
- Full migration of all Chatbook’s in‑repo MCP tools to Unified MCP in phase 1.
- Building a multi‑tenant RBAC admin UI in Chatbook (defer to tldw_server).
- Replacing tldw_server deployment patterns; Chatbook focuses on local/desktop use.


## Users & Use Cases
- Individual users: run a local MCP server to power Claude Desktop and use MCP tools from Chatbook.
- Power users/devs: configure modules (e.g., media, RAG) and point them at Chatbook databases.
- Operators: view health/metrics locally and adjust runtime knobs (rate limits, concurrency) for heavy workflows.


## Assumptions & Dependencies
- Unified MCP code available at `../tldw_server2/tldw_server_API/app/core/MCP_Unified`.
- Chatbook continues as a Textual app; optional browser serving via `tldw_chatbook/Web_Server/serve.py`.
- Clients:
  - Claude Desktop (stdio) supported via bridge.
  - Other clients (WS/HTTP) can connect directly.
- Local, single‑user mode acceptable by default; JWT/RBAC configurable for advanced users.
- Module maturity: core modules required for Chatbook workflows should be treated as stable for integration, while experimental Unified MCP modules remain opt-in in the generated `mcp_modules.yaml`.


## Architecture
High‑level topology:
- Chatbook UI ↔ Chatbook MCP Client (WS) ↔ Unified MCP Server (WS/HTTP)
- Claude Desktop ↔ Stdio Bridge ↔ Unified MCP Server (HTTP)

Key components:
- Unified MCP Server (imported): `tldw_server_API/app/core/MCP_Unified/server.py`, endpoints under `/api/v1/mcp/*`.
- MCP Stdio Bridge (new): translates stdio JSON‑RPC ↔ Unified MCP HTTP `POST /api/v1/mcp/request`.
- MCP Orchestrator (new): starts/stops Unified MCP from Chatbook (subprocess or in‑process FastAPI lifespan).
- Chatbook MCP Client: reuse `tldw_chatbook/MCP/client.py` to connect via WS and surface capabilities in UI.

Deployment notes:
- Default bind to loopback only, with API key auth enabled.
- Configured via Chatbook settings (env‑first overrides passed to Unified MCP at start).


## Functional Requirements
1) Server lifecycle management
- Start/stop Unified MCP server from Chatbook.
- Choose run mode: in‑process (import + lifespan) or uvicorn subprocess (recommended for isolation).
- Auto‑generate single‑user API key on first run; store locally (never hardcode in code).
- Persist user preferences in `tldw_chatbook/config.py` under `[mcp]` and write a modules YAML.

2) Stdio bridge for Claude Desktop
- New entrypoint (e.g., `python -m tldw_chatbook.MCP --stdio-bridge`) that:
  - Reads JSON‑RPC from stdin, forwards to Unified MCP `POST /api/v1/mcp/request`.
  - Supports `initialize`, `tools/list`, `tools/call`, `resources/list|read`, `prompts/list|get`.
  - Passes Authorization header or `X-API-KEY` from local config.
  - Preserves IDs, propagates Unified MCP errors (rate limit, permission, validation, internal).

3) Client usage in Chatbook
- Use `tldw_chatbook/MCP/client.py` to connect to WS endpoint and:
  - List tools/resources/prompts.
  - Execute tools with args; display results; optionally save outputs to notes/DB.
  - Show server status, uptime, and connection counts.

4) Configuration & customization
- Extend `tldw_chatbook/config.py` `[mcp]` with:
  - `enabled` (bool), `provider` (`builtin|unified`), `transport` (`stdio|http` for bridge hints), `unified.port`, `unified.api_key`, `unified.ws_allowed_origins`, `unified.single_user_mode`, `unified.modules_yaml`, `unified.env_overrides`.
- Generate a default `mcp_modules.yaml` pointing modules at Chatbook DBs.

5) Security
- Defaults: loopback bind, API key required, rate limiting enabled per Unified MCP defaults.
- Optional JWT/RBAC integration (advanced): surface toggle and pointers; keep off by default.
- No wildcard CORS by default; WS allowed origins restricted to localhost.

6) Observability
- Surface `/api/v1/mcp/status` in UI. Optionally enable `/metrics` (admin-only toggle).
- Minimal log viewer to tail MCP server logs within Chatbook.

7) Compatibility
- Claude Desktop integration documented via the stdio bridge command.
- Keep `tldw_chatbook/MCP/server.py` as fallback during beta; prefer Unified MCP for WS/HTTP.
- Use the fallback when Unified MCP is disabled explicitly, fails to start within the configured timeout, or does not pass the initial health check.


## Non‑Functional Requirements
- Performance: local tool discovery < 500 ms; stdio bridge overhead negligible; rate limit windows configurable.
- Reliability: graceful shutdown, client auto‑reconnect, circuit breaker settings honored per module.
- Security: secure defaults, no secrets in code, environment‑first configuration, loopback binding by default.


## Data & Module Configuration
- Provide `tldw_Server_API/Config_Files/mcp_modules.yaml` (created/updated by Chatbook) to autoload modules.
- Example (single‑user local):
```yaml
modules:
  - id: media
    class: tldw_Server_API.app.core.MCP_Unified.modules.implementations.media_module:MediaModule
    enabled: true
    name: Media
    settings:
      db_path: tldw_chatbook/DB/Media_DB_v2.db
    max_concurrent: 16
    circuit_breaker_threshold: 3
    circuit_breaker_timeout: 30
```
- Optionally ship a `mcp_tool_categories.yaml` with ingestion/read categorization for rate limits.


## API & Protocol
- JSON‑RPC methods proxied via Unified MCP:
  - `initialize`, `tools/list`, `tools/call`, `resources/list`, `resources/read`, `prompts/list`, `prompts/get`
- HTTP endpoints (selected):
  - `POST /api/v1/mcp/request`, `GET /api/v1/mcp/tools`, `POST /api/v1/mcp/tools/execute`
  - `GET /api/v1/mcp/status`, `GET /api/v1/mcp/health`, `GET /api/v1/mcp/metrics` (admin gated)
  - `WS /api/v1/mcp/ws`
- Error mapping in bridge/UI:
  - Rate limit → 429 / JSON‑RPC code -32002
  - Permission → 403 / JSON‑RPC code -32001
  - Validation → 400 / JSON‑RPC code -32602
  - Internal → 500 / JSON‑RPC code -32603


## UX Requirements
- New “Integrations > MCP” panel in Chatbook UI:
  - Status: running/stopped, port, uptime, WS connections.
  - Controls: Enable/Disable, Start/Stop, Provider switch (builtin/unified), Regenerate API key, Open logs.
  - Discovery: list tools/resources/prompts (filter by module), view schemas.
  - Actions: call tool with arguments; show results; save outputs to notes.
  - Config: pick modules YAML, edit DB paths, WS allowed origins, rate/session knobs (safe subset), single‑user toggle.
  - Onboarding wizard: enable MCP, generate API key, print stdio bridge command for Claude Desktop.

- Stdio bridge usage example (Claude Desktop custom server):
```
python -m tldw_chatbook.MCP --stdio-bridge \
  --api-key "<your-local-api-key>" \
  --endpoint "http://127.0.0.1:<port>/api/v1/mcp/request"
```


## Implementation Plan & Milestones
- M0: Wiring & toggles
  - Extend `[mcp]` config in `tldw_chatbook/config.py` (provider, unified.* keys).
  - Add orchestrator to start/stop Unified MCP with env vars and uvicorn subprocess (logs to file + console).

- M1: Stdio bridge
  - Implement `tldw_chatbook/MCP/bridge_stdio.py` translating stdio JSON‑RPC ↔ Unified HTTP.
  - Package CLI flags and document Claude Desktop setup.

- M2: Client UI integration
  - Add MCP panel/screen to browse tools/resources/prompts and call tools via WS using `tldw_chatbook/MCP/client.py`.

- M3: Module configs & Chatbook defaults
  - Generate a default `mcp_modules.yaml` targeting Chatbook DBs; helper for DB path selection.
  - Optional: provide a thin Chatbook module in Unified MCP if needed beyond existing modules.

- M4: Hardening & docs
  - Add quick‑start and security guide; expose metrics toggle (admin only).
  - Finalize fallback behavior to builtin server if Unified MCP is unavailable.


## Testing Plan
- Unit tests
  - Bridge request/response translation, error propagation, id handling.
  - Config serialization (env → process), API key storage/regeneration.

- Integration tests
  - Start Unified MCP in single‑user mode; assert `tools/list`, `resources/list`, `prompts/list`, `tools/call`.
  - WS client sessions, rate window enforcement, session TTL cleanup.

- E2E manual
  - Claude Desktop → stdio bridge → Unified MCP: initialize, list tools, call tool.
  - Chatbook UI: discover tools, run tool, save result to notes.


## Rollout
- Ship behind `[mcp].enabled=true` with `provider=unified` in dev builds; keep builtin FastMCP as fallback.
- Default to loopback bind, API key required, and metrics endpoint disabled.
- Provide migration notes and stdio bridge setup steps.


## Acceptance Criteria
- Start/stop Unified MCP from Chatbook; status and logs visible in UI.
- Claude Desktop connects via stdio bridge; `initialize`, `tools/list`, and `tools/call` succeed.
- Chatbook UI lists tools/resources/prompts over WS and executes a tool with visible results.
- Default config passes Unified MCP validation and binds to loopback with API key auth.
- Health endpoint works; Prometheus metrics available when enabled (admin‑only).


## Open Questions
- Deprecation window for `tldw_chatbook/MCP/server.py` once the bridge is stable?
- Default location of `mcp_modules.yaml` (repo vs user config directory) for single‑user setups?
- Provide a convenience shell script/binary wrapper for the bridge to simplify Claude Desktop setup?


## Risks & Mitigations
- API/drift between bridge and Unified MCP → Keep bridge thin; proxy `/request` with minimal transformation.
- Cross‑repo dependency on a relative path → Allow env var (`TLDW_SERVER_ROOT`) and document; long‑term publish Unified MCP as a package.
- User security misconfiguration → Guided wizard with validation, opinionated safe defaults, warnings in UI.


## File Touchpoints (planned)
- Chatbook (this repo)
  - `tldw_chatbook/config.py` — extend `[mcp]` with unified provider keys.
  - `tldw_chatbook/MCP/bridge_stdio.py` — new stdio→HTTP translator.
  - `tldw_chatbook/MCP/runner_unified.py` — new orchestrator to start/stop Unified MCP.
  - `tldw_chatbook/Screens/MCP_Integration.py` — new UI panel for control/discovery.
  - `tldw_chatbook/MCP/__main__.py` — CLI flags for provider/stdio bridge/port.

- Unified MCP (sibling repo — no code changes required for MVP)
  - `tldw_server_API/app/core/MCP_Unified/__init__.py` — used for `get_mcp_server()`.
  - `tldw_server_API/app/api/v1/endpoints/mcp_unified_endpoint.py` — HTTP/WS endpoints consumed by bridge/client.
  - `tldw_Server_API/Config_Files/mcp_modules.yaml` — generated/updated by Chatbook.


## Success Metrics
- 90%+ of local runs succeed starting Unified MCP with default settings.
- Claude Desktop stdio sessions successfully initialize and list tools within < 2s.
- < 1% bridge translation errors over 1k tool calls in local testing.
- Users can execute at least one built‑in Unified MCP tool and persist the result via Chatbook UI.


## References
- Chatbook MCP config: `tldw_chatbook/config.py:2429`
- Chatbook MCP client: `tldw_chatbook/MCP/client.py`
- Unified MCP server: `../tldw_server2/tldw_server_API/app/core/MCP_Unified/server.py`
- Unified MCP endpoints: `../tldw_server2/tldw_server_API/app/api/v1/endpoints/mcp_unified_endpoint.py`
- Unified MCP README: `../tldw_server2/tldw_server_API/app/core/MCP_Unified/README.md`
