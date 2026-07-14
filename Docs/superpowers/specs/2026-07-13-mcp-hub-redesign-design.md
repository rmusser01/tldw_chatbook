# MCP Hub Redesign — Design

- **Date:** 2026-07-13
- **Status:** Approved design, pending implementation plan
- **Scope anchor:** Operator console (not full tldw_server governance parity)
- **Visual/IA references:** `Docs/Design/New_UI/MCP.png` (mockup), tldw_server2 webui MCP Hub (`apps/packages/ui/src/components/Option/MCPHub/`), `Docs/Design/master-shell-design-system-contract.md`

## 1. Context & problem

The MCP destination (`UI/Screens/mcp_screen.py` → `UI/MCP_Modules/unified_mcp_panel.py`, compact workbench) is a working but expert-hostile control-plane browser: Source/Server/Scope/Section `Select`s on the left, a text-dump detail pane, and an action runner where the user picks an action and hand-edits a raw JSON payload. The mode strip (Servers | Tools | Permissions | Audit) is a decorative label. The panel carries screen-local CSS with hardcoded hex colors, violating the design-system contract.

The service layer underneath is sound and stays: `UnifiedMCPControlPlaneService` fronts a **Local** control plane (stdio external-server profiles, JSON-file stores) and a **Server** side (`tldw_api/mcp_unified_client.py` → tldw_server `/mcp` + `/mcp/hub` routes: tool registry, readiness, external servers, credential slots, permission profiles, audit).

Separately, a functional gap: MCP tools are managed but **not usable in chat** — nothing bridges MCP server tools into `Tools/tool_executor.py`.

## 2. Goals

1. Replace the panel's view layer with a mode-tabbed workbench (Servers | Tools | Permissions | Audit) following the Console rail + canvas + inspector convention, styled entirely with `ds-*` contract classes.
2. One IA over both sources (Local / Server) with a persistent source switch; server-only features render as explicit "available with a tldw_server connection" states in Local mode.
3. Port the webui's **readiness state machine** as the single source of truth for status display and available actions.
4. Replace raw-JSON action UX with structured forms (add-server wizard, schema-driven Test Tool runner).
5. **Chat bridge**: register enabled MCP tools into the chat tool-calling path, gated by an Allow/Ask/Off permission model with inline approval UX.
6. Include chatbook's own app-as-MCP-server (`MCP/server.py`, `[mcp]` config) as a pinned built-in row.

## 3. Non-goals / out of scope

- Full webui parity: workspaces (path scopes / workspace sets / shared workspaces), governance packs, capability mappings, policy assignments. Server-side governance stays the webui's job.
- Editing tldw_server permission profiles (shown read-only; edited in the webui).
- Named local permission profiles (schema is profile-ready; UI ships a single `default` profile).
- Binding permissions to personas/characters (future; couples to the Roleplay program).
- Full resource/prompt browsing (read-only listing + test-read only; see §13).
- Local governance-rule editing UX (`governance_rule.*` actions remain reachable via the Advanced action runner, §13).

## 4. Decisions from brainstorming

| Question | Decision |
|---|---|
| Hub scope | Operator console: 4 modes, advanced governance deferred/passthrough |
| Local vs Server | One IA, persistent source switch; server-only features shown, not hidden |
| Chat bridge | In scope, gated by Hub permissions |
| Permission binding | Global `default` profile now; schema profile-ready for later |
| App-as-MCP-server | Included as pinned built-in row in Servers mode |
| Overall shape | Approach A: mode-tabbed workbench; replace view, keep service contract |
| Tool state model | Single per-tool control: **Inherit / Allow / Ask / Off** (Off = not registered with the model; store keeps `deny` internally for future profile overrides) |

## 5. Screen architecture

`MCPScreen(BaseAppScreen)` remains the destination (route `mcp`, legacy alias `tools_settings`). Its content becomes a `DestinationWorkbench`-based triad. The compact-workbench view internals of `UnifiedMCPPanel` are replaced; the `unified_mcp_service` contract and `save_state`/`restore_state` seam are kept. This supersedes the 2026-05-14 plan's "keep `UnifiedMCPPanel` intact" constraint (service contract kept, view replaced).

```
┌ MCP ─ Manage MCP servers, tools, permissions, and audit ────────────────┐
│ [ Servers ] [ Tools ] [ Permissions ] [ Audit ]      Source: Local ▾    │  ← real DestinationModeStrip
├───────────────┬───────────────────────────────┬─────────────────────────┤
│ RAIL          │ CANVAS (per mode)             │ INSPECTOR (ds-inspector)│
│ ▸ Source      │ tables / forms / matrix       │ state + why + actions   │
│ ▸ Servers     │ for the active mode           │ (readiness-driven)      │
│ ▸ Scope (srv) │                               │                         │
├───────────────┴───────────────────────────────┴─────────────────────────┤
│ shortcut bar: [1-4] mode  [a] add server  [t] test tool  [r] refresh    │
└──────────────────────────────────────────────────────────────────────────┘
```

- **Mode strip**: real navigation (click + number keys 1–4), persisted in screen state.
- **Rail** (persistent across modes, collapsible Console-rail sections):
  - **Source**: Local / Server switch.
  - **Servers**: all servers with readiness badges — built-in chatbook server pinned first (⌂), then local, then server-side, with source chips (`.source-local` / `.source-server`). Selecting a server scopes every mode's canvas; an "All servers" entry clears the scope. Badges fill progressively from cached readiness (workers refresh; a slow server never blocks rail render).
  - **Scope** (Server source only): Personal/Team/Org + entity (relocated from today's selects).
- **Inspector**: always answers *what is this object's state, why, and what can I do*. Action buttons come exclusively from the readiness reason→actions table. No free-form JSON payload TextArea (except the Advanced escape hatch, §14).
- **Styling**: `ds-*` contract classes only; screen-local hex CSS removed. `.density-compact` / `.density-comfortable` supported. All affordances keyboard-operable (divider resize gets keyboard equivalents or fixed `fr` widths); the non-obscuring-focus contract test extends to this screen.
- **State restore**: tolerant of the old panel's persisted shape — unknown/missing keys ignored, default mode Servers. Persisted: mode, source, selected server, filters.

## 6. Readiness model (`MCP/readiness.py`)

Ported from the webui (`mcpHubReadiness.ts`), used by both sources:

- **Display states**: `needs_setup | checking | ready | needs_attention | no_tools | stale`.
- **Reason codes** (priority-ordered): `not_configured → auth_missing → runtime_unavailable → preflight_failed → unreachable → discovery_failed → config_changed → discovery_not_run → no_tools_returned → catalog_expired → partial_capability`.
- **Reason → allowed-actions table**: the *only* source of which action buttons render — server rows, inspector, empty states, audit remediation.
- Local source computes state from connection/discovery state; Server source maps the server-reported readiness payload onto the same enums.
- Hub-level aggregate ("3 of 5 servers ready — 1 needs auth").
- Ships with the redaction utility (§13) as its display companion.

## 7. Servers mode

**Overview (no server selected)**: aggregate readiness line; servers table — **Name | Transport | Status | Tools | Auth | Scope**. Status is a `ds-status-badge`; problem rows get a one-line `ds-recovery-callout` beneath the table ("web-search: Needs Auth — Open credentials"). Local servers' Auth column shows `env (2)` / `none` (no fake auth schemes). Primary action: **Add Server**.

**Add-server flow** (inline canvas form): setup-type chooser, then type-specific fields:
- **Local stdio** — name, command, args, working dir, env vars (write-only once saved).
- **HTTP/SSE** — name, URL, headers.
- **Import JSON** — paste *or file path* (terminal paste of large blobs is fragile); live parse preview of what will be created + validation errors before commit.

Validation lint: warn when a secret-looking value appears in `args` (visible in `ps`); suggest env. Save lands the server in `discovery_not_run`; the inspector offers the one right next action ("Refresh tools").

**Detail (server selected)**: config summary (transport, command/URL, env redacted), tool count, read-only **resources & prompts** listing with counts (test-read/test-get via inspector), and a **Credentials** section:
- Server source: the slot model — define slots (name, kind, privilege class) → auth-template mappings (slot → header/env target, prefix/suffix) → write-only slot secret.
- Local source: masked env/secret fields.
- **Write-only secret pattern everywhere**: blank = keep existing, typed = replace, explicit Clear. Secrets are never read back.

**Lifecycle actions** (inspector, readiness-filtered): Connect / Disconnect / Test / Refresh tools / Edit / Delete. Delete confirms and cascades cleanup of that server's tool overrides; audit history is retained. All async ops are per-server workers with explicit timeouts and a visible Cancel during `checking` (stdio connects can hang).

**Built-in chatbook server** (pinned ⌂ row): honest controls only — enable + expose toggles (tools/resources/prompts) and **"Copy client config snippet"** (ready-to-paste MCP-client JSON pointing at chatbook) instead of fake connect buttons. *Correction (verified during Phase 1, §17 fact 4):* the built-in server is **stdio-only** — it runs solely when an MCP client launches chatbook via `python -m tldw_chatbook.MCP`, never in-process; the HTTP transport branch raises `NotImplementedError` and no port binding exists, so no transport/port controls are offered until HTTP genuinely ships. Toggle changes show an explicit "applies to the next client launch" note (config is read at server start). Its exposed tools appear read-only in Tools mode.

## 8. Tools mode

**Canvas**: one catalog across all servers with cached discovery results (filterable by rail selection), grouped by server; tools from non-ready servers render marked `stale` or "blocked by server" rather than vanishing. (Only the chat bridge's send-time assembly is ready-servers-only, §11.) Row: tool name, risk class (high/med/low), capability tags (`mutates`, `network`, `process`), and the single **Inherit/Allow/Ask/Off** state (effective state shown; editable here and in Permissions mode — same store). Filter bar: server, risk, state, text search. Virtualized rendering for large catalogs.

**Tool identity**: `(source, server_id, tool_name)` everywhere — catalog, permission store, execution log. Server IDs are stable internal IDs, never display names.

**Inspector (tool selected)**: description, parameter summary, server + transport, risk, last used / last error, "blocked by server" annotation when tldw_server's own policy denies a tool (a local Allow must not pretend it works). Actions: **Test Tool**, state setter, jump to Permissions.

**Test Tool**: structured parameter form generated from the tool's JSON schema (string/number/bool/enum; raw-JSON fallback editor only for unrenderable schemas). Result in the inspector: status, duration, redacted output. Test runs are user-initiated management actions: testing an Off tool is allowed with explicit confirmation (once the permission store exists, Phase 4+), and every test is recorded in the execution log.

**Empty state = diagnosis**: inspect readiness and show exactly one primary action — "No servers configured → Add server" / "web-search needs auth → Open credentials" / "Discovery not run → Refresh tools".

## 9. Permissions mode

**Model**: per-tool state **Inherit / Allow / Ask / Off**; precedence tool override → server default → global default. Absence of a key = Inherit. `Off` maps to `deny` in the store (reserved for future profile deny-overrides). Newly discovered tools follow the server default; server default defaults to **Ask** — usable immediately, nothing silently auto-runs.

**Guardrails**:
- **Rug-pull guard**: an explicit Allow stores a hash of the tool definition (description + parameter schema). A hash mismatch — checked at discovery *and* during effective-state resolution, so a changed tool never executes under a stale Allow between discoveries — downgrades the effective state to Ask, sets `config_changed` / `needs_attention`, and emits an audit entry ("web_search definition changed since you allowed it — review and re-allow").
- **High-risk inherit floor**: newly discovered tools tagged `mutates`/`process` never silently inherit a server-default Allow — they arrive as Ask with a notice.
- **Global kill switch**: master "MCP tools in chat: on/off" toggle, honored by send-time assembly, independent of per-tool states.

**Canvas**: the matrix — rows = tools grouped by server; pinned "Server default" row per group and "Global default" row on top; Space cycles Inherit → Allow → Ask → Off; overrides visually marked; capability tags shown read-only (they are metadata, not client-enforceable toggles). **Policy preview strip** at the bottom: plain-language effective-policy sentences, scoped to the rail selection. Inspector explains which rule produced the effective state.

**Server-source note**: this store is chatbook's client-side gate and applies regardless of source. tldw_server profiles are shown read-only (`list_permission_profiles`) with a pointer to the webui. Precedence statement: local gate runs first; server-side denials (403) surface as execution errors + readiness/audit events, never silently.

**Store** (`mcp_permissions.json`, schema-versioned, profile-ready; consistent with the MCP layer's JSON stores; atomic replace, last-write-wins across concurrent instances; unknown schema version → back up file and start fresh, never crash):

```json
{ "schema_version": 1,
  "kill_switch": false,
  "profiles": { "default": {
      "global_default": "ask",
      "servers": { "<source>:<server_id>": {
          "default": "ask",
          "tools": { "<tool_name>": { "state": "allow|ask|deny", "definition_hash": "…" } } } } } } }
```

## 10. Audit mode

Two canvas sub-views:

**Executions (both sources)**: chatbook's bounded execution log — every MCP tool run (chat bridge or Hub test): timestamp, tool key, initiator (chat/test), decision (allowed / approved / denied / blocked-by-server), duration, outcome, error summary. Args/results stored redacted + truncated; arg capture is a setting (default on; can be disabled entirely). Storage: `mcp_execution_log.jsonl` — append-only, size-based rotation, two generations (~1000 entries total). Filters: decision, server, initiator, text. Inspector: entry detail + drill actions "Open tool" / "Adjust permission" (mode switch + object selection).

**Findings (Server source)**: tldw_server audit findings via the existing client — severity/type filters, message, suggested remediation. **Open drills into the owning mode + object**; findings with a client-executable fix (e.g. "discovery stale → Refresh tools") get an inline action routed through the readiness action table. Local source shows the standard server-only state.

## 11. Chat bridge

**Assembly is send-time, not stateful**: when a chat message is sent with tools enabled, the pipeline queries the control plane for the effective tool list — catalog of **ready** servers ∩ effective state ≠ Off ∩ kill switch off — and instantiates adapters alongside built-in tools. No registration state to go stale. No silent auto-connect mid-send; when enabled servers aren't connected, chat shows a subtle "N MCP servers enabled but not connected" affordance linking to the Hub.

**`MCPToolAdapter(Tool)`** wraps `(source, server_id, tool_name, schema)`:
- `get_name()` → `mcp__<server>__<tool>`, sanitized to `[a-zA-Z0-9_-]`, ≤64 chars (truncate-with-hash; post-sanitization collisions get numeric suffixes). Pretty name in UI, namespaced name to the LLM.
- `get_parameters()` → passthrough of the MCP JSON schema.
- `execute()` → gate, then route: Local → stdio client session `call_tool`; Server → `MCPUnifiedClient.execute_tool`. Per-call timeout (default 60s, configurable), cancellable; result truncated/sanitized before re-entering the conversation. Non-text results (images, embedded resources) return a typed placeholder ("[image result — not yet supported]").

**Gate at call time**: Allow → run. Off → defensive refusal (shouldn't be registered; belt-and-braces). **Ask → batch approval**: one approval widget per assistant turn listing all pending calls (models emit parallel tool calls), individually approvable with "Approve all". Options: **Approve once / Approve for session / Always allow / Deny**.
- "Approve for session" = app-run lifetime, per tool, in-memory.
- "Always allow" writes through to the store *with the definition hash* (rug-pull guard applies).
- Timeout (default 2 min) → deny, with an explicit result to the model: "user did not approve within the time limit; do not retry".
- Approval prompts are transient UI; tool call + result persist as schema-v7 tool messages. App crash mid-approval → deny, logged.
- Switching conversations or sending a new message while approvals are pending resolves them as **denied**.

Every decision and execution appends to the execution log.

## 12. Service layer & configuration

**View code**: new mode widgets live under `UI/MCP_Modules/` (one module per mode, plus shared rail/inspector pieces), replacing the compact-workbench internals of `unified_mcp_panel.py`.

**New service modules**: `MCP/readiness.py` (§6), `MCP/permission_store.py` (§9), `MCP/execution_log.py` (§10), and the shared redaction utility (§13).

**`UnifiedMCPControlPlaneService` grows typed methods** shared by the new UI and the chat bridge: `catalog()`, `effective_tool_states()` (definition-hash check enforced here, §9), `set_tool_state()`, `execute_tool()`, `readiness_for(server)` — plus a change-notification hook (Textual messages) so open screens refresh on connect/disconnect/discovery/permission changes. The existing action-descriptor/`run_action` seam remains for everything not yet typed and backs the Advanced escape hatch (§14).

**Configuration knobs** (consolidated; env → config.toml → defaults per app convention): per-call tool timeout (default 60s), approval timeout (default 2 min), execution-log arg capture (default on), log rotation size. The kill switch lives in the permission store only (runtime-mutable, no config knob).

## 13. Error handling & security

- Every error renders as **readiness reason → recovery copy → one primary action** (`ds-recovery-callout`), never a raw traceback. Disabled is visually distinct from broken (preserves the 2026-05-14 QA fix).
- **One redaction utility** (ported webui patterns: secret-ish keys, URL query params, CLI arg arrays), applied at every display and log boundary: server detail, diagnostics, test results, execution log, config previews.
- **Server source offline**: render last-known data marked `stale` with an unreachable banner + retry; the Hub stays useful read-only.
- All connect/discovery/test/execute operations run in per-server workers with visible progress, timeouts, Cancel.
- Secrets: write-only everywhere; never read back, never logged.

## 14. Continuity & migration

- **Advanced escape hatch**: the legacy action runner (action select + JSON payload) survives as an "Advanced" inspector section through the transition so no existing capability disappears mid-stream (notably `governance_rule.save/preview/delete`, `runtime.access.preview`, `resource.read`, `prompt.get`). Removed from default view in Phase 6; governance-rule actions (out of operator scope) remain reachable there.
- **Resources & prompts**: read-only listing in server detail + inspector test-read/test-get. Full browsing deferred.
- **task-88 coordination** (Settings ▸ MCP Defaults): once the Hub owns operational MCP config, that Settings category hosts file-config defaults only and points at the Hub.
- **Permission migration is safe by construction**: no file → global default Ask; no bridge existed before, so nothing regresses.
- **Phase-6 cleanup**: retire `unified_mcp_sections.py` text renderers and `LAYOUT_MODE_FULL` if unused; verify the `tools_settings` alias still routes.

## 15. Testing

- **Unit**: readiness reason→action table; permission precedence (inherit/override/floor); definition-hash downgrade; name sanitization + collisions; JSONL rotation; adapter gate paths (allow / ask-approve / ask-deny / timeout) with a scripted approver; permission-store unknown-schema-version handling.
- **Property-based** (Hypothesis): redaction utility.
- **Integration**: control-plane typed methods against a scripted stdio echo MCP server fixture (Local) and a mocked `MCPUnifiedClient` (Server).
- **UI**: `app.run_test()` pilots including geometry checks (plain-Vertical clipping lesson); focus-contract test extended to the new surfaces; textual-serve screenshot QA at 2050×1240 with real CSS per the established capture recipe.

## 16. Phasing (six PRs, dependency order, each independently shippable)

1. **Readiness module + screen shell** — real mode strip, rail, Servers mode read-only (table/detail/inspector), `ds-*` migration, Advanced escape hatch. No mutating service changes (adds read-only readiness derivation).
2. **Servers mutations** — add-server wizard, credentials UX, built-in server row, lifecycle actions with workers/timeouts.
3. **Tools mode** — catalog, structured Test Tool runner (ungated; confirmation on risky tools), diagnostic empty states.
4. **Permissions mode** — store, matrix, policy preview, rug-pull hash, kill switch; Test Tool gains gate awareness.
5. **Chat bridge** — adapter, gate, batch approval UX, execution log + Audit ▸ Executions.
6. **Audit ▸ Findings (server) + polish** — drill-ins, inline remediation, density/shortcut pass, legacy cleanup.

Each screen-visible phase gets screenshot QA and explicit user approval before merge, per the established gate. Feature work happens in worktrees off `origin/dev`; backlog task IDs assigned against `origin/dev`.

## 17. Risks & planning-time verifications

1. **Chat tool-loop status**: CLAUDE.md says tool-calling "detection works, execution pending." Verify the actual state of `ToolExecutor` integration in the chat pipeline before sizing Phase 5 — the bridge may inherit finishing the execute-and-continue loop.
2. **Async boundary**: `MCPClient` is asyncio; chat workers may be thread workers. `execute()` needs correct loop bridging (`run_coroutine_threadsafe` or equivalent) — confirm against the actual pipeline.
3. **Server readiness payload shape**: confirm `MCPUnifiedClient` readiness responses map cleanly onto the ported enums; add an adapter layer if the server's reason codes drift.
4. **Built-in server restart semantics**: confirm what `[mcp]` changes can apply live vs. require restart; the UI's "restart required" state depends on it.
5. **`DestinationModeStrip` interactivity**: the shared widget is currently used as a static label; verify it supports interactive mode switching or extend it in Phase 1.
