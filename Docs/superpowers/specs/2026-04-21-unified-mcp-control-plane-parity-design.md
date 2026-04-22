# tldw_chatbook Unified MCP Control Plane Parity Design

**Date:** 2026-04-21  
**Status:** Approved for spec review  
**Primary Repo:** `tldw_chatbook`  
**Reference Repo:** `tldw_server`  
**Supersedes:** `2026-04-21-local-mcp-runtime-parity-vertical-design.md`

## Goal

Define a real `Unified MCP` control plane inside `tldw_chatbook` that gives Chatbook parity with the server's Unified MCP surfaces for control, access, and administration of MCP servers and tools.

The destination must be one MCP control plane with explicit source-scoped panes:

- `Local`
- `Server`

Within the `Server` pane, the UI must provide explicit scope switching for the authenticated scopes the user is allowed to manage:

- `Personal`
- `Team`
- `Org`
- `System/Admin`

This is not a local-runtime-first MCP shell. This is a unified control plane that lets Chatbook manage both:

- Chatbook-owned local MCP runtime and local external MCP profiles
- server-owned Unified MCP runtime, catalogs, registry, approvals, governance, and administration surfaces

## Why The Previous Spec Is Replaced

The previous spec treated `Local MCP Runtime` as the main target and deferred the server-side control plane. That is not the product goal.

The corrected goal is explicitly:

- server parity inside Chatbook for Unified MCP control/access/administration
- present Unified MCP as one destination with source-scoped panes
- include real server-side administration surfaces, not just status or discovery
- include actual scoped governance and catalog management

The new spec therefore replaces the earlier local-runtime-centered framing rather than patching it.

## Context

Chatbook already has meaningful MCP building blocks:

- `tldw_chatbook/MCP/server.py`
- `tldw_chatbook/MCP/client.py`
- `tldw_chatbook/MCP/tools.py`
- `tldw_chatbook/MCP/resources.py`
- `tldw_chatbook/MCP/prompts.py`
- `tldw_chatbook/Docs/Design/MCP.md`
- `tldw_chatbook/config.py` under `[mcp]`

Those pieces already cover:

- a Chatbook-owned local MCP server implementation
- an MCP client for external MCP servers
- local MCP tools/resources/prompts for Chatbook content

What Chatbook does **not** currently have is a unified product surface around those pieces:

- no real MCP control plane destination
- no source-aware local/server MCP administration UI
- no explicit scope switching for server governance/admin contexts
- no app-owned service boundary that unifies local MCP and server Unified MCP
- no API-client layer for the server's Unified MCP + MCP Hub administration surfaces

On the server side, the relevant contracts are not one file. They are a family of Unified MCP and MCP Hub surfaces.

### Server Unified MCP runtime/discovery surfaces

- `tldw_server/tldw_Server_API/app/api/v1/endpoints/mcp_unified_endpoint.py`

Key categories exposed there include:

- server status and health
- metrics
- tools
- tool execution
- modules and module health
- resources
- prompts
- visible tool catalogs
- curated external catalog and connection test helpers

### Server scoped catalog management surfaces

- `tldw_server/tldw_Server_API/app/api/v1/endpoints/mcp_catalogs_manage.py`

Key categories exposed there include:

- org-scoped tool catalogs
- team-scoped tool catalogs
- catalog entry create/delete operations within those scopes

### Server MCP Hub administration and governance surfaces

- `tldw_server/tldw_Server_API/app/api/v1/endpoints/mcp_hub_management.py`

Key categories exposed there include:

- tool registry and registry summary
- capability mappings
- governance packs and trust policy
- permission profiles
- path-scope objects
- workspace-set objects and memberships
- shared workspaces
- policy assignments and overrides
- approval policies and approval decisions
- effective policy and external-access views
- ACP profiles
- external server registry
- auth templates
- credential slots and secret management
- credential bindings
- governance audit findings

The server surface is intentionally broad and scope-sensitive. Chatbook parity therefore cannot be a single "MCP status" tab. It needs a structured control plane that keeps source and server scope explicit while still feeling like one destination.

## Product Decisions

The following decisions are fixed for this vertical:

- Chatbook exposes one `Unified MCP` destination, not separate local and server MCP screens.
- That destination uses a top-level source switch:
  - `Local`
  - `Server`
- This destination is an explicit exception to the app-wide single-active-source pattern.
- The MCP destination does **not** use the global runtime source as its only authority selector.
- Instead, the destination carries its own explicit `mcp_source` selection while still remaining source-separated.
- Mixed local/server lists are still deferred.
- Only one source pane is visible at a time.
- Inside the `Server` pane, Chatbook uses explicit scope switching:
  - `Personal`
  - `Team`
  - `Org`
  - `System/Admin`
- When the user has access to multiple teams or orgs, the selected scope type also requires an entity picker.
- The `Server` pane must expose real administration surfaces, not only discovery/status views.
- The `Server` pane must expose the server-side administration scopes the authenticated user is actually allowed to manage.
- The `Server` pane must not flatten scope-sensitive records into one merged list with badges only.
- The `Local` pane remains Chatbook-owned and locally authoritative.
- The `Server` pane remains server-owned and server-authoritative.
- Chatbook supports one active server target at a time inside this destination, but explicit server switching is expected behavior.
- The `Server` pane must never merge administration records across multiple configured servers.
- No local shadow copy of mutable server MCP admin records is introduced.
- Durable local storage is allowed for local pane records and for read-only UI caches where clearly non-authoritative.
- Secret values for server-managed MCP resources must not be stored as local durable shadow copies.
- The control plane should live under one destination in the existing app shell. Implementation may initially host it in the current tools/settings area, but the MCP modules must be structured as if they were their own destination.
- Runtime policy remains the authority for local and server MCP actions, but the policy layer must support an MCP-destination-specific source selection rather than assuming the global runtime source always applies.
- Runtime policy for this destination must use fixed action codes from one authoritative registry and block invalid actions centrally from day one.

## User Decisions Captured

- The goal is server parity inside Chatbook for Unified MCP control/access/administration of MCP servers and tools.
- Unified MCP should be presented as one control plane with source-scoped local and server panes under the same destination.
- The `Server` pane must include actual server-side MCP administration surfaces such as external server registry, catalogs, and approvals/governance controls.
- Chatbook should expose all Unified MCP administration scopes the authenticated user is allowed to manage on the server, including org/team catalog and governance surfaces.
- The `Server` pane should use explicit scope switching rather than one merged view with badges.

## In Scope

### Destination and shell

- Add one `Unified MCP` destination in Chatbook.
- Add a top-level MCP source switch between `Local` and `Server`.
- Add explicit server scope switching within the `Server` pane.
- Add source-specific and scope-specific persisted screen state for the destination.

### Local pane

- Local MCP overview/status
- Local runtime readiness and launch metadata
- Local registered tools/resources/prompts inventory
- Local external MCP profile registry
- Local connect/disconnect/test for external MCP profiles
- Local approvals/access rules backing local MCP server/profile/tool control

### Server pane: runtime/discovery

- Server MCP status
- Server health
- Server metrics when permitted
- Server modules and module health
- Server tools/resources/prompts
- Server tool execution/test actions where permitted
- Visible server tool catalogs
- Curated external catalog browse and connection-test helpers

### Server pane: scoped catalog administration

- Personal/team/org/system-aware catalog browsing where the server exposes it
- Org/team tool catalog CRUD and catalog-entry CRUD through the scoped management endpoints
- Server visible-catalog browse aligned to the authenticated scope

### Server pane: external servers and credentials

- External server registry CRUD
- External server import from catalog
- External auth template view/update
- Credential slot CRUD
- Secret set/clear actions through server-owned secret endpoints
- Credential binding views and mutation paths where permitted

### Server pane: governance and approvals

- Permission profiles CRUD
- Policy assignments CRUD
- Approval policies CRUD
- Approval decisions create/list as exposed by the server
- Effective policy views
- Effective external-access views
- Policy override views/mutations where exposed by the hub

### Server pane: broader MCP Hub administration

- Tool registry browse/summary
- Capability mappings preview/CRUD
- Governance packs browse/import/upgrade/trust policy
- ACP profiles CRUD
- Path-scope objects CRUD
- Workspace-set objects and memberships
- Shared workspaces CRUD
- Governance audit findings browse

### Foundation work

- Add a dedicated Unified MCP service/controller boundary in Chatbook
- Add server API-client coverage for the required MCP endpoints
- Add runtime-policy support for MCP destination source selection and source-specific action enforcement
- Add regression coverage for source switching, scope switching, permissions, and secret handling

## Out Of Scope

- Collapsing local and server records into one merged list
- Local durable shadow copies of server-owned mutable MCP admin records
- Silent local fallback for denied or unavailable server MCP actions
- Hiding scope-specific server administration differences behind generic badges only
- Replacing the server's authority for server-owned MCP objects
- Full redesign of the whole app navigation shell unrelated to MCP

## Approaches Considered

### Option A: Local-first MCP shell with server administration deferred

Use one local-runtime control surface and add a light server status panel later.

Why not chosen:

- it is the wrong product target
- it underdelivers the requested server parity
- it would repeat the mistake from the superseded spec

### Option B: One unified MCP control plane with source switch and explicit server scope switch

Use one destination with:

- top-level source switch: `Local` | `Server`
- nested server scope switch: `Personal` | `Team` | `Org` | `System/Admin`
- one shared control-plane vocabulary, but source-specific and scope-specific panels behind it

Why chosen:

- it matches the requested product shape
- it preserves explicit source authority
- it supports full server admin parity without pretending local and server records are the same objects
- it scales to multiple teams/orgs/admin contexts without merged-list confusion

### Option C: Separate Local MCP and Server MCP destinations

Create one local destination and one server admin destination.

Why not chosen:

- it weakens the "one control plane" requirement
- it duplicates navigation and mental models
- it makes cross-source parity harder to understand and test

## Chosen Model

Chatbook will expose one `Unified MCP` control plane destination with explicit source-scoped panes.

### Top-level structure

1. `Local`
2. `Server`

### Nested `Server` structure

1. `Personal`
2. `Team`
3. `Org`
4. `System/Admin`

Within each source/scope context, the destination exposes MCP sections using a consistent vocabulary:

- `Overview`
- `Inventory`
- `Catalogs`
- `External Servers`
- `Governance`
- `Advanced`

Not every section appears in every source/scope context. The point is a consistent control-plane frame, not a fake one-size-fits-all object model.

## Architecture

### 1. Source Model

This destination is a deliberate exception to the current "one active source for the whole app" behavior.

Rules:

- the rest of the app may continue to use global runtime source selection
- the Unified MCP destination keeps its own explicit `mcp_source` state
- `mcp_source` can be `local` or `server`
- only one MCP source pane is visible at a time
- actions in the destination are evaluated against the selected MCP source, not implicitly against the app-global source

This is not a mixed view because the destination still shows one source at a time. It is a source-switched control plane.

### 2. Server Scope Model

When the `Server` pane is active, Chatbook must also track:

- `active_server_id`: selected configured server target
- `server_scope_kind`: `personal`, `team`, `org`, or `system_admin`
- `server_scope_ref`: selected team/org identifier when applicable

Rules:

- only one configured server target is active at a time
- switching servers is explicit user intent, not an implicit side effect of navigation
- server-specific screen state should be partitioned by `active_server_id`
- `Personal` shows server MCP surfaces scoped directly to the authenticated user context
- `Team` requires an explicit team selector if the user can manage more than one team
- `Org` requires an explicit org selector if the user can manage more than one org
- `System/Admin` is shown only when the user's claims/permissions allow it
- the UI never merges `Team`, `Org`, and `System/Admin` results into one list
- the UI never merges records from different configured servers into one list
- mutation actions must carry the currently selected scope context explicitly

### 2a. Configured Server Target Model

Explicit server switching requires a real configured-target model, not just a transient `active_server_id`.

Required record:

- `ConfiguredServerTarget`

Minimum fields:

- `server_id`
- `label`
- `base_url`
- `auth_mode`
- `auth_reference`
- `is_default`
- `last_known_server_label`
- `last_known_reachability`
- `last_known_auth_state`
- `last_connected_at`
- `updated_at`

Rules:

- configured server targets live in dedicated Chatbook-owned storage
- `auth_reference` points to existing config-backed or secure token material; it must not require duplicating raw secrets into a second ad hoc store
- only one configured target is active in the Unified MCP destination at a time
- switching targets changes the server pane authority context but does not mutate the app-wide runtime source
- the first implementation may bootstrap the initial `ConfiguredServerTarget` from the existing single `tldw_api` config block
- until a richer multi-target settings surface exists, that bootstrap path is the compatibility bridge for current installs
- once the dedicated target registry exists, Unified MCP server selection reads from that registry rather than deriving identity directly from one global `tldw_api.base_url`

### 2b. Server Scope Discovery And Reset Contract

Server scope switching cannot be a UI-only toggle. It must be backed by an authenticated access-context model.

Recommended contract:

- when a server becomes active, Chatbook resolves a fresh `ServerAccessContext`
- `ServerAccessContext` contains:
  - `server_id`
  - `can_use_personal_scope`
  - `manageable_team_ids`
  - `manageable_org_ids`
  - `can_use_system_admin_scope`
  - permission/version metadata needed to gate sections
- the access context may come from claims, dedicated server membership endpoints, or a composed bootstrap call, but the resulting model must be normalized inside Chatbook
- server scope switch options are derived from the active `ServerAccessContext`, not hard-coded
- when switching `active_server_id`, Chatbook clears loaded records from the previous server immediately, restores the last still-valid state for the new server if one exists, and otherwise falls back to the most specific allowed scope in this order:
  - `Personal`
  - first allowed `Team`
  - first allowed `Org`
  - `System/Admin`
- if a previously saved team/org selection is no longer valid on the new or current server, that selection is dropped before any read or mutation requests are sent
- if permissions or memberships change during a session refresh, the destination revalidates the current server scope and evicts any now-invalid selection before continuing
- if no server scopes are usable after refresh, the `Server` pane remains visible but becomes an explicit unavailable state rather than rendering stale privileged data

### 3. Service Architecture

This vertical needs a top-level orchestrator plus two source-specific service families.

Recommended structure:

- `UnifiedMCPControlPlaneService`
- `UnifiedMCPContextStore`
- `ConfiguredServerTargetStore`
- `LocalMCPControlService`
- `ServerUnifiedMCPService`
- MCP destination controllers/panels under a dedicated UI module family

Responsibilities:

#### `UnifiedMCPControlPlaneService`

- holds current destination source/scope state
- holds current destination-local MCP context, distinct from app-global runtime source state
- dispatches operations to local or server services
- normalizes section models for the UI
- applies source/scope labels
- owns screen-state persistence for MCP control plane context

#### `UnifiedMCPContextStore`

- persists Unified MCP destination-local selection state
- restores MCP context without relying on the app-wide `runtime_policy_snapshot`
- partitions persisted state by selected server target where applicable

#### `ConfiguredServerTargetStore`

- owns configured server target records
- resolves available targets for the explicit server selector
- handles compatibility bootstrap from legacy single-target config

#### `LocalMCPControlService`

- wraps existing `tldw_chatbook/MCP/*`
- manages local runtime readiness and metadata
- manages local external MCP profile registry
- manages local connection lifecycle
- manages local catalog/inventory models
- manages local approvals/access rules

#### `ServerUnifiedMCPService`

- wraps new server API-client calls for Unified MCP and MCP Hub endpoints
- handles server scope switching and endpoint routing
- translates server responses into destination models
- never stores server-owned mutable records as authoritative local state
- keeps only ephemeral or clearly labeled cache state where necessary

### 4. Local Pane Model

The local pane should expose the same control-plane vocabulary where it is honest to do so.

#### Local `Overview`

- MCP dependency availability
- Chatbook local MCP runtime identity
- launch/install metadata
- local exposure toggles from config
- connected local external profile counts
- last local errors

#### Local `Inventory`

- locally registered tools
- locally registered resources
- locally registered prompts

The local inventory should be derived from local registration metadata, not from a fake self-connection loopback.

#### Local `External Servers`

- local external MCP profile registry
- create/update/delete profiles
- connect/disconnect/test profiles
- last-known discovery snapshot per profile

#### Local `Governance`

- local approval/access rules
- local profile-level and local tool-level access rules
- local source labels and effective decision views

The local governance model does not need to mirror every server governance object. It needs to be strong enough to make local MCP administration credible and explicit.

### 5. Server Pane Surface Map

The server pane should map the actual server surfaces into coherent UI sections.

#### 5a. `Overview`

Backed by:

- `/api/v1/mcp/status`
- `/api/v1/mcp/health`
- `/api/v1/mcp/metrics` when permitted

Surface:

- server status
- health
- metrics summary
- connection counts
- module counts
- permission-aware metrics access

#### 5b. `Inventory`

Backed by:

- `/api/v1/mcp/modules`
- `/api/v1/mcp/modules/health`
- `/api/v1/mcp/tools`
- `/api/v1/mcp/resources`
- `/api/v1/mcp/prompts`
- `/api/v1/mcp/tools/execute`
- `/api/v1/mcp/tool_catalogs`

Surface:

- modules and health
- tool/resource/prompt inventory
- tool execution or test actions where permitted
- visible tool catalogs

#### 5c. `Catalogs`

Backed by:

- `/api/v1/mcp/catalog`
- `/api/v1/mcp/catalog/test-connection`
- scoped org/team catalog endpoints from `mcp_catalogs_manage.py`

Surface:

- visible catalogs
- curated external catalog browse
- connection-test helper
- org/team catalog CRUD
- org/team catalog-entry CRUD

When the selected server scope has no applicable catalog mutation rights, the section should degrade to browse-only with explicit explanation.

#### 5d. `External Servers`

Backed by the MCP Hub external-server family under `/api/v1/mcp/hub/...`

Surface:

- external server registry CRUD
- import from catalog
- auth template view/update
- credential slot CRUD
- set/clear secrets through server endpoints
- credential bindings and external-access views where relevant

Chatbook must never store server-side secret values durably as local shadow state.

#### 5e. `Governance`

Backed by MCP Hub governance families:

- permission profiles
- policy assignments
- approval policies
- approval decisions
- effective policy
- external access

Surface:

- permission profile CRUD
- policy assignment CRUD
- approval policy CRUD
- approval decision creation/list
- effective-policy preview
- effective external-access views

#### 5f. `Advanced`

Backed by:

- tool registry
- capability mappings
- governance packs
- governance pack trust policy and upgrade history
- ACP profiles
- path-scope objects
- workspace-set objects and members
- shared workspaces
- governance audit findings

Surface:

- tool registry summary and browse
- capability-mapping preview and CRUD
- governance-pack browse/import/upgrade
- trust-policy edit when permitted
- ACP profile CRUD
- path/workspace/shared-workspace administration
- audit findings browse

This section is still in scope for the vertical. It is just lower-frequency and should not crowd the core inventory/catalog/governance workflows.

### 6. Policy Architecture

The runtime-policy system must support this destination explicitly.

The current registry separation between:

- `local_mcp_runtime`
- `remote_mcp_control_plane_governance`

is directionally correct, but not yet sufficient for this destination.

The destination needs action-level MCP entries that distinguish at least:

- local runtime overview/inventory/profile/governance actions
- server overview/discovery actions
- server catalogs actions
- server external-server administration actions
- server governance actions
- server advanced governance actions

Those entries should not remain a coarse pair of broad MCP buckets forever. The target model is a single authoritative registry of fixed action codes for this destination, with policy checks happening at action level rather than at vague feature-family level.

Minimum action-code matrix for this vertical, normalized to the current Chatbook runtime-policy convention of `<resource>.<action>.<source>`:

- local overview:
  - `mcp.runtime.observe.local`
- local inventory:
  - `mcp.inventory.list.local`
  - `mcp.inventory.observe.local`
- local external profiles:
  - `mcp.external_profiles.list.local`
  - `mcp.external_profiles.configure.local`
  - `mcp.external_profiles.launch.local`
  - `mcp.external_profiles.trigger.local`
  - `mcp.external_profiles.observe.local`
- local governance:
  - `mcp.governance.list.local`
  - `mcp.governance.configure.local`
  - `mcp.governance.approve.local`
  - `mcp.governance.observe.local`
- server overview:
  - `mcp.runtime.observe.server`
- server inventory:
  - `mcp.inventory.list.server`
  - `mcp.inventory.observe.server`
  - `mcp.tools.trigger.server`
- server catalogs:
  - `mcp.catalogs.list.server`
  - `mcp.catalogs.configure.server`
  - `mcp.catalogs.trigger.server`
  - `mcp.catalogs.observe.server`
- server external servers:
  - `mcp.external_servers.list.server`
  - `mcp.external_servers.configure.server`
  - `mcp.external_servers.trigger.server`
  - `mcp.external_servers.observe.server`
- server credentials and bindings:
  - `mcp.credentials.list.server`
  - `mcp.credentials.configure.server`
  - `mcp.credentials.observe.server`
- server governance:
  - `mcp.governance.list.server`
  - `mcp.governance.configure.server`
  - `mcp.governance.approve.server`
  - `mcp.governance.observe.server`
- server effective policy/external access:
  - `mcp.effective_access.observe.server`
- server advanced administration:
  - `mcp.advanced.list.server`
  - `mcp.advanced.configure.server`
  - `mcp.advanced.trigger.server`
  - `mcp.advanced.observe.server`

The exact registry schema may still use resource/action pairs internally, but these canonical ids must remain in the existing registry shape rather than introducing a second MCP-specific naming convention. UI visibility, enabled-state decisions, service preflight checks, and hard-stop enforcement must all derive from this same authoritative registry.

Backward-compatible aliasing from the current coarse MCP buckets is acceptable during rollout, but all new Unified MCP code should target the canonical ids above.

The policy layer must also support an explicit MCP-destination source context rather than using only the app-global runtime source.

Recommended design rule:

- actions inside this destination are evaluated with an explicit `selected_source`
- `selected_source` can be `local` or `server`
- when `selected_source=server`, evaluation also considers server configured/authenticated/reachable state
- when `selected_source=server`, evaluation also includes the selected `active_server_id`
- server scope selection does not grant authority by itself; it only shapes which permitted records and mutations are available
- invalid or unregistered MCP actions are rejected centrally rather than silently tolerated by individual screens
- UI affordances should be driven from the same action registry used by enforcement so that visibility and execution rules do not drift apart
- MCP-destination source switching must not call the app-wide authoritative runtime source setter; it operates on destination-local context only

### 7. Secret Handling

#### Local pane

- persisted local external MCP profiles must not store raw secret values in plaintext
- local persisted env maps may store non-secret literals and placeholder references such as `${MY_TOKEN}`
- missing placeholder variables fail explicitly at connect/test time

#### Server pane

- server secrets are server-owned
- Chatbook may send secret values to server endpoints when the user performs set/update actions
- Chatbook must not retain server secret values durably after the mutation
- secret fields shown in the UI should be write-only / set-clear oriented wherever possible

### 8. UI Structure

The destination should be implemented as dedicated MCP modules even if it initially lives inside the existing tools/settings shell.

Recommended layout:

#### Level 1: Source switch

- `Local`
- `Server`

#### Level 2 when `Server`

- explicit configured-server selector when more than one server target exists
- `Personal`
- `Team`
- `Org`
- `System/Admin`

#### Level 3: Section switch

- `Overview`
- `Inventory`
- `Catalogs`
- `External Servers`
- `Governance`
- `Advanced`

This structure is intentionally explicit. It is meant to keep source and scope comprehensible, not collapsed.

### 9. State Persistence

Persist destination UI state for:

- selected MCP source
- selected active server id
- selected server scope kind
- selected team/org id when applicable
- selected section
- last selected record ids per section where useful

This state must be persisted as a Unified MCP destination-local context record rather than piggybacking on the app-wide `runtime_policy_snapshot`.

Required record:

- `UnifiedMCPContext`

Minimum fields:

- `selected_source`
- `selected_active_server_id`
- `selected_server_scope_kind`
- `selected_server_scope_ref`
- `selected_section`
- `per_server_state`

Rules:

- `UnifiedMCPContext` is restored independently of the app-global runtime source snapshot
- MCP context restore must not be invalidated solely because the app-global source changed elsewhere
- `per_server_state` may remember last-valid section and scope selections by `server_id`, but only for records that still pass scope validation
- destination-local source switching must update `UnifiedMCPContext` without mutating the app-global runtime source state

Do not persist:

- live server connection assumptions
- server secret values
- server-owned mutable records as local truth

### 10. Error Handling

Required behavior:

- missing local MCP dependencies produce explicit local-pane status, not broken widgets
- missing server configuration disables the server pane with explicit guidance
- missing server auth or invalid session blocks server mutations explicitly
- unauthorized scopes are hidden or disabled with clear explanation
- section-level failures do not destroy the whole destination
- stale read-only caches, if shown, are clearly labeled as non-authoritative
- server secret mutations are treated as write-only operations

### 11. Testing

This destination needs heavy regression coverage because it introduces a sanctioned dual-source screen and a large permission-sensitive server administration surface.

Test areas:

- MCP destination source switching independent of global runtime source
- active server switching and server-specific state partitioning
- server scope switching and entity selection
- local pane service behavior
- server pane client routing by scope
- unauthorized scope/action blocking
- local secret-placeholder validation
- server secret write-only behavior
- catalog CRUD routing for org/team scopes
- external server registry CRUD routing
- approval/governance CRUD routing
- advanced governance browse/mutation routing where applicable
- destination state persistence and restoration

The service/client layers should be heavily unit-tested so the TUI layer can stay thin.

## File Structure Direction

The exact filenames can vary, but the design should push toward these boundaries:

- `tldw_chatbook/MCP/unified_control_plane_service.py`
- `tldw_chatbook/MCP/unified_context_store.py`
- `tldw_chatbook/MCP/server_target_store.py`
- `tldw_chatbook/MCP/local_control_service.py`
- `tldw_chatbook/MCP/server_unified_service.py`
- `tldw_chatbook/MCP/local_store.py`
- `tldw_chatbook/tldw_api/mcp_unified_client.py`
- `tldw_chatbook/UI/MCP_Modules/`
- `tldw_chatbook/UI/Screens/unified_mcp_screen.py` or equivalent dedicated MCP destination module

Existing files that should be extended rather than bypassed:

- `tldw_chatbook/app.py`
- `tldw_chatbook/MCP/server.py`
- `tldw_chatbook/MCP/client.py`
- `tldw_chatbook/config.py`
- `tldw_chatbook/runtime_policy/registry.py`
- `tldw_chatbook/runtime_policy/enforcement.py`
- `tldw_chatbook/runtime_policy/bootstrap.py`
- `tldw_chatbook/runtime_policy/source_state.py`

## Execution Slices

This is one vertical, but it is too large to land safely as one undifferentiated patch. The implementation plan should stage it.

### Slice 1: Unified shell and source/scope foundation

- new Unified MCP destination
- source switch
- configured server target registry/bootstrap
- destination-local MCP context persistence
- server scope switch
- service scaffolding
- server API client scaffolding
- runtime-policy support for destination-scoped source evaluation using canonical MCP action ids
- local/server overview and inventory browse

### Slice 2: Catalogs and external servers

- local external profile registry
- server catalogs surfaces
- server external-server registry
- auth templates
- credential slots
- connection test helpers

### Slice 3: Governance core

- local approvals/access rules
- server permission profiles
- policy assignments
- approval policies
- approval decisions
- effective policy / external access

### Slice 4: Advanced MCP Hub administration

- tool registry
- capability mappings
- governance packs
- ACP profiles
- path/workspace/shared-workspace objects
- audit findings

## Acceptance Criteria

This vertical is successful when all of the following are true:

- Chatbook exposes one Unified MCP destination
- that destination has explicit `Local` and `Server` panes
- when multiple server targets are configured, the destination supports explicit active-server switching without mixed cross-server views
- configured server targets are backed by a dedicated Chatbook-owned target registry rather than being derived only from one global `tldw_api.base_url`
- the `Server` pane has explicit `Personal`, `Team`, `Org`, and `System/Admin` scope switching where permitted
- the `Local` pane controls Chatbook-owned local MCP runtime and local external MCP profiles
- the `Server` pane exposes real server-side Unified MCP administration surfaces rather than only status/discovery
- server catalogs, external servers, approvals, and governance controls are manageable from Chatbook according to the user's actual permissions
- source and scope remain explicit at all times
- MCP control-plane visibility and mutation blocking are driven by a single authoritative action registry with fixed action codes
- Unified MCP context persistence remains independent of the app-global runtime source snapshot
- no merged local/server MCP record list is presented
- server-owned mutable records remain server-authoritative

## Recommendation

Proceed with `Option B`: one Unified MCP control plane destination with:

- explicit top-level source switching
- explicit server scope switching
- real server-side Unified MCP administration coverage
- dedicated local and server service layers behind one orchestrating control-plane shell

This is the only design here that actually matches the stated goal of server parity inside Chatbook for Unified MCP control/access/administration while preserving clear source authority.
