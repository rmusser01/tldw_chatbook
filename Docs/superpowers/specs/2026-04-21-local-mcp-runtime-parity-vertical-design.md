# tldw_chatbook Local MCP Runtime Parity Vertical Design

**Date:** 2026-04-21  
**Status:** Approved for spec review  
**Primary Repo:** `tldw_chatbook`  
**Reference Repo:** `tldw_server`

## Goal

Define the first serious `Local MCP Runtime` parity slice for `tldw_chatbook` as a standalone-owned runtime surface.

This slice is intentionally local-first and local-authoritative:

- Chatbook must be able to expose its own MCP server locally without the server present
- Chatbook must be able to manage locally configured external MCP connections without the server present
- Chatbook must own its own local MCP approvals and governance backing
- remote MCP hub/catalog governance remains a different surface and is explicitly deferred

The target outcome is not "server MCP parity inside Chatbook." The target outcome is a credible Chatbook-owned local runtime that later remote governance work can coexist with, rather than replace.

## Context

This vertical is not greenfield.

Chatbook already contains a meaningful MCP package:

- `tldw_chatbook/MCP/server.py`
- `tldw_chatbook/MCP/client.py`
- `tldw_chatbook/MCP/tools.py`
- `tldw_chatbook/MCP/resources.py`
- `tldw_chatbook/MCP/prompts.py`
- `tldw_chatbook/Docs/Design/MCP.md`
- `tldw_chatbook/config.py` under `[mcp]`

That existing package proves several important things:

- Chatbook already knows how to expose a local stdio MCP server
- Chatbook already knows how to connect to external MCP servers through the client wrapper
- Chatbook already has local tool/resource/prompt implementations for a meaningful subset of its data

What is missing is the runtime product surface around those pieces:

- no dedicated app-owned MCP runtime service layer
- no authoritative local runtime state model attached to the app
- no dedicated persistence model for MCP connection profiles, catalog snapshots, or local approvals
- no real TUI surface for MCP runtime status, connections, catalog browsing, or approval policy
- no runtime-policy adoption around MCP runtime actions beyond registry seeding

On the server side, the relevant contract is broad and multi-tenant:

- `tldw_server/tldw_Server_API/app/api/v1/endpoints/mcp_unified_endpoint.py`
- `tldw_server/tldw_Server_API/app/core/MCP_unified/server.py`
- `tldw_server/tldw_Server_API/app/core/MCP_unified/protocol.py`

The server MCP contract includes:

- HTTP and WebSocket request handling
- multi-user auth and session handling
- metrics and module health
- tool, resource, prompt, and module enumeration
- catalog access
- connection testing
- remote governance hooks

That contract is useful as a capability reference, but it is not the right direct implementation target for the first Chatbook slice. Chatbook is single-user, offline-capable, and already has an existing stdio MCP package. The first parity slice should crosswalk the server concepts that matter to a standalone local runtime:

- runtime status
- capability catalog
- connection definitions
- connection testing / launch state
- approvals and governance backing

It should not import the server's full hub/governance/control-plane model into Chatbook.

## Product Decisions

The following decisions are fixed for this vertical:

- `Local MCP Runtime` remains a Chatbook-owned local surface.
- `Remote MCP Control Plane / Governance` remains a separate later vertical.
- Chatbook must not mirror remote hub/catalog governance into the local MCP store in this slice.
- The first MCP runtime UI lives inside the existing `Tools & Settings` destination, not a new top-level screen.
- The first UI slice exposes four areas:
  - `Status`
  - `Catalog`
  - `Connections`
  - `Approvals`
- The first slice uses the existing `tldw_chatbook/MCP/*` package as the implementation base rather than replacing it.
- The first slice introduces a dedicated local runtime service boundary rather than wiring UI directly to `MCP/server.py` and `MCP/client.py`.
- The first slice introduces a dedicated local persistence layer for MCP runtime data that does not belong in ad hoc widget state.
- Static runtime defaults remain in `config.toml`; dynamic MCP runtime records move to dedicated local storage.
- The first slice is `stdio`-first for both the local Chatbook MCP runtime surface and external MCP connections.
- The local Chatbook MCP server is treated as an on-demand `stdio` launch target in the first slice, not as a persistent managed in-app daemon.
- HTTP transport configuration may remain visible as config, but first-slice lifecycle management is not built around HTTP serving.
- The first slice supports:
  - enable/disable local MCP runtime configuration
  - inspect local MCP runtime readiness, dependencies, and launch metadata
  - create/update/delete local external-connection profiles
  - connect/disconnect/test local external-connection profiles
  - browse discovered tools/resources/prompts
  - manage local profile-level approval decisions
- The first slice does not require a general manual tool-execution UI.
- The first slice does not persist raw secret values for MCP connection profiles.
- Persisted MCP env overrides may store non-secret literals and environment-variable references only.
- The first slice keeps approvals profile-level only; capability-level approval prompts and rules are deferred until a real execution surface exists.
- The first slice does not implement remote catalog import, remote governance sync, or approval mirroring.
- The first slice does not create a mixed local/remote MCP view.
- Runtime policy remains the authority for local MCP runtime actions and blocking behavior.

## User Decisions Captured

- Chatbook should have its own local notifications and MCP governance backing for local operations without the server present.
- Remote workflows are acceptable as remote surfaces and should not drive local MCP design.
- Local MCP runtime and remote MCP governance must remain separate.
- A single authoritative runtime-policy registry should govern action-level access.
- Invalid actions should be blocked centrally from day one.
- Single active server is sufficient for now, but later server switching must remain possible.
- Mixed local/remote views are deferred.

## In Scope

- Add a Chatbook-owned local MCP runtime service layer.
- Add app bootstrap wiring for the MCP runtime service.
- Add dedicated local persistence for:
  - external MCP connection profiles
  - catalog snapshots / discovered capabilities
  - profile approval rules
  - last-known runtime status
- Add a `Tools & Settings` MCP runtime section with `Status`, `Catalog`, `Connections`, and `Approvals`.
- Add lifecycle handling for:
  - local Chatbook MCP runtime readiness / launch metadata / status
  - external MCP client connect/disconnect/test
- Add source-aware runtime-policy enforcement for:
  - `mcp.runtime.list.local`
  - `mcp.runtime.configure.local`
  - `mcp.runtime.launch.local`
  - `mcp.runtime.trigger.local`
  - `mcp.runtime.observe.local`
- Add explicit local governance backing through stored profile approval decisions.
- Add normalized capability browsing for:
  - tools
  - resources
  - prompts
- Add regression coverage for:
  - runtime lifecycle state
  - connection profile CRUD
  - discovery snapshot behavior
  - approval decision persistence
  - runtime-policy blocking
  - shutdown and restart cleanup behavior
  - secret-placeholder validation

## Out Of Scope

- Remote MCP hub governance
- Remote catalog management
- Approval mirroring or sync with the server
- Mixed local/remote MCP views
- General manual tool execution UI for every discovered external capability
- Capability-level approval prompts or rules
- Persisted plaintext secret values for MCP connection profiles
- A persistent managed in-app MCP daemon for the local Chatbook server
- HTTP/WebSocket parity with the server runtime
- Rebuilding the existing `tldw_chatbook/MCP/*` package from scratch
- A new top-level MCP screen
- Full server MCP endpoint parity

## Approaches Considered

### Option A: Mirror the server unified MCP control plane inside Chatbook

Recreate the server's request/batch/status/catalog model inside Chatbook and treat the server implementation as the direct blueprint.

Why not chosen:

- the server runtime is multi-user and auth-heavy
- Chatbook is single-user and local-first
- this would blur local runtime and remote governance too early
- it would bias the client toward server semantics instead of standalone ownership

### Option B: Build a local runtime shell over the existing Chatbook MCP package

Add a local runtime service, local store, runtime-policy adoption, and a TUI surface around the existing `MCP/server.py` and `MCP/client.py` implementations.

Why chosen:

- it preserves the already-working stdio MCP foundations
- it aligns with the user's standalone-first requirement
- it keeps local governance local
- it creates a clean later boundary for separate remote governance work
- it fits the existing pattern used by other parity rows: scope service + app bootstrap + UI shell

### Option C: Defer local MCP until remote governance is designed

Treat MCP as primarily a later remote-control-plane problem and avoid shipping a local runtime slice now.

Why not chosen:

- it contradicts the explicit priority on local operations without the server
- it would leave a parity row called out as high-value still structurally unowned in Chatbook
- it would force later remote work to define local behavior implicitly rather than explicitly

## Chosen Model

This vertical introduces a dedicated Chatbook-owned `Local MCP Runtime` product slice.

The runtime has two responsibilities:

1. manage Chatbook's own local MCP server lifecycle and exposure settings
2. manage locally configured external MCP connections and the discovered local catalog around them

The runtime keeps four distinct surfaces inside `Tools & Settings`:

- `Status`: whether the local Chatbook MCP runtime is enabled, whether it is running, what transport is active, and summary connection health
- `Catalog`: discovered tools/resources/prompts from the local runtime and connected external servers
- `Connections`: CRUD plus connect/disconnect/test for local external MCP connection profiles
- `Approvals`: locally stored governance decisions for MCP connection and capability usage

The critical ownership rule is:

- local MCP runtime state is authored and owned locally inside Chatbook
- remote MCP governance stays remote-only and separate
- later remote catalog import may reference or augment local catalog views, but must not replace local authority

## Architecture

### 1. Authority Model

This vertical must inherit the runtime-policy model already seeded in the registry.

The controlling rules are:

- local MCP runtime actions are local-only actions
- MCP runtime operations must be blocked centrally through `ServicePolicyEnforcer`
- UI preflight checks may use the same decisions for early feedback
- server connectivity or server auth must not be prerequisites for local MCP runtime behavior
- remote governance actions must not be surfaced as if they are part of the local runtime

Required action ids for the first slice are:

- `mcp.runtime.list.local`
- `mcp.runtime.configure.local`
- `mcp.runtime.launch.local`
- `mcp.runtime.trigger.local`
- `mcp.runtime.observe.local`

The intended meaning is:

- `list`: browse runtime records and catalog state
- `configure`: edit runtime settings, connection profiles, and profile approval rules
- `launch`: connect/disconnect external MCP profiles and surface local launch-target metadata
- `trigger`: run bounded local runtime actions such as test-connection or refresh-discovery
- `observe`: inspect runtime health, connection status, and catalog snapshots

### 2. Persistence Model

This slice should split MCP data into two persistence layers.

#### 2a. Config-owned static runtime defaults

Existing `[mcp]` configuration in `config.toml` remains the source for static operator-facing defaults such as:

- runtime enabled flag
- server name and version
- default transport
- top-level exposure toggles
- default limits

#### 2b. Dedicated local MCP runtime store

Dynamic MCP runtime records should not be pushed into arbitrary config blobs. They need their own local store because they are runtime-authored data with list/detail/update semantics.

The dedicated local store should hold:

- external connection profiles
- discovered capability snapshots
- profile approval rules
- last-known runtime state
- last error and last successful connection metadata

Recommended storage choice:

- use a dedicated local SQLite store rather than JSON blobs

Why:

- connection profiles, approvals, and discovered capabilities are list-oriented and mutable
- later audit/history or richer filters are easier to add
- the app already uses dedicated local stores for other durable client-owned state

#### 2c. Secret handling rule

This slice must not create a new plaintext secret store by accident.

Rules:

- persisted connection profiles must not store raw secret values
- persisted `env_overrides` may contain:
  - non-secret literal values
  - placeholder references such as `${MY_TOKEN}`
- if a connection profile depends on a referenced environment variable and the variable is missing at connect/test time, the action must fail explicitly
- any future persisted secret-entry UX is deferred until it can reuse an approved encryption or secret-storage path

### 3. Service Layer

This vertical should add a dedicated local runtime service boundary rather than letting the UI talk directly to `MCP/server.py` or `MCP/client.py`.

Recommended components:

- `LocalMCPRuntimeService`
- `LocalMCPRuntimeStore`
- `LocalMCPRuntimeModels`
- optional small controller/helper layer for `Tools & Settings`

The service owns:

- runtime status derivation
- connection profile CRUD
- connection lifecycle actions
- catalog discovery persistence
- profile approval CRUD
- policy enforcement at the hard-stop seam
- translation between UI-friendly models and raw `MCPClient` / `TldwMCPServer` calls

The service must be app-bootstrapped the same way newer parity seams are:

- construct once in `app.py`
- attach to the app instance
- inject `ServicePolicyEnforcer`
- expose one authoritative local runtime interface to the UI

### 3b. Local catalog authority

The local Chatbook-owned catalog must not depend on a loopback MCP client session in the first slice.

Rules:

- the local Chatbook runtime catalog is derived directly from service-owned metadata about the registered local tools, resources, and prompts
- external profile catalogs are discovered through `MCPClient` introspection and then persisted as snapshots
- the service is the only layer that merges local manifest data with external snapshot data for UI display

This avoids the complexity of spawning a loopback subprocess or fake self-connection just to list Chatbook's own local MCP capabilities.

### 4. Runtime Data Model

The exact class names can vary, but the first slice should standardize these records.

#### `MCPRuntimeStatus`

Fields:

- `enabled`
- `dependencies_available`
- `launch_mode`
- `transport`
- `server_name`
- `server_version`
- `connected_profile_count`
- `healthy_profile_count`
- `last_status_refresh_at`
- `last_error`

#### `MCPConnectionProfile`

Fields:

- `profile_id`
- `display_name`
- `transport`
- `command`
- `args`
- `env_overrides` (non-secret literals or `${ENV_NAME}` references only)
- `auto_connect`
- `approval_mode`
- `description`
- `created_at`
- `updated_at`
- `last_connection_state`
- `last_connected_at`
- `last_error`

First-slice constraint:

- transport is effectively `stdio`
- future transport types may be modeled, but they are not first-slice execution targets

#### `MCPCapabilitySnapshot`

Fields:

- `profile_id` or `runtime_source_key`
- `capability_kind` (`tool`, `resource`, `prompt`)
- `name`
- `title`
- `description`
- `schema_json` or raw metadata
- `discovered_at`
- `connection_state_at_discovery`

#### `MCPApprovalRule`

Fields:

- `rule_id`
- `subject_kind` (`profile`) in the first slice
- `subject_key`
- `decision` (`allow`, `ask`, `deny`)
- `scope` (`local`)
- `notes`
- `updated_at`

### 5. Governance And Approvals

This slice must give Chatbook its own local governance backing rather than leaving MCP decisions implicit.

The first-slice model should be intentionally simple:

- approvals are local records
- approvals never sync to the server in this slice
- approvals are profile-level only in the first slice
- the default mode is conservative rather than silent allow

Recommended first-slice behavior:

- newly created profiles default to `ask`
- `ask` is evaluated on `connect` and `test` actions only in the first slice
- explicitly denied profiles cannot be connected or tested
- approval rules are editable from the `Approvals` tab

Because general manual tool execution is out of scope, first-slice approvals primarily justify:

- connection permission
- future-safe capability governance backing
- explicit local ownership of MCP policy

This is still worth landing now because it prevents later remote governance work from becoming the accidental first approval system users encounter.

Capability-level approvals are deferred until Chatbook has a real execution surface that would make those prompts meaningful.

### 6. Runtime Lifecycle

The first slice should manage lifecycle deliberately rather than pretending that importing MCP code means the runtime is active.

Required behavior:

- the local Chatbook MCP runtime readiness and launch metadata can be inspected from the UI
- runtime enable/disable and readiness refresh update `Status` immediately
- connection profiles can be connected/disconnected from the UI
- profile test/refresh actions do not require a locally managed Chatbook runtime daemon
- runtime state survives screen refreshes because it is owned by the service, not the widget
- live MCP client sessions must be disconnected on app shutdown
- on app restart, persisted runtime state must never assume that prior sessions are still connected
- deleting a connected profile must disconnect it first before removing its durable record

Important rule:

- `enabled`, `ready`, and live `connected` state are different states

Examples:

- a runtime can be configured as enabled but not ready because MCP dependencies are missing
- a runtime can be ready while zero external profiles are currently connected
- after restart, the runtime may still be enabled while every prior live profile connection is treated as disconnected until re-established

### 7. Catalog And Discovery Rules

`Catalog` is an observe surface, not a fake execution console.

The first slice should show discovered:

- tools
- resources
- prompts

from:

- the local Chatbook runtime manifest
- connected external MCP profiles

Rules:

- discovery snapshots are persisted locally
- disconnected profiles may still show last-known catalog entries, clearly labeled as stale
- profile and local-runtime entries must be labeled by source so users know where a capability came from
- local Chatbook-owned capabilities must remain distinguishable from external profile capabilities
- `Refresh Discovery` for a disconnected profile must fail explicitly rather than silently reconnecting
- if a disconnected profile has cached capabilities, the UI may continue to show them as stale until the user explicitly tests or reconnects that profile

The catalog should not imply:

- remote governance state
- server-owned catalogs
- automatic sync

### 8. UI Surface

The first slice lives inside `Tools & Settings`.

This is the right shell because:

- MCP runtime is currently operational/configuration-heavy
- it avoids introducing another top-level destination too early
- it fits the existing settings-oriented ownership of `[mcp]` config

Recommended layout:

#### `Status`

- runtime enabled/disabled
- dependencies available / missing
- launch mode (`on_demand_stdio`)
- transport and top-level exposure summary
- local runtime identity
- connected profiles summary
- last error / last refresh summary
- buttons:
  - `Refresh Status`
  - `Copy Launch Command`
  - `Show Install Instructions`

#### `Catalog`

- filter by `All`, `Local Runtime`, or a specific profile
- grouped sub-sections for `Tools`, `Resources`, and `Prompts`
- source labels and stale-state labels
- simple detail panel for metadata/schema preview

#### `Connections`

- list of connection profiles
- create/edit/delete profile actions
- connect/disconnect/test actions
- explicit status badges

#### `Approvals`

- list of profile-level rules
- edit decision between `allow`, `ask`, `deny`
- show what subject the rule applies to
- show when it was last changed

Implementation guidance:

- do not turn `Tools_Settings_Window.py` into a larger monolith
- the shell can remain there, but MCP-specific UI logic should move into dedicated modules/controllers

### 9. Error Handling

The first slice must fail clearly.

Required behavior:

- missing MCP optional dependencies produce explicit runtime status and blocked actions rather than broken widgets
- failed profile connections surface last error in `Connections` and `Status`
- stale catalog data is labeled as stale, not shown as current truth
- policy-denied actions surface runtime-policy messages
- unsupported transport or unsupported profile shape should fail at the service boundary, not only in the UI
- missing environment-variable placeholders required by a profile must fail explicitly at connect/test time
- restart recovery must clear any persisted assumption that an MCP session is still live

### 10. Testing

This slice needs regression coverage because it adds a new durable runtime surface rather than just a UI wrapper.

Test areas:

- runtime-policy blocking for `mcp.runtime.*.local`
- local MCP runtime store CRUD
- runtime service status derivation
- connection profile connect/disconnect/test flows with mocked MCP client
- discovery snapshot persistence and stale-label behavior
- approval rule persistence and resolution
- placeholder-only secret reference validation
- shutdown disconnect and restart cleanup behavior
- UI controller/widget behavior for:
  - refresh status / launch metadata presentation
  - connect/disconnect/test profile
  - catalog refresh
  - approval update

The service layer should be heavily unit-tested so the TUI layer remains thin.

## File Structure Direction

The exact filenames can vary, but the design should push toward these boundaries:

- `tldw_chatbook/MCP/local_runtime_service.py`
- `tldw_chatbook/MCP/local_runtime_store.py`
- `tldw_chatbook/MCP/local_runtime_models.py`
- `tldw_chatbook/UI/Tools_Modules/mcp_runtime_controller.py`
- `tldw_chatbook/UI/Tools_Modules/mcp_runtime_panel.py`
- targeted tests under `Tests/MCP/` and `Tests/UI/`

Existing files that should be extended rather than bypassed:

- `tldw_chatbook/app.py`
- `tldw_chatbook/UI/Tools_Settings_Window.py`
- `tldw_chatbook/MCP/client.py`
- `tldw_chatbook/MCP/server.py`
- `tldw_chatbook/config.py`

## Acceptance Criteria

This slice is successful when all of the following are true:

- Chatbook exposes a real local MCP runtime surface in `Tools & Settings`
- the surface works without any server configured
- local runtime actions are policy-gated through `mcp.runtime.*.local`
- users can inspect local Chatbook MCP runtime readiness and launch/install metadata
- users can enable/disable local Chatbook MCP runtime configuration
- users can create/update/delete local external MCP connection profiles
- users can connect/disconnect/test local external MCP connection profiles
- users can browse discovered local MCP tools/resources/prompts
- users can manage locally stored MCP approval decisions
- the product does not imply that remote governance or remote catalog management are already part of local MCP runtime

## Follow-On Work Explicitly Deferred

- remote governance client
- remote catalog import/sync
- mixed local/remote MCP views
- manual general-purpose tool execution UI
- richer audit/history views
- HTTP/WebSocket transport parity
- multi-server governance bridging

## Recommendation

Proceed with `Option B`: build a dedicated Chatbook-owned local runtime shell over the existing MCP package, attach it to app bootstrap and runtime policy, persist local runtime state separately from static config, and surface it inside `Tools & Settings` as `Status`, `Catalog`, `Connections`, and `Approvals`.

This is the narrowest slice that:

- gives Chatbook real standalone MCP ownership
- respects the user's local-first governance requirement
- keeps remote governance clearly deferred
- builds on existing code instead of pretending MCP work has to start from zero
