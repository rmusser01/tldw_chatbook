# ADR-107: Portable Tool-use Packs

- **Status:** Proposed
- **Date:** 2026-08-31
- **Task:** [TASK-25713](../tasks/task-25713%20-%20Design-portable-Tool-use-Pack-export-and-import.md)
- **Design:** [Portable Tool-use Packs design](../../Docs/superpowers/specs/2026-08-31-tool-use-pack-design.md)
- **Related:** [ADR-032](032-local-agent-tool-permission-boundary.md),
  [ADR-069](069-console-project-instruction-local-state-and-preflight.md),
  [ADR-074](074-portable-actor-packs-and-local-persona-visual-runtime.md), and
  [ADR-079](079-workspace-assistant-defaults.md)

## Context

Chatbook has named Tool policy profiles in `MCPPermissionStore`, workspace
references to those profiles, a unified tool catalog, and several tool providers.
It does not have a portable policy format. Copying the raw permission-store JSON
would carry global state and destination-specific inheritance, miss disabled or
disconnected definitions, preserve stale Allow values, and confuse policy with MCP
connection configuration.

An imported profile is security-sensitive even when it carries no executable code:
an exact Allow can authorize a later tool call. Import also crosses two existing
boundaries. ADR-079 makes workspaces reference named profiles that inherit from
`default`; ADR-032 places builtin, local, and external MCP tools under permission
identities with different fallback behavior. The design must not make import a
workspace bind, weaken existing runtime floors, or turn profile removal into a
fallback grant.

## Decision

### 1. Define one deterministic, policy-only profile per pack

`.tldw-tool-pack/v1` contains exactly one flattened Tool policy profile in a
deterministic two-file ZIP. It carries safe portable ids, effective Allow/Ask/Deny
states, Ask/Deny fallbacks, stable permission identities, and contract fingerprints.
It excludes tools, skills, plugins, MCP connection/server configuration, credentials,
commands, environment, endpoints, global kill switch, workspace/Persona data,
approval history, session grants, and runtime-install instructions.

Tools+Skills packaging and plugin/runtime installation are a separate future schema,
ADR, and trust boundary. A V1 Tool-use Pack is never executable content.

### 2. Snapshot complete permission-addressable behavior, not visible catalog state

Export inventories only authorities actually governed by `MCPPermissionStore`,
including code-owned builtins, the built-in MCP server, local tools, and local
external MCP definitions available live or from validated cache. Display-only
server-source tools, skills, orchestration tools, and non-addressable capability
tools are excluded.

Export resolves named-profile inheritance, rug-pull downgrade, and high-risk floors
through the existing pure resolvers, then records the flattened effective result.
Destination/config/Persona/workspace gates and the global kill switch remain outside
the pack. Every future-tool fallback is clamped to Ask or Deny; no broad Allow
fallback can be exported or installed.

### 3. Import is review-first, exact, and unbound

Import inspection is side-effect free. Automatic mapping requires exact authority,
server key, raw tool name, and contract fingerprint. A user may explicitly map one
source MCP server to one installed destination server; labels, fuzzy matching,
many-to-one mapping, connection configuration, and secret transfer are forbidden.

Exact Allow and Ask rules may be installed. Missing or changed Allow/Ask rules are
omitted and never retained for future exact autoactivation. An unresolved Deny may
be retained under its reviewed identity because it can only restrict. The compiled
profile uses only Ask/Deny defaults plus exact exceptions and never a server/global
Allow default.

Commit repeats archive and destination validation, checks the reviewed destination
id and store digest, and atomically installs only if the id is still absent and no
active or archived workspace references it, including through a dangling reference.
The final reference check and install share the lifecycle lock used by every binding.
Import does not overwrite, silently suffix, select, or bind the profile. The
successful result is accurately called an **unbound profile**, not a dormant profile.

### 4. Keep runtime policy and binding authority separate

`MCPPermissionStore` remains the only policy authority. Schema version remains 1;
minimal additive `profile_metadata` records imported origin, digest/revision,
first-bind requirement, receipt id, and compact counts. Detailed bounded import
receipts live outside the hot-path store and are not authority.

All instances targeting the same resolved permission-store path share one
process-wide reentrant mutator lock. Complete-profile install, metadata update, and
tombstone replacement reload and validate under that lock, enforce profile and byte
caps, then atomically replace the file. Chatbook is the single-process authority;
the decision makes no arbitrary cross-process writer guarantee.

Workspace assistant defaults remain binding authority. The first bind of an imported
profile requires a fresh, one-use, short-lived token bound to the exact profile
digest/revision, workspace, Persona/memory settings, and intended assistant-default
payload. The central `set_assistant_defaults` mutation validates and consumes the
token; UI confirmation alone is insufficient. Existing local/auto-managed profiles
and workspace backfill are unchanged. Import by itself cannot change any existing
workspace's effective policy.

### 5. Removal leaves a deny tombstone

A referenced profile cannot be removed, including references from archived
workspaces, and an active Console/Test Tool runtime lease also blocks removal. An
unreferenced removable profile is atomically replaced with a hidden, permanent Deny
tombstone rather than deleted. A validated tombstone sentinel makes every permission
resolver return Deny before profile inheritance; MCP-global and explicit
`agent:builtin` Deny values provide defense in depth for the current namespaces. It
is not reusable and counts toward storage caps.

This prevents a stale or not-yet-resolved workspace reference from falling through
to `default`, which ADR-079 specifies for an unknown profile. Already dispatched
invocations are not claimed to be revocable; their runtime lease blocks removal.
`default`, `ws-` profiles, and tombstones are non-removable. Bind, lease, and removal
share one lifecycle coordinator so none can pass a stale reference/use check while
another commits.

### 6. Use the canonical management and editing surfaces

Settings gains a modular Tool Profiles panel for import/export, origins, references,
receipts, binding state, and removal. The existing MCP Permissions surface gains a
Tool policy profile selector and remains the only rule editor. Every read, mutation,
re-allow, preview, and Test Tool operation is explicitly scoped to the selected
profile. Deprecated settings surfaces are not extended.

## Alternatives considered

| Alternative | Why rejected |
| --- | --- |
| Export the raw permission-store profile JSON | It preserves destination inheritance and stale values, omits a complete authority inventory, and lacks a stable portable trust contract. |
| Define a provider-neutral semantic policy DSL | It would invent a second policy engine and drift from the runtime authority. V1 snapshots the behavior the current resolver actually enforces. |
| Export multiple profiles or workspace bindings together | It introduces identity graphs, overwrite policy, and cross-workspace mutations before the single-profile trust boundary is proven. |
| Import and bind in one action | A valid archive could immediately alter live agent authority. Unbound install makes that lifecycle explicit and reversible. |
| Match servers/tools by display name or LLM-facing name | Those names are not stable identities and can collide or change with projection order. |
| Preserve unresolved Allow or Ask for future tools | A future definition could acquire policy without exact review. Only Deny is safe to retain unmatched. |
| Hard-delete an unreferenced profile | Unknown named profiles inherit `default`, so stale/in-flight references could widen. A Deny tombstone fails closed. |
| Bump permission-store schema version | Unknown versions trigger backup/reset and would risk destroying live permissions; the metadata addition is compatible and additive. |
| Store detailed receipts in the permission file | It duplicates up to 2,000 identities in a hot-path file and increases every resolver load. A separate bounded receipt store preserves review evidence. |
| Include skills, plugins, or MCP server setup | That carries executable/configuration trust, dependencies, secrets, installation, updates, and revocation—materially different from portable policy. |

## Consequences

### Benefits

- Users can move a reviewed Tool policy without moving secrets, server setup, or
  executable content.
- Exact current Allows remain useful while changed/missing contracts fail closed.
- Import alone is inert with respect to existing workspaces, and first use receives
  a current-state confirmation.
- Complete inventory and builtin-aware fallbacks reflect runtime behavior more
  faithfully than model-visible catalog export.
- Tombstones preserve fail-closed behavior for stale and not-yet-resolved references.

### Costs and constraints

- Export requires complete definitional inventory, so unavailable authority metadata
  can block export instead of producing a partial pack.
- Manual external-server mapping adds a review step when local ids differ.
- Minimal profile metadata plus a separate bounded receipt store add two profile-local
  persistence concerns without changing the permission schema version.
- The permission store gains in-process multi-instance serialization; arbitrary
  concurrent external writers remain unsupported.
- Tombstones consume permanent profile ids and capacity until a future versioned
  migration defines a stronger safe-reclamation proof.
- Windows-safe archive validation is covered in V1, but native Windows publication
  and picker support require separate live verification before being claimed.

## Links

- [Portable Tool-use Packs design](../../Docs/superpowers/specs/2026-08-31-tool-use-pack-design.md)
- [TASK-25713](../tasks/task-25713%20-%20Design-portable-Tool-use-Pack-export-and-import.md)
- [ADR-032: Local Agent Tool Permission Boundary](032-local-agent-tool-permission-boundary.md)
- [ADR-069: Console project-instruction local state and preflight](069-console-project-instruction-local-state-and-preflight.md)
- [ADR-074: Portable Actor Packs and local Persona Visual runtime](074-portable-actor-packs-and-local-persona-visual-runtime.md)
- [ADR-079: Workspace assistant defaults](079-workspace-assistant-defaults.md)
