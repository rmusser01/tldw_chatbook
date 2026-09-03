# ADR-107: Portable Tool-use Packs

- **Status:** Accepted
- **Date:** 2026-08-31
- **Task:** [TASK-29232](../tasks/task-29232%20-%20Design-portable-Tool-use-Pack-export-and-import.md)
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
states, Ask/Deny fallbacks, stable permission identities, and contract fingerprints
covering name, description, input schema, and policy-relevant risk tags.
It excludes tools, skills, plugins, MCP connection/server configuration, credentials,
commands, environment, endpoints, global kill switch, workspace/Persona data,
approval history, session grants, and runtime-install instructions.

Tools+Skills packaging and plugin/runtime installation are a separate future schema,
ADR, and trust boundary. A V1 Tool-use Pack is never executable content.

### 2. Snapshot complete permission-addressable behavior, not visible catalog state

Export inventories only authorities actually governed by `MCPPermissionStore`,
through a code-owned portability registry: code-owned builtins, the built-in MCP
server, local/raw-shell/Virtual CLI tools, and local external MCP definitions
available live or from validated cache. Display-only server-source tools, skills,
orchestration tools, and non-addressable capability tools are excluded and reported.
A newly permission-addressable but unclassified namespace blocks export instead of
silently disappearing.

Export resolves named-profile inheritance, rug-pull downgrade, and high-risk floors
through the existing pure resolvers, then records the flattened effective result.
Destination/config/Persona/workspace gates and the global kill switch remain outside
the pack. Every future-tool fallback is clamped to Ask or Deny; no broad Allow
fallback can be exported or installed.

Named-profile propagation is a shipping prerequisite across every included provider.
Resolution, by-key gates, persistent approvals, and profile-scoped session approvals
must use the same captured profile id; importing a profile must never cause an
approval to land in `default` or affect a different workspace.

### 3. Import is review-first, exact, and unbound

Import inspection is side-effect free and uses a strict, non-mutating permission
snapshot. Corrupt or unknown-version live policy bytes are left exactly untouched;
inspection never invokes the permission store's legacy backup/reset recovery.
Automatic mapping requires exact authority, server key, raw tool name, and portable
contract fingerprint. A user may explicitly map one source MCP server to one
installed destination server; labels, fuzzy matching, many-to-one mapping,
connection configuration, and secret transfer are forbidden.

Exact Allow and Ask rules may be installed. A risk-tag-only change invalidates an
exact mapping. The portable fingerprint is validation evidence, not the runtime
`definition_hash`; activation recomputes the latter from the exact destination
definition. Missing or changed Allow/Ask rules are omitted and
never retained for future exact autoactivation. An unresolved Deny may
be retained under its reviewed identity because it can only restrict. The compiled
profile uses only Ask/Deny defaults plus exact exceptions and never a server/global
Allow default.

Activation always writes an explicit safe named MCP-global fallback plus the
independent builtin and per-server fallbacks, all Ask/Deny only. Therefore an unseen
destination server cannot inherit a broad Allow from `default`.

Commit repeats archive and destination validation, checks the reviewed destination
id and store digest, and atomically installs only if the id is still absent and no
active or archived workspace references it, including through a dangling reference.
The final reference check and install share the lifecycle lock used by every binding.
Import does not overwrite, silently suffix, select, or bind the profile. The
successful result is accurately called an **unbound profile**, not a dormant profile.

### 4. Keep runtime policy and binding authority separate

`MCPPermissionStore` remains the only policy authority. Schema version remains 1.
An imported profile has a durable `profile_kind: "tool_pack_imported"`
discriminator and a required, validated `tool_pack_lifecycle` object containing the
first-bind marker, policy digest/revision, receipt link, and compact counts. A
tombstone has the corresponding tombstone kind/origin. Legacy profiles have neither.
If one field exists without the other, either is malformed, or kind and origin
disagree, all resolvers fail closed and the profile cannot bind, export, or edit.
Detailed bounded import receipts live outside the hot-path store and are provenance,
never authority.

The store adds `read_snapshot_strict()`: a non-mutating schema/nested-shape read that
returns immutable policy plus a generation digest and never creates, renames,
normalizes, backs up, or resets the live file. Tool Pack inspection, export,
revalidation, and outcome reconciliation use this seam, not legacy `load()`.

All instances targeting the same resolved permission-store path share one
process-wide reentrant mutator lock/profile fence. Complete-profile install,
lifecycle update, and tombstone replacement reload and validate under that lock,
enforce profile and byte caps, then atomically replace the file. The fixed lock order
is lifecycle coordinator, permission-store fence, then workspace SQLite transaction;
no code acquires it in reverse. Chatbook is the single-process authority; the
decision makes no arbitrary cross-process writer guarantee.

Workspace assistant defaults remain binding authority. The first bind of an imported
profile requires a fresh, one-use, short-lived token bound to the exact profile
digest/revision, workspace, Persona/memory settings, action, and complete intended
assistant-default payload. Every entry point that creates, sets, replaces, clears,
provisions, or backfills defaults—including inline defaults during workspace
creation—uses the central guard; UI confirmation alone and direct-service calls
cannot bypass it. Bind holds the store fence from final strict token/profile
validation through workspace commit and marker clear, closing the edit-after-review
race. Existing local/auto-managed profiles are behaviorally unchanged but traverse
the lifecycle-aware write boundary. Import by itself cannot change any existing
workspace's effective policy.

Receipts are written through private mode-`0600` temporaries, atomically replaced,
fsynced, and capacity-reserved before profile
authority commits. Referenced receipts are not auto-evicted; startup only reclaims
expired, unreferenced, unowned orphans. A missing receipt degrades provenance without
changing policy or bypassing the authoritative first-bind marker. Install, removal,
and publication reconcile exact state after ambiguous post-replace failures and
report an explicit uncertain outcome rather than guessing.

### 5. Removal leaves a deny tombstone

A referenced imported profile cannot be removed, including references from archived
workspaces, and an active Console/Test Tool runtime lease for its exact captured id
also blocks removal. V1 does not add deletion for local, legacy, workspace-managed,
invalid-lifecycle, or already tombstoned profiles. An unreferenced valid imported
profile is atomically replaced with a hidden, permanent Deny tombstone rather than
deleted. A compact receipt is staged first under a new id and linked by the tombstone;
the former detailed receipt becomes eligible for bounded orphan-grace cleanup only
after strict outcome reconciliation. The validated tombstone
discriminator/lifecycle pair makes every permission resolver return Deny before
profile inheritance; MCP-global and explicit
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
Tool policy profile selector and remains the only rule editor. Every row/action
captures the selected profile id, selector generation, and profile digest/revision;
profile switches or edits make pending actions stale instead of retargeting them.
Every read, mutation, re-allow, preview, persistent/session approval, and Test Tool
operation is explicitly scoped to that captured profile. Deprecated settings
surfaces are not extended.

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
| Reuse legacy `load()` for inspection | Its recovery path may rename corrupt policy and return defaults. Review must be byte-preserving and fail closed through a strict read seam. |
| Make lifecycle information optional display metadata | A missing marker could silently convert a reviewed imported policy into an ordinary bindable profile. Kind/lifecycle consistency is runtime authority. |
| Copy the portable contract digest into runtime `definition_hash` | They intentionally hash different framed fields. Activation validates the portable contract, then recomputes the runtime hash from the destination definition. |
| Store detailed receipts in the permission file | It duplicates up to 2,000 identities in a hot-path file and increases every resolver load. A separate bounded receipt store preserves review evidence. |
| Fall back to non-atomic export overwrite | A partial or redirected destination could be published. Unsupported secure primitives must produce a stable failure instead. |
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
- A profile discriminator/lifecycle object plus a separate bounded receipt store add
  two profile-local persistence concerns without changing the permission schema
  version.
- The permission store gains strict snapshot validation, in-process multi-instance
  serialization, and profile-scoped accessors; arbitrary concurrent external writers
  remain unsupported.
- Tombstones consume permanent profile ids and capacity until a future versioned
  migration defines a stronger safe-reclamation proof.
- Windows-safe archive validation is covered in V1, but native Windows publication
  and picker support require separate live verification before being claimed.

## Links

- [Portable Tool-use Packs design](../../Docs/superpowers/specs/2026-08-31-tool-use-pack-design.md)
- [TASK-29232](../tasks/task-29232%20-%20Design-portable-Tool-use-Pack-export-and-import.md)
- [ADR-032: Local Agent Tool Permission Boundary](032-local-agent-tool-permission-boundary.md)
- [ADR-069: Console project-instruction local state and preflight](069-console-project-instruction-local-state-and-preflight.md)
- [ADR-074: Portable Actor Packs and local Persona Visual runtime](074-portable-actor-packs-and-local-persona-visual-runtime.md)
- [ADR-079: Workspace assistant defaults](079-workspace-assistant-defaults.md)
