# Portable Tool-use Packs — Design

Status: Approved in brainstorming; awaiting written-spec review

Date: 2026-08-31

Owner: Console tool policy / MCP permissions / Settings

Related:

- [ADR-107: Portable Tool-use Packs](../../../backlog/decisions/107-portable-tool-use-packs.md)
- [ADR-079: Workspace assistant defaults](../../../backlog/decisions/079-workspace-assistant-defaults.md)
- [ADR-032: Local Agent Tool Permission Boundary](../../../backlog/decisions/032-local-agent-tool-permission-boundary.md)
- [ADR-069: Console project-instruction local state and preflight](../../../backlog/decisions/069-console-project-instruction-local-state-and-preflight.md)
- [ADR-074: Portable Actor Packs and local Persona Visual runtime](../../../backlog/decisions/074-portable-actor-packs-and-local-persona-visual-runtime.md)
- [MCP Hub redesign design](2026-07-13-mcp-hub-redesign-design.md)
- [Workspace assistant defaults design](2026-08-29-workspace-assistant-defaults-design.md)
- [TASK-25713](../../../backlog/tasks/task-25713%20-%20Design-portable-Tool-use-Pack-export-and-import.md)

ADR required: yes

ADR path: `backlog/decisions/107-portable-tool-use-packs.md`

Reason: this design establishes a portable permission-policy contract, an
archive trust boundary, imported-profile lifecycle semantics, deletion behavior,
and a confirmation guard spanning the permission store and workspace registry.

## Summary

V1 introduces `.tldw-tool-pack`, a deterministic archive containing exactly one
flattened Tool policy profile. Import is review-first and installs a new,
unbound profile; it never overwrites an existing profile and never binds the
profile to a workspace. The first later workspace bind requires a fresh,
digest-bound confirmation.

The pack carries policy only. It does not carry tools, skills, plugins, MCP
server configuration, credentials, workspace or Persona data, or runtime
installation instructions. A separate Tools+Skills design may later compose
policy with installable capabilities, but it must not silently extend this
format or its trust assumptions.

## Goals

- Export one named profile as a complete snapshot of effective behavior for the
  permission-addressable tool inventory.
- Import that snapshot without changing the effective permissions of any
  existing workspace.
- Preserve exact current Allow decisions only when destination identity and
  contract fingerprints match.
- Fail closed for unknown, changed, missing, malformed, stale, or oversized
  input while retaining unresolved Deny rules when doing so cannot grant access.
- Keep the permission store as the sole runtime policy authority and workspace
  assistant defaults as the sole binding authority.
- Give users a compact, auditable management and review flow in the canonical
  Settings and MCP Permissions surfaces.

## Non-goals

- Packaging or installing tool implementations, skills, plugins, MCP servers,
  dependencies, executables, or runtime modifications.
- Exporting MCP connection profiles, commands, arguments, environment values,
  endpoints, authentication, managed-secret references, or discovery caches.
- Exporting Personas, persona policy rules, workspace bindings, the global kill
  switch, configuration gates, project-instruction bindings, approval history,
  or session grants.
- Synchronizing packs or profiles with `tldw_server`.
- Supporting more than one Tool policy profile per archive.
- Claiming native Windows picker/publication support in V1. The archive contract
  is Windows-safe; native Windows behavior is verified separately.

## Terminology

- **Tool policy profile**: one named permission profile in
  `MCPPermissionStore`. UI copy uses this term to distinguish it from an MCP
  connection/server profile.
- **Permission-addressable tool**: a tool whose invocation posture is resolved
  through `MCPPermissionStore` under a stable server key and raw tool name.
- **Contract fingerprint**: a SHA-256 digest of the raw tool name plus its
  normalized description and parameter schema. The portable identity is the
  surrounding `(authority, server_key, tool_name, contract fingerprint)` tuple.
  Keeping `server_key` outside the digest permits an explicitly reviewed server
  mapping while still detecting a changed tool contract. The digest is not proof
  that implementation behavior is identical.
- **Unbound profile**: an installed profile that no active or archived workspace
  references. It is not called dormant because the permission store has no
  enabled/disabled profile state.
- **Pending Deny**: a Deny rule retained without a current exact destination
  match. It can only restrict a future matching identity and never grant it.
- **Install**: commit a reviewed profile and its metadata to local policy
  storage. Installation is not workspace binding and not capability installation.

## 1. Portable archive contract

### 1.1 Envelope

The file extension is `.tldw-tool-pack`. The archive is a ZIP containing exactly
these two regular files:

1. `tool-pack.json`
2. `profile/profile.json`

There are no optional members. Import reads the two declared members directly;
it does not extract the archive. Any compression method other than `ZIP_STORED`,
directory entries, encrypted entries, symlinks,
hard links, devices, nested archives, duplicate names, case-folded collisions,
absolute paths, backslashes, dot segments, Windows device names, extra members,
and undeclared members are rejected.

Export uses `ZIP_STORED`, UTF-8 member names, no archive/member comments or extra
fields, a fixed `1980-01-01 00:00:00` ZIP timestamp, `create_system = 3`, and
regular-file mode `0644`. Members are written in the order above. Canonical JSON
uses UTF-8 without BOM, strings validated as NFC, sorted object keys, compact
separators, no insignificant whitespace, and one trailing newline. Arrays whose
order has no semantic meaning are sorted by their documented identity key.

Identity strings are never silently normalized: a non-NFC source identity fails
export/import, while user-authored display text and suggested ids are normalized
before review. Exact and Unicode-case-folded collisions among fallback or tool
identities are rejected.

The same source snapshot, portable display metadata, and producer version must
therefore produce byte-identical archives. There is no export timestamp.

### 1.2 Manifest

`tool-pack.json` has exactly this shape:

```json
{
  "schema": "tldw.tool-pack/v1",
  "producer": {"name": "tldw_chatbook", "version": "1.0.0"},
  "required_features": [],
  "profile": {
    "suggested_id": "research-tools",
    "display_name": "Research tools",
    "payload": "profile/profile.json"
  },
  "files": [
    {
      "path": "profile/profile.json",
      "size": 1234,
      "sha256": "0000000000000000000000000000000000000000000000000000000000000000"
    }
  ],
  "content_digest": "0000000000000000000000000000000000000000000000000000000000000000"
}
```

`required_features` is empty in V1. Import rejects an unknown required feature
rather than ignoring it. `files` declares the payload only, since the manifest
cannot inventory itself. `content_digest` is SHA-256 over this framed byte
sequence: the ASCII schema id, a NUL byte, canonical manifest JSON with
`content_digest` omitted, a NUL byte, then the exact payload bytes. The payload
file hash and size are independently verified before parsing.

`producer` is untrusted provenance for review display, never compatibility or
authorization evidence. `suggested_id` is only a suggestion; import review owns
the exact destination id. Exporting `default` or a reserved `ws-` profile must
ask for or derive a portable, nonreserved suggestion without including a
workspace id. Hidden Deny tombstones are not exportable; re-export of an ordinary
imported profile snapshots only its current effective policy.

### 1.3 Profile payload

`profile/profile.json` has exactly this shape:

```json
{
  "schema": "tldw.tool-profile/v1",
  "fallbacks": [
    {"authority": "mcp", "server_key": "*", "state": "ask"},
    {
      "authority": "builtin",
      "server_key": "agent:builtin",
      "state": "ask"
    }
  ],
  "tools": [
    {
      "authority": "mcp",
      "server_key": "local:docs",
      "tool_name": "search",
      "state": "allow",
      "contract_sha256": "0000000000000000000000000000000000000000000000000000000000000000"
    }
  ]
}
```

Allowed authorities are exactly `mcp` and `builtin`; allowed states are
`allow`, `ask`, and `deny`. Every current
permission-addressable tool appears once in `tools`, with its flattened effective
state. `contract_sha256` is required for Allow and Ask. It is optional for Deny,
which permits a stale stored Deny to remain restrictive without pretending the
old contract is current.

The contract fingerprint is SHA-256 of canonical JSON with exactly
`tool_name`, `description`, and `input_schema`; description line endings are
normalized to LF, all strings must be NFC, schema objects have sorted keys, and
arrays retain their declared order. Authority and server key deliberately remain
outside this digest and are checked by the surrounding portable identity/mapping.

Each fallback is Ask or Deny, never Allow. There is one MCP global fallback and
one fallback for every exported server, including MCP servers with an explicit
source server default and `agent:builtin`, whose resolver ignores the MCP global.
Keys are sorted by
`(authority, server_key)`; tools by `(authority, server_key, tool_name)`.

The strict portable id grammar is
`[a-z0-9][a-z0-9._-]{0,127}` after user-visible slug normalization. `default`
and the `ws-` prefix are reserved. Display names are NFC Unicode, 1–200 code
points. Authority, server, tool, producer, and version strings are nonempty and
bounded; identity strings are at most 512 UTF-8 bytes and producer fields at
most 128 UTF-8 bytes.

### 1.4 Hard bounds

Import and export fail instead of truncating:

| Limit | V1 maximum |
| --- | ---: |
| Archive bytes | 5 MiB |
| Manifest bytes | 256 KiB |
| Profile payload bytes | 4 MiB |
| ZIP members | exactly 2 |
| Tool entries | 2,000 |
| Distinct source servers | 256 |
| Fallback entries | 257 |
| Manual server mappings | 256 |
| JSON nesting depth | 12 |
| Parsed JSON nodes per file | 50,000 |
| Installed profiles, including `default` and tombstones | 128 |
| Projected canonical permission-store bytes | 8 MiB |
| One import receipt | 4 MiB |
| Receipt store total | 32 MiB |

An existing permission store over an install cap remains readable and editable,
but imports that grow it are refused. Tombstones count toward profile and byte
caps because their reserved ids and fail-closed behavior are durable authority.

## 2. Complete permission-addressable snapshot

### 2.1 Inventory source

Export must not use only `ToolCatalogRegistry`'s model-visible catalog. That
catalog can exclude denied, disabled, disconnected, or stale tools and can include
runtime orchestration or capability tools that the permission store does not
govern. Instead, `catalog_snapshot.py` builds a complete definitional inventory
from permission-addressable authorities:

- code-owned in-process builtins under `agent:builtin`, resolved with
  `resolve_builtin_state`;
- built-in MCP tools under `builtin:tldw_chatbook`;
- local tools under `local:__local__`;
- local external MCP connection profiles under `local:<profile_id>`, using live
  definitions or their validated cached definitions.

Current remote/server-source tools that are display-only and do not pass through
the local permission gate are excluded. Runtime orchestration tools such as
spawn/wait/load, skills, capability-gated Library tools outside a permission-store
namespace, raw CLI, and any other non-addressable catalog entry are excluded.

If an included authority cannot provide a complete definitional inventory, export
fails with `tool_pack.export.inventory_incomplete`; it does not silently export the
visible subset. Stored Deny rules with no live definition may be added as pending
Denies without a fingerprint. Stored Ask or Allow rules without a definition are
omitted and reported before export; they are never serialized as portable grants.

### 2.2 Flattening

Export loads one immutable permission-store payload and one immutable inventory
snapshot. It resolves every current tool through `resolve_effective_state` or
`resolve_builtin_state`, including named-profile inheritance, definition-hash
downgrades, and high-risk inherited-Allow floors. The flattened result is what
the source runtime would enforce before Persona/config/workspace narrowing; raw
stored values are not exported.

For each namespace/server, export also resolves the posture for an unseen tool.
Resolved Allow is clamped to Ask; Ask stays Ask; Deny stays Deny. This becomes the
safe fallback. Builtins receive their own fallback because they do not inherit the
MCP global default. The global kill switch is deliberately excluded and pure
state resolvers are used so a temporarily enabled kill switch does not rewrite a
profile into all Deny.

Configuration availability, Persona policy rules, project-instruction binding
authority, read-only workspace restrictions, ephemeral restrictions, and
capability gates are not profile data. The destination evaluates those runtime
gates after selecting the imported profile, so a pack can never bypass them.

### 2.3 Safe export publication

Export captures the chosen parent and destination identity at picker acceptance,
validates the `.tldw-tool-pack` name and regular-file/no-symlink boundary, and builds
the complete archive in a private temporary file. Immediately before publication it
revalidates the captured parent and destination. An overwrite requires explicit
confirmation for that exact destination identity; a missing destination appearing,
an existing destination changing, a parent substitution, or a nonregular target
fails with `destination_changed`. Publication uses a same-parent temporary file and
atomic replace where the host supports it. Cancellation or failure removes only the
validated private temporary file and never the destination.

This is a Tool-Pack-specific use of the captured-destination pattern. V1 does not
refactor or share Actor Pack internals, and Windows-native publication remains the
separate verification claim in §11.

## 3. Import review and mapping

### 3.1 Inspection is side-effect free

Inspection performs bounded ZIP admission, exact schema validation, digest
verification, profile-id normalization, destination inventory capture, and mapping
analysis without writing the permission store or workspace database. The review
object is immutable, process-local, and expires after 15 minutes.

The proposed destination id must be absent from the permission store **and** have
zero active or archived workspace references, including a dangling reference to a
currently missing profile. Otherwise the result would not be unbound and review
requires another id.

The review displays:

- pack display name, producer, content digest, and exact proposed destination id;
- a clear **unbound** notice and a statement that import does not install tools;
- source fallback posture and Allow/Ask/Deny counts;
- exact matches, changed contracts, missing tools, pending Denies, and omitted
  Ask/Allow rules;
- disconnected-but-cached destination servers distinctly from connected servers;
- any explicit source-server to destination-server mapping.

The only commit action is **Import unbound profile**. There is no import-and-bind
shortcut.

### 3.2 Exact and manual mapping

Automatic matching requires exact authority, source server key, raw tool name,
and contract fingerprint. LLM-facing sanitized names are never identities because
they can change with collision order and provider projection.

When an external MCP server id differs, review may explicitly map one source server
to one installed destination server. Mapping never uses labels, fuzzy matching, or
heuristics. It is one-to-one: two source servers may not map to one destination,
and duplicate resulting destination tool identities fail review. A disconnected
destination may be selected only when its stored validated definitions provide
the exact contract fingerprints. Mapping controls only stable policy identities;
it never imports commands, environment, endpoints, credentials, or secrets.

For exact matches, Allow remains Allow and Ask remains Ask. A changed or missing
Allow/Ask is omitted as an exact rule. The destination's compiled safe fallback
still applies and can only Ask or Deny. An unresolved Deny may remain under the
source identity or its reviewed mapped identity as a pending Deny because that
cannot grant access. No unresolved Ask/Allow is retained for future exact
autoactivation.

### 3.3 Revalidation and installation

Commit re-reads the archive, repeats all admission and digest checks, refreshes the
destination inventory, and verifies the exact review digest, mappings, destination
id, store digest, caps, absence of an existing profile, and absence of active or
archived references to that id. If anything changed,
the review is stale and the user must inspect again. Import never overwrites or
silently suffixes a profile. If review chose `research-2`, commit either installs
exactly `research-2` or fails.

The activation compiler writes safe per-server defaults (Ask/Deny only) plus exact
tool exceptions into a new named permission profile. It never materializes a broad
Allow default. For a reviewed destination server, its fallback is the pack's
clamped source fallback. For an unmapped/missing source server, Ask fallback and
Ask/Allow entries are omitted; a Deny fallback or Deny entry may be retained under
the source key. Current tool entries equal to a compiled fallback need not be
stored as overrides; the installed effective result must nevertheless reproduce
the reviewed matched snapshot.

Import writes a bounded detailed receipt first, then atomically installs the
profile and minimal metadata. A receipt orphaned by a failed authority commit is
safe and removed by best-effort cleanup. The successful profile install is the
authority boundary; a receipt never grants permission.

## 4. Storage and concurrency

`MCPPermissionStore.SCHEMA_VERSION` remains `1`. Bumping it is prohibited for this
feature because unknown versions trigger backup/reset and could destroy existing
permissions. The additive top-level `profile_metadata` object is keyed by profile
id and contains only:

- origin (`local`, `workspace_managed`, `imported`, or `tombstone`);
- pack content digest and local import time for imported profiles;
- `first_bind_confirmation_required`;
- receipt id and compact match/omission counts;
- current canonical profile content digest and revision.

Legacy schema-1 payloads remain valid and gain no metadata until a related feature
mutates them. Detailed unresolved identities live in a separate bounded Tool Pack
receipt store so the hot-path permission payload does not duplicate up to 2,000
tool identities.

The profile content digest covers only the canonical normalized profile object, not
store timestamps or metadata. Every permission mutator that changes a profile also
updates that profile's digest/revision in the same locked save. For an imported
profile whose first-bind marker remains set, this invalidates every outstanding bind
token without clearing the marker. Malformed optional metadata is quarantined from
display and reported, but never causes the schema-1 permission payload to be backed
up or reset.

The permission store gains narrow complete-profile operations: install-if-absent,
replace-with-tombstone, and metadata update. All mutators for the same resolved
store path share one process-wide reentrant lock, including separate
`MCPPermissionStore` instances. A mutator reloads while holding the lock, validates
its expected store/profile digest, applies one complete change, enforces caps, and
atomically replaces the file. This removes in-process lost updates. Chatbook remains
a single-process authority; arbitrary concurrent writers in another process are
unsupported, although ordinary pre-commit staleness is detected by digest. The
feature makes no cross-process locking claim.

Profile installation also holds the lifecycle coordinator lock across its final
workspace-reference check and permission-store save. Every workspace defaults
mutation that sets or clears `tool_policy_profile_id` holds that same lock, even for
local and currently missing profile ids. Therefore a dangling-reference bind cannot
race installation: either binding commits first and import refuses the referenced
id, or import commits first and the later bind encounters the imported first-bind
guard.

## 5. First workspace binding

Installation leaves the profile unbound. On the first attempt to place an imported
profile in `WorkspaceAssistantDefaults.tool_policy_profile_id`, the central
workspace mutation boundary requires confirmation. The guard is not UI-only.

A `ToolProfileBindingGuard` composes the workspace registry and permission store:

1. Review recomputes the current profile, its digest and revision, Allow/Ask/Deny
   counts, target workspace, Persona id, persona memory mode, and full intended
   assistant-default payload.
2. Explicit confirmation issues an opaque, process-local, one-use token with a
   10-minute TTL bound to all those values and the intended `set` action.
3. `LocalWorkspaceRegistryService.set_assistant_defaults` calls the guard before
   its database transaction whenever the target profile has
   `first_bind_confirmation_required`.
4. The guard atomically consumes the token immediately before the transaction.
   A failed transaction requires a fresh review; a token can never be replayed.
5. After a successful database commit, the marker is cleared best-effort while
   the lifecycle lock is held. A crash or metadata-write failure may repeat the
   prompt, but it cannot create an unconfirmed binding.

Any profile content mutation changes the digest/revision and invalidates outstanding
tokens. Existing local and auto-created profiles pass unchanged. Provisioning and
backfill remain unchanged because their `ws-` profiles are not imported. Binding is
a separate user action after import, and the first-bind modal displays the current
profile rather than trusting the historical import receipt.

The existing `confirm_read_write` memory gate remains independent. A Tool-profile
confirmation token does not satisfy it, and a memory confirmation does not satisfy
the imported-profile guard; a UI may present both facts in one modal only if the
registry receives and validates both acknowledgements separately.

Import, bind, and removal share one lifecycle coordinator lock. Binding holds it through
the workspace database commit; removal holds it while checking active and archived
references and writing the tombstone. If bind commits first, removal sees the
reference. If removal wins first, binding sees a tombstone and fails.

## 6. Removal and tombstones

Hard deletion is unsafe: ADR-079 makes an unknown named profile inherit from
`default`, so a stale workspace reference or in-flight run could widen. Removal
therefore follows these rules:

- `default`, any `ws-` profile, and an existing tombstone are non-removable.
- Removal refuses while any active or archived workspace references the profile or
  while a Console run/Test Tool operation holds a runtime lease for it.
- Successful removal atomically replaces the profile with a hidden Deny tombstone
  and marks its metadata origin `tombstone`.
- The tombstone profile object carries a validated `tombstone: true` sentinel.
  Every permission resolver checks that sentinel before named-profile inheritance
  and returns Deny for every permission-addressable authority. The stored profile
  also sets MCP global fallback Deny and explicitly sets `agent:builtin` Deny as a
  current-schema defense in depth. It contains no Allow or Ask entry.
- The id remains reserved permanently, is hidden from normal profile pickers, counts
  toward caps, and cannot be reused or imported over.
- A resolver that has not yet admitted a call, or that reloads the old profile id,
  observes Deny rather than destination `default`. Removal does not pretend to
  revoke a tool invocation that was already authorized and dispatched; the runtime
  lease prevents removal while such work is active.

Console runs and management tests acquire/release the same coordinator's lightweight
profile lease around use. Removal deletes or compacts the detailed receipt only after
the tombstone is durable.
Re-export always snapshots the profile's current effective policy; original omitted
rules do not reappear from provenance or receipts.

## 7. UI design

### 7.1 Settings → Tool Profiles

The canonical `UI/Screens/settings_screen.py` composes a modular Tool Profiles
panel from `Widgets/Settings_Widgets/tool_profiles_panel.py`; the feature does not
add more large inline logic to the already oversized screen and does not modify
deprecated settings surfaces.

The panel lists non-tombstoned profiles with origin (`local`,
`workspace-managed`, `imported`), exact id, bound/unbound status, and reference
count. Detail shows effective fallback posture, Allow/Ask/Deny counts, active and
archived workspace references, minimal provenance, first-bind state, and the
durable import receipt. Actions are Import, Export, Remove, and **Edit permissions**.

Import opens the review in §3. First bind opens the separate current-state modal in
§5. Imported profiles may be edited before first bind; editing preserves the marker
and invalidates outstanding tokens.

### 7.2 MCP Permissions profile selector

The current MCP Permissions matrix edits only `default`, so V1 adds a **Tool policy
profile** selector. Every effective-state read, global/server/tool mutation,
definition re-allow, policy preview, and Test Tool approval is passed the selected
`profile_id`. Switching profiles invalidates the matrix's cached effective states.
The Settings **Edit permissions** action deep-links to this selector rather than
duplicating the rule editor.

Testing a tool under the selected profile is a management action, not workspace
binding. Existing runtime/config/Persona/project gates still apply. A persistent
approval writes to the selected profile through normal permission-store APIs.

### 7.3 Interaction constraints

Review and detail are compact and scrollable, render untrusted strings as plain text
without Rich markup, restore focus after modals, and follow the repository keybinding
contract. No screen binds terminal-convention keys or shadows global shortcuts, and
footer hints advertise only implemented actions. Normal and compact terminal layouts
are mounted in UI tests.

## 8. Module ownership

New pure/service code lives under `tldw_chatbook/Tool_Packs/`:

- `contracts.py`: strict schemas, canonical encoding, bounds, identifiers, and
  stable error categories;
- `catalog_snapshot.py`: permission-addressable inventory adapters and fingerprints;
- `export.py`: flattening and immutable export snapshot;
- `publication.py`: Tool-Pack-specific captured-destination safe publication;
- `importer.py`: bounded inspection, mapping, and immutable review objects;
- `activation.py`: revalidation and safe permission-profile compilation;
- `receipt_store.py`: bounded, non-authoritative import receipts;
- `binding.py`: binding review/token guard and lifecycle coordination;
- `service.py`: presentation-facing orchestration.

`MCPPermissionStore` remains the policy authority and gains only atomic profile and
metadata primitives. The workspace registry remains binding authority and invokes a
narrow guard protocol. Settings and MCP Permissions remain presentation. V1 may reuse
the captured-destination pattern from Actor Packs, but it does not refactor or couple
Actor Pack publication code.

No database schema migration is expected. The permission JSON additions and receipt
file are profile-local data, not portable content.

## 9. Failure model and observability

Errors expose stable, path-free categories and bounded user copy. Logs contain only
safe ids, counts, digests, and exception type; they never contain source/destination
paths, commands, arguments, environment, endpoints, credentials, descriptions,
untrusted archive text, or receipt contents.

| Operation | Categories |
| --- | --- |
| Export | `profile_unavailable`, `profile_invalid`, `inventory_incomplete`, `too_large`, `destination_invalid`, `destination_changed`, `cancelled`, `publication_failed` |
| Import | `archive_invalid`, `schema_unsupported`, `feature_unsupported`, `manifest_invalid`, `inventory_invalid`, `payload_invalid`, `identity_duplicate`, `mapping_invalid`, `too_large`, `review_stale`, `capacity_exceeded`, `store_changed`, `activation_failed` |
| Bind | `confirmation_required`, `confirmation_stale`, `confirmation_expired`, `confirmation_invalid` |
| Remove | `referenced`, `non_removable`, `stale` |

The public code prefix is `tool_pack.<operation>.<category>`. Exceptions may carry
private diagnostic causes internally, but UI and logs use the stable category.
Cancellation, publication failure, stale review, capacity failure, and any validation
failure leave no partial installed profile or workspace binding.

## 10. Privacy contract

Pack schema rejects rather than ignores fields that could carry local or secret
state. Export tests recursively assert the absence of:

- source/destination paths, commands, arguments, environment, endpoints, URLs,
  credentials, tokens, secret references, or discovery runtime state;
- tool descriptions or schemas in plaintext (only contract digests travel);
- approval/execution logs, session grants, permission-store timestamps,
  `config_changed`, profile metadata, or import receipts;
- workspace ids/names/bindings, Persona ids/names/policy, project-instruction state,
  and the global kill switch.

Stable server keys, raw tool names, safe display name, suggested portable id,
producer, state, and fingerprints are intentionally portable policy metadata and are
shown before export/import. A reserved workspace profile id is never used as the
default suggested id, preventing accidental workspace-id disclosure.

## 11. Verification strategy

### Contract and archive

- Golden byte tests prove deterministic exports and canonical JSON.
- Malformed ZIP, duplicate/case-folded members, traversal, Windows device names,
  linked/special/encrypted entries, extra members, nested content, bad hashes,
  oversized/deep JSON, unknown schema/features/fields, and forbidden privacy keys
  fail with pinned categories.
- Boundary tests cover every stated size/count/string/id limit.

### Inventory, flattening, and mapping

- Complete-inventory fixtures include disabled, denied, disconnected, and cached
  definitional tools while excluding every non-addressable provider.
- Resolver parity tests prove named inheritance, rug-pull downgrade, high-risk floor,
  builtin-specific fallback, safe Allow-to-Ask fallback clamping, and exclusion of
  kill-switch/config/Persona/workspace gates.
- Exact and manual mapping tests cover one-to-one enforcement, collisions, changed
  contracts, cached destinations, omitted Ask/Allow, and pending Deny.
- Property tests assert that inspecting or importing an unbound profile never changes
  the effective permissions of any existing active or archived workspace, including
  workspaces with dangling references to the proposed destination id.

### Storage, binding, and removal

- Schema-1 compatibility, corrupt optional metadata, install-if-absent, projected
  caps, receipt-orphan cleanup, stale store/name races, and multiple store instances
  sharing one path lock receive targeted tests.
- Token tests cover one-use, TTL, digest/revision/payload binding, mutation
  invalidation, direct registry-call bypass attempts, transaction failure, and safe
  repeated confirmation after marker-clear failure.
- Removal tests cover active and archived references, runtime leases, bind/remove
  races, reserved profiles, hidden tombstones, permanent id reservation,
  resolver-level tombstone short-circuit, builtin/global Deny defense in depth, and
  a stale unresolved reference resolving Deny rather than `default`.

### UI and performance

- Textual pilots mount normal and compact layouts; exercise import review, manual
  mapping, first bind, removal, selector/deep-link behavior, plain-text rendering,
  focus restoration, and keybinding/footer contracts.
- A maximum-size export loads the permission payload once and each authority inventory
  once; activation performs one locked store reload and one atomic save. Tests assert
  those structural bounds, preventing per-tool disk loads. Canonicalization is
  `O(n log n)` for at most 2,000 tools and never runs on the Textual event loop.
- A non-gating benchmark records inspect/activation time and peak memory at maximum
  bounds so regressions are visible without a platform-brittle wall-clock promise.

Targeted suites covering touched modules are required. A full repository sweep is
run only when explicitly requested, following repository testing policy.

### Windows claim

Platform-independent tests enforce POSIX member names, Windows device-name rejection,
no extraction, portable ids, and path-free errors. V1 does not claim that native
Windows file picking, captured-destination identity, or atomic publication has been
live-verified. That claim belongs to a separate Windows support task and Windows host.

## 12. Implementation slicing

The implementation plan should split this design into independently reviewable work:

1. contract, canonical archive, metadata/receipt bounds, and permission-store atomic
   primitives;
2. permission-addressable inventory, flattening, export, and safe publication;
3. import inspection, exact/manual mapping, activation, and receipts;
4. binding guard, lifecycle coordinator, and deny tombstone removal;
5. modular Settings profile management and MCP Permissions profile selector;
6. targeted security, concurrency, UI, performance, and Windows-contract verification.

The plan must preserve dependency order and must not combine Tools+Skills or plugin
installation exploration with these V1 implementation tasks.

## 13. Future Tools+Skills exploration boundary

A future Tools+Skills Pack is a different product and trust boundary. Before design,
it must answer at least:

- whether a pack references installed tools/skills or carries executable content;
- how publisher identity, signatures, review, quarantine, dependency resolution,
  update/revocation, licensing, and rollback work;
- whether installation is restart-bound, skill-copy based, plugin based, or a future
  runtime plugin contract;
- how a policy pack behaves when its required capability is absent or removed;
- which process owns install authority and what sandbox/permission review applies.

`.tldw-tool-pack/v1` must never be interpreted as executable or installable content.
Any combined format requires a new schema and ADR.
