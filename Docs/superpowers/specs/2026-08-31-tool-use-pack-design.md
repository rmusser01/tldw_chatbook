# Portable Tool-use Packs — Design

Status: Approved

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
- [TASK-29232](../../../backlog/tasks/task-29232%20-%20Design-portable-Tool-use-Pack-export-and-import.md)

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
  normalized description, parameter schema, and policy-relevant risk tags. The
  portable identity is the surrounding
  `(authority, server_key, tool_name, contract fingerprint)` tuple.
  Keeping `server_key` outside the digest permits an explicitly reviewed server
  mapping while still detecting a changed tool contract. The digest is not proof
  that implementation behavior is identical.
- **Unbound profile**: an installed profile that no active or archived workspace
  references. It is not called dormant because the permission store has no
  enabled/disabled profile state.
- **Pending Deny**: a Deny rule retained without a current exact destination
  match. It can only restrict a future matching identity and never grant it.
- **Install**: commit a reviewed profile and its lifecycle authority to local policy
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
`external_attr = (S_IFREG | 0o644) << 16`. The canonical writer also pins
`create_version = 20`, `extract_version = 20`, `flag_bits = 0`,
`internal_attr = 0`, and volume/disk start `0`; it emits no data descriptor.
Members are written in the order above. Cross-runtime golden vectors, rather than
one library's defaults, define the authoritative bytes.

Canonical JSON uses UTF-8 without BOM, strings validated as NFC, sorted object
keys, compact separators, non-ASCII characters unescaped, non-finite numbers
forbidden, no insignificant whitespace, and one trailing newline. Arrays whose
order has no semantic meaning are sorted by their documented identity key. Strict
decoding rejects duplicate object keys, `NaN`, positive/negative infinity, lone
Unicode surrogates, invalid UTF-8, and non-NFC identity strings before schema
validation. File-size admission happens before decoding; depth and node limits are
checked immediately after this strict parse.

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
`tool_name`, `description`, `input_schema`, and `policy_risk_tags`; description line
endings are normalized to LF, all strings must be NFC, schema objects have sorted
keys, and schema arrays retain their declared order. `policy_risk_tags` is the sorted,
deduplicated set of normalized tags that the provider's permission resolver actually
uses for inherited-Allow flooring—not arbitrary display labels. A change to those
tags therefore makes an exact Allow/Ask mapping stale even when name and schema are
unchanged. Authority and server key deliberately remain outside this digest and are
checked by the surrounding portable identity/mapping.

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
| Import-admission profile count, including `default` and tombstones | 128 |
| Import-admission projected canonical permission-store bytes | 8 MiB |
| One import receipt | 4 MiB |
| Receipt store total | 32 MiB |

These are import-admission caps, not new global limits on ordinary workspace
provisioning. An existing permission store over either cap remains readable and
editable, but imports that grow it are refused. Tombstones count toward profile and
byte caps because their reserved ids and fail-closed behavior are durable authority.

## 2. Complete permission-addressable snapshot

### 2.1 Inventory source

Export must not use only `ToolCatalogRegistry`'s model-visible catalog. That
catalog can exclude denied, disabled, disconnected, or stale tools and can include
runtime orchestration or capability tools that the permission store does not
govern. Instead, `catalog_snapshot.py` builds a complete definitional inventory
from a code-owned portability registry. Every provider whose posture is resolved
through `MCPPermissionStore` must be classified as included or explicitly
nonportable; a new unclassified permission namespace makes export fail rather than
silently disappearing from the pack. V1 includes:

- code-owned in-process builtins under `agent:builtin`, resolved with
  `resolve_builtin_state`;
- built-in MCP tools under `builtin:tldw_chatbook`;
- local tools under `local:__local__`, including raw-shell rows after the raw-shell
  runtime's permanent Allow-to-Ask floor is applied;
- read-only Virtual CLI tools under `local:__virtual_cli__`;
- local external MCP connection profiles under `local:<profile_id>`, using live
  definitions or their validated cached definitions.

Code-owned local and Virtual CLI definitions are captured in the unbound fallback
context: configured `[console].workspace_root` or app cwd, with no selected
project-instruction binding or admitted-root aliases. Those workspace-specific
schema additions are deliberately not profile data. If a later bound run projects a
different contextual schema, the existing destination runtime definition-hash guard
downgrades the stored exact Allow to Ask; export never embeds a workspace locator or
tries to make one profile's Allow silently valid across different root schemas.

Current remote/server-source tools that are display-only and do not pass through
the local permission gate are excluded. Runtime orchestration tools such as
spawn/wait/load, skills and managed-skill approval tools, capability-gated Library
tools outside a permission-store namespace, and any other non-addressable catalog
entry are excluded. The export review reports every excluded category and count;
“complete” means all V1-classified permission-addressable Tool authorities, never
only the currently model-visible subset.

If an included authority cannot provide a complete definitional inventory, export
fails with `tool_pack.export.inventory_incomplete`; it does not silently export the
visible subset. Stored Deny rules with no live definition may be added as pending
Denies without a fingerprint. Stored Ask or Allow rules without a definition are
omitted and reported before export; they are never serialized as portable grants.

### 2.2 Flattening

Export obtains one strict, immutable, non-mutating permission-store snapshot and one
immutable inventory snapshot. It resolves every current tool through the provider's
actual pure policy adapter: `resolve_effective_state`, `resolve_builtin_state`, or
the raw-shell two-state floor layered on the former. This includes named-profile
inheritance, definition-hash downgrades, and high-risk inherited-Allow floors. The
flattened result is what the source runtime would enforce before
Persona/config/workspace narrowing; raw stored values are not exported.

For each namespace/server, export also resolves the posture for an unseen tool.
Resolved Allow is clamped to Ask; Ask stays Ask; Deny stays Deny. This becomes the
safe fallback. Builtins receive their own fallback because they do not inherit the
MCP global default. The global kill switch is deliberately excluded and pure
state resolvers are used so a temporarily enabled kill switch does not rewrite a
profile into all Deny.

Configuration availability, Persona policy rules, project-instruction binding
authority, read-only workspace restrictions, ephemeral restrictions, and
capability gates are not profile data. App-run session approvals are also excluded
from the snapshot. The destination evaluates those runtime gates after selecting the
imported profile, so a pack can never bypass them.

Before pack export/import can ship, named-profile propagation must be correct for
every included runtime. In particular, Console `BuiltinToolGate` resolves
`agent:builtin` under the run's captured profile; local, Virtual CLI, raw-shell, and
MCP providers send persistent approvals to that same profile; all provider gates and
by-key fallbacks receive it; and in-memory session approvals are keyed by
`(profile_id, server_key, tool_name)`. This is a prerequisite correctness repair,
not optional UI polish: otherwise a run can resolve an imported profile but write an
Allow into `default`, changing unrelated workspaces.

### 2.3 Safe export publication

Export first consumes the normalized path returned by central path validation, then
captures the chosen parent and destination identity at picker acceptance, validates
the `.tldw-tool-pack` name and regular-file/no-symlink boundary, and builds the
complete archive in a private temporary file. Immediately before publication it
revalidates the captured parent and destination. V1 fails closed with
`publication_unsupported` when the destination already exists because the supported
POSIX primitives do not provide compare-and-swap replacement against the captured
inode and digest. For an absent destination, publication flushes and fsyncs the
complete private file, atomically hard-links it into the captured parent with
no-replace semantics, removes the private name, and fsyncs the parent directory. A
destination appearing, parent substitution, or nonregular target fails without being
overwritten. A host lacking the required secure primitive fails with
`publication_unsupported`; non-atomic overwrite is forbidden. Failure after the link
may have committed reconciles the destination identity/digest and reports either
success or the distinct committed-but-`durability_uncertain` outcome; it never tells
the user no archive was published when the new file may already be visible.

This is a Tool-Pack-specific use of the captured-destination pattern. V1 does not
refactor or share Actor Pack internals, and Windows-native publication remains the
separate verification claim in §11.

## 3. Import review and mapping

### 3.1 Inspection is side-effect free

Inspection consumes the normalized path returned by central path validation before
its suffix, descriptor, and identity checks. It performs bounded ZIP admission,
exact schema validation, digest verification, profile-id normalization, destination
inventory capture, and mapping analysis without writing the permission store or
workspace database. Permission
authority is read only through the strict snapshot API in §4; inspection must never
call legacy `MCPPermissionStore.load()`, whose corrupt/unknown-version recovery can
rename the live file. Invalid authority storage fails with
`tool_pack.import.store_invalid` and leaves every byte untouched. The review object
is immutable, process-local, and expires after 15 minutes.

The proposed destination id must have no exact or Unicode-case-folded collision with
an existing profile/tombstone and have zero active or archived workspace references,
including a dangling reference to a currently missing profile. Otherwise the result
would not be unbound and review requires another id.

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
tool exceptions into a new named permission profile. It always writes the reviewed
MCP-global Ask/Deny fallback, then the independent `agent:builtin` fallback, then
every reviewed per-server Ask/Deny default and exact exception. It never materializes
a broad Allow default, and the named global fallback prevents an unseen destination
server from inheriting a broad Allow from `default`. For a reviewed destination
server, its fallback is the pack's clamped source fallback. For an unmapped/missing
source server, Ask fallback and Ask/Allow entries are omitted; a Deny fallback or
Deny entry may be retained under the source key. Current tool entries equal to a
compiled fallback need not be stored as overrides; the installed effective result
must nevertheless reproduce the reviewed matched snapshot.

The portable `contract_sha256` (which also covers policy risk tags) is never copied into the runtime
`definition_hash` field. Mapping uses the portable hash defined in §1.3; for each
matched external Allow, activation recomputes the existing runtime
`definition_hash(description, input_schema)` from the exact destination definition
and stores that value. Hash-free code-owned namespaces retain their established
runtime policy after the import-time exact contract check.

Import durably writes a bounded detailed receipt first, then atomically installs the
profile and its authoritative lifecycle sentinel. If an exception occurs around the
authority replace, activation performs a strict reload: an exact lifecycle/profile
digest is reconciled as installed, absence is failure, and any third state returns
`activation_uncertain` without automatic retry. The successful profile install is
the authority boundary; a receipt never grants permission.

## 4. Storage and concurrency

`MCPPermissionStore.SCHEMA_VERSION` remains `1`. Bumping it is prohibited for this
feature because unknown versions trigger backup/reset and could destroy existing
permissions.

### 4.1 Strict snapshot reads

`MCPPermissionStore.read_snapshot_strict()` is a new non-mutating authority seam.
It reads the current bytes once, strictly validates schema 1 and nested profile
shape, and returns an immutable payload plus exact file/store digest. A missing file
returns a fresh in-memory default with a missing-file generation token. Corrupt JSON,
unknown schema, invalid nested structure, or an I/O error raises a typed snapshot
error without renaming, backing up, normalizing on disk, creating, or resetting any
file.

Export, import inspection, store digesting, activation revalidation, and post-commit
reconciliation must use this strict seam. They never call legacy `load()`, whose
documented recovery behavior may rename a corrupt file and return defaults. Ordinary
runtime callers retain existing `load()` behavior; Tool Pack review cannot trigger
that recovery as a side effect.

### 4.2 Authoritative profile lifecycle

An imported profile carries two additive fields inside the profile itself: the
durable discriminator `profile_kind: "tool_pack_imported"` and one required
`tool_pack_lifecycle` object:

```json
{
  "schema": "tldw.tool-pack-lifecycle/v1",
  "origin": "imported",
  "pack_digest": "0000000000000000000000000000000000000000000000000000000000000000",
  "imported_at": "2026-08-31T00:00:00Z",
  "first_bind_confirmation_required": true,
  "receipt_id": "tp-00000000000000000000000000000000",
  "receipt_digest": "0000000000000000000000000000000000000000000000000000000000000000",
  "counts": {"matched": 0, "omitted": 0, "pending_deny": 0},
  "policy_digest": "0000000000000000000000000000000000000000000000000000000000000000",
  "revision": 1
}
```

This discriminator/sentinel pair—not optional display metadata—is binding authority.
Legacy/local profiles have neither field and remain valid. If either field exists,
both must be present and consistent. An imported discriminator with a missing,
malformed, or partially missing lifecycle object; a lifecycle object without the
discriminator; an unknown discriminator; or an origin/kind mismatch resolves Deny
with origin `lifecycle_invalid`, cannot be bound/exported/edited, and requires
explicit repair. It is never reclassified as a legacy local profile. A tombstone
uses `profile_kind: "tool_pack_tombstone"` and the exact tombstone lifecycle variant:
`origin: "tombstone"`, `first_bind_confirmation_required: false`, preserved pack
provenance, removal time, compact receipt id/digest, policy digest, and revision. It
has the resolver-level Deny behavior in §6.

`policy_digest` covers the canonical normalized policy fields plus `profile_kind`,
excluding the lifecycle object and store timestamp. Every permission mutator that
changes an imported profile also
increments its revision and updates its policy digest in the same locked save. The
first-bind marker stays set, so every outstanding token becomes stale. Detailed
unresolved identities remain in the separate receipt store instead of duplicating
up to 2,000 entries in the hot-path permission payload.

### 4.3 Atomic mutation and lock order

The permission store gains profile-scoped accessors, tombstone-aware resolvers, and
narrow complete-profile operations: install-if-absent, update-with-expected-revision,
and replace-with-tombstone. Raw/default getters gain an explicit `profile_id`; every
mutator and low-level save for the same resolved store path shares one process-wide
reentrant lock, including separate `MCPPermissionStore` instances. A mutator reads
under that lock, validates expected store/profile digest, applies one complete
change, enforces import caps where applicable, and atomically replaces the file.

The global lock order is:

1. Tool-profile lifecycle coordinator;
2. resolved permission-store path lock/profile mutation fence;
3. workspace SQLite transaction.

No code acquires these in reverse. First bind holds the profile mutation fence from
final token/revision validation through the workspace commit and best-effort marker
clear, so an ordinary permission edit cannot land between confirmation and binding.
Profile installation holds both outer locks across its final workspace-reference
check and permission-store save. Every workspace defaults mutation that creates,
sets, replaces, or clears `tool_policy_profile_id` holds the lifecycle lock, even
for local and currently missing ids. Thus a dangling-reference bind cannot race
installation: either binding commits first and import refuses the referenced id, or
import commits first and the later bind encounters the imported first-bind guard.

This removes in-process lost updates and closes bind/edit races. Chatbook remains a
single-process authority; arbitrary concurrent writers in another process are
unsupported, although ordinary pre-commit staleness is detected by digest. The
feature makes no cross-process locking claim.

### 4.4 Receipt durability and recovery

Receipts are canonical, bounded records written through mode-`0600` private temporary
files, atomic replace, file fsync, and parent-directory fsync before profile authority
can commit. The
receipt id and receipt digest are linked from `tool_pack_lifecycle`; a referenced
receipt is never automatically evicted. Import reserves receipt-store capacity
before writing, and release/cleanup is idempotent.
Receipt ids use the exact grammar `tp-[0-9a-f]{32}` from 128 random bits; creation
retries an authenticated-name collision rather than replacing an existing receipt.

Startup reconciliation removes only receipts that no installed profile references,
that no live review/commit owns, and whose bounded orphan grace period has elapsed.
It never deletes a referenced receipt. A missing/corrupt referenced receipt degrades
provenance display and is repairable, but does not change policy or bypass the
authoritative first-bind marker. Removing an old unbound imported profile can compact
policy bytes and detailed receipt storage after the orphan grace period, but its
permanent tombstone cannot free a profile-count slot. A profile-count exhaustion
error therefore offers no false automatic recovery; increasing/migrating that cap is
a future versioned decision. Imports never truncate receipt details silently.

## 5. First workspace binding

Installation leaves the profile unbound. On the first attempt to place an imported
profile in `WorkspaceAssistantDefaults.tool_policy_profile_id`, the central
workspace mutation boundary requires confirmation. The guard is not UI-only and
does not depend on an import receipt being present.

A `ToolProfileBindingGuard` composes the workspace registry and permission store:

1. Review acquires the lifecycle coordinator, reads a strict permission snapshot,
   and recomputes the current discriminator/lifecycle validity, policy digest and
   revision, effective Allow/Ask/Deny counts, target workspace, Persona id, persona
   memory mode, and full intended assistant-default payload. It shows the global and
   builtin fallback posture, every current Allow-bearing server fallback, every
   stored exact Allow (including currently unavailable or rug-pull-downgraded rows),
   every current effective Allow, and which known Allows are high-risk. An Allow
   whose current risk/definition cannot be verified is called out as unavailable,
   not omitted; expandable Ask/Deny detail is available without relying on the
   historical receipt.
2. Explicit confirmation issues an opaque, process-local, one-use token with a
   10-minute TTL bound to all those values and the intended `set` action.
3. Every registry entry point that can create, set, replace, clear, provision, or
   backfill assistant defaults—including `create_workspace(...,
   assistant_defaults=...)` and direct service calls—routes through one binding
   guard before it can write a non-null profile id. There is no lower-level
   persistence bypass.
4. Commit acquires locks in the §4.3 order, re-reads the profile strictly under the
   store mutation fence, and validates the token, lifecycle marker, current policy
   digest/revision, exact intended defaults, reference state, and action. It then
   atomically consumes the token and starts the workspace transaction while retaining
   the store fence. A failed transaction requires fresh review; a token is never
   replayable.
5. After the workspace commit, the guard clears
   `first_bind_confirmation_required` with an expected-revision profile update while
   still holding both outer locks and the store fence. If the workspace commit is
   known successful but marker persistence fails, the binding remains safe and the
   next operation prompts again. If the database commit outcome is uncertain, the
   guard reconciles the exact intended defaults before reporting
   `binding_uncertain`; it never clears the marker on an unverified outcome.

Any profile content mutation changes the digest/revision and invalidates outstanding
tokens. Existing local and auto-created profiles pass without a Tool-Pack prompt,
but still traverse the lifecycle-aware mutation boundary so they cannot race an
installation/removal. Provisioning and backfill remain behaviorally unchanged
because their `ws-` profiles are not imported. Binding is a separate user action
after import, and the first-bind modal displays the current profile rather than
trusting the historical import receipt.

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

- Only a valid `tool_pack_imported` profile is removable in V1. `default`, local or
  otherwise legacy profiles, any `ws-` profile, invalid-lifecycle profiles, and an
  existing tombstone are non-removable; other profile-deletion behavior is outside
  this feature.
- Removal refuses while any active or archived workspace references the profile or
  while a Console run/Test Tool operation holds a runtime lease for it.
- Successful removal atomically replaces the profile with a hidden Deny tombstone
  whose required discriminator/lifecycle pair is
  `profile_kind: "tool_pack_tombstone"` / `origin: "tombstone"`.
- Every permission resolver checks that validated tombstone pair before named-profile
  inheritance and returns Deny for every permission-addressable authority. The stored profile
  also sets MCP global fallback Deny and explicitly sets `agent:builtin` Deny as a
  current-schema defense in depth. It contains no Allow or Ask entry.
- The id remains reserved permanently, is hidden from normal profile pickers, counts
  toward caps, and cannot be reused or imported over.
- A resolver that has not yet admitted a call, or that reloads the old profile id,
  observes Deny rather than destination `default`. Removal does not pretend to
  revoke a tool invocation that was already authorized and dispatched; the runtime
  lease prevents removal while such work is active.

Console runs and management tests acquire/release the same coordinator's lightweight
profile lease for the exact captured `profile_id` from policy selection until the
last governed invocation/test completes. Before authority replacement, removal
durably stages a bounded compact tombstone receipt under a new id. The tombstone links
that receipt; after strict reconciliation proves the tombstone durable, the old
detailed receipt is merely unreferenced and later follows normal orphan-grace cleanup.
An uncertain outcome retains both receipts and returns `outcome_uncertain` for
explicit recovery.
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
durable import receipt. Actions are Import, Export, Remove, and **Edit permissions**;
Remove is enabled only for valid imported profiles that satisfy §6.

Import opens the review in §3. First bind opens the separate current-state modal in
§5. Imported profiles may be edited before first bind; editing preserves the marker
and invalidates outstanding tokens.

### 7.2 MCP Permissions profile selector

The current MCP Permissions matrix edits only `default`, so V1 adds a **Tool policy
profile** selector. Every effective-state read, global/server/tool mutation,
definition re-allow, policy preview, and Test Tool approval is passed the selected
`profile_id`. Each rendered row, inspector, confirmation, re-allow action, and test
captures that `profile_id` together with the selector generation and strict profile
digest/revision. Switching profiles increments the generation and invalidates the
matrix's cached effective states; a stale event is rejected rather than retargeted to
the newly selected profile.
The Settings **Edit permissions** action deep-links to this selector rather than
duplicating the rule editor.

Testing a tool under the selected profile is a management action, not workspace
binding. Existing runtime/config/Persona/project gates still apply. A persistent
approval writes to the exact captured profile through normal permission-store APIs;
by-key gates and the profile-scoped session-approval key use the same captured id.
Audit events record only the safe profile id, action, result, and digests—not tool
arguments or receipt content.

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

`MCPPermissionStore` remains the policy authority and gains strict non-mutating
snapshots, profile-scoped raw accessors/resolvers, a shared path mutation fence, and
atomic complete-profile/lifecycle primitives. Included providers must propagate the
captured profile through resolution plus persistent and session approval paths before
portable profiles can ship. The workspace registry remains binding authority and
invokes a narrow guard protocol from every defaults-write entry point. Settings and
MCP Permissions remain presentation. V1 may reuse
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
| Export | `profile_unavailable`, `profile_invalid`, `store_invalid`, `inventory_incomplete`, `too_large`, `destination_invalid`, `destination_changed`, `cancelled`, `publication_unsupported`, `publication_failed`, `durability_uncertain` |
| Import | `archive_invalid`, `schema_unsupported`, `feature_unsupported`, `manifest_invalid`, `inventory_invalid`, `payload_invalid`, `identity_duplicate`, `mapping_invalid`, `too_large`, `review_stale`, `capacity_exceeded`, `store_invalid`, `store_changed`, `destination_referenced`, `activation_failed`, `activation_uncertain` |
| Bind | `confirmation_required`, `confirmation_stale`, `confirmation_expired`, `confirmation_invalid`, `lifecycle_invalid`, `binding_uncertain` |
| Remove | `referenced`, `in_use`, `non_removable`, `stale`, `outcome_uncertain` |

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
  `config_changed`, profile lifecycle/discriminator fields, or import receipts;
- workspace ids/names/bindings, Persona ids/names/policy, project-instruction state,
  and the global kill switch.

Stable server keys, raw tool names, safe display name, suggested portable id,
producer, state, and fingerprints are intentionally portable policy metadata and are
shown before export/import. A reserved workspace profile id is never used as the
default suggested id, preventing accidental workspace-id disclosure.

## 11. Verification strategy

### Contract and archive

- Cross-runtime golden byte tests prove deterministic exports, every pinned ZIP
  header/attribute, and canonical JSON.
- Malformed ZIP, duplicate/case-folded members, traversal, Windows device names,
  linked/special/encrypted entries, extra members, nested content, bad hashes,
  oversized/deep JSON, unknown schema/features/fields, and forbidden privacy keys
  fail with pinned categories.
- Strict JSON tests reject duplicate keys, non-finite numbers, invalid UTF-8, lone
  surrogates, and non-NFC identity strings before schema activation.
- Boundary tests cover every stated size/count/string/id limit.

### Inventory, flattening, and mapping

- Complete-inventory fixtures include disabled, denied, disconnected, and cached
  definitional tools, raw-shell, Virtual CLI, and every included MCP namespace while
  reporting every excluded non-addressable provider. A registry tripwire proves a
  newly permission-addressable but unclassified namespace blocks export.
- Resolver parity tests prove named inheritance, rug-pull downgrade, high-risk floor,
  builtin-specific fallback, safe Allow-to-Ask fallback clamping, and exclusion of
  kill-switch/config/Persona/workspace gates.
- Included-provider tests prove resolution, by-key gates, persistent approvals, and
  session approvals all use the same captured named profile; a Console builtin/local/
  Virtual CLI/raw-shell/MCP approval under an imported profile never mutates
  `default` or grants another profile.
- Exact and manual mapping tests cover one-to-one enforcement, collisions, changed
  contracts, risk-tag-only changes, cached destinations, omitted Ask/Allow, and
  pending Deny.
- Activation tests prove the named MCP-global fallback is always explicit and safe,
  so a later unseen server cannot inherit an Allow from `default`, and prove portable
  contract hashes are validated but destination runtime `definition_hash` values are
  independently recomputed.
- Property tests assert that inspecting or importing an unbound profile never changes
  the effective permissions of any existing active or archived workspace, including
  workspaces with dangling references to the proposed destination id.

### Storage, binding, and removal

- Strict-snapshot tests prove missing files return an in-memory generation, valid
  schema-1 is immutable, and corrupt/unknown-version/nested-invalid bytes remain
  byte-for-byte unchanged with no backup, reset, normalization, or file creation.
- Compatibility tests cover legacy profiles with neither discriminator nor lifecycle,
  every missing/malformed/mismatched discriminator/lifecycle combination resolving
  Deny, install-if-absent, projected caps, stale store/name races, and multiple store
  instances sharing one path fence. Every legacy permission mutator preserves the
  discriminator/lifecycle pair byte-for-byte except for its required atomic
  revision/digest update, and refuses to edit an invalid pair.
- Receipt tests inject crashes before/after receipt replace and profile replace,
  verify capacity reservation/release, idempotent cleanup, bounded orphan-grace
  reconciliation, preservation of referenced/live-review receipts, and degraded
  provenance without marker bypass when a referenced receipt is unavailable.
- Token tests cover one-use, TTL, digest/revision/full-payload binding, mutation
  invalidation, every registry create/set/replace/clear/provision/backfill bypass
  attempt (including `create_workspace` with inline defaults), transaction failure,
  uncertain-commit reconciliation, independent `confirm_read_write`, and safe repeated
  confirmation after marker-clear failure.
- Concurrency tests enforce the lifecycle → store-fence → SQLite lock order and prove
  an ordinary profile edit, direct defaults writer, install, bind, and removal cannot
  interleave between final token validation and workspace commit.
- Removal tests cover active and archived references, runtime leases, bind/remove
  races, reserved profiles, hidden tombstones, permanent id reservation,
  resolver-level tombstone short-circuit, builtin/global Deny defense in depth, and
  a stale unresolved reference resolving Deny rather than `default`, plus exact
  post-replace reconciliation and uncertain-outcome receipt preservation.

### UI and performance

- Textual pilots mount normal and compact layouts; exercise import review, manual
  mapping, first bind, removal, selector/deep-link behavior, plain-text rendering,
  focus restoration, and keybinding/footer contracts.
- Selector race tests capture profile id, selector generation, and profile
  digest/revision in every row/inspector/confirmation/re-allow/Test Tool event; a
  profile switch or edit makes the event stale instead of applying it elsewhere.
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
Publication fault tests on supported hosts also cover unavailable secure primitives,
pre-replace cleanup, post-replace digest reconciliation, and the distinct
committed-but-`durability_uncertain` result.

## 12. Implementation slicing

The implementation plan should split this design into independently reviewable work:

1. strict non-mutating permission snapshots, profile discriminator/lifecycle
   validation, shared mutation fencing, and profile-scoped provider propagation;
2. contract, canonical archive, receipt bounds/durability, and permission-store
   complete-profile primitives;
3. permission-addressable inventory, flattening, export, and safe publication;
4. import inspection, exact/manual mapping, activation, and crash reconciliation;
5. binding guard across every workspace-default writer, lifecycle coordinator,
   runtime leases, and Deny tombstone removal;
6. modular Settings profile management and profile-safe MCP Permissions selector;
7. targeted security, concurrency, UI, performance, and Windows-contract verification.

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
