# Native package inspection

`inspect_package(root: Path, *, dialect=None)` reads an existing canonical absolute
package directory. It does not execute commands, launch MCP, fetch schemas,
install a package, or grant trust. Use a worker for this bounded filesystem work.
`PackageInspection` is an immutable record; component definitions are canonical
JSON strings and collections cannot be mutated in place after hashing.

## Authoring

Use Agent Plugins 1.0.0 `plugin.json`, immediate `skills/<name>/SKILL.md` children,
and optional root `mcp.json`. Unknown manifest fields are diagnosed and ignored.
Invalid skills and MCP entries are isolated from valid siblings. A missing or
unsupported portable schema rejects the portable interpretation.

`extensions.io.github.rmusser01.chatbook` requires integer `version: 1`. Its only
optional fields are `commands`, `rules`, `agents`, `hooks`, `requires`, and
`variables`. Paths are contained package files; there is no implicit extension
folder scan. Commands require `name`/`description`, rules `name`/`mode`, and agents
`name`/`description`/`tools` YAML frontmatter. An empty agent tools array means no
tools. Hook files use the closed v2 envelope from ADR-163. Tool/model mappings
are explicit blockers where they still need host configuration.

Dependencies use typed local IDs such as `skill:review` and `hook:protect-writes`.
Missing dependencies, duplicate IDs, and cycles block affected components.
Unrecoverable recognized extension constraints block activation conservatively;
unrelated unknown extension namespaces do not. Native variables require closed,
typed declarations, cannot override `PLUGIN_ROOT`/`PLUGIN_DATA`, and secret
variables cannot supply package defaults. Required unset variables block use.

Every parsed component starts disabled, with parsed-only evidence. Support is a
format interpretation result, not runtime compatibility or permission. Hook
transformer/guard effects and MCP transport declarations remain explicit for
later runtime owners. No publication into live skills/tools occurs in this
foundation module.

## Snapshot operation and identity

`materialize_package(source, destination)` requires an absent destination under
an existing canonical parent. It rejects all unsafe or incomplete source captures
before writing, creates a private destination, writes regular files, and verifies
its resulting content digest. Existing destinations are never overwritten.
Internal regular-file links become regular files. Directory/external/broken
links, special files, reparse points, and ambiguous platform spellings are refused.
Case-folding and Unicode normalization collisions are conservatively rejected.

- `content_digest` binds sorted relative file paths, exact bytes, and executable
  bits of the materialized representation. It reproduces on destination inspection.
- `source_digest` additionally binds normalized internal link targets.
  `link_targets` retains that provenance after links become regular files.
- `effective_digest` binds native normalized identity, component definitions,
  instruction-file hashes, dependencies, variables, blockers, and interpretation.
  Ignored portable fields alone do not change it. Both content and effective
  digests are review inputs; neither grants trust.
- `source_identity` names the captured source; `materialized_identity` names the
  verified destination when materialization succeeded. Evidence timestamps are
  deliberately outside deterministic identity.

Inspection retains safe portable components after narrow path errors, but an
incomplete capture has no content/source digest and cannot be materialized.
Source and destination mutations are checked through anchored descriptors and
identity comparisons. This is a package acquisition boundary, not a sandbox
against arbitrary same-user code or a substitute for the later trust/lease owner.

Limits: 256 KiB/depth 32 per definition JSON; 100 MiB expanded, 10,000 files,
10 MiB/file, path depth 32; 512 normalized components; 64 native hooks. Traversal
also caps directories at `10,000 * 32`, bounding empty-directory trees. Every
visited entry consumes its file/directory budget before filename validation;
unreadable entry metadata consumes the file budget. Retained filesystem diagnostics
share a 10,000-entry cap. Files are read in 64 KiB chunks and counted before retention. Frontmatter is bounded to
256 KiB and rejects aliases, duplicate fields, and non-JSON values. JSON rejects
nonfinite numbers (including exponent overflow) and non-UTF-8-representable strings
or keys before normalization. Numeric timeout bounds precede float conversion.

## Qualification boundaries

POSIX descriptor APIs and no-follow support are required. Automated filesystem
checks currently run on macOS; Windows fails closed as `platform_capture_unqualified`.
Windows reparse behavior is not claimed qualified. Vendor markers retain candidate
identity and ambiguity but are explicitly unsupported/unqualified until a vendor
adapter is delivered. A valid native root defaults to its native interpretation;
independent vendor roots require an explicit choice. An inline OpenAI overlay
replaces the compatibility-file overlay wholesale.

See [ADR-162](../../backlog/decisions/162-managed-agent-plugins.md),
[ADR-163](../../backlog/decisions/163-expanded-console-hook-runtime.md), and
[fixture provenance](../../Tests/Plugins/fixtures/PROVENANCE.md).

Portable remote MCP definitions reject userinfo or fragment delimiters in URLs,
case-insensitive duplicate headers, invalid HTTP field characters, and leading or
trailing field whitespace. Loopback HTTP and ordinary HTTPS definitions retain
the same inspection boundary; syntax validation never grants network access.

## Private registry and runtime owner (foundation)

`PluginRuntimeOwner(root)` takes the **plugins directory derived from the actual
resolved user-data profile**. `try_acquire()` returns false for a competing owner;
unsupported storage or lock failures raise and leave plugin execution unavailable.
Other application features do not depend on acquiring this lock. The stable
`runtime.lock` file is retained on close, never unlinked to release ownership.
Parent aliases such as `/tmp` resolve consistently; a symlink replacing the owned
root or lock is refused. Instances belong to one serialized worker and one process.

`PluginRegistry(root / "registry.sqlite3")` opens an existing validated read-only
view, including committed WAL frames. It cannot create or migrate the store.
Supply `owner=acquired_owner` for mutations. Each `transaction()` checks current
ownership before starting and before commit, rolls back on failures, and returns
only after SQLite commit succeeds. Disk settings explicitly require WAL,
`synchronous=FULL`, and macOS `fullfsync=ON`. A later authority coordinator may
publish its protected commit certificate only **after** this return. These rows
are unauthenticated metadata and do not grant trust or execution permission.
`:memory:` is available for nonpersistent data tests; it grants no runtime owner.

Schema v1 is packaged in `migrations/001_initial.sql`; v2 adds separate mutable
`revision_trust` and independently retained `tombstones` through
`002_authority.sql`. Exact predecessor validation, upgrade and versioning share
one owned transaction; reopening verifies exact schema and database integrity.
A read-only v1 view requires the owner to perform that upgrade first.
Tables hold installations, immutable revision/component records, selections,
activation, sources, mappings, authority generations, operation intents,
data-root generations/deletion fences, process evidence, and receipts. Persist
F1 inspection records via `model_dump(mode="json")`, retaining the distinct
support, selection, availability and provenance fields. Installation activation
defaults to disabled. `activation.intent` distinguishes explicit `inherit`,
`enabled` and `disabled`; absence is a separate state. Authority scopes use
`scope_kind` (`installation`, `global_default`, `workspace`) plus `workspace_id`;
non-workspace scopes require an empty workspace ID and workspace scopes require
a nonempty stable ID. A workspace named `installation` cannot collide with the
installation-wide scope. Public list methods require a page of 1–50 rows.

Before spawning, the host calls `reserve_launch(operation_id, installation_id,
workspace_id, revision_digest)` and durably obtains a token. After spawning,
`publish_process(token, provenance)` retains exact non-secret host-captured
process identity. `settle_process(token, confirmed=False)` retains unresolved
ownership; true is a trusted host assertion that all owned writers stopped,
never a conclusion from lock acquisition or a stale PID. These APIs do not spawn,
signal, kill or reconcile any process automatically. Pending/unresolved records
survive close and owner death, and a new owner cannot reserve work for an affected
installation until host reconciliation settles them. Other installations remain
available. `list_processes` exposes bounded evidence for reconciliation.

`active_revision_leases` counts pending launches and active runs, including
unresolved active work. `set_process_kind` distinguishes a published active run,
idle connection and archived history; idle/history are not active run leases.
Their unresolved process evidence still blocks reuse after owner death: no run
lease does **not** mean that a data root has no surviving writers. Process evidence
intentionally survives installation/revision removal without cascading deletion.
Fencing, authenticated authority, coordinator recovery and consumer admission
arrive in subsequent foundation tasks; the registry alone authorizes none of them.

### Runtime storage qualification

The owner is currently qualified on **64-bit macOS local APFS**. The small native
Darwin `statfs64` probe uses the SDK structure and requires `MNT_LOCAL` plus APFS. HFS has no separate test-volume evidence
here and remains unqualified. Unknown platforms, filesystem types and probe failures refuse
ownership. Known synchronized ancestors (CloudStorage, Mobile Documents, Dropbox,
OneDrive, Google Drive) are refused before creating directories. Arbitrary
third-party syncing cannot be detected generically and is **unsupported** even
when a local lock succeeds. Network/sync semantics, Windows and Linux are not
qualified. This is host process ownership, not sandbox containment or proof that
escaped descendants and external side effects have stopped.


## Authenticated authority (F3)

`PluginAuthorityStore` is an internal primitive for the existing plugin owner and
coordinator. **F4/app callers must hold the acquired F2 runtime owner before
bootstrap, prepare, certify, marker advancement or reset.** Secondary instances
remain read-only. UI must use the coordinator/owner gate, never call these
mutating primitives around it. This store creates no second owner or permission
service, and its signature does not grant component eligibility or tool access.

Use `default_plugin_authority_dir(local_skills_store_dir)` for the fixed protected
`trust/plugins` namespace. It has its own 32-byte salt and schema-v1 metadata,
separate scrypt/HMAC purpose keys, and marker service `tldw_chatbook.plugin_trust`
with scoped account `managed-plugins:generation-marker:v1:<scope>`. Standalone
skill accounts, manifests, salts, key caches and snapshots remain unchanged.
There is no plugin key cache: unlock authenticates current material on each start.

Explicit `bootstrap(passphrase)` creates only a canonical empty generation-zero
snapshot and the reserved `bootstrap` marker. It refuses existing/partial setup;
`unlock` never bootstraps. An unavailable secure keyring blocks mutation.
`FilePluginMarkerStore` is reduced rollback protection and requires explicit
`accept_reduced_protection=True`; constructing it is not acceptance. The posture
API distinguishes setup, locked, unavailable, recovery and reduced protection.

`authority_projection(operation_result=...)` reads one consistent SQLite snapshot
and validates a closed complete logical schema. It preserves missing workspace
rows versus explicit Inherit/Disabled, missing review rows versus reviewed false,
selection, dependencies/blockers, package/source/link/adapter identity, mappings,
credential binding references/generations, revocations, tombstones and data roots.
It omits transient availability, inspection timestamps, process cleanup claims,
source caches and receipts. `sources.source_json` stays untrusted acquisition/
discovery metadata: later catalog/update consumers must add a closed authenticated
artifact/update-origin binding before treating it as reviewed provenance. Source
refresh/deletion cannot change an installed revision's authority.
The closed operation result is supplied by the
coordinator; `read_operation` returns only an untrusted phase hint plus a validated
intended result. Neither is proof of commitment.

Component definitions and variable declarations are authenticated digest references,
including malformed/unsupported definitions and their blockers. Reconstruction
must reinspect retained immutable package material under the exact authenticated
interpretation, then match package/component/variable digests and constraints.
Catalog overlays require their own retained authenticated material. Missing bytes,
changed material or unavailable adapters require recovery; never fetch, execute,
drop constraints or infer defaults. Mapping target/configuration references require
the owning service's equivalent digest/binding validation. Credential snapshots
contain stable reference IDs, authority generations and identity/audience/scope
bindings; current token values and expiry are resolved by that service at use time.

Publication ordering is `prepare(snapshot, old, new)` → durable owned registry
transaction returns → `certify_commit(old, new)` → `advance_marker(old, new)`.
Certificate issuance is coordinator-internal. Prepared data has no certificate;
a registry phase flag cannot authorize issuance. `verify_transition(operation_id)`
authenticates exact old/new tuples, snapshot and optional separate certificate.
An old marker plus prepared-only evidence cannot advance. `verify_snapshot(marker)`
allows reconstruction after registry loss, and `verify_current()` selects exactly
the marker-named snapshot. F4 implements actual crash/commit orchestration.

The recovery digest is SHA-256 over the domain-separated canonical plaintext
snapshot. AES-GCM binds the complete marker header; prepared and committed HMACs
use different purposes and keys. Operation artifact names hash their IDs. Files
are create-only; exact retries authenticate equality and resynchronize durability.
The private-file helper pins protected file identities, fsyncs file and parent,
and requires `verified_private`. This was exercised on local macOS/APFS; no
Windows, network/synchronized filesystem or power-loss guarantee is inferred.

`reset(operation_id=...)` requires an explicit caller-reviewed operation ID under
the same owner gate. F4/app retains that ID with the review. A new reviewed reset
uses a new ID; retries reuse the exact original ID. Reset clears only the plugin
marker, drops that namespace's session keys and archives encrypted evidence under
the protected trust parent (`plugins-reset-<scope>-<random>`).

A protected sibling `plugins-reset-state.json` binds its exact store scope and
retains up to 1,000 closed reset records: operation ID, safe archive leaf, original
directory device/inode, and pending/completed phase. A pristine reset records an
explicit completed no-archive outcome. Pending state is durable before any archive
rename; exact archive and parent synchronization precede completed publication.
A visible completed receipt is synchronized again on recovery/readiness checks,
since its final publication may itself have failed synchronization.

Pending or invalid receipts report Recovery required and block bootstrap/unlock.
Another ID cannot replace a pending reset. Same-ID retries qualify and return only
that ID's retained archive or no-archive outcome, even after later resets and a new
bootstrap; they do not clear the new marker or drop the new session's keys. At the
1,000-record cap, a new reset fails closed; no IDs or archives are evicted. These
cleanup receipts never authorize execution. Archive deletion/pruning has no API
in this foundation task.

Reset returns the archive path (or None for a recorded pristine reset). It does
not delete archives or standalone data. Failures retain available evidence and
raise; do not report success after a partial reset. Rebootstrap creates only
empty authority and never imports previously reviewed packages. Exact marker
advancement retries also republish through the configured marker backend, so
visible marker bytes cannot substitute for successful durable publication.

## Reviewed installs and recovery (F4)

Construct `PluginCoordinator(registry, authority, owner)` on the **same dedicated
non-UI worker** that creates, uses and closes its real SQLite registry. Install one
persistent asyncio event loop on that worker before construction. Call `review`,
`bootstrap`, `reset`, `published_snapshot`, and the async `commit`/`recover` methods
on that loop; wrong-thread/loop calls fail before storage mutation. App integration
owns worker startup and message marshalling. Do not send this live connection to
arbitrary `asyncio.to_thread` calls or construct a new loop for each request.

Explicit owner-gated `bootstrap(passphrase)` sets up empty authority. After normal
startup unlock, await `recover()` before reviewing a new install. The internal
store remains an owner-gated primitive; UI must not bypass the coordinator.
`reset(operation_id=...)` requires an explicit reviewed reset and its retained ID.
Reset invalidates pending reviews and fences publication; it does not establish
that existing package bytes or runtime users are trusted or stopped.

`review(inspection, selection=(...), workspace_id=...)` captures a fresh
installation identity, exact inspected bytes/source/link/interpretation,
components/dependencies, selected component IDs, explicit target workspace,
and the complete current authenticated state (including mapping/configuration
references and their absence). The frozen review has a session token and a
15-minute expiry. Changed package inputs, authority, selection or target invalidate
it. Embedded manifest overlays are verified with package bytes; unknown or missing
external overlay artifacts cannot be reconstructed. F4's public mutation installs
new local packages **disabled and untrusted**. Workspace targeting does not enable
the install. Explicit activation/trust, replacement drain, catalogs and their
additional review inputs are later lifecycle integrations.

Retain one operation ID when awaiting `commit(review, operation_id)`. The owned
package copy is reinspected and its files/directories synchronized before protected
preparation. Candidate rows exist only inside the guarded uncommitted SQLite
transaction while `authority_projection` builds the complete snapshot. The snapshot
and intent are durable **before that transaction exits**. The sequence is:

1. Reconcile previous transitions and revalidate the exact review.
2. Materialize immutable bytes and persist the complete protected snapshot/intent.
3. Return successfully from the real durable SQLite transaction.
4. Create the separate authenticated commit certificate.
5. Verify agreement and durably advance the exact marker tuple.
6. Publish the authenticated projection and acknowledge the operation.

`progress`, when set to a host callback, reports `materialized`, `prepared`,
`registry_committed`, `certified`, `marker_advanced`, and `published` milestones.
It contains no package bodies or secrets. A raised callback leaves the same
recoverable state as a lost response at that boundary. `published_snapshot()`
refuses fenced or mismatched state; a projection cannot enable itself by editing
registry flags. Publication is metadata eligibility, never a filesystem, tool,
network, credential or execution grant.

`recover()` returns closed `OperationReceipt` values: operation ID, phase,
commitment evidence status, and a non-secret recovery reason. Prepared-only work
with the old registry aborts. A new-looking registry without post-commit proof
requires reviewed recovery. A matching certificate with the old marker completes
its transition. A secure new marker and complete snapshot can reconstruct lost
SQLite state independently of registry phase. Ambiguous transitions, malformed
journals or missing/mismatched retained bytes stay fenced. Recovery performs no
fetch, package execution, credential-grant synthesis or process signalling.

Reconstruction reinspects retained material under its authenticated dialect and
adapter, verifying exact content, definitions, variable digests, dependencies,
selection and blockers. It restores the complete logical snapshot, including
unrelated installations, root fences and tombstones. Original acquisition links
are copied files: source/link provenance comes from authenticated references and
does not require the original acquisition directory. Missing owning-service
verification for configuration/credential mappings blocks reconstruction instead
of weakening those references.

Registry reconstruction preserves every surviving process row and records an
idempotent **unresolved unknown-runtime-user** row for each affected installation,
with no invented PID. F2 `reserve_launch` blocks these rows even after another
restart. Restoring authority never proves earlier processes stopped; host
reconciliation is still required. F8 adds exact durable root-user relationships.

`PluginAuthorityStore.list_transitions(limit=1..50, offset=...)` provides bounded
public authenticated discovery without relying on a surviving registry. Consume
all pages before reconciliation. The inventory accepts at most **1,000 retained
transitions**; capacity refuses new preparation before creating a new snapshot,
while exact existing-operation retries and recovery remain available. F4 never
prunes snapshots, intents or certificates, including the marker's current snapshot.
Lifecycle retention integrates this inventory bound with protected-current and
in-flight evidence later; it is not intended as a lifetime installation limit.

Qualification: real controlled owner death and fresh-process recovery on local
macOS/APFS, including missing/rolled-back SQLite before and after marker publication.
Tests isolate config/data/marker storage before imports and verify imported module
and effective profile provenance. This is not hardware power-loss, real OS-keychain,
Windows, Linux, synchronized-root or network-filesystem qualification.
