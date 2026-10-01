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

Retain the issued `review.operation_id` when awaiting `commit(review, review.operation_id)`. The owned
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


## Native Console skills (F5)

The app lazily owns one `PluginService` beneath `config.get_user_data_dir()`.
Protected authority uses `default_plugin_authority_dir(default_local_skills_store_dir(profile_root))`,
inside the existing sensitive Skills trust subtree. Package/registry storage remains
under the separate `profile_root/plugins` directory.
Construction, Skills listing, and synchronous Console configuration capture do
not open plugin storage or keyring. Explicit async operations start one persistent
storage thread/event loop; coordinator, SQLite, protected authority, and runtime
owner are created, used, and closed on that thread. The existing connection-owner
context also retires newly opened native workspace lookup handles when this
worker ends, preserving previously borrowed and custom owners. Call `bootstrap(passphrase)`
for first setup, or `unlock(passphrase)` for an existing profile; unlock completes
recovery before publishing metadata. UI management is a separate increment.

Typed `review_install`, `review_trust`, and `review_activation` return expiring
`PluginReview` records; `commit(review, operation_id)` uses the same authenticated
durable pipeline. Installation alone is disabled and untrusted. Activation
reviews specify `workspace_id=None` for the global default or the actual captured
workspace ID and exact `inherit`, `enabled`, or `disabled` intent. Global defaults
cannot inherit. The Console sentinels `global` and `workspace-default` are not
valid new override targets; only an explicit `None` edits the global default.
Already authenticated historical sentinel records retain generation fencing.
Missing/archived named workspaces refuse use rather than falling
back to global. Explicit workspace disable wins over the global default.

Schema v3 (`003_installation_alias.sql`) persists the authenticated installation
alias. A first installation uses the package name; a collision adds the first
eight installation-ID characters, extending the checked suffix if needed. The
alias belongs to the installation and must remain unchanged through future
update/rename/removal operations. Human names are `alias:skill-id`, owned record
IDs are `plugin:installation-id:skill:skill-id`, and model tools use a separate
63-character `plugin_` plus hashed identity. Duplicate model names are excluded.
Old snapshots without aliases retain their exact canonical bytes/digests and gain
no alias, trust, or runtime grant. Standalone edits, deletion, overwrite imports,
export, scripts, and standalone trust paths refuse owned identities.

Native portable `SKILL.md` frontmatter remains strict. Chatbook behavior uses
string entries under `metadata`:

```yaml
metadata:
  context: "fork"                 # "inline" (default) or "fork"
  user_invocable: "true"          # exact "true" / "false"
  disable_model_invocation: "false"
  argument_hint: "[change description]"
allowed-tools: ""                  # explicitly no child tools
```

Malformed recognized metadata blocks execution. Unknown metadata is inert.
Manual-only skills remain explicitly invocable but are absent from model tool
catalogs. Empty allowed-tools never inherits the parent's tools. Nonempty tool restrictions require reviewed host references. Skill model overrides
remain explicitly unsupported; agent model mappings retain the actual parent
provider/endpoint floor. Package code runs only through selected owned hooks
and the existing command/MCP runtime owners.

Console freezes a metadata-only maximum at capture, with its actual workspace and
fresh pending turn identity. That cache is eligibility only. `admit` revalidates
current authenticated authority and exact retained bytes, intersects the frozen
maximum, and restores metadata from authenticated definitions. New enablements
wait for a new turn. Current checks also run before injection, invocation, reads,
launch/dispatch, and approval acceptance. Unrelated scope changes do not revoke
an unchanged admitted scope. Package bodies/arguments enter attributed untrusted
user context as whole blocks, limited to 8 KiB each and 32 KiB per send. Literal
arguments are not template-expanded. Final agent sends account for all surviving
host-attributed instruction and file-result blocks. Live typed text carries only
content-free attribution beside its text; provider JSON and durable projections
exclude that sidecar. Marker-looking user or standalone text never creates it.
History pruning removes plugin blocks whole, and removed blocks do not consume
later sends' budgets. Console's existing skill spawn runner,
permission review, child tool ceilings, budgets, and lineage remain authoritative.

`bind_run` records exact actual root/child IDs, optional fleet handle, parent ID,
originating pending/turn/workspace/revision, lease, cancellation callback and
terminal event before host effects. A pending identity cannot widen, change its
turn, or bind a second unrelated root. `LivePluginFences.live_lock` protects only
short live admission/record operations; storage and lease cleanup stay outside
it. `fences.seal` and `live_runs()` do not wait on the storage loop. Sealing alone
does not implement cancellation: F6 consumes these exact callbacks and owners.
A rejected pre-spawn reservation is settled because the host has not launched.
Already bound work remains retained until actual `complete_run` terminal evidence;
returning/cancelling a caller or retiring pending custody does not settle children.
Direct-provider custody also retains the existing gateway worker futures through
`ConsoleProviderStreamSignals.provider_work_callback`; consumer exit settles only
after every retained worker/closer reports actual terminal completion.
Permanent app shutdown drains Console and then closes the plugin service;
navigation keeps its owner alive. Update/revision drain and retention use that owner;
M4 supplies plugin MCP ownership. These later paths must preserve these identities
and current-authority checks.

## Scoped stop and uninstall (F6)

Use app-owned `PluginService.disable(target)` or `uninstall(installation_id)` for
one awaited action. For caller cancellation/retry, retain the synchronous handle
from `begin_disable(target)` / `begin_uninstall(installation_id)` and await
`finish_revocation(request)`. `RevocationTarget` is frozen:

- `RevocationTarget(id, workspace_id, False)` disables one named workspace.
- `RevocationTarget(id, None, False, global_default=True)` changes the explicit
  default and stops its captured inheritors; explicit workspace overrides remain.
- `RevocationTarget(id, None, True)` disables every scope, setting the default
  false and existing overrides disabled. Uninstall always covers every scope.

Missing/contradictory modes and reserved workspace names are rejected. Reviewed
activation commits that disable, including Enabled → Inherit under a false
default, use the same immediate path and retain their original reviewed intent.

The facade seals live admission and transfers exact host cancellation before its
first storage-worker await. Pending admissions, context checks, result acceptance
and future hook/checkpoint consumers use the same snapshot fence. Call
`check_entries_live(entries)` for immediate refusal before asynchronous checks;
a pass is **not** authorization and must still be followed by the normal current
checks. Independently configured handlers have no plugin admission carrier and
retain their existing authority. Namespace marker changes alone do not cancel
unrelated scopes. Explicit reconciled enable creates fresh live generations;
old tokens never revive. Failed/pending disable scopes remain fenced until their
persistence is reconciled, including an explicit fresh request after an aborted ID.

`OperationReceipt` separates `committed` and `phase` from `runtime_stopped`,
`cleanup_pending`, and content-free error types. `complete` means publication
finished; `recovery_required` with `committed=True` can mean SQLite committed but
its certificate/publication still needs recovery. `session_only` means disable
was not saved and gives **no restart guarantee**. Runtime stop uses actual host
completion events and retained process evidence; a cancelled caller/Future never
proves completion. An unobserved process inventory cannot prove a clean stop.
`revocation_status(request)` observes retained outcomes without storage IO.
`RevocationFailure` retains the original exception in `original_error`/`__cause__`
for the owning adapter; public diagnostics must use the receipt's closed fields.

Caller cancellation leaves retained persistence/cleanup running. Retry the exact
request handle; it retains at most one original reviewed issued mutation identity.
Conflicting reuse raises `ValueError`. A prepared-only
abort retains its aborted result without replay; reconcile, then make an explicit
fresh request with a fresh ID. A committed registry without a certificate remains
recovery-required. Across restart, `lookup_operation(identity)` looks up either the issued operation
ID or retained request-handle nonce. It never starts a mutation. An unavailable or
expired identity requires an explicit new action, not inferred replay permission.

Uninstall commits the tombstone and removes only installation-owned trust,
selection, mappings and registrations before package-file cleanup. Saved data and
process evidence survive. Unknown/idle surviving owners keep cleanup pending;
retry the same uninstall after confirmed drain to remove retained package files.
Independent credentials and connection owners remain with their existing
services. M4 supplies actual plugin MCP request/connection ownership; hook-v2
consumers use this admission fence without a parallel runtime.

A pending request also captures its live scope versions. A newer disable in that
scope (or everywhere) supersedes the old unfinished request. Retrying it after a
fresh enable raises `RevocationConflict` before worker access; it cannot mint a
new disable review or cancel fresh work. Completed IDs remain idempotent.
`runtime_stopped=None` means the worker has not yet observed enough process
ownership evidence; UI must show unknown/cleanup pending, never confirmed stop.
An explicit Inherit review under a true default can resume fresh admissions after
reconciliation, with the same permanent refusal of old tokens.


## Reviewed revisions, retention and archived continuation

`review_revision(installation_id, source_root)` captures the exact current baseline,
new package bytes and existing selection intersection. Added components remain
unselected. Review reserves an inactive ticket; `apply_revision(review,
review.operation_id)` synchronously activates only that exact ticket before worker
waiting. Existing admitted work completes; new admission and late children refuse.
`revision_drain.blockers(token)` reports actual run/handle/workspace/lease custody.
Cancel releases only its proposal fence. `cancel_work(token)` explicitly requests
cancellation and still waits for actual terminal evidence. Cancelled Apply waiters
leave the existing worker task owned. `review_rollback(installation_id, digest)`
creates a fresh review under current selections/policy; it restores package bytes,
not old permission. Mutable-data compatibility remains unknown.

Initial packages keep `packages/<installation>` identity; replacement roots use
`packages-revisions/<installation>/<digest>`. Uninstall owns both layouts.
After update, internal retention keeps current plus at most two inactive revisions
unless protected, expires eligible inactive material after 30 days, and enforces a
2 GiB total package/cache/staging quota plus 100 MiB free reserve before allocation.
`retain_revisions(installation_id)` explicitly reconciles cleanup or maintenance.
A distinct host-issued `retain` operation commits reference compaction first.
Its bounded authenticated result freezes exact directory/root/anchor identities;
restart cleanup refuses replacements and preserves pending evidence. File failure
reports `cleanup_pending`; commitment never means file cleanup succeeded.

Terminal transition/result history keeps a contiguous authenticated suffix, bounded
at 1,000 entries/30 days when eligible. Current and unresolved evidence remains
protected; all-protected capacity refuses a new mutation. A fixed v2 trust-metadata
cutover authenticates the finite legacy-ID map before pruning; old snapshots and
salt remain unchanged. Issued identities authenticate the expected generation and
review/result, not permission or proof of commitment. Newer uncertified SQLite
hints without their protected intent still require recovery; expired old hints may
be reconciled after restoring an old SQLite backup.

Managed durable continuations use the existing private field's closed V2 envelope.
The Console captures actual admitted capabilities at the real run binding and the
store finalizer signs every resolved checkpoint body with its exact durable owner.
Resume verifies namespace, body, owner, revision/definitions, scoped authority and
qualified data coverage before context assembly and again at admission. V1 receives
an explicit zero-managed ceiling. Valid foreign/remapped V2 remains private history
but cannot resume as local managed work. No import/fork re-signs it.

Fleet pins remain host-held inside the existing bounded retained transcript.
`send_to_agent` verifies before definition lookup/slot consumption, intersects the
current parent ceiling and repeats validation at actual child binding. Missing
required pins refuse retention/resume; pin bytes count toward transcript limits.
These pins never restore old approvals and never survive fleet restart.

The staging cleanup adapter accepts explicit reconciled terminal producer custody
older than 24 hours and an exact physical directory identity. Unknown, active,
recovery-referenced or replaced staging stays protected and counts toward quota.
I3/I4 must qualify their future acquisition/cache producers against this seam.
F8 must qualify root-user joins/attachment/generation changes; package-only native
skills have qualified-no-data coverage, while unqualified data users refuse exact
resume. No acquisition, hook or MCP execution owner is created here.

## Saved data: exact roots and host ownership (F8)

The host creates roots through `review_data_creation` / `create_data` and receives
an immutable `DataRootRef` (original installation, root ID, generation, absolute
path). A same-name reinstall receives no old data. Uninstall retains the original
root owner; explicit `review_data_attachment` followed by `delete_data` performs
a reviewed, drained attachment/detachment and advances the root generation.
The original owner and physical path never change.

Before giving any process or operation access, host adapters call
`reserve_data_user(..., roots=..., cancel=...)`; reserve includes every root and
publishes/readbacks the session's dirty checkpoint before returning. Publish the
exact process provenance through `publish_data_user`. Later grants use
`RootUsage.acquire`; release a grant only after its actual handle is closed.
`settle_data_user(token, confirmed=True)` means the owned process exited and was
reaped, or the whole retained operation actually ended. Failed stop/unknown exit
must use `confirmed=False`. Idle connections, readers, pending launches, old
revisions and all workspaces sharing a root remain blockers. An empty revision
lease count or missing join is never terminal proof. Package-only host work must
explicitly reserve `root_coverage="qualified_none"`; default/legacy is unknown.

`review_data_deletion` captures exact roots. `delete_data` immediately fences the
reviewed roots, then commits waiting and deleting phases with distinct issued
operation IDs before unlinking. It waits for actual root users, and an abandoned
waiter does not cancel its retained operation. `cancel_data_work` requests stop
only from registered host callbacks for those roots; callback return alone does
not settle a process. `cancel_data_deletion` cancels the proposal before deleting
and clears only its own fence after safe authority reconciliation. After deleting
begins, cancellation/IO failure retains the authenticated pending group and
fence. Read-only receipt lookup never performs cleanup. Resume requires
`review_data_cleanup_resume` and the original exact authenticated group/action.
A pending attachment remains an attachment after restart. Every phase keeps the
original review deadline; expiry leaves pending recovery for a fresh explicit
review, including when the leaf has already been removed.

The separate bounded runtime checkpoint uses the existing selected marker
backend. Graceful `aclose` seals ordinary service admission and rejects queued
ordinary callbacks on the worker. Already running calls, users and retained
cleanup callbacks must finish before clean is written/read back against the
final current authority marker. A refused shutdown still accepts exact
`settle_data_user` / `complete_run` terminal evidence, while new authority work
remains refused. A package/activation commit before shutdown is included in that
final binding. Dirty, missing, mismatched or malformed checkpoint evidence fences
all retained roots, including orphans. SQLite restoration cannot establish that
an omitted writer stopped. `review_data_reconciliation(...,
confirm_quiescence=host_proof)` / `reconcile_data` requires explicit whole-root
host proof, repeated at application; a user assertion or empty SQL is not proof.
Authenticated `cleaned_absent` roots remain tombstones: reconciliation confirms
the expected absence through exact current ancestry and preserves the historical
leaf binding. An unexpected leaf, replaced/missing ancestry or a missing
`present` root refuses.

Native root binding is qualified for same-boot local APFS on Darwin: no-follow
owner/data anchors and leaf descriptors bind exact device/inode/birth seconds and
nanoseconds plus the native boot session UUID. A changed, missing or malformed
boot identity refuses access/reuse/deletion. After a system reboot, retained data
requires reviewed host quiescence reconciliation and rebinding/generation advance;
pending destructive groups cannot silently rebind. Package-only use remains
available. A boot UUID does not prove inode non-reuse, stop writers or contain
arbitrary external code. Windows, network filesystems and arbitrary external
writers are unqualified. File-marker mode retains its explicit reduced coherent
rollback protection; it does not become secure-marker protection.

Adapter handoffs: **H2** hook launches and **M4/H6** MCP/runtime owners must reserve
all roots before launch/handle delivery, retain joins while idle, and settle only
actual terminal ownership; use the shared host callback seam for reviewed stop.
**I6** must display waiting/deleting/recovery-required outcomes and specific
`root_*` recovery reasons, distinguish Cancel operation from Cancel work, and
route explicit host-proof reconciliation/cleanup resume through these APIs.
**I7** must use service shutdown's admission-close/drain/final-clean order and
preserve a failed shutdown's dirty evidence. None of these handoffs adds a second
process owner, force-delete path, or plugin-controlled authority.


MCP hook invocation (H6) uses the existing normal tool owner, exact live
schema/profile/persona/permission checks and already-connected eligible
sessions. Only qualified original-wire typed results can create hook effects.
Required nested postevents settle before result authority is rechecked; validation
never replays the tool or recreates an approval. Initialization uses a private
prospective Console view and requires independently qualified dependency
readiness. Unknown managed graph declarations remain unavailable until native
registration supplies them. Teardown cannot connect, prompt or create context;
Interrupt and SessionEnd keep their original one/three-second deadlines.
Cancellation retains the original request/validator and source/root custody
until actual terminal evidence, even after the visible waiter returns.


## Selected native capabilities (I1)

`list_components(workspace_id)` returns metadata for selected, excluded and
unavailable components. `capture_maximum`/`admit` freeze the same dependency-complete
selection for skills, commands, rules, agent presets, hooks and MCP. Missing
requirements refuse their dependent component without blocking unrelated material;
new selection or enablement takes effect on a later run.

Manual commands and rules use `alias:command:id` and `alias:rule:id`. Commands
accept a strict JSON object containing exactly their declared named string arguments.
Arguments remain separate literal user context; nested mentions are never expanded.
Always rules have stable installation/component ordering. Complete instruction
blocks remain attributed untrusted user context, with the existing 8/32 KiB limits.

`review_configuration` also accepts `tool_references={component_id: {label:
"builtin:calculator"}}` and `models={component_id: "provider::model"}`. Owned MCP
references use an already reviewed exact `local:profile::tool` mapping and require
the MCP component in that material's dependency closure. Commit remains the existing
protected review owner; these references grant no tool permission. Native skill
constraints narrow the actual inline/fork catalog. Agent tools inherit only when
explicitly declared `inherit`; an empty list remains empty, including through the
actual host child, owned MCP invocation and approval paths. Declared models must
exist in the host catalog and match the actual parent's provider route.

Owned hook definitions use the same immutable snapshot and scheduler. `${PLUGIN_ROOT}`
resolves to retained bytes, `${PLUGIN_DATA}` to a bound shared data root, and declared
nonsecret defaults expand once. Unresolved values remain unavailable. F2 reserves
actual command/root custody before spawning; revocation retains it until the process
and pipes settle. Hook dependencies are checked for the actual tool definition and
live attributed material, so an unrelated ordinary tool does not inherit another
component's failed initialization requirement. Owned MCP composition delegates to the
existing connection, permission, discovery, source and result-currentness owners.
