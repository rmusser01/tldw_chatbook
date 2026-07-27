# Local Privacy Containment Design

Date: 2026-07-23
Status: User-approved design; verified review corrections applied
ADR: [ADR-029](../../../backlog/decisions/029-local-private-data-boundary.md)
Backlog:
[TASK-943](../../../backlog/tasks/task-943%20-%20Establish-private-path-boundary-and-harden-config-bootstrap.md),
[TASK-489](../../../backlog/tasks/task-489%20-%20Apply-private-storage-boundary-to-every-SQLite-owner-and-backup.md),
[TASK-490](../../../backlog/tasks/task-490%20-%20Harden-persistent-log-and-tool-cache-file-lifecycles.md),
[TASK-491](../../../backlog/tasks/task-491%20-%20Make-config-persistence-use-one-effective-path-and-live-runtime-boundary.md),
[TASK-492](../../../backlog/tasks/task-492%20-%20Remove-private-payloads-from-persistent-diagnostics-and-tool-history.md),
[TASK-493](../../../backlog/tasks/task-493%20-%20Contain-legacy-Notes-sync-paths-and-preserve-file-modes.md),
[TASK-494](../../../backlog/tasks/task-494%20-%20Complete-metadata-only-boundary-across-remaining-production-diagnostics.md)

## Summary

Chatbook will enforce one explicit boundary for the verified local privacy
exposures before repairing deletion/migration, evaluation/worker, packaging,
or application-state findings.

The tranche will:

- preserve lexical path selection long enough to reject links and untrusted
  parent namespaces;
- create and harden the effective config, every Chatbook-owned file-backed
  SQLite database and backup, rotating application logs, MCP execution logs,
  and the optional tool-result cache;
- make the config module the only `config.toml` persistence owner without
  violating existing restart-only or Console-session precedence contracts;
- remove private payload values from persistent diagnostics and bounded tool
  history;
- pin legacy Notes traversal beneath one canonical root, reject path aliases
  that can reach outside content, and preserve existing modes.

Existing eligible files are automatically hardened on POSIX. New private
artifacts are created without relying on the process umask. Windows and
platform-specific ACL posture are reported honestly rather than inferred from
POSIX mode calls.

## Verified Problem

The audit and design review reproduced these outcomes:

- first-run `config.toml` and new SQLite databases at `0644`;
- full prompts in INFO logs;
- tool arguments and results in logs and unbounded process history;
- provider and summarization request/response content and API-key fragments in
  logs;
- MCP execution arguments and result excerpts written to a `0644` JSONL log;
- the optional `tool_results.cache` containing full results at `0644` and
  loading attacker-replaceable bytes with `pickle.load`;
- encryption reporting success after modifying an unrelated default config;
- providers retaining stale imported settings after save;
- decrypted configuration exported to a plaintext backup through a direct UI
  writer;
- absolute or traversing `log_filename` values escaping the application data
  directory;
- a Notes-root symlink importing outside-root content;
- a Notes-root hardlink importing the contents of an outside file;
- an existing private note changing from `0600` to `0644` after sync.

The review also found 31 direct `sqlite3.connect` call sites across 18
production modules. A sampled database migration cannot establish an
application-wide privacy invariant.

## Goals

- Automatically contain eligible existing private artifacts on POSIX.
- Create private artifacts without a group/world-readable interval.
- Preserve lexical path evidence and reject attacker-writable path namespaces.
- Refuse unsafe config/database operations rather than silently continuing.
- Cover every production SQLite owner and backup, not a sample.
- Disable only unsafe persistent file sinks while retaining terminal/UI logs.
- Eliminate executable deserialization from the optional tool-result cache.
- Make the effective config path the only `config.toml` persistence target.
- Ensure later provider/security reads observe a successful save.
- Preserve restart-only storage and Console session-precedence contracts.
- Keep persistent diagnostics useful without retaining private payload values.
- Bound tool execution history without changing immediate tool results.
- Keep legacy Notes operations beneath a pinned canonical root.
- Preserve user-selected modes on existing Notes and create new synchronized
  Notes as `0600` on POSIX.
- Report mode/ACL enforcement honestly across platforms.

## Non-Goals

- Keyring migration, new encrypted credential storage, or secret rotation.
- A raw-payload support bundle or persistent diagnostic mode.
- Recursive permission changes outside Chatbook-owned directories.
- A general-purpose repository secret scanner.
- A portable Windows ACL implementation.
- Cross-process config lost-update protection.
- Live storage relocation or reconnection; ADR-004 keeps those settings
  restart-bound.
- Changing Console effective-session precedence from ADR-006.
- The file-backed Notes projection, mutation journal, recovery database, or
  authority changes described by ADR-021.
- Data deletion/vector retention, schema migration, eval, worker, packaging,
  or application-state work; those remain later tranches.

## Terminology

| Term | Meaning |
| --- | --- |
| Lexical selected path | The absolute, normalized spelling selected by defaults or the user before symlink resolution. It is retained for diagnostics and link detection. |
| Canonical identity | The verified resolved object or pinned directory descriptor used for identity/containment after lexical safety checks. |
| Trusted namespace | A parent chain in which no unauthorized local user can replace or pre-create the next path component. POSIX sticky-directory ownership is evaluated explicitly rather than treating every writable ancestor as equivalent; a shared sticky parent may protect an existing owned leaf but is insufficient for creating a missing selected leaf. |
| Private file | The effective config, credential-bearing config backup, application database/sidecar/backup, rotating application or MCP log, versioned tool cache, or another artifact explicitly added to the checked inventory. |
| Application-owned directory | A Chatbook-specific default config, data, cache, or log directory, not an arbitrary parent selected by a user. |
| Effective config path | The lexical target selected by `TLDW_CONFIG_PATH` before the default, paired with a separately verified canonical identity. |
| POSIX-mode private | A verified current-user-owned regular file with mode `0600`, or application-owned directory with mode `0700`. This does not claim that an uninspected platform-specific ACL is private. |
| Metadata-only diagnostic | A record containing operation identity and measurements but no prompt, message, request/response body, credential, tool-argument value, or tool-result value. |
| Lexical Notes root alias | The original selected root spelling used to find compatible existing sync metadata. |
| Pinned Notes root | The opened canonical root identity used for every containment operation during one legacy sync pass. |

## Architecture

### Dependency-leaf private-path boundary

A small stdlib-only utility owns private artifact selection, inspection,
creation, and POSIX mode hardening. It imports neither configuration nor
logging code; callers choose policy and emit diagnostics from its structured
result.

The result distinguishes:

- `created_private`;
- `hardened_private`;
- `already_private`;
- `unsafe_parent`;
- `wrong_owner`;
- `link_or_non_regular`;
- `operation_failed`;
- `unverified_platform`.

Path selection does not call `Path.resolve()` before security checks. The
utility keeps the lexical absolute path and separately establishes canonical
identity.

On POSIX:

1. every traversed component is inspected without following links;
2. the namespace is rejected when another local user could replace a
   component—or pre-create a missing selected leaf—under the defined
   ownership, write-bit, and sticky-bit model;
3. the final object is opened without following a link and verified as the
   expected type and current-effective-user-owned before `fchmod`;
4. supported atomic replacements create the temporary file relative to a
   verified parent descriptor and replace relative to that same descriptor;
5. a postcondition verifies identity, type, ownership, and mode.

Application-owned private directories use `0700`; private files use `0600`.
For a custom config or database path, Chatbook does not chmod arbitrary
parents. The selected path is accepted only if its namespace is already
trusted.

Caller failure policy is explicit:

- unsafe config reads/writes fail closed;
- unsafe file-backed database opens/backups fail closed;
- an unsafe application or MCP file-log target disables only that file sink;
- eligible historical files are hardened, while ineligible ones remain
  untouched and produce a bounded redacted diagnostic.

On Windows the utility returns `unverified_platform`. Normal platform APIs may
continue where the caller's contract permits, but Settings must not label the
result owner-only or ACL-secure.

### Complete SQLite owner and backup lifecycle

TASK-489 begins with a checked inventory of every production SQLite connection
and backup entry point. Each is classified as:

- private file-backed;
- in-memory;
- supported URI/read-only;
- explicitly excluded with a reason.

No file-backed owner may remain in a “sampled” category. A source or registry
guard requires newly added production owners to choose a classification and
the approved private connection seam.

For a new private database, Chatbook validates the namespace and exclusively
creates the file as `0600` before the first `sqlite3.connect`. Existing
eligible databases are hardened before connection. Because Python's SQLite API
cannot portably connect through an already opened file descriptor, custom
database paths require a namespace that unauthorized users cannot rename or
replace during pathname-based connect. In-memory and supported URI/read-only
connections retain their existing semantics.

Real application journal modes are tested. WAL, SHM, and rollback journal
files must remain private on supported POSIX platforms. Default databases live
inside `0700` application directories, and custom database parents must
already meet the trusted-namespace precondition. Chatbook-created backups use
the same private creation and connection path.

### Persistent application, MCP-log, and tool-cache lifecycle

`logging.log_filename` is a filename, not a path. Absolute values, separators,
`.`/`..`, and traversal are invalid. The resulting path must remain directly
beneath the canonical application log directory.

The rotating application handler creates each active generation through a
private `_open` implementation. Startup hardens eligible active and rotated
generations. The application log directory is `0700`, and each generation is
`0600`.

`MCPExecutionLog` uses the same private parent, create, append, rotation, and
existing-generation posture. Line counting and reads of both active and
rotated generations are anchored to the verified parent and do not follow
links. An unsafe generation cannot be counted for append/rotation and disables
that file sink; an unsafe generation encountered by `read_recent` is skipped
with a bounded redacted diagnostic rather than followed. Its payload contract
is tightened separately by TASK-492.

The optional tool-result cache no longer uses pickle. It writes a versioned,
strictly validated, size-bounded JSON representation atomically as `0600`.
Results that cannot be represented without changing their cache-hit contract
remain in the existing bounded in-memory cache and are not persisted.

Eligible legacy `tool_results.cache` files are automatically hardened and then
left inert. They are never deserialized, migrated, or silently deleted.

If an application or MCP log cannot be secured, its file sink is not installed
or is disabled before the next write. Terminal, Rich, and in-app logging
continue.

### Exclusive config persistence and live-runtime boundary

The config module owns:

- first creation;
- batched setting save and deletion;
- encryption enable, disable, and password change;
- shutdown encryption persistence;
- reset/default creation;
- raw-TOML display, replacement, and recovery;
- config backup/export.

The unrelated `~/.tldw_cli_config.toml` fallback is removed. App and UI modules
call config APIs rather than opening either the default or effective target.

Every operation:

1. selects one lexical effective path;
2. verifies its canonical identity and trusted namespace;
3. holds the in-process config serialization boundary across the complete
   read-modify-write and cache-generation commit;
4. uses descriptor-anchored atomic private replacement where supported;
5. verifies the postcondition before reporting success.

Config backups copy the serialized on-disk representation. When encryption is
enabled, export does not serialize the decrypted runtime mapping back to
plaintext. Backups are created privately.

The raw-TOML editor also reads the serialized on-disk representation. It never
renders a decrypted full-config mapping while encryption is enabled. Raw
replacement must validate and encrypt sensitive plaintext before commit or
fail closed; it cannot silently downgrade an encrypted configuration.
Disabling encryption remains an explicit operation through the normal config
API.

Successful writes publish a new generation-aware immutable or defensive
runtime snapshot. Provider request boundaries and security/credential views
resolve that current snapshot rather than importing `settings` or caching
request-sensitive credentials at module scope.

This does not make every application setting live:

- ADR-004 storage defaults remain next-launch values and do not reconnect or
  move active databases;
- ADR-006 Console session overrides and effective resolution continue to take
  precedence over persisted provider defaults;
- unrelated startup application state remains deferred to the later
  application-state decomposition.

A production-source guard rejects direct `config.toml` writes outside the
config owner, mutable `settings` imports, and module-scope snapshots of
request-sensitive credential/provider values. It does not reject documented
restart-bound constants merely because they originate in config.

Cross-process lost-update protection remains a separate task. Atomic
replacement prevents partial files but is not misrepresented as a portable
interprocess lock.

### Metadata-only diagnostics and bounded history

TASK-492 establishes a checked repository-wide inventory of every
Chatbook-owned production diagnostic that can reach the persistent application
or MCP sinks. Each owner is assigned to TASK-492, TASK-494, or a reviewed
non-persistent/excluded classification. The inventory and source guard prevent
new production owners or persistent-sink topology from bypassing
classification.

TASK-492 remediates the verified high-risk Chat, cloud/local provider,
summarization, ToolExecutor, and MCP owners. TASK-494 completes the same
metadata-only contract for every remaining application domain, including RAG
and search, ingestion, media/database, Notes/sync, subscription/web, and
UI/application orchestration.

Allowed metadata includes:

- provider, model, operation, streaming state, status code, duration, timeout,
  retry count, message count, and payload/result length;
- tool name, registered-schema argument names, status, duration, cache status,
  and result type/size.

Unknown tool argument keys are counted, not persisted, because provider-created
keys can themselves contain user content.

Excluded at every persistent diagnostic level, including DEBUG:

- prompts, messages, system prompts, request dictionaries, and response
  dictionaries/bodies;
- raw exception/HTTP body text;
- API keys or partial key fragments;
- tool argument values and tool result values.

The ToolExecutor public return contract is unchanged. Its history becomes a
100-record bounded collection containing only identity, timestamps, status,
duration, approved argument names, cache status, and result type/size.

MCP execution JSONL records become metadata-only too; existing secret-key
redaction is not treated as permission to retain non-secret private argument
or result values. Eligible legacy active and rotated generations are
atomically rewritten to the metadata-only schema on their next read or append;
torn and non-object rows are dropped during that migration.

The application file handler is an admission boundary, not a content-redaction
filter: Chatbook records persist only when emitted through the strict metadata
helper, and third-party records remain available to UI/terminal handlers but
are not admitted to disk. The unused legacy Metrics Loguru setup no longer
creates independent file sinks that bypass this boundary.

### Descriptor-anchored legacy Notes containment

The lexical selected root is retained as a lookup alias for existing
`sync_root_folder` records. The root is then resolved, verified, and opened
once; that pinned canonical identity governs the full sync pass.

On POSIX, scanning and mutation walk relative components from the pinned root
using directory descriptors and no-follow flags. Reads use the final verified
file descriptor. Writes create a private temporary file relative to the
verified parent descriptor and replace relative to that same descriptor.
Final-file and intermediate-parent replacement are separate tested failures.

Legacy sync rejects:

- descendant file or directory symlinks;
- junctions and reparse points;
- multiply linked regular files;
- cross-device nested mounts;
- non-regular files;
- any resolved or opened identity outside the pinned root.

This deliberately rejects in-root aliases too. A user may select the root
itself through a link because its resolved identity becomes the explicit
pinned root.

On Windows, an entry is skipped whenever the supported runtime cannot verify
reparse, containment, or replacement safety. Diagnostics identify the skipped
entry without claiming POSIX-equivalent guarantees.

Existing files retain their permission bits during replacement. New files use
`0600` on POSIX. The generic atomic-write helper's global default does not
change because it also serves non-private export paths.

Existing rows stored with the lexical root spelling are matched without
weakening containment. Their root metadata is normalized to the canonical
spelling only as part of a successful sync metadata update; a failed or
rejected pass does not orphan those records.

## Repository Credential Hygiene

The exact repository-root entries `/openai-api-key.txt` and
`/moonshot-api-key.txt` are added to `.gitignore`. A `git check-ignore`
regression test keeps the guard owned by TASK-943.

The files are not opened, deleted, moved, or claimed to contain valid
credentials. This is not a secret scanner and does not affect tracked content,
subdirectories, or differently named files.

## Error Handling and Diagnostics

- Security errors never include file contents, credentials, prompts, response
  bodies, exception response text, or tool payload values.
- Path diagnostics may include the selected path, expected posture, object
  type, platform support status, and remediation.
- A failed hardening attempt never deletes or replaces the target.
- Historical ineligible artifacts are retried on the next startup.
- Duplicate degraded-posture diagnostics are bounded by artifact category and
  path during one process lifetime.
- One rejected Notes entry does not stop unrelated safe entries.
- No success message is emitted before the operation and postcondition check
  complete.

## Testing Strategy

All production changes follow red-green-refactor TDD.

### Private-path and config-bootstrap tests

- fresh config creation is `0600` under a `0022` umask;
- existing eligible `0644` config becomes `0600`;
- application-owned directories become `0700`;
- lexical final/intermediate links are rejected before resolution;
- target and parent replacement attempts fail;
- custom attacker-writable parents fail closed;
- a missing target in a shared sticky parent fails closed;
- wrong-owner and non-regular simulations are unchanged;
- Windows and platform ACL posture are not overstated;
- only the two exact repository-root credential filenames are ignored.

### SQLite tests

- every production connection/backup owner appears in the checked inventory;
- each file-backed constructor creates/hardens privately;
- in-memory and URI/read-only semantics remain intact;
- WAL, SHM, and rollback-journal modes are private;
- backup targets use the same policy;
- target/parent replacement and unsafe custom parents fail closed.

### Persistent file-lifecycle tests

- absolute/traversing log filenames are rejected;
- active and rotated application/MCP logs remain private;
- MCP read/count/write paths reject unsafe parents, links, and replaced
  generations without following them;
- unsafe log targets disable only the affected file sink;
- the versioned cache is atomic, private, bounded, and schema-validated;
- unsupported cache results remain memory-only;
- corrupt JSON and legacy pickle cache files are never executed.

### Config ownership and runtime tests

- every mutation, shutdown, raw replacement/recovery, and export operation
  affects only `TLDW_CONFIG_PATH` when set;
- no unrelated fallback config is created;
- encrypted export remains encrypted at rest;
- the raw editor never displays a decrypted full-config mapping while
  encryption is enabled, and raw save cannot downgrade encryption;
- file and cache generation commit together under concurrent in-process use;
- the next provider/security read observes a successful save;
- storage paths remain restart-bound;
- Console session overrides retain precedence;
- source guards reject direct writes and request-sensitive module snapshots.

### Diagnostic and tool-history tests

The TASK-492 sentinel matrix covers its inventoried owners and:

- success;
- HTTP error;
- parsing error;
- timeout;
- streaming;
- cache hit/miss.

Captured standard/loguru logs, MCP JSONL, and the real rotating file must not
contain any sentinel. Metadata assertions prove diagnostics remain useful.
More than 100 tool calls prove history bounding while immediate return values
remain unchanged.

### Remaining diagnostic-domain tests

TASK-494 proves that every other inventoried production diagnostic is
metadata-only or has a reviewed non-persistent/excluded classification.
Parameterized sentinels cover RAG/search, ingestion, media/database,
Notes/sync, subscription/web, and UI/application orchestration through normal,
debug, and error paths and the real rotating file sink. The inventory/source
guard fails when a production logger owner or persistent sink is added without
classification.

### Notes tests

- selected-root symlink compatibility;
- outside-root and in-root descendant symlinks;
- descendant directory links and Windows reparse/junction simulation;
- outside-content hardlinks;
- nested-device simulation;
- final-file and intermediate-parent replacement;
- safe file alongside a rejected entry;
- lexical-root metadata compatibility and successful normalization;
- existing `0600`, `0640`, and `0644` preservation;
- new-file `0600` creation.

## Delivery Decomposition

1. TASK-943 establishes the dependency-leaf private-path boundary, hardens
   config bootstrap, and owns the exact repository ignore guard.
2. TASK-489 applies the boundary to the complete SQLite/backup inventory.
3. TASK-490 applies it to application/MCP logs and replaces executable tool
   cache persistence.
4. TASK-491 consolidates config persistence and its scoped live-runtime
   snapshot.
5. TASK-492 removes private payloads from persistent diagnostics and bounds
   tool history.
6. TASK-493 hardens legacy Notes traversal and mode preservation.
7. TASK-494 completes the metadata-only persistent-diagnostic boundary across
   every remaining production domain.

TASK-489, TASK-490, TASK-491, and TASK-493 depend on TASK-943. TASK-492 depends
on TASK-490, and TASK-494 depends on TASK-492. Each task is independently
reviewable and receives its own implementation plan, red-green tests,
verification, and review gate.

TASK-331 now retains only the separate built-in file-tool sandbox/governance
work; executable tool-cache persistence belongs exclusively to TASK-490.

All seven privacy tasks complete before the deletion/migration tranche begins.

## ADR Check

ADR required: yes

ADR path: `backlog/decisions/029-local-private-data-boundary.md`

Reason: ADR-029 already establishes the controlling privacy, filesystem,
configuration, logging, and legacy Notes decisions. These review corrections
make the implementation design satisfy that accepted policy; they do not
change it, so no replacement ADR is required.

Related decisions:

- [ADR-004: Settings Storage Defaults Restart Boundary](../../../backlog/decisions/004-settings-storage-defaults-restart-boundary.md)
- [ADR-006: Provider-Aware Generation Settings](../../../backlog/decisions/006-provider-aware-generation-settings.md)
- [ADR-012: Provider Credential Settings Boundary](../../../backlog/decisions/012-provider-credential-settings-boundary.md)
- [ADR-021: File-Backed Notes Disk Authority and Recovery Replica](../../../backlog/decisions/021-file-backed-notes-disk-authority-and-recovery.md)
