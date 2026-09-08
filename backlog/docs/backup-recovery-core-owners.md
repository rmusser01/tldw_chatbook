# Core local recovery qualification

TASK-31989 implements [ADR-126](../decisions/126-complete-local-backup-and-recovery.md)
for the current conversation, media, prompt, collection and ingestion databases.
This is an internal capture boundary. Complete backup, archive publication and
installation replacement remain unavailable until their later contracts qualify.

`DB.recovery_core.core_adapters()` returns frozen installed declarations without
configuration bootstrap, optional engines, store constructors, migration, or SQLite
opens. Explicit profile context and the existing canonical selectors honor custom
paths. Discovery does not infer absence from disabled features, open SQLite to find
assets, or mark unresolved required owners complete. The executor registers these
adapters through `owner_registry.register`; importing this module does not mutate
that registry or start a service.

## Schema and content

| Owner | Qualified current version | Recovery SQLite authority |
| --- | --- | --- |
| db.chachanotes.primary | 42 | recovery.core.chachanotes |
| db.media.primary | 6 | recovery.core.media |
| db.prompts.primary | 4 | recovery.core.prompts |
| db.library_collections | 1 | recovery.core.library_collections |
| db.library_ingest_jobs | 5 | recovery.core.library_ingest_jobs |

`recovery_core_schema.py` contains exact `sqlite_schema.sql` ordered by type/name,
including indexes, triggers, virtual FTS tables and their shadow tables. The catalog
was captured once from actual fresh installed domain constructors under Tests
isolation at base `3769c1d8f72038dd18f9c7fda1b496bdf5ef3bc7`; its header records
SQLite provenance. The generation query was `SELECT sql FROM sqlite_schema WHERE
sql IS NOT NULL ORDER BY type,name`. The named schema test compares the catalog
against fresh stores and installed version constants. Schema SQL is never normalized
or executed by these adapters. Historical versions and physical schema variants are
explicitly unsupported; `SchemaPolicy.migration_steps` is empty. Future schema
changes require renewed qualification, not a version-label edit.

Capture uses the registered SQLite backup seam, including committed WAL. It does
not copy sidecars or serialize selected domain records. Soft-deleted rows, primary
keys, relationships, scalar/extra message attachments, character images, reaction
images, flashcard bytes and full FTS state survive unchanged. Cancellation aborts
capture and leaves any interrupted output in private staging for executor disposal;
that output is never a successful capture result. Existing destinations are refused.

Core-owned binary assets are BLOBs; managed visual/persona locators are relative.
`relocate` validates the current candidate and preserves these stable identities.
Absolute note-sync/dictionary and ingest-source locators are owned by separate
adapters or external user input and remain inert; this method never rewrites or
activates them. The primary database declares dependencies on notes, dictionaries,
persona assets and other attachment owners. Later owners must reconcile their actual
referenced byte inventories and activation state. Study/quiz data already inside
ChaChaNotes share this physical owner; later qualification must not duplicate its
physical payload ownership.

`validate_dependencies(item, candidate, candidates)` accepts only the existing final
profile/owner-qualified dependency IDs declared by `item`. It validates actual local
collection/media/note/prompt/conversation links and local ingest media IDs against
qualified staged peer databases. Server-origin ingestion links retain their remote
identity and require no local media row. Unknown collection reference types remain
unsupported. Presence checks for later-owned assets are prerequisites only; their
owner validators still must verify every required byte and tombstone.

## Native capture authority

Ordinary SQLite/file admission still refuses inside maintenance. The executor must
hold the actual fixed `default_bootstrap_root()/admission` authority with every
selected source namespace **and `bootstrap.unbound`**, drain actual storage/startup
leases, and use the opaque session yielded by `Admission.maintenance(...)`:

```python
with authority.maintenance(namespaces_including_unbound, timeout) as session:
    with session.capture_scope(exact_sources, private_staging_directory):
        adapter.capture(item, destination, cancel)
```

Capture checks the native authority directory identity, intact local profile bindings,
held namespace membership and directional source ownership. An unrelated Admission
root over the same files cannot authorize capture. Staging must already exist with
private permissions and be disjoint from all registered source/control roots.
Sources are exact regular files; only read-only opens of their physical identities
are allowed. Writes are limited to regular nonaliased staging descendants. Ordinary
owners receive no exemption. The capability is scoped to its issuing PID/thread
and native session identity; copied, expired and cross-thread objects are refused.

Each actual capture SQLite connection is tracked and native-closed before scope
exit even if its Python reference escapes or a custom close method fails. A native
retirement failure conservatively retains the actual native lock stacks until
process exit and reports `capture_resources_not_retired`; no destructors release or
retry that unresolved authority. This failure is not qualified recovery completion.
No writable capture file-stream API is exposed in this slice.

Holding the unbound gate prevents an unknown/unbound process from opening an alias
of the selected source. Verified disjoint bound profiles retain their own namespaces.
The resource protocol still excludes arbitrary external/legacy writers. Task 9/10
must qualify owner/startup drain; this task does not release process holds on UI intent.

## Evidence boundary

Targeted tests construct all five real stores, capture retained committed WAL,
compare complete SQLite dumps and explicit FTS/BLOB/deletion evidence, exercise
cancellation, malformed schemas, profile-specific dependencies, escaped connections,
fixed/unbound authority and independent-process blocking. Source main/WAL bytes and
main mtime are preserved; reader-induced access-time and SQLite-managed shared-memory
coordination are not claimed immutable. No user configuration, credentials or data
are used. Only this host's existing qualified native admission backend is exercised.

Established fixed authority is opened with `Admission.open_existing`, which never
creates missing files/directories. `admission_authority` verifies the exact unbound
marker roots and registered physical identity under a shared registry lock. It
registers only when this caller positively created the first marker. Missing
namespace, lost control root, corrupt registry or replaced marker refuses without
repair. This avoids an idempotent-looking `register` taking an exclusive registry
lock on every ordinary owner open and blocking disjoint profiles during capture.
