# ADR-240: Workspace project context and folder bound indexing

Status: Accepted (user approved the design on 2026-10-20)
Date: 2026-10-20
Design: [Workspace aware project context](../../Docs/superpowers/specs/2026-10-20-workspace-aware-project-context-design.md)
Related guidance tasks: [TASK-35930](<../tasks/task-35930 - Guide-project-questions-toward-admitted-workspace-files.md>),
[TASK-35931](<../tasks/task-35931 - Allow-mixed-Console-grounding-without-forced-Library-attribution.md>),
[TASK-35932](<../tasks/task-35932 - Qualify-workspace-guidance-with-preserved-Library-choices.md>).
Related keyword tasks: [TASK-36002](<../tasks/task-36002 - Store-folder-bound-keyword-cache-and-durable-authority-contracts.md>),
[TASK-36003](<../tasks/task-36003 - Scan-project-folders-in-bounded-resumable-pages.md>),
[TASK-36004](<../tasks/task-36004 - Own-keyword-index-jobs-refresh-recovery-and-revocation.md>),
[TASK-36005](<../tasks/task-36005 - Search-keyword-indexes-through-admitted-read-only-file-tools.md>),
[TASK-36006](<../tasks/task-36006 - Manage-optional-keyword-indexes-in-Settings-and-qualify-release.md>).
Keyword indexing is in implementation; semantic indexing remains a separate planned deliverable.
Related decisions: ADR-030 derived-index lifecycle, ADR-079 Console Library
authority, ADR-082 private scratch, ADR-101 pinned workers, ADR-102 run-admitted
roots, ADR-126 recovery, ADR-174 binding exclusions

## Context

Attaching a project folder grants file-tool access without staging its contents.
A natural project question can consequently receive an answer based on unrelated
Library excerpts when the model does not explore the folder. Existing workspace
notes explain root addressing but do not specify project exploration behavior.
One Console evidence formatter also supplies Library-only grounding instructions
that can compete with using file-tool evidence.

The user selected workspace-aware guidance while explicitly preserving saved
Library choices. They also requested optional indexing for thousands or 10k+
folders/items, choosing a folder-bound cache rather than retained Library copies.
This requires an ownership and revocation boundary distinct from Library ingest.

## Decision

### Workspace questions use capability aware guidance

Extend the existing model-facing workspace note after the final advertised-tool
filter. For project questions, guide the model to inspect relevant admitted
folders, README/manifests, and source as needed. Keep multiple-root selection and
scratch addressing explicit. The note grants no access and is never constructed
from the process-global active workspace in place of the owning run's authority.

Console grounding permits both Library references and workspace evidence with
distinct provenance. Dedicated Library answer generation retains its supplied-
evidence-only contract. An irrelevant Library bundle does not establish that
workspace files are unavailable.
Canonical Console citation repair accepts a mixed answer without Library labels;
their absence must not manufacture attribution. Actual unknown/malformed labels
remain invalid, while generic local capture retains mandatory-marker defaults.

No Library policy migration or automatic toggle occurs. Saved Never/Automatic,
Blocked/Allowed, manually staged sources, and scope intersection retain ADR-079
semantics, even when Automatic supplies unrelated evidence to a project turn.

### The optional index belongs to a local folder binding

Indexing is an explicit per-binding opt-in, disabled by default. A device-local
project-index service and separately versioned SQLite store own its manifest,
FTS, jobs/checkpoints, generations, and cleanup. Files remain authoritative;
cached text and vectors are derived. Root identity, locator fingerprint, local
data authority, binding identity, configuration, and index epoch scope readiness.
The existing registry owns a durable monotonic authority revision/tombstone;
the cache owns a durable active-job claim and transactional manifest revision.
App-local mutation notifications are not a cross-process authority witness.
Enablement authorizes application indexing reads only within the live binding
and selected processing scope; it does not authorize assistant tool calls.

Project content is not persisted as Library Media/Notes, is absent from global
Library retrieval, and does not inherit Library access or item scopes. The cache
and its settings/consents are excluded from content sync/export. Rebuildable
cache data is excluded from ordinary backups and integrates with the existing
storage-admission, activation, maintenance, and borrower-retirement contracts.

Keyword FTS works without optional embedding dependencies. Semantic indexing is
a further opt-in using existing embedding/chunking/vector primitives with a
separate project namespace and lifecycle. Cloud embedding disclosure requires
index-specific destination consent; Library permission does not provide it.
Project-local model loading is offline-only, including ordinary profiles, and
adapter/factory reporting does not disclose credential-bearing endpoints.

### Discovery does not widen file authority

An optional read-only `fs_search_index` spec uses the existing local provider,
root aliases, permissions, review, budgets, and path-target preparation. It may
query only a compatible project generation for the selected run-admitted root.
Binding exclusions and authority filters apply before ranking and again before
disclosure. Validate each selected candidate against the live pinned file and
its content hash; return live excerpts rather than stale cache text. A denied
read, removed binding, or incompatible root cannot be bypassed through the cache.
The binding's effective `fs_read` permission is a narrowing floor for excerpt
delivery: Blocked refuses and Ask requires review even when index search is
otherwise Allowed.
A conditional Ask uses existing one-call review options and exact run/call-ID
provenance at dispatch. A saved search grant, a sibling call stamp or a forged
broader decision cannot satisfy that floor or persist a new standing grant.
Carry the requesting run's frozen/accumulated exclusion floor into keyword and
semantic selection; a live registry read alone cannot restore access removed for
that run. If exact bounded semantic filtering is unavailable, use keyword search.

Indexed text never becomes automatic project instructions. Search has the same
binding-root instruction scope as existing grep; exact follow-up file reads use
the current nested-instruction activation path. Library labels remain reserved
for their owning Library reference bundles.

### Builds are bounded jobs and revocation precedes cleanup

Incremental scanning uses bounded pages and queues with a durable frontier, not
a whole-tree list. A fixed-purpose contained scanner process lives for one job;
it reuses root-pinning/process-containment primitives and does not change the
one-shot agent file-worker protocol. The parent service owns transactional
publication and rechecks authority and epoch before each committed batch.
Fresh authority is admitted per bounded directory/read unit, not frozen for the
whole worker lifetime. A short registry-then-cache serialization fence covers
publication/disclosure admission against concurrent registry mutation, including
other processes; never hold it across worker/network work or cleanup.

Keyword entries are eligible only after their manifest/text/FTS batch commits.
Semantic generations read bounded pages of one checked keyword manifest revision
and publish through an atomic compatible ready pointer. Partial
coverage, skipped processing-policy files, budget exhaustion, and interrupted
jobs are reported truthfully. Refresh and restart Resume are explicit in the
first release; no watcher dependency or scan-on-every-send is introduced.

Disable, Clear, detach, retarget, root replacement, and data-authority changes
revoke query/publication eligibility before retryable physical cleanup. New
exclusions narrow access immediately at the next query/publication boundary.
An old job cannot resurrect cleared or revoked data. Detachment cannot retract
tool output already disclosed into a conversation or provider request.
Management actions compare expected scope/job state before effects, so a delayed
Enable cannot undo Clear and an old Clear cannot remove a later generation.

Explicit Clear reclaims validated v2 derived payload across the binding's old
root and authority generations after revoking current access. Keep durable
control epochs/claim tombstones and stable lifetime files. Inventory and acquire
native lifetimes deterministically outside registry fences; code-owned namespace
metadata is cleanup ownership only, never file authority. Fresh v2 caches enable
incremental auto-vacuum before table creation and compact in bounded in-place
steps, preserving DB identity. Migration never forces VACUUM or replaces the DB.
[SQLite auto-vacuum](https://www.sqlite.org/pragma.html#pragma_auto_vacuum) and
[incremental vacuum](https://www.sqlite.org/pragma.html#pragma_incremental_vacuum)
describe that storage mode; they do not prove the application's quota accounting.
Process smaller namespaces first and reserve whole-DB journal/record headroom
against all binding-owned bytes before cleanup. Insufficient space or unproven
legacy state remains Off with a truthful retryable outcome. No nonexistent
maintenance-disposal route is advertised. Tests must show a normally admitted
near-capacity generation can be cleared and indexing enabled again without
losing revocation history or exceeding the journal/temp-inclusive quota.


The first release indexes local UTF-8 project text. Remote/SSH indexing and
binary/media extraction require separate designs. Settings owns management;
Console shows status and a route to Settings. Library controls stay distinct.

### Runtime identity and native job lifetime (execution clarification)

Issue the cache's opaque data-authority identity outside registry fences while
holding its actual storage/activation admission. Combine the admitted profile,
data/store identity and verified installation/activation generation, then expose
a bounded memory reader for that pinned lifetime. Ordinary preference reloads do
not change it; profile changes and restore/rollback invalidate old readiness.
Unknown or lost admission refuses execution. Existing config/admission counters
and unverified catalog records are not authority identities.
Retain the actual execution pin per management/query operation and continuously
from writer dispatch through native DB owner-thread exit and scanner retirement.
An idle service keeps no forever lease that would block maintenance. Frozen idle
identity/status is eligibility data only; later operations reacquire and verify
the admitted selector/activation identity outside authority fences.
The frozen authority callback and service projections introduce no config, receipt
or cache-stat reads inside those fences. Existing mandatory per-store activation
revalidation remains required when entering the separately admitted cache
operation beneath the registry fence. It is not replaced with an unchecked
borrow or skipped to optimize the critical section; its established metadata
checks can extend lock duration, so no hard real-time fence bound is claimed.

A stable private native lifetime lease per binding spans writer dispatch, native
DB closure, writer-thread exit and scanner process retirement. The scanner
inherits the lease capability at process creation, before its containment/root
handshake. POSIX inherited flock ownership and Windows inherited share-denying
file-object handles permit a fresh process to prove prior holders retired by
exclusive reacquisition; byte-lock inheritance is not assumed. Retained process
objects, PIDs, elapsed time and a child-main-only lock cannot prove this after a
parent crash. The fixed-purpose child creates no descendants; any future child
must retain the same lifetime or recovery must refuse proof.

Prepare exact-claim retirement evidence outside all registry/cache locks and
retain that native acquisition through claim replacement. Clear retains the
stable lease/control object, which remains declared, accounted, device-local
and excluded from ordinary backup. This single-writer/lifetime witness grants no
file or activation permission. Existing authority and admission gates still
apply. Platform inheritance and hard-parent-death tests qualify the native seam.

Production cache namespaces use
`project_index/<opaque binding hash>/<complete-key namespace>/workspace_project_index.db`.
The stable accounting hash includes workspace and binding IDs under the admitted
private data root, so prior authority generations remain in the same bounded
accounting group. Each authority has its own stable
`<sha256(data_authority_id)>.lifetime.lock` under that parent; these are stable
across locator/root changes and retained by Clear. Complete-key generation
namespaces still include the data authority. Counting prior-generation bytes
never grants authority or compatible readiness. Actual storage/activation pins
prevent concurrent generation replacement while native work remains. This makes
obsolete generation bytes and native control objects part of one explicit
bounded per-binding accounting domain. Unknown or
unbounded entries fail admission. No ownership is inferred from a project locator.
The earlier flat unreleased scaffolding is not inferred, migrated or deleted;
its state cannot be advertised as production-ready. Cache schema v2 uses an
explicit additive migration for authority witnesses and per-job scan membership.
A v1 claim without the issued lifetime-capability contract cannot obtain retirement
proof merely because a new lease acquisition succeeds.

Windows lease acquisition uses nonzero data access and share mode zero;
attribute-only access is not an exclusion witness. Inheritance keeps a reference
to the same file object. [CreateFileW](https://learn.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-createfilew),
[handle inheritance](https://learn.microsoft.com/en-us/windows/win32/sysinfo/handle-inheritance).
On Linux, close descriptors without an explicit unlock: LOCK_UN through any
inherited duplicate would release the shared-description lock early.
[flock](https://man7.org/linux/man-pages/man2/flock.2.html).
Use explicit child handle/FD lists and restore temporary Windows inheritability
on every path; unrelated children must not receive this capability.
[Python 3.12 subprocess](https://docs.python.org/3.12/library/subprocess.html#subprocess.STARTUPINFO.lpAttributeList).
These source contracts do not replace actual crash/retirement qualification.
## Alternatives considered

| Alternative | Reason not selected |
| --- | --- |
| Change saved Automatic/Allowed choices when a folder is attached | Conflicts with the user's explicit requirement and ADR-079 ownership. |
| Add a deterministic project-versus-Library intent router | Adds classification/state behavior before proving existing controls plus guidance insufficient. |
| Automatically ingest the project as Library Media | Gives retained Library records different lifetime and authority from their folder binding. |
| Reuse the shared Library RAG service as the project index | Couples Library source/activation contracts to live filesystem authority and risks cross-source retrieval. |
| Load every file or a complete directory tree into each prompt | Fails the requested scale and adds unnecessary context, I/O, and disclosure. |
| Add a resident filesystem daemon and mandatory semantic stack | Keyword indexing and explicit incremental jobs meet the initial need with fewer dependencies and lifecycle owners. |

## Consequences

Guidance can ship independently and works without an index. Saved Library choices
continue to affect pre-send retrieval; the model must evaluate relevance rather
than treating their presence as exclusive project evidence. Guidance improves
behavior but cannot guarantee every model's tool selection, so real-model
qualification is required.

Indexing adds a versioned storage owner, explicit job recovery, a tool contract,
and revocation tests. Its storage and semantic cost are bounded and optional.
Current-file validation adds query latency, but a fast stale cache cannot become
an alternate disclosure path after binding access changes.

Implementation task records and plans must link this ADR and include
storage/recovery qualification before the index is enabled in production.

### Settings execution handoff
Index cards carry exact workspace and selected-binding data; any root/status admission is captured in background workers. Binding ID alone is not file authority. Console reuses safe modal dismissal and the existing Workspaces-category Settings navigation contract; no binding deep-link or new navigation API is introduced. Existing row replacement and per-binding job/epoch/visit fences remain required. See the keyword plan and retained UI contract probe.
