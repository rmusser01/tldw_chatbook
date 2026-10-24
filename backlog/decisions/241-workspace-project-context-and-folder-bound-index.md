# ADR-241: Workspace project context and folder bound indexing

Renumbered from ADR-240 during latest-dev integration on 2026-10-22. The landed MCP/Settings decision retains ADR-240; historical project-index receipts and chronological rulings use their original source-era number. The decision and accepted contracts are otherwise unchanged.

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
Guidance and keyword indexing shipped in PR #3204. Semantic indexing is a separate optional delivery under the amendment below.
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

### Run executor snapshot for keyword capture
The local provider may consume an immutable snapshot of the exact run executor's captured local DirectoryChain through a read-only code-owned accessor. This is data provenance, not permission or live admission; both search/read review, the exact run guard and exclusions, operation pin, and final disclosure fence remain required. Remote executors expose no local chain. Ordinary file-tool wire/result and lifecycle contracts remain unchanged.

### Live query retirement
Live Capture queries do not take the exclusive durable-writer lifetime lease because they access no cache handle in the worker. They retain ordinary admitted operation ownership through actual scanner retirement. Unproven retirement retains that admission and a service-owned failure record, refuses further affected work and makes close fail explicitly; cancellation or an ordinary missing/stale file is not native proof. No PID/elapsed-time substitution or automatic permission recovery is introduced.

### Saved processing policy display
Settings reads immutable processing policy through an exact-scope admitted service operation backed by the existing store.policy(scope). It performs that read in a worker after status capture; a changed scope refuses/reloads. No constructor I/O, new persisted model, direct UI store ownership or implicit policy reset is introduced.

The existing last_scan_at field is the timestamp of the latest published page, so Settings labels it Last scan activity. Ready separately reports completion under the stated policy. This delivery does not introduce a separate last-success history or timestamp.

### Aggregate index read budget
The app-owned service shares its at-most-two index-read budget across durable writer scans and live Capture queries. Permits survive until actual reader retirement, including failure paths; writer cache/lifetime ownership remains separate. A whole-operation permit may conservatively reduce query concurrency while a build runs. Waiting occurs outside publication/disclosure fences and locks needed by cancellation. Writer Refresh/Resume uses nonblocking admission before pins/fences/service locks; when both slots are occupied it returns the existing LifetimeBusy outcome without changing scope/job state or queuing deferred work. UI copy offers retry and does not infer that another build is running. This service boundary does not impose a machine-wide budget on ordinary file tools or independent applications.


### Index worker registry connection ownership
Index writer and supervisor threads own any workspace-registry connections they open. Runtime supplies the existing finite operation_owned_connection context; it grants no file/storage authority and holds no SQL transaction. Both thread boundaries positively retire their newly opened native connections before releasing required lifetime/pin/read ownership or advertising complete retirement. A failed close retains ownership and reports unproven retirement; another thread does not close the handle. Borrowed pre-existing caller connections retain their existing ownership. This implements ADR126/228 maintenance drain rather than caching admission decisions.

### Bounded multi-directory keyword units
K5 scale qualification found per-directory control-plane cost dominating real scanner work. A keyword read unit may therefore span several directories, with fresh parent authority/exclusions/exact active-job admission before that unit and a separate current registry-then-cache atomic publication fence afterward. This supersedes the implementation plan's earlier one-directory-page preflight; it does not freeze authority for the worker lifetime. The entire unit, across all directories, is bounded by256 discovered entries,256 attempted directories including empty ones,256 raw iterator inspections and256 pending paths, with existing source/frame/read limits. An authority change after admission still prevents publication at the final fence and is enforced before the next unit.
The closed index-only protocol advances to v2; ordinary one-shot file-tool protocols and query permission semantics stay unchanged. Immutable aggregate units retain independent directory identities/completion receipts, precomputed chunks/line mappings and whole-unit replay digest. Publish all entries before applying completion/pruning, atomically; incomplete receipts do not prove coverage. Capacity or stale authority rejects the unit whole. Continuations receive fresh admission; crash recovery re-enumerates rather than trusting offsets.
Private derived-cache schema v3 adds an indexed attempt_revision to durable directory rows. Terminal attempts use the active monotonic job revision; explicit new run/Resume advances it, allowing incomplete paths to retry once while completed paths remain done. Newly discovered children remain eligible regardless of lexical order. This replaces the tree-sized in-memory attempted set with bounded queries and a durable per-run marker, preserving v2 controls/frontier/claim and v1 unproven-retirement quarantine. Existing activation/admission/native lifetime and backup exclusions still apply.
Alternatives: retained admission intervals add ownership plumbing while leaving per-directory method/fence overhead; publication-only batching preserves most read-preflight cost. Both were declined in favor of the measured-cost-directed bounded unit. New small-phase and literal full-pipeline qualification must validate benefit and resource/authority/recovery boundaries; source analysis is not a performance claim.

### Complete keyword invocation registry ownership
Finite registry connection ownership starts before index provider preflight/guard/exclusion callbacks and before Console scope preparation, and also covers direct service/key calls. A narrow registry_connection_scope exposes the runtime's existing connection-ownership context; nested contexts preserve borrowed handles. It grants no authority, retains no approval result and introduces no token broker. Ordinary tools are unchanged.
The index query operation pin/read slot spans actual Capture and cache-native retirement. An outer workspace-registry connection is a separate native owner with its own counted storage lease; it may finish after inner search retirement but must positively close before the complete provider invocation returns evidence. Maintenance remains blocked by whichever native owner is still active. Failed registry closure propagates without successful evidence and retains its counted connection; no foreign-thread close substitutes for proof. Writer/supervisor registry cleanup still precedes their retirement flags/lifetime/pin release. Tests distinguish inner query retirement from complete invocation retirement and full-profile startup enrollment.

The finite index registry scope must verify native retirement, because generic kernel close intentionally preserves retryable handles and can return after refusal/failure. An index-specific runtime adapter captures borrowed/new/replacement handle identity without opening it, reuses existing operation_owned_connection, and verifies the exact newly owned handle through native SQLite closure plus the existing registered-core retirement witness. Unknown/live closure raises ProjectIndexRetirementUnproven; counted native ownership remains. Borrowed handles and generic DB close semantics are unchanged. New outer index/UI/preparation scopes use the same strict adapter rather than hiding a failure behind a plain outer owner. This is native connection lifetime proof, never authority or activation caching.

### Cache v3 migration capacity and native lifetime
The eligibility index migration is a binding-owned write, not free setup. Production supplies a migration-only constructor context: read old version on admitted connection outside SQL transaction, acquire the exact stable binding capability nonblocking, account all binding namespaces/sidecars, constrain DB pages with DELETE-journal and existing1MiB reserve, then recheck schema and migrate atomically. Existing pages exceeding that budget refuse before mutation. Capability ownership spans commit/rollback and any failed-constructor native close; unknown closure retains ownership and fails affected work/close explicitly. v3 opens take no migration lease, preserving status/search during a build. Validatedv2 cleanup_only performs no migration/index allocation, preserving Clear under pressure; v1 unproven quarantine remains. The generic excluded-cache recovery owner has no payload catalog change. Direct standalone store fixtures use a bounded local default and do not infer project binding authority.

Cache v3 also fulfills the existing path/filename keyword requirement: the same FTS stores path and text, each chunk indexes its path, and a valid file producing no text chunks has one zero-length path-search chunk. Existing rows are rebuilt/backfilled in SQL under the same migration lifetime/capacity transaction. This does not change permissions or result semantics: exclusions precede ranking/counts and every filename/excerpt still requires live source identity/hash validation; empty-file evidence has an honest zero-length excerpt. No separate search owner or mode is introduced. Text-only historical performance receipts do not substitute for final path+text qualification.

A migration constructor whose native close cannot be proven leaves its store/capability pair retained by the runtime factory, at most one unresolved pair per affected binding. Further construction for that binding refuses, and application index close reports unresolved after draining normal work. No foreign-thread close, retry broker or premature capability release is introduced. Counted native ownership remains until positive retirement or process exit.

### Management recovery without migration and responsive close
Settings/Console use an admitted immutable management snapshot (existing status, policy, upgrade-required bool) from validated nonmigrating cache access. It preserves exact scope/job data and shows upgrade required rather than query-ready legacy coverage, making Clear available even when migration lacks capacity. Clear itself uses that nonmigrating path. Normal query/upgrade consumers remain latest-schema paths; management visits do not trigger DDL. No new status enum or persisted field is needed.
Service/runtime close move initial lock snapshots off the event loop inside their complete shielded drain, preserving actual retirement before cancellation returns and unchanged dispatch lock order.
Processing defaults include all otherwise eligible UTF8 text. Empty inclusion/suffix tuples mean all and must round-trip faithfully; user narrowing is explicit. An arbitrary four-extension default or replacement of a valid partly empty policy would silently omit project manifests. Existing security/resource limits and saved Library controls are unchanged.

The scanner and SQL candidate filters agree that empty inclusion/suffix tuples are unrestricted within existing eligibility/security limits. The previous worker exclude-all interpretation is corrected before default-all UI release; mandatory archive/generated/sensitive and source limits remain independent.


## K5 query performance qualification and bounded correction

Ruling: Retain two explicit performance boundaries. The original accepted spec target remains cached canonical lexical_candidates p95 <=500ms, including that method's production activation, guard, SQL filtering/ranking and return. The later root-added admitted-candidate <=500ms target additionally includes the real registry/root/chain fence, management-status read and floor merge. Source audit confirms the expanded benchmark is faithful, not a fixture error; all six expanded cells failed and nested/common canonical577.305ms also failed. Accept the expanded target as an explicit unmet requirement deviation for this delivery rather than weaken authority or change generic storage/config admission; preserve and report both boundaries and all adverse receipts. Cost is roughly0.5-1.2s measured cached admission before live validation on this Windows machine and an unfulfilled stronger target. Original canonical500 remains a gate, and final whole-branch review must independently judge this deviation. Address only the measured repeated exclusion normalization with a per-query immutable predicate; no cross-query cache, shared admission change or new store/service API.
The canonical metric does not include initial store/open/read-pin acquisition, current registry fence or live Capture; those real costs are separately retained as admitted-candidate and whole-tool timings. No raw-SQL-only timing replaces canonical decorated lexical_candidates. The source audit and fix2 brief are retained with the delivery evidence.


## Final integration correction contract

Ruling: Address the complete final whole-delivery review in one consolidated index-specific fix wave: ImportantI1-I4 and MinorM1-M3. ReopenK5 and extend its AC before code; original task gates/receipts remain historical rather than retroactively green. Complete admitted operations must remain owned through outer native registry closure, detached derived caches need durable cleanup-only ownership/retry without source authority, processing prefixes use the already accepted scanner casefold rule at both boundaries, and committed compatible Building rows remain searchable with partial results. Also bound proven-completed run/error history, correct scanner public types/docs and retain dirty processing input on status reload. Cost is focused service/runtime/registry/provider-controller/UI integration and new native/behavior regressions; no generic storage/operation framework, permission broadening, Library mutation or semantic work. I2 concrete durable cleanup-record format/placement and retry entrypoint require a source-backed proposal and root record BEFORE that implementation, not a guessed global collector.

Disable deliberately revokes index use while retaining its owned cache until explicit Clear. Detach removes read/publication authority immediately and must leave sufficient bounded durable cleanup ownership to reclaim already-owned derived namespaces after restart, even when source binding/root authority is gone. Cleanup-only metadata must never grant source access, query eligibility or key reconstruction from a matching path; native capabilities, validated catalogs, exact accounting group, current revocation/replacement checks, positive retirement and quota headroom remain mandatory. Any new cleanup receipt is control metadata only and charged to the binding budget; no project bodies/Library copies or general collector. Its exact storage/permission/migration treatment must be recorded before code.

Index close must reject new finite operations and drain already admitted management, query, key/UI/preparation operations through their outer native registry proof as well as cache/worker retirement. A finite lifecycle record is not retained authority. Repeated waiter cancellation leaves the event loop responsive while physical owners drain; unknown closure remains represented/reported.

Processing include-prefix comparison preserves the current scanner casefold behavior in SQL, including Unicode, exact directory and slash-boundary descendants. Mandatory/run/current exclusions and actual filesystem containment are unchanged. Building is query-eligible only for committed compatible rows under the existing exact scope/root/policy/manifest/liveCapture gates; results are partial, never complete Ready coverage. No uncommitted read or silent scope adoption.

ADR required: yes, amend existing ADR241; no duplicateADR. Reason: clarify index-owned lifecycle/cleanup contract and direct correction of existing accepted producer-consumer/search behavior; no global storage/permission interface redesign.


## Detached cache cleanup ownership receipt (final I2 correction)

Ruling: Accept the narrow I2 cleanup ownership design: fixed binding-group cleanup-owner-v1.lock holds at most4096UTF8JSONbytes {version:1,workspace_id,binding_id,authority_revision}, enrolled before cached content under live exact ownership and actual data/native admission. It is immutable cleanup-only control metadata, never a source key/read grant. Reuse native BindingLifetime handle I/O and existing namespace cleanup; no new schema, generic collector or pathname inference. Post-commit detach/archive hints queue offthread attempts; Settings Workspaces Retry removed-folder cache cleanup exposes bounded cold-restart retry with truthful completed/pending/refused counts. Cost is a small durable private control file, index-only lifecycle/background retry integration and one Settings action; missing/corrupt/unproven legacy detached receipts remain refused until proven ownership can be enrolled while live, not silently erased.

The exact stable accounting group is the existing hash(workspace_id,binding_id) under admitted DataRoot/project_index. Receipt JSON has a closed version1 schema, strict nonempty bounded IDs/revision and maximum4096bytes; it contains no locator/root/source bodies or Library records. The owned cache budget includes this file and native creation headroom. Existing native receipt handle is opened no-follow/single-link under the verified private group; receipt-specific bounded native read/write/flush/truncate uses the held handle, never pathname replacement or a general agent file API. Missing detached receipts are opened existing-only/refused, never created by retry. Interrupted creation may be repaired only with exact live ownership and all existing group native lifetime proofs; malformed detached ownership refuses. New native I/O must preserve private-parent/alias assumptions and positive actual handle retirement.

Acquire receipt and stable native lifetimes outside registry fences. Capture exact live revision before enrollment, perform bounded write/flush outside the registry transaction, then recheck live exact ownership/revision before configure/publication; a changed binding cannot gain index authority from a receipt. Existing proven v2/v3 namespaces may enroll only while live; validated legacy cleanup/Clear must remain usable under capacity pressure without requiring metadata allocation or migration first. Unproven detached legacy receipt cases remain explicit refusals, not cleanup success.

Detached cleanup retains current data admission, receipt capability and all observed generation lifetimes plus the CURRENT data-generation lifetime, repeats bounded complete namespace/catalog inventory, and verifies receipt IDs/group plus a later durable tombstone with absent binding or the same archived workspace. Active/recreated binding refuses cleanup; a distinct replacement binding at the same path has a different group. Current revocation checks and native capability serialization must prevent a same-ID recreate/Enable race from deleting new state. Registry fences span only bounded ownership/revocation acceptance, never native acquisition, joins, worker I/O or bulk reclamation. Existing cleanup_eligible, clear_owned_namespace/reclaim and whole-binding journal headroom remain mandatory. Unknown layouts/native closure keep ownership/pending failure rather than query permission or successful removal.

Attempts/retries run offloop under the complete I1 service operation owner and drain during close; no retained authority, source scan, auto-Refresh, watcher, generic collector or new Library behavior. Durable receipt remains available across process exit so UI retry does not require the old folder row or source reattachment. Disable deliberately retains its cache until live Clear; detach/archival gets revoke-first cleanup/retry. Retained empty control/CAS/receipt metadata is distinct from cached source bodies.

Additional authorized exact product path: tldw_chatbook/Workspaces/project_index/lifetime.py, fixed-purpose bounded receipt capability operations only. New focused regression file: Tests/Workspaces/test_project_index_detached_cleanup.py. No new production file/schema/dependency authorized by this amendment; request exact path if a clean local module is concretely needed. ADR required yes, existing ADR241 amended before code.


### Building compatible coverage proof

Ruling: Permit a narrow bool-only committed-compatible-row existence check for Building scope/query eligibility, sharing the exact SQL processing/exclusion predicate with lexical ranking. Raw indexed counts include newly excluded rows and cannot satisfy I4 zero-compatible coverage. Preserve exact scope/current and applicable run floors/native admission, never fabricate production authority/guards; no names/counts are disclosed by the check. The earlier fix2 no-new-API restriction applied to its performance-only correction, not this final I4 requirement. Cost is additional admitted SQL only on Building eligibility and a small internal store/service seam; Ready lookup remains on its normal predicate path and is requalified on final source.


## Semantic implementation amendment after PR3204 merge (2026-10-22)

Implements the approved optional feature on current cachev3, preserving Library choices and keyword-only operation. Add migration004/fresh v4, independent semantic settings/revision and immutable generation/member/vector tables in the existing SQLite owner. Members own path/identity/hash/lines and float32 bytes rather than mutable keyword chunk IDs. No new Library collection, Chroma/HNSW persistence, dependency, source scanner or generic VectorStore change. Reuse actual embeddings under exact code-issued project scope before constructor/activation/native/model effects; default Library routing is unchanged.

Choose bounded streaming exact cosine for the initial backend. Apply current and requesting-run policy/exclusion predicates before top-k, then RRF60 and existing100candidate/10hit/16KiB live-validation boundary. Source/vector pages are at most32records/1MiB, total retained/queued source at most4MiB, no transaction/cursor across provider await, and old compatible Ready survives staging failure. A new Ready pointer CAS checks scope, semantic revision, consent/settings/model/artifact and one captured keyword manifest. A later keyword manifest does not rewrite an old immutable snapshot; live validation still rejects changed files. O(eligible members*dimension) is an explicit cost, with actual20k latency/memory/quality qualification before release.

Use one binding-group allocation-v1.lock through existing native BindingLifetime for finite SQL units outside registry fences; it is a fixed zero-payload known control file, not permission or a new persistence owner. Count it and all owned regular files in the closed inventory. Cooperating keyword/semantic writes are sequential. The gate is try-only and may be acquired after a retained job lifetime; every gate holder uses only try-only receipt/lifetime acquisition, unwinds on busy, and never joins/waits while holding it. Cleanup returns Pending for a live owner. Acquire it before a registry fence, close each actual store before release, and keep scanner/provider/Capture/join/retry waits outside the unit. Set actual project SQLite timeout/busy_timeout to zero and retry only outside the unit. This narrows constructor, frontier/admit/emit, search, revoke and supervisor finalization lifetimes without weakening authority or blocking keyword lookup for a whole build. Every semantic-enabled cache mutation, including settings/control/claim/status/failure/checkpoint/CAS/pruning/migration and competing keyword writes, uses pre-write allowance and actual native settlement. Status/search do not silently allocate control. Preserve standalone keyword availability/limits on systems without semantic capability; semantic-specific qualification is not an embedding dependency for keyword.

A finite cleanup/receipt control unit may own the group gate while sequential namespace SQL units borrow that exact live lease. Borrowing is limited to the same group and process/thread/execution; stale or copied contexts and other groups refuse. The control owner holds no SQLite handle. At most one native SQL store/unit is active in the group at a time (same-store nested methods may reuse it); each namespace closes and inventories before the next opens. Overlapping independent reservations could budget against stale other-file lengths and are forbidden. Admit and count bounded receipt growth under the gate, then inventory again. Any child store, receipt, capability or gate settlement uncertainty retains/quarantines the exact outer gate and operation owner. No borrow extends over joins, retries, scanner/provider waits or file Capture; cleanup capability acquisition remains try-only and a live owner yields Pending.

Public read admission uses the unchanged 3s outside-unit retry bound. Measured consecutive finite writer units can starve status at its first constructor admission, so ProjectIndexService may record an ephemeral pending-read count per exact binding group only after status/management/policy receives transient LifetimeBusy. Registration always clears on success, timeout, close or any other exit. After a successful background SQL operation has completely released its fence, native store and allocation gate and finished inventory, a same-service writer with a pending reader yields once for at most25ms using a stop-aware wait, unless stopped or closing. No unconditional per-unit delay, gate-held wait, cached authority/status or new persistent owner is introduced. At most one bounded yield follows a successful unit, so a stream of readers cannot indefinitely suspend the writer. This is a cooperative acquisition window for the existing5ms read retry, not a cross-process fairness or hard scheduling guarantee; foreign contention still reports Busy at3s. Native synchronized contention and physical stop/close tests must verify the chosen policy.

Review clarification for binding-wide admission: before a write-capable outer SQL unit enters its target allocation_store or opens its native handle, the existing group gate owns a bounded, serial, existing-only/read-only peer preflight through the established project SQLite owner. Positively close and inventory each peer before another peer or target opens. Cache only a boolean for this exact finite gate-held unit and discard it on release. Qualify the group when any validated namespace has semantic consent enabled, a non-null ready-generation reference, or generation/member rows. Disabled settings alone, schema-v4 tables alone, and an Off row without generation data do not force semantic runtime qualification; their bytes/freelist still count under the keyword allowance. Unknown/corrupt/aliased/recovery-sidecar peers or uncertain close refuse. There is no permanent semantic marker or new persistence owner. Preflight write-capable constructor/management units, including regular constructors that may migrate; keep cleanup-only/read units read-only, and refuse unexpected read-to-write nesting rather than opening a peer beside a live target. Pure keyword groups retain keyword admission without semantic source qualification. No gate waits, registry-fence reversal or overlapping native SQL store is permitted.

Persisted semantic settings v1 use configuration catalog keys, not raw model paths or connection settings: provider_id matches [a-z][a-z0-9_-]{0,63}; model_id matches [A-Za-z0-9][A-Za-z0-9_.-]{0,127}; provider_ref is exactly provider:v1:<provider_id>; config_ref is exactly model:v1:<model_id>. The model key selects the existing EmbeddingConfigSchema.models entry (which already contains its provider and configuration); no separate configuration-profile system is introduced. Project semantic configuration must refuse unsupported key syntax without rewriting saved Library configuration. destination_fingerprint, config_fingerprint and a non-null approved_destination_fingerprint are exactly sha256:v1:<64 lowercase hexadecimal characters>. Different well-formed approval/destination digests still project Paused. Syntax validation does not prove a caller's arbitrary ID is not a secret; Task2 must resolve code-issued actual provider/model catalog membership and construct fingerprints from non-secret canonical fields before effects or persistence. Task1 rejects endpoint/userinfo/query/assignment/freeform reference material before persistence and exposes no raw invalid values in errors. These metadata formats do not change schema-v4 ownership or create a catalog/runtime in Task1.

For verified single-main DELETE journaling, memory temporary structures, cache_spillOFF and no savepoints/ATTACH/VACUUM, reserve J=2S+N(P+8), with S=65536, old pagesN and page sizeP. Other owned regular-file lengthsO, separately permitted control growthE and held-connection page capC must satisfy C>=N and O+E+C*P+max(observed_journal,J)<=1073741824. Verify actual file length, PRAGMA return/readback, effective temp/locking/synchronous/secure-delete/FK settings; no negative-allowance clamp or assumed freelist savings. Abort SQL on the first SQLite error. The initial analytical source is exact SQLite3.49.1/source873d4e274b4988d260ba8354a9718324a1c26187a4ab4c1cc0227c03d0f10e70; another linked build needs a source-backed envelope before semantic writes. Native scaled near-capacity peak tests supplement rather than replace this proof.

Before opening a semantic-admitted namespace, refuse nonempty journal/WAL/SHM, unknown sidecars or unretired handles instead of triggering unbudgeted recovery or deleting evidence. Keep exact connection/gate through commit/rollback/native close/inventory. Generic kernel commit can fail outside its body rollback arm: project owner handles settlement without modifying the generic kernel. Ambiguous rollback/close retains quarantine/ownership; physically settled but logically ambiguous commit requires fresh admitted CAS reread before reporting/retry. Preserve nonmigrating v2/v3/v4 Clear/cleanup and existing v1 quarantine; do not erase old Ready to finance staging.

The analytical envelope covers cooperating new-version processes after old-native retirement. It is not an OS quota or a certificate for concurrent older binaries ignoring the gate, external file writers, filesystem allocation-unit overhead or arbitrary extensions. Observed old/mixed ownership refuses semantic allocation; mixed-version universal enforcement is outside this delivery. Do not misstate the prior fixed-reserve keyword policy as a demonstrated overrun. Unsupported semantic setup degrades honestly without Library changes or model download.

Provider/model/destination choice and consent remain independent per binding. Persist restart-safe secret-free references/fingerprints, resolve actual credentials only at the factory, compare current destination/config before calls/publication, and pause changed destinations before approval. Actual success/retry/failure logs and propagated errors must not expose raw credential-bearing URLs. Local models use an already complete snapshot with offline-only loading, including restored profiles; preserve the existing broad optional-dependency gate. Enabling keyword never implies embedding consent. S1a/S1b/S2/S3 are serialized implementation tasks, with actual real-model qualification distinct from deterministic lifecycle tests.

ADR required: yes, this amendment to existing ADR241 precedes semantic code. It replaces stale schema-v2/Chroma allocation assumptions while keeping the accepted product intent and authority boundaries. Plan: [semantic implementation](../../Docs/superpowers/plans/2026-10-20-workspace-project-semantic-index.md). Primary SQLite references and exact/runtime qualification limits are retained with that plan and semantic QA.


R1 implementation refinement (recorded before owner-policy edit): the actual first GREEN failed because db.workspace_project_index permits only PRIVATE_FILE and refuses READ_ONLY_URI. Authorize only this existing owner's allowed kinds to become _PRIVATE_AND_READ_ONLY, using the existing checked mode=ro, must_exist=True path for serial peer inspection. Keep its production_module, private-path validation, backup/sync exclusion, foreign-source flag, recovery flags, memory exclusion and every other owner unchanged. This introduces no source grant or generic-kernel behavior. Record the new WorkspaceProjectIndexDB._probe_peer call as a separate read_only_uri C99 row in backlog/docs/sqlite-private-owner-inventory.md and deliberately extend only the exact connection census range to include C99 in Tests/DB/test_private_sqlite_inventory.py, retaining its guards. Add actual project-owner read-only/missing-target/write-refusal controls and run matching inventory/registry nodes; no inventory regeneration or unrelated baseline absorption.

Missing allocation gate bootstrap: before first gate creation, a regular write-capable constructor must inspect every existing namespace serially through existing-only read-only handles, with validated inventory and positive native close. An existing v4 target OR peer refuses without creating the missing gate, including repeated/restarted attempts. Only an empty or fully validated legacy v2/v3 group may proceed to a separate gate-held write-capable unit, which rechecks current peers before opening the target. Read/status never creates a gate. Ambiguous aliases, unknown catalogs, recovery sidecars or close uncertainty refuse; uncertain ownership is retained. The preliminary scan grants no write authority and cannot substitute for the gate-held recheck. Costs if wrong: safe refusal or additional finite peer-open work; no repair-by-retry of missing v4 control.


Demonstrated Windows offline-loader correction: actual qualified factory reached model_recovery._closure and failed because that owner imports stdlib os while pinned_directory returns the existing platform filesystem descriptor. Python Windows stdlib os.listdir(fd) is unsupported. Reuse tldw_chatbook.Utils.platform_files.os in model_recovery.py, as native_files already does, so listdir/open/fstat use the existing handle-relative no-follow Windows implementation and unchanged stdlib POSIX behavior. This is a narrow import/adapter correction under existing ADR126 and ADR241, not a new storage/source permission or Windows backend. Do not add path-based enumeration fallback, weaken alias/single-link/identity/manifest/closure limits, change the generic platform shim or bypass model/recovery gates. Retain the actual offline13/16/17 failures and run native closure success/refusal tests plus directly affected model-reference/recovery controls and actual blocked-network offline factory. Report any additional owner need before edits.

### Task2 review correction: retained project model close failure

Ruling: Use only the existing ProjectIndexService operation owner to strongly retain failed project model retirement. Add a private projection_close_pending token-to-actual-wrapper map to _Operation and narrow private mark/settle/check helpers under the service lock. Register the actual project wrapper before factory/model close; keep its actual factory/cache record and carrier/retained parent on any failure, raise/log only classified project_embedding_close_failed without raw cause, and remove only its token after positively successful retry. Carrier close and wrapper use/rebind refuse while its retirement is pending; the service-rooted operation -> wrapper -> carrier -> actual parent chain must survive caller reference loss and GC. A self-contained wrapper/carrier cycle alone is insufficient. Default Library close behavior is unchanged.

Keep projection-close pending separate from generic retirement_unproven. It must not blanket-deny registry_connection_scope or keyword reads merely because a model close failed; new semantic use on the affected binding refuses, while actual keyword lookup remains usable. During explicit service shutdown, detect/report a known unresolved projection close before waiting forever on its not-done operation; retain its real owner and permit positive same-wrapper cleanup retry. Never clear an unrelated native/store/capability uncertainty, claim cleanup success, or release the parent because an exception was swallowed. Serialize close/rebind and prevent embed reuse after partial closure. No SQLite handle/gate/fence spans model close, joins or waits.

Authorized additional exact product path is tldw_chatbook/Workspaces/project_index/service.py, limited to this existing operation-owner retention/reporting seam. Existing Task2 projection_scope.py, embeddings_wrapper.py and actual factory paths may consume it; no generic recovery framework, new persistence owner, source grant or Library behavior change. Test actual factory record retention after injected close failure, sanitized logs/errors, caller-reference loss/GC, carrier and affected-semantic admission refusal, actual keyword lookup availability, explicit shutdown unresolved result, and successful physical retry followed by parent retirement. Root inspected actual factory close (cache.clear only after all record closes), service operation roots/admission and shutdown waits before this amendment.

ADR required: yes, amend existing ADR241 before the cross-module owner change. ADR path: backlog/decisions/241-workspace-project-context-and-folder-bound-index.md. Reason: narrow completion of accepted physical retirement and keyword independence, reusing existing project operation owner; no duplicate ADR.

### Task3 background build model retirement retry

Ruling: A service-owned semantic build whose actual wrapper close fails must keep its original owning worker execution alive, retaining the same wrapper/factory/carrier, operation, binding lease and pin. Add only `ProjectIndexService.retry_semantic_close(job_id)` as a nonblocking signal to that retained semantic run; one signal retries the same wrapper close on its original owner execution. This signal grants no source, model, query or publication authority and permits no further embedding. It remains usable after `aclose()` reports unresolved retirement. Wrong, completed or unrelated job identifiers cannot retire another run. Failed retry retains the actual owners; positive close, carrier/native/registry retirement and operation completion precede release. No SQLite connection, allocation gate or registry fence spans retry waits or native close.

The run waits for an explicit signal without busy spinning; repeated cancellation/Stop/Clear/detach/shutdown cannot unwind its retained ownership merely to finish a coroutine. Service shutdown promptly reports known pending retirement rather than joining that worker indefinitely, and may complete after a positive retry. Keyword reads remain available while affected semantic admission refuses. Cover actual close failure, reference loss, repeated cancellation, shutdown reporting, failed and successful same-owner retry, wrong-job refusal and physical retirement.

This is a narrow completion of existing ADR241 model-retirement ownership in the existing service and Task3 path map, not a new persistence owner, durable control, foreign-thread close or general retry broker. The earlier no-retry-broker statement concerns failed SQLite constructor retirement and remains unchanged. Synchronous query close-failure behavior must be explicitly reviewed and reported before inventing an additional executor. Cost if wrong: a retained background owner may require process exit; do not hide that limitation or report cleanup success.

ADR required: yes, amend existing ADR241. ADR path: backlog/decisions/241-workspace-project-context-and-folder-bound-index.md. Reason: make the accepted failed-model-close retry usable under the service-owned async build execution without changing authority or storage.

### Task3 synchronous query close failure and bounded retry signal

Ruling: Keep query embeddings on the existing synchronous tool worker. Do not add a query thread pool, per-tool event loop, transferable source carrier or foreign-thread model close. If close fails, perform at most one immediate same-wrapper retry. Persistent failure keeps that same owner execution and its actual wrapper/factory/carrier/operation/pin alive in a non-spinning signal wait until positive same-wrapper close. No further embedding, query disclosure or publication occurs from that pending owner. Capture any query result only after positive cleanup. Cancellation or outer tool timeout cannot release those physical owners.

Superseding the earlier job-id-only signature before implementation: use `ProjectIndexService.retry_semantic_close(key)` to nonblockingly signal only already-retained pending close owners for that exact binding key, including background builds and the existing synchronous query worker. Cleanup signaling does not grant source authority, alter consent/revisions, query data or start embeddings; it remains available after service close reports pending. Wrong/retired/unrelated keys do not release any owner. One attempt per signal, with failed retry retaining the actual owner. Existing run/operation records own the event and safe pending state; no new persistence or general executor.

Bound semantic query admission to one active query projection per service, nonblocking with no queued query text: contention degrades that invocation to keyword-only candidates. This accepted throughput ceiling bounds the exceptional retained query worker count to one and avoids a pool of waiting semantic queries. Keyword reads must have an available independent read slot while this query waits; retire any cache SQL handle/gate/fence/read slot before the close-retry wait, retaining only actual model/carrier/operation/data capabilities needed for retirement. If the current search structure cannot safely release the slot, move query projection and its close before entering `_search_store`; do not release native owners on an assumption.

The remaining exceptional user cost is one tool worker waiting until explicit cleanup retry succeeds; the UI and keyword search must remain responsive. Task4 must expose the safe pending-close reason and a per-binding Retry cleanup action through the existing card, including after a failed shutdown. Test actual pending query close on its original worker, repeated tool cancellation, bounded additional semantic admission/keyword fallback, live keyword hit, wrong-key signal, repeated failed retry, successful same-owner retry and physical carrier/operation retirement. No synthetic carrier or global permission relaxation substitutes for this proof.

ADR required: yes, amend existing ADR241. ADR path: backlog/decisions/241-workspace-project-context-and-folder-bound-index.md. Reason: bounded same-owner cleanup for the accepted finite sync-query integration. Cost if wrong: a held tool worker remains unresolved, reported rather than silently retired; independent review must check both ordinary and failed close paths and keyword independence.

### Task3 finite registry units before model work

Ruling: Reuse `project_index_registry_connections(database)` and service `registry_connection_scope()` to retire admission-created WorkspaceDB handles before actual semantic model/provider work, physical model close or close-retry waits. The actual outer runtime context records its pre-existing native/core handle and yields a same-thread callback for finite retirement; nested service/project scopes consume that same outer owner. Prefer the existing yielded context seam over another service constructor callback. The callback may close only a newly owned or replacement connection, verifies that exact native handle is closed and absent from the core registry, and retains final proof for every such handle. A still-live borrowed caller handle is never closed: refuse semantic model admission before effects and allow ordinary keyword fallback. An absent handle needs no new open. Do not change generic DB close/ownership behavior.

Invoke that narrow finite-retirement seam after issuer/current-state admission and before actual model construction/encode/close or retry waits. Authorize the exact additional Task3 path `RAG_Search/projection_scope.py` for `_current_state` and associated project admission integration only. Existing service/runtime/semantic paths are already in the task map. Preserve complete operation/native lifetime accounting through final scope exit, repeated cancellation and replacement handles; a bare `database.close()` callback without entry ownership and positive native/core proof is insufficient. Any uncertain retirement propagates and remains owned rather than permitting model effects. No registry transaction/fence or allocation gate spans the close.

This is a semantic boundary correction under existing ADR241, not a new registry, persistence owner, generic connection framework or source grant. Library default calls and ordinary keyword query semantics remain unchanged. Verify actual production file-backed WorkspaceDB connections are physically retired at model entry/held wait, replacement handles are also retired, a borrowed live caller handle is preserved/refuses before model work, failed native retirement stays represented, and default keyword/model-free controls still work.

ADR required: yes, amend existing ADR241. ADR path: backlog/decisions/241-workspace-project-context-and-folder-bound-index.md. Reason: complete finite SQL ownership at the accepted project-model boundary. Cost if wrong: unavailable semantics under a borrowed handle, or retained unknown native ownership; neither may be hidden as successful zero results or repaired by closing caller state.


### Task3 explicit OpenAI transport retirement

Ruling: The held-provider lifecycle test exposed that the OpenAI factory record has close=None while its embedding closure owns a requests.Session. Reference loss or factory-cache clearing is not deterministic transport retirement. Extend only the private _openai_embedder seam in tldw_chatbook/Embeddings/Embeddings_Lib.py to return the embedding callable and the existing session.close callback; the single production _build caller stores that callback in its existing CacheRecord. Adapt the direct consumer in Tests/RAG/test_embeddings_lib_logging.py and focused factory/held-provider controls. The existing factory refcount/condition, cache record, same-owner wrapper close and pending-retirement path remain the owners; do not introduce a transport manager, executor, new persistence or permission boundary. Requests/response behavior, destination approval, credentials, saved Library policy/item scopes, and keyword operation remain unchanged. Shared factory retirement now explicitly closes its already-owned OpenAI session for both project and default callers; do not silently substitute project-only no-op closure.

Before source changes, retain the actual None-callback diagnosis and the aborted test as adverse evidence, not proof that the original callback failed. Test explicit real Session close on the original worker after in-flight embed completion, injected failure retention and successful retry, exact cache/wrapper/carrier retirement, and default-factory closure plus sanitized provider logs. No SQL handle/gate/fence spans close. No new model quality run is required for this HTTP-only seam. Cost if wrong: close failure remains truthfully pending under the existing owner; independent Task3 review includes this producer delta.

ADR required: yes, amend existing ADR241. ADR path: backlog/decisions/241-workspace-project-context-and-folder-bound-index.md. Reason: complete deterministic transport retirement at the existing embedding factory boundary without a new owner or storage contract.


### Task3 eviction retains failed transport retirement

Ruling: The new explicit OpenAI session callback makes both existing idle and LRU eviction paths part of deterministic project transport retirement. They currently pop a zero-refcount CacheRecord before close, losing the owned callback if close raises. Correct only those branches in Embeddings_Lib.py to invoke the existing callback before removing the exact record under the existing factory condition. A failed close propagates while the actual record/callback remains retained for the existing wrapper/factory close path; no replacement model is built and no new ownership framework is introduced. Successful eviction retains existing capacity, refcount and concurrency behavior. Project wrappers already prevent embedding reuse after close starts. Preserve failure evidence, test both idle and LRU actual callback failure/retention and successful physical close retry, and cover the directly affected factory concurrency contract. Do not repeat cached-model quality or held HTTP qualification when this isolated ordering delta does not reach those previously exercised paths.

ADR required: yes, amend existing ADR241. ADR path: backlog/decisions/241-workspace-project-context-and-folder-bound-index.md. Reason: retain the already-owned session across failure at both existing cache retirement entrances. Cost if wrong: safe failed embedding and retained resources until explicit existing close succeeds; never a lost native owner or false retirement.


### Mounted semantic consent and release coverage

Use the existing perbinding Settings card and independently revisioned semantic snapshots; explicit model/destination setup occurs on a worker, while compose/status stays model-free. Destination changes stage Paused before approval; cancelled disclosure causes no provider work and no saved Library mutation. A pending Retrycleanup retains the actual old service/key after failed shutdown and only signals its already-owned worker; signal acknowledgement is not physical retirement. Compatible priorReady with failedrefresh is displayed truthfully. RequiredPRcoverage follows the bounded actual-runtime ADR103 amendment; actual cached-model app relevance, retained20k scale, optionaldependency independence and tracked raw/sourceproof remain separate release gates. No new persistence owner, global Library collection or semantic auto-merge.

Task4 consent refinement: the destination description shown by the disclosure dialog and the staged settings approved after it must derive from one inspected catalog snapshot. If its provider/model/destination/config fingerprint differs from the staged revision, refuse before opening the dialog and ask for a new stage. The service's later current-catalog check and existing epoch/revision CAS still apply; no UI-only approval substitutes for them.

Task4 precision correction: a cached-model diagnostic on a CUDA-capable Windows host selected `device="cpu"` yet the existing Hugging Face embedder loaded float16 parameters on CPU because it chose dtype from CUDA availability before resolving the requested device. Resolve the actual target device before `AutoModel.from_pretrained`; choose float32 for CPU and float16 only for actual CUDA, preserving `auto`/CUDA and existing factory, cache, close and authority ownership. This is numerical correctness, not an assertion that CPU precision caused the retained 20k throughput failure: that attempt was operator-stopped after 4,864 chunks and raw timing attributes most delay outside the embedder call. Any service performance change or second retained scale run needs a separate measured ruling.

Task4 measured local batching amendment (recorded before source changes): the stopped 20k run logged about 1.70 seconds inside each Hugging Face embedding body and about 5.90 seconds between body logs. A separate four-page actual-carrier comparison measured about 6.32 seconds per complete decorated wrapper call, including repeated real projection admission; deterministic vectors on a comparable 20k keyword state did not reproduce the model-carrier interval. Reduce only local Hugging Face source/provider admission frequency by fetching and embedding at most 128 contiguous chunks with at most 1 MiB encoded source text. Hosted and non-Hugging Face providers keep the existing 32-row source/provider page. Before any native append, validate the complete provider result count and dimensions. Split the returned local vectors into separately admitted, contiguous native append transactions of at most 32 rows and at most 1 MiB encoded vector/source payload each; query vector pages remain at most 32 rows and 1 MiB. The unchanged paged query rank retains one outer `store.sql_unit()`/registry fence across those pages, so the page limit is not a per-page query transaction claim. Recheck live source/manifest, scope, semantic epoch/revision, consent and destination at each committed slice and before the final Ready CAS; advance the durable checkpoint only after each successful contiguous slice. Do not hold SQL, registry fence or allocation gate across embedding await. Preserve the 4 MiB queued-source and 1 GiB binding limits, native allocation formula, existing owner and cancellation/retirement gates. Raw encoded 128-vector local results are bounded to 8 MiB at the supported maximum dimension 16,384 per representation; this is not a bound on total Python object/RSS memory. Verify a 129-plus-chunk tail, source-byte trim, maximum-dimension split, wrong-count and wrong-dimension refusal before append, cancellation/source change between slices, actual native retirement and memory before another retained 20k qualification. This prospective amendment does not certify improved throughput or relax the original 3,600-second scale bound.


## Final semantic review correction contract

ADR required: yes, amend existing ADR241.
ADR path: backlog/decisions/241-workspace-project-context-and-folder-bound-index.md.
Reason: clarify existing eligible-source, exact provider destination, independent consent and physical-owner contracts; no schema, backend or dependency change.

A Partial keyword cache may retain rows outside its current processing policy. Semantic source paging must apply the existing current policy and authoritative binding exclusions before reading text, using bounded keyset SQL. Append validation and final completeness count the same eligible set, with fresh service authority at each unit. Retained forbidden rows cannot reach a provider or prevent an otherwise complete eligible generation from becoming Ready.

Project OpenAI-compatible transport refuses HTTP redirects, including 307/308, before any resend; the explicit project factory option is set only by the issued project wrapper. Default Library transport retains its existing behavior. A redirect refusal is an unavailable provider outcome, never new destination consent.

New semantic work remains denied while any same-binding model close is pending. Already-admitted carriers retire independently: close verifies issued identity, live parent, original execution/thread, zero live children and only its own operation's pending model-close tokens. A peer's ordinary or failed close cannot poison physical retirement. Own failed close and unknown native closure remain retained.

Semantic ranking captures its independently consented compatible Ready generation. Candidate and final disclosure fences revalidate the captured semantic revision/settings and that exact compatible generation. A mismatch returns unavailable without stale semantically selected hits. Keyword scope/manifest alone cannot attest semantic consent.

Tests that perform native-positive semantic allocations explicitly require the production qualified SQLite source/version/options; unsupported runtimes skip only those positives with a precise reason. Portable keyword/refusal controls remain active. Required Windows preflight and positive assertions remain fail-closed.


### Pending nested-carrier operation ownership correction

ADR required: yes, amend existing ADR241; ADR path: backlog/decisions/241-workspace-project-context-and-folder-bound-index.md. Reason: preserve the existing actual operation owner across nested issued carriers, without a new owner/schema/permission boundary.

Nested registry scopes borrow the same physical operation. Carrier retirement therefore refuses while ANY actual wrapper token remains pending in that retiring parent operation, including a nested carrier for a different binding. A positive independent operation may still retire while another operation closes. New semantic admission retains its existing binding-key pending filter. Issued identity, original execution/thread, live parent and native-child checks remain mandatory; positive same-owner wrapper retry precedes both nested carrier and outer operation retirement. Add one real-issued/native/factory regression covering same-binding and different-binding nested owners, plus existing independent-peer/own-failure/native-child controls. The already-qualified441bb9 cached-model result remains historical for this guard delta; no model/20k replay is implied.
