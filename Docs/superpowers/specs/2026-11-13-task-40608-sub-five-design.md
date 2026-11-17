# TASK-40608: finite operations and shared presentation state

The owner explicitly authorized implementation on 2026-11-13 after rejecting the previous 84.5-second README and 162.0-second edit results. Both original deterministic-loopback journeys must take less than five seconds of application time, excluding only measured human disclosure and approval waits, in all three candidate runs of three sequential baseline/candidate pairs. The previous percentage gate does not establish usability.

ADR required: yes. ADR paths: `backlog/decisions/126-complete-local-backup-and-recovery.md` and `backlog/decisions/231-console-polling-and-full-state-reconciliation.md`. The task changes finite ownership and presentation reconciliation boundaries, while preserving their checks.

## Chosen approach

Reuse existing operation owners and Console presentation machinery. Move repeated work to complete finite local operations and independently keyed detached display sections. Do not add a second scheduler, a permission TTL, a generic connection pool or a new storage schema. Increasing TTLs and further per-helper verdict caches are insufficient. A wholesale service/actor rewrite is an alternative to assess, not authorization to discard the existing safety contracts.

## Shared presentation state

Build on `config_sync._Cadence`, `_projection_inputs`, `publish_console_presentation`, `ConsoleReadinessConfigProjection`, `ConsoleContextReadSnapshot` and `_ConsoleRegistryDisplayReads`. Routine rendering consumes current detached facts; unchanged render ticks must not start native reads merely to discover that rendering is unchanged. Native reconciliation is coalesced and retains each existing source/owner/publication guard.

Keep distinct dependencies for config-derived display, character scope/rows, context memory facts and workspace bindings/physical status. Ordinary streaming/payload changes must not invalidate a durable memory-presence fact or unrelated workspace bindings. Use exact repository/database/session/conversation owners and existing local revisions; where no complete memory-write revision exists, add a narrow post-commit repository revision covering its real writers. Preserve the lineage-dependent path for positive memory. Unsaved conversations/no repository may project raw context without a DB callback when policy is already available.

Existing active/config/context and physical workspace availability reconciliation bounds remain one second; idle character reconciliation remains two seconds. Registry display metadata retains its original service-identity/local-mutation-generation invalidation; the first implementation incorrectly added a new one-second registry-metadata expiry by conflating it with physical availability. Remove that added polling. A local generation is not a cross-process signal, but this performance task does not add a new cross-process registry-label refresh feature. Age timed observations after successful physical completion, coalesce overlap, and never renew failed/obsolete reads. Existing external file/DB/folder reconciliation and fresh action checks remain.

FULL has real effects, including stale-default settings handoff, runtime/provider gates, attachment reconciliation and acknowledgement. Keep explicit FULL, owner changes, terminal settlement and trailing replay under their original ownership. Separating pure rendering must not omit these effects. Custom, subclassed and in-memory callbacks retain their original routes. Display data never grants tool, approval, storage or persistence authority.

## Finite DB operation batches

Use `run_owned_db_call(database, callback)` for complete synchronous local phases, rather than adjacent tiny callbacks. Preserve original transaction boundaries, check order and publication semantics inside each phase. Keep its connection until the complete finite callback returns; borrowed handles remain caller-owned.

For concrete multi-DB phases, use exact captured DB objects and their existing `operation_owned_connection` and `core_operation` contexts. Extend repository operation reuse only to an already-issued live same-participant ancestor (A-to-B-to-A), never another instance that happens to name the same file. Revalidate source/path, read-only state, process/thread/task, pause and live custody; restore and revalidate the previous operation in all exits. Independent participants still receive independent ordinary admission.

Keep SQLite's independent connection/resource lease, native target preparation, current named-parent/leaf checks, and uncertain-close handling. No operation lease spans provider or human waits. Retained descriptors do not prove current path ancestry after rename/recreate. Existing primary-plus-related-path admission does not grant every member independent borrowable authority; do not generalize that capability without a separate reviewed design.

## Proof and iteration

### Replacement assessment

A cleaner replacement would expose complete typed storage commands and detached
presentation snapshots, with one explicit owner for each finite command. Its
admission boundary would gather the union of that command's dependencies once,
evaluate the existing rules from that evidence, and retain fresh checks after
waits and before protected effects/publication. Renderers would consume snapshots;
existing bounded reconciliation and local changes would request one coalesced
refresh. This removes repeated admission caused by helper composition while
retaining independent SQLite and native-resource retirement.

The problem is multiplication, rather than one simple O(N) database query:
source count, shared ancestry, nested helper boundaries and recurring consumers
multiply the native work. The private-directory allocation path also contained a
depth-squared traversal, removed in Task6. Snapshot coalescing and vector evidence
acquisition address different factors; fewer scalar admissions alone do not
prove less total work or lower latency.

Replacing the entire storage layer at once would also require reproducing
restore/maintenance exclusion, custom sources, independent same-file owners,
transaction borrowing, cancellation, external changes and uncertain-close
custody. The selected migration introduces complete operations at existing
boundaries while keeping those proven owners. A native whole-observation backend
could reduce Python/Win32 transitions, and a separate storage process could
isolate UI-thread interpreter contention, but neither has demonstrated the
required journey speed. Both add build or process-lifetime/authority-transfer
contracts. They remain alternatives pending a controlled critical-path result,
not a claimed shortcut to five seconds.

The process alternative needs a command protocol, not a replacement of the
thread executor with `ProcessPoolExecutor`. Current callbacks capture live
services, thread-local SQLite transactions, operation tokens and publication
locks. Source receipts also contain Python object identities. A child must
admit and retire its own resources and return detached values; those values
cannot serve as parent execution leases. Cancellation and maintenance must stop
new requests, await physical retirement and account for uncertain effects before
shutdown. A bounded read-only permission command is the smallest useful
prototype. Its parent must recheck the selected source after completion, and
permission publication needs an explicit owner. No process backend is selected.

### Qualification

Task20 adds a checked missing-index branch to stock Windows Console preparation.
With current retained data/binding authority and a positively retired named
observation, an absent `project_index` base yields no optional search scope and
creates no cache. Public status, management and existing/custom bases retain
their original paths. This is a negative snapshot for one run; it does not
freeze absence or grant permission. ADR241 and ADR126 define the source and
uncertain-retirement boundaries.

The finite local file-tool decision may project its kill switch and initial
effective state from one fresh checked permission payload after workspace
authority validation. Stock no-effect eligibility, original refusal ordering and
source/profile checks bound this reuse; it ends before any stamp persistence,
approval callback or execution. Review before an approval and invocation after
it remain separate reads. Custom/SSH/promotion routes keep existing behavior,
and a failed switch read does not cache failure for the later state read.
ADR126 and plan Task19 define the causal Windows controls. The unrelated
POSIX-only analytical footprint path is not a Windows optimization candidate.

Measured follow-through also batches ordinary activation derivations within exact finite boundaries. Stock warm Agents guard entries retain ordered independent acquisition and per-source decisions while sharing initial/final fully validated ordinary control inputs. The existing MCP canonical preparation may share the same inputs for repeated pre-lock generation selection, discarding the observation before the source-lock wait. Restored/custom/cold routes retain their original checks, and native refusal never becomes a fallback retry. All post-lock and effect/publication checks stay fresh. These are single-entry observations, not cross-operation permission or witness caches; ADR126 and plan Tasks7/8 define their qualification.

Add causal native-work-count tests, not just wall-clock thresholds. Test unchanged/streaming ticks, local invalidation, external reconciliation, terminal FULL, custom readers, cancellation/teardown, operation ancestry, independent same-file owners, pause and uncertain retirement. Use targeted suites only.

First qualify one candidate against the absolute gate using the unchanged original journey harness. If it fails, attribute the remaining critical-path work and continue implementing within these constraints. Do not repeat expensive full paired qualification until a candidate is plausibly below the gate. Final proof is the three sequential matched pairs with frozen product/harness hashes, raw times, approval/disclosure intervals, admission counts and every original grounded-answer/read-only/approval/exact-diff assertion. Do not mark the task Done or claim five seconds from work counts or instrumented diagnostics.

### Finite native input groups

**Measured outcome: rejected.** Task46 preserved its targeted correctness/retirement guarantees but slowed five of six complete mounted pairs; median public acquisition rose 52.08% idle and 109.80% active. No production integration or original-journey screen follows. ADR126 records this disposition; the design below is the retained experimental contract. Task51 selected the reviewed [complete project-index key owner](2026-11-14-task-40608-project-index-key-owner-design.md) as a disabled first storage-command slice outside the Textual interpreter. ADR126/230 are amended before implementation; no production routing or performance result follows.

Task44 exercised the lab metadata batch throughout both original journeys, with
positive retirement and no fallback, but measured 154.8950/164.7428 seconds of
adjusted README/edit time. The component improvement did not establish useful
journey latency. Task45 therefore selects a smaller implementation experiment
than rewriting the complete input frontend: group adjacent native operations
behind the existing Windows facade and retain the Python traversal/read policy.
[ADR126](../../../backlog/decisions/126-complete-local-backup-and-recovery.md#task-40608--finite-native-input-groups-2026-11-14)
defines the selected ownership and compatibility boundary.

The selected scope starts only in an eligible ordinary warm reuse call after
the real counted lease exists. It covers original bootstrap traversal, ordered
bounded record reads, optional independent selector fingerprint, independent
admission traversal/native identity, qualification, actual registry lock/wait
and read, the conditional metadata observation, and original nested cleanup.
Grouping is suspended for qualification and the yielded original union; these
remain independent existing owners. Final parent provenance stays outside the
scope. Unknown/custom/ambient-owned cases use the original route before effects;
late changes refuse rather than replay.

Use one finite extension context and direct facade groups for adjacent
create/info/type, volume/device, root duplicate/close and metadata info/type
calls. Keep audit buffers, getters, CRT reads/closes, ACL projection, name
sorting, limits and Python predicates. Do not reproduce the rejected adapter's
broad Python definition walk around every scalar call. Original callback and
post-wait checks remain. The first callback-free selected entry action
suppresses recursive discovery before any callback-capable validation;
already-suppressed entries stay original, internal helpers receive context
explicitly, and guaranteed exit cleanup restores prior state. Native identity
and cleanup validation follow the same rule. Unrelated audit-hook work stays
outside the scope.

Prepublish selected resources and transfer/close attempts before interruption
or Python result consumption. Preserve original close nesting and exception
causes, retain uncertain records, and never retry a possibly completed close.
The security-allocation seam preserves default behavior and the standard
`lru_cache(maxsize=128)` wrapper/key. Expose the seam only for the exact cached
invocation, consume it at miss entry, pass creators explicitly thereafter, and
mask/restore it across every audit/callback. Qualification and yielded union
have no seam. No per-context key or filesystem-verdict cache is allowed.
Fixed CPython exception calls avoid
ctypes swallowing callback failures. Use the installed CPython 3.12.10 Windows
x64 ABI/CRT for this disabled diagnostic; other ABIs remain original.

The first causal comparison is current stock versus these input groups with the
original metadata union in both arms. Gate18, original QA/Trace/runner and prior
artifacts stay unchanged; stage the experiment separately. Prove bound/unbound
input equality, selector distinctions, mismatch skipping, original refusal
predicates, audit-hook refusal/reentrancy, source/ambient drift, physical lock
wait plus fresh post-wait observations, interruption/transfer/close uncertainty,
separate concurrent contexts and complete retirement before timing. Reuse the
existing mounted public-acquisition fixtures and sequential pair order, with
all setup/guards/callbacks/lease-close costs reported. Do not run another full
journey merely because the native groups alone are faster. This experiment
establishes no five-second claim or production binary support.
