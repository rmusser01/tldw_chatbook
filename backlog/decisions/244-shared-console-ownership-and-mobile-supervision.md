# ADR-244: Shared Console ownership and mobile agent supervision

Status: Accepted 2026-10-25 (design only; written review and corrections complete)
Date: 2026-10-25
Related Task: [TASK-36728](../tasks/task-36728%20-%20Design-shared-Console-ownership-and-mobile-agent-supervision.md)
Spec: [Shared Console ownership and mobile supervision](../../Docs/superpowers/specs/2026-10-25-shared-console-mobile-supervision-design.md)
Plan: [Shared Console mobile supervision implementation](../../Docs/superpowers/plans/2026-10-25-shared-console-mobile-supervision-implementation.md)
Supersedes: [ADR-234](234-served-mobile-companion-surface.md), its separate mobile execution/storage contract
Amends: [ADR-236](236-served-desktop-approval-relay.md), its mounted-card, browser-lifetime and Canvas-coupled relay contract
Implementation: Not started; existing production behavior remains until verified cutover

## Context

The operator needs to inspect and correct existing Console agents from a phone
over VPN, including individual subagents, and switch between existing chats.
Work must survive browser closure and network loss while `--serve` remains alive.
Served mode is first; native process attachment follows later.

The existing Console already supports viewless execution. `ConsoleRuntime` owns
the store/controller/bridge and accepted turns above the screen lifetime. The
current mobile service instead constructs a separate engine graph over three
databases under `user_data_dir()/mobile`. That isolates phone-created chats,
workspace bindings and run history from the normal Console. Its partial desktop
approval relay does not make those chats shared. Separately, served terminal
WebSocket teardown stops the owning AppService child.

`AgentRunsDB.reconcile_orphaned_runs` also assumes one app instance per data
directory: its default constructor sweep can mark another process's live runs as
errors. Consolidating paths without qualifying that recovery owner is unsafe.

The private control broker also disconnects on event-queue overflow, while its
relay drainer starts only on first mobile use. Applying owner-loss shutdown to
that unchanged transport would stop healthy agents after ordinary event traffic.

The storage split is an implementation divergence documented in ADR-234's
2026-10-16 amendment. It must be corrected as part of this work, not retained as
a second execution system behind an improved mobile UI.

## Decision

1. All served interfaces use canonical profile conversation, workspace and run
   stores. Remove mobile-only database selection and the separate mobile engine
   construction. Existing database responsibilities stay separate; phone-created
   chats become ordinary Console chats.
2. Keep execution in the existing Console runtime and domain owners. Retain one
   served child App/runtime per canonical profile/storage identity, containing
   that profile's open chats. A small served registry routes clients and live
   chat identities. The executing child/native process holds the OS-backed fence
   for the actual secured `agent_runs.db` resource before execution-capable opens,
   run recovery and dispatch. Profile tuples cannot partition a shared run file;
   validate aliases and safe creation identity. The parent owns routing only.
   Parent loss closes child admission and begins shutdown without releasing the
   fence until physical retirement. Native/served entry points, explicit sweeps
   and auxiliary execution writers must honor it; non-owning reads cannot sweep
   or migrate another owner's store. No new engine or scheduler is introduced.
3. Browser connections are revocable attachments. Retain the owning process and
   output drain across disconnect, and restore terminal modes (mouse, bracketed
   paste, cursor, alternate screen and sync) plus a full render on reattachment,
   without restarting the driver input thread or agent. Permit one terminal
   input/viewport attachment per Console at a time plus concurrent independent
   phone views. Stale attachments
   cannot mutate or tear down their successor. Server shutdown and explicit
   native lifecycle actions retain their cancellation boundaries.
4. Extend the existing authenticated mobile API and bounded process transport
   with runtime-owned snapshots, events and explicit commands. Control transport
   availability is independent of Canvas enablement; Canvas capabilities retain
   their existing session fences and feature revocation. Mobile view selection
   never changes the desktop active tab as an implementation shortcut.
   The served lifecycle starts private event draining before child publication
   and retains it without phone subscribers or mobile-surface use. Projection
   overflow drops/coalesces replaceable events and retains bounded reset state
   until a current snapshot reconciles affected views. It cannot close the host
   control channel, release its execution fence or trigger owner-loss shutdown.
   Native decisions and command responses/receipts retain their authority;
   genuine parent/control-channel loss still invokes bounded shutdown.
5. Commands identify the owning epoch/runtime, exact chat/session incarnation
   and applicable run/handle/round/revision. Reuse native admission, queues,
   hooks, workspace authority, permissions, decision resolution and subagent
   mailboxes. Add explicit-target seams where native active-tab operations need
   them; keep existing active-tab callers on the same underlying behavior. Phone
   Send owns immutable submitted text and checked target revisions. Extend native
   admission without capturing, clearing or invalidating on desktop drafts,
   attachments, prefill or staged RAG. Those phone composition features are not
   introduced in this release; normal committed chat context still applies.
6. Assign mutations stable command IDs and body-free receipts. Reconnection
   reconciles authoritative outcomes and a bounded snapshot/event boundary;
   it never blindly replays a mutation. Preserve unknown outcomes through owner
   loss instead of claiming exactly-once external effects. Server-issued command
   tickets bind action, target and epochs to an authenticated immutable five-minute
   monotonic submission expiry. Keep deduplication evidence for valid tickets;
   pruning never makes an expired ID executable. Retain unknown outcomes through
   review/recovery and reuse native turn/round identities where sufficient.
   Capacity pressure refuses new non-idempotent submissions without blocking
   exact-target native Stop. Ticket expiry does not cancel accepted work. Answer
   decisions once through their retained owner, including background/viewless rounds.
7. Consolidate legacy mobile data using existing maintenance/capture/recovery
   machinery with explicit coverage of legacy stores and referenced files.
   Journal source/destination identities, conflict mappings, per-DB committed
   batches and final verification. Preserve complete history, references and
   authority restrictions. Legacy source identity is physical, not implied by the
   active configuration; review ambiguous destination attribution and claim each
   source once with a receipt bound to its reviewed destination. Quarantine all
   imported automatic-work/result-delivery authority, including completed runs
   with undelivered results, until fresh native review. Preserve truthful statuses
   and delivery stamps; ordinary startup recovery and new manual turns do not
   approve imported chains. Fence writes through interruption; cut over only
   after verification. Keep originals as recovery copies; no automatic deletion
   or replay of imported pending agent work is authorized.
8. Build on the existing `/m` page, auth and static assets. Open on Chats and
   provide live transcript/decisions, main-agent controls, and direct subagent
   guidance/Stop with an always-visible target. Keep page-local navigation and
   drafts independent, make connection uncertainty visible, and queue no offline
   commands. Existing browser Library reading remains available.
9. Scope phone control first to the served runtime. Native discovery/attachment
   requires the later phase; exclusive native-or-served execution per canonical
   profile is the first release's coexistence limit. An active native owner or
   another served server prevents competing execution, while ordinary read-only
   access can continue without recovery sweeps. Never treat unregistered native
   rows as remotely controllable or infer owner death from a timeout. Existing
   dispatch recovery and maintenance admission remain separate authorities.

## Alternatives considered

| Alternative | Reason rejected |
| --- | --- |
| Keep mobile databases and add more desktop relay features | Preserves split chat/workspace identities and a second engine that can drift from native hooks, permissions and runtime behavior. |
| Point the mobile engine at the desktop DB paths | Sharing rows does not share live controllers, pending decisions, mailboxes or cancellation; duplicate runtime ownership remains. |
| Move agent execution into a new headless daemon | Viewless execution and its native owners already exist. A new daemon would expand the task and duplicate established runtime behavior. |
| Make the character-grid terminal the primary phone UI | Does not provide adequate touch navigation, reading or explicit per-worker targeting for this use case. |
| Switch desktop active tabs to implement phone commands | Races with desktop input and can apply the wrong draft, settings or command target. |
| Persist and replay every disconnected command | Can repeat uncertain sends, approvals or tool effects. Reconciliation must preserve uncertainty instead. |
| Delete legacy mobile stores or leave them indefinitely active | Deletion loses user history; leaving a normal second write path preserves the rejected design. Verified consolidation and retained recovery copies resolve both. |

## Consequences

The implementation needs served-host lifecycle/attachment handling, a shared
ownership registry, explicit-target native seams, authenticated projections and
command receipts, and a recoverable legacy consolidation. This is more than a
responsive stylesheet change. It does not replace the agent loop or invent a
second runtime state machine.

Durable mapping/receipt changes require the normal schema version increments,
executed SQL migrations, exact schema-catalog and backup/restore coverage.
Legacy mobile backup coverage must be established rather than assumed.
Consolidation cannot claim atomicity across three independent DB owners; the
journal and startup fence govern partial progress.

The operator's app authentication continues to authorize phone control; VPN
access alone grants no app permissions. Existing local-file exclusions, project
instruction consent, capture/privacy policies and tool gates remain in force.
Server/child crashes remain normal durable recovery boundaries. UI snapshots,
timeouts and persisted run rows do not prove dispatch or external effects ended.

The spec records targeted storage/runtime/API tests and real served-browser
qualification. Full-suite execution remains opt-in. Written-spec review precedes
the implementation plan and atomic implementation tasks; this ADR records the
approved design direction without claiming shipped behavior.

### Native lifecycle implementation details

Legacy source and canonical rollback are two fixed native capture partitions under
one positively qualified exclusive maintenance session. The native archive format
assigns one schema version per owner; legacy81/8/25 and canonical82/9/26 share real
owner IDs and cannot honestly share that slot. Projection derives solely from the
admitted installed inventory. Both encrypted archives bind the original source,
reviewed destination, operation and exact verification identities; no import writes
or release of the enclosing source/maintenance custody occurs between captures.
Generic mixed-version refusal remains unchanged. Empty source has a no-data proof,
never an invented source archive or legacy database.

Missing canonical ancestors and stores use empty native private siblings. Typed
intent precedes creation, exact temp inode/parent/final witnesses precede native
no-replace publication, and final observation records the same inode. Recovery may
finish only the witnessed object at exactly temp or final. Unknown pre-witness
objects and foreign replacements remain retained and refused. No name-only
adoption, overwrite or expanded caller path authority is introduced.

Credential review uses disposable native capture before actual capture intent.
It returns body-free native issues without acknowledgement or publication. Exact
separately acknowledged manual-only issue sets are immutable per partition in the
reviewed preview, preparation and capture intent. Actual capture re-exports and
checks both issue sets and captured bytes; destination confirmation never grants
credential consent. Manual-only archives are coherent but do not establish
automatic credential rollback readiness. Required missing/unreadable/unsupported
material refuses. The conservative unused-default-key rule and its shared native
predicate are specified in ADR-126; it permits no fabricated keys or records.

The private lifecycle retains source-addressed claim custody and real native owner
retirement. It provides no public import success, completion, rollback terminal or
cutover. Cold source validation keeps Chat package server exports lazy so reading
Goal models cannot bootstrap config inside recovery; explicit public service
access preserves its normal runtime and admission behavior.

### Imported Goal digest provenance and native mutation guards

[ADR-227](227-native-console-goal-runs.md) binds a Goal request to its canonical
serialized request hash. Consolidation maps only declared typed binding and
owner references and recomputes that request hash using the native serializer.
The original checkpoint fingerprint hashes the complete GoalIterationResult,
which need not remain stored. The original artifact digest also identifies the
source result review. Preserve both as historical provenance; rewriting saved
checkpoint JSON cannot reconstruct either original execution or fresh approval.
The owning import batch receipt must retain original and mapped request hashes
and original digest/identity associations together with mapped records and holds.

Exact imported Goal, chain and attempt holds prevent checkpoint acceptance,
current completion verification, result approval, provisioning, resume and
recovery/state transitions from reusing or rewriting that evidence. Guards belong
at the actual native transactional owners, including direct GoalRunsStore callers,
and at service callbacks before external effects. Pure history/evidence reads,
unrelated fresh Goals/manual work and cancellation of unheld live work remain
available. No hold-release or new result-review workflow is introduced.

Source row fingerprints encode each actual SQLite scalar with its type: NULL,
integer, real, text or BLOB. BLOB bytes use hex and real values use Python's stable
hex float representation; unknown values refuse. Native AgentRuns result columns
can contain BLOB or TEXT values with identical printed representations, despite
passing exact schema/domain validation. Metadata digests and namespace identities
keep their existing format. An old prepared source fingerprint that no longer
matches requires native recovery/fresh review; it is never relabeled or accepted
through a lossy compatibility fallback.

Explicit native Goal payload erasure remains available under ADR-227's existing
eligible-status, unsettled-root and immutable-launch trigger checks. It retains
accounting/identity tombstones, every import hold and every batch receipt. Neither
import nor automatic recovery invokes erasure. Later completed-cutover verification
must respect native erased-payload tombstones and must never restore erased bodies
from retained source archives merely to reproduce a historical receipt.

### Native imported-history file custody

Consolidation retains the registered `agents.history` owner's captured bytes at
`user_data_dir(target_config)/agent_history/imports/<destination-identity>/<source-fingerprint>`.
Both components come from the same qualified operation's selectors and encrypted
capture. Each authenticated original root has a separate stable root-ID component;
its relative topology and original framed-record IDs remain unchanged. Legacy
store discovery does not treat the canonical import namespace as legacy input.
Ordinary native recovery discovery owns and captures the imported namespace.

A separate MaintenanceSession-issued file capability binds the actual live claim,
operation, native journal, selectors, captures and fixed destination. It cannot
accept caller paths, file lists or mapper callbacks, and does not expand the three
DB targets. Typed intent, staged, observed and verified evidence precedes/records
native no-replace publication of the complete tree. Staging remains private and
pinned. Recovery accepts only the exact witnessed object at temp XOR final, with
unchanged parents, topology and bytes; unknown pre-witness objects and foreign
conflicts are retained and refused. Existing bytes are never accepted by name.

File evidence establishes custody, not a committed database batch. Later native
owner receipts must bind original root/run IDs to mapped runs/conversations and
file locations/hashes before historical lookup becomes available. Lookup must
retain native recovery admission, byte witnesses, bounded paging and original-ID
primary/subagent filtering. No global fallback, scratch activation, writable root,
permission, hook, tool grant or blind JSON/path rewriting follows. Bindings,
exclusions and change-snapshot root meaning remain historical; change tracking
continues to belong to its separate owner. Unknown associations fail complete
preflight before any database import. The private file increment supplies this
contract; the run-owner increment qualifies actual committed-receipt consumers.

The private file path reuses the existing descriptor owner through extraction,
staging, capacity/pinning, recursive flush, native publication/qualification and
artifact observation. Optional private helper custody does not change ordinary
validation or grant paths. Uncertain closes retain native exclusion and are never
retried. Pre-import canonical absence is native `unused` producer evidence;
existing trees retain ordinary complete capture coverage. The existing restore
selection handles same-operation local topology; cross-profile `agents.history`
relocation stays `owner_relocation_unverified`. Public rollback after file
publication plus committed DB batches remains the later terminal qualification.
Custody reuse retains native per-component `O_NOFOLLOW`; it does not substitute
bootstrap's trusted-system-symlink traversal. The resource owner supplies close
bookkeeping only, never an alternate pathname resolver.


### Fixed Chat/Workspace consolidation and shared selector proof

The native mobile pending owner re-observes every prepared selector's content
fingerprint and physical witnesses: source, target and the complete reviewed
candidate cohort. This proof is shared by native database admission, each native
owner transaction and history-file custody. Same-byte replacement is a changed
reviewed object. Enrollment content/namespaces alone are insufficient; no
history-only or public-wrapper guard substitutes for this owner check. Ordinary
non-mobile admission remains unchanged.

Chat/Workspace import uses fixed table/column dispositions and a pure typed
cross-owner identity/dependency plan before effects. Unused identities survive;
reuse requires native schema and authority equivalence. Colliding UUIDs may map
deterministically; numeric identities stay integers and their allocations commit
with the owning rows, exact holds and receipt. Only declared typed references
map, including actual console_trace_calls.run_id. Original file roots, binding
locators, exclusions and instruction restrictions retain their meaning without
activation, permission grants or manual-send release. Unsupported populated
families or unresolved dependencies refuse prewrite with an actionable reason.

Only Console trace maintenance/graph epoch/migration/compaction and character
search state/revision receive operational-singleton disposition: exact typed
source row digests/identity associations plus authenticated capture provenance in
the Chat receipt, with no body-bearing shadow history. Destination lease, GC,
cursor and index authority remain native; ordinary mapped inserts retain their
normal monotonic/index effects. Busy/unresolved maintenance and populated
unsupported GC marks/scopes/runs refuse; no expiry repair. Every imported exchange
must have a complete readable semantic graph, independent of the destination
migration cursor. Schema version, FTS and sqlite_sequence remain native metadata
or derived state, never source schema replay.

The installed nine-entity sync frontier uses fixed typed payload references and
native mapped hashes. Native INSERT-generated current proof remains unique;
required representable v-1 live-message proof survives, and deleted entities
retain only body-free native tombstones. Missing, ambiguous or unrepresentable
required proof refuses. Original proof digests remain capture provenance, never
receipt body copies. Source Sync profiles, grants, outbox and receipts are not
imported, and consolidation does not reconcile Sync. Configured native restore
keeps its idempotent outbox policy; qualification must cover continuation,
previous hashes, deletion privacy and zero provider/tool dispatch.

Default remains workspace-default, with its Chats placement and no imported
runtime bindings. Source activation/timestamps cannot replace desktop selection;
Default policy/content conflicts require explicit prewrite refusal rather than
silent settings merge or ensure_default_workspace repair/deletion. Named
conflicts use native casefold/suffix uniqueness and retain original names in
identity associations. Citation identity is outside the singleton exception:
only ADR126's exact unused-default predicate permits inert seed disposition.
Populated key/fingerprint/provenance requires an actually readable native mapping
or prewrite refusal; no second context, key provisioning or invented approval.

The private Chat/Workspace increment cannot claim complete import. The run-owner
increment supplies Goal/AgentRuns mappings, all-owner preflight before file
publication and the first database batch, receipt-first resume and full closure;
public terminal recovery/cutover remains separate. ADR227's historical Goal
digests and explicit removed-payload tombstones retain the rules above. Retained
captures never authorize resurrection of erased bodies. These fixed rules avoid
replaying mutable process authority or silently discarding user history; failures
require revising the specific disposition while preserving recovery copies.

### Fixed legacy-exchange preflight disposition

A-chat refuses a populated `message_exchanges` table before imported rows, exact
holds, batch receipts or history-file publication. Already-normalized
`console_trace_*` graph rows and artifact BLOBs remain in the fixed import scope.
Native `LegacyTraceMaintenance.run_batch` deletes each old exchange only
after `normalize_exchange` reports verified semantic equivalence, and its cursor
is local to that database. Copying either the raw exchange or its source cursor
cannot establish readable imported trace history. The mapper neither invokes a
second normalizer nor mutates the source.

The actionable refusal identifies the populated legacy exchange family and
requires the supported native source-normalization path, followed by a fresh
review/capture. It does not promise that a repair command exists or authorize
writes while the current consolidation operation is pending. Retirement/recovery
of that operation belongs to the terminal recovery workflow. The cost is an
explicitly blocked legacy capture until its native semantic history is ready;
the retained source and authenticated captures remain unchanged.

### Private typed Chat/Workspace planning boundary

The pure planning increment reads only authenticated fixed owner members from the
admitted source and rollback captures. It verifies the actual capture declaration,
sealed archive and member size/hash before restricted disposable SQLite validation.
Its connections and descriptors belong to the existing maintenance resource owner;
uncertain physical retirement retains staging and native exclusion. A returned
context manager alone is not proof of retirement. No caller path, connection,
factory or authority predicate substitutes for these native checks.

The plan records fixed table-and-column dispositions and typed identity domains,
including composite graph references and integer ordering. It preserves native
message metadata and checkpoint discriminators, provider continuation, normalized
trace BLOBs, generated-memory lineage/selection proofs and the nine sync frontiers.
Only declared ownership references map; opaque provider identities and historical
digests retain their meaning. Mapped generated-memory prefix proofs use the native
persisted-lineage representation. Persona and permission-profile references require
actual captured native authority equivalence, never equal names or labels.

This boundary produces immutable transformations and body-free source associations
for later native owner transactions. It produces no imported row, hold, receipt,
file publication or successful public composition. The later write increment must
revalidate destination identity reservations and commit rows, identity/provenance
associations, exact holds and receipt atomically per owner. AgentRuns retains full
all-owner preflight and Goal/run handlers; terminal recovery retains cutover. This
split makes native transformation proofs reviewable before effects, at the cost
of serial review and revalidation; it does not reduce required history/settings
coverage or turn source evidence into current execution permission.

Captured raw authority reuse keeps the fixed capture partitions. Canonical raw
material may satisfy a source Persona/profile reference only when authenticated
native owner/member provenance proves it covers that exact source reference and
destination meaning. Same selector, path, label or ID alone proves nothing;
different selectors may share one actually proved owner. Equivalence includes the
complete native effective authority: named-profile inheritance, default/global
kill policy, lifecycle and provenance. No raw policy or grant is imported.
Missing, unknown, ambiguous, changed or non-equivalent authority produces an
actionable body-free exact-reference blocker before effects, without rejecting
unrelated history or otherwise supported defaults.


### Stable native source-claim control discovery

The fixed source-addressed claim/lock declarations retain stable discovery scope
across native missing-to-created transitions, including distinct enrolled selector
aliases. Only the exact safe intentionally-excluded control objects produced by
native mobile inventory are outside discovery's automatic physical-alias grouping.
Every declaration remains in the complete scope digest; unknown, unsafe and other
recovery controls keep ordinary classification. This does not declare a shared
payload identity or change generic alias/capture policy.

Native claim contents, operation/journal/session identity, current source fingerprint
and the complete selector cohort's physical/path/ancestor/mode witnesses remain
mandatory at their existing boundaries. An excluded control row cannot authorize
adoption of a substituted claim, release old ownership, or suppress payload/shared
owner changes. Qualify actual distinct-selector encrypted capture and authenticated
SQLite reads before typed planning; this correction supplies no import, receipt,
history publication, public recovery or cutover authority.


### Authenticated fixed SQLite alias member selection

After the native control-discovery correction, distinct-selector encrypted capture
reaches the private SQLite reader with multiple native declarations for one fixed
owner file. The reader may select one deterministic logical-ID representative only
after fresh native discovery/projection and current physical identity prove the
same shared owner, and every relevant authenticated producer matches its native
logical ID, owner, status, dependencies and shared group. Every included alias must
have its exact authenticated member with equal size/hash semantics. Missing,
changed, mixed or unproved declarations refuse; owner/path/group labels alone do
not qualify a shared resource. All-native-unused declarations without payloads
retain absence and grant no shared-file authority.

This consumes the existing native capture's alias semantics; it does not widen
fixed families/paths, change complete inventory or source/cohort/claim checks, or
issue additional connections, schemas, transactions or publication authority.
Captured-SQLite custody, physical retirement and quarantine remain unchanged.


### Fixed captured Persona and permission-reference proof

The private identity/Workspace planning increment may read only the `personas`
and `mcp.permissions` raw owners already included by the canonical capture
partition. The actual live captured-mobile, maintenance, source-claim, operation,
selector and capture-cohort objects issue this fixed read/proof scope. No caller
path, connection, callback, predicate or asserted shared/authenticated flag
confers authority; no raw owner is added to the legacy capture partition.

Before returning immutable private reference evidence, the scope authenticates
the exact producer/member/schema/bytes and requalifies source and destination
owner coverage, physical identity and current native generation/root-binding/
activation provenance under maintained exclusion. A same-selector or shared-owner
match still requires that complete proof. Equal paths, labels, IDs or logical
scope digests alone are insufficient. Retained backups/history and inactive or
unreviewed generations are recovery evidence, never active permission policy.

Use installed pure Persona/MCP validation and resolution, including inherited
defaults, global kill policy, lifecycle and provenance. A policy digest alone
does not represent full effective authority. A necessary narrow read-only native
gate must preserve the existing execution gate and its error/default behavior.
Do not instantiate/load/repair authority stores or borrow execution admission
and ambient effective-profile state to inspect the captured material. Missing,
unknown, mismatched or unprovable dependencies yield body-free exact-reference
prewrite blockers while unrelated supported history/defaults remain supported.

The scope reuses admitted archive/custody primitives with strict nofollow checks,
actual retained pins, bounded bytes and sticky uncertain retirement. New reached
acquisition/read/substitution/retirement boundaries require actual native proof,
including captured shared-owner and active-generation positives and absent,
mismatched, inherited-policy and lifecycle negatives. Bodies remain private and
out of logs, receipts and public plan metadata. This grants no policy activation,
raw-policy import, key provisioning, owner write, hold, receipt or publication.

This finite seam supplies an already-required proof that the SQLite-only reader
cannot express. It does not replace the complete fixed table/column manifest,
typed identity reservations or Workspace transformation plan; C/T/S remain
required pending handlers and the integrated plan is reviewed before owner writes.
If native proof cannot be retained, the dependency remains refusing; qualification
may require a narrow correction or serial extraction before the plan completes.


### Workspace backfill frontier and audit payload disposition

The installed `workspace_agent_backfill` assistant_defaults key is a database-wide
completion frontier. Preserve its exact source row/state as typed historical
capture provenance, while retaining canonical marker rows and pending work. Do
not import the source marker, coerce explicit-none flags, provision raw authority
records, or add a per-workspace lifecycle. Preserve source default/null/archive
flags. For imported named, nonarchived rows with NULL defaults and explicit_none
false, differing source/canonical completion booleans refuse before effects with
an exact-workspace blocker. Matching states preserve native eligibility. Default,
archived, explicit-none and already-referenced rows are exempt; marker mismatch
alone cannot block them. Unknown populated source marker keys refuse explicitly.
Later owner-write/recovery qualification must verify actual post-import startup
and no accidental authority creation; a pure plan proves no startup effect.

Workspace handoff audit rows support their fixed fields and typed audit/workspace
identities only when manifest_json is a valid empty object, retaining the original
text and original/mapped identity association. Nonempty, invalid or unknown
manifests refuse with the exact audit/workspace reference before effects. The
installed schema supplies no versioned manifest reference grammar; no guessed
recursive rewrite, silent drop or invented grammar is safe. This narrow refusal
leaves ordinary supported Workspace and unrelated history semantics intact.

These dispositions preserve both databases' intent without replaying a global
completion marker or pretending opaque audit references are understood. Some
legacy rows consequently require a supported explicit default choice, matching
native completion state or a separately reviewed manifest grammar before import.
Originals and captures remain intact when preflight refuses.


### Typed identity and ordering reservations

Unused genuine identities survive. Global memory-selection and fleet sequence
positions are ordered domains, separate from stable selection/message identities:
allocate their complete source order deterministically against destination
occupancy/native frontiers rather than preserving an unused position that would
reverse history after a lower collision. Keep original/mapped position
associations; C/S may rewrite only proved typed references. Do not infer relative
chronology between profiles. Per-parent/per-segment ordinals remain local to
their mapped owners and are never shifted as a global sequence. Unrepresentable
ordering or relationships refuse with the exact source association.

Native validators govern every identity's format/version. Deterministic collision
identities must satisfy those actual validators; a blanket UUIDv5 transformation
is invalid where the native record requires UUIDv4. Ordinary integer identities
remain integers; no ID-looking strings or arbitrary JSON values are rewritten.


### Fixed I/C/T/S native family ownership (2026-10-26)

Ruling: Set finite family ownership from the approved Console-history/dependency scope, not the retired draft's zero-row blockers or a generic whole-library merger. I supplies explicit literal table/column dispositions and typed domains; C owns required Chat history/context/settings and its native dependent content: the nine frontier entities' actual rows (messages, conversations, notes, character_cards, keywords, keyword_collections, chat_dictionaries, world_books, world_book_entries), their typed note/folder/keyword/collection/link and character/world-book/dictionary/conversation associations needed for native closure, Console response-rule revisions/bindings and their actual typed supporting records, and Canvas documents/revisions plus declared message/Canvas provenance. Preserve native body/version/deletion/restriction semantics; no arbitrary extension/body-string ID rewriting. C also owns typed logical assistant/Buddy/visual associations when required by chat/settings; physical asset reuse/closure must be actually proved through existing admitted captured native owners, otherwise an exact missing/unprovable dependency refuses before effects. This does not authorize I to widen raw capture/read owners or invent an asset publisher. Any concrete missing native file/proof seam is routed before implementation. T retains complete normalized trace/composite/BLOB and citation semantics; S retains nine native sync-frontier/proof transformation and integrated qualification, consuming C's actual entity maps. A-runs retains actual run/Goal handlers and ALL-owner preflight/closure. Mark these C/T/S families pending/refusing until implemented; absence of their current handler is not permanent unsupported status.

Standalone non-Console study/learning and visual/asset-library families without a supported Console dependency handler may have explicit unsupported-populated dispositions in this release, with exact family/record blockers and originals retained. Source Sync profiles/grants/routing/control/cursor/outbox/activation/reconciliation state remains non-importable; this includes Notes publication/organization sync intents/checkpoints/adoption/frontiers whose replay changes external dataset authority. Populated unsupported protocol/control/receipt families refuse explicitly rather than being silently omitted, replayed or treated as disposable caches. Ordinary Notes content/organization needed by the nine entity graph is C data, distinct from those Sync protocol obligations. Existing narrowly enumerated native derived/seed/singleton exceptions remain exact; no prefix/count/FK-only rule can classify a table or convert an unknown producer into a seed. A known inert installed seed must use an exact native predicate; every other populated unknown family refuses. If one allegedly standalone/unsupported record is required by an imported chat's native context, surface the exact dependency; do not relabel a required chat setting unsupported to avoid its handler. C may route further concrete native semantics, but no generic recursive copy or blanket library expansion is authorized.

Reason: spec7.3/7.6 requires conversations/settings and dependent records for readable history, explicitly permits unknown populated preflight refusal, and the native nine-entity frontier at ChaChaNotes_DB21514–21566 includes notes/character/organization/dictionary/lore rows. Hydration722 uses real character identity, dispatch checkpoints60 use typed note/media scope and message metadata declares Canvas origin; these are required dependencies, while Notes publication/org sync tables carry external routing, immutable requests and authority frontiers. Cost if wrong: a newly discovered Console dependency needs a narrowly reviewed handler/file-proof path or a conservative exact-reference blocker, and C may need a source-grounded internal split; standalone library/Sync-protocol sources may remain blocked. No required chat data is silently dropped, source authority replayed or public cutover permitted before complete closure.


### Fixed captured schema catalog for I (2026-10-26)

The private native `_CapturedSQLite.schema_catalog()` may return immutable exact table/column tuples from its already admitted, schema-validated fixed family connections. It accepts no SQL, path, table, predicate or reader callback. Verify the actual retained pin and live captured/session/operation scope before and after each family metadata batch, and retain the existing SQLite/resource limits, custody and native control checks. This supplies I's literal manifest only; it adds no data/owner authority or capture coverage.

The reached full-manifest test (`task-8-c2-i-manifest-green-output.txt`) failed `sqlite_resource_limit` at `mobile_captured_sqlite.page` after 81.47 seconds while repeating metadata reads for 230 literal source/canonical table definitions. Batching only native schema metadata replaces those redundant page calls without extending timeouts or falling back to broader data reads. Actual catalog equality, unknown column/schema refusal and connected custody qualification remain required. If incorrect, correct this narrow method and requalify before I review.


### Evals retained metadata during owned mobile capture (2026-10-26)


Ruling: Fix the reproduced Evals retained-definition passive-read boundary narrowly within current I before claiming restored-generation proof. Reuse the existing native _mobile_discovery_pending admission predicate for the exact actual _DiscoveryScope/MaintenanceSession/journal/prepared selector+namespace cohort, only for the current operation's recovery_pending case; no new token, caller assertion or ordinary execution admission. Root read _retained_definition_paths358–360, its subsequent binding/committed-or-rolled-back receipt/activation/manifest/root validation, bootstrap.startup_permission501–542 and _mobile_discovery_pending662–688. The native pending fence must continue refusing ordinary startup, eval execution, unrelated/foreign/stale discovery and recovery_scope_uncertain; every existing paired-generation, selector/root binding, receipt/plan/manifest/hash/retained-original check must still run unchanged. No adapter filtering, inventing empty definitions, skipping issues/activation, configuration reassignment or broad bootstrap permission change. The existing helper can observe its native data through this positively issued maintenance read exactly as other passive generation owners do. Qualify the actual restored/MCP-reviewed capture then raw proof, and wrong/no/foreign scope/selector/namespace/pending-state negatives plus targeted ordinary Evals retained/rollback discovery controls. No whole-suite or generic control-custody retrofit; inherited control-reader lifetime remains explicitly unqualified, while all new I resources retain positive custody. Snapshot/amend ADR policy before this Evals branch's code/tests, accurately listing existing dirty I files; keep I BASEb062 and complete I review range. Reason: real complete preflight succeeds, then the same operation's own pending revalidation fails before I on a read-only retained metadata adapter; existing native mobile discovery already supplies the exact bound passive exception without granting startup or eval authority. Cost if wrong: the narrow Evals discovery branch needs correction/requalification and restored-generation sources remain blocked before import; no broader execution, capture owner or source mutation is authorized.


### Captured Evals isolated descriptor verification

The Evals passive discovery path may read the same verified isolated-profile descriptor during its own native mobile recovery_pending operation. Factor the existing committed association, journal receipt, catalog identity/digest, activation-owner, prepared mapping and root-coverage checks once. The ordinary launch-state/descriptor entry and all launch, selection and requirements consumers retain their startup-permission gate and refusal behavior. A separate private Evals metadata entry may pass only the actual native current-mobile discovery/session/journal/selector/namespace predicate; it accepts no bypass flag or callback and does not grant execution. All descriptor validation remains mandatory, including exact selected config/profile/control and paired witness. Missing, foreign or retired scopes, mismatches and recovery_scope_uncertain refuse.

This corrects a second reached boundary after the first Evals pending gate: its shared isolated descriptor also enforced startup admission during passive capture. The alternative of weakening the shared launch gate is rejected because it would admit ordinary profile consumers. Qualify unwrapped restored/MCP-reviewed capture and raw reference proof, ordinary launch/requirements refusal under pending, and focused profile/catalog/retained controls. Existing inherited internal control-reader custody remains independently unqualified.


### Native note-link token relocation during private consolidation

The captured Note identity map may relocate only the identity capture group of
the installed `ChaChaNotes_DB._NOTE_LINK_TARGET_RE` grammar in `notes.content`.
Preserve the surrounding `(note://...)` syntax, visible link text and all other
body bytes exactly. Process every occurrence, including repeated links and
self-links; native backlink extraction excludes self-links from its edges, not
from the body's identity tokens. An unknown or unproved target, or a token that
cannot retain the native grammar, produces an exact body-free source/dependency
blocker before effects. Never guess a canonical match, replace arbitrary prose
or recursively rewrite extension JSON.

This is necessary because native `replace_note_links` regenerates `note_links`
from the note body on every write. Mapping only the relational edges would
silently revert their meaning on the next save. Validate the mapped body through
the installed extractor, native write and backlink reader, including collisions,
repeated/self/deleted notes and unchanged non-reference text. Later native sync
proof construction uses that actual relocated body, while source capture retains
the original body/digest association without a body-bearing receipt shadow or
replaying source Sync authority. A failed proof blocks writes; originals remain
retained.

Contemporary state at this amendment: released HEAD
`d74175b0fd319aef9d17f94d697bbbfd7fdf0895` already contains the private captured
SQLite reader, literal table manifest, typed identity reservations, Workspace
planning and captured Persona/MCP reference proof. No content-graph production
module or tests exist yet; only the private field-ownership matrix and initial
source-boundary handoff were written. Existing controller TASK-36766 metadata
remains outside this amendment. This records current policy before its new code
and tests; it does not reconstruct earlier A-files or hook chronology.


### Historical Notes locators in captured content planning

Retire only the legacy Notes `file_path_on_disk`/`sync_root_folder` presence
query from `_inspect_store`'s external-reference check. Those fields accompany
the complete DB note body and preserve historical locators, sync hashes and
state; their presence alone proves neither missing body bytes nor activated
file authority. Preserve all their values in the private content plan without
filesystem probes, external reads/copies, hash falsification, activation or
Sync protocol replay. Keep the external-reference default, dictionary and
visual-asset queries, and every preview count/fingerprint/schema/physical,
capture and claim check. Populated unsupported source protocol obligations
continue to refuse through the fixed manifest.

The actual preview currently refuses before preparation and encrypted capture
for this historical-field case. Qualify the real preview → native preparation →
encrypted capture → content-preservation path with unchanged profile bytes,
modes and journal events during planning. Do not simulate admission by changing
a fixture guard. Native `notes_sync_legacy.snapshot_legacy_notes_sync` performs
metadata/identity/freshness probes and creates PAUSED roots, CANDIDATE bindings
and pending-review plans; explicit fresh review/activation remains necessary.
Later startup qualification must separately exercise the marker-absent native
migrator and canonical cutover-marker semantics. Pure content planning does not
prove startup behavior or grant filesystem access. A genuinely unrepresentable
dependency still blocks before writes with its exact source identity.

Contemporary state: HEAD `0faa12dc91` already records native note-token policy.
Uncommitted partial `mobile_content_graph.py`, its captured-reader entry point,
`test_mobile_content_graph.py` and a test-only native seed callback now exist.
Two captured Note token tests passed; two native core tests reached missing
entity/uniqueness behavior. Earlier NOT NULL/attachment-guard fixture failures
are not product RED evidence. The historical Notes fixture reaches the actual
pre-capture presence refusal. This amendment precedes changing that guard; it
does not describe the current partial content handler as complete. Original
`d74175b0fd319aef9d17f94d697bbbfd7fdf0895` review base and all subsequent private
plan/integrated/write gates remain in force. Controller TASK-36766 metadata is
outside this amendment, and historical hook limitations remain unchanged.


### Imported response-rule history retains native execution restrictions

The private source-to-canonical projection follows
`DB/recovery_response_rules.prepare_imported_rules` and ADR-232 decision 10:
source enabled bindings become disabled with `binding_revision + 1`; disabled
and excluded bindings keep their state and revision. Source pending assessments
become cancelled in both relational state and the closed assessment JSON state
field, using the native canonical JSON encoding. Completed, cancelled and stale
assessments retain their state. This is an import; the exact local rollback
exception does not apply. Preserve authenticated original rows separately and
retain typed original-to-mapped associations and historical RuleSource fields,
including effective digests. Known conversation/message/branch references map
through their authenticated domains. Unproved global profile namespaces still
produce exact dependency blockers; historical strings never activate a scope.

Do not invoke the mutating native preparation helper against captured source or
canonical owners, change canonical/rollback rows, execute checks or repairs,
create receipts, or invent fresh authority/digests. Inconsistent native records
produce body-free source blockers. Qualify this pure projection with one native
import-restriction comparison in a disposable test owner and unchanged source
and canonical snapshots. This clarifies earlier broad state-preservation prose:
historical originals stay exact; imported execution-bearing states obey the
installed restriction. If wrong, correct and requalify this finite projection
before writes while retained originals remain untouched.

Contemporary state at HEAD `63ea05de48`: dirty C1 core/Note/message graph and
Canvas modules, captured-reader entry, narrow Notes preview guard change,
native fixture extension and three C1 test modules already exist. The rule
module now implements independent revision/validation/fixture/draft mapping;
binding and assessment restriction projection has not been written. Actual
rule RED reached missing validation rows/global-scope blocker in 111.98s after
native capture; Canvas native import/read consumers are still pending. The
earlier malformed Canvas fixtures failed native guards and are not product RED.
This snapshot precedes restriction-projection code and test changes. The exact
C review BASE `d74175b0fd319aef9d17f94d697bbbfd7fdf0895`, full integrated gate,
controller-only TASK-36766 metadata, and historical hook limitations are retained.


### Captured normalized trace reconstruction and historical locators

Private trace planning invokes native read-only reconstruction on the existing
retained immutable source and canonical Chat connections. Its internal transaction
adapter yields real cursors without BEGIN/COMMIT, authorizer relaxation or a new
reader role. No caller SQL, path, connection, cursor, factory or callback escapes.
Every retained member keeps its pre/post custody, named-object, hash, resource,
scope, claim, journal and cancellation checks and original limits. Bounded,
recursively immutable results bind the actual manifests and exact message/trace
inputs; neither record construction nor sample tests establishes readability.

Mapped preservation requires a closed field-specific isomorphism over every native
reader input: graph incidence, ordering and lineage; selected message fields and
sidecars; revisions and retired locators; frozen policies, bindings and masks;
headers and their declared structural metadata; exact artifact bytes and digests.
Native verification, omission, sanitizer-unavailable and uncertainty results retain
their original associations and cannot be upgraded. Uncovered or changed semantic
inputs refuse before effects. Disposable mapped native consumer tests qualify the
transformation independently; they provide no per-input capture or authority proof.
There is no production mapped SQL sink or staging writer. Later owning writes must
still establish legal native lifecycle and trigger ordering.

Message and conversation identity reservations also cover historical locators in
actual semantic-revision source_message_id/source_conversation_id fields. A live
identity and all its historical revisions share one reservation. Allocation checks
canonical live and historical occupancy plus every source/reserved target, rejects
ambiguous source ownership, and preserves predecessor, sequence and uniqueness
relations. A retired live_message_id remains NULL with its original retirement
stamp; no message, conversation, content or live authority is invented. Historical
materialized-envelope BLOBs retain their bytes and audit identities. Only declared
structural header references map; matching text inside content is not a reference.

The retained scope may invoke native citation context, shared no-fingerprint-history
and local-authority validators. Their bounded immutable result states the captured
context and SQL invariant only. It does not prove credential UNUSED status, backend
or key availability, readable fingerprints, namespace equivalence, or permission to
adopt canonical identity. Credential coverage and actual source/canonical namespace
and readability proof remain separate requirements. Missing proof produces exact
dependency blockers; canonical identity is never overwritten, keys are not
provisioned, and C-owned authority slots are discharged only with their full proof.

These constraints avoid a second production reconstruction database while requiring
complete input closure. An omitted native dependency could change history or expose
content, and a mistaken historical identity or key assumption could misattribute
ownership; any such gap must therefore refuse before effects. Plans remain private
and noninsertable pending integrated, run-owner and owning-write qualification.


### Opaque trace grouping and normalized legacy reconstruction

A normalized call's whole nonempty run_id is an opaque historical grouping
identity, separate from AgentRuns execution ownership. Runtime UUIDs, paired
runtime tags and already-normalized legacy tags share this native string grammar;
no colon parsing or textual equality creates an execution reference. Each source
group receives one deterministic reservation against canonical groups and every
source/reserved target, preserving call_sequence and typed original/mapped history.
Explicit AgentRuns dependencies retain their separate owner domain.

The fixed retained-connection reader also invokes the native LegacyTraceNormalizer
read_calls consumer for already-normalized legacy_snapshot calls. It uses the same
immutable native cursors, custody checks and original resource budgets, without
normalization, batch state, migrations or writes. Closed-input proof covers all
legacy metadata/request/response artifacts with exact bytes and native omissions,
uncertainty, verification, legacy_snapshot provenance and recorded_call_only
chronology. Historical metadata run_tag remains exact audit data inside its BLOB.

Original and mapped raw reader order remain auditable. Group reservations may change
lexical cross-group iteration without inventing cross-group chronology. Recorded
timestamps, within-group sequence, native lineage/event order and actual
ConsoleTraceProjection selection must survive mapping. Native semantic-key,
selection, ordering, tie and duplicate controls must establish that remapped groups
or call identities lose or merge no captures. Originally unavailable or uncertain
history remains so. A concrete collision that cannot preserve consumer behavior
with this fixed map and exact bytes produces an exact prewrite conflict; it does
not license a generic remapper, BLOB rewriting or invented chronology.

This prevents trace grouping from acquiring execution authority and prevents silent
history loss through viewer deduplication. It grants no source mutation,
insertability, normalization permission or later lifecycle-trigger bypass.


### Captured local-character association projection

A nonNULL C-owned local-character authority association may map from the captured
source default namespace to the captured canonical default namespace only with
native per-input ownership proof and the complete accepted C conversation/card map.
The fixed retained adapter obtains local authority through the native context
validator and invokes CharacterConversationSearchRepository against the actual
joined conversation and card. Its resolved-row predicate is necessary but not
sufficient: caller restrictions for local runtime, character kind, canonical decimal
assistant identity, captured authority, live joined card and conversation visibility
and deletion must also hold, with exact row provenance.

The mapped association uses the accepted mapped conversation/card/reservation and
canonical native context; it never substitutes a similarly named or occupied
canonical card. The same native predicate and complete caller filters qualify the
mapped inputs. Independent actual original and mapped native transcript-preview and
revalidation controls must cover distinct contexts, occupied card-ID remapping, and
foreign, unresolved, missing, deleted and explicit-None cases. Disposable mapped
consumers qualify the transformation; they do not create source authority.

Ownership and present UI eligibility remain distinct. Provably owned inactive
history retains its exact deletion and visibility state; two false predicates do
not prove foreign, missing or ambiguous ownership. Any unproved association keeps
an exact body-free per-conversation blocker and original provenance. Explicit None
stays None; remote and persona identities never enter this local-character rule.
Canonical local identity is unchanged. Citation profiles, fingerprint keys,
permissions and filesystem bindings retain their separate proof obligations.

This finite projection prevents a foreign or unavailable character chat from
becoming another character's live chat. No generic adapter, caller cursor or SQL,
indexing, repair, key provisioning, capture widening, owning write or authority
activation is granted; every existing retained-scope fence and resource limit
continues to apply.


### Preconstruction bounds for captured native trace reads

The captured trace operation owns one nonrenewable aggregate 16 MiB budget for
inputs, proof and native materialization. One fixed private budget object propagates
explicitly through its actual native reader, service, semantic projection and legacy
read paths. Optional private parameters default to None so ordinary native consumers
retain their existing behavior; no product option, caller callback, generic factory,
global state, cache, per-call allowance or budget renewal is introduced.

Budget reservations precede decoding, cloning, concatenation and repeated-reference
materialization, and every list, dictionary and result accumulation. Coverage includes
header components, semantic envelopes and batches, continuations, thinking, media and
base64, system composition, legacy messages and call tuples. A bounded raw value may
be inspected; charging an already completed unbounded reconstruction is insufficient.
Nested consumers receive the identical shared object and cannot silently use an
unbudgeted path.

A distinct body-free exhaustion exception propagates as a resource refusal. Native
reconstruction-unavailable handling, sanitizers and uncertainty conversion never
swallow it or represent exhausted resources as uncertain historical content. Existing
custody, context, schema, authorizer, cancellation, deadline and step restrictions
remain independent and unchanged. Fixed trace tuple acquisition uses actual retained
cursors in batches no larger than 1000, with full installed metadata and all six
members' complete pre/post checks; there is no general reader or mapped SQL writer.

Repeated-reference and pre-decode refusal controls must exercise the real captured
entry. Bounded complete, partial and normalized legacy output and ordinary unbudgeted
native callers require focused preservation controls. A missed nested path risks
excess allocation; overcharging or catching exhaustion risks false historical loss.
These are implementation qualification requirements, never new import authority.

A separate incidence-size preflight was considered and rejected: it would duplicate
native projection semantics without a complete faithful bound for JSON, base64,
masks, sanitizer and legacy expansion. The shared explicit native allowance keeps
those reservations beside the operations that allocate the representations.

### Retired world-book entry identity occupancy

The existing integer `world_book_entry` identity domain includes live rows and
native hard-delete proofs from the fixed captured `sync_log` table in both source
and canonical partitions. A qualifying proof has the exact native entity,
operation, version and body-free `{id, world_book_id}` payload. Its IDs obey
SQLite integer grammar and `entity_id` equals the native text cast of `id`;
booleans, coercions and fabricated positive-only constraints are invalid.

One native identity reserves once, including when a retained tombstone shares a
live identity. Allocation excludes canonical retired identities as well as live,
source and newly reserved targets under the existing signed-integer bounds.
Each original proof retains its parent association and authenticated provenance;
its `world_book_id` must resolve through the existing typed parent map or yield
an exact dependency refusal. Malformed, body-bearing or ambiguous required proofs
refuse before effects.

A historical reservation creates no live entry, parent, body or authority and
does not prove frontier completeness. Native retention and unique INSERT current
proof remain unchanged. The fixed captured read interface is sufficient; no new
SQL, reader, capture or callback capability is introduced. Pure plans remain
noninsertable until their integrated and native owning gates are qualified.

This union is necessary because native hard-delete retention keeps deletion
proofs after their entry rows disappear. Reserving only live IDs could associate
an imported deletion with a different canonical entry. Actual native deletion,
body privacy, retired/live collision, parent closure and no-effects retirement
checks qualify this rule; payload samples alone do not.

### Shared trace and sync proof materialization allowance

The fixed S composition owns one existing 16 MiB trace-read allowance. Its
private call into T passes that same object; subsequent fixed sync input,
JSON/proof, serialization and result construction consume its remaining
allowance. Standalone T with no supplied private allowance still creates its
original one. Captured public entry points take no arguments and return only
immutable proof values, never the mutable allowance.

New sync raw-row aggregation, decoding, cloning, serialization and repeated
reference expansion reserve before construction. A bounded raw page does not
permit an uncharged aggregate. Exhaustion propagates as the existing distinct
body-free resource refusal, never as uncertain or unavailable content. There is
no second allowance, reset, cache, larger cap or unbudgeted fallback.

This bounds the T and S representation work receiving the allowance; it does
not measure Python heap or retroactively bound existing R/I/C allocations.
Native six-member custody, scope, claim, journal, schema, cancellation, deadline,
step and authorizer checks remain unchanged. Actual composed predecode and
repeated-reference boundaries, standalone T behavior and no-effects retirement
must be qualified with the real allowance. A smaller injected limit is not
capacity evidence. This prevents disconnected proof phases from expanding the
same operation outside its existing bound without introducing a new reader or
capability.

### Exact pre-feature AgentRuns source24 compatibility

Fixed captured consolidation admits only native-validated legacy Chat81,
Workspace8 and AgentRuns24 or25; canonical AgentRuns26 is unchanged. The actual
authenticated owner declaration and observed installed catalog version must agree.
Source24 bytes, schema and version remain intact: no execution constructor,
migration, repair or synthesized25 catalog is permitted.

The sole version-bound table absence is `console_command_receipts` in an actual
legacy runs24 member authenticated by the active captured scope and validated
against its complete installed native catalog. Fixed manifest coverage accounts
for that absence, and typed identity/history occupancy may consume it as absent.
No table, row or proof is fabricated. Physical schema metadata remains exact and
public fixed `page()` still refuses nonexistent tables. Runs25 receipts remain
real A-runs-owned history. A missing25 table, unexpected24 table, declared/observed
version disagreement, unknown shape/type or arbitrary read error must refuse.
Whole optional-member absence remains a separate existing rule.

All six member pins, hashes, names, ancestors, schema, session, claim, journal,
authorizer, cancellation, deadline, step/resource and physical-retirement fences
remain. There is no new reader, capture owner, SQL role, callback, factory, cache,
allowance, reset or cap change. Plans remain noninsertable. Qualification must
exercise actual native81/8/24 preview, preparation, encrypted capture, identity
and complete integrated planning, exact absence/version negatives, unchanged25/26
controls, unchanged source/canonical/journal state and physical retirement before
any owning write or public cutover is released.


### Fixed runtime-owner and command historical facts

The source automatic-work runtime-owner singleton is immutable historical
provenance only. Canonical runtime authority or its absence remains unchanged.
Reservation/attempt owner IDs and chain clock-owner, monotonic, wall-clock and
accounting anchors retain their original historical meaning. Equal owner strings
do not discharge exact imported holds or existing native guards.

Source25 command receipts retain all twelve original recorded fields, validated
with the native decoder. Original epoch, Console target, fingerprint, status,
reason, timestamps and native ID are historical facts. Shape validation cannot
authenticate the unavailable HMAC key or monotonic execution context. Separate
mapped associations cover only proved persistent native ownership references;
session, incarnation, handle, epoch and unproved turn/native-ID strings stay
opaque. Source24's authenticated receipt-table absence remains distinct.

Each historical receipt carries an exact unresolved passive-history/retention
duty. The later owning writer must qualify bounded complete retention and reads
outside live command capacity, control observation, invalidation, pruning and
dispatch. This pure disposition creates no storage schema, hold type, release
workflow or native consumer, and never copies these facts into the active table.

### One bounded pure Run/Goal preflight composition

A fixed no-argument runs_preflight() on the actual issued captured SQLite scope
owns one existing 16 MiB _TraceReadBudget. A private optional _read_budget argument
passes that same object through integrated planning and its existing Sync/Trace
chain; the same remaining allowance covers every newly introduced Run/Goal,
operational fact, history proof, identity-domain and closure representation.
Reserve before raw accumulation, decoding/model construction, clones, indexes,
repeated references, serialization and retained result containers. Public
no-argument integrated planning and standalone default-None behavior remain.

The immutable output binds the actual source fingerprint, capture manifests and
original/typed mapped identities and carries unresolved native duties. It is
unconditionally noninsertable even when pending tuples are replaced. No mutable
allowance, caller plan, SQL/path, factory, callback, cached verdict or authority
predicate is accepted. Existing six-member qualification and scope, claim,
journal, schema, pin, hash, name, ancestor, authorizer, cancellation, deadline,
step and physical retirement fences remain, with a final complete native recheck.

This bounds new composed representations, not total Python heap or pre-existing
R/I/C allocations. Resource exhaustion propagates as the distinct body-free
_TraceReadLimit; no phase gets a new allowance or omitted fields. Archive/history
limits remain independent. This API creates no new history-reader/capture role,
execution owner, publication or write authority.


### Pure captured agents.history qualification

The actual issued captured SQLite/Run-preflight operation may qualify only its
own authenticated source agents.history partition and sealed archive. Reuse the
native history _members inventory/root/topology proof and archive reader's
_regular, _central_preflight, _local_ranges, _member_chunks and verify_sealed
primitives under the same existing custody. Preserve full member size, SHA/CRC,
producer/topology and sealed-archive qualification before and after consumption.
The existing archive/member limits and cancellation remain independent.

The same remaining preflight representation allowance pays before new metadata,
bytes, native framing decode, models, associations and retained proof containers.
Return immutable original root/run/member/size/digest facts and separately proved
mapped associations. Accept no caller paths, selectors, manifests, streams,
callbacks or independent history capability. Saved plans do not carry authority.

Native frames require complete byte association to actual source Run parentage
and native root semantics. A filename, header kind, equal ID or digest alone is
insufficient; the native parser's skipped torn/unframed/trailing bytes cannot be
claimed as complete ownership. Preserve raw bytes, original frame IDs and partial
or uncertain state. Unknown or foreign ownership, missing members and incomplete
association cause exact body-free blockers. This adds no other physical/Git/Goal
evidence owner and never relocates shadow Git metadata or core.worktree.

No history publication, journal event, imported hold/receipt, source write or
runtime/scratch/tool grant occurs. Existing HistoryFiles read_member still needs
published verified state, SQLite/history scopes stay mutually exclusive, and
runtime history needs later committed native receipts. All six native members,
actual scope/claim/journal/selector/resource fences and final rechecks remain;
uncertain retirement retains native exclusion.


### Retained change-note snapshot locators and allocator admission (R103)

The outer pure Run preflight extends the existing integer snapshot domain with
retained non-NULL note locators from both captured partitions. Native integer
bounds, live snapshot and note run/root ownership, canonical historical occupancy
and every previously reserved target constrain the extension. Existing I/S and
Workspace proofs and reservations remain exact; a conflicting base target is
refused. Missing snapshots remain missing, with original/mapped per-note
provenance and unchanged excerpts, line text and delivery state. No synthetic row
or NULL replacement repairs history.

The fixed native `runs.sqlite_sequence` entry for `change_snapshots` supplies
source/canonical allocator provenance. Its source value is never copied. The pure
result records the required mapped frontier and an unresolved native allocator
admission duty, atomic with eventual owning rows/maps/receipts, so a future INSERT
cannot attach an old note to a new snapshot. Unproved frontiers and int64
exhaustion refuse. This authorizes no sequence mutation, new capture/read role or
writer. Full diff/revert still requires actual Git bytes or its exact pending
physical consumer duty; native self-contained note history survives pruning.


### Fixed pure-history selector parsing under the same allowance (R104)

Only the actual active captured SQLite history operation may read its own source
selector through the exact preview provenance. Preserve native selector admission:
owned regular file, pinned nofollow parent/name, original ancestors/fingerprint,
and existing 1 MiB MAX_RECORD. Ordinary modes such as 0644 remain valid; unrelated
private-copy 0600/nlink1 restrictions do not apply. Descriptors and parent pins use
the same native custody. Reserve the same remaining preflight allowance before a
bounded size+1 read within the native limit; exact length, descriptor/name/ancestor
identity and original SHA must agree before parsing. Growth/change/short reads or
unproved provenance refuse and retire normally.

The installed inventory `_parse_profile_config` remains the UTF8/TOML/reserved-key/
Pydantic authority, with an optional private budget defaulting to None for existing
callers. Before decoding, a bounded lexical accounting pass reserves raw/text,
values/keys, arrays, inline tables, table prefixes (including implicit dotted
prefixes) and parser representations. It does not validate or reinterpret TOML;
native comments, quoted/multiline strings/keys and array-table grammar remain
native. Additional actual decoded-tree/model clones are reserved before construction.
No JSON-only charge, gross guessed multiplier or heap-measurement claim substitutes
for native structure accounting.

Parse checked bytes once and retain only immutable required native RunLog root/name
association facts. Any private `_members` plumbing checks the actual scope and
captured-object identity; ordinary behavior remains unchanged. Preserve the native
environment/directory-name fallback and inventory/root/topology semantics. There is
no caller selector/config/reader, cache, new capture role, altered permission,
publication, runtime grant or writer release. R101's same allowance and R102's
source/archive/member/fence/cancellation/final-check/retirement requirements remain.


### Native passive command-history storage and reads (R105)

AgentRuns26 retains imported command facts in its existing five-column
`mobile_consolidation_batches` table, which previously had no installed JSON
payload grammar, completion validator or reader. This new fixed contract changes
no schema, catalog or owner and never inserts into `console_command_receipts`.

The exact key `history.command` contains only `version: 1`,
`kind: "command_history"` and nonnegative integer `record_count`.
Each `history.command:<exact original command_id>` child contains only
`version: 1`, `record` with all twelve original SQL fields, and
`run_associations` with separately proved original/mapped Run pairs for actual
`target.run_id`. Original epoch/target JSON strings, native ID, clocks, state,
fingerprint, reason and timestamps remain verbatim. Native decoding validates
shape, never historical HMAC/monotonic authentication or current execution.
Session/incarnation/handle/epoch/unproved native IDs remain opaque.

A private retention primitive borrows the actual admitted native owner transaction
and original composed allowance. It preflights every complete record before
insertion. The eventual real Runs writer commits passive children and manifest
atomically with actual rows, maps, provenance, exact holds and a separate owning
receipt. This passive manifest certifies only command retention, never full Runs
completion. Component fixtures and saved planning rows do not grant import
readiness; no consolidation caller is enabled by this primitive.

The native reader requires both operation ID and source fingerprint on an
actually admitted AgentRuns handle. It validates strict stored key/version/type,
command-key, source, count and association consistency in one read snapshot.
Unknown, foreign, duplicate, incomplete or malformed data refuses without active
receipt fallback. A missing manifest refuses; a valid zero-count manifest alone
means completed empty retention. Authenticated source24 table absence remains
separate provenance for the later owner receipt. Pagination binds the actual
operation/source and existing child, retaining complete-record coverage.

New reader limits are default 50 / maximum 100 records, a 16 MiB encoded-byte ceiling per
child, and one nonrenewable 16 MiB representation allowance per page. SQL BLOB byte
lengths precede Python text/body materialization for both master and children.
Keys/cursors/metadata, raw text, JSON, native decoding/clones and results are
admitted before construction. Pages may return fewer complete records with a
continuation; they never skip/truncate records or return a nonprogressing empty
page. A single unreadable record refuses before any retention writes; its new
representations are charged on the original composed allowance without reset or
replacement. Coverage queries remain bounded with no full-JSON fallback. The
historical source count has no live 4,096-row quota.

Passive history has no expiry, invalidation, pruning, unresolved-control
observation, routing, claim, dispatch or permission integration. Native database
backup/lifecycle retains these records. Existing applicable privacy/deletion
semantics remain; future history deletion is an owning transaction, never live
receipt pruning. No deletion API is introduced.

### Native owner admission and history access (R106)

Captured SQLite/history readers and a native consolidation owner cannot be active
simultaneously. Their transition must preserve the actual operation/session/claim/
journal/capture cohort across physical reader retirement. Saved/replaced plans
and pending-duty strings never supply that authority; pure completion guards
remain hard refusals. A concrete readiness protocol and runtime RunLog receipt/
membership contract require separate qualification before implementation.
Maintenance HistoryFiles or an arbitrary path/Run ID grants no runtime access.

All populated owners and required dispositions must be supported and qualified
before file publication or owning import effects. Fixed Chat/Workspace/Run
transactions preserve legal lifecycle, maps/provenance/holds/receipts, snapshot
allocator admission, receipt-first recovery and actual definition/current Goal/
recovery consumers. Original-ID RunLog access must bind committed receipts and
distinguish root ownership, framed membership and historical references.
Passive component delivery grants no partial public activation or cutover.

### Same-operation native reader retirement and preopen admission (R107)

One private synchronous coordinator entered from the actual native-issued
`_CapturedMobile` accepts only password and cancellation. It derives complete
source/integrated/Run input internally under one original representation allowance;
`_plan_runs` may receive that allowance privately. New frame, association and
validation representations are admitted before construction. Exact session, claim,
journal, capture, selector, source fingerprint, manifest/member/schema and native
row proofs stay inside the synchronous operation frame. Caller plans, callbacks,
factories, flags, handler names and returned handoff tokens grant nothing. All pure
completion guards remain unconditional refusals.

Before native path creation, owner-open publication, constructors or import effects,
all captured SQLite/cursor/pin/history/crypto/worker/pipe/FD/staging resources must
physically retire. Cleared local markers or inactive Python objects are insufficient;
actual session resource cohorts and unresolved-close outcomes must be checked.
The source claim and maintenance exclusion deliberately remain live. Uncertain
retirement retains existing quarantine and never permits a retry to erase doubt.
Preopen revalidation binds the same actual cohort, prepared journal, captured
artifacts, selectors, source/target physical witnesses and cancellation. Subsequent
owner/file effects require the same checks using legitimate native journal and
receipt changes. Authenticated prepared snapshots under exclusion are not fresh
postretirement live-row fingerprints. Readers and allowances are never reopened
or renewed to cross this boundary.

The serial component may qualify actual same-operation physical retirement followed
by existing native owner admission on disposable fixtures without importing rows.
The production coordinator must refuse exact body-free required writer, physical
or consumer duties before effects until fixed owning handlers qualify them. A
component success is not all-owner readiness, and these temporary blockers are
not a permanent unsupported-content policy. No helper may bypass that production
guard via a flag, fake plan or callback. The next fixed-writer increment supplies
atomic rows/maps/provenance/holds/owning receipts inside this coordinator and proves
populated positive operations. Crash recovery reconstructs admission and consults
actual committed receipts first; saved frames and journal progress are not receipts.
Undefined definition, current Goal/binding, runtime RunLog membership, allocator
and Git dispositions still need their concrete native contracts.

### Fixed owning receipts and historical native consumers (R108)

The fixed coordinator must complete all supported Chat, Workspace and Run writers
inside the same operation frame. Existing Chat82/Workspace9/Runs26 tables remain
unchanged. This contract supersedes R106/R107's undefined receipt, definition and
RunLog consumer dispositions; it does not relax their custody, original allowance,
physical retirement, exact pre-effect refusal or hard pure-completion guards.

Each owner atomically commits its actual rows, typed identity mappings, provenance,
exact installed holds and a separate `mobile_consolidation_batches` completion.
Runs includes the existing passive `history.command` master/children in that same
transaction. A passive command receipt never certifies Runs completion. Native
transactions revalidate the original cohort and inspect committed owning receipts
first on fresh admission. Exact source/owner, full census/digest and current mapped
rows/holds establish only that owner's pending-import completion. Partial masters,
matching primary keys, journal progress and saved plans grant no completion.

Masters have keys `chat`, `workspaces`, `runs` and exactly these JSON fields:
`version` (integer 1), `kind` (`owner_completion`), `owner` (respectively
`db.chachanotes.primary`, `db.workspaces`, `db.agent_runs`),
`source_manifest_sha256`, `canonical_manifest_sha256` (lowercase 64-hex or null
only for authenticated absent partitions), `entry_count` (nonnegative integer),
`entries_sha256` (64-hex), `command_history_count` (nonnegative integer for Runs,
null otherwise), `source_runs_version` (24/25 for present source Runs, null for
actual absent source Runs and other owners). A source24 member has its real
capture hash and explicit zero passive command manifest; it is not source25-empty
or an absent member. Booleans are never accepted as integers.

Children are `<master>.entry:<canonical decimal ordinal>` numbered consecutively
from zero, without leading zeros. Each has exactly `version:1`, `kind`, `data`.
Kind order is `identity`, `row`, `provenance`, `runlog_root`, `runlog`,
`history_publication`; within each kind order by canonical `data` JSON bytes.
JSON uses UTF-8, ensure_ascii=False, allow_nan=False, sorted keys and compact
comma/colon separators. Reject duplicate keys and require exact re-encoding.
Numeric SQL ordering uses fixed-prefix key length then binary key order. The
master digest hashes, for every child in numeric order, uint64 big-endian key
byte length, key bytes, uint64 payload byte length, payload bytes. No gaps,
aliases, foreign source/owner/version or incomplete candidate is accepted.

Identity data has exactly `domain`, `original`, `mapped`: an installed typed domain
with its declared owner/table and native scalar or composite identity shape.
Domain names are data, never SQL identifiers or authority. Row data has exactly
`table`, `original`, `mapped`, `source_sha256`, `mapped_sha256`; identities are
arrays of native string/integer identity scalars and table is a fixed manifest
entry. Hash complete columns in manifest order with SHA256 over ASCII
`tldw-mobile-row-v1` plus NUL, length-framed UTF-8 table, uint64 column count,
then length-framed column name, scalar tag and length-framed scalar payload.
Tags are N (null/empty), I (canonical signed SQLite integer ASCII, excluding bool),
R (IEEE754 big-endian binary64), T (exact UTF-8), B (exact BLOB bytes). No repr,
normalization, recursive payload rewriting or JSON-body substitute is permitted.

Provenance data has exactly `family`, `record`; the eleven literal grammars are:

- `goal`: `table` (goal_runs/checkpoints/evidence/reports/iterations with their
  actual `goal_` prefix), `original` (identity array), `original_hash`,
  `mapped_hash` (64-hex), `artifact_digest` (64-hex/null).
- `note_snapshot`: `original_note` (SQLite integer), `original_snapshot`,
  `mapped_snapshot` (SQLite integer/null), `original_run`, `original_root`
  (strings), `missing` (boolean). A historical root never grants file access.
- `snapshot_allocator`: `source_sequence`, `canonical_sequence` (nonnegative
  SQLite integer/null for absent sequence row), `required_frontier` (0..2**63-2).
  Exactly one Runs record, including an empty snapshot population.
- `sync_source`: `change_id` (SQLite integer), `entity` (messages, conversations,
  notes, character_cards, keywords, keyword_collections, chat_dictionaries,
  world_books or world_book_entries), `original_id` (string), `operation`
  (create/update/delete), `timestamp`, `client_id` (strings), `version` (positive
  integer), `source_digest`, `source_manifest_digest` (64-hex).
- `sync_frontier`: `entity` (same enum), `original_id`, `mapped_id` (string/SQLite
  integer), `version` (positive integer), `operation` (same enum), `timestamp`,
  `client_id` (strings), `payload_sha256` (64-hex), `payload_hash` (native sha256:64-hex/null),
  `previous_payload_sha256` (bare64-hex/null), `previous_hash` (native sha256:64-hex/null), `previous_operation`
  (same enum/null), `previous_timestamp`, `previous_client_id` (string/null),
  `current_disposition` (native_insert_current_unique,
  replace_native_create_with_body_free_tombstone or retain_native_body_free_hard_delete),
  `source_change_ids` (integer array), `owner_table` (fixed table),
  `owner_original` (identity array). Receipt bodies are hashes only.
- `operational`: `table`, `original` (identity array), `source_digest`,
  `source_manifest_digest` (64-hex), `disposition`. Exactly the five
  character_conversation_search_{dirty,documents,generations,revision,state}
  tables pair with retain_destination_authority_native_derived; exactly
  console_trace_{graph_epoch,migration_state,maintenance_state,compaction_state}
  pair with retain_canonical_operational_authority.
- `runtime_owner`: `singleton` (integer 1), `owner_id` (string). Historical
  automatic_work_runtime_owner facts never replace current runtime authority.
- `semantic`: `table` (fixed table), `original` (identity array), `handler`
  (C2b/C2c/C3/T/S/A-runs/I), `domain` (fixed typed domain), `original_id`,
  `mapped_id` (string/SQLite integer).
- `citation`: `original` (identity array), `source_manifest_digest`,
  `canonical_manifest_digest` (64-hex), `disposition`
  (sql_no_history_context_not_adopted).
- `trace_incidence`: `original_call_id`, `mapped_call_id` (strings),
  `source_manifest_digest`, `canonical_manifest_digest` (64-hex), `inputs`
  (array of [fixed table, identity array]).
- `workflow`: `table` (message_trajectory_metadata/transcript_annotations),
  `original` (identity array), `source_identity`, `projected_identity` (strings),
  `meaning` (historical_source_event/native_sidecar_event for trajectory;
  native_message_anchor for annotation).

No arbitrary dataclass/asdict or semantic fallback is durable grammar. Exact
original and mapped bodies remain in owning rows and authenticated retained
captures. Receipts do not duplicate Sync, Goal request or trace reconstruction
bodies. Native privacy removal remains authoritative; runtime lookup must never
retrieve capture bodies or replay old hashes to undo later edits/erasure.

Encoded master/child size is at most16MiB, checked by SQL BLOB byte length before
Python body fetch. Each runtime lookup/page has one nonrenewable16MiB
representation allowance, default50/max100 returned receipt entries. All relevant
committed master/child grammar, complete census, ordering and digest validate
before any body or filtered roster is returned. Count/length queries and bounded
pages are used, never aggregate JSON or unbounded bodies. The existing PK indexes
operation+batch only; no mapped-ID index is implied. Bounded complete relevant
Runs receipt scans are required. Missing masters, malformed/foreign/incomplete or
oversized candidates and allowance exhaustion fail the whole consumer call
closed, never silently become native or yield a partly filtered result. Stores
without import candidates retain native behavior.

All fixed writer/consumer dispositions and prospective receipt readability must be
qualified on the original writer allowance before the first effect. Future native
publication/commit observations alone may be precharged at their exact bounded
shapes, filled only from actual observations after the legal effect, and validated
on the same remaining allowance before owner commit. A placeholder is never
proof of missing semantics. One existing fixed A-files streaming publication phase
is permitted after captured SQLite physical retirement and before DB owners. It
uses the same frame's already-qualified authenticated member proofs, without
replanning/reopening SQL, another allowance or a new reader role. All history,
crypto, file descriptors and staging resources retire before DB owner admission;
maintenance exclusion remains on failure.

Imported definitions retain every SQL field. Only enabled-roster queries exclude
exact `agent_definitions` rows identified by the installed `agent_definition`
identity domain in valid committed Runs mappings. Unfiltered
list/get remains historical/editor access. Both bridge and AgentService/named spawn
use the filtered roster. Preset provider/model precedence bypasses ad-hoc override
allowlists, so historical enabled fields cannot advertise current instructions or
routing. Editing an imported ID does not release it; fresh native IDs follow current
validation, duplicate-name constraints, routing and tool narrowing. No flag
rewrite, fuzzy matching, permission grant, new hold kind or release API is added.

The admitted canonical AgentRuns owner exposes
`read_imported_run_log_page(mapped_run_id, operation_id, source_fingerprint,
*, cursor=None)`. No caller root/path/manifest or maintenance HistoryFiles object
supplies runtime authority. Committed identity and actual per-root framed
membership resolve original IDs. Root ownership, historical MANIFEST references
and actual frames are distinct. A healthy root owner may locate its own filtered
(empty if descendant-only) evidence; it never inherits descendant frames.

`runlog_root` data has exactly `root_id`, `location`, `artifact`, one per original
root. `runlog` data has exactly `logical_id`, `root_id`, `relative_path`, `location`,
`size`, `sha256`, `owner`, `framed_runs`, `historical_run_ids`, `segment_ids`,
`artifact`: IDs/path strings; nonnegative integer size; raw member64-hex/null digest;
original/mapped owner pair or null; arrays of original/mapped framed pairs, original
historical IDs and segment IDs. Location is a confined relative path. Artifact
witnesses are actual native _Object values with exact `path`, `device`, `inode`,
`kind`, `size`, `sha256`: absolute path, nonnegative integers, file/directory kind,
and64-hex witness digest. File raw digest and native witness digest are separate.
`history_publication` has exactly `destination_identity`, `source_fingerprint`,
`location`, `artifact`, bound to the real journal observation. Its creation artifact
may be an ancestor: check binding/path/device/inode, but do not freeze that ancestor's
old size/tree digest against legitimate future sibling imports. Requested root and
member witnesses plus their complete census detect extras, omissions, symlinks,
foreign files and byte-identical inode replacement.

The only namespace is the actual prepared target profile's user_data_dir,
`agent_history/imports/<destination_identity>/<source_fingerprint>`, then native
_HistoryMember.location. Runtime requires the actual current selected canonical
DB/profile association and genuinely enrolled StorageLease.execution_context with
non-None namespaces. UNBOUND/unqualified lease acquisition is not admission.
Separate actual root/member leases precede no-follow descriptor-relative opens;
installed agents.history enrollment and pins remain through hashing and paging.
Check named/held identities before and after; physically retire handles before
leases. Failed close retains the actual lease and quarantine, never retrying a
numeric descriptor or clearing flags. Ordinary runtime admission still refuses
pending maintenance; it is not bypassed to manufacture a positive test.

The native pager receives only private pinned descriptors, never a validated path
it reopens. Preserve ORIGINAL Run-ID filtering, original historical headers,
100 fragments/256000 content bytes/4500000 scan bytes, MAX_HEADER_BYTES16384+9
allowance and UTF-8 continuation. Empty pages with progressing continuation remain
valid, distinct from R105 command paging. Pre-admit receipt/tree/name/JSON/frame/
output representations on the same runtime allowance. The requested root's declared
aggregate regular-file bytes must be <=16MiB before verification; at most two
bounded root/selected-member digest passes plus the unchanged native parser scan
are allowed. Never scan unrelated publication ancestors. Writer preflight must
qualify this complete runtime readability on its original allowance.

Existing exact Goal+chain mutation guards preserve get/evidence/privacy removal.
R103 allocator advancement remains atomic, without synthetic snapshots or ID reuse.
Git requires genuine native physical proof or exact pre-effect refusal. This ruling
grants no detached-trace import exception, voice terminal-call grant reuse, captured
Git role, source flag rewrite or disabled trigger. Those exact native cases must be
resolved separately. Public cutover, full Task8 and later review/hardware gates
remain mandatory; receipt delivery alone is not task completion.

### Exact unrepresentable native trace cases (R109)

A populated detached owner whose attached-fenced descendants have no proved
original conversation mapping or lawful native replay schedule fails the entire
preflight before any file publication, owner path/open/schema, row, mapping, hold
or receipt effect. The source native CHECK requires detached owners to have no
conversation; descendant mutations require an attached effective owner and the
owner update trigger permits only one-way detach (v55→56). Never invent a parent,
borrow an unrelated conversation, revive a detached prefix or rewrite chronology.
This is not an attached=0 blanket refusal: an empty lawful detached graph or an
actually proved original relationship may be qualified under unchanged constraints.

A source `post_dispatch_promoted` call fails whole-operation preflight at its exact
call/provenance dependency. The v71→72 native guard requires a current Voice-only
exact-connection terminal-insert grant; source SQL is not that current authority.
The transaction-owning voice importer rejects an existing owner transaction and
cannot be nested or repurposed as a historical copier. No terminal INSERT, grant
reuse, crash_durable_reserved relabel, immutable-provenance/status/clock rewrite,
trigger disabling or manufactured execution is authorized.

Both refusals preserve every original byte and identify only the exact body-free
owner/call and native reason. They do not permit silent omission, partial import or
leaving ordinary attached/crash_durable_reserved trace families unsupported. Native
source controls must demonstrate no first effect alongside the actual supported
populated positive. A future historical-restore seam requires its own explicit
source/transaction/custody/schema/authorization contract; this decision creates no
new seam or implied future work. Conservatively blocked profiles are an explicit
cost; fabricated parentage and execution authority are unacceptable alternatives.

### Native Sync hash spelling (R110)

The sync_frontier `payload_hash` and `previous_hash` retain the exact native
`sha256:` prefix followed by 64 lowercase hexadecimal characters, or null under
the existing frontier meaning. `payload_sha256` and `previous_payload_sha256`
remain separate bare-hex digests of raw payload strings. Do not strip the prefix,
substitute either digest, recompute unrelated hashes or change the producer version.
The initial bare-hex spelling was disproved by the actual native projection test
before owner writes; this correction preserves that evidence and original data.
Semantic provenance domains likewise retain the finite _SettingsSuccessor
producer/table/handler meanings, distinct from identity reservation domains.
No aliases, extra provenance families or authority are introduced.


### Cold original-capture recovery and coherent checks (R111)

An unresolved journal owner may outlive its process. Only a genuinely fresh native
exclusive MaintenanceSession for the original pending operation, with reacquired
source claim and no _MobileScope ever issued in that admission, may read the
original authenticated archives before recovering that owner. Preserve actual
scope issuance history and registry identity. Before creating any reader resource,
require native physical retirement of every prior scope, nested claim uncertainty,
execution handle, staging resource and retained lock. Current live owners and
ambiguous closure still refuse; no caller recovery flag or substitute scope list
grants access.

The actual issued reader binds the unmatched owner-open ordinal/content and the
observed journal tail sequence/previous/content fingerprint using native event
encoding, together with its original operation/cohort. Re-read the journal at
every check and require those observations unchanged. Only that reader's own
healthy resources may be live. Preserve all exact session/claim/archive/member/
candidate/schema/selector/target/cancel checks; no current target SQL or new role.
Pre-admit new anchor representations on the original operation allowance; no
budget, native 30-second/5M-step limit, or attempted-operation marker renewal.

After complete original all-owner preflight and physical reader retirement,
existing native owner recovery may settle an out-of-order stale owner under its
actual locks. This recovery-only opening inspects the expected receipt first and
physically retires; it does not import merely to clear journal state. Ordinary
owning transactions validate the complete local receipt/R105 census first, then
exact mapped rows, holds, Sync and allocator: valid committed receipt skips
insertion, absence permits only already-qualified writes, partial or mismatch
refuses. Journal events never substitute for committed receipts.

One fixed internal same-call path may share a fresh coherent journal read and
pending/prepared validation across the nested claim, captured lifecycle and
SQLite checks. Keep lock ordering and all physical descriptor/name/private claim
receipt/archive/candidate/hash/namespace/target checks. Standalone public checks
remain complete. No public trusted-record argument, reusable observation, cache,
callback, skipped hash or cross-call memoization is introduced. Subsequent calls
repeat current checks.

Qualify with actual process death after owning COMMIT and before retirement, cold
exclusive admission, original ciphertext reuse, complete three-owner receipts and
no duplicate insertion. Live owner, replaced scope/cohort, same-valued changed
owner event/tail, held lock/failed resource, changed archive/claim/target and
repeated attempt controls remain refusals. Prior ordinary retired-owner resource
failures and inclusive instrumentation are separate evidence, not performance or
replay correctness claims. RunLog publication and all remaining integration and
review gates remain required.


R111 native discovery child custody clarification: the existing captured-reader
inventory call may issue a native private SQLite-copy discovery child. The
native `_discovery_reads` issuer checks the actual registered current parent
before creating the child and pre-admits its finite identity binding on the
same original operation allowance. The actual admission registry binds only
that exact issued child, its originally constructed private-copy object and
both original resource-list identities; the reader retains the registry map's
identity. This gives no new inventory, SQL, source or target authority.

During stale-owner capture checks, only that active current-TLS child in the
same session may remain live with its own healthy resources. Native reciprocal
child/copy checks and installed source/read-only/root restrictions still apply.
Copied, substituted, foreign, retired-as-live, replaced-map/list and failed-close
states refuse. Retired children undergo the ordinary physical-retirement checks;
no retained map entry excuses resources or uncertainty. Before every owner
path/schema/transaction effect the no-reader-exception guard still requires all
actual reader, discovery and private-copy resources physically retired.


### Imported checkpoint Discard and live Fleet capacity (R112)

A complete, valid committed Chat receipt census identifies imported message,
fleet_message and exact checkpoint rows. An imported valid checkpoint remains
visible as QUARANTINED/imported_history with its actual checkpoint and assistant,
and only the native Discard action. Invalid, incomplete, conflicting or exhausted
evidence offers no action. The actual store action claim and controller/native
commit/repository transaction retain their current identity, state and all three
version comparisons. Only explicit Discard may settle that imported checkpoint:
retain USER, use the native discarded assistant, delete the exact checkpoint.
Fresh subsequent Send uses ordinary new identities and permissions. Reconcile,
retry, dispatch CAS, provider handoff and non-discard settlement cannot mutate or
execute imported recovery. No other imported hold or authority is released.

Runtime queue/preparation projections are None without rewriting stored facts.
Even forced hydration cannot recreate an imported queue, pending hydration, hook
admission or continuation. Historical ancestry remains data. Current offered
Discard and its actual native claim remain necessary; a terminal string alone
is not user authorization. Ordinary nonimported behavior stays unchanged.

Each complete consumer call uses one original 16-MiB allowance for the entire
strict canonical numeric/framed, source/operation-bound receipt census, bounded
fetches of at most 100 and every subsequent representation. Preadmit SQL encoded
byte lengths before body fetch; native character limits remain character limits.
Fleet load, append and in-memory restoration/capacity exclude exact committed
imported IDs before FIFO selection and global/child message and character limits.
Append uses that same live subset inside its actual immediate transaction.
Invalid evidence refuses delivery and mutation; raw historical rows, FIFO
sequences and explicit erase remain. Later explicit Discard/edit/erase never
causes historical row hashes to restore old data.

### Original scope after first history-root publication (R113)

Only the actual issued mobile partition in its original admission, source claim,
prepared operation, authenticated source/canonical capture and selector cohort
may reconcile the first creation of the installed canonical agent_history root.
Fresh native journal/pending/cohort checks and the same original budget apply on
every call. The authenticated original before-image and approved scope must prove
that exact root unused, resolved_path=None and metadata=None. Require one native
history intent selecting that wholly absent root, exact fixed imports/destination/
source namespace, parent witness, manifest/member digests and valid native phases.
Current no-follow named/held root and ancestor identities/kinds must match the
actual recorded publication and admission. Unknown, foreign, linked, replaced or
ambiguous objects refuse. A staged-only rename qualifies only when the publication
has the staged directory identity and the temporary name is absent under no-follow
checks, including broken links. Ordinary staging before rename stays with the
existing publisher.

Complete current discovery still runs. Solely for original-source scope and
inventory, reconstruct that one exact unused declaration and exclude its created
canonical descendants. Re-run native classification and selection and require the
unchanged prepared/partition approved digest. Every unrelated change refuses; no
published file gains a source/SQL role. This inverse is neither byte verification
nor write readiness. After all SQLite resources physically retire, the existing
single AFiles phase independently verifies full actual publication custody,
digest, census and member witnesses before owner effects. Extra/replaced files
remain retained with pending exclusion. Precharge new metadata/journal/identity/
comparison representations on the original allowance; no cache, alternate digest,
replan, new stream while SQLite lives or renewed budget. Runtime ancestor witness
semantics remain unchanged across later legitimate imports.

PublicB native final verification, cutover, completion and pending retirement,
and ordinary reader E2E on this same produced target, remain separate mandatory
gates. Neither R112 nor R113 releases maintenance or authorizes user migration.


### Explicit local Goal payload erasure (R114)

Implementation status: the earlier source write was rejected for missing human
authorization and did not occur. Explicit human approval for code and isolated
tests was subsequently received ("Approve code and isolated tests"). The bounded
transaction-local implementation follows the contract below; this approval does
not authorize deleting actual user data. The earlier RED evidence and rejected/no-edit history remain
separate from subsequent qualification.

Only the existing native remove_payloads transaction may exclude proven imported
historical unsettled accounting from its privacy decision. Keep its allowed Goal
statuses, already-removed idempotence and native deletion/result shape; general
unsettled_root and all current execution/mutation predicates stay unchanged.
Complete the relevant strict Runs receipt census on one original 16-MiB allowance,
with SQL encoded-byte preadmission and fetch<=100. Resolve the whole actual native
allowance-root family and its descendant Run/Goal/chain/wake/chat-start/progress,
attempt and reservation relationships. Require each exact committed mapped row
and applicable matching unreviewed hold; bounded current scalar-row digests must
still match that operation's mapped projection. Missing, conflicting, changed,
reviewed, exhausted or unproved evidence refuses without body mutation.

Exact imported reserved/uncertain/token-committed and prepared/accepted/review-
required facts may be historical for local erase only. Their uncertainty and
accounting remain unchanged: no settlement, cancellation, clock/owner/reservation
or authority release is implied. New/current/nonimported or unproved-uncertain
family facts still block. Canonical runtime-owner authority is neither compared
to nor replaced by historical source provenance. Only existing privacy body
removals (including native Goal payload-reservation deletion), emptied request,
removed/payload_removed state and revision update occur. All holds remain. Old
hashes/captures never restore erased bodies. Native owned-fixture qualification
is separate from ordinary access to the still-pending produced target after B.


### Native history observation resource accounting (R108 implementation clarification)

The fixed writer derives frozen numeric observation ceilings from its actual
authenticated member census and publication prefix, before any publication effect.
Bounded descriptor enumeration checks each name before retention, relative-path
construction, sorting or file streaming; the complete native DFS encoding, object
identity and named/held comparisons remain. The original one-call allowance prices
three full publication observations, four separately priced root/path/custody
probes (including a genuinely absent root), and each actual root/member observation. Iterator close uncertainty retains native
custody and maintenance exclusion. No refund, new allowance or caller-ready value
is introduced; prospective unsigned-64-bit scalar widths are discarded codec
reservations, and actual observed objects alone become committed receipt evidence.

The ordinary observer `_limits=None` retains its existing behavior. Bounded mode
requires actual supported descriptor enumeration and a finite native filename
ceiling before effects. The installed Windows scandir facade eagerly materializes
listdir output and is not streaming evidence; no wrapper bypass, Windows rewrite,
or cross-platform qualification is implied. Unavailable bounded native resources
refuse before effects. Current Mac/POSIX verification and later phone gates are
distinct. Source representation accounting and the exact failed-path schedule are
recorded in the owning implementation allocation ledger; it is not a heap or
performance claim.

### Public mobile completion and recoverable cutover (R115)

This contract is recorded before Public B behavior tests/code, at approved native
owner HEAD `417bba869a0337feba56fe8eaea77d9b0ec0d711`. It implements the remaining
[Task36766](../tasks/task-36766%20-%20Consolidate-legacy-mobile-history-with-recoverable-ownership-and-quarantine.md)
and [Task8 plan](../../Docs/superpowers/plans/2026-10-25-shared-console-mobile-supervision-implementation.md),
not a new importer or capture role. The contemporary native Python3.12 SHA/AST/
status checkpoint is retained as `task-8-public-cutover-start-checkpoint.json` in
the plan SDD workspace; it includes absent new filenames and the controller-owned
dirty task. The approved proposal snapshot SHA256 is
`1989fa4a33efca6debd11fe47ad725b22e25c3db5f6a5349b29fcd1e0722d6e4`;
R115 source-line SHA256 is
`92e3f075d84b45b891bb2b0467f7cdc8a5e2626df143bb74d85012991f4c6312`.
Those are pre-code design evidence, not implementation or runtime qualification.

One actual native coordinator retains the issued maintenance session, source flock,
capture/selector cohort and original16MiB representation allowance through terminal
publication and exact pending retirement. New raw JSON, newly read lineage journal
representations, hashes, SQL bodies, names, witnesses and output copies are admitted
BEFORE construction/fetch/decoding. No caller record, plan, callback, token, role or
ready flag grants completion. No per-link/page/move budget renewal, new deadline,
reader reopening or extra R108 history observation is permitted. Existing native
capture/parser limits remain distinct. Physical reader/owner/iterator/crypto/file/
staging retirement and uncertain-close quarantine remain mandatory.

New records are strict version1 with exact fields and duplicate-key rejection.
Hashes are lowercase64hex SHA256. Ref is `{sequence,sha256}` with sequence0..99999;
it hashes the actual same-journal `journal._encoded` event envelope, including
operation, sequence, previous and evidence. Native `_evidence_digest` retains its
sorted compact ASCII-escaped JSON encoding. Owner master hashes use exact stored
R108 UTF8 bytes and census hashes retain the existing length-framed child codec;
these encodings are not interchangeable. Counts use signed SQLite nonnegative
integers, excluding bool; new physical device/inode observations use unsigned64.
Paths retain the native absolute/no-parent/no-NUL predicate and representation
bounds; captured member paths retain1024bytes. Journal100000-event/64MiB-record,
bootstrap1MiB-record/4096-record and existing archive ceilings grant no additional
representation allowance.

`mobile_committed` has exactly version, kind(imported|no_data), source_fingerprint,
destination_identity, prepared Ref, claim Ref, captures(max2 purpose+verified Ref),
targets(exactly3 TargetFinal), owners(exactly3 OwnerFinal for imported or0 for
no_data), history Ref|null, pending Object and bootstrap Directory. TargetFinal
has owner_id, root-to-main witnesses(2..256), sidecars(exactly WAL/SHM/journal),
schema_version(exact82/9/26). OwnerFinal has owner_id, master_sha256, entry_count,
entries_sha256, command_history_count(Runs integer, others null). All final main
witnesses are present and physical. Three missing masters never prove no_data.

Complete native receipts, mapped rows, holds, Sync, allocator and current history
files are verified inside the real operation before terminal; all owner/reader
handles then physically retire. Under locked journal, recheck actual claim,
selectors, profile bindings, pending Object and native bootstrap Directory,
append+barrier mobile_committed, then descriptor-relative retire ONLY that exact
pending pointer and barrier while maintenance and bootstrap.unbound remain held.
The terminal is the rollback cutoff, not execution permission or hold release.
Service success requires actual scope retirement; failed close remains fenced.

A fixed terminal-only recovery entry reacquires native maintenance and the same
source flock for present or absent pending AFTER durable terminal. It rechecks
actual terminal/source/target/profile continuity, complete immutable receipts/maps
and current retained members. It does not weaken `_mobile_pending`, replay import,
fabricate replacement activation, recreate a pointer or treat missing pending
before terminal as success. `require_mobile_cutover` is observational: no import,
repair/finalizer on HTTP, and no lifetime historical-row equality after lawful native
edit, Discard or privacy erasure. Native replacement identities require actual
installed continuity; path equality never accepts an arbitrary new inode.

Verified empty/absent source uses source_empty plus canonical-only capture, zero
imported effects and no legacy constructors. Absent canonical mains initialize
BEFORE terminal through actual mobile_consolidation_owner and installed
mobile_path_intended/staged/observed, owner-open/schema/mode/sidecars/owner-retired
transitions. The witnessed empty inode is published before constructors; final
terminal stores real82/9/26 main identities. No future identity wildcard or fake
import receipt is needed (`storage_admission._mobile_create_path`,
`_mobile_publish_creation`, `mobile_consolidation_owner`).

Explicit pre-terminal rollback freezes direction at mobile_rollback_started and
restores EXACT authenticated coherent pre-import SQLite snapshot/state, original
absence and appropriate native modes/sidecar dispositions. Native adapters use
copy_private_sqlite: uncaptured main/WAL layout/inode and guessed WAL are never
claimed. Original source/ciphertexts and displaced current DB/sidecar/history
objects remain retained; unknown objects never become owned by assertion.

Fixed rollback evidence fields are:

- started: version, prepared Ref, claim Ref, canonical_capture Ref,
  source_fingerprint, destination_identity.
- stage_intended: version, started Ref, owner_id, logical_id, member_path,
  payload_sha256, payload_size, target, temporary, parent Directory, Metadata;
  stage_observed: version, intended Ref, staged Object.
- move_intended: version, started Ref, ordinal, Subject, step(retain_current|
  restore_snapshot|retain_created), source Object, destination,1..2native MoveParents;
  move_observed: version, intended Ref, destination Object,1..2DirectoryStates.
- metadata_intended: version, started Ref, Subject, Object, previous/applied Metadata,
  parent Directory; metadata_observed: version, intended Ref, Object, Metadata.
- rolled_back: version, started Ref, source_fingerprint, destination_identity,
  exactly3 native targets, exactly3 RestoreFinals, retained Refs(max781), pending
  Object, bootstrap Directory. RestoreFinal has owner_id, stage Ref|null, current
  Witness, snapshot_sha256|null; absent originals have null stage/hash.

Subject is exactly owner_id, component(main|wal|shm|journal|history|created_parent),
path. Null owner is allowed only for history/created_parent. Paths derive only from
three mains/nine sidecars, one actual history publication and proved created-chain
entries(max3*256), at most781 before dedup. At most3 staged snapshot payloads;
per-parent native census max100000 entries and the same whole rollback-call16MiB
allowance. Complete coverage/custody/representation preflight precedes staging or
moves; intended evidence precedes each effect, and recovery accepts only exact
before/after native no-replace positions. Metadata changes are journaled separately.
No eager unknown listdir, unpriced recursive helper or untracked raw descriptor is
implicitly qualified by replacement helper reuse. Actual mobile rollback evidence
may select a narrow native rename-gap admission counterpart, never caller roots or
fabricated replacement records. Cancellation keeps recoverable exclusion and never
selects automatic rollback. Verified rolled_back retires pending through the same
terminal discipline but is NOT cutover. After committed, old-snapshot rollback
always refuses, even while pending. Only authenticated completed-mobile canonical
capture binding is added to existing later-rollback review/current-target safety
capture/replacement; no stale automatic overwrite.

After verified rolled_back, explicit reviewed successor may use ONLY unchanged
source and the SAME physically proved restored destination/controlroot, with new
operation ID and fresh full preview/captures under the original never-unlinked
source flock. Original fixed claim/lock remain immutable. Successor claim is
ONLY `new_journal.root/mobile-source-claim.json`, not a source-adjacent new inventory
row. New prepared predecessor is null or operation_id/control_root/claim_path/
claim Object/rolled_back Ref; old absent-field records stay byte-valid. Predecessor
may append one successor_intended with version, rolled_back Ref, new operation_id,
control_root, claim_path, prepared Ref. Cross-journal references explicitly identify
their journal and reciprocally match actual immutable records. Fixed journal artifact
recognition requires a real typed successor, exact local claim-intended/observed and
derived path; no generic ignored filename or recursive claim resolver.

Source lock acquisition moves ahead of preparation in one native lexical lifetime.
Prepared/pending/parent-child claim gaps recover the SAME new ID, never a second
reservation; no nested journal locks. Follow authenticated links, not glob, at most
4096 and one shared16MiB allowance with raw bytes admitted before parse/copy.
Reject cycles, branching, duplicate IDs, foreign source/control/destination and
unknown files. No pruning, original claim rewrite/deletion or scope normalization.
Changed live source remains reviewed-new-operation-required; changed-history merge
and post-committed successor are outside this contract.

Public service/CLI binds actual immutable preview and typed selected destination,
native password handling and explicit same-operation review/finish/rollback/status;
a bool alone is never physical consent authority. Qualify canonical Personas rename
BEFORE preview, cold default selection and real retirement with source81 untouched.
Qualify Notes paused candidate visibility with absent/present marker and repeated
startup; any reproduced missing candidate boundary needs a concrete contract before
policy changes. Final evidence must use the SAME legitimately completed produced
target through ordinary cold enrollment/RunLog/command/trace/Discard/Goal/privacy/
manual-turn consumers, keeping old automatic/binding authority held and zero old
provider/tool dispatch. Exact visual/trace/Git/worktree and platform limits remain.
Full original Task8 review precedes Task9; no hardware/TaskDone claim follows from
this decision. Incorrect completion risks premature writes/stranded fences;
incorrect rollback/lineage risks lost pre-import state or duplicate history.

### R116: explicit canonical initialization before the no-data review

The installed full capture rejects missing required Chat/Workspace stores before
partition projection (`Backup_Recovery/capture.py:348–392`). It cannot authenticate
an original absent target by capturing a database initialized during consolidation.
R116 therefore narrows R115's all-absent canonical preference: an operator starts
the selected canonical app with its supported `--config PATH`, closes it and waits
for physical retirement, then obtains a **fresh** preview and complete inventory.
The ordinary app's Notes/Workspace wiring initializes those required native owners;
legacy storage stays absent or untouched. Consolidation does not initialize them
silently, relabel inventory failures, allow a partial capture, or add a capture role.

Public start applies the same complete-inventory refusal before preparation, pending
publication and source claim. Its guidance must preserve every actual inventory
issue and explain initialization/closure/fresh review where required target owners
are absent. Optional absent stores already supported by a complete capture, such
as AgentRuns, still use R115's actual journaled native first creation and terminal
witnesses. No-data has no imported receipts or legacy constructors. Its rollback
snapshot is the real post-initialization, pre-import state, including supported
optional absence; it does not cover pre-initialization absence.

The native operator qualification observed actual canonical Chat and Workspace
constructors, app-owned shutdown and process exit, followed by a separate complete
capture preview with absent legacy input; Runs remained absent (PID41789, one test
passed in 4.90 seconds). This establishes the prerequisite only. Full no-data
cutover, ordinary same-target access and repeated gate verification remain separate
requirements. R115's terminal, rollback, lineage, custody and budget rules remain.
The immutable R116 evidence snapshot and the contemporaneous eight-path SHA/AST/
status checkpoint are retained in the Task8 implementation evidence directory.

### R117: first enrollment at explicitly selected consolidation start

Ordinary canonical startup initializes stores but does not publish a native profile
binding. The actual R116 follow-through failed before preparation at that missing
association (PID76126, one failure and two independent early-refusal controls
passing). A selected consolidation start may now enroll only genuinely missing
source/target profiles through installed `bind_profile`, after complete inventory
and exact per-selector configured-root/container proof. Existing bindings remain
unchanged. Preview remains observational; enrollment is not a generic user mode.

Prove the configured source data container covers the exact fixed claim parent;
never enroll an arbitrary absent-source ancestor or assign another profile's data
through a union of inventories. Observe private complete containers, preserving
native historical-alias, overlap and busy/incompatible-client checks. The default
config parent is never enrolled over bootstrap/service controls. Only installed
`_fixed_control_exclusion` and `_service_control_exclusion` authenticate those exact
excluded subtrees; include actual selector/native leaves and trees. Unknown
siblings, unsupported shared narrower ownership or uncovered claim parents refuse.

Registration precedes maintenance because native admission forbids nesting it.
Then hold one real session covering unbound and all relevant overlapping namespaces,
including candidate writers; recheck physical roots, registry mappings, selectors,
candidate cohort, complete inventory and containers before profile publication.
Keep that exclusion through post-binding review, preparation, capture and import.
Partial enrollment is retained and must validate on a later explicit start.

Binding can reveal config siblings that unbound discovery did not expose. A fresh
native preview/inventory must preserve the exact reviewed source/destination,
selectors/candidates and user-data/capture/rollback/credential scope. Account only
for authenticated excluded control metadata changed by enrollment. Any additional
user data or unproved equivalence returns `mobile_fresh_review_required` before
preparation/pending/claim/capture; it requires another explicit review and start.
No automatic retry, readiness flag or silent expansion of consent is permitted.

Issue the same original mobile budget before new observations and carry the actual
object through maintenance and capture/import. Pre-admit every new raw/decoded/
census/hash representation, retain the native 1 MiB selector bound and parser
accounting, and physically retire all resources on every exit. No replacement
plan, new SQLite role, caller-built authority or budget renewal is introduced.
The immutable R117 source snapshot and current native Python 3.12 SHA/AST/status
checkpoint precede this implementation. R115/R116 rollback, terminal and lineage
requirements remain; full Task8 review still precedes Task9.

### R118: exact native default-service capture output

The actual ordinary-unbound no-data start reached native preparation and the
canonical capture intention, then refused `output_overlap`: the authenticated
default recovery service container includes its own installed `control-work`
directory. Preserve that refusal evidence and the existing failed operation.

Permit only the already native-issued, live, non-review mobile capture partition
to exempt the exact independently authenticated `recovery.control:service` item
from this one output-overlap check. Require its actual maintenance/session scopes,
claim, pending pointer and journal; literal legacy/canonical purpose; current
intended destination; default service control; exact installed work parent; and
the existing SHA256(operation_id)-purpose encrypted-backup filename. Reuse the
installed private service validator with bounded custody and the same original
mobile allowance. Prove admitted effective profile roots do not cover the service
container/output. Preserve physical parent/item continuity across every check.

Capture preflight and both archive-writer checks receive the actual partition.
All other source paths, including omitted producers, fixed bootstrap, capture
staging, unknown controls and normal external-output rules remain protected.
Forged, retired, review-only or custom-control partitions gain no exception;
credential review keeps its disposable temporary output. Native publication,
private modes and durable intended/staged/observed evidence remain unchanged.

This bounded exception introduces no replacement authority, caller boolean,
new role, output format, broad ancestor exclusion or unbounded validation pass.
The immutable R118 source snapshot and native Python 3.12 SHA/AST/full-status
checkpoint precede dependent code. Qualify the default no-data terminal and
ordinary same-target use, plus the exact refusal controls. R115–R117 and the
stopped rollback optimization remain unchanged; no integrity waiver is granted.

### Task 8 execution clarification: ordinary native config publication (R119)

The completed no-data target passed the ordinary gate, but an actual installed
theme save advanced its persisted profile fingerprint and made both same-process
and restarted gates reject the old selector proof. R119 permits this advancement
only in the ordinary observational gate after immutable committed-terminal and
pending-absence proof. Pending import, finish-pointer recovery, rollback and
successor preparation retain their original selector evidence.

Keep the exact reviewed selector cohort and strict unchanged-selector checks.
Changed config leaf bytes/inode require the actual current positive native
`bootstrap._binding`, equal current observed fingerprint, exact registry and
effective-root association, unchanged activation where applicable, and namespaces
within the operation's admitted set. Ancestor identities/modes remain exact; the
leaf must remain owned, private, regular and nonaliased. Parse bounded current
config through installed resolvers: preserve the claimed source, all three target
stores, and every selector's candidate membership and reviewed destination. A
previously unrelated selector joining the source group still refuses. Chosen
targets use authenticated terminal transitions; other candidates retain their
original physical witnesses.

After journal and owner reads, recheck the same current selector observations,
bindings and registry before returning. All new representations use the same
gate allowance and physical custody, including the native 1 MiB selector bound.
The gate writes no record, refreshes no binding and grants no execution authority.
External edits, redirects, cohort changes, unbound changed leaves, ancestor or
target replacement, concurrent publication and pending operations still refuse.

The immutable R119 snapshot and native Python 3.12 SHA/AST/full-status checkpoint
precede policy code. Earlier retained scripts had private HOME/network isolation
but omitted explicit profile guards; preserve that narrower evidence. Integrated
no-data explicit finish, native config save, restart and refusal children must
install both normal profile and network guards before application imports.
R115–R118 receipts, source claims, rollback integrity and lineage are unchanged.

### Task 8 execution clarification: fixed rollback source binding (R120)

The retained intended-only Workspace rename exposed a missing rollback restart
entry. A newly introduced live semantic source preview then refused because the
bounded RunLog reader requires an actual captured import scope. R120 removes
only that new preview from pre-cutover rollback continuation: restoration imports
and rewrites no live legacy source content. The original source fingerprint stays
an immutable operation reference, never a claim of freshly read semantic content.

Keep the actual claim/flock, original strict selectors and pending association,
physical source-root/ancestor witnesses, fresh supplied-password authentication
of the original canonical archive, exact immutable manifest/stage/move/metadata
receipts, and installed fixed-owner validation on fresh disposable candidates.
Use actual native discovery scope/custody and retire its resources before moves.
Do not rediscover a required live Workspace during its authenticated rename gap.
Every source/current/displaced/capture object remains retained; unknown physical
state, bad password/cipher/manifest/schema/stage or ambiguous close still refuses.

Import, finish and reviewed successor source checks remain strict, including the
unchanged-source limit. No unbudgeted preview fallback, fabricated captured reader,
new role, caller authority, source constructor or repair SQL is introduced.
One recovery-call allowance, native archive limits, full journal verification and
known before-effect capacity checks still apply. The earlier restart checkpoint
preceded archive-reader plumbing; the separate R120 checkpoint precedes this
rollback-only policy edit. Qualify the same retained operation under both guards
and report initial refusal separately from explicit recovery completion.

### Task 8 execution clarification: initial and held preview budgeting (R121)

The actual unbound populated public-start test reached legacy Chat inspection in
the initial genuine `_PreviewScope` without the one already-issued allowance.
Earlier no-data success did not exercise this path. R121 binds the immutable
enrollment-reader snapshot and requires budgeting both initial generic discovery
and the two subsequent native-held re-reviews, before materialization.

Keep the installed order: complete observational preflight and physical retirement,
namespace registration, all-overlap plus unbound maintenance, strict current
re-review, first binding, strict post-binding re-review, then preparation. Attach
the same early-issued allowance and actual custody to the existing `_preview_reads`
issuer/`_PreviewScope`, then to the actual `_DiscoveryScope` and its owned copy
scope. Check exact current identity, PID/thread/active state and source safety;
maintenance additionally enforces its real session roots. A budget grants no new
source, role, writer, captured-reader or caller-record authority.

Retain the entire installed owner registry, declarations, unknown-child/alias
census and Complete semantics. Thread optional budgeting through shared inventory,
file, fixed SQLite and required adapter helpers. No unpriced fallback, omitted
owner, parallel reduced inventory or incomplete-status relabel is permitted.
Pre-admit raw/decoded config, path/model/collection/resource/source-state/digest
representations and SQLite payloads before allocation, decoding or fetching.
For actual immutable private candidates, use fixed type/length/count projections,
admitted catalog identifiers, bounded pages and unchanged native restrictions.
Schema/metadata rereads consume that same allowance; copy-byte/member/catalog/
step/time limits remain independent. Defaults preserve ordinary existing callers.

RunLogs gains a distinct genuine current preview/discovery branch deriving its
names from `_run_log_names` and its locators from fixed installed queries on those
private copies. Its captured branch keeps the exact `_CapturedSQLite` requirement;
no caller-supplied names are admitted on the new branch. Directory iteration stays
bounded and custody-owned with unknown-child refusal. Preserve exact source
fingerprints, selector bytes/physical witnesses and the whole candidate cohort;
any post-binding scope change still requires fresh review before preparation.

The R121 checkpoint precedes policy changes and covers only its named paths.
Qualify actual populated unbound start, initial and held prefetch exhaustion before
effects, selector/candidate changes and physical retirement/source preservation.
A parameter-presence assertion is only RED evidence, not completed budget proof.
R115–R120 and the stopped verification-removal restriction remain unchanged.

### R122 — Separate public coordinator accounting from the original native proof bound

The original R93/R95 retained T plus S/native input and proof path still has ONE
shared nonrenewable 16 MiB `_TraceReadBudget`. Its ordinary constructor and all
optional/default planner behavior remain unchanged. R115–R121 had extended that
same allowance to every new public coordinator phase; a real minimal Core81 read
alone now consumes 14,909,407 representation bytes, and full unbound start refuses
in initial Complete inventory. Those failures are not success or fresh-review proof.

A genuine local mobile service start or explicit finish/rollback worker now issues
one fixed private pair: a 512 MiB cumulative public-operation aggregate and its
ONE original 16 MiB native child. Every child reservation debits both atomically;
refusal by either leaves both unchanged. No arbitrary constructor capacity/parent,
configuration/API limit, per-phase renewal, refund, hidden retry or new authority
is introduced. An accounting link does not grant paths, captured input or writes.
The fixed aggregate is an engineering envelope, not measured completion or a
resident-memory/single-object guarantee. All smaller native limits remain.

The actual maintenance admission ledger keeps slot 2 as the original child and
appends a distinct aggregate slot. Genuine pair association is checked whenever
that slot is present. Initial PreviewScope and held DiscoveryScope coordinator
reads use the same aggregate; original CapturedSQLite/T/S/history/planning/row/
receipt/sync/owner proof consumers retain the same child. New public selector,
claim, journal-frame, terminal and rollback envelopes explicitly use the aggregate.
Do not relabel original proof variables or create an unparented replacement child.
Private native callers without public issuance retain their original 16 MiB path.
Ordinary cutover observation, status, standalone readers and generic backup/restore
keep their existing default behavior. Before-effect nonconsuming capacity checks
must consider both counters whenever the child is involved.

This supersedes only the later global-single-16-MiB accounting extension and its
coordinator identity plumbing. Every prefetch/advance-allocation check, Complete
owner inventory, native validation, original full journal verification, custody,
failed-close fence, selector/source/target check, hold and receipt remains required.
No further metadata-equivalence machinery is needed merely to fit that old outer
cap. The stopped rollback integrity-check removal stays stopped.

Qualification must preserve original/default 16 MiB exhaustion, atomic coupled
refusal, one pair per actual worker, same native child through reconstruction and
owner proof, same aggregate across public phases, before-effect refusal, and the
remaining R121 adapter/prefetch coverage. Record actual phase deltas while forwarding
all charges unchanged on full populated/no-data/recovery paths under both guards;
512 MiB is not a success claim. Ordinary schema/default regressions and the original
Task8 gate remain required. The immutable R122 proposal snapshot and precode
`task-8-public-operation-budget-r122-checkpoint.json` record the exact boundary.


### Current Core82 shared Subscription catalog integration

Task8's native Core82 receipt table must remain readable in the installed shared
Subscription/Core file. The existing Subscription full-catalog declarations stop
at embedded Core80 although the native Core81 constructor predates this feature;
the current Core82 addition is a separate Task8 compatibility obligation. Append
only exact Core82 hybrid successors of the already frozen Core80 lineage, using
the installed eleven Core81 DDL declarations and the exact Core82 migration DDL.
Retain both already-qualified historical source-index absence and the current
native Subscriptions constructor's exact source index; each is a fixed full
catalog, never a wildcard or optional-object matcher. Their embedded Core stamp
must be82; Subscription2/3 retain their existing identical-DDL version policy.
Do not introduce a bare-Core81 accepted hybrid family. Preserve exact unknown
object/SQL/stamp refusals and all native SQLite/schema validation. No live source
is upgraded, repaired, or normalized by recovery. The source-origin evidence and
pre-edit checkpoint remain in the Task8 evidence directory.


### R124 — Correct the public cumulative envelope from measured phase costs

The public512MiB envelope exhausted on a small native profile before both required
captures completed. R124 supersedes ONLY that R122 literal with one fixed2GiB
aggregate, retaining the512-byte bookkeeping reservation, original16MiB native
child, ordinary16MiB default, exact no-argument private pair/ledger identities,
atomic coupled reservations and both-bound nonconsuming pre-effect checks. All
current journal, owner, schema, credential, scope, physical/custody and independent
resource limits remain. There is no configurable capacity, per-phase renewal,
refund, cache, hidden retry, or migration authority added to the served runtime.

Observed separate epochs consumed413.398MiB before first capture on fresh start,
58.257MiB before first capture on explicit continuation,295.873MiB for one complete
legacy capture, then157.869MiB of canonical work before refusal. These are
cumulative reservations, not measured heap usage or one combined successful run.
The2GiB choice allows engineering margin for the broader canonical phase and
public completion; it does not promise all profiles fit. Preserve bounded refusal
and require actual complete-path phase/child/source/guard evidence. The original
native16MiB proof remains an independent unchanged sub-bound.

Checkpoint and immutable R124 snapshot precede the edit. Five coupling/default/
refusal controls and a genuinely fresh guarded populated start with readable-empty
isolated credential storage are required. Previous retained-operation approvals
are consumed; this accounting decision does not authorize another such action.


### R125 — Live ancestor custody and mobile canonical-copy provenance

Controller Ruling125 and the immutable
`.superpowers/sdd/2026-10-25-shared-console-mobile-supervision-implementation/task-8-public-lineage-copy-r125-snapshot.md`
(SHA256 `f4427e275d736099e046ead15760350399e12eea9503705a263a9f286affc228`)
resolve the A6 child-frame composition and retained-copy evidence boundaries.

After the reciprocal parent reservation and new immutable claim are durable,
the actual SourceClaim authenticates ancestors under separate sequential native
SH locks before the child EX frame. It retains their actual pinned directory,
record, lock and claim descriptors under the same original source flock and
custody. Every ancestor use repeats complete names/physical/raw-hash checks and,
when records are needed, fresh raw decoding with the original full canonical,
sequence, hash, grammar and reference validation. There is no decoded-record
cache, caller-ready token, generic authority, nested ancestor flock, or temporary
removal of the current frame. All current child verification remains. The closure
ends with native claim custody; failed retirement keeps quarantine. A fresh call
reauthenticates every ancestor. Existing4096-link and journal limits remain, with
all new representations pre-admitted to the same actual public2GiB aggregate;
native16MiB and ordinary16MiB limits remain unchanged and no third worker budget
is issued.

Mobile canonical-copy provenance is distinct from replacement rollback proof:
mobile receipts contain ciphertext/manifest/partition hashes, not a plaintext ZIP
hash. Mobile no-archive calls authenticate source evidence only and cannot attest
a supplied plaintext digest. Mandatory actual SealedArchive paths at planning,
staging and publication bind encrypted_source to retained ciphertext, exact
manifest/policy, plaintext archive.digest to the reviewed plan and actual native
stage/publication receipt before effects. Fixed builtin/config/preserved/created
callers must use real mobile source/manifest/producer/current-target facts or
refuse unsupported shapes; they cannot invent an original replacement plan or
_Rollback. Existing replacement-copy checks remain exact. Later restoration still
requires Complete current-target review, explicit new operation/password and a
new encrypted safety capture before replacement. Committed mobile operations
never gain same-operation pre-import rollback.

Qualification remains open:57976 proves the actual first-operation frame identity
and pair only;63574 exposed a private16MiB fixture capture refusal before ordinary
consumer assertions. Neither is successor or later-copy success. The controller
owns the full original Task8 gate before Task9.

### R126 — Existing journal frame during public input planning

The first small-profile successor exhausted the fixed public aggregate during
input planning, before import. Diagnostic2496 preserved the initial local-claim
JSON reservation refusal; subsequent staging retirement also lacked capacity.
No descriptor close failure occurred. All failed bytes/quarantine remain evidence.

For a genuine public worker, derive the existing coupled native slot2/public slot4
association from the actual maintenance admission, then use `_locked_mobile`
only around the initial `captured.sqlite_reads` context and fixed planning,
input/group/hold/receipt preparation, final reader check and physical retirement
(`mobile_native_operation.py`, initial read block). The frame uses the actual
journal, session and SourceClaim custody; R125 ancestor custody is issued first.
It physically closes before history publication, mode, owner or target writes.
The unchanged private path uses its original unframed16MiB allowance.

All initial canonical/sequence/grammar checks, every nested full frame census,
raw hash and physical verification, source/ancestor/SQLite/candidate checks and
physical retirement remain. No new cache, caller record/ready token, cross-call
state, budget change/refund/reset or stopped rollback optimization is introduced.
Original native16MiB and public2GiB remain atomically coupled; ordinary defaults
remain16MiB. Only actual repeated materialization within this existing continuously
locked read phase is avoided. The immutable R126 snapshot and supplemental
precode checkpoint preserve exact source and failed-run chronology. Qualification
must cover private/default/current-frame behavior and the actual successor,
ordinary gate, counters, source preservation and guards after retirement.

### R127 — Existing journal frames during pure owner verification

After R126, successor33857 reached the Chat transaction but exhausted the public
aggregate during receipt verification. No successor commit or descriptor-close
failure was proven; subsequent rollback and retirement checks also refused.

For the genuine current public admission and its original coupled slot2/slot4
pair, use the existing `_locked_mobile` only inside the three fixed pure bodies
`verify_receipt`, `verify_rows`, and `verify_recovery_history`. Their installed
SELECT, census, row, sync, codec and command-history validators remain unchanged.
Each frame authenticates the actual journal/session/SourceClaim custody under the
original source flock; R125 ancestor custody already exists. Every initial
canonical/grammar check and nested full census/raw-hash/physical/ancestor check
remains. Each frame physically exits before the caller's next write, commit,
rollback, sidecar observation or owner retirement. No whole transaction, owner
construction, history publication, mode or target write is wrapped. Private
callers retain original unframed16MiB behavior; public2GiB/native16MiB and ordinary
16MiB defaults are unchanged. No cache, callback authority, ready flag, cap
change, refund, reset or stopped integrity optimization is introduced.

The initial R126 read frame is separate. R127 does not close the outstanding
before-effect capacity gap: codec capacity alone does not bound new coordinator
verification, rollback and retirement tails. A source-derived nonconsuming bound
and actual before-write refusal must still be qualified, forwarding all real
charges unchanged. A successful framed import alone proves neither that bound
nor the remaining full Task8 acceptance. The immutable R127 ruling/snapshot and
current native3.12 SHA/AST/status checkpoint precede this sequencing change.


### R128 — Existing native frame for the two fixed owner transactions

Controller Ruling128 permits exactly the stale-owner and normal fixed import
transaction sites to nest actual owner construction, existing PUBLIC journal
frame, then native transaction. The frame spans BEGIN, fixed SQL and COMMIT or
ROLLBACK, and fully verifies/closes before owner retirement or journal publication.
The native exclusive-in-frame refusal remains unchanged. Three pure verification
closures require the exact current journal/session/custody/cancel/public-pair frame
and retain full entry/normal-exit verification; they cannot issue a nested frame.
Private native callers remain unframed with the original16MiB; public2GiB and
all atomic reservations/checks remain. This is not an arbitrary callback facility:
the fixed import paths register no completion callbacks. Core cursor versus
Workspace/Runs connection Scope counts, getter liveness, holds, savepoints and
commit/rollback remain native. No decoded cache or caller readiness is introduced.

The before-first-data-write capacity bound remains separately open. Current-read
integer deltas can be terms only with source proof of an unchanged bounded path;
they cannot alone cover future sidecar/owner-retirement representations. No larger
cap, sampled-only estimator, refund/reset or validation removal is approved.
Evidence: R128 immutable transaction assessment and precode checkpoint in SDD;
76005 remains a failed successor,95465 remains a restored-target observer refusal.


### R129 — Direct native canonical restoration in the ordinary observer

Only ordinary require_mobile_cutover may recognize the direct, explicitly reviewed
native replacement restoring this exact mobile operation's canonical copy. The
immutable mobile terminal/claim consumes the source once; restored pre-import rows
need not retain imported batches. This does not synthesize receipts or reimport.
Current positive paired activation and native binding must identify the same-control
committed replacement, whose receipt-bound plan.local_snapshot matches the original
mobile operation/control/evidence. Native plan/manifest/installed/activation/terminal
and immutable required-record evidence remain exact; actual current three-store
paths, installed inodes/private ancestors and bounded nonmigrating native schema
observations establish the current generation. Existing sourceflock, claim, captures,
selector cohort/routes, ordinary storage hold and all pending refusals remain.
Both terminal histories, current binding and physical objects are rechecked.

Original-generation checks remain unchanged. Only the direct-restoration branch
replaces the now-inapplicable original target inode/imported-live-census assertions.
Legitimate subsequent native data/config writes must remain usable through same
physical generation and positive binding, without permanent old payload/profile
leaf hashes. No owner activation or execution permission is granted. Ordinary16MiB
and custody/pricing remain; no public pair/new role/service/mutation scope. Other
SafetyCopies/multiple generations/unsupported config or created-builtin histories
are not generalized. Evidence: immutable R129 snapshot, precode checkpoint and
actual95465 restored gate RED versus89275 replacement GREEN, separately retained.


### R130 — Exact early terminal pricing and fixed public accounting envelope

Controller Ruling130 corrects R127/R128's over-interpretation: R115 requires fixed
terminal representations before the first target effect, not a guarantee that all
future repeated import SQL/control/cleanup work fits before the first INSERT.
Per-materialization pre-admission and fail-closed pending/quarantine/recovery remain;
complete rollback representation/custody coverage before staging/moves remains.
No generalized sampled-delta estimator is introduced. A fixed disposable prospective
terminal shape prices exact native path/count/hash/scalar/model/encoding dimensions
after readonly planning/readers retire and before history/owner effects. NoData uses
its empty-owner form before owner creation. Actual completion derives fresh real
proof; EVERY late reservation/check remains, with no prepaid credit/refund/token.

The public coordinator envelope becomes fixed4GiB after actual post-R128 successor
exhaustion8599 at Workspace receipt verification after Chat retirement. This is an
engineering bound on cumulative representations, not a heap measurement, exact
remaining-cost prediction or universal completion guarantee. Original coupled
native16MiB, ordinary16MiB, atomic charging and independent native limits remain.
The exact precode checkpoint precedes capacity/helper/tests. Qualify early refusal,
five pair/default controls, private16, one fresh successor, and genuine late refusal
with durable fences plus explicitly requested fresh-process same-op recovery.
Retainedb025 authorization remains consumed. R129 observer edit remains separately
blocked by automatic review pending direct human authorization.


### Task8 cold native Subscription owner import correction

The Subscriptions database class/schema import must remain inert with respect to
application configuration. Retain its patchable module-level `get_cli_setting`
name as a typed lazy forwarder to the actual configuration getter: preserve all
positional/keyword arguments, omitted defaults and exceptions without caching or
fallback. Only the existing subscription threshold read imports configuration;
its native startup/configuration admission and pre-transaction ordering remain.
This corrects the standalone recovery fixture's observed eager startup lease,
without a fixture shutdown bypass, changed catalog or new runtime authority.

### Task 8 pure Media migration import correction

The real-home standalone fixture (64638) exposed a second eager configuration import: Media's v6→v7 migration imports the documented pure `Chunking._template_conversion`, but package initialization eagerly loaded the runtime engine and optional-dependency configuration. Preserve the exact existing 16 package exports with standard lazy resolution to their actual defining objects. Pure conversion imports must remain inert; accessing runtime exports still runs their native configuration and frozen-default behavior. This changes neither migration/schema semantics nor startup admission, and introduces no fixture lease-retirement bypass.

### R131 — unchanged installed builtin members in a mobile canonical copy

A new native replacement may preserve ordinary installed builtin package members from the authenticated mobile canonical capture only when current Complete inventory, unchanged selectors/positive bindings, exact installed root, producer/dependency/topology and each manifest size/hash match. Preview excludes precisely these package roots from destinations; the existing safety_scope/preserve rules and public pending refusals remain. Staging/publication use their actual new plan/session and native private bound-source copy/dependency validation; no fabricated old replacement plan, historical created-source proof or package mutation. Changed package/configuration/history shapes refuse.

Use one genuine current observational allowance, or one ordinary16MiB allowance per lexical validation, throughout new maps/models/paths/hashes/copies. Optional private accounting arguments on existing copy/digest loops charge every read/view/partial-write slice before allocation. Current scope allowance identity and exact native budget type are checked; CaptureScope keeps its original os.read path and gains no field or authority. None callers, native limits/custody/retirement and all existing validation remain unchanged. This avoids a >215MiB worst-case whole-copy slice forecast for the shipped31-asset Samira pack without changing any cap. R129 restored observer and R130 capacity/terminal pricing remain unapplied pending separate human approval.

### R132 — Notes source boundary assumption (withdrawn by R133)

The original design requires refusal for unproved dependencies, not invention of a legacy File Notes owner. R132 withdraws R115's tentative positive imported-file-locator candidate test: legacy notes with non-NULL `file_path_on_disk` or `sync_root_folder` require the exact native legacy dependency. Do not strip these fields, substitute the canonical owner, import partially, or grant file authority. Positive App tests use imported plain notes and genuinely admitted canonical File Notes state, natural absent/present migration markers, actual canonical Personas rename before a new preview, repeated cold startup, and no automatic activation. Imported linked-note candidate visibility remains unqualified.

Execution finding before test refinement: the installed Core semantic validator has this refusal, but mobile capture currently invokes semantic dependency validation only for recovered media; legacy declarations contain no Notes dependency edge. The requested capture-level refusal is therefore not yet established. Record the actual native refusal point before claiming public coverage; no new production dependency dispatch is authorized by this test clarification. The ignored R132 call-boundary artifact preserves exact source anchors and the earlier unexecuted linked-note fixture text.

R131 execution authorization update: after the original automatic-review rejection
and unchanged production bytes were preserved, the human explicitly approved the
exact bundled-artwork recovery code and isolated tests (question
call_cd6c9bc1fbbe471b9457618eab28daa3). Resume only the existing R131 contract through
normal tool review. Native16MiB, public2GiB, all custody/admission/default behavior
and package immutability remain. Empty manifest members use a positive native
probe limit with exact zero-size equality; accounting imports remain conditional
on an explicitly supplied allowance. Runtime support is not yet qualified.
R129/R130 remain stopped pending their separate human decisions.

### R133 — Correct the actual Notes capture entry path

R133 supersedes R132's claimed mobile capture refusal and plain-only test narrowing.
Core's semantic dependency validator is called by generic replacement staging and
publication, not the actual mobile capture/import path. Function presence did not
prove an entry-point fence. Preserve the R132 immutable evidence as chronology;
no linked-note fixture refinement or App run occurred under that assumption.

Restore R115's actual linked-note source/canonical capture, public import and
App-owned startup qualification, including natural absent/present marker cases.
First observe what a genuinely captured nonconflicting source accepts; label
capture separately from import/terminal/startup success. Imported file locators
remain historical data and grant no authority. Do not insert a generic Core
capture check, create a new legacy File Notes owner, substitute dependencies or
strip source fields. If actual accepted imported notes are skipped by a present
marker, record the proven candidate-enrollment gap before Notes policy changes.
Concrete guarded App bodies still need controller inspection before runtime.

### R134 — Native source namespaces without profile rebinding

The actual existing-bound public Samira fixture (35527) failed held discovery with `capture_source_outside_scope`. Initial complete discovery includes the installed package, but missing-only enrollment omitted its native capture namespace for existing profiles. Obtain the SAME reviewed inventory's native capture names before maintenance for both existing and missing profiles, retain existing profile records/fingerprints/roots/names unchanged, then hold and recheck all native registry/alias facts and exact fresh review. Native registration remains non-nested and all new helper/record/path representations use the same public allowance and custody; None/default behavior remains unchanged.

Only the authenticated mobile canonical-copy branch may acquire source exclusion for the unchanged global readonly package via this native current capture registry rather than pretending it is a private profile root. Its actual locked copy, acquired verified archive, current Complete plan, positive selector/store bindings, exact members/source bytes and new replacement session/receipts remain mandatory. Generic replacement profile-based builtin admission and non-package constraints remain unchanged. Package files remain read-only; source namespaces grant no agent/tool access. R131 human approval covers this necessary composition through normal review. Actual whole replacement and real stage ordinary16 fit remain unqualified; R129/R130 remain stopped, public2GiB/native16 unchanged.

R134 execution status: automatic review rejected the first production namespace/accounting edit before process creation, judging its namespace-registration boundary outside the existing artwork copy/hash approval. Both affected files remain byte-identical to the precode checkpoint. That admission composition is stopped pending explicit human authorization; R131's separately approved native copy/hash/member controls remain implemented and narrowly qualified. Current public capacity remains2GiB; R129/R130 remain stopped too.

R134 authorization update: the user explicitly approved the exact native backup-source namespace registration/holding code and isolated tests for already-bound profiles. The earlier automatic-review rejection remains recorded. Implementation proceeds through normal review with unchanged profile records, agent permissions, public2GiB/native16MiB limits and existing physical admission checks. R129/R130 remain awaiting separate authorization.


### R135 — Native ephemeral scratch and ordinary agent admission

The actual cold mounted Console v7 refuses the stock run writer's real temporary scratch at ordinary persistent admission before provider dispatch. Compose the existing ADR082 manager-issued snapshot/real lease with ADR126 selected-profile and owner admission, without registering temporary roots or weakening generic StorageLease/registry checks. Authenticate the exact stock AgentService/writer callbacks, actual partial(manager.lease, issued snapshot), manager creation PID and current physical generation. Preserve config, Runs, Core, Workspace, default sandbox, MCP, imported holds, cancellation and tool permission observations.

Only the genuine ephemeral writer observation uses that composition: from actual selected-config lease verify unchanged config/agents.history activation and source-scope checks. All other paths retain ordinary admission. Fixed native worker decorators transfer only a private issued association, validate parent at capture/entry and acquire independent actual worker leases/admission. Already running workers retire under their own frame even if their waiter cancels; copied foreign contexts and forked manager use refuse. Cleanup remains native lease-counted and leaves no persistent namespace for deleted scratch.

The stock writer's bind, append and byte-write sites also independently acquire storage; route those exact sites through a fixed private context requiring the current authenticated writer association and exact native root/logdir/generated segment/manifest, otherwise the original persistent acquire_storage. All actual write/close callbacks, path containment/sensitive checks, physical leases and publication remain. No generic temporary exemption or new public capability. Before production save the exact writer/worker caller supplement and current SHA/AST/status checkpoint; qualify negative identity/profile/hold/cancellation controls before a fresh mounted manual flow. Current public2GiB and original native/ordinary16MiB are unchanged; R129/R130 remain separately stopped.


### Task8 R136: native device-local Notes lock metadata

The fixed `user_data_dir(config)/notes_sync_locks` directory belongs to the existing excluded Notes device state. Native coordinator construction was observed to make Complete discovery fail before PRESENT-marker import (27084). Exclusion requires complete bounded native no-follow metadata enumeration: exact private0700 root, only direct children with64 lowercase hex plus `.lock`, empty0600 UID-owned single-link regular files, stable pinned root and child identities. Absent/empty/native retained locks are valid; unknown, nested, linked, nonempty, broad-mode, changed or unobservable content stays blocking. No content, portable owner, import destination, active lease, sync grant, marker change or cleanup is introduced. Existing native inventory limits/current allowance/custody apply; no new scanner, allowance or scope. Precode identities are in `task-8-public-r136-precode.json`; actual acquire/release closure and positive/refusal controls precede another PRESENT App flow. R129/R130 remain separately pending.


R135 implementation evidence: after explicit human code/test approval, the reviewed native composition passed bound/default admission, stock callback mutation refusal, independent worker cancellation, process identity and default regression controls. Fresh mounted45390 then completed the actual New Chat/Send path, native drains and ordinary persisted reopen on a public-produced target; imported identity-scoped holds/accounting stayed unchanged, with one scripted generation and zero catalog dispatches. This is character-estimate fallback/scripted-provider qualification, not a new authority contract or real-provider/hardware claim. The original rejection and all failed epochs remain recorded.


R134 held-preview accounting clarification: the fixed authenticated mobile canonical-copy path carries one original ordinary16MiB allowance explicitly through its new source-prefix, selector/mapping, bootstrap/journal and framed-observation representations, and its already budgeted package read/copy/view/slice work. The actual native `_capture_bound_sources` context remains the sole source-read authority; generic capture admission is unchanged. No PreviewScope may be issued during maintenance, no scope fields or generic role are added, and nested fixed callbacks reuse this exact allowance without reset. Original generic verify_sealed/plaintext and encrypted hash passes, Zip/config/native planner internals retain their existing ArchiveLimits and behavior; this counter is not a whole-engine or RSS bound. Factor the existing authenticated candidate prefix and preserve positive current nonpackage profile bindings, complete target/member/selector/store checks, cancellation and physical retirement. Optional private forwarding defaults preserve existing callers; component identity/refusal/cancel/exhaustion/default checks precede another full replacement. R129/R130 remain blocked separately.


### R137 — existing-marker imported Notes candidate review

A completed mobile import may add a canonical linked note after this device already has the current Notes cutover marker. Present-marker startup uses complete current canonical Chat owner receipts only to select imported note IDs, with one original ordinary16MiB allowance for the added projection/materializations. Any existing binding for the same scope/note in any state excludes it before version/file probes. Only genuinely new disjoint paused candidates may be added through native Notes evidence/planning/private persistence; complete current root geometry and binding ownership are rechecked in that transaction. Existing active/paused roots, bindings, operations, leases and watchers remain unchanged. Invalid, overlapping or oversized evidence refuses the new enrollment visibly. No legacy sync-session/config-root replay, note/file mutation, activation, schema or new trigger ledger is introduced.

Both boot construction and captured runtime start-evidence retain existing config/root checks, adding only exact CURRENT-marker evidence from an already existing Notes store and a constant-row receipt probe on the existing selected Core owner. No Notes store is created on the zero-profile/no-receipt path. The added callback runs after present-marker/cutover/sole-process/closing gates; absent-marker migration and marker write ordering remain unchanged. Native Core SELECT transactions are not physical mode=ro/no-checkpoint claims; only newly owned thread handles close, borrowed handles remain. Canonical baseline and private persistence are not cross-database atomicity.

A defaulted exact boolean import_review_required in the runtime and Library snapshots projects a separate markup=False Static using the existing status class. It does not replace action/failure messages or change status/next_action/root controls. Fixed copy: “At startup, some file links could not be added. Notes remain available in Notes; review folder setup before syncing.” No new review/activation action is advertised. R137 source snapshot and precode evidence precede implementation; actual receipt/default/refusal/overlap/start-gate/closure/UI controls must pass before another full PRESENT flow.


### R138 — fixed mobile source verification boundaries

When a genuine explicit/current native allowance already exists, derive fixed mobile source evidence and its terminal fact from one actual Journal-owned shared-lock read with the complete native parser, canonical grammar, pending/namespaces, custody and evidence checks. No caller rows, proof cache, new issuer or cross-call reuse. Original no-current/None generic AND mobile dispatch reads and budget behavior remain unchanged; an allowance selects accounting composition, never source authority/family.

The planner's adjacent source and preservation checks use one full verification only for the actual returned mobile canonical-copy proof. Generic rollback proofs retain both original verification calls, mode validation and hash counts. This explicitly removes one redundant fixed-mobile archive verifier invocation at this single boundary; entry/planner/exit still freshly verify all predicates, and the extra original entry manifest verification remains. Original generic hashes/Zip/config/planner ArchiveLimits and the original ordinary16 accounting domain remain unchanged. No predicted fit, cap/reset/frame or source grant. The source snapshot and precode artifact precede implementation; actual parse/count/tamper/terminal/pending/default/custody controls precede a new held-prefix component.


### R139 — local allocation reuse for explicit-budget file hashes

The existing native reader may reuse one prepaid local memoryview/bytearray of at most64KiB only for explicit-budget, digest-only, non-collecting reads with no native scope budget. This removes real repeated os.read byte allocations; it does not remove any source/archive/package verification. Charge width+1024 before the buffer, keep every actual loop's768 transient charge and the original requested-size/positive EOF/growth/cancellation checks. Native scoped buffers, defaultNone, collecting/non-digest readers, copy/write paths and missing-readv fallback retain their original behavior. No new buffer API, helper/role, scope field, cache, authority or allowance. Actual lease/identity/custody and physical retirement before lease release remain mandatory.

The source-only56174 arithmetic is not measured heap, a complete bound or predicted fit. Original ordinary/native16 and public2GiB remain; the adjacent verify_sealed-only alternative is not approved. Save precode identities before implementation; finite real native parity/short-read/empty/growth/identity/cancel/exhaustion/fallback/retirement controls and a source/static report precede a new full held-preview attempt. R129/R130 remain independently pending.


### R140 — one full archive validation at the canonical entry

Only the fixed mobile canonical preview omits the acquired archive from its first source verifier call. That call still owns the fresh native journal/terminal/evidence read and original-cipher match; the immediately following full archive verifier uses the actual proof just returned and validates the acquired archive before mapping/planning. Planner and exit retain full source plus archive checks. This removes the redundant adjacent entry invocation and its extra temporal sample: canonical archive verifiers4→3 and verify_sealed6→5 per completed fixed mobile preview; source reads/original matches3 and target/package passes2 remain. All predicates stay at entry/planner/exit, without an atomic-observation promise. Stable source failures precede archive checks; actual family change still refuses before mapping, possibly earlier than the old redundant generic archive validation. Cancellation and errors propagate with original native behavior.

This supersedes only the earlier duplicate-entry count promise. The fixed caller always issues or receives its original ordinary allowance, including outer None, and intentionally changes its entry count in both cases. Outside it, generic/no-current/None source dispatch, crypto/Zip/native limits, reads and defaults stay unchanged. No new API, returned-document cache, authority, frame, field, capacity or refund; public2GiB/native16/ordinary16 remain. Source arithmetic is not a fit claim. R129/R130 remain separately unimplemented pending approval.

Saved precode and focused caller-unit RED/GREEN count/order/actual-proof/budget/cancel/failure/default controls precede a new native sentinel. Its bounded test observer records existing phase counters and scope fields in finally at source_context_exit_before_staging; a staging count0 there is not an overall no-staging assertion. No new source read, allowance or cleanup is introduced. Actual integration retirement/staging/SafetyCopy remain unqualified until their own gates.


### R141 — capture-scoped digest byte workspace

The existing native CaptureScope may own one private byte workspace, initially None, for the explicit-budget digest-only non-collecting readv branch. Prepay 65,536+1,024 on the same validated allowance before its first bytearray/memoryview allocation. Reuse bytes only: each file/pass still hashes its own actual returned bytes and performs every native admission, identity, bounded read/EOF, cancellation and custody check. No budget/evidence/cache field, API, helper, role or limit changes. Ordinary adaptive buffers, native Preview/Discovery buffers, None/default/collect/non-digest/fallback and copy/write paths remain unchanged.

Clear the byte field in the existing retirement finally immediately after active=False, before unchanged physical cleanup, including verification/resource failures. Dropping bytes is neither FD retirement nor secure erasure; traceback locals can retain views without authority. Different native scopes remain independent. This supersedes only the earlier no-scope-byte-field/cross-call byte workspace restriction. A first tiny/empty eligible captured file now prepays the full66,560 and may refuse earlier near exhaustion; no capacity increase or refund. Public2GiB/native16/ordinary16 and all R140 check counts remain.

Save precode identities and prove actual native two-file/multipass reuse, distinct scopes, tiny-file admission, defaults, failures and physical retirement before any new held-preview sentinel. The failed84580 cleanup evidence and source allocation arithmetic do not establish full preview/staging/SafetyCopy fit. R129/R130 remain separately pending.


### R142 — preserved installed sources at native SafetyCopy capture

The actual R141 full replacement reached native stage and then refused SafetyCopy capture because the selected profile correctly does not own bundled artwork. Keep the original profile-containment route first and unchanged. Only an included preserved builtin outside those roots may freshly authenticate the committed mobile source, acquired archive, prepared receipt, unchanged installed member closure and all current witnesses. Its current held native namespace may cover the exact package or a broader directory; pin and reopen that current root, retain all profile checks for other sources, and grant no profile binding or agent permission. All three existing original-source checks remain fresh.

One distinct original ordinary16MiB allowance is issued after actual stage retirement for this SafetyCopy source-validation phase, forwarded unchanged through the fixed callbacks. It covers the new journal, selector, geometry and witness representations; unchanged generic archive/copy/encryption/SQLite work retains its original independent limits and accounting domain. Track one descriptor bucket on the actual maintenance session before new opens, borrow it only under exact current custody, reject failed buckets before opens, and retain native quarantine on ambiguous close. Reject self/physical journal identity before nested source SH; no new lock protocol, cache, role, namespace or allowance reset.

The authoritative R142 snapshot and task-8-public-r142-precode.json precede the reviewed five-source implementation. Finite genuine source/namespace/default/refusal/custody controls and a component report precede any new full integration. Retained98188 is not continued. Public2GiB/native16 remain unchanged; R129/R130 remain separately pending and no restored observer or early-pricing behavior is authorized here.


### R143 — lexical member-parent reuse within each fresh SafetyCopy check

R142 native57145 authenticated the first two complete source checks but exhausted its single original16MiB phase allowance during the third overlap loop. Preserve that failed operation and all original checks/cost units. Within each fresh fixed helper only, preprice a local parent-to-existing-member projection; pin each actual parent once, stat every leaf and verify the unchanged manifest/witness frame, then reopen the current parent name while the original FD remains held and compare device/inode. Check cancellation immediately before reopen pricing/IO. Keep the separate package/covering-root reopens, all14overlap observations, all35leaf checks and all3fresh source/archive/nativeSafetySource checks.

This removes real repeated parent-walk allocations, not checks at the three source boundaries. Group order is first parent appearance and member order within each parent; interleaved parents can change first-refusal timing. No global atomicity or rename-and-restore detection is promised. All representations and opens remain prepaid on the same original allowance and registered custody; no FD/proof cache, new helper/API/role, cap/reset, generic-path or permission change. Ambiguous close retains native exclusion/quarantine.

The exact R143 snapshot and precode artifact precede code. Local mechanics tests using explicit mocked upstream evidence qualify only representation/custody, then one fresh exact committed-source three-check component decides real fit. A passing exact component permits one combined genuine broader/unprefixed/multiroot registration variant before root review; full replacement remains gated. R129/R130 remain separately pending; public2GiB/native16 are unchanged.

### R144 — distinct write-path observations within one fresh SafetyCopy check

The fixed mobile SafetyCopy source helper may validate each equal existing write Path once per fresh call, in first-appearance order. All logical owner mappings and effects remain unchanged. Its local set/lookup/growth representations are prepaid with existing units; different lexical paths/aliases retain independent native overlap checks. This supersedes only R143's fourteen repeated overlap observations: the qualified fixture has ten distinct paths, so overlap samples decrease14→10 and resolves28→20 within each pure read boundary. Supported writers remain excluded; opportunistic detection of hostile changes between formerly repeated samples is reduced, and no global filesystem atomicity is promised. All three fresh journal/archive/namespace/member/parent checks, original native16/public2GiB/defaults and custody remain. No set survives the call or becomes authority. See immutable R144 snapshot and exact eight-line proposal; broader geometry qualification remains separate.


### R129/R130 explicit human approval and current-source rebase (2026-10-31)

The human explicitly approved code and isolated tests for both existing decisions; the prior missing-authorization block is resolved (SDD `task-8-r129-r130-explicit-human-approval.md`). The old rejected drafts and failing epochs remain historical evidence. Implementation is pending root drift review, not yet applied or qualified. No new ADR/ruling is required.

R129 remains an ordinary16-only direct restore of this mobile operation’s canonical copy: native committed replacement/receipt-bound plan/manifest/activation and current private-copy schema plus main-generation/ancestor continuity; original claim, retained encrypted captures/history, current selectors and pending fences remain checked. Ordinary imported-state census stays unchanged outside that authenticated transition. Ephemeral WAL/content timestamps are not permanent generation identity; every native copy and physical safety check still runs. No fabricated imported receipts, reimport, owner grant or activation bypass.

R130 changes only the public cumulative envelope to4GiB and constructs/discards fixed maximum-width terminal representations before no-data owner creation or populated history/owner effects. Every actual late reservation remains; no escrow, refund, future-SQL estimator or universal-fit/RSS claim. Existing JSON sizing covers non-BMP escaping for every new serialization/hash. Private no-public calls still use their original16 and must be checked proportionally. Ordinary/native16, atomic pair coupling, independent archive/SQL limits and full rollback preflight remain unchanged. Current production stays2GiB until the reviewed delta is applied.

Finite sequence: new local identity/manifest/Unicode/dimension controls plus existing five pair/default controls and affected capture-names debit; early refusal before first owner for populated/no-data; one proportional original private/preterminal native16 control; then one fresh direct-restore positive with native data/theme-save and tamper/refusal checks, and one fresh successor. Existing late-budget real-home two-process evidence is retained; any rerun is a new isolated fixture with unchanged reviewed harness, only if needed to qualify changed prepricing. No old-fixture continuation or accepted App/Notes/CLI repeats.


### R130 cold-resume implementation correction — original invocation reuse (2026-11-01)

Existing ADR244/R130; no new ADR or accounting policy. The first changed-code real-home Workspace cold run failed its original16 allowance before stale-owner settlement. Its new early-pricing helper repeated native journal/prepared/pending work. The reviewed v2 correction uses pricing-only local references from the actual PRE-R130 entry preopen for no-data or post-reader preopen for populated, after original validation and retirement. Default callers still return None and retain their original call shape/checks. The new two-reference collector and shallow sizing views are prepaid and discarded before effects; no readiness, cache, native scope or authority is issued. All future original-record Ref encoding costs, fixed maximum terminal representations, actual final checks/hashes/charges and independent native16/public4 bounds remain. No new unbudgeted invocation replaces the deleted duplicate work.

SDD `task-8-r130-cold-pricing-v2-reviewed-release.md` governs the exact two-file correction, affected local/default/codec controls, genuine populated/no-data refusal and private16 preterminal proof, then one fresh observed Workspace73→0. Actual fit remains unqualified until that run; the prior failed fixture and rejected v1 accounting rationale remain preserved. If green, the already released B–E integrations continue serially. No full Task8 completion follows from these components.

### R130 Event tree accounting correction (2026-10-31)

The cold3 native16 resume exhausted at final JSON scanning (requested256, remaining58). Before another runtime, the reviewed helper-only schedule replaces two coarse eight-text-copy Event-stage reservations with separate complete-tree reservations: the native terminal model dump, conservative Event evidence validation, and each of the two Event dumps. Constructor/model/field containers and a shallow six-field sizing view are prepaid separately. The first eight-copy `_MobileCommitted`/Path validation, existing pre-R130 input reuse, future Ref coverage, non-BMP serialization, record bound, JSON roundtrip, all final charges and native16/public4 limits remain unchanged. No authority or readiness value escapes. Pure retained-shape arithmetic is not a fit claim. Source contract and finite verification: `.superpowers/sdd/2026-10-25-shared-console-mobile-supervision-implementation/task-8-r130-event-tree-reviewed-release.md`; prior failed epochs remain preserved.

### R145 - production recovery qualification scope (2026-10-31)

Both production captured-consolidate entry points use a genuine public pair. Keep
the current early prototype and every late check on all callers; no private
exception or accounting change is made. The larger private Workspace cold4 fixture
now legitimately exhausts original16 after early pricing, and remains a recorded
failure rather than a parity or complete-path claim. R130 promises bounded
refusal, not universal fit. Replace only the added private-success gate with the
same committed Workspace/unmatched-owner graph through actual public start and
fresh-process public finish, each genuine4GiB/16MiB pair. The smaller private
three-receipt component retains its actual passing scope. Genuine restored-gate
B and C-E may proceed independently; complete public crash/receipt/no-replay and
all original Task8 qualification still precede Task9. Source contract: SDD
`task-8-r145-public-recovery-gate.md`.


### R146 — Complete ordinary cutover observation envelope

The B1 direct-restoration integration exposed an actual ordinary16MiB refusal while loading the native receipt-bound plan. Both full journal parses, plan/manifest validation, schema, retained captures/history and current exit rechecks remain required. R146 explicitly supersedes the earlier whole-cutover-call16MiB promise with one fixed nonrenewable64MiB coordinator allowance, issued by a private no-argument function before observation and debited512 for issuance. Both restored and unrestored ordinary cutover calls use it; it is not paired with a new16MiB child and never grants storage or execution authority.

Generic trace/Notes constructors and private migration remain16MiB. Public migration remains4GiB with the original coupled16MiB native child. Every existing validation, lease, source flock, physical/custody/retirement check and independent SQL/archive limit stays unchanged. No later reset, refund, cached proof or caller-selected capacity is introduced.64MiB is the controller's bounded engineering choice within the user's restored-profile implementation approval, not a user-specified number or RSS/universal-fit guarantee. Retained B1/builtin1892 selected-charge floors exclude other work; fresh B/C qualification decides actual fit. Preserve original failures and immutable decisions.

Before code save precode and finite local issuer/default/exhaustion controls, then one fresh B with actual success/failure counter evidence. Conditional C/D/E follow only on green. Public Workspace fresh-process qualification remains separate under R145; no full Task8/Task9 release follows.


### Task9 canonical read and saved identity seams

Core and Media expose explicit `open_read_only` constructors over existing current
schemas. Their dedicated read owner policies retain ordinary storage admission,
native participant registration and physical close, with SQLite mode=ro and no
initialization, migration, reconciliation or execution lease. Missing or incompatible
stores are unavailable. Parent Library reads use the canonical conversation service
and existing notes/media filtering; GET never starts a child. These are ordinary
read roles, not recovery, integrity or foreign-source capabilities.

Native runtime saved-open reserves the conversation before its first hydration
await and shares that operation through cancellation. Native persistence publishes
saved identity through the store's existing session state; lookup always checks
current sessions before hydrating. Host-level create/open use the same durable
command receipt owner as session actions and never select a desktop tab.


### Task9 R147 explicit first-host bootstrap

An authenticated, CSRF-protected `POST /api/v1/mobile/console/host` accepts only
`{protocol_version: 1}`. After verified cutover it reuses the canonical registry,
awaits connection and positively checks native run-store execution admission,
then returns the exact `ConsoleHostIdentity`. It creates no native chat, provider
request or ticket. An explicit Create/Open may bootstrap then issue one exact-host
ticket; races refuse. Reads and reconnects never bootstrap, reticket or replay.


### Task9 connected native project callback correction

The first native remote Send reached durable persistence, then the unchanged
project-provider composition invoked `project_authority_guard` on the controller
loop. Its local `on_owning_loop` unconditionally used Textual `call_from_thread`,
which refuses on that loop. Preserve the four original activation/authority/
dispatch callers and every project/source/permission check: invoke directly only
on a non-null matching controller owner loop; otherwise retain worker marshalling
and the existing no-App fallback. Catch only absence of a running loop, never a
callback exception. Original first-Send exception evidence is in Task9 agent10;
qualify actual native Send plus owner-loop/worker/error/default routing controls.
This is a routine shared bug fix under ADR244, with no new authority or API.


### Task9 cold viewless source metadata dependency

The un-warmed Library-mounted native first Send refused before execution because
`_resident_skill_wiring()` had no resident `Widgets.compact_model_bar` source
metadata. The existing cold stock catalog proof deliberately requires that
module's original source capsule; a mounted Console normally imports it. Explicit
remote Send loads this same dependency through `run_preparation_read`, observed by
controller/runtime physical teardown, before unchanged received-configuration
capture. Its import reads/compiles Textual parent source metadata, so it runs off
the owner loop. Exact native target and all five revisions are rechecked after
that await. No widget is mounted, trust/plugin getter invoked or cold/custom/
memory/source/permission qualification relaxed. Preserve cold refusal evidence
and qualify the actual un-warmed first Send plus no-retention negative control.


### Task9 fix1 — shared saved-session readiness and cleanup custody

ADR required: yes; this clarifies existing ADR244, with no new execution or file
authority. F02–F07 use ONE store-owned saved-conversation operation, reserved
before I/O/native creation's first durable commit. Every observer shields the
owned task/future. A provisional row is not a usable published owner: readiness
includes progress, native Library policy and membership reconciliation. Actual
failure/owner shutdown retains physical retirement and rolls back exact
provisional state once; observer loss alone cannot cancel another opener.

Presentation callers receive explicit exact cleanup custody. Terminal workspace,
Buddy and the outer Character activation retain it until their final successful
commit. Another successful opener adopts/publishes the ready owner and revokes
that cleanup capability. Failed solitary unpublished restores still roll back.
No blanket rollback suppression, hidden duplicate session, or extra registry.
Cold saved publication defaults to the durable ID to retain Goal's existing
sealed attempt/session contract; never renumber a live first-persisted owner,
and refuse conflicting explicit IDs. Remote scope comes from the durable row,
including explicit global, independently of terminal selection.

Native new_chat/fork reserve their preallocated durable ID in this same operation
before commit/copy. Their intended settings, assistant identity, opening draft
and existing new-session project controls must be durable before a phone can
observe publication; direct native completion joins the same readiness. Do not
overwrite an accepted phone turn or widen project/file/permission authority.
Source-close/orphan cleanup remains exact. Native shutdown drains owned physical
work and detached exceptions are consumed. Focused regressions cover mixed
native/phone readiness, presentation/cancellation, Goal iteration in both orders,
native create/fork publication races and close/reopen. F01 separately bounds the
saved discovery DTO (including scope/activity); F08 refreshes only current
nonstarting registry discovery and retains stale target refusal.

Fix1 implementation qualification: the common operation also joins the existing
first-identity reservations for manual Save and durable first Send; these already
live owners stay controllable while saved-open observers await publication. Native
new_chat controls commit through the existing project-context writer in the native
atomic creation transaction; fork's first commit and copy share the reservation.
Shutdown carries saved-read cancellation through existing physical native cleanup.
Host discovery refresh does not move a running event subscription: register and
disconnect use the exact child captured with its parent buffer.
