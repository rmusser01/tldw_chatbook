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
