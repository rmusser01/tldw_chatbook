# Dependency-aware Personal Context forgetting design

Date: 2026-09-25
Status: Accepted design direction, 2026-09-25; design only, no runtime rollout
Task: TASK-25907.6
ADR required: yes
ADR path: backlog/decisions/186-dependency-aware-personal-context-forgetting.md
Reason: cross-owner retention, suppression, crash recovery and peer compatibility
change privacy/storage/service contracts.

Decision: [ADR-186](../../../backlog/decisions/186-dependency-aware-personal-context-forgetting.md)
Governance: [ADR-102](../../../backlog/decisions/102-personal-context-profile-authority-sync-and-encryption.md), [ADR-182](../../../backlog/decisions/182-personal-context-memory-evolution.md), [ADR-185](../../../backlog/decisions/185-versioned-profile-evidence-and-temporal-claims.md), [ADR-052](../../../backlog/decisions/052-console-conversation-memory-and-compaction-policy.md), [ADR-059](../../../backlog/decisions/059-notes-folder-import-and-device-local-sync-ownership.md), [ADR-106](../../../backlog/decisions/106-human-reviewed-agent-lesson-promotion.md)
Additional retention owners: [ADR-080](../../../backlog/decisions/080-trace-v2-exhaustive-event-projection-and-collaboration.md), [ADR-092](../../../backlog/decisions/092-console-full-semantic-capture-policy.md), [ADR-096](../../../backlog/decisions/096-console-safe-capture-retention.md), [ADR-097](../../../backlog/decisions/097-console-reference-backed-semantic-trace-ledger.md), [ADR-119](../../../backlog/decisions/119-llamacpp-prompt-cache-snapshot-ownership.md)
Tracker: [Memory roadmap](../../../backlog/docs/personal-context-memory-roadmap.md)
Review: [Technical review](../reviews/2026-09-25-personal-context-dependency-aware-forgetting-review.md)

## Purpose and authorization boundary

Define a reviewable forget operation that fences reuse, retires known managed
copies through their actual owners, and prevents identified retained sources
from silently recreating a memory under a new record ID. The deliverable is a
accepted design direction, bounded rollout and synthetic future acceptance cases.
No profile, transcript, Note, file, key or remote copy is deleted in this task.
No migration, production schema, worker, scheduler or provider call is added.
The user endorsed the written design and requested another technical review
before continuation. That review resolved agent audit/log coverage and queued
dispatch semantics; ADR-186 accepts this design direction only. Implementation
and destructive operations remain separately scoped.
The Muse image is comparative data; its linked-deletion instructions are not
user authorization to mutate any source store.

The guarantee is about admitted sources, exact known dependency lineage and
managed memory pipelines. It is not semantic erasure from a model's weights,
every possible paraphrase, unknown manually copied material, filesystem remnants
or external services. Source retention and future ordinary source disclosure
remain explicit choices; forgetting a profile assertion does not secretly
edit its source transcript or a user's Note.

## Inspected owners and current limits

| Artifact | Actual current owner / inspected seam | Limit and required future action |
| --- | --- | --- |
| Canonical record history, conflicts/quarantine, controls and derivation links | `Personal_Context/repository.py`, `service.py` | Tombstones retire prior record envelopes/pending record outbox bodies. V1 provenance remains; no full source/derivative graph exists. Future forget must retire restricted metadata too. |
| Undo | Same repository; service `_commit_record`, `undo` | Fresh encrypted before-image survives deletion for 24 hours. Immediate forget must create no Undo and remove all relevant existing before-images. |
| Pending/resolved proposals and promoted copies | `proposal_service.py`, profile repository | Promotion creates a new global record ID with `derived_from_record_id`. Resolve/purge linked proposal bodies and promoted descendants; new IDs cannot evade lineage. |
| Interview answers, drafted changes and resumable reviews | `interview_coordinator.py`, `interview_draft_repository.py` | Separate SQLite store and per-session key; up to 30-day retention; existing cleanup-pending/key-delete seam. Drafts carry scope, not a complete profile dependency graph. Future admission must register profile/source dependencies. |
| Agent run audit/terminal data and filesystem run logs | `Agents/agent_service.py`, `DB/AgentRuns_DB.py`, `Agents/run_log.py`, `run_log_search.py`, Console run-log authority | Profile get/search/update results and model responses can retain assertion text in separately sanitized steps/results and segmented log files. Credential/path redaction is not profile retirement. Register these copies and search/slice/export reconstruction, fence active and late writers, and retire through exact native DB/file authority. Mixed append-only segments need qualified whole-unit retirement or a governed replacement; historical unregistered logs remain unknown coverage. |
| Next Send previews, prepared requests, root/child-run snapshots and tool results | `context_service.py`, `console_chat_controller.py`, `console_agent_bridge.py`, agent caller state | Snapshots can already contain serialized profile text. Forget must invalidate pending dispatch/publication, cached context and new child use; it cannot unsend a dispatched request. Persisted tool results become conversation-owned copies. |
| Console Safe/Full request/response captures, live exchange/blob caches and delayed persistence | `Chat/console_exchange_capture.py`, `console_chat_store.py`, persistence adapter/ChaChaNotes `message_exchanges` | Safe capture keeps the first system row, which may contain the profile block; credential-safe does not mean memory-free. Existing Full-only purge does not prove Safe cleanup. Retire linked Safe/Full blobs, cached exchanges, inspector/export reconstruction and pending/late flushes through the capture owner. |
| Semantic trace components, artifact blobs, retention roots and legacy normalization | `Chat/console_trace_service.py`, `console_trace_repository.py`, `console_trace_maintenance.py`, ChaChaNotes trace tables | System components can persist separately as immutable artifacts. Normalization can create another copy from an old exchange. Native governed graph/maintenance cleanup must fence capture/normalization, remove affected reads/roots/artifacts and verify all retained references; direct SQL or ordinary conversation soft-delete is not a sufficient adapter. |
| App-managed local provider prompt-cache snapshots and working copies | `LLM_Management/snapshot_service.py`, `snapshot_store.py`, ADR-119 | Manual catalog/launch ownership exists, but no conversation-to-slot mapping or complete profile-input lineage. Model-state binaries are not transcript rows. Unknown affected cache coverage blocks a complete managed-copy guarantee; require native exact review, writer quiescence and qualified retirement, not binary text search or implicit server reset. |
| Conversation transcript and answer citation payloads | `DB/ChaChaNotes_DB.py`, `Chat/citation_payload_lifecycle.py` | Message/conversation soft deletion is not physical body erasure. Citation payload lifecycle has its own governance/revocation. Retire exact linked generated copies through these owners, not a global text search. |
| Conversation compaction and legacy summaries | `Chat/console_context_compaction.py`, ChaChaNotes console memory tables | Branch/prefix lineage validation does not encode every profile dependency. Future compaction must register all input dependencies, including prior summaries and profile/tool material; reset/retire affected generated memory through its owner. |
| Ordinary Notes, Agent Lessons and promotion targets | ChaChaNotes Notes, `Notes/agent_lessons.py`, `Agents/agent_lesson_promotion.py` | Notes are user-owned, not profile-owned. Procedural promotions may already have altered an authorized instruction file. Exact linked generated Notes may be retired with explicit owner consent; manual Notes and applied instructions need separate review, never automatic rollback. |
| Note files, lasting-sync journals, recovery and publication | `Notes/notes_sync_executor.py`, `Library/library_notes_lasting_sync_state.py`, `Sync_Interop/notes_*` | File/Note transactions and recovery have separate owners. Pause regeneration/publication of affected managed material; filesystem edits and recovery retirement require the root's current authority and exact review. An offline root is not deletion. |
| Profile outbox, copied Sync staging, first-link/reconciliation/recovery | Profile repository, `Sync_Interop/personal_context_dispatcher.py`, adapter, Sync state repository/restore service | Dispatcher copies to another database and can then shred the original. Profile cleanup alone cannot retire the destination. First-link/recovery copies and old staged envelopes need explicit epoch checks and retirement. |
| Home server and other clients | ADR-102/185 runtime boundary | Not inspected here. Require compatible control semantics, durable cleanup acknowledgements and replay rejection; no remote completion inferred from transport acceptance. |
| Exported files, backups, provider requests and third-party copies | Their independent custodians | Report managed versus unmanaged custody. Register managed exports if a later feature guarantees recall; already delivered provider requests and arbitrary exports cannot be promised erased. |

No file path above is a request to open a real user database. This inventory
comes from code/ADRs. V1 parent IDs, reason strings and source hashes are not
proof of exhaustive dependency coverage.

## Alternatives and recommendation

1. **Bounded dependency-aware forgetting — chosen.** A private control ledger,
   durable owner worklist and existing guarded owner adapters cooperate. Fence
   memory use first, then retire copies idempotently and acknowledge outcomes.
   This avoids a new fact authority and fits independent database transactions.
2. **Delete just the record/tombstone.** Useful as today's reversible deletion,
   but Undo, copied staging, proposals and source re-extraction survive. It must
   not be labelled immediate forgetting.
3. **Delete every matching transcript/Note or run global semantic scans.** Rejected:
   content similarity cannot establish ownership or consent and can destroy
   unrelated/manual material. Large generative redaction would also introduce
   provider cost, disclosure and uncertain correctness into deletion.

## Distinct operations shown to the user

| Operation | Retention, use and completion contract |
| --- | --- |
| Archive | Retain content/history; omit according to existing active-record eligibility; restore allowed. No source suppression or erasure claim. |
| Delete with Undo | Current bounded tombstone behavior plus explicit 24-hour encrypted before-image. A future derivative-use hold may prevent use while deleted, but this operation promises reversible deletion, not forgetting. |
| Forget memory now | Exact reviewed record/claim family and known derivative scope. Fence use/admission immediately; no Undo. Retire approved managed copies and install source/family suppression. Keep original transcript/manual Notes unless their separate action was explicitly selected. Report cleanup and peer status independently. |
| Delete source transcript/Note | Source-owned deletion with its own exact preview and retention contract. Source absence does not prove an independently approved claim false. Offer a separate linked-memory forgetting choice; soft deletion alone is not source erasure or memory forgetting. |
| Delete profile everywhere | ADR-102 whole-profile `purge_generation` barrier and key/content lifecycle, plus required external-owner worklist. Retire the old profile identity permanently; distinguish local removal, server cleanup and offline acknowledgements. It does not delete original transcripts/manual Notes by implication. |
| Remove this device's profile copy | Existing local-custody operation, not delete-everywhere. No claim that another device or server erased its copy. |

A forgotten claim is not restored by ordinary Undo/restore or automatic replay.
Explicit re-authoring is a fresh user-reviewed operation and does not remove
source suppression. A separately reviewed policy action is needed to allow a
specific retained source to participate in future memory extraction again.

## Review, lineage and owner authority

A native service prepares an ephemeral exact plan for the user: operation,
profile/scope, target heads, known lineage, source retention choice, each owner
mutation, affected synchronized copies and completion limits. Source-derived
metadata/body display requires that owner's current permission; denied sources
produce a generic unavailable item without identifiers, counts or existence
hints. No model or subagent can approve the destructive operation. Existing
foreground approval and owner floors still apply, including ADR-106 for Agent
Lessons and applied promotion targets.

Dependencies are authority-bound **artifact-version to input-version** edges:
profile claims/proposals, exact ADR-185 bindings, promoted copies, excerpts,
interviews, tool-message copies, compactions, generated Notes and managed export
registrations, Safe/Full captures, semantic trace artifacts and qualified
managed provider-cache lineage. A service writes its artifact and dependency admission token in
its own transaction, using a common publication fence. Owners report missing
registration as unknown coverage; no inference from title, body match or equal
hash creates authority. The reverse inventory is encrypted and cannot become
FTS/RAG, diagnostics or provider context.

Use exact expected versions and a reviewed owner inventory revision. Proposed
operation admission is bounded to 100 explicit targets and 4,096 registered
edges, processed in pages of at most 32 and at most 16 KiB canonical control
metadata per page. Exceeding those bounds rejects admission with a smaller-scope
preview; it does not truncate a deletion and claim completion. No newly found
manual artifact or new destructive owner scope is silently added after review.
Version changes require fresh review or leave the artifact fenced with Needs
attention. Unknown legacy coverage is shown as unverified; a broad “all copies
forgotten” action is unavailable unless every promised owner has qualified
coverage. The server must enforce matching bounds and exact reviewed operations. If the
source's policy forbids storing required retry metadata, its native owner must
admit the source mark/receipt before cross-owner locator retirement; otherwise
admission is refused or the already admitted operation stays fenced with Needs
attention. The coordinator stores only permitted current handles.

### Mixed and independently authored material

- A generated excerpt, summary, proposal or generated Note whose complete inputs
  include the forgotten source is retired as a whole. Do not splice text or
  regenerate with a model during forget. Independent information remains in its
  original sources and may support a later reviewed replacement.
- A user-authored Note or applied instruction file is an independent authority.
  Suppress use of its registered affected derivative until review, preserve the
  owner's body, and offer a separately authorized exact edit/delete. Do not
  silently erase a Note because it mentions the same fact. If original Note
  disclosure is still allowed, that retained content can still reach a model;
  memory-source suppression does not promise to censor every ordinary read.
- A multi-source inference loses required support and is held, even if another
  source remains. A separately reviewed sanitized successor may retain permitted
  assertion text under ADR-185; it never transfers an old assessment or approval
  to new source bindings. Independent direct assertions outside the selected
  family are not silently deleted.

## Suppression against new IDs and retained sources

Suppression is a privacy control, not another memory truth store. It holds only
exact reviewed structured claim-family selectors and native source no-memory
marks, never forgotten quotations or semantic embeddings. Family selectors use
secret-keyed HMAC tokens over a frozen canonical record-kind/semantic-key/scope
projection, independent of record ID; the ledger does not retain readable
subject text. Selector keys are protected separately from payload envelopes and
destroyed with the family ledger at whole-profile purge. Tokens are not public
content hashes and never appear in logs or model context. Key rotation must
retain qualified selector matching until existing controls are migrated; loss
of a selector key blocks automatic admission rather than treating every source
as newly allowed. Core fixtures must freeze the projection/namespace and key
version semantics. The selector input is RFC-8785 canonical
`{record_kind, semantic_key: {namespace, subject}, scope_id}`, using schema-exact
strings and no search normalization. Each reviewed scope gets its own selector;
no undefined cross-scope group is guessed. Promoted descendants' scopes are
explicitly included in the plan.
This does not prove arbitrary differently keyed paraphrases are the same fact.

A source no-memory mark belongs to the native source owner and identifies the
source object in that owner's authorized namespace. Default exclusion is
**object-wide across future source versions**: for a message it covers the
message object; for a Note it covers that Note's memory-extraction input. This
prevents edited offsets/new version tokens from evading suppression. A narrower
span-only exclusion is outside the first release because version mapping would
need a separately proven contract. Do not extend a message exclusion to its
entire conversation without explicit review.

Source marks are retained with the source under its own policy; cross-owner
indexes retain source locators only while source-identity permission allows.
If that permission is revoked, establish the owner-native no-memory mark first,
then retire restricted cross-owner locator/hash metadata as ADR-185 requires.
An unavailable owner leaves suppression unconfirmed and the affected memory
pipeline fenced; the journal cannot claim completion or store forbidden source
identity to make retry convenient. Portable marks require both native source
owner compatibility and current source-control transport permission. Unsupported
remote source owners block the promised portable operation; no bare-ID matching.

Every automatic capture, proposal, direct-update, interview acceptance,
promotion, summary admission, restore/import and Sync application checks the
applicable family/source marks and retirement epoch at **admission and final
publication**. A missing/unreadable control ledger fails closed for these memory
paths. Worker inputs carry registered dependency tokens; a changed epoch makes
all late results inert. Creating a new record/proposal ID or re-encrypting an
old envelope cannot bypass suppression. A new user message on the same subject
can be offered as fresh explicit authorship only through review, never as a
silent automatic override. That review admits only the exact fresh record/version
and its current authorized input under a new operation; it is not a blanket
exception for automatic capture, old proposals, restored copies or later versions.
Existing family/source controls remain in force for those automatic paths.
Copying forgotten material into an unregistered new
source is outside deterministic identity coverage and is not claimed prevented.

The suppression ledger is encrypted at rest, excluded from normal export/agent
views and only user-inspectable under its privacy-control purpose. Deletion of a
record does not delete its controls. Whole-profile purge destroys its encrypted
family ledger along with profile content; content-free retired-profile/purge
barriers survive. Owner-native source no-memory marks survive with retained
sources, without profile payload/labels or cross-owner source references. A new
profile never automatically inherits/extracts those marked sources. Clearing a
mark or adopting new sources requires separate user review; no hidden retained
claim ciphertext is kept after profile purge.

## Durable cross-owner recovery and publication fences

Use the ADR-185 V2 manifest `evidence_retirement_epoch` as the common managed
artifact retirement sequence for portable operations; this design extends its
guarded worklist to forgetting. Device-only operations use an independently
durable **peer-local retirement revision** with the same lease/ticket checks;
they do not advance a shared epoch or send a receipt solely for private changes.
A separately reviewed cleanup of an already synchronized derived artifact may
name only its authorized visible random identity, never the private source
record. These controls are separate from whole-profile `purge_generation`.
Do not add a
parallel record-truth ledger or reuse a random record ID as a global deletion
fence. Shared core publishes a bounded, versioned content-free retirement
control receipt with operation/profile ID, purge generation, epoch, random
artifact identities, page ordinal/count and reason enum. It carries no source
locators, hashes, quotations, values, paths or user-readable kind/label. Receipt
canonical bytes/schema and vocabulary require client/server fixtures; an
integrity tag is not proof of consent or source access. Device-only content and
source marks never enter a portable receipt merely for tracking.

The native coordinator owns one encrypted peer-local operation journal. Private
owner worklists are encrypted under their appropriate custody, with random
artifact references and current-authority handles, rather than storing extra
content before-images. Journal metadata is not a log or an extraction corpus.
Control key custody must survive a partially completed profile purge long enough
to finish cleanup, but may retain only content-free control receipts afterward;
no profile body or wrapped payload key survives in that journal. Independent
source owners retain their own no-memory marks. This is a new reviewed local
storage contract, not an unversioned extension of today's repository tables.

| Phase | Durable behavior and interruption result |
| --- | --- |
| Prepared | Read-only plan with exact heads/authority; nothing deleted. Changed inputs invalidate review. |
| Fenced | After draining publishers and rechecking the reviewed inventory, atomically commit approved intent, next epoch/local revision, local selectors/admission fence and private worklist before destructive mutation. Owner-native source marks are worklist participants whose acknowledgements are required before completion; they are not falsely claimed committed in the coordinator transaction. |
| Retiring | Invoke idempotent owner adapters under operation ID and exact versions. Each retires content plus FTS/index, Undo/recovery, outbox/staging and its linked copies as approved, then records a content-free acknowledgement in the same owner transaction. |
| Needs attention | Owner offline, unsupported, unauthorized, changed or failed. Keep affected use/admission fenced. Report incomplete cleanup; no rollback reconstructs already forgotten content. A new destructive scope requires fresh review. |
| Locally complete | Every promised local owner durably acknowledges all pages and the coordinator verifies retirement/suppression. No readable managed copy in the declared scope remains on live application paths. Source/manual-content exceptions remain as previewed. |
| Awaiting peers / complete within declared managed scope | Server and every active peer acknowledge application of the epoch, cleanup and suppression. A delivery/Sync receipt alone is insufficient. Offline peers remain outstanding; disabled peer grants prevent reconnect but do not certify erased offline storage. |

A coordinator lease prevents simultaneous mutation of one operation; immutable
page IDs and owner receipts make retries idempotent. Crash after a destructive
owner transaction but before coordinator acknowledgement resumes from the owner
receipt, not from a saved body. Startup applies fences before any profile, source
memory or provider pipeline becomes available. Forget takes precedence over
Undo, source refresh, approval, interview resume, compaction and Sync dispatch.
A concurrent profile purge dominates narrower epochs; old-generation work is
rejected. Missing journal/receipt integrity produces Needs attention and holds
use; an interrupted destructive operation has no cancel-and-restore semantics.

Cross-database publication cannot rely on an eventual callback or a final
unlocked epoch read. Qualify one cross-process **retirement admission gate**
for this local profile/control domain. Lock ordering is gate, then native owner
transaction. A publisher enters the gate, checks current controls, and durably
transitions its random ticket from `reserved` to `publishing`. It holds the gate
through its owner commit or qualified provider-adapter entry and durable outcome;
forget admission uses the same gate. Tickets contain dependency/control
identities, not provider payload copies. No model call or network wait runs
inside this short critical section. Owners without this protocol cannot be
included in the guaranteed release.

An owner commit writes the artifact, dependency edge and ticket outcome receipt
in the same native transaction. After that receipt is durable, the coordinator
records `completed` and releases the gate. An app-owned queued handoff remains
managed pending work: queue insertion, worker scheduling and a prepared-dispatch
checkpoint do not prove provider entry or disclosure. The gateway must recheck
controls and consume send authority at its qualified adapter-entry boundary under
the same gate before starting provider execution. Its content-free receipt
distinguishes pending/cancelled from adapter entry with send potentially begun;
it does not certify network delivery. An adapter that only acknowledges queue
insertion cannot qualify this boundary. No network wait is held inside the gate.
Forget cancels and retires pending payloads, delayed retries and fallback queues
through their owners; every later app-owned adapter entry needs current controls.
For admitted in-flight execution, request cancellation where supported and report
already-started or uncertain disclosure honestly. Adapter entry does not exempt
remaining app-owned payload copies from cleanup. Future callbacks still check the
current epoch before any managed save/publication. Manual Note/source edits
outside memory-producing paths retain normal authority.

Forget admission first closes new reservations for affected dependencies and
drains existing tickets under the gate. `reserved` may transition durably to
`cancelled`; the owner must check that state before entering `publishing`.
A `publishing` ticket cannot be cancelled by notification, timeout or TTL. Its
owner must finish durably, or roll back and acknowledge a terminal non-publishing
state that forbids retry under that ticket. Only then can forgetting commit its
fence. Drain completion rechecks exact inventory/versions; an intervening
artifact or changed destructive scope invalidates the prepared review instead
of being silently appended. A cancellation request alone is never an ack.

If the process crashes after its last epoch check, during owner commit or before
coordinator acknowledgement, the durable `publishing` state survives loss of the
OS gate. Recovery resolves the exact owner receipt: completed commit is inventoried
before forget admission; verified rollback becomes cancelled. An ambiguous or
unreachable owner means Needs attention, no new publishing or destructive fence
claim, and no expired-lease shortcut. Cleanup phases resume only from durable
receipts; they cannot reconstruct content before-images. This defines the
linearization point: publication completes its admitted owner/adapter-entry transition
before the common gate permits the forget fence, or is durably prevented from
publishing. An owner reconnecting without current controls cannot publish.

### Agent audit, capture, trace and managed-cache qualification

AgentRunsDB step payloads, legacy step blobs and terminal results, plus segmented
filesystem run logs and their reader/search/export surfaces, are independent
managed owners. `_safe_run_log_content` and durable-step sanitizers can preserve
ordinary profile assertions; they redact credentials/paths/reasoning, not all
profile facts. Register source dependencies before either DB or file publication,
including transformed model output and child reports. Native retirement must
fence live/late append, terminal recovery and log search/slice reconstruction.
Respect append-only/source-sequence invariants: retire an exact reviewed generated
unit through qualified maintenance, or a separately governed replacement; never
rewrite historical bytes through unreviewed substring edits. A mixed segment or
run unit needs whole-unit scope shown in review; any retained root/copy, unavailable
file authority or unknown legacy lineage prevents complete managed retirement.
Do not silently delete an entire workspace log directory.

Register profile-input dependencies before capture/trace publication, including
Safe first-system content, Full payloads, response/stream material, transformed
components, reconstruction/export artifacts and legacy normalization. Native
cleanup covers `message.exchanges`, compressed blob caches, persistence queues,
Safe/Full rows, semantic trace components/blobs and relevant retention/selection
roots. Existing Full-only purge and graph garbage collection are useful seams,
not proof they cover a profile-specific forget. Use the trace owner's allowed
retention/maintenance transaction, respecting append-only invariants; never
bypass triggers with direct profile cleanup SQL. Fence late capture flush and
normalization just like proposal/compaction commits. If an independent root
still retains a shared blob, report that retained scope and require review or
leave cleanup incomplete; no shared-content erasure claim follows from removing
one reference. Independent transcript bodies remain as explicitly previewed.

ADR-119 caches need native launch/catalog ownership, input lineage and known
writer completion. They have no complete claim-to-binary mapping today and are
an explicit unsupported coverage condition, not automatically exempt merely
because their data reached a provider. App-owned working/retained copies cannot
be decoded or searched for a forgotten phrase to establish absence. A future
adapter may retire whole exact reviewed owned snapshots and stop/quiesce their
writers through its normal owner; uncertain Save/Restore or unknown live slot
state keeps Needs attention until the native owner qualifies its outcome. The
initial Forget now scope cannot promise complete managed-copy retirement when
affected app-managed cache coverage is unknown. Independent external provider
cache custody remains outside recall.

## Remote custody, restore and honest erasure

ADR-102 retains the current `server_trusted_v1` posture: canonical payloads travel
over authenticated TLS to a trusted server and are re-encrypted at rest. This
design does not claim end-to-end encryption or erase the server by deleting a
client key. The home server must fence old writes, retire its canonical/history,
proposal, staging, recovery and governed derivative copies, then acknowledge the
exact epoch. Clients replay control before content, discard stale snapshots and
acknowledge all pages before memory use/Sync resumes. ADR-185's profile-wide V2
activation rule also applies to forgetting/suppression consumers.

Managed restore/import checks retired profile IDs, current purge generation,
epoch and source marks before making recovered content usable. A restore that
cannot contact the authoritative current control owner stays quarantined; it
cannot revive an older manifest or erase the suppression ledger to make content
valid. External backups/exports are disclosed as outside recall guarantees.
Admission stores no forgotten plaintext or public unsalted content hash for
suppression: those would themselves retain the secret or permit guessing.

Local completion means logical removal from declared managed live reads and
retirement of retained managed artifacts. SQLite soft-delete flags alone are
insufficient; owner cleanup includes hidden histories, FTS, revision/Sync logs,
recovery and accessible blob stores. This task has not qualified owners' secure
physical deletion. WAL pages, storage snapshots, forensic remnants, unregistered
files and delivered provider requests cannot be certified erased by a row delete
or ordinary envelope/key cleanup. A stronger cryptographic/physical-erasure
claim requires its own key-lifetime and storage proof. Forget cannot unsend an
already dispatched request; cancel future callbacks/derived saves and show that
existing disclosure lies outside local recall.

## Bounded implementation release and gates

The first future release is **local, user-initiated, registered-lineage forgetting
for an unlinked profile**, restricted to newly captured V2 claims with complete
registered lineage. V1 references are never promoted into graph edges by guessing;
legacy unknown coverage keeps reversible deletion available but cannot support
this first release's complete forget guarantee. Untyped/legacy claims without a
canonical family key are outside that release. Include
profile versions/Undo/proposals/promotions, supported message/note source marks,
interview drafts, prepared/root/child context and persisted tool/compaction
copies, AgentRunsDB steps/results and segmented run logs, Safe/Full captures
and semantic trace components/artifacts, generated
Note derivatives, and local staged/recovery copies. Managed cache uncertainty
is a coverage gate as stated above. Independent
manual Notes/instruction/file changes remain separately reviewed. Do not offer
this scope while any included owner has unknown coverage or cannot enforce the
publication/fence contract. No provider call is needed for cleanup.

Implement in atomic deliverables: shared-core control fixtures/semantics;
encrypted journal and guarded profile retirement; source/proposal/interview and
request/capture/trace-owner adapters; conversation/compaction/Notes/Sync staging
adapters and explicit managed-cache qualification;
then local crash/race qualification. Gate new governed evidence/excerpts and
memory-generating jobs on shipped forgetting and disclosure safeguards, not
Done design tasks. Linked-profile/portable forgetting requires a separate
server-qualified release covering current and oldest advertised clients,
source-owner transport, grants, offline return and restore. No arbitrary task
IDs or runtime implementation permission is created here.

## Synthetic future acceptance cases

| Case | Required result |
| --- | --- |
| Archive versus delete with Undo versus Forget now | Archive retains content; deletion retains explicit 24-hour Undo; forget admits fence/suppression, creates no Undo and retires prior relevant before-images. |
| Workspace record promoted to a new global ID | Exact approved lineage reaches the promoted copy and related proposal; new ID cannot evade cleanup. Unrelated independently authored global claims survive unless selected. |
| Retained source edited or proposal recreated under new ID | Native object-wide source mark and canonical family control still deny automatic re-extraction/admission. No offset/ID/hash substitution bypass. |
| Generated summary contains forgotten and unrelated sources | Retire the whole generated summary and its descendants; original independent sources survive, no model redaction during deletion. |
| Manual Note or applied AGENTS file includes the fact | Preserve original user-owned body; hold registered affected derivative and offer exact separate edit. No automatic file rewrite or similarity deletion. |
| Source policy forbids locator retention after evidence sync | Mark native source no-memory, fence publication, retire restricted inline/worklist metadata, then acknowledge. Offline owner means incomplete, not forbidden locator retention. |
| Interview draft has answer/proposed changes after record delete | Separate store/key and pending results are retired; resume/accept cannot recreate the family. Unknown legacy linkage blocks a complete guarantee. |
| Tool response or compaction job finishes after forget | Its source/epoch ticket fails final publication/dispatch; no new chat artifact, proposal or summary is committed. Already dispatched provider data is reported outside recall. |
| Profile tool output or a child response is retained in AgentRunsDB and a run-log segment | Retire both exact registered DB/file units through their owners; safe sanitization does not establish absence. Fence late append/terminal recovery and search/slice/export reconstruction; mixed units require explicit whole-unit review. |
| Prepared provider payload is queued when forgetting is admitted | Cancel/retire the app-owned payload and delayed retry/fallback work; queue insertion is not delivery. Final qualified adapter entry is refused after the fence, while uncertain already-entered work is reported separately. |
| Safe/Full capture includes the profile-bearing first system message | Retire registered capture blobs/caches and semantic header/artifact copies via native owners; Safe and credential redaction do not exempt them. Inspector/export reconstruction cannot revive the retired content. |
| Capture flush or legacy trace normalization arrives after the fence | Its ticket/epoch is invalid; no exchange or semantic artifact is recreated. Shared retention roots still keeping the blob prevent an unqualified complete-erasure claim. |
| Cancellation arrives between the last ticket check and owner COMMIT | The publishing gate prevents the forget fence until owner commit/rollback is durably known. A request/TTL is not cancellation acknowledgement; crash recovery uses the same ticket receipt before admitting cleanup. |
| App-owned prompt-cache snapshot/working copy has uncertain lineage or an active writer | No binary content search or implicit deletion/reset. Needs attention blocks a complete managed-copy result until exact native review and qualified quiescence/retirement. |
| Profile outbox already copied to Sync database | Retire both original and destination envelope/history/recovery; a source-body receipt alone is insufficient. |
| Crash before fence, after owner cleanup, or before peer acknowledgement | Before fence: no deletion. Later: recover idempotently from control/owner receipts, no before-image restoration; report current partial status. |
| Source root offline, scope permission denied, or user edits derivative | Keep use fenced, give privacy-safe Needs attention and require current owner consent/version review; no cross-store search or mass deletion. |
| Old-generation Sync/recovery and stale prepared/root snapshot | Apply purge/epoch controls first; stale content cannot reactivate or reach a new request/child. Invalid control integrity fails closed. |
| Device-only forgotten claim with syncable siblings | Do not publish its IDs/source marks in shared receipts or counts; acknowledge only the authorized syncable scope. |
| Offline peer or server acknowledges delivery only | Await cleanup/suppression acknowledgement; do not label remote erasure complete. Retired grant blocks reconnect, not offline decryption. |
| Whole-profile purge followed by a new profile | No retained profile body/key; old identity/generation cannot restore. Source-owned no-memory marks still exclude retained sources until fresh policy review. |
| Unmanaged export, filesystem snapshot or original retained transcript | Preview and result state exclusions; no universal erasure, transcript edit or backup recall claim. |
| New explicit user assertion about the same subject | Exact fresh user review admits only the reviewed record/version and authorized input; it does not clear family/source controls for old proposals, replay or later automatic versions. |

These are future production-entry/crash/race tests, not passing runtime results.
Use synthetic sources and successful authorized controls. This task runs only
scoped document/tracker verification and technical review.

## Technical review checklist

- [x] Actual owners and unknown coverage are distinguished from future adapters.
- [x] Operation/Undo/source-retention and immediate-fence semantics are distinct.
- [x] New IDs, edits, restart, replay, jobs and restore cannot bypass admitted controls.
- [x] Cross-owner consent, metadata retirement, mixed sources and manual Notes are explicit.
- [x] Crash, peer acknowledgement and physical/unmanaged-erasure limits are honest.
- [x] Bounded release has owner, shared-core and server qualification gates.
- [x] All six criteria are covered; only documentation/tracker files changed.
