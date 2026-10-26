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
