# ADR-138: Portable workflow definitions and local execution

- Status: Accepted by the user on 2026-09-08
- Date: 2026-09-08
- Historical source task/design: TASK-32077 and Docs/superpowers/specs/2026-09-08-workflows-local-first-parity-design.md on preserved codex/workflows-local-file-to-note.
- Current task: [TASK-32601](../tasks/task-32601%20-%20Restore-the-authoring-only-Workflows-editor-on-dev.md)
- Current delivery: [Authoring-only dev port](../../Docs/superpowers/specs/2026-09-14-workflows-authoring-dev.md)
- Revision: the user approved the revised Chatbook contracts and safety defaults on 2026-09-08. The paired-server synchronization protocol still requires counterpart agreement and implementation; this acceptance does not assert server support.
- Supersedes: N/A

## Current dev delivery scope (user approved, 2026-09-14)

This port delivers authoring only: create/edit/save/import/export, recoverable drafts,
and the reviewed three-pane UI. Local execution is unavailable in this delivery.
The original runtime decisions below remain historical design context, not tasks
for this slice. The later helper-owned lock/schema-v5/PID recovery proposal is
withdrawn from active scope and is not imported here. Existing ADR-125 private
SQLite utilities and ADR-150 UI tokens govern integration. Preserve migrations
v1-v4 for document-store compatibility; introduce no execution lock or execution
API and do not claim completion of the wider original milestone.

### Stable-file exchange contract (user approved, 2026-09-15)

The user approved retaining v1 file-picker import/export under a stable-file
assumption: do not externally move, replace or relink the live workflow database
or its sidecars while the store is open; do not move, replace or relink the
selected JSON file or its containing path during exchange. Normal SQLite-managed
writes and sidecar lifecycle are not prohibited by this assumption.

Retain the feature-local metadata precheck: reject visible database/sidecar
aliases, symlinks, multiple hard links and non-regular files before generic file
I/O. Existing private-file checks, permissions, size bounds and overwrite
confirmation remain required. This precheck is not a path lease or retained-inode
proof. Concurrent substitution after validation, or moving a live inode away
from its known database name, can bypass it; raw opening/closing of that inode
can still disrupt SQLite locking. That race is accepted outside this delivery's
operating contract, not technically fixed or qualified as safe.

This narrowly amends the authoring spec's unconditional file-exchange guarantee;
it does not change ADR-125's shared SQLite validation/proof implementation or
weaken other storage owners' contracts. No new helper operation, process, lease,
schema or runtime is authorized. Deferring picker exchange and introducing an
isolated JSON-I/O boundary were considered; the user chose the bounded existing
implementation. Static-analysis debt, review and integration gates are not waived.

### Task-specific static qualification (user approved, 2026-09-15)

After the final authoring review, the user separately approved a no-new-static-debt
gate for TASK-32601: fix findings introduced by this slice and retain documented
baseline failures. Compare against the clean dev base `77eb2601a6` with the same
Ruff version/configuration. Match diagnostics to unchanged source spans, not only
net counts; inspect unmatched or changed-block findings and correct introduced
issues. New/rewritten files must pass scoped lint and formatting. Attribute
remaining formatter edits to baseline code rather than declaring whole files clean.

This replaces the whole-file-clean prerequisite for this task only. Existing
713 Ruff findings and five formatter failures were reported before qualification;
report actual remaining counts afterward, without suppressions, broad reformatting,
CI/config changes or removing runtime checks to satisfy lint. Other behavioral,
visual, privacy and review requirements remain. The stable-file limitation above
is unchanged. This is not permission to merge, push, execute workflows or change
shared SQLite infrastructure, nor an inherited exception for other tasks.

### PR review hardening (2026-09-15)

The requested PR integration includes bounded collection reads: workflow heads,
revision history and local drafts accept a page size of 1–100 (default 20) and
a nonnegative offset, rejecting booleans and nonintegers before opening a
transaction. Lists retain their existing deterministic order. UI selectors expose
previous/next pages; a page is not an assertion that no other items exist.
Workflow search applies across the entire library, using bounded name/identity
reads and Unicode casefold matching before retrieving matching full definitions.
Selected-head lookup is independent of the visible page, so paging/search cannot
change a draft's identity or disable recovery/export. Reads run through existing
workers and SQLite transactions; no schema, connection owner or storage subsystem
is added. Offset pages are a live view, not a snapshot across external writes.

New structural admission also limits definitions to 500 steps, 64 container levels
and 100,000 value/container nodes, including opaque subtrees. The 16 MiB raw-text
limit is unchanged. These are local authoring bounds, not server schema claims.
An iterative bounded check precedes recursive projection/dependency traversal;
bounded rejected raw edits retain the previous valid form projection and remain
recoverable. Existing above-limit saved content must remain exportable without
requiring projection or silently rewriting it. Derived display/required-field
reads reuse a prepared projection, never a new full parse per field. The measured
500-step/large-scalar case spent 28 seconds on 1,335 complete parses versus 24 ms
for one projection; remove this multiplicative work before adding concurrency.
If bounded remaining pure analysis exceeds 100 ms, use the existing worker seam
and freshness checks; draft ownership and widget mutation remain on the app loop.

## Context

The user approved a Workflows redesign with a workflow library, step navigator, and overview/focused-card canvas. Forms are continuous and collapsible. V1 must execute workflows locally without tldw_server; branching follows in v2 and parallelism in v3. Definitions should be shareable and synchronized across Chatbook and tldw_server where reasonably possible.

The server dev baseline `6cd2745f696af04668a61c20b84ab8a9e69ca5e4` provides an ordered-step API definition, an extensible step catalog, and immutable saved versions. Its web canvas export is a different nodes/edges representation. Workflow visibility is private and workflows are absent from the current Sync v2 domains. Chatbook has reusable local domain services but no corresponding general workflow editor/engine.

## Decision

1. Use the server API workflow definition as the common execution/exchange document. Preserve stable step IDs, config, and metadata. UI projections and local view preferences do not become an alternative execution format.
2. Provide a Chatbook-owned sequential engine and a single run-state service over private workflow persistence. Reuse local services through explicit adapters. Do not embed the server application or create a second implementation of each underlying domain service.
3. Select Local or a configured server for a complete run. Unsupported local capabilities fail preflight; there is no automatic remote fallback. Report local execution separately from external-network requirements.
4. Execute immutable saved snapshots. Draft changes cannot modify running work. Interrupted or uncertain side effects require recovery review rather than automatic replay.
5. Propose a versioned `metadata.tldw_workflow` namespace for global workflow/revision IDs, revision parents, and portable requirements. Bind actual resources and credentials per installation. Adopt this namespace through a paired server/client contract before claiming cross-client sync support.
6. V1 supports reviewed canonical JSON import/export and explicit fetch/publish against existing servers. A separate v1 sync workstream adds an advertised `workflow.definition` domain, durable revision checks, recoverable server version projection, and explicit conflicts. It must not overload an existing notes/chat domain or imply current server support.
7. Sync saved definition revisions only in v1. Local drafts, credentials, grants, active runs, results, and view state retain their local owners. File sharing creates an explicit reusable copy/provenance relationship and does not grant execution authority.
8. Preserve unsupported branching/parallel definitions. Editing/execution capability follows the v1/v2/v3 roadmap, not whether a parser can retain the JSON.
9. Materialize reviewed ordinary defaults/run values and requirement-owned destination bindings into explicit run inputs. Capture non-secret selections, target identity, contract revisions, limits, and launch-operation identity in a private manifest. Resolve credentials only through the bound credential owner or a proven adapter-specific runtime-secret path. The server is not assumed to interpret the proposed metadata namespace.
10. Evaluate the whole definition's effects before admission/export, including completion callbacks and opaque metadata. Non-empty completion hooks block local v1 execution; remote execution requires destination/output disclosure and normal egress authorization. Preservation never grants execution authority or certifies opaque content secret-free.
11. Retain physical attempt ownership until all owned work exits. Separate engine timeout, adapter request timeout, and durable human-response deadline. Repeated launch delivery uses the same operation identity; an uncertain effect or unconfirmed remote launch cannot automatically create a replacement attempt/run. Run budgets persist through retry/restart, and policy increases are explicit recorded amendments.
12. Recover drafts independently of saved revisions, including invalid editor buffers. Navigation waits for durable draft persistence or explicit loss acknowledgment. Results identify their original run/revision/target/attempt even while a different draft is edited. Define keyboard focus transitions for collapsed/hidden regions and typed input references as part of the application structure.
13. Deliver a local file-to-note end-to-end slice first, retain the complete 21-step v1 target, and retain paired-server workflow sync as an explicit v1 milestone. This sequencing does not move branching from v2 or parallelism from v3.

## Alternatives considered

| Alternative | Reason not selected |
| --- | --- |
| Adopt the web editor's nodes/edges export as the common format | It differs from the execution API and currently loses metadata/version information. |
| Require a server for every run | Does not meet the approved local execution requirement. |
| Import the server's complete runtime | Couples the TUI to FastAPI, server databases, and server scheduler dependencies. |
| Reimplement LLM/RAG/media services inside workflows | Duplicates established behavior and creates avoidable parity drift. |
| Offload unsupported steps automatically | Changes execution authority and data movement during a supposedly local run. |
| Sync by name/version or last-write-wins | Names and integer IDs are not portable identity; concurrent edits can be lost. |
| Add workflow content to an existing sync domain | Changes a negotiated contract without corresponding server support. |
| Flatten imported control flow for v1 | Changes the meaning of a shared workflow. |
| Auto-merge concurrent step edits | Array order, references, and prompts can conflict semantically even when fields differ. |
| Save only valid revisions; retain no invalid editor buffer | Ordinary navigation or a restart could lose unfinished authoring work, even though run history is durable. |
| Generate execution controls directly from discovered config schemas | The inspected server distinguishes step-level controls from adapter configuration and human-response deadlines; schema presence alone does not establish runtime semantics. |
| Free capacity as soon as a run is marked timed out/cancelled | An owned tool worker may still be running and producing effects, allowing unsafe overlapping retries. |
| Block the first usable workflow on all baseline adapters and server sync | A smaller end-to-end slice can validate the shared contracts while preserving the eventual v1 scope. |

## Consequences and constraints

- Adapter availability is separate from definition readability and form coverage; a type or schema entry is not a working local adapter.
- Existing servers remain usable for explicit exchange, while sync readiness depends on negotiated support and metadata-preserving clients.
- The new store must register with private-path, backup, migration, and lifecycle ownership under ADR-029 and the current backup design.
- Runtime/source authority remains with ADR-033 owners; no new root AppState is introduced.
- Workflows owns authoring and run summaries; Console retains detailed live activity, consistent with ADR-011.
- Existing tool permission, path, and credential boundaries continue to apply to imported workflows.
- The approved local profile is workflow-runtime scoped, not a machine-wide quota; domain-owned notes/media are not workflow artifact cleanup targets. Remote limits are advertised as enforced only when the server actually supports them.
- Safety/privacy differences from the server, including payload-bearing log handling and unresolved-expression rejection, are named conformance restrictions rather than hidden format changes.
- This ADR records accepted Chatbook architecture, not approval to implement a new server protocol or a claim of completed parity.
- Identifier allocation was checked against local remote refs and worktrees and must be rechecked before integration.

## Related decisions

### Local provider identity correction (user-directed, 2026-09-09)

The illustrative Ollama mapping is not a product restriction. The first local
slice supports configured `llama_cpp` and `ollama` identities over the existing
captured OpenAI-compatible chat-completions transport. Run setup selects the
provider and model from its captured configuration; the actual provider, model,
endpoint and bounded settings remain in the local binding and run manifest.
Revalidation and dispatch must preserve those facts. A llama.cpp endpoint must
never be represented as Ollama just to pass admission.

This corrects the implementation's accidental provider lock-in without adding
a transport or broadening egress authority. Numeric-loopback/keyless admission,
zero automatic retries, redirect/proxy refusal, absolute request deadlines and
physical attempt ownership remain unchanged. Unsupported providers stay explicit;
there is no automatic fallback. Existing Ollama bindings remain valid. Portable
examples and historical captures do not establish installed-provider availability.

### Task 7 local composition clarification (controller approved)

TldwCli captures the local service/configuration facts once, before setup awaits.
For the verified secured lexical absolute profile directory P, profile_id is
`str(uuid5(NAMESPACE_URL, "tldw-chatbook:workflow-profile:" + P.as_uri()))`.
For the exact validated configured Notes user N, actor_id is
`str(uuid5(UUID(profile_id), "review-actor:" + N))`. For the coherent Notes
owner's verified actual lexical absolute database path D, database_id is
`str(uuid5(NAMESPACE_URL, "tldw-chatbook:notes-database:" + D.as_uri()))`.
Notes client attribution is N. The logical database ID is independent of actor
and profile. These IDs are not authentication, secret protection, global
installation identity, or the existing local character authority UUID. Existing
no-follow path/device/inode, opaque Notes destination and profile checks remain
authoritative. Paths are not resolved through aliases; usernames are not
normalized. Directory/path spelling/database selection/user changes alter the
corresponding identity after restart and may make old recovery unavailable.
There is no automatic migration or retarget. Persistent random IDs were rejected
for this slice because they need a new durable owner and migration policy.

Workflow construction happens once on the actual application loop, after local
Notes and permission services exist. A retained setup operation is joined by
readiness/navigation/shutdown; it never replaces a failed graph. Authoring and
history may remain available when execution is unavailable. Admission closure is
a separate public runtime operation from physical stop: existing admitted work
remains owned while drafts flush, then stop/drain settles it before dependent
storage closes. Failed settlement preserves the error and recovery ownership;
remaining application lifecycle owners must still be drained. Caller cancellation
does not mean a shielded owner finished.

Notes destination acquisition is exception-safe even when it opens a per-client
connection before validation/token publication fails. Workflow setup and each
runtime lane close only their borrowed current-thread Notes connection through
the Notes owner's public destination seam, retaining routing cache/token identity.
Cleanup must not perform a fresh route lookup, create/rebind a database, or close
another thread's connection. A legitimately captured old destination can still
be cleaned up after routing changes. Cleanup failures do not skip other drains
or hide a durable settlement failure.

Ordinary Quit prepares the app-owned draft through the existing pre-quit guard,
regardless of the active screen or overlay, before committing exit. Failed
persistence leaves the exact draft/provenance interactive for Retry or staying;
it never discards automatically. Successful DraftSession.close seals accepted
edits before final audio/cache cleanup. Final/forced shutdown still closes
admission, drains drafts and runtime, releases borrowed connections, then closes
storage; retaining an unsaved in-memory buffer after forced teardown is not
reported as durable recovery.

### Task 7 review: exact result inspection (controller approved)

Retry presentation reads a fail-closed hint from the run owner's exact current
step/attempt, durable retryable/replay-safe/non-uncertain outcome, physical status
and effective attempt limit. Computing the hint never mutates state or authorizes
a retry; the existing control owner revalidates admission when invoked.

Open result is read-only access to an exact successful Notes attempt, not mutable
Library routing. RunService.note_result_request returns only the detached durable
request whose successful receipt belongs to that run and step.
WorkflowRuntime.read_note_result validates run/profile/revision and retains the
read as owned work on its serialized state lane. Cancellation cannot abandon the
read; shutdown drains it before borrowed connections or dependent stores close,
and closure refuses new reads. LocalWorkflowServices.read_note_result uses the
existing captured Notes destination and public get_note_by_id, checking profile,
database, client, lexical path, physical identity and returned Note id/client/
deleted state before/after reading. Existing strict permission refusals remain.
There is no create, effect approval, route repair, replacement owner or fallback.

The UI distinguishes immutable historical receipt/provenance from current content
of the same retained Note. A normal later edit is not an uncertain effect or a
receipt mismatch. Missing/deleted or changed authority is explicitly unavailable;
unrelated content is not disclosed. Multiple Notes results require exact step
selection. The read-only viewer does not promise editable Library navigation and
does not place Note content in Console handoff metadata. No schema or identity
policy changes are introduced.

### Task 5 ownership clarification (controller approved)

The local milestone enforces one workflow execution owner per Workflows DB with
a lazily acquired, nonblocking OS lock on a private, stable, never-unlinked
sibling file. Ownership lasts through parked waits until all physical workers,
observers, admitted controls and state writes drain. A second app may author and
inspect but cannot execute/recover against that DB while the owner remains live.
This stronger per-database lifetime exclusion has a deliberate second-app
execution cost; it is not a queue or an application-wide lockout. The app's
advisory instance-lock policy is unchanged. In-memory stores share their guard
through the same WorkflowsDB object.

The DB owner supplies validated store path/identity, rejects replacements and
unsafe aliases, and serializes ownership transfer of run/attempt/capacity rows
in one transaction under the guard. Only actual OS contention means another
owner; lock errors fail closed. PID, age, or lock-file existence are not evidence
of liveness. Runtime shutdown drains but never closes its injected DB; the app
coordinator flushes drafts and closes stores after runtime settlement.

File-backed execution/recovery is unavailable before lock-file mutation if
required pinned no-follow descriptor/identity safeguards are unavailable. This
includes the existing Windows ordinary-open fallback. ADR-029's Windows ACL
privacy posture remains UNVERIFIED for existing storage/authoring; this milestone
does not add Windows ACL/reparse/native-lock implementation or claim live Windows
qualification. Simulated capability-loss tests prove refusal/no mutation only.
Recovery reads/reconciles without automatic dispatch; imported decisions never
provide an in-memory effect grant.

Fix round1 clarifies the protocol origin: schema4 records
`legacy_ownership_unverified=1` for **every** prior schema, including schema3 with
an existing bound lock. An earlier process may never have held the new lock and
may admit work after a migration even when no runs/capacity were visible. Only
original schema0 initialized under the same BEGIN IMMEDIATE migration transaction
may clear that marker. Pre-v4 prototype stores therefore remain unavailable for
execution/recovery, while authoring/inspection and their stored ownership/capacity
remain intact. This deliberate compatibility cost is not completed legacy recovery;
there is no takeover/reset API, PID/age inference or child-exit-based clearing.
A separately qualified future migration would be needed to establish safe legacy
execution. Shipped migrations1/2/3 remain immutable.

The approved response-duration range is unchanged. Schema4 positive deadlines use
`utc-us-v1:` plus 25 zero-padded decimal digits: exact UTC microseconds since
0001-01-01T00:00:00Z. The largest aware UTC clock plus signed-int64 seconds fits.
Comparisons use the same canonical encoding, without float timestamps, saturation,
rounding or datetime overflow. NULL alone means unlimited; opened_at remains ISO
UTC. A public read/display seam in the private workflow deadline owner prevents
Task7 from parsing this format or overflowing on display; legacy ISO is readable
for inspection, never compared lexically against the new tag.

Recovery preserves durable cancel/timeout intent in existing terminal status and
original error_code fields, including across failed binding/permission/receipt
revalidation and reopen. A verified late Notes receipt may settle only the
original attempt/effect and ledger, never forward outputs or a successor step.
Task7 reads deadlines through `Workflows.utc_deadlines.read_deadline(value, now)`:
the frozen DeadlineView carries unlimited, expired, iso_utc and exact display;
a finite out-of-datetime-range value has no ISO but is never unlimited.

### Task 3 implementation clarification

Workflow execution uses an additive strict read on the existing MCP permission
store and an opt-in strict BuiltinToolGate check. Missing, unreadable, corrupt,
or malformed authority is unavailable; execution neither repairs it nor treats
the incumbent fallback/cache as fresh permission. Kill and deny outrank a
one-effect approval. Ordinary load/check defaults and Console caching retain
their existing behavior. The existing resolver remains the only rule owner.
Approval is a single-use in-memory handoff tied to run, step, effect, resolved
config and captured bindings; durable wait decisions remain the run owner's
responsibility. Private model usage is separate from portable output keys.
Notes composition explicitly binds its logical database-owner ID, profile and
client attribution to the captured service's real DB path and identity; these
identities are distinct and cannot be supplied as a permission grant by a
portable definition.

Task 3 review clarification: Notes destination identity must be captured from
the per-client database actually selected by the Notes domain owner, not only
its mutable template. An opaque owner-issued destination token guards create
and receipt-read routing; unbound legacy Notes operations keep their defaults.
The captured model timeout is an absolute network deadline covering connect,
headers and body, not merely a socket inactivity timeout or a check between
chunks. Any cancellation resource is owned and drained before adapter return.
The bounded Requests path uses per-session HTTP/HTTPS pool connection classes,
remaining connect budget and an owned timer armed before post. A cancellation
socket duplicate retains the live connection through TLS wrapping; expiry shuts
it down, and registration after expiry refuses immediately. Timer cancellation
and join plus socket/response/session cleanup precede physical return. No request
runs in an untracked helper, and default callers keep their existing transport.

- [ADR-008: Sync v2 client contract alignment](008-sync-v2-client-m1-contract-alignment.md)
- [ADR-011: Workbench UI system](011-chatbook-workbench-ui-system.md)
- [ADR-029: Local private data boundary](029-local-private-data-boundary.md)
- [ADR-031: TUI keybindings](031-tui-keybinding-and-footer-hint-conventions.md)
- [ADR-033: Application session state ownership](033-application-session-state-ownership.md)
- [ADR-073: Notes interoperability constraints](073-notes-sync-round-trip-and-interoperability-constraints.md)
- [ADR-068: Local research execution precedent](068-local-research-execution-engine.md)
- [ADR-126: Local backup and recovery](126-complete-local-backup-and-recovery.md)
