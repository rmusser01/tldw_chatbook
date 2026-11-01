# Shared Console ownership and mobile agent supervision

Date: 2026-10-25
Status: Design accepted 2026-10-25; implementation and scoped code reviews are present at code22bae10d3dd6. Fresh UAT on8491688b was13P1F; later retention checks passed but the first mouse failure remains unclassified. Task11/AC1 is reopened; physical phone/VPN remains unverified.
Task: [TASK-36728](../../../backlog/tasks/task-36728%20-%20Design-shared-Console-ownership-and-mobile-agent-supervision.md)
Decision: [ADR-244](../../../backlog/decisions/244-shared-console-ownership-and-mobile-supervision.md)
Implementation plan: [Shared Console mobile supervision](../plans/2026-10-25-shared-console-mobile-supervision-implementation.md)
Delivery evidence: [Task11 qualification packet](../qa/2026-10-25-shared-console-mobile-supervision/README.md). Automated/emulated checks and physical phone/VPN qualification are recorded separately; this status does not claim release completion.

```text
ADR required: yes
ADR path: backlog/decisions/244-shared-console-ownership-and-mobile-supervision.md
Reason: Changes storage and live execution ownership, served process lifetime,
        authenticated control contracts, and legacy data consolidation.
```

## 1. Purpose and approved scope

A single operator connects over VPN to inspect, correct, and stop agents already
running in Console, switch between existing chats, answer decisions, and steer
individual subagents. Start with `--serve`; native terminal attachment follows in
a later phase. A browser or VPN disconnect must not end accepted work while the
served server and its owning app process remain alive.

This work removes the separate mobile storage and execution island. `/m` becomes
another interface to ordinary Console chats. Phone-created chats use the same
configured storage, workspace registry, runtime behavior, and permission owners
as chats created through the served terminal. Existing mobile history must be
consolidated safely before retiring its old write path.

The first release includes the Chats-first phone workflow, live transcript and
tool activity, pending decisions, main-agent Send/Steer/Redirect/Stop, and direct
subagent inspection, guidance, and Stop. Native process registration, multi-user
accounts, offline command execution, new agent scheduling, and automatic replay
after a server crash are outside this release. Existing Library reading remains
available; this is not a mobile rewrite of Settings, ingestion, or persona editing.

## 2. Current code and the actual gaps

The inspected baseline is `51782694ac47ddfc223d83aa03e942a22496e809`.
These are source facts, not runtime verification results:

| Existing owner | Source and behavior |
| --- | --- |
| App-owned Console execution | [console_runtime.py](../../../tldw_chatbook/Chat/console_runtime.py): `ConsoleRuntime` owns store, gateway, bridge, controller and accepted-turn custody. `leave_console` detaches a view; `dispose` ends the runtime. Viewless construction and execution already exist. |
| Agent execution and subagents | [console_agent_bridge.py](../../../tldw_chatbook/Chat/console_agent_bridge.py): `run_reply` constructs the existing `AgentService`; `fleet_snapshot`, `steer_subagent` and `cancel_subagent` expose the live fleet. Steering uses the conversation-owned mailbox. |
| Current mobile island | [mobile_service.py](../../../tldw_chatbook/Web_Server/mobile_service.py): `_build_engine` creates another controller/store/bridge over `mobile/chat.db`, `mobile/workspaces.db` and `mobile/runs.db`, with a `HeadlessAppContext`. `_restore_persisted_session` reopens mobile history. |
| Served process lifetime | [serve.py](../../../tldw_chatbook/Web_Server/serve.py): `handle_websocket` constructs an `AppService`, binds its output callbacks to the socket, and stops it on disconnect. The installed textual-serve `stop` sends the child a quit message. |
| Existing process transport | [control_protocol.py](../../../tldw_chatbook/Canvas/control_protocol.py): authenticated, bounded parent/child requests and events. Existing relay messages cover approvals and run status. Broker startup and revocation currently follow Canvas enablement. |
| Projection overflow hazard | [control_protocol.py](../../../tldw_chatbook/Canvas/control_protocol.py) closes the child control connection when its bounded event queue fills. [serve.py](../../../tldw_chatbook/Web_Server/serve.py) starts the relay drainer only on first mobile use. Retaining this combination would turn ordinary projection traffic into execution-owner loss under the new shutdown policy. |
| Current relay projection | [served_relay_publisher.py](../../../tldw_chatbook/Web_Server/served_relay_publisher.py): approvals follow mounted Console cards. Background and viewless decisions therefore need a runtime-owned projection. |
| Existing command authority | [console_chat_controller.py](../../../tldw_chatbook/Chat/console_chat_controller.py): main-agent steering, redirect and Stop select `store.active_session_id`. Remote calls need explicit targeting without changing that selection. |
| Shared-run-store hazard | [AgentRuns_DB.py](../../../tldw_chatbook/DB/AgentRuns_DB.py): construction defaults to `reconcile_on_init=True`; `reconcile_orphaned_runs` assumes one app per data directory and can mark another process's active runs as errors. Shared reads must not trigger that sweep. |
| Intake and decisions | [console_received_dispatch.py](../../../tldw_chatbook/Chat/console_received_dispatch.py), [console_received_intent.py](../../../tldw_chatbook/Chat/console_received_intent.py) and [console_interrupt_rounds.py](../../../tldw_chatbook/Chat/console_interrupt_rounds.py): received-input admission, retained decisions, answerability and resolution already have native owners. |
| Recoverable storage operations | [backup-recovery.md](../../Architecture/backup-recovery.md): existing capture, maintenance admission, journals and startup recovery fences govern storage replacement. Legacy mobile stores need explicit inventory coverage. |

Execution does not need to be moved into a new headless engine. The missing pieces
are shared ownership and access, process retention, transport reattachment, and
consolidation of the storage split recorded in ADR-234's amendment.

## 3. Ownership and canonical data

Use the profile's existing configured conversation and workspace databases and
the normal sibling `agent_runs.db`. Do not collapse the repository's distinct
database responsibilities into one file. Remove frontend-specific database
selection and the mobile-only controller/store/bridge construction.

Retain one existing served child App as the execution host for each admitted
canonical profile/storage identity. It owns one `ConsoleRuntime` containing the
profile's open chats. The parent owns host lifetime, authenticated attachment
routing and a small live ownership registry; it does not become another agent
controller or scheduler. Repeated device connections never construct a second
execution host for the same canonical stores.
Snapshots are projections. The existing runtime, controller, store, bridge,
decision host, queues, tool policies and persistence remain authoritative.

The served registry identifies that host by server epoch and stable Console ID, and
live chats by their owning host, session ID and session incarnation. Saved chats
also carry canonical database/profile identity and conversation ID. Reserve a
saved conversation's served owner before hydration; concurrent opens attach to
that owner. Register the durable identity when an unsaved chat is persisted.
Never use a title, device ID, raw path supplied by the browser, or tab position as
execution identity. An unsaved chat remains inspectable under its live identity
until its existing persistence or close rules apply.

A new device discovers existing served hosts and chats. Opening a chat routes to
its owner. Start a host only if no compatible retained host exists; do not create
one for every phone or reconnect. Saved chats without a live served owner resume
through normal Console hydration and admission, including dispatch recovery.
Reading a saved conversation does not restart its agent.

Canonical run-store safety requires an OS-backed process execution fence for the
actual admitted `agent_runs.db` resource. Key it by the secured physical store,
including safe creation identity when the DB does not yet exist, rather than a
profile name or hash of the whole configured path tuple. Two differently named
chat DBs in one directory share that sibling run store and must contend for the
same fence. Validate aliases; refuse an alias that cannot be fenced consistently.

The executing process owns the fence: the served child for served execution,
the native App process for native execution. The web parent owns routing only;
its disappearance must not release an executing child's authority. Acquire
before any execution-capable run-store open, orphan reconciliation, automatic
work recovery or dispatch, and retain through physical owner cleanup. An empty
`ConsoleRuntime` holder or a read-only view need not acquire execution authority.
Constructor flags alone are insufficient: explicit recovery and auxiliary run
writers must honor the same owner check. Maintenance and schema migration keep
their existing separate admission requirements.

A live native owner or second served server prevents competing execution for
that resource. Non-owning reads use a read-only path without schema migration or
reconciliation. Neither a heartbeat timeout, parent death nor absence from the
served registry proves the execution owner dead. A served child that loses its
parent/control channel stops new admission and automatic wakes, begins existing
bounded runtime shutdown, and retains the fence until its process/physical work
has retired. A surviving or hung child keeps replacements blocked; no replacement
server may steal its fence or sweep its runs. OS-released ownership after actual
process death permits the existing uncertainty-aware recovery, never blind replay.
Projection congestion must not close that channel or trigger this shutdown path.

Independently launched native processes are not yet registered remote controls.
Do not advertise their persisted run rows as controllable live runs or
automatically retry their pending dispatches. The later phase adds native
discovery/attachment to the common contract. The first release's deliberate
limit is exclusive native-or-served execution for one profile, with concurrent
desktop-browser and phone access to its served runtime. Upgrade/consolidation
requires older app processes to quiesce before the new ownership contract applies.

## 4. Server lifetime and terminal reattachment

Server startup completes normal storage admission and any pending consolidation
recovery before admitting ordinary writes. Hosts start lazily on authenticated
Console/mobile use and remain owned until explicit Console closure, app exit,
child failure, or served-server shutdown. Closing a browser, signing out a
device, or revoking its session detaches that client and its authority; it does
not implicitly Stop a run. Existing explicit Stop, session-close and app-shutdown
semantics remain distinct.

The served lifecycle owner starts the private event drainer before a retained
child can publish events and keeps it running for that host's lifetime, including
before the first mobile request, with zero phone subscribers or with the mobile
surface disabled. Draining and publication never wait on a browser subscriber.
If a private projection buffer fills, drop/coalesce replaceable projection events
and retain a bounded reset-needed marker until a current snapshot establishes a
new sequence boundary. That snapshot reconciles transcript, run/fleet state and
pending decisions from their native owners. Projection loss never discards a
command response/receipt or resolves a native decision. It must not close the
host's authenticated control channel, release its execution fence, replace the
host or trigger owner-loss shutdown. Genuine parent/control-channel failure
still follows the shutdown contract above.

Move socket callbacks behind a retained transport attachment. Keep draining the
child's output while no terminal is attached, without writing to a dead socket,
blocking the app, or retaining an unbounded terminal-output history. A newly
attached terminal receives a reset/full-render handshake at its dimensions
before incremental updates. Restore the driver's application/alternate-screen,
mouse reporting, bracketed-paste, cursor and terminal-sync state in that handshake;
these modes are otherwise emitted only at driver startup. Do not restart the
driver's input thread or restart the agent to obtain this state. Replaying only
the deltas produced during absence is insufficient. Phone transcript
subscriptions do not resize the terminal.

One retained Console has at most one full-terminal input/viewport attachment at
a time. An authenticated replacement attachment invalidates and closes the old
terminal attachment; it does not replace the runtime. A stale socket cannot send
keys, resize, revoke or shut down the successor. Multiple phone subscriptions
can coexist with that terminal. Selecting a chat on the phone is local to that
phone's view and never calls `switch_session` on the desktop as a side effect.

Separate runtime-control availability from Canvas feature enablement while
reusing the existing authenticated transport. Canvas disablement revokes Canvas
capabilities only. Preserve browser-session fences, child secrets and capability
checks on Canvas routes when terminal attachment ownership changes; stable
runtime ownership is not a transferable browser Canvas capability.

Shutdown closes admission, disconnects clients, and invokes existing app/runtime
teardown for every retained host, with bounded cleanup and honest outcomes.
There is no transparent continuation guarantee after server or child process
death. Durable history and the existing dispatch recovery surface remain the
recovery authority.

## 5. Authenticated reads, commands and reconnects

Extend the existing `/api/v1/mobile/` surface and its authenticated stream. Reuse
session/CSRF authentication, device management, TLS and trusted-proxy policy,
input bounds and same-origin protections. VPN connectivity does not grant app
authority. The mobile-surface flag still controls exposure of `/m` and its API.

The process adapter exposes bounded chat listings, cursor-paginated transcript
and agent history, current run/fleet snapshots, and retained decisions. Obtain
live state from its owning runtime, including while Console is unmounted or a
chat is in the background. Do not scrape widgets or read credentials, automatic
instruction bodies, arbitrary filesystem data, or unrestricted DB rows into
metadata projections. Preserve current thinking/capture visibility policies.

| Operation | Owning path and target |
| --- | --- |
| Open/create chat | Canonical Console session creation or saved hydration; atomic served ownership reservation. |
| Send | Runtime-owned admission with a command-owned phone input snapshot and the target session's checked settings/workspace. Reuse the existing session claim, preparation and dispatch owners without borrowing desktop composition. |
| Main-agent Steer/Redirect/Stop | Existing controller behavior extended with explicit target session/incarnation and current run identity. Existing active-tab entry points delegate to the same underlying operation. |
| Subagent guidance/Stop | Existing bridge/coordinator path, scoped to owning conversation and exact live handle/run. Historical or finished workers are read-only. |
| Answer decision | Existing retained round resolver, scoped to owning runtime, session, run, kind and round ID. Includes approvals, agent questions and other pending native confirmations with a supported phone projection. |

Remote preparation and dispatch must not temporarily switch global active-session
state, borrow the desktop composer, clear its draft, or rebuild configuration
from the phone's viewed chat. The current `ConsoleSessionInputSnapshot` captures
and validates the stored desktop draft/revision, attachments and one-shot prefill;
its acceptance path clears that exact draft. It cannot be reused unchanged for
independent phone text. Extend the existing input/admission seam with an immutable
phone command snapshot carrying its exact session/incarnation, submitted text,
command identity, and checked settings/context revisions. Revalidate session,
configuration, permissions and admission under their existing owners; desktop
typing alone must neither invalidate nor clear a received phone command.

First-release phone Send has no attachments, one-shot prefill or staged RAG
launch. Those desktop composition values remain untouched, including when a
different chat is active. The target chat's committed context policy, history,
native retrieval and checked generation settings still apply. Phone Send must
not snapshot global desktop staging or use `set_session_draft` as a transport.
Only the originating page consumes its exact accepted input; late acknowledgement
must not clear newer edits on that page or any draft on another device. This is
one new input origin into the existing claim/preparation/dispatch path, not a
second controller or a relaxed bypass of native validation.

An unsupported decision remains visibly pending
with a reason and an Open terminal path; it is never silently denied or granted.
Supported decisions use the same native answerability/timeout policy. Track
answerable attachments so one device disconnecting cannot mark a decision
unanswerable while another device can still answer it. A successful decision
resolves once; all projections then remove or mark that exact round resolved.

Each mutation carries an opaque command ID, protocol version, exact target
identity and relevant expected revision/run/round. The owning runtime admits
each ID at most once and records a bounded receipt before issuing its effect.
The same ID and same payload return the existing receipt; reuse with different
content is refused. Receipts distinguish pending, accepted/queued, applied,
refused and outcome-unknown. Accepted Stop or queued guidance does not claim
physical cancellation or guidance consumption has already completed.

Command IDs are server-issued submission tickets bound to the action, target,
server/host epoch and an immutable monotonic expiry, five minutes from issue.
Issue when the user submits; this is not a typing deadline, approval timeout or
run budget. The issuer authenticates that binding even if a receipt is absent;
the client cannot renew a ticket's deadline while keeping the same command ID.
Expired or prior-epoch tickets can query outcomes but cannot issue an effect.
Ticket expiry does not cancel an already accepted turn, Stop or queued guidance.

Retain deduplication evidence for every still-valid ticket; never evict it to
admit another effect. Capacity pressure refuses new non-idempotent submissions
before dispatch. Reuse existing turn/queue/round outcomes as receipts where they
already establish identity. Exact-target native Stop must remain available under
receipt pressure: repeated Stop is the same cancellation request, not a new run
or a reason to allocate an unbounded command history. Completed/refused receipt
details may be pruned only after their ticket is invalid. Pending or unknown
outcomes remain retained/referenced through native recovery or explicit review.
After details are pruned, report expired/unavailable outcome; absence never means
the command is fresh or safe to resend.

Receipts needed across owner loss are durable, profile-local and body-free;
command text follows existing transcript/steering capture policy. Retain a
payload fingerprint for mismatch checking and the target/outcome facts. Where
acceptance and its receipt cannot share a native transaction, an interrupted
handoff remains outcome-unknown and cannot be automatically issued again.
This is duplicate-dispatch prevention, not a promise of exactly-once external
side effects. Apply normal schema-version/file-backed migration and recovery
coverage rules to any new durable receipt/mapping records.

On disconnect, disable mutations and retain unsent text in the live page. Do not
persist drafts in browser localStorage or queue offline commands. Reconnect
authenticates afresh, checks the server/host epoch, reconciles outstanding command
IDs and takes a current snapshot. Subscribe and snapshot with an event-sequence
boundary so changes cannot fall into a read/subscribe gap. Bound subscriber
queues and replay; a cursor older than retained events triggers a snapshot reset.
Private projection loss requires the same reset for affected subscriptions;
an incomplete projection cannot be reported as current.
The snapshot includes current decisions, run/fleet state and transcript cursor.
No socket reconnect automatically resends a mutation.

Unknown prior-epoch outcomes remain visible for review under existing recovery
rules. A finished/replaced run, resolved round, closed session or stale protocol
gets a refusal with current state. Never redirect a stale command to whichever
chat or worker happens to be active now. Expired authentication returns the
existing sign-in flow, preserving the distinction from a network outage.

## 6. Phone workflow and presentation

Open on Chats, listing ordinary saved and live served chats without duplicate
saved/live rows. Show workspace, latest activity, and running/waiting/finished
state with text and an attention marker for required decisions. An unavailable
host is shown as unavailable, not finished. Saved history is still readable
through its canonical owner/service when live execution is unavailable.

Opening a chat shows the live transcript and expandable tool activity. Pending
approvals and questions are answerable in the chat and discoverable from the
attention surface. An Agents view lists the main agent and subagents; a worker
detail shows its activity, live status and queued guidance. Finished worker
history remains readable without advertising live controls.

The composer always names the target chat and, when selected, the subagent.
Switching chat or worker preserves separate unsent text in that page and updates
the target before controls are enabled. Main-agent Send queues a follow-up under
the existing Console queue policy; Steer delivers guidance at the existing safe
point; Redirect interrupts the current model response under existing semantics;
Stop requests existing cooperative cancellation. Subagents expose guidance and
Stop, with mailbox acceptance distinct from consumption. Do not invent subagent
Redirect support. Show native refusals instead of silently converting an action
into another kind of send.

Use the existing plain HTML/CSS/JS `/m` assets, manifest and sign-in paths. Keep
the current no-service-worker posture. Compose the existing design language's
semantic statuses, visible focus, control hierarchy, disclosure and error rules;
Textual `$ds-*` TCSS is not directly executable browser CSS. Reuse served CSS
variables and document any necessary browser equivalents rather than adding a
second unrelated palette or a frontend framework. Phone controls must have
usable touch targets, no hover-only information, and visible Back/close actions.
The soft keyboard must leave the composer and its action reachable. Long content
must wrap or have a contained expansion/scroll affordance.

Keep content inert or pass it through the existing admitted renderer; never
insert raw tool output or provider HTML. Connection state and last confirmed
update are explicit. Unsent drafts survive a transient connection loss while
the page stays open; refresh/browser termination has no draft-durability promise.
Existing mobile-only model, hook and project-instruction exceptions disappear
for new canonical chats; permissions and review gates remain native-owned.

## 7. Consolidating legacy mobile data

Consolidation is part of this correction. Changing DB paths or retiring the old
engine without preserving its data does not satisfy the design.

1. **Preflight and fence.** Discover the exact legacy directory and canonical
   destination candidates from admitted configuration. Legacy source identity is
   its physical directory/stores, not the configuration that starts first:
   configurations can share `user_data_dir()/mobile` while independently
   overriding canonical DB paths. Existing mobile records do not prove which
   configuration owns each chat. Present the source, destination and preserved
   scope for review; require an explicit destination decision when attribution
   is ambiguous. Never silently assign mixed history to the current profile.
   Claim the physical source once and bind the operation/cutover receipt to both
   source identity/fingerprint and reviewed destination resources. Another
   configuration must see that claim and cannot re-import or redirect it; a
   changed source requires a fresh reviewed operation. Stop new legacy-mobile
   admission, let active work settle or require explicit Stop, and quiesce
   affected source/destination owners using existing maintenance admission.
   Never force-close a running writer or open legacy databases with constructors
   that mutate/reconcile them merely to inspect them. Refuse unsafe paths,
   unsupported schemas and unresolved source/destination aliases.
2. **Capture.** Extend existing recovery inventory/participants for the three
   legacy mobile stores and their referenced files before invoking normal
   coherent capture. Confirm SQLite WAL-consistent copies, exact destination
   rollback coverage, selected scope and existing credential/encryption policy.
   A directory copy or an archive missing the legacy stores is insufficient.
3. **Plan identities and dependencies.** Inventory conversations, message trees,
   generation/context settings, workspace records/bindings/exclusions, run trees,
   step history and referenced artifacts/logs. Include dependent records needed
   to read that history. Preserve IDs when unused; reuse records only when
   equivalent under the installed schema and authority rules. Otherwise allocate
   deterministic journaled mappings and rewrite all references consistently.
   Identical titles, paths or names are never sufficient for equivalence.
4. **Preserve authority.** Reconcile the built-in Default workspace through its
   native invariant. Keep conflicting named workspaces distinct. Preserve
   exclusions, explicit restrictions, project-instruction control and relevant
   historical settings; do not import new grants, activate unreviewed roots or
   turn data restoration into dispatch approval. Quarantine all imported
   automatic-work chains, wake/chat-start claims, goals and pending-result
   delivery authority as review-required, including completed runs. A terminal
   subagent with an undelivered result can otherwise trigger a provider follow-up
   when normal startup recovery seeds results from history. Preserve original
   run statuses, results, accounting and truthful delivery stamps; never forge
   `wake_delivered_at` to suppress that work. Publish the quarantine atomically
   with imported authority records and retain it across every journal resume
   before runtime admission. Ordinary startup recovery is not fresh approval.
   Release only through an explicit native review of the exact imported authority;
   unsupported release stays paused. Reading imported history or sending a new
   manual turn does not approve an old imported automatic chain.
5. **Apply and resume.** Use existing owner transactions and a durable operation
   journal. Record source fingerprints, destination identity/revisions, mappings,
   committed batches, verification and cutover. Couple each batch's data and
   idempotency receipt in its owning DB transaction. Do not claim one transaction
   spans all three databases. A crash fences new writes until recovery resumes
   or rolls back the same operation; it never starts an untracked second import.
6. **Verify and cut over.** Check mapped identity/reference closure, record and
   branch preservation, settings/restrictions, readable history and actual
   referenced artifacts. Unknown populated source data must be covered or make
   preflight fail; never silently discard it. Only then publish the completed
   cutover and enable canonical mobile writes. An empty/absent legacy directory
   is a verified no-data cutover, not a reason to create legacy DBs.
7. **Retain recovery copies.** Stop using the legacy stores for normal reads and
   writes, keep originals/captures as explicit recovery copies, and prevent
   double execution through the retired mobile engine. No automatic deletion is
   part of this release. Repeating a completed operation verifies the receipt
   and mappings rather than importing again.

Rollback before cutover restores affected data through existing recovery
ownership. After canonical writes are admitted, never overwrite newer data with
a stale pre-cutover backup; the existing reviewed recovery flow governs any
later rollback. Record source schemas and retain exact source bytes rather than
guessing how an unsupported source should be transformed.

## 8. Component boundaries and delivery scope

| Component | Responsibility and limits |
| --- | --- |
| Served lifecycle/attachment owner | Retain and route to one existing child host per canonical profile, await that child's run-store fence, start private event draining before publication independently of phone subscribers, attach terminal transport, drain detached output, fence stale attachments and close hosts on server shutdown. The parent does not hold the child's execution authority. |
| Existing Console runtime and domain owners | Hold execution authority in the executing process; admit command-owned phone inputs through the existing session claim, resolve exact commands/decisions, execute agents/tools and persist outcomes. Add explicit-target seams here; do not add a parallel mobile state machine. |
| Existing control transport and mobile API | Authenticate, validate bounded requests, route to the owning process, expose snapshots/events/receipts and apply backpressure. Projection overflow coalesces/resets projections while preserving the host control channel and native outcomes. Runtime control is independent of Canvas enablement. |
| Existing recovery subsystem plus consolidation adapter | Inventory/capture retired mobile data, map and apply records transactionally, recover interrupted work and publish verified cutover. It does not replay agent work. |
| `/m` page | Render shared chat/run state, keep page-local navigation/drafts and submit explicit user commands. It owns presentation, not runtime lifecycle or permissions. |

One reviewed specification covers these related contracts. Implementation must
be split into atomic, independently testable tasks in dependency order: ownership
and retention; canonical storage/consolidation; authenticated projections and
explicit commands; phone presentation and end-to-end qualification. No task
should be declared complete on a UI-only demonstration that still uses the mobile
engine. Task creation and the detailed implementation plan follow written-spec
review. Native Console attachment gets its own concrete design once the served
contract is implemented and qualified; no speculative native daemon is added now.

## 9. Failure behavior

| Event | Required result |
| --- | --- |
| Browser close, VPN drop, device sign-out/revoke | Detach that client's input/subscriptions/capabilities; accepted work continues under its owner. |
| Slow/disconnected client | Bounded projection buffering; no blocked agent thread, output pipe or unbounded history. |
| Private projection queue fills | Bounded coalescing/drop plus snapshot reset; the same host, control channel and execution fence remain live, including before any phone connects. Native decisions and command outcomes remain authoritative. |
| Replacement terminal attachment | Full repaint into the new terminal; old keys/resize/cleanup cannot affect the new attachment. |
| Canvas disabled | Canvas unavailable; mobile runtime controls remain available under ordinary authentication. |
| Lost command acknowledgement | Query its receipt/current owner; no automatic replay or success inferred from a timeout. |
| Competing decision answers | One resolver wins; later answers return already resolved/current state. |
| Chat/worker finishes or changes before command | Refuse the stale target; do not retarget, relaunch or change desktop selection. |
| Child or served-server death | Live controls unavailable; pending receipts may be unknown; normal durable recovery governs restart. |
| Parent dies while child survives | Child closes admission and starts shutdown while retaining its run-store fence; a replacement cannot execute or reconcile that store until ownership actually retires. |
| Expired command or pruned receipt | Outcome query only; do not renew the same ID, infer non-execution or dispatch it again. |
| Shared legacy source with ambiguous destination | Hold the source claim and require reviewed destination mapping; no startup-profile default import. |
| Interrupted consolidation | Storage admission remains fenced; resume/rollback the journaled operation without duplicate imports. |
| Source conflict, unknown schema or missing dependency | Refuse cutover with the original data retained and an actionable issue. |

## 10. Verification and acceptance

Use real isolated SQLite stores and existing runtime test seams. Product code
changes require targeted unit/integration tests and static checks for the touched
areas. Run the full test suite only if the user opts in.

- Prove the same canonical chat/history/settings/workspace identities are seen
  from terminal and phone, including new phone-created chats and resumed history.
  Assert the retired mobile engine and its DB constructors are not used.
- Race native/served execution and two served servers against the same canonical
  stores. Prove one execution owner, no false orphan sweep by readers or rejected
  contenders, and no takeover on a timeout. Include two chat DB filenames sharing
  one sibling `agent_runs.db`, aliases, non-owning explicit recovery calls, and a
  web-parent crash while its child remains alive. Prove the child's fence blocks
  replacement/reconciliation until physical retirement. Exercise actual owner
  exit/crash recovery separately from browser disconnection.
- Start a real served child, run a controlled agent/tool sequence, close the
  desktop browser socket, reconnect via the phone and then terminal, and verify
  the same process, session and run continue. Verify a full terminal redraw,
  mouse clicks, cursor/mode state, multiline bracketed paste and resize after
  reattachment, without restarting the driver input thread. Test stale keys,
  resize and cleanup rejection. Separately test clean server teardown.
- Before any mobile request, start a real served host with zero phone subscribers
  and emit more events than the private broker queue can hold. Repeat with the
  mobile surface disabled and with a slow subscriber. Verify lifecycle-owned
  draining and force private projection overflow: buffers and reset state stay
  bounded, the same child/process/run and execution fence remain live, and the
  authenticated control channel stays connected. A current snapshot must restore
  transcript, fleet and pending-decision state without resurrecting resolved
  rounds or losing command responses/receipts. Separately sever the actual
  parent/control channel and verify the existing bounded shutdown still occurs.
- Exercise two chats and multiple subagents concurrently. Read/steer/redirect/
  Stop exact supported targets from the phone; verify unrelated runs, desktop
  selection, settings and drafts are unchanged. Show queued versus consumed
  subagent guidance honestly and test the target finishing during submission.
- Leave different unsent drafts on desktop and phone for the same chat, with
  desktop attachments, prefill and staged RAG; submit and queue phone text while
  desktop typing continues. Assert exact phone text once, unchanged desktop
  composition, no borrowed attachment/context and no rejection caused only by
  desktop draft edits. Revalidate genuine settings, binding and admission races;
  a late phone receipt cannot clear newer phone edits.
- Exercise background and viewless decisions, multiple answerable clients,
  competing resolutions, disconnect during a decision and Canvas disabled.
  Verify native permission, timeout, hook and project-instruction behavior.
- Inject disconnects before admission, after admission and after effect but
  before acknowledgement. Reuse command IDs, change payload under the same ID,
  change owner epoch, overflow replay and race snapshot/subscription. Assert no
  duplicate accepted mutation and no false success for an unknown outcome. Cover
  ticket expiry, authenticated expiry tampering, valid-ticket retention under
  pressure, querying after pruning, unknown-outcome retention and Stop while the
  non-idempotent receipt budget is full. Accepted work survives ticket expiry.
- Interrupt consolidation at each journal stage and between DB batches. Retry
  against populated canonical stores; cover conflicting IDs/workspaces, message
  branches, settings, exclusions, run parentage, artifacts, unsupported data,
  empty sources and repeat-after-cutover. Verify backup scope, source retention,
  rollback, schema-catalog changes and recovered execution remains gated. Include
  two configurations sharing one legacy source but selecting different canonical
  DBs; require reviewed attribution and refuse a second destination claim. Import
  a completed parent/child pair with an undelivered terminal child result and an
  otherwise settled active automatic chain. Native startup/recovery and journal
  resume must produce no automatic provider/tool call before explicit review,
  while preserving truthful delivery history. A new manual turn does not release
  the quarantined old chain.
- Check phone-sized layouts, touch/keyboard access, soft-keyboard viewport,
  long/inert content, independent drafts/targets, offline display and sign-in
  recovery in the real served browser path. Emulated disconnection verifies the
  app contract; actual phone/VPN qualification is recorded separately.

Capture source/commit identity and actual commands/results for the eventual
verification. A schema label, mocked list response, static source assertion or
phone screenshot alone is not evidence the shared runtime or migration works.

## 11. Governing decisions and review state

[ADR-244](../../../backlog/decisions/244-shared-console-ownership-and-mobile-supervision.md)
supersedes ADR-234's frontend-specific execution/storage contract and amends
ADR-236's mounted-card, browser-lifetime and Canvas-coupled relay contract.
Its metadata makes the transition explicit; this document does not claim the
replacement is implemented. Existing ADR-233 authentication/device policy,
ADR-094 accepted-turn ownership, ADR-126 recovery and ADR-150/161 design-language
boundaries remain governing.

Architecture, phone workflow and consolidation/failure/verification sections were
approved in conversation. The user reviewed the written spec and approved the six
initial source-backed corrections and the subsequent projection-overflow correction.
Each correction has matching verification scenarios above. Self-review and
independent review found no remaining issues in the projection-overflow correction;
genuine owner loss retains its shutdown boundary.
The linked implementation plan maps this specification to 11 atomic Backlog tasks
with native interfaces, targeted checks and delivery gates. Product implementation
has not started. No production code, real user-data migration, provider call or
runtime qualification is part of this design/planning work.
