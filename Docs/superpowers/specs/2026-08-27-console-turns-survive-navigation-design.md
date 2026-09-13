# Console turns survive screen navigation

**Date:** 2026-08-27

**Status:** Approved for implementation planning

**Task:** [TASK-22514](../../../backlog/tasks/task-22514%20-%20Console-turns-survive-screen-navigation.md)

**ADR:** [ADR-094](../../../backlog/decisions/094-console-turn-lifetime-and-navigation-boundary.md)

**Baseline:** `e4923b2135ee858d58214c69da3a0b53565e4638`

## Summary

Every accepted Console turn continues when the user navigates to another screen.
Navigation replaces only the Console view: provider streaming, the main assistant or
agent loop, turn-launched tools and media work, delegated agents, and pending human
decisions remain owned by the app-level Console runtime. Returning to Console attaches
a fresh screen to the same runtime and reconciles the continuing or completed turn
from the live app-owned store.

Navigation is not a cancellation boundary. Stop, explicit session close, and confirmed
application quit remain cancellation boundaries. Application quit is still the final
lifetime boundary; this design does not make turns survive process exit or restart.

When a hidden turn needs approval, Chatbook shows one app-wide notice, marks Console as
needing attention, and pauses the decision clock until the relevant card is answerable.
When a hidden turn completes or terminally fails, Chatbook shows one bounded notice and
keeps a durable local-only unseen marker until the owning session has rendered the
terminal result.

## ADR check

**ADR required:** yes

**ADR path:**
`backlog/decisions/094-console-turn-lifetime-and-navigation-boundary.md`

**Reason:** this feature changes the long-lived runtime, task-ownership, cancellation,
approval-clock, and local attention-persistence boundaries shared by the app, Console
controller, Chat screen, and navigation shell. It supersedes the screen-scoped
user-turn policy recorded by TASK-1143 and TASK-15860 acceptance criterion 2.

No schema migration is expected. `conversation_local_marks` already stores arbitrary
validated mark strings; its validator gains a local-only
`console_unseen:<terminal-receipt-id>` namespace. The matching terminal transcript
commit and receipt mark must share the existing ChaChaNotes transaction rather than
rely on cross-database rollback.

## User promise

- Pressing Send transfers the turn to app-owned runtime custody before the composer is
  cleared.
- Switching screens never stops an accepted Console turn or denies its pending human
  decisions.
- Returning to Console shows the same continuing stream or terminal result without
  duplicate or missing transcript content.
- A hidden approval never auto-approves, never fails closed merely because no Console
  screen is mounted, and does not lose decision time while it cannot be answered.
- A hidden completion or terminal failure produces one app-wide notice and a durable
  Console attention marker that remains until the owning result is presented.
- Stop affects the selected turn or chain. Session close affects that session. Confirmed
  app quit affects the whole runtime. None of those scopes silently widens.
- Forced process termination may interrupt work. Existing durable dispatch recovery
  remains the only owner of restart repair and recovery semantics.

## Goals

- Move accepted-turn task ownership from `ChatScreen` to the existing app-owned
  `ConsoleRuntime`.
- Preserve the existing controller and store as the only run-state, queue, cap,
  streaming, and transcript authorities.
- Freeze send-time inputs so detaching the view cannot change a turn's dictionaries,
  world information, staged RAG, Library policy, tools, identity, or settings midway.
- Make every human-decision round view-independent while preserving explicit approval
  and fail-closed cancellation behavior.
- Make background outcomes discoverable without forcing navigation or flooding the
  current screen.
- Keep detached execution free of DOM queries, transcript polling, and per-token shell
  updates.
- Retain deterministic, bounded teardown when the user explicitly stops work, closes a
  session, or quits.

## Non-goals

- Continuing turns after a confirmed app quit, crash, host shutdown, or restart.
- Keeping a `ChatScreen` cached, mounted, or hidden behind other destinations.
- Running microphone capture, realtime voice sessions, audio playback, open modals,
  animations, or screen timers after Console unmounts.
- Treating unsent composer text, staged attachments, or a modal action as an accepted
  turn before runtime custody succeeds.
- Changing provider retry policy, per-session admission rules, global concurrency caps,
  prompt-queue ordering, or the existing dispatch-recovery contract.
- Adding a global Stop button or allowing approval decisions from non-Console screens.
- Showing an exact global attention count. The shell promises a boolean marker; Console
  session surfaces retain detailed status.
- Persisting raw turn requests, context snapshots, attachment bytes, tool output, or
  approval payloads in navigation or attention state.

## Existing constraints and starting point

The app already owns one `ConsoleRuntime`; `ChatScreen` attaches and detaches its view
hooks. The runtime owns the `ConsoleChatStore`, provider gateway, agent bridge, and
`ConsoleChatController`, and the store already survives navigation. Headless supervisor
wake turns can run with no Console mounted.

The remaining screen-scoped behavior is explicit and must be changed deliberately:

- `_submit_console_native_draft` is awaited by a Textual worker owned by `ChatScreen`,
  so unmount cancellation can propagate into an ordinary user turn.
- `ConsoleRuntime.leave_console()` invokes the controller's per-visit teardown, which
  cancels ordinary user-originated tasks and binds parked decisions to a visit event.
- multiple screen-supplied hooks are consulted on the send path; clearing them to
  viewless defaults is correct for machine-authored wake turns but would silently
  remove user-selected context from a surviving ordinary turn;
- tool approvals can be retained headlessly, while skill-install and skill-script
  confirmations currently fail closed without their screen hooks;
- terminal outcome notification hooks are screen-owned and are cleared on detach; and
- the old navigation confirmation and teardown notice tell users that active Console
  work will be cancelled.

Screens remain fresh instances. Prior defects proved that keeping a removed screen
alive lets self-rearming sync workers query a dead DOM and can terminate the whole TUI.
This design therefore extends the runtime/view boundary; it does not weaken the
fresh-screen navigation invariant.

## Terminology

- **Runtime custody:** the app-owned runtime has synchronously registered an immutable
  request and is responsible for either running it or producing a recoverable refusal.
- **Durable acceptance:** the existing first-send/promotion boundary has committed the
  user turn and its authoritative transcript identity. Before this boundary, a refusal
  may restore the draft; after it, failure is a transcript outcome.
- **Turn-owned work:** provider streaming, agent execution, queued continuation,
  turn-launched tools or generated media, delegated work, and human decisions reachable
  from one accepted turn.
- **View-owned work:** DOM rendering, transcript and cost timers, microphone and audio
  resources, open modals, animations, selection, focus, and unsent composition.
- **Attachment generation:** a monotonically increasing claim token identifying the
  currently attached Console view. It is unrelated to response variants.
- **Decision-active time:** elapsed approval time while the owning decision card is
  mounted, visible, and answerable in Console.
- **Terminal acknowledgement:** the owning session is active and the mounted transcript
  has synchronized the terminal row. Exact scroll-viewport intersection is not
  required.

## Ownership model

| Concern | Authority | Lifetime |
| --- | --- | --- |
| Store, controller, provider gateway, agent bridge | `ConsoleRuntime` | App |
| Accepted-turn task handles and custody records | `ConsoleRuntime` | Handoff through terminal cleanup |
| Run state, queues, caps, cancellation events | Existing `ConsoleChatController` | Existing controller policy |
| Messages, active lineage, session drafts | Existing `ConsoleChatStore` | App/runtime |
| Terminal unseen mark | ChaChaNotes `conversation_local_marks` | Durable until acknowledged |
| Pending approval attention | Runtime/app memory plus existing controller round registry | Until decision, cancellation, or app exit |
| DOM hooks, timers, selection, focus, realtime media | `ChatScreen` | One mounted view |
| Restart recovery | Existing dispatch-recovery owner | Existing policy |

The runtime registry is intentionally not a second run-state machine. It keeps the
minimum strong references needed to own scheduling, cancellation routing, finalization,
and cleanup. The controller and store remain authoritative for all domain state.

## Send handoff and frozen context

### Custody boundary

The screen captures lightweight send-time state and transfers it to a main-loop runtime
handoff. The handoff registers the custody record before scheduling execution and
returns a stable turn ID. Only that successful return permits the screen to clear the
composer.

Registration-before-scheduling closes the navigation race between creating a task and
recording its owner. The screen never awaits the long-running runtime task. It may
subscribe for repaint hints while attached, but the task's completion and cleanup do
not depend on that subscription.

The request transfers accepted attachment objects rather than deep-copying their byte
payloads. Existing provider construction may consume the prompt and accepted
attachments under its current policy; runtime custody must not broaden that provider
payload or create a new raw copy in `repr`, logs, attention state, navigation labels,
sync, export, or unrelated remote APIs.

### Frozen inputs

The request captures immutable values or app-owned service references for every
screen-derived input whose value belongs to the send:

- exact session and conversation identity;
- provider/model selection and session generation/settings;
- agent/ordinary-chat mode and enabled tool policy;
- character/persona identity and presentation name;
- dictionary and world-information selection;
- staged RAG/evidence and Library access policy;
- project/workspace context and authority snapshot;
- prompt-history contribution; and
- accepted attachments and their session ownership.

Heavy or asynchronous preparation—provider readiness, RAG retrieval, Library reads,
project-instruction activation, and request construction—runs after custody inside the
runtime task. It consumes the frozen request and app-owned services, never a widget or
bound screen method. This keeps Send responsive without letting detachment change the
turn's meaning.

### Refusal and recovery

Custody refusal leaves the current composer untouched. A refusal after custody but
before durable acceptance records a recovery entry keyed by turn ID, including the
session-owned draft and accepted attachment references. It must not overwrite an older
recovery entry for the same session. A fresh Console view can restore or discard each
entry explicitly.

After durable acceptance, the draft never reappears as unsent text. Provider or tool
failure becomes a stopped or failed transcript outcome under existing policy.

## Turn execution

The runtime schedules the controller operation on the app event loop and retains its
task before it can run. Existing controller admission, prompt queues, global and
per-session caps, provider resolution, streaming, tools, agent loop, and persistence
remain unchanged except where they currently consult live screen hooks.

Streaming writes the app-owned store and existing persistence path. While a view is
attached, a content-free repaint hint may coalesce a transcript sync. While detached,
no DOM query, Textual worker, transcript timer, cost timer, or navigation update runs
for deltas. Reattachment performs one full reconciliation, then restarts ordinary
coalesced rendering from the live store.

The runtime registers one authoritative terminal observer. Screen hooks repaint only;
they do not finalize tasks or emit duplicate outcome notices. A done callback always
retrieves task exceptions, finalizes once, and releases the request/context/attachment
references after any recovery or attention projection is complete.

## Navigation and view lifecycle

Navigation away performs view detachment, not controller visit cancellation.

Detachment must be guaranteed early in unmount or through a `finally` path. Audio,
realtime, persistence-flush, timer, video, or other view-cleanup failures must never
leave callbacks bound to a dead screen. Detachment:

- clears the enumerated screen hook set only when the detaching screen still owns the
  current attachment generation;
- stops screen timers and releases ephemeral view resources;
- preserves runtime tasks, controller cancellation events, queues, rounds, store state,
  and turn-owned operations; and
- performs no navigation confirmation about losing accepted work.

The existing overlapping-screen rule remains: a newly attached Console may claim the
runtime before an outgoing Console finishes unmounting. The outgoing generation's
later detach and callbacks are no-ops and cannot clear or mutate the successor's hooks.

Returning to Console attaches fresh hooks, reads the runtime registry and store, mounts
the active session's pending decision projection when one exists, synchronizes the
transcript, and resumes view timers. It does not automatically switch sessions or clear
unrelated attention.

## Human decisions while detached

Tool approval, skill-install confirmation, and skill-script confirmation share one
view-independent decision contract:

1. The controller registers the round and retained payload by stable round/request ID
   before attempting any UI projection.
2. The runtime records pending attention and emits at most one sanitized app-wide
   notice for that ID.
3. If no answerable card is mounted, the turn waits without approving or denying.
4. The decision clock counts only decision-active time. Detachment pauses it; mounting
   the owning card resumes the remaining allowance. A round born headlessly starts its
   clock on first answerable mount.
5. Returning to Console derives one ordered pending-decision projection for the active
   session across all three decision types. Multiple rounds remain retained and are
   presented in stable existing order; resolving one does not clear siblings.
6. An explicit decision, per-turn Stop, session close, or app disposal resolves the
   round through its existing fail-closed path.

Navigation never auto-approves, consumes a decision timeout, or binds the round to a
replaceable per-visit cancellation event. Per-turn cancellation and permanent runtime
shutdown remain authoritative.

## Attention and notification behavior

### Terminal results

Each user-visible **completed** or **terminally failed** transcript commit mints a
stable opaque terminal receipt ID and writes a local-only
`console_unseen:<terminal-receipt-id>` mark for the owning durable conversation in the
same ChaChaNotes transaction. The receipt ID is stable across an idempotent retry of
that terminal commit. The write happens regardless of whether Console appears visible.
A terminal row carries its receipt ID into the live store and display projection; a
currently attached view clears only that exact namespaced mark after acknowledging the
matching row.

Always-write-then-acknowledge makes completion/unmount races favor a retained marker
and avoids guessing visibility before the transcript paints. Separate receipts allow
several terminal outcomes in one conversation: an acknowledgement for an older row
cannot clear a newer result. The user-visible transcript plus receipt marks are the
ChaChaNotes authority. `AgentRunsDB` and shell presentation are derived projections and
are not included in a fictitious cross-file transaction.

First-send promotion must provide the durable conversation identity before terminal
commit. No accepted durable result may become unmarkable because the original session
started temporary.

The namespaced marks use the existing `conversation_local_marks` table and are absent
from conversation sync, export, prompt payloads, and unrelated remote APIs. They are
separate from `FLEET_UNSEEN`; supervisor wake delivery may continue to clear its own
fleet mark without erasing a hidden ordinary-turn result.

Explicit user Stop and confirmed session-close cancellation create no new terminal
attention: those actions are themselves acknowledgement and, for a closed session,
there is no result surface to revisit. App-shutdown cancellations create no mark or
toast because the app is closing. A cancellation not attributable to one of those
explicit reasons is classified by the controller as a terminal failure and follows the
failed-receipt path.

### Shell presentation

When a terminal receipt mark is created for a session that has not acknowledged it,
the app shows one sanitized toast: completion is informational and terminal failure is
an error. Bursts may coalesce into a bounded summary toast, but every underlying receipt
mark remains.
Notification failure never rolls back terminal persistence.

The main and overflow navigation surfaces show a boolean, non-color-only Console
attention glyph and an explanatory tooltip. They do not promise an exact count.
Freshly mounted navigation reads durable marks, so a temporarily unavailable bar or a
restart cannot erase attention. Session tabs and conversation rows retain the detailed
run/approval markers already owned by Console.

Opening Console does not clear the global marker. Resolving a pending decision clears
that decision's in-memory attention; terminal attention clears one receipt only after
the matching terminal acknowledgement. The global glyph disappears only when neither
durable terminal receipt marks nor pending decisions remain.

## Cancellation and terminal races

| Trigger | Scope and behavior |
| --- | --- |
| Screen navigation | Detach view only; accepted turns and rounds continue. |
| Stop | Signal and cancel the selected turn/chain; revoke only its rounds; create no unseen receipt. |
| Session close | Fence new/queued submissions for the session, confirm when active, then cancel and await that session's turns, rounds, and delegated children; create no unseen receipt. |
| Confirmed app quit | Revision-pin the active set, fence runtime admission, cancel all turns and rounds without new attention, drain bounded cleanup, then close gateway and databases. |
| View/render failure | Report bounded UI failure; runtime and persistence continue. |
| Recoverable tool failure | Persist the ordinary tool result; the controller may continue the turn. |
| Terminal provider/tool failure | Commit terminal failure plus unseen state and notify when unacknowledged. |
| Forced process death | Existing persistence/recovery policy applies; no continuation promise. |

Stop and normal completion share one atomic terminalization gate. If terminal content
was durably committed first, a later Stop is a no-op. If cancellation was recorded
first, late provider/tool output and callbacks are rejected by the turn generation and
cannot overwrite the stopped result.

Session close marks the session closing before it snapshots work, so no new or queued
submission can enter the set behind the cancellation pass. Quit confirmation is pinned
to the active-turn revision; if the set changes before admission is fenced, the
confirmation refreshes rather than silently widening consent.

Cancellation waits are bounded. After a cooperative grace window, Chatbook cancels the
async wrapper and advances a generation fence. An underlying non-cancellable thread may
finish, but its late writes, approval resolutions, callbacks, and terminalization are
ignored. Shutdown must not hang indefinitely on provider or worker behavior.

## Failure handling

- A runtime task exception is consumed, sanitized, and converted into the existing
  terminal failure contract; it never becomes a Textual `WorkerFailed` that exits the
  app.
- Failure before durable acceptance preserves turn-keyed recovery. Failure afterward
  preserves the accepted transcript and terminal state.
- A failing view hook or notification surface is best-effort and never authoritative.
  The store, terminal transaction, and decision registries remain intact.
- A failing unread clear leaves a stale marker and retries through the next ordinary
  attention reconciliation; it never clears optimistically.
- A failed terminal transaction publishes neither terminal acknowledgement nor cleared
  attention. Retrying the idempotent transaction produces one terminal result and one
  mark.
- Existing dispatch checkpoints and recovery own restart repair. This feature adds no
  second orphan scanner or restart executor.

## Privacy and bounded retention

The custody record may temporarily contain sensitive turn inputs because the provider
request needs them. That record is process-local, excluded from `repr` and logs, and
released after terminal cleanup. Navigation and attention state store only stable
opaque IDs, safe titles, phases, and booleans.

Notifications use existing display-label sanitation and never include prompt bodies,
tool arguments/results, exception text, credentials, attachment names containing local
paths, or project-instruction content. Failure logs use exception types and stable
codes under existing redaction policy.

Each namespaced unseen receipt stores only conversation ID, a mark type containing one
opaque terminal receipt ID, and timestamps. It remains local-only and does not enter
sync or export. Pending approval payloads retain existing in-memory security semantics
and die on explicit turn/session cancellation or app exit.

The runtime drops terminal task records after attention projection. Recovery entries
remain only until restored or discarded. Tests must prove completed request/context
objects become collectible and repeated turns do not grow the registry.

## Accessibility and narrow layouts

The navigation marker uses a glyph and tooltip, not color alone. It must remain legible
in the main bar and overflow menu at 80x24, 100x30, and wide layouts without clipping
route labels or changing their activation targets. Approval and terminal notices name
the sanitized owning session/workspace when available and direct the user to Console;
they never move focus or navigate automatically.

## Verification strategy

### Layered automated coverage

Pure runtime/controller tests cover every origin and decision type. Mounted Textual
tests cover representative end-to-end navigation paths. Tests use injected clocks,
events, and cancellation deadlines rather than sleeps.

Required deterministic cases include:

- custody registration is visible before task scheduling and an immediate real
  `NavigateToScreen` cannot cancel or lose the request;
- ordinary chat, main-agent, turn-launched tool/media, queued continuation, and
  delegated-agent paths remain runtime-owned after unmount;
- send-time context stays byte/identity stable after view hooks are detached or a new
  screen changes its settings;
- navigating during preparation, provider readiness, streaming, tool execution,
  approval, and terminal commit preserves the correct outcome;
- returning mid-stream reconciles from the app-owned store, then receives later deltas
  without duplication;
- tool, skill-install, and skill-script decision clocks pause headlessly and resume only
  when their owning card is answerable;
- multiple rounds in one session and turns across several sessions remain isolated;
- rapid away/back cycles and overlapping view claims cannot let the outgoing screen
  clear its successor's hooks;
- Stop, session close, and quit honor their exact scopes, including concurrent
  completion and an uncooperative provider/thread;
- injected unmount cleanup, notification, navigation-bar, and render failures never
  affect runtime completion;
- each completed/failed terminal row and its exact namespaced receipt mark are both
  present or both absent across injected ChaChaNotes transaction failures;
- acknowledgement of one terminal receipt cannot clear a later receipt for the same
  conversation, while explicit Stop/session-close/app-shutdown cancellation creates no
  receipt or outcome toast;
- a fresh marks service and app/runtime over the same temporary SQLite database restore
  the global marker before Console has ever mounted;
- active-session terminal synchronization clears only the acknowledged conversation;
- simultaneous outcomes may coalesce their toast while retaining every mark;
- detached execution performs zero DOM queries, transcript polling, and per-token
  navigation updates; and
- terminal registry cleanup releases request/context/attachment objects and remains
  bounded over repeated turns.

The returning-view tests assert four distinct facts: the live app-owned store contains
the rows, the mounted transcript renders identifying content, the next provider payload
uses the same active lineage, and durable rows exist. A database append alone is not
accepted as UI evidence.

### Privacy and architecture pins

Tests inspect every new durable and presentation owner—ChaChaNotes marks, runtime
records, logs, notifications, navigation state, sync payloads, exports, and provider
request construction—to prove the new ownership layer creates no extra raw-content
copy. Weak-reference collection covers post-terminal memory release.

Architecture tests narrowly assert that `ChatScreen` delegates accepted-turn creation
to the runtime, owns no resulting task, and remains absent from retained callbacks after
detach. They also preserve `ConsoleRuntime` as the single shared controller/store
construction and disposal owner. They do not pin incidental helper names.

For the critical lifecycle invariants, red/green evidence includes locally inverting
the named behavior: restoring screen-owned cancellation, advancing a hidden decision
clock, clearing all unseen state on Console mount, or accepting a late post-cancel
terminal write must make the corresponding focused test fail. This is targeted
test-validation evidence, not a new repository-wide mutation-testing framework.

### Real UI and live provider verification

The mounted badge tests exercise production hierarchy and styles at 80x24, 100x30, and
wide sizes, including the overflow menu, tooltip, focus, and route activation. They
wait independently for authoritative state, mounted DOM, and compositor paint.

One isolated live-provider journey waits for the first real streamed delta, navigates
through the production route, proves additional deltas or completion occur while
Console is absent, returns, and verifies identifying transcript content and the marker
clear. A local provider is preferred. Any billed provider call requires explicit user
approval.

Before targeted or live verification, run the repository import-provenance probe. Live
state uses a disposable `HOME`, XDG config/data/cache roots, explicit
`TLDW_CONFIG_PATH`, and scratch `[paths].data_dir`; unrelated catalog networking is
disabled. The real profile is fingerprinted before and after. No bare interpreter probe
may import the app without equivalent isolation.

Per repository policy, implementation verification uses targeted suites unless the
user explicitly opts into a full repository sweep.

## Documentation changes

- Replace User Guide statements that ordinary and agent turns are Console-screen-scoped.
- Explain that navigation is safe, approvals wait and notify, terminal outcomes mark
  Console until viewed, and confirmed app quit still cancels work.
- Remove the obsolete navigate-away cancellation confirmation and returning teardown
  notice from UX documentation and tests.
- Link TASK-22514, ADR-094, this design, and the eventual implementation plan.

## Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| A surviving turn reads a new screen's settings | Freeze send-time values and use app-owned services only. |
| A detached task calls a dead view | Enumerated hook detach plus attachment-generation fencing. |
| Navigation cancels the awaiting screen worker | Screen never owns or awaits the accepted runtime task. |
| Invisible approval expires or fails closed | Retained unified rounds and decision-active clocks. |
| Terminal completion races unmount or another completion | One atomic namespaced receipt mark per terminal row followed by exact-receipt acknowledgement. |
| Stop races completion | One atomic terminalization gate and late-write generation fence. |
| Quit hangs on a worker thread | Bounded cooperative drain followed by wrapper cancellation and fencing. |
| Multiple stores imply false rollback | ChaChaNotes transcript+mark is the authority; other databases are projections. |
| Runtime registry becomes a second controller | Registry stores lifetime handles only; controller/store remain authoritative. |
| Sensitive context leaks through new state | Process-local redacted records, ID-only attention, exhaustive owner tests. |
| Badge breaks compact navigation | Boolean glyph, production hierarchy, narrow-size and overflow tests. |

## Acceptance mapping

TASK-22514 acceptance criteria map directly to the sections above:

1. Runtime custody, frozen context, execution, and view lifecycle.
2. Human decisions while detached.
3. Cancellation and terminal races.
4. Turn execution plus navigation reconciliation.
5. Verification strategy.
6. Documentation changes.
7. Attention and notification behavior.
8. Send handoff plus privacy and bounded retention.

No implementation begins until this written spec passes independent review and the
owner approves the reviewed document.
