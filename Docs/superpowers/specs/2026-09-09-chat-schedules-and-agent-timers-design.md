# Local chat schedules and agent timers

Date: 2026-09-09
Status: Proposed written design; the user approved the core flow and added agent tool access. Implementation has not started.
Task: [TASK-32195](../../../backlog/tasks/task-32195%20-%20Design-local-chat-schedules-and-agent-self-timers.md)
ADR required: yes
ADR path: [ADR-143](../../../backlog/decisions/143-local-chat-schedules-and-agent-timers.md)
Reason: Durable scheduling authority, automatic execution accounting, agent tools, and the Scheduling/Console interface need one explicit contract.

## 1. Approved behavior and scope

Attach a schedule to an existing Console conversation. The user enters
`/schedule <instructions>`, reviews cadence, timezone and limits, and saves it.
When due, Chatbook runs a native chat turn using the conversation's current
context and posts a labeled scheduled response to that conversation. Enabled
tools remain usable through existing permission and resource controls. The
composer draft belongs to the user throughout.

Schedules execute while the Chatbook process is running, including while another
screen is visible. Definitions survive app restart. After an absence, eligible
missed occurrences collapse into one catch-up turn. A pending approval appears
as **Needs attention**. Chat and Schedules both expose pause and cancellation.

The user's addition makes this capability available as agent tools, including
one-time timers: an agent can arrange a future turn, receive a durable schedule
ID, and finish its current turn without keeping a worker asleep.

**Scope assumption for written review:** agent-created timers target the main
native agent in the same persisted conversation. Resuming a spawned subagent's
private state is a separate persistence feature. This draft uses the main-agent
option from the unanswered scope question. A timer restores saved instructions
and conversation context, not a
Python stack or a provider's old continuation token. Native function calling and
the existing fenced tool protocol must both work.

This milestone covers native Console execution. It does not establish an OS
background service, execution after quitting, server offload, remote MCP exposure
of Chatbook's scheduling tools, or timer support inside other agent engines.

## 2. Approach and existing owners

Two storage approaches fit the existing architecture:

1. **Recommended: native schedules in AgentRunsDB, projected into Scheduling.**
   Schedule definitions, authority, occurrences and budget reservations share a
   durable transaction boundary. Scheduling still owns cadence calculation,
   queue polling and the common Schedules surface. Watchlists and briefings
   already demonstrate the projection pattern for an independently owned store.
2. **Automation definitions in ScheduledTasksDB with native run state elsewhere.**
   This centralizes definitions but requires a cross-database protocol for
   schedule edits, revocation, occurrence claiming and budget acceptance. The
   current reminder store uses NORMAL synchronization and post-dispatch marking;
   those semantics alone are insufficient for accepting billable automatic work.

Choose the first approach. Do not copy schedule definitions into two databases.
Do not turn reminders into executable prompts or wrap each occurrence in a
complete goal-solving loop. A scheduled turn is its own automatic origin.

Relevant existing implementation:

- `Scheduling/scheduler/loop.py` and `queue.py`: app-owned polling and due tasks.
- `Scheduling/services/{scheduling_service,watchlist_projection,briefing_projection}.py`:
  common facade and projections into Schedules.
- `Chat/console_runtime.py`: app-owned controller, including viewless execution
  and approval restoration when Console reopens.
- `Chat/console_goal_runs.py`, `Agents/automatic_work_runtime.py`,
  `DB/automatic_work.py`: native automatic admission, ownership and accounting.
- `Agents/tool_catalog.py`, `Agents/run_context.py`, and the permission bridge:
  canonical tool registration and trusted calling-run identity.

The checked-out scheduling implementation predates some server-owner filtering
described by TASK-18940. That task also records remaining server integration work.
Treat server support and prerequisite integration as separate dependencies;
never dispatch a server-scoped definition through this local path.

## 3. User interaction

`/schedule` opens the same form without prefilled instructions. The slash popup
discovers it. Literal/paste handling follows the established command grammar.
The command does not send a model message or start execution before Save.
Freeze the exact reviewed fields before asynchronous validation, disable duplicate
submission, and retain one creation identity for retries of that Save. Validation
failure restores the form without replacing its instructions or the user's draft.

The form contains:

- Instructions and an optional short title.
- One-time delay/date, fixed interval, or a calendar preset with timezone.
- The target chat and captured model/tool/workspace selection.
- A readable next-run preview and **Runs while Chatbook is open**.
- Run limits and an optional end date or total run count. A human-created
  recurrence may explicitly continue until paused, under the daily limits below.
- **Allow agent follow-up timers**, off by default for a human-created schedule.
  Its explanation states that follow-ups share the displayed run limits and can
  execute between ordinary recurrence slots, at most once per minute per grant.

Ordinary chat history is read afresh at dispatch; provider, allowed tools and
workspace/resource selections are captured on create/edit. Changing the model
selector for an unrelated manual turn does not retarget a saved schedule.
Changed or unavailable bindings stop execution with an actionable explanation.
Settings and permissions may narrow the capture at any time.

An unsaved chat is persisted through its existing owner before schedule creation;
failure leaves the form and draft intact and creates no executable definition.
The target is the persisted conversation, not the currently selected screen tab.
Forking/copying a chat does not copy its schedules. Normal edits to the target
chat's active conversation path affect the next not-yet-accepted turn.

The chat shows a compact schedules control with status and next due time. The
existing Schedules queue shows the same records with **Open chat**, **Edit**,
**Pause/Resume**, and **Cancel**. A row's creator distinguishes user setup from
an agent tool call. Detail includes instructions, bounds and recent occurrences.
Use the canonical F9 Settings surface for feature controls. Preserve the existing
keybinding rules and implement every advertised action.

Scheduled output carries a machine-origin notice and ordinary assistant/tool
messages with schedule and occurrence provenance. It never impersonates a new
manual user message. Streaming and non-streaming delivery must both persist to
the correct conversation when its view is absent.

## 4. Cadence and occurrence semantics

The shared validated timing value has exactly one of these forms:

| Kind | Input | Stored meaning |
| --- | --- | --- |
| `delay` | Positive integer `seconds` | One UTC `run_at`, resolved once when creation is durably accepted. |
| `at` | An offset-aware timestamp | One UTC `run_at`; new times already in the past are rejected. |
| `interval` | Integer `seconds`, at least 60 | UTC anchor plus a fixed period, independent of completion time. |
| `calendar` | Five-field cron expression and IANA timezone | Civil-time cadence with UTC occurrence identities. |

The UI generates the calendar representation from its presets; an advanced cron
input uses the same validator as the tool. Use the existing croniter/zoneinfo
dependencies. Validate finite integer bounds, reject booleans, and reject unknown
timezones or excessive/unsupported expressions. Delay timers may be shorter than
the poll interval; the preview explains that dispatch occurs on a scheduler tick,
whose current default is 30 seconds. This is not a precision alarm clock.

Calendar policy is explicit: skip a nonexistent local time during the spring
transition; run an ambiguous repeated local time once, at its first occurrence.
Test that policy around the timezone transition instead of assuming the cron
library supplies it. Intervals retain their UTC anchor across timezone changes.

An occurrence has a durable ID and a unique `(schedule_id, definition_revision,
nominal_due_utc)` key. A transaction claims a due occurrence and advances the due
cursor. Run-now and a due tick must compete for that same occurrence when one is
already due. An early manual Run-now is a distinct occurrence, consumes the same
allowance, and leaves the regular cadence in place. Agent tools cannot bypass
delay/cadence limits through Run-now.

There is at most one pending or executing occurrence per schedule. When the chat
is busy, later due slots coalesce into that pending work and advance the cursor;
they do not accumulate provider calls. Record the covered due interval and skipped
count. On reopen, make one catch-up occurrence for eligible overdue work and
advance to the next future slot. An expired, paused or exhausted schedule cannot
run a catch-up. Resume starts at the next future slot; it does not replay the
period deliberately paused.

Never replay accepted work with an uncertain outcome. Coalescing applies to due
work that has not executed, not to retrying a possibly completed tool effect.

## 5. Agent tool contract

Register a descriptor-backed `ScheduleToolProvider` with the existing catalog.
Use the `schedule:` provider namespace and these model-visible names:

| Tool | Purpose |
| --- | --- |
| `schedule_create` | Save instructions and a one-time or recurring timing value; optionally request a finite total run count. |
| `schedule_get` | Inspect one schedule in the calling chat, including its saved instructions and effective limits. |
| `schedule_list` | Read a bounded page of summaries for the calling chat. |
| `schedule_update` | Change instructions/timing or pause/resume an owned schedule using its expected revision. |
| `schedule_cancel` | Revoke future occurrences and request cooperative cancellation of an active occurrence. |

Example model arguments, with ownership supplied only by the runtime:

```json
{
  "instructions": "Check whether the export has finished and report the result.",
  "timing": {"kind": "delay", "seconds": 300},
  "max_runs": 1
}
```

```json
{
  "instructions": "Review new test results and summarize actionable failures.",
  "timing": {"kind": "interval", "seconds": 1800},
  "max_runs": 10
}
```

Successful create/update returns `schedule_id`, `revision`, `status`, normalized
timing, `next_run_at`, and effective remaining limits. A successful save does not
claim that a future run has executed. Return a structured refusal or
`needs_attention` state when authorization is missing. Cancellation is idempotent.

The calling run supplies conversation, owner, source chain, scheduling grant and
resource authority through trusted runtime context. Reject model-supplied owner,
conversation, grant, approval, budget-counter and execution-engine overrides.
Resolve `current_run_id()` against runtime-owned run records and a live calling
authority; a bare string or direct provider invocation is insufficient. Child
runs and unsupported engines neither advertise nor invoke the v1 tool capability.

`get` and `list` see only the calling conversation. Mutation additionally checks
the source authority: a scheduled turn may manage its grant's schedules, while
managing an unrelated schedule requires fresh human authority. A human-directed
main turn still uses the ordinary tool permission gate for those mutations.
Denied or stale calls change nothing. Updates require the current revision and
never increase an inherited grant's total runs, deadline, scope or budgets.
The grant also records its approved timing policy. With follow-ups disabled,
automatic updates may postpone a not-yet-accepted occurrence, change its
instructions, or pause/cancel the schedule; they cannot accelerate its cadence
or create another active timer. The original cadence remains authoritative after
a postponed occurrence. Creating follow-ups or widening this policy requires
human review. With follow-ups enabled, timers may use remaining allowance after
the grant's original first due time, with at least 60 seconds between accepted
starts across the grant. Admission enforces that interval even across different
timer IDs and app restart. Model arguments cannot enable follow-ups or move the
grant's original first due time earlier.

Creation has a runtime-owned idempotency key derived from the actual provider
tool-call identity, run and operation. Carry that identity across worker timeout,
retry, continuation and both tool protocols; do not trust a model's invented
idempotency or approval token. Repeating the same identity and normalized request
returns the original record and due time. A different request with the same
identity fails. For a new call identity, deduplicate an exact live request under
the same grant using normalized instructions, timing input, target and limits;
the unresolved delay input is part of that key, not a freshly recomputed due
time. The UI can explicitly request a separate creation. Completed records retain
their operation identities for retry handling, without treating all later
intentional uses of the same text as duplicates.
For fenced calls without provider IDs, persist a generation/call ordinal under
the run before tool invocation. Native IDs are also scoped to the run and
generation. Propagate the resolved identity into the actual tool worker; parsing
the same saved call again must not assign it a new logical operation identity.

Tool descriptions teach the model to write self-contained instructions, state
that timers run only while Chatbook is open, and explain that it can finish now.
Scheduling does not keep the current agent invocation or a worker thread alive.

## 6. Authority and finite automatic work

Separate a schedule's recurrence from its authority to spend resources. Both
are durable, inspectable and revocable.

- **Human form:** Save creates a scheduling grant for the displayed target,
  timing, tools and limits. An ongoing recurrence is an explicit standing
  authorization, still subject to per-run and per-conversation daily caps.
- **Agent in a human-origin main turn:** the existing mutation permission gate
  can authorize a finite grant. A one-time timer defaults to one total run;
  recurring timers default to ten and may request at most thirty. Multiple
  timers created by one source turn share that turn's grant; repeated creates
  do not add fresh allowances. The first accepted create fixes its finite total;
  a later create returns the remaining allowance or refuses an increase. Tool
  descriptions define `max_runs` as the total across this timer and its future
  descendants, so an agent planning multiple follow-ups requests the total up
  front. Increasing the grant requires the human form.
  These finite agent-created grants allow follow-ups within the fixed total;
  their permission review/result states that explicitly. A human-created grant
  permits follow-ups only when its separate control was enabled.
- **Agent in a scheduled turn:** created/updated timers inherit the same grant,
  remaining total run count, expiry, resource scope and daily accounting. A
  one-time grant with no remaining runs cannot schedule another wake. The agent
  can request a larger finite initial allowance for intended follow-up timers.
  A follow-up cannot start before the original first due time. Subsequent starts
  retain the minimum interval, total count, daily caps and optional end date even
  if the agent replaces one recurring timer with a series of one-time timers.
- **Other automatic origins:** without an applicable scheduling grant, a tool
  may save a bounded proposal requiring user review, but cannot activate a new
  grant. A fleet wake or goal iteration must not escape its existing chain's
  limits or deadline by creating a timer. Generic stored tool permission alone
  cannot upgrade an automatic proposal into new spending authority.

Count accepted scheduled turns across every descendant timer of a grant, not per
timer row. Removing or recreating a timer does not return that count. A scheduled
turn may create a new bounded occurrence chain only by atomically debiting the
standing scheduling grant; it cannot mint a grant. This explicitly amends the
otherwise strict ADR-134 rule that automatic work cannot create new allowances.

Initial proposed defaults, editable by the user through canonical settings and
narrowed by a schedule's approved values:

| Bound | Default |
| --- | ---: |
| Live schedule definitions per conversation, including paused/proposed | 8 |
| Scheduled turn starts per conversation per UTC day | 96 |
| Physical model calls per conversation per UTC day | 128 |
| Budget tokens per conversation per UTC day | 2,000,000 |
| Automatic child launches per conversation per UTC day | 6 |
| Model calls per occurrence | 8 |
| Budget tokens per occurrence | 128,000 |
| Maximum output per model call | 8,192 |
| Native steps per occurrence | 64 |
| Elapsed execution time per occurrence, including preparation/approvals | 240 seconds |
| Maximum saved instruction size | 16 KiB UTF-8 |

These are initial policy choices, not measured throughput or model-quality
claims. Existing stricter executor, workspace and resource limits still win.
Zero disables automatic execution for the corresponding limit; it does not mean
unlimited. Settings cannot retroactively widen a saved grant. An agent cannot
change these settings through the scheduling tool.

Keep the master control under `[scheduling] chat_schedules_enabled`; the feature
is available by default, with no work until a user-authorized schedule exists.
Turning it off prevents creation/reactivation and automatic acceptance, and
signals active work through normal cancellation checks. Reading history,
pausing and cancelling remain available. It cannot free occupied resources or
erase charges. Use a separate scheduled-turn settings policy rather than
overloading `autowake_enabled` or `goal_runs_enabled`.

The user's standing schedule authorizes daily allowance windows. UTC keys never
depend on the model or the schedule's display timezone. Reserve each physical
call against its occurrence, grant and conversation window in the same FULL
transaction, using the existing exact-request accounting. Descendant calls and
children count too. Calls already reserved before midnight settle in their
original window. Detect clock rollback before admitting a new window; do not
reopen consumed windows or reset uncertain reservations on restart.

A daily ceiling defers eligible future work to the next authorized window and
shows **Daily limit reached**. A finite grant's exhaustion requires a human edit;
it never auto-renews. Per-occurrence exhaustion ends that occurrence, records its
actual outcome and permits the next authorized cadence slot. Unknown usage
retains its conservative reservation and requires recovery review. These are
admission bounds in budget-token units, not currency guarantees.

## 7. Storage and runtime interfaces

AgentRunsDB gains versioned tables for schedule definitions, scheduling grants,
occurrences and daily reservations. Keep prompt/settings bodies private and
bounded; list/status/diagnostic projections carry IDs and safe metadata only.
Use optimistic revisions, parameterized SQL, deterministic occurrence/message
identities and FULL-synchronized admission. Preserve accounting tombstones and
active/uncertain dependencies through existing pruning paths.

Suggested focused modules and their responsibilities:

- `Scheduling/chat_schedule_models.py`: validated definition, cadence, grant and
  result contracts; no widgets or model calls.
- `Scheduling/services/chat_schedule_service.py`: create/edit/control and
  permission/authority validation through AgentRunsDB's scheduling store.
- `DB/chat_schedules.py`: transactional definition/grant/occurrence operations;
  the existing automatic ledger remains the accounting owner.
- `Scheduling/services/chat_schedule_projection.py`: body-free `ScheduledTask`
  records for local rows only, used by both the queue and Schedules screen.
- `Scheduling/scheduler/handlers/chat_schedule_handler.py`: durable due-work
  handoff to the coordinator, returning promptly without awaiting generation.
- `Chat/console_scheduled_turns.py`: an app-owned coordinator that restores the
  correct conversation and admits bounded native turns through the controller.
- `Agents/schedule_tool_provider.py`: descriptor-backed tool adapter over the
  service, with trusted per-run context and existing permission review.
- A Console scheduling module and form: thin adapters over that same service.

Extend `ConsoleSubmissionOrigin` and `AutomaticWorkContext` with an explicit
scheduled-turn policy. Every dispatch/accounting path must resolve policy from
trusted origin; it must not accidentally use fleet autowake or goal settings.
Preserve existing full-request tool checks, selected resources, CLI/skill/MCP
permissions and project-instruction revalidation after waits.

On a due tick, the handler durably records pending work and returns. The
coordinator owns execution tasks separately so reminders and other schedules
continue while an LLM or tool is running. Queue refresh on definition/control
changes uses the existing callback mechanism. Busy work uses bounded retries and
capacity notifications; one blocked chat cannot starve another eligible chat.

Restore runtime dependencies from persisted configuration on first due work even
if Console has never opened in this process. Share existing app ownership and
startup recovery with goals/fleet; constructing this service must not trigger a
second global recovery. One app owner admits work. A second process sharing the
same store cannot claim the same schedule; loss of ownership never proves that
an old physical tool or remote effect has stopped.

## 8. Busy chats, edits and cancellation

Only one native primary may occupy the target conversation. Use the existing
manual capacity reservation and user-priority probe before automatic admission.
Do not start into a live user turn, queued manual turn or active composition.
Once executing, existing Stop/queue controls remain available; leaving Console
does not itself cancel scheduled work.

Claim, permission review and actual turn acceptance are separate states. Recheck
the definition revision, grant, target conversation, provider/resources and
permissions immediately before acceptance and after asynchronous preparation.
An edit or pause won before that transaction prevents the old pending work from
starting. The accepted turn uses its immutable snapshot. Later edits affect
future turns, with an explicit **Cancel current response** action when needed.

Pause removes future/pending eligibility and preserves history. Cancel also
signals an active turn cooperatively and shows **Cancelling** until cleanup
settles. A cancellation request never refunds already accepted work or releases
physical ownership early. Stopping only the current response consumes that
occurrence; it does not silently disable an otherwise active schedule.

Pausing/cancelling a grant's root schedule applies to its descendant timers too;
otherwise a visible root control could leave an agent-created follow-up running.
Cancelling a descendant affects that descendant only. Resuming a descendant cannot
reactivate a paused or revoked grant. The UI labels root controls as applying to
all related timers, and the tool result includes the affected schedule IDs.

Deleting/archiving the target chat revokes its scheduling grants and stops future
work; it never recreates the conversation. Workspace removal, resource identity
changes and inaccessible provider configuration surface a blocked state. Restoring
the chat or a permission does not silently reactivate a revoked schedule.

## 9. Failure, recovery and visibility

Use separate definition and occurrence states. Definition states include active,
paused, needs_attention, exhausted, cancelled and completed. Occurrences distinguish
pending, accepted, succeeded, failed, timed_out, cancelled and recovery_required.
Do not reuse the reminder status **Missed** to conflate dispatch failure with an
overdue occurrence.

At restart, pending work with no durable acceptance may be reclaimed under the
same occurrence identity. Accepted work without a conclusively durable terminal
result becomes recovery_required and pauses the affected schedule/grant. A future
cadence slot must not conceal an unresolved previous effect. Other independent
schedules remain eligible.

Chat persistence uses deterministic notice/reply IDs derived from the occurrence
and existing message ownership. If transcript persistence succeeds but terminal
accounting fails, retain the visible response and reconcile or require review;
do not make another provider call to repair the record. No exactly-once external
effect claim is made: acceptance deduplication and conservative uncertainty
handling bound what Chatbook can establish.

Provider-unavailable preflight stays pending with a bounded retry delay and an
actionable status. A definitively failed or timed-out accepted occurrence is
recorded and the next ordinary cadence remains eligible. Three consecutive such
failures pause the schedule for attention. A pending permission/uncertain effect
always blocks that schedule immediately instead of producing repeated prompts or
effects each tick. Resuming requires the relevant permission or recovery action,
not merely dismissal of a notification.

Permission review may suspend the currently accepted occurrence through the
existing native approval mechanism, under its execution deadline. Surface the
specific pending approval and prevent subsequent occurrences while it waits. If
the bound expires, cancel and settle that occurrence; later approval cannot
resurrect its stale invocation. The user can resolve permissions and resume the
schedule for a future occurrence. A proposal lacking a scheduling grant requires
the separate explicit schedule review, which displays the complete normalized
future-work authorization before enabling it.

Agent creation and schedule controls are visible in the transcript/activity
history. Finished responses mark their conversation unread when not in view.
Use existing notification delivery for completed replies and required attention;
do not emit notifications for every unchanged pending tick. Notification policy
is configuration, not instructions injected into the model. This release does
not need an LLM-based change detector or claim to suppress semantically unchanged
answers.

## 10. Verification and implementation boundaries

Run targeted tests only. Use fake time, real SQLite and the real Console/runtime
route with a deterministic recording provider and tool implementation. Tests
must observe actual admission, provider/tool calls, transcript persistence and
accounting, not merely a successful service return or an empty screen.

Required cases:

1. Mounted `/schedule` completion, prefill, Save, cancel and disabled paths;
   creation never clears a draft or fires before confirmation.
2. One-time/interval/calendar cadence, timezone gaps/folds, end dates, precision
   copy, coalescing and one catch-up after restart. Pause/resume and Run-now must
   retain their defined semantics.
3. Due tick racing Run-now, two DB handles/process owners, duplicate create tool
   calls, stale edits and cancellation before/after acceptance. Count physical
   provider invocations and preserved occurrence IDs.
4. Navigation away, another active chat, and a fresh process that never opens
   Console. Assert delivery to the correct persisted conversation and unchanged
   composer/attachments, with a recording streaming and non-streaming provider.
5. A busy chat and an unavailable provider beside a ready chat. Assert bounded
   attempts, ready-chat progress and uninterrupted reminder dispatch while a
   scheduled tool is deliberately held open.
6. Actual model-generated scheduling calls through both native and fenced
   protocols, normal approval, denial, missing run context, forged target/grant,
   cross-conversation access and unsupported child calls.
7. A scheduled agent creates successor timers. Across repeated definitions,
   cancellation/recreation, UTC midnight and restart, root total runs and daily
   model/token/child limits remain authoritative; an automatic origin without a
   grant cannot activate a new one. Fixed human cadence cannot be accelerated
   without follow-up authorization; root pause/cancel covers descendants.
8. Revoked resources/permissions after an asynchronous wait; per-call approval
   identity must match the exact immutable arguments reviewed.
9. Crash boundaries before/after acceptance, provider completion, transcript
   commit and accounting commit. Unknown work is not replayed or treated as free.
10. Schema migration and rollback, retention of active/uncertain references,
    metadata privacy, and targeted existing goal/fleet/tool/scheduling regressions.

Before implementation, create a plan with independently testable slices covering
durable lifecycle/admission, native dispatch, composer/Schedules controls, and
agent tools with combined recovery qualification. Implement both frontends over
one service; a tools-only scheduler is not this feature. Link ADR-143 from that
plan and its execution tasks. Allocate migration/task IDs against current refs
at implementation time rather than reserving guessed future numbers here.

The worktree is the existing native-goal branch. Its draft PR already contains
divergent prerequisites; this design neither resolves nor hides those integration
dependencies. No runtime implementation, full-suite sweep, or new live provider
qualification is claimed by this document.
