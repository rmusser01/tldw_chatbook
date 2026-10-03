# Chat schedules and agent timers: design review

Date: 2026-10-02

Assessment: revise the contracts below before implementation planning. The local,
same-chat workflow and agent timer capability remain the approved scope.

## Scope and evidence

Reviewed the [specification](../specs/2026-09-09-chat-schedules-and-agent-timers-design.md),
[proposed ADR-143](../../../backlog/decisions/143-local-chat-schedules-and-agent-timers.md)
and [design task](../../../backlog/tasks/task-32195%20-%20Design-local-chat-schedules-and-agent-self-timers.md)
at immutable commit `627cb86f77692530c0a83eda955321f2e78c8236`, whose parent is
`20bb97cdded8f45cc77fd9848d232e5dc951b24c`.
Two independent, read-only reviews covered authority/accounting and
runtime/lifecycle behavior; findings were checked against the source.

The owned worktree supplies the source references below. A read-only comparison
with the newer shared checkout also checked scratch authority, workspace defaults,
hooks, design tokens and the existing local-automation handler. That checkout is
based on `9ba96ebb626dd010f14d093b0e8d40f37c715d32` and has unrelated uncommitted
changes; it is integration evidence, not a tested or modified dependency.
No application code changed and no runtime tests were run for this design review.

The draft already has useful safeguards: trusted caller identity, shared
descendant allowances, explicit follow-up authorization, immutable accepted turns,
cooperative cancellation and conservative uncertain-effect recovery. Preserve
those safeguards while closing the gaps below.

## Findings requiring design revisions

### 1. P1: establish live ownership before startup recovery

Spec lines 366–371 require one app owner and safe behavior when a second process
opens the same store. Existing [startup recovery](../../../tldw_chatbook/DB/automatic_work.py)
at lines 1109–1127 unconditionally replaces the owner and marks foreign unfinished
work uncertain. [ADR-135](../../../backlog/decisions/135-fleet-completion-delivery-and-crash-recovery.md)
lines 144–146 explicitly exclude cross-process admission and live-work reclamation.
Starting process B while A executes can therefore revoke healthy A without
stopping A's physical worker. Occurrence deduplication alone cannot establish
the promised cross-process primary exclusion.

Specify a database-scoped live-owner acquisition boundary before recovery and
admission. A secondary process must not steal the owner or recover live work.
Define dead-owner recovery and which manual/automatic admission paths participate
in the same-chat guarantee; an expired timestamp alone is not proof of quiescence.
Qualification must use two genuinely live processes, a blocked tool in A, B's
startup, A's exit and late cleanup. B must neither replace a live owner nor admit
conflicting work, and dead-owner uncertainty must remain charged.

### 2. P1: bind the human-reviewed task to its grant

Spec lines 211–215 allow an automatic turn to replace a root's instructions even
when follow-ups are disabled. Lines 247–249 bind the grant to target, timing,
tools and limits but omit the reviewed task. A schedule saved to report disk usage
could replace future instructions with file cleanup under stored
`schedule_update` and file-mutation permissions. Permission to use a tool does
not establish authorization for the substituted recurring task.

Keep human-reviewed root instructions immutable to automatic updates by default.
Require an exact-task human review for replacement, or explicitly display and
authorize a separate instruction-edit delegation. Preserve any deliberately
authorized finite follow-up capability. Test that stored tool permissions and
follow-ups disabled cannot authorize a root body change, while a human-reviewed
replacement can.

### 3. P2: re-arm the live queue after cursor advancement

Spec lines 360–364 require refresh after definition/control changes but omit
ordinary due-cursor changes. The [scheduler loop](../../../tldw_chatbook/Scheduling/scheduler/loop.py)
removes due projections at line 172 and otherwise reloads every 60 ticks.
With the documented 30-second poll, a one-minute recurrence can disappear from
the queue after its first dispatch until the approximately 30-minute reload.

Require notification/requeue after every committed cursor advance and relevant
retry-time change. Reuse the newer
[automation handler's callback contract](https://github.com/rmusser01/tldw_chatbook/blob/9ba96ebb626dd010f14d093b0e8d40f37c715d32/tldw_chatbook/Scheduling/scheduler/handlers/automation_handler.py#L402)
rather than building a second queue mechanism. Verify three consecutive ordinary
due times through the real loop without a forced reload or intervening edit;
other reminders must progress while the scheduled response runs.

### 4. P2: define one-time edit, resume and Run-now transitions

Three rules conflict or lack a one-time outcome: early Run-now leaves the regular
cadence in place (lines 146–148); Resume selects the next future slot (156–157);
and an edit fences pending work after claim has advanced its cursor (145,
381–385). A one-time timer has no next slot. With spare grant allowance, early
Run-now can also execute its original due time again; an instructions-only edit
after claim can discard its sole occurrence.

Add a transition table covering before claim, pending, accepted and terminal
states. Recommend that early Run-now consumes the one-time occurrence, overdue
Resume requires explicit retiming, and instructions-only edits transfer the
unaccepted nominal occurrence to the new revision without spending allowance.
Timing edits should explicitly supersede pending work. Test every transition with
remaining grant allowance greater than one so exhaustion does not hide duplication.

### 5. P2: decide what happens to slots due during execution

Lines 151–160 allow at most one pending or executing occurrence and coalesce
unexecuted work; lines 381–386 make accepted work immutable. For a three-minute
response on a one-minute cadence, the intervening slots cannot unambiguously
coalesce, queue or disappear. This can produce an unintended immediate follow-up
or lose the skipped-count/cursor history.

Define pending coalescing separately from slots due after acceptance. The smallest
policy is to record slots covered by active execution as skipped and resume at
the next future slot after settlement. Specify cursor and covered-interval
updates, including approval waits. Test slow completion, paused approval,
cancellation and a daily-window boundary; accepted work must stay immutable.

### 6. P2: extend replay receipts to every mutation

Lines 222–236 provide creation receipts, while line 209 makes stale updates
no-ops. If an update commits revision 6 and its result is lost, replay with
expected revision 5 returns a stale failure although the operation succeeded.
This falls short of ADR-143 line 44's mutation-idempotency contract.

Transactionally store a normalized request hash and result receipt for update and
control operations as well as create. Resolve a receipt before checking the
revision; identical replay returns the original result, and changed arguments
under that identity fail. Test committed-but-lost update, pause, resume and cancel
results through both tool protocols, including the affected descendant IDs.

### 7. P2: bound retained records, not just live definitions

The eight-live-definition ceiling at line 286 does not bound completed/cancelled
definitions, grants, occurrence history, reservations or receipts. Lines 230–232
retain completed operation identities, and 329–334 supply no aggregate retention
contract. Repeated create/cancel cycles can grow storage indefinitely while using
one live slot; an ongoing human recurrence has the same problem. Existing goal
payload quotas do not cover these new tables.

Define aggregate retained-data admission limits and safe compaction for settled
payloads, receipts and daily windows. Pin active/uncertain dependencies. Expired
operation identities must be refused rather than treated as a new creation.
Test many lifecycle cycles, old-call replay and unresolved reservations together:
retention stays bounded, replay creates no work and uncertainty survives pruning.

### 8. P2: specify scheduled instructions in later model history

Lines 116–119 define transcript provenance but not later provider-history replay.
The existing [controller](../../../tldw_chatbook/Chat/console_chat_controller.py)
sends automatic instructions transiently as USER at lines 3844–3851, persists
them as a machine-origin SYSTEM row at 3876–3883, and excludes SYSTEM rows from
later history at 13188–13193. Reusing that path unchanged can leave assistant
answers to different scheduled tasks in context without the requests they answer.

Choose an origin-preserving projection for scheduled request/reply pairs. The
projection must not turn stored automatic content into a new human grant or user
prompt-history entry. Test the next manual question and next scheduled request
after streaming, non-streaming, remount and restart. This is a missing contract,
not a claim that the deliberate existing fleet-wake replay policy is incorrect.

## Integration requirements and improvements

- Enumerate persistable capture fields. Under current ADR-082, scratch locators
  and generations are live-session authority and cannot be persisted or
  reattached. A scheduled turn needs fresh scratch in its current runtime;
  persistent files require an explicitly selected binding. Explain that temporary
  scratch files do not survive restart and never fall back to `workspace_root`.
- Re-resolve the current workspace permission profile and persona policy on each
  run under ADR-079. Reuse the current turn-configuration seam and deny-only hook
  boundary under ADR-148, including automatic-origin invocation. New `schedule:`
  tools must respect advertising, confirmation floors, caps and PreToolUse denial.
- Clarify lines 426–438: a timed-out tool result is not physical settlement.
  The existing tool timeout helper can abandon a still-running thread. Keep a
  hard scheduled wall deadline even where ordinary human approval pauses execution
  time; retain ownership/reservations and block the next slot while an effect is
  uncertain. A permission timeout before dispatch can settle without claiming an
  external effect occurred. Test both paths and a late worker completion.
- Specify app-level cold startup without first opening Console, shared primary
  capacity across goals/fleet/schedules, and real controller/bridge hook
  registration. Exercise this through runtime attachment, not a preinitialized
  backend fixture.
- Compose the form, compact control and Schedules rows from ADR-150's design
  tokens and interaction rules. Use the canonical F9 settings surface, source CSS
  modules and Python 3.12. This is an integration update to the September draft.
- Defer advanced cron input and chat-specific Run-now UI if useful for the first
  implementation. The approved composer form and finite agent timers do not need
  those extra controls. This is optional scope reduction, not a required change.

## Backlog integration collision

The read-only ref/worktree sweep found two different `TASK-32195` records. Their
first-add provenance is:

| Record | First-add commit | Author time |
| --- | --- | --- |
| This scheduling design | `627cb86f77692530c0a83eda955321f2e78c8236` | 2026-09-09 19:16:04 -07:00 |
| Report web-search backend failures correctly to agents | `e3ce4d3f4c50a9fe513b42a872e63c2f902526c0` | 2026-09-09 21:25:19 -07:00 |

The later record is present on current `dev`. Under the repository's add-commit
rule, that later claimant must move with all its references. No shared files or
identifiers were changed during this review. Reconcile the collision and repeat
the live identifier sweep before integration; do not rename the earlier design
solely because the later claimant merged first.

## Readiness

Keep ADR-143 Proposed and the design task In Progress. Revise the eight contracts,
carry the integration requirements into the implementation plan, and review the
resulting transitions and admission boundaries before implementing runtime code.
ADR check: this review introduces no accepted architecture decision; it reviews
the already proposed ADR-143 and identifies amendments it needs.

Documentation checks for this review cover links, Markdown fences, whitespace,
task status and preservation of the existing user-authored goal-plan bytes.
Runtime qualification remains future work, with the concrete cases above required
when the feature is implemented.
