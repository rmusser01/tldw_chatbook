# Task 62: completed plan retention and lookup cost

Bounded source-only audit, 2026-11-17. No imports, native execution, tests or
product edits. Line anchors reflect the live source while the separate
cancellation-order correction was being prepared; that correction is not
reviewed here.

## Concrete finding

`ConsoleLocalControls._plans` (`console_control_dispatch.py:422`) is populated at
`:1061` and has no removal/clear path in its owner or `ConsoleTurnRetirement`.
`_PendingPlan` retains its original sealed plan, Task and acknowledgement Future.
The sealed plan includes store/database, session, preparation, inputs and
acceptance state (`console_durable_commit_plan.py:385`). Successful retirement
clears the retirement object's references but leaves these independent roots.

With N historical plans in this service epoch:

- New ordinal calculation scans all N keys (`console_control_dispatch.py:1046`).
- Original native effect admission scans all N plan values (`:1170`) even though
  the supplied plan has an exact original turn/task owner.
- Waiting-plan publication (`:994`) and source fencing (`:1364`) also scan all N.

Each is O(N) in lifetime plan history. With a bounded number of plans per turn,
the accumulated lookup work over T turns can grow quadratically. This is a
source fact, not a measurement of its contribution to the current journey time.

`_turns` is likewise never removed (`:419,1630`). Its entries retain the original
identity, custody record, Task and source graph. It also has necessary historical
identity roles: duplicate ReceiveTurn is refused by membership (`:1586`), and
ACK checks compare the original session/incarnation (`:1429`). Drain currently
iterates/gathers every retained task (`:946`). Removing `_plans` alone therefore
does not eliminate all completed-turn references.

## Why blindly deleting finished records changes behavior

`CommitPlanAck` dispatch (`console_control_dispatch.py:1420-1445`) rejects a
missing/abandoned/conflict/uncertain plan. A matching ACK for an acknowledged,
started **or finished** plan succeeds without a second native entry. A differing
fingerprint/request/session/incarnation fences that turn, marks conflict,
cancels the original task, and refuses. The command must still pass the existing
service-epoch/sequence/request-ID checks first.

Thus removing an abandoned record preserves its existing late-ACK refusal;
removing a finished record turns a previously accepted matching duplicate into
`no longer pending`. Deleting its `_turns` identity also loses the explicit
duplicate-ReceiveTurn barrier. Keeping only a boolean finished flag is not enough
to distinguish matching duplicates from conflicts.

The process client already drops per-received plans on a **positive** retired
message (`console_execution_process.py:529-531`), so normal parent ACKs stop at
that boundary. This does not justify silently changing the local controls'
matching duplicate semantics. Existing duplicate-ACK coverage sends both ACKs
before awaiting work completion (`Tests/Chat/test_console_control_dispatch.py:370-389`);
it does not establish post-retirement behavior as tested.

## Minimal safe integration point and options

Use `ConsoleTurnRetirement._settle`'s `confirmed` branch
(`console_execution_retirement.py:345-377`), before clearing its task/record/source
and retiring its registry entry. At that point the original task has finished,
logical custody is closed, source custody is closed, every retained read has
positive retirement, and every original effect has positively retired. Under the
existing controls lock, match the exact turn key, identity, custody record and
Task before releasing any related plan references. Do not compact on task.done
alone, cancellation, receipt Applied, or uncertain/negative retirement.

The smallest changes can stay within those existing owners:

1. Keep the next ordinal on the original per-turn owner instead of counting the
   whole service map. Retain it through every retry in that turn; never reuse an
   ordinal after abandoning a plan while the turn can still issue another.
2. Retain the exact pending-plan reference/key on that same original turn owner
   at registration. Native admission can directly check the existing map entry
   is that pending record, then run all current task/source/state/fence checks.
   This removes the lifetime scan without relaxing provenance.
3. After positive turn retirement, release the heavy sealed-plan/task/Future
   references. Preserve only the existing ACK identity/fingerprint/terminal
   outcome needed for duplicate/conflict behavior, plus the original turn's
   identity needed for replay/session checks. The already keyed entries can be
   compacted; no new active/settled global registry is necessary. Any compaction
   must also ensure drain skips already-proven terminal owners instead of
   retaining or gathering their Tasks. Failed/uncertain owners remain unchanged.

The memory tradeoff is explicit: preserving exact replay and matching duplicate
semantics for an unlimited service epoch still retains small identity receipts
proportional to history. That is different from retaining entire plans, Tasks and
session graphs. Removing even those small receipts requires a separately agreed
terminal/replay lifetime contract; this audit does not propose one.

No broader registry redesign or new architecture round is needed to address the
two hot lookups and release the heavy references at the existing positive
retirement boundary. The current cancellation-order fix should be verified
before this independent change is integrated.
