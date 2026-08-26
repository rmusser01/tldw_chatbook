---
id: TASK-15860
title: 'Headless wake: fire the supervisor auto-wake with no Console screen mounted'
status: Done
assignee: []
created_date: '2026-08-13 13:47'
labels:
  - console
  - agents
  - architecture
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fleet PR 3a-2 (`feat/fleet-autowake`) delivers auto-wake wherever a Console
controller exists: the wake fires immediately when a Console screen is mounted
(the user can be in any session — the toast/badge reach them app-wide), and
when no controller exists the durable FLEET_UNSEEN mark plus the per-run
`agent_runs.wake_delivered_at` ledger stage the wake, which is claimed
synchronously at the next Console mount.

The remaining gap is a supervisor that ACTS with no Console mounted at all
("headless wake" — e.g. the user stays on Library while a child finishes; the
toast reaches them but the supervisor does not act until Console is opened).
The bridge and controller are per-screen (screens are never cached;
`ChatScreen.on_unmount` runs `controller.shutdown()`), so there is nothing to
run a wake turn INTO while Console is unmounted. Closing the gap requires
moving bridge/controller ownership above the screen (app-level lifetime),
which is an architectural change deliberately excluded from PR 3a-2 (see the
plan's "Deliberately NOT in this PR" section,
`Docs/superpowers/plans/2026-08-13-supervisor-fleet-pr3a2-autowake.md`).

Everything else headless wake needs is already built and is shared substrate:
the hardened terminal signal + fan-out seam, the durable unseen-completion
mark, the wake-delivery ledger, and the full wake machinery (kill switch,
app-wide serialization, user-wins-ties, approval floor, AGENT_WAKE
authorization). The only delta is WHERE the wake runs.

**State update 2026-08-14 (wake-integrity arc, tasks 15970/15971):** the
"no Console at all" case has NARROWED. On current dev a Chat screen can
stay resident (mounted, controller live) after nav-away — observed live
and harness-reproduced: a navigation issued while a pushed screen sits
above Chat pops the modal off the stack, not Chat — and the coordinator's
design ruling made off-screen delivery from such a resident screen the
INTENDED behavior (the user learns of it via the settle toast + the
FLEET_UNSEEN ◈ mark, which an off-view delivery now leaves SET until
viewed; `ConsoleFleetWakeCoordinator._conversation_in_view` /
`ChatScreen._console_wake_conversation_in_view`). The genuinely-headless
gap this task owns is therefore restart / first boot — the window before
the first Console open, where no controller has ever existed — plus any
nav path that truly unmounts the Chat screen (the plain nav path still
does: `on_unmount` → `controller.shutdown()`). The description above
predates that ruling; scope accordingly.

**Correction 2026-08-14 (task-16300) — the narrowing above is REVERSED.**
The state that narrowed it (a Chat screen resident after nav-away) was a
screen-stack leak, not a residency model: `App.switch_screen` pops only
the top of the stack, so a navigation under a pushed screen replaced the
MODAL and left Chat running behind the new screen, against the invariant
`app.py`'s `_create_navigation_screen` documents. It is fixed —
navigation now reduces the stack to its content screen first, so leaving
Console unmounts it and shuts the controller down on every path. **The
open coordinator call recorded here is therefore closed toward
always-unmount, and the staged-wake path grows back to covering every
nav-away**: this task's ACs cover restart, first boot, AND ordinary
navigation away from Console again. The 15971 off-view ruling is
unaffected — it applies to a Console that is mounted but not looked at (a
modal covering it, a different session tab active), which never depended
on residency.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 A finished background sub-agent wakes its supervisor while no Console screen is mounted, under the same `autowake_enabled` gate, caps, and approval floor as the mounted case
- [x] #2 The ownership move does not regress documented screen-scoped semantics (leaving Console still cancels streaming turns and denies parked approvals; survivors keep running)
- [x] #3 Every wake invariant holds headless: no USER transcript row, exactly-once via the `wake_delivered_at` ledger, no phantom wake after restart
- [x] #4 The User Guide's honest-limits paragraph about the headless gap (Docs/User_Guide/console/agent-runs-and-tools.md) is removed or rewritten when the limit no longer holds
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Shipped across eight landings plus this close-out; each has its own report
under `Docs/superpowers/plans/2026-08-14-headless-wake-*.md`, and the
close-out is `…-closeout-report.md`.

**The change, in one line:** the Console runtime (agent bridge +
`ConsoleChatController` + the message store) is owned by the app, not by
`ChatScreen`; a screen attaches as a VIEW and detaches at unmount, and
`ConsoleFleetWakeCoordinator._attempt` refuses only a DISPOSED controller
(app exit), never a merely-ended visit. Ownership (`f09bb991d`), lifetime,
viewless hook defaults, store continuity, the gate itself, headless
approval, and wake-at-launch landed as separate, separately revertable
PRs, in that order, on the owner's staging condition.

**Close-out (this task's last commits).** Plan Task 7's four invariants
are proven together on current dev in
`Tests/UI/test_console_headless_invariants_gate.py`, which closes the
three gaps per-landing coverage left: the no-USER-row assertion on the
kill-switch RELEASE path, "OFF loses nothing durable" asserted on the
persisted ROWS, and app-wide serialization asserted where only
`_delivering` can enforce it (a second conversation with an idle session).
Both new gates are mutation-tested. Plan Task 8 rewrote the User Guide's
wake sections, added the spec's superseding note, closed its follow-up
row, and recorded two lessons.

**Two honest residues, documented rather than hidden.**
1. A process killed between a wake turn's acceptance and its ledger stamp
   re-announces the completion exactly once at the next launch (never
   loses it, never repeats beyond one). Measured in a test and then
   reproduced live. The User Guide claimed the stronger thing; corrected.
2. **task-17500** — a headless approval round's card mounts empty and
   cannot be answered until the user clicks that session's tab, which
   also stalls every other conversation's owed wake behind it. Found by
   the live pass, not by tests; a mounted round renders fully, which is
   the control making it headless-specific.

**Live verification** (dev `524194c15`, real `claude-sonnet-5`, isolated
scratch profile, tmux): wake in another session, wake while on Library,
wake at launch with Console never opened, exactly-once on a second
relaunch, and the kill switch off/on — all driven and checked against the
app's own databases. Details and panes in the close-out report.
<!-- SECTION:NOTES:END -->
