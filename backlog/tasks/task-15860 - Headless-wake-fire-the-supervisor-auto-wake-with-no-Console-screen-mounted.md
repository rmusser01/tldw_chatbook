---
id: TASK-15860
title: 'Headless wake: fire the supervisor auto-wake with no Console screen mounted'
status: To Do
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
predates that ruling; scope accordingly. Also note: whether the
resident-screen state itself (the modal-atop-Chat stack leak) is a nav
bug to fix or the intended residency model is the coordinator's open
call — see task-16210 (filed by the wake-integrity arc with the harness
repro); if it is fixed toward always-unmount, the staged-wake path grows
back to covering every nav-away and this task's ACs cover more ground
again.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 A finished background sub-agent wakes its supervisor while no Console screen is mounted, under the same `autowake_enabled` gate, caps, and approval floor as the mounted case
- [ ] #2 The ownership move does not regress documented screen-scoped semantics (leaving Console still cancels streaming turns and denies parked approvals; survivors keep running)
- [ ] #3 Every wake invariant holds headless: no USER transcript row, exactly-once via the `wake_delivered_at` ledger, no phantom wake after restart
- [ ] #4 The User Guide's honest-limits paragraph about the headless gap (Docs/User_Guide/console/agent-runs-and-tools.md) is removed or rewritten when the limit no longer holds
<!-- AC:END -->
