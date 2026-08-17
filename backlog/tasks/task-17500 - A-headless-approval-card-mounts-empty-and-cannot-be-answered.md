---
id: TASK-17500
title: 'A headless approval card mounts empty and cannot be answered'
status: To Do
assignee: []
created_date: '2026-08-17 13:55'
labels:
  - console
  - agents
  - approvals
  - bug
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the task-15860 close-out live pass (2026-08-17, dev `524194c15`,
real Anthropic model, isolated scratch profile).

When a woken supervisor turn reaches a risk-tagged tool while no Console
screen is mounted, everything up to the card works: an app-wide toast
names it ("Agent in "…" needs approval to use a tool. Open Console to
review — nothing runs until you answer."), the session takes its `◆`
badge, and the status bar reads `Approvals: 1 pending`. Opening Console
then shows an approval card that is **visible but empty** — the
"Approval required" title and nothing else: no tool row, no arguments,
no Approve/Deny controls. The run stays blocked and the user has no way
to answer the thing they were just told to come and answer.

Switching Console session tabs (which re-derives the card through
`switch_session`) renders the SAME round correctly and completely, so the
payload is intact and the round is genuinely answerable — it is the
open-Console path specifically that mounts a body-less card. A round
armed while Console was MOUNTED renders fully from the start, which is
the control that makes this headless-specific.

The consequence is larger than one stuck card, because deliveries are
serialized app-wide (one `_delivering` per runtime): while the blocked
round sits unanswerable, **every other conversation's owed wake is held
too**. Observed live — a second conversation's completion sat undelivered
with its `◈` mark set until the blocked round was denied, at which point
it delivered immediately.

This falsifies both the shipped User Guide sentence ("the card is
waiting, already mounted, the moment you open Console") and the headless-
approval landing's own acceptance criterion ("a round armed while
detached and still armed at attach must mount its card, not be silently
re-parked").
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 A risk-tagged tool call in a wake turn that armed while Console was unmounted shows a fully rendered, answerable approval card (tool name, arguments, decision controls) the first time the user opens Console — with no session switch required
- [ ] #2 The same holds on the launch-wake path (the round armed before any Console screen ever existed in the process)
- [ ] #3 A regression test drives the failure through the real attach path and fails on current dev before the fix
- [ ] #4 While an approval round is unanswered, other conversations' owed wakes are either delivered or the blocking is documented as intended — the current behaviour (all app-wide wakes silently held behind one unanswerable card) is decided one way or the other, not left implicit
- [ ] #5 `Docs/User_Guide/console/agent-runs-and-tools.md` describes what actually ships
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Reproduce first; the seams below are where the live evidence points, not a
diagnosis.

1. Reproduce in a test: arm a risk-floored round through a wake turn with
   the view detached, then attach a fresh `ChatScreen` through the real
   navigation API and assert the card's rows/buttons exist (not merely
   that the card is displayed). Expect RED.
2. `ConsoleRuntime.attach_view` calls `_bind_view_hooks()` before
   `remount_pending_approval()`, so the hook ordering is right; the
   suspect is WHERE in the screen lifecycle the attach happens.
   `attach_view` runs from `_ensure_console_chat_store` during
   `restore_state`, i.e. BEFORE the incoming screen's widgets are
   composed, and `ChatScreen.sync_task_resume_state` swallows
   `QueryError` when `#console-task-surface` does not exist yet
   (`chat_screen.py`). Check whether anything re-pushes
   `_task_resume_state` into the widgets once they mount.
3. `ChatTaskCards.sync_state` shows the container from
   `has_pending_approval()` (truthy dict) but populates the card from
   `approval.get("calls")` — so a payload that arrives with no calls, or
   never arrives at the widgets at all, produces exactly the observed
   "title only" state. Distinguish those two before fixing.
4. Mutation-test the new test against the fix.
<!-- SECTION:PLAN:END -->
