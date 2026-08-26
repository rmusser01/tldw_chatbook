---
id: TASK-17500
title: A headless approval card mounts empty and cannot be answered
status: Done
assignee:
  - '@claude'
created_date: '2026-08-17 13:55'
updated_date: '2026-08-17 21:53'
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
- [x] #1 A risk-tagged tool call in a wake turn that armed while Console was unmounted shows a fully rendered, answerable approval card (tool name, arguments, decision controls) the first time the user opens Console — with no session switch required
- [x] #2 The same holds on the launch-wake path (the round armed before any Console screen ever existed in the process)
- [x] #3 A regression test drives the failure through the real attach path and fails on current dev before the fix
- [x] #4 While an approval round is unanswered, other conversations' owed wakes are either delivered or the blocking is documented as intended — the current behaviour (all app-wide wakes silently held behind one unanswerable card) is decided one way or the other, not left implicit
- [x] #5 `Docs/User_Guide/console/agent-runs-and-tools.md` describes what actually ships
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Baseline gate on untouched branch (READ counts)
2. Probe the harness ordering: full production e2e (nav-away), measure the painted frame + display chain of the approval card at first open
3. RED: new suite driving both headless paths through real navigation, asserting on the PAINTED card (compositor text) and the display chain; deterministic delivery of the card's deferred initial-hide after the mount sync (the live slow-tty ordering)
4. Fix: make the approval card's initial hide construction-time (no deferred call_after_refresh clobber); same for the task-cards container
5. Mutation-test the fix; re-run gate; live re-verify the close-out scenario in tmux with a scratch profile
6. Docs: User_Guide agent-runs-and-tools.md + serialization statement (AC#4 decided as documented-by-design); report
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Mechanism (proven by state reachability, then reproduced): the card's initial batch-body hide was DEFERRED mount work (on_mount -> call_after_refresh(_hide_batch_body)); a real terminal's slow first Console paint delivers it AFTER the screen's one-shot 0.05s mount sync has rendered the headless round, unrendering it -- the only writer sequence that can produce the observed card-visible/body-hidden state. Not a third payload overwrite: the state funnel was correct; the clobber was at the widget. run_test paints in microseconds (hide lands first), which is why the merged e2e never saw it -- and its .approval-row/renderable assertions see through display:none anyway.

Fix: every initial hide is construction state (ChatApprovalCard.__init__, compose-built batch body, ChatTaskCards.__init__); both on_mount handlers and _hide_batch_body deleted; set_batch made all-or-nothing (containers resolved before any state mutation). The observed defect state is unreachable by construction.

Red: Tests/UI/test_console_approval_first_open_render.py -- 5 of 6 red at ee6c3d709 through the real navigation on BOTH headless paths, asserting on the PAINTED frame (compositor text) and answering through the rendered control; live ordering made deterministic by capturing the card's after-refresh work and delivering it post-sync. Probe file pins that the harness's natural ordering is favourable. 5 mutations run, all KILLED (incl. re-adding the deferred hide and reverting each construction hide); Edit-based restores verified git-clean.

AC#4 decided: the app-wide hold behind an unanswered approval is the close-out's serialization invariant working as designed; now answerable (card renders) and documented plainly in the User Guide for a later owner ruling. Serialization unchanged.

Gate (read counts): baseline 1584 passed / final 1591 passed (delta = exactly the 7 new tests), 0 failed both sides; +130 widget-adjacent; doc contract 69. Live re-verified in tmux (scratch profile, real Anthropic model): toast on Library, first Console open painted the FULL card (write_file (high risk), args, all controls) with no session switch, stable ~8 min; round ended via documented quit-denies; no file written. Report: Docs/superpowers/plans/2026-08-17-task-17500-report.md. Residue named: one-shot mount sync (lost-sync cousin, no red), sibling cards' on_mount hides, task-15661.
<!-- SECTION:NOTES:END -->
