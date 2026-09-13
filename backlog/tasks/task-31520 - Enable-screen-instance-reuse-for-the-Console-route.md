---
id: TASK-31520
title: Enable screen-instance reuse for the Console route
status: Done
assignee: []
created_date: '2026-09-04 21:30'
labels:
  - performance
  - console
dependencies:
  - task-24452
priority: high
---

## Description (the why)

TASK-24452 landed opt-in screen-instance reuse (`ScreenRoute.reusable`:
construct once, install, suspend instead of unmount) and proved it on Home.
The measured headroom for Console is the largest in the app: a warm
installed-instance switch to ChatScreen cost 158 ms CPU against 750-820 ms
for today's fresh construction (-80%, interleaved arms, 2026-09-04), and
Console's ~559-widget re-mint per visit disappears. Enablement is gated on
a lifecycle audit: `ChatScreen.on_unmount` tears down ~a dozen subsystems
per visit (sidebar-state flush, terminal workspace detach, auto-speak,
image-edit cancels, transcript-sync/fleet-survivor/cost-TTL timers,
roleplay persistence, hands-free, realtime, dictation), and with reuse
those must become suspend/resume-aware instead -- otherwise they keep
ticking on a hidden screen. TASK-1143 F5's leave-confirmation ("navigating
away cancels runs") also changes meaning when leaving no longer unmounts.

## Acceptance Criteria (the what)

- [x] Every `ChatScreen.on_unmount` teardown step is dispositioned for reuse (19-step audit; dispositions encoded in `on_screen_suspend`'s docstring, including what deliberately does NOT run there and why)
- [x] No Console-owned timer fires while the Console is suspended (guard test arms the intervals, navigates away through the real seam, and asserts all timer handles are stopped -- mutation-tested red)
- [x] Repeat visits to the Console resume the same instance and construct materially fewer widgets (guard tests; plus the initial-push retain fix means even the FIRST return visit is warm)
- [x] Console switch CPU improves in an interleaved A/B against the fresh-instance baseline (arrival CPU 726 -> 212 ms, -71%; wall -42%; MBBM on a quiet machine, 10 visits/arm)
- [x] Console behaviour across a switch-away-and-back cycle: active runs/approvals now SURVIVE (the deliberate contract change, guard-tested); dictation/realtime/hands-free abandon on leave exactly as before; known accepted delta: an open Terminal workspace view survives the round trip
- [x] The TASK-1143 F5 leave-confirmation copy/behavior is reconciled: `confirm_navigation` is a pure always-allow (nothing is lost), `confirm_quit` unchanged (app exit still cancels); the three dialog-contract tests rewritten to the reuse-era journeys

## Implementation Plan (the how)

Audit completed 2026-09-04 (19-step on_unmount disposition + Textual
mechanics). The crux (item 19): `leave_console_runtime` must NOT run at
suspend -- runs/approvals survive navigation by design, and a naive
suspend wiring permanently kills the prompt queue (`begin_visit` never
re-fires for a reused screen; the runtime ref is memoized and
restore_state is skipped on reuse). Also hands-off: the H3 image-edit
screen pointer (clearing strands completions), `_drain_pending_console_
videos` (its closed-flag has NO reset path -- wiring it to suspend
permanently disables video gen), roleplay-persistence abandon (keep
running is strictly better), the hands-free store tap (idempotent
install), the previews cache.

1. `on_screen_suspend` (new): release the conversation-settings-return
   claim (claim side already resume-aware); stop+flush the sidebar-state
   debounce; stop TTS into the hidden screen
   (`invalidate_console_speech_context`); `_console_auto_speak.unmount()`;
   stop the four timers (transcript-sync 0.2s, fleet-survivor 1s,
   cost-TTL 10s, draft-spend one-shot); abandon hands-free/realtime/
   dictation (preserves today's mic/privacy behavior; terminal, no
   resume half).
2. `on_screen_resume` additions: `_console_auto_speak.mount()`
   (idempotency-guarded); restart the transcript-sync timer when the
   viewed session's run is active (new logic); the fleet survivor-tick
   hedge (`_maybe_start_console_fleet_survivor_tick`, exists, idempotent).
3. TASK-1143 reconciliation: `confirm_navigation` becomes always-allow
   for tab switches (nothing is cancelled any more); `confirm_quit`
   copy/behavior unchanged (app exit still cancels via
   `dispose_console_runtime`). Update the stale comment at the app.py
   confirm seam.
4. Flip the chat route `reusable=True`.
5. Guards: console reuse test file -- same-instance resume, timer
   quiescence while suspended, active-run-survives-navigation,
   resume restarts the sync timer for an active run; mutation-test the
   suspend stops and the resume restart.
6. Interleaved A/B vs merge-base; paired-arm the Console UI suites.

Known accepted behavior deltas (PR body): leaving Console no longer
cancels in-flight user turns / denies parked approvals (they resume
visibly on return); the Terminal workspace view survives a tab
round-trip; H3 image edits keep running across navigation.

## Implementation Notes

Flipped `reusable=True` on the chat route after executing the 19-step
audit. `on_screen_suspend` (new): releases the conversation-settings
handoff claim, flushes the sidebar debounce (task-held async), stops TTS
into the hidden screen, unmounts auto-speak, stops the four per-visit
timers, abandons hands-free/realtime/dictation (preserving today's
mic/privacy behavior on leave). Its docstring records what deliberately
does NOT run there and why each would break its feature -- the crux being
`leave_console_runtime`: the runtime view stays ATTACHED across suspend,
so runs, queued prompts, and parked approvals survive navigation
(mutation-tested: a suspend-time detach fails the guard). `on_screen_
resume` re-arms auto-speak, restarts the transcript-sync poll when a run
is still in flight (new logic -- no other path restarts it;
mutation-tested), and fires the survivor-tick hedge.

`confirm_navigation` is now a pure always-allow: the busy-fleet dialog
warned about a loss that no longer happens; `confirm_quit` still
delegates (app exit really cancels). The three tests pinning the dialog
were rewritten to reuse-era journeys through the real navigation seam:
lossless busy-fleet navigation with runtime attachment, a navigation
storm never raising a dialog, and the gate being side-effect-free with
the queue manager open.

Bonus fix with its own win: `_push_initial_screen` never RETAINED a
reusable initial screen, so the default-tab Console paid one full
re-mint on the first return; it now retains like a navigated-to screen
-- even the first return visit is warm (first arrival measured 171 ms
CPU, same as steady state).

Measured (interleaved MBBM, quiet machine, 10 visits/arm): Console
arrival CPU 726 -> 212 ms (-71%), wall 1,280 -> 745 ms (-42%); a warm
Chat+Library round trip 1,210 -> 652 ms CPU (-46%).

Verified: reuse suites 15 passed (runtime-detach and resume-restart
mutations red); lifecycle-adjacent selection (seam guard, parallel runs,
residency, prompt queue, store continuity) 82 passed with the one red
pre-existing on pristine dev (workspace-test seam violations);
test_console_native_chat_flow spot-check; ruff clean on touched files.

Files: `UI/Screens/chat_screen.py`, `UI/Navigation/screen_registry.py`,
`app.py`, `Tests/UI/test_console_screen_reuse.py` (new),
`Tests/UI/test_console_parallel_runs.py`,
`Tests/UI/test_console_prompt_queue.py`, `Tests/UI/test_screen_reuse.py`.
