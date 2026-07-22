---
id: TASK-351
title: Reduce or mask slow first feedback after primary Console actions
status: Done
assignee: []
created_date: '2026-07-20 14:21'
updated_date: '2026-07-22 02:20'
labels:
  - console
  - ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Cluster of measured instances: (a) Enter-to-send: 350ms after Enter the composer still held the full text and the transcript said 'No messages yet' (j4-03); in another session the first transcript change after Enter took 7.3s (echo of the user's own message), though a later send echoed in 1.0s; (b) rail conversation clicks: in one session clicking 'No tools: explain...' produced no visible change within 1.2s and clicking 'Write a detailed...' never opened it at all during ~10s of subsequent interaction (transcript stayed on the empty 'Chat 1'), while in the next session the same click opened in 0.6s - silent intermittent failure with no pressed/loading feedback; (c) clicking 'Inspector' during a run showed no response within 0.7s (j4-32 still collapsed) and the panel was simply found open in a later frame.

**Repro:** (a) Type a prompt, press Enter, watch composer/transcript in the first second; (b) right after app start, click a saved conversation in the rail Chats list and wait - sometimes nothing happens; (c) click 'Inspector' during a run.

**Verifier note:** Not covered by the perf ledger: task-280/259 fixed tick/DB-on-loop and transcript derivation, and no item covers first-send echo latency, silently-failing rail conversation clicks, or delayed Inspector toggle. The 7.3s worst-case echo and dead rail clicks are intermittent single-session measurements (hence medium confidence) but three independent sub-symptoms point at on-loop work in the send/resume paths (e.g. first-send agent-bridge/MCP catalog init). No pressed/loading acknowledgment on rail conversation rows is verifiable by design (plain Buttons, no busy state). P2 appropriate.

**Source:** Console UX expert review 2026-07-20 (finding j4-first-feedback-latency-cluster; P2, verdict NEW, medium confidence) — full report `Docs/superpowers/qa/console-ux-expert-review-2026-07-17/README.md`, app under review at origin/dev cad9e271d, J4 streaming journey. Evidence: `j4-03-sent-immediate.png`, `j4-18-first-change.png`, `j4-26-partial-after-restart.png`, `j4-32-inspector-midrun.png`, `j4-33-streaming3.png` (regression/P1 captures committed alongside the report; the full 339-capture set is on the review machine).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A warm send echoes the user's message in the transcript the instant the submit is accepted (when the composer clears), rather than waiting for the next 0.2s transcript poll — closing the "composer cleared but transcript still says 'No messages yet'" gap
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Measured, not assumed.** With llama.cpp live, `resolve_for_send` is 24ms cold
/ <1ms warm — the review's 7.3s echo is a cold-server/first-connection artifact,
NOT reproducible warm. Server-side instrumentation of a warm cold-start first
send showed the whole controller path finishes by ~56–64ms (user row in the
store at ~37ms, composer clears at ~51ms, MCP compose only ~5ms), yet the
transcript echo landed at ~577–674ms. The gap is entirely UI-side: the native
transcript only repaints on a **0.2s poll** (`_start_console_transcript_sync_
timer` → `_poll_transcript`), and the first poll is heavy (`refresh_messages`
first render ~178ms + the rest of `_sync_native_console_chat_ui` ~166ms). So the
composer cleared at ~50ms while the transcript still read "No messages yet" for
~600ms — reading as "not sent".

**Fix:** `_on_console_submission_accepted` (the seam that already clears the
composer at acceptance, and only fires once submit_draft confirms the turn
proceeds) now also kicks an immediate `_sync_native_console_chat_ui()` via
`run_worker(exclusive=True, group="console-sync")` — the same call the poll's own
self-requeue uses, so it coalesces against a running poll through the existing
`_console_sync_in_progress` guard. The echo no longer waits for the poll phase.

Live result: cold-send echo ~600ms → ~450–470ms in the textual-serve harness
(the residual is `refresh_messages` first-render + the shared sync ordering, both
out of scope). More importantly the echo is now decoupled from poll/stream
timing.

**Rejected approach:** a leaner "transcript-first" echo (refresh only the
transcript surface, guarded by a new in-progress flag) reached ~230ms but the
skip-and-drop guard could permanently suppress the post-stop "[stopped]"
refresh — it broke three stop/stream tests. The shipped full-sync kick reuses
the proven skip-and-**requeue** coalescing (`_console_sync_requested`), so a
needed refresh is never lost.

**Verified:** new RED→GREEN regression test `test_console_send_echoes_user_
message_before_transcript_poll` (disables the poll so only the acceptance echo
can surface the message); full `test_console_native_chat_flow.py` suite (188)
green. Files: `tldw_chatbook/UI/Screens/chat_screen.py`,
`Tests/UI/test_console_native_chat_flow.py`.
<!-- SECTION:NOTES:END -->

## Scope note

This task delivers sub-symptom (a) for the **warm** send path (provider ready).
The remaining pieces of the original finding move to **task-457**: the
**cold**-provider first-send echo (needs an optimistic user-append *before* the
readiness probe, a real blocked-send behaviour change), the rail-conversation
pressed/loading state (sub-symptom b), and the Inspector-during-run
acknowledgment (sub-symptom c).
