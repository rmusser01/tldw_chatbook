---
id: TASK-337
title: Make Console Stop acknowledge immediately and reliably cancel generation
status: Done
assignee:
  - '@claude'
created_date: '2026-07-20 14:21'
updated_date: '2026-07-21 07:38'
labels:
  - console
  - ux
  - regression
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Two Stop interruptions, two different behaviors, zero feedback in both. Run A (j4-23/24): clicking Stop froze the transcript mid-word at 'A [streaming]'; 3s later the message still carried [streaming] and the Stop button was still active - the UI looked stuck-running with no confirmation; after an app restart the persisted message contained two additional paragraphs that were never displayed, i.e. generation continued after the UI froze and the user's view diverged from what was saved. Run B (j4-36): Stop cleared [streaming] and reverted the button within ~2.5s - but again with no 'stopped' toast, no event line in the 'Transcript / Event Stream', and no state change anywhere else.

**Repro:** Send a long prompt -> when tokens start streaming, click Stop in the composer bar -> observe no acknowledgment; in one of two trials the message stayed '[streaming]' with the Stop button active, and reloading the conversation later showed extra content generated after the freeze.

**Verifier note:** Run A (frozen '[streaming]', Stop still active, generation continued in background, persisted content diverged from display) is the exact race family task-227 (Done 2026-07-16, in this worktree) claims fixed: 'a stop that lands during an in-flight agent bridge thread always persists the run as cancelled'. Either an unfixed window remains or a later change regressed it — one occurrence in two trials, races are nondeterministic, hence medium confidence. The zero-acknowledgment half is real and previously unexercised (ledger gap-not-exercised-2026-07 lists Stop mid-stream as never verified): _stop_console_generation_from_visible_action (chat_screen.py:9785-9793) emits no success toast/event row, only a warning when nothing is running. P1 stands for the freeze+divergent-persistence behavior.

**Source:** Console UX expert review 2026-07-20 (finding j4-stop-feedback-unreliable; P1, verdict REGRESSION, medium confidence) — full report `Docs/superpowers/qa/console-ux-expert-review-2026-07-17/README.md`, app under review at origin/dev cad9e271d, J4 streaming journey. Evidence: `j4-23-just-after-stop.png`, `j4-24-post-stop-settled.png`, `j4-28-click-partial-message.png`, `j4-36-just-after-stop.png` (regression/P1 captures committed alongside the report; the full 339-capture set is on the review machine).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Stop visibly acknowledges immediately (button state change + 'Stopping...')
- [x] #2 Stop reliably cancels the provider request
- [x] #3 An explicit 'stopped by user' record appears in the transcript/event stream, and persisted content matches what was displayed
- [x] #4 A regression test pins the restored behavior
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Recon the stop path: handle_console_stop_generation -> controller stop -> gateway/agent-bridge cancellation; identify the freeze window the review hit (one of two trials)
2. AC1 immediate acknowledgment: Stop button flips to a Stopping state synchronously at the click
3. AC2/AC3: reliable cancel + explicit stopped-by-user transcript record; persisted content == displayed
4. TDD with a gateway that ignores/delays cancellation; live-verify against llama
<!-- SECTION:PLAN:END -->

## Implementation Notes

AC1: the Stop click acknowledges synchronously — the button flips to
"Stopping…"/disabled before the controller is even asked to stop, and the
label is restored after the sync (the composer bar's ``sync_action_state``
governs visibility/variant but never the label, so without the restore a
later run's Stop button would read Stopping…).

AC3: ``stop_active_run`` appends an explicit "Response stopped by user."
SYSTEM transcript row (durable, unlike the transient run-state chip copy);
``shutdown()`` passes ``record_user_stop=False`` — teardown is not a user
action.

AC2 (reliable cancel / persisted == displayed): pinned with a
chunk-then-park bridge-thread test (task-227 rig): stop mid-stream, release
the surviving thread, assert the stopped message never grows past the stop
point. This passes on current code — the store's terminal-status guard
already holds the line — and now cannot regress silently. The review's
run-A display freeze could not be reproduced on current dev: the #726 sync-
timer work guarantees a post-stop render (one-last-tick), and a
deterministic coalesced-sync test pins that guarantee with the bridge
thread still parked (the exact live shape where the submit-side sync
cannot run for minutes).

Verified live against llama.cpp: Stop mid-stream renders "[stopped]" +
"System  Response stopped by user." with the Stop button gone and status
Ready. Files: `Chat/console_chat_controller.py`,
`UI/Screens/chat_screen.py`.
