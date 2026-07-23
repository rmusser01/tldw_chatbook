---
id: TASK-344
title: >-
  Stop the Console rail from asynchronously swapping its section stack after
  switches and sends
status: Done
assignee:
  - '@claude'
created_date: '2026-07-20 14:21'
updated_date: '2026-07-21 20:12'
labels:
  - console
  - ux
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Buffer dumps show the rail as the browse list at 13:19:30 (j2-10) and as the context stack at 13:19:42 (j2-12) with zero rail interaction in between; the same swap followed the earlier rail-click switch, and end states differ between similar action sequences (after one switch the list stayed, after another it was replaced). The swap displaced a click in practice: a click aimed at a listed conversation row landed on the Context section because the stack flipped between reading and clicking (my scripted C5 click failed exactly the way a human's would, and instead a stray click starred a different row unnoticed). While in context mode the conversation list is simply gone from the rail with no visible way back other than waiting or Ctrl+K.

**Repro:** Switch to a saved conversation (rail click or Ctrl+K) and/or send a message; watch the rail for the next ~15s without touching it: Starred/Workspaces/Chats are replaced by Context/Model/Agent/Details (or vice versa) at an unpredictable moment.

**Verifier note:** Phenomenon confirmed by screenshots: j2-10 shows the Session section with the full conversation browser (search row + Starred/Workspaces/Chats); j2-12, 12s later mid-run with zero rail interaction, shows the browser gone (summary-only legacy render, Context/Model/Agent/Details now at the click positions the list occupied). Every sync path routes through _build_console_workspace_context_state which always attaches conversation_browser (chat_screen.py:4702-4738), so this async restructure is unintended, not a designed mode. Not covered by tick-ttl-2s-gating (that blesses content lag, not structural swap) nor task-149 (scroll survival). Displaced a click in practice (accidental star). Mechanism not fully pinned, hence medium confidence on root cause, high on the defect itself.

**Source:** Console UX expert review 2026-07-20 (finding j2-rail-unpredictable-mode-swap; P1, verdict NEW, medium confidence) — full report `Docs/superpowers/qa/console-ux-expert-review-2026-07-17/README.md`, app under review at origin/dev cad9e271d, J2 returning power user journey. Evidence: `j2-10-just-sent.png`, `j2-12-reply-later.png`, `j2-06-switcher-open.png`, `j2-44-dot-click.png`, `j2-57-mangle-trialB.png` (regression/P1 captures committed alongside the report; the full 339-capture set is on the review machine).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The rail should be a stable landmark: keep the conversation list persistently reachable (fixed sections or an explicit user-controlled toggle), and never restructure itself asynchronously after the user has stopped acting
<!-- AC:END -->

## Implementation Notes

Root cause (both findings, one bug): `_sync_native_console_chat_ui` calls
`_sync_console_workspace_context()` on every 0.2s run tick, and
`ConsoleWorkspaceContextTray.sync_state` does `refresh(recompose=True)`
UNCONDITIONALLY — so during a streaming run the conversation browser tore
down and rebuilt ~5x/second, visibly vanishing (344's async swap / 349's
list disappearing) and displacing clicks.

A widget-level equality guard is forbidden (TASK-251 pinned
`test_console_workspace_context_tray_sync_state_always_recomposes`): the
unconditional recompose ALSO self-heals a real DOM/state desync — a
full-screen recompose sets the fresh tray's `.state` but its rows can be
superseded before they settle, so `.state` says X while the DOM shows
nothing, and the next tick's recompose repaints it. Instrumentation
confirmed this live (a search-selection tick showed `changed=False` yet
`dom_rows=0`), which is exactly why a plain equality guard reintroduces the
"row not found" failures.

Fix: a SCREEN-side guard scoped to ACTIVE RUNS only. During a run tick
(`_console_transcript_sync_timer is not None`) with unchanged workspace
state, skip the tray push — no search/resume happens mid-run, so the
self-heal isn't needed. Every non-run sync keeps the original
always-recompose, so search/resume click-targeting is untouched.

Verified: new `test_console_workspace_context_tray_not_recomposed_when_state_unchanged`
(RED first), the full tick-gating + workspace-context-rail suites, and the
50 search/resume/browser flow tests all pass. File:
`UI/Screens/chat_screen.py` (`_sync_console_workspace_context`).
