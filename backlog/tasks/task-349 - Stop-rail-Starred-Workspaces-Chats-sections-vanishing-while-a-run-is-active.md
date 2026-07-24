---
id: TASK-349
title: Stop rail Starred-Workspaces-Chats sections vanishing while a run is active
status: Done
assignee:
  - '@claude'
created_date: '2026-07-20 14:21'
updated_date: '2026-07-21 20:12'
labels:
  - console
  - ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
During generation the rail drops its whole conversation area: j4-33 (mid-stream) shows only Session/Context/Model/Agent/Details with the Starred/Workspaces/Chats lists gone; j4-04d shows the same during the thinking gap. Immediately after the run ends the sections return (j4-36). A user glancing at the rail mid-run sees their conversation list apparently deleted, and the whole rail layout jumps twice per send.

**Repro:** Note the rail's Chats list -> send any prompt -> while the run is active, look at the rail: Starred/Workspaces/Chats are gone -> when the run finishes they return.

**Verifier note:** Confirmed real and consistent: j4-04b/04d/33 (Session section reduced to a bare 'Workspace Default' row — Switch, scope, search, Starred/Workspaces/Chats all absent) vs j4-36 (restored while Stop button still visible, i.e. immediately at run end). Not covered: tick-ttl-2s-gating covers ≤2.8s refresh LAG, not wholesale disappearance; rail-layout-quiet-focus/rail-conversations-bounded say nothing about runs. The state builder always attaches the browser (_with_console_conversation_browser_state, chat_screen.py:4702+), so suspect the per-0.2s-tick exclusive console-sync/legacy-alias worker kicks (chat_screen.py:5438-5444) repeatedly cancelling the tray rebuild while ticks run, or run-time recompose height clipping. P2 confirmed severity — twice-per-send layout jump plus apparent data loss.

**Source:** Console UX expert review 2026-07-20 (finding j4-rail-sections-vanish-during-run; P2, verdict NEW, high confidence) — full report `Docs/superpowers/qa/console-ux-expert-review-2026-07-17/README.md`, app under review at origin/dev cad9e271d, J4 streaming journey. Evidence: `j4-33-streaming3.png`, `j4-04d-gap-40s.png`, `j4-36-just-after-stop.png`, `j4-04b-gap-5s.png` (regression/P1 captures committed alongside the report; the full 339-capture set is on the review machine).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The rail should keep its content stable during runs (at most disable interactions), never removing the conversation list wholesale
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
