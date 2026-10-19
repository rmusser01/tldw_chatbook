---
id: TASK-35640
title: >-
  Console tab switch follow-ups: click record tied to its press, queued clicks,
  repaint request, workspace bind
status: In Progress
assignee:
  - '@claude'
created_date: '2026-10-10 19:05'
updated_date: '2026-10-19 19:45'
labels:
  - console
  - tabs-sessions
dependencies: []
references:
  - TASK-33622.7
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-up to TASK-33622.7 from its pre-merge re-review. Four gaps remain in the tab-switch fix: a session-tab click's composer record sits in a controller-wide slot, so a click whose Pressed never arrives can carry later typing into the next activation of that tab by any route (Alt+N, Ctrl+K); two clicks queued behind a busy screen lose the second click's record when the first activation clears it; the repaint request made when an activation loses its claim after switching only takes effect if a sync pass is already running; and the workspace bind helper skips silently when it cannot reach the session controller, with three of its four paths untested.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A session-tab click's composer record is used only by the activation that its own press starts; an activation by any other route (Alt+1..9, Ctrl+K) never uses an earlier click's record
- [ ] #2 When two tab clicks are queued behind a busy screen, no key typed after them stays in the chat being left or is lost, each clicked tab keeps its own draft, and the keys keep their order
- [ ] #3 An activation that loses its claim after switching gets the new chat repainted whether or not a sync pass is running at that moment
- [ ] #4 Each workspace switch path that binds the composer (open, switch, resume, failed-open restore) is covered by a test that fails without the bind
- [ ] #5 The Console User Guide's tab-switch typing claim matches the behaviour
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. RED (mounted Pilot tests on the real ChatScreen + store, bootstrap_profile): a click whose Pressed never arrives, then typing in B and Alt+N/Ctrl+K activation of that tab -> typing must stay in B; two clicks queued behind a held activation, each followed by keys -> no key stays in A or is lost; an activation that loses its claim after switching with no pass running -> a sync pass runs; workspace switch/new-chat/resume/failed-open restore with a held sync pass -> composer bound to the new chat.
2. Tie the click record to its press: ConsoleSessionTabButton records the composer on the click and attaches the record to the Pressed that click posts; the session controller's tab-press handler takes it from that message. No controller-wide slot, so queued clicks carry their own records.
3. Repaint request: set _console_sync_requested when a pass is running, else start one (send_acknowledgement._resync's shape).
4. Workspace bind: a required injected collaborator called directly (doubles get it), tests for the untested paths, mutation-checked.
5. User Guide wording; related suites vs dev by test id; preflight; live check.
<!-- SECTION:PLAN:END -->
