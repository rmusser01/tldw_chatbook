---
id: TASK-32388
title: >-
  Library Conversations: after a workspace hop, an Undo receipt names the wrong
  workspace
status: Done
assignee:
  - '@codex'
created_date: '2026-09-11 10:30'
updated_date: '2026-09-16 21:44'
labels:
  - library
  - conversations
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
task-32107 made "Use as source" link a conversation into the active workspace in one undoable step, with a receipt and an "Undo link" beside it, and made Undo remove the membership the receipt names rather than whichever workspace happens to be active. A residual remains: switch to a second workspace that also holds that membership and the receipt on screen still names the first one, so Undo correctly removes the membership the receipt names while the user reads it as acting on the workspace they are now in. Recorded in task-32107's own notes at close.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Switching the active workspace stands the stale link receipt down, or restates it for the workspace now active
- [x] #2 Undo never removes a membership the receipt on screen does not name
- [x] #3 The workspace-hop case is covered by a test alongside the existing link/undo pins
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/005-console-workspace-server-readiness.md; backlog/decisions/150-design-token-system-and-design-language.md; backlog/decisions/161-component-pattern-library.md
Reason: repair the existing ephemeral link receipt and exact-target Undo contract without changing workspace ownership, persistence or application structure.

1. Trace receipt creation, compose/retained sync and Undo against real registry membership; reproduce the same-conversation workspace hop, using distinct workspaces accepted by registry validation.
2. Add focused mounted regressions beside existing link/Undo pins. Project receipt visibility from its workspace ID and current active workspace at both render seams; use the same eligibility at Undo admission while retaining the original exact ID target.
3. Verify stale/hidden Undo has no membership effect and returning to the original workspace can undo its named link. Cover retained sync and recomposition.
4. Run targeted neighboring tests/static checks and a real TldwCli private-profile journey at compact/wide sizes in both themes; verify exact membership/message persistence and normal exit.
5. Obtain independent review, record bounded evidence and update task status before a local commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Workspace link receipts and Undo now project only into the active workspace they name, using a shared ID/eligibility check in initial reader composition, retained synchronization and Undo admission. Returning with the same conversation loaded restores the original receipt; stale/hidden/delayed Undo cannot remove a membership after a workspace change or stale list. The unlink still targets the retained exact workspace ID.

Changed LibraryScreen, its conversation-reader controller, mounted link/Undo regressions and the Library user guide. QA runner and evidence: Docs/superpowers/qa/2026-09-16-conversation-link-receipt/README.md.

Verification: 144 distinct targeted tests passed (112 neighboring checks plus final 59 with 27 overlapping). Five focused cases failed against baseline production code; removing the eligibility guard made the added stale-list case wrongly unlink Alpha, then restoration passed. Four real-app private-profile journeys passed at 170x48 and 80x24 in both themes; eight captures were inspected. Exact final membership was Beta only, with the one conversation/message unchanged, ten healthy databases and three unchanged default-profile config hashes. Normal app return, shell exit 0 and PID absence preceded cleanup of only the owned terminal.

Independent review and final follow-up found no actionable regressions. Native runner lint/format and changed test-function formatting pass; comparison of three existing Python files shows no new findings against 212 inherited lint findings and inherited whole-file formatting debt. Design/component governance, generated bundle freshness and diff whitespace checks pass. No full suite or remote provider journey ran.

Plan deviation: the proposed duplicate display-name fixture was invalid because registry validation requires unique workspace names; removed it and excluded its failures from red evidence. Workspace hops in native qualification use the real registry API while Library remains mounted, so Console workspace-switcher navigation is not qualified. The final typed registry-error catch was verified by the final mounted unavailable-context case after the native successful-path run.

ADR required: no. Existing ADR-005 (backlog/decisions/005-console-workspace-server-readiness.md), ADR-150 (backlog/decisions/150-design-token-system-and-design-language.md) and ADR-161 (backlog/decisions/161-component-pattern-library.md) govern this repair. No new lesson or architecture decision was needed. TASK-32700 Import remains separately blocked by host semaphore allocation; its criteria are unchanged.

2026-09-16 follow-up: TASK-32700 subsequently completed real local Import success and restart recovery after the semaphore gate cleared. The blocked statement above is historical; current evidence is Docs/superpowers/qa/2026-09-16-ingest-lifecycle/resume/README.md.
<!-- SECTION:NOTES:END -->
