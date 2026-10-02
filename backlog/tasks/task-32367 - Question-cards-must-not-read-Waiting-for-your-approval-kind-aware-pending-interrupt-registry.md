---
id: TASK-32367
title: >-
  Question cards must not read Waiting for your approval (kind-aware
  pending-interrupt registry)
status: Done
assignee:
  - '@codex'
created_date: '2026-09-11 01:55'
updated_date: '2026-10-02 00:52'
labels:
  - console
  - approvals
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`has_pending_approval_round` covers all five interrupt kinds (including the question card) while the inspector counts mounted approval cards only, so the transcript activity line can say "Waiting for your approval" while a question card is up. Lane B's minimum fix (R15) corrected the false invariant comment in chat_screen.py; the registry itself is still kind-blind. Found by lane B's final review (task-32345 area).

Source: approval-card / MCP-permissions fix wave 2026-09-10/11 (plan `Docs/superpowers/plans/2026-09-10-approval-card-fix-wave.md`, review snapshot `.impeccable/critique/2026-09-10T17-31-53Z__hatbook-widgets-chat-widgets-chat-approval-card-py.md`); rider recorded in the lane ledger, not fixed in the wave.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A pending question card never produces the "Waiting for your approval" activity line; it produces copy that names a question
- [x] #2 The pending-interrupt registry exposes the interrupt kind to the activity classifier and the inspector count agrees with it
- [x] #3 A test pins one question card + one approval card → the line names the approval, and a lone question card → the question copy
- [x] #4 Keyboard review focuses the visible pending decision card when a tool approval is queued behind another confirmation, while retaining visible approval priority.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/094-console-turn-lifetime-and-navigation-boundary.md; backlog/decisions/067-indefinite-human-approval-waits.md; backlog/decisions/195-console-live-tool-call-presentation.md
Reason: Complete the existing kind-aware pending-decision projection into Inspector status/counts; no new owner or registry.
1. Mount actual ChatScreen with real question and approval worker rounds; reproduce Inspector mismatch for a lone question and multiple approval rounds while retaining existing activity precedence.
2. Reuse pending kind registry/shared copy. Add one locked kind-count accessor; derive active-session Inspector approval count and Live work copy from it, retaining compatibility fallback for missing legacy controller seams.
3. Cover lone question, question plus approval, two queued approvals, resolution, sibling session and detach/remount with production-shaped round ownership and visible cards.
4. Run targeted Inspector/activity/pending-lifetime tests and native Console journey; update user guide and QA.
5. Self-review integrated changes, resolve PR feedback and require final-head checks before normal merge.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Inspector now derives pending copy and tool-approval counts from the existing session-owned kind registry through a locked kind-count accessor. Queued approval rounds count; questions and skill/worktree confirmations do not. Existing broad interrupt predicate and shared activity precedence are preserved. Eighteen focused checks cover lone question, mixed precedence, queued count2→1, sibling isolation, navigation and fresh remount using actual workers/cards/bridge snapshots. Console guide updated. Existing ADR-067/094/195 apply; no new ADR. Independent review clear; evidence and fixture limits are in Docs/superpowers/qa/2026-10-01-console-tool-ux-followups.md.

Qodo queued-decision finding fixed in the shared review router: scan painted cards in approval-first order independently of queued-round counts. Three mounted cases verify Alt+A, Inspector and attention-tab entry points, with approval priority retained. Independent review is clear.

Final combined tree rebased onto dev31d4f9b764: 44 targeted Console/close/attribute/latency checks passed; one empty exemption set skipped as expected. Fresh derived-artifact preflight passes, with census122 and unchanged startup cap. Final native approval07 and Close08 hashes match production sources; private profiles unchanged. PR2953 retains final-head review/CI/normal-merge checkpoint.
<!-- SECTION:NOTES:END -->
