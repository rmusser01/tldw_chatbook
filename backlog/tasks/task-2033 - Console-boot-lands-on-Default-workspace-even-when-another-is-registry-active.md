---
id: TASK-2033
title: Console boot lands on Default workspace even when another is registry-active
status: Done
assignee: []
created_date: '2026-08-03 00:45'
updated_date: '2026-08-29 05:45'
labels:
  - console
  - workspaces
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in TASK-1980 live UAT. Settings → Workspaces "Set active" marks a
workspace active in the registry (list shows "UAT Review (active)"), but
after an app restart the Console workspace switcher shows "Default
(everyday chats) (current)" — the Console context does not start on the
registry-active workspace, and the startup session is created tool-less
under Default. May be deliberate ("Switching changes Console context only")
but it surprises: the one thing "Set active" visibly promises is where you
land next. Owner decision wanted: either boot Console on the registry-active
workspace, or rename/re-copy the Settings affordance so it stops implying
that.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Decision recorded: Console boot honors registry-active workspace, or the Settings "Set active" copy/behavior is changed to match reality
- [x] #2 The chosen behavior is implemented and tested
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: This closes a fresh-session edge case in the accepted Settings ownership
and Console resume-time reconciliation contracts already recorded by ADR-028 and
TASK-18310; it introduces no new ownership or persistence boundary.

1. Add a failing reconciliation regression for an empty Console store with a
   named registry-active workspace, asserting both first-session ownership and
   store workspace context.
2. Change the existing resume-time reconciliation seam so a fresh store follows
   the same registry-active activation sequence as a divergent existing session,
   while preserving the explicit saved-conversation startup exception.
3. Add a mounted startup regression and run the focused workspace/controller and
   Console lifecycle tests that cover the changed seam.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Aligned an app-owned but sessionless Console store with the registry-active
  workspace before compose creates the first ordinary tab, and made resume-time
  reconciliation activate a registry workspace when no session exists.
- Preserved explicit saved-conversation precedence across compose and hydration,
  preventing a global conversation from inheriting named-workspace authority or
  leaving an orphan bootstrap tab. Stale active-session identities now repair
  before core synchronization can read them.
- Added unit and mounted regressions for ordinary named-workspace startup,
  explicit global resume, one-tab ownership, stale-session recovery, and a
  missing ID-only resume falling back to a usable registry-active tab.
- A fresh independent review found that failed startup resume could otherwise
  leave the Console sessionless. The ordered resume worker now releases resume
  authority, runs ordinary workspace reconciliation, and refreshes the native
  Console only when the opener fails and no prior session exists.
- Existing user-guide copy already documents that Settings **Set active** takes
  effect on the next Console visit, so no documentation wording changed.
- ADR check: no new ADR. ADR-028 and TASK-18310 already own the workspace/context
  and resume-time reconciliation contracts; this change closes their fresh-store
  edge case without changing persistence, authority, or ownership boundaries.
- Verification: 171 focused tests passed with one unrelated `origin/dev` fixture
  test deselected (`test_attach_and_detach_cover_exactly_the_same_slot_set`, whose
  bare `ChatScreen` lacks the pre-existing `_library_activity` fixture member),
  followed by 35 focused startup/resume/reconciliation tests after the review
  fix. Ruff checks and `git diff --check` passed. Independent follow-up review
  reported no Critical or Important findings and a **Ready to merge: Yes**
  verdict. Its only Minor note was addressed by extending the mounted fallback
  regression to assert the active session's rendered tab in addition to store
  and active-workspace state.
- Required-CI follow-up: reviewed the two production-diagnostic inventory
  deltas statement by statement before regeneration. This task adds one static
  debug event for failed saved-chat startup reconciliation; current `dev` also
  contributes one Library Skills warning containing only a fixed operation name
  and `type(exc).__name__`. Neither statement interpolates user content,
  secrets, paths, URLs, or exception messages, and sink topology is unchanged.
  Regenerated `Docs/security/production-diagnostic-inventory.json` records 541
  owner files, 1,270 TASK-492 calls, 7,401 TASK-494 calls, and 8 sink files.
<!-- SECTION:NOTES:END -->

## Decision

Fresh Console startup honors the registry-active workspace, so Settings keeps
its existing **Set active** wording and behavior. An explicit saved-conversation
resume remains authoritative over that startup default; a workspace-less/global
conversation resolves to the capability-safe built-in Default workspace. A
viewless Console runtime remains global until a Console view requests its store.
