---
id: TASK-19702
title: 'Default workspace never sees change review or git modes'
status: To Do
assignee: []
created_date: '2026-08-21'
labels:
  - console
  - change-review
  - workspaces
dependencies:
  - TASK-16801
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Conversations in the Default workspace never see the Change Review screen's
contents, and therefore never see TASK-16801's `Working tree (current)` entry
either — even though the `[change_review] git_actions` switch defaults ON.

The chain, traced during TASK-16801's whole-branch review:
`Workspaces/registry_service.py` refuses runtime bindings for the Default
workspace at the service layer, so `folder_binding_roots` is always empty; the
Console agent bridge only records change snapshots when it has roots
(`and change_roots`), so no snapshots exist; and the review screen derives its
candidate roots from snapshot rows plus the roots the opener passes. With
neither source populated, the screen is empty and git modes are invisible.

This predates TASK-16801 — the Review screen is already empty in the Default
workspace on dev — so it was ruled ship-with-follow-up rather than a merge
blocker. The tempting shortcut was explicitly rejected during that review:
falling back to `[console] workspace_root` or the process CWD would offer
confirmed commit and push against a repository that was never bound to the
workspace, which is worse than showing nothing.

There is a related gap worth resolving in the same breath: `[console]
workspace_root` grants local-tool WRITE access to a directory that neither
change tracking nor git modes can see, so an agent can edit files that the
review surface will never show the user.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 A Default-workspace conversation either sees its changes in Change Review, or the screen explains why it cannot, rather than rendering an empty state that reads as "nothing changed"
- [ ] #2 Any root that becomes reviewable is a root the workspace actually binds — no fallback to CWD or to a bare config key that bypasses workspace binding
- [ ] #3 The `[console] workspace_root` write/visibility mismatch is resolved or explicitly documented: a directory an agent can write must not be invisible to change review without the user being told
- [ ] #4 Tests cover a Default-workspace conversation end to end, asserting the chosen behaviour rather than the current silent emptiness
<!-- AC:END -->
