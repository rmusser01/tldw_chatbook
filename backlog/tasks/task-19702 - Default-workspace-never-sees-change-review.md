---
id: TASK-19702
title: 'Default workspace never sees change review or git modes'
status: Done
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
- [x] #1 A Default-workspace conversation either sees its changes in Change Review, or the screen explains why it cannot, rather than rendering an empty state that reads as "nothing changed"
- [x] #2 Any root that becomes reviewable is a root the workspace actually binds — no fallback to CWD or to a bare config key that bypasses workspace binding
- [x] #3 The `[console] workspace_root` write/visibility mismatch is resolved or explicitly documented: a directory an agent can write must not be invisible to change review without the user being told
- [x] #4 Tests cover a Default-workspace conversation end to end, asserting the chosen behaviour rather than the current silent emptiness
<!-- AC:END -->

## Implementation Notes

Chose disclosure over fabrication. The screen now distinguishes the two
things an empty history can mean, and names the writable-but-untracked
directory when one exists.

**Premise verified before building** (the task was traced during a review,
so I re-checked it against the real registry rather than trusting the
write-up). `add_folder_binding(DEFAULT_WORKSPACE_ID, ...)` raises
`WorkspaceRegistryServiceError: Default workspace does not allow runtime
bindings.` — folder bindings delegate to `save_runtime_binding`, which
refuses Default — while the same call on a non-Default workspace succeeds.
So a Default-workspace conversation genuinely can never have a tracked root.

**AC #1 — the empty state now states its cause.** "No file changes recorded
for this conversation." is a claim the app can only support when the
conversation HAS tracked roots; with none, nothing was ever watched. With no
roots the copy now says no folder is bound, says explicitly that this is not
a report that nothing changed, and points at Settings ▸ Workspaces plus the
Default-workspace limitation. With roots present the original copy is
unchanged, because there it is true. This is the same honesty rule the
working-tree pane already follows by saying "unavailable" rather than
"clean" when every root fails (TASK-16801).

**AC #2 — no fabricated root.** Deliberately does NOT fall back to
`[console] workspace_root` or the process CWD to manufacture something
reviewable. That was rejected in TASK-16801's whole-branch review and is
worth restating: it would offer confirmed commit and push against a
repository the workspace never bound.

**AC #3 — the write/visibility mismatch is disclosed.** Confirmed in
`Chat/console_chat_controller.py`: the agent's file-tool confinement root
comes from `tool_configuration["workspace_root"]` or `[console]
workspace_root`, **falling back to `os.getcwd()`**, while change tracking
follows bound folders. Nothing keeps them in agreement. When a configured
root is not covered by a tracked root, the banner now names it and says
changes made there will not appear in this review.

**Not changed:** the Default workspace's inability to bind folders. That is
a deliberate registry rule with its own rationale; this task's remit was
that the user must not be left with a silent gap, not that the rule should
be relaxed.

**Files:** `tldw_chatbook/UI/Screens/change_review_screen.py`,
`Tests/UI/test_change_review_current_mode.py` (one pre-existing assertion
updated: `test_pseudo_entry_absent_without_candidate_roots` asserted the old
copy in exactly this scenario; its subject — no pseudo-entry without
candidate roots — is unchanged).
