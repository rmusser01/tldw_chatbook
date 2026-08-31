---
id: TASK-25712
title: Workspace and tree-chat action menus in the Console Context rail
status: Done
assignee:
  - '@Robert'
created_date: '2026-08-31 04:34'
updated_date: '2026-08-31 05:33'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Extend the TASK-23200/25709 asterisk-menu pattern to the Workspaces tree: workspace nodes open a workspace action menu (Activate, New chat, Rename, RAG scope, More>Archive) and chat rows reuse the conversation action menu. Retires the tree's single contextual Star button.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A workspace tree node's trailing asterisk (click) and the m binding open the workspace action menu with Activate/New chat/Rename/RAG scope/More>Archive
- [x] #2 A tree chat row's trailing asterisk (click) and the m binding open the existing conversation action menu
- [x] #3 Activate is bulleted and disabled for the already-active workspace; RAG scope states its active-workspace precondition when disabled
- [x] #4 New chat on a non-active workspace activates it and creates the session there
- [x] #5 Rename and Archive route through the existing workspace seams
- [x] #6 The contextual Star button and its selection-context line are retired; the s star binding remains
- [x] #7 Both new menus dismiss via Escape-anywhere and click-outside per ADR-068 (registry + TASK-25709 paths)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Tests first: pure workspace-menu model suite; widget suite for shape/paging/escape; wiring suite for tree click + m binding + routing + star retirement + dismissal parity.\n2. Chat/console_workspace_actions.py pure model (target, build menu, pages, gating).\n3. Widgets/Console/console_workspace_action_menu.py sibling widget + WeakSet registry + messages.\n4. console_workspace_tree.py: trailing asterisk labels, click hit-test on the asterisk cell, m binding, WorkspaceTreeMenuRequested messages.\n5. chat_screen.py: mount/route seams (activate, activate+new chat, rename, archive, rag scope), conversation-menu reuse for tree chats, dismissal paths extended to the workspace registry, Dismissed handler focuses any widget by id.\n6. console_workspace_context.py: retire tree star button + selection-context line.\n7. Targeted suites, lint, boot census; task notes.
<!-- SECTION:PLAN:END -->

## Renumbering

TASK-19601 owner rule: this task originally took id 25710, which collided
with the older arrival ``task-25710 - Home-content-recents-stream-resume-
banner.md`` (created 2026-08-30 23:39 vs this task's 2026-08-31 04:34; the
older id keeps). Renumbered to TASK-25712 with this provenance section; the
task-file frontmatter, doc comments, and test references were updated with
it. Earlier commit messages on this branch still name TASK-25710.

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Extended the TASK-23200/25709 row-menu pattern to the Workspaces tree, per
the approved design (workspaces tree nodes + tree chat rows; full action set).

- **Pure model** (`Chat/console_workspace_actions.py`): `WorkspaceMenuTarget`
  + `build_workspace_menu` — root Activate/New chat/Rename…/RAG scope…/More ▸,
  More page Archive. Activate is bulleted+disabled for the active workspace;
  RAG scope gates on active (the scope-picker seam is active-workspace
  scoped) with a stated reason. Separately tested (7 tests).
- **Widget** (`Widgets/Console/console_workspace_action_menu.py`): sibling of
  the conversation menu with its own `WeakSet` registry, `restore_focus`
  dismissal semantics, memoized single-shot detach, and paging. The two
  registries share one screen-level contract: the outside-click pass, the
  Escape guards (composer-home, collapsed-composer, hands-free, realtime),
  and the one-menu-at-a-time mount helper all fold BOTH kinds, so every
  TASK-25709 dismissal behavior carries over (pinned by parity tests).
- **Tree affordances** (`console_workspace_tree.py`): workspace/conversation
  labels carry a trailing ` *` appended AFTER truncation (budget reduced by
  the affordance width — shared `_label_budget` used by render AND the
  tooltip fit decision, so a tooltip can't fire for a row that only looks
  truncated); `render_label` records each row's asterisk x-zone, cleared per
  projection pass, and a pointer press inside the zone posts
  `WorkspaceTreeMenuRequested` instead of selecting. New `m` binding opens
  the menu at the cursor row (guarded on a measured region — a collapsed
  section's 0×0 tree has no anchor).
- **Screen routing** (`chat_screen.py`): `on_workspace_tree_menu_requested`
  mounts the right menu; `on_workspace_action_chosen` routes through existing
  seams (activate, activate-then-create for New chat, rename modal, archive
  confirmation, scope picker). Tree chat rows reuse
  `ConsoleConversationActionMenu` with a target built from marks-service
  truth. Menu-opener focus restore is now by DOM id on any widget (the tree
  is not a Button).
- **Retirement**: the contextual Star/Unstar button, its selection-context
  line, the dead left_rail press branch, and the timer-inventory pin are
  gone; `sync_workspace_tree_context` remains as the cursor recorder.
  Tests pinning the retired chrome were rerouted (disclosure test keeps the
  persisted-disclosure invariants; boundary perf test now pins the stronger
  "no reconcile at all") or deleted (selection-copy/tooltip write tests,
  star visibility/geometry tests).
- **Verification**: 243 passed across 13 suites (model, new menu suite 8/8,
  conversation menu, routing, context rail, rail reconciliation, cursor
  layout, timer inventory, hands-free, composer draft, selection-dismissal
  perf, boot census green — ADR-097 ratchet intact since all new module
  references are lazy). The 7 failures in the sweep (4 hands-free wiring, 2
  tree tooltips, 1 timer census) were each verified red on clean origin/dev
  by stashing this change. Ruff clean on all task files.
- ADR check: no new ADR — applies ADR-068's dismiss contract and the
  TASK-23200 menu pattern to two more surfaces; ADR-097 boot-ratchet honored.
  Files: 4 new (model, widget, 2 test files), 8 modified.
<!-- SECTION:NOTES:END -->
