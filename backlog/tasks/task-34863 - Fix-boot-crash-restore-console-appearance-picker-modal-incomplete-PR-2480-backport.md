---
id: TASK-34863
title: >-
  Fix boot crash: restore console appearance picker modal (incomplete PR-2480
  backport)
status: Done
assignee: []
created_date: '2026-09-12 00:33'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
App crashed after splash close: working-tree WIP workspace.py imports Widgets.Console.console_appearance_picker_modal, which was never swept in from PR #2480 (commit 683390f414 on dev, not an ancestor of HEAD).
<!-- SECTION:DESCRIPTION:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Root cause: partial backport of PR #2480 into the working tree -- workspace.py + Chat/console_appearance.py present, modal + tests missing. Fix: (1) restored console_appearance_picker_modal.py, Tests/UI/test_console_appearance_picker.py, Tests/Chat/test_console_appearance.py from 683390f414 (all deps verified present; console_appearance.py identical to PR); (2) ported the PR's color-only placeholder logic into this branch's divergent _switcher_icon_prefix (console_session_switcher_modal.py); (3) declared 7 WIP modal launches on the root edge of CONSOLE_MODAL_LAUNCH_EDGES and rode the documented inventory_only_types escape hatch for the 6 modals lacking dismissal contracts (follow-up: write proper contracts). Verified: 46/46 appearance tests, 118/118 modal dismissal tests, headless real-app boot x2 reaches ChatScreen past splash (skip 2.4s, auto-close 7.0s). Known pre-existing non-fatal issue: user media DB schema v9 > code v6 (migrated by a newer dev build), logs CRITICAL at boot but does not block it. ADR: not required - restoration of reviewed code + test inventory updates.
<!-- SECTION:NOTES:END -->
