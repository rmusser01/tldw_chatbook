---
id: TASK-31245
title: Add Character chats mode to CtrlK switcher
status: To Do
assignee: []
created_date: '2026-09-04 02:09'
labels:
  - console
  - switcher
  - characters
  - ux
dependencies:
  - TASK-31244
references:
  - >-
    Docs/superpowers/specs/2026-09-03-character-conversation-navigation-design.md
  - >-
    Docs/superpowers/plans/2026-09-03-character-conversation-navigation-implementation.md
priority: high
---

## Renumbering provenance

Renumbered from TASK-31237 on 2026-09-04. The final pre-commit worktree sweep
found the older `Reader uses its vertical space` task created at 01:50; it keeps
TASK-31237 under the older-arrival rule. This unshipped task moves with all plan
and dependency references.

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Extend the Console session switcher into a complete operational switchboard for active tabs, history, and local character conversations while preserving all incumbent target-trust behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Ctrl+K exposes Active, History, and Character chats with F3 cycling and truthful visible hints.
- [ ] #2 Blank Active Enter still targets the most recently used other tab; explicit navigation and nonblank queries activate only the committed highlighted identity.
- [ ] #3 Active and History share their per-visit query and labeled zero-match widening; Character chats owns a separate query and never widens.
- [ ] #4 Character rows use the approved two-line grammar plus one stable selected-only detail region, with no unselected snippets.
- [ ] #5 The modal stays mounted through typed cancellable activation, freezes the committed row, ignores post-commit Escape, and cannot duplicate opens.
- [ ] #6 The exact 52x20 row budget, focus order, cell-aware truncation, paging, pointer press target, F2 restrictions, and Cancel reachability are enforced.
- [ ] #7 Context shows Continue search in Character chats only when this mode is available and transfers a validated query without pretending Meaning exists.
- [ ] #8 Targeted trust, modal dismissal, activity, keyboard, focus, geometry, zero-result, and exact-resume tests pass with production CSS.
<!-- AC:END -->
