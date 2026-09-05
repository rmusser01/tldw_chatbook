---
id: TASK-31245
title: Add Character chats mode to CtrlK switcher
status: In Progress
assignee:
  - codex
created_date: '2026-09-04 02:09'
updated_date: '2026-09-05 16:51'
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

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Extend the Console session switcher into a complete operational switchboard for active tabs, history, and local character conversations while preserving all incumbent target-trust behavior.
<!-- SECTION:DESCRIPTION:END -->

## Renumbering provenance

Renumbered from TASK-31237 on 2026-09-04. The final pre-commit worktree sweep
found the older `Reader uses its vertical space` task created at 01:50; it keeps
TASK-31237 under the older-arrival rule. This unshipped task moves with all plan
and dependency references.

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
- [ ] #9 Navigation and Keyword delivery is isolated on frozen dev with the original five task boundaries, no Meaning runtime or controls, and all applicable later correctness fixes.
- [ ] #10 Fresh targeted tests, startup comparison, resource ownership, static checks and bounded Pilot evidence are recorded with inherited failures and unavailable external evidence explicit.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Release isolation (2026-09-05): replay the exact original 23-commit Tasks 1–5 prefix onto e990738b; reconcile provisional ADR116 to ADR120 while preserving shipped Schedules116 and schema65; audit later non-Meaning hunks; qualify targeted behavior, startup, resource cleanup, static checks and production-styled Pilot at 52x20 and 120x50. ADR required: no new ADR. ADR path: backlog/decisions/120-character-conversation-navigation-and-local-semantic-search.md. Reason: existing independent Keyword delivery contract. Keep this task In Progress for unresolved gates. Binding scope/plan: Docs/superpowers/specs/2026-09-05-character-keyword-release-scope.md and Docs/superpowers/plans/2026-09-05-character-keyword-release-isolation.md.
<!-- SECTION:PLAN:END -->
