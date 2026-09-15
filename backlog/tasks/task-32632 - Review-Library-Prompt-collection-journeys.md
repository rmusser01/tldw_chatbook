---
id: TASK-32632
title: Review Library Prompt collection journeys
status: Done
assignee:
  - '@codex'
created_date: '2026-09-15 16:10'
updated_date: '2026-09-15 16:30'
labels:
  - library
  - prompts
  - design-system
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue the component review through local Prompt collections, verifying keyboard entry, catalog actions, membership staging and recovery while preserving Prompt drafts and existing service boundaries.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Collections opens from More actions and returns to visible membership controls at wide and compact sizes in dark and light themes.
- [x] #2 Catalog search, paging, create, rename, name-collision recovery and cancellation preserve readable controls and entered text.
- [x] #3 Membership Done stages changes, Cancel retains the prior set, and Apply persists independently from unsaved Prompt content with usable failure recovery.
- [x] #4 Targeted automated checks and isolated native verification document the reviewed collection journeys.
- [x] #5 The user guide, workflow audit and task notes describe the verified behavior and remaining limitations.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/086-library-adaptive-reader-shell.md; backlog/decisions/150-design-token-system-and-design-language.md; backlog/decisions/161-component-pattern-library.md (existing)
Reason: Review and repair existing collection interaction, focus and draft continuity; preserve TASK-198 local ownership, membership transactions and immutable session guards.

1. Exercise production-CSS keyboard journeys with real SQLite through Basic More actions, manager Done/Cancel and membership Apply at 170x48 and 80x24 in both themes. Reproduce defects before repair.
2. Check catalog search/page, create/rename/collision and retry continuity; retain exact request and identity guards. Resolve targeted baseline failures according to evidence.
3. Run affected UI, domain and governance checks only. Verify a private native profile with synthetic collections, final compositor captures, normal exit and read-only persistence checks.
4. Update the guide, workflow audit and QA evidence; self-review, close this Backlog task and commit locally. No full repository sweep or integration into dev.

Allocation: fresh fetch; reachable paths and 42 live worktrees maximum 32631; no content claim for 32632 across 313 refs. CLI offered conflicting 32631; corrected the new record to 32632 before implementation.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
More actions → Collections now reveals Info from Basic, closes the menu and opens from the visible Manage control. Done and Cancel return to that control and explicitly scroll it into a compact viewport. The manager preserves entered names and unsubmitted search text through child replacement. Membership Apply restores retry/Manage focus only when no newer focus owns it, preserving unsaved Prompt fields and content.

Verification: 180 distinct targeted checks pass (66 collection UI, 22 adjacent History/action journeys, 66 domain/state and 26 governance/wiring). New journey and native-runner files pass Ruff and formatting; existing files have no new diagnostics. Six final native captures were rendered and inspected at 170x48 dark and 80x24 light. The actual TldwCli/LinuxDriver used 12 private storage roots and exclusive ownership; normal Ctrl+Q returned exit 0 to zsh. Read-only SQLite confirms both original v1 Prompts, renamed active collections and exact applied memberships. The owned session is closed. No full repository sweep or provider verification was run.

Updated the controller, manager modal, Screen delegator and canvas routing; added real-SQLite keyboard journeys, corrected two baseline readiness/count assertions, and updated the guide, workflow audit, QA record and focus/viewport lesson. No storage, service boundary or token values changed. Existing limitations: newly staged names can show Collection #ID until Apply and compact status text can require scrolling. Native fault injection and large catalogs were covered by targeted tests rather than this native run.

ADR required: no; applies existing backlog/decisions/086-library-adaptive-reader-shell.md, 150-design-token-system-and-design-language.md and 161-component-pattern-library.md, preserving TASK-198 contracts. Evidence: Docs/superpowers/qa/2026-09-15-prompt-collections/README.md. Next: Use in Console; integration into dev remains pending.
<!-- SECTION:NOTES:END -->
