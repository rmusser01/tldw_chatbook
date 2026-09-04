---
id: TASK-31243
title: Add trusted character navigation recovery and Roleplay browse
status: To Do
assignee: []
created_date: '2026-09-04 02:08'
labels:
  - console
  - roleplay
  - library
  - navigation
dependencies:
  - TASK-31242
references:
  - >-
    Docs/superpowers/specs/2026-09-03-character-conversation-navigation-design.md
  - >-
    Docs/superpowers/plans/2026-09-03-character-conversation-navigation-implementation.md
priority: high
---

## Renumbering provenance

Renumbered from TASK-31235 on 2026-09-04. The final pre-commit worktree sweep
found the older `Sort chooser renders every option` task created at 01:50; it
keeps TASK-31235 under the older-arrival rule. This unshipped task moves with
all plan and dependency references.

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deliver the first complete local character-conversation navigation slice: draft-safe departure, typed exact Console activation, Library-owned unavailable-link repair, and full Roleplay browse/search/preview.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every caller uses one typed cancellable activation contract and Console changes destination only after commit, with rollback preserving the prior tab on failure.
- [ ] #2 Escape cancels only before commit; duplicate activation cannot open duplicate sessions; success is returned only after the exact destination is visible.
- [ ] #3 Roleplay navigation captures all incumbent card, Persona visual, attachment, and in-flight-save drafts and requires Save and continue, Discard and continue, or Stay.
- [ ] #4 Roleplay provides local per-character keyset browse, Keyword search, read-only preview, exact resume, Back to Console, and stable focus in the approved 52x20 progression.
- [ ] #5 Library accepts a typed repair context, shows historical evidence and same-authority candidates, requires explicit confirmation, and performs compare-and-set repair.
- [ ] #6 Repair failure preserves source data and focuses Refresh; success invalidates indexes and restores the requested return anchor.
- [ ] #7 Existing Roleplay card editing, imports, exports, visual and attachment workflows, and transcript-to-Console draft remain unchanged.
- [ ] #8 Targeted race, focus, compact-layout, keyboard, pointer, draft-loss, exact-resume, and unavailable-recovery tests pass.
<!-- AC:END -->
