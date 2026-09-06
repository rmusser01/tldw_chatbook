---
id: TASK-31243
title: Add trusted character navigation recovery and Roleplay browse
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-04 02:08'
updated_date: '2026-09-06 00:27'
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

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deliver the first complete local character-conversation navigation slice: draft-safe departure, typed exact Console activation, Library-owned unavailable-link repair, and full Roleplay browse/search/preview.
<!-- SECTION:DESCRIPTION:END -->

## Renumbering provenance

Renumbered from TASK-31235 on 2026-09-04. The final pre-commit worktree sweep
found the older `Sort chooser renders every option` task created at 01:50; it
keeps TASK-31235 under the older-arrival rule. This unshipped task moves with
all plan and dependency references.

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

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/120-character-conversation-navigation-and-local-semantic-search.md
Reason: Implements existing activation, draft-veto, repair and surface ownership contracts.
1. Start TASK-31243 after Task 2 merges; capture targeted incumbent baseline.
2. Write failing navigation-payload validation tests.
3. Implement versioned payload module and navigation keys.
4. Write failing activation state-machine tests.
5. Implement Console-owned activation coordinator.
6. Write failing aggregate Roleplay draft-veto tests.
7. Implement app-owned pre-navigation coordinator.
8. Write failing Library repair interaction tests.
9. Implement Library-owned repair presentation.
10. Write failing Roleplay browse and compact-flow tests.
11. Extend incumbent Roleplay controller and widgets.
12. Build CSS and run targeted Task 3 gate.
13. Perform isolated real-TUI verification and commit.
Delivery adaptation: reuse seven original Task3 commits through 653b9e3c0 plus owning later fixes; preserve merged Task2 schema/APIs and newer dev ownership. New adaptations require fresh RED/GREEN evidence. Task3 Console bodies belong in workspace/modules. Native evidence remains pending; Pilot is compositor evidence only. No full-suite runs or remote actions.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Packaged the approved Task3 implementation on merged Task2 (2b4973971e5dcf101c5a6ddcc55aa082ff22f814): typed exact activation with cached-runtime rollback/token ownership, aggregate Roleplay draft veto, complete three-field keyset browse, retained Keyword snapshot-time copy, and Library-owned confirmed CAS repair with bounded candidate continuation. Preserved owning late b157/8a3/c1 fixes without Task4/5 consumers. New Console bodies live in workspace with named late-bound wiring; screen changes remain lifecycle/state glue. ADR required: no; existing backlog/decisions/120-character-conversation-navigation-and-local-semantic-search.md governs. Fresh adaptation RED/GREEN and current targeted evidence are documented in Docs/QA/task-31243/README.md and local task-3-report.md. Main selected gate:569passed1fixturefailure; corrected core68passed, resource fixture10passed with zero registered worker connections, final compact/capture9passed, queue-veto7passed. Focused Ruff/format and CSS reproducibility pass; inherited screen ratchet remains red and was not raised. One bounded compact-preview paint correction preserves full labels/body and pointer Back focus. Native isolated-terminal walkthrough remains unavailable; do not mark Done or claim all acceptance evidence complete until controller review and missing platform evidence are addressed.

Final six-capture controller confirmation: full Back/Send/Open labels and visible transcript are resolved at52x20; a right-edge In inspector-area fragment remains beside Send. This is a known compact visual concern, not a visual pass. No further polish pass was started; independent review will triage it alongside inherited ratchet deltas. Generated SVG whitespace-only blank lines were normalized for git diff --check without changing rendered content.
<!-- SECTION:NOTES:END -->
