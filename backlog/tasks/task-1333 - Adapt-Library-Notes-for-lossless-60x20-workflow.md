---
id: TASK-1333
title: Adapt Library Notes for lossless 60x20 workflow
status: In Progress
assignee:
  - '@codex'
created_date: '2026-07-31 00:19'
updated_date: '2026-07-31 00:24'
labels: []
dependencies: []
references:
  - backlog/decisions/011-chatbook-workbench-ui-system.md
  - backlog/decisions/015-shell-destination-ia.md
  - backlog/decisions/021-file-backed-notes-disk-authority-and-recovery.md
  - backlog/decisions/022-textual-8-runtime-floor.md
documentation:
  - Docs/superpowers/specs/2026-07-30-library-notes-adaptive-60x20-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make the existing Library Database Notes workflow fully keyboard-usable at 60x20 without losing in-session edits, while preserving current storage, sync, export, and Console-handoff behavior as the first phase toward a dedicated capable Notes workbench.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 At 60x20 a keyboard-only user can navigate Library to Notes Navigator to Editor to Context and back with every existing Notes capability reachable
- [ ] #2 Resizing across the compact breakpoint preserves the canonical note draft, caret, selection, scroll, focus intent, selected note, and Edit or Preview presentation
- [ ] #3 Edits made during any in-flight normal or overwrite save are serialized and persisted without a newer revision being marked saved prematurely
- [ ] #4 Reload never replaces edits made after the reload request began, and save failures or conflicts retain the draft and veto unsafe navigation
- [ ] #5 Preview, Context, conflict, and confirmation transitions preserve relevant Notes widget identity, and any screen-level recompose rehydrates draft, caret, selection, scroll, and focus
- [ ] #6 All visible compact Notes controls remain within the viewport with one defined scroll owner per active region and no page-level horizontal overflow
- [ ] #7 Geometry-critical fallback, source, and generated CSS stay aligned and are covered by parity checks
- [ ] #8 Focused unit, Pilot, regression, static, and full-project verification required by the repository Definition of Done passes using isolated synthetic data
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Review the approved design and governing ADRs in Docs/superpowers/specs/2026-07-30-library-notes-adaptive-60x20-design.md, backlog/decisions/011-chatbook-workbench-ui-system.md, backlog/decisions/015-shell-destination-ia.md, backlog/decisions/021-file-backed-notes-disk-authority-and-recovery.md, and backlog/decisions/022-textual-8-runtime-floor.md.
2. Add failing pure-state tests for canonical draft revisions, serialized normal/overwrite saves, conflict resolution, and conditional reload.
3. Add failing real-bundle LibraryHarness tests for Notes-scoped compact navigation, keyboard capability access, geometry, focus, resize preservation, snapshot rehydration, and CSS parity.
4. Implement pure immutable Notes session transitions in library_notes_state.py and orchestration in library_screen.py without changing storage schema or sync ownership.
5. Implement stable Notes presentation surfaces and responsive CSS from _agentic_terminal.tcss, align DEFAULT_CSS, and regenerate the bundle.
6. Run focused tests, static/CSS checks, full project verification, self-review, and isolated synthetic 60x20 plus wide UAT; document evidence and deviations.

ADR required: no new ADR
ADR path: backlog/decisions/011-chatbook-workbench-ui-system.md
Reason: ADR-011 already governs stable workbench composition, explicit state snapshots, and responsiveness. ADR-015, ADR-021, and ADR-022 remain related boundaries; Adapt changes no schema, file authority, sync policy, route ownership, security boundary, provider boundary, or dedicated-workbench architecture.
<!-- SECTION:PLAN:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 All acceptance criteria are checked with linked evidence
- [ ] #2 Automated unit and Pilot integration tests cover new state and interaction logic
- [ ] #3 Static, formatting, CSS generation/parity, focused regression, and full-project checks required by the repository pass
- [ ] #4 Relevant design and implementation documentation is updated, including the ADR check
- [ ] #5 Self-review confirms no storage, sync-authority, security, or unrelated Library-canvas regressions
- [ ] #6 Implementation Notes summarize the approach, trade-offs, modified files, verification, and any plan deviations
<!-- DOD:END -->
