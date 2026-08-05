---
id: TASK-1333
title: Adapt Library Notes for lossless 60x20 workflow
status: In Progress
assignee:
  - '@codex'
created_date: '2026-07-31 00:19'
updated_date: '2026-07-31 01:42'
labels: []
dependencies:
  - TASK-400
references:
  - backlog/decisions/011-chatbook-workbench-ui-system.md
  - backlog/decisions/015-shell-destination-ia.md
  - backlog/decisions/021-file-backed-notes-disk-authority-and-recovery.md
  - backlog/decisions/022-textual-8-runtime-floor.md
  - backlog/decisions/027-portable-database-note-session-coordinator.md
documentation:
  - Docs/superpowers/specs/2026-07-30-library-notes-adaptive-60x20-design.md
  - Docs/superpowers/plans/2026-07-30-library-notes-adaptive-60x20.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make the existing Library Database Notes workflow fully keyboard-usable at 60x20 without losing in-session edits, while preserving current storage, sync, export, and Console-handoff behavior as the first phase toward a dedicated capable Notes workbench.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 At 60x20 and wide widths, a keyboard-only user can reach every Notes capability enumerated in the design specification's Existing capability parity table
- [ ] #2 Resizing across the fixed 120-cell workbench breakpoint preserves the complete Focus identity tuple defined by the design specification and never replaces the canonical draft
- [ ] #3 The portable DatabaseNoteSessionCoordinator imports no Textual or File Notes type and serializes normal and overwrite saves without marking a newer revision saved prematurely
- [ ] #4 Reload never replaces edits made after its request began, conflict actions cannot race or duplicate, and failures retain the draft with actionable recovery text
- [ ] #5 The Stable Composition surfaces retain identity across presentation toggles, and every whole-screen recompose passes through one central coordinator-preserving capture and rehydration seam
- [ ] #6 Wide Editor and Preview preserve direct access to keywords, metadata, Console handoff, Copy, Markdown/text export, and Delete without requiring Context
- [ ] #7 Geometry-critical fallback, source, and generated CSS stay aligned and are covered by parity checks
- [ ] #8 TASK-400 is Done and supported dependency metadata constrains Textual to >=8.0.0,<9 before TASK-1333 can be completed
- [ ] #9 Focused coordinator, state, Pilot, accessibility, lifecycle, ADR-011 responsiveness, regression, static, and full-project verification passes using isolated synthetic data
- [ ] #10 Payload validation never truncates or rewrites canonical title, body, or keyword content; invalid drafts remain dirty with a typed actionable veto and cannot report Saved
- [ ] #11 Discard-new-note and general Delete atomically block mutation/save admission, revalidate session tokens before the service call, and cannot race newly typed edits
- [ ] #12 At 60x20 every dynamic Notes state matches its exact 15-row allocation, all controls stay in bounds, and long markup-like titles remain one-row plain-text headers without altering stored text
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Complete dependency TASK-400 so supported metadata and CI enforce Textual >=8.0.0,<9.
2. Review Docs/superpowers/specs/2026-07-30-library-notes-adaptive-60x20-design.md and governing decisions backlog/decisions/011-chatbook-workbench-ui-system.md, backlog/decisions/015-shell-destination-ia.md, backlog/decisions/021-file-backed-notes-disk-authority-and-recovery.md, backlog/decisions/022-textual-8-runtime-floor.md, and backlog/decisions/027-portable-database-note-session-coordinator.md.
3. Add failing pure-state and coordinator tests for canonical drafts, lossless payload validation vetoes, serialized normal/overwrite saves, conflict-operation gating, conditional Reload, destructive admission, untouched-new-note discard eligibility, normalized detail loading, typed flush outcomes, and host independence.
4. Add failing real-bundle LibraryHarness tests for invariant outer-width breakpoint measurement, every exact 60x20 dynamic-state row budget, truthful ellipsized empty states, direct Sort/Sync choices, Notes-scoped compact navigation, complete focus-tuple preservation, unsafe-session precedence, local accelerators/compact footer priority, destructive edit-race prevention, wide inline utility compatibility, keyboard capability access, long titles, transfer feedback, central recompose rehydration, and CSS parity.
5. Implement pure immutable Notes state in library_notes_state.py and the ADR-027 DatabaseNoteSessionCoordinator plus async normalized service port in library_notes_session.py.
6. Host the coordinator from library_screen.py, add the central recompose seam and Textual focus/timer/service adapters, gate destructive mutations, and preserve existing storage, sync, and route ownership.
7. Implement stable Notes presentation surfaces, compatible wide inline utilities, compact action grouping, persistent labels, direct option controls, actionable status/empty copy, contextual footer guidance, and exact compact geometry; update _agentic_terminal.tcss and DEFAULT_CSS, then regenerate the bundle.
8. Run focused tests, ADR-011 heartbeat/backlog/timer/route soak, static/CSS checks, full project verification, self-review, and isolated synthetic 60x20 plus wide UAT; document evidence and deviations.

ADR required: yes
ADR path: backlog/decisions/027-portable-database-note-session-coordinator.md
Reason: ADR-027 defines the new long-lived host-independent Database Note draft/save/conflict/validation/destructive-admission/flush boundary while preserving Library route ownership and keeping ADR-021 File Notes authority separate.
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
