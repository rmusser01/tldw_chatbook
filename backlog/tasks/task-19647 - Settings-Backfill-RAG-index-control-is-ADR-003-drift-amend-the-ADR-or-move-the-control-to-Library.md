---
id: TASK-19647
title: Settings 'Backfill RAG index' control is ADR-003 drift — amend the ADR or move the control to Library
status: To Do
assignee: []
created_date: '2026-08-21'
labels:
  - adr
  - tech-debt
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The "Backfill RAG index" control (`UI/Screens/settings_screen.py:2332`, binding `settings_rag_backfill`) sits in a Settings slice that ADR-003 explicitly excludes: *"RAG indexing, embedding model lifecycle, chunking templates, collection management, and workspace eligibility remain outside this Settings slice"*, and its rejected-alternatives table names adding RAG indexing to Settings as rejected. The control landed under task-541 with **no ADR amendment** (verified when the parity spec was written: no ADR supersedes or amends 003) — undocumented drift, not sanctioned precedent.

Filed from the chunking template parity design spec §11 item 7 / §10.0 (`Docs/superpowers/specs/2026-08-21-chunking-template-parity-design.md`; ADR-078). Re-verified live 2026-08-21: `settings_screen.py` changed on `origin/dev` since the spec's pin but none of the diff lines touch the Backfill control. The parity sub-project deliberately put its own re-chunk action in Library honoring ADR-003 as written; this task stops the boundary from staying ambiguous.

The outcome is a decision recorded **and executed**: amend ADR-003 to sanction the control's placement, or move the control to the Library RAG surface.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The decision (amend or move) is recorded with its rationale in the Implementation Notes and executed — an ADR board and a UI that disagree afterwards is a fail
- [ ] #2 If amended: ADR-003's text names the exception with the task reference; if moved: binding, footer hints, worker, and `Docs/User_Guide/settings.md` references move with it and the Settings suite stays green
- [ ] #3 The parity sub-project's re-chunk control and this control end up with a documented mutual-exclusion story (they touch the same stores — see spec §10.3's worker-guard ruling) or a documented statement that none is needed
<!-- AC:END -->
