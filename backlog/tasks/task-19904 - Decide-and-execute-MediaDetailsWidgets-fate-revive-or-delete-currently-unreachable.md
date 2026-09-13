---
id: TASK-19904
title: Decide and execute MediaDetailsWidget's fate — revive or delete (currently unreachable)
status: To Do
assignee: []
created_date: '2026-08-21'
labels:
  - tech-debt
  - ui
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Widgets/media_details_widget.py` (`MediaDetailsWidget`) has no production importer — it is unreachable from any registered route — and its template `Select` is hardcoded to `[("Default", "default"), ("Custom Configuration", "custom")]` (`:167`), never populated from the DB. It is also the only existing writer of `Media.chunking_config`.

Filed from the chunking template parity design spec §11 item 1 (`Docs/superpowers/specs/2026-08-21-chunking-template-parity-design.md`; ADR-078). Re-verified live 2026-08-21: zero importers repo-wide (grep), and the file is untouched between the spec's pin (`e31a18d45`) and current `origin/dev`.

A revive-or-delete **fork** is not an outcome: the task is done when a decision is recorded with reachability evidence **and executed**. Note the pairing: deleting this widget orphans `Widgets/chunk_preview_modal.py` (its only import is `media_details_widget.py:753`) — TASK-19642 owns that module's disposition; execute the two together or in a declared order.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A revive-or-delete decision is recorded with reachability evidence (importer graph at implementation time) in the Implementation Notes, and the chosen branch is executed in the same task — not left open
- [ ] #2 If deleted: no dangling references remain repo-wide (imports, CSS selectors, tests, packaging); if revived: the template Select is populated from the Media DB via the service layer and the widget is mounted on a reachable production surface
- [ ] #3 The `Media.chunking_config` writer situation is resolved either way — a live writer exists after revive, or deletion is coordinated with the parity sub-project's new writer (spec §9.2) so the column never ends up with zero writers and no ruling
<!-- AC:END -->
