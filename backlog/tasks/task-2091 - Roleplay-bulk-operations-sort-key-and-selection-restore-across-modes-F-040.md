---
id: TASK-2091
title: >-
  Roleplay: bulk operations, sort key, and selection restore across modes
  (F-040)
status: Done
assignee: []
created_date: '2026-08-03 17:25'
updated_date: '2026-08-04 11:45'
labels:
  - ux-review
  - roleplay
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
No multi-select delete/export; sort is click-cycle only; restore_state round-trips only Characters mode. See Docs/superpowers/qa/2026-08-03-library-roleplay-mcp-ux-review.md.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Multi-select delete/export exists in the library pane,Sort is keyboard-accessible,Selection survives mode round-trips for all modes,Tests updated
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Failing tests: pane m-marks rows (marker, 'N marked' count, PersonaMarksChanged message, prune on refresh, clear on mode switch); s key cycles sort only in characters/personas; screen bulk delete (one confirm, backend loop, summary notify) and bulk JSON export (SelectDirectory -> per-item files); restore round-trips personas/dictionaries selections. 2. Pane: _marked_ids + 'm'/'s' bindings + marked-row rendering. 3. Screen: PersonaMarksChanged handling, inspector.set_marked_count gating (PNG disabled with reason under marks), _begin_delete_marked + _export_marked_json workers. 4. restore_state/_apply_pending_restore: accept personas/dictionaries/lore modes via _apply_mode before selecting; update the two gate tests. 5. Suites + ruff. ADR required: no - extends existing selection/pane patterns; no schema or contract change.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
(a) Bulk ops: library pane marks ('m' toggles, ● glyph, 'N marked' count summary, prune-on-refresh, clear-on-mode-switch, PersonaMarksChanged to the screen); inspector Delete/Export JSON retarget the marked set with scope-saying tooltips, Export PNG disabled with 'Bulk export is JSON only.'; bulk delete = one confirm then per-item backend delete (per-item fetch_character_by_id for versions) + one refresh + summary notify; bulk JSON export = SelectDirectory once then one <name>.json per card (plain cards - the TTS include checkbox is a per-selection decision). (b) Sort keyboard: pane 's' binding posts the same PersonaSortCycleRequested as the button (no-op where sort doesn't apply), tooltip + footer hint disclose it. (c) restore_state now accepts all chip modes and _apply_pending_restore runs _apply_mode for non-Characters modes before re-selecting - the task-434 fallback floor is raised; the two gate tests became restore tests. Files: personas_messages.py, personas_library_pane.py, personas_inspector_pane.py, personas_screen.py; tests in test_personas_{library_pane,workbench,workbench_state}.py. Verified: pane 35, bulk class 5, state 9, race/isolation included in gate 439 passed (workbench+dict+lore+scale); ruff clean. ADR: not required (extends existing selection/pane idioms; no schema/boundary change).
<!-- SECTION:NOTES:END -->
