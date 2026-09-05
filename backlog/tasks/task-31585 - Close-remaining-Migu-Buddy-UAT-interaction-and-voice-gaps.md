---
id: TASK-31585
title: Close remaining Migu Buddy UAT interaction and voice gaps
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-05 03:32'
updated_date: '2026-09-05 03:48'
labels: []
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Port the remaining reproducible Buddy UAT fixes onto current dev while preserving its newer Buddy, Console, and governance architecture.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Native Buddy move and resize commit final release coordinates when mouse-move events are coalesced.
- [x] #2 Read-back uses trusted Console speech and clears presentation after playback completion or failure.
- [x] #3 Project-instruction recovery refusal releases the run and preserves the unsent draft without replacing newer edits.
- [x] #4 Diagnostic inventory excludes nested virtual environments while retaining application modules.
- [x] #5 Persona Visual import and edit publication succeeds without weakening file identity and containment checks.
- [ ] #6 Targeted tests and scoped static checks pass on the actual PR tree; earlier live evidence and remaining OpenAI credential limitation are clearly distinguished.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Compare older UAT changes with current dev, port only unsuperseded fixes with failing seam regressions, run targeted tests and governance checks, review the final diff, and create a PR against dev. ADR required: no new ADR. Existing ADR-074 Persona Visual/Buddy, ADR-037 trusted speech, ADR-069 project instructions, and ADR-029 private diagnostics govern the repairs; no new ownership boundary or storage schema.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Ported only remaining UAT fixes onto dev 68f9d865: final-release Buddy geometry, profile-owned publication, trusted readback lifecycle, setup terminalization and transient echo exclusion, visible-owner-fenced draft restoration, nested environment pruning, and two metadata-only lifecycle diagnostics. Existing ADR-074/037/069/029 apply. Native/publication/importer 171 passed; independent focused review 15 passed; final focused selection 23 passed. Broader targeted runs have 6 Console and 2 diagnostic failures, all reproduced on pristine base, alongside the existing Library Iterable F821; retain AC6 and In Progress status and publish draft PR. Evidence, exact limits and earlier hardware provenance: qa/buddy-uat-2026-09-05/README.md. No unrelated dirty-checkout work included.
<!-- SECTION:NOTES:END -->
