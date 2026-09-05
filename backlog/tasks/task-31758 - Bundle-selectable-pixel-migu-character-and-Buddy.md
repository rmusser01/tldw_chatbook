---
id: TASK-31758
title: Bundle selectable pixel-migu character and Buddy
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 23:02'
updated_date: '2026-09-05 23:15'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Include the approved pixel-migu art as a selectable character and Buddy without requiring imports on a fresh install.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Fresh profiles can select pixel-migu as a character with 18 usable expressions and a Buddy with all baseline runtime states.
- [x] #2 Restarts preserve customizations, deletions, existing active choices, and user content.
- [x] #3 Wheel and sdist contain the final assets and installed-package fresh-profile verification passes.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Follow Docs/superpowers/plans/2026-09-05-pixel-migu-builtins.md. ADR required: yes. ADR path: backlog/decisions/122-bundled-pixel-migu-character-and-buddy.md. Reason: bundled content and first-install Persona ownership; reuse existing storage/runtime contracts.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Included the approved pixel-migu artwork as an optional character with 18 Shared Visual Identity expressions and a separate Buddy Persona with 64 frames and all nine baseline runtime states. Character creation is atomic and provenance-based; Buddy installation uses the existing Actor Pack coordinator after background recovery. Existing selections, tombstones, user edits and forks remain unchanged. Bundled character resources stay immutable; Buddy copies use per-attempt profile directories and retain committed assets on interrupted returns. ADR-122 records the decision; the user guide includes selection instructions and expression preview. Packaging explicitly includes all 89 resources; the existing build fixture now copies the declared packages source root. Verified 75 character/resolver/provenance tests and 20 Buddy/import-closure/distribution tests, plus the release checker acceptance test. Both direct wheel and sdist-rebuilt wheel resolved all18 expressions and9states from read-only installed resources on fresh profiles. Missing-image artifact rejection, rollback, concurrent services and postcommit interruption are covered. Scoped Ruff lint/format and git diff --check passed. Independent code review findings were fixed. One preexisting deferred-startup test fails identically on pristine origin/dev because a4-second wait precedes the7-second splash; no new regression. Full suite not run. Recorded the cleanup/cache lesson in lessons-testing-evidence.md.
<!-- SECTION:NOTES:END -->
