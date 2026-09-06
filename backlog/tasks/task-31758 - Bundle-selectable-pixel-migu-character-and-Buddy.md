---
id: TASK-31758
title: Bundle selectable pixel-migu character and Buddy
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 23:02'
updated_date: '2026-09-06 00:09'
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
Follow Docs/superpowers/plans/2026-09-05-pixel-migu-builtins.md and ADR-122. Review follow-up fixes are complete. CI follow-up: inspect the two added startup diagnostic statements, regenerate the canonical diagnostic inventory, verify every derived-artifact checker locally, and record the evidence. Then push, obtain clean current-head Qodo and CI, and merge. ADR required: no new ADR; this corrects generated evidence without changing the existing ADR-122 behavior.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Included the approved pixel-migu artwork as an optional character with 18 Shared Visual Identity expressions and a separate Buddy Persona with 64 frames and all nine baseline runtime states. Character creation is atomic and provenance-based; Buddy installation uses the existing Actor Pack coordinator after background recovery. Existing selections, tombstones, user edits and forks remain unchanged. Bundled character resources stay immutable; Buddy copies use per-attempt profile directories and retain committed assets on interrupted returns. ADR-122 records the decision; the user guide includes selection instructions and expression preview. Packaging explicitly includes all 89 resources; the existing build fixture now copies the declared packages source root. Verified 75 character/resolver/provenance tests and 20 Buddy/import-closure/distribution tests, plus the release checker acceptance test. Both direct wheel and sdist-rebuilt wheel resolved all18 expressions and9states from read-only installed resources on fresh profiles. Missing-image artifact rejection, rollback, concurrent services and postcommit interruption are covered. Scoped Ruff lint/format and git diff --check passed. Independent code review findings were fixed. One preexisting deferred-startup test fails identically on pristine origin/dev because a4-second wait precedes the7-second splash; no new regression. Full suite not run. Recorded the cleanup/cache lesson in lessons-testing-evidence.md.

Qodo review follow-up: rebased onto dev 47bfde54. Added Google-style Buddy API sections, transactional character provenance reads, a named resource byte limit, confined validated publication paths, and filewise importlib.resources.as_file support for ZIP-backed packages on Python 3.11+. Preserved no-follow leaf checks after a red/green symlink regression. Retained the existing strict Persona Visual wire-schema validator instead of duplicating it in Pydantic; invalid renderer, empty animations and unknown-field tests prove rejection before publication. ZIP-backed resources resolve all 31 authored states. Verified 44 focused seed/import/distribution/provenance tests, then 25 Buddy/distribution tests after the symlink correction; separate existing character lifecycle/resolver gate passed 74 tests. Scoped Ruff and diff checks pass. No new ADR: this preserves ADR-122 contracts.

Final CI follow-up: rebased onto current dev 2b497397 (schema 66), and all 45 targeted seed/resource/distribution/provenance tests passed. The required generated-artifact job then exposed two unrecorded startup diagnostics. Reviewed the exact added statements: app.py emits only fixed Buddy retry text; config.py records only the exception class, without payload, path, URL or secret. Regenerated the canonical production diagnostic inventory: two owner counts/digests and aggregate count only, with no sink topology change. All six derived-artifact checkers now pass locally (CSS, profile paths, diagnostic inventory, backlog IDs, schema table allowlist and index-plan pins), and git diff --check is clean. Added the observed missing-inventory-check incident to the existing guarded-resource lesson. No runtime behavior changed and no new ADR is required; ADR-122 still applies.
<!-- SECTION:NOTES:END -->
