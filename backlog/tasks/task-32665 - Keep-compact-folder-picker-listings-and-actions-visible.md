---
id: TASK-32665
title: Keep compact folder picker listings and actions visible
status: Done
assignee:
  - '@codex'
created_date: '2026-09-16 02:05'
updated_date: '2026-09-16 02:26'
labels:
  - library
  - design-system
  - file-picker
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue the component review by making the directory picker usable at compact terminal sizes, where loaded folders currently have no visible rows and the typed path is too narrow. Preserve selection, cancellation and resize continuity.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Loaded directory rows and the highlighted folder paint at 80×24 in both themes, with keyboard navigation into and back out of a real folder.
- [x] #2 Folder path, Select, Cancel and validation remain readable and reachable at compact sizes, without losing the listing.
- [x] #3 Resizing compact to wide and back preserves typed path, focus and selection; neighboring picker and caller contracts pass targeted checks.
- [x] #4 Production CSS, isolated native evidence, static checks and documentation qualify the repair without imports, installs, provider requests or raised budgets.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/150-design-token-system-and-design-language.md; backlog/decisions/160-progressive-file-picker-listings.md; backlog/decisions/161-component-pattern-library.md (existing)
Reason: Routine responsive layout repair of the existing folder picker; no listing service, ownership, selection contract or token-value change.

1. Reproduce compact listing and path/action geometry with production CSS and real private folders. Measure the dialog, fixed chrome, listing viewport and input row to locate lost space.
2. Add failing keyboard and paint journeys for folder listing/navigation, typed-path validation, Select/Cancel and resize retention at 80×24 and 170×48 in both themes.
3. Repair the owning compact layout using existing tokens and retained widgets. Keep the existing wide presentation; do not change discovery or selection authority. Cover sibling pickers if shared CSS is affected.
4. Rebuild source CSS; run targeted picker, caller and token/bundle checks, static and size comparisons to 254e9f527f, and focused review.
5. Run private native navigation/selection/cancel/error and resize journeys, inspect captures in one batch, verify unchanged fixture contents/zero jobs and normal exit. Update guide/audit/QA/task and commit locally. No full suite, installation, ingestion, remote/provider request, push or dev integration.
6. Focused review reproduced long-path validation hiding the compact listing, stale error on successful navigation to the current folder, and ENAMETOOLONG escaping the shared typed-directory validator. Add regressions and repair these under AC 2 before closeout; preserve full validation text in the widget while abbreviating only its compact display. Existing ADR 160 path validation contract applies.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Compact folder pickers now paint loaded/empty/scrolled rows and keep the typed path and Select/Cancel visible at 80×24. Responsive token-backed classes retain mounted controls, path selection, highlighted folder and focus through resize. Shared path validation catches metadata-probe errors; successful current-folder corrections clear stale diagnostics; compact long errors ellipsize without consuming the listing.

154 distinct targeted checks pass, including 15 new production-CSS picker journeys, 16 Library caller journeys and neighboring picker/token/bundle/CSS-budget checks. Boot CSS is 615,348/634,050 bytes; token values and budgets are unchanged. New/adjusted test Python and native runner pass Ruff/formatting; changed production ranges are formatted, with zero added Ruff diagnostics (38 inherited base-dialog diagnostics remain). Final independent review found no actionable issue.

Private TldwCli/LinuxDriver journeys at 170×48 dark and 80×24 light qualify keyboard parent/child navigation, empty folders, Select/Cancel, error recovery and live resize. Eight final SVGs and QA evidence are recorded in Docs/superpowers/qa/2026-09-16-compact-picker/README.md. Ten healthy databases, zero media/messages/jobs, unchanged synthetic source/exact source tree and normal exit 0 verified; owned session closed. Parakeet availability is simulated only to enable the UI controls. No installation, ingestion, model execution, remote request, full suite, push or dev integration.

Review expanded validation coverage under AC 2 before repair. Test-only deviations: isolated listing fixtures from setup siblings, and waited for the actual provider Select instead of its parent to remove a mount race; no wait-budget increase. Existing testing lessons already cover child-mount readiness and same-value event assumptions; no new general lesson needed.

Updated the file-picker guide, prior picker QA checkpoint and Library workflow audit. ADR required: no; existing backlog/decisions/150-design-token-system-and-design-language.md, 160-progressive-file-picker-listings.md and 161-component-pattern-library.md govern this routine layout/validation repair. Next review is remaining per-type ingest options, then queue activity/recovery.
<!-- SECTION:NOTES:END -->
