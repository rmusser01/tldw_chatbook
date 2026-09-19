---
id: TASK-32664
title: Review Parakeet import directory controls and picker continuity
status: Done
assignee:
  - '@codex'
created_date: '2026-09-16 01:36'
updated_date: '2026-09-16 01:55'
labels:
  - library
  - ingest
  - design-system
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue the ingest options review with the local Parakeet model-folder controls. Correct the unstyled overflow test and qualify directory selection and cancellation without losing the staged import or keyboard context.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Directory input and Browse are fully visible and keyboard reachable under production CSS at compact and wide sizes in both themes.
- [x] #2 Selecting a folder updates only the staged import option while preserving other draft fields and a visible usable focus destination.
- [x] #3 Cancelling the picker preserves the draft and returns visible focus; unavailable dependency controls remain disabled with their reason.
- [x] #4 Targeted and isolated native evidence document real versus simulated availability, with no import, install or provider request and no raised budgets.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/014-library-ingest-service-authority-and-recovery.md; backlog/decisions/150-design-token-system-and-design-language.md; backlog/decisions/161-component-pattern-library.md (existing)
Reason: Correct test CSS provenance and preserve existing form/picker interaction; no storage, runtime, authority or token-value change.

1. Correct the existing directory geometry test to load shipped app and screen styles, based on the paired CSS probe (unstyled input100w versus styled1fr). Preserve render-only harnesses for unrelated unit tests.
2. Add production-CSS journeys for enabled/disabled directory controls, actual Tab/Shift+Tab/Enter, real picker selection and cancellation, staged metadata and option retention at170x48 and80x24 in both themes. Simulate optional package availability only at the UI capability seam; never load a model or submit an import.
3. If a continuity defect reproduces, repair the owning picker callback using existing in-place option updates and focus behavior. Avoid whole-screen recomposition or new layout values for a row that already fits.
4. Run targeted new and neighboring ingest/picker/token checks, compare static diagnostics and file sizes to6144043362, obtain focused review and verify a private native wide/compact journey with exact persistence and normal exit.
5. Update guide, QA, audit, task and any generalizable testing lesson; commit locally. No full suite, extraction, installation, provider/server request, push or dev integration.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Selecting a Parakeet model folder now updates the retained Input instead of recomposing LibraryScreen, preserving title cursor, draft fields and visible Browse return focus. Same-folder selection explicitly refreshes the gate after disarming consent, fixing the stale confirmation found in independent review. The historical row-overflow test now loads APP_STYLESHEETS: the paired probe proved a harness CSS omission rather than a production row defect.

Verification:279 distinct targeted checks pass (208 canvas/structural/token/bundle/picker +16 new directory journeys +55 existing consent), zero new Ruff diagnostics, formatted new files/changed ranges, git diff --check clean. Four baseline selection failures reproduced cursor reset and offscreen compact focus; same-folder confirmation regression failed before its repair. Independent follow-up review found no remaining actionable findings. No budget increased; inherited screen size ceiling remains documented in size comparison.

Actual TldwCli/LinuxDriver private run-003 passes at170x48 dark/80x24 light, including Select/Cancel/same-folder consent and keyboard re-entry. Real unavailable tooling is recorded; enabled controls simulate only UI package availability. Ten healthy SQLite databases, zero Media/messages/ingest jobs, unchanged source, empty selected folder, no ERROR/CRITICAL log lines, normal exit0 and owned session closed. No import, installer, model, provider/server request, fullsuite, push or dev integration.

Updated QA (Docs/superpowers/qa/2026-09-16-ingest-options/README.md), prior-report correction, workflow audit, user guide and testing-evidence lesson. Initial native profile needed its configured private DB parent created; corrected fresh runs passed. Next review:compact directory-picker list layout (loaded row not visible), then remaining per-type options and queue recovery. This slice qualifies typed-path Select/Cancel, not compact list navigation or extraction.

ADR required:no; existing014,150,161 apply as linked in Implementation Plan and QA. Changed library_screen.py callback, canvas test CSS setup, new directory journey tests and QA/docs only.
<!-- SECTION:NOTES:END -->
