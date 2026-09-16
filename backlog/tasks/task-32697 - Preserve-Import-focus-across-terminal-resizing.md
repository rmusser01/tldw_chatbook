---
id: TASK-32697
title: Preserve Import focus across terminal resizing
status: Done
assignee:
  - '@codex'
created_date: '2026-09-16 05:35'
updated_date: '2026-09-16 05:48'
labels:
  - library
  - ui
  - focus
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Changing terminal width or height should keep the current Import control visibly focused and preserve the staged draft, disclosure state, and queue. The native component audit observed focus transfer to Library navigation on resize.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Import fields and queue controls remain visibly focused through compact, wide, and height-only resize round trips with their values and identity retained.
- [x] #2 A newer user focus move wins over deferred resize work, including movement outside Import; resize does not reload sources or write preferences.
- [x] #3 Targeted neighboring focus and layout checks plus isolated native evidence qualify the repair without running a real import.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/086-library-adaptive-reader-shell.md; backlog/decisions/150-design-token-system-and-design-language.md; backlog/decisions/161-component-pattern-library.md
Reason: restore the existing destination-owned focus contract during in-place geometry changes without altering layout policy, persistence, or queue execution.
1. Add production-CSS resize journeys for Import metadata, options, queue Details, and Recent imports with draft, identity, and no-data-work assertions. Reproduce the rail fallback before changing production.
2. Exclude Import from Notes semantic-focus restoration at the existing Prompt route guard; schedule the canvas current-focus reveal after its own resize.
3. Verify newer focus inside and outside Import wins, including callbacks delayed until after explicit movement. Run targeted Import, Prompt, Notes, resize-cost and governance checks.
4. Run a private TldwCli terminal journey with actual resize events and synthetic queue jobs; inspect wide/compact captures, review the diff, document results and commit locally.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Resizing Import now preserves the current control instead of replaying a Notes focus tuple that falls back to the rail. The canvas reveals current focus after viewport layout, while newer focus inside or outside Import remains authoritative. Draft text, selection, disclosure, widget identity and queue state stay intact.

Changed LibraryScreen and LibraryIngestCanvas, added 12 production-CSS resize journeys, updated the Import user guide, and recorded the isolated native runner, six inspected captures and receipts in Docs/superpowers/qa/2026-09-16-ingest-resize/README.md. The red baseline reproduced rail focus theft; a subprocess mutation removing only the new canvas reveal reproduced an offscreen focused title.

Validation: 98 targeted tests passed (92 Import/Prompt/resize/governance plus six Notes neighbors); 32 native resize steps passed in both themes at 80x24, 170x48 and 170x24. Native TldwCli exited normally; private databases remained healthy and empty of media/messages/ingest jobs, source bytes and default-profile hashes were unchanged, and no ERROR/CRITICAL app log lines appeared. No real import or provider/server operation ran. Independent review found no actionable findings.

Zero new Ruff diagnostics; new files and changed ranges formatted; diff whitespace checks passed. Inherited static debt remains: 205 LibraryScreen and six canvas Ruff diagnostics, and the LibraryScreen size ratchet fails at 35,215 lines versus 33,204 allowed (baseline 35,214); its companion check passes. No budgets were raised and no full repository suite ran.

ADR required: no. Existing ADR-086, ADR-150 and ADR-161 govern this repair; paths are linked in the implementation plan and QA README. No layout policy, visual token, persistence or execution boundary changed. Plan followed; provider-specific recovery remains outside this resize slice.
<!-- SECTION:NOTES:END -->
