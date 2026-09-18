---
id: TASK-32629
title: Review Library Prompt Copy and Markdown export journeys
status: Done
assignee:
  - '@codex'
created_date: '2026-09-15 14:47'
updated_date: '2026-09-15 15:19'
labels:
  - library
  - prompts
  - design-system
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue the component review through the visible saved-Prompt Copy and Export controls, including actual file-picker interaction, Markdown fidelity, focus return and failure recovery.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Copy Markdown and Export are keyboard-reachable through More actions in wide and compact production layouts, preserving the current editor.
- [x] #2 Successful Copy and Export preserve legacy and structured Prompt content through the canonical Markdown parser without changing the source row.
- [x] #3 Export opens a readable filename field, accepts a real keyboard save, and returns focus to its action after cancellation or completion.
- [x] #4 Clipboard unavailability/failure and export write failure report truthful feedback and retain a usable editor for retry.
- [x] #5 Targeted regression checks and isolated native journeys record the results without a full repository sweep.
- [x] #6 Markdown export documentation accurately describes empty-value normalization and preserved multiline content.
- [x] #7 Copy feedback and export results remain readable without covering focused actions or the next save dialog at compact size.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/040-versioned-prompt-artifacts-and-safe-improvement-transactions.md; backlog/decisions/086-library-adaptive-reader-shell.md; backlog/decisions/150-design-token-system-and-design-language.md; backlog/decisions/161-component-pattern-library.md (existing)
Reason: Review and repair existing Prompt copy/export controls and modal recovery without changing Markdown, clipboard, storage or ownership contracts.

1. Exercise the saved Prompt More actions path through Copy Markdown and the actual FileSave dialog with production CSS at 170x48 and 80x24, dark and light; compare copied/exported content through the canonical parser and verify the source row is unchanged.
2. Check keyboard filename entry, cancel and successful save focus return, clipboard failure/unavailability, and export write failure/retry. Add failing regressions before repairing verified gaps using existing modal, feedback and token patterns.
3. Run affected targeted tests and applicable governance, then an isolated native app journey with real file output and explicit clipboard handoff evidence; record measured sizes and fresh shutdown evidence.
4. Update the user guide and workflow audit as needed, complete task evidence, self-review and commit locally. Full suites and integration into dev are outside this review.

Allocation: CLI proposed 32629; a reachable-history and live-worktree scan found maximum 32628, and content checks across 309 refs found no prior 32629 reference. The allocated ID is available.

Observed documentation drift: prompt_markdown_export.render_prompt_markdown still describes empty-lane section bleeding and multiline truncation as current limitations, although the parser has already been fixed and its regression file verifies both behaviors. Correct the renderer docstring to describe current empty-value normalization and multiline preservation; verify the parser/renderer round-trip file.

Native compact finding: stacked Copy/unavailable and Export/cancel notifications cover focused More actions and the FileSave Save button for several seconds. Route Copy outcomes and Export results to the existing Prompt status area, revealing it without moving focus. Preserve application notifications for export results whose original editor is no longer active. Add channel/paint regressions first, keep source/callback identity guards, and rerun the affected action tests plus native flow without waiting for toast expiration.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Copy and Export outcomes now reveal the existing inline Prompt status, keeping focused actions and the next FileSave dialog unobstructed at compact size. The reporter checks the current screen, editor route and original export identity; delayed results elsewhere retain application notifications. Live fields, source rows, Markdown structure and mutation guards are preserved. The renderer docstring now reflects existing empty-value normalization and multiline preservation.

Verification: 104 targeted checks pass (19 new journeys, 42 neighboring cases, 17 parser round trips, 13 token/bundle checks, 13 controller wiring checks). Two regressions were observed failing before their respective repairs: missing inline feedback and hidden-editor feedback after screen navigation. Final native run-004 (PID 41434) passes at measured 170x48 dark and 80x24 light, including real file failure/retry and OSC52 handoff. Compact Ctrl+Q returns exit 0 to the observed shell; post-exit read-only SQLite confirms the source remains live at version 1. New tests/probe pass Ruff; changed functions are formatted with no added baseline diagnostics. No full repository sweep, external provider call or OS clipboard delivery qualification.

Modified: Library Prompt controller/export feedback routing, adjacent feedback assertions, new Copy/Export journey tests, renderer documentation, Prompt user guide and workflow audit. Evidence: Docs/superpowers/qa/2026-09-15-prompt-export/README.md. Updated lessons-live-verification.md with the native toast-obstruction incident. Self-review complete; no storage, dependency, licence or permission-boundary changes.

ADR required: no. Existing backlog/decisions/040-versioned-prompt-artifacts-and-safe-improvement-transactions.md, 086-library-adaptive-reader-shell.md, 150-design-token-system-and-design-language.md and 161-component-pattern-library.md govern this routine feedback repair. No deviation beyond the documented native finding and current-screen refinement. Integration into dev remains pending; History, Collections and Use in Console remain in the broader review.
<!-- SECTION:NOTES:END -->
