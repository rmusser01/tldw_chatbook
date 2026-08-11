---
id: TASK-14812
title: Unify Console model selection into a searchable picker
status: In Progress
assignee:
  - '@codex'
created_date: '2026-08-10 21:52'
updated_date: '2026-08-11 01:31'
labels:
  - console
  - models
  - ux
dependencies:
  - TASK-3600
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Give Console users one keyboard-first model control that supports fast selection from the current provider catalog without requiring a separate search field or routine manual model entry.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Alt+M and full Console settings use the same searchable model-selection interaction
- [x] #2 Typing filters the current provider's full endpoint-scoped catalog without starting a catalog request on each keystroke
- [x] #3 Keyboard users can open, filter, choose, clear, and cancel without losing the current model
- [x] #4 Loading, empty-catalog, unavailable-catalog, and current-model-not-listed states are explicit and actionable
- [x] #5 A clearly separated custom model ID escape hatch remains available
- [x] #6 Provider changes refresh the choices and cannot retain a model from the previous provider
- [x] #7 Custom model IDs are validated as bounded single-line text before they can reach downstream provider calls
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Characterize the existing model Select, search, custom-entry, readiness, and provider-switch contracts in both Console surfaces.
2. Evolve ModelSearchPicker into a controlled searchable picker that loads an uncapped provider catalog once, filters locally, preserves the committed model while searching, and exposes explicit catalog/current/custom states.
3. Mount the same visible picker in Alt+M and full Console settings, with existing model state controls retained only as hidden compatibility adapters for validation and save logic.
4. Route provider changes, model selections, custom values, focus, apply, and cancel through the shared picker without carrying a model across providers.
5. Add pure and mounted regressions for one-load filtering, keyboard selection/cancel, empty and unavailable catalogs, current-not-listed state, custom IDs, and provider switching; mutation-test the one-load and cross-provider guards.
6. Run scoped Ruff, picker/Console tests, architecture checks, and live Textual verification where the environment permits.

ADR required: no
ADR path: backlog/decisions/020-automatic-model-catalog-refresh.md
Reason: ADR-020 already defines catalog authority, uncapped search, transient current-model preservation, and fallback behavior; this task consolidates the existing UI without changing those boundaries.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Reworked `ModelSearchPicker` into the controlled, keyboard-first model control used by both Alt+M and full Console settings. It loads each provider's uncapped endpoint-scoped catalog in a worker, filters only in memory, preserves the committed selection during search/cancel, and provides explicit loading, empty, unavailable, unlisted-current, no-match, and custom-ID states.
- Routed provider switches, catalog selection, custom IDs, focus recovery, apply, and manual endpoint discovery through the shared picker. Existing Select/Input controls remain hidden compatibility adapters for established validation and draft logic.
- Added a provider-scoped overlay for results from the full modal's explicit unsaved-endpoint probe, so probed models are immediately searchable without another catalog request.
- Made the Alt+M popover vertically bounded and scrollable after mounted geometry testing showed an open result list grew to 34 rows on a 24-row terminal.
- Added mounted coverage for keyboard selection and Escape, one-load filtering, query-during-load recovery, explicit states, custom IDs, provider isolation, manual-discovery overlays, shared-modal save behavior, and compact-terminal geometry. Mutation checks proved the no-request-per-key and cross-provider-reset tests fail when their guards are removed, then passed after restoration.
- Rebased verification on current `origin/dev`: 19 picker tests passed; 45 Console rail/popover tests passed; the broader full-modal `-k model` slice passed all 49 tests; the 85-test catalog/resolver/picker set and the targeted gateway model-recovery test passed. Scoped Ruff, fatal-error Ruff for the legacy screen module, and `git diff --check` passed. Mutation checks proved the no-request-per-key and cross-provider-reset tests fail when their guards are removed, then pass after restoration.
- The blocking-I/O architecture suite passes all six tests after its repository scanner was corrected to decode UTF-8 source explicitly on Windows. TASK-14878 now makes the unrelated symlink containment test skip clearly when the host account lacks link privilege; its file passed 22 tests with that one capability skip. Repository-wide completion remains blocked by the pre-existing screen-size ratchet, which reports `chat_screen.py` at 19,743 lines versus its 17,727 budget even though this patch removes three lines, and by the unrelated timing-sensitive gateway concurrency test. TASK-3600 is still in progress, so this task remains In Progress despite its acceptance criteria being met.
- ADR required: no. Existing ADR-020 remains the governing catalog-authority and fallback decision.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

Qodo review remediation: validated custom model IDs with the shared input-validation helpers and a 256-character single-line boundary; restored the committed catalog model after an uncommitted filter loses focus; deferred blur collapse briefly so pointer clicks complete before layout reflow; renamed ModelPickerInput to conform to class naming; and replaced the provider-error and Windows privilege literals with named constants. Added picker and modal regressions. Verification: 24 picker tests passed, 49 Console settings model tests passed, 45 Console rail/popover tests passed, the targeted provider 400 recovery test passed, and the file-tool suite passed 22 tests with one explicit symlink-capability skip. Scoped Ruff and git diff --check passed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->
<!-- SECTION:FINAL_SUMMARY:END -->
