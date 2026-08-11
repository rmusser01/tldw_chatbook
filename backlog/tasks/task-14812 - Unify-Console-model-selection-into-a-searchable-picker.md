---
id: TASK-14812
title: Unify Console model selection into a searchable picker
status: In Progress
created_date: 2026-08-10 21:52
dependencies:
- TASK-3600
labels:
- console
- models
- ux
assignee:
- '@codex'
updated_date: 2026-08-10 22:30
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

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Reworked `ModelSearchPicker` into the controlled, keyboard-first model control used by both Alt+M and full Console settings. It loads each provider's uncapped endpoint-scoped catalog in a worker, filters only in memory, preserves the committed selection during search/cancel, and provides explicit loading, empty, unavailable, unlisted-current, no-match, and custom-ID states.
- Routed provider switches, catalog selection, custom IDs, focus recovery, apply, and manual endpoint discovery through the shared picker. Existing Select/Input controls remain hidden compatibility adapters for established validation and draft logic.
- Added a provider-scoped overlay for results from the full modal's explicit unsaved-endpoint probe, so probed models are immediately searchable without another catalog request.
- Made the Alt+M popover vertically bounded and scrollable after mounted geometry testing showed an open result list grew to 34 rows on a 24-row terminal.
- Added mounted coverage for keyboard selection and Escape, one-load filtering, query-during-load recovery, explicit states, custom IDs, provider isolation, manual-discovery overlays, shared-modal save behavior, and compact-terminal geometry. Mutation checks proved the no-request-per-key and cross-provider-reset tests fail when their guards are removed, then passed after restoration.
- Verification: 19 picker tests passed; 45 Console rail/popover tests passed; the broader full-modal `-k model` slice passed all 49 tests; the 85-test catalog/resolver/picker set and all 144 gateway tests passed. Scoped Ruff, fatal-error Ruff for the legacy screen module, and `git diff --check` passed. Mutation checks proved the no-request-per-key and cross-provider-reset tests fail when their guards are removed, then pass after restoration.
- The blocking-I/O architecture suite now passes all six tests after its repository scanner was corrected to decode UTF-8 source explicitly on Windows. A bounded full-suite run collected 8,584 tests but reached only 3% in 15 minutes, with nine failures before timeout; the first reproduced independently as an unrelated host-level `WinError 1314` because this Windows account cannot create the symlink required by the file-tool test. Repository-wide completion remains blocked by that host constraint and the pre-existing screen-size ratchet, which reports `chat_screen.py` at 19,743 lines versus its 17,727 budget even though this patch removes three lines. TASK-3600 is still in progress, so this task remains In Progress despite its acceptance criteria being met.
- ADR required: no. Existing ADR-020 remains the governing catalog-authority and fallback decision.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
