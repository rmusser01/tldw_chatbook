---
id: TASK-32191
title: Preserve guided Web Search input and save completion across navigation
status: Done
assignee:
  - '@codex'
created_date: '2026-09-10 00:32'
updated_date: '2026-09-10 00:47'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Configuring or rotating search-backend credentials must preserve typed input and finish an explicit save even when the user navigates away. Reopened Settings must show the same draft, selected backend and accurate save state.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Queued input is preserved when asynchronous setup or test status refreshes occur, including masked credentials.
- [x] #2 An explicit Web Search save completes across category and Settings destination navigation; restored state reflects the actual outcome.
- [x] #3 Edits and clears retain their correct baseline across navigation, and save conflicts or failures preserve unsaved work without leaking secrets.
- [x] #4 Backend selection and draft state survive Settings recreation; test evidence remains invalidated after navigation.
- [x] #5 Mounted regressions, targeted compatibility checks, independent review and documentation establish the final behavior.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce pending-input loss and interrupted save completion through the mounted production Settings panel before changing implementation.
2. Preserve the live guided-search session in the existing in-memory screen-state store, rebind callbacks safely, and run persistence with app lifetime. Retain selected backend and baseline without a second store.
3. Synchronize pending widget values before async result/save decisions; preserve input during status-only refresh and apply results only to the captured configuration and revision.
4. Verify successful and failed saves across destination recreation, queued masked input, selection/clear retention, and stale test evidence. Run targeted raw/guided Settings compatibility and independent review; update the walkthrough and review ledger.

ADR required: no
ADR path: backlog/decisions/012-provider-credential-settings-boundary.md and backlog/decisions/033-settings-commit-models-three-honestly-labeled.md
Reason: repair guided draft/async ownership within the existing staged-settings contract and memory-only screen-state owner, using the already established raw-editor lifetime pattern; no new persistence, permission, or provider boundary.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Guided Web Search now retains its live memory-only session through Settings recreation, so the selected backend, masked draft, original conflict baseline, and save outcome survive navigation. Persistence and explicit probes use app-owned workers; view callbacks are safely rebound and stale test evidence is invalidated on new-view mounting as well as teardown.

Pending widget input is captured before status refresh, save/test decisions, Clear, and Revert. A committed write with reload failure reports Saved to disk / Restart and retains the committed editing baseline; failed writes retain the unsaved draft with a credential-safe message. No new visual structure or provider behavior was introduced.

Validation: initial mounted regressions reproduced four input/navigation/save failures. Additional regressions reproduced probe callback-order invalidation, Clear/Revert intent loss, and post-write reload misreporting; all now pass. Final targeted run: 84 passed in 122.07s across test_settings_web_search_lifecycle, test_settings_web_search, test_settings_raw_draft, test_settings_footer_hints, test_screen_state_store, two state-round-trip/malformed-state nodes in test_settings_configuration_hub, and test_probe_import_provenance. Independent re-review: nine lifecycle cases passed, no remaining findings. Full Ruff and formatting passed on both guided modules and the new test; scoped static analysis and changed-range formatting passed on the legacy Settings screen; git diff --check passed. One existing RequestsDependencyWarning remains. No full suite, live provider calls, or participant study was performed.

Updated the Settings walkthrough, local Web Search surface contract, original UX review follow-through, and the Textual lessons incident. Core files: settings_web_search.py, settings_web_search_panel.py, settings_screen.py; regressions: Tests/UI/test_settings_web_search_lifecycle.py. Self-review and independent review completed. Changes remain uncommitted in the isolated codex/shared-search-backend-default worktree.

ADR required: no. Existing ADR-012 (backlog/decisions/012-provider-credential-settings-boundary.md) and ADR-033 (backlog/decisions/033-settings-commit-models-three-honestly-labeled.md) cover this repair to staged settings and existing memory-only navigation ownership. No plan deviation or new permission, persistence, dependency, or provider boundary.

PR integration: moved onto dev 86a8054edb, preserving current TLS, profile/footer, privacy and config publication behavior. Final evidence and baseline limitations: Docs/superpowers/reviews/2026-09-09-search-settings-pr-integration.md (525 targeted cases passed across two runs, three live cases skipped).
<!-- SECTION:NOTES:END -->
