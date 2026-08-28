---
id: TASK-2331
title: Runs toolbar Refresh reloads and Re-run gives feedback
status: Done
assignee:
  - '@codex'
created_date: '2026-08-04'
updated_date: '2026-08-28 03:04'
labels:
  - watchlists
  - bug
dependencies: []
references:
  - >-
    Docs/superpowers/specs/2026-08-27-watchlists-runs-refresh-rerun-feedback-design.md
  - Docs/superpowers/plans/2026-08-27-watchlists-runs-refresh-rerun-feedback.md
  - backlog/decisions/042-watchlists-reader-first-ia.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Observed live during UAT batch-2 verification and confirmed in review
(`runs_pane.py:396-397`): the Runs toolbar's Refresh button only re-arms the
action buttons (`_update_action_buttons`) — it never reloads the runs list.
"Re-run source" does run the check but gives no visible feedback that it
started or finished (compounded by Refresh being dead). Pre-existing before
the batch-2 branch; now more visible since run rows carry real accounting.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Refresh reloads authoritative run rows and the selected run's fresh detail while preserving that selection by identity when it still exists.
- [x] #2 Refresh clears an authoritatively deleted selection, retains the complete mounted snapshot on transient failure, and never publishes results after a backend switch.
- [x] #3 Re-run source immediately exposes a disabled `Re-running...` state and reports honest local-complete, server-started, skipped, and failure outcomes consistent with TASK-2309.
- [x] #4 Local Check now and Re-run source share one canonical source guard; server Re-run uses and deduplicates by its required job identity; different targets remain independent.
- [x] #5 Discriminating affected Watchlists tests, modified-file Ruff, and `git diff --check` pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add typed Runs Refresh/Re-run intents and non-recomposing busy presentation with focused pane tests.
2. Forward local `source_id` and server `job_id` through the existing controller/scope launch seam.
3. Implement generation-checked staged Runs refresh with selection pinning, narrow not-found handling, backend guards, and grouped detail publication.
4. Unify local Check-now/Re-run concurrency identity, preserve distinct server source/job namespaces, and add honest Re-run feedback/cleanup.
5. Run only affected Watchlists tests, modified-file Ruff and formatter checks, branch diff checks, self-review, and Backlog completion hygiene.

ADR required: yes

ADR path: `backlog/decisions/042-watchlists-reader-first-ia.md`

Reason: the repair establishes a cross-module/runtime identity and publication boundary: canonical local subscription keys, distinct server source/job namespaces, controller forwarding through the existing scope seam, and screen-owned concurrency plus monotonic Runs publication/selection authority. ADR-042 was amended to record it without introducing a new backend API, storage, or navigation architecture.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented Runs Refresh and Re-run feedback across `RunsPane`, the Watchlists collections screen, and backend controller seams. Refresh uses staged, generation/backend-guarded publication with identity-pinned selection and transient-failure snapshot retention; Re-run carries backend-specific source/job identity, shares the canonical local operation guard, preserves independent server targets, and keeps busy/toast outcomes honest.

Post-format verification evidence: `pytest -q Tests/Watchlists/test_watchlists_runs_pane.py` — 37 passed, 1 warning, exit 0; `pytest -q Tests/Watchlists/test_watchlists_backend_controller.py Tests/Watchlists/test_watchlist_scope_service.py` — 48 passed, 1 warning, exit 0; `pytest -q Tests/UI/test_watchlists_run_detail.py` — 48 passed, 2 warnings, exit 0; `pytest -q Tests/UI/test_watchlists_check_now_progress.py` — 10 passed, 2 warnings, exit 0; `pytest -q Tests/UI/test_watchlists_check_now_skipped.py Tests/UI/test_watchlists_check_now_failure.py` — 24 passed, 2 warnings, exit 0; exact Task-4 rerun/check nodes in `test_watchlists_destination_shell.py` — 13 passed, 2 warnings, exit 0. Total: 180 passed, 0 failed, exit 0. Ruff check on all eight modified Python files passed (exit 0); Ruff format check reports all eight already formatted (exit 0). `git diff --check origin/dev...HEAD` passed (exit 0).

The prior broad exploratory `-k "runs or run_detail or rerun or check_now or launch_run"` selection is not a gate: it includes eight unchanged unmounted-route deep-link tests with baseline `NoActiveAppError` failures. The exact 13 changed Task-4 destination-shell nodes above are the gate and pass. Mechanical formatting is isolated in `2220218666` (`style: format Watchlists runs files`), with 558 insertions/452 deletions and AST_UNCHANGED=True for all eight files compared with pre-format HEAD. ADR required: yes; `backlog/decisions/042-watchlists-reader-first-ia.md` was amended on 2026-08-27 to record canonical local identity, distinct server source/job namespaces, existing controller/scope forwarding, and screen-owned concurrency/publication authority.
<!-- SECTION:NOTES:END -->
