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
Implemented Runs Refresh and Re-run feedback across `RunsPane`, the Watchlists collections screen, and backend controller seams. Refresh uses staged, generation/backend-guarded publication with identity-pinned selection and transient-failure snapshot retention; Re-run carries backend-specific source/job identity, shares the canonical local operation guard, preserves independent server targets, and keeps busy/toast outcomes honest. Final integration fix `63e32bda924c4a4345e73bfc6fe715ae93b4a7b4` makes the authority split explicit: authoritative list/load/explicit Refresh intents own the list-publication token and pending authority, accepting them invalidates older ticks, periodic ticks use a separate newest-wins epoch with backend/selection ABA guards and pending suppression, and backend transitions invalidate both authorities.

Final integration verification evidence at `63e32bda924c4a4345e73bfc6fe715ae93b4a7b4`: `pytest -q Tests/Watchlists/test_watchlists_runs_pane.py` — 37 passed, 1 warning, exit 0; `pytest -q Tests/Watchlists/test_watchlists_backend_controller.py Tests/Watchlists/test_watchlist_scope_service.py` — 48 passed, 1 warning, exit 0; `pytest -q Tests/UI/test_watchlists_run_detail.py` — 49 passed, 2 warnings, exit 0; `pytest -q Tests/UI/test_watchlists_check_now_progress.py` — 10 passed, 2 warnings, exit 0; `pytest -q Tests/UI/test_watchlists_check_now_skipped.py Tests/UI/test_watchlists_check_now_failure.py` — 24 passed, 2 warnings, exit 0; exact Task-4/final rerun/check nodes in `test_watchlists_destination_shell.py` — 14 passed, 2 warnings, exit 0. Total: 182 passed, 0 failed, exit 0. Ruff check on all eight modified Python files passed (exit 0); Ruff format check reports all eight already formatted (exit 0). `git diff --check origin/dev...HEAD` passed (exit 0).

Post-rebase Qodo follow-up validated and fixed two findings. Runs refresh/tick publication now also requires Runs to remain the active section, so delayed work cannot replace a newer Source or other Inspector selection; a gated mounted regression proved the bug RED before the one-guard fix and GREEN afterward. The four new public Runs watchers now document their parameters in Google style. The final focused gate is 183 passed, 0 failed, and 3 warnings (the prior 182 cases plus the navigation-race regression), exit 0; the 87 directly affected run-detail/pane cases also pass. Ruff lint and format checks pass for the changed Python files, and branch diff checks pass. The remaining dependency, descriptor-count, and temporary cleanup warnings are pre-existing test-environment noise rather than failures.

The prior broad exploratory `-k "runs or run_detail or rerun or check_now or launch_run"` selection is not a gate: it reports 1 failed, 139 passed, 120 deselected, and 2 warnings; the sole failure is `test_leaving_runs_clears_pending_run_deep_link`, an unchanged unmounted reactive `NoActiveAppError` baseline. The exact 14 changed Task-4/final destination-shell nodes above are the gate and pass. Mechanical formatting is isolated in `2220218666` (`style: format Watchlists runs files`), with 558 insertions/452 deletions and AST_UNCHANGED=True for all eight files compared with pre-format HEAD. ADR required: yes; `backlog/decisions/042-watchlists-reader-first-ia.md` was amended on 2026-08-27 to record canonical local identity, distinct server source/job namespaces, existing controller/scope forwarding, and screen-owned concurrency and publication authority, including the final list/tick authority split.

The required Derived Artifacts check exposed a stale production-diagnostic inventory digest for `watchlists_collections_screen.py`. Review confirmed the call count remains 87, the eight statement deltas are the intended refresh/re-run refactor or formatting changes, and the new dynamic refresh helper receives only five fixed internal strings—no user content, secrets, paths, or URLs. `Docs/security/production-diagnostic-inventory.json` was regenerated only after that statement-level review.
<!-- SECTION:NOTES:END -->
