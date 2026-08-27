---
id: TASK-2331
title: Runs toolbar Refresh reloads and Re-run gives feedback
status: In Progress
assignee:
  - '@codex'
created_date: '2026-08-04'
updated_date: '2026-08-27 21:49'
labels:
  - watchlists
  - bug
dependencies: []
references:
  - >-
    Docs/superpowers/specs/2026-08-27-watchlists-runs-refresh-rerun-feedback-design.md
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
- [ ] #1 Refresh reloads authoritative run rows and the selected run's fresh detail while preserving that selection by identity when it still exists.
- [ ] #2 Refresh clears an authoritatively deleted selection, retains the complete mounted snapshot on transient failure, and never publishes results after a backend switch.
- [ ] #3 Re-run source immediately exposes a disabled `Re-running...` state and reports honest local-complete, server-started, skipped, and failure outcomes consistent with TASK-2309.
- [ ] #4 Local Check now and Re-run source share one canonical source guard; server Re-run uses and deduplicates by its required job identity; different targets remain independent.
- [ ] #5 Discriminating affected Watchlists tests, modified-file Ruff, and `git diff --check` pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add typed Runs Refresh/Re-run intents and non-recomposing busy presentation with focused pane tests.
2. Forward local `source_id` and server `job_id` through the existing controller/scope launch seam.
3. Implement generation-checked staged Runs refresh with selection pinning, narrow not-found handling, backend guards, and grouped detail publication.
4. Unify local Check-now/Re-run concurrency identity, preserve distinct server source/job namespaces, and add honest Re-run feedback/cleanup.
5. Run only affected Watchlists tests, modified-file Ruff and formatter checks, branch diff checks, self-review, and Backlog completion hygiene.

ADR required: no

ADR path: `backlog/decisions/042-watchlists-reader-first-ia.md`

Reason: the existing Watchlists reader-first screen and pane boundaries already govern this repair; no persistence, service ownership, backend API, dependency, or navigation architecture changes.
<!-- SECTION:PLAN:END -->
