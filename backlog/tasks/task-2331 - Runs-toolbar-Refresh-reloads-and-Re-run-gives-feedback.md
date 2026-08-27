---
id: TASK-2331
title: Runs toolbar Refresh reloads and Re-run gives feedback
status: In Progress
assignee:
  - '@codex'
created_date: '2026-08-04'
updated_date: '2026-08-27 21:30'
labels:
  - watchlists
  - bug
dependencies: []
references:
  - Docs/superpowers/specs/2026-08-27-watchlists-runs-refresh-rerun-feedback-design.md
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

- [ ] Refresh reloads authoritative run rows and the selected run's fresh detail while preserving that selection by identity when it still exists.
- [ ] Refresh clears an authoritatively deleted selection, retains the complete mounted snapshot on transient failure, and never publishes results after a backend switch.
- [ ] Re-run source immediately exposes a disabled `Re-running...` state and reports honest local-complete, server-started, skipped, and failure outcomes consistent with TASK-2309.
- [ ] Check now and Re-run source refuse duplicate work for the same canonical source while leaving different sources independent.
- [ ] Discriminating affected Watchlists tests, modified-file Ruff, and `git diff --check` pass.
<!-- AC:END -->
