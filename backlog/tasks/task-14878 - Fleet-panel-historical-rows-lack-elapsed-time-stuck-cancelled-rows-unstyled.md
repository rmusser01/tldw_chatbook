---
id: TASK-14878
title: 'Fleet panel: historical rows lack elapsed time; stuck/cancelled rows unstyled'
status: To Do
assignee: []
created_date: '2026-08-11 04:01'
labels: []
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Two cosmetic gaps disclosed and accepted during supervisor-fleet PR 2b Task 4. (1) A fleet row's elapsed segment renders only for LIVE rows, which carry FleetHandle timestamps; historical/resumed rows show none. The data is not missing at the source — AgentRunsDB.get_run()/list_runs() return created_at/updated_at (SELECT *) — it is _derive_historical_snapshot that drops them when building SubAgentSummary. Fix by threading created_at/updated_at onto SubAgentSummary and computing a wall-clock elapsed for historical rows. (2) Row status color variants exist for running/done/error/blocked only; stuck and cancelled fall through to plain $ds-text-primary. Reviewer confirmed this does not misread as running (the accent $ds-status-running is visually distinct from the default foreground, and each status carries its own glyph: running ●, stuck ⚠, cancelled ✗), so it is genuinely cosmetic — but stuck is an attention-worthy state and deserves its own treatment.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Historical/resumed fleet rows show an elapsed time derived from the run's DB timestamps
- [ ] #2 stuck and cancelled rows have their own color variants, distinct from running and from the default foreground
<!-- AC:END -->
