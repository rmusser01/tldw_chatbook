---
id: TASK-15200
title: 'Fleet panel: historical rows lack elapsed time; stuck/cancelled rows unstyled'
status: To Do
assignee: []
created_date: '2026-08-11 04:01'
updated_date: '2026-08-11 14:05'
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
- [ ] #3 A historical/resumed row's secondary line is restored from the sub-agent run's own persisted steps JSON — the same shape _summarize_persisted_step already reads for the primary's historical steps; _derive_historical_snapshot never reads it for subagent records and both historical row builders hardcode secondary_text=''
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
2026-08-11, PR2b Task 6 live verification: the gap is BROADER than this task's original scoping. It is not only historical/resumed (post-restart) rows — the moment the WHOLE TURN ends, fleet_snapshot() empties and every row in that run reverts, same session and seconds later, from the live rendering (elapsed + secondary line + token count) to the sparse historical one (name/task only). Observed live: rows that had shown elapsed, result text and a token figure lost all three within seconds of the reply completing. So restoring this detail from the run DB fixes the common same-session case, not just the restart case. Docs corrected to describe the transience honestly rather than claiming durable token spend.
<!-- SECTION:NOTES:END -->
