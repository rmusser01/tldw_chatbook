---
id: TASK-15200
title: 'Fleet panel: historical rows lack elapsed time; stuck/cancelled rows unstyled'
status: Done
assignee:
  - '@codex'
created_date: '2026-08-11 04:01'
updated_date: '2026-09-08 05:01'
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
- [x] #1 Historical/resumed fleet rows show an elapsed time derived from the run's DB timestamps
- [x] #2 stuck and cancelled rows have their own color variants, distinct from running and from the default foreground
- [x] #3 A historical/resumed row's secondary line is restored from the sub-agent run's own persisted steps JSON — the same shape _summarize_persisted_step already reads for the primary's historical steps; _derive_historical_snapshot never reads it for subagent records and both historical row builders hardcode secondary_text=''
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing real-DB/bridge and compositor tests for child-owned detail, approximate elapsed timestamps, invalid/unfinished timestamps, and stuck/cancelled styling.
2. Extend SubAgentSummary with saved timestamps and bounded detail; share one historical-record projection between cached summaries and raw-row fallback. Preserve existing budget tokens and run IDs.
3. Display valid terminal timestamp spans with ~ to distinguish wall-clock bookkeeping from live monotonic duration. Use semantic warning/muted colors for stuck/cancelled in component CSS and regenerate the stylesheet.
4. Verify targeted bridge/fleet/history/component UI tests, rendered words and colors, scoped lint/format, and the CSS build. Update guide, review ledger, and notes.
ADR required: no
ADR path: N/A
Reason: Routine restoration of existing fleet presentation under Docs/superpowers/specs/2026-08-08-supervisor-agent-fleet-design.md section 7 and ADR-017/043; no storage, execution, or authority change. Timestamp approximation is explicit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
2026-08-11, PR2b Task 6 live verification: the gap is BROADER than this task's original scoping. It is not only historical/resumed (post-restart) rows — the moment the WHOLE TURN ends, fleet_snapshot() empties and every row in that run reverts, same session and seconds later, from the live rendering (elapsed + secondary line + token count) to the sparse historical one (name/task only). Observed live: rows that had shown elapsed, result text and a token figure lost all three within seconds of the reply completing. So restoring this detail from the run DB fixes the common same-session case, not just the restart case. Docs corrected to describe the transience honestly rather than claiming durable token spend.

2026-09-07 repair: historical summaries now carry saved timestamps, budget counters, and bounded child-owned result/last-step detail. Cached and raw-record row builders share the same projection. Valid terminal timestamp spans render with ~ because updated_at includes bookkeeping; missing, invalid, reversed, and unfinished spans are omitted. Stuck/cancelled primary text uses warning/muted semantic colors. Updated component and generated CSS, Console guide, and the orchestration review ledger.

Verified with real SQLite/bridge regressions and rendered Console checks at 180x48 and 120x35. The combined TASK-15200/15201 pass has 398 passing targeted tests across disjoint DB/bridge, fleet/agent UI, historical/parallel UI, and component/CSS groups. New files pass Ruff/format; changed production ranges are formatted and add no lint findings versus pass start; scoped whitespace and diagnostic inventory checks pass. Self-review fixed double truncation and tested actual painted detail. No full suite or live provider used.

ADR required: no; this restores existing presentation under ADR-017/043 and supervisor fleet spec section 7, without a schema or authority change. Navigation in the paired task follows ADR-132. Validation limitation: existing TASK-3070 screen-size gate remains red (17,600 lines before this pass; 17,601 after its history callback, budget 17,570; unchanged 591 methods). The limit was not raised. Changes remain uncommitted.
<!-- SECTION:NOTES:END -->
