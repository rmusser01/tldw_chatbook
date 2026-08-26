---
id: TASK-16483
title: Readable bundle and artifact inspection in the Research window
status: Done
assignee:
  - '@robert'
created_date: '2026-08-16 03:31'
updated_date: '2026-08-16 03:35'
labels:
  - research
  - ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The window renders loaded bundles and artifacts as raw JSON dumps, and its first-artifact heuristic assumes the server bundle shape (name-to-content) so local bundles select the run record instead of an artifact. Make run outputs legible: structured rendering for the known artifact types and a sensible default that opens the report.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Bundle summaries render run status plus an artifact inventory for the local bundle shape and a names summary for the server shape instead of raw JSON,Known artifact types render structurally (report markdown as text, verification metrics, budget usage, claims with statuses, sources list) with a pretty-JSON fallback for unknown content,The default artifact after loading a bundle prefers the report and never selects the run record for local bundles,Loading a bundle auto-loads the default artifact so the report is visible immediately,Tests cover both bundle shapes, each structured renderer, the fallback, and the window auto-load path
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. TDD UI/Research_Modules/bundle_rendering.py: pure render_bundle_summary (local run+artifacts shape and server name-to-content shape), render_artifact (structured per known type with pretty-JSON fallback), default_artifact_for_bundle (report-first, never the run record)
2. Wire the window: bundle detail uses the summary renderer, artifact detail uses the structured renderer, and loading a bundle auto-loads the default artifact
3. Tests plus lint plus task close
ADR required: no - presentation-only rendering over existing artifacts
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- New `UI/Research_Modules/bundle_rendering.py` (pure functions, both record shapes tolerated): `render_bundle_summary` (run status/phase/query + artifact inventory for the local shape; name listing for the server name-to-content shape), `render_artifact` (header + structured content: report markdown as capped text, verification metrics incl. citation/gate lines, claims with statuses, sources list, budget usage with the estimated-tokens flag; pretty-JSON fallback), and `default_artifact_for_bundle` (report-first, never the local shape's `run` record — fixing the old first-key heuristic that selected it).
- Window: bundle detail renders the summary; artifact detail renders structurally; `load_selected_run_bundle` auto-loads the default artifact (setting the artifact input when mounted), so Load Bundle immediately shows the report.
- Verified TDD: 6 renderer tests + 1 window auto-load test written first and watched failing; suites 181 passed; ruff clean.
<!-- SECTION:NOTES:END -->
