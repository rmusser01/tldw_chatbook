---
id: TASK-21513
title: Daily Reports surface and demo
status: In Progress
assignee: []
created_date: '2026-08-29 22:08'
updated_date: '2026-08-29 22:14'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Surface scheduled watchlist briefings as 'Daily Reports' on the Artifacts screen, notify on scheduled briefing completion, and add a one-click live demo that seeds a real Daily Brief watchlist (RSS sources, preset, daily cadence) and runs it immediately - text brief plus TTS audio when a voice profile exists. Spec: Docs/superpowers/specs/2026-08-29-daily-reports-demo-design.md; ADR-079.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Artifacts Reports slot lists recent briefings across watchlists with play/open actions and an empty-state demo CTA
- [ ] #2 Scheduled briefing completion dispatches a 'briefing' notification through NotificationDispatchService
- [ ] #3 One-click demo seeds watchlist+sources+preset+24h cadence idempotently and generates a text brief live
- [ ] #4 Demo synthesizes audio when a TTS voice profile + pydub exist; otherwise skips audio with a Settings hint and still succeeds
- [ ] #5 Watchlists screen shows a dismissible demo banner only while no briefing schedule exists
- [ ] #6 No new tables, columns, or dependencies
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Execute Docs/superpowers/plans/2026-08-29-daily-reports-demo.md task-by-task (ADR-079 filed; worktree daily-reports).
2. Read-path: SubscriptionsDB.list_recent_briefings + daily_reports_view.
3. BriefingJobHandler completion notifications (category briefing).
4. Artifacts screen Reports slot + demo CTA.
5. DailyReportDemoService: preflight/seed/live text brief.
6. Demo audio stage with graceful degradation.
7. App wiring + Watchlists banner.
8. Live verification (scratch profile) + close-out.
<!-- SECTION:PLAN:END -->
