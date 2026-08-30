---
id: TASK-21513
title: Daily Reports surface and demo
status: Done
assignee: []
created_date: '2026-08-29 22:08'
updated_date: '2026-08-30 05:57'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Surface scheduled watchlist briefings as 'Daily Reports' on the Artifacts screen, notify on scheduled briefing completion, and add a one-click live demo that seeds a real Daily Brief watchlist (RSS sources, preset, daily cadence) and runs it immediately - text brief plus TTS audio when a voice profile exists. Spec: Docs/superpowers/specs/2026-08-29-daily-reports-demo-design.md; ADR-079.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Artifacts Reports slot lists recent briefings across watchlists with play/open actions and an empty-state demo CTA
- [x] #2 Scheduled briefing completion dispatches a 'briefing' notification through NotificationDispatchService
- [x] #3 One-click demo seeds watchlist+sources+preset+24h cadence idempotently and generates a text brief live
- [x] #4 Demo synthesizes audio when a TTS voice profile + pydub exist; otherwise skips audio with a Settings hint and still succeeds
- [x] #5 Watchlists screen shows a dismissible demo banner only while no briefing schedule exists
- [x] #6 No new tables, columns, or dependencies
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

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented per Docs/superpowers/plans/2026-08-29-daily-reports-demo.md; spec Docs/superpowers/specs/2026-08-29-daily-reports-demo-design.md; ADR-079. Live verification passed on scratch profile /tmp/daily-reports-verify (pane captures run6-*.txt): demo seeded watchlist+3 RSS sources, live DeepSeek brief complete (40 items), briefing notifications recorded, audio skipped by design, banner absent after seeding, persistence across relaunch, isolation held (0 writes to real data dir). Findings: stored Anthropic key invalid at provider; deepseek-v4-flash default model burns BRIEFING_MAX_TOKENS on reasoning -> empty-content failures. Follow-up filed: TASK-21514. All AC checked.
<!-- SECTION:NOTES:END -->
