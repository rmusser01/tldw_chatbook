---
id: TASK-16483
title: Console selection feedback actions phase 3
status: Done
assignee: []
created_date: '2026-08-16 05:27'
updated_date: '2026-08-17 05:33'
labels: []
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Review feedback actions (Request changes / LGTM / Comment) on selections in agent output, routed as the next user message via the prompt queue
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Diff rows support line-granularity selection (unified-diff projection, snap to whole diff lines, reverse-video strip)
- [x] #2 Menu offers Request changes / LGTM / Comment only when the selection sits in agent output (widened per product decision 2026-08-16: assistant prose, tool, and diff rows; USER rows excluded)
- [x] #3 Without an active run, Request changes + LGTM render disabled with a visible hint; Comment stays enabled
- [x] #4 Comment modal collects an optional comment (empty submit sends without a comment; cancel/escape abandons)
- [x] #5 Feedback composes header + quoted selection + optional comment and routes via the prompt queue as the next user message (queues behind an active run; composer draft untouched)
- [x] #6 All selection/feedback/dismissal/transcript suites green; no new failures vs pre-existing baselines
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Per Docs/superpowers/plans/2026-08-15-console-selection-feedback-phase3.md (tasks 1-6: diff-row selection protocol → menu feedback entries + run gating → transcript wiring → comment modal → screen handler + prompt-queue dispatch → wrap-up)
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
DEAD-BUTTON / MESSAGE-SELECTION FOLLOW-UPS (2026-08-16, all user-verified in a real terminal, tip 8c6819606):

- 78cd9aeba: the drag-release Click reached the transcript's on_click with just_finished already consumed (row guard's or-chain skipped consume_release_click and did not stop the event) -> _remove_selection_menu() wiped the selection before menu actions read it ('buttons only work once'; queue-race dependent, first menu of a session usually won). Both row guards now consume BOTH tokens and stop the artifact; the transcript's on_click checks suppression before any dismissal cleanup. Regression: two consecutive drag->ask-side-chat rounds via raw driver-shaped events.
- 86f5807c9: plain clicks' synthesized Click routes to the mouse CAPTURER (the drag-arm captures on press; release happens after the Click was forwarded), so the row never saw clicks and mouse click-to-select never toggled in real terminals (pilot clicks bypass capture). The transcript's on_click re-dispatches capture-routed clicks to the targeted row (event.control walk) with the suppression guard. Regression: real-shaped plain click toggles then untoggles a message.
- 8c6819606: spike instrumentation (mouse event logs, Button.press/App.on_event wraps) removed; lessons recorded in backlog/docs/lessons-live-verification.md (capture reroute + one-shot-token consumption + widget-level event dumps).
<!-- SECTION:NOTES:END -->
