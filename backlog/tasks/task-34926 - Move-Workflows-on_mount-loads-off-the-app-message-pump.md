---
id: TASK-34926
title: Move-Workflows-on_mount-loads-off-the-app-message-pump
status: To Do
assignee: []
created_date: '2026-10-13 14:51'
labels:
  - performance
  - ui-perf
  - wave-3-screen-switch
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
WorkflowsScreen.on_mount awaits _initialize_authoring() — which itself awaits ensure_workflow_authoring() and then a full-screen await self.recompose() DURING mount (workflows_screen.py:126-157) — and then awaits controller.load(). The repo documents this exact trap elsewhere (personas_screen.py:1938: awaiting here ran it on the App's message pump — the app stopped responding). First navigation to Workflows freezes the whole TUI until load completes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 on_mount returns before authoring init and load complete — loads run in workers or call_after_refresh,No await self.recompose() during mount,Saved selection/section/workflow-views state still restored correctly (test)
<!-- AC:END -->
