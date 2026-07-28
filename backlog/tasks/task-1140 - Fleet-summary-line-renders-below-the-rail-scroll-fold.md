---
id: TASK-1140
title: 'Fleet summary line renders below the rail scroll fold'
status: To Do
assignee: []
created_date: '2026-07-27 18:05'
labels: [console, fleet-ux, uat]
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
UAT (Docs/superpowers/qa/parallel-agents-uat-2026-07-27, F1): #console-agent-fleet-summary sits at the bottom of the Agent rail section below the viewed session's status/step bullets, so after any agent run it is off-screen unless the user wheel-scrolls deep into the rail — defeating the spec's "at a glance" intent. Headless proof: region y=48 in a 44-row viewport with an all-True display chain, so the existing render-path test passes while nothing is visible. Move the line to the top of the Agent section (or pin it outside the scrollable flow) and strengthen the test to assert viewport intersection, not just the display chain.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 With a busy fleet and any amount of rail content, the fleet line is visible without scrolling.
- [ ] #2 A test asserts viewport intersection (region within the visible viewport), failing against the current placement.
<!-- AC:END -->
