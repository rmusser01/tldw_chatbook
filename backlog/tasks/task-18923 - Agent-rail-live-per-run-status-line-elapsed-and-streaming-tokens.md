---
id: TASK-18923
title: 'Agent rail: live per-run status line (elapsed + streaming tokens)'
status: To Do
assignee: []
created_date: '2026-08-19 09:55'
updated_date: '2026-08-19 09:55'
labels:
  - console
  - agents
  - ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Port of hermes-agent's live token-flow spinner idea (2026-08-19 hermes-release review). While a run streams, the Console Agent rail shows only "running · step N" and token totals arrive post-hoc on the cost chip. Add a live status line during streaming: elapsed time plus the tokens received so far for the in-flight reply, updating at most ~1/s (reuse the once-a-second survivor tick timer pattern, task-15664) and tearing down when idle. Extend the same treatment to live children in the fleet panel. Figures must be honest: provider-reported usage where available, else an explicitly-labeled local count — never a fabricated dollar cost.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 During streaming the Agent section shows a live line with elapsed time and the current turn's token count, updating at most once per second and stopping when nothing is live (idle CPU unaffected)
- [ ] #2 Token figures use provider-reported usage or are explicitly labeled approximate/local; no cost estimate is invented for the live figure
- [ ] #3 Live children in the fleet panel show the same live elapsed/token treatment where the child's usage is observable
- [ ] #4 The tick reuses/stops per the survivor-tick discipline: self-stopping when nothing is live, no per-chunk repaint cost
- [ ] #5 Tests pin the render, the 1/s cadence bound, idle teardown, and honest-labeling of non-provider counts
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no.
ADR path: N/A.
Reason: presentation over existing run/usage data; no storage or boundary change.

1. Thread incremental per-turn usage (provider-reported where available) into the run status state
2. Extend the 1/s rail tick to paint the live line for the primary run and live children
3. Tests (cadence, teardown, labeling) + agent-runs-and-tools.md update
<!-- SECTION:PLAN:END -->
