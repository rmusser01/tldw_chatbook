---
id: TASK-915
title: 'Sticky manual collapse of Agent rail section while fleet busy'
status: To Do
assignee: []
created_date: '2026-07-27 03:55'
labels: [console, fleet-ux, polish]
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Agent rail section auto-opens whenever the fleet summary has content (parallel-agents train). Manually collapsing it while the fleet is busy holds only until the next agent-section payload change, which re-forces it open; the persisted preference is honored again once the fleet quiets and is never corrupted. Add a transient "user dismissed during this busy window" flag so the collapse sticks until the fleet quiets or a new run starts.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Collapsing the Agent section during a busy fleet sticks across payload changes within that busy window.
- [ ] #2 Auto-open still triggers for a newly busy fleet after quiet; persisted preference still never overwritten by the transient force.
<!-- AC:END -->
