---
id: TASK-1143
title: 'Screen navigation silently kills the agent fleet'
status: To Do
assignee: []
created_date: '2026-07-27 18:05'
labels: [console, ux, agents, uat]
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
UAT (Docs/superpowers/qa/parallel-agents-uat-2026-07-27, F5): navigating away from Console (e.g. to Settings to change the run cap) unmounts the screen, shuts down the controller, and denies every in-flight/parked run — by design (instance lifecycle) — but nothing warns before, and nothing reports after: returning shows a fresh Console with no markers, toasts, or record of the killed runs. Users running parallel background agents lose them by opening Settings. Add a confirm-on-navigate when the fleet is busy, and/or a returning notice ("N runs were cancelled when you left Console"), and document the lifecycle in the user guide.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Navigating away with in-flight or parked runs either asks for confirmation or leaves a visible record on return.
- [ ] #2 Never auto-approves; deny-on-teardown semantics unchanged.
- [ ] #3 User guide documents that runs are Console-screen-scoped.
<!-- AC:END -->
