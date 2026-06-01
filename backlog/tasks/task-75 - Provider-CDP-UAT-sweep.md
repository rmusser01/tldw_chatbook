---
id: TASK-75
title: Provider CDP UAT sweep
status: In Progress
assignee: []
created_date: '2026-06-01 00:46'
updated_date: '2026-06-01 00:47'
labels:
  - qa
  - providers
  - console
  - cdp
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Verify every testable Chatbook Console provider through rendered Textual-web/CDP using isolated config and redacted provider credentials.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Runtime provider inventory is extracted from Chatbook code
- [ ] #2 Inventory records model source and local endpoint reachability
- [ ] #3 Hosted providers with usable keys are tested through CDP
- [ ] #4 Local/custom providers are skipped unless endpoint is reachable
- [ ] #5 Each passed provider receives a second assistant reply in the same Console session
- [ ] #6 External failures are classified separately from Chatbook defects
- [ ] #7 Raw API keys do not appear in evidence
- [ ] #8 QA report and residual risks are recorded
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Build redacted provider inventory and isolated CDP launch helpers.
2. Generate provider inventory and QA report skeleton.
3. Launch Chatbook through Textual-web/CDP with isolated HOME/XDG config/data.
4. Run a manual two-turn provider sweep through the rendered app.
5. Fix and rerun only Chatbook-caused provider defects.
6. Record evidence, residual risks, and task closeout.
<!-- SECTION:PLAN:END -->
