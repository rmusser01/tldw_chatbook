---
id: TASK-31664
title: Align style picker tests with split CSS and reachable rows
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 17:52'
updated_date: '2026-09-05 18:01'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore real styled picker verification after CSS ownership and scrollable row layout changes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Style selectors are checked across maintained source and all generated app-owned stylesheets.
- [x] #2 Style insertion tests perform a successful real click on the visible target row and retain exact draft and command assertions.
- [x] #3 Full style picker tests and scoped static checks pass without product changes.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce dev CSS pin failures and real-click failure; instrumentation showed the target row at y37 and Pilot.click returned False inside the scrollable picker. 2. Check generated selector ownership through existing app_css_text union; explicitly scroll style insertion target rows into view and assert the real click succeeds. Keep bare modal interaction unit harness and all exact draft/parser/dismissal assertions. 3. Run full style picker file and scoped lint/format plus self-review. ADR required: no. ADR path: backlog/decisions/097-boot-budget-ratchets.md (existing). Reason: test-only alignment with split stylesheet ownership and real scrollable row interaction; no runtime layout or behavior changes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Replaced stale boot-bundle-only selector pin with existing app_css_text union of generated app-owned sheets; every original selector remains required in both maintained source and generated output. Real-CSS row was clipped: Pilot.click returned False. Four screen insertion scenarios now scroll the target row into view and assert the actual click succeeds, retaining exact dismissal/draft/parser checks. Bare modal behavior harness unchanged. Original failures reproduced on latestdev; final full style picker file31passed27.99s; full-file Ruff lint and changed-region formatting pass; self-review/diff check clean. No product changes or new ADR; existing ADR-097 stylesheet ownership applies.
<!-- SECTION:NOTES:END -->
