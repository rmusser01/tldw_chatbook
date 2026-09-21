---
id: TASK-32882
title: Keep MCP Test Tool reachable in compact inspectors
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-21 05:48'
updated_date: '2026-09-21 05:48'
labels:
  - mcp
  - ui
  - accessibility
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Terminal users must be able to inspect a selected tool, edit its arguments, review permission and run or close the test without controls falling outside the compact viewport. Existing native 80x24 evidence cannot reach Test Tool.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Keyboard users can reach and read Test Tool, argument fields, permission preview, Run and Close in dark and light themes at 80x24, with wide behavior preserved.
- [ ] #2 Resize and focus changes preserve entered arguments, selected tool identity and existing permission or execution behavior.
- [ ] #3 Targeted regressions and real native error/retry journeys qualify compact reachability and clean private-profile shutdown; separate visual approval gates the layout PR.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no new ADR.
ADR path: backlog/decisions/161-component-pattern-library.md (ADR-150 design language also applies).
Reason: restore keyboard reachability in the existing MCP inspector, preserving its current layout and service/permission boundaries.
1. Reproduce compact Test Tool and argument-field clipping on the current merged runtime with real-app paint, clipping and focus checks; inspect existing scroll owners and token-backed layout.
2. Give the inspector a usable scrolling viewport using the existing Textual container and design tokens. Keep focus and arguments stable through size changes; preserve tool, permission and execution authority.
3. Run targeted reachability and neighboring inspector regressions, static/derived-artifact guards and independent review.
4. Reuse the admitted private real-stdio error/retry journey for fresh dark/light compact and wide native evidence with full labels, hit targets, source provenance and clean lifecycle checks. Present a bounded follow-up PR and its own visual approval after PR2769 closes.
<!-- SECTION:PLAN:END -->
