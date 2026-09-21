---
id: TASK-32882
title: Keep MCP Test Tool reachable in compact inspectors
status: Done
assignee:
  - '@codex'
created_date: '2026-09-21 05:48'
updated_date: '2026-09-21 07:37'
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
- [x] #1 Keyboard users can reach and read Test Tool, argument fields, permission preview, Run and Close in dark and light themes at 80x24, with wide behavior preserved.
- [x] #2 Resize and focus changes preserve entered arguments, selected tool identity and existing permission or execution behavior.
- [x] #3 Targeted regressions and real native error/retry journeys qualify compact reachability and clean private-profile shutdown; separate visual approval gates the layout PR.
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
5. Current-dev QA compatibility: PR2741 removed validate_username, which the shared native CLI still imports. Preserve the exact documented tmux identifier contract through a strict shared input-validation helper; retain all profile/path admission checks and verify the existing cross-runner invalid-CLI/positive-profile cases before native qualification.
6. Address PR2770 Qodo validation feedback by using the shared input-validation boundary for tmux names; preserve exact ASCII/length/leading-character admission, cover non-coercion and newline rejection, and rerun native admission checks. Owner approved the compact layout on 2026-09-21; this QA-only repair does not change it.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented token-backed inspector scrolling, natural form height and wrapped actions; deferred resize reveal preserves the current focused editor and arguments. The 510-case initial inventory and four clean native dark/light compact/wide journeys qualified the layout, approved by the owner on 2026-09-21. PR2770 Qodo feedback is addressed by a strict shared Pydantic tmux identifier validator preserving the exact ASCII/length/leading-character contract, without coercion or trailing-newline acceptance. All 153 direct/admission/fixture cases and eight import checks pass; pre-existing validation AST and all other captured sources remain unchanged. Existing ADR-150/161 apply; no new ADR. Evidence: Docs/superpowers/qa/2026-09-21-mcp-inspector-compact/README.md. Remains In Progress pending current-head CI, review and current-dev merge checks.
<!-- SECTION:NOTES:END -->
