---
id: TASK-133
title: Polish Console setup-state usability
status: Done
assignee:
- '@codex'
labels:
- console
- ux
- uat
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Improve the Console blocked/setup state so users can understand why chat is unavailable, what action restores it, and how workspace authority applies without relying on scattered text or disabled controls.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Console setup-required state groups provider problem, impact, and recovery action in one coherent callout.
- [x] #2 Transcript empty state teaches the next useful actions instead of showing only a sparse placeholder.
- [x] #3 Left workspace/context authority information is structured into scannable rows or badges without hiding key limits.
- [x] #4 Inspector collapsed/setup state is purposeful and does not add unreadable rail noise.
- [x] #5 Composer disabled controls expose setup-required recovery clearly enough for keyboard and visual users.
- [x] #6 Changes are covered by focused mounted regressions and verified with an actual CDP/Textual-web screenshot.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add focused mounted regressions for the setup-required Console state: recovery copy/action grouping, transcript empty-state guidance, workspace authority row labels, inspector setup summary, and disabled composer reason.
2. Update Console display-state copy and composition only; avoid changing provider, workspace, persistence, send, or message-action behavior.
3. Keep the approved three-column shell and terminal-grid visual style, but reduce redundant border noise where nested frames do not communicate state.
4. Rerun the new focused regressions plus relevant Console layout/session-settings tests.
5. Capture an actual Textual-web/CDP screenshot and keep the task open until the rendered state is approved.

ADR required: no.
ADR path: N/A.
Reason: This is a focused Console UI polish and regression-harness change. It does not change storage/schema, sync policy, provider/runtime boundaries, service contracts, security policy, or long-lived product architecture.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added setup-state regressions for provider recovery grouping, instructional empty transcript copy, structured workspace authority rows, inspector setup guidance, and setup-aware composer copy.
- Updated Console setup UX so provider blockers show a single problem/impact/action callout, empty transcripts teach the next steps, the composer names the required recovery, and the inspector exposes setup/send-blocked/recovery rows when expanded.
- Reworked workspace authority metadata into explicit label/value rows and normalized the ACP handoff label to a user-facing `Handoff` row while preserving ACP task/run detail in the value.
- Verification: `python -m pytest -q Tests/UI/test_console_native_chat_flow.py --tb=short` passed, `python -m pytest -q Tests/UI/test_console_session_settings.py -k "provider or model or endpoint or credential or generation or summary" --tb=short` passed, and `git diff --check` passed.
- CDP/Textual-web evidence captured for visual approval:
  `Docs/superpowers/qa/console-uat-parallelization/task-133-console-setup-polish-cdp-2026-06-22.jpg`
  `Docs/superpowers/qa/console-uat-parallelization/task-133-console-setup-polish-inspector-cdp-2026-06-22.jpg`
- Visual approval received on 2026-06-22.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Console setup-state usability now presents a coherent recovery path across the provider callout, transcript empty state, workspace authority panel, inspector, and composer. The work is covered by mounted regressions, focused session-settings verification, and approved CDP/Textual-web screenshots.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
