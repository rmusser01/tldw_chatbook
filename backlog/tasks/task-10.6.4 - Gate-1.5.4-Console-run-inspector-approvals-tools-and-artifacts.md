---
id: TASK-10.6.4
title: 'Gate 1.5.4: Console run inspector approvals tools and artifacts'
status: Done
assignee: []
created_date: '2026-05-07 03:37'
labels:
  - product-maturity
  - phase-3-knowledge-study
  - gate-1-5-console
dependencies:
  - TASK-10.6.3
documentation:
  - Docs/superpowers/specs/2026-05-06-screen-design-adaptation-audit-design.md
  - >-
    Docs/superpowers/plans/2026-05-07-gate-1-5-console-internals-decomposition.md
parent_task_id: TASK-10.6
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Move run tool approval RAG and artifact state into the Console inspector/action model so operators can understand and recover live agentic work.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Run inspector shows live-work provenance tool readiness approval status RAG/source state artifact availability and recovery actions from Console display-state seams.
- [x] #2 Approval tool-call and Chatbook artifact actions remain reachable from Console with target-specific disabled reasons when unavailable.
- [x] #3 Mounted tests cover blocked provider missing RAG/source and pending approval/artifact states.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Extend Console display-state seams with a typed run-inspector/action model for live work provider/tool readiness RAG/source status approvals and Chatbook artifact availability.
2. Add failing unit regressions for the inspector state and disabled action reasons in Tests/Chat/test_console_display_state.py.
3. Add mounted regressions for blocked provider missing RAG/source pending approval and Chatbook artifact states in the existing Console UI test coverage.
4. Implement the ConsoleRunInspector widget and wire it into ChatScreen without replacing the transcript/composer internals.
5. Preserve existing route IDs and action handlers while adding target-specific unavailable reasons.
6. Run focused verification and update task notes/QA evidence before commit/PR.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Added a typed Console inspector action model with explicit enabled state and target-specific disabled reasons for approval review tool-call review and Chatbook save/open actions.
- Added the native `ConsoleRunInspector` widget and wired ChatScreen to derive provider readiness tool count approval count RAG/source state and Chatbook artifact availability from current Console seams.
- Preserved the existing Console transcript and composer surfaces while making the live-work action structurally reachable in the mounted TUI layout.
- Extended Chatbook artifact live-work primary actions so Console can route artifact launches back to Artifacts without inventing a new persistence path.
- Replaced fixed Artifacts Chatbook launch waits with selector-based waits in the touched mounted live-work tests.
- Verification: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Chat/test_console_display_state.py Tests/UI/test_console_internals_decomposition.py Tests/UI/test_console_live_work_handoffs.py Tests/UI/test_chat_approvals_and_resume.py --tb=short` passed with 75 tests and known warnings; `git diff --check` passed.
<!-- SECTION:NOTES:END -->
