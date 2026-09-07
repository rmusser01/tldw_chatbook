---
id: TASK-26045
title: Permission-request context summaries
status: Done
assignee: []
created_date: '2026-08-31 22:14'
updated_date: '2026-08-31 23:04'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Advisory rationale + opt-in fast-LLM summaries on Console approval cards per ADR-090
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Model context lines render on approval rows
- [x] #2 External summary fires once per round per mode
- [x] #3 Nothing advisory persists or alters verdicts
- [x] #4 Targeted tests green
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Implement per Docs/superpowers/plans/2026-08-31-permission-request-summaries.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Spec: Docs/superpowers/specs/2026-08-31-permission-request-summaries-design.md; ADR: backlog/decisions/090-permission-request-context-summaries.md

Implemented across 11 commits on feat/permission-request-context-summaries (83ab1ba55..c57d2c531), executed task-by-task with per-task review plus a final whole-branch review (all clean).

- Approach: rationale captured at parse time onto `ToolCall` (explicit fence `rationale` key, else the turn's preamble text), threaded through the existing `MCPPendingCall` → wire payload → `ChatApprovalCard` chain for all three tool owners; external summary is a sync service (`Chat/permission_summary_service.py`) fired once per round from every approval mount/remount path on its own thread, delivered via a guarded UI bridge that patches only the summary line. Both lanes display-only: capped (500 capture / 240 display, tail-biased), control-stripped, escape()-rendered, never persisted, never verdict inputs.
- Key decisions: `MCPPendingCall` also carries the tool `description` (captured at row-build time) for the summarizer prompt; the summary is payload-carried (source of truth for remounts) with `set_summary` as a live patch; config is `[permission_summary]` with `mode = off|fallback|always` (default off); lock order in the trigger path is config-lock-before-approval-lock per config.py's documented order.
- Accepted deviation from spec §Configuration: provider/model are free-text Inputs (plan-mandated) rather than spec's "seeded pickers" — a typo'd provider fail-opens to inactive; follow-up opportunity is picker/validation feedback in Settings.
- Known pre-existing failures observed but NOT caused by this work (verified at base): `test_selected_root_swap_fails_closed_before_local_invoke` (hook arity), `test_first_open_paints_an_answerable_card_nav_away_path` (paint timing), `test_attach_and_detach_cover_exactly_the_same_slot_set` (harness construction order) — worth a tracked fix separately.
- Testing: 620+ targeted tests across nine suites + fix-wave reruns, all green; no full sweep run (repo policy).
- Modified/added: `Agents/agent_models.py`, `Agents/agent_runtime.py`, `Agents/mcp_tool_provider.py`, `Agents/local_tool_provider.py`, `Chat/console_chat_controller.py`, `Chat/permission_summary_service.py` (new), `Chat/approval_display.py` (new), `Chat/console_runtime.py`, `Widgets/Chat_Widgets/chat_approval_card.py`, `Widgets/Chat_Widgets/chat_task_cards.py`, `UI/Screens/chat_screen.py`, `UI/Screens/settings_screen.py`, `config.py`, plus seven test files.
<!-- SECTION:NOTES:END -->
