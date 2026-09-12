---
id: TASK-32034
title: Track owned agent operations and bound abandoned tool workers
status: Done
assignee:
  - '@codex'
created_date: '2026-09-08 05:19'
updated_date: '2026-09-08 05:54'
labels:
  - agents
  - console
dependencies:
  - TASK-32019
references:
  - backlog/decisions/134-fleet-admission-and-automatic-work-budgets.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Timed-out tool calls can leave local workers running after their caller returns. Capacity must follow actual worker ownership so repeated timeouts and cancellation cannot create unbounded hidden work.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Actual tool-worker and model-lifeline ownership survives timeout, cancellation, run terminalization, and bounded cleanup joins until the worker or driver finishes.
- [x] #2 The Console runtime admits at most eight tool workers by default, with two slots reserved for manual work; one run cannot launch another tool while its timed-out worker remains alive.
- [x] #3 Trusted manual or automatic origin is supplied by the submit path and inherited by children without reading message or tool text.
- [x] #4 Failed starts and repeated cleanup release exactly once; delayed cleanup stays visible in a metadata-only runtime snapshot and never touches billing.
- [x] #5 Barrier-based tests prove limits across conversations, thread-start failure, late completion, detached cleanup, and manual capacity under automatic load.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no (implements existing ADR-134).
ADR path: backlog/decisions/134-fleet-admission-and-automatic-work-budgets.md
Reason: implements the approved ownership and tool-capacity contract without a new boundary.
1. Prove physical worker retention, automatic/manual admission, failed starts, and delayed lifeline cleanup with gated tests.
2. Add app-owned capacity and per-execution operation ownership; propagate trusted submit origin through services and children.
3. Enforce configured tool capacity before both threaded and inline invocations; release only on actual completion.
4. Wire model drivers and runtime disposal/replacement to retained ownership.
5. Run focused runtime, tool, controller, and lifecycle verification; update delivered-policy documentation and review ledger.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented ADR-134 operation ownership and conservative tool admission. ConsoleRuntime shares one capacity ledger across services, conversations, view/bridge replacement, and shutdown. Controller-supplied origin is inherited by children. Timed-out/cancelled tool workers retain slots until worker finally; per-run retries refuse while the worker lives. Inline calls also reserve capacity. Model drivers release only after actual cleanup/loop closure, including bounded-join overruns; inline children use a separate scope from fleet settlement counters. Failed starts and repeated cleanup release exactly once. Current settings are read before admission; invalid totals fall back and reserved slots clamp safely. No schema, dependency, billing, diagnostic-content, or sink change.

Core files: Agents/execution_capacity.py, Agents/agent_service.py, Chat/console_agent_bridge.py, Chat/console_chat_controller.py, Chat/console_runtime.py. Updated config comments, user guide, ADR implementation status, plan, review ledger, and baseline tool probe. New gated unit/service/lifecycle tests cover 8 workers across conversations, 6 automatic plus 2 manual, inline bypass, lowering limits, failed starts, late completion, and retired ownership.

Verification: 382 targeted service/bridge/continuation/runtime/headless/wake tests; final integration 145 tests (one overlaps the earlier headless run); another 40 targeted controller tests. New files pass Ruff lint/format, modified production files add no lint findings, changed formatting and whitespace checked. Self-review complete. The repository diagnostic gate is red solely for unrelated TTS_Events/tts_events.py digest drift; all five edited production files preserve their starting diagnostic/sink entries and the new module has none. No full suite or live provider was run.

ADR required: no new ADR; implements backlog/decisions/134-fleet-admission-and-automatic-work-budgets.md. Child admission and automatic-chain limits remain pending under TASK-32035 through TASK-32037.
<!-- SECTION:NOTES:END -->
