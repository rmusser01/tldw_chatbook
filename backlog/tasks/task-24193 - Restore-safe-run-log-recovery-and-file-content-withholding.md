---
id: TASK-24193
title: Restore safe run-log recovery and file-content withholding
status: Done
assignee: []
created_date: '2026-08-29 05:03'
updated_date: '2026-08-29 05:17'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore full-fidelity recovery handles for safe run-log records while keeping local file tool results non-recoverable at the durable filesystem boundary.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The real search_run_log closure test exercises recoverable safe tool output through a non-file built-in.
- [x] #2 The test still proves history truncation, run-log pointer recovery, and marker visibility beyond the legacy 400-character render cap.
- [x] #3 Sensitive read_file output remains non-recoverable and its existing privacy regression stays green.
- [x] #4 Safe run-log content is not reduced to the 4,000-character Trace summary cap before the writer applies its own record ceiling.
- [x] #5 Local file tool results are withheld based on tool identity even when earlier runtime path redaction has removed the original locator.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/080-trace-v2-exhaustive-event-projection-and-collaboration.md
Reason: this repairs the implementation of ADR-080's existing safe-content recovery and local-file omission boundary; no storage, service, or privacy contract changes.

1. Replace the contradictory read_file fixture with a calculator result containing the same large safe marker payload.
2. Separate full-fidelity run-log sanitization from the bounded Trace summary path and withhold file-tool results by tool identity.
3. Preserve the real AgentService, run-log writer/search closure, catalog dispatch, history truncation, and provider fence path.
4. Run the focused recovery and sensitive-file privacy tests, the full module, scoped lint/format/diff checks, and a maxfail full-suite probe.
5. Self-review the implementation against realistic recovery/privacy mutations, document evidence, and close the task.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Replaced the contradictory `read_file` recovery fixture with a large safe calculator result, then separated filesystem run-log fidelity from the bounded Trace summary limit. Known file-content tools—including the model-visible `skill_file` runtime alias—are now classified by tool identity at both durable seams, so prior locator redaction cannot make their payloads recoverable. Safe records retain full content until the run-log writer's own ceiling; altered records retain privacy-safe audit rows without recovery handles.

Verification: 115 focused Agent/Trace tests passed; scoped Ruff and diff checks passed. A full `--maxfail=1` probe advanced past the former 1,765-test failure to 2,403 passed before reaching the independent Console wave-6 line-count ratchet. ADR required: no. ADR path: `backlog/decisions/080-trace-v2-exhaustive-event-projection-and-collaboration.md`.
<!-- SECTION:NOTES:END -->
