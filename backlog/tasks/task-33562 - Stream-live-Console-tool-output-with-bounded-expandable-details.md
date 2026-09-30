---
id: TASK-33562
title: Stream live Console tool output with bounded expandable details
status: Done
assignee:
  - '@codex'
created_date: '2026-09-30 01:15'
updated_date: '2026-09-30 02:40'
labels:
  - console
  - tools
  - ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let users inspect partial text from streaming-capable tools during execution, while keeping each call in its existing stable Console row and preserving honest terminal outcomes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Supported tool and skill-script text appears while the call is still running; tools without live text retain final-result behavior.
- [x] #2 Live text updates the existing three-line preview and expanded details without changing row identity or focus.
- [x] #3 Output is bounded, incrementally decoded, isolated per execution, and late or malformed updates cannot affect another call.
- [x] #4 Partial text uses existing display projection and remains session-only; final results and interrupted outcomes remain honest.
- [x] #5 Permissions, budgets, process cleanup and final-only tool behavior are preserved; targeted automated and live UI checks pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/205-console-partial-tool-output.md
Reason: optional producer/runtime output observation with a session-only privacy boundary.
Follow Docs/superpowers/plans/2026-09-29-console-partial-tool-output.md: implement and test the bounded observer, skill stdout/stderr and exact-token MCP progress; project into stable Console rows; verify targeted tests, native UI, docs and review.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented optional session-only live output in the existing stable Console tool row. Skill scripts incrementally decode bounded stdout/stderr; native MCP stdio calls observe exact-token, monotonic progress messages. Unsupported producers retain final-only behavior. A context-bound stdlib observer coalesces bounded snapshots and cannot delay tool completion beyond a 100ms close grace; late abandoned-worker output and malformed/stale protocol updates are rejected. Runtime display projection keeps partial bodies out of durable trace, trajectory, provider history and diagnostics. Timeout/Stop retain visible partial text with honest outcome labels; real final results keep their existing authority.

ADR required: yes; backlog/decisions/205-console-partial-tool-output.md records the runtime/producer/privacy contract. Changed Agents output scope/runtime/service, skill runner, MCP client, Console activity/presentation, focused tests and guide. No new dependency, schema, setting or permission was introduced. Final focused verification: 48 passed; broader existing checks and native journeys are qualified in Docs/superpowers/qa/2026-09-29-console-tool-followups.md. Read-only review issues were fixed and rechecked.

MCP progress uses the shared strict Pydantic boundary, preserves integer counter precision and quietly rejects malformed or nonfinite data. Progress truncation reuses the named observer cap; the denial helper declares its exact string verdict-map type.
<!-- SECTION:NOTES:END -->
