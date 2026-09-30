# Console Partial Tool Output Implementation Plan

> **For agentic workers:** Use executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Show supported tool output during execution in the existing Console row.

**Architecture:** An optional context-bound observer carries bounded snapshots from producers through service workers. The runtime applies existing display projection; the Console correlates updates to the exact proposal. Final results retain their existing authority.

**Tech Stack:** Python 3.12+, stdlib threading/contextvars/codecs, Textual 8.x, existing MCP JSON-RPC transport.

**Spec:** Docs/superpowers/specs/2026-09-29-console-partial-tool-output-design.md

## Global Constraints

- No new dependencies, configuration, settings, schema changes, or keybindings.
- Retain 16,000 characters; coalesce to ten updates per second.
- Partial text is session-only and display-projected.
- Preserve permissions, cancellation, containment, budgets, and final-result behavior.
- Use existing ADR-150 tokens and disclosure components.

ADR required: yes
ADR path: backlog/decisions/205-console-partial-tool-output.md
Reason: introduces an optional runtime/producer observation contract with a privacy boundary.

## Implementation

- [x] Add a focused failing held-execution check for bounded output, runtime-only observation, and stable row updates.
- [x] Add the minimal optional output scope in Agents/tool_output.py; explicitly bind its sink in agent_service.py's existing worker. In agent_runtime.py, project partial snapshots only to on_tool_activity around script/catalog dispatch.
- [x] Publish incrementally decoded stdout/stderr from skill_script_runner.py without changing containment or final capture; add a real held-child regression.
- [x] Handle exact-token MCP progress notifications in MCP/client.py; verify isolation, malformed/late messages, and unchanged final results.
- [x] Update console_tool_activity.py to correlate partial events and retain interruption text; verify mounted disclosure identity and final replacement.
- [x] Run affected producer/runtime/Console tests, lint, native walkthrough, and fresh read-only review. Update guide and task notes.
- [x] Publish the verified PR against dev and address review feedback.

Integration is tracked by [PR #2923](https://github.com/rmusser01/tldw_chatbook/pull/2923); merge requires its final checks and review to pass.
