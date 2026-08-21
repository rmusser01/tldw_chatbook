---
id: TASK-16322
title: Add nested AGENTS.md activation before Console tools
status: Done
assignee:
  - '@codex'
created_date: '2026-08-20 15:32'
updated_date: '2026-08-21 04:14'
labels:
  - console
  - agents
  - security
dependencies:
  - TASK-16320
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Activate narrower repository instructions atomically before path-aware Console tool batches so every model chain receives applicable guidance before approval or execution.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 `ToolCatalogRegistry` exposes the same first-registrant-wins owner to dispatch and preflight, and only that resolved owner can report structural path targets.
- [x] #2 Local and built-in path-aware tools implement every approved filesystem, patch, glob/grep, git, and outside-binding scope rule without parsing opaque process, skill, MCP, or command text.
- [x] #3 Nested discovery is lazy and O(depth), pins stable content for one dispatch, refuses changed-after-start sources, enforces the shared nested byte budget, and renders admitted guidance broad-to-specific.
- [x] #4 A typed preparation hook runs before unchanged security review; any newly required guidance atomically defers the whole batch before approval or execution, while sanitized preparation failure proceeds to normal review without changing verdicts.
- [x] #5 Deferred batches preserve tool-call IDs, order, and cardinality with runtime-owned protocol stubs followed by a separate ephemeral context update valid for OpenAI-compatible, Anthropic, Gemini, and fenced/local transports.
- [x] #6 Parent and subagent chains share one activation ledger and budget but track delivery independently; terminal omissions warn and defer each affected chain at most once, and concurrent admission is deterministic within the first-lock-wins policy.
- [x] #7 Automatic nested instruction bodies never enter persistent or diagnostic channels, while explicit file reads and model quotations retain ordinary persistence behavior.
- [x] #8 Focused path-mapping, collision, runtime, concurrency, provider-grammar, persistence-leak, and regression tests pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add the registry-owned immutable path-target contract and a single cached first-wins owner resolution seam shared by dispatch and preflight.
2. Implement complete local and built-in filesystem, patch, glob/grep, git, and outside-binding target mapping by reusing validated provider paths and the shared patch parser.
3. Add lazy nested resolution plus the run-local activation ledger, byte/token admission, per-chain delivery receipts, and deterministic concurrency behavior.
4. Thread typed preparation before unchanged security review, then add provider-safe retry context transport and persistence-leak protections across parent and subagent chains.
5. Run the approved focused, regression, static, sentinel, performance, and live-verification gates; document evidence and close the task.

ADR required: yes
ADR path: backlog/decisions/069-console-project-instruction-local-state-and-preflight.md
Reason: ADR-069 already defines the cross-module path-aware provider contract, preparation/security boundary, and ephemeral shared-ledger ownership; no new ADR is needed.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented registry-owned first-wins path targeting, lazy nested resolution with shared run-local activation state, preparation-before-review deferral, provider-correct retry grammar, and content-free persistence/diagnostic boundaries. ADR: [ADR-069](../decisions/069-console-project-instruction-local-state-and-preflight.md); no new ADR was required. Delivery code landed in the registry/runtime series rooted at [af94887ba](https://github.com/rmusser01/tldw_chatbook/commit/af94887ba) and the Console integration series rooted at [06b4989f4](https://github.com/rmusser01/tldw_chatbook/commit/06b4989f4); accepted heads are 3eb701a45 and 4c21cd37d.

Verification: the exact Task12 aggregate produced 748 passed plus only the two loopback-bind PermissionError setup nodes in test_console_provider_gateway; the identical two-node command produced the exact same errors on clean base 5047b6962 after escalation was unavailable. Runtime/concurrency ran 20 clean iterations of 65 tests (1,300 passes). All 17 new files pass Ruff check and format; the existing-file scan matches the clean-base 28 non-F821 diagnostics and sole RunLogWriter F821 baseline. Sentinel QA passed and only /tmp/chatbook-agents-md-delivery2/provider-spy.json contains the automatic body; SQLite dump, pytest XML, and run log are clean. git diff --check and the Delivery2 scope scan are clean. Complete UX/docs/performance/live UAT remain intentionally outside Delivery2 under Task12 and are owned by Delivery3.
<!-- SECTION:NOTES:END -->
