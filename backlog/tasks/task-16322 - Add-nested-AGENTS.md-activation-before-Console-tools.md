---
id: TASK-16322
title: Add nested AGENTS.md activation before Console tools
status: In Progress
assignee:
  - '@codex'
created_date: '2026-08-20 15:32'
updated_date: '2026-08-21 00:43'
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
- [ ] #1 `ToolCatalogRegistry` exposes the same first-registrant-wins owner to dispatch and preflight, and only that resolved owner can report structural path targets.
- [ ] #2 Local and built-in path-aware tools implement every approved filesystem, patch, glob/grep, git, and outside-binding scope rule without parsing opaque process, skill, MCP, or command text.
- [ ] #3 Nested discovery is lazy and O(depth), pins stable content for one dispatch, refuses changed-after-start sources, enforces the shared nested byte budget, and renders admitted guidance broad-to-specific.
- [ ] #4 A typed preparation hook runs before unchanged security review; any newly required guidance atomically defers the whole batch before approval or execution, while sanitized preparation failure proceeds to normal review without changing verdicts.
- [ ] #5 Deferred batches preserve tool-call IDs, order, and cardinality with runtime-owned protocol stubs followed by a separate ephemeral context update valid for OpenAI-compatible, Anthropic, Gemini, and fenced/local transports.
- [ ] #6 Parent and subagent chains share one activation ledger and budget but track delivery independently; terminal omissions warn and defer each affected chain at most once, and concurrent admission is deterministic within the first-lock-wins policy.
- [ ] #7 Automatic nested instruction bodies never enter persistent or diagnostic channels, while explicit file reads and model quotations retain ordinary persistence behavior.
- [ ] #8 Focused path-mapping, collision, runtime, concurrency, provider-grammar, persistence-leak, and regression tests pass.
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
