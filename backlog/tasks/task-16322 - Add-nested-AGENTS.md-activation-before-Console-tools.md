---
id: TASK-16322
title: Add nested AGENTS.md activation before Console tools
status: To Do
assignee: []
created_date: '2026-08-20 15:32'
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
- [ ] `ToolCatalogRegistry` exposes the same first-registrant-wins owner to dispatch and preflight, and only that resolved owner can report structural path targets.
- [ ] Local and built-in path-aware tools implement every approved filesystem, patch, glob/grep, git, and outside-binding scope rule without parsing opaque process, skill, MCP, or command text.
- [ ] Nested discovery is lazy and O(depth), pins stable content for one dispatch, refuses changed-after-start sources, enforces the shared nested byte budget, and renders admitted guidance broad-to-specific.
- [ ] A typed preparation hook runs before unchanged security review; any newly required guidance atomically defers the whole batch before approval or execution, while sanitized preparation failure proceeds to normal review without changing verdicts.
- [ ] Deferred batches preserve tool-call IDs, order, and cardinality with runtime-owned protocol stubs followed by a separate ephemeral context update valid for OpenAI-compatible, Anthropic, Gemini, and fenced/local transports.
- [ ] Parent and subagent chains share one activation ledger and budget but track delivery independently; terminal omissions warn and defer each affected chain at most once, and concurrent admission is deterministic within the first-lock-wins policy.
- [ ] Automatic nested instruction bodies never enter persistent or diagnostic channels, while explicit file reads and model quotations retain ordinary persistence behavior.
- [ ] Focused path-mapping, collision, runtime, concurrency, provider-grammar, persistence-leak, and regression tests pass.
<!-- AC:END -->
