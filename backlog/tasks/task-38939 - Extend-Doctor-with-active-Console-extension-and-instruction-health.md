---
id: TASK-38939
title: Extend Doctor with active Console extension and instruction health
status: To Do
created_date: 2026-11-05 16:09
labels:
- console
- diagnostics
- plugins
priority: medium
references:
- https://uxplanet.org/anthropic-just-shipped-claude-2-0-bbb46a9c1383
- TASK-25906
documentation:
- Docs/Architecture/tool-catalog.md
- Docs/Architecture/mcp-hub.md
- Docs/Architecture/console-run-hooks.md
- Docs/Architecture/console-file-authority.md
- backlog/decisions/162-managed-agent-plugins.md
- backlog/decisions/163-expanded-console-hook-runtime.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Help users understand setup problems involving skills, managed plugins, MCP connections, hooks, and project-instruction readiness from the existing Doctor surface. Preserve useful diagnostics even when the current model or configuration is unavailable.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The existing Doctor report includes bounded health summaries for skill/catalog conflicts, managed-plugin activation blockers, known MCP connection state, hook configuration, and authorized project-instruction readiness.
- [ ] #2 Checks use the existing authoritative readiness/diagnostic owners and distinguish invalid configuration, unavailable data, disabled capabilities, and healthy idle state.
- [ ] #3 Findings identify an existing remediation/settings destination where one exists; recommendations never silently edit settings, enable plugins, change permissions, or discard instructions.
- [ ] #4 The default report performs no network calls, provider requests, external command hooks, or MCP startup; any active probe remains explicit and separately authorized.
- [ ] #5 Reports disclose no credentials or automatic instruction bodies and remain scoped to the selected workspace's permitted metadata.
- [ ] #6 Targeted checks demonstrate isolated per-check failure, no-working-config behavior, honest unavailable states, and continued correctness of the original Doctor checks.
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Implementation planning records ADR required: yes/no, ADR path, and reason; any required new or amended ADR is accepted and linked before implementation begins.
- [ ] #2 Applicable targeted automated checks, formatting/lint checks, and live UI or real-provider evidence are recorded before completion; full test sweeps require user opt-in.
- [ ] #3 Relevant user guidance and living Architecture documentation describe the delivered behavior and its limits.
<!-- DOD:END -->
