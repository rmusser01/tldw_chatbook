---
id: TASK-32681
title: Preserve typed MCP tool results through client services
status: Done
assignee:
  - '@codex'
created_date: '2026-09-16 04:26'
updated_date: '2026-10-01 04:21'
labels:
  - plugins
  - implementation
  - mcp
dependencies:
  - TASK-32645
documentation:
  - Docs/superpowers/plans/2026-09-15-plugin-mcp.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Retain the protocol information needed to distinguish tool errors from successful structured results while keeping ordinary display compatible.

Design: Docs/superpowers/specs/2026-09-15-managed-plugins-design.md; Docs/superpowers/specs/2026-09-15-expanded-hook-runtime-design.md. ADR required: yes. ADR paths: backlog/decisions/162-managed-agent-plugins.md; backlog/decisions/163-expanded-console-hook-runtime.md. This task implements the accepted contracts without creating a separate permission or runtime owner.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Stdio/client/service tool results retain content, structuredContent, isError and metadata before any display projection, with separate transport-failure identity.
- [x] #2 Malformed error flags, serialization overflow and tool errors cannot turn into successful hook effects.
- [x] #3 Existing non-hook consumers retain compatible presentation through an explicit projection while typed consumers receive the complete result.
- [x] #4 Production client and service tests cover structured-only, text-only, mirrored, error-bearing and oversized results without executing third-party plugins.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR paths: backlog/decisions/163-expanded-console-hook-runtime.md; backlog/decisions/162-managed-agent-plugins.md
Reason: implements the accepted typed protocol evidence and shared execution/audit contract.
1. Trace current stdio/client/service/provider ownership and port reviewed M1 fixtures; establish RED on the actual stdio boundary.
2. Integrate the reviewed M1 increment preserving current transport settlement, recovery guards, schema and permission owners.
3. Qualify complete original results, strict failures, write/settlement truth, cancellation and single bounded audit publication with controlled peers.
4. Run affected MCP/provider neighbors and owned/static comparisons; record evidence and limits, self-review ACs, complete via CLI and commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Preserved complete typed MCP results and exact bounded wire evidence through stdio/client/local/unified/provider boundaries with an explicit compatible display projection. Strict error flags and fixed body-free diagnostics fail closed; host dispatch state and a shared one-use publication claim preserve cancellation/timeout and audit truth. Retained current producer/recovery guards, same-task deadlines and actual native child settlement; corrected constructor/profile and observation-signature fixtures. Final targeted qualification: 453 passed in 172.38s, no warnings/skips. Existing deadline/source/inspection failures reproduced on frozen pre-M1 production and are documented separately. New files Ruff+format, changed Python parse/whitespace and shared diagnostic comparisons pass. ADR-162/163 apply. Evidence/platform limits: Docs/superpowers/reviews/2026-09-30-expanded-hooks-integration.md.
<!-- SECTION:NOTES:END -->
