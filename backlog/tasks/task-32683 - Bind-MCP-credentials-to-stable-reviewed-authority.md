---
id: TASK-32683
title: Bind MCP credentials to stable reviewed authority
status: Done
assignee:
  - '@codex'
created_date: '2026-09-16 04:27'
updated_date: '2026-10-01 04:54'
labels:
  - plugins
  - implementation
  - mcp
dependencies:
  - TASK-32682
  - TASK-32670
documentation:
  - Docs/superpowers/plans/2026-09-15-plugin-mcp.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep normal token renewal usable while ensuring account, endpoint or scope changes cannot inherit stale plugin authority.

Design: Docs/superpowers/specs/2026-09-15-managed-plugins-design.md; Docs/superpowers/specs/2026-09-15-expanded-hook-runtime-design.md. ADR required: yes. ADR paths: backlog/decisions/162-managed-agent-plugins.md; backlog/decisions/163-expanded-console-hook-runtime.md. This task implements the accepted contracts without creating a separate permission or runtime owner.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The owning credential service exposes stable reference, principal, issuer, audience/endpoint and scope bindings with separate authority and storage revisions.
- [x] #2 Verified renewal within unchanged authority resolves current credentials at dispatch without invalidating plugin trust; changed or unknown authority requires reconciliation.
- [x] #3 Explicit header/token mappings and supported OAuth flows use existing host credential services; unsupported auth is a specific unready state and no vendor connector grant is imported.
- [x] #4 Secret sentinels stay out of trust snapshots, catalogs, receipts, debug defaults and errors; origin changes, failed refresh, opaque replacement and store migration have successful and refusal controls.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR paths: backlog/decisions/162-managed-agent-plugins.md; backlog/decisions/163-expanded-console-hook-runtime.md
Reason: implements accepted stable credential authority through existing host secret storage and authenticated plugin snapshots.
1. Read the actual config/keyring/transport/trust flow and reviewed M3 source/test increment; preserve current recovery and native cleanup ownership.
2. Establish missing-binding RED through owned transport/service controls after M2 successful controls.
3. Integrate reviewed bindings, dispatch-time secret resolution, origin/generation checks, schema migration and body-free projections using current protected writers.
4. Run targeted credential/transport/trust and affected neighbors plus static/resource checks; record limits, review ACs and commit M3-owned files.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Integrated stable protected MCP credential references/generations, verified renewal versus changed/unknown authority, selected-origin current secret resolution, explicit supported header policy and fixed unsupported OAuth. Schema 3 stores references only and migrates through the current protected writer; authenticated recovery validates complete mappings through the actual local owner. Preserved current recovery/producer/native cleanup seams. Fixed proven worker-start capacity leak and raw exception exposure with a real transport refusal/retry test. Existing ADR-162/163 apply; changed client/local_store/local service/HTTP/config/coordinator/recovery, new credential owner and qualification docs/tests. Evidence: 124 credential/transport/trust tests and 234 affected transport/schema/lifecycle/coordinator/recovery neighbors passed without warnings/skips; new Ruff/formatter and shared no-add lint/syntax/whitespace pass. Limits: fake protected backends/keyring API and controlled peers on macOS; no real keychain, generic OAuth, external vendor or full-suite qualification. Details: Docs/superpowers/reviews/2026-09-30-expanded-hooks-integration.md.
<!-- SECTION:NOTES:END -->
