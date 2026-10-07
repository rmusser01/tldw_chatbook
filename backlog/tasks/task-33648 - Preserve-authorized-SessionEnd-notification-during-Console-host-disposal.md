---
id: TASK-33648
title: Preserve authorized SessionEnd notification during Console host disposal
status: Done
assignee:
  - '@codex'
created_date: '2026-10-02 18:48'
updated_date: '2026-10-07 02:06'
labels:
  - console
  - hooks
dependencies: []
references:
  - backlog/decisions/163-expanded-console-hook-runtime.md
  - backlog/decisions/197-console-hook-configuration-review.md
  - Docs/superpowers/reviews/2026-09-29-agent-orchestration-burndown.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
An independently reproduced expanded-hooks limitation suppresses a granted standalone SessionEnd notification during host disposal. Exact session close succeeds, but host disposal closes ordinary runtime and permission authority before notification launch. Review a bounded shutdown policy without reopening normal tool or model authority.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Actual saved and granted v2 configuration emits the intended SessionEnd notification once during host disposal, with a successful exact-session-close control.
- [x] #2 Changed or revoked definitions and ordinary post-disposal tool or model calls remain refused; no general authority is reopened.
- [x] #3 Real process, ticket and physical cleanup settle through cancellation, with targeted negative controls and canonical ADR assessment.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes (assessment/amendment of the existing teardown and consent boundary)
ADR path: backlog/decisions/163-expanded-console-hook-runtime.md; backlog/decisions/197-console-hook-configuration-review.md
Reason: preserve a narrow, already authorized host-issued teardown delivery while ordinary runtime/permission authority closes; no new runtime or grant.
1. Trace saved v2 config, exact grants, real Console disposal and existing SessionEnd ownership; settle the minimal transfer design against ADR-163/197 before source edits.
2. Add a real saved/granted SessionEnd process regression that fails during disposal, alongside exact-session-close and changed/revoked-definition controls.
3. Implement only the shared bounded teardown authority fix; retain ordinary post-disposal refusal and the original notification/physical-cleanup deadlines.
4. Verify targeted lifecycle/permission/cleanup tests, including cancellation, and independent scoped review. Record real process settlement and preserved limits before checking criteria.

Prospective mixed-projection correction after independent source review: reproduce disposal with a real installed/activated native plugin and saved/granted standalone hook. Reuse the exact original host event and engine execution through a minimal private per-delivery projection witness; require callback object identity, active/non-cancelled membership and fixed deadline, with fresh binding/finally reset and no external callback under the memory lock. Preserve public same/copied/ambient-context replay refusal and all existing grant/plugin/effect/model guards. Keep earlier Changes-required package and exact-base inherited teardown evidence; obtain a new frozen source review before closure.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the shared bounded SessionEnd fix in hook lifecycle/engine, HookPermissions and Console runtime, with real saved/granted and mixed-native regression controls. Only the exact host-issued, effect-free standalone notification may survive disposal; one private active-delivery projection witness authenticates its callback. Existing launch locks reread canonical config and the original store/grant token; ordinary permission targets stay closed, deadlines/process custody unchanged. ADR assessment/amendment: backlog/decisions/163-expanded-console-hook-runtime.md and 197-console-hook-configuration-review.md.

Final affected selection: 62 PASS, F/E/S0, XML62.277s (wall64.635s), including exact close/both disposal paths, actual native activation/startup, changed/revoked/reapproved/missing store refusal, public same/copied replay and real cancellation/reaping/ticket settlement. Independent corrected immutable review Ready/no actionable findings; prior projection P2 Changes-required package preserved. Portable raw/XML/receipts/static/reviews and source hashes: Docs/superpowers/qa/2026-10-06-task33648-session-end/{report.md,manifest.json,evidence.tar.gz}.

Limits: related selection remains NON-GREEN (179 passed bodies + one mounted executor teardown error), reproduced on complete exact6feb base with the same300s timeout. No unique cause/broad suite/aggregate FD/non-macOS certificate. Sandbox-only root-identity setup failure, earlier RED/setup/skip/cache/temp warnings retained; no cleanup/cap workaround. Runtime retains30 inherited lint signatures/format debt, zeroNEW; other5 changed files lint/format clean. Replay tests dynamically supply the existing continuation context; replacing inherited new delivery tuple is verified by source trace. No permission owner, plugin/MCP/model/effect authority or cleanup allowance is widened.

Generalizable mixed-projector incident and required real-owner regression pattern recorded in backlog/docs/lessons-hook-teardown.md; corrected-source hashes remain unchanged.
<!-- SECTION:NOTES:END -->
