---
id: TASK-33648
title: Preserve authorized SessionEnd notification during Console host disposal
status: In Progress
assignee:
  - '@codex'
created_date: '2026-10-02 18:48'
updated_date: '2026-10-07 01:45'
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
- [ ] #1 Actual saved and granted v2 configuration emits the intended SessionEnd notification once during host disposal, with a successful exact-session-close control.
- [ ] #2 Changed or revoked definitions and ordinary post-disposal tool or model calls remain refused; no general authority is reopened.
- [ ] #3 Real process, ticket and physical cleanup settle through cancellation, with targeted negative controls and canonical ADR assessment.
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
