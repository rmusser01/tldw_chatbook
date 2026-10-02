---
id: TASK-32679
title: Wire session child and compaction hook boundaries
status: Done
assignee:
  - '@codex'
created_date: '2026-09-16 04:24'
updated_date: '2026-10-02 02:03'
labels:
  - plugins
  - implementation
  - hooks
dependencies:
  - TASK-32678
documentation:
  - Docs/superpowers/plans/2026-09-15-expanded-hooks.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Expose the approved lifecycle events where their effects can be applied safely to the owning run and context.

Design: Docs/superpowers/specs/2026-09-15-managed-plugins-design.md; Docs/superpowers/specs/2026-09-15-expanded-hook-runtime-design.md. ADR required: yes. ADR paths: backlog/decisions/162-managed-agent-plugins.md; backlog/decisions/163-expanded-console-hook-runtime.md. This task implements the accepted contracts without creating a separate permission or runtime owner.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 SessionStart uses cancellable provisional admission and publishes dependent capabilities/context only after controlling requirements succeed; tab focus and package inspection emit no session events.
- [x] #2 SubagentStart narrows inherited tools and budgets, and SubagentStop contributes only to an active parent checkpoint without restarting settled work.
- [x] #3 PreCompact supplies required compactor input and PostCompact fences subsequent input after committed summaries; runtime context blocks are reassembled without duplication.
- [x] #4 Manual-only UserPromptSubmit, idle hook-set replacement, SessionEnd, required-context failures and independent component success are covered at actual Console and agent boundaries.
- [x] #5 Enabled standalone v2 definitions appear in Console review and canonical Settings, require exact persistent consent before execution, and changed or revoked definitions cannot bypass that consent at session admission, launch, or effect acceptance.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes (amend existing accepted contracts; no new permission or storage owner). ADR paths: backlog/decisions/163-expanded-console-hook-runtime.md and backlog/decisions/197-console-hook-configuration-review.md. 1. Qualify the reviewed H4 lifecycle, child, context, and compaction checkpoint against current Console boundaries with private-profile regressions. 2. Reuse the existing HookPermissions owner and review modal for v2 exact definitions; preserve legacy identities and guided Settings fields, expose v2 definitions through the existing advanced editor, and guard actual launches with current grant epochs. 3. Port the reviewed H4 producers without replacing newer admission, workspace, recovery, or worker guards. 4. Run scoped lifecycle, compaction, child, consent, Settings, and neighboring runtime tests; compare unchanged baselines for fixture failures. 5. Self-review, document current evidence and limitations, check acceptance criteria, mark Done via CLI, and commit the exact task files.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Reused reviewed H4 session/child/compaction producers and shared checkpoint/context retirement. Current-dev additions reuse exact persistent HookPermissions consent for v2 definitions, launch serialization, Console review and canonical Settings Advanced Config; malformed source blocks admission. Common durable hydration now publishes exact parent IDs while preserving batched version reads. Final core 404 passed with no skips/warnings; consent/UI 150 passed; frozen 1,000-turn stress passed. Two unchanged recovery failures reproduce against pre-H3 source and are explicitly recorded for continuation assessment. Test-owned runtime/worker cleanup removed aggregate descriptor warnings. Full Ruff/format on 22 owned files; shared diagnostics unchanged; syntax, whitespace and design-token checks pass. Existing ADR163/197 amended and linked, no new storage/dependency/permission owner. Detailed commands/limitations: Docs/superpowers/reviews/2026-09-30-expanded-hooks-integration.md.

PR #2946 rebase onto dev 83c2c9810d5d09406c85e1f027541b846ead93aa reuses its exact committed user-parent field and updates two runtime test doubles to the required coordinator binding contract. Admission group: 123 passes and one retained upstream TASK-32873 xfail; mounted composer/fork/Stop integration: 15 passes. New timestamp writes reuse ADR-173 canonical UTC helper; no production admission guard was weakened.

PR #2946 Qodo review: initial context authority collection runs on a worker with the existing operation-owned connection; session/disposal fences are rechecked after the await. Actual accepted Console-turn evidence verifies off-UI-thread collection. Complete lifecycle/continuation group: 80 passes with no skips. No exact-currentness gate was removed. ADR-163 applies; evidence: Docs/superpowers/reviews/2026-09-30-expanded-hooks-integration.md.

PR #2946 rebased onto dev 84247cb843 with production repairs unchanged. Fixture controls retain the collection-time private profile, real coordinator binding, full immutable snapshots plus owned turn attribution and bounded worker entry. All 203 selected Console cases pass across retained/isolated corrected runs; final teardown passes 13 cases in 431.771s with no skips, preserving exact cancellation, revocation, resource settlement and no replay under the existing best-effort observer deadline. Mounted collapse/resize and all four pending-Stop size variants pass; boot 22 passes at census 1031/1033 with unchanged limits. ADR-162/163/197 apply; no new ADR. Exact failed-run dispositions, artifacts and scope limits are in Docs/superpowers/reviews/2026-09-30-expanded-hooks-integration.md.
<!-- SECTION:NOTES:END -->
