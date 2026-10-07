---
id: TASK-33648
title: Preserve authorized SessionEnd notification during Console host disposal
status: To Do
assignee: []
created_date: '2026-10-02 18:48'
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
