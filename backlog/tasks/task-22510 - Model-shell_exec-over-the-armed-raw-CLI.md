---
id: TASK-22510
title: Model shell_exec over the armed raw CLI
status: To Do
assignee: []
created_date: '2026-08-27 04:53'
updated_date: '2026-08-27 04:54'
labels:
  - console
  - tools
  - security
  - agents
dependencies:
  - TASK-18926
  - TASK-22509
references:
  - backlog/decisions/094-raw-and-virtual-cli-execution-boundaries.md
  - Docs/superpowers/specs/2026-08-26-raw-and-virtual-cli-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let a model request the same dangerous one-shot host-shell executor as the Console user command, but only through an unmistakable command-visible approval boundary with no persistent silent grant.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 One shell_exec model tool accepts command, shell selector, optional absolute initial directory, and a timeout that cannot exceed 300 seconds
- [ ] #2 The tool schema is registered only while persistent raw CLI unlock, per-launch arming, local tools, and the global tool kill switch all permit it; invocation rechecks every gate
- [ ] #3 Raw-shell permission exposes only Ask and Off, and any stored or hand-edited Allow value is coerced to Ask at runtime
- [ ] #4 Every approval row safely displays the complete command, shell, initial directory, timeout, host-authority warning, and whether a session decision covers future commands
- [ ] #5 Approval offers Run once, Allow all raw shell commands for this Console session, and Deny; the session grant is process-memory-only and clears on disarm or restart
- [ ] #6 Repeated shell_exec calls in one model batch retain per-call identity so Run once can never approve an undisplayed command; a session-wide approval may cover later calls only after its scope is stated
- [ ] #7 Disarming or disabling persistent unlock denies pending raw approvals, prevents post-approval launch races, clears session grants, and begins bounded cleanup of active raw commands
- [ ] #8 Approved calls reuse TASK-18926 raw execution, streaming, sanitization, timeout, cancellation, result, and cleanup-certainty contracts without a second executor
- [ ] #9 Model output enters ordinary bounded agent tool history and local run logs, while generic diagnostics retain only content-free execution metadata
- [ ] #10 Focused tests cover catalog and invocation gate matrices, stored-Allow coercion, per-call approval isolation, session grant lifetime, approval-disarm races, and mounted approval and Tools UI behavior
- [ ] #11 The Console tools and Privacy and Security documentation distinguish direct user commands from model shell_exec authorization
<!-- AC:END -->
