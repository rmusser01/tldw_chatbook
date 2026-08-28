---
id: TASK-22510
title: Model shell_exec over the armed raw CLI
status: Done
assignee: []
created_date: '2026-08-27 04:53'
updated_date: '2026-08-28 15:11'
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
- [x] #1 One shell_exec model tool accepts command, shell selector, optional absolute initial directory, and a timeout that cannot exceed 300 seconds
- [x] #2 The tool schema is registered only while persistent raw CLI unlock, per-launch arming, local tools, and the global tool kill switch all permit it; invocation rechecks every gate
- [x] #3 Raw-shell permission exposes only Ask and Off, and any stored or hand-edited Allow value is coerced to Ask at runtime
- [x] #4 Every approval row safely displays the complete command, shell, initial directory, timeout, host-authority warning, and whether a session decision covers future commands
- [x] #5 Approval offers Run once, Allow all raw shell commands for this Console session, and Deny; the session grant is process-memory-only and clears on disarm or restart
- [x] #6 Repeated shell_exec calls in one model batch retain per-call identity so Run once can never approve an undisplayed command; a session-wide approval may cover later calls only after its scope is stated
- [x] #7 Disarming or disabling persistent unlock denies pending raw approvals, prevents post-approval launch races, clears session grants, and begins bounded cleanup of active raw commands
- [x] #8 Approved calls reuse TASK-18926 raw execution, streaming, sanitization, timeout, cancellation, result, and cleanup-certainty contracts without a second executor
- [x] #9 Model output enters ordinary bounded agent tool history and local run logs, while generic diagnostics retain only content-free execution metadata
- [x] #10 Focused tests cover catalog and invocation gate matrices, stored-Allow coercion, per-call approval isolation, session grant lifetime, approval-disarm races, and mounted approval and Tools UI behavior
- [x] #11 The Console tools and Privacy and Security documentation distinguish direct user commands from model shell_exec authorization
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Execute Docs/superpowers/plans/2026-08-26-model-shell-exec.md in seven test-driven slices: 1. Define the shell_exec schema and Ask/Off-only resolver. 2. Add command-visible per-call approval and process-memory session grants. 3. Register the provider only while unlock, arm, local-tools, and global-kill-switch gates permit it. 4. Revoke pending approvals and active commands on disarm. 5. Correlate live executor progress with the existing agent tool marker. 6. Show always-visible Ask/Off-only policy and availability in Tools. 7. Document, live-verify, self-review, and complete focused verification. ADR required: yes. ADR path: backlog/decisions/094-raw-and-virtual-cli-execution-boundaries.md. Reason: ADR-094 defines conditional model shell exposure, Ask/Off-only permission, command-visible approval, process-memory session grants, shared executor reuse, and disarm race behavior.
<!-- SECTION:PLAN:END -->

## Implementation Notes

Implemented one Ask/Off-only `shell_exec` model tool over the existing
app-owned raw CLI runtime defined by ADR-094. The provider is cataloged only
while every live gate is open, validates and discloses the complete command,
keeps per-call approvals and session grants in process memory, rechecks
authority before dispatch, and revokes raw-only rounds, stamps, grants, and
active work on Disarm. Call-correlated bounded progress reuses the existing
Console tool marker, model calls stay in ordinary agent history/run logs, and
the Tools and Privacy & Security documentation now distinguish policy
discoverability, launch arming, direct-user commands, and model authorization.

Verification:

- Focused shell/provider/approval/Tools UI suite: 62 passed.
- Raw executor and Console fork-title suite: 265 passed.
- Mounted production Console probe: real approval card, Run once/Deny,
  session grant, raw execution, Disarm, and restart behavior passed.
- Ruff lint across all changed Python files and `git diff --check`: passed.
- Complete Codex Security feature scan
  `90f3b427-9a59-4b6a-b1e9-1c8be02124d9` and import-order delta scan
  `a78d894d-d0ca-4746-839f-f043427d2d2f`: no confirmed findings.

The first mounted launch exposed a fresh-interpreter Chat/Library import cycle
that in-process tests had hidden. The fix lazily resolves the Console title
helper without changing its validation seam, adds a fresh-interpreter import
regression, and records the incident in
`backlog/docs/lessons-live-verification.md`.

ADR required: yes.
ADR path: `backlog/decisions/094-raw-and-virtual-cli-execution-boundaries.md`.
Reason: ADR-094 is the governing execution and authorization boundary; no new
architectural decision was introduced during implementation.
