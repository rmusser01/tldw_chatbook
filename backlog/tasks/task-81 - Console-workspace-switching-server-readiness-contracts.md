---
id: TASK-81
title: Console workspace switching server-readiness contracts
status: Done
labels:
- console
- workspaces
- server-readiness
- ux
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Prepare Console workspace switching for future tldw_server and ACP handoff support without claiming or implementing a sync engine. Console should keep local workspace switching usable now, preserve explicit staging rules, avoid hiding global Library or Notes content, and expose honest unavailable/readiness states for future server workspace and ACP task/run package migration paths.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Console local workspace switching remains backed by the local registry fallback and does not depend on tldw_server sync.
- [x] #2 Workspace switching changes Console operating context and conversation scope, but Library/Notes/global browse visibility remains intact.
- [x] #3 Console clearly distinguishes local-only, server-unavailable, remote-only, conflict, and runtime-missing workspace states using existing workspace authority/sync language.
- [x] #4 Console supports explicit copy/reference/metadata-only/local-only handoff eligibility states for sources and conversations before staging them into the active workspace.
- [x] #5 ACP task/run package handoff is represented as a future migration target with visible unavailable, readiness, failure, and audit details.
- [x] #6 No background sync engine is implemented or implied; server-backed hydration remains behind an adapter boundary that can be wired only when the server API is available.
- [x] #7 Focused regressions cover local fallback, server-unavailable states, cross-workspace gating, and ACP handoff blocked/ready states. Actual CDP screenshots are captured before approval.
- [x] #8 Console auto-creates a safe built-in `Default` workspace when no active workspace exists, keeps normal chat/conversation visibility available, and blocks or sanitizes filesystem/runtime bindings until the user creates an explicit workspace.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/005-console-workspace-server-readiness.md
Reason: TASK-81 defines long-lived workspace authority, sync/handoff policy, server adapter boundaries, and ACP task/run package readiness contracts.

1. Create ADR 005 to lock the local-first/server-readiness boundary: no sync engine, local registry fallback, explicit adapter states, and ACP handoff as visible future target.
2. Write failing pure regressions for Console workspace server-readiness display state covering server-unavailable, remote-only, conflict, runtime-missing, source/conversation transfer policy labels, and ACP handoff blocked/ready/failure/audit states.
3. Write failing mounted Console regressions asserting the workspace rail renders those states without enabling sync or hiding local fallback switching.
4. Implement minimal workspace contract/display models and Console rail rendering using the existing registry seam and runtime binding metadata, without adding background sync or server hydration.
5. Add a registry-backed built-in `Default` workspace that is auto-created only when no active workspace exists, is local-only/not-configured, preserves chat/conversation visibility, rejects runtime bindings, and sanitizes stale bindings so filesystem/file tools stay disabled by default.
6. Run focused Workspace and Console tests, run diff checks, capture actual rendered CDP screenshots, and record QA evidence before requesting approval.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added ADR 005 to document the local-first Console workspace boundary, server-readiness adapter seam, transfer-policy staging model, and ACP task/run handoff readiness contract without implementing sync.
- Extended workspace display state and Console left rail rendering to show server fallback/unavailable/remote/conflict/runtime-missing states, copy/reference/metadata-only/local-only handoff rows, and ACP handoff readiness/audit copy.
- Added `Default` workspace constants and `LocalWorkspaceRegistryService.ensure_default_workspace()`, wired app startup to create it when no active workspace exists, and blocked/sanitized runtime bindings on `Default` so file/tool access remains disabled until a user creates an explicit workspace.
- Review follow-up: optimized Default runtime binding sanitization so common read paths first check for stale bindings and skip the write transaction when none exist.
- Updated Console rail persistence tests so the built-in default workspace uses the real `workspace-default` namespace instead of the old display-only `global` sentinel.
- Verification: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Workspaces --tb=short` passed with 42 tests.
- Verification: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_console_workspace_context_rail.py Tests/UI/test_console_live_work_handoffs.py Tests/UI/test_console_persistent_rails.py --tb=short` passed with 93 tests and one existing dependency warning.
- Verification: `git diff --check` passed.
- QA screenshots: `Docs/superpowers/qa/console-workspace-server-readiness-task81-cdp-v2.png` and `Docs/superpowers/qa/console-workspace-default-task81-cdp.png`.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
