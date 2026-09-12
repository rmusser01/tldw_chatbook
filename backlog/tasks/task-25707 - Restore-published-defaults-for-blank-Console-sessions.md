---
id: TASK-25707
title: Restore published defaults for blank Console sessions
status: Done
assignee:
  - '@codex'
created_date: '2026-08-30 19:48'
updated_date: '2026-08-30 19:58'
labels:
  - console
  - settings
  - workspaces
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Ensure every eligible blank Console creation path continues to use the app-published new-chat defaults after workspace-assistant startup selection was introduced, so live provider controls or readiness snapshots cannot leak into unrelated new chats.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Ctrl+T and temporary blank Console sessions use the app-published provider, model, model-profile values, and default generation
- [x] #2 Workspace-created blank sessions use the same app-published settings snapshot
- [x] #3 Workspace default personas are stamped onto the published blank snapshot without weakening explicit source-owned session behavior
- [x] #4 Targeted Console session and workspace-default regression tests plus static checks pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Confirm the regression and ownership contract against ADR-095 and ADR-079, and verify no duplicate in-flight fix.
2. Change the plain new-session startup fallback to use the app-owned published blank-settings snapshot while preserving workspace-persona stamping and explicit source-owned carry behavior.
3. Extend the focused workspace-default seam tests so published-default provenance remains explicit, then run the Console session-controller and workspace-default targeted suites plus Ruff and diff checks.
4. Self-review the final diff, complete the acceptance criteria and implementation notes, and follow the normal dev PR/review/merge workflow.

ADR required: no
ADR path: backlog/decisions/095-conversation-owned-console-generation-settings.md and backlog/decisions/079-workspace-assistant-defaults.md
Reason: This is a narrow regression fix restoring the already-decided blank-chat ownership and workspace-persona startup contracts; no storage, security, or interface boundary changes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Restored ADR-095 blank-chat ownership by making the plain new-session selector always start from the app-published blank settings snapshot instead of cloning active controls or active-session settings. ADR-079 workspace default personas are still resolved and stamped onto that fresh snapshot; source-owned controller, handoff, character, retry, continue, and branch paths remain separate and unchanged.

Updated workspace startup regressions to distinguish published defaults from active controls, and repaired the synthetic context-policy constructor fixture to model the current controller contract explicitly. Ruff formatting was applied only to the three touched Python files.

Verification: 67 targeted tests passed; Ruff lint passed; Ruff format check passed; git diff --check passed. Existing warnings are limited to third-party requests/audioop deprecations.

ADR required: no. Existing ADR-095 and ADR-079 govern the repaired behavior.

Modified files: tldw_chatbook/UI/Console_Modules/session.py; Tests/Chat/test_workspace_default_session.py; Tests/Chat/test_console_context_policy_lifecycle.py.
<!-- SECTION:NOTES:END -->
