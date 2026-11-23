---
id: TASK-42475
title: "The Console has no accessor for its pending startup work, so three tests hand-copy private flags to wait for it to be quiet"
status: To Do
assignee: []
created_date: 2026-11-23 09:42
updated_date: 2026-11-23 09:42
labels:
- follow-up
- console
- tests
- hardening
priority: medium
references:
- tldw_chatbook/UI/Screens/chat_screen.py
- Tests/UI/test_console_workspace_context_rail.py
- Tests/UI/test_console_mcp_approval.py
- Tests/UI/test_console_rail_reconciliation.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during round 22 of the follow-up burn-down (2026-11-23), by TASK-42474's review. Checked on `fu/r22-railrecon` (the round-22 stack a999508ae7 plus TASK-42474); paths start at the repository root.

**Why.** Three tests wait for the Console's startup work to finish before they push or count state, and each keeps its own hand-copied list of private `ChatScreen` attributes:
- `Tests/UI/test_console_workspace_context_rail.py:124`, `_wait_for_console_startup_settled` (TASK-42148): `_console_attach_reconciled`, `_console_sync_in_progress`, `_console_sync_requested`, `_console_control_bar_sync_scheduled`, and the worker list.
- `Tests/UI/test_console_mcp_approval.py`, the `unsettled()` check at about line 1659: the same flags plus `_console_control_bar_replay_whole_sync` and a console-sync worker check.
- `Tests/UI/test_console_rail_reconciliation.py`, `_wait_for_inspector_quiet` (TASK-42474): TASK-42148's flags plus `_console_config_retries` (a pending config-rebuild retry) and a copy of the credential poll's own pre-gate (`subscription_credential_status(background=True)`, `subscription_readiness_revision()`, the shared connection evidence's `version`), compared with `_console_credential_pre_gate` so the wait also covers a first background credential read that has not finished.

The lists already disagree. TASK-42474 found three startup producers (the credential poll's first pass, the first background credential read, and a config-rebuild retry) that TASK-42148's list never saw; under load they re-synced Inspector owners inside a test's observation window. A renamed private attribute switches a check off silently: several are read with `getattr(..., False)`, so a rename reads as "quiet". None of the three sees a startup producer added later.

The product owns this knowledge. A `ChatScreen` method that returns the names of its pending startup work (empty when quiet), with a test that pins each producer to a name, would give the tests one signal that fails loudly when a producer is renamed or added.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The Console screen exposes one read-only accessor that names each piece of its pending startup and sync work (attach, native sync running or requested, control-bar sync or replay owed, a credential read or credential-poll pass owed, config-rebuild retries, Console-owned workers) and returns nothing when quiet
- [ ] #2 A test pins the accessor: driving each producer into its pending state makes the accessor name it, and a quiet screen returns nothing
- [ ] #3 The three tests' waits use the accessor instead of their own attribute lists, and keep their two-consecutive-checks rule and their assertions
- [ ] #4 The three test files pass in repeated runs under parallel load (at least 10 runs each, or a soak recorded in the notes)
<!-- AC:END -->
