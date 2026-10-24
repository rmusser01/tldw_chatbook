---
id: TASK-36696
title: Keep pending MCP file reads from replacing newer import drafts
status: To Do
created_date: 2026-10-24 17:34
labels:
- textual-patterns
- reliability
priority: medium
documentation:
- Docs/superpowers/qa/2026-10-24-textual-book-after-pr3228.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
MCP file-import completion looks up whichever import panel currently exists and replaces its text. Later typing, or Cancel followed by a fresh import form, can therefore lose its draft to an older file selection.

### Evidence and scope

Verified after PR #3228 merged, against dev `96cae7eec00473bbfb5687ccfd0685eba2785add`. Actual Import -> From file -> picker Path callback -> real to_thread read. Both same-panel typing and Cancel/reopen replacement retain later-draft before release, then receive older fixture JSON. Untouched-panel control loads normally. Fake hub only supplies catalog; no credentials or network.

Book basis: PDF145–146 and158 / printed136–137 and149 explains widget queues and asynchronous handlers. Originating panel/draft identity is an application-specific deduction.

Affected owners: [mcp_workbench.py](../../tldw_chatbook/UI/MCP_Modules/mcp_workbench.py), [mcp_profile_form.py](../../tldw_chatbook/UI/MCP_Modules/mcp_profile_form.py).

### Existing ownership and decisions

TASK-32791 covers workspace-root drafts, TASK-32780/32787 Tool Pack review/write lifetimes, TASK-32839 recovery-review completion and TASK-36361 other Settings drafts. Their complete acceptance criteria do not own MCP server JSON file-import publication.

Related tasks: [TASK-32791](../tasks/task-32791%20-%20Preserve-MCP-root-drafts-and-ordered-save-outcomes.md), [TASK-32780](../tasks/task-32780%20-%20Keep-Tool-Profile-reviews-within-the-initiating-Settings-visit.md), [TASK-32787](../tasks/task-32787%20-%20Preserve-Tool-Profile-writes-across-Settings-recreation.md), [TASK-32839](../tasks/task-32839%20-%20Keep-restored-MCP-review-completion-scoped-and-refresh-passive-views.md), [TASK-36361](../tasks/task-36361%20-%20Preserve-newer-Appearance-Storage-and-Console-Behavior-drafts-after-pending-Saves.md).

ADR required: no.
ADR paths: [033-application-session-state-ownership](../decisions/033-application-session-state-ownership.md), [240-mcp-destination-retirement-tools-hub-in-settings](../decisions/240-mcp-destination-retirement-tools-hub-in-settings.md).
Reason: Routine stale UI publication repair within existing import-panel/workbench ownership and the canonical Settings host; no new service or permission boundary.

[Post-merge book audit and evidence](../../Docs/superpowers/qa/2026-10-24-textual-book-after-pr3228.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 An untouched originating import panel receives its explicitly selected file normally.
- [ ] #2 Typing or clearing the import text after file selection remains visible when the older read completes, including invalid raw drafts.
- [ ] #3 Cancel, replacement or reopening invalidates the old panel read so it cannot publish into a new import panel.
- [ ] #4 Mounted checks cover the actual picker callback and filesystem await with same-panel, replacement and ordinary controls.
<!-- AC:END -->
