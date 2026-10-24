---
id: TASK-36697
title: Reject malformed MCP import shapes without losing the editable panel
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
The MCP JSON importer assumes args is iterable and env is a mapping. Valid JSON with malformed field containers escapes Preview validation, causing the error keepalive to retire the import panel and its editable draft.

### Evidence and scope

Verified after PR #3228 merged, against dev `96cae7eec00473bbfb5687ccfd0685eba2785add`. Real Preview of args:42 raises TypeError; env:[not-an-object] raises AttributeError. Valid args:[]/env:{} control passes. Installed keepalive guard keeps the host alive while the failing panel/editor detach. No verified whole-app production crash claim.

Book basis: Collateral concrete input-boundary defect found while tracing the book’s queued Preview/panel lifecycle model. Container validation is an existing application requirement, rather than a specific book design.

Affected owners: [mcp_import.py](../../tldw_chatbook/MCP/mcp_import.py), [mcp_profile_form.py](../../tldw_chatbook/UI/MCP_Modules/mcp_profile_form.py).

### Existing ownership and decisions

TASK-241 remains open for regex alignment, mutation remount, slot confirmation and lifecycle cancellation; its complete criteria do not cover malformed containers. TASK-26013 validates spawn-command shapes, TASK-14922 protocol/packaging. TASK-32533 and TASK-33621.13 keep handler errors alive while retiring failing pumps; they do not validate these JSON shapes.

Related tasks: [TASK-241](../tasks/task-241%20-%20MCP-Hub-engineering-follow-ups-from-Phase-2-3-ledgers.md), [TASK-26013](../tasks/task-26013%20-%20MCP-server-spawn-configuration-guard.md), [TASK-14922](../tasks/task-14922%20-%20Resolve-PR-1490-post-merge-review-findings.md), [TASK-32533](../tasks/task-32533%20-%20Library-Notes-the-app-exits-with-an-unhandled-InvalidSelectValueError-during-the-first-timer-edit-journey-and-the-crash-handler-records-no-site.md), [TASK-33621.13](../tasks/task-33621.13%20-%20Console-Choose-folder-for-project-instructions-freezes-the-whole-app-Esc-clicks-and-CtrlQ-ignored.md).

ADR required: no.
ADR paths: [161-component-pattern-library](../decisions/161-component-pattern-library.md), [240-mcp-destination-retirement-tools-hub-in-settings](../decisions/240-mcp-destination-retirement-tools-hub-in-settings.md).
Reason: Routine validation and recoverable errors within the existing importer and host; no new service or security policy.

[Post-merge book audit and evidence](../../Docs/superpowers/qa/2026-10-24-textual-book-after-pr3228.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Malformed args/env container shapes produce a clear import validation error without an uncaught handler exception.
- [ ] #2 Rejected source text stays editable, Import remains disabled, and stale preview candidates cannot be submitted.
- [ ] #3 Correcting the text and pressing Preview recovers normally; supported valid and omitted field shapes retain existing behavior.
- [ ] #4 Parser and mounted Preview regressions cover both observed malformed shapes, a valid control and the installed keepalive semantics.
<!-- AC:END -->
