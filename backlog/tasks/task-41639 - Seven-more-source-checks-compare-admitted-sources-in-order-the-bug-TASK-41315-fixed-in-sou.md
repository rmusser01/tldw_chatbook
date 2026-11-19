---
id: TASK-41639
title: "Seven more source checks compare admitted sources in order, the bug TASK-41315 fixed in sources_changed()"
status: To Do
assignee: []
created_date: 2026-11-17 19:05
labels:
- follow-up
- agents
- console
- mcp
priority: high
references:
- tldw_chatbook/Agents/legacy_hook_ownership.py
- tldw_chatbook/Chat/console_plugin_hook_preparation.py
- tldw_chatbook/Chat/console_reference_preparation.py
- tldw_chatbook/Chat/console_workflow_runs.py
dependencies:
- TASK-41315
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during the TASK-41306..41320 follow-up burn-down (2026-11-17, merged as #3581..#3593). Checked on dev 97f8c58c0c; line numbers are that tree's, and paths are under `tldw_chatbook/` unless they start with `Tests/` or `backlog/`.

**Why.** TASK-41315 found that every Console agent turn with a context-returning MCP hook was refused before its first model call. `sources_changed()` compared the admitted sources as an ordered sequence, and the same set captured in a different order read as a change. The fix compares them as a set. The same `captured_sources(...) != sources` comparison remains here:
- `Agents/legacy_hook_ownership.py:205`: likely live (same refusal path)
- `Chat/console_plugin_hook_preparation.py:177`: plausibly live
- `Chat/console_reference_preparation.py:312`: plausibly live
- `Chat/console_workflow_runs.py:241`, `:390`, `:947`, `:2263`
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each site compares admitted sources without depending on order, or a comment there says why order matters
- [ ] #2 A test per live site captures the same sources in a different order and asserts no refusal; red before the fix
<!-- AC:END -->
