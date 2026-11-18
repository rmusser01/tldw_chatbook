---
id: TASK-41631
title: "The Console live-work MCP label says \"Not wired\" instead of the unbound-store reason"
status: To Do
assignee: []
created_date: 2026-11-17 19:05
labels:
- follow-up
- console
- mcp
- ux
priority: low
references:
- tldw_chatbook/Chat/console_live_work.py
dependencies:
- TASK-41308
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during the TASK-41306..41320 follow-up burn-down (2026-11-17, merged as #3581..#3593). Checked on dev 97f8c58c0c; line numbers are that tree's, and paths are under `tldw_chatbook/` unless they start with `Tests/` or `backlog/`.

**Why.** When the MCP store is unbound, the Inspector row now names the reason (TASK-41308), but the live-work summary in `Chat/console_live_work.py:378` still sets `mcp_label = "Not wired"`. The two surfaces disagree, and "Not wired" suggests a setup step the user can take, when the actual cause is that the profile's MCP settings file can't be used.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 With the MCP store unbound, the live-work MCP label carries the same reason as the Inspector row
- [ ] #2 Other "Not wired" cases (no store problem) keep their current label
- [ ] #3 A test covers both cases
<!-- AC:END -->
