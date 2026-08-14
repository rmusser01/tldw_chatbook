---
id: TASK-15996
title: 'Update stale DEFAULT_CSS references for consolidated classes'
status: To Do
assignee: []
created_date: '2026-08-14 01:10'
labels:
  - docs
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Four in-tree references still say `DEFAULT_CSS` for classes that now declare `BUNDLED_CSS` after TASK-15450: `Constants.py:1390` (documents the KEEP-IN-SYNC contract from task-264 — a reader will now look in the wrong attribute), `UI/Workbench/help.py:94`, `UI/MCP_Modules/mcp_tools_mode.py:140`, `UI/MCP_Modules/mcp_inspector.py:848`. Cosmetic except the Constants.py contract note. Sweep for others while there (`grep -rn DEFAULT_CSS` filtered to comments/docstrings naming consolidated classes). Found during the TASK-15450 CSS-consolidation review (PR #1616, merged `c3ed2854a`); evidence in the session review record and `Docs/Design/2026-08-11-input-latency-audit.md`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 All four cited references updated to name BUNDLED_CSS (and the contract note reads correctly)
- [ ] #2 A grep sweep confirms no other stale comment/docstring references remain for the 57 consolidated classes
<!-- AC:END -->
