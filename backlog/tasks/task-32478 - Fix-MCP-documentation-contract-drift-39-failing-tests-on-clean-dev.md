---
id: TASK-32478
title: Fix MCP documentation-contract drift (39 failing tests on clean dev)
status: To Do
assignee: []
created_date: '2026-09-11 23:54'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Pre-existing on origin/dev (verified 2026-09-11, unrelated to the MCP Hub UX program): Tests/MCP/test_mcp_documentation_contract.py has 39 failures -- the documented standalone/private inventory lists drifted from the code enumerators the contract test reads. Sample failure: Docs/Design/MCP.md documents 'Library tools excluded from standalone (24)' but the actual code list has 21 entries; the same drift hits Docs/User_Guide/mcp.md (26 mcp.md-scoped failures) and release-recovery-setup.md. The contract test IS the oracle: regenerate each documented count + backtick name list from the code enumerators until the suite greens.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 test_mcp_documentation_contract.py fully green on dev,Every documented count matches its code enumerator,Name lists match verbatim
<!-- AC:END -->
