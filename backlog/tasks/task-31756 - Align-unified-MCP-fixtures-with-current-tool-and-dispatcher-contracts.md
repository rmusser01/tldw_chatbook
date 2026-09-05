---
id: TASK-31756
title: Align unified MCP fixtures with current tool and dispatcher contracts
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 21:43'
updated_date: '2026-09-05 21:44'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Two unified stdio tests fail before exercising their intended assertions: the Library manifest fixture still expects three retired collection tools, and the fake connection constructor omits the current server-request dispatcher keyword. Real cold subprocess coverage already passes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The in-process manifest retains exactly the current21Library tool names and no retired collection tools while wire exposure remains builtin-only
- [x] #2 The fake legacy connection explicitly accepts and verifies the default server-request dispatcher and retains the exact gateway line-limit assertions
- [x] #3 Complete unified stdio and canonical Library capability files pass with no runtime changes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve complete baseline27passed2failed and inspect retirement commit5dd1077df6 plus canonical capability and client constructor seams.
2. Update only stale test inventory and explicit fake-constructor keyword, preserving exact wire and byte-limit checks.
3. Run both complete files and static checks, review diff and record evidence.
ADR required: no
ADR path: N/A
Reason: Test-only alignment with existing retired tool surface and dispatcher seam, not a protocol change.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Test-only repair: current21-tool Library manifest explicitly excludes all3retired collection names; wire remains exactly BUILTIN_TOOL_NAMES. Legacy fake constructor accepts and asserts server_request_dispatcher=None without changing its1,048,576-byte output limit. Complete unifiedstdio plus canonicalLibrarytools67passed14.69s, including realcoldmodule and legacychild integration; baseline27passed2failed in unifiedfile. Wholefile Ruff/format and diffchecks pass. XML:/private/tmp/tldw-31756-mcp-contracts-fixed.xml. No runtime/protocol/limits change; ADR not required.
<!-- SECTION:NOTES:END -->
