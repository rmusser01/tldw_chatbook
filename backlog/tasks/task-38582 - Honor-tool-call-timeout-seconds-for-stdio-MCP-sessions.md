---
id: TASK-38582
title: Honor tool_call_timeout_seconds for stdio MCP sessions
status: Done
created_date: 2026-11-02 14:26
labels:
- bug
- mcp
priority: high
updated_date: 2026-11-03 02:54
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The stdio transport hardcodes a 10 second wire timeout ignoring the configured knob and kills the server process on any slower tool call, bricking the session and hiding the real cause as transport unavailable. See bug-sweep-2026-11-02-dev.md section 1.3.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Configured timeout reaches the stdio session request deadline
- [x] #2 A tool call slower than 10 seconds completes instead of killing the server
- [x] #3 Teardown happens only on transport uncertainty not plain deadlines
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
tool_call_timeout_seconds plumbed end-to-end to stdio sessions; healthy-transport deadlines fail that request with typed mcp_tool_timeout and keep the server alive. Commit 8da2053aab.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
