---
id: TASK-38598
title: Pin skill remote fetch connections to validated addresses
status: In Progress
created_date: 2026-11-02 14:28
labels:
- security
- skills
priority: medium
updated_date: 2026-11-02 14:51
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The SSRF check resolves once then lets httpx resolve independently and honors proxy env vars, allowing DNS rebinding or proxy bypass of the validated address; the MCP HTTP transport already does this correctly. See bug-sweep-2026-11-02-dev.md section 1.7.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 HTTP client uses trust_env=False
- [ ] #2 Connection is pinned to the validated address or the connected peer IP is revalidated
- [ ] #3 Behavior is consistent with the MCP streamable_http transport policy
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
