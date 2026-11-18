---
id: TASK-38599
title: Fix UTF-8 detection for file references
status: Done
created_date: 2026-11-02 14:28
labels:
- bug
- chat
priority: medium
updated_date: 2026-11-03 02:55
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The binary heuristic refuses UTF-8 files whose first 8 KB exceed 30 percent non-ASCII so CJK or Cyrillic notes are silently not injected when referenced with @file. See bug-sweep-2026-11-02-dev.md section 1.7.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 At-file references accept UTF-8 files whose first 8 KB exceed 30 percent non-ASCII
- [x] #2 Binary heuristic applies only after a failed UTF-8 decode
- [x] #3 Test covers a CJK markdown file
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Strict UTF-8 decode first; printable-ASCII heuristic only on UnicodeDecodeError; size cap unchanged; CJK and Cyrillic fixtures accepted, binary refused. Commit a8cdf9ece5.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
