---
id: TASK-37655
title: Unicode-normalize path exclusion and sensitive-path matching
status: To Do
created_date: 2026-10-30 10:35
labels:
- security
- tools
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
On APFS, and other filesystems that ignore Unicode normalization when opening files, an NFD spelling of a path reaches a file that the user excluded under its NFC spelling. The shared exclusion and sensitive-path matcher (local_tool_impls._relative_target_is_safe / _is_relative_sensitive_path) casefolds but never normalizes. So a user's binding exclusion with a non-ASCII name (for example naïve.txt, or a folder like Données/) can be bypassed, and the file's content reaches a model. The same applies to names git reports in the other normalization. Found during the TASK-35582 (PR #3286) local review: a reviewer captured an excluded NFD-spelled file as review source. The PR closed its own selection path with an exact-name listing check, but the shared helper is older and also serves fs_read. Built-in sensitive names are ASCII and unaffected.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 An exclusion or sensitive-path rule matches a path regardless of the NFC/NFD spelling of either the rule or the candidate path
- [ ] #2 fs_read and code-review source capture both refuse an NFD-spelled selection of an NFC-excluded file, and an NFC-spelled selection of an NFD-excluded one
- [ ] #3 Regression tests prove both directions and that ASCII case-insensitive matching is unchanged
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
