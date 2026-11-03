---
id: TASK-38585
title: Escape model- and user-authored strings rendered as Rich markup
status: Done
created_date: 2026-11-02 14:27
labels:
- bug
- ui
priority: high
updated_date: 2026-11-03 02:55
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Canvas card labels containing model-chosen titles crash the app with MarkupError on every transcript render when the title contains a closing tag, and bracketed text is silently swallowed; conversation dialog titles have the same flaw. See bug-sweep-2026-11-02-dev.md section 1.5.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Canvas card label renders bracketed titles safely without MarkupError
- [x] #2 Conversation selection dialog title is escaped
- [x] #3 Regression tests cover both sites
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Both sites render raw strings with markup=False - verified strictly stronger than escaping on Textual 8.2.8 which would paint literal backslashes; tests assert bracketed content is preserved. Commit 774b17b7e8.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
