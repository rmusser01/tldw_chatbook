---
id: TASK-38585
title: Escape model- and user-authored strings rendered as Rich markup
status: To Do
created_date: 2026-11-02 14:27
labels:
- bug
- ui
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Canvas card labels containing model-chosen titles crash the app with MarkupError on every transcript render when the title contains a closing tag, and bracketed text is silently swallowed; conversation dialog titles have the same flaw. See bug-sweep-2026-11-02-dev.md section 1.5.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Canvas card label renders bracketed titles safely without MarkupError
- [ ] #2 Conversation selection dialog title is escaped
- [ ] #3 Regression tests cover both sites
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
