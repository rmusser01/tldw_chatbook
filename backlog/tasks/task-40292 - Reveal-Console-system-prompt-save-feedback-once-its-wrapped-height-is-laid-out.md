---
id: TASK-40292
title: Reveal Console system-prompt save feedback once its wrapped height is laid out
status: Done
created_date: 2026-11-10 06:40
assignee:
- '@claude'
labels:
- console
- ui
- bug
dependencies: []
priority: low
references:
- https://github.com/rmusser01/tldw_chatbook/pull/3480
- https://github.com/rmusser01/tldw_chatbook/pull/3490
- tldw_chatbook/Widgets/Console/console_system_prompt_modal.py
- Tests/UI/test_console_system_prompt_layout.py
updated_date: 2026-11-12 15:30
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed on 2026-11-10 as TASK-40129 (keep the system-prompt editor's actions visible at 80x24) on the unmerged #3490 branch; renumbered and narrowed on 2026-11-12. TASK-40129 was then taken on another branch, and the 80x24 layout itself reached dev with PR #3480 (TASK-39147).

After a Library save, the editor scrolls its feedback line into view right after updating it. That scroll can run against the line's old one-row height, so wrapped feedback can end with its last lines under the docked action row. Under load this made `test_prompt_editor_resize_and_library_feedback_preserve_actions` fail 3 times in 24 runs.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Wrapped Library save feedback is fully painted above the docked actions once its height is laid out.
- [x] #2 The editor's layout test passes repeatedly under parallel load.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
The save-status line is a small `Static` subclass that scrolls itself into view from its own `Resize`, once its wrapped height is known; the existing post-update scroll stays for feedback that does not change height. Verified on 2026-11-10: the resize-and-feedback case failed 3 of 24 runs under six-way load with the post-update scroll alone and passed 36 of 36 with the Resize reveal. A height-only terminal resize does not scroll the body, and a width resize scrolls it the same with or without the reveal.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
