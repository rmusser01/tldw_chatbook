---
id: TASK-37360
title: Preserve Media filter typing and editing context across browse publication
status: To Do
created_date: 2026-10-29 04:49
labels:
- textual-patterns
- library
- media
- inputs
priority: medium
references:
- tldw_chatbook/Widgets/Library/library_media_canvas.py
- tldw_chatbook/UI/Library_Modules/library_media_controller.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep ordinary typing and editing in the Library Media filter intact while loading and accepted browse results publish. A user pausing between letters currently loses the previously typed prefix, so the visible results no longer reflect the query they were composing.

**Evidence:** PR3290's unchanged Reader-identity control failed on source3e414ff824 with applied query `t` and two rows after native `roduct` keys. A native observation with the same keys spaced1.2seconds apart reproduced that exact terminal state. Loading/results create fresh filter Inputs; restored focus selects their prefill, and the next printable character replaces it. The trace shows that character reaching the current attached field, so a stale retained widget is not required. The uninstrumented CI interleaving is inferred from this sufficient matching witness; natural frequency and latency are unmeasured. One native Paste prepares the separate Reader-identity test successfully but does not repair typed input. [Investigation and raw-receipt map](../../Docs/superpowers/qa/2026-10-28-textual-followups-implementation/ci-repairs/media-filter-investigation.md).

**Existing owners:** [TASK-32350](<task-32350 - Library-Media-list-the-filter-box-and-the-applied-filter-can-disagree-—-add-a-persistent-scope-line.md>) owns truthful applied-scope presentation, not preservation of this typed draft. [TASK-28013](<task-28013 - Library-media-list-in-canvas-filter-input-and-sort-control.md>) introduced the in-canvas filter/sort. [TASK-15457](<task-15457 - Library---convert-per-click-whole-screen-recomposes-to-canvas-scoped-sync.md>) and [TASK-281](<task-281 - Library-targeted-updates-instead-of-whole-screen-recompose.md>) concern broader screen-to-canvas update costs. [TASK-34230](<task-34230 - Media-focus-guard-is-released-before-the-recompose-it-covers-when-canvas-sync-falls-back-to-a-whole-screen-recompose.md>) concerns an unmounted-canvas whole-screen fallback guard; this witness uses an attached live Items field. [TASK-34000.18](<task-34000.18 - Library-Media-Read-later-trash-and-Edit-metadata-act-on-the-item-opened-by-link-not-row-1.md>) owns Reader action identity and is preserved by the native Paste setup. [TASK-34000.37](<task-34000.37 - Library-keyboard-focus-survives-dialogs-and-filter-submits-F6-reaches-every-canvas-F1.md>) covers other keyboard/dialog/filter-submit journeys. Coordinate those boundaries rather than duplicating their completed or broader scopes. The fresh417-ref/368-tree/55-worktree task screen found no discovered task acceptance scope for this exact ordinary-debounce prefix replacement.

**ADR assessment:** New ADR expected:no; ADR path:N/A. Existing [ADR-033](../decisions/033-application-session-state-ownership.md), [ADR-086](../decisions/086-library-adaptive-reader-shell.md), [ADR-150](../decisions/150-agent-chat-fork-and-spawn.md) and [ADR-161](../decisions/161-component-pattern-library.md) preserve destination ownership, adaptive layout and design patterns. Reassess only if implementation changes those boundaries. No new query engine, storage contract or remote fallback is requested.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Native ordinary and paced typing across loading and accepted Media browse publications retains the complete current filter draft and its caret/selection/focus instead of replacing earlier characters.
- [ ] #2 An earlier browse completion cannot overwrite later accepted filter edits or editing context; the scope line continues to describe the actually applied results until newer results are accepted.
- [ ] #3 Clear/type/sort/paging behavior and dirty Reader off-page identity/highlight/action contracts retain their established outcomes.
- [ ] #4 Mounted native controls expose the original prefix-replacement mechanism and verify the repaired publication boundaries, with all holds drained inside the App lifetime and unchanged test/privacy guards.
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
