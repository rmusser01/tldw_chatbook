---
id: TASK-832
title: Decide the direction of the shared rail glyph constants
status: Done
assignee: []
created_date: '2026-07-26 22:08'
updated_date: '2026-07-27 20:33'
labels:
  - tech-debt
  - architecture
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Widgets/destination_rail.py re-declares the collapse/expand glyph literals rather than importing them from Chat.console_glyphs, so the shared widget carries no Chat-layer dependency. A test asserts the two stay equal. The final review of PR #940 argued this installs a hidden bidirectional lockstep: neither module can change its glyphs without a test in a third file going red, and that enforcement is invisible to static tools. It proposed inverting so destination_rail owns the constants and console_glyphs re-exports them. The counter-argument is that inverting makes the Chat layer import from Widgets, which is the worse dependency direction. This needs a decision before a second destination adopts the base.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A decision is recorded with its rationale in the repo,The chosen direction is implemented and the now-redundant guard test is removed or rewritten to match,Exactly one module defines each glyph constant,No module imports across the layer boundary in the rejected direction
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Decision recorded as ADR-034 (backlog/decisions/034-shared-rail-disclosure-glyphs.md) and implemented.

destination_rail owns GLYPH_EXPANDED/GLYPH_COLLAPSED; Chat/console_glyphs re-exports them; UI/Evals/library_rail imports from the shared module and no longer depends on Chat. Each constant is defined once.

The task's counter-argument -- that inverting makes Chat import from Widgets, 'the worse dependency direction' -- did not survive measurement: Widgets->Chat is 40 imports on dev, Chat->Widgets is 1. The direction called worse is the rare one. Decisive evidence that these glyphs were never Console vocabulary: UI/Evals/library_rail.py, a Lab destination, was importing them from Chat.console_glyphs.

The guard test now asserts 'is', not '==': identity can only hold with a single definition, so it fails if anyone re-declares the literal, where equality would pass for a re-introduced duplicate.
<!-- SECTION:NOTES:END -->
