---
id: TASK-832
title: Decide the direction of the shared rail glyph constants
status: To Do
assignee: []
created_date: '2026-07-26 22:08'
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
