---
id: TASK-41130
title: "Seed the Library Collections reader layout from the measured width so the first compose does not close the Library pane"
status: To Do
created_date: 2026-11-15 02:20
labels:
- follow-up
- library
- focus
priority: low
references:
- tldw_chatbook/UI/Library_Modules/library_collections_state.py
- tldw_chatbook/UI/Library_Modules/library_collections_controller.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed in round 14 of the 2026-10-23 burn-down (closeout, 2026-11-15). Checked against the round-14 stack top c7d4a41717 (on dev beab5c1e4e); line numbers are that tree's.

**Why.** On a first visit the Collections reader shell composes with a width-0 layout, which closes and disables the Library pane, so a focus restore aimed at the Collections row finds it unfocusable. TASK-40328 review r14ls-rev2 fixed the symptom with a one-time retry; seeding the layout from a real width removes the cause.

**Evidence.** `library_collections_state.py:156-162`: `reader_layout` defaults to `resolve_adaptive_reader_layout(0, AdaptiveReaderLayoutPreferences(), LIBRARY_COLLECTIONS_READER_PROFILE)`. `library_collections_controller.py:270` (the retry): "a first visit composes this shell with the width-0 …" layout; the shell's first real layout sync reopens the pane.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The first Collections compose uses a layout resolved from the screen's measured width, so the Library pane is open and focusable in the first frame
- [ ] #2 The focus retry is removed if it is no longer needed, and the Enter-on-Collections-row test still passes
- [ ] #3 A test checks the first frame's pane state, red before the change
<!-- AC:END -->
