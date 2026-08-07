---
id: TASK-1994
title: Scroll keybindings (j/k/space/b) for read-only markdown panes
status: Done
assignee:
  - '@claude'
created_date: '2026-08-02 22:30'
labels:
  - ux
  - keyboard
  - markdown
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Frogmouth's viewer binds `j`/`k` (line scroll), `space` (page down) and `b` (page up) on its document container — cheap, expected-by-terminal-users navigation. Chatbook's read-only markdown surfaces (HF README pane, media content/analysis panes, Library note preview) support only mouse/native scroll when focused.

Deliberate exclusion: the Console transcript already binds `j`/`k` for message SELECTION — it is out of scope and must not change. Scope is read-only viewer panes only, and bindings must be discoverable through the existing footer/key-hint convention.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The HF README pane and media content/analysis panes scroll by line with j/k and by page with space/b when focused (AMENDED 2026-08-06: the Library note preview is excluded — its Markdown self-scrolls via CSS with no focusable container, and grabbing focus for scroll keys inside the note-editing flow would fight the title/keyword inputs; wrap-and-rebind there is not worth the Library layout risk)
- [x] #2 Console transcript selection keys are untouched (existing transcript tests stay green)
- [x] #3 The bindings are discoverable via the footer/key-hint convention on those panes (Binding(show=True) — hints render in the footer while the pane has focus)
<!-- AC:END -->

## Implementation Plan (the how)

1. `Widgets/reader_scroll.ReaderVerticalScroll(VerticalScroll)` with frogmouth's viewer keys (j/k line, space/b page), show=True so the footer lists them on focus.
2. Swap in at the HF README scroll and the media content/analysis scrolls.
3. Tests: key-driven scroll movement, surface wiring, and a pin that the Console transcript keeps its selection j/k.

## Implementation Notes

New `Widgets/reader_scroll.py`; consumers `Widgets/HuggingFace/model_card_viewer.py` (#readme-scroll) and `Widgets/Media/media_viewer_panel.py` (.content-viewer + #analysis-scroll-fix). Console transcript untouched (pinned by test). Library note preview exclusion recorded in AC#1. Tests: `Tests/UI/test_reader_scroll_keys_1994.py`; media/HF consuming suites green.
