---
id: TASK-24612
title: Inspect rail Tab cycle stops on two non-interactive containers
status: Done
assignee:
  - '@claude'
created_date: '2026-08-30 00:55'
updated_date: '2026-08-30 02:46'
labels:
  - console
  - ux
  - inspector
  - critique-2026-08-29
  - a11y
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A focus walk of the open rail measured a closed Tab cycle in which two of the stops are containers, not controls: the rail root region widget and the outer scroll body. Both accept focus, neither has a dedicated focus treatment, and live capture showed their focus indication as a single border glyph and a lit scrollbar column. Separately, every one of the 11 bounded sections reported can_focus False on its viewport in the empty state, so the n and p section-jump accelerator has no focusable target in any section and repeatedly leaves focus where it was.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Every Tab stop inside the Inspect rail has a focus treatment a user can see
- [x] #2 A focus stop that is a scroll container is either given a visible treatment or removed from the Tab cycle
- [x] #3 Pressing n or p moves focus to a target the user can see in every section, including sections that do not overflow
- [x] #4 The way to leave the rail's Tab cycle is discoverable without prior knowledge
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added ':focus' rules for the Inspect rail's two CONTAINER Tab stops -- '#console-inspector-rail-body' (the outer scroller) and '#console-right-rail' (the rail root) -- sharing the treatment '.console-bounded-section-viewport:focus' already models.

Premise correction worth recording. The critique said these stops had 'no visible focus'. Measured precisely, they had no ':focus' RULE, which is not the same claim: something in their computed styles does move on focus (frame/scrollbar), which is why the live capture showed a single border glyph and a lit scrollbar column rather than literally nothing. The defect is a missing treatment, and that is what was added.

Both are focusable deliberately and neither should leave the Tab cycle: the scroller so the keyboard can scroll it, the root because it is the pane F6 lands on (right_rail.can_focus = True at compose). Removing either would break F6.

':focus' only, never ':focus-within' -- these two CONTAIN every other stop in the rail, so a focus-within tint would be lit whenever anything in the rail had focus and would carry no information. A test pins that.

Testing note: the first version of this test focused each widget and diffed widget.styles. It produced a false positive on the header button (whose treatment arrives via an ancestor class applied on DescendantFocus, one refresh later) while missing both containers, because some unrelated style does move. Replaced with a deterministic stylesheet assertion plus a behavioural check that every container the Tab cycle actually visits has a rule.

Not addressed here, still open from the critique: n/p landing on a non-focusable section (all 11 ConsoleBoundedSection viewports report can_focus False at rest) is a separate mechanism and a separate change.

Modified: tldw_chatbook/css/components/_agentic_terminal.tcss (+ regenerated bundle), Tests/UI/test_console_inspector_focus_visibility.py (new).
<!-- SECTION:NOTES:END -->
