---
id: TASK-39156
title: "The Console composer's expanded row and actions row keep a stale cached visual style from mount, so their background differs from a fresh computation before any breakpoint flip"
status: Done
assignee: ['@claude']
created_date: 2026-11-08 06:54
updated_date: 2026-11-10 16:59
labels:
- follow-up
- console
- ui
- textual
priority: low
references:
- tldw_chatbook/Widgets/Console/console_composer_bar.py
- tldw_chatbook/css/features/_console.tcss
- tldw_chatbook/UI/Console_Modules/frame.py
- Tests/UI/test_console_projection_restyles.py
dependencies:
- TASK-39132
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 12 of the 2026-10-23 burn-down (review r12p-rev1 of TASK-39132, round-12 branch fu/r12-cprod; the probe is in the coordinator's scratch directory). Checked on dev b590a15afd, 2026-11-08. Line numbers are dev's.

**The probe.** A scratch plugin, not committed, wraps the `_styles` helper in `Tests/UI/test_console_projection_restyles.py`. For every widget under the shell, it compares the cached `visual_style` with one computed after it clears `_visual_style` and `_visual_style_cache`.

**On dev.** Run on `test_a_header_breakpoint_flip_restyles_only_what_its_rules_style` (1 passed, 5.6 s, load ~26). At its first `_styles` call, before any breakpoint flip, two widgets differ:
- `Horizontal#console-composer-expanded` and `Horizontal#console-composer-actions`;
- cached `Style(background=Color(36, 47, 56), …)`, against a fresh `Style(background=Color(25, 68, 102), …)`;
- their parent, `ConsoleComposerBar#console-native-composer`, sets background `(1, 120, 212)`.

**In review.** On the round-12 head the review saw the same two nodes stale after the compact and header flips too (4 entries); on the round's base, the same two. The review reported that these rows paint blank cells from the stale style. That visual effect was not reproduced for this filing.

**The rules.** `#console-composer-expanded` is at `css/features/_console.tcss:996` and `#console-composer-actions` at `:1062`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 After mount, and after each compact and header breakpoint flip, both rows' cached visual style equals a fresh computation; a test is red on dev, showing the two stale nodes
- [x] #2 A capture or cell probe, before and after, records in the notes what the stale style painted (blank cells, wrong colour, or nothing visible)
- [x] #3 The test runs on a PR lane (never the fast PR contract), with three timings recorded
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce: add a test to `Tests/UI/test_console_projection_restyles.py` (whole-file UI lane entry) comparing both rows' cached `visual_style` with a fresh computation after mount, across compact (34/35) and header (24/25) flips, and after a blur.
2. Find why they are stale: Textual 8.2.8 keys a widget's `visual_style` on its OWN `styles._cache_key`, but it reads the ancestors' background; the composer's focus/draft restyle (TASK-33620.15.8 / TASK-36511.2) restyles only the bar and its draft text.
3. Fix at that restyle: when the bar's own visual style moved, drop the other descendants' cached style (`notify_style_update`) and repaint them, without restyling them.
4. Cell probe before/after for the notes; run the two restyle files.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Cause.** Textual 8.2.8's `Widget.visual_style` is cached against the widget's own `styles._cache_key`, yet composes every ancestor's background. `ConsoleComposerBar.watch_has_focus` (TASK-36511.2; also reached from `_sync_interaction_classes`, TASK-33620.15.8) restyles only the bar and `#console-command-visible-text`. The focused rule (`#console-native-composer.console-composer-focused`) moves the bar's background, so descendants without a background of their own -- `#console-composer-expanded` and `#console-composer-actions` -- kept the unfocused panel colour cached from mount.

**Fix** (`tldw_chatbook/Widgets/Console/console_composer_bar.py`, `watch_has_focus`). After the scoped restyle, if the bar's own `visual_style` changed, every other bar descendant gets `notify_style_update()` (Textual's own cache clear, as a restyle does) and `refresh()`; no extra stylesheet `apply`, so the TASK-33620.15.8 / TASK-36511.2 restyle-count pins still hold. Breakpoint flips needed no change: the rows were stale from mount and simply stayed stale.

**Test** `test_the_composer_rows_follow_the_bar_background_through_breakpoint_flips` (`Tests/UI/test_console_projection_restyles.py`, a whole-file entry in `scripts/ui_pr_gate_census.txt`, so on the UI PR lane, not the fast contract). Red on base ae9f9e8d21: `AssertionError: after mount`, both rows listed (`#console-composer-expanded: Color(36, 47, 56) != Color(25, 68, 102)`, same for actions). Green after. Timings (`-n 4`, alone): 10.67 s, 13.43 s, 14.54 s call (load 45-70).

**What the stale style painted (AC #2).** Scratch cell probe (not committed; `Region` render of each row, cells where `get_widget_at` returns the row itself: 2 in the expanded row, 18 in the actions row at 160x40), at mount, after the 34-row compact flip, after the 24-row header flip and after a blur. Before the fix the cached style was stale at the first three (`36,47,56` against the bar's `25,68,102`), yet every row-owned cell painted `25,68,102` (and `36,47,56` after the blur), identical to after the fix. So: nothing visible. A container renders `Blank(self.background_colors[1])`, which Textual computes fresh, so the stale cache did not reach these cells; the review's "blank cells" report stays unreproduced.

**Covering runs.** `Tests/UI/test_console_projection_restyles.py` + `Tests/UI/test_console_composer_state_restyle.py` (`-n 4`, load ~60): 25 passed, 1 failed. The failure, `test_a_compact_flip_restyles_only_what_its_rules_style` ("40->34: restyled 28 of 540 shell nodes; the compact rules style 6"), fails the same way with this change reverted (same command, load ~50), so it is not this change's.
<!-- SECTION:NOTES:END -->
