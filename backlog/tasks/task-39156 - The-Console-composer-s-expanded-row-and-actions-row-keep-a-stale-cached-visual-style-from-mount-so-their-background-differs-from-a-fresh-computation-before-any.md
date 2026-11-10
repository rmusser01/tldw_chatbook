---
id: TASK-39156
title: "The Console composer's expanded row and actions row keep a stale cached visual style from mount, so their background differs from a fresh computation before any breakpoint flip"
status: To Do
created_date: 2026-11-08 06:54
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
- [ ] #1 After mount, and after each compact and header breakpoint flip, both rows' cached visual style equals a fresh computation; a test is red on dev, showing the two stale nodes
- [ ] #2 A capture or cell probe, before and after, records in the notes what the stale style painted (blank cells, wrong colour, or nothing visible)
- [ ] #3 The test runs on a PR lane (never the fast PR contract), with three timings recorded
<!-- AC:END -->
