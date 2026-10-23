---
id: TASK-36386
title: 'Recover boot CSS headroom: generated widget sheets copy Python indentation, and one mid-session modal rides the boot parse'
status: To Do
created_date: 2026-10-22 23:58
labels:
- follow-up
- css
- performance
priority: medium
references:
- tldw_chatbook/css/build_css.py
- tldw_chatbook/css/widget_defaults_self.tcss
- tldw_chatbook/css/widget_defaults_scoped.tcss
- Tests/Performance/test_boot_css_byte_budget.py
dependencies:
- TASK-35649
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the review of TASK-35649 (PR #3217, 2026-10-22). Once TASK-35649 lands, `MAX_BOOT_PARSED_CSS_BYTES` has 775 B of headroom (607,315 of 608,090 B). The PR's review measured 392 B of boot CSS added on dev in the week before, so the next ordinary CSS PR is close to a red build, the failure mode ADR-097's Context names. Two ways to shed bytes with no rendering change:
- **Indentation.** `widget_css.render_stylesheets` copies each Python block's indentation verbatim into the generated `widget_defaults_self.tcss` and `widget_defaults_scoped.tcss`. At #3217's head, the two sheets carry 30,848 B of leading whitespace and 1,212 blank lines. About 19.9 KB of that is copied per-block Python indent; the rest is the CSS's own declaration indent.
- **FileNotesHistoryDialog.** After TASK-35649 it is a mid-session ModalScreen on the boot parse (992 B). The repo's route for deferring a rarely opened mid-session modal is `MODAL_OWNED_SPLIT_SHEETS` (build_css.py around :386-399; TASK-33628.2).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The generated widget-defaults sheets carry no copied per-block source indentation, and the CSS bundle still reproduces (preflight)
- [ ] #2 FileNotesHistoryDialog's rules leave the boot parse through MODAL_OWNED_SPLIT_SHEETS, or the task records why they stay
- [ ] #3 Every affected widget renders the same at 211x44 and 235x52 before and after
- [ ] #4 The boot CSS budget's headroom is re-measured and recorded
<!-- AC:END -->
