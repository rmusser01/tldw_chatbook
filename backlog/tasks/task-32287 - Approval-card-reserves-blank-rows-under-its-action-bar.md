---
id: TASK-32287
title: Approval card reserves blank rows under its action bar
status: Done
assignee: []
created_date: '2026-09-10 19:16'
updated_date: '2026-09-10 21:22'
labels:
  - console
  - approvals
  - ui
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A one-row batch renders a 17-row card at 50 rows, with roughly ten empty rows between the action bar and the bottom border; at 80x30 the action bar was not on screen. No CSS exists for the card container, its action bar or ChatTaskCards; cause not traced. Source: live approval-card / MCP-permissions UX review 2026-09-10 against dev 3315241674 (snapshot .impeccable/critique/2026-09-10T17-31-53Z__hatbook-widgets-chat-widgets-chat-approval-card-py.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The card's height matches its content (title, optional deadline and summary, rows, action bar) plus padding.
- [x] #2 At 80x24 a one-row card shows its action bar.
- [x] #3 A mounted test pins the height for one-row and three-row batches.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Three unstyled containers kept Textual's `height: 1fr` default and reserved space the card's content never used, so `.ds-approval-card`'s own `height: auto` could not shrink it: `#approval-batch-body` (a Container) and `#approval-batch-actions` (a Horizontal) expanded to fill the card -- a one-row batch measured 46 lines in a 50-row harness, with the ten-plus blank rows under the Approve-all/Submit/Deny-all bar the live pass reported -- and `#console-task-surface` (ChatTaskCards) took a 1fr share of the session column one level up: 17 rows at 200x50 for an 11-row card, but only 6 rows at 80x24, where it clipped the action bar out of the compositor entirely (the bar's own coordinates hit-tested to the transcript's empty state). Fix is three `height: auto` rules in the bundle source `css/components/_agentic_terminal.tcss` (bundle + agentic split sheet regenerated with build_css.py); no widget or Python change. Production after: one-row card 17 -> 14 rows at 200x50 (transcript 18 -> 21), and at 80x24 the Submit button now hit-tests to itself. Three mounted tests in Tests/UI/test_chat_approval_card.py pin it: one-row card <= 12 lines and three-row <= 22 under the real bundle in a ChatTaskCards harness, plus an 80x24 production-Console test that asserts get_widget_at on the Submit button's own centre returns the button (region alone was not evidence -- pre-fix the region was on screen while the widget was clipped away). Known ceiling, not regressed here: a row is 7 lines tall since tasks 5-7, so a 3-row batch still needs 22 rows and squeezes the transcript at 80x24; the rows container's max-height: 15 scroll cap bounds it. Files: tldw_chatbook/css/components/_agentic_terminal.tcss, tldw_chatbook/css/tldw_cli_modular.tcss, tldw_chatbook/css/screen_agentic_console.tcss, Tests/UI/test_chat_approval_card.py.

Review round 1: the same `1fr`-inside-`auto` ballooning applied to the other cards `ChatTaskCards` hosts once the surface went `auto` -- `#chat-resume-panel`, `#chat-skill-install-card` and `#chat-skill-script-card` each measured 50 rows in a 50-row harness (the two skill cards through their unstyled `Horizontal` button rows, one level down), so all five ids now carry `height: auto`. `ChatQuestionCard` needed nothing: it already ships `height: auto; max-height: 24` in its own BUNDLED_CSS and measured 14 rows throughout. The harness also pinned `CSS_PATH` to the bundle alone, which never loads the console split sheet holding the `#console-task-surface` rule; it now uses `APP_STYLESHEETS`, and a fourth test (`test_every_task_surface_card_hugs_its_content`) pins each sibling card at <= 8 lines and the surface at exactly its visible children's height.
<!-- SECTION:NOTES:END -->
