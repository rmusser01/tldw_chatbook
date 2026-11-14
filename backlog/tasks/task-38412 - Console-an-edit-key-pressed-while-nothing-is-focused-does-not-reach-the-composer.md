---
id: TASK-38412
title: >-
  Console: an edit key pressed while nothing is focused does not reach the
  composer
status: To Do
assignee: []
created_date: '2026-10-31 02:37'
labels:
  - console
  - composer
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
ChatScreen.on_key treats the Console composer as the default target while nothing is focused. A printable key and Ctrl+C follow that, but Left then Backspace leave the draft unchanged ('abc' stays 'abc'). The keymap characterisation test test_an_edit_key_reaches_the_composer_while_nothing_is_focused pins the expected 'ac' and is red on dev 7536ac3b87. Its precondition used to race the Console's activation focus restore, which hid this; TASK-37870 made that precondition deterministic and exposed the real assertion. Either the routing or the test's expectation is wrong.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 With nothing focused, Left and Backspace edit the Console draft as the composer would, or the owner rules the test's expectation stale and it is changed to match
- [ ] #2 test_an_edit_key_reaches_the_composer_while_nothing_is_focused passes under the plain command
<!-- AC:END -->
