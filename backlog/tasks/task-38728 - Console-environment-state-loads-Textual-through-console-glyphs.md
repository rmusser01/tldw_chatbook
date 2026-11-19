---
id: TASK-38728
title: "Console environment state loads Textual through console_glyphs"
status: To Do
created_date: 2026-11-04 02:52
labels:
- follow-up
- architecture-review-2026-10-25
- console
priority: low
references:
- tldw_chatbook/Chat/console_environment_state.py
- tldw_chatbook/Chat/console_glyphs.py
dependencies:
- TASK-36944.4
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-36944.4 moved the inspector section state into the Textual-free Chat/console_inspector_state.py. Chat/console_environment_state.py still imports Chat/console_glyphs, which imports Widgets.destination_rail, so importing environment state loads Textual and keeps a Chat-to-Widgets edge in layer_direction_allowlist.tsv.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Importing tldw_chatbook.Chat.console_environment_state in a fresh interpreter loads no textual module, asserted by a test
- [ ] #2 The console_glyphs to Widgets.destination_rail allowlist row is deleted, or the task notes name what still needs it
- [ ] #3 The Console environment and inspector tests show no new failures against dev
<!-- AC:END -->
