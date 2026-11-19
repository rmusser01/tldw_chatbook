---
id: TASK-39119
title: "Automatic Console turns anchor their mix-in attempt to an earlier, unrelated user message, and are blocked when that message is not yet saved"
status: To Do
created_date: 2026-11-07 02:03
labels:
- follow-up
- console
- agents
priority: medium
references:
- tldw_chatbook/Chat/console_chat_controller.py
- tldw_chatbook/Chat/console_chat_models.py
- tldw_chatbook/DB/console_prompt_mixins.py
- tldw_chatbook/DB/recovery_core_schema.py
- Tests/UI/test_console_fleet_wake_ui_freshness.py
dependencies:
- TASK-39044
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 11 of the 2026-10-23 burn-down; checked on dev 2684e5453f, 2026-11-07. TASK-39044's notes (round-11 branch fu/r11-modals) report it as the remaining load flake in the fleet freshness node. Line numbers are dev's.

**The anchor.**
- Before a reply, the controller picks the last USER row before the assistant row as the mix-in attempt's anchor (`console_chat_controller.py:25022-25034`), with no check of the turn's origin. It then writes the attempt against it (`:25035-25085`).
- Automatic origins are `AGENT_WAKE`, `AGENT_CHAT_START` and `GOAL_ITERATION` (`console_chat_models.py:96-97`). Their nearest USER row belongs to an earlier, manual turn.

**What blocks the turn.**
- The insert trigger `console_prompt_mixin_attempt_anchor_insert` (`DB/recovery_core_schema.py:1167`) refuses an anchor that is not yet a saved message (`mixin_anchor_ownership`), and so do `DB/console_prompt_mixins.py:173`, `:421` and `:443`.
- The turn is then blocked with "Mix-in attempt could not be saved; response was not sent." The anchor requirement dates from 7970fbcada (2026-11-01).

**Seen in practice.**
- `test_wake_turn_in_a_nonviewed_session_flips_the_tab_glyph_off_running` failed this way in a round-11 stress run on the modals branch ("BLOCKED … 'Mix-in attempt could not be saved…'"). TASK-39044's notes record 2 of 6 failures at load 58.
- The list also said `test_composer_blocked_copy_names_the_wake_not_provider_setup` "fails on base the same way". That did not reproduce: it passed 2 of 2 on dev under a bootstrap marker, and no log shows it failing with this copy.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A wake, agent chat start or goal iteration never records its mix-in attempt against a user message from an earlier turn
- [ ] #2 Such a turn is never blocked because an earlier user message is not yet saved; a deterministic test holds that save back and is red on dev
- [ ] #3 Manual-turn anchoring and refusal are unchanged, and the existing mix-in tests pass by name
- [ ] #4 The freshness node passes 10 of 10 at `-n 4`, with the load recorded
<!-- AC:END -->
