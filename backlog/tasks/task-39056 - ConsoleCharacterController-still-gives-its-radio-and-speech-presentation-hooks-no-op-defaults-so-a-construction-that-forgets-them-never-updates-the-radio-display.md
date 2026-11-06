---
id: TASK-39056
title: "ConsoleCharacterController still gives its radio and speech-presentation hooks no-op defaults, so a construction that forgets them never updates the radio display"
status: To Do
created_date: 2026-11-06 10:09
labels:
- follow-up
- console
- architecture
priority: low
references:
- tldw_chatbook/UI/Console_Modules/character.py
- tldw_chatbook/UI/Console_Modules/wiring.py
- Tests/Architecture/test_console_wave6_inventory.py
- Tests/UI/test_console_character_avatar.py
- Tests/UI/test_console_persona_visual_identity.py
- Tests/UI/test_console_character_avatar_copy_budget.py
- Tests/UI/test_console_character_controller.py
dependencies:
- TASK-36549
- TASK-36550
- TASK-38988
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 10 of the 2026-10-23 burn-down; checked on dev b2c618185f, 2026-11-06, and on the Architecture branch that lands TASK-38988 (the state after it merges). TASK-38988's notes list it as a follow-up for once TASK-36549 and TASK-36550 land. Both are Done on dev, since 2026-10-24.

- `ConsoleCharacterController.__init__` gives both hooks no-op defaults (`UI/Console_Modules/character.py:98-99`, from b333cebdfe):
  - `speech_presentation: Callable[[], Any | None] = lambda: None`;
  - `render_radio: Callable[..., None] = lambda *args, **kwargs: None`.
- The only production caller, `UI/Console_Modules/wiring.py:1780-1784`, passes both.
- TASK-38988 adds an allowance for the defaults: `optional = {"speech_presentation", "render_radio"}` in `test_character_controller_has_only_named_non_dom_dependencies` (`Tests/Architecture/test_console_wave6_inventory.py:1880`, branch tip).
- These test constructors omit both hooks and would need them:
  - `Tests/UI/test_console_character_avatar.py:146`;
  - `Tests/UI/test_console_persona_visual_identity.py:239` and `:294`;
  - the dependency builders at `Tests/UI/test_console_character_avatar_copy_budget.py:152` and `Tests/UI/test_console_character_controller.py:52`.
- No PR lane runs those files.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Both hooks are required keyword arguments with no default, and the `optional` allowance in `test_console_wave6_inventory.py` is gone
- [ ] #2 With the allowance removed and the defaults still in place, the wave6 dependency node is red (shown before the source change)
- [ ] #3 Every test constructor passes both hooks, and the covering files pass, with counts recorded
<!-- AC:END -->
