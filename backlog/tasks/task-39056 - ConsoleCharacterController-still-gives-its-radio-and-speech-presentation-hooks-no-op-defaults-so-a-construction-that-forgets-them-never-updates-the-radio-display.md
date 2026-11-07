---
id: TASK-39056
title: "ConsoleCharacterController still gives its radio and speech-presentation hooks no-op defaults, so a construction that forgets them never updates the radio display"
status: Done
assignee:
- '@claude'
created_date: 2026-11-06 10:09
updated_date: 2026-11-06 17:59
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
Found in round 10 of the 2026-10-23 burn-down; checked on dev b2c618185f, 2026-11-06, and on Architecture PR #3436 (head b7c8e809b3), which lands TASK-38988. Line numbers are dev's; ones marked #3436 are from that head. TASK-38988's notes list it as a follow-up for once TASK-36549 and TASK-36550 land. Both are Done on dev, since 2026-10-24.

- `ConsoleCharacterController.__init__` gives both hooks no-op defaults (`UI/Console_Modules/character.py:98-99`, from b333cebdfe):
  - `speech_presentation: Callable[[], Any | None] = lambda: None`;
  - `render_radio: Callable[..., None] = lambda *args, **kwargs: None`.
- The only production caller, `UI/Console_Modules/wiring.py:1780-1784`, passes both.
- TASK-38988 adds an allowance for the defaults: `optional = {"speech_presentation", "render_radio"}` in `test_character_controller_has_only_named_non_dom_dependencies` (#3436 `Tests/Architecture/test_console_wave6_inventory.py:1880`).
- These test constructors omit both hooks and would need them:
  - `Tests/UI/test_console_character_avatar.py:146`;
  - `Tests/UI/test_console_persona_visual_identity.py:239` and `:294`;
  - the dependency builders at `Tests/UI/test_console_character_avatar_copy_budget.py:152` and `Tests/UI/test_console_character_controller.py:52`.
- No PR lane runs those files.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Both hooks are required keyword arguments with no default, and the `optional` allowance in `test_console_wave6_inventory.py` is gone
- [x] #2 With the allowance removed and the defaults still in place, the wave6 dependency node is red (shown before the source change)
- [x] #3 Every test constructor passes both hooks, and the covering files pass, with counts recorded
<!-- AC:END -->

## Implementation Plan

1. Pass both hooks (`speech_presentation`, `render_radio`) in every test constructor: the two in `test_console_persona_visual_identity.py`, the one in `test_console_character_avatar.py`, and the dependency builders in `test_console_character_avatar_copy_budget.py` and `test_console_character_controller.py`.
2. Remove the `optional` allowance from `test_character_controller_has_only_named_non_dom_dependencies` and show the node red while the defaults are still in place.
3. Make both hooks required keyword arguments (drop the defaults in `ConsoleCharacterController.__init__`); production wiring already passes both.
4. Run the covering files and record the counts.

## Implementation Notes

**Status: Done.**

- `ConsoleCharacterController.__init__` (`UI/Console_Modules/character.py`) now declares `speech_presentation` and `render_radio` as required keyword arguments, with no defaults. The production constructor (`UI/Console_Modules/wiring.py`) already passed both.
- Every test constructor passes both hooks as explicit no-ops: `Tests/UI/test_console_character_avatar.py` (`_bare_console_screen`), the two in `Tests/UI/test_console_persona_visual_identity.py`, and the dependency builders in `Tests/UI/test_console_character_avatar_copy_budget.py` and `Tests/UI/test_console_character_controller.py`.
- `test_character_controller_has_only_named_non_dom_dependencies` (`Tests/Architecture/test_console_wave6_inventory.py`) no longer has the `optional` allowance; it requires every keyword-only dependency to have no default.

**Evidence.**
- Red first (AC#2): with the allowance removed and the defaults still in place, the node failed with `assert False` (`all(default is None ...)`). After the source change, the wave6 file passed 33 of 33, serial.
- Covering files at `-n 4`, load 29-51. The four constructor files ran 109 passed and 1 failed. An earlier run that also included `Tests/UI/test_console_controller_wiring.py` gave 155 passed and 3 failed. None of the failures comes from this change:
  - `test_console_controller_wiring.py::test_character_controller_is_constructed_with_late_bound_screen_edges` fails the same way at base 2684e5453f (`AttributeError: 'types.SimpleNamespace' object has no attribute '_character_console_session_defaults'`, `wiring.py:1807`).
  - `test_console_character_avatar.py` has a timing flake under load, with different nodes each run: first `test_refresh_never_raises_on_bad_image` and `test_visual_identity_cache_uses_the_complete_resolution_identity[...]`, then `test_character_geometry_epoch_consumes_only_one_followup`, then `test_character_geometry_replacement_is_equality_guarded_and_keeps_focus`. At base the same file failed two other nodes in one run. The bad-image, epoch and equality-guard nodes, rerun together, passed both here and at base (3 passed each).

Files: `tldw_chatbook/UI/Console_Modules/character.py`, `Tests/Architecture/test_console_wave6_inventory.py`, the four `Tests/UI` files above.
