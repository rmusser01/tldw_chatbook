---
id: TASK-1744
title: Shared card-to-prompt assembly with full-card fidelity in Console
status: In Progress
assignee: []
created_date: '2026-08-01 11:16'
updated_date: '2026-08-01 20:44'
labels:
  - evals
  - roleplay
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The character-probe eval engine (`tldw_chatbook/Evals/character_probe/prompt.py`) composes a
card's system prompt from every field that shapes voice, including `message_example` and
`post_history_instructions`. Console's own card->prompt joiner
(`UI/Screens/chat_screen.py::_character_session_prompt_seed`) sends only `system_prompt`,
`personality`, `description`, and `scenario` -- it omits both fields today, so a character in
Console does not behave exactly as its author wrote it, and a probe run is not byte-identical to
what Console actually sends.

This was originally meant to be a non-issue: the phase 1 design spec called for the eval to reuse
Console's existing card->prompt path rather than duplicate it. That reuse turned out to be
impossible as scoped -- the only implementation lived in `UI/Screens/chat_screen.py`, and the
eval engine may not import from `UI/` (the eval must stay a pure, UI-independent package). The
engine ended up with its own, more complete assembly instead, which is the behavior the human
has ruled correct: full-card fidelity stays, and Console should be brought up to match it, not
the other way around.

The goal is to extract one card->prompt function that both the character-probe engine and Console
use, with Console gaining `message_example` and `post_history_instructions` so characters behave
as authored in real chat, not just in probe runs. This also finally removes the duplicate prompt
logic the original spec wanted to avoid but could not deliver, since the shared function can now
live somewhere both the engine and the UI layer are permitted to import from.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 One shared card->prompt function is used by both the character-probe engine and Console's chat path -- no second, independently-maintained copy of the assembly logic remains in either place
- [ ] #2 Console's assembled system prompt includes `message_example` and `post_history_instructions` content for cards that carry it, matching what the character-probe engine already sends
- [ ] #3 A test proves both callers produce identical prompt assembly output for the same card and steering input
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a shared, non-UI composer `compose_character_card_text` to
   `Character_Chat/Character_Chat_Lib.py` (next to `replace_placeholders`,
   which it calls): takes plain fields (name, system_prompt, personality,
   description, scenario, message_example, post_history_instructions,
   user_name), joins them in the engine's existing field order/labels
   (system_prompt and post_history_instructions unlabelled; personality/
   description/scenario/message_example labelled), resolves {{char}}/
   {{user}} macros once over the joined text, and returns "" when every
   field is empty. Both callers already know how to defer this module's
   heavy imports (Pillow, CharactersRAGDB) via a local import inside the
   composing function, so this adds no new import-time cost to either side.
2. Rewire `Evals/character_probe/prompt.py::compose_system_prompt` to build
   its `card_text` via the shared function instead of its own inline
   card_parts/join/resolve_card_macros block, then keep steering-prepend
   logic as today. `resolve_card_macros`/`build_messages` (first_message
   handling) stay untouched -- they are a different concern (single-field
   macro resolution), not field assembly.
3. Rewire `UI/Screens/chat_screen.py::_character_session_prompt_seed` to
   call the same shared function for system_prompt (passing card fields
   including message_example/post_history_instructions), keep Console's own
   "Stay in character." fallback when the shared function returns "" (the
   Console adapter continues to own this fallback, not the shared
   function), and leave greeting composition (first_message +
   replace_placeholders) unchanged.
4. Update the two Console callers (~6888, ~13722) only if the call
   signature changed -- it does not, so no changes expected there beyond
   what already flows from step 3.
5. Update `Tests/UI/test_character_session_prompt_seed.py`: fix the one
   pinned exact-equality assertion that changes under the new label
   ("Description: ..." prefix), add a test proving message_example/
   post_history_instructions now reach Console's system prompt, and add a
   byte-identical cross-boundary parity test (real card dict fed to
   `_character_session_prompt_seed`, an equivalent `CardSnapshot` fed to
   `compose_system_prompt`, asserting equal output) with macros in the
   fields to prove resolution parity too.
6. Run `Tests/Evals/character_probe`, `Tests/UI/test_character_session_prompt_seed.py`,
   and `Tests/UI/test_evals_*` in the foreground; confirm the engine's
   existing prompt tests pass unchanged (proves the refactor is behavior-
   preserving on the engine side) and report red->green accurately.
<!-- SECTION:PLAN:END -->
