---
id: TASK-1744
title: Shared card-to-prompt assembly with full-card fidelity in Console
status: Done
assignee: []
created_date: '2026-08-01 11:16'
updated_date: '2026-08-01 22:39'
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
- [x] #1 One shared card->prompt function is used by both the character-probe engine and Console's chat path -- no second, independently-maintained copy of the assembly logic remains in either place
- [x] #2 Console's assembled system prompt includes `message_example` and `post_history_instructions` content for cards that carry it, matching what the character-probe engine already sends
- [x] #3 A test proves both callers produce identical prompt assembly output for the same card and steering input
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

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Extracted the ONE card->prompt joiner both the character-probe eval engine
and Console's chat path now share: `compose_character_card_text` in
`Character_Chat/Character_Chat_Lib.py`, next to `replace_placeholders`
(which it calls). It takes plain fields -- name, system_prompt, personality,
description, scenario, message_example, post_history_instructions,
user_name -- not either caller's own container type (CardSnapshot or a raw
Mapping), so both sides adapt their own shape into it rather than the
shared function knowing about either. It joins fields in the engine's
existing order/labels (system_prompt and post_history_instructions
unlabelled; personality/description/scenario/message_example labelled),
resolves {{char}}/{{user}} macros once over the joined text, and returns ""
when every field is empty.

Home: Character_Chat_Lib.py, not a new lighter module. The engine already
had a precedent for this exact tradeoff -- resolve_card_macros already did
a LOCAL import of replace_placeholders from this same heavy module (Pillow,
CharactersRAGDB, world_book_import, TTS profile portability all load at its
module scope) specifically so the engine package's own module-scope import
stays light. compose_system_prompt now does the identical local import for
compose_character_card_text, so this refactor adds zero new import-time
cost to the engine; it was already paying to compose a prompt at all.
Pillow is also a base (non-optional) dependency of this project, so there
is no missing-optional-dep hazard, only import-time weight, and that weight
is already deferred.

Callers:
- Evals/character_probe/prompt.py::compose_system_prompt now builds its
  card_text via the shared function, then keeps its own steering-prepend
  logic (steering is a target-level, non-card concept with no Console
  equivalent, so it stays local to the engine). resolve_card_macros and
  build_messages's first_message handling are untouched -- separate concern
  (single-field macro resolution for the greeting-equivalent opening turn).
- UI/Screens/chat_screen.py::_character_session_prompt_seed now builds its
  system prompt via the same shared function, passing all seven card
  fields (previously only four). The empty-card fallback ("Stay in
  character.") stays OWNED BY CONSOLE, applied as `... or "Stay in
  character."` on the shared function's output -- not folded into the
  shared function, because the engine's own empty-card behavior is
  different and deliberate (an intentionally blank system message, per
  compose_system_prompt's docstring/test). Both existing call sites
  (now ~6912, ~13746 after the docstring insert shifted line numbers)
  needed no changes -- the
  function's signature and return shape are unchanged.

Console's assembled system prompt is a real, user-visible change: it now
includes message_example and post_history_instructions content, and
personality/description/scenario gain "Personality:"/"Description:"/
"Scenario:" labels it did not have before, with "\n\n" separators instead
of "\n". This was the point of the task, not a side effect -- the eval
predicts what Console sends only if both build byte-identical text.

Tests: Tests/UI/test_character_session_prompt_seed.py -- fixed the one
pinned exact-equality assertion that changes under the new label
(description now reads "Description: ..."), added a test proving
message_example/post_history_instructions now reach Console's prompt, and
added test_console_and_engine_compose_byte_identical_system_prompts: a real
card dict fed through _character_session_prompt_seed and an equivalent
CardSnapshot fed through compose_system_prompt (steering=None) must produce
byte-identical output, with {{char}}/{{user}} macros in every field to
prove resolution parity, not just field-inclusion parity. The greeting path
is asserted separately in the same test and is unchanged.

Found and deliberately left alone: a THIRD, independently-maintained
card->prompt joiner, build_preview_system_prompt in
UI/Persona_Modules/personas_preview_controller.py (Personas workbench
preview pane, not Console's chat path or the eval engine). It has its own
divergent behavior (folds a seeded greeting into the system row, its own
empty-fallback). The task's AC and the dispatching instructions scoped this
work to exactly _character_session_prompt_seed and compose_system_prompt;
unifying the preview pane too is out of scope here and would be scope
creep beyond what was asked. Flagging as a natural follow-up, not filing a
new task since one wasn't requested.

Verification: Tests/Evals/character_probe (142 passed, including the 22
pre-existing compose_system_prompt/build_messages tests passing UNCHANGED
-- proof the engine-side refactor is behavior-preserving) +
Tests/UI/test_character_session_prompt_seed.py (5 passed) together;
Tests/UI/test_evals_authoring_e2e/bench_editor/cell_continuation_e2e/
continuation_e2e/deletion_guard/empty_states/results_grid/screen/
snippet_editor/steering_e2e.py (376 passed, 1 pre-existing failure --
test_two_ui_authored_targets_one_steered_light_up_column_mode_delta,
task-1611's "one-model-row-ever" UI bug, reproduced identically against
the pre-task-1744 base commit 7550ddeb6 in a throwaway git worktree, fully
unrelated to card->prompt assembly); Tests/UI/test_console_native_chat_flow.py
-k character (55 passed); Tests/UI/test_personas_workbench.py -k
"character or start_chat" (127 passed); Tests/UI/test_personas_preview.py
(39 passed, confirms the third joiner above is untouched).

Modified files: tldw_chatbook/Character_Chat/Character_Chat_Lib.py (new
compose_character_card_text), tldw_chatbook/Evals/character_probe/prompt.py
(compose_system_prompt delegates), tldw_chatbook/UI/Screens/chat_screen.py
(_character_session_prompt_seed delegates), Tests/UI/test_character_session_prompt_seed.py
(one assertion updated, two tests added), Docs/superpowers/specs/2026-08-01-character-probe-eval-design.md
(marked the Console-divergence section resolved).

Fix round 1 (post-review): converged the THIRD joiner too --
build_preview_system_prompt in UI/Persona_Modules/personas_preview_controller.py
(Personas workbench preview pane). The reviewer's finding that changed the
scope call: before this task, the preview builder was byte-identical to
OLD Console (same 4 fields, same unlabelled "\n" join) -- preview and
Console never disagreed. task-1744's Console change moved Console and left
the preview behind, creating a NEW divergence on a user-facing surface
right next to "Start Chat" -- exactly the class of bug this task exists to
remove. Converged it: build_preview_system_prompt now builds its card text
via compose_character_card_text (all 7 fields, name resolved the same way),
then folds the seeded greeting in on top exactly as before.

Decision on the empty-card fallback (explicitly asked for): the preview
KEEPS ITS OWN fallback decision -- `folded_result or "Stay in character."`,
computed AFTER greeting-folding -- rather than adopting Console's fallback,
which fires the instant the card alone has no prompt fields. This is
deliberate, not an accident: the preview folds a seeded greeting into the
system row (task-1531, so strict providers still see the greeting the user
already read), a step Console's live-session seed has no equivalent of
(Console keeps the greeting as its own separate chat turn). A card with no
prompt fields but a real seeded greeting must still show the
greeting-derived system text in the preview, not the generic fallback --
collapsing straight to Console's fallback rule would silently drop that
greeting content from the preview's own provider-facing prompt. When
greeting="" (the parity test's condition), both fallback rules agree
exactly, which is what the byte-identical test proves.

Test approach followed the review's instruction: extended the existing
parity test in Tests/UI/test_character_session_prompt_seed.py (renamed
test_console_and_engine_compose_byte_identical_system_prompts ->
test_console_engine_and_preview_compose_byte_identical_system_prompts)
to a third caller -- build_preview_system_prompt(card, greeting="") --
rather than adding a separate test, so the point ("ALL surfaces agree") is
one assertion set, not three scattered ones. Tests/UI/test_personas_preview.py's
three build_preview_system_prompt tests: two updated deliberately (label
now present, never loosened -- exact-equality assertions still exact,
just with "Description: " prepended), one (empty-record fallback) needed
no change. Added two new tests there: message_example/post_history_instructions
now reach the preview (mirrors the Console-side AC #2 test), and a persona-
profile record (no personality/description/scenario/message_example/
post_history_instructions keys at all, matching PersonasScreen._profile_record's
actual shape) still composes cleanly -- confirms no persona-specific
branching was needed, absent fields already read the same as empty ones.

All three surfaces (Console, character-probe engine, Personas preview) now
share exactly one card->prompt composer: Character_Chat_Lib.compose_character_card_text.

Verification (foreground, /private/tmp/tldw-venv/bin/python -m pytest):
Tests/UI/test_character_session_prompt_seed.py Tests/UI/test_personas_preview.py
Tests/UI/test_personas_workbench.py Tests/Evals/character_probe -p no:randomly
-> 475 passed, 0 failed (5 pre-existing unrelated asyncio-mark warnings on
sync test functions in test_personas_preview.py, present before this round
too).

Modified files (fix round 1): tldw_chatbook/UI/Persona_Modules/personas_preview_controller.py
(build_preview_system_prompt delegates to the shared composer),
Tests/UI/test_personas_preview.py (2 assertions updated, 2 tests added),
Tests/UI/test_character_session_prompt_seed.py (parity test extended to
three callers).

PR review round: fixed a whitespace-only-field bug in compose_character_card_text
(Character_Chat/Character_Chat_Lib.py). The four labelled fields
(personality/description/scenario/message_example) tested the RAW value's
truthiness before labelling, so a field containing only spaces/tabs/a bare
newline was treated as present: it got a label prepended, and the label
text alone ("Personality:") survived the final per-part .strip(), so (a)
every affected session shipped a dangling label with nothing after it, and
(b) a card whose fields were ALL whitespace never composed to "", so
Console's "Stay in character." fallback (which checks for an empty
composed string) could never fire -- the card silently got a bare label as
its entire system prompt. This directly contradicted the function's own
docstring, which already claimed "" for "every field ... empty or
whitespace-only".

Fix: test `field.strip()` for presence instead of `field` itself, for
every field (labelled AND the two unlabelled ones, system_prompt/
post_history_instructions, even though those two already happened to
degrade correctly via the existing join-time filter -- unified for
consistency per the review). The RAW (unstripped) value is still what gets
embedded in the label f-strings, so a genuine value's own interior
whitespace stays byte-exact -- only the presence TEST changed, not what
gets written when a field is present.

TDD discipline followed literally: wrote 5 direct unit tests in a new file
(Tests/Character_Chat/test_compose_character_card_text.py -- no prior
direct-unit-test coverage of this function existed, only caller-level
coverage), ran them against the UNFIXED code first and confirmed exactly
the 2 bug-targeted tests failed (whitespace-only labelled field, all-
whitespace card) while the 3 others already passed (unlabelled-field
whitespace handling, interior-whitespace preservation, real-value
edge-trimming) -- confirming those 3 document pre-existing correct
behavior, not coincidental passes. Also added 2 caller-level tests in
Tests/UI/test_character_session_prompt_seed.py
(test_whitespace_only_card_falls_back_to_stay_in_character,
test_whitespace_only_card_agrees_with_the_preview_builder) and verified
THOSE fail against the unfixed function too (temporarily restored the
pre-fix file via `git show HEAD:...`, ran the two new tests, confirmed
both red, restored the fix) before trusting the fix made them green.
Re-ran the three-way byte-identical parity test
(test_console_engine_and_preview_compose_byte_identical_system_prompts) --
still holds, unaffected (its fixture card has no whitespace-only fields).

Verification (foreground): Tests/UI/test_character_session_prompt_seed.py
Tests/UI/test_personas_preview.py Tests/UI/test_personas_workbench.py
Tests/Evals/character_probe Tests/Character_Chat/test_compose_character_card_text.py
-p no:randomly -> 482 passed, 0 failed (same 5 pre-existing unrelated
asyncio-mark warnings as the prior round).

Reviewer's second finding (multi-line test docstrings in
test_personas_preview.py needing a blank line after the summary) was ruled
a false positive by the coordinator -- no action taken.

Modified files (PR review round): tldw_chatbook/Character_Chat/Character_Chat_Lib.py
(whitespace-presence fix), Tests/Character_Chat/test_compose_character_card_text.py
(new, 5 direct unit tests), Tests/UI/test_character_session_prompt_seed.py
(2 new caller-level whitespace tests).
<!-- SECTION:NOTES:END -->
