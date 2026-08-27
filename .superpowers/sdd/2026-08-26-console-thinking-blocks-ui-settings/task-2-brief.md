# Task 2 brief — Console thinking disclosures

## Scope

Reuse `ConsoleActivityDisclosure` and transcript-owned expansion state to render
only supported Assistant thinking evidence. Live evidence opens once, then the
first answer/tool/terminal boundary collapses it unless the user has manually
toggled that disclosure. Historical and late terminal evidence starts collapsed
and keeps its full body unmounted until requested.

## Contracts

- Thinking activities are projected from the selected Assistant generation's
  validated `ThinkingEnvelope`; opaque/missing evidence renders no row.
- Existing Assistant-turn, answer, disclosure, focus, selection, scroll, tool
  expansion, pruning, and windowing identities remain intact across updates.
- Trusted hashed activity IDs are the only DOM/selection IDs.
- Mouse, Enter, Space, and existing `o` toggle the same state; a manual toggle
  cancels the pending automatic collapse.
- Displayable detail is the exact stored text. Proprietary detail is exactly
  `Proprietary thinking obfuscated - not available` and never comes from storage.
- Collapsed detail is not mounted. Copy/Inspector resolution can still obtain the
  full text from the owning envelope; Assistant answer copy/speech remains answer-only.

## Non-goals

No Settings work, controller, dependency, binding, footer hint, animation,
parallel disclosure widget, or persistence/provider change.
