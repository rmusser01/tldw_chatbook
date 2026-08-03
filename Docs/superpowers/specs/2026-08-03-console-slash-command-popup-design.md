# Console Slash-Command Popup — Design

Date: 2026-08-03
Status: Approved (design)

## Summary

When a user types `/` in the console composer (`ConsoleComposerBar` on `ChatScreen`), a floating
suggestion popup appears above the composer listing available slash commands and skills, filtered
live as the user keeps typing. Up/Down navigates, Enter/Tab inserts the completion into the
composer (never executes), Escape dismisses. Style and interaction follow the claude-code /
kimi-code / codex-cli inline completion idiom.

## Context

- Composer: `tldw_chatbook/Widgets/Console/console_composer_bar.py` — `ConsoleComposerBar`, a
  hand-rolled editor (draft segments + rendered `Static`). Only used by `ChatScreen`
  (`tldw_chatbook/UI/Screens/chat_screen.py`, instantiated at ~7586).
- Command grammar: `tldw_chatbook/Chat/console_command_grammar.py` — `ConsoleCommandRegistry`
  with three registered commands: `/prompt [name]`, `/system [name]`, `/skills [name] [args]`,
  plus a `KIND_FALLBACK` resolver (`tldw_chatbook/Chat/console_skill_resolver.py`) that resolves
  bare `/skill-name` to the skills handler.
- Dispatch: `chat_screen.py` `_dispatch_console_command()` (~9137); unknown-command hint derives
  from `registry.available_names()`.
- All composer keys are handled at screen level in `ChatScreen.on_key` (~11307); `escape` is
  claimed by `BINDINGS` (priority binding ~591, non-priority ~599).
- No anchored/floating overlay pattern exists yet; existing pickers are `ModalScreen`s
  (`ConsoleModelPopover`, `ConsolePromptPickerModal`, `ConsoleSkillPickerModal`) or inline
  show/hide widgets (`ModelSearchPicker`).

## Decisions (from brainstorming)

- **Scope**: popup lists the three registry commands AND every resolvable skill as `/skill-name`.
  `/skills` additionally gets argument completion: after `/skills ` the popup lists skill names,
  filtered as the user keeps typing.
- **On select**: insert text into the composer with a trailing space; never execute immediately.
- **Rendering**: floating overlay (new pattern), screen-owned, positioned above the composer.
- **Architecture**: screen-owned overlay widget; key routing in `ChatScreen.on_key` / escape
  actions, where all composer key handling already lives.

ADR required: no. Self-contained UI feature following existing widget/screen patterns; no
storage, sync, security, or cross-module contract changes. (The floating-overlay positioning is
new, but it is a local Textual styling concern, not an architectural boundary.)

## Components

### 1. `ConsoleCommandPopup` — `tldw_chatbook/Widgets/Console/console_command_popup.py` (new)

A `Widget` wrapping an `OptionList`.

- Rows render claude-code style: `/name` + description.
- Hidden by default (`display: none`); shown/hidden by the screen.
- Floating: `overlay: screen` (or equivalent layer) with absolute `offset` computed from the
  composer region so the popup's bottom edge sits just above the composer's top edge. Offset is
  recomputed on open and on screen resize. Height capped (~10 rows); `OptionList` scrolls beyond
  that. Width tracks the composer width.
- Never takes focus (non-focusable; focus stays conceptually on the composer — same idiom as
  `ConsolePromptPickerModal`'s synthetic highlight).
- Public API: `show_suggestions(list[Suggestion])`, `hide()`, `is_open` (property or reactive),
  `move_highlight(delta)`, `accept_selected() -> Suggestion | None`.

### 2. Suggestion provider — `tldw_chatbook/Chat/console_command_suggestions.py` (new)

Pure function(s), no Textual imports, trivially unit-testable:

```python
@dataclass(frozen=True)
class Suggestion:
    insert_text: str   # full replacement text for the token being completed
    label: str         # display name, e.g. "/prompt" or "/skill-name"
    description: str

def suggestions_for_draft(
    draft: str,
    registry: ConsoleCommandRegistry,
    skill_names_with_descriptions: list[tuple[str, str]],
) -> list[Suggestion] | None:
    ...
```

Behavior:

- `None` → popup stays hidden (draft doesn't match any completion context, or paste segments
  present — the caller checks `composer.has_paste_segments()`).
- Command mode: draft matches `^/(\S*)$` → entries for `/prompt`, `/system`, `/skills`
  (note: `ConsoleCommand` has no description field — the provider holds a small static
  name → description map for the three commands; skill entries use the resolver's
  descriptions) plus one entry per skill as `/skill-name`, filtered case-insensitively by the
  typed prefix. `insert_text` = `/name ` (trailing space).
- Skill-arg mode: draft matches `^/skills\s+(\S*)$` → entries per skill filtered by the partial
  argument; `insert_text` = `/skills <name> ` (full-draft replacement, keeps implementation
  simple). Once the user types a second argument (`/skills name extra`) the draft no longer
  matches and the popup hides — intentional.
- Empty filtered result → `[]` → caller hides the popup.

Skill names/descriptions come from the same source the fallback resolver uses — the screen
already holds them as `self._console_skill_candidates` (chat_screen.py:1587) — and passes them
in so the provider stays pure.

### 3. `ChatScreen` wiring — `tldw_chatbook/UI/Screens/chat_screen.py`

- Mount `ConsoleCommandPopup` in `compose()` as a sibling overlay of the composer.
- After every composer mutation (the existing paths that call `insert_text`, deletes, paste,
  `clear_draft`, `load_draft`), recompute: if `has_paste_segments()` → hide; else call
  `suggestions_for_draft(composer.draft_text(), registry, skills)` and show/hide the popup
  (recomputing overlay offset when showing).
- Key routing, when popup is open, **before** existing branches:
  - In `on_key` (~11307): `up`/`down` → `move_highlight(∓1)`; `enter`/`tab` → accept
    (runs before the `activate_focused_paste_token()`/send path at ~11379); printable keys fall
    through to the composer as today (popup re-filters via the mutation hook).
  - `escape`: claimed by `BINDINGS`, so both escape actions (priority ~591, non-priority ~599)
    check popup visibility first and close the popup instead of their normal behavior.
- Accept behavior: replace the whole draft with `suggestion.insert_text` (command mode draft is
  just the partial command; arg mode replacement is the full `/skills <name> ` string), close the
  popup — except accepting `/skills` in command mode, whose insert text ends with a space and
  naturally re-triggers skill-arg mode via the mutation hook.

## Data flow

1. User types `/` → `on_key` inserts into composer → mutation hook → provider returns entries →
   popup opens above composer.
2. User types `sk` → each keystroke re-filters → list shrinks to `/skills` + matching skills.
3. Up/Down moves highlight; Enter inserts `/skills ` → mutation hook sees arg mode → popup lists
   skill names → typing filters → Enter inserts `/skills <name> ` → popup closes.
4. Escape at any point closes the popup without touching the draft; Enter with popup closed sends
   as today.

## Error handling / edge cases

- Empty filter result → popup hides (no "no results" row).
- Paste segments in draft → no popup (consistent with command dispatch being gated on
  `has_paste_segments()`).
- Terminal too small / popup near top → height cap + `OptionList` scrolling.
- Screen resize while open → recompute the overlay offset from the composer's new region (the
  composer stays mounted, so its region is always available; recompute rather than hide).
- Popup open + click elsewhere / focus change → popup stays driven purely by draft text, so no
  special handling needed.

## Testing

- Unit tests (`Tests/Chat/test_console_command_suggestions.py`): provider in command mode
  (all entries on `/`, prefix filtering, case-insensitivity, skills included), arg mode
  (filtering, insert text), non-matching drafts → `None`, empty filter → `[]`.
- Textual `run_test()` pilot tests (extend the existing console/chat screen test module):
  type `/` → popup visible with entries; type `/sk` → filtered; Down + Enter → draft becomes
  `/skills ` and popup shows skill args; Escape → popup hidden, draft unchanged; Enter with popup
  closed → normal send path unaffected.

## Files

- New: `tldw_chatbook/Widgets/Console/console_command_popup.py`
- New: `tldw_chatbook/Chat/console_command_suggestions.py`
- Modified: `tldw_chatbook/UI/Screens/chat_screen.py` (mount, mutation hook, key routing)
- New tests: `Tests/Chat/test_console_command_suggestions.py`, pilot tests near existing
  console composer tests.
