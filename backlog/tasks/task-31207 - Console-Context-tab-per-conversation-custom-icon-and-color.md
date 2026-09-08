---
id: TASK-31207
title: 'Console Context tab: per-conversation custom icon and color'
status: Done
assignee:
  - '@zcode'
created_date: '2026-09-03 12:00'
labels:
  - console
  - ui
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Conversations in the Console Context tab's Conversations section are hard to
differentiate at a glance. Cursor-style customization: each conversation gets a
small icon control at the far left of the row (left of the conversation name,
star stays on the right); clicking it opens a picker with an emoji grid (search +
recents, reusing the dormant `Widgets/emoji_picker.py`) and a fixed color
palette. The icon renders colored via Rich markup; the color tints the icon
only, leaving selected/broken CSS states untouched. Unset conversations show a
dim placeholder glyph and are otherwise unchanged.

Storage rides the existing synced `conversations.metadata` JSON column under a
new namespaced key `console_appearance: {icon, color}` — the exact pattern
`console_speech`/`console_roleplay_context` already use — so it follows the
conversation across sync with no schema migration.

ADR required: no. Uses the established `conversations.metadata` namespaced-key
storage pattern (precedents: `console_speech`, `console_roleplay_context`); no
schema, sync, interface, or security boundary changes.

Follow-ups filed separately: TASK-31208 (icon in the Ctrl+K switcher) and
TASK-31209 (expanded color options).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Clicking the icon control on a conversation row opens the appearance picker modal (emoji search grid + color swatches + Apply/Clear/Cancel); Apply persists and refreshes the browser, Clear resets to unset
- [x] #2 Set icon renders colored to the LEFT of the conversation name in the Context tab Conversations section (icon control is the leftmost element of the row; star remains at the right); unset rows show a dim placeholder glyph and are otherwise unchanged
- [x] #3 Icon+color persist in `conversations.metadata` under the `console_appearance` key through the normal optimistic-locked update path and survive app restart / browser re-sync (round-trip evidence)
- [x] #4 Row geometry preserved: with the icon control mounted, the title button and star button regions stay fully on-screen (region containment assertions, not just display/text — per lessons-testing-evidence)
- [x] #5 ASCII mode renders a fallback glyph via the existing glyph-fallback map; malformed stored values (bad JSON, multi-grapheme icon, non-hex color) are sanitized per member at read time without crashing the tray
- [x] #6 Icon/color participates in the tray's structural recompose key so an applied change repaints without a full reload; existing browser-state/tray/workspace suites stay green (targeted runs)
<!-- AC:END -->

## Implementation Plan
<!-- SECTION:PLAN:BEGIN -->
1. Study exact code paths: `Widgets/Console/console_workspace_context.py`
   (row composition, `_marker_prefixed_name_lines` budget,
   `_conversation_browser_rows_height`, structural recompose key),
   `Workspaces/conversation_browser_state.py`, `UI/Console_Modules/workspace.py`
   row builders + star applier, `Chat/chat_persistence_service.py`
   (`merge_console_speech_preferences` pattern),
   `Widgets/emoji_picker.py` internals, a Console picker modal template
   (`console_reaction_picker_modal.py`), `ChatScreen.on_button_pressed` routing.
2. New `Chat/console_appearance.py`: metadata key constant, color palette
   (fixed hex values), validation (single grapheme, cell width <= 2, palette
   color), and a sanitize function for reads.
3. Persistence: read-merge-write `set_console_conversation_appearance` in
   `Chat/chat_persistence_service.py` mirroring the speech-preferences merge
   (optimistic-locked `update_conversation`). Tests first: round-trip + merge
   preserves sibling keys + sanitization cases.
4. Browser state: add `icon`/`color` to `ConsoleConversationBrowserInputRow`
   and `ConsoleConversationBrowserRow` (default None); parse sanitized
   appearance from `metadata` in the three row builders in `workspace.py`
   (native rows read appearance for their conversation ids). Tests first:
   passthrough + defaults.
5. Rendering: icon `Button` (id `console-conversation-icon-{index}`) as the
   leftmost child of the row `Horizontal`, explicit fixed width via CSS, dim
   placeholder when unset, colored Rich-markup label when set; register ASCII
   fallback glyph; verify row-height math unchanged and title budget accounts
   for the control; include icon/color in the tray structural recompose key.
6. Picker modal `Widgets/Console/console_appearance_picker_modal.py` modeled on
   the reaction picker: embedded emoji grid + search + recents (reuse
   `emoji_picker.py` internals), palette swatch row, local selection state,
   Apply/Clear/Cancel; posts an appearance-selected message carrying
   conversation id + icon + color.
7. Wiring: handle `console-conversation-icon-*` in `ChatScreen.on_button_pressed`
   (star-handler pattern), push modal, apply via persistence service off the UI
   thread, then trigger browser re-sync.
8. Tests: state passthrough, persistence round-trip, render evidence via
   `export_screenshot()` (icon visible left of title), neighbor geometry
   containment (title + star regions on-screen), ASCII fallback, sanitization;
   run targeted suites for touched modules.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- **Approach.** Cursor-style per-conversation icon+color for the Console
  Context tab. Storage is the synced `conversations.metadata` JSON under
  `console_appearance` (no migration, no ADR — established namespaced-key
  pattern). The icon control is a third fixed-width button at the far left
  of each row (left of the name; star stays right), rendered through Rich
  markup; the color tints the icon only, so selected/broken CSS states are
  untouched.
- **Data flow.** One batched read per row derivation:
  `db.get_conversations_metadata_by_ids` -> `ChatConversationService.
  get_conversation_appearances` -> applied in `_merge_console_browser_rows`
  (the single choke point native/membership/persisted rows all cross, so
  merged native rows that supersede the persisted row still get decorated).
  Writes go `set_conversation_appearance` (bounded optimistic-lock retry —
  streaming replies bump the version constantly) from a worker, mirroring
  the star-toggle discipline (task-15471), then invalidate the persisted
  rows TTL cache and re-sync the tray.
- **Validation/sanitization** (`Chat/console_appearance.py`): icon must be
  one grapheme (variation selectors allowed) of cell width <= 2; color is
  `#rrggbb` format (forward-compatible with a future larger palette, per
  TASK-31209). Reads sanitize per member; a payload whose key set is not
  exactly {icon, color} drops whole (speech-style version-skew guard).
- **ASCII mode** degrades a set icon to a colored `*` and the unset
  placeholder to a dim `+` (glyph vocabulary in `Chat/console_glyphs.py`;
  `▢ -> +` added to `ASCII_GLYPH_FALLBACKS`) — arbitrary user emoji cannot
  map meaningfully to ASCII.
- **Recompose guard**: icon/color are fields on the frozen browser row
  dataclasses, so the tray's state-equality skip (task-15454) repaints on
  appearance changes with no signature changes; pinned by an equality test.
- **Picker**: `Widgets/Console/console_appearance_picker_modal.py` reuses
  the dormant `emoji_picker.py` grid + recents store; result-driven dismiss
  (Apply -> appearance, Clear -> all-unset appearance, Cancel -> None),
  pushed with `push_screen(..., callback=...)` like the workspace rename
  modal. Row chrome width 6 -> 11 (`_BROWSER_ROW_CHROME_WIDTH`); CSS
  bundle regenerated via `build_css.py`.
- **Tests** (new/extended): `Tests/Chat/test_console_appearance.py` (31),
  persistence round-trip + retry class in `test_chat_persistence_service.py`
  (8), browser-state passthrough/equality (3), 5 rail tests (render,
  geometry containment + left-of-name ordering, ASCII, disabled native
  rows, routing), 6 picker tests. Targeted sweep: 833 passed; the only
  failures were 12 pre-existing `test_console_native_chat_flow.py` breaks
  from in-flight `_retrieval`/wiring changes already in the working tree
  (unrelated to this task; `wiring.py` is mid-refactor in the dirty tree).
- **Modified files**: `Chat/console_appearance.py` (new),
  `Chat/console_glyphs.py`, `Chat/chat_conversation_service.py`,
  `DB/ChaChaNotes_DB.py` (batched metadata read), `Workspaces/
  conversation_browser_state.py`, `UI/Console_Modules/workspace.py`,
  `Widgets/Console/console_workspace_context.py`,
  `Widgets/Console/console_appearance_picker_modal.py` (new),
  `Widgets/glyph_fallback.py`, `UI/Screens/chat_screen.py` (routing branch),
  `css/components/_agentic_terminal.tcss` + regenerated bundle.
- **Lesson recorded**: collapsed-left-rail zero-region trap in
  `lessons-testing-evidence.md` (rail must be opened explicitly before
  region assertions; standalone probes lie via the developer's real config).
<!-- SECTION:NOTES:END -->
