# Personas Workbench Design (Characters + Personas)

Date: 2026-06-09
Status: Approved design, pre-implementation
Route: `personas` (legacy `ccp`, `characters`, `prompts` routes resolve here at completion)

## Purpose

Rebuild the Character/Persona creation, viewing, editing, and management surface as a
destination-native workbench on the `personas` route, following the agentic-terminal
design system with Console as the style anchor. This replaces the current two-hop
arrangement (thin snapshot shell on `personas` -> "Open Personas" -> half-converted
legacy `ccp` workbench).

## Context: state of the redesign

- The design system is enforced by contract: `Docs/Design/agentic-terminal-visual-system.md`
  and `Docs/Design/master-shell-design-system-contract.md` define `$ds-*` tokens, shared
  classes (`.ds-destination-header`, `.ds-panel`, `.ds-inspector`, `.ds-field-row`,
  `.ds-toolbar`, `.ds-recovery-callout`, `.ds-status-badge`), state classes, readable
  status labels, and the layout grammar (header -> mode strip -> 3-pane workbench ->
  footer shortcut bar). New visual patterns require a contract update first.
- Console (`chat_screen.py`) is the hardened anchor; Library has its UX fixes; Notes was
  rebuilt on the destination workbench pattern (`notes_screen.py` +
  `Note_Widgets/notes_workbench_panes.py`) and is the freshest reference implementation.
- Characters/Personas lags: `personas_screen.py` is a thin attach-to-Console shell;
  `ccp_screen.py` (~1,800 lines) has a first-pass workbench shell with only Characters
  mode wired; Personas/Prompts/Dictionaries/Lore modes are placeholders. A stray
  `ccp_screen.py.bak` remains.
- `Docs/Design/New_UI/Characters.png` and `Personas.png` are layout references, not pixel
  targets.

## Scope decisions (approved)

- **Modes in scope:** Characters and Personas (full create/view/edit/manage). Prompts,
  Dictionaries, and Lore remain visible placeholder modes on the strip (the contract
  forbids hiding them); they are follow-up work.
- **Routing:** one workbench owns the `personas` destination. The legacy `ccp` route (and
  `characters`/`prompts` legacy routes) resolve to it. The thin snapshot shell and the
  two-hop navigation go away.
- **Approach:** new destination-native shell with maximum reuse of the working behavior
  layer (handlers, character editor/card widgets, import/validation libs), restyled to
  the design system. Stub widgets and sidebar-era chrome are rebuilt or retired.
- **Data scope:** local-only. Source-authority labels render per the design system so
  server support slots in later without layout change.
- **Must-have workflows:** import/export cards, attach/start chat in Console, preview
  conversation, library search/filter, and viewing saved conversations associated with
  the selected character.

## Layout

```
+------------------------------------------------------------------------------------+
| Top nav (unchanged): Home Console Library ... Personas ...                          |
+------------------------------------------------------------------------------------+
| Personas | Behavior profiles for chat & agents | Ready | Local      [+ New]        |  <- .ds-destination-header
+------------------------------------------------------------------------------------+
| Modes: [Characters*] [Personas] [Prompts] [Dictionaries] [Lore]                     |  <- DestinationModeStrip
+--------------------+--------------------------------------------+------------------+
| LIBRARY            | WORK AREA                                  | INSPECTOR        |
| [search________]   | View mode: character card (read-only)      | Selected target  |
| New | Import       |   -- or --                                 | Validation panel |
| > Detective Sam    | Edit/Create mode: ds-field-row form        | Conversations    |
|   Lab Assistant    |   [Save] [Cancel]  (.is-unsaved tracking)  | Readiness        |
|   Tutor (Unsaved)  |   -- or --                                 | [Attach Console] |
|                    | Saved conversation (read-only view)        | [Start Chat]     |
| 12 characters      +--------------------------------------------+ [Export]         |
|                    | PREVIEW CONVERSATION (collapsible)         | [Delete]         |
|                    | [Test Reply] [Reset] [Open in Console]     |                  |
+--------------------+--------------------------------------------+------------------+
| Ctrl+N new | Ctrl+F search | Ctrl+S save | Ctrl+Enter attach | <ShortcutContext>   |  <- footer
+------------------------------------------------------------------------------------+
```

- **Library pane (left):** mode-scoped list with live search/filter, `.ds-toolbar` with
  New and Import (Import in Characters mode only), row-level status badges
  (`.is-active`, `.is-unsaved`), count line. Selecting a row loads view mode.
- **Work area (center):** view/edit stack. Read-only card view by default; Edit/New swaps
  in a `.ds-field-row` form with explicit Save/Cancel (no autosave - these are behavior
  definitions). The same region renders a selected saved conversation read-only. The
  preview conversation docks at the bottom, collapsible, ephemeral.
- **Inspector (right):** `.ds-inspector` with selected-item identity, validation summary,
  conversations panel, attachment readiness (readable labels), and the action group:
  Attach to Console, Start Chat, Export, Delete (destructive styling + confirm).
  Recovery callouts render here.
- **Personas mode** is the identical skeleton with persona-profile fields (name, system
  prompt, style/policy fields per the scope service schema); no image or card
  import/export (those are Characters-mode concepts).

## Component architecture

New files (small, focused, Notes-pattern):

```
tldw_chatbook/
  UI/Screens/personas_screen.py          <- rebuilt as the workbench screen (shell,
                                            mode switching, message routing only)
  Widgets/Persona_Widgets/
    __init__.py
    personas_library_pane.py             <- search input + toolbar + mode-scoped list
    personas_inspector_pane.py           <- identity, validation, conversations,
                                            readiness, actions
    personas_preview_pane.py             <- ephemeral preview conversation
    persona_profile_card_widget.py       <- read-only persona view (replaces stub)
    persona_profile_editor_widget.py     <- ds-field-row persona form (replaces stub)
```

Reused in place (import path unchanged, restyled to `ds-*` classes):

- `Widgets/CCP_Widgets/ccp_character_card_widget.py` (~555 lines)
- `Widgets/CCP_Widgets/ccp_character_editor_widget.py` (~866 lines)
- `Widgets/CCP_Widgets/ccp_conversation_view_widget.py` (~471 lines, read-only saved
  conversation view)
- `UI/CCP_Modules/ccp_character_handler.py`, `ccp_persona_handler.py`,
  `ccp_validators.py`, `ccp_messages.py`
- `Character_Chat/Character_Chat_Lib.py` (import/validation/parsers, conversation
  listing), `DB/ChaChaNotes_DB.py`, `Character_Chat/character_persona_scope_service.py`
- Existing `ChatHandoffPayload` attach/start-chat wiring

Handlers keep their parent-screen reference pattern; where they query legacy widget IDs
(e.g. `#ccp-character-list`), the new panes adopt those stable IDs rather than rewriting
the handlers. Verified exception: `ccp_character_handler` queries the sidebar-era
`#conv-char-character-select` Select in two refresh paths; that widget does not exist in
the new layout, so those two call sites get guarded or removed. The handlers have no
coupling to `CCPScreenState`, so the slim state replacement is safe.

Retired at the end: `ccp_screen.py` (+ `.bak`), `ccp_sidebar_widget.py`,
`ccp_sidebar_handler.py`, the thin snapshot shell content of the old `personas_screen.py`,
and their dead tests.

## State and data flow

A slim `PersonasWorkbenchState` dataclass on the screen replaces `CCPScreenState`:
`active_mode`, `selected_kind`, `selected_id`, `edit_mode` (view/edit/create),
`is_unsaved`, `search_query`. The selection/search/preview message classes
(`CharacterSelected`, `PersonaSelected`, `PreviewReplyRequested`, `LibrarySearchChanged`,
...) are defined in the new `Persona_Widgets` package - NOT imported from
`ccp_screen.py`, where the current equivalents live, because that file is retired in
migration step 4. The richer `CCP_Modules/ccp_messages.py` hierarchy stays for
handler-level events.

```
pane widget --post_message()--> screen @on() handler --> handler module
     ^                                                       | run_worker / async
     |                                                       v
  reactive update / targeted refresh <-- call_from_thread -- ScopeService / Chat_Lib / DB
```

- All DB/service calls run in workers (>100ms rule); list refreshes `exclusive=True`.
- Optimistic-locking conflicts from `update_character_card` surface as a
  `.ds-recovery-callout` in the inspector ("Reload latest / overwrite"), never a silent
  failure.
- Preview flow: the preview pane builds an ephemeral message list (greeting + user
  turns), calls the same provider gateway Console uses, and streams the reply into the
  pane. Verified: `ConsoleProviderGateway` is dependency-injected (http client, config
  provider, `chat_api_call` fn) with no Console-screen coupling, so the preview pane
  constructs its own instance from app config. Nothing persists. "Open in Console"
  converts the preview into a `ChatHandoffPayload`.
- Routing: `shell_destinations.py` canonical overrides flip so `ccp`, `characters`, and
  `prompts` resolve to `personas`. Command-palette entries keep working.
- Footer: the screen registers a `ShortcutContext` on mount (New / Search / Save /
  Attach) and clears it on unmount per the contract. The key choices shown in the layout
  diagram are illustrative; final bindings are resolved against existing global bindings
  at implementation time.

## Workflows

**Create / Edit.** "+ New" opens the editor in create mode; selecting a library row shows
the read-only card with an Edit button. Any field change sets `.is-unsaved` on the row,
header, and inspector. Save runs validation (`ccp_validators` + `validate_v2_card` for
characters); failures render field-level messages in the inspector validation panel and
block Save. Cancel, row switches, and mode switches with unsaved edits prompt a confirm
guard.

**Import (Characters mode only).** File picker -> `import_and_save_character_from_file`
(PNG-embedded JSON and raw JSON; v1/v2/ccv3). Success refreshes the library and selects
the new character in view mode. Parse/validation failure shows a recovery callout with
the parser's reason. Duplicate names follow existing lib behavior and report the outcome.

**Export.** Inspector action, enabled when a saved character is selected. Offers both
JSON and PNG card formats via the existing `export_character_card_to_json` /
`export_character_card_to_png` functions and a save-file dialog (PNG is the de-facto
card interchange format). Personas export as plain profile JSON. Unsaved edits disable
Export with a "Save before exporting" tooltip.

**Preview conversation.** Collapsed by default. Expanding seeds the greeting
(placeholders replaced via `replace_placeholders`). Test Reply streams a response using
the current default provider/model from config; provider problems show a readable status
("Provider unavailable - configure in Settings"), never a traceback. Reset clears to the
greeting. Preview reads the in-editor draft, so unsaved edits affect the next Test Reply
without saving.

**Saved conversations.** When a character is selected, the inspector's Conversations
panel lists recent saved conversations (title + last-updated, via
`list_character_conversations`) with a count. Selecting one opens it read-only in the
center work area (reusing `CCPConversationViewWidget`), with two actions: Continue in
Console (handoff with character + conversation context) and Open in Library (deep-link;
Library remains the owner of full conversation browsing/management). In Personas mode
the panel renders only if profile-linked conversations exist in the data; otherwise it
is hidden.

**Attach to Console / Start Chat.** Same `ChatHandoffPayload` semantics as today: Attach
stages the selected card as behavior context; Start Chat navigates to Console with the
character active. Both disabled with reason-tooltips when nothing is selected or the
selection has unsaved edits.

**Search/filter.** Library search filters as you type - local filtering over the loaded
list first, falling back to `search_characters` / FTS when the loaded list was truncated
by the page limit. Count line shows "n of m" while filtered. Esc clears.

**Delete.** Inspector action, destructive styling, confirm dialog naming the item. Soft
delete via existing DB methods; library refreshes and selection moves to the next row.

## Empty, loading, and error states

Each pane owns its states per the Library UX-fix pattern:

- Library list: skeleton "Loading..." then rows or an actionable empty state
  ("No characters yet - [New] [Import]").
- Service unavailable / policy denied: `.ds-recovery-callout` with owner/problem/next
  action, reusing `DestinationRecoveryState` / `policy_denied_recovery_state`.
- The snapshot-timeout guard from the current screen carries over so the route never
  hangs in a loading state.
- All statuses are readable labels ("Ready", "Blocked", "Unsaved"); no color-only
  signaling.

## Testing

Mounted Textual tests asserting stable IDs/classes and readable status text (never raw
colors), using real in-memory SQLite:

- Route renders header, mode strip, and three panes; mode switch swaps library + work
  area.
- Select -> view -> edit -> unsaved guard; validation blocks save.
- Import success and failure paths.
- Conversations panel lists and opens a saved conversation; Continue in Console posts
  the right handoff payload.
- Attach/Start Chat post the right `ChatHandoffPayload`.
- Legacy `ccp` route resolves to the personas destination.
- Footer shortcut context is set on mount and cleared on navigation.

Existing `test_ccp_screen.py` assertions migrate or retire with the screen. Visual QA via
the established textual-web/CDP screenshot workflow, captured under `Docs/superpowers/qa/`.

## Migration sequence (each step shippable)

1. New workbench shell on the `personas` route with Characters mode wired
   (library/view/edit/inspector); legacy `ccp` route still available as fallback.
2. Personas mode with the new persona card/editor widgets.
3. Preview pane, conversations panel, import/export, search.
4. Flip route overrides (`ccp` -> `personas`); retire `ccp_screen.py` + `.bak`,
   `ccp_sidebar_widget`, `sidebar_handler`, the old thin-shell content, and dead tests.

## ADR

ADR required: yes
ADR path: backlog/decisions/ (number assigned at creation), "Personas destination-native
workbench and CCP route retirement"
Reason: long-lived UX/application structure - route consolidation, module retirement,
and cross-module interface changes. The ADR lands before implementation step 1 and is
linked from the Backlog task, implementation plan, and implementation notes.

## Out of scope

- Server-backed character/persona CRUD (authority labels only).
- Prompts, Dictionaries, and Lore modes (placeholders remain).
- Assignments tables from the New_UI mockups.
- Any new visual pattern not already in the design-system contract.
