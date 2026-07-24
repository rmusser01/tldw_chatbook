# TASK-435 — Resolve the Personas/Roleplay naming split → "RP&CD"

- **Date:** 2026-07-23
- **Task:** TASK-435 (RP/character-card UX review). One public name per meaning.
- **Branch base:** origin/dev (tip `0f5904e6b`).

## Problem

The destination has three overlapping names: the nav label says **"Personas"**, the screen header says **"Roleplay"**, and an in-screen **mode** is also **"Personas"** (user identity). A user hunting "character chat" must guess, and "Personas" means something different at each level.

## Resolution (user-chosen)

The destination's public name is **"RP&CD"** (abbreviation) / **"Roleplay & Chat Dictionaries"** (full). **"Personas"** is reserved for the in-screen user-identity mode.

- Cramped nav rail shows the abbreviation **"RP&CD"** (matching the other one-word tab labels).
- The full name **"Roleplay & Chat Dictionaries"** appears where there is room: the screen header, the command-palette command, and tooltips.
- These are the abbreviated and full forms of **one** name, so nav ⇔ header are mutually consistent (AC#1); "Personas" now means only the mode.

## The naming surfaces (all user-visible destination names)

1. `tldw_chatbook/UI/Navigation/shell_destinations.py` — the `personas` `ShellDestination` `label="Personas"` (nav rail via `destination.label`, `main_navigation.py:176`).
2. `tldw_chatbook/Constants.py` — `TAB_DISPLAY_LABELS[TAB_CCP]` and `[TAB_PERSONAS]` = `"Personas"`.
3. `tldw_chatbook/UI/Screens/personas_screen.py:584` — `DestinationHeader(title="Roleplay", ...)`.
4. `tldw_chatbook/app.py` — the tab tooltip map: `TAB_PERSONAS` (`:704`, "Open Personas for …") and `TAB_CCP` (`:712`, "Switch to Personas for …"). **Both** name the destination "Personas".
5. `tldw_chatbook/Widgets/Persona_Widgets/personas_state.py:25-30` — `MODE_LABELS["personas"] = "Personas"` (the mode — **unchanged**, its reserved meaning).

## Design

### 1. `shell_destinations.py` — the destination
```python
ShellDestination(
    "personas",                       # destination_id — internal, UNCHANGED
    "RP&CD",                          # label  (was "Personas")  → nav rail
    "personas",                       # primary_route — internal, UNCHANGED
    "Characters, personas, dictionaries, and behavior profiles.",   # purpose (unchanged)
    "Manage behavior profiles and persona context.",                # tooltip (unchanged)
    ("ccp", "conversations_characters_prompts", "characters", "roleplay"),  # legacy_routes += "roleplay"
    full_label="Roleplay & Chat Dictionaries",   # NEW → accessible_label / palette / display
),
```
- `accessible_label` (`= full_label or label`, `shell_destinations.py:21`) now returns "Roleplay & Chat Dictionaries" → the palette command reads "Open Roleplay & Chat Dictionaries for …".
- `legacy_routes` gains **only** `"roleplay"` so the new public name resolves as a route (`resolve_shell_route("roleplay")` → `personas`) and is a palette search term. `get_tab_display_label("roleplay")` safely returns the title-cased fallback `"Roleplay"` (`Constants.py:105-107`, `.get(tab_id, tab_id.replace("_"," ").title())` — no raise for a non-tab route).
- **"personas" needs no addition:** `_destination_alias_terms` (`app.py:796-808`) already seeds the term set with `destination.destination_id` **and** `destination.primary_route` — both `"personas"` — so `"personas"` remains both routable (primary route) and palette-searchable without touching `legacy_routes`. (This is what previously supplied the capitalized `"Personas"` term — the destination `label`; after the rename the lowercase `"personas"` term persists via id/primary_route while the label term becomes `"RP&CD"`.)

### 2. `Constants.py` — tab display labels
`TAB_DISPLAY_LABELS[TAB_CCP]` and `[TAB_PERSONAS]`: `"Personas"` → `"RP&CD"`.

### 3. `personas_screen.py:584` — header
`title="Roleplay"` → `title="Roleplay & Chat Dictionaries"`. (Header title is a single-line `height:1` Static, `_workbench.tcss:28`; a longer title clips gracefully on very narrow terminals — no wrap/layout break.)

### 4. `app.py` — tab tooltips (704, 712)
Rename the destination in both, and drop the retired "prompts" (Task 7) while editing:
- `TAB_PERSONAS`: `"Open Personas for characters, prompts, dictionaries, and behavior profiles"` → `"Open Roleplay & Chat Dictionaries for characters, personas, dictionaries, and behavior profiles"`.
- `TAB_CCP`: `"Switch to Personas for characters, personas, prompts, dictionaries, and world books"` → `"Switch to Roleplay & Chat Dictionaries for characters, personas, dictionaries, and world books"`.

### 5. Mode label — unchanged
`MODE_LABELS["personas"] = "Personas"` stays. This is now the single meaning of "Personas".

### Not changed (non-goals)
- `destination_id`/`primary_route` = `"personas"`, and all `TAB_*` id constants — internal ids, not display names.
- No file/class/module renames (`personas_screen.py`, `PersonasScreen`, `CCP_Modules/*` docstrings, etc.).
- The mode name and the `purpose`/`tooltip` content strings that legitimately describe contents (they mention lowercase "personas"/"dictionaries" — the modes/content, which is correct).

## Testing

Update the label/tooltip-asserting tests and add discoverability + rendering assertions:
- `Tests/UI/test_shell_destinations.py` (~:14): the `personas` destination `label` is `"RP&CD"`, `full_label`/`accessible_label` is `"Roleplay & Chat Dictionaries"`; `legacy_routes` includes `roleplay` (and the pre-existing ccp/conversations_characters_prompts/characters). `resolve_shell_route("personas")`, `("ccp")`, `("roleplay")`, `("characters")` all resolve to destination `personas` (AC#2 — `"personas"` still resolves via primary_route).
- `Tests/UI/test_command_palette_shell_routes.py` (~:95): the existing `assert "Personas" in alias_terms["personas"]` becomes `assert "RP&CD" in alias_terms["personas"]` and `assert "Roleplay & Chat Dictionaries" in alias_terms["personas"]`; keep `{"ccp","conversations_characters_prompts","characters"} <= alias_terms["personas"]` and add `roleplay`; assert `"personas"` itself is still a term (via destination_id/primary_route). Assert the `"&"` survives literally in the `"RP&CD"` term (renders, not mangled).
- `Tests/UI/test_command_palette_providers.py` (~:356): update the expected label from `"Personas"` to `"RP&CD"` (or the accessible label, whichever the assertion targets).
- `TAB_DISPLAY_LABELS`: assert `get_tab_display_label(TAB_CCP) == get_tab_display_label(TAB_PERSONAS) == "RP&CD"`.
- The `test_destination_visual_parity_correction.py:1104` `"Personas"` button is the **mode chip** (Characters/Personas/Dictionaries/Lore) — must stay `"Personas"`; confirm it is untouched.
- Live/pilot check: the nav rail button renders `"RP&CD"` literally (the `"&"` is not a Rich-markup metacharacter, so no escaping is needed — assert it appears).

`Tests`/`tests` are byte-identical dupes in this repo — edit whichever the build treats as source and keep them consistent (verify).

## Risks / mitigations

- **`"&"` rendering:** Rich markup only treats `[`/`]` specially, so `"&"` renders literally in the nav button, header, and palette; a rendering assertion pins it.
- **New `"roleplay"` legacy route:** registers `_ROUTE_MAP["roleplay"] → personas`; `get_tab_display_label("roleplay")` returns the safe title-cased fallback `"Roleplay"`. `"personas"` is left out of `legacy_routes` (already routes via primary_route + is already a search term via destination_id) — no redundant/odd registration.
- **Header clipping on very narrow terminals:** single-line `height:1` clip, no layout break; the abbreviation "RP&CD" already covers the cramped nav rail.
- **`Tests`/`tests` duplication:** update both / the build source; keep consistent.

## Non-goals

- Renaming the internal `destination_id`, route keys, `TAB_*` constants, files, classes, or the `CCP_Modules` package.
- Changing the mode name or the modes list.
- A broader copy pass on the screen (that is TASK-444's remit).
