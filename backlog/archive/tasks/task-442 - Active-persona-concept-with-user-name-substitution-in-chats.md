---
id: TASK-442
title: Active persona concept with user-name substitution in chats
status: Done
assignee:
  - '@claude'
created_date: '2026-07-21 09:38'
updated_date: '2026-07-24 18:53'
labels:
  - roleplay
  - ux
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed from the RP/character-card UX review (Docs/superpowers/qa/rp-ux-review-2026-07-21/report.md). Personas (who you are) are currently one-shot staged text for Console; there is no default/active persona, no import, and the preview always renders the user as "you"/"User" (placeholders replace {{user}} with the literal "User"). An RP user expects to pick a persona once and have their name/description flow into greetings, placeholder substitution, and sends.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A persona can be marked active/default and persists across sessions
- [x] #2 Preview and character sends substitute the active persona's name for {{user}} and label the user's messages with it
- [x] #3 With no active persona, current behavior is unchanged
<!-- AC:END -->

## Reconciliation note (rebase, 2026-07-24)

A concurrent session DROPPED this task as mis-specified, correctly observing
that the ORIGINAL AC wording ("substitute the active persona's name for
{{user}}") inverted the macro semantics as written. The app author then
personally corrected the semantics in-session and directed the corrected
implementation — which matches the drop-note's own prescription exactly:
{{character}}/{{persona}} added as CHARACTER-side aliases, and {{user}} fed
the USER's real name (from the active user profile) instead of the literal
"User". This branch delivers that corrected version, plus the
persona->user-profile rename the author additionally directed. The drop-note
is preserved in git history; this record supersedes it.

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Design doc: `Docs/superpowers/specs/2026-07-24-roleplay-active-user-profile-design.md`
(brainstormed + 2 review passes). Executed as 5 sequential sub-tasks on branch
`claude/roleplay-active-persona`, each with its own RED→GREEN TDD cycle and
verification gate:

1. **T1 — placeholder aliases + active-profile resolver.** Add `{{character}}`/
   `{{persona}}` as character-side aliases to `replace_placeholders`
   (`Character_Chat_Lib.py`). Add `tldw_chatbook/Character_Chat/active_user_profile.py`:
   config-backed pointer (get/set/clear) + a never-raising
   `resolve_active_user_profile_name(service) -> str | None` resolver.
2. **T2 — the "persona" → "user profile" rename** (7 axes, user-facing AND
   internal, this same cycle per explicit user direction). See the rename map
   below. Byte-compat: accept-old-write-new on every persisted surface.
3. **T3 — "Set as my name" marking UX.** Inspector action + library-row
   indicator + delete-active-clears-pointer, wired through T1's pointer API
   only (no raw config writes).
4. **T4 — substitution at the three send/preview sites,** using the config
   pointer resolved through the app's already-constructed
   `local_character_persona_service` (no new service instances).
5. **T5 (this pass) — byte-compat pins + bookkeeping.** Audit T1/T3/T4's
   existing tests against the plan's byte-compat checklist, add tests only for
   genuinely uncovered items, fix a stale docstring, close out this task file.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
### Placeholder table (verbatim, `Character_Chat_Lib.replace_placeholders`)

| Token | Substitutes | Source |
|---|---|---|
| `{{user}}`, `{{random_user}}`, `<USER>` | **The USER's name** — the name the user sets for themself in chat | The active user profile's name; fallback `"User"` when none is set |
| `{{char}}`, `{{character}}`, `{{persona}}`, `<CHAR>` | **The AI character's name** — who the user chats WITH | The character card's name |

`{{character}}`/`{{persona}}` are T1's new character-side aliases — they
resolve to the character's name regardless of the active-profile pointer (a
deliberate, user-requested behavior change for texts containing those tokens,
previously rendered literally; independent of AC3's byte-compat, which covers
only the user-side tokens and token-free text).

### Active pointer + "Set as my name" UX + rename-follows-pointer

The active profile is a **single config value**: `[character_defaults]
active_user_profile = "<name>"`, written only through
`active_user_profile.py`'s `set_active_user_profile` / `clear_active_user_profile`
/ `get_active_user_profile_pointer` (never a raw config write). This is
deliberately a different concept from the P3b per-record `is_active`/`enabled`
Switch (a filtering flag, untouched).

T3 added the marking UX: an inspector action ("Set as my name" / "Clear")
gated to `user_profile` selections (slots into the task-443 kind-aware action
matrix), an active-row indicator (`●` prefix + `.is-active-profile` class) in
the library list, and an always-visible "Chatting as: X" inspector summary.
Deleting the active profile clears the pointer before the delete-refresh runs
(no dangling-by-design). A follow-up fix in the same sub-task made **renames
follow the pointer**: renaming the currently-active profile updates the
pointer to the new name instead of going dangling.

A dangling pointer (profile deleted/renamed outside this flow, or corrupted)
resolves as `None` — treated identically to "no active profile," never an
error and never a blocked send.

### The three substitution sites + service-instance wiring

`resolve_active_user_profile_name` calls `service.list_user_profiles()`
**synchronously**, which rules out `app.character_persona_scope_service` (its
`list_user_profiles` is `async`). Every site reaches the resolver through
`getattr(<app handle>, "local_character_persona_service", None)` — the same
object the app constructs once in `app.py::_wire_character_persona_services`
(traced: same Python instance the workbench's local backend uses, zero
split-brain). `getattr(..., None)` plus the resolver's never-raise contract
means a missing/broken service silently reads as no-active.

1. **Preview seeding** (`UI/Persona_Modules/personas_preview_controller.py`,
   `_load_greetings`): `replace_placeholders(g, name, user_name or "User")` for
   every greeting (primary + alternates + reload/restore). `pane.set_speakers(user=user_name)`
   is called **only** when a profile is active — no active profile leaves the
   pane's default "you" label untouched.
2. **Start-Chat handoff greeting** (`UI/Screens/chat_screen.py::_start_character_console_session`):
   `replace_placeholders(first_message, name, active_user_name or "User")`;
   `ConsoleSessionSettings.user_profile_label` is set from the active name
   **only** when a profile is active, otherwise the dataclass default
   `"General"` is untouched.
3. **Enhanced-chat display path** (`Event_Handlers/Chat_Events/chat_events.py::display_conversation_in_chat_tab_ui`):
   `resolve_active_user_profile_name(...) or app.app_config.get("USERS_NAME", "User")`
   — the fallback preserves this site's **pre-existing** `USERS_NAME`
   config-based semantics byte-exactly (this site's fallback was never the bare
   `"User"` literal, unlike sites 1/2).

Console native transcript user rows have no nameable speaker slot
(`console_transcript.py::_message_role_label` derives the label from the role
enum only) — per the design's explicit scope note, nothing was added there;
the label surfaces stay the preview pane lines and the session's
`user_profile_label` chip ("As: {label}").

### The rename ("persona" never means the user again) — 7 axes

| # | Axis | Old → New |
|---|---|---|
| 1 | Mode id (state.active_mode + ~40 literals) | `"personas"` → `"user_profiles"` |
| 2 | User-facing labels (mode chip, descriptors, Settings category description, inspector "Type:", status lines) | "Persona(s)" → "User Profile(s)" |
| 3 | Selection kind (~20 literals) | `"persona_profile"` → `"user_profile"` |
| 4 | DTOs (app-side imports) | `PersonaProfileCreate/Update/Response` → `UserProfileCreate/Update/Response` (re-export aliases) |
| 5 | Messages + widget file | `EditPersonaRequested`/`PersonaProfileSaveRequested`/`PersonaProfileEditorWidget` → `EditUserProfileRequested`/`UserProfileSaveRequested`/`UserProfileEditorWidget` (file renamed) |
| 6 | `persona_label` field (2 dataclasses, ~23 refs) | `user_profile_label`; display copy `"Persona: {label}"` → `"As: {label}"` |
| 7 | Service internals (`local_character_persona_service.py`, `CharacterPersonaScopeService`) | `persona_store_path`/`list_persona_profiles`/`_persona_profile_view`/etc. → `user_profile_*` equivalents |

**The `tldw_api` wire-mirror boundary:** `tldw_api/character_persona_schemas.py`
DEFINES the DTOs and mirrors the server's actual REST contract, which calls
them personas — this module (and `tldw_api/client.py`'s method names) is the
**one sanctioned internal "persona" remnant** for the user-profile concept;
every app-side importer switches to the new re-export aliases instead.

**Compat shims (accept-old, write-new):**
- Saved screen state: `restore_state` only ever restores `active_mode ==
  "characters"`, so an old `"personas"`-mode blob was already discarded before
  this rename — no shim needed, pinned by a compat-free-by-construction test.
- `persona_label` inside serialized `ConsoleSessionSettings`
  (`chat_screen.py::_restore_console_settings`): reads accept the old key and
  map it onto `user_profile_label` before the existing unknown-key-drop filter
  runs; writers emit only the new key. Default `"General"` unchanged.
- On-disk profile JSON / store path: unchanged (only Python identifiers moved,
  not the JSON field contract or the file path value).

**B2 boundary (kept, explicitly out of scope):** the *workbench* family —
`PersonasScreen`/`personas_screen.py`/`Persona_Widgets/`/`PersonasPreviewPane`/
`personas_pane_messages.py`, the `#personas-*` DOM ids, and the
`personas_workbench`/`personas_preview` saved-state dict keys — names the
**product/route** ("Roleplay & Chat Dictionaries" workbench), not the user;
renaming it is a separate, much larger axis with zero user-reference semantics
and was deliberately not pulled into this cycle. Settings category **title**
(not description) was likewise adjudicated during T2 review to become
"Roleplay" (not "User Profiles") since the category demonstrably covers both
Characters and user-profile content.

### Deferred residuals (explicitly out of scope for this task)

- **Active→cleared pane-label staleness:** if the active profile pointer
  changes mid-conversation, the preview pane only relabels already-rendered
  character lines retroactively (task-437 precedent); already-rendered user
  lines keep whatever label was current when they were appended, and clearing
  the pointer does not retroactively relabel back to "you" until the next
  reseed. Same class of behavior the pane already had for character renames.
- **Profile description/traits flowing into the system prompt** (model-visible
  prompt engineering) — out of scope per the design doc; own future follow-up.
- **`ccp_persona_handler.py` internal identifiers:** its user-facing
  `_notify`/error strings were de-persona'd during T2's review fix wave, but
  its internal method names, view-name strings, and its own extensive
  "persona" naming (e.g. `refresh_persona_list`, `load_persona`,
  `PersonaMessage`) were left untouched — out of the T2 rename map's named
  surface, a much larger separate refactor.
<!-- SECTION:NOTES:END -->

### Grep-gate survivor enumeration (whole-branch review, Finding 2)

Remaining "persona" tokens in `tldw_chatbook/`, each justified:
- The `persona_label` → `user_profile_label` compat shim itself (accept-old-write-new; by design).
- `server_character_persona_service.py` / `tldw_api/client.py` wire methods + the
  `tldw_api/character_persona_schemas.py` class names — the sanctioned server-API
  wire-mirror boundary (app layer uses the `UserProfile*` aliases).
- The pre-existing `assistant_kind`/`persona_memory_mode`/`ccp_persona`
  conversation-ownership subsystem (Chat_Functions, ChaChaNotes_DB,
  chat_conversation/persistence services, chat models/tabs) — a DIFFERENT,
  older "persona" concept classifying conversation assistants; zero hunks in
  this branch; never in the rename map.
- Kept-by-design internal ids: `#console-persona-chip`,
  `console-settings-persona-readonly`, `WorkbenchMode(id="persona", ...)`,
  DOM ids `#personas-*`, state-dict keys `personas_workbench`/`personas_preview`.
- `ccp_persona_handler.py` internal identifiers (user-facing strings de-personaed;
  internals are the documented residual).
