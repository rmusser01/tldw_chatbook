# Roleplay task-442 — Active User Profile + Name Substitution (and the "persona" → "user profile" rename) — Design

**Date:** 2026-07-24
**Program:** RP-UX follow-ups sweep, Batch 4 (B1 #815 / B2 #824 / B3 #830 merged).
**Status:** design approved (brainstorming); ready for implementation plan.

## Goal

Let the user pick **who they are** once — an **active user profile** — and have their name flow into greetings and chats everywhere `{{user}}` appears, with their messages labeled by it. And fix the naming that caused confusion in the first place: **"persona" never refers to the user in this app again** — the user-side concept is renamed **User Profiles**, end-to-end (user-facing labels AND internal identifiers), with full persistence compatibility.

## Decisions (settled in brainstorming — binding)

1. **Placeholder semantics (the user's explicit correction — get this right):**

   | Token | Substitutes | Source |
   |---|---|---|
   | `{{user}}`, `{{random_user}}`, `<USER>` | **The USER's name** — the name the user sets for themself in chat | The active user profile's name; fallback `"User"` when none is set |
   | `{{char}}`, `{{character}}`, `{{persona}}`, `<CHAR>` | **The AI character's name** — who the user chats WITH | The character card's name |

   `{{character}}` and `{{persona}}` are NEW aliases added to `replace_placeholders` (`Character_Chat_Lib.py:404`), joining the existing `{{char}}`/`<CHAR>`. The user-side tokens NEVER receive the character's name. Texts containing none of the new tokens are byte-identical.

2. **The active pointer lives in config** (single value ⇒ single-active by construction; persists across sessions natively): `[character_defaults] active_user_profile = "<profile name>"`, written via the codebase's existing config-save seam (the `console.provider_defaults` precedent). Marked via an inspector action on user-profile selections; NOT via the P3b `is_active`/`enabled` Switch (that is a per-record ENABLED flag with filtering semantics — a different concept, untouched).

3. **The rename ships in this same cycle, user-facing AND internal** (user decision: "do not leave internal naming as another task"). Boundary below.

## Part A — the feature

### A1. Resolver

`resolve_active_user_profile_name(...) -> str | None` — reads the config pointer, loads the profile via the (renamed) user-profile service, returns its name. Unset pointer, or a **dangling** pointer (profile deleted/renamed) ⇒ `None` (treated as no-active, never an error; a notify-level hint at most). Cheap: config + one file-backed profile read, resolved at the call sites' natural cadence (selection sync / send start), never per-tick.

### A2. Marking UX

- Inspector action for `user_profile` selections (slots into task-443's kind-aware matrix): **"Set as my name"** on an inactive profile / **"Clear"** on the active one.
- The active profile is indicated in its library list row (e.g. a marker on the row) and in the inspector summary ("Chatting as: Sam").
- Deleting the active profile clears the pointer (write-through) — no dangling-by-design.

### A3. Substitution sites (AC2)

The three call sites that hardcode `"User"` receive the resolved name (fallback `"User"` — AC3 byte-compat):
1. Preview greeting seeding — `personas_preview_controller.py:113` (`replace_placeholders(g, name, "User")`).
2. **Start-Chat handoff greeting** — `chat_screen.py:10271` (the Console session's opening message).
3. Enhanced-chat character path — `chat_events.py:4374` (+ whatever `user_name` params its `Character_Chat_Lib` callees already take — thread the same value; do not fork the logic).

### A4. Labeling (AC2b)

- Preview: `set_speakers(user=<name>)` (the existing 438-era seam) when an active profile exists — the user's transcript lines carry their name.
- Console: Start-Chat sets the session's `user_profile_label` (the renamed `persona_label` — an existing display string) from the active profile's name. Whether Console transcript user-rows have a nameable speaker slot is verified at plan time; if they don't, we do NOT invent one this cycle (the label surfaces are the preview lines + the session's label chip).

### A5. Byte-compat default (AC3)

No active profile ⇒ resolver `None` ⇒ `"User"` fallbacks ⇒ `set_speakers` untouched ⇒ label untouched ⇒ every current output byte-identical. Pinned by tests.

## Part B — the rename ("persona" never means the user)

### B1. Rename map (user-profile concept — IN scope)

| Old | New |
|---|---|
| Mode id `"personas"` (state.active_mode value + ~40 literals) | `"user_profiles"` |
| User-facing labels: "Personas" mode chip, `_MODE_DESCRIPTORS` entry, Settings category, inspector "Type:" copy, status/footer lines | **"User Profiles"** (copy style per the P0/444 plain-language patterns) |
| Selection kind `"persona_profile"` (~20 literals) | `"user_profile"` |
| DTOs `PersonaProfileCreate` / `PersonaProfileUpdate` / `PersonaProfileResponse` (16 files) | `UserProfileCreate` / `UserProfileUpdate` / `UserProfileResponse` |
| Messages `EditPersonaRequested`, `PersonaProfileSaveRequested`; widget `PersonaProfileEditorWidget` | `EditUserProfileRequested`, `UserProfileSaveRequested`, `UserProfileEditorWidget` |
| `ConsoleSessionSettings.persona_label` (~23 refs; serialized) | `user_profile_label` (with load-compat, B3) |
| Service naming: the persona halves of `local_character_persona_service.py` / `CharacterPersonaScopeService` (method/param/attr names like `persona_store_path`, `list_persona_profiles`, `_persona_profile_view`) | user-profile naming (`user_profile_store_path`, `list_user_profiles`, …); module/class renames included where they name the user-profile concept |

### B2. Explicit boundary (OUT of scope — the workbench axis)

`PersonasScreen` / `personas_screen.py` / `Persona_Widgets/` / `PersonasPreviewPane` / `personas_pane_messages.py` and the widget family name the **workbench** (user-facing "Roleplay & Chat Dictionaries" since P0), not the user. Renaming that family is the workbench-rename axis (massive import churn, zero user-reference semantics) and stays out unless separately requested. The spec records this boundary deliberately.

### B3. Persistence compatibility (LOAD-BEARING — accept old, write new)

Three persisted surfaces carry the old names and MUST keep working for existing users:
1. **Saved screen state:** `active_mode: "personas"` in persisted workbench state, and `persona_label` inside serialized `ConsoleSessionSettings` (screen-state + session persistence). Readers accept BOTH old and new keys/values (old value normalized on load: `"personas"` → `"user_profiles"`, `persona_label` → `user_profile_label`); writers emit only the new. Tests round-trip a pre-rename serialized blob.
2. **On-disk profile JSONs:** the profile files' own schema fields keep loading unchanged (the rename is of Python identifiers/DTO class names, not the JSON field contract — verify at plan time which JSON keys, if any, say "persona" and apply the same accept-old-write-new rule to those). The store DIRECTORY default stays wherever existing users' files are (rename the parameter, not the path), or reads fall back to the old path if the default moves — existing files must load with zero user action.
3. **Config:** any existing config keys referencing personas keep being read (accept-old); the new `active_user_profile` key is net-new.

The P1e/P2 standing rule applies: guard EVERY read of persisted/imported content (isinstance/str-normalize/coerce).

## Error handling

Dangling active pointer → `None` + (at most) a gentle notify on the marking surface; never a crash, never a blocked send. Config write failures surface via the existing `_notify` error path. All substitution falls back to `"User"` on any resolution failure. Rename compat readers never raise on old-shaped state (accept-old is total).

## Testing

- **Resolver:** set → resolve; clear; dangling (deleted profile) → None; config persistence round-trip (write → fresh read).
- **Placeholders:** the full token table (`{{user}}`/`{{random_user}}`/`<USER>` → user name; `{{char}}`/`{{character}}`/`{{persona}}`/`<CHAR>` → character name); byte-compat for token-free text.
- **Substitution:** preview seeding + Start-Chat greeting + enhanced-chat path with an active profile (name appears) and without (byte-identical "User" output — the AC3 pin).
- **Labeling:** preview `set_speakers` with/without; session label set on Start-Chat.
- **Marking UX:** inspector action sets/clears the pointer; list-row indicator; delete-active clears pointer.
- **Rename:** the full suite passes under the new names; compat tests load a pre-rename saved state (`"personas"` mode + `persona_label`) and an existing profile JSON; grep-gate test (optional): no NEW user-facing string says "Persona" for the user-side concept.

## Global constraints (for the plan)

- Placeholder table exactly as §Decisions-1 — user-side tokens NEVER get the character name.
- Active pointer = config single-value; the P3b `enabled`/`is_active` flag untouched.
- Accept-old-write-new on ALL three persisted surfaces; existing user files/state load with zero action.
- No active profile ⇒ byte-identical behavior (pinned).
- Rename boundary per §B2 (workbench family stays).
- No DB migration (profiles are file-backed; nothing schema-touching).
- Established process constraints: concurrent-session hazard on the personas files (survey drift MID-cycle too — the B2 lesson); implementers stage only task files; foreground tests; file-backed DB fixtures where DB is involved; Tests/UI asyncio rules.

## Out of scope

- Profile description/traits flowing into the system prompt (model-visible prompt engineering — own follow-up).
- Console transcript user-row speaker slots if none exist today.
- The workbench-family rename (§B2).
- Import of personas/user-profiles from external formats.
