# Character card tools and built-in Character Creator skill — design

Status: draft for review · 2026-09-25 · base `origin/dev` 461668df00

## 1. Intent

**What the user asked for:** a user can ask the Console assistant to create a new
character card, or to update/modify an existing one, and a Character Creator
skill guides the assistant through it. The skill ships with the app and is on by
default.

**Success looks like:** on a fresh install, "make me a character who …" produces
an interview, a full draft the user reads and revises in chat, and — after one
approval — a saved character (optionally with an avatar) the user can chat with
immediately. "Make Aria more sarcastic and give her a new avatar" edits only what
was asked, never silently overwrites newer edits, and never loses text.

**Decisions made during brainstorming (user-confirmed):**

| Question | Decision |
| --- | --- |
| Scope | Card text fields + avatar (no lorebook linking, no delete) |
| Avatar source | Generate via the configured Image Gen backend **or** a local file path |
| Defaults | Skill active and tools on by default; every save asks for approval |
| Skill delivery | Built-in read-only source in the app package, plus an on-demand "Customize" copy (the local `seed_builtin_skills`) |
| Tool home | A `character_*` family in `LocalToolProvider`, modelled on `watchlists_*` |
| Tool shape | Separate read and write tools (not one multi-action tool, not whole-card JSON) |
| Server mode | Local-only in v1, matching the app (the Personas editor saves locally) |

## 2. Rejected alternatives (and why)

- **Legacy gated `Tools/` builtins** (`create_note` style): the older path; the
  Console's data tools no longer use it.
- **The Library tool contract** (`library_*`): exposed only when
  `[console] direct_library_tools` is on — a *retrieval-mode* setting, so users
  on RAG-only retrieval would silently lose character authoring; its descriptors
  are auto-registered on the MCP server (would expose a write tool to external
  clients); its service is local-only; the tool count is pinned by tests.
- **One `manage_character` tool:** a single permission row cannot keep reads and
  writes at different risk levels; vaguer schema.
- **Whole-card JSON get/save:** the model must resend every field to change one;
  a truncated reply can erase fields.
- **Seeding the skill into the user store at launch** (server-style): a seeded
  copy is outside the user's trust manifest and can be blocked from the model for
  exactly the users who set up skill trust.

## 3. Components

### 3.1 `CharacterToolService` (new, `Character_Chat/character_tool_service.py`)

Synchronous service wrapping `LocalCharacterPersonaService`, which validates
with the same `CharacterCreateRequest`/`CharacterUpdateRequest` schemas (field
limits, `image_base64`) and enforces `expected_version`. (The Personas editor
itself writes through `ccp_character_handler` to the DB; this service is the
schema-validated path.) Owns bounding, the truncation guard, avatar resolution
and the change notification. No UI imports.

- Refuses every call when the active Console session's runtime backend is
  `server`: "Character editing is local-only; switch this chat to local to create
  or edit characters."
- Holds, per agent run, the set of `(character_id, version, field)` returned
  truncated and not since read in full (the truncation guard, §4.4).

### 3.2 Tool specs (in `Agents/local_tool_provider.py`)

Three `LocalToolSpec`s, all `exposure=CONSOLE_ONLY`:

| Tool | Tags | Execution policy | Approval effects |
| --- | --- | --- | --- |
| `character_search` | `()` | `BOUNDED_ABANDONABLE` | `PRIVATE_READ` |
| `character_get` | `()` | `BOUNDED_ABANDONABLE` | `PRIVATE_READ` |
| `character_save` | `("mutates",)` | `DEFINITIVE_AFTER_START` | `MUTATES_LOCAL` (+ `LLM_SPEND` when generating on a paid backend) |

`character_save` supplies `approval_arguments` (a summary; §4.3) and gets a
`timeout_for` override when an avatar is requested, derived from the Image Gen
request timeout plus grace (precedent: `web_deep_search`).

### 3.3 Gate

`[tools] character_tools_enabled`, **default ON** — special-cased like
`ask_user` (`ASK_USER_DEFAULT_ENABLED`), hand-listed in `all_tool_gates()` so the
MCP hub's Tool gates pane shows a switch with title and blurb. The local-tools
master switch (default ON) still governs the family.

### 3.4 Avatar helper (new, `Character_Chat/character_avatar.py`)

`resolve_avatar(request, card) -> AvatarOutcome` with no UI dependencies:

- `generate`: `compose_expression_prompt` (or the model's prompt) →
  `Image_Generation.worker.build_request` → `run_generation`. Refuses up front
  when no image backend is configured.
- `file`: path checked with `path_validation`; bytes read with the same size and
  type limits as the Personas editor's avatar upload.
- `remove`: clears the image.

### 3.5 Built-in skills source (`Skills_Interop/`)

- Files ship at `tldw_chatbook/assets/skills/character-creator/SKILL.md`, listed
  in `[tool.setuptools.package-data]` (`include-package-data = false`).
- `LocalSkillsService._visible_records()` = index records ∪ built-in records; a
  user skill with the same name wins. **All read paths use it**: list, get,
  `get_context`, `read_skill_file`, `execute_skill`, and the Console's per-send
  capture (`console_chat_controller.capture_skill_context_maximum`, which reads
  `_load_index()` directly today). **Write paths keep the raw index.**
- Built-in records carry `source="builtin"`, `trust_status="builtin"`, and skip
  the trust service (no manifest/keyring reads).
- Integrity: a SHA-256 per built-in file is pinned in code and verified at load;
  a mismatch blocks the skill with reason "built-in skill modified". Built-ins may
  not contain scripts (loader refuses).
- Read-only: update/delete of a built-in name is refused ("Built-in skills are
  read-only. Use Customize to make your own copy.").
- Customize = local `seed_builtin_skills(overwrite=False)` (today a stub): copies
  the chosen built-in into the user's store, where it then overrides the built-in
  and goes through normal trust.
- Off switch: `[skills] disabled_builtins = []`, read from the cached config
  snapshot (never a per-send `get_cli_setting`).
- Library ▸ Skills: built-in rows show Customize and the toggle, hide Edit/Delete,
  and are labelled "overridden" when a user copy exists (the copy is labelled
  "overrides built-in"). If the Skills list cannot host a read-only row cleanly,
  the toggle falls back to Settings.
- Local mode only.

### 3.6 Change notification

After a successful save the service posts an app-level `CharacterCardChanged`
message (via `call_from_thread`). The Personas screen reloads that character when
showing it; if its editor has unsaved changes it keeps them and shows "This
character was changed elsewhere" instead of discarding them.

## 4. Tool contracts

### 4.1 `character_search`

Args: `query` (optional; empty lists all), `limit` (1–25, default 10), `offset`
(default 0). Returns, per character: `id`, `name`, one-line truncated
`description`, `tags`, `version`, `has_avatar`, `updated_at`; plus `next_offset`
when more exist. Output bounded to a fixed size.

### 4.2 `character_get`

Args: `id`; optional `field` + `offset` to page one long field. Returns every
editable field, `version`, `has_avatar` (never image bytes). Each long field is
bounded with `truncated: true` and `next_offset`; truncated fields are recorded
for the truncation guard.

Editable fields (from `CharacterCreateRequest`/`CharacterUpdateRequest`):
`name`, `description`, `personality`, `scenario`, `first_message`,
`message_example`, `system_prompt`, `post_history_instructions`,
`creator_notes`, `creator`, `character_version`, `alternate_greetings`, `tags`.
**`extensions` is excluded** — other features store data there (visual identity,
actor packs).

### 4.3 `character_save`

- **Create:** no `id`; `name` required; other fields optional.
- **Update:** `id` + `expected_version` + only the fields to change. Omitted =
  unchanged; empty string = clear; list fields are replaced whole.
- **`avatar`** (optional): `{"source":"generate","prompt"?}` |
  `{"source":"file","path"}` | `{"source":"remove"}`.
- **Approval card** (`approval_arguments`): create/update, character name,
  changed field names with sizes, avatar action (+ backend and a cost note for
  paid backends, + the file path for `file`). Never the full field text.
- **Order after approval:** validate → resolve avatar (generate/read) → **one**
  create/update carrying text and image → one version bump → notify. If the
  avatar fails, save the text alone and report the avatar failure.
- Returns `id`, new `version`, `changed_fields`, `avatar`:
  `saved | failed: <reason> | none`.

### 4.4 Errors (all returned as tool text; nothing raises into the agent loop)

Not found · stale version ("re-read with character_get") · duplicate name ("A
character named X already exists (id N); update it or choose another name") ·
field over limit (field + limit) · update to a field last seen truncated ("read
the full field first") · server-mode refusal · no image backend · generation
failed/timed out · avatar path rejected / too large / unsupported type · partial
success ("Saved; avatar failed: …"). Approval denied writes nothing.

### 4.5 Logging

Metadata only (tool name, outcome, error type) — never field text or paths.
Update `Docs/security/production-diagnostic-inventory.json` with reviewed rows.

## 5. The Character Creator skill (content outline)

1. **Create:** ask at most ~5 questions (concept, role, tone, setting, how the
   user will chat with it); draft every field (third-person description and
   personality, scenario, first message in the character's voice using
   `{{char}}`/`{{user}}`, `<START>`-format example messages, optional system
   prompt); **show the full draft and revise until the user approves; only then
   call `character_save`.**
2. **Edit:** `character_search` → `character_get` (page long fields in full) →
   propose a before/after → confirm → `character_save` with `expected_version`.
3. **Avatar:** offer to generate from the card's appearance, or use a file.
4. **After saving:** offer to start a chat with the character via the existing
   new-chat tool.
5. **Permissions:** the first time a read asks for approval, mention once that the
   two read tools can be set to Allow in the MCP hub.

## 6. Testing

TDD; real in-memory SQLite.

- **Service:** create; partial update; `extensions` untouched; stale version;
  duplicate name; truncation guard (blocks, then allows after a full read);
  server-mode refusal; field limits.
- **Avatar:** fake image backend — single write, single version bump; generation
  failure saves text and reports; file validation.
- **Provider:** three tools, Console-only; `mutates` save still asks under Allow;
  gate default ON and switchable; `DEFINITIVE_AFTER_START`; timeout override;
  approval card contains no raw field text.
- **Skill:** present in `get_context` **and** the real per-send capture path, with
  a counting keyring proving no trust read; digest mismatch blocks; user copy
  overrides; `disabled_builtins` honoured without per-send config reads;
  Customize copies; edit/delete refused; `SKILL.md` present in the built wheel.
- **End to end:** one real-app agent run with a scripted model (search → draft →
  save → approve) asserting the DB row and the Personas refresh message. (Which
  scripted-provider harness exists is confirmed in the plan.)

## 7. Out of scope (v1)

Delete; lorebook/world-book linking; `extensions`; card import/export; server
(tldw_server) mode; exposing the tools to external MCP clients; reusing an image
generated earlier in the chat as the avatar.

## 8. Open items to confirm in the implementation plan

- The scripted-provider harness for the end-to-end test.
- Whether Library ▸ Skills can render a read-only built-in row (else the toggle
  moves to Settings).
- How the tool service learns the active Console session's runtime backend
  (turn context vs. session store).
- That `LocalCharacterPersonaService` maps `image_base64` onto the DB `image`
  column on create and update; if it does not, the single write sets the DB
  `image` field directly in the same `update_character_card` call.
