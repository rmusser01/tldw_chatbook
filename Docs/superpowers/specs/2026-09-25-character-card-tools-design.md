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

- Wired per turn by `console_chat_controller` through a `_character_wiring(session_id)`
  helper, following the existing `_todo_wiring(session_id)` /
  `_ask_user_wiring(session_id)` precedent (the controller builds a fresh
  `LocalToolProvider` per turn with the session id in hand). The wiring supplies
  the session's runtime backend and an injected change notifier.
- Refuses every call when that session's runtime backend is `server`:
  "Character editing is local-only; switch this chat to local to create or edit
  characters."
- **Truncation guard (per Console session, not per turn):** a turn-scoped memory
  would forget a truncated read in turn N before the save in turn N+1 (the usual
  "propose, then confirm" flow). The guard state lives on the controller per
  session (as todo state does) and records fields read *in full* at a given
  version. Rule at save time: a field whose **stored** value is longer than the
  `character_get` per-field bound may be updated only if this session has read it
  in full at the current version; shorter fields are never truncated and need no
  record.

### 3.1a Local character service fixes (required, and a latent bug)

`LocalCharacterPersonaService.create_character`/`update_character` validate
`image_base64` but pass it straight to `ChaChaNotes_DB`, which only understands an
`image` key holding bytes — **so images sent through this service are silently
dropped today** (only the card-import path in `Character_Chat_Lib` decodes
`image_base64`). Its `update_character` also strips `None` values, so it cannot
remove an image. Fix: decode and validate `image_base64` into `image` bytes
(reusing the import path's decoder and limits), and add an explicit
`clear_image=True` on update. Covered by service-level tests independent of the
tools.

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

- `generate`: `compose_expression_prompt(state="avatar", …)` (or the model's
  prompt), with the user's configured expression style template when one is set →
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
- Library ▸ Skills: the skill pane is a full editor (dirty tracking, Save/Discard
  vetoes, trust review) with no read-only mode, so built-ins do **not** open it.
  A built-in row carries a "Built-in" badge and opens a **read-only preview**
  (rendered Markdown) with **Customize** and an **Enabled** switch. A user copy is
  labelled "overrides built-in"; the built-in row then reads "overridden". The
  editor is untouched.
- Local mode only.

### 3.6 Change notification

After a successful save the service calls its injected notifier (supplied by the
wiring), which posts an app-level `CharacterCardChanged` message thread-safely;
the service itself imports no UI. The Personas screen reloads that character when
showing it; if its editor is dirty (existing change-based tracking,
`_mark_dirty`) it keeps the edits and shows "This character was changed
elsewhere" instead of discarding them.

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
Every result is sized to fit the agent runtime's head-first tool-result cut
(`RunBudget.max_tool_result_chars`, 16,000 chars by default) and the local
provider's 32 KB cap, measured after JSON escaping: the full card gives each
field ~1,000 chars (`CHARACTER_FIELD_READ_BOUND`, also the guard threshold),
and a `field` page ~12,000 (`CHARACTER_FIELD_PAGE_BOUND`). Text the model never
received is never counted as read (final review C1).

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
- **Service fixes (§3.1a):** `image_base64` round-trips to stored bytes on create
  and update; `clear_image` removes it; invalid base64 is rejected.
- **Truncation guard across turns:** a full read in one turn permits the save in
  the next; a truncated-only read never does.
- **End to end:** a mounted Console run modelled on
  `Tests/UI/test_console_watchlists_mounted_uat.py` (approvals as in
  `test_console_mcp_approval.py`): search → draft → save → approve, asserting the
  DB row and the Personas refresh message.

## 7. Out of scope (v1)

Delete; lorebook/world-book linking; `extensions`; card import/export; server
(tldw_server) mode; exposing the tools to external MCP clients; reusing an image
generated earlier in the chat as the avatar.

## 8. Resolved during spec review

- Session backend and per-session state: `_character_wiring(session_id)` on the
  controller (precedent `_todo_wiring`).
- End-to-end harness: `test_console_watchlists_mounted_uat.py` pattern.
- Library ▸ Skills: read-only preview, not the editor (§3.5).
- Local service image handling: broken today; fixed in §3.1a.
