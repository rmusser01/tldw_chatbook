# TASK-962 / TASK-983 — implementation report

Worktree: `/Users/macbook-dev/Documents/GitHub/wt-983-930`, branch `fix/mcp-server-pass-and-raw-toml`.

## TASK-962 — Tools_Settings_Window raw-TOML save path

**Finding: already fixed on current `dev`. No source change made.**

`UI/Tools_Settings_Window.py::_save_raw_toml_config` no longer does the
described `open(DEFAULT_CONFIG_PATH, 'w') + toml.dump`. It calls
`config.replace_cli_config(config_data)`, which:

- resolves the target file via `get_cli_config_path()` (== the profile-aware
  `_get_effective_config_path()`, honoring `TLDW_CONFIG_PATH`), and
- persists via `_write_raw_cli_config_unlocked()` → `atomic_private_write_text()`
  (temp-file-plus-rename write, hardened file mode).

This was already true on `dev` before this session started (per task-962's
own Implementation Notes, the old code path was removed in a prior
dev-reconciliation commit, `1df0c4cb4`). Confirmed by reading the current
source and by running the three existing regression tests in
`Tests/UI/test_tools_settings_window.py`
(`test_save_raw_toml_config_writes_effective_path_not_default_decoy`,
`test_save_raw_toml_config_is_atomic_on_serialization_failure`,
`test_save_raw_toml_config_roundtrips_with_no_profile_override`) — all 3 pass.

### Before / after repro

No "before" exists to reproduce against on this branch (the bug was already
gone). To verify the *current* behavior end to end, a manual repro script
was run under a fully redirected `HOME` (never touching the real user
profile):

- `TLDW_CONFIG_PATH` → a temp profile file seeded with
  `plaintext_sentinel = "ORIGINAL_SENTINEL_VALUE"`.
- `config.DEFAULT_CONFIG_PATH` monkeypatched to a separate temp decoy file
  seeded with `plaintext_sentinel = "DECOY_SHOULD_NOT_CHANGE"` (simulating
  the old hardcoded-wrong-path target).
- Called `config.replace_cli_config({"plaintext_sentinel": "UPDATED_SENTINEL_VALUE", ...})`
  (the same call `_save_raw_toml_config` makes).

**Result — which file actually changed:**

| File | Before | After |
|---|---|---|
| Profile file (`_get_effective_config_path()`'s result) | `ORIGINAL_SENTINEL_VALUE`, mode `0o600` | `UPDATED_SENTINEL_VALUE`, mode `0o600` (unchanged) |
| Decoy `DEFAULT_CONFIG_PATH` | `DECOY_SHOULD_NOT_CHANGE` | `DECOY_SHOULD_NOT_CHANGE` (untouched) |

The write landed exclusively in the effective/profile path, the decoy was
never touched, the file mode was preserved (not widened), and the returned
in-memory config round-tripped every configured value/type correctly
(verified in the repro's captured output).

No files modified for this task.

---

## TASK-983 — MCP notes tools called a nonexistent API

### The core fix

`TldwMCPServer._init_databases()` constructed
`NotesInteropService(self.chachanotes_db)` — one positional argument (a
`CharactersRAGDB`) bound to the real class's first parameter,
`base_db_directory: Union[str, Path]`. Every construction failed before the
server could open a single connection.

**Decision: wire it correctly, don't remove it.** A fully working
implementation of the same feature already exists in this codebase
(`Tools/note_management_tools.py`'s `CreateNoteTool`/`SearchNotesTool`, and
`app.py`'s own `NotesInteropService` construction), proving the feature is
finished and intentional — only this one call site was wired wrong. Fixed
to match that established pattern exactly:

```python
self.notes_service = NotesInteropService(
    base_db_directory=get_chachanotes_db_path().parent,
    api_client_id=CLI_APP_CLIENT_ID,
    global_db_to_use=self.chachanotes_db,
)
```

`create_note` and `search_notes` were then rewired to the real API:

- `create_note(title, content)` → `notes_service.add_note(user_id=, title=, content=)`.
  Dropped `tags`/`template` from the tool's own parameter schema entirely:
  neither `NotesInteropService` nor the `notes` table has any such concept
  (no tags column, no template system) — they were never real, just assumed.
- `search_notes(query, limit)` → `notes_service.search_notes(user_id=, search_term=, limit=)`.
  The old call passed `query=`/`limit=` with no `user_id` (a `TypeError` on
  the real signature) and then read results via attribute access
  (`note.id`, `note.updated_at`) — the real method returns plain **dicts**
  keyed by the real `notes` table columns, which has `last_modified`, not
  `updated_at`.

`user_id` is resolved via a new `_resolve_notes_user_id()` helper mirroring
`Tools/note_management_tools.py::_resolve_user_id` (`load_settings()["USERS_NAME"]`
or `"default_user"` — an attribution value, not a visibility partition).

Proved end to end (not just import) in `Tests/MCP/test_server_notes_service.py`:
bypasses `__init__` via `__new__` (same technique the existing
TASK-854/968 tests use, since the optional `mcp` package isn't installed in
this venv), runs the real `_init_databases()`, swaps a recording fake in for
`FastMCP` so the actual `create_note`/`search_notes` closures can be
captured and called directly, and creates + full-text-searches a note
against a temp on-disk database with `HOME`/`TLDW_CONFIG_PATH` redirected to
a temp profile (never touching the real user config/DB). 3/3 pass. The
service now constructs without needing the `_PermissiveFakeService` stub
the TASK-854/968 tests had to use — their docstrings were updated to note
this (stub left in place there anyway, to keep those files' own scope
narrow).

### Whole-module pass

Per the mandate to check "every service construction, config lookup,
import, and tool handler," `MCP/server.py` and the three collaborator
modules its tool/resource/prompt handlers delegate to (`tools.py`,
`resources.py`, `prompts.py`) were read in full and every call checked
against the real class it targets.

**Fixed (unambiguous — a real method/column exists with the obviously
intended value, no design call needed):**

| Site | Was | Now |
|---|---|---|
| `tools.py::chat_with_character` | `get_cli_setting("API", f"{provider}_api_key", "")` | `get_api_key(provider)` (same defect TASK-968 already fixed at the sibling call site in `server.py::chat_with_llm`) |
| `resources.py::list_recent_conversations` | `chachanotes_db.get_recent_conversations(limit=)` (doesn't exist) | `chachanotes_db.list_all_active_conversations(limit=)` (already ordered by recency) |
| `resources.py::list_recent_notes` | `chachanotes_db.get_recent_notes(limit=)` (doesn't exist) | `chachanotes_db.list_notes(limit=)` (already ordered by recency) |
| `resources.py::get_media_resource`, `prompts.py::analyze_media_prompt` | `media_db.get_media_transcript(media_id)` (never existed as an instance method) | module-level `get_media_transcripts(db, media_id)`, most recent transcript taken |
| `prompts.py::summarize_conversation_prompt`, `generate_document_prompt` | `chachanotes_db.get_conversation_messages(id)` (never existed) | `chachanotes_db.get_messages_for_conversation(str(id))` (already fixed at the identical shape elsewhere in this same module) |

**Also found — wrong dict-key access (not missing methods, wrong column
names), surfaced because they broke the new regression tests against real
seeded rows, then fixed the same way:**

- `media['media_type']` / `media['created_at']` (`resources.py`, `prompts.py`) —
  the `Media` table's real columns are `type` and `ingestion_date`.
- `char.get('greeting')` / `char.get('example_dialogue')` / `char.get('updated_at')`
  (`resources.py::get_character_resource`) — `character_cards`' real columns
  are `first_message`, `message_example`, `last_modified`.
- `conv.get('updated_at')` / `note.get('updated_at')` (`resources.py`) — both
  tables have `last_modified`, not `updated_at`.

All covered by 8 new tests in `Tests/MCP/test_tools_resources_prompts_real_methods.py`.

**Filed rather than guessed (genuine design decisions) — TASK-985:**

- `tools.py::search_conversations` calls `chachanotes_db.search_all_content(...)`,
  which does not exist anywhere in the codebase at all. The real analog,
  `search_conversations_by_content(search_query, limit)`, returns
  conversation rows with **no inline content column**, so the tool's
  `preview` formatting (`result["content"][:200]`) can't be ported as-is —
  what a preview should be sourced from instead is a product decision.
- `resources.py::get_rag_chunk_resource` calls
  `media_db.get_chunk_by_id(int(chunk_id))`, which also does not exist. The
  real accessor, `get_chunk_text(db, chunk_uuid)`, takes a **UUID string**,
  not an int id, and returns **bare text only** — no `media_id`/
  `start_char`/`end_char`/`embedding_id` (that table has no `embedding_id`
  column at all). The resource's whole id scheme and metadata contract need
  reconciling with what the DB can actually provide, not guessing.

**Noted, not fixed — harmless dead metadata (a feature never built, not a
crash; always-empty via defensive `.get()`):**

- Notes have no `tags`/`template` columns (`get_note_resource`'s tags/template
  metadata is always empty; keyword *linking* is a separate real API,
  `link_note_to_keyword`, this resource never calls).
- `character_cards` has no `message_count` (`list_available_characters` /
  `get_character_resource` always report `0`; would need a join/count query).
- `Media` has no `duration` column (`get_media_resource`'s duration metadata
  is always empty).

### Tests / files

- Added: `Tests/MCP/test_server_notes_service.py` (3 tests),
  `Tests/MCP/test_tools_resources_prompts_real_methods.py` (8 tests).
- Modified: `tldw_chatbook/MCP/server.py`, `tools.py`, `resources.py`,
  `prompts.py`.
- Updated stale scope-disclaimer docstrings in
  `Tests/MCP/test_server_media_db_path.py` and
  `Tests/MCP/test_server_character_service.py` (they used to explain a
  `NotesInteropService` stub was needed because of "an unrelated,
  pre-existing defect out of this task's scope" — that defect is now fixed
  for real; the stub is kept in those files anyway to keep their own scope
  narrow).
- Filed `backlog/tasks/task-985` for the two defects that need a design
  decision (id scanned against local `backlog/tasks/` max, 984, and
  `origin/dev`'s tree max, 1010 — 985–1009 were free on both; re-scanned
  after filing, still no collision).

### Test results (foreground, this session)

- `Tests/MCP/`: 377 passed.
- `Tests/UI/test_tools_settings_window.py`: 49 passed, 16 skipped
  (pre-existing `AppTest not available in this version of Textual` /
  one intentionally-hard-to-mock case — unrelated to this change).
- `Tests/Utils/`: 562 passed.
