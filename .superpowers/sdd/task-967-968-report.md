# TASK-967 / TASK-968 report

Worktree: `/Users/macbook-dev/Documents/GitHub/wt-967-968`, branch `fix/chatbook-accessors-and-mcp-service`, cut from `origin/dev`.

## TASK-967 — Route Chatbook window and wizard files through the path accessors

### What I found

Task-865's Implementation Notes named the *exact* deferred finding this task was filed to capture:
`ChatbookCreationWindow.py`, `ChatbookExportManagementWindow.py`, `ChatbookCreationWizard.py` and
`ChatbookImportWizard.py` each built an ad-hoc `db_paths` dict from
`self.app.config_data.get('database', {})` with hardcoded fallback literals like
`'~/.local/share/tldw_cli/tldw_prompts_db.db'`, and `ChatbookCreationWindow.py`'s output dir was a bare
`Path.home() / '.local' / 'share' / 'tldw_cli' / 'chatbooks'` with a bare `.mkdir()`.

**That specific defect was already fixed on current dev before this task started** — all four files now
call the real accessors (`get_chatbook_database_paths()`, `get_private_chatbooks_dir()`), and zero literal
`~/.config/tldw_cli` or `~/.local/share/tldw_cli` string remains in any Chatbook window/wizard file
(verified by grep across all 10 candidate files, and by the existing, passing
`Tests/Chatbooks/test_chatbook_database_paths.py::test_chatbook_surfaces_do_not_embed_database_defaults`,
parametrized across all four files).

### A related-but-out-of-AC-scope finding, reported not silently fixed

Three live files — `ChatbookExportManagementWindow.py`, `Chatbooks_Window_Improved.py`,
`Wizards/ChatbookCreationWizard.py` — default the *visible* chatbooks export/scan directory to
`Path.home() / "Documents" / "Chatbooks"` (already correctly hardened via
`secure_private_directory`/`secure_chatbook_directory` in all three). This is not the AC's named literal
class (it's not `~/.config/tldw_cli` or `~/.local/share/tldw_cli`), and it diverges from
`get_private_chatbooks_dir()` (`~/.local/share/tldw_cli/<user>/chatbooks`), which
`ChatbookCreationWindow.py` + `Tools_Settings_Window.py`'s modal-based creation flow already use for the
same conceptual directory. Switching the three wizard/window files to `get_private_chatbooks_dir()` would
silently stop the management window from finding chatbooks a user already exported via the wizard flow —
the exact live-data-relocation risk the task's constraints told me to stop and report on rather than act
on. **Filed as a cross-window inconsistency worth its own follow-up**, not resolved here.

### What I fixed

- `tldw_chatbook/UI/Chatbooks_Window.py` (confirmed dead code — not imported anywhere in the live app
  except a `.skip`'d integration test) still built its export path from raw `config.get()` +
  `.expanduser()` + a bare `.mkdir()`. Routed it through `Chatbooks/database_paths.secure_chatbook_directory`
  (the same helper `Chatbooks_Window_Improved.py` already uses), matching the `chatbook_importer.py`
  hardening pattern the task pointed at.
- Deleted `Tests/Chatbooks/conftest.py`'s unused `mock_app_config` fixture and `MockWizardApp` class: dead
  test scaffolding referenced by zero tests, re-spelling the exact stale db-path/export-directory literals
  the earlier production fix already replaced — left in place it would have been misleading debris of the
  "test repeats a literal instead of deriving it" shape the task warns about.

### Verification

- `Tests/Chatbooks/` — 159 passed, 1 skipped.
- `Tests/UI/test_chatbook_action_recovery_tooltips.py`, `test_chatbook_management_server_jobs.py`,
  `test_chatbooks_screen_server_actions.py`, `test_file_picker_action_tooltips.py` — 22 passed.

### Status

Backlog task-967 marked **Done**, AC #1 checked.

---

## TASK-968 — Fix the missing `CharacterInteropService` referenced by MCP server

### Decision: remove, not resolve

`Character_Chat/Character_Chat_Lib.py` is a free-function module — it has never had a single class in it
(confirmed by listing every top-level `class`/`def`). `CharacterInteropService` was never renamed or moved;
it was never implemented at all. `self.character_service` (built from it in `_init_databases()`) was never
read anywhere else in `MCP/server.py` or the rest of the `MCP/` package. The character-related tools that
*are* wired up — `chat_with_character`, `list_characters` — go through `self.tools` (`MCPTools`), which
already reads character rows directly off `self.chachanotes_db` via `get_character_card_by_id()` /
`list_character_cards()`. There was no unfinished feature behind the reference to wire up — only a
guaranteed `ImportError` on every call to `_init_databases()`. **Removed** the import and the
`self.character_service = ...` line, with an explanatory comment in place.

### Before / after reproduction

Technique: bypass `TldwMCPServer.__init__` (which requires the optional `mcp` package, not installed in
this venv) via `__new__`, then call the real, unmodified `_init_databases()` directly — the same technique
`Tests/MCP/test_server_media_db_path.py` already used for TASK-854. `NotesInteropService` is stubbed with a
permissive fake because its own call-signature mismatch (see "broader look" below) is a separate,
pre-existing defect that would otherwise mask the specific thing being reproduced.

Commands (run from the worktree, `HOME`/`TLDW_CONFIG_PATH` redirected to a scratch dir so nothing touches
real config/databases; `PYTHONPATH` pinned to the worktree so `tldw_chatbook.__file__` resolves there, not
the main checkout):

```bash
SCRATCH=<scratchpad>
# BEFORE: git HEAD's server.py (pre-fix) copied temporarily into place
cp <scratchpad>/server_before.py tldw_chatbook/MCP/server.py
HOME="$SCRATCH/home968" TLDW_CONFIG_PATH="$SCRATCH/home968/.config/tldw_cli/config.toml" \
  PYTHONPATH=/Users/macbook-dev/Documents/GitHub/wt-967-968 \
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python <scratchpad>/repro_968.py
# -> _init_databases() RAISED: ImportError: cannot import name 'CharacterInteropService' from
#    'tldw_chatbook.Character_Chat.Character_Chat_Lib'   (exit 1)

# AFTER: restore the fixed server.py
cp <scratchpad>/server_current_fixed.py tldw_chatbook/MCP/server.py
HOME="$SCRATCH/home968" TLDW_CONFIG_PATH="$SCRATCH/home968/.config/tldw_cli/config.toml" \
  PYTHONPATH=/Users/macbook-dev/Documents/GitHub/wt-967-968 \
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python <scratchpad>/repro_968.py
# -> _init_databases() SUCCEEDED; has character_service attribute = False   (exit 0)
```

`repro_968.py` (in the session scratchpad) just monkeypatches `NotesInteropService` to a permissive fake at
its source module, constructs `TldwMCPServer.__new__(...)`, calls `._init_databases()`, and prints
success/failure plus whether `character_service` exists. The same assertions are now pytest-enforced in
`Tests/MCP/test_server_character_service.py` (no scratch files needed to re-run them).

### Broader look at `MCP/server.py` (as asked — checked, not everything fixed)

- **Fixed** (in scope, low-risk, single call site): `chat_with_llm`'s API-key lookup called
  `get_cli_setting("API", f"{provider}_api_key", "")` directly instead of `config.get_api_key(provider)`,
  the declared accessor. `get_api_key()` checks three tiers — the newer `api_settings.<provider>` structure
  (with its own env-var-name indirection), then the legacy `[API]` section (the *only* tier the direct call
  covered), then a bare `{PROVIDER}_API_KEY` env var. The direct call silently missed a key configured via
  either of the other two tiers, returning `"No API key configured"` even when a real one existed.
  Repointed to `get_api_key(provider)`; the now-unused `get_cli_setting` import was removed.
- **Reported, not fixed** (own design decision, out of this task's scope): `self.notes_service =
  NotesInteropService(self.chachanotes_db)` is *also* broken, independently of the character bug.
  `NotesInteropService.__init__` requires `(base_db_directory: str|Path, api_client_id: str,
  global_db_to_use=None)` — the call passes only one positional arg (a `CharactersRAGDB` instance) where a
  directory path is required, and omits the required `api_client_id` entirely. (This is exactly why
  `test_server_media_db_path.py` has to stub `NotesInteropService` with a permissive fake just to get past
  `_init_databases()` at all.) Even a corrected construction wouldn't help downstream: the `create_note`
  tool calls `self.notes_service.create_note(title=, content=, tags=, template=)`, but the real class has
  no `create_note` method (only `add_note(user_id, title, content, note_id=None)` — no `tags`/`template`
  params, and a required `user_id` the tool call never supplies); `search_notes` calls
  `self.notes_service.search_notes(query=, limit=)`, but the real signature is
  `search_notes(user_id, search_term, limit=10, fts_match_query=None, *, id_allowlist=None)` — wrong kwarg
  name (`query` vs `search_term`) and again a missing required `user_id`. Fixing this needs a real design
  decision (where does an MCP-context `user_id` come from?) that's beyond this task's scope — recommend
  filing separately.
- **Verified clean**: `MCPResources` and `MCPPrompts` constructors, and every method `server.py` calls on
  `self.resources`/`self.prompts` (`get_conversation_resource`, `get_note_resource`,
  `get_character_resource`, `get_media_resource`, `get_rag_chunk_resource`, `list_recent_conversations`,
  `list_recent_notes`, `summarize_conversation_prompt`, `generate_document_prompt`, `analyze_media_prompt`,
  `search_and_synthesize_prompt`, `character_writing_prompt`) all exist with matching signatures.
  `MCPTools`'s other methods used by `server.py` (`search_conversations`, `get_conversation_history`,
  `export_conversation`, `perform_rag_search`) also verified present with matching signatures.
  `get_chachanotes_db_path()`/`get_media_db_path()` (TASK-854's fix) remain correct. `ingest_media` is an
  explicit `TODO`/placeholder, not a silent-failure case.

### Files changed

- `tldw_chatbook/MCP/server.py` — removed the dead `CharacterInteropService` import + assignment (with
  explanatory comment); repointed `chat_with_llm`'s API-key lookup to `get_api_key`.
- `Tests/MCP/test_server_media_db_path.py` — dropped the now-unnecessary `CharacterInteropService` stub,
  updated the module docstring to reflect the fix.
- `Tests/MCP/test_server_character_service.py` (new) — 4 tests: no `CharacterInteropService`
  import/reference remains (AST-based, so it doesn't false-flag the explanatory comment),
  `_init_databases()` succeeds without the old stub and has no `character_service` attribute,
  `Character_Chat_Lib` still has no such class (sanity check on the remove-vs-resolve decision), and
  `chat_with_llm` calls `get_api_key` rather than `get_cli_setting` (AST-based).

### Verification

- `Tests/MCP/test_server_character_service.py` + `test_server_media_db_path.py` +
  `test_server_chat_repoint.py` — 10 passed.
- `Tests/MCP/` (full directory) — 366 passed.
- `Tests/Utils/` (full directory) — 562 passed.

### Status

Backlog task-968 marked **Done**, AC #1 checked.

---

## Commits

See git log on `fix/chatbook-accessors-and-mcp-service` for the SHAs (not pushed).
