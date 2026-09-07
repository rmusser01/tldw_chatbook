# Library ▸ Prompts + Console Injection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship Library ▸ Prompts (canonical prompt CRUD on the Notes-canvas template) and Console injection (`/prompt`, `/system`, system-prompt plumbing) per `Docs/superpowers/specs/2026-07-12-library-prompts-console-injection-design.md`.

**Architecture:** Phase 1 adds a prompts source to the Library shell (rail row + count seam, list/editor canvases, import/export, search source) through `PromptScopeService`/`LocalPromptService` — never raw DB from UI. Phase 2 adds a pure slash-command registry consulted ahead of `submit_draft`, a prompt picker, and a real per-session system prompt plumbed into every provider-message build and persisted with the conversation.

**Tech Stack:** Python 3.11+/Textual 8.2.7, SQLite (Prompts_DB FTS5, ChaChaNotes), pytest (real in-memory DBs; UI via real App subclass + `app.run_test()` — `textual.app.AppTest` does not exist).

## Global Constraints

- Base branch: `origin/dev`; work in worktree branch `claude/prompts-library-spec`. Local checkout `dev` is stale — never trust it.
- Tests run ONLY via `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest` (system python3 is 3.9 and breaks collection). Referred to below as `$PY -m pytest` with `PY=/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python`.
- TCSS: edit `tldw_chatbook/css/components/_agentic_terminal.tcss`, then regenerate with `$PY tldw_chatbook/css/build_css.py`; commit BOTH source and `tldw_chatbook/css/tldw_cli_modular.tcss`. Widget `DEFAULT_CSS` must parse standalone (no `$ds-*` without local fallbacks).
- Escape ALL user/server-derived text with `rich.markup.escape` before Button labels / markup-rendered Statics.
- Session system prompt default is **none** — never seed from `chat_defaults.system_prompt`.
- Prompt names: globally unique; NO auto-suffixing in the Library editor; `add_prompt` on a soft-deleted name returns `(None, None, msg)` with "soft-deleted" in msg — that is a distinct user-facing outcome, not success.
- Config reads at interaction time go through `tldw_chatbook.config.load_settings()` / scope services — no boot-snapshot reads (task-177 lesson).
- Every phase ends with the five Console/Library UI suites relevant to touched files green plus live served captures at 2050×1240; per-screen user approval before merge.
- Commit after every task; messages `feat(prompts): …` / `feat(console): …` etc. ending with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

---

## Phase 0 — Investigation

### Task 0: Decide the system-prompt persistence seam

**Files:**
- Read: `tldw_chatbook/DB/ChaChaNotes_DB.py` (conversations schema + `add_conversation`/`update_conversation`), `tldw_chatbook/Chat/console_chat_store.py` (`_persistence_scope`, conversation create/update calls)
- Create: `Docs/superpowers/plans/2026-07-12-prompts-task0-decision.md`

**Interfaces:**
- Produces: a decision note naming EITHER (a) an existing conversations metadata column that will store the system prompt (exact column, read/write call sites) OR (b) a schema migration (`system_prompt TEXT NULL` on `conversations`, schema version bump, `migrations/` entry). Task 13 consumes this decision.

- [ ] **Step 1:** Read the conversations `CREATE TABLE` + migrations in `ChaChaNotes_DB.py`; list every free/metadata-ish column (the add_conversation docstring mentions assistant/scope/topic metadata). Grep their readers: `grep -rn "<column>" tldw_chatbook/ --include=*.py | grep -v Tests`.
- [ ] **Step 2:** Apply the rule: a column qualifies only if (1) TEXT-typed and nullable, (2) not consumed anywhere with conflicting semantics, (3) round-trips through `update_conversation` with sync triggers intact. If none qualifies, choose the migration.
- [ ] **Step 3:** Write the decision note (chosen seam, exact call sites Task 13 will use, one-paragraph rationale). Commit: `docs(prompts): task-0 persistence decision`.

---

## Phase 1 — Library ▸ Prompts

### Task 1: `count_prompts` seam + rail row `Prompts (N)`

**Files:**
- Modify: `tldw_chatbook/Prompt_Management/local_prompt_service.py`, `tldw_chatbook/Prompt_Management/prompt_scope_service.py`
- Modify: `tldw_chatbook/Library/library_shell_state.py` (Browse section rows — Notes precedent at the `LibraryRailSectionState` builders), `tldw_chatbook/UI/Screens/library_screen.py` (`_list_local_source_snapshot` area: add prompts count fetch; rail row constant `LIBRARY_ROW_BROWSE_PROMPTS = "library-row-browse-prompts"` beside the existing `LIBRARY_ROW_BROWSE_*` constants)
- Test: `Tests/Library/test_library_prompts_seam.py` (new), extend `Tests/UI/test_library_shell.py`

**Interfaces:**
- Produces: `LocalPromptService.count_prompts() -> int` (async; exact count of non-deleted prompts), `PromptScopeService.count_prompts(mode: str = "local") -> int`, rail row id `LIBRARY_ROW_BROWSE_PROMPTS`, snapshot key `"prompts"` in `_local_source_records` carrying `(count, page_records)`.
- Consumes: `PromptsDatabase.list_prompts(page, per_page)` returns `(results, total_pages, page, total_items)` — count = `total_items` with `per_page=1`.

- [ ] **Step 1: Failing seam test** in `Tests/Library/test_library_prompts_seam.py`:

```python
import pytest
from tldw_chatbook.DB.Prompts_DB import PromptsDatabase
from tldw_chatbook.Prompt_Management.local_prompt_service import LocalPromptService

@pytest.mark.asyncio
async def test_count_prompts_counts_non_deleted(tmp_path):
    db = PromptsDatabase(tmp_path / "prompts.db", client_id="test-client")
    svc = LocalPromptService(db)  # match the service's real constructor; adjust if it takes a provider callable
    assert await svc.count_prompts() == 0
    db.add_prompt(name="alpha", author="t", details="d", user_prompt="hello")
    db.add_prompt(name="beta", author="t", details="d", user_prompt="world")
    assert await svc.count_prompts() == 2
    db.soft_delete_prompt("alpha")
    assert await svc.count_prompts() == 1
```

- [ ] **Step 2:** `$PY -m pytest Tests/Library/test_library_prompts_seam.py -v` → FAIL (`count_prompts` missing).
- [ ] **Step 3:** Implement `count_prompts` on `LocalPromptService` (Google docstring; use `list_prompts(page=1, per_page=1)` total_items; run sync DB work exactly the way the service's existing list method does — thread offload if that is its pattern) and a `PromptScopeService.count_prompts(mode="local")` passthrough mirroring how its other read actions route (policy id `prompts.count.local` if the registry requires one — mirror `skills.list.local` registration style).
- [ ] **Step 4:** Test passes; commit `feat(prompts): count_prompts seam`.
- [ ] **Step 5: Failing UI test** in `Tests/UI/test_library_shell.py`: build the Library screen with a fake prompt scope service on the app exposing `count_prompts` → assert the rail renders a row `Prompts (2)` with id `LIBRARY_ROW_BROWSE_PROMPTS` (mirror the existing `Notes (N)` rail test in the same file — copy its harness setup verbatim, rename).
- [ ] **Step 6:** Wire rail row + snapshot: add the Browse row in `library_shell_state.py` after Notes; in `library_screen.py` fetch the count in the same snapshot worker that fills notes/conversations counts (`_list_local_source_snapshot` / its callers), storing under `_local_source_records["prompts"]`. Row click selects the prompts list canvas (canvas kind lands in Task 3 — for now selecting sets `_library_selected_row_id` and shows the existing placeholder-empty canvas path so the row is inert-but-selectable).
- [ ] **Step 7:** UI test passes; full file `$PY -m pytest Tests/UI/test_library_shell.py -q` stays green (240+ baseline). Commit.

### Task 2: Pure state builders `library_prompts_state.py`

**Files:**
- Create: `tldw_chatbook/Library/library_prompts_state.py`
- Test: `Tests/Library/test_library_prompts_state.py`

**Interfaces:**
- Produces (all pure, dataclasses frozen):
  - `PromptListRow(prompt_id: int, name: str, secondary: str)` — `name` raw (canvas escapes at render), `secondary` = `"<author> · <kw1, kw2> · <age>"` parts omitted when empty.
  - `build_prompts_list_state(records, *, query: str, sort: str, now: datetime) -> PromptsListState` with `PromptsListState(rows: tuple[PromptListRow, ...], count: int, sort: str)`; `sort in {"newest", "name"}`; `query` filters case-insensitively over name+keywords.
  - `PromptEditorState(prompt_id, name, author, details, system_prompt, user_prompt, keywords_csv, version, created, modified)` + `build_prompt_editor_state(detail: Mapping) -> PromptEditorState` (maps `fetch_prompt_details` output; keywords list → csv).
  - `classify_prompt_save_error(result_id, message, exc: Exception | None) -> str` returning one of `"name-in-use" | "soft-deleted-name" | "conflict" | "ok" | "error"` — `soft-deleted` substring in message + `result_id is None` → `"soft-deleted-name"`; `ConflictError` → `"conflict"`; unique-name IntegrityError/message → `"name-in-use"`.
- Consumes: record mappings shaped like `fetch_prompt_details` / `list_prompts` rows (keys: `id, name, author, details, system_prompt, user_prompt, keywords, last_modified/created_at, version`).

- [ ] **Step 1: Failing tests** covering: newest sort orders by modified desc; name sort alpha-ci; query matches name and keyword; secondary omits empty author; `classify_prompt_save_error(None, "…soft-deleted…", None) == "soft-deleted-name"`; `classify_prompt_save_error(None, "", ConflictError("x")) == "conflict"` (import the real `ConflictError` from `tldw_chatbook.DB.Prompts_DB` — verify its module of origin first and use that import). Write ~10 focused tests with literal expected tuples.
- [ ] **Step 2:** FAIL run. **Step 3:** Implement the module (~120 lines; reuse the relative-age helper the notes state uses — import it if importable, else copy the exact function with attribution comment). **Step 4:** PASS run. **Step 5:** Commit `feat(prompts): pure list/editor state builders`.

### Task 3: List canvas + screen wiring

**Files:**
- Create: `tldw_chatbook/Widgets/Library/library_prompts_canvas.py` (list part)
- Modify: `tldw_chatbook/UI/Screens/library_screen.py` (canvas kind `"prompts"`, row-press → list, filter/sort/toolbar handlers), `tldw_chatbook/Library/library_shell_state.py` if canvas-kind registry lives there
- Test: `Tests/UI/test_library_prompts_canvas.py` (new)

**Interfaces:**
- Consumes: Task 1 snapshot records + Task 2 `build_prompts_list_state`.
- Produces: widget `LibraryPromptsListCanvas(state: PromptsListState)` with row Button ids `library-prompt-row-<id>`; messages/handler contract: pressing a row calls screen handler `handle_library_prompt_row(prompt_id)`; toolbar ids `library-prompts-sort`, `library-prompts-import`, `library-prompts-export`; filter input id `library-prompts-filter`.

- [ ] **Step 1:** Open `tldw_chatbook/Widgets/Library/library_notes_canvas.py` and copy its list-compose structure (header count line, filter Input, single-row `Horizontal(classes="ds-toolbar")`, row Buttons with escaped labels) as the starting file — keep class names/ids prompts-specific. This is a structural template copy, not shared code; prompts and notes diverge (two-part editor, no sync).
- [ ] **Step 2: Failing UI test**: mount the canvas in a bare test App with a 3-row state → assert 3 row buttons, escaped bracket-title rendering (`[draft] X` appears literally), toolbar children share one row y (behavior/content assertions preferred; the harness lacks app CSS).
- [ ] **Step 3:** Implement; wire `library_screen.py`: canvas kind `"prompts"` rendered when `_library_selected_row_id == LIBRARY_ROW_BROWSE_PROMPTS`; filter (Enter) re-builds state with query; sort button cycles newest↔name. Fresh state from `_local_source_records["prompts"]`.
- [ ] **Step 4:** Tests pass; `test_library_shell.py` green. **Step 5:** Commit `feat(prompts): list canvas`.

### Task 4: Editor canvas, explicit Save, conflict outcomes, delete

**Files:**
- Modify: `tldw_chatbook/Widgets/Library/library_prompts_canvas.py` (editor part), `tldw_chatbook/UI/Screens/library_screen.py`
- Test: extend `Tests/UI/test_library_prompts_canvas.py`, `Tests/Library/test_library_prompts_seam.py`

**Interfaces:**
- Consumes: `LocalPromptService` CRUD (`create_prompt`/`update_prompt`/`soft_delete_prompt` — use the service's real method names, read the file first), Task 2 editor state + `classify_prompt_save_error`.
- Produces: editor ids `library-prompt-name`, `-author`, `-details`, `-system` (TextArea), `-user` (TextArea), `-keywords`; actions row ids `library-prompt-save`, `-insert-console`, `-export`, `-copy`, `-delete`; screen methods `handle_library_prompt_row(prompt_id)` (opens editor), `_save_library_prompt()`; dirty tracking flag `_library_prompt_dirty` participating in the screen's existing `flush_pending_work()` (prompts branch mirrors the notes branch: unsaved → prompt-to-save veto semantics identical).

- [ ] **Step 1: Failing tests** (write all before implementing):
  - open row → editor shows all six fields populated from a real seeded prompts DB;
  - edit name to an existing name → Save → outcome Static id `library-prompt-save-status` shows the exact copy `Name already in use — pick another or open the existing prompt.`;
  - save onto a soft-deleted name → `A deleted prompt holds this name — restore it or choose another.`;
  - stale version (bump version behind the editor's back via a second service call) → conflict bar with Overwrite/Reload buttons (`library-prompt-conflict-overwrite` / `-reload`);
  - nav-away with dirty editor → `flush_pending_work` returns the same veto shape notes uses (assert against the notes test for the exact contract — read `Tests/UI/test_library_shell.py` notes flush tests first and mirror);
  - delete → back to list, count decremented.
- [ ] **Step 2:** FAIL run. **Step 3:** Implement editor compose + save flow: Save button → gather fields → route via scope service; classify outcome with `classify_prompt_save_error`; success → refresh snapshot + status `Saved.`; keywords csv → list. Delete uses soft delete + confirm-free (dim button, single press acceptable — match the notes delete affordance exactly). **Step 4:** PASS. **Step 5:** Commit `feat(prompts): editor with explicit save + conflict outcomes`.

### Task 5: Import + per-prompt export with round-trip

**Files:**
- Modify: `tldw_chatbook/Widgets/Library/library_prompts_canvas.py`, `tldw_chatbook/UI/Screens/library_screen.py`
- Create: `tldw_chatbook/Prompt_Management/prompt_markdown_export.py`
- Test: `Tests/Library/test_prompt_export_roundtrip.py` (new) + canvas tests

**Interfaces:**
- Consumes: `Prompts_Interop.parse_markdown_prompts_from_content`, `import_prompts_from_files` (read signatures first; import skips duplicates — verify: if it overwrites by default, pass/force the non-overwrite mode and count skips).
- Produces: `render_prompt_markdown(detail: Mapping) -> str` emitting EXACTLY the shape `parse_markdown_prompts_from_content` reads (inspect the parser to learn its expected headings/frontmatter and mirror it); screen import flow: `Import…` → path Input row (validate via the same `validate_path_simple` the ingest form uses) → outcome line `N imported · M skipped (duplicate name)`.

- [ ] **Step 1: Failing round-trip test**:

```python
def test_prompt_markdown_export_roundtrips(tmp_path):
    detail = {"name": "Release note", "author": "me", "details": "d",
              "system_prompt": "sys text", "user_prompt": "user text",
              "keywords": ["release", "notes"]}
    text = render_prompt_markdown(detail)
    parsed = parse_markdown_prompts_from_content(text)
    assert len(parsed) == 1
    p = parsed[0]
    assert (p["name"], p["system_prompt"], p["user_prompt"]) == ("Release note", "sys text", "user text")
```

(Adjust parsed-record key names to the parser's real output after reading it — then the assertions are exact.)
- [ ] **Step 2:** FAIL. **Step 3:** Implement exporter (~40 lines) by mirroring the parser's grammar. **Step 4:** PASS. **Step 5:** Wire editor `Export .md` (FileSave-style path like the notes export — reuse its dialog helper; sanitize filename from name) and toolbar `Import…` (path input + worker + outcome line; duplicates counted as skipped). UI tests for both outcomes. **Step 6:** Commit `feat(prompts): import + round-trip markdown export`.

### Task 6: Prompts as a Search source

**Files:**
- Modify: `tldw_chatbook/Library/library_local_rag_search_service.py`, `tldw_chatbook/Library/library_rag_state.py` (source toggle + provenance), `tldw_chatbook/UI/Screens/library_screen.py` (`_open_library_item_by_id` gains `source_type == "prompt"` → prompts editor)
- Modify: `tldw_chatbook/Prompt_Management/Prompts_Interop.py` (or `local_prompt_service.py`) — optional `fts_match_query` pass-through on `search_prompts` mirroring the notes seam added in task-185
- Test: extend `Tests/Library/test_library_local_rag_search_service.py`, `Tests/Library/test_library_rag_state.py`

**Interfaces:**
- Produces: search source key `"prompts"` with toggle label `Prompts (N)`; result rows `source_type="prompt"`, `record_id=<prompt id>`; opening lands in the prompts editor.
- Consumes: Task 1 count seam (toggle count), prompts FTS via `search_prompts(..., fts_match_query=...)`.

- [ ] **Step 1: Failing service test** (extend the real-DB fixture): seed a prompt whose user part contains "feedback loops"; search "feedback loop" with prompts enabled → one prompt hit (plural expansion proof rides task-185's builder); search with prompts deselected → zero prompt rows.
- [ ] **Step 2:** FAIL. **Step 3:** Add the seam: `search_prompts` optional `fts_match_query` (same conditional-forwarding pattern the notes seam uses — read `notes_scope_service.search_notes` for the shape); service queries it alongside notes/media/conversations; state builder adds the ✓ toggle + count and provenance rows. **Step 4:** PASS + `test_library_shell.py` + `Tests/UI/test_product_maturity_gate16_library_search_rag.py` green. **Step 5:** Open-path: `_open_library_item_by_id("prompt", id)` selects prompts rail row + opens editor; UI test. **Step 6:** Commit `feat(prompts): search source with plural expansion parity`.

### Task 7: Personas placeholder retirement + routing + dead code

**Files:**
- Modify: `tldw_chatbook/UI/Screens/personas_screen.py` (remove "prompts" from `MODE_CHIP_ORDER` + `_apply_mode` branch + placeholder copy), `tldw_chatbook/UI/Navigation/screen_registry.py` + `tldw_chatbook/UI/Navigation/shell_destinations.py` (`prompts` alias → library with nav-context selecting `LIBRARY_ROW_BROWSE_PROMPTS`; follow the `notes → library` alias precedent exactly — grep `"notes"` in both files)
- Delete: `tldw_chatbook/Event_Handlers/prompt_ingest_events.py`; remove `CCPPromptHandler` from `tldw_chatbook/UI/CCP_Modules/` exports
- Test: extend `Tests/UI/test_screen_navigation.py` (alias test), personas tests

**Interfaces:** Produces: route alias `prompts → library` + `LIBRARY_NAV_CONTEXT` selection of the prompts row (name the context key exactly as the notes alias does).

- [ ] **Step 1:** `grep -rn "prompt_ingest_events\|CCPPromptHandler" tldw_chatbook Tests` — confirm zero non-definition references; if any exist, STOP and report instead of deleting.
- [ ] **Step 2: Failing nav test**: `NavigateToScreen("prompts")` lands on LibraryScreen with the prompts rail row selected (mirror the notes-alias test in `test_screen_navigation.py` / `test_library_shell.py`).
- [ ] **Step 3:** Implement alias + retirement + deletions. `$PY -c "import tldw_chatbook.app"` clean. **Step 4:** Personas suite + navigation suite green. **Step 5:** Commit `feat(prompts): prompts route to Library; retire personas placeholder + dead code`.

### Task 8: Phase-1 gate — suites + live captures (STOP for user approval)

- [ ] **Step 1:** `$PY -m pytest Tests/Library Tests/UI/test_library_shell.py Tests/UI/test_library_prompts_canvas.py Tests/UI/test_destination_shells.py Tests/UI/test_screen_navigation.py Tests/UI/test_non_obscuring_focus_contract.py -q` → all green.
- [ ] **Step 2:** Serve from the worktree (textual-serve recipe: scratchpad `serve_qa.py`, playwright bundled chromium 2050×1240, route-abort `https://**` only, gate on `body.-first-byte`); seed 4 real prompts via `PromptsDatabase.add_prompt` into the profile's prompts DB (one with `[draft] X [wip]` name, one with a multi-KB user part). Capture: rail `Prompts (4)`, list, editor, name-conflict outcome, import outcome line, search hit with prompts source, `prompts` alias landing. Save to `Docs/superpowers/qa/library-prompts-2026-07/` with a README per QA convention.
- [ ] **Step 3:** Commit QA evidence. **STOP — present captures for user approval before Phase 2.**

---

## Phase 2 — Console: grammar, /prompt, /system

### Task 9: Pure grammar module

**Files:**
- Create: `tldw_chatbook/Chat/console_command_grammar.py`
- Test: `Tests/Chat/test_console_command_grammar.py`

**Interfaces:**
- Produces:

```python
@dataclass(frozen=True)
class ConsoleCommand:
    name: str            # "prompt"
    argument_hint: str   # "[name]"
    handler_id: str      # "insert-prompt"

@dataclass(frozen=True)
class CommandParse:
    kind: str            # "command" | "fallback" | "unknown" | "not-command"
    name: str = ""
    args: str = ""

class ConsoleCommandRegistry:
    def register(self, command: ConsoleCommand) -> None: ...
    def register_fallback_resolver(self, resolver: Callable[[str, str], CommandParse | None]) -> None: ...
    def parse(self, draft_text: str) -> CommandParse: ...
    def available_names(self) -> tuple[str, ...]: ...

def default_console_registry() -> ConsoleCommandRegistry:  # registers /prompt + /system
```

`parse` rules: text not starting with `/` or containing paste-token markers → `not-command`; `/word rest` where word matches a registered name (case-insensitive) → `command`; else each fallback resolver gets `(word, rest)` and may return a parse; else `unknown` with `name=word`.

- [ ] **Step 1: Failing tests** (literal cases): `/prompt release note` → `("command","prompt","release note")`; `/PROMPT x` case-insensitive; `/usr/bin/thing` → unknown name `usr` (note: tokenizer splits on whitespace AND a bare `/usr/bin` is one token — decide: token = chars up to first whitespace, name = token[1:], so name `usr/bin/thing`; assert exactly that); `hello` → not-command; `/system` → command with empty args; fallback resolver claiming `myskill` wins over unknown; resolver returning None falls through.
- [ ] **Step 2:** FAIL. **Step 3:** Implement (~90 lines, no Textual imports). **Step 4:** PASS. **Step 5:** Commit `feat(console): slash-command registry with fallback-resolver hook`.

### Task 10: Composer interception + unknown-command Enter-again

**Files:**
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py` (`_send_console_message_from_visible_action` ~:6473 — parse BEFORE readiness/submit; store `self._console_command_registry = default_console_registry()` on screen init; armed-state field `_console_unknown_send_armed: str | None`), `tldw_chatbook/Widgets/Console/console_composer_bar.py` (only if a draft-text accessor is missing — the grammar needs the plain-text draft and a paste-token presence flag; add read-only helpers if absent)
- Test: `Tests/UI/test_console_command_composer.py` (new)

**Interfaces:**
- Consumes: Task 9 `parse`. Produces: transcript-local hint row (system-style, persist=False — reuse the run-status system-row mechanism from the task-182 error rows; grep `persist=False` in `console_chat_controller.py`) with copy `Unknown command /<name> — available: /prompt, /system. Press Enter again to send as text.`; handler dispatch map `{"insert-prompt": self._console_command_insert_prompt, "apply-system": self._console_command_apply_system}` (methods stubbed here returning a "not wired yet" notify, replaced in Tasks 12/14).

- [ ] **Step 1: Failing UI tests**: (a) draft `/nope x` + Enter → hint row rendered, draft unchanged, nothing sent (controller submit spy not called); (b) second Enter with unmodified draft → submitted as plain text; (c) edit draft between Enters → hint again, not sent; (d) draft with a collapsed-paste token starting with `/` → sent normally (not-command); (e) `/prompt` dispatches the insert-prompt stub.
- [ ] **Step 2:** FAIL. **Step 3:** Implement: in the send action, before readiness gating, `parse(draft_plain_text)`; `not-command` → existing path; `command`/`fallback` → dispatch, consume nothing on failure; `unknown` → if `_console_unknown_send_armed == draft_text` → clear arm + fall through to normal send; else set arm + hint row. Clear arm on any composer change (hook the composer's change event — find its existing changed-message and subscribe). **Step 4:** PASS + `Tests/UI/test_console_native_chat_flow.py` green. **Step 5:** Commit `feat(console): composer command interception with Enter-again escape`.

### Task 11: Prompt picker modal

**Files:**
- Create: `tldw_chatbook/Widgets/Console/console_prompt_picker_modal.py`
- Test: `Tests/UI/test_console_prompt_picker.py`

**Interfaces:**
- Produces: `ConsolePromptPickerModal(mode: str, initial_query: str, prompt_search: Callable[[str], Awaitable[list[Mapping]]])` — `mode in {"insert", "apply-system"}`; dismisses with `Mapping | None` (the chosen prompt detail). Filter Input id `console-prompt-picker-filter`, row Buttons `console-prompt-picker-row-<id>`. In `apply-system` mode rows with empty `system_prompt` render dimmed + suffix `(no system part)` and refuse selection (press → inline reason Static, no dismiss). Empty store → single line `No saved prompts yet — create them in Library ▸ Prompts.` `prompt_search` is FTS-backed (bounded page ≤ 25) and called fresh per filter change (debounce 200ms).
- Consumes: scope-service search from Task 6.

- [ ] **Step 1: Failing tests**: type-to-filter calls `prompt_search` with the query; Enter on highlighted row dismisses with that record; apply-system mode blocks empty-system rows with the reason line; Esc dismisses None; empty store line. Use a fake async `prompt_search`. Escape names in labels (bracket-name literal test).
- [ ] **Step 2:** FAIL. **Step 3:** Implement (ModalScreen, keyboard-first: filter autofocus, ↑/↓ move, Enter select). **Step 4:** PASS. **Step 5:** Commit `feat(console): prompt picker modal`.

### Task 12: `/prompt` handler + insertion + Library Insert-in-Console

**Files:**
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py` (`_console_command_insert_prompt`, insertion helper `_insert_prompt_text_into_composer(text, *, replace: bool)`), `tldw_chatbook/Widgets/Console/console_composer_bar.py` (insertion API: set/append text through the same path paste uses so the collapse threshold applies — find the paste-handling entry point and expose `insert_text_as_paste(text)`), `tldw_chatbook/UI/Screens/library_screen.py` (editor `Insert in Console` → `ChatHandoffPayload`-free direct route: `NavigateToScreen("chat")` + app-level pending-insert field consumed by ChatScreen on mount/resume — mirror how `open_chat_with_handoff` stages, but into the composer, append-not-replace)
- Test: extend `Tests/UI/test_console_command_composer.py`, `Tests/UI/test_library_prompts_canvas.py`

**Interfaces:**
- Consumes: Tasks 9-11; prompt resolution: exact-name ci match via scope service `fetch_prompt_details(name=...)`, else prefix match over a bounded `search_prompts` page; 0/2+ → picker with `initial_query=args`.
- Produces: `/prompt <unique>` replaces draft with the prompt's `user_prompt` via paste semantics; picker selection same; Library insert appends when draft non-empty; blocked-composer Library insert → toast `Finish provider setup to insert prompts.`

- [ ] **Step 1: Failing tests**: unique name replaces draft (short body inline — assert composer text); multi-KB body becomes a collapsed paste token (assert token segment present, not raw text); ambiguous → picker opened with query prefilled (fake picker capture); Library insert with existing draft `abc` → composer contains `abc` + token/text appended; blocked Console → toast, prompt not lost from Library editor.
- [ ] **Step 2:** FAIL. **Step 3:** Implement resolution + insertion + Library route. **Step 4:** PASS + chat-flow suite green. **Step 5:** Commit `feat(console): /prompt insertion with paste semantics + Library insert-in-console`.

### Task 13: System-prompt plumbing + persistence

**Files:**
- Modify: `tldw_chatbook/Chat/console_session_settings.py` (`ConsoleSessionSettings.system_prompt: str | None = None` — flows through existing asdict/whitelist serializers like `source` did), `tldw_chatbook/Chat/console_chat_controller.py` or `chat_screen.py` `_provider_messages_for_session` (prepend `{"role": "system", "content": ...}` when non-empty), persistence per Task 0 decision (metadata column write on apply + read on conversation resume, or migration first), `tldw_chatbook/Chat/console_chat_store.py` (persist/restore call sites)
- Test: `Tests/Chat/test_console_session_settings.py`, `Tests/UI/test_console_session_settings.py`, `Tests/Chat/test_console_chat_controller.py`

**Interfaces:**
- Produces: `system_prompt` on session settings (per session/tab, default None); provider messages prepend it for submit AND regenerate AND continue; persistence: applied prompt survives restart+resume (test with a real ChaChaNotes DB: create conversation, apply, reload store, assert restored).
- Consumes: Task 0 decision note.

- [ ] **Step 1: Failing tests**: settings default None; messages built with a system prompt start with the system message (submit path); regenerate path includes it (extend the existing regenerate message-build test); persistence round-trip per Task 0 seam; serializer whitelist carries the field (mirror the `source`-field test).
- [ ] **Step 2:** FAIL (and if Task 0 chose migration: write the migration + schema-version test first, in this task, before the round-trip test). **Step 3:** Implement. **Step 4:** PASS + `Tests/Chat -k console` green. **Step 5:** Commit `feat(console): real per-session system prompt plumbed into all send paths`.

### Task 14: `/system`, system editor modal, rail preview, palette

**Files:**
- Create: `tldw_chatbook/Widgets/Console/console_system_prompt_modal.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py` (`_console_command_apply_system` real impl; Model rail section `System: <preview>` line — build alongside the model line in the rail sync, `text_wrap="nowrap"`/ellipsis like the task-186 fix; click → modal; palette entries in `tldw_chatbook/UI/console_command_provider.py`: `Insert prompt…`, `Edit system prompt`)
- Test: `Tests/UI/test_console_system_prompt.py` (new), extend palette test if one exists

**Interfaces:**
- Consumes: Tasks 11 (apply-system picker mode), 13 (apply/clear via session settings).
- Produces: modal ids `console-system-prompt-text` (TextArea), buttons `-apply`, `-save-library`, `-clear`, `-cancel`, scope line `Applies to this session.`; rail Static id `console-rail-system-line` with `System: none` dim state; `/system <name>` applies system part (empty-part inline error `Prompt "<name>" has no system part.`); `Save to Library…` routes through Task 4's save flow (name prompt via a small Input row inside the modal; same outcome copy).

- [ ] **Step 1: Failing tests**: bare `/system` opens modal with current text; Apply updates session settings + rail preview text; Clear → `System: none`; `/system <name>` with system part applies + preview updates; empty system part → inline error, settings unchanged; Save to Library duplicate name shows the Task 4 copy; palette lists both entries.
- [ ] **Step 2:** FAIL. **Step 3:** Implement modal + rail line + handler + palette. **Step 4:** PASS + `Tests/UI/test_console_persistent_rails.py`, `test_console_session_settings.py`, workbench-contract suite green. **Step 5:** Commit `feat(console): /system + system prompt modal + rail preview`.

### Task 15: Phase-2 gate — suites + live captures (STOP for user approval)

- [x] **Step 1:** Broad sweep → GREEN (1844 passed / 1 load-flake `test_library_shell_search_history_row_reruns_query`, passes in isolation; + 229 console re-run + 86/127 fix-covering suites).
- [x] **Step 2:** Live verification against llama.cpp @127.0.0.1:9099 on a fresh profile (7 seeded two-part prompts): captured `/prompt` picker + insertion (short inline + collapsed-paste), unknown-command hint + Enter-again send, `/system` apply + `System:` rail preview, system editor modal, empty-system-part inline error, and one real send whose **provider request body was captured off the wire** (`messages` = `[system, user]`, system role present) via the identical `chat_api_call("llama_cpp", …)` path through a logging proxy — request-capture method used (stated in README; the served UI worker resolves its endpoint independently of the runtime proxy swap, so the on-wire capture was taken headlessly through the same handler). Captures → `Docs/superpowers/qa/console-prompts-2026-07/` + README.
- [ ] **Step 3:** Update spec/backlog: mark the plan checkboxes, commit evidence. **STOP — present captures for user approval; PR only after approval.** ← AWAITING USER APPROVAL

---

## Self-review notes (kept for executors)

- Service method names in Tasks 1/4/6 (`LocalPromptService` constructor, CRUD names, `search_prompts` signature) MUST be read from the source before writing tests — the plan pins behavior and shapes, not unverified names; where a name differs, keep the plan's semantics and use the real name.
- The five Console suites were green at branch base; any new failure in them is yours.
- chat_screen.py is high-churn: rebase onto origin/dev before Phase 2 and re-run the chat-flow suite first.
