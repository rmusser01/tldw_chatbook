# Console Prompt Draft Shelf Implementation Plan

> **For Codex:** Execute this plan task by task with strict red-green-refactor
> cycles. Keep the solution inside the existing Prompt database, scope service,
> Prompt Workbench, and Console controller boundaries.

**Goal:** Add a local, explicitly managed shelf for unsent Console drafts and
connect it to the existing Prompt Workbench and Library Prompt collection.

**Architecture:** Prompts database schema v5 owns a non-synced
`LocalPromptDrafts` table introduced by the v4-to-v5 migration.
`LocalPromptService` owns its CRUD operations, and `PromptScopeService`
provides the local-only normalized contract. `ConsolePromptsModal` adds Draft
Shelf as a source and a plain-text editor mode, while
`ConsolePromptsController` owns exact composer capture, keep/clear, caret
insertion, and session persistence.

**Tech Stack:** Python 3.12, SQLite, Textual 8.x, pytest.

**Spec:** `Docs/superpowers/specs/2026-09-25-console-prompt-draft-shelf-design.md`

**ADR required:** yes
**ADR path:** `backlog/decisions/184-local-console-prompt-draft-shelf.md`
**Reason:** This adds a durable local storage owner and a local-only service
contract that must remain outside Prompt sync/export/server behavior.

## Global Constraints

- Preserve unrelated changes and the existing Prompt improvement/provider path.
- Do not add a Ctrl+S or other terminal-convention binding (ADR-031).
- Never evict shelf entries automatically; cap at 100 and refuse the 101st.
- Never clear the composer before persistence succeeds.
- Never remove a shelf entry as a side effect of Library promotion.
- Reuse PromptScopeService for Prompt/collection writes; do not call provider or
  database adapters from widgets.
- Use design-system tokens and existing Prompt Workbench component patterns.
- Run targeted tests only unless the user explicitly requests a full sweep.

## Review Focus

- Exact collapsed-paste content reaches storage, while save failures preserve
  the composer.
- Capacity enforcement is atomic under concurrent writers.
- Draft rows cannot enter sync log, Prompt FTS, Chatbook export, or server APIs.
- Stale browse/detail/edit completions cannot replace newer state.
- Keyboard focus, dirty guards, and Escape restore behavior remain coherent.
- Promotion reports a successful Prompt create honestly if collection
  assignment fails afterward.

### Task 1: Local draft-shelf contract

**Files:**

- Modify: `tldw_chatbook/Prompt_Management/prompt_scope_service.py`
- Modify: `tldw_chatbook/DB/Prompts_DB.py`
- Create: `Tests/Prompt_Management/test_prompt_draft_shelf.py`
- Modify: `Tests/Prompts_DB/test_prompts_db_retained_history.py`

1. Write failing service tests for create/detail, exact Unicode and multiline
   content, newest-first bounded paging/search, update version conflicts, hard
   delete, local-only routing, and the 101st-create refusal.
2. Add the minimal `LocalPromptDrafts` schema through the Prompts database
   v4-to-v5 migration, then add typed capacity/conflict outcomes and CRUD in
   `LocalPromptService`.
3. Add normalized local-only draft methods to `PromptScopeService`; use an
   immediate transaction for count-plus-insert and parameterized SQL throughout.
4. Run the new service test file and the affected PromptScopeService tests.

### Task 2: Save from composer without data loss

**Files:**

- Modify: `tldw_chatbook/Widgets/Console/console_composer_menu_modal.py`
- Create: `tldw_chatbook/Widgets/Console/console_prompt_draft_save_dialog.py`
- Modify: `tldw_chatbook/UI/Console_Modules/prompts.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Modify: `tldw_chatbook/UI/console_command_provider.py`
- Create: `Tests/Console/test_prompt_draft_save.py`

1. Write failing tests for the menu/palette entry, exact canonical draft
   capture, Save and keep, Save and clear only after success, failure
   preservation, and the full-shelf recovery path.
2. Add one stable menu action and one Console action method shared by Menu and
   command palette.
3. Implement the compact keep/clear dialog and controller persistence flow.
4. Persist the cleared session draft only after a successful Save and clear.
5. Run the new tests plus existing composer-menu and composer snapshot tests in
   isolated targeted processes.

### Task 3: Draft Shelf inside Prompt Workbench

**Files:**

- Modify: `tldw_chatbook/Widgets/Console/console_prompts_state.py`
- Modify: `tldw_chatbook/Widgets/Console/console_prompts_browse.py`
- Create: `tldw_chatbook/Widgets/Console/console_prompt_draft_editor.py`
- Modify: `tldw_chatbook/Widgets/Console/console_prompts_modal.py`
- Modify: `tldw_chatbook/UI/Console_Modules/prompts.py`
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss`
- Modify (generated): `tldw_chatbook/css/tldw_cli_modular.tcss` via
  `python tldw_chatbook/css/build_css.py`
- Create: `Tests/Console/test_prompt_draft_workbench.py`

1. Write failing pure-state/widget tests for Draft Shelf source switching,
   bounded paging/search, stale-worker rejection, filter focus, arrow/Enter
   row selection, explicit empty/error/full states, and responsive action
   reachability.
2. Add Draft Shelf row rendering and a plain-text draft editor within the
   existing workbench shell.
3. Implement versioned Update, two-press Delete, and insert-at-current-caret
   through injected controller callbacks. Reuse the existing dirty guard and
   composer-focus restoration.
4. Add promotion inputs for required Prompt name and optional existing local
   collection; create through PromptScopeService, then assign membership, and
   retain the shelf entry.
5. Build CSS, run token governance, and run the new widget tests plus affected
   Prompt Workbench tests in process isolation where required by the known dev
   profile-lifetime baseline.

### Task 4: Documentation, live UAT, and closeout

**Files:**

- Modify: `Docs/User_Guide/console/chat-basics.md`
- Modify: `Docs/User_Guide/library/prompts.md`
- Modify: `backlog/decisions/README.md`
- Modify: `backlog/decisions/184-local-console-prompt-draft-shelf.md`
- Modify: `backlog/tasks/task-18930 - Console-prompt-stash-browser-save-browse-and-edit-partial-drafts-against-the-Library-prompt-collection.md`

1. Document save/keep/clear, the 100-entry block, Draft Shelf editing,
   insertion, deletion, and Library promotion.
2. Run targeted service, widget, controller, PromptScopeService, database
   migration/collection, CSS governance, and lint checks.
3. Run a real Console UAT at narrow and wide sizes covering first-time and
   power-user flows, collapsed paste preservation, focus, full shelf, deletion,
   promotion, and collection assignment; revise until the approved workflow is
   coherent.
4. Perform a self-review and a fresh code review, resolve actionable findings,
   re-run verification, accept ADR-184, check every acceptance criterion, add
   concise implementation notes, and mark TASK-18930 Done.
