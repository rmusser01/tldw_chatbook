# Older Roleplay Conversations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a keyboard user load and open every saved local conversation for the selected Roleplay character in stable batches of 20, with honest loading, retry, empty, and exhausted states.

**Architecture:** Extend the existing local `CharactersRAGDB` listing seam with composite seek pagination, keep one memory-only browse session in `PersonasConversationsController`, and let `PersonasInspectorPane` append ordinary conversation rows while replacing only its special tail row. Reuse the existing preview and Resume/draft/Library actions unchanged.

**Tech Stack:** Python 3.11+, SQLite, Textual 8.x, pytest/Pilot, generated Textual CSS bundles.

---

## Working rules

- Use @superpowers:test-driven-development for every production behavior change.
- Use @superpowers:systematic-debugging before changing code in response to any unexpected failure.
- Use @ponytail in full mode: prefer the existing DB/controller/inspector seams, one typed tail message, and no new dependency, schema migration, index, setting, shortcut, or general pagination framework.
- Use @textual-tui for ListView focus, selection, async worker, and layout behavior.
- Use @superpowers:verification-before-completion before any completion, commit, PR, or merge claim.
- Run only the targeted checks listed here. Do not run the full repository test suite unless the user separately opts in.
- Do not hand-edit `tldw_chatbook/css/widget_defaults_self.tcss` or `tldw_chatbook/css/widget_defaults_scoped.tcss`; regenerate them from `BUNDLED_CSS`.

## Task 0: Record the implementation contract

**Files:**

- Modify: `backlog/tasks/task-22453 - Make-older-local-character-conversations-discoverable-in-Roleplay.md`
- Reference: `Docs/superpowers/specs/2026-08-27-task-22453-older-roleplay-conversations-design.md`
- Reference: `Docs/superpowers/plans/2026-08-27-task-22453-older-roleplay-conversations-implementation.md`

- [ ] **Step 1: Confirm the task is in progress before production edits**

Run:

```bash
sed -n '1,160p' 'backlog/tasks/task-22453 - Make-older-local-character-conversations-discoverable-in-Roleplay.md'
```

Expected: `status: In Progress`, assignee `@codex`, and all four approved acceptance criteria are present.

- [ ] **Step 2: Verify the recorded implementation plan**

The task file already records this plan because planning happens before production edits. Compare its section with the canonical text below and update it only if it has drifted; do not add a second `## Implementation Plan` heading.

```markdown
## Implementation Plan

1. Add deterministic `(last_modified, id)` seek pagination to the existing local character-conversation DB query, preserving legacy offset callers.
2. Add presentation-only conversation tail states and append behavior to the Roleplay inspector.
3. Orchestrate 21-record sentinel reads, retry, deduplication, and stale-result ownership in the conversations controller and screen.
4. Verify database, keyboard, focus, layout, preview-action parity, and isolated live behavior with targeted checks.

ADR required: no
ADR path: N/A
Reason: This is a routine extension of the existing local Roleplay discovery query and inspector list; it does not change storage, ownership, synchronization, security, or service boundaries.

Detailed plan: [2026-08-27-task-22453-older-roleplay-conversations-implementation.md](../../Docs/superpowers/plans/2026-08-27-task-22453-older-roleplay-conversations-implementation.md)
```

- [ ] **Step 3: Confirm scope before implementation**

Run:

```bash
git diff --check
git status --short
```

Expected: only the approved design, this plan, and the Backlog task documentation are changed; no production code is modified yet.

- [ ] **Step 4: Commit the planning record**

```bash
git add Docs/superpowers/plans/2026-08-27-task-22453-older-roleplay-conversations-implementation.md 'backlog/tasks/task-22453 - Make-older-local-character-conversations-discoverable-in-Roleplay.md'
git commit -m "docs: plan older roleplay conversation discovery"
```

## Task 1: Add the database seek-pagination contract

**Files:**

- Create: `Tests/DB/test_character_conversation_seek_pagination.py`
- Modify: `tldw_chatbook/DB/ChaChaNotes_DB.py:9530`

- [ ] **Step 1: Write deterministic ordering and boundary tests**

Create a real `CharactersRAGDB(tmp_path / "chacha.db", client_id="seek-test")` fixture and close it in fixture teardown. Seed global, non-deleted conversations for the default local character with explicit text IDs, then set `last_modified` through a parameterized transaction so the order is deterministic.

Add tests proving:

1. the first read orders by `last_modified DESC, id DESC`, including tied timestamps;
2. a second read using the final visible row's timestamp and ID starts strictly after that row and has no overlap;
3. a legacy positional call `(character_id, limit, offset)` still returns the expected slice;
4. another character, non-global scope, and soft-deleted records stay filtered out.

- [ ] **Step 2: Write mutation and validation tests**

Add tests proving:

1. inserting a newer conversation between reads does not duplicate or displace unchanged older rows;
2. deleting a row from the already traversed page does not skip a remaining older row;
3. supplying only one cursor component raises `InputError` before `execute_query` is called;
4. supplying a complete seek cursor with a nonzero offset raises `InputError` before `execute_query` is called.

For the fail-fast assertions, monkeypatch `database.execute_query` to raise `AssertionError("SQL should not run")`; the expected exception must remain `InputError`.

- [ ] **Step 3: Run the new tests and observe the expected failure**

Run:

```bash
python3 -m pytest Tests/DB/test_character_conversation_seek_pagination.py -q
```

Expected: failures because `get_conversations_for_character()` has no seek cursor and ties are not ordered by ID.

- [ ] **Step 4: Implement the smallest compatible query extension**

Change the public signature to retain all existing positional parameters and add two keyword-only values:

```python
def get_conversations_for_character(
    self,
    character_id: int,
    limit: int = 50,
    offset: int = 0,
    *,
    before_last_modified: str | None = None,
    before_id: str | None = None,
) -> list[dict[str, Any]]:
```

Before starting metrics or SQL:

- raise `InputError` unless both cursor values are supplied or both are omitted;
- raise `InputError` when a complete seek cursor is combined with nonzero `offset`.

Use these two query shapes only:

```sql
-- Existing offset mode, now deterministic for tied timestamps.
WHERE character_id = ? AND deleted = 0 AND scope_type = 'global'
ORDER BY last_modified DESC, id DESC
LIMIT ? OFFSET ?
```

```sql
-- Seek mode.
WHERE character_id = ? AND deleted = 0 AND scope_type = 'global'
  AND (
    last_modified < ?
    OR (last_modified = ? AND id < ?)
  )
ORDER BY last_modified DESC, id DESC
LIMIT ?
```

Keep the existing return type, metrics, contextual logging, and propagated `CharactersRAGDBError` behavior. Update the docstring with the cursor contract and `InputError` cases. Do not add an index or migration.

- [ ] **Step 5: Run the DB tests green**

Run:

```bash
python3 -m pytest Tests/DB/test_character_conversation_seek_pagination.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit the DB contract**

```bash
git add tldw_chatbook/DB/ChaChaNotes_DB.py Tests/DB/test_character_conversation_seek_pagination.py
git commit -m "feat(roleplay): add character conversation seek pagination"
```

## Task 2: Add inspector tail states and append-only rendering

**Files:**

- Modify: `tldw_chatbook/Widgets/Persona_Widgets/personas_pane_messages.py:74`
- Modify: `tldw_chatbook/Widgets/Persona_Widgets/personas_inspector_pane.py:496`
- Modify: `Tests/UI/test_personas_inspector_pane.py:959`
- Regenerate: `tldw_chatbook/css/widget_defaults_self.tcss`
- Regenerate: `tldw_chatbook/css/widget_defaults_scoped.tcss`

- [ ] **Step 1: Write message and tail-state tests**

Add `OlderConversationsRequested` to the inspector test imports and add Pilot tests proving:

1. a successful initial page with `has_more=True` renders ordinary rows followed by **Load 20 older conversations**;
2. focusing the ListView, highlighting that tail, and pressing Enter posts exactly one `OlderConversationsRequested` and no `ConversationRowSelected`;
3. append loading changes only the tail to **Loading older conversations...**, keeps it highlightable, and repeated Enter posts nothing;
4. an initial failure renders the two visible lines **Load failed.** and **Retry conversations** and Enter posts the typed request;
5. an append failure preserves existing row widgets, renders **Retry older conversations**, and Enter posts the typed request;
6. empty and exhausted copy are distinct, readable, and inert.

Drive keyboard behavior with `pilot.press("enter")`; do not call the handler directly.

- [ ] **Step 2: Write widget-identity and highlight tests**

Add tests that retain the first mounted conversation `ListItem` object, append another page, and assert:

1. the original object is still mounted and is the same object;
2. the appended rows post their real conversation IDs through `ConversationRowSelected`;
3. when the focused ListView still highlights the exact loading tail, success highlights the first newly appended row while the ListView retains focus;
4. when focus moved elsewhere, completion changes neither focus nor highlight;
5. when the user highlighted another conversation row, completion preserves that highlight;
6. when no new row is accepted, the inspector does not try to advance to a nonexistent row.

- [ ] **Step 3: Write the production-style narrow-layout test**

Using the existing production `tldw_cli_modular.tcss` harness, render the initial and append retry tails at narrow and standard supported widths. Assert the special tail has at least two lines of height and that **Retry conversations** / **Retry older conversations** is present in the rendered Static. Keep ordinary conversation title rows one line with ellipsis.

- [ ] **Step 4: Run the inspector tests and observe the expected failures**

Run:

```bash
python3 -m pytest Tests/UI/test_personas_inspector_pane.py -q
```

Expected: the new message, tail APIs, append behavior, and wrapping style do not exist yet.

- [ ] **Step 5: Add one dedicated typed action message**

In `personas_pane_messages.py`, add one parameterless `OlderConversationsRequested(Message)` class. It represents Load and Retry activation; the controller's current phase and retained cursor determine which exact request runs. Do not encode Load/Retry as synthetic conversation IDs.

- [ ] **Step 6: Implement presentation-only tail ownership**

In `PersonasInspectorPane`, retain the existing `_conversation_lookup` for durable rows and add only the minimal tail state needed to identify the exact mounted tail and whether it is actionable.

Evolve the inspector API as follows:

- `show_conversations_loading()` clears the list and shows disabled **Loading conversations...**;
- `show_conversations(rows, *, empty_copy=None, has_more: bool | None = None)` remains compatible with silent clears when `has_more is None`; when pagination state is supplied it renders empty, Load, or exhausted state as appropriate;
- `show_older_conversations_loading()` replaces the current tail with an enabled-but-inert loading tail without rebuilding durable rows;
- `show_conversations_failure(*, initial: bool)` either replaces the initial list or only the append tail with the appropriate two-line Retry state;
- `append_conversations(rows, *, has_more: bool)` removes/replaces only the tail and mounts ordinary rows using the same row construction and lookup path as the initial page.

Factor row construction into one small private helper so initial and appended rows cannot diverge. Resolve duplicate DOM IDs with the current suffix logic, but durable ID deduplication remains the controller's responsibility.

In the ListView Selected handler:

- post `ConversationRowSelected` only for IDs in `_conversation_lookup`;
- post `OlderConversationsRequested` only when the selected object is the exact current tail and that tail is actionable;
- do nothing for busy, empty, and exhausted tails.

When swapping Load for Loading, preserve the tail index only if the focused ListView highlighted the old tail. On append completion, inspect current focus and `highlighted_child` immediately before replacement; advance to the first new row only if it is still the exact loading tail. Never call `focus()` on completion.

- [ ] **Step 7: Add source CSS and regenerate bundles**

Add a dedicated conversation-tail class to `PersonasInspectorPane.BUNDLED_CSS` that permits automatic/two-line height and wrapping, while leaving `.personas-conversation-row` at one line.

Run:

```bash
python3 tldw_chatbook/css/build_css.py
python3 tldw_chatbook/css/check_bundle_sync.py
```

Expected: bundle generation succeeds and the sync check exits 0. Inspect the generated diff and keep only changes derived from the inspector source CSS.

- [ ] **Step 8: Run the inspector tests green**

Run:

```bash
python3 -m pytest Tests/UI/test_personas_inspector_pane.py -q
```

Expected: all inspector tests pass.

- [ ] **Step 9: Commit the inspector contract**

```bash
git add tldw_chatbook/Widgets/Persona_Widgets/personas_pane_messages.py tldw_chatbook/Widgets/Persona_Widgets/personas_inspector_pane.py Tests/UI/test_personas_inspector_pane.py tldw_chatbook/css/widget_defaults_self.tcss tldw_chatbook/css/widget_defaults_scoped.tcss
git commit -m "feat(roleplay): add older conversation list states"
```

## Task 3: Orchestrate stable page loading in Roleplay

**Files:**

- Modify: `tldw_chatbook/UI/Persona_Modules/personas_conversations_controller.py:64`
- Modify: `tldw_chatbook/UI/Screens/personas_screen.py:261`
- Modify: `tldw_chatbook/UI/Screens/personas_screen.py:6331`
- Modify: `Tests/UI/test_personas_workbench.py:3684`

- [ ] **Step 1: Replace list-wrapper stubs with a cursor-aware DB test double**

Before adding new behavior tests, update `Tests/UI/test_personas_workbench.py` so conversation-list tests stub `PersonasScreen._character_db()` with a small test double exposing:

```python
get_conversations_for_character(
    character_id,
    limit=50,
    offset=0,
    *,
    before_last_modified=None,
    before_id=None,
)
```

The double should record calls and accept a callable/page queue so tests can gate, fail, and retry exact boundaries. Continue stubbing `retrieve_conversation_messages_for_ui` separately for preview tests.

Replace every `monkeypatch` of `conversations_controller_module.list_character_conversations` in this file; after the controller moves to the DB seam there must be no stale test dependency on the exception-swallowing helper.

Run:

```bash
rg -n "list_character_conversations" Tests/UI/test_personas_workbench.py
```

Expected: no matches.

- [ ] **Step 2: Write first-page and append tests**

In `TestConversationsPanel`, add controlled pages with IDs and timestamps and prove:

1. the first DB call requests 21 records with no cursor;
2. 21 returned records render the first 20 plus the Load tail;
3. keyboard activation requests the next page with the twentieth visible row's exact `(last_modified, id)` boundary;
4. the sentinel is not rendered on the first page and becomes the first row returned/rendered by the next query;
5. successful append preserves the open preview and its action buttons;
6. selecting the oldest appended row enters the same preview and exposes **Resume chat**, **Send to Console draft**, and **Open in Library**.

- [ ] **Step 3: Write retry, duplicate-dispatch, and exhaustion tests**

Using threading events plus the existing worker completion helpers, prove:

1. Enter changes Load to append loading before the DB gate is released;
2. repeated Enter while gated results in one DB call;
3. an append exception keeps all rows and the cursor, shows Retry, and a keyboard retry sends the identical cursor;
4. an initial exception shows Retry rather than **No saved conversations.**;
5. zero records show the empty state;
6. a successful page of at most 20 records shows **All conversations shown.**;
7. a duplicate durable ID in a malformed later page is not appended a second time.

Use exact events/`workers.wait_for_complete()` for worker progress; do not use a fixed pause as proof that a background request finished.

- [ ] **Step 4: Write stale ownership and focus tests**

Gate a page read and prove that its continuation is ignored after each of:

1. selecting another local character;
2. switching away from Characters mode;
3. resetting and starting a newer list attempt for the same character.

Also prove that append completion does not focus the ListView when focus moved to the preview or another control, and does not change a different highlighted conversation row.

- [ ] **Step 5: Run the workbench tests and observe the expected failures**

Run:

```bash
python3 -m pytest Tests/UI/test_personas_workbench.py::TestConversationsPanel -q
```

Expected: new page orchestration and the screen's typed-message handler do not exist yet.

- [ ] **Step 6: Add bounded browse-session state**

In `PersonasConversationsController`, remove the `list_character_conversations` import and add a page-size constant of 20. Keep only memory-local state:

- selected list character ID;
- ordered durable row map plus loaded ID set;
- next `(last_modified, id)` cursor;
- `has_more` flag;
- current list phase (`initial-loading`, `ready`, `append-loading`, `initial-retry`, or `append-retry`); and
- exact attempt object.

`reset()` must invalidate the active list attempt and clear all browse state in addition to its existing preview cleanup.

- [ ] **Step 7: Implement the direct 21-record read path**

Make `load_conversations(character_id)` reset the browse session for that character and start a first page. Add one public controller method for the typed Load/Retry request; it starts work only when the current phase is actionable and no request owns the session.

Each worker must:

1. capture character ID, cursor, initial/append mode, and a unique attempt object;
2. call `self.screen._character_db().get_conversations_for_character(...)` directly with `limit=21` and the cursor keywords only for later pages;
3. propagate the distinction between exception and successful empty data to separate UI continuations;
4. convert at most the first 20 valid records into `(id, title, last_modified)` rows;
5. use the 21st record only to set `has_more`;
6. derive the next cursor from the last accepted visible row;
7. reject any ID already accepted by the browse session before append.

If a malformed page contains records but accepts no new durable row, log a warning and terminate with exhausted state rather than offering an infinite Load loop.

- [ ] **Step 8: Guard every UI continuation**

Centralize the ownership predicate. Apply success or failure only when:

- the screen is mounted in Characters mode;
- the selected entity is the same local character;
- the retained cursor equals the captured boundary; and
- the controller's active attempt is the exact captured object.

Invalidate the attempt before rendering the result. On append failure, preserve rows, loaded IDs, and cursor; change only the phase/tail. On success, update state once, then call the inspector's initial or append API. Do not emit a duplicate toast.

- [ ] **Step 9: Add the thin screen message handler**

Import `OlderConversationsRequested` in `personas_screen.py` and add one handler next to `_handle_conversation_row_selected`. Stop the message, guard Characters/local-character mode, and delegate to the controller's public page-request method. Do not route through the unsaved-edit guard: loading more metadata does not discard edits or open/close a preview.

- [ ] **Step 10: Run the focused workbench class green**

Run:

```bash
python3 -m pytest Tests/UI/test_personas_workbench.py::TestConversationsPanel -q
```

Expected: all conversation-panel tests pass.

- [ ] **Step 11: Run the whole modified workbench test module**

Run:

```bash
python3 -m pytest Tests/UI/test_personas_workbench.py -q
```

Expected: all tests pass, including fixtures outside `TestConversationsPanel` that previously replaced the legacy listing helper.

- [ ] **Step 12: Commit the controller integration**

```bash
git add tldw_chatbook/UI/Persona_Modules/personas_conversations_controller.py tldw_chatbook/UI/Screens/personas_screen.py Tests/UI/test_personas_workbench.py
git commit -m "feat(roleplay): load older character conversations"
```

## Task 4: Verify the complete feature and close the Backlog task

**Files:**

- Verify: all modified production and test files
- Modify: `backlog/tasks/task-22453 - Make-older-local-character-conversations-discoverable-in-Roleplay.md`
- Modify only if an actual reusable incident occurred: `backlog/docs/lessons-testing-evidence.md` or `backlog/docs/lessons-live-verification.md`

- [ ] **Step 1: Run the complete targeted automated matrix**

Run each command independently and retain its output:

```bash
python3 -m pytest Tests/DB/test_character_conversation_seek_pagination.py -q
python3 -m pytest Tests/UI/test_personas_inspector_pane.py -q
python3 -m pytest Tests/UI/test_personas_workbench.py -q
python3 tldw_chatbook/css/check_bundle_sync.py
python3 -m ruff check tldw_chatbook/DB/ChaChaNotes_DB.py tldw_chatbook/UI/Persona_Modules/personas_conversations_controller.py tldw_chatbook/UI/Screens/personas_screen.py tldw_chatbook/Widgets/Persona_Widgets/personas_pane_messages.py tldw_chatbook/Widgets/Persona_Widgets/personas_inspector_pane.py Tests/DB/test_character_conversation_seek_pagination.py Tests/UI/test_personas_inspector_pane.py Tests/UI/test_personas_workbench.py
git diff --check origin/dev...HEAD
```

Expected: every command exits 0. If `ruff` is not installed in the active environment, use the repository's installed lint entry point and record the exact substitute; do not claim lint passed from absence of the tool.

- [ ] **Step 2: Self-review the branch against the approved boundaries**

Run:

```bash
git diff --stat origin/dev...HEAD
git diff origin/dev...HEAD -- tldw_chatbook/DB/ChaChaNotes_DB.py tldw_chatbook/UI/Persona_Modules/personas_conversations_controller.py tldw_chatbook/UI/Screens/personas_screen.py tldw_chatbook/Widgets/Persona_Widgets/personas_pane_messages.py tldw_chatbook/Widgets/Persona_Widgets/personas_inspector_pane.py
rg -n "list_character_conversations" tldw_chatbook/UI/Persona_Modules/personas_conversations_controller.py Tests/UI/test_personas_workbench.py
```

Expected: no search matches; no schema/index/config/search/filter/prefetch/persistence/shortcut changes; the existing preview, Resume, draft, and Library implementations are unchanged.

- [ ] **Step 3: Perform isolated live keyboard acceptance**

Create an explicit scratch config outside the repository with `[paths].data_dir` pointing to the same scratch root and with model-catalog networking disabled. Set `TLDW_TEST_MODE=1`, `XDG_CONFIG_HOME`, `XDG_DATA_HOME`, `XDG_CACHE_HOME`, and `TLDW_CONFIG_PATH` to that scratch profile before importing or launching the app. Do not redirect stderr; Textual renders there.

Seed the scratch `CharactersRAGDB` through its public methods with one local character and at least 45 global, non-deleted conversations whose titles make their order visible. Fingerprint the real config, real data database, worktree, and generated CSS before launch.

Launch:

```bash
python3 -m tldw_chatbook.app
```

In the real TUI:

1. open Roleplay and select the seeded character;
2. tab/focus into Conversations and use arrows plus Enter to activate **Load 20 older conversations** twice;
3. confirm each page appends, the final tail is **All conversations shown.**, and focus/highlight behavior matches the design;
4. select the oldest row and confirm the read-only preview exposes **Resume chat**, **Send to Console draft**, and **Open in Library**;
5. quit normally.

Afterward, compare the fingerprints. Expected: the real profile/data and tracked worktree/CSS are byte-identical to their pre-launch state. Keep only bounded, path-free evidence; do not commit the scratch profile or transcripts.

- [ ] **Step 4: Update task completion records**

Only after the targeted matrix and live acceptance pass:

1. check all four acceptance criteria in the task file;
2. add a concise `## Implementation Notes` section covering the seek query, controller attempt/cursor ownership, append-only inspector tail, tests, generated CSS, and the no-ADR decision;
3. add `ADR required: no`, `ADR path: N/A`, and the same boundary reason to the notes;
4. add a lessons entry only if implementation exposed a real, reusable incident; do not invent one;
5. change the task frontmatter status directly from `In Progress` to `Done`.

Do not address this five-digit task through `backlog task <id>` or `backlog task edit <id>`; Backlog CLI 1.44.0 can silently create a malformed `task-task- - .md`. Edit the source task file with `apply_patch`, then run:

```bash
sed -n '1,240p' 'backlog/tasks/task-22453 - Make-older-local-character-conversations-discoverable-in-Roleplay.md'
git status --short backlog/
test ! -e 'backlog/tasks/task-task- - .md'
```

Expected: status `Done`, all acceptance criteria checked, Implementation Plan and Implementation Notes present.

- [ ] **Step 5: Commit task closeout**

```bash
git add 'backlog/tasks/task-22453 - Make-older-local-character-conversations-discoverable-in-Roleplay.md' backlog/docs
git commit -m "docs: complete older roleplay conversation task"
```

If `backlog/docs` has no intentional lesson change, stage only the task file.

- [ ] **Step 6: Run final verification from a clean worktree**

Run:

```bash
git diff --check origin/dev...HEAD
git status --short
git log --oneline origin/dev..HEAD
```

Expected: the final diff check exits 0, status is clean, and the planning, DB, inspector, controller, and closeout commits are present. Invoke @superpowers:requesting-code-review before integration, then use @superpowers:finishing-a-development-branch for the user-approved PR/merge path.
