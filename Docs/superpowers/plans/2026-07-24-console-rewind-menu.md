# Console `/rewind` Menu — Implementation Plan (SP2)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Ship `/rewind` in the native Console: a menu over the conversation's prior user prompts with **Restore to here** (non-destructive tree navigation + composer refill) and **Summarize up to here** (boundary summary that compacts the provider context while the transcript stays fully visible).

**Architecture:** Per the approved spec `Docs/superpowers/specs/2026-07-24-console-rewind-menu-design.md` (decisions D1–D3 locked; spec-review fixes applied — READ IT FIRST, it is the contract). Restore is pure `set_active_leaf` navigation (SP1 primitives) + the `/prompt`-style composer refill. Summarize stores `context_summary` + `summary_boundary_message_id` as local-only conversation columns (v24→v25, the `active_leaf_message_id` no-sync pattern), generated via the session's resolved provider, applied at the dispatch choke point ONLY when the boundary message is present in the payload, and surfaced in the transcript as a render-derived banner (never a transcript node).

**Tech Stack:** Python ≥3.11, Textual, SQLite (ChaChaNotes), pytest.

## Global Constraints

- **Spec is the contract** — especially its spec-review fixes: render-derived banner (NEVER a SYSTEM/any transcript node); compaction only-when-boundary-in-payload (future-info-leak rule) hooked at the dispatch choke point; restore parent by **id lookup** in `active_path_message_ids` (never view-positional); restore-to-empty documented limitation (do not engineer around).
- **Migration:** ChaChaNotes columns `context_summary TEXT` + `summary_boundary_message_id TEXT` — **re-verify `_CURRENT_SCHEMA_VERSION` on latest dev at implementation AND merge time** (three collisions in this program so far; the number below assumes v24→v25 but the actual FROM version is whatever dev holds). Migration-only (CREATE TABLE frozen at v4), NO `conversations_sync_*` trigger redefinition, bare-UPDATE setters that never bump `version`/`last_modified` (copy the `active_leaf_message_id` pattern verbatim: guarded ALTER method + `_MIGRATE_*_SQL` version-bump-only constant + `.sql` reference file + `migration_steps` registration).
- **Summarization call:** session's resolved provider via the Console gateway, non-streaming, off-thread, exclusive `console-run` worker group; failure = no-op + notify (never partial state). New editable `Internal_Prompts` entry (follow the registry pattern; register catalog + default).
- **Blocked-state gates:** restore and summarize both refuse while `controller.run_state.is_send_allowed` is False (mirror regenerate's screen-side gate + notify copy).
- **Tests:** real in-memory SQLite / real store+controller harnesses (reuse `Tests/Chat/test_console_regenerate_branching.py` + `Tests/UI/test_console_resume_active_path.py` patterns). `./.venv/bin/python -m pytest`. Known baselines (`test_anthropic_native_tools`, `test_chat_functions`, `test_console_native_chat_flow` continue/regenerate + order-dependent flakes) are pre-existing — ignore.
- **NO `git stash`** in any subagent (shared stash stack). Explicit `git add <files>` only — never `-A`/`.superpowers`/`.claude/settings.local.json`. Commit per task; end messages with `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.
- **Base note:** this branch forked from dev b19317f5e (pre-Phase-C #827). Before Task 1 begins, rebase onto latest dev (which should include #827) and re-verify the research anchors below.

## Research anchors (verified 2026-07-24 on dev b19317f5e — re-check after rebase)

- Grammar registry + 4 commands: `console_command_grammar.py` (`ConsoleCommand(name, argument_hint, handler_id)` :50-62; `default_console_registry()` :166-202).
- Parse at send choke point BEFORE readiness gating: `chat_screen.py:10996-11009`; dispatch maps `_CONSOLE_COMMAND_NAME_TO_HANDLER_ID` :11120-11125 + `dispatch_map` in `_dispatch_console_command` :11139-11165; `/prompt` end-to-end template :11174-11293; composer refill `_insert_prompt_text_into_composer` :11295-11325 (uses `insert_text_as_paste`; paste-collapse for long text).
- Modal templates: `console_session_switcher_modal.py` (tagged-union result `ConsoleSwitcherChoice(kind, entry)` :24-29, Button-stack rows, push+callback site `chat_screen.py:1353-1366`); keyboard discipline (non-focusable rows + synthetic highlight) from `console_prompt_picker_modal.py` :11-32.
- Tree/store: `messages_for_session` :833-840 (active-path view; may contain display-only TOOL rows), `active_path_message_ids` :962-971 (tree-only, root→leaf), `set_active_leaf(sid, id|None)` :933-960 (None = clear; KeyError on unknown), `active_leaf` :928-931.
- Payload/choke: `_provider_messages_for_session` `console_chat_controller.py:3252-3265` → `_provider_message_payloads` :3281+ (hard-filters to USER/ASSISTANT; SYSTEM rows never reach payloads); `bound_messages_to_window` call at dispatch choke point :2667-2679 (single place, covers agent + direct branches); `_leading_system_message` :3247-3250.
- History budget API: `console_history_budget.py` — `bound_messages_to_window(...) -> BoundResult` :103-185 (preserves leading system prefix), `count_console_messages_tokens` :31-79.
- Internal prompts: `Internal_Prompts/catalog.py` CATALOG + `get_internal_prompt(key)` (see `console_agent_bridge.py:56-63` for the usage pattern).
- Local-only column pattern to copy: `set_conversation_active_leaf`/`get_conversation_active_leaf` in `ChaChaNotes_DB.py` (bare UPDATE, no version bump) + `_migrate_from_v23_to_v24` + `chachanotes_v23_to_v24_conversation_active_leaf.sql`.

---

## File Structure

- **Modify** `tldw_chatbook/Chat/console_command_grammar.py` — `/rewind` constants + registration.
- **Create** `tldw_chatbook/Widgets/Console/console_rewind_modal.py` — the menu modal (`ConsoleRewindChoice` result).
- **Modify** `tldw_chatbook/UI/Screens/chat_screen.py` — dispatch entries + `_console_command_rewind` handler + restore callback + summarize worker + banner data plumb.
- **Modify** `tldw_chatbook/DB/ChaChaNotes_DB.py` (+ new `DB/migrations/*.sql`) — two summary columns + setters/getters (vNN→vNN+1).
- **Modify** `tldw_chatbook/Chat/console_chat_store.py` — session-level summary state accessors (in-memory + write-through, mirroring the active-leaf pattern) + restore/resume validation of the boundary.
- **Modify** `tldw_chatbook/Chat/console_chat_controller.py` — `summarize_up_to(message_id)` (provider call + storage) + boundary compaction at the dispatch choke point.
- **Modify** `tldw_chatbook/Widgets/Console/console_transcript.py` — render-derived summarize banner above the boundary message.
- **Modify** `tldw_chatbook/Internal_Prompts/` — new `console.rewind_summarize` prompt entry.
- **Tests** per task + `Tests/integration/test_console_rewind_e2e.py`.

---

### Task 1: Grammar + dispatch + handler skeleton + rewind modal (Restore path complete)

**Files:**
- Modify: `tldw_chatbook/Chat/console_command_grammar.py`, `tldw_chatbook/UI/Screens/chat_screen.py`
- Create: `tldw_chatbook/Widgets/Console/console_rewind_modal.py`
- Test: `Tests/Chat/test_console_command_grammar.py` (extend), `Tests/Chat/test_console_rewind_modal.py`, `Tests/UI/test_console_rewind_restore.py`

**Interfaces:**
- Produces: `REWIND_COMMAND_NAME="rewind"` / `REWIND_COMMAND_ARGUMENT_HINT=""` / `REWIND_COMMAND_HANDLER_ID="rewind"` registered in `default_console_registry()`; `ConsoleRewindChoice(kind: Literal["restore","summarize-up-to"], message_id: str, prompt_text: str)` frozen dataclass; `ConsoleRewindModal(ModalScreen[ConsoleRewindChoice | None])` taking `prompts: tuple[RewindPromptRow, ...]` (row = frozen: `message_id`, `index_label`, `preview`); screen handler `_console_command_rewind` + restore callback.

- [ ] **Step 1 (RED):** grammar test — `/rewind` parses to `KIND_COMMAND` with `name="rewind"`; `available_names()` includes it. Modal test — constructing with N rows renders N buttons; selecting a row then the "Restore to here" action dismisses `ConsoleRewindChoice("restore", ...)`; Escape dismisses None (mirror the switcher-modal test file's style). Restore test — on a store with U1→A1→U2→A2: choose U2 → active path becomes `[U1, A1]`, composer receives U2's text (spy `_insert_prompt_text_into_composer`), pointer persisted; choose U1 → `set_active_leaf(None)`, empty path; streaming run-state → blocked + notify, no mutation.
- [ ] **Step 2:** run RED.
- [ ] **Step 3:** implement. Grammar: constants + one `registry.register(...)`. Screen: import constants; add to BOTH dispatch maps; `_console_command_rewind` builds rows from `store.messages_for_session(sid)` filtered `role is USER` (newest first; single-line preview via existing truncation helpers; no prompts → notify "Nothing to rewind." and return), pushes the modal with a callback. Callback (restore kind): **id-lookup rule** — `path = store.active_path_message_ids(sid)`; `i = path.index(choice.message_id)`; `target = path[i-1] if i else None`; gate on `is_send_allowed` (notify `CONSOLE_RUN_ALREADY_RUNNING_COPY` if blocked); `store.set_active_leaf(sid, target)`; `_insert_prompt_text_into_composer(choice.prompt_text, replace=True)`; `_focus_console_composer_if_needed(force=True)`; `await self._sync_native_console_chat_ui()`. Modal: model on `ConsoleSessionSwitcherModal` (Button rows, tagged dismissal, Escape=None) with a two-level flow (select prompt row → action row appears with Restore / Summarize-up-to-here / Never mind); borrow the prompt-picker's non-focusable-rows + synthetic-highlight if arrow-nav is added. The "summarize-up-to" kind dismisses correctly but the screen callback notifies "Summarize lands in Task 3." until Task 3 wires it.
- [ ] **Step 4:** GREEN + `Tests/Chat/test_console_command_grammar.py Tests/UI/test_console_native_chat_flow.py -k "command or rewind"` sweep.
- [ ] **Step 5:** Commit — `feat(console): /rewind command + menu modal + restore-to-here`.

---

### Task 2: Summary storage — columns, setters, store state (no behavior yet)

**Files:**
- Modify: `tldw_chatbook/DB/ChaChaNotes_DB.py`; Create: `tldw_chatbook/DB/migrations/chachanotes_vNN_to_vNN1_conversation_context_summary.sql`
- Modify: `tldw_chatbook/Chat/console_chat_store.py`
- Test: `Tests/DB/test_chachanotes_context_summary_migration.py`, `Tests/Chat/test_console_chat_store_summary.py`

**Interfaces:**
- Produces: columns `context_summary TEXT`, `summary_boundary_message_id TEXT` on `conversations` (one migration, both columns); `set_conversation_context_summary(conversation_id, summary: str | None, boundary_message_id: str | None) -> None` (single bare UPDATE, both fields atomically, no version bump → no sync row) + `get_conversation_context_summary(conversation_id) -> tuple[str | None, str | None]`; store accessors `session_context_summary(session_id) -> tuple[str | None, str | None]` / `set_session_context_summary(session_id, summary, boundary_native_id)` holding in-memory state (keyed by native id; write-through persists the boundary's **persisted** id via the db seam, mirroring `_persist_active_leaf`), restored on resume (`restore_persisted_session` maps the persisted boundary id back to a native id when it is on the loaded tree; dangling → unset).

- [ ] **Step 1 (RED):** migration tests mirror `test_chachanotes_active_leaf_migration.py` exactly: fresh DB at the new version with both columns; setter round-trip; **the write bumps neither `version` nor `sync_log`**. Store tests: set/get in-memory; write-through captured by a recording persistence db-seam fake; resume maps persisted→native and drops dangling boundaries.
- [ ] **Step 2:** RED. **First action of this task: check `_CURRENT_SCHEMA_VERSION` on the rebased base and use FROM=that, TO=that+1 everywhere** (constant, method name, dict key, .sql filename, test asserts).
- [ ] **Step 3:** implement — copy the `active_leaf_message_id` migration + accessor pattern verbatim (guarded ALTER ×2 columns, SQL constant = version bump only, `.sql` reference file, `migration_steps` entry, `_CURRENT_SCHEMA_VERSION` bump).
- [ ] **Step 4:** GREEN + full `Tests/DB/ -k "schema or migration or chacha"` chain sweep.
- [ ] **Step 5:** Commit — `feat(db): local-only conversation context-summary columns (vNN->vNN1)`.

---

### Task 3: Summarize-up-to-here — internal prompt, controller method, choke-point compaction, banner

**Files:**
- Modify: `tldw_chatbook/Internal_Prompts/` (catalog + default for `console.rewind_summarize`), `tldw_chatbook/Chat/console_chat_controller.py`, `tldw_chatbook/UI/Screens/chat_screen.py`, `tldw_chatbook/Widgets/Console/console_transcript.py`
- Test: `Tests/Chat/test_console_rewind_summarize.py`, transcript banner test in `Tests/UI/test_console_native_transcript.py`

**Interfaces:**
- Produces: `async ConsoleChatController.summarize_up_to(message_id: str) -> ConsoleSubmitResult` — gates (active-run; boundary id must be a USER message ON the active path; provider resolve), builds the span text (active path from the previous boundary — or root — up to but excluding `message_id`, USER/ASSISTANT rows only, prior summary prepended when rolling), calls the resolved provider non-streaming via the gateway with `get_internal_prompt("console.rewind_summarize")`, caps the span with `count_console_messages_tokens` guidance, and on success `store.set_session_context_summary(sid, summary, message_id)`; failure → no state change + blocked result with visible copy. Runs via `run_worker(..., exclusive=True, group="console-run")` from the screen callback ("Summarizing conversation…" run state).
- Produces: **choke-point compaction** — at the dispatch choke point (immediately before `bound_messages_to_window`), when the session has a summary AND the boundary message's payload row is PRESENT in `provider_messages` (match by the ids used to build the payload; thread native ids alongside or match content-safe keys — implementer picks the mechanically reliable option and documents it): drop payload rows before the boundary row and append `"\n\n[Conversation summary of earlier turns]\n" + summary` to the leading system message. When absent (pre-boundary regenerate etc.): payload untouched (the leak rule — cover with a dedicated test).
- Produces: transcript banner — render-derived: when the active session has a summary and the boundary native id is on the active path, `console_transcript` renders a non-interactive rule/banner line ABOVE the boundary message row ("⤵ N earlier turns summarized for context — full history above"); no node, no store mutation, disappears when inert.

- [ ] **Step 1 (RED):** controller tests with the fake-gateway harness: happy path stores summary+boundary; provider-not-ready → no state; non-USER/off-path boundary → blocked; rolling re-summarize includes the prior summary and moves the boundary. Compaction tests: boundary-in-payload → pre-boundary rows replaced + summary in system prefix + `bound_messages_to_window` preserves it; boundary-absent (regenerate a pre-boundary message) → payload identical to no-summary (THE leak test); dangling boundary → untouched. Banner test: set summary state → banner renders above the boundary row; restore to before boundary → banner gone.
- [ ] **Step 2:** RED.
- [ ] **Step 3:** implement (wire the Task-1 "summarize-up-to" modal kind to the worker; internal-prompt entry with a concise instruction default emphasizing decisions/state/facts-to-carry-forward).
- [ ] **Step 4:** GREEN + regression: `Tests/Chat/ -k "rewind or summar or budget or controller" Tests/UI/test_console_native_transcript.py -q`.
- [ ] **Step 5:** Commit — `feat(console): summarize-up-to-here boundary compaction + banner`.

---

### Task 4: End-to-end + regression sweep

**Files:** Test: `Tests/integration/test_console_rewind_e2e.py`

- [ ] **Step 1:** E2E over real DB+store+controller (fake provider): converse U1→A1→U2→A2 → `/rewind` restore to U2 → path `[U1,A1]` + composer holds U2 text → send edited prompt (forks a sibling — SP1 interplay) → summarize-up-to the new tip's prompt → payload for the next send is compacted (summary in system prefix, pre-boundary rows gone) while `messages_for_session` still shows full history → persist → drop → resume → summary+boundary restored, banner state derivable, next payload still compacted → restore to before the boundary → payload back to full history (summary inert). Also: `sync_log` purity for the summary writes.
- [ ] **Step 2:** run + focused regression sweep across the touched suites; name baselines; zero new failures. Fix + document any surfaced integration bug (do not paper over).
- [ ] **Step 3:** Commit — `test(console): /rewind end-to-end (restore + boundary summarize + resume)`.

---

## Self-Review

**Spec coverage:** trigger+menu+restore (T1); storage columns/no-sync/resume mapping (T2); summarize provider call + leak-rule compaction + render-derived banner (T3); e2e incl. SP1 interplay + resume + sync purity (T4). Spec-review fixes all carried into task contracts (id-lookup rule T1; banner-not-a-node T3; boundary-in-payload rule T3 with a dedicated leak test; restore-to-empty limitation is documented, no task engineers around it). D3 provider-via-gateway in T3. Migration renumber guard in T2 + Global Constraints.

**Placeholder scan:** interfaces + test contracts are concrete; two bounded implementer choices are explicitly delegated with criteria (payload-row↔native-id matching mechanism in T3; modal arrow-nav borrowing in T1). No TBDs.

**Type consistency:** `ConsoleRewindChoice(kind, message_id, prompt_text)` across T1/T3; `set_session_context_summary(session_id, summary, boundary_native_id)` / `session_context_summary -> (summary, boundary_native_id)` across T2/T3/T4; DB setter takes the **persisted** boundary id (write-through maps native→persisted, mirroring `_persist_active_leaf`).
