# Follow-ups 504 + 551 — Chat-load repair + server-backend user-profile resolution — Design

**Date:** 2026-07-24
**Program:** RP-UX follow-ups (post-sweep; B1 #815 / B2 #824 / B3 #830 / B4 #849 merged).
**Branch:** `claude/roleplay-followups-504-551` off dev `238ac3041`.
**Status:** design approved by user (with AC-1 keywords amendment + reachability-follow-up approach confirmed).

## Part 1 — task-504: repair `display_conversation_in_chat_tab_ui`

### Scout-verified ground truths

- The function (`Event_Handlers/Chat_Events/chat_events.py:4147-4423`) is **currently unreachable from the live UI**: its three callers (save `:3323`, clone `:3592`, load-selected `:3797`) fire only from buttons composed in `settings_sidebar.py`, whose `create_settings_sidebar()` has no callers since task-412. The tabs wrapper (`chat_events_tabs.py:319`) has no production callers either.
- The dead `#chat-right-sidebar` query (`:4247`) is the first failure; the title/UUID/keywords ids written at `:4287-4295` (`#chat-conversation-title-input`, `#chat-conversation-uuid-display`, `#chat-conversation-keywords-input`) are **composed nowhere** — they would raise QueryError even with `:4247` removed. Same for the six `#chat-character-*-edit` ids (`:4248-4285`).
- The not-found branch (`:4158-4188`) writes the same dead ids inside its own guard.
- Live equivalents in `EnhancedSettingsSidebar._compose_chat_details()` (prefix-composed, `id_prefix="chat"`): **`#chat-chat-title`** (`Input`, `enhanced_settings_sidebar.py:436-440`) and **`#chat-chat-id`** (`Input`, disabled, `:428-433`). Zero readers/writers today; no `Input.Changed` router anywhere matches these ids (sidebar's own `@on(Input.Changed)` filters on `"settings-search"`; `app.on_input_changed` matches exact other ids) → programmatic `.value` writes are side-effect-free. **There is no live keywords field.**
- Twin dead pattern: `handle_chat_clear_active_character_button_pressed` queries `#chat-right-sidebar` + the six character-edit ids (`:4892-4907`); its success notify sits after the dead queries, so today it always degrades to the error notify.
- The shared test fixture `Tests/fixtures/event_handler_mocks.py` registers `#chat-right-sidebar` (bare MagicMock, `:188`), the three `#chat-conversation-*` ids (`:122-124`) and the six `#chat-character-*-edit` ids (`:153-158`) — this is what masks the bug in the existing tests. Unknown selectors already raise `QueryError` (`:250-253`).
- `TabContext.GLOBAL_WIDGETS` (`Chat/tabs/tab_context.py:~40-46`) enumerates the three dead `#chat-conversation-*` ids; `Tests/.../test_chat_events_tabs.py:868-874` pins that `#chat-conversation-title-input` is not redirected.
- `:4421-4423` logs "Displayed conversation …" unconditionally, even when the except path ran.

### Changes

1. **`display_conversation_in_chat_tab_ui`:**
   - Delete the `#chat-right-sidebar` block (`:4247-4285`) — container query + six character-edit writes (AC #2 "dead write attempt removed": no live equivalents exist).
   - Repoint conversation details to the live sidebar ids: title → `#chat-chat-title` (`Input.value`), conversation id → `#chat-chat-id` (`Input.value`). **Drop the keywords write** (no live field; AC #1 amended accordingly).
   - Wrap the sidebar-detail writes (title/id + `#chat-system-prompt` character branch) in **their own scoped `try/except QueryError`** (warning log, no notify) so a missing sidebar field can never again abort the message-log mount. The message-log population, `#chat-input` focus, and token counter run regardless of sidebar-detail outcome.
   - Repoint the not-found branch (`:4164-4171`) the same way (title → `#chat-chat-title` "Error: Not Found", id → `#chat-chat-id`; keywords write dropped).
   - Move the `"Displayed conversation"` info log into the success path.
   - Keep the outer `except QueryError` as last-resort for the truly load-bearing queries (`#chat-log`, `#chat-input`).
2. **`handle_chat_clear_active_character_button_pressed`:** remove the dead container + six field queries; keep the (live) `#chat-system-prompt` reset and the "Active character cleared." success notify.
3. **Fixture honesty (`event_handler_mocks.py`):** remove `#chat-right-sidebar`, the three `#chat-conversation-*` entries, and the six `#chat-character-*-edit` entries; add `#chat-chat-title` (`Input`, `value=""`) and `#chat-chat-id` (`Input`, `value=""`). Also drop the `#chat-right-sidebar` `query_one` wiring block (`:265-269`). **Blast-radius rule:** run the suites that consume `mock_app`; if a test of some *other* legacy handler (outside 504's scope) breaks on a removed id, restore just that id with a `# DEAD-ID: not in live tree; kept for legacy-handler test — see follow-up task` comment and record it in the follow-up task.
4. **Tabs surfaces:** in `TabContext.GLOBAL_WIDGETS` swap the three dead `#chat-conversation-*` ids for `#chat-chat-title` + `#chat-chat-id` (keep `#chat-system-prompt`); update `chat_events_tabs.py` wrapper id references and the `test_chat_events_tabs.py` redirect pins consistently. Tabs-mode raw-literal behavior of the inner function (`#chat-log`/`#chat-input` not routed through `_tabbed_chat_selector`) is pre-existing and unchanged.
5. **Regression tests (AC #3):** against the live-shaped fixture — (a) load populates `#chat-chat-title`.value with the metadata title, `#chat-chat-id`.value with the conversation id, and mounts the message widgets into `#chat-log`; (b) a sidebar-detail QueryError does not prevent the message-log mount (scoped-guard pin); (c) the task-442 substitution twins keep passing unchanged.
6. **Task hygiene:** amend task-504 AC #1 (title + UUID + message log; keywords dropped — no live field) with a note *before* implementation; file the follow-up task: "Conversation load/save/clone entry points missing from live Chat tab UI" (covers reachability + residual dead surfaces: `app.py` right-sidebar watchers `:8613/:8626`, dead button routers `app.py:6925/:9299/:9445`, orphan CSS, `settings_sidebar.py` module retirement). Backlog-ID collision protocol at file time AND merge time.

### Boundaries (OUT of scope)

- No new UI (no keywords field, no load/save/clone buttons — that is the follow-up task).
- Other legacy handlers that query dead ids keep their behavior (only the `:4895` twin is cleaned, because it is the same audit family).
- `chat_events_sidebar_resize.py`, `app.py` right-sidebar watchers, orphan CSS: follow-up task.
- The live `toggle-chat-right-sidebar` *button* id family is a different id — untouched.

## Part 2 — task-551: async active-user-profile resolution for the server backend

### Scout-verified ground truths

- `resolve_active_user_profile_name(service)` (`Character_Chat/active_user_profile.py:71-97`) validates the pointer against the **sync local** service only; server-only profiles fail validation → `{{user}}` silently falls back (Qodo #849-4; lenient variant would break the deliberate dangling→None).
- `CharacterPersonaScopeService.list_user_profiles` (`character_persona_scope_service.py:637-661`) is **async**, `mode: str = "local"`, normalizes mode (`{"local","server"}` else ValueError), enforces policy `character.persona.list.<mode>`, delegates to the backend, `_maybe_await`s. Server backend → `TLDWAPIClient.list_persona_profiles` → GET `/api/v1/persona/profiles`, returns raw parsed JSON; **every failure raises** (`APIConnectionError`/`AuthenticationError`/`APIResponseError`/`ValueError`/`PolicyDeniedError`); no result caching. Payload may be a bare list or `{"items": [...]}` (CCP handler precedent `ccp_persona_handler.py:85-91`); records carry `"name"`.
- Scope service instance: `app.character_persona_scope_service` (`app.py:3863`); its local backend wraps the SAME `local_character_persona_service` instance.
- Backend-mode sources: app-authoritative `app.get_authoritative_runtime_source()` (`app.py:4961-4967`, default "local"; bootstrap downgrades server→local when unconfigured); workbench-local `persona_handler.current_mode()` (`ccp_persona_handler.py:28-43`, window state first, app fallback, default "local").
- The three task-442 sites: (a) preview `_active_user_name` (`personas_preview_controller.py:69-84`, sync) consumed by sync `_load_greetings` (`:109-159`) whose two callers are async (`reset_for_character:178`, `handle_character_loaded:379`); (b) `chat_screen._start_character_console_session` (`:10398`, async, call `:10445-10447`); (c) `chat_events.display_conversation_in_chat_tab_ui` (`:4147`, async, call `:4205-4207`, preserves `USERS_NAME` fallback).
- Existing tests: resolver units (`Tests/Character_Chat/test_active_user_profile.py`), site twins (`Tests/UI/test_personas_preview.py:613-716`, `Tests/UI/test_chat_first_handoffs.py:485-560`, `Tests/Event_Handlers/Chat_Events/test_chat_events.py:777-860`) — all set only `local_character_persona_service`, never a scope service.

### Changes

1. **New async resolver** in `active_user_profile.py`:
   ```
   async def resolve_active_user_profile_name_async(
       scope_service, *, mode: str | None, local_service=None
   ) -> str | None
   ```
   - Normalize mode (strip/lower; `None`/unknown → `"local"` — never raise on garbage).
   - **mode ≠ "server" → return `resolve_active_user_profile_name(local_service)`** unchanged (zero new failure modes in local mode: no policy-enforcement hop, no scope-service dependency; existing semantics byte-identical).
   - **server:** pointer `None` or `scope_service` `None` → `None`. Else `await scope_service.list_user_profiles(mode="server")`; accept bare-list or `{"items": [...]}` payloads; match `str(record.get("name") or "") == pointer` → pointer; no match → `None` (dangling→None preserved). **Any exception** (connection, auth, policy, ValueError) → debug log → `None` — never raises into a send path; `{{user}}` falls back exactly as today. No result caching (calls happen at selection/send cadence only). Server mode validates against the server only — no local fallback on server failure (a pointer is resolved by the backend the surface is using, or not at all).
2. **Site (b) `chat_screen._start_character_console_session`:** swap to `await resolve_active_user_profile_name_async(getattr(app, "character_persona_scope_service", None), mode=<app mode>, local_service=getattr(app, "local_character_persona_service", None))` where `<app mode>` = guarded `app_instance.get_authoritative_runtime_source()` (fallback `"local"`). Greeting + `user_profile_label` consumption unchanged.
3. **Site (c) `chat_events.display_conversation_in_chat_tab_ui`:** same swap with `app`; keep the `or app.app_config.get("USERS_NAME", "User")` fallback byte-exact.
4. **Site (a) preview controller:** resolve at the async boundary — new `async def _resolve_user_name_async()` on the controller using the **workbench's** mode (`self.screen.persona_handler.current_mode()`, guarded, fallback `"local"` — the preview validates against the backend the workbench is browsing, mirroring what the list surfaces show); `reset_for_character` and `handle_character_loaded` await it and pass the result into `_load_greetings(user_name=...)` (new keyword arg, sentinel default falls back to the existing sync `_active_user_name()`); `set_speakers` logic unchanged. Sync internals stay sync.
5. **Mode-source rule (one precedence rule per surface):** chat surfaces (b)(c) use the app-authoritative runtime source; the workbench preview (a) uses the workbench mode. Each surface validates against the backend it actually serves.

### Byte-compat (AC #3)

No pointer → the resolver returns `None` before any backend call → all fallbacks identical → the task-442 twins pass **untouched** (their fixtures set no scope service and no runtime source → guarded mode defaults to `"local"` → exact sync path).

### Tests

- Resolver units (`Tests/Character_Chat/test_active_user_profile.py`): server happy path (fake async scope service, bare list), `{"items": [...]}` shape, server dangling → None, scope-service raises → None, `mode="local"`/`None`/garbage delegates to sync path, `scope_service=None` in server mode → None, never-raises.
- Per-site server-mode substitution test: fake scope service + mode source pinned to `"server"` → the profile name substitutes at each of the three surfaces. Existing twins unchanged.

### Boundaries

- No caching layer, no retry/backoff (selection-cadence network calls; client timeouts govern).
- No policy-enforcement path added in front of local-mode resolution.
- Naming: all new identifiers are user-profile-named; "persona" never refers to the user (the `character_persona_scope_service` attribute name is the sanctioned both-halves service name).
- Preview label staleness on clear/switch and description-into-system-prompt residuals (task-442 archive) stay out.

## Global constraints (carried from the program)

- Test env prefix: `HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest ... -q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread`; Tests/UI has `asyncio_mode=auto` — don't mix Tests/UI with other dirs in one invocation.
- Implementers stage ONLY their task's files; never `git add -A`; never touch `.superpowers/`.
- Subagents prepend a non-leading `cd` to every Bash call (`true; cd <worktree>; ...` — a hook strips a LEADING `cd`).
- No background/broad test sweeps; never broad-pkill pytest.
- Backlog-ID collision check (dev max + archive + open branches) at task-file time and re-verified at merge.
- PR to `dev`; Qodo adjudication; STOP for explicit user merge-go.
