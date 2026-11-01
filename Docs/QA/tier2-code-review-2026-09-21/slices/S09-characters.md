# S09 — `Character_Chat/` + `Prompt_Management/` + `Internal_Prompts/`

**Coverage:** files read in full: 5 | sampled: 22 | mechanical only: 39 (of 66).
Mechanical coverage was AST sweeps over all three package dirs — function-body import resolution (**100 imports,
all resolve**), duplicate-method detection (repo-wide), per-module importer census, `except:pass` /
`get_cli_setting` / `run_worker` / `to_thread` / `fetchall` greps, and a timestamp-expression sweep.

## Findings

### P1 [D1] — An imported character card whose `character_book` survives conversion permanently kills world-info for that character, and the failure is logged at DEBUG only
- Where: `Character_Chat/Character_Chat_Lib.py:1357-1358` (emits `scan_depth`/`token_budget` uncoerced);
  `Character_Chat/world_info_processor.py:119-120`, `:153-154`, `:305`; swallowed at
  `Character_Chat/world_info_resolver.py:123-127` and silently at `Chat/console_runtime.py:621-622`.
- Evidence: ran `import_character_card_from_json_string` on a card with
  `character_book = {"name":"Bad book","entries":[{"keys":[123],"content":""}]}`:
  the entry is skipped (no content) → `parse_v2_card:1825` logs *"yielded no usable entries; leaving any legacy key
  intact"* → **the raw book stays in `extensions`**. The stored book is `{"scan_depth": null, "token_budget": null,
  …}` (`parse_character_book` copies `.get("scan_depth")` with no default and no coercion) →
  `WorldInfoProcessor._process_character_book` → `.get("token_budget", 500)` returns **`None`** (key present) →
  `process_messages:305` `self.token_budget > 0` → **`TypeError`**. Then
  `resolve_world_info_injection(...)` → returned the text unchanged, `matched=0`, **one `logger.debug`**.
  Second reachable shape (stored `world_books` rows) raises in `__init__` at `:153` (`max(0, None)`). Four further
  crash shapes in `_process_entry` on raw card data: `keys:[123]` → `AttributeError`; `secondary_keys:[{…}]` +
  `selective` → same; `entries:"str"` / `entries:["str"]` → `'str' object has no attribute 'get'`.
- Why it matters: **the user's lorebook silently never fires on any send, forever, with no visible error.** The same
  file already has `_coerce_int` (`world_info_processor.py:44`) applied to `insertion_order`/`priority` **because of
  this class of bug**, with a pinning test for it — `scan_depth`/`token_budget`/`keys` were missed.
- Recommended correction: route `scan_depth`/`token_budget` through the existing `_coerce_int`, and coerce
  `keys`/`secondary_keys` elements with the existing `world_book_import._as_str_list`. Raise the resolver's swallow
  from `.debug` to `.warning` so a dead lorebook is discoverable. · Size: S · Confidence: verified
- Pinning test: `Tests/Character_Chat/test_world_info_diagnostics.py::test_null_insertion_order_does_not_crash_sort`
  pins the **sibling** fields as a requirement; nothing pins these.

### P1 [D1/D4] — Chat-dictionary regex keys bypass the ReDoS validator that world-info entries use; an imported card can wedge a send thread indefinitely
- Where: compile `Character_Chat/Chat_Dictionary_Lib.py:174-221` (`_compile_key_internal` — **no length cap, no
  catastrophic check**); match `:224-231`, `:605-634`, `:653-681`. Sources: card extensions → `:1294`, stored
  `entries_json` → `:1221`. Applied at `Chat/console_chat_controller.py:22462` (`await asyncio.to_thread(applier,…)`).
- Evidence: `ChatDictionary.from_dict({"key": "/(a+)+$/", "content": "pwn"})` compiles;
  `match_whole_words([entry], "a"*26+"b")` took **2.59 s** (exponential — 35 chars ≈ minutes).
  **The in-repo validator rejects the same pattern:** `validate_regex_pattern("(a+)+$")` → *"Regex pattern is too
  complex (nested quantifiers can hang matching)"*. Length: this module accepted a **5,000-char** pattern vs
  `MAX_REGEX_PATTERN_LENGTH = 500`.
- Why it matters: `world_info_regex.py`'s own docstring states this risk and **the module exists to close it**; it is
  adopted by `world_info_processor.py:220` and `world_book_import.py:125` but **not by the sibling matcher in the
  same package.** Because the match runs under `asyncio.to_thread`, the UI survives but **the send never returns,
  `CancelledError` cannot stop the spinning thread, and one default-executor thread is pegged for the process
  lifetime.**
- Recommended correction: call `world_info_regex.validate_regex_pattern(...)` in `_compile_key_internal` before
  `re.compile`, downgrading to literal on `ValueError` — the fail-closed shape
  `world_info_processor._process_entry:216-222` already uses. **Canonical home already exists.**
- Size: S · Confidence: verified

### P2 [D1/D2] — Two confirmed live sync-sqlite-on-the-event-loop instances, including the `_maybe_await` call site the lead could not reproduce
- **(A, persona path):** `Character_Chat/character_persona_scope_service.py:378-392 get_character` →
  `:289 return await self._maybe_await(method(*args, **kwargs))` →
  `local_character_persona_service.py:759-763` (**sync**) → `DB/ChaChaNotes_DB.py:9194 get_character_card_by_id` →
  `execute_query`. UI entry: `UI/Screens/chat_screen.py:16819 self.set_timer(…, self._consume_pending_chat_handoff)`
  (**an async callback on the message pump, not a worker**) → `:17564` → `UI/Console_Modules/session.py:4165` →
  `:4224 await get_character(character_id, mode=runtime_backend)` with `runtime_backend == "local"`.
- **(B, prompts path):** `Widgets/Console/console_prompts_modal.py:349-353 self.run_worker(self.reload_browse(), …)`
  — **a coroutine, `thread` unset** → asyncio Task on the UI loop → `:746` → `UI/Console_Modules/prompts.py:285-289`
  → `Prompt_Management/prompt_scope_service.py:1338 await self._maybe_await(service.list_prompts(...))` →
  `:462-474` (**sync**) → `DB/Prompts_DB.py:3142 cursor.execute(...)`.
- Evidence: read the full chain at every hop; `grep -n "to_thread"` over both packages → the only offloads are
  `chat_dictionary_scope_service.py:114` and `prompt_scope_service.py:1465` (`count_prompts` **only**).
  `character_persona_scope_service.py` has **zero**.
- Why it matters: **this answers the brief's standing question — `_maybe_await`'s sync argument doing real blocking
  sqlite on the loop, on the persona path.** B is worse in frequency (every Console Prompts modal open and page change).
- Recommended correction: **B has an in-repo precedent** — `UI/Library_Modules/library_prompts_controller.py` calls
  the same scope methods through `library_screen.py:12993 _run_library_service_call(..., isolate_in_worker=True)`,
  which forces `asyncio.to_thread(run_finite_local_worker, …)`; **the Console modal never adopted it.**
- Size: M · Confidence: verified
- Already covered: task-32804.12 (To Do) — **these are its missing concrete instances.**

### P2 [D4] — Three hand-rolled UTC timestamp writers persist non-canonical shapes; the ADR-173 guard is structurally blind to all three
- `local_character_persona_service.py:160-161 _now()` (15 call sites → `last_modified`/`updated_at` → persona-store
  JSON file); `local_chat_dictionary_service.py:205-206 _now()` (→ history JSON file);
  `buddy_conversion.py:440` (→ `converted_at` → **exported Actor Pack payload**).
- Evidence: all three are `datetime.now(timezone.utc).isoformat()` (microseconds + `+00:00`) vs
  `Utils/timestamps.py:63 utc_now_iso()` → `…mmmZ`. `scripts/check_timestamp_writers.py` matches exactly two shapes:
  `.utcnow` and `.isoformat()` on a **zero-arg** `.now()`. **All three pass `timezone.utc` positionally, so
  `not func.value.args` excludes them.** Running the guard with all three in tree: *"0 datetime.utcnow() site(s),
  0 naive datetime.now().isoformat() occurrence(s)"* → **OK**.
- Why it matters: TASK-32803.5 is **Done** and the guard reports green — **so the repo believes this is closed. It is
  not, and the guard cannot report it.** · Size: S (swap) / M (guard extension) · Confidence: verified

### P2 [D3] — `Prompt_Management/Prompt_Engineering.py` (590 lines) is unimportable dead code
- `:12 from tldw_Server_API.app.core.Chat.Chat_Functions import chat_api_call` → **`ModuleNotFoundError: No module
  named 'tldw_Server_API'`**. Zero importers in `tldw_chatbook/` or `Tests/`; not in `__init__.py`'s `_LAZY_EXPORTS`.
  Live references: the v1 review's own slice notes (**noticed, never filed**), `backlog/tasks/task-474` (wants its
  metaprompt onboarded), and a stale `Docs/Development/Developer_Guide.md:72` entry.
- Recommended correction: delete the module and the Developer_Guide line — but **extract the metaprompt text first**,
  because task-474 (To Do) is the only thing that still needs it. · Size: S · Confidence: verified

### P3 [D3] — `CharacterPersonaScopeService` defines `_enforce_policy` twice; the shadowed copy references an attribute that does not exist, and no linter catches the shape
- `:126` (shadowed) and `:139` (live), same class body, separated only by `_maybe_await` at `:134`. All **68**
  `self._enforce_policy(...)` call sites pass one argument = the live arity, so no runtime break. The dead copy reads
  `self._ACTION_IDS`, which is referenced only on that line and **is never defined on the class**.
  `ruff 0.16.6 --select F811 --isolated` → *All checks passed*; a minimal two-def probe **is** flagged, but a probe
  reproducing the exact shape (**intervening `async def`**) is **not** — reproduced in an isolated 22-line file.
- *(Lead's note: this generalised into a repo-wide census — see D4 §1 and `phase4-verification.md` "S09".)*
- Size: S · Confidence: verified

### P3 [D3] — `Chat_Dictionary_Lib.py` logs 36 live call sites through stdlib `logging`, including user dictionary keys and content previews
- `:6 import logging`; 36 calls at `208, 215, 427, 472, 500, 527, 540, 546, 568, 598, 625, 633, 646, 650, 670,
  844-1018`. The module **also** imports loguru at `:15`. `Logging_Config.py` configures the stdlib root logger and
  loguru independently (**no `InterceptHandler`**), so these bypass the loguru sink chain. `:633` logs `entry.key`
  (user/card text); `:670` logs `entry.content[:50]` **and `text[:50]` — the user's message.** `:208`/`:215` are
  user-facing warnings (a dictionary regex silently downgraded to literal) that **never reach the app log**.
- Size: S · Confidence: verified
- Already covered: task-32806.7 covers "user content to logs" **for the loguru path**; this is the stdlib escape hatch.

### P3 [D3] — `Prompts_Interop.py:1204-1596` is a 393-line `__main__` demo (24.6% of a 1,600-line library module)
- It is the stated reason for three "for example usage" imports at `:26`, `:31`, `:32`. The
  `tempfile_no_secure.tsv` row for this file is **demo-only, not a production path**. · Size: S

### P3 [D3] — `Character_Chat/ccv3_parser.py` is a 0-byte module whose only import target does not exist, inside a method with zero callers
- Imported at `UI/CCP_Modules/ccp_character_handler.py:1098` inside `handle_export_character` (`:1091`), which has
  **no `@on`, no binding, zero tests, and one hit total (its own `def`)**. The sibling comment at `:162` already
  documents that `ccv3_parser.import_character_card_json` "is an empty module" — **the export half was left behind.**
  The `except Exception` at `:1113` would swallow the `ImportError` into a log line if it ever ran.
  **Of 100 function-body imports across the three slice packages, this is the only unresolvable one.** · Size: S

### P3 [D4] — Two byte-identical private-helper pairs in this slice, no drift
- `_normalize_extensions`: `world_book_manager.py:784` ≡ `local_chat_dictionary_service.py:1103`.
- `_portrait_content_type`/`_portrait_mime_type`: `persona_visual_identity.py:457` ≡
  `Persona_Buddy/controller.py:308` — **an 11-module cluster**, see D4 §4. · Size: S · Owner: task-32808.9

## Candidate triage
**RETIRED — the two hypotheses the lead asked about, both answered NO:**
- **`_reject_json_constant` drift across 7 copies: no drift.** All 7 raise unconditionally; only the exception
  *type* differs (`ValueError` ×5, `PermissionStoreSnapshotError`, `_InvalidJsonTokenError`). `parse_constant` is
  called for NaN/Infinity/-Infinity in every case, **so no copy accepts a constant another rejects.** The real
  (minor) gap is **coverage, not drift**: `visual_identity.py` has 8 `json.loads` sites but strict kwargs on only 6 —
  `:2019` and `:2747` are bare, and both re-read app-written DB values. P3 at most.
- **`_coerce_bool` `bool(value)`-vs-`default` divergence: does NOT reach stored card data.** `[]`/`{}` diverge;
  `0.0` does **not** (all four take the `(int,float)` branch). Storage path traced: card-extension `ChatDictionary`
  objects are consumed **read-only** by the send path; the only writers take entries from API/UI payloads, not card
  data — and `to_dict()` re-emits `self.enabled` already normalized to a real `bool`, **so a list can never be
  persisted.** The divergence is **read-time only**: a crafted card with `"enabled": []` yields a silently-OFF chat
  dictionary where the same value on a world-book entry yields ON. P3 consistency, not data.
**CONFIRMED:** `character_persona_scope_service` — `get_character` is the UI-reached, sqlite-hitting method (P2), and
the `_maybe_await` D1 call site at `:289`. **Ruled out with reasons:** `list_characters`/`search_characters` (UI
callers hard-code `mode="server"`; the local list path bypasses the service entirely via
`personas_screen.py:3727-3820 asyncio.to_thread(get_character_page_for_ui, db, …)`); all character/world-book/
session/message/exemplar CRUD (**zero UI callers**); `list_chat_greetings`/`*_chat_preset` (local impls do hit
sqlite and UI wrappers exist, **but those wrappers themselves have zero callers — dead**).
`prompt_scope_service` — **confirmed partial**: one `to_thread` in 2,148 lines (`count_prompts` only); every other
method uses the unoffloaded shape. **The Library screen's calls to the same methods ARE correctly offloaded.**
**RETIRED:** `except_exception_return` ×10 — all documented never-raise contracts with the behaviour stated in the
docstring. `function_body_import` ×82 — **AST-resolved all 100 across the three packages against
`importlib.util.find_spec`; every one resolves.** The one broken import is outside the three dirs (filed P3).
`Internal_Prompts/authoring.py:51,98` dotted `get_cli_setting` — **round-trips correctly**, tested against a scratch
`TLDW_CONFIG_PATH`: `save_override` wrote `[internal_prompts.agents.…]`, `_override_table` read it back,
`override_state` → `customized=True`, and `get_internal_prompt` returned the override text.
`fetchall_no_limit` ×4 and `fetchall_dynamic_sql` ×5 — fully parameterized; literal-only concatenation; paged
queries carry `LIMIT ? OFFSET ?` with `limit = min(limit, 100)`. `lock_and_execute` — `:2583 BEGIN IMMEDIATE` is a
deliberate hand-rolled reservation spanning a filesystem `rmdir` that `db.transaction()` cannot express, guarding
`connection.in_transaction` first and rolling back in `finally`. `re_compile_in_def` ×12 — small alternations
rebuilt per call, hitting `re`'s 512-entry cache. `tempfile_no_secure` ×2 — one inside a
`secure_private_directory(...).verified_private` gate with `rmtree` in `finally`, one in the dead `__main__`.
`expression_set_io.py` zip handling — **retired**: `MAX_ZIP_MEMBERS=64`, `MAX_MEMBER_BYTES=16MB`,
`MAX_TOTAL_BYTES=64MB` enforced at every read; **members are used only as zip keys, never filesystem paths**;
Windows separators normalized; JSON members size-checked before `zf.read`.
`_set_status`/`_toast` non-adopters — **retired for this slice**: `grep -c "def _set_status\|def _toast"` across all
three dirs = **0** (non-Textual modules).
**PARTIALLY CONFIRMED, DEFERRED:** PIL bombs in the *parsers* — `extract_json_from_image_file` has
`_MAX_CARD_DECODE_PIXELS = 50_000_000` guarding its only decode; unguarded decodes remain at
`Character_Chat_Lib.py:1036` and `:4485`, and `expression_set_io.py:565-578` is byte-capped but not pixel-capped.
`visual_identity.py:1891` is the correct pattern. **Already covered: task-32806.8** — not re-filed.
`Character_Chat_Lib.py:225,3445,3628` strftime — confirmed but **P3**: naive local time reaching
`conversations.title`, a free-text auto-title, not a sortable column. Invisible to the guard (strftime shape).
**UNVERIFIED:** `token_est_len_div4` `world_info_processor.py:678` (a fallback only when `Utils/token_counter` is
unavailable; owned by task-32808.10); `id_keyed_dict` `Chat_Dictionary_Lib.py:800,977,1000` (module-level functions,
not widgets, so the recompose hazard does not apply; sites not read).
**God modules >5k lines: none in slice** (largest `Character_Chat_Lib.py` 4,781). Owned by task-32809.2.

## D4 observations for repo-wide Phase 3
1. **Shadowed method definitions — repo-wide census, 11 classes, and `ruff F811` catches none of them.** AST scan of
   every class body (excluding `@property.setter`/`@overload`), then arity-matching every call site against the
   *surviving* definition:

   | file | class | method | shadowed→live | calls | mismatched |
   |---|---|---|---|---|---|
   | `Evaluations_Interop/evaluation_scope_service.py` | `EvaluationScopeService` | `_enforce_policy` | 106 (3 args) → 137 (1 arg) | 49 | **`:129` passes 3 args → `TypeError`** |
   | `Study_Interop/study_scope_service.py` | `StudyScopeService` | `_enforce_policy` | 130 → 227 | 56 | none (dead) |
   | `Character_Chat/character_persona_scope_service.py` | `CharacterPersonaScopeService` | `_enforce_policy` | 126 → 139 | 68 | none (dead) |
   | `UI/Screens/chat_screen.py` | `ChatScreen` | `on_button_pressed` | 24578 → 25009 | — | **a Textual event handler fully shadowed** |
   | `UI/Screens/chat_screen.py` | `ChatScreen` | `_restore_collapsible_states` | 24960 → 25205 | 0 | dead |
   | `Widgets/voice_input_widget.py` | `VoiceInputWidget` | 3 methods | 449→560, 399→581, 408→590 | 0 | dead ×3 |
   | `UI/Speech/speech_catalog_mixin.py` | `SpeechCatalogMixin` | `_reserve_voice_request_token` | 607 → 1795 | 2 | dead |
   | `UI/STTS_Window.py` | `AudioBookGenerationWidget` | `_get_model_for_provider` | 1196 → 1464 | 2 | dead |
   | `UI/Dictation_Window_Improved.py` | `ImprovedDictationWindow` | `_show_troubleshooting` | 732 → 1044 | 1 | dead |

   **Lead's corrections after verification:** the `Evaluations_Interop` `TypeError` **is reproducible but the
   containing method has no caller**, so it is latent (P3), not the live P0 the table implies; and the `ChatScreen`
   shadow is **real but not user-facing** — an AST diff found zero string constants present only in the dead body,
   so it reads as duplicated-then-extended, not lost functionality. See `phase4-verification.md`.
   **The generalisable result is the census and the linter gap, not either instance.** A ~20-line AST check in
   `scripts/preflight.sh` closes the whole class.
2. **Non-canonical UTC writers invisible to the ADR-173 guard.** The common shape is
   `datetime.now(<tz>).isoformat()` — the guard's `not func.value.args` test **excludes every aware writer by
   construction**, so *every* aware-but-non-canonical writer in the repo is currently unseen.
3. **Sync-local-service-on-the-loop is a per-service coin flip, not a policy.** In this slice:
   `chat_dictionary_scope_service.py:114` offloads, `character_persona_scope_service.py:289` does not,
   `prompt_scope_service.py` offloads exactly one of ~20 methods. **The Library screen has a working pattern the
   Console surfaces never adopted.** This is task-32804.12's real shape: not "a few stragglers" but an unadopted
   convention across ~38 services.
4. **Image magic-byte sniffing — 11 modules, a helper exists but in the wrong package.**
   `Image_Generation/adapters/image_format_utils.py:118 format_from_bytes` + `:130 content_type_for_format` have
   8 importers, **all inside `Image_Generation/`**. Re-rolls: `Character_Chat/persona_visual_identity.py:457`,
   `Persona_Buddy/controller.py:308`, `Character_Chat_Lib.py:2730,4401`, `UI/Screens/personas_screen.py`,
   `Tools/web_tool_impls.py`, `Actor_Packs/{creation,contracts,export}.py`, `Canvas/limits.py`. No drift in the two
   diffed. **The canonical home should be `Utils/`, not `Image_Generation/adapters/` — that placement is why nobody
   adopted it.**
5. **`_coerce_bool` (4 in-slice) / `_coerce_int` (2)** — the canonical home should be `Character_Chat/` module-level,
   **not** `Utils/Utils.coerce_bool_flag` (which stringifies ints, so `5` → `default` where these four return
   `True`). Recording it so Phase 3 does not re-litigate.
6. **Documented-deliberate duplication worth recording as such:** `Chat_Dictionary_Lib.py:1466-1472` states it
   duplicates the `metadata.active_dictionaries` parse **on purpose** to protect a byte-for-byte regression pin.

## Left UNVERIFIED
| Claim | Why | Command |
|---|---|---|
| The ReDoS wedge permanently burns a default-executor thread (rather than hanging one send) | needs a live app; executor sizing is runtime-dependent | a harness firing 8 concurrent `asyncio.to_thread` catastrophic matches, then timing a 9th trivial `to_thread` call |
| The world-info failure is visible in the Personas Try-it panel as the raw exception string | `_handle_tryit` catches and calls `show_error(f"Couldn't run the preview: {exc}")` — read only, not rendered | boot the app, Personas ▸ Lore ▸ Try-it with a book whose stored `scan_depth` is NULL |
| `handle_export_character` truly has no UI binding (vs a `BINDINGS`/action-string dispatch) | grep found only the `def`; Textual action dispatch by string name would not appear | `rg -n "export_character\|action_export" tldw_chatbook/UI/ tldw_chatbook/css/` |
| `world_book_manager.get_world_books_for_conversation` nests `db.transaction()` inside an outer one — safe only if the context manager is reentrant | read only; tests pass, implying reference-counting, but `DB/base_db.py` not read | `sed -n '/def transaction/,/yield/p' tldw_chatbook/DB/base_db.py` |
| `id_keyed_dict` rows `Chat_Dictionary_Lib.py:800,977,1000` are benign | not read; module-level functions, so the recompose hazard should not apply | `sed -n '795,805p;972,1005p' tldw_chatbook/Character_Chat/Chat_Dictionary_Lib.py` |
| `world_info_processor.py:678` `len(content)//4` is only a fallback | `:689` imports `Utils.token_counter` in the function body; the helper's name not confirmed to resolve | `rg -n "def " tldw_chatbook/Utils/token_counter.py \| head` |
