# S09 — `Character_Chat/` + `Prompt_Management/` + `Internal_Prompts/` validation

Zero commits touched any of these three directories between the review commit `3722a85748`
and `HEAD`. All 10 findings checked directly against current code. One finding (`console_chat_
controller.py`'s to_thread call site, referenced by Finding 2) sits in a file that DID change in
the 25 commits — verified below that the change (docstrings only, unrelated lines) does not
affect the finding.

## 1. P1 [D1] — an imported card's surviving `character_book` permanently kills world-info, logged at DEBUG only
- Verdict: CONFIRMED
- Site now: `Character_Chat/Character_Chat_Lib.py:1357-1358` (`parse_character_book`:
  `"scan_depth": book_data.get("scan_depth")`, `"token_budget": book_data.get("token_budget")` —
  no default, no coercion) → `Character_Chat/world_info_processor.py:119-120`
  (`self.token_budget = self.character_book.get("token_budget", 500)` returns `None` when the key
  is present-but-null, not the 500 default) → `:305` (`if apply_token_budget and self.token_budget
  > 0:` → `TypeError` on `None > 0`) → swallowed at `world_info_resolver.py:123-127`
  (`except Exception: logger.opt(exception=True).debug(...); return message_text, 0`)
- Proof: direct read confirms the exact chain. `grep -n "_coerce_int"
  tldw_chatbook/Character_Chat/world_info_processor.py` shows it applied to `insertion_order`/
  `priority` (`:232/235/299/300`) but never to `scan_depth`/`token_budget`. `grep -n
  "scan_depth|token_budget" Tests/Character_Chat/test_world_info_diagnostics.py` shows every test
  value is a real int (500/250/300/40) — none passes `None`, confirming the pinning gap.

## 2. P1 [D1/D4] — chat-dictionary regex keys bypass the in-repo ReDoS validator
- Verdict: CONFIRMED
- Site now: `Character_Chat/Chat_Dictionary_Lib.py:174-221` (`_compile_key_internal`: `re.compile
  (pattern_to_compile, self.key_flags)` directly, catching only `re.error`, no length cap, no
  complexity check); applied via `asyncio.to_thread(applier, ...)` at
  `Chat/console_chat_controller.py:22485/22502` (line shifted from the review's cited `:22462` —
  see Note)
- Proof: `python -c "from tldw_chatbook.Character_Chat.Chat_Dictionary_Lib import ChatDictionary,
  match_whole_words; cd = ChatDictionary.from_dict({'key': '/(a+)+\$/', 'content': 'pwn'});
  match_whole_words([cd], 'a'*26+'b')"` → compiles cleanly (`is_regex: True`) and
  `match_whole_words` took **2.596 s** for a 26-char input (matches the review's 2.59 s exactly).
  A 5,001-character key also compiles with no rejection (`MAX_REGEX_PATTERN_LENGTH = 500` in
  `world_info_regex.py:19` is never consulted here). `python -c "from
  tldw_chatbook.Character_Chat.world_info_regex import validate_regex_pattern;
  validate_regex_pattern('(a+)+\$')"` → raises `"Regex pattern is too complex (nested quantifiers
  can hang matching)."` — the exact in-repo validator that would have caught it, confirmed
  imported by `world_info_processor.py:11` and `world_book_import.py:16` but not by
  `Chat_Dictionary_Lib.py` (`grep -n validate_regex_pattern` on the latter → no hit).
- Note: `Chat/console_chat_controller.py` DID change in the 25 commits since the review (`git log
  -- tldw_chatbook/Chat/console_chat_controller.py` → one commit, `6171f1d945`), which shifted the
  `to_thread(applier, ...)` call site from line ~22462 to ~22485/22502. `git diff
  3722a85748..HEAD -- tldw_chatbook/Chat/console_chat_controller.py` shows the change is
  docstring-only additions at lines 598-635, nowhere near the dictionary-applier code — no
  substantive effect on this finding.

## 3. P2 [D1/D2] — two live sync-sqlite-on-the-event-loop chains through `_maybe_await`
- Verdict: CONFIRMED
- Site now (A, persona): `Character_Chat/character_persona_scope_service.py:378-391
  get_character` → `:289 await self._maybe_await(method(*args, **kwargs))` →
  `local_character_persona_service.py:758-763` (sync `get_character` → `_require_db().
  get_character_card_by_id`); `_maybe_await` itself (`:134-137`) is `if
  inspect.isawaitable(value): return await value; return value` — the sync call has already run
  by the time it reaches this function. UI entry: `UI/Screens/chat_screen.py:16819
  self.set_timer(..., self._consume_pending_chat_handoff)` (async timer callback on the message
  pump) → `UI/Console_Modules/session.py:4165-4228 await get_character(character_id,
  mode=runtime_backend)`.
  Site now (B, prompts): `Widgets/Console/console_prompts_modal.py:349-353
  self.run_worker(self.reload_browse(), exclusive=False, group=...)` — a coroutine with no
  `thread=True` → `UI/Console_Modules/prompts.py:285-290` → `Prompt_Management/
  prompt_scope_service.py:1338 await self._maybe_await(service.list_prompts(...))` → `:462-474`
  sync `list_prompts` → `DB/Prompts_DB.py` cursor execute.
- Proof: `grep -n "to_thread" tldw_chatbook/Character_Chat/character_persona_scope_service.py` →
  zero hits. `grep -n "to_thread" tldw_chatbook/Prompt_Management/prompt_scope_service.py` →
  exactly one hit (`:1465`, `count_prompts` only). Contrast confirmed:
  `UI/Library_Modules/library_prompts_controller.py:918-919` calls the same `count_prompts`
  through `self._run_library_service_call(..., isolate_in_worker=True)` — the Library screen's
  correct pattern the Console modal never adopted. `backlog/tasks/task-32804.12` status: To Do.

## 4. P2 [D4] — three hand-rolled UTC timestamp writers, invisible to the ADR-173 guard by construction
- Verdict: CONFIRMED
- Site now: `local_character_persona_service.py:160-161 _now()`,
  `local_chat_dictionary_service.py:205-206 _now()` (both `datetime.now(timezone.utc).
  isoformat()`), `buddy_conversion.py:440` (`datetime.now(UTC).isoformat(timespec="seconds")`)
- Proof: `grep -n "func.value.args|func.value.keywords" scripts/check_timestamp_writers.py` →
  the naive-writer check is `not func.value.args and not func.value.keywords` — a zero-arg
  `.now()` only. All three writers pass a `tz` argument positionally, so the guard's own AST
  predicate excludes them by construction. `python scripts/check_timestamp_writers.py` → reports
  "0 ... OK" with all three live in tree.

## 5. P2 [D3] — `Prompt_Management/Prompt_Engineering.py` (590 lines) is unimportable dead code
- Verdict: CONFIRMED
- Site now: `Prompt_Management/Prompt_Engineering.py:12` — `from tldw_Server_API.app.core.Chat.
  Chat_Functions import chat_api_call`
- Proof: `python -c "import tldw_chatbook.Prompt_Management.Prompt_Engineering"` →
  `ModuleNotFoundError: No module named 'tldw_Server_API'`. `grep -rln "Prompt_Engineering"
  tldw_chatbook/ Tests/` → zero hits outside the file itself; not in `__init__.py`'s
  `_LAZY_EXPORTS`. `backlog/tasks/task-474` status: To Do (the only thing still needing the
  metaprompt text before deletion).

## 6. P3 [D3] — `CharacterPersonaScopeService` defines `_enforce_policy` twice; the dead copy references an undefined attribute
- Verdict: CONFIRMED
- Site now: `character_persona_scope_service.py:126` (shadowed, `def _enforce_policy(self, mode:
  str, action: str)`, references `self._ACTION_IDS.get(action)`) and `:139` (live, `def
  _enforce_policy(self, action_id: str)`), separated by `async def _maybe_await` at `:134`
- Proof: `grep -n "_ACTION_IDS"` → exactly one hit, `:129` inside the dead copy — never defined on
  the class. `grep -c "self\._enforce_policy("` → 68 call sites, all matching the live 1-arg
  signature. `ruff check --select F811 --isolated
  tldw_chatbook/Character_Chat/character_persona_scope_service.py` → "All checks passed!" — the
  linter misses it because of the intervening method between the two defs. Cross-checked against
  `qa/tier2-code-review-2026-09-21/phase4-verification.md:597-620`, whose independent re-trace of
  the repo-wide shadowed-method census reaches the same conclusion for this class.

## 7. P3 [D3] — `Chat_Dictionary_Lib.py` logs 36 sites through stdlib `logging`, bypassing loguru, including user content
- Verdict: CONFIRMED
- Site now: `Chat_Dictionary_Lib.py:6 import logging` alongside `:15 from loguru import logger`
- Proof: `grep -c "^\s*logging\.\(warning\|info\|debug\|error\)"
  tldw_chatbook/Character_Chat/Chat_Dictionary_Lib.py` → 36. Read confirms `:633`
  (`logging.debug(f"... Matched string entry: {entry.key}")`) logs user/card key text and `:670`
  (`logging.debug(f"Applying replacement ... {entry.content[:50]}... in text: {text[:50]}...")`)
  logs both dictionary content and the user's own message text.

## 8. P3 [D3] — `Prompts_Interop.py` is 1,600 lines, ~25% of it a `__main__` demo
- Verdict: CONFIRMED
- Site now: `Prompt_Management/Prompts_Interop.py` (1,600 lines), `if __name__ == "__main__":` at
  `:1204`
- Proof: `wc -l` confirms 1,600 lines; demo block spans `:1204-1600` (396/1600 ≈ 24.75%, matching
  the review's ~24.6%).

## 9. P3 [D3] — `ccv3_parser.py` is a 0-byte module whose only import target has zero callers
- Verdict: CONFIRMED
- Site now: `Character_Chat/ccv3_parser.py` — `wc -c` → 0 bytes
- Proof: `grep -n "ccv3_parser" UI/CCP_Modules/ccp_character_handler.py` → imported at `:1098`
  inside `handle_export_character` (`:1091-1113`). `grep -rn "handle_export_character"
  tldw_chatbook/ Tests/` → exactly one hit, the `def` itself — zero callers, zero tests. The
  `except Exception` at `:1113` would swallow the resulting `ImportError` into a log line.

## 10. P3 [D4] — two byte-identical private-helper pairs, no drift
- Verdict: CONFIRMED
- Site now: `_normalize_extensions` at `world_book_manager.py:784` and
  `local_chat_dictionary_service.py:1103`; `_portrait_content_type`/`_portrait_mime_type` at
  `persona_visual_identity.py:457` and `Persona_Buddy/controller.py:308`
- Proof: `grep -n` on all four sites confirms exact function names and line numbers as cited.

TOTALS: confirmed=10 fixed=0 wrong=0 demoted=0 promoted=0
