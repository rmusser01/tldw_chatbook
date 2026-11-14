# Phase 4 — verification log

Every P0/P1 from a completed slice, re-run by me (the orchestrator) after the slice filed it. Rules: worktree root as cwd, `PYTHONPATH=<worktree>`, isolated `HOME`/`XDG_*`/`TLDW_CONFIG_PATH` profile, `.venv/bin/python` from the main checkout. The app was never run; no full-suite run.

A finding that reproduced is marked CONFIRMED; one did not and is marked CONTESTED; one was corrected upward.

---

## P0 — Media bare-DML transaction leak (DB-media-base)

```
$ PYTHONPATH=$WT $PY <SCRATCH>/home/repro/media_repro2.py
after bare update_keywords_for_media: in_transaction = True
after later add_media_with_keywords:  in_transaction = True | mid2 = 2
after close+reopen: media rows = 1 (expected 2) | keywords linked to mid1 = ['k1'] (expected ['k1','k2']) | read_it_later rows = 0 (expected 1)
```
**CONFIRMED — data loss.** Caller chain traced by me (the slice had left this inferred, which is why it filed P1):
```
Media/media_reading_scope_service.py:1799  import_reading_items        (async; runs the sync local service inline via _maybe_await)
Media/local_media_reading_service.py:2848  import_reading_items
                                     :2873  execute_reading_import_job
                                     :3383  _execute_reading_import_job   <- per-row loop, no db.transaction()
                                     :3485  _materialize_reading_import_row
                                     :3525  db.update_keywords_for_media(...)   <- bare DML on the held connection
DB/Client_Media_DB_v2.py:5530             "Assumes called within an existing transaction"
```
`awk` over each frame confirmed no enclosing `with db.transaction()` on the chain. The other two callers of these functions (`Library/meeting_speaker_rename.py:378/394`, `local_media_reading_service.py:4875`) *are* inside `db.transaction()` blocks. **Promoted to P0.**

## P1 — flashcard due-date, mixed timestamp shapes (DB-chacha)

```
$ PYTHONPATH=$WT $PY <SCRATCH>/repro_flashcard_due_day.py
T-form  next_review='2026-09-18 01:23:49+00:00' now='2026-09-18 02:23:49'  next_review<=now -> 0 ; count_due=0
space   next_review='2026-09-18 01:23:49'       now='2026-09-18 02:23:49'  next_review<=now -> 1 ; count_due=1
review 1h inside a 30-day window, stats(days=30)['reviews'] -> {'total_reviews': 0, 'avg_rating': None}   (control, further inside: total_reviews 1)
```
**CONFIRMED.** A card due an hour ago is not returned as due, and the stats window drops the boundary day.

Supporting census I ran separately (`verify_timestamp_formats.py`): 31 ChaChaNotes tables have `DEFAULT CURRENT_TIMESTAMP` columns (space separator); the Python writers store `2026-09-18T02:19:55.699Z`; `create_deck` and friends let the default fire, so `decks.created_at` is `2026-09-18 02:20:19`. One `ORDER BY` in the whole file normalises with `julianday()` (`:11456-11471`, conversation seek pagination, pinned by `Tests/DB/test_character_conversation_seek_pagination.py:141`); 29 other `ORDER BY <timestamp>` sites in that file and 100 across `DB/` + `Chat/` compare the text directly.

## P1 — `config.get_api_key()` placeholders + precedence split (ENTRY-config)

Scratch config `[api_settings.openai] api_key = "YOUR_KEY"`, `[api_settings.anthropic] api_key = " sk-ant-modern "`, env `ANTHROPIC_API_KEY=sk-ant-from-env`:
```
get_api_key('openai')           -> 'YOUR_KEY'          | bridge openai_api.api_key    -> None
get_detected_api_providers()    -> []
get_api_key('anthropic')        -> 'sk-ant-from-env'   | bridge anthropic_api.api_key -> 'sk-ant-modern'
```
**CONFIRMED, all three.** The placeholder is handed to five live spend/readiness call sites; `doctor`'s provider check can never see a configured provider; and the two accessors disagree about env-vs-config precedence with *both* orders pinned by test names (so that half is a decision needing an ADR, not a bug fix).

Same run also retires a long-standing belief: `get_cli_setting("chat.images","save_location","~/Downloads")` → the configured value, `("dictation.spoken_feedback", False)` → `True`, 1-arg dotted → the configured value. Dotted-section lookups work.

## P1 — MCP permission store resets on a transient read error (TOOLS-MCP)

```
$ PYTHONPATH=$WT $PY - <<'EOF'   # my own probe, <SCRATCH>/perm_probe2/
before: kill_switch= True | file exists: True
WARNING  MCP permission store at '.../mcp_permissions.json' is unreadable/corrupt ([Errno 13] Permission denied: ...); backing it up and resetting to defaults.
after chmod 000 load(): kill_switch= False | store exists: False | .bak exists: True
```
**CONFIRMED.** One EACCES moves the user's whole policy file aside and resolves from permissive defaults; the next mutator persists the reset.

## P1 — `CalculatorTool` is unbounded (TOOLS-MCP)

```
$ PYTHONPATH=$WT $PY -c "... asyncio.run(CalculatorTool().execute(expression=\"'ab' * 10**7\"))"
P4 CalculatorTool 'ab'*10**7 -> 0.002s, result len=20000000
$ ... expression="7**10**6"
ValueError: Exceeds the limit (4300 digits) for integer string conversion
```
**CONFIRMED.** A model-supplied expression allocates a 20 MB string in 2 ms (the slice measured 200 MB in 20 ms at `10**8`), and the `**` path computes a multi-million-bit integer before anything can reject it.

## P1 — streaming handlers leak the response on Stop (LLM)

```
$ PYTHONPATH=$WT $PY <SCRATCH>/llm_repro_generatorexit.py
openai      close() OK                                    response.close() called=True
anthropic   RuntimeError: generator ignored GeneratorExit  response.close() called=False
deepseek    RuntimeError: generator ignored GeneratorExit  response.close() called=False
groq        RuntimeError: generator ignored GeneratorExit  response.close() called=False
mistral     RuntimeError: generator ignored GeneratorExit  response.close() called=False
openrouter  RuntimeError: generator ignored GeneratorExit  response.close() called=False
google      RuntimeError: generator ignored GeneratorExit  response.close() called=False
cohere      RuntimeError: generator ignored GeneratorExit  response.close() called=False
```
**CONFIRMED — 7 of 8.** Same script also confirms two P2s: anthropic 429 → `requests.post calls=1 / Session.post calls=0` (the Retry adapter is mounted on a session the request never uses), and the deepseek stream yields `['Hello ', 'world', 'Hello world']` → a consumer that joins gets the summary twice.

## P1 — Gemini/Cohere streamed turns record no usage (LLM)

`usageMetadata` appears exactly once in all of `LLM_Calls/` (`LLM_API_Calls.py:4018`) and that read is on the **non-streaming** response; the Cohere `message-end` handler (`:2832-2860`) maps `finish_reason` and logs, extracting no usage. **CONFIRMED by reading** (the slice reproduced the dropped chunk with a mocked stream).

## P1 — `analyze("koboldcpp"|"tabbyapi")` always errors (LLM)

```
$ PYTHONPATH=$WT $PY -c "import inspect; from tldw_chatbook.LLM_Calls import Summarization_General_Lib as L; ..."
P4 isgeneratorfunction summarize_with_kobold   -> True
P4 isgeneratorfunction summarize_with_tabbyapi -> True
P4 isgeneratorfunction summarize_with_openai   -> False
```
**CONFIRMED.** Both are generator functions, so the caller's type check at `:807` returns `"Error: Unexpected result type …"`. Already filed as task-17387; the new evidence is that the recovery wrapper turns it into a user-visible error string.

## P1 — `doctor` reports every optional group missing (UTILS)

```
$ PYTHONPATH=$WT $PY -c "from tldw_chatbook.Utils.doctor import check_optional_dependencies as c; ..."
P4 UTILS P1 doctor: warn | 70 optional feature group(s) not installed: PIL, aiohttp, audio_processing, av, beautifulsoup4, ...
torch installed: True | registry torch: False | numpy installed: True | registry numpy: False
```
**CONFIRMED.** The visible face of task-25704/task-287 — cite those, don't re-file.

## P1 — Console left-rail Model section never updates (UI-chat)

```
$ grep -rn 'console-model-section-(provider|model)' tldw_chatbook/
tldw_chatbook/UI/Screens/chat_screen.py:9416   (the query)
tldw_chatbook/UI/Screens/chat_screen.py:9420   (the query)
$ PYTHONPATH=$WT $PY -m pytest <SCRATCH>/probes/test_probe_model_section_stale.py -q -s
PROBE before='0.60' after='0.60' provider_rows=[]
```
**CONFIRMED.** Nothing composes those two ids; the summary said temperature `9.87` and the row stayed `0.60`.

## P1 — `_open_video_with_os` always raises (UI-chat)

```
type in class dict: function
signature: (path: pathlib.Path) -> None
TypeError: ChatScreen._open_video_with_os() takes 1 positional argument but 2 were given
```
**CONFIRMED.** No `self`, no `@staticmethod`; every production call raises inside a swallowing `except`.

## P1 — idle credential poll costs 30–49 ms/tick (UI-chat)

```
$ PYTHONPATH=$WT $PY -m pytest <SCRATCH>/probes/test_probe_credential_poll_cost.py -q -s
PROBE readiness_ms=29.360 poll_tick_ms=30.245 app_config_ms=9.824 attach_reconciled=True
```
**CONFIRMED** (the slice measured 41/37/11.7 on its run). A 0.25 s timer spending ~30 ms is ~12 % of idle loop time.

## P1 — one Library/RAG settings keystroke costs 300–1,000 ms (UI-settings)

```
$ PYTHONPATH=$WT:$WT/Tests/UI $PY - <<'EOF'   # my own rebuild of the slice's probe
P4 keystroke#0: 1031.8 ms, get_cli_setting calls=170
P4 keystroke#1:  289.6 ms, get_cli_setting calls=32
P4 keystroke#2:  300.9 ms, get_cli_setting calls=32
```
**CONFIRMED** (the slice measured 1,343 ms then 388–426 ms). Runs on the calling thread, which in production is the event loop.

## P1 — media Reader keypress runs sync sqlite on the loop (UI-library)

```
$ PYTHONPATH=$WT $PY - <<'EOF'   # LibraryCollectionsDB + ReviewSetService, 500-item active set
P4 get_active_review_set (500 items): median 4.477 ms, p95 6.896 ms, items=500
```
**CONFIRMED** (the slice measured 5.064 / 6.415 ms and traced 2–4 such reads per `]`/`[` press).

## P1 — folder import blocks the UI (ENTRY-app) — **partly CONTESTED**

```
$ PYTHONPATH=$WT $PY <SCRATCH>/bench_ingest_submit.py
n=  100 store=True  listener=True   submit_total=    21.4 ms  (0.21 ms/job)
n= 1000 store=False listener=True   submit_total=  1646.1 ms  (1.65 ms/job)   <- O(n^2) listener
n= 1000 store=True  listener=False  submit_total=    30.5 ms  (0.03 ms/job)   <- per-job commit
n= 1000 store=True  listener=True   submit_total=  1702.9 ms  (1.70 ms/job)
```
The **O(n²) listener half reproduces exactly** (100 → 1,000 files is a 103× cost increase for a 10× input). The **per-job commit half does not**: the slice measured 2.80 ms/job, I measured 0.03 ms/job with the same script. That difference is environmental (fsync behaviour), so the finding stands on the listener and the commit claim is recorded as contested inside it. Corrected magnitude: ~1.7 s at the 1,000-file scan-limit maximum, ~21 ms at 100 files.

## Supporting measurements (not findings themselves)

```
P4 get_cli_setting: 11.125 ms/call | load_settings: 10.276 ms/call | load_cli_config_and_ensure_existence: 11.832 ms/call   (N=200, warm, isolated env)
```
This retires the 2026-07 performance audit's "config reads are cache-backed" as a *cost* claim — the value is cached, but each read pays a storage-admission handshake. Three P1s are downstream of it.

```
$ PYTHONPATH=$WT $PY -m pytest Tests/Architecture/test_screen_size_ratchet.py -q
FAILED ...[tldw_chatbook/UI/Screens/chat_screen.py]
FAILED ...[tldw_chatbook/UI/Screens/library_screen.py]
FAILED ...::test_task_22507_4_does_not_worsen_chat_screen_base
3 failed, 2 passed
```
Red at the reviewed SHA. The Library controllers' own ratchet is red in 16 of 43 rows (measured by the UIM-library slice).

```
$ PYTHONPATH=$WT $PY -c "... Utils.text.sanitize_filename('../../etc/passwd\x00\x07x.txt') ..."
Utils.text.sanitize_filename  -> '....etcpasswd\x00\x07x.txt'      (NUL and BEL survive)
path_validation.validate_filename -> ValueError: Filename cannot contain path separators
```

```
$ PYTHONPATH=$WT $PY <SCRATCH>/repro_conn_ctx_commit.py
control (transaction()-based read) title after abort: t
candidate (get_connection()-ctx read) title after abort: LEAKED
```

```
$ PYTHONPATH=$WT $PY <SCRATCH>/agents_probe_persist.py
RESULT: _persist RAISED OperationalError: database is locked
db calls: [('insert_steps_at_indices', 1), ('get_run',)]
terminal write attempted: False
```

Dead-helper re-check, independent of the slice that raised it (exact dotted-path `rg` over the package plus a `rg` over the 97,863-test collect-only inventory): `Utils.ui_helpers` 0/0 · `Utils.pagination` 0 (the word "pagination" appears in 70 test files; the *module* in none) · `Widgets.base_components` 0 (3 CSS-census tests name the file) · `Utils.cost_estimation` 0 (the collect hits are `test_evals_deletion_guard` asserting the removed widget stays removed) · `Utils.debug_helpers` 0 · `Utils.ingestion_preferences` 0 · `Utils.splash_animations` 0 · `truncate_content` 0/0 · `ensure_directory_exists` 0/0.

---

# Wave 2 — the six slices that finished after the second usage limit

## P1 — RAG search cache stops caching permanently (RAG) — **worse than filed**

`_prune_expired_async` (`RAG_Search/simplified/simple_cache.py:1055-1087`) deletes expired entries with a bare `del self._cache[key]` and never touches `_current_memory_bytes`, while the eviction paths at `:686` and `:1166` both decrement it.

```
after 40 put+expire cycles: entries=0  counter=540 KB  cap=1024 KB
cache STOPPED accepting writes at prune cycle 42: entries=0 counter=1022 KB cap=1024 KB
end: entries=0 counter=1022 KB cap=1024 KB
recovery path exists? ['clear', 'clear_async']   # neither is called automatically
```
**CONFIRMED, and worse than the slice filed it:** the cache is not degraded, it is a permanent no-op for the rest of the process — every RAG search after that point is a cold search.

## P1 — Transformers "Browse models dir" can never open a picker (EVENTS)

```
textual_fspicker installed: False
vendored (Third_Party/textual_fspicker): False
llm_management_events_transformers.py:153:  from textual_fspicker import FileOpen
```
**CONFIRMED.** The package is neither installed nor a dependency; the repo vendors a *differently named* module. The handler's own comment at `:27` claims the import is dynamic, which is true and does not help — it fails every time.

## P1 — local LLM servers are stopped without a process group (EVENTS)

```
start_new_session in server_lifecycle.py : 0
start_new_session elsewhere in package   : 17
server_lifecycle.py:488  process.terminate()
server_lifecycle.py:570  process = subprocess_module.Popen(command, **kwargs)
```
**CONFIRMED by reading.** A forked worker survives `terminate()`, so the app clears the handle and reports "stopped" while the port is still held. Nine other subprocess sites in this repo already use `start_new_session` + `killpg`.

## P1 — Console HTML-escapes text for a terminal surface (CHAT-rest-1)

```
'R&D Report'  -> 'R&amp;D Report'
'A &amp; B'   -> 'A &amp;amp; B'      # already-encoded text escaped twice
'<tag>'       -> '&lt;tag&gt;'
```
**CONFIRMED.** `Library/library_rag_state.py:392` documents this exact defect, found in live UAT and fixed there; the Console path still ships the pre-fix behaviour, pinned by a test that asserts the escaping.

## P1 — prompt search binds every match as one `IN (?,?,…)` list (DB-rest)

```
sqlite3 bound-parameter probe: 32766=ok  32767=OperationalError  40000=OperationalError
```
**CONFIRMED** (the limit; the 6.2 / 29.3 / 93.3 ms timings at 2k/10k/32k matches are the slice's). The same file's `search_library_prompts_page` already uses the correct subquery shape.

## P1 — `AgentRunsDB` reconcile runs from a compose path (DB-rest)

`AgentRuns_DB.py:285` calls `reconcile_orphaned_runs()` from `__init__` (guarded once per file per process); `settings_screen.py` constructs the DB inside `_render_detail_pane` (`:20025`), a compose generator. **CONFIRMED** (shape; the 295 ms at 250k step rows is the slice's).

## P1 — Library semantic search runs ChromaDB synchronously on the loop (RAG)

`rag_service.py:1592` calls the synchronous `vector_store.search_with_citations`; the caller chain reaches `library_screen.py:34872 async def _execute_library_rag_search`. **CONFIRMED by reading** (212 ms cold / 3.5 ms warm is the slice's measurement).

## Also confirmed in wave 2

- `pipeline_integration.py:122` imports `Event_Handlers.Chat_Events.chat_rag_events_simplified` — **the file does not exist**. A function-body import of a deleted module, exactly the class the prompt predicted mocked tests never catch.
- The Prompts sibling of the P0 transaction defect stays P2: `PromptsDatabase.update_keywords_for_prompt` has the same bare-DML shape, but its only public wrapper (`Prompt_Management/Prompts_Interop.py:230`) has **zero callers** anywhere in the package. A trap, not a live loss.
- `Media_Creation_Events/swarmui_events` is not merely dead but unimportable (`ImportError: cannot import name 'GenerationResult'`).

---

# Wave 3 — the eight slices added after the premature stop

These eight slices contained **three of the four P0s in this review**.

## P0 — Escape kills the app (W-library)

`Widgets/Library/library_prompts_canvas.py:549-555`: `more_actions_open` survives a recompose into the import branch, which never composes `#library-prompt-more-actions-region`; `on_key`'s unguarded `query_one` then raises `NoMatches` into `App._exception`. The slice reproduced it twice (a two-click sequence, then Escape). One-line root fix: reset the flag in `sync_state`.

## P0 — "Hide advanced" during a load kills the app (UIM-nav-mcp-persona)

Reproduced by the slice end-to-end through real `pilot.click`s → `WorkerFailed: NoMatches`. **Mechanism confirmed by me from source:**
```
:3294  run_worker(partial(self._load_advanced_section, ...), group="mcp-adv-section", exclusive=True)
:3418  self.query_one("#mcp-adv-content", Static).update(...)   <- after `await self._service.load_section(section)`
:1639  _hide_advanced ->  await collapsible.remove()             <- removes the subtree, cancels nothing
```
`exclusive=True` only cancels another worker in the *same group*, so the in-flight load resumes into a removed subtree. The toggle is re-enabled before the load is scheduled, so the window is the whole round trip.

## P0 — sync callable passed to `run_worker` (W-top)

```
:246 run_worker(self._get_devices_safe)  coroutine=False  thread_kwarg=False
```
Confirmed against Textual 8.2.8's own source:
```python
# textual/worker.py  Worker._run_async
elif callable(self._work):
    raise WorkerError("Request to run a non-async function as an async worker")
```
`Worker._run` catches that generically and `exit_on_error` defaults to `True`, so the app exits. Reachability traced by me: route `stts` (`screen_registry.py:194`) → `UI/Screens/stts_screen.py` → `STTSWindow` → `UI/Dictation_Window_Improved.py:732 _show_troubleshooting` pushes the dialog.

**Self-correction:** my own first enumeration reported *two* call sites. `:226` is `run_worker(self._initialize_audio())` — a `@work(exclusive=True, group=...)`-decorated **async** method, already called, and my regex stopped at the first `)`. `inspect.iscoroutinefunction` returns False on the decorated attribute, which is exactly the false positive a decorator produces; `inspect.unwrap` shows `unwrapped_coroutine=True`. Retracted before it reached the summary. One site, not two.

## P1 — `rich.markup.escape` does not protect Textual 8 (W-console-1 + W-library, independently)

```
rich RE_TAGS pattern: ((\\*)\[([a-z#/@][^[]*?)])

'[TODO] Q3 plan'    escape-> '[TODO] Q3 plan'    Content.from_markup(...).plain -> ' Q3 plan'
'[WIP] Draft plan'  escape-> '[WIP] Draft plan'  Content.from_markup(...).plain -> ' Draft plan'
'[IMPORTANT]'       escape-> '[IMPORTANT]'       Content.from_markup(...).plain -> ''
'[draft] x'         escape-> '\[draft] x'        Content.from_markup(...).plain -> '[draft] x'
```
**CONFIRMED.** Only tags starting `[a-z`, `#`, `/` or `@` are escaped; Textual 8's `Content.from_markup` eats every other bracket tag. A note or conversation titled `[IMPORTANT]` renders as an **empty row**.

Two slices found this independently — ~40 sites: Library media/conversations/trash/notes/prompts/skills/rail/Home row titles (shared `library_rail.py:377 _visible_row_title` + 27 more) and 12 Console sites (conversation titles, queued prompts, rewind previews, scope/tag/reaction labels, attachment filenames, tooltips, Canvas labels).

Two things make this worse than a normal escaping bug. The repo **already shipped the correct escaper** — `Library/library_rag_state.py:520 _escape_all_brackets`, whose own comment names the `[TODO]` case — and applies it only to RAG snippets. And every pinning test for the broken sites uses a **lowercase** tag, the one case the broken escape handles, so the suite is green.

The same class runs the other way at two sites: `Chat/console_display_state.py:93` and `Chat/console_prompt_queue.py:171` escape *into* markup-off surfaces, so `R&D Report` reaches the user as `R&amp;D Report` and `summarize [draft]` as `summarize \[draft]`.

## Also confirmed in wave 3

- `Widgets/confirmation_dialog.py:117` has no `markup=False`, so a title containing `[/…]` raises `MarkupError` inside `compose()` and the irreversible "Delete stored Full captures" confirmation never appears. 45 modules import that dialog.
- 21 of 66 top-level `Widgets/` modules (7,512 lines, 30 % of that slice) have zero production importers, with ~15 test files existing only to keep them alive.
- `Chat/document_generator.py` is dead in production and its dead code hides a real bug: `db.add_note(title, content, conversation_id)` passes the conversation id as the note's `note_id`, so a second document raises `ConflictError`.
- `Utils/Utils.py:253 truncate_content()` has zero importers against **52** inline `[:N] + "..."` re-rolls (up from the 24 measured in Tier-1 alone).
