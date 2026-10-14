# CHAT-rest-3 — tldw_chatbook/Chat/ (last third, non-console-core), 62 files, 40109 lines

STATUS: complete.

NOTE ON SLICE NAMING: the orchestrator brief called this slice "the `citation_*` subsystem".
`Chat/citation_trace_repository.py` exists but is NOT in my file list (it belongs to a sibling
slice). The confirmed sibling finding "1 `threading.Lock` + 62 `.execute(`, highest lock-to-SQL
ratio" was given against `citation_trace_repository.py`; the file with that shape in MY slice is
`console_trace_repository.py` (0 locks, 62 executes) plus `console_trace_runtime.py` (1 RLock).
I checked both, results below.

## Coverage

Honest totals: **7 files read in full** (2,456 lines), **20 sampled by named ranges** (~6,900 lines actually read of 22,900), **35 mechanical only** (the AST/grep passes named per row). ~9,400 of 40,109 lines read. I chose depth on the trace/settlement, marks, import-seam and credential paths over breadth; the weakest coverage is `console_voice_promotion.py` (1369) and `console_visual_evaluation.py` (1286, dead so deprioritised), then `console_voice_input.py`'s 1,200 unsampled lines.

| file | lines | status |
| --- | --- | --- |
| console_trace_repository.py | 3809 | sampled by symbol cluster: 290-520 (validators + `import_post_dispatch_trace`), 1987-2135 (lineage readers), 2326-2510 (redaction spans), 3660-3740 (epoch/selects); the 793-1700 post-dispatch write/reconcile block read only at its transaction boundaries |
| console_trace_runtime.py | 686 | sampled: 40-60 (`_utc_now`), 82-100 (`RLock` init), 150-235 (all three lock+transaction regions), 380-400, 440-460, 530-545 (the flagged SQL) |
| console_trace_settlement.py | 977 | **read in full** |
| console_transaction_contribution.py | 225 | sampled: read for `with ... get_connection()` DML (none) — otherwise mechanical only (AST: function-body-import resolution + `re.compile`-in-body; grep: locks, `get_cli_setting`, UTC/timestamp helpers, logger-vs-secret, `object_pairs_hook`, reachability) |
| console_turn_context.py | 674 | sampled: 35-190 (consent seam + `_detached_selection`) |
| console_turn_grouping.py | 250 | mechanical only (AST: function-body-import resolution + `re.compile`-in-body; grep: locks, `get_cli_setting`, UTC/timestamp helpers, logger-vs-secret, `object_pairs_hook`, reachability) |
| console_turn_preparation.py | 851 | sampled: 1-60 (bounds/validators) — otherwise mechanical only |
| console_visual_benchmark.py | 242 | reachability-verified only (transitively dead) |
| console_visual_evaluation.py | 1286 | reachability-verified (0 prod importers) + symbol index read; body NOT read — dead module, see task-19571 |
| console_visual_transcript.py | 669 | mechanical only (AST: function-body-import resolution + `re.compile`-in-body; grep: locks, `get_cli_setting`, UTC/timestamp helpers, logger-vs-secret, `object_pairs_hook`, reachability) |
| console_voice_attempts.py | 928 | sampled: 250-330, 580-620, 885-928 (cancellation/teardown; all except-candidates) |
| console_voice_controls.py | 8 | **read in full** (8 lines) |
| console_voice_eligibility.py | 39 | **read in full** (39 lines) |
| console_voice_input.py | 2264 | sampled: 140-200, 280-420 (config resolution), 780-1210 (warm-up/config readers), 1330-1420 + 1960-2230 (`_state_lock` regions) |
| console_voice_preflight.py | 56 | **read in full** (56 lines) |
| console_voice_process.py | 1129 | sampled: 60-80 (`threading.Lock`), 120-130, 600-615 (the flagged function-body import) — otherwise mechanical only (AST: function-body-import resolution + `re.compile`-in-body; grep: locks, `get_cli_setting`, UTC/timestamp helpers, logger-vs-secret, `object_pairs_hook`, reachability) |
| console_voice_process_effects.py | 861 | sampled: 820-861 (drain/teardown) — otherwise mechanical only |
| console_voice_promotion.py | 1369 | sampled: 30-40 (`_required_string`) — otherwise mechanical only (AST: function-body-import resolution + `re.compile`-in-body; grep: locks, `get_cli_setting`, UTC/timestamp helpers, logger-vs-secret, `object_pairs_hook`, reachability). **Weakest coverage in this slice, together with console_visual_evaluation** |
| console_voice_settings.py | 319 | mechanical only (AST: function-body-import resolution + `re.compile`-in-body; grep: locks, `get_cli_setting`, UTC/timestamp helpers, logger-vs-secret, `object_pairs_hook`, reachability) |
| console_voice_supervisor.py | 217 | sampled: 45-200 |
| console_voice_trace_gateway.py | 1496 | sampled: 30-60, 335-360 (`_utc_now`/`_thaw`), 440-480 + 535-710 (dispatch/lifecycle), 955-1110 (`ProvisionalTraceRegistry` under `RLock`); 1110-1496 read as the lock-discipline index only (every `_locked` helper enumerated, bodies not read) |
| console_voice_trace_promotion.py | 781 | sampled: 1-110 (validators), 440-600 (`PostDispatchTraceCall.__post_init__`) |
| console_voice_tts_bridge.py | 653 | mechanical only (AST: function-body-import resolution + `re.compile`-in-body; grep: locks, `get_cli_setting`, UTC/timestamp helpers, logger-vs-secret, `object_pairs_hook`, reachability) |
| console_voice_worker.py | 418 | sampled: 190-418 (both lock-holding classes) |
| console_workspace_actions.py | 157 | mechanical only (AST: function-body-import resolution + `re.compile`-in-body; grep: locks, `get_cli_setting`, UTC/timestamp helpers, logger-vs-secret, `object_pairs_hook`, reachability) |
| console_worktree_recovery.py | 399 | mechanical only (AST: function-body-import resolution + `re.compile`-in-body; grep: locks, `get_cli_setting`, UTC/timestamp helpers, logger-vs-secret, `object_pairs_hook`, reachability) |
| conversation_archive_actions.py | 311 | mechanical only (AST: function-body-import resolution + `re.compile`-in-body; grep: locks, `get_cli_setting`, UTC/timestamp helpers, logger-vs-secret, `object_pairs_hook`, reachability) |
| conversation_local_marks_service.py | 540 | **read in full** |
| cost_display.py | 190 | mechanical only (AST: function-body-import resolution + `re.compile`-in-body; grep: locks, `get_cli_setting`, UTC/timestamp helpers, logger-vs-secret, `object_pairs_hook`, reachability) |
| custom_endpoint_registry.py | 549 | sampled: 90-145 (the pydantic boundary model + validators) |
| document_generator.py | 601 | **read in full** |
| library_activity.py | 719 | sampled: 295-310, 480-515, 680-700 (`unique_object`, `_value`/`_field`) — otherwise mechanical only (AST: function-body-import resolution + `re.compile`-in-body; grep: locks, `get_cli_setting`, UTC/timestamp helpers, logger-vs-secret, `object_pairs_hook`, reachability) |
| library_preparation.py | 302 | sampled: 165-200 (`unique_object`) — otherwise mechanical only (AST: function-body-import resolution + `re.compile`-in-body; grep: locks, `get_cli_setting`, UTC/timestamp helpers, logger-vs-secret, `object_pairs_hook`, reachability) |
| llamacpp_think_filter.py | 261 | mechanical only (AST: function-body-import resolution + `re.compile`-in-body; grep: locks, `get_cli_setting`, UTC/timestamp helpers, logger-vs-secret, `object_pairs_hook`, reachability) |
| local_reasoning.py | 367 | mechanical only (AST: function-body-import resolution + `re.compile`-in-body; grep: locks, `get_cli_setting`, UTC/timestamp helpers, logger-vs-secret, `object_pairs_hook`, reachability) |
| local_server_discovery.py | 589 | **read in full** |
| message_metadata.py | 656 | mechanical only (AST: function-body-import resolution + `re.compile`-in-body; grep: locks, `get_cli_setting`, UTC/timestamp helpers, logger-vs-secret, `object_pairs_hook`, reachability) |
| permission_summary_service.py | 337 | mechanical only (AST: function-body-import resolution + `re.compile`-in-body; grep: locks, `get_cli_setting`, UTC/timestamp helpers, logger-vs-secret, `object_pairs_hook`, reachability) |
| prompt_history.py | 309 | mechanical only (AST: function-body-import resolution + `re.compile`-in-body; grep: locks, `get_cli_setting`, UTC/timestamp helpers, logger-vs-secret, `object_pairs_hook`, reachability) |
| prompt_template_manager.py | 138 | reachability-verified only (0 prod importers/mentions) |
| provider_catalog.py | 82 | sampled: the 7 `legacy` markers — otherwise mechanical only (AST: function-body-import resolution + `re.compile`-in-body; grep: locks, `get_cli_setting`, UTC/timestamp helpers, logger-vs-secret, `object_pairs_hook`, reachability) |
| provider_continuation.py | 771 | sampled: 190-215, 430-460, 555-700 (all 4 except-candidates + the parse boundary) |
| provider_endpoint_contract.py | 443 | mechanical only (AST: function-body-import resolution + `re.compile`-in-body; grep: locks, `get_cli_setting`, UTC/timestamp helpers, logger-vs-secret, `object_pairs_hook`, reachability) |
| provider_failures.py | 126 | **read in full** |
| provider_readiness.py | 751 | sampled: the 5 `legacy` markers + the 3 function-body imports — otherwise mechanical only (AST: function-body-import resolution + `re.compile`-in-body; grep: locks, `get_cli_setting`, UTC/timestamp helpers, logger-vs-secret, `object_pairs_hook`, reachability) |
| provider_setup_persistence.py | 1838 | sampled: 1-130 (constants/ownership), 225-250, 400-500 (credential reprs), 1820-1838; scanned mechanically for logging + secret leaks |
| provider_test_evidence.py | 1261 | mechanical only (logging/secret sweep: 0 logger refs) |
| provider_usage.py | 290 | mechanical only (AST: function-body-import resolution + `re.compile`-in-body; grep: locks, `get_cli_setting`, UTC/timestamp helpers, logger-vs-secret, `object_pairs_hook`, reachability) |
| rag_scope.py | 632 | mechanical only (AST: function-body-import resolution + `re.compile`-in-body; grep: locks, `get_cli_setting`, UTC/timestamp helpers, logger-vs-secret, `object_pairs_hook`, reachability) |
| reply_sentence_sequencer.py | 579 | sampled: checked for per-token `get_cli_setting`/`re.compile` (none) — otherwise mechanical only (AST: function-body-import resolution + `re.compile`-in-body; grep: locks, `get_cli_setting`, UTC/timestamp helpers, logger-vs-secret, `object_pairs_hook`, reachability) |
| sampling_params.py | 325 | mechanical only (AST: function-body-import resolution + `re.compile`-in-body; grep: locks, `get_cli_setting`, UTC/timestamp helpers, logger-vs-secret, `object_pairs_hook`, reachability) |
| scope_picker_listers.py | 541 | **read in full** |
| server_chat_conversation_service.py | 335 | sampled: the 8 function-body imports (all resolved by real import) — otherwise mechanical only (AST: function-body-import resolution + `re.compile`-in-body; grep: locks, `get_cli_setting`, UTC/timestamp helpers, logger-vs-secret, `object_pairs_hook`, reachability) |
| server_chat_loop_service.py | 82 | **read in full** |
| stream_stall_watchdog.py | 232 | sampled: 170-185 (`_REGISTRY_LOCK`) — otherwise mechanical only (AST: function-body-import resolution + `re.compile`-in-body; grep: locks, `get_cli_setting`, UTC/timestamp helpers, logger-vs-secret, `object_pairs_hook`, reachability) |
| thinking_blocks.py | 434 | sampled: 1-40 (bounds), 200-240 (json seam) |
| trace_export_profiles.py | 55 | mechanical only (AST: function-body-import resolution + `re.compile`-in-body; grep: locks, `get_cli_setting`, UTC/timestamp helpers, logger-vs-secret, `object_pairs_hook`, reachability) |
| trajectory.py | 1859 | sampled: 110-250 (path redaction), 325-440 (`_field`/`_parse_timestamp`/`_normalize_message`), symbol index for the rest; the 440-1900 derivation/graph block read only at function signatures |
| trajectory_export.py | 1508 | sampled: 60-215, 1040-1100, 1240-1345 (build/atomic-write/timestamps); rest mechanical only |
| trajectory_import.py | 1124 | sampled: 1-260 (read/validate seam), 941-1124 (record/snapshot builders); 260-941 (`_validate_v2`) mechanical only |
| usage_recorder.py | 122 | mechanical only (AST: function-body-import resolution + `re.compile`-in-body; grep: locks, `get_cli_setting`, UTC/timestamp helpers, logger-vs-secret, `object_pairs_hook`, reachability) |
| voice_phrase_sequencer.py | 87 | mechanical only (AST: function-body-import resolution + `re.compile`-in-body; grep: locks, `get_cli_setting`, UTC/timestamp helpers, logger-vs-secret, `object_pairs_hook`, reachability) |

## Findings

### P1 [D1] — `load_imported_trace()` lets `UnicodeDecodeError` and `RecursionError` escape the import seam; the only caller catches `TrajectoryImportError` alone, so picking the wrong file in the trace-import picker raises out of a Textual action handler
- Where: `tldw_chatbook/Chat/trajectory_import.py:80-108` (`_read_document`: `path.read_text(encoding="utf-8")` guarded by `except OSError` only; `json.loads` guarded by `except json.JSONDecodeError` only). Caller: `tldw_chatbook/UI/Screens/trajectory_screen.py:2026-2029` (`await asyncio.to_thread(load_imported_trace, path)` / `except TrajectoryImportError`). The picker at `:2054-2060` deliberately offers an **"All Files"** filter alongside "Trace files".
- Evidence: `cd /Users/macbook-dev/Documents/GitHub/tldw-review && source <SCRATCH>/env.sh && PYTHONPATH=$WT $PY <<'EOF' ...` (script inline in the run; writes a latin-1 `.json` and a 20000-deep `[[[...]]]` `.json`, then calls `load_imported_trace`) ->
  ```
  1) *** UNHANDLED UnicodeDecodeError : 'utf-8' codec can't decode byte 0xe9 in position 50: invalid continuation byte
  2) *** UNHANDLED RecursionError : maximum recursion depth exceeded while decoding a JSON array from a unicode string
  3) handled TrajectoryImportError: Cannot read trajectory trace file '/var/folders/...   <- the OSError path IS handled
  ```
  Row 3 is the negative control: the author handled unreadable files, and missed decode failures.
- Why it matters: `action_open_trace`'s own docstring promises "Import failures surface as an error notification carrying the actionable message from the shared validator". For any non-UTF-8 file the user picks through "All Files" (a PNG, a `.sqlite`, a latin-1 JSON) that contract is broken and the exception leaves the action handler. `app.py:19025 _handle_exception` (TASK-32533) keeps the *screen* alive only when the raise came through `_PUMP_DISPATCH_FRAMES` with `pump is not self`; otherwise it calls `super()._handle_exception` and **the app exits**. Which branch an async binding-action takes is the unverified part (see Left UNVERIFIED) — the escape itself is verified.
- Recommended correction: widen the two guards in `_read_document` to the idiom this repo already uses ~15 times — `except (OSError, UnicodeDecodeError)` on the read and `except (json.JSONDecodeError, RecursionError, ValueError)` on the decode — re-raising `TrajectoryImportError`. Exact precedent in this same slice: `Chat/library_activity.py:501`, `Chat/library_preparation.py:190`, `Chat/local_server_discovery.py:451` (`except (RecursionError, UnicodeDecodeError, ValueError)` on the same `json.loads` shape), `Chat/trajectory.py:153`.
- Size: S · ADR: no · Confidence: **verified** (escape), inferred (crash-vs-toast)
- Pinning test: none. `grep -rn "UnicodeDecodeError\|RecursionError" Tests/Chat/test_trajectory_import*.py` -> no hits.
- Already covered: none

### P2 [D1] — `conversation_local_marks.updated_at` receives two different UTC string shapes, and the table's only ORDER BY is lexical
- Where: writer A `Chat/conversation_local_marks_service.py:93-95` (`_now()` -> `datetime.now(timezone.utc).isoformat()`, **microsecond** precision, or *no* fraction when `microsecond == 0`); writer B `Chat/chat_persistence_service.py:453` and `:879` and `Chat/console_dispatch_repository.py:1058,1110` (`db._get_current_utc_timestamp_iso()` -> `DB/ChaChaNotes_DB.py:8700-8713`, **millisecond** precision). Both funnel into the same INSERT at `conversation_local_marks_service.py:258-268`. Lexical readers: `:427` `ORDER BY updated_at DESC`, `:455` same.
- Evidence: `cd /Users/macbook-dev/Documents/GitHub/tldw-review && source <SCRATCH>/env.sh && PYTHONPATH=$WT $PY <SCRATCH>/repro_marks_two_shapes.py` ->
  ```
  service._now()                    -> 2026-09-18T15:40:07.404621Z
  db._get_current_utc_timestamp_iso -> 2026-09-18T15:40:07.404Z
  stored shapes in one column: [('conv-ms', '2026-09-18T12:34:56.789Z'), ('conv-us', '2026-09-18T12:34:56.789012Z')]
  ORDER BY updated_at DESC -> ['conv-ms', 'conv-us']
  expected newest-first     -> ['conv-us', 'conv-ms']  (conv-us is 12us later)
  INVERTED
  ```
  Mechanism: on a shared `SS.mmm` prefix the ms form continues with `'Z'` (0x5A) and the us form with a digit (0x30-0x39), so the **older** ms row sorts as newer.
- Why it matters: two shapes are already committed to one column. The inversion is reachable today only *within* one `mark_type`, and no `mark_type` is currently written by both helpers (`starred`/`fleet_unseen` -> writer A only via `set_mark`; `console_unseen:*`/`console_terminal_outcome:*` -> writer B only via `set_console_terminal_with_cursor`) - so this is a live shape split with a latent ordering bug one writer away, not a bug a user hits today. The `# task-15471` comment at `:65-84` asserts "every star and fleet mark in the process goes through this instance", which is true of `set_mark` but not of `set_mark_with_cursor`, the public entry the two ms-shape callers use.
- Recommended correction: make `ConversationLocalMarksService._now()` delegate to `db._get_current_utc_timestamp_iso()` (or move that helper to `Utils/` and have both call it), so the column has one shape. S, mechanical.
- Size: S · ADR: no · Confidence: **verified**
- Pinning test: none found asserting either shape; `Tests/Chat/test_conversation_local_marks_service.py` exercises `get_mark`/ordering but not the string format.
- Already covered: none (this is an instance of the review-wide timestamp class, headline finding).

### P2 [D1/D2] — `trajectory_import._read_document` reads a user-picked file with `Path.read_text()` and no size ceiling
- Where: `Chat/trajectory_import.py:89-91`
- Evidence: read only — no `st_size` check, no streaming; the whole file is materialised as one `str` and then again as a parsed object. `grep -n "MAX_\|st_size" trajectory_import.py` -> no hits (the only cap constants in the trio are `PREVIEW_MAX_CHARS` in `trajectory_export.py`).
- Why it matters: same picker, same "All Files" filter; a multi-GB pick is a `MemoryError` that escapes exactly as above. Compare `Chat/local_server_discovery.py:39` `MODEL_PROBE_RESPONSE_MAX_BYTES = 1MB`, which bounds a *remote* read on the same kind of JSON.
- Recommended correction: `path.stat().st_size` ceiling before the read, raising `TrajectoryImportError`; same constant style as `MODEL_PROBE_RESPONSE_MAX_BYTES`.
- Size: S · ADR: no · Confidence: inferred (command that would settle it: write a 3 GB `.json` under `<SCRATCH>` and call `load_imported_trace` on it)
- Pinning test: none
- Already covered: none

### P2 [D2] — the scope picker's tag vocabulary rebuilds the whole notes keyword-usage table on every tag query, although the expensive part does not depend on the query
- Where: `Chat/scope_picker_listers.py:462-483` (`_notes_tag_usage`: 1 `list_keywords(limit=200)` + up to **200** `get_notes_for_keyword(kid, limit=1000)` calls, each materialising up to 1000 full note rows only to take `len()`), called from `:519-524` inside `build_keyword_tag_lister`'s `_tag_lister(query)`. The `query` is used only at `:528-537`, an in-memory substring filter over the already-built `counts` dict.
- Evidence: `cd /Users/macbook-dev/Documents/GitHub/tldw-review && source <SCRATCH>/env.sh && PYTHONPATH=$WT $PY` with a real `CharactersRAGDB` seeded 200 keywords x 20 notes ->
  ```
  caps: vocab=200 usage_sample=1000
  200 keywords x 20 notes each -> _notes_tag_usage took 22.4 ms, 200 tags
  second identical call (no cache): 22.1 ms
  ```
  20 notes/keyword is 2% of the `_NOTES_TAG_USAGE_SAMPLE_LIMIT` ceiling; cost is linear in rows materialised, so a user at the cap pays ~50x that per query.
- Why it matters: `Widgets/Console/console_scope_picker_modal.py:739` debounces and `:749` runs it `exclusive=True, group=TAG_SEARCH_WORKER_GROUP`, so this is off-loop and de-duplicated within a keystroke burst — but every *settled* query re-pays the full scan for a result that is identical each time. The modal also calls `_tag_lister("")` at `:514` on open.
- Recommended correction: compute `counts` once per modal (or memoise it on the closure with a short TTL) and apply only the substring filter per query. The docstring at `:447-449` already names the root cause ("no single-query usage-stats seam exists on the notes side, unlike media's `get_keyword_usage_stats`") — a `get_keyword_usage_stats` equivalent on `ChaChaNotes_DB` is the other, larger fix.
- Size: S (memoise) / M (add the notes-side stats query) · ADR: no · Confidence: **verified** (measured)
- Pinning test: none found.
- Already covered: none

### P2 [D3] — `Chat/document_generator.py` (601 lines) has zero production callers; the dead code hides a real API misuse that would corrupt note ids the day it is wired
- Where: whole module `Chat/document_generator.py`; the defect is at `:595` — `self.db.add_note(title, full_content, conversation_id)` passes a conversation id into `CharactersRAGDB.add_note`'s third **positional** parameter, which is `note_id`, not a conversation field.
- Evidence:
  1. Reachability: `grep -rn --include='*.py' "document_generator" tldw_chatbook/ Tests/ | grep -i import` -> six hits, **all in `Tests/`** (`Tests/Chat/test_thinking_privacy_surfaces.py:33`, `test_provider_continuation_privacy.py:19`, `test_assistant_generation_state_roundtrip.py:19`, `Tests/Internal_Prompts/test_document_generation_migration.py:45,297`, `Tests/Utils/test_config_nested_settings.py:163`). The only production mention is a *comment* in `Internal_Prompts/document_generation_prompts.py:12-22`.
  2. The defect, run for real:
  ```
  add_note signature: (self, title: str, content: str, note_id: Optional[str] = None, *, cursor: sqlite3.Cursor | None = None) -> Optional[str]
  1st note id returned: conv-abc-123  <- equals the conversation_id, not a generated note id
  note row id: conv-abc-123 title: T1
  2nd note for same conversation RAISED: ConflictError Note with ID 'conv-abc-123' already exists. (Entity: notes, ID: conv-abc-123)
  ```
  So the first generated document silently takes the conversation's id as its note primary key, and the second document for that conversation is refused outright.
- Why it matters: 601 lines carry their own provider table (`chat_with_openai/anthropic/cohere/groq/openrouter/deepseek`, `:104-120`) that duplicates the dispatcher in `Chat/Chat_Functions.py`, three `get_cli_setting` prompt tables, and a `pyperclip` dependency — all maintained by tests alone. The `ConflictError` above is the proof that no caller has ever exercised it. Same module, `:589`: `datetime.now().isoformat()` — **naive local time** written into a user-visible `Generated: ...` note header, the only non-UTC timestamp I found in this slice (another instance of the review's timestamp class, currently unreachable).
- Recommended correction: delete the module and the six test imports, or wire it and fix `:595` to `self.db.add_note(title, full_content)` plus `:589` to `datetime.now(timezone.utc)`. A deletion needs a call, not a code review — the tests that import it assert things about *prompt migration*, not about this class.
- Size: S (delete) / M (wire + fix) · ADR: no · Confidence: **verified**
- Pinning test: `Tests/Internal_Prompts/test_document_generation_prompt_parity.py` and `test_document_generation_migration.py` pin the module's *prompt literals* as a requirement, so the file cannot be deleted without touching them. Nothing pins `create_note_with_metadata`.
- Already covered: none

### P2 [D3] — 2,349 of this slice's 40,109 lines are in modules with zero production importers; only one of the five is in the census that owns the decision
- Where, with the check that resolved each (production = `tldw_chatbook/`, excluding the module itself and `Tests/`):
  | module | lines | production importers | kept alive by |
  | --- | --- | --- | --- |
  | `Chat/console_visual_evaluation.py` | 1286 | **0** | `Tests/Chat/test_console_visual_evaluation.py`; also has a row in the `GatewayCallsiteRecord` census at `Chat/console_trace_provenance.py:384-391` (a *string* path, not an import) |
  | `Chat/console_visual_benchmark.py` | 242 | 1 — and that one is `console_visual_evaluation.py:39`, i.e. transitively dead | `Tests/Chat/test_console_visual_transcript.py` |
  | `Chat/document_generator.py` | 601 | **0** | 6 test files (see the finding above) |
  | `Chat/prompt_template_manager.py` | 138 | **0** — zero mentions anywhere in `tldw_chatbook/` | `Tests/Chat/test_prompt_template_manager.py`; `Docs/Development/Developer_Guide.md:49` still documents it as live |
  | `Chat/server_chat_loop_service.py` | 82 | **0 consumers of `ServerChatLoopService`**, but `Chat/__init__.py:9` re-exports it, so it is imported at **boot** (it appears in `Tests/Performance/boot_budget_snapshots/boot_import_modules.txt`) | `Tests/Chat/test_server_chat_loop_service.py` |
- Evidence: `grep -rn --include='*.py' "<module>" tldw_chatbook/ | grep -v "Chat/<module>.py:"` per row (outputs above); plus `grep -rn --include='*.py' "ServerChatLoopService" tldw_chatbook/ Tests/` -> production hits are only the `__init__.py` re-export and the class definition.
- Why it matters: `console_visual_evaluation.py:833` makes a real `stream_chat` gateway call and is registered in the production gateway-callsite census, so a derived-artifact check now pins a dead module in place. `server_chat_loop_service` costs boot-import time for a class nothing constructs.
- Recommended correction: **none from me** — `task-19571` ("Decide once what to do with 170 unreachable modules, 78 of which still carry a test suite", status `To Do`) owns this as a wire-or-retire policy decision and explicitly names `Chat/console_visual_evaluation.py` (1,285) in its subsystem table. The useful output of this review is the four rows its table does *not* name: `console_visual_benchmark.py`, `document_generator.py`, `prompt_template_manager.py`, `server_chat_loop_service.py`. `server_chat_loop_service` is the one with a cost attached today (boot import) and is a one-line `__init__.py` fix independent of the policy.
- Size: S (drop the `__init__` re-export) / L (the census decision) · ADR: no · Confidence: **verified**
- Pinning test: each row has one; that is the census's own complaint.
- Already covered: **task-19571**

### P2 [D4a] — `Utils/Utils.py:253 truncate_content()` has **zero importers** while 52 inline re-rolls of exactly its body live in the package; my slice's copy is byte-identical in behaviour
- Where: helper `tldw_chatbook/Utils/Utils.py:253-261` (`content[: max_length - 3] + "..."`). Copy in this slice: `Chat/provider_failures.py:50-53` (`if len(detail) > 240: detail = detail[:237] + "..."`). Nearest neighbours outside this slice: `Chat/console_generate_image.py:230`, `Chat/console_fleet_wake.py:132`, `UI/Research_Modules/bundle_rendering.py:99`, `Tools/code_audit_tool.py:286,429,487,531`, `UI/STTS_Window.py:568,679,842,1428`, `UI/Evals/bench_editor.py:249`, `UI/Evals/inspector.py:213`, `UI/MCP_Modules/mcp_inspector.py:458`.
- Evidence:
  - `grep -rn --include='*.py' "truncate_content" tldw_chatbook/ Tests/ | grep -v "Utils/Utils.py:"` -> **no output**: the helper is dead.
  - `grep -rn --include='*.py' -E '\[: *[A-Za-z_0-9]+ *(- *[0-9]+)? *\] *\+ *"(\.\.\.|…)"' tldw_chatbook/ | wc -l` -> **52**
  - Behavioural identity for this slice's copy: `truncate_content(s, 240) == s[:237] + "..."` -> `True`, `len == 240`.
- Why it matters: this is the brief's "dead helper with >=10 re-rolls" shape exactly. The drift is real and user-visible in the copies: some use `"..."`, some `"…"` (one codepoint vs three), some truncate to `limit` and some to `limit - 1`, and several (`UI/STTS_Window.py:679,842`) append the ellipsis *unconditionally*, so a 40-character string is shown as `"short text..."` with nothing elided.
- Recommended correction: import `Utils.Utils.truncate_content` at the copies whose semantics match (the `-3` family), or move a single `truncate(text, limit, *, marker="…")` into `Utils/Utils.py` beside `elide_path_middle` and retire the rest. Canonical home: `Utils/Utils.py`, where the helper already is.
- Size: M (52 call sites, and the `"…"` vs `"..."` marker has to be chosen) · ADR: no · Confidence: **verified**
- Pinning test: none on `truncate_content` (`grep` above returns nothing from `Tests/` either).
- Already covered: none. NOTE: this is a repo-wide class; a sibling slice reviewing `UI/` or `Tools/` will see the same helper. Reported here because the census is in my evidence and `provider_failures.py:53` is my row.

### P3 [D3] — `ConversationLocalMarksService.get_mark()` has no production caller; its docstring documents a delivery feature that was never wired
- Where: `Chat/conversation_local_marks_service.py:349-390` (docstring at `:353-358` claims "PR3a-2 Task 5: the auto-wake mount-claim uses `created_at` as the since-when boundary")
- Evidence: `grep -rn --include='*.py' "get_mark(" tldw_chatbook/` -> one hit, the definition itself. Test-only callers: `Tests/UI/test_console_agent_progress.py:93,111`, `Tests/Chat/test_conversation_local_marks_service.py:273+`.
- Why it matters: the docstring is the only statement of an auto-wake boundary rule; a reader will believe it is enforced somewhere.
- Recommended correction: delete the method, or strip the claim to what the method does.
- Size: S · ADR: no · Confidence: verified
- Pinning test: `Tests/Chat/test_conversation_local_marks_service.py` pins the behaviour (so it is a *decision* to keep the API), but nothing pins the docstring's claim.
- Already covered: none



### P3 [D4b] — the strict-JSON parser (duplicate-key rejection + constant rejection) is hand-rolled 49 times across the package, twice inside this slice as 30-line structural clones, with no `Utils/` home
- Where (this slice): `Chat/provider_continuation.py:205-221` and `Chat/thinking_blocks.py:221-235` are the same function with a different `_fail` arity; both are followed by a near-identical `_exact_mapping`. Also `Chat/library_activity.py:487-502`, `Chat/library_preparation.py:174-193`. Adjacent `Chat/` copies outside my slice: `console_dispatch_checkpoint.py:275-292`, `console_trace_regex_worker.py:58-67`.
- Evidence: `grep -rn --include='*.py' "object_pairs_hook" tldw_chatbook/ | wc -l` -> **49**; `grep -rn --include='*.py' "parse_constant" tldw_chatbook/ | wc -l` -> **40**; `grep -rn --include='*.py' "def strict_json\|strict_json_loads\|def loads_strict" tldw_chatbook/` -> **four separate private `_strict_json_loads` definitions** (`Chat/thinking_blocks.py:221`, `Chat/provider_continuation.py:204`, `LLM_Calls/qwencloud_streaming.py:65`, `LLM_Calls/hosted_chat.py:673`) plus two public near-twins (`Chat/console_generation_settings_metadata.py:343 strict_json_metadata_object`, `Tool_Packs/contracts.py:969 strict_json_object`). Nothing in `Utils/`.
- Why it matters (the drift, which is the D4 test): the *exception guard* around the call differs per copy. `library_activity.py:501`, `library_preparation.py:190` and `console_dispatch_checkpoint.py:291` catch `(TypeError, ValueError, json.JSONDecodeError, RecursionError)`; `provider_continuation.py` and `thinking_blocks.py` rely on a caller-level `except Exception` (which does cover `RecursionError`, since it subclasses `RuntimeError` — I checked). The only copy whose guard is *wrong* is in `Chat/trajectory_import.py` and is written up as the P1 above. That is the concrete cost of 49 hand-rolls: one of them is missing a case and nothing makes that visible.
- Recommended correction: one `Utils/strict_json.py` helper — `strict_json_loads(text, *, error=ValueError, message=...)` that bundles `parse_constant`, `object_pairs_hook` and the decode-exception tuple — and convert the copies as the modules are next touched. Canonical home: `Utils/`, per the brief's preference.
- Size: L (49 sites, 6+ distinct exception contracts) · ADR: no (a `Utils/` helper is not a cross-module interface change) · Confidence: **verified** (the census; the drift claim is checked at the six `Chat/` sites, not at all 49)
- Pinning test: each copy is pinned by its own module's tests; no test names the shared behaviour.
- Already covered: none

## Candidate dispositions

| candidate (file:line pattern) | disposition |
| --- | --- |
| **dotted_section_setting (5)** | |
| console_voice_input.py:907 `dictation.warm_model_before_capture` | **retired** — per brief, dotted lookups resolve since TASK-1771; read the function: session-start read, not a hot path |
| console_voice_input.py:965 `dictation.acoustic_barge_in` | **retired** — same |
| document_generator.py:76 / :85 / :94 `prompts.document_generation.*` | **retired as a config finding**; the module they live in is dead — see the `document_generator` finding |
| **dup_shape (7)** | |
| `aclose@console_voice_process.py:125` + 7 others | **retired** — one-line trampolines with unrelated bodies; a shape match, not a duplication |
| `unique_object` x7 (library_activity:487, library_preparation:174, console_dispatch_checkpoint:275, console_trace_regex_worker:58, watchlists_tool_service:1257, workspace_tool_protocol:407, permission_store:388) | **confirmed** — see the strict-JSON D4b finding (census widened to 49 `object_pairs_hook` sites) |
| `_require_ledger@console_agent_bridge:2827` + 6 | **retired** — guard-clause shape, different subjects |
| `_field@trajectory_export:147` / `_value@library_activity:302` / `_field@library_activity:685` + 3 | **confirmed (P3)** — the "read a name from a Mapping-or-object" accessor, 6 copies; canonical form is `Chat/trajectory.py:328 _field` (the only one with a docstring naming it "the single seam that tolerates them all"). No behavioural drift found; consistency only |
| `_clean_text`/`_optional_text`/`_normalize_optional_text`/`_text_or_none`/`_cleaned_path_setting` (5) | **confirmed (P3)** — "strip, empty -> None"; `Utils/` has no home for it; drift is only in whether `None` input is accepted |
| `_thaw` x5 (console_provider_gateway:1745, console_trace_service:6565, console_voice_trace_gateway:351, console_trace_final_values:779, console_prepared_request:110) | **confirmed (P3)** — recursive Mapping/tuple -> dict/list thaw; the copy I read (`console_voice_trace_gateway.py:351-356`) is 5 lines and identical in shape to the `dup_verbatim` trio. Canonical home: `Chat/console_prepared_request.py:110 thaw_json` is already the public-named one |
| `detach_owner@console_voice_attempts:259` + 3 | **retired** — unrelated bodies |
| **dup_verbatim (6)** | |
| `_clean_text`/`_optional_text`/`_text_or_none`/`_cleaned_path_setting` (4) | **confirmed (P3)**, folded into the dup_shape row above |
| `_thaw` x3 | **confirmed (P3)**, folded above |
| `_field@trajectory_export:147` + `_field@library_activity:685` | **confirmed (P3)**, folded above |
| `get_graph_epoch@console_trace_repository:1717` + `_graph_epoch@console_trace_maintenance:1359` | **confirmed (P3)** — and `get_graph_epoch` has **0 production callers, 52 test callers** (`grep -rn "\.get_graph_epoch(" tldw_chatbook/` -> none outside the def): a test-only observability method duplicated as a private helper in the module that actually needs it |
| `_required@console_voice_trace_promotion:55` + `_required_string@console_voice_promotion:34` | **confirmed (P3)** — same bounded-non-empty-string validator; both in the voice-promotion pair |
| `_as_dict@chat_loop_scope_service:48` + `_as_dict@server_chat_loop_service:43` | **confirmed (P3)** — and one of the two modules is dead (see the dead-module finding) |
| **except_exception_pass (14)** | |
| console_trace_settlement.py:739 | **retired** — the `try` guards an equality *check*; falling through raises `_SettlementConflict` two lines later. Read 720-741 |
| console_turn_context.py:51 | **retired** — `# noqa: BLE001 -- review never blocks a send`, returns the documented empty admission |
| console_voice_attempts.py:599, :608, :904 | **retired** — cooperative-cancellation and task-exit paths; `asyncio.CancelledError` is re-raised separately at each |
| console_voice_process_effects.py:849 | **retired** — drain-on-teardown, `CancelledError` re-raised when the work was not itself cancelled |
| console_voice_supervisor.py:167 | **retired** — `except BaseException` consuming a done-callback's `result()`; the standard "never let a callback kill the loop" shape |
| console_voice_worker.py:338, :400 | **retired** — :338 re-raises immediately after; :400 is per-diagnostic-callback isolation |
| provider_continuation.py:450, :573, :601, :689 | **retired** — all four carry `# the public boundary must not retain private parser context` and re-raise `ContinuationValidationError(...) from None`. Deliberate privacy design |
| trajectory.py:338 | **retired** — `# noqa: BLE001 - sqlite3.Row/dataclass/str all differ here`, falls through to `getattr` |
| **except_exception_return_per_file (12)** | |
| scope_picker_listers.py (9) | **confirmed as a pattern, retired as a defect** — all nine are read-only picker listers degrading to empty, with `:376-379` documenting why widening is unsafe. Real gap: the file has **zero** logging (`grep -c "logger\." -> 0`), so a broken notes DB shows an empty picker with no trace. P3 |
| local_server_discovery.py (1) | **retired** — `# noqa: BLE001 - discovery must degrade for injected clients`, returns honest failure copy |
| the other 10 files (1 each) | **not individually examined**; sampled 4 (console_trace_settlement, console_voice_attempts, provider_continuation, thinking_blocks) and all four were the documented boundary shape |
| **fetchall_dynamic_sql (8)** | |
| console_trace_repository.py:713, 2024, 2098, 3277, 3369, 3605; console_trace_runtime.py:394, 453 | **retired** — the only interpolated fragments are (a) a column name picked from two string literals (`ensure_redaction_spans`/`read_redaction_spans`: `"semantic_revision_id"` vs `"artifact_id"`) and (b) a fixed `HAVING`/`AND` clause with its value still bound as `?`. No user data reaches the SQL text. Read 2326-2500 and 1987-2135 |
| **fetchall_no_limit (14)** | |
| console_trace_repository.py:665, 2375, 2437, 2465, 2642, 2828, 3010, 3107, 3759 | **retired** — each is keyed on a single owner/policy/header/segment id whose row count is bounded by a schema CHECK or a validator (`MAX_PROMOTED_TRACE_CALLS = 8`, `MAX_SURFACE_REPLACEMENT_SPAN`, one row per `(policy, source, field_path)`). The one genuinely open table (`read_source_redaction_spans`, :2500) already does `LIMIT 10001` + `raise ValueError("redaction_span_limit")` — the correct pattern, present in the same file |
| console_trace_runtime.py:540 | **retired** — `WHERE call_id = ?` |
| console_trace_settlement.py:363 `recover_open_calls` | **retired, and promoted to Verified-fine** — bounded by the `julianday()` grace window; see Verified-fine for the shape test |
| conversation_local_marks_service.py:450 `list_console_unseen_marks` | **confirmed (P3)** — the `LIMIT` is applied in Python at `:465-466` *after* `fetchall()`, so a user with many unacknowledged receipts fetches all of them. Bounded in practice because `acknowledge_console_unseen` deletes them |
| conversation_local_marks_service.py:472 `has_console_unseen_marks` | **confirmed (P3)** — fetches every matching row to compute one `any()`; a `LIMIT`-less existence check |
| conversation_local_marks_service.py:519 `console_terminal_outcome` | **retired** — keyed on one `(conversation_id, receipt_id)`; at most 3 rows |
| **function_body_import_per_file (18 files, 50 imports)** | **all retired** — I resolved every one of the 50 against the filesystem AND then imported the 9 that my AST pass could not see (star re-exports through `tldw_chatbook/tldw_api/__init__.py` and `Widgets/Console/__init__.py`): all 9 resolve at runtime. No function-body import in this slice targets a deleted module. Command + output in Verified-fine |
| **inline_truncate (1)** provider_failures.py:53 | **confirmed (P2)** — the `truncate_content` D4a finding |
| **legacy_markers_per_file (12)** | **not examined individually** — sampled `provider_catalog.py` (7 markers) and `provider_readiness.py` (5): every one is the word "legacy" in a docstring describing the legacy `[API]` config table, which is a live contract per CLAUDE.md, not dead code |
| **lock_and_execute (3)** | |
| console_trace_runtime.py (1 lock / 4 executes) | **retired** — `with self._lock, operation_owned_connection(self.database), self.database.transaction() as cursor:` (`:162`, `:193`, `:381`): lock acquired *outside* the transaction, every statement inside `db.transaction()`. Correct order, no raw sqlite |
| console_trace_settlement.py (1 lock / 1 execute) | **retired** — `_queue_lock` guards only the in-memory `OrderedDict`; the one `execute` (`recover_open_calls`, :363) is inside `database.transaction(immediate=True)` and outside the lock |
| conversation_local_marks_service.py (1 lock / 10 executes) | **retired** — `_list_cache_lock` guards only the id-list cache; all 10 executes are inside `self.db.transaction()`. The generation-counter race note at `:73-81` is correct as written |
| **mutable_class_attr (2)** custom_endpoint_registry.py:110 `model_config`, :119 `params` | **retired** — pydantic v2 `BaseModel`, which deep-copies mutable field defaults per instance. Verified by running it: two instances, mutate one's `params`, the other stays `{}`, `a.params is b.params -> False`, pydantic 2.12.5 |
| **os_replace_no_atomic (1)** trajectory_export.py:1319 | **retired** — the pattern is the *correct* atomic write: `mkstemp` in the destination directory (mode 0600), `os.fdopen`, `os.replace`, `unlink` on any `BaseException`. See Verified-fine for the one caveat |
| **plain_readback (1)** console_turn_context.py:172 `str(item.label)` | **retired** — `item` is a `ConsoleStagedSource` dataclass, not a Textual widget; `.label` is a plain `str` field being copied in `_detached_selection`. No markup round-trip |
| **raw_1024x1024 (8)** | **all retired** — every one is the right-hand side of a *named module constant* (`MAX_PROMOTED_TRACE_BYTES = 64 * 1024 * 1024`, `MODEL_PROBE_RESPONSE_MAX_BYTES = 1024 * 1024`, ...). This is the pattern the rule wants, not the one it hunts |
| **seed_name__now (1)** conversation_local_marks_service.py:94 | **confirmed (P2)** — the two-shapes-in-one-column finding |
| **seed_name__strict_json_loads (2)** provider_continuation.py:204, thinking_blocks.py:221 | **confirmed (P3)** — the 49-site strict-JSON D4b finding |
| **seed_name__utc_now (2)** console_trace_runtime.py:48, console_voice_trace_gateway.py:343 | **confirmed as shapes, retired as defects** — `console_trace_runtime._utc_now()` is microsecond-or-bare-`Z`; `console_voice_trace_gateway._utc_now()` is `timespec="microseconds"`, so always 6 digits. Both land in `console_trace_calls`, so that column *does* carry two shapes — but every comparison on it is `julianday()`, which normalises them (measured, see Verified-fine), and the one lexical-looking parse (`console_trace_repository.py:553`) is guarded upstream (see Retired) |
| **tempfile_no_secure (1)** trajectory_export.py:1313 | **retired** — `tempfile.mkstemp` *is* the secure API (0600, no race); the insecure one is `mktemp` |
| **try_import_guard (1)** console_voice_input.py:172 | **retired** — `require_dependency("faster_whisper", ...)` inside a function is the repo's `optional_deps` contract, and the `except Exception` logs with `logger.opt(exception=True).debug` before returning `False` |

## Verified-fine

- **`console_trace_settlement.recover_open_calls` compares timestamps across shapes correctly.** The predicate is `julianday(COALESCE(response_started_at, dispatch_started_at, created_at)) <= julianday(?, '-N seconds')` (`:363-375`) — and those three columns genuinely carry different shapes (`created_at` defaults to SQLite `CURRENT_TIMESTAMP`, the others come from `_utc_now()`). Measured on sqlite 3.49.1:
  ```
  '2026-09-18 12:34:56'              julianday -> 2461302.0242592595
  '2026-09-18T12:34:56.789012Z'      julianday -> 2461302.024268391
  '2026-09-18T12:34:56.789Z'         julianday -> 2461302.024268391
  '2026-09-18T12:34:56.789012+00:00' julianday -> 2461302.024268391
  '2026-09-18T12:34:56Z'             julianday -> 2461302.0242592595
  ```
  all five satisfy the predicate identically. This is the shape-safe pattern the rest of the codebase's 100 lexical comparisons should copy; worth naming in the review's headline timestamp finding as the existing in-repo answer.
- **No function-body import in this slice targets a missing module or name.** AST pass over all 62 files -> 50 function-body imports, 9 flagged; importing all 9 for real under the isolated env resolved every one (`tldw_chatbook.tldw_api.{ConversationUpdateRequest,ChatLoopStartRequest,ChatKnowledgeSaveRequest,ConversationShareLinkCreateRequest,CharacterChatSessionCreate,ChatLoopApprovalDecisionRequest}` and `tldw_chatbook.Widgets.Console.VoicePreviewProjection` — all `OK`). They are star re-exports, which is exactly the false positive the brief warns about.
- **Zero `re.compile` inside a function body anywhere in the 62 files.** AST pass, not grep.
- **No secret reaches a log from this slice's credential path.** `grep -c "logger\|logging"` -> **0** in both `provider_setup_persistence.py` (1838 lines) and `provider_test_evidence.py` (1261 lines). `credential_value` is `field(repr=False)` (`:233`) and the three state classes carry hand-written `__repr__`s that emit only non-secret fields (`:404`, `:480`, `:569`, `:702`).
- **`local_server_discovery.py` is not an egress bypass.** Auto-discovery filters to `{127.0.0.1, localhost}` *before* any request (`is_localhost_url`, `:177-202`, applied in `_add` at `:225-227`); every probe sets `follow_redirects=False` (`:431`), forces `Accept-Encoding: identity` and rejects any other content-encoding (`:94-97`), caps the body at 1 MB (`:39`, `:103-110`), and sanitises every server-supplied model id (`:263-274`). `probe_models_endpoint` has no host filter but is the explicitly user-typed URL in the settings modal, documented as such at `:10-14`.
- **`write_trajectory_export` is a correct atomic write** (`trajectory_export.py:1293-1327`). Caveat worth one line, not a finding: there is no `os.fsync(fd)` before `os.replace`, so the docstring's "readers never observe a partial file" holds for concurrent readers but not across a power loss.
- **`conversation_local_marks`' list cache is race-correct.** The generation counter at `:84/:89/:418/:434` closes the populate-after-invalidate window exactly as its comment claims; I traced all four writers.
- **`console_trace_repository.import_post_dispatch_trace` does not hold a connection for DML.** `db.get_connection()` at `:462` is used only to assert `not connection.in_transaction`; every write goes through `db.transaction(immediate=True)` (`:467`). The transaction-class defect the brief describes is absent here.
- **`get_cli_setting` is not on a hot path in this slice.** All 19 call sites in `console_voice_input.py` are session/press-scoped config resolution; `reply_sentence_sequencer.py` (the per-token module) and `console_voice_process.py` (the per-chunk module) have zero.

## Retired

- **`console_trace_repository.py:553-555`'s `datetime.fromisoformat(x[:-1] + "+00:00")`** — named in the orchestrator brief as "a fragile parse". Symptom real, cause wrong. Every value reaching it is a `PostDispatchTraceCall` field, and that dataclass's `__post_init__` (`console_voice_trace_promotion.py:576-582`) runs `_observed_timestamp`, which is `:78-84`: `if not encoded.endswith("Z"): raise ValueError(...)` followed by the same parse. The `[:-1]` is therefore total on its domain. What survives is a P3 D4: the repository re-implements a validated parse instead of importing `_observed_timestamp` from the module it already imports four other names from.
- **`citation_trace_repository.py` "1 lock + 62 executes"** — not in my slice. In my slice the equivalent file, `console_trace_repository.py`, has **zero** locks and 62 executes, all taking a caller-owned `cursor`; the single `RLock` in the trace subsystem is `console_trace_runtime.py:92` and is correctly ordered outside the transaction (see dispositions).
- **`ConsoleTraceSettlementCoordinator` silently drops the oldest queued settlement past `max_pending=64`** (`console_trace_settlement.py:885-890`). I raised this as silent trace loss, then retired it: `dropped_count` is asserted as a requirement by `Tests/Chat/test_console_trace_call_lifecycle.py:386` and `Tests/Chat/test_console_trace_settlement.py`, i.e. the bounded drop is a stated decision, and the data is diagnostic rather than user content. Worth knowing: the counter has **no production reader** (`grep -rn "dropped_count" tldw_chatbook/` -> only `console_history_budget`'s unrelated field), so the drop is invisible at runtime.
- **`ProvisionalTraceRegistry.begin_attempt`'s O(n) duplicate scan** (`console_voice_trace_gateway.py:1049-1054`) — a linear scan over `_states.values()` per attempt where a second dict keyed by `(promotion_id, attempt_id)` would be O(1). Retired: `_states` is bounded by `MAX_PROVISIONAL_TRACE_APP_BYTES / MAX_PROMOTED_TRACE_BYTES` = at most 2 live attempts by construction.

## Left UNVERIFIED

| claim | why not verified | literal command to run |
| --- | --- | --- |
| The `UnicodeDecodeError`/`RecursionError` from `load_imported_trace` **exits the app** rather than being absorbed by `app.py:19025`'s TASK-32533 keep-alive | Needs the running TUI; the keep-alive fires only when the raise passes through `_PUMP_DISPATCH_FRAMES` with `pump is not self`, and whether an async binding-action qualifies cannot be settled by reading | `tmux -L verify new-session -d -s t 'cd /Users/macbook-dev/Documents/GitHub/tldw-review && TLDW_CONFIG_PATH=<scratch profile> .venv/bin/python -m tldw_chatbook.app'` then open a trajectory screen, press `o`, pick a non-UTF-8 `.json` written to the scratch dir, and `tmux -L verify capture-pane -p -t t` — a "Something went wrong in …" toast means keep-alive (P1); a dead pane means exit (P0) |
| `Chat/console_trace_runtime.py`'s `db.transaction(immediate=True)` write runs **on the Textual event loop** | `_reserve_trace_call` (`console_provider_gateway.py:2992`) is sync, but one of its two callers is `async def _authorize_llamacpp_fallback` (`:5694`); whether that coroutine is scheduled on the app loop or a worker's own loop needs the running app | add `assert threading.current_thread() is threading.main_thread()` reporting inside `ConsoleTraceBoundaryFactory._verify_owned_recovery` on a scratch copy and drive one Console send under `tmux -L verify`, or instrument with `asyncio.get_running_loop()` identity vs `app._loop` |
| `trajectory_import` OOMs on a multi-GB picked file | did not want to write a 3 GB file into the shared scratch dir | `python -c "open('/tmp/big.json','w').write('['+'0,'*400000000+'0]')"` then `load_imported_trace('/tmp/big.json')` |
| The `_notes_tag_usage` cost at the real `_NOTES_TAG_USAGE_SAMPLE_LIMIT=1000` ceiling | measured at 20 notes/keyword (22.4 ms); building the 200x1000 fixture is minutes of setup | same repro script with `NPER = 1000` |
| Whether the six modules with zero production importers are already rows inside task-19571's unpublished 170-module census (only `console_visual_evaluation.py` is named in its table) | the census itself is not in the repo, only its summary | ask the task owner, or re-run the lane's AST BFS from `app.py` |
