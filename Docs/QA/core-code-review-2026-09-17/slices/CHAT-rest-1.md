# CHAT-rest-1 — tldw_chatbook/Chat/ (first 50 files, minus the 6 reviewed separately), 43,946 lines

Worktree: `/Users/macbook-dev/Documents/GitHub/tldw-review` (detached at origin/dev d8fb4053f9). Nothing modified.
`$WT` = that path. `<SCRATCH>` = `/private/tmp/claude-501/-Users-macbook-dev-Documents-GitHub-tldw-chatbook/5cdb48ca-db0f-47a4-930a-9ef5b33bceed/scratchpad`. `$PY` = the venv python (3.12.11).

## Coverage

Honest summary: **~11,500 of 43,946 lines read closely; the rest covered by symbol census + pattern sweep.**
Every file went through an AST/pattern sweep (mutable defaults, mutable class attrs, `except:`/`except Exception`, `json.loads` guards, `re.compile` placement, `get_cli_setting`/`load_settings`, `get_connection()` DML, `fetchall`, locks, timers/`query_one`/`run_worker`, `eval`/`exec`/`pickle`/`subprocess`, timestamp helpers and parsers, canonical-JSON helpers, strict-JSON `object_pairs_hook` families, Chat→UI imports). The citation_* subsystem (~14,000 lines) is the weakest coverage: I read its persistence seams, swallow sites, serializers and timestamp/locator validators, but not its model layer end to end.

| file | lines | read in full / sampled (which ranges) / mechanical only |
|---|---|---|
| Chat_Deps.py | 111 | read in full |
| Chat_Functions.py | 3476 | sampled: 1–220, 852–1350 (`chat_api_call` whole), 1401–1443, 1443–1960 (`chat()` head + payload build), 1985–2050, 2622–2882 (export path); symbol census of whole file |
| __init__.py | 19 | read in full |
| answer_citations.py | 379 | mechanical only |
| assistant_generation_state.py | 114 | read in full |
| attachment_core.py | 472 | read in full |
| character_expression_playback.py | 229 | read in full |
| chat_conversation_scope_service.py | 649 | sampled: 100–220 (the `_maybe_await`/`asyncio.to_thread` reference shape, task-283) |
| chat_conversation_service.py | 1617 | sampled: 27–90, 293–360, 812–940; symbol census of whole file |
| chat_handoff_messages.py | 107 | mechanical only |
| chat_handoff_models.py | 203 | mechanical only |
| chat_loop_scope_service.py | 169 | read in full |
| chat_models.py | 140 | read in full |
| chat_persistence_service.py | 3849 | sampled: 1–260, 388–560, 560–790 (voice-pair reconciler whole), 855–908, 975–1015, 1995–2030, 2431; symbol census of whole file |
| citation_artifact_ownership.py | 571 | sampled: 380–460 (reconciliation loop), 470–571 (shared-DB contract) |
| citation_evidence_models.py | 714 | mechanical only |
| citation_legacy_migration.py | 1614 | sampled: 105–130, 585–600, 640–665, 1362–1400, 1590–1600 |
| citation_payload_lifecycle.py | 1270 | sampled: 90–110, 370–460 (tombstone upsert/purge), 1240–1260 |
| citation_provenance_runtime.py | 29 | read in full |
| citation_repair.py | 407 | sampled: 30–50, 365–400 |
| citation_service_factory.py | 103 | read in full |
| citation_source_locators.py | 1177 | sampled: 1–100 (locator/path validators), 720–760 |
| citation_trace_adapters.py | 402 | mechanical only |
| citation_trace_builder.py | 898 | mechanical only (plus sweep hits at 56–68, 320, 815) |
| citation_trace_identity.py | 547 | mechanical only |
| citation_trace_models.py | 1284 | sampled: 30–50, 970–1000 |
| citation_trace_repository.py | 4719 | sampled: 260–320, 370–520, 650–700, 1680–1700, 3920–4010, 4080–4200, 4580–4640; symbol/pattern census of whole file |
| console_activity_receipts.py | 406 | read in full |
| console_appearance.py | 175 | read in full |
| console_assistant_defaults.py | 101 | read in full |
| console_auto_speak.py | 100 | read in full |
| console_auxiliary_routing.py | 49 | read in full |
| console_canvas_controller.py | 1959 | sampled: 344–420 (init + lock), 600–660; symbol census + full lock-site listing |
| console_capture_policy_repository.py | 311 | sampled: 85–165, 230–245 + full import/logging census |
| console_chat_fork.py | 857 | sampled: 73–175, 352–460; symbol census of whole file |
| console_chat_models.py | 1449 | sampled: 425–435, 830–845, 1110–1130; symbol/pattern census |
| console_command_grammar.py | 363 | sampled: 140–180 |
| console_command_suggestions.py | 188 | sampled: 1–60 |
| console_context_compaction.py | 2664 | sampled: 1828–1925, 2458–2560, 2650–2664; symbol census of whole file |
| console_context_policy.py | 470 | mechanical only |
| console_context_repository.py | 1925 | sampled: 1120–1240, 1275, 1400–1530, 1870–1890 |
| console_context_window.py | 238 | read in full |
| console_conversation_actions.py | 343 | mechanical only |
| console_conversation_activation.py | 384 | sampled: 60–130, 270–370 (every `_maybe_await` site) |
| console_conversation_hydration.py | 718 | sampled: 100–180, 250–260 |
| console_conversation_markdown.py | 281 | sampled: 1–70, 255–270 |
| console_cost_tracker.py | 1068 | sampled: 85–200 (`TokenEstimateCache`), 390–445, 1029–1068; symbol census |
| console_dispatch_checkpoint.py | 673 | sampled: 60–80, 265–300 |
| console_dispatch_repository.py | 1716 | sampled: 100–180, 300–340, 390–420, 740–770, 950–1130, 1180–1200, 1305–1335 |
| console_display_state.py | 2239 | sampled: 1–100, 1001–1200; full `_safe_display_text` call-site census + symbol census of whole file |

## Findings

### P1 [D4b] — Console evidence/inspector rows HTML-entity-escape text destined for a terminal surface, so a Library title "R&D Report" reaches the user as "R&amp;D Report"; the Library surface fixed exactly this bug and Console did not
- Where: `tldw_chatbook/Chat/console_display_state.py:8` (`from html import escape as html_escape`) and `:93-95 _safe_display_text`. 18 call sites: `:496, 507, 508, 519, 528, 829, 849, 852, 854, 855, 901, 902, 903, 1158, 1159, 1170, 1171`. Already-fixed sibling copy: `tldw_chatbook/Library/library_rag_state.py:369-389 _sanitize_display_text` + `:392 _unescape_and_rescrub`.
- Evidence: `cd $WT && source <SCRATCH>/env.sh && PYTHONPATH=$WT $PY - <<EOF` importing both helpers →
  ```
  'R&D Report'      console='R&amp;D Report'              library='R&D Report'
  'Alice & Bob'     console='Alice &amp; Bob'             library='Alice & Bob'
  '<b>Release</b>'  console='&lt;b&gt;Release&lt;/b&gt;'  library='<b>Release</b>'
  'R&amp;D Report'  console='R&amp;amp;D Report'          library='R&D Report'   <-- Console double-escapes
  ```
- Why it matters: these rows render in a Textual/Rich `Static` with markup OFF — stated by `ConsoleStagedEvidenceRow`'s own docstring (`console_display_state.py:1096-1102`) and by `Tests/UI/test_console_staged_evidence_strip.py:156-158`. Rich never decodes HTML entities, so the escape protects nothing on this surface and corrupts the displayed string; a title that arrives already entity-encoded is escaped twice. `library_rag_state.py:392`'s docstring records the identical bug found in **live UAT on 2026-08-03 (task-15 finding 1)** — "a Note containing 'Alice & Bob' rendered as 'Alice &amp; Bob' in the evidence card" — and fixed there. The same Library note staged into Console still shows the pre-fix string.
- Recommended correction: replace `_safe_display_text`'s `html_escape` with the Library tail (`html.unescape` → re-run the dangerous-pattern scrubber → `escape_markup`), promoted out of `Library/library_rag_state.py` into a shared `Utils/` display-text helper as the canonical home; `escape_markup` is the only escape a markup-off Rich surface needs. Heed `library_rag_state.py:392`'s own warning: the un-escape must be followed by re-running the scrubber, so lift the whole tail, not just the `html.unescape`.
- Size: M · ADR: no · Confidence: verified (string drift reproduced; the on-screen render is inferred from the Library live-UAT precedent on the identical widget class — see UNVERIFIED)
- Pinning test: `Tests/UI/test_console_staged_evidence_strip.py::test_strip_state_escapes_untrusted_library_titles` asserts `row.title == "[bold]pwn[/bold] &lt;script&gt;"` — **it states the current behaviour as a requirement**, so this is a standing decision the later task-15 ruling contradicts; fixing Console means re-ruling that test, not just editing the helper. Same shape pinned at `Tests/UI/test_console_staged_context.py:369-370`, `Tests/UI/test_console_live_work_handoffs.py:1790`, `Tests/UI/test_console_internals_decomposition.py:4184-4185`.
- Already covered: none (task-15 covered the Library copy only)

### P2 [D4b] — the compaction admission fence's canonical-JSON digest exists as three byte-identical private copies, and a fourth strictness variant of the same serializer sits beside them
- Where — the digest triple (identical): `Chat/console_context_compaction.py:2657 _digest_json`, `Chat/console_context_repository.py:1876 _digest_json`, `Chat/console_chat_controller.py:22946 ConsoleChatController._manual_json_digest`. The canonical-serializer triple, also identical but **stricter** (`allow_nan=False`): `citation_trace_repository.py:265`, `citation_legacy_migration.py:111`, `citation_source_locators.py:728`. A seventh variant: `citation_trace_models.py:976 _canonical_json_bytes`. ~20 such helpers repo-wide.
- Evidence: `cd $WT && source <SCRATCH>/env.sh && PYTHONPATH=$WT $PY -c "..."` →
  ```
  digest triple identical: True c9dad5b6d97afe79
  canonical triple identical: True
  citation _canonical_json   NaN -> ValueError(Out of range float values are not JSON compliant: nan)
  console _digest_json       NaN -> 'c238cb6e9ac5407491b0988102620ab9445290d2'
  ```
- Why it matters: the three `_digest_json`/`_manual_json_digest` copies compute the value `_manual_admission_matches`/`_automatic_admission_matches` (`console_context_compaction.py:2563, 2611`) compare **across module boundaries** to decide whether a compaction may commit. Three private copies in three files means a one-line change to any one (adding `default=str`, dropping `sort_keys`) silently breaks the fence, and no test compares copy against copy. The NaN strictness split is already live drift inside the same serializer family.
- Recommended correction: one `Utils/` canonical-JSON module exporting `canonical_json(value) -> str` and `canonical_json_digest(value) -> str`, strict (`allow_nan=False`) as the single contract; the Chat copies import it. `Skills_Interop/skill_trust_crypto.canonical_json` and `Actor_Packs/contracts.canonical_json_bytes` are existing candidates to promote rather than writing a new one.
- Size: M · ADR: no · Confidence: verified
- Pinning test: none compares the copies
- Already covered: none

### P2 [D4b] — `conversation_local_marks` accepts two UTC timestamp shapes, and the voice-promotion reconciler's validator rejects the shape the table's own public writer produces
- Where: writer A `Chat/conversation_local_marks_service.py:94-95 _now()` (`datetime.now(timezone.utc).isoformat().replace("+00:00","Z")`), used by the public `set_mark()` at `:233`. Writer B `chat_persistence_service.py:453` and `:879`, `console_dispatch_repository.py:1058` — `db._get_current_utc_timestamp_iso()` (`DB/ChaChaNotes_DB.py:8700`, `timespec="milliseconds"`). Validator: `chat_persistence_service.py:195-212 _is_canonical_utc_timestamp`, used at `:700` and `:767`; failure raises `RuntimeError("Voice promotion persistence conflict.")` at `:723`.
- Evidence: `cd $WT && source <SCRATCH>/env.sh && PYTHONPATH=$WT $PY -c "..."` →
  ```
  marks_service._now()                   len=27 value='2026-09-18T14:46:16.004249Z' accepted=False
  db._get_current_utc_timestamp_iso()    len=24 value='2026-09-18T14:46:16.004Z'    accepted=True
  marks _now() at microsecond==0         len=20 value='2026-09-18T12:00:00Z'        accepted=False
  set_mark() accepts receipt mark type -> 'console_unseen:11111111-1111-4111-8111-111111111111'
  set_mark() accepts outcome mark type -> 'console_terminal_outcome:<uuid>:complete'
  ```
- Why it matters: `_is_canonical_utc_timestamp` hard-codes "length 24 ending in Z" or "length 19"; the marks service's own `_now()` produces neither (27 normally, 20 at microsecond==0). Today every *receipt* mark happens to be written through the cursor-scoped path that uses the DB helper, so nothing breaks — but `set_mark()`'s `_mark_type()` validates and **accepts** exactly the receipt mark types the validator later inspects (reproduced above), so one call through the public API turns an idempotent voice-promotion retry into `RuntimeError` on an already-committed row. Same class the CHAT-bridge sibling rated D4b, with a concrete single-table consumer attached.
- Recommended correction: delete `ConversationLocalMarksService._now()` and use `self.db._get_current_utc_timestamp_iso()` (already the canonical helper and already used by the other three writers); alternatively make `_is_canonical_utc_timestamp` parse rather than length-match. Canonical home: `DB/ChaChaNotes_DB._get_current_utc_timestamp_iso`.
- Size: S · ADR: no · Confidence: verified for the shape mismatch and the API acceptance; the *reachable* failure is inferred (no shipped caller passes a receipt mark to `set_mark()` today)
- Pinning test: none (`rg _is_canonical_utc_timestamp Tests/` → no hits)
- Already covered: none

### P2 [D1] — `ConsoleCapturePolicyRepository` has three bare `except Exception: return …UNAVAILABLE` arms and the module imports no logger, so a failed write of the per-conversation capture/PII-redaction policy leaves no diagnostic at all
- Where: `Chat/console_capture_policy_repository.py:91-96` (read), `:157-158` (`replace_detail`), `:239-240` (`replace_privacy`). Imports at `:3-10` contain no `loguru`/`logging`.
- Evidence: `grep -n "logger\|import" tldw_chatbook/Chat/console_capture_policy_repository.py` → only `sqlite3`, `dataclasses`, `enum`, `CaptureDetail`, `CharactersRAGDB`. `grep -n -A3 "except Exception" …` → the three arms, none logging.
- Why it matters: `console_chat_controller.py:4935` and `:5081` read `status is CapturePolicyWriteStatus.UNAVAILABLE` and degrade the user to "session only". A genuine defect (a `TypeError` in `_upsert`, a schema drift, a bad `CaptureDetail`) is therefore indistinguishable from "the database was busy" and produces zero log output — the user's PII-redaction preference silently fails to persist with nothing to debug from. The write is inside `db.transaction()`, so no partial write occurs; what is lost is the diagnostic.
- Recommended correction: bind a module logger and `logger.opt(exception=True).warning(...)` in each arm (Console's own convention, e.g. `chat_persistence_service.py:118`), and narrow the catches to `sqlite3.Error` plus `(TypeError, ValueError)` so a programming error is not laundered into a storage verdict.
- Size: S · ADR: no · Confidence: verified (code + import census; not reproduced as a runtime failure)
- Pinning test: none
- Already covered: none

### P2 [D3] — `Chat/console_conversation_hydration.py` reaches into a UI module for a **private** helper, inverting the very Chat←UI layering the module exists to remove
- Where: `Chat/console_conversation_hydration.py:137-139` (function-body import) and `:173` (call) of `_apply_console_message_attachments` from `UI/Console_Modules/message.py:186`.
- Evidence: `grep -rn "_apply_console_message_attachments" --include='*.py' .` → defined once at `UI/Console_Modules/message.py:186`; used at `UI/Console_Modules/message.py:926` and `Chat/console_conversation_hydration.py:138,173`. Nothing else.
- Why it matters: `UI/Console_Modules/message.py:611-614` states the tree walk was moved into `Chat/` precisely so "the launch wake — which has to hydrate a conversation with no screen at all — shares this policy instead of copying it". Importing back into UI for the attachment-folding half restores a screen-layer dependency on a headless path, through a leading-underscore name whose contract UI is free to change, and the function-body import hides it from any module-level dependency check.
- Recommended correction: move `_apply_console_message_attachments` to `Chat/console_chat_models.py` (which already owns `ConsoleChatMessage` and its attachments tuple) as a public name; UI imports it from there.
- Size: S · ADR: no · Confidence: verified
- Pinning test: none
- Already covered: none

### P2 [D2] — `attachment_core`'s three config readers memoize nothing and each costs ~7 ms; a single image attachment fans out to ~5 of them
- Where: `Chat/attachment_core.py:84-102 _chat_images_setting` (shared reader), `:105 supported_image_formats`, `:147 max_image_bytes`, `:164 image_resize_max_dimension`, `:184 attachment_filter_specs`. Per-attachment callers: `Utils/file_handlers.py:112` (`ImageFileHandler.can_handle`, once per candidate file), `:600-606`, `Event_Handlers/Chat_Events/chat_image_events.py:67, 76, 194, 206`.
- Evidence: `cd $WT && source <SCRATCH>/env.sh && PYTHONPATH=$WT $PY -c "..."` (warm, n=20 each) →
  ```
  get_cli_setting('chat','images')          7.415 ms/call
  supported_image_formats()                 7.563 ms/call
  max_image_bytes()                         6.923 ms/call
  attachment_filter_specs()                 7.457 ms/call
  ```
  Independently reproduces the sibling's "~11 ms per `get_cli_setting` even when cached" on this machine.
- Why it matters: per-attachment, not per-tick, so bounded (~35 ms for one image) — but it is 5 process-wide config handshakes to answer three questions that cannot change within one attachment operation, and `supported_image_formats()` additionally re-runs `svg_rendering_available()` (which imports `optional_deps` and probes cairosvg) each time.
- Recommended correction: not a per-call cache in `attachment_core` (config is mutable at runtime); resolve the three values once per attachment operation at the entry point (`process_attachment_path` / `chat_image_events`' entry) and pass them down. If a cache is preferred, the fix belongs in `config.get_cli_setting`, where the ~7 ms actually is — that is the sibling's finding, not this one.
- Size: M · ADR: no · Confidence: verified (cost measured; the "~5 per attachment" fan-out is counted statically from the call sites above, not instrumented)
- Pinning test: none
- Already covered: none — the brief's "config reads are cache-backed" known-deliberate item covers the *caching design*, not this call count

### P3 [D3] — `Chat_Functions.py` uses stdlib `logging` for ~100 calls and loguru for the rest, in one file
- Where: `Chat/Chat_Functions.py:21` (`import logging`) and `:30-35` (`logger = loguru.logger.bind(...)`). The loguru name is used inside `chat_api_call`; stdlib `logging.*` is used throughout `chat()` (`:1547` onward) and the export/save helpers.
- Evidence: `grep -n "logging\." tldw_chatbook/Chat/Chat_Functions.py | wc -l` → 100+; the loguru name appears only in the `chat_api_call` block.
- Why it matters: only the loguru sink is routed through this app's configuration and `Utils/log_sanitizer`; the stdlib calls bypass it. I checked each stdlib call's arguments — they log lengths, booleans and type names, never content or keys — so there is **no secret or transcript leak today**; the risk is that the next line added there is not sanitized, and that half the file's diagnostics land in a different sink. `chat()` is still reachable (`Event_Handlers/worker_events.py:41` → `app.chat_wrapper` → `UI/MediaWindow_v2.py:1881`).
- Recommended correction: convert the `logging.*` calls to the bound `logger`; `Logging_Config.py` is the one file allowed both.
- Size: M (mechanical but ~100 call sites) · ADR: no · Confidence: verified
- Pinning test: none
- Already covered: none

### P3 [D4b, documented] — `_split_leading_token` exists three times, by explicit decision
- Where: `Chat/console_command_grammar.py:151`, `Chat/console_chat_controller.py:3041 _split_skill_command_word`, `UI/Console_Modules/skill.py:111 _split_console_skill_name_args`. Bodies identical (4 lines).
- Evidence: read all three; `console_chat_controller.py:3043-3050` documents the duplication as deliberate ("a deliberate small duplicate rather than an import").
- Why it matters: consistency only. There is no stdlib equivalent — `str.split(None, 1)` collapses whitespace runs and strips, while this splits on the first whitespace *character* and preserves the remainder verbatim — so the helper is warranted, just not three times.
- Recommended correction: make it public in `console_command_grammar` (`split_leading_token`) and import it; that module already owns command tokenization.
- Size: S · ADR: no · Confidence: verified
- Already covered: none

### P3 [D3] — `UI/Console_Modules/message.py:597 _console_message_role_from_persisted` is a dead verbatim copy of `Chat/console_conversation_hydration.py:109`
- Where: `UI/Console_Modules/message.py:596-610` vs `Chat/console_conversation_hydration.py:108-122`.
- Evidence: `grep -rn "_console_message_role_from_persisted" --include='*.py' .` over the whole worktree (Tests included) → exactly 3 hits: the two definitions and ONE call, at `Chat/console_conversation_hydration.py:257`. The enclosing class's `_console_messages_from_conversation_tree` (`:612`) delegates to the Chat function.
- Why it matters: dead code that reads as the live role-mapping policy; the next person fixing a role-mapping bug has even odds of editing the copy nothing calls.
- Recommended correction: delete the UI static method.
- Size: S · ADR: no · Confidence: verified
- Already covered: none

### P3 [D4a] — `citation_artifact_ownership.py` rolls its own SQL-identifier validator instead of `DB/sql_validation.validate_identifier`
- Where: `Chat/citation_artifact_ownership.py:37 _SQL_IDENTIFIER`, applied at `:501-504` before the f-string `PRAGMA main.foreign_key_list("{owner_table}")` at `:533`. The helper: `DB/sql_validation.py:558 validate_identifier`, 11 importing modules.
- Evidence: `grep -rln "sql_validation" --include='*.py' tldw_chatbook` → 11 files, none in `Chat/`.
- Why it matters: the inline regex is correct and applied before interpolation, so this is **not** an injection finding; it is one more independent definition of "what a safe identifier is". The related site `chat_persistence_service.py:522-541` also interpolates identifiers but takes them from a frozen module constant, which is fine.
- Recommended correction: import `validate_identifier`.
- Size: S · ADR: no · Confidence: verified
- Already covered: none

## Candidate dispositions

| candidate (file:line pattern) | disposition |
|---|---|
| dup_shape `unique_object`×7 (`library_activity:487`, `library_preparation:174`, `console_dispatch_checkpoint:275`, `console_trace_regex_worker:58`, `watchlists_tool_service:1257`, `workspace_tool_protocol:407`, `permission_store:388`) | **confirmed as a family, only 1 in slice.** `console_dispatch_checkpoint.py:289-291` catches `(TypeError, ValueError, json.JSONDecodeError, RecursionError)` — correct. Copies that do NOT catch `RecursionError` are listed under *Adjacent*. |
| dup_shape `_require_ledger`/`runner_task`/`_require_sync_scope_service`/`terminal_turn`/`_play_console_video`… | **retired** — a 5-copy "shape" match over unrelated 3-line guard/raise bodies (`chat_conversation_scope_service.py:105` is `if x is None: raise ValueError(...)`, `return x`). Nothing shared to extract. |
| dup `_clean_text`(chat_conversation_service:27) / `_optional_text`(trajectory:1044) / `_normalize_optional_text`(MCP) / `_text_or_none`(MCP) / `_cleaned_path_setting`(RAG) | **confirmed, P3, not written up separately** — 4 byte-identical `str(value).strip() or None` bodies plus near-copies (9 repo-wide by `grep -c "return text or None"` after `text = str(value).strip()`). Two-line helper; canonical home `Utils/`. Folded into the D4 picture. |
| dup `_split_leading_token`×3 | **confirmed** → P3 finding above |
| dup `_manual_json_digest`/`_digest_json`×3 | **confirmed** → P2 finding above |
| dup_shape `reference_by_id`(citation_evidence_models:418) / `_tool_for_row_key` / `get_section` | **retired** — three unrelated "look up by id, return None" accessors on different types. Shape-matcher false positive. |
| dup `_maybe_await`×3 (`chat_loop_scope_service:32`, `chat_conversation_scope_service:144`, `MCP/unified_control_plane_service:568`) | **retired as a P1** (see Retired) — the `_maybe_await(sync_call())` hazard cannot fire here: `ServerChatLoopScopeService` is never constructed outside `Tests/`, and its only `server_service` implementation is fully `async def`. |
| seed `_maybe_await`@`console_conversation_activation:79` | **retired** — the only sync callback any caller injects (`UI/Console_Modules/workspace.py:5729`) is in-memory + `query_one`, no I/O. |
| dup `_empty_profile_context_snapshot` (`console_chat_controller:3537` / `console_chat_models:1395`) | **unverified (check)** — both sites outside my read ranges; `diff <(sed -n '3537,3550p' …controller.py) <(sed -n '1395,1408p' …models.py)` |
| dup `_metadata_object` (`console_appearance:166` / `console_speech_preferences:109`) | **confirmed, P3** — byte-identical 10-line JSON-or-mapping coercion; both files share the "one namespaced key in `conversations.metadata`" contract (`console_appearance.py:1-12`). Canonical home: a shared `console_metadata` helper for the three namespaced-metadata modules. |
| dup `_format_size`(attachment_core:64) / `_human_size`(Widgets/Console/console_transcript:639) | **confirmed as documented duplication, P3** — `console_transcript.py:640` names `attachment_core._format_size` and says it duplicates it on purpose (brief's known-deliberate list). 10 byte-size formatters repo-wide. |
| dup `_console_message_role_from_persisted`×2 | **confirmed** → P3 finding (UI copy is dead) |
| dup `_as_dict` (`chat_loop_scope_service:48` / `server_chat_loop_service:43`) | **confirmed, P3** — identical 7-line Mapping/`model_dump`/`dict()` coercion in two files of the same (test-only) subsystem. Not worth fixing while nothing constructs it. |
| `except_exception_pass` `citation_artifact_ownership.py:439` | **retired** — the inner `record_provenance_operation_failure` is a best-effort note inside an outer arm that already recorded `failed`, `failed_ids` and `reason` into the returned `ArtifactReconciliationResult`. No information lost. |
| `except_exception_return` `citation_artifact_ownership.py:384` | **retired** — returns a populated `ArtifactReconciliationResult` carrying `_reconciliation_reason(exc)`. Converts, does not swallow. |
| `except_exception_return` `citation_repair.py:391` | **retired** — guards injected `count_fn`/`window_fn`, returns `False` fail-closed (repair prompt not sent). |
| `except_exception_return` `citation_trace_repository.py:509` | **confirmed but P3** — `canonical_citation_writes_ready` returns `False` on any identity-load failure with no log, so citations degrade silently. One `logger.debug` closes it; the module has a logger. |
| `except_exception_return` `console_assistant_defaults.py:97` | **confirmed but P3** — degrades to a user-visible notice but logs nothing, so a `TypeError` in `build_persona_agent_system_prompt` is indistinguishable from an absent Persona. |
| `except_exception_return` ×3 `console_capture_policy_repository.py` | **confirmed** → P2 finding above |
| `except_exception_return` `console_chat_models.py:430` | **retired** — `return console_pending_round_copy(())` with an explicit `# copy must never break a render` rationale; the fallback is the correct empty copy. |
| `fetchall_dynamic_sql` ×5 `console_dispatch_repository.py` (141, 314, 401, 619, 744) | **retired** — every one is `_OWNER_SELECT`/`_ACTIVE_OWNER_SELECT` (module-level SQL constants) concatenated with a literal `WHERE …= ?`; `grep -nE 'f"""\|f"SELECT\|" \+ \|join\('` over the file returns nothing. No user data reaches the SQL text; row counts bounded by one conversation's active path. |
| `fetchall_no_limit` ×27 | **retired as a class, two exceptions noted.** The 4 `chat_persistence_service` rows are `WHERE id IN (?, ?)` — 2 rows. The citation rows are all `profile_id`+`trace_id`-scoped, bounded by one trace. `citation_artifact_ownership:511/533` are `sqlite_master` lookups over 2 names. The 3 `console_context_repository` rows (1411, 1482, 1526) are per-conversation/per-message lineage reads, bounded by conversation length. Exceptions, both **out of slice**: `conversation_local_marks_service.py:449-466 list_console_unseen_marks` (no SQL `LIMIT`; truncated in Python after `fetchall()`) and `:474 has_console_unseen_marks` (`fetchall()` of every unseen mark to answer a boolean; `LIMIT 1` would do). |
| `function_body_import_per_file` ×15 files | **examined; all resolve.** `ruff check --select E9,F63,F7,F82 tldw_chatbook/Chat/` → **All checks passed!**, and every module I imported loaded. Cycle-breaking or optional-dep in each case I read (`attachment_core`'s `config`/`optional_deps`/`PIL`; `chat_persistence_service`'s `Backup_Recovery`). The `console_conversation_hydration` one is a **layering** problem (P2 above), not an import-resolution one. One redundant stdlib import: `attachment_core.py:333 from io import BytesIO`. |
| `legacy_markers_per_file` ×17 | **retired** — sampled `citation_legacy_migration.py` (15) and `citation_trace_adapters.py` (19): these are `legacy_inferred`/`legacy_conversation_id` **domain vocabulary** for the pre-canonical citation origin, not TODO/deprecation markers; `console_context_compaction.py`'s 22 are `LegacyMemorySnapshot`/`_NoLegacyMemory` type names. Mechanical false positive. |
| `lock_and_execute` `citation_trace_repository.py:0 locks=1 executes=62` | **retired** — the one lock (`:454`) guards *barrier registration only* (`:467-…`), never SQL. The 62 `execute`s are cursor-scoped inside `db.transaction()` or read-only `get_connection().execute(...)`. |
| `loguru_and_logging` `Chat_Functions.py` | **confirmed** → P3 finding (no secret leak; each stdlib call logs lengths/bools/type names) |
| `raw_1024x1024` ×11 | **retired** — all are byte-limit constants (`100*1024*1024`, `10*1024*1024`, `16*1024`) and `_format_size`'s divisors. No image-dimension literal among them. |
| `seed_name__clean_text` `chat_conversation_service.py:27` | **confirmed** — see the `_clean_text` family row |
| `seed_name__format_size` `attachment_core.py:64` | **confirmed as documented duplication** |
| `seed_name__normalize_mode` ×2 (`chat_loop_scope_service:22`, `chat_conversation_scope_service:64`) | **retired** — different vocabularies (a `ChatLoopBackend` enum vs a `"local"|"server"` string) and different failure modes. Not the same function. |
| `seed_name__now` `chat_conversation_service.py:321` | **retired as a storage risk** — its single use is `record_message_rag_context`'s `last_modified` in a JSON sidecar file, single-writer, never compared against a DB timestamp. |
| `strftime` ×2 `Chat_Functions.py:2658, 2745` (`%Y%m%d_%H%M%S`) | **confirmed but P3** — both use naive `datetime.now()` (local time, no tz) for an export *filename* and the export JSON's `"timestamp"` field. Filename is fine; the JSON field is local-time in a document whose other timestamps are UTC ISO. Cosmetic. |
| `try_import_guard` `Chat_Functions.py:1128, 1172` | **retired** — both are `from .usage_recorder import active_recorder` inside `try/except Exception` with the rationale `# accounting must never break a call`, and both log at debug. Not an optional-dep guard. |
| `try_import_guard` `chat_persistence_service.py:227` | **confirmed but P3** — `RecoveredMessageReferences` binding failure degrades with `logger.debug` only, so recovered-media cleanup can silently never run. Logged, but at debug. |
| `try_import_guard` `console_assistant_defaults.py:54` | **confirmed** — same row as its `except_exception_return` above (P3, no logging) |

## Verified-fine

- **No timers, no `query_one`, no `run_worker`, no `call_from_thread` anywhere in the 50 files.** `grep -nE "run_worker|query_one|set_interval|set_timer|call_from_thread"` over the slice returns two hits, both inside *comments*. The slice is genuinely pure-logic + repository code, so the "unguarded `query_one` in a timer callback" class cannot occur here.
- **No mutable default arguments and no shared mutable class attributes.** AST scan over all 50 files → `mutable default args: NONE`; `mutable class attrs: ['console_activity_receipts.py:62 ConsoleActivityReceiptService._FLEET_STATUS']`, a read-only lookup table.
- **No `except:`, `eval(`, `exec(`, `pickle.`, `subprocess`, or `os.system` in the slice.**
- **Every `re.compile` in the slice is at module scope** (17 hits, all top-level constants). The one in-function regex is `console_context_compaction.py:2554`'s `re.search` with a literal pattern, which `re`'s own 512-entry cache covers.
- **`ruff check --select E9,F63,F7,F82 tldw_chatbook/Chat/` → All checks passed!** (matches the stated 0-fatal Tier-1 baseline).
- **Only two `get_cli_setting`/`load_settings` call sites in the whole 44k-line slice** (`attachment_core.py:99`, `Chat_Functions.py:2000`), neither per-tick nor in a `compose()`. `Chat/` is essentially free of the sibling's config-read hot-path class; the one instance is the P2 above.
- **`citation_trace_repository`'s `id()`-keyed capability registries are safe.** `_issued_prepared_writes`/`_issued_active_results`/`_issued_artifact_owner_requests` (`:429-447`) key on `id(obj)` but store a `weakref.ref` alongside and verify `issued[0]() is obj` before honouring a hit (`:3994, 4001-4003`), with a finaliser that pops only its own entry (`:3936-3940`). id-reuse after collection cannot produce a false positive — this is *not* the "recompose cache keyed by `id()`" hazard.
- **No bare DML on a raw `get_connection()`.** All 21 `get_connection()` sites in the slice are `SELECT`s (each checked: `chat_persistence_service:790, 986`; `citation_artifact_ownership:506`; `citation_legacy_migration:647, 1370, 1394`; `citation_trace_repository:387, 576, 1686, 1902, 2009, 2580, 2842, 3145, 3161, 3594, 3720, 3782, 4018, 4083`; `console_dispatch_repository:314`). Every write goes through `db.transaction()`.
- **`console_context_window.py` is a model for outbound HTTP**: `check_url_or_raise_async` + `origin_set` from `Utils/egress` before the request (`:176-180`), `follow_redirects=False`, a 256 KB streamed body cap (`:203-206`), `autoload=false` so metadata inspection cannot load a model as a side effect (`:189-193`), credentials excluded from `repr` via `field(repr=False)` (`:43`).
- **`citation_source_locators._safe_relative_path` (`:67-82`) is not a `path_validation` bypass.** It is a pydantic `AfterValidator` on a *locator string* that never touches the filesystem (no root to resolve against), and is strictly stronger than a traversal check: it rejects leading `/`, `\`, `~`, a drive prefix, any backslash, any colon, any control character, and any `""`/`"."`/`".."` segment.
- **The Chat side of the sibling's strict-JSON/`ContinuationValidationError` concern (LLM P2) does not fire.** All three Chat-side call sites are safe: `Chat_Functions.py:2823` uses the *tolerant* `read_provider_continuation_json` (`provider_continuation.py:577-589`, `except Exception: return SafeContinuationRead(checkpoint=None, warning=…)`), and the three `parse_provider_continuation_json` sites (`console_dispatch_repository.py:1025, 1183, 1318`) each catch `ContinuationValidationError` and convert to `ConsoleDispatchCheckpointValidationError`. The missing depth cap is also moot at that boundary because its `except Exception` catches `RecursionError`.
- **The `Chat/` timestamp cluster is otherwise single-shape per table.** `console_auxiliary_attempts.started_at/finished_at` has one writer (`console_context_compaction.py:1905, 2198, 2521` → `+00:00`); `rag_payload_tombstones.retain_until` has one writer and its SQL `max()` over `+00:00` ISO strings is lexicographically monotone (`'+'` = 0x2B sorts below every digit and below `'.'`, so a no-fraction value correctly precedes a fractional one in the same second). `citation_trace_repository.py:317`'s `datetime.fromisoformat(row["observed_at"])` without `Z` handling is fine on 3.12.
- **`console_cost_tracker.TokenEstimateCache`** verifies every hit against the full `(model, provider, rows)` signature before serving it (`:145-147`), so a key collision costs a recompute and never a wrong number. LRU-bounded at 4096.
- **`character_expression_playback.prepare_expression`** accounts its RGBA budget under `_BUDGET_LOCK`, reserves before `n_frames`/`seek`, releases the over-reservation exactly once, and has a `finally` that releases on every failure path. `frame_at`'s `elapsed_ms % total` cannot divide by zero: the all-zero-delay case is collapsed to a single frame at `:201-213` and the single-frame case returns early at `:80-81`.

## Retired

- **`ServerChatLoopScopeService._maybe_await(sync_call())` is not an event-loop block.** Symptom real (all five public methods evaluate `self.server_service.<call>(...)` eagerly before awaiting: `chat_loop_scope_service.py:107, 121, 138, 155, 166`); cause wrong. `grep -rn "ServerChatLoopScopeService" --include='*.py' .` → constructed **only** in `Tests/Chat/test_server_chat_loop_service.py:198, 234, 250`; production re-exports it from `Chat/__init__.py:7` and never instantiates it. `grep -n "async def" tldw_chatbook/Chat/server_chat_loop_service.py` → the only `server_service` implementation defines `start_run`/`list_events`/`approve`/`reject`/`cancel` all as `async def`, so `_maybe_await` always receives a coroutine. The file remains a **dead production surface** (D3) that never received task-283's `asyncio.to_thread` treatment — the wrong template for whoever later wires it up.
- **`console_conversation_activation._maybe_await`** — the only sync injected callback does no I/O.
- **`citation_repair.py:391` / `citation_artifact_ownership.py:384, 439` / `console_chat_models.py:430` swallows** — each converts to a typed result carrying its reason, or is a documented best-effort note inside an already-recorded failure.
- **`lock_and_execute` on `citation_trace_repository.py`** — the lock never spans SQL.
- **`fetchall_dynamic_sql` ×5** — the "dynamic" SQL is module-constant concatenation with no interpolation.
- **`raw_1024x1024` ×11 and `legacy_markers` ×17** — mechanical false positives (byte-limit arithmetic; domain vocabulary).
- **Deep-JSON `RecursionError` in the slice** — the one strict-JSON parser in my 50 files (`console_dispatch_checkpoint.py:289-291`) catches it explicitly. I confirmed the hazard is real in principle (`json.loads('{"a":'*20000 + '1' + '}'*20000, object_pairs_hook=dict)` → `RecursionError; isinstance ValueError: False`; it parses fine at 5000, so the required depth is large) but it does not land here.

## Adjacent (out of slice, for routing — not mine to own)

- `Chat/console_trace_regex_worker.py:231-233` parses the PII worker subprocess's stdout with `json.loads(..., object_pairs_hook=_unique_json_object)` and catches `(UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError)` — **not** `RecursionError`, unlike its sibling copies at `Chat/library_activity.py:501` and `Chat/library_preparation.py:190`. Same gap at `Tools/watchlists_tool_service.py:1034, 1201` and `MCP/permission_store.py:810, 858`.
- `Prompt_Management/Prompt_Engineering.py:12` imports `from tldw_Server_API.app.core.Chat.Chat_Functions import chat_api_call` — a package that does not exist in this repo. Noticed while enumerating `Chat_Functions` importers.
- `Chat/conversation_local_marks_service.py:449-466, 472-478` — `fetchall()` with no SQL `LIMIT` (one truncates in Python afterwards; the other materialises every row to answer a boolean).

## Left UNVERIFIED

| claim | why not verified | literal command to run |
|---|---|---|
| The P1 escaping drift is visible on screen (Console staged-evidence strip / inspector tray shows `R&amp;D Report`) | Brief forbids running the app; the drift is proven at the helper level and the widget class is the same `Static` the Library live-UAT used | `.claude/skills/verify/SKILL.md` recipe: `tmux -L verify new-session -d -s v 'cd /Users/…/tldw_chatbook && .venv/bin/python -m tldw_chatbook.app'`, stage a Library note titled `R&D Report` as Console evidence, then `tmux -L verify capture-pane -p -t v | rg 'R&'` |
| `_empty_profile_context_snapshot` is a verbatim duplicate between `console_chat_controller.py:3537` and `console_chat_models.py:1395` | Both sites lie outside my read ranges (the controller is another reviewer's slice) | `diff <(sed -n '3537,3550p' tldw_chatbook/Chat/console_chat_controller.py) <(sed -n '1395,1408p' tldw_chatbook/Chat/console_chat_models.py)` |
| `PreparedExpression.close()` is always called, so the 64 MB `_preparation_bytes` budget cannot leak permanently | The callers are in the UI layer, outside my slice; the contract is documented (`character_expression_playback.py:70`) but unenforced | `rg -n "PreparedExpression|prepare_expression|\.close\(\)" tldw_chatbook/UI tldw_chatbook/Widgets`, then check every `prepare_expression` result reaches a `close()` on unmount |
| The "~5 config reads per image attachment" fan-out count | Counted statically from call sites; not instrumented at runtime | `cd $WT && source <SCRATCH>/env.sh && PYTHONPATH=$WT $PY -c "import tldw_chatbook.config as c, asyncio; n=[0]; f=c.get_cli_setting; c.get_cli_setting=lambda *a,**k:(n.__setitem__(0,n[0]+1), f(*a,**k))[1]; from tldw_chatbook.Chat.attachment_core import process_attachment_path; asyncio.run(process_attachment_path('/path/to/test.png')); print(n[0])"` |
| `citation_evidence_models.py` (714), `citation_trace_adapters.py` (402), `citation_trace_identity.py` (547), `citation_trace_builder.py` (898), `console_context_policy.py` (470), `console_conversation_actions.py` (343), `chat_handoff_*.py` (310), `answer_citations.py` (379) contain no D1 | Mechanical-only coverage — pattern sweeps passed but I did not read their logic | `cd $WT && source <SCRATCH>/env.sh && $PY -m pytest Tests/Chat/ -q` plus a close read of each |
