# tldw_chatbook bug sweep / best-practices audit — 2026-11-02

Base: `dev` @ `3755417af5` **plus the uncommitted staged working tree** (the in-flight
radio-portraits / console first-send refactor). Method: 14 parallel per-module audit
agents (deep-read of central + recently-changed + highest-risk files, pattern sweeps for
bug classes, targeted live reproductions), plus a mechanical `ruff --select F,B,ASYNC,PLE`
pass over the whole package, plus independent re-verification of every P0 claim by the
coordinating session. Read-only: no files were modified.

Every P0 below was independently re-verified (signature/MRO inspection, code reading, or
test reproduction) after the module agent reported it.

---

## 1. Critical — verified breakage on committed dev (P0/P1)

### 1.1 [P0] Every shared-service RAG search raises `TypeError`
`RAG_Search/simplified/enhanced_rag_service_v2.py:~353-363` forwards `query_embedding=query_embedding`
to `super().search(...)`, but `RAGService.search` (`rag_service.py:1205`) has no such parameter
(verified by signature inspection: base params are `query, top_k, search_type, filter_metadata,
include_citations, score_threshold, metadata_allowlist, keyword_source_types`). `get_shared_rag_service()`
returns `EnhancedRAGServiceV2`, so the Library RAG answer path, chat sidebar, and MCP search all enter
through the broken override — every call raises `TypeError`, including the default `query_embedding=None`.
Introduced by commit `859c76f104` (2026-10-11, TASK-34390), whose message claims a passthrough that was
only ever added to `_semantic_search`, not to `RAGService.search`. Unit tests stay green because their
doubles mirror `RAGService.search` instead of driving the real V2→base chain — the exact trap the V2
docstring documents for an earlier, identical incident (`keyword_source_types`).
**Fix:** add `query_embedding: Any = None` to `RAGService.search` and thread it into
`_semantic_search_scoped`/`_semantic_search` (or drop the kwarg from the V2 forwarding), and add one
contract test that drives the real V2→RAGService chain.

### 1.2 [P0] Audio pause/resume self-deadlock — one Pause click wedges all TTS audio for the session
`TTS/audio_player.py:203` uses a non-reentrant `threading.Lock`; `pause()` (lines 445-461) takes the
lock, then calls `self.get_position()` and `self.stop()` — both re-acquire the same lock. `resume()`
has the same shape via `self.play()`. Affects every player without native pause: **afplay (macOS
default)**, ffplay/paplay/pw-play/aplay on Linux. Runtime-proven by the audit agent (constructed the
real player; `pause()` never returned in 3s, `get_state()` then also blocked). The Speech-Lab Pause
button (`UI/Speech/speech_playground_pane.py:211` → `speech_playback_mixin.py:864` → single-worker
executor) deadlocks the executor thread permanently; later `AsyncAudioPlayer` calls queue forever, and
TTS stop paths calling `stop()` via `asyncio.to_thread` block that thread too.
**Fix:** `threading.RLock()`, or release the lock before calling `get_position()`/`stop()`/`play()`.

### 1.3 [P1] stdio MCP wire timeout hardcoded to 10s — kills the server and ignores config
`MCP/client.py:51` `_REQUEST_TIMEOUT_SECONDS = 10.0`; `connect_to_server` never receives the configured
`[mcp] tool_call_timeout_seconds` (default 60s at `unified_control_plane_service.py:2776`). On timeout,
`request()` tears down and **kills the server process** (verified live: returncode −15, misattributed
`mcp_transport_unavailable`). Any stdio tool legitimately slower than 10s fails, and in passive-hook
contexts (`hooks may not launch`) one slow call bricks every later MCP call in that scope.
**Fix:** plumb the configured timeout into the session; only tear down on transport uncertainty, not a
plain per-request deadline.

### 1.4 [P1] Eval LogProbRunner fabricates predictions; A/B testing pairs wrong samples together
- `Evals/eval_runner.py:2477` — `_get_text_logprob` returns `-inf` for every choice (OpenAI branch fires
  a real, paid request first, then discards it); `max()` over equal values always picks index 0. Every
  `task_type="logprob"` run scores "always answer A", wastes one API call per choice, and persists
  `-Infinity` into metrics (invalid strict JSON on export). Verified live by the agent.
- `Evals/ab_testing.py:284,408,446` — results from the two concurrently-running models are zipped by
  completion order (`ORDER BY created_at`, second-resolution) rather than paired by sample id; the
  merged view shows model B's answer to a *different question* beside model A's, and paired CIs are
  computed across mismatched samples.
**Fix:** make LogProb honestly fail (or wire real logprobs); pair A/B samples by `sample_id` intersection.

### 1.5 [P1] Model-authored canvas title rendered as live Rich markup — crash loop
`Widgets/Console/console_canvas_card.py:179` renders the card label (containing the model-chosen canvas
title, length-only-validated) in a `Static` without `markup=False`/escaping. A title containing `[/]`
raises `MarkupError` out of the app run; because the title persists in message metadata, the crash
recurs on every transcript render of that message. Bracketed text (`[DRAFT]`) is silently swallowed.
Both variants reproduced live on Textual 8.2.8. Everywhere else in the transcript escapes; this is the
outlier. Same class, lower frequency: `Widgets/conversation_selection_dialog.py:41` (user conversation
title unescaped).
**Fix:** `Static(escape_markup(label), markup=False, …)`.

### 1.6 [P1] Watchlists sources pane: phantom RowHighlighted hijacks selection
`UI/Watchlists_Modules/sources_pane.py:1785-1812` — table rebuilds post a phantom `RowHighlighted` that
`select_source_by_id` acts on; the other five panes in the package guard with `highlight_is_user_driven`
(`Widgets/table_click_select.py:120`), this one doesn't. Typing in the sources search box can retarget
the inspector — Check-now/Preview/Delete act on a row the user never chose. Phantom event arrival
empirically verified with a Textual pilot.
**Fix:** add the shared guard (or the `DataTableClickSelectMixin`).

### 1.7 [P2→notable] Other committed-dev correctness bugs (condensed)
- `DB/Client_Media_DB_v2.py:2497` — FTS5 external-content update via `INSERT OR REPLACE` never purges
  old tokens (reproduced in-memory: old words stay searchable forever after overwrite).
- `LLM_Calls/LLM_API_Calls.py:931,309,3960` + `LLM_API_Calls_Local.py:347` — `iter_lines(decode_unicode=True)`
  decodes SSE as ISO-8859-1 when the endpoint omits `charset=utf-8` (common for local llama.cpp/vLLM
  endpoints) → silent mojibake of all non-ASCII output.
- `Embeddings/Embeddings_Lib.py:581` — OpenAI-compatible embeddings responses not sorted by `index`;
  out-of-order providers silently mis-associate vectors with chunks, and the ADR-223 SQLite cache then
  persists the wrong vector keyed by the correct text hash.
- `RAG_Search/simplified/vector_store.py:591-594,1579-1581` — transient Chroma errors are swallowed to
  `[]` and then **cached empty for the TTL hour**; InMemoryVectorStore l2 scores double-normalized and
  inverted (threshold filters backwards); score scales differ between backends.
- `RAG_Search/simplified/indexing_helpers.py:307-323` — partial embedding failure silently loses chunks
  forever (marked indexed, backfill skips them, no user-visible signal).
- `MCP/permission_store.py:213-223` — publication-fence `uncertain` flag is never cleared; one transient
  disk-full during a permission save permanently fails-closed all passive-hook MCP grants until restart
  (introduced by committed `e7dc47f289`).
- `Chatbooks/chatbook_importer.py` + wizard — REPLACE and MERGE conflict strategies are offered in the
  UI but never implemented (REPLACE creates a duplicate-titled note; characters fail with ConflictError;
  MERGE silently becomes RENAME; `merge_notes` is dead code).
- `UI/Wizards/ChatbookImportWizard.py:400,1060` (+ CreationWizard) — full import/export runs
  synchronously on the event loop inside async workers → whole TUI freezes for the duration.
- `Tools/tool_executor.py:285-290` — CalculatorTool silently discards keyword args: `round(x, ndigits=2)`
  → wrong numeric result from an always-on auto-approved tool.
- `Evals/eval_runner.py:1404,2274` — `stop_sequences` silently dropped by the param whitelist (verified
  live); `_extract_choice` fabricates "A" when nothing matches (biases accuracy toward first choice).
- `Utils/optional_deps.py:639-651` — an installed-but-broken optional package (e.g. broken torch dlopen)
  raises non-ImportError out of `check_dependency`; with eager checking enabled the app cannot boot.
- `Web_Scraping/Article_Extractor_Lib.py:537` — timeout path retries with no backoff and never raises
  `MaxRetriesExceededError`; `retries = 0` returns `""` silently.
- `Skills_Interop/skill_remote_fetch.py:282` — SSRF check-then-connect gap: no `trust_env=False`, no IP
  pinning → DNS rebinding / proxy-env bypass of the validated address (MCP's own HTTP transport does
  this correctly).
- `Character_Chat/character_card_formats.py:115-121` — chub format detector unreachable (v2 pattern
  matches first); chub/agnai fields silently dropped.
- `Chat/Chat_Functions.py:2303-2319` — legacy `chat()` returns raw exception text *as the assistant
  reply* (reachable via `worker_events.chat_wrapper`).
- `Chat/console_references.py:214-222` — `_looks_binary` refuses UTF-8 files whose first 8 KB are
  >30 % non-ASCII (CJK notes silently not injected via `@file`).

---

## 2. The dominant risk: half-landed staged work in the index

The uncommitted staged tree (radio-portraits + console first-send refactor) is **incomplete and partly
dead-on-arrival**. If committed as-is it breaks at import/runtime:

- **DB (P0):** three staged ChaChaNotes migrations (`v81_to_v82_canvas_sibling_sequences.sql`,
  `v82_to_v83_portrait_performance.sql`, `v83_to_v84_portrait_regions.sql`) are never applied —
  `_CURRENT_SCHEMA_VERSION` is still 81 and `migration_steps` ends at `_migrate_from_v80_to_v81`
  (verified). Four call sites (`DB/portrait_region_settings.py:208`,
  `UI/Persona_Modules/portrait_performance.py:357`, `portrait_regions.py:57`,
  `portrait_composition_generation.py:159`) call `db.get_character_portrait_performance` /
  `get_character_portrait_regions` — **neither method exists anywhere** (verified). Any portrait UI
  action → `AttributeError`. Recovery schema catalog also stops at V81.
- **Chat (P1):** staged `console_library_search.py` and `console_unsent_turn.py` fail to import
  (`ImportError: cannot import name 'LIBRARY_INITIALIZATION_TIMEOUT_SECONDS' / 'SHELF_RELEASABLE_PAUSES'`
  — verified by the agent with the repo venv). Staged `console_local_review_hook.py:116` calls
  `provider.apply_index_read_decisions(...)`, which `LocalToolProvider` does not define. Four more
  staged modules (`console_first_send_modules`, `console_library_preparation`,
  `console_project_index_preparation`, `console_response_trace`) have zero importers. Meanwhile the
  user-facing bug they were written to fix is still live: first Automatic-Library send still runs the
  RAG runtime build inside the search timeout budget in `console_chat_controller.py:7404-7424`.
- **Agents (P1):** staged `first_request_schemas.py` is DOA — imports `FirstRequestSchemaPlan` from
  `agent_models` (it lives in `agent_service.py`) and calls `dataclasses.replace(...,
  schema_input_budget_tokens=...)` on a field that doesn't exist (both verified live). Staged
  `tool_disclosure_diagnostics.py` has zero callers and hardcodes `agent_kind=primary`.
- **UI (P2):** the portrait-performance UI is entirely unwired — `portrait_performance_editor.py` posts
  `CharacterPerformanceRequested` (no handler exists anywhere), `PortraitPerformanceController` is never
  instantiated, `portrait_radio.py` / `compose_delta.py` / `library_note_link_picker.py` /
  `composer_paste_panel.py` have zero callers/mount sites.
- **Audio (P0):** staged caller landed ahead of its dependency —
  `UI/Persona_Modules/portrait_performance.py:775` passes `presentation=` to
  `TTSEventHandler.speak_guarded_utterance` (`tts_events.py:994`), which doesn't accept it (verified):
  every confirmed "Play sample" raises `TypeError`. The sink-side half of task-36544 is genuinely
  missing (nothing publishes to `SpeechPresentation` yet), and the new call path has zero test coverage.
- **Working-tree test health:** every UI test that constructs the real `TldwCli` dies at bootstrap with
  `Backup_Recovery/raw_participants.py:275: RecoveryRequired: raw_source_selection_changed`
  (reproduced: `Tests/UI/test_watchlists_source_row_click_selects.py` → 1 failed in 1.0 s). Root cause is
  in the raw-participant registration gate, likely staged fallout — needs triage by the staged-work owner.
- Also staged-red: `Tests/DB/test_migration_file_sources.py` fails on disk and already failed at HEAD for
  a separate detection gap (7 false positives — runtime-read migrations via `with_name(...)` the guard
  regex can't see — drowning the signal it exists to provide).

**Recommendation:** treat the staged set as one unit: finish or unstage. Do not push/commit it piecemeal.

---

## 3. Module-by-module summaries

**Chat/** — Committed engine (controller/store/bridge/custody/reference-authority/scratch-space) is
exceptionally disciplined: no findings in stop/abort paths, lock nesting, SQL, secrets handling, httpx
lifecycle. Problems concentrate in the staged half-refactor (above) plus: `document_generator.py`
bypasses `API_CALL_HANDLERS` with a stale provider dict (moonshot/zai/etc. raise "Unknown provider");
legacy `chat()` returns exception text as the reply; `_looks_binary` refuses CJK-heavy UTF-8.

**Agents/ + Tools/ + runtime_policy/** — Core seams heavily hardened (fs_* sandbox choke point with
per-entry rechecks, process-group kill with honest custody retention, deny-by-default policy engine);
no `AVAILABLE_TOOLS` global; argv-only hooks. Findings: staged DOA modules (above); Calculator kwarg
drop (1.7); `invoke_by_name` can raise on non-finite floats despite a never-raise contract (contained
by callers); `_notify` missing the `is_closed()` guard its sibling has; unbounded `_WRITE_LOCKS` growth;
final hook-drain stage is an unbounded wait.

**MCP/ + Plugins/ + interop** — Security-first and matching `Docs/Architecture/mcp-hub.md`; grant
revalidation/refresh sound (no stale-grant/revocation-bypass constructed); injection, spawn-guard,
redaction all clean. Findings: the 10s wire timeout (1.3); fence no-recovery (1.7); SSRF pinning gap in
skill fetch (1.7); settlement can wait forever on an unreapable child; connection killed for failures of
requests never sent.

**DB/** — Core machinery mature: statement-by-statement migrations in transactions with re-entry guards,
BaseException rollback everywhere, optimistic locking in WHERE, no injection path found, all modules
import clean. Findings: staged portrait schema unwired (section 2); migration-guard test red (section 2);
Client_Media FTS drift (1.7); `vacuum()`/`check_integrity()` leak the connection on error; new
`Workspace_Project_Index_DB` splits SQL on raw `";"` (fragile) and has zero test coverage.

**UI/Screens + Navigation** — Navigation contract, keybinding conventions (ADR-031/152), worker/thread
discipline, reactive usage, design tokens: all honored and largely test-enforced. Findings: dead
`ViewChangeMessage.Requested` posts (7 sites); `ContentsRebuilt`/`ChunkingTemplatesChanged`/
`ReminderFormSubmitted` dead messages; 37 unused imports in the two monolith screens; no lint gate in CI.

**UI/Console_Modules + Speech + Widgets/** — Transcript pruning, streaming fence-parity throttle,
playback-timer retirement, dictation timers: all clean under deep reading. Findings: canvas-card markup
crash (1.5); conversation-dialog title unescaped; staged portrait UI unwired (section 2); orphaned
`Widgets/Media/` (3.1k lines, test-only); `Widgets/NewIngest/` is a test-compat shim (don't "fix" it);
`form_components.create_form_field` select branch silently drops `default_value`.

**Other UI module groups** — Watchlists phantom-selection (1.6); chatbook wizards block the loop (1.7);
legacy `Dictation_Window_Improved`/`Voice_Cloning_Window` bind banned keys (ctrl+c/p/d/r, f1);
`Outputs_Panel`/`Sharing_Panel` unreachable (~1.2k dead lines); all 69 custom messages verified handled
except the dead ones noted; every staged `Settings_Modules` file clean.

**LLM stack** — Registration parity complete and test-enforced both directions; hosted SSE state machine
rigorous; catalog refresh single-writer/bounded/atomic; no key logging. Findings: charset mojibake (1.7);
`add_model_capability` leaves `is_vision_capable` lru_cache stale (latent); local sessions built outside
the ADR-222 registry; dead double-brace debug log; embeddings error body logged raw; streaming timeout
ignores config in `chat_with_openai`; minor doc drift.

**RAG stack** — FTS5 escaping airtight; incremental timestamps round-trip exactly; chunking/fusion
correct; daemon races handled; optional deps guarded. Findings: the P0 (1.1); embeddings order (1.7);
cached-empty / silent chunk loss / score-scale issues (1.7); one red test from a stale double
(`test_semantic_honest_states.py:645`); `rag.md` doc drift re ADR-223 cache.

**Notes/ + Character_Chat/ + Chatbooks/** — Notes sync invariants all actually enforced (conflict always
surfaced, journal-before-mutation, crash resume verified, watcher races serialized, quarantine deletes);
portrait pipeline trio clean. Findings: chatbook REPLACE/MERGE advertised but unimplemented (1.7);
chub/agnai parse gaps; unbounded card-file reads (hardening); `_json_depth_is_bounded` dead.

**Audio/ + TTS/ + STT/ + Media** — PCM streaming stack and STT lifecycle clean. Findings: pause deadlock
(1.2); staged presentation kwarg (section 2); VAD path drops sub-frame tail bytes for non-default buffer
durations; `stop_recording` can raise out of `terminate()`; 9 unused imports.

**Evals/ + Library/ + Local_Ingestion + Web_Scraping + Subscriptions** — Fresh source-reader code
excellent (fail-closed, isolated experiments); ingest job state machine guarded; SSRF-guarded fetches;
idempotent subscription upserts. Findings: LogProb + A/B (1.4); `_extract_choice` fabrication and
`stop_sequences` drop (1.7); scraper timeout path; CI z-value hardcoded at 1.96; BLEU math nonstandard;
orchestrator mutates caller's overrides dict.

**App core + Utils** — Security-critical surfaces clean (AES-256-GCM with per-value salt+nonce, atomic
TOML writes with parse-back, cipher-as-credential fenced everywhere, path/SQL/input validators solid,
egress SSRF guard thorough, navigation/shutdown disciplined). Findings: `check_dependency` boot crash
(1.7); dead `tab_events.py` module; dead `model_download` branch; scrypt N=2¹⁴ below guidance (deliberate
tradeoff, documented); dead `encrypt_config()` bypassing the sensitive-key predicate; every settings save
strips all config.toml comments (may be accepted); `state/` mostly test-only with a plaintext-password
field waiting for an accidental writer.

**Services + long tail** — No P0/P1: web surfaces fail-closed (CSRF, rate limits, constant-time compares,
containment), Image/Video generation match their ADR design, scheduler heavily defended. Hygiene:
`WebClipper/`, `Sharing/`, `Outputs/` are bytecode-only husks (live code moved to `*_Interop`); latent
scheduler poison-row crash on naive datetime (hand-edited data only); artifact-share controller allows
unauthenticated non-loopback bind with the gate only in the UI dialog; symmetric unused-import cruft
copied between Image/Video packages; two planning docs shipped inside the package.

---

## 4. Mechanical pass (ruff `F,B,ASYNC,PLE`, whole package)

1,219 findings. Breakdown: `PLE1205` 236 (logging-format), `B904` 233 (raise-without-from),
`B905` 190 (zip without strict=), `F401` 126 (unused imports), `B009`/`B007`/`B010` ~170,
`B023` 36 (loop-variable capture in closures — the ones spot-checked by agents were false positives,
but worth a triage), `B006` 14 (mutable defaults), `F821` 8 (all checked by agents to be
annotation-only under `from __future__ import annotations` — no runtime bugs), rest singletons.
Full list: `/tmp/ruff_audit_full.txt`. Zero F821 runtime issues; `compileall` clean on 3.12.11.

**Gap:** none of the 24 CI workflows runs any linter.

---

## 5. Cross-cutting recommendations

1. **Finish or unstage the staged set as one unit** (section 2) — it contains import-broken files,
   calls to nonexistent methods, and unwired UI; committing piecemeal will break dev for everyone.
2. **Add a signature-contract test for the RAG V2→base chain** — this class of drift has now happened
   twice on the same override; test doubles mask it.
3. **Add a lint gate to CI** (`ruff check --select F,B,ASYNC,PLE` at minimum, `--fix` for F401 first).
4. **Fix the migration-hygiene guard test** (extend its regex to `with_name(...)`/f-string readers) so
   its signal is trustworthy again; it is currently red at HEAD.
5. **Wire `tool_call_timeout_seconds` through the stdio MCP session** and stop killing servers on
   per-request deadlines.
6. Consider a tiny "escape or `markup=False`" lint/idiom check for every `Static`/`Label` fed
   model- or user-authored strings (two live instances found; the codebase otherwise does this right).

## 6. Coverage honesty

~1.6 M LOC across 100+ packages; deep-read focused on the central, recently-changed, and
highest-risk files in each module (~180 files deep-read total), with pattern sweeps and targeted live
reproductions covering the remainder. The long tail of small `*_Interop` packages was audited at
hygiene level only. Not audited: `Tests/` as a subject in itself, CSS token values, docs content.
