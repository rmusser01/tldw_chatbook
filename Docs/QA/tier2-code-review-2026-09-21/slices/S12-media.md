# S12 — `Media/` + `Media_Creation/` + `Media_Playback/` + `Image_Generation/` + `Video_Generation/`

**Coverage:** files read in full: 11 | sampled: 17 | mechanical only: 24 (of 52).
Full: `Media_Playback/{player_pipeline,stream_resolve,availability}.py`,
`Image_Generation/adapters/image_format_utils.py`, `prompt_refinement.py`, both `worker.py`, the trivial leaves.
Sampled: `Media/local_media_reading_service.py` (~900 of 6,752 + full AST scan), `media_reading_scope_service.py`,
`server_media_reading_service.py`, `comfyui_image_adapter.py`, both `config.py`, both `request_validation.py`,
`http_client.py`, `swarmui_adapter.py`, `video_store.py`, `comfyui_video_adapter.py`, `swarmui_client.py`,
`image_generation_service.py`, `generation_templates.py`, `media_reading_normalizers.py`, both `adapter_registry.py`.
Mechanical only: 24 — swept for `re.compile`-in-body, dotted `get_cli_setting`, loguru+stdlib mixing,
`exclusive=True`, subprocess-without-timeout, and Pillow ingress. All clean on those axes.

## Findings

### P1 [D1] — Every image adapter's format-conversion path decodes backend-supplied image bytes with no pixel cap; an 87 KB PNG expands to 90.25 M pixels and converts successfully
- Where: `Image_Generation/adapters/image_format_utils.py:140-170` (`maybe_convert_format`), reached from
  `validate_and_convert_image_output:173` — called by swarmui:62,77, fal:248, gemini:165, novita:52,
  modelstudio:83, openrouter:78, together:60, stable_diffusion_cpp:117.
- Evidence: hand-built 9500×9500 8-bit-grayscale PNG (**87,802 bytes, 90,250,000 px**) through
  `validate_and_convert_image_output(data, "image/png", "jpg", max_bytes=10MB)` → **no exception, conversion
  completed**; instrumented `Image.open` confirmed the call was reached.
  `grep "PILLOW_DECOMPRESSION" image_format_utils.py` → no match;
  `request_validation.PILLOW_DECOMPRESSION_WARNING_MAX_PIXELS = 89478485`.
- Why it matters: the only cap on this path is `_enforce_max_bytes` (**compressed** bytes). A hostile or compromised
  generation backend returns a small PNG whose IHDR declares huge dimensions; `img.convert()`/`img.save()` allocates
  W·H·3. Pillow's own backstop is 2×`MAX_IMAGE_PIXELS` (178,956,970 px ≈ 700 MB RGBA), so everything below that
  decodes. Measured amplification at the tested size: **87 KB → ~271 MB, ≈3,000×.** The bytes arrive from
  `fetch_image_bytes(url_from_backend_json)` — a real trust boundary.
- Recommended correction: apply the repo's existing guard at the top of `maybe_convert_format` — read
  `Image.open(...).size` before any `convert`/`save`, reject `w*h > PILLOW_DECOMPRESSION_WARNING_MAX_PIXELS`
  (already exported from `Image_Generation/request_validation.py:37`), and add
  `Image.DecompressionBombWarning`/`Error` to the `except`. **Canonical home already exists; this is adoption.**
- Size: S · Confidence: **verified**
- Already covered: **partially — TASK-32806.8 (In Progress).** The brief named `request_validation.py:36,337` and
  `comfyui_image_adapter.py:1372` as already carrying the guard; **the shared converter every *other* adapter routes
  through does not.** If .8 is scoped only to those two files it is now insufficient.

### P1 [D1] — 117 of 138 local-mode scope-service call sites run synchronous sqlite (and in one case a blocking HTTP scrape) inline on the event loop; only 21 use the threading seam built for exactly this
- Where: `Media/media_reading_scope_service.py` — `_call_local_leaf` (the correct seam) used at **21** sites; direct
  `await self._maybe_await(service.<method>())` at **117**, incl. `:1065 empty_media_trash`,
  `:1081 search_media_metadata`, `:1129 reprocess_media`, `:1381 permanently_delete_media_item`,
  `:1582/:1590 save_reading_item`.
- Evidence: `grep -c '_call_local_leaf('` → 22 (21 + def); `grep -c '_maybe_await('` → 119 (117 + def + the one
  inside `_call_local_leaf:197`). **Traced `:1582`** → `local_media_reading_service.py:2091 save_reading_item` →
  `:2124 scraper(normalized_url, ...)` → `:4276 _default_url_article_scraper`, which detects a running loop and
  submits to a `ThreadPoolExecutor(max_workers=1)` — then calls `future.result()` **with no timeout**, blocking the
  calling (loop) thread for the full duration of the HTTP article fetch. `:1065 empty_media_trash` →
  `local_media_reading_service.py:642`: unbounded `SELECT id FROM Media WHERE is_trash = 1` `.fetchall()` then a
  per-row `permanently_delete_item` loop, all on the loop.
- Why it matters: `_call_local_leaf`'s own docstring states the reason ("`run_worker(coroutine)` does NOT leave the
  event loop"). `save_reading_item` is the worst instance — **not slow sqlite but a network round-trip with no
  timeout freezing the UI.** *(Lead's note: this is the concrete `_maybe_await` D1 instance the review was looking
  for. I traced the `Character_Chat` scope service and found no live one there; this is the live one.)*
- Recommended correction: route the 117 through `_call_local_leaf` (mechanical — same signature); separately give
  `_default_url_article_scraper`'s `future.result()` a timeout.
- Size: M · Confidence: verified (code-traced; not observed live)
- Already covered: **TASK-32804.12 (To Do)** — this adds the exact count and identifies the network-I/O site as the
  priority within it.

### P1 [D1] — `_clone_git_repository` is the only `subprocess.run` in the slice with no `timeout=`; a hostile or slow git host wedges the ingest worker forever
- Where: `Media/local_media_reading_service.py:4680` (called from `:4491`).
- Evidence: 4 `subprocess.run` sites in the slice. `stream_resolve.py:186` has `timeout=YTDLP_TIMEOUT_SECONDS`,
  `player_pipeline.py:90` has `timeout=30`, `stable_diffusion_cpp_adapter.py:93` has
  `timeout=self._config.sd_cpp_timeout_seconds`. This one has none.
- Why it matters: `git clone` against a pathological remote never returns; the job is uncancellable and the worker
  is gone for the session. The same call also has no size bound — `_sync_filesystem_source_items:4503` then SHA-256s
  every file in the clone into an unbounded in-memory dict.
- Size: S (timeout) · Confidence: verified

### P1 [D1] — The yt-dlp stream branch egress-validates one hop, then hands the URL to ffmpeg, which follows redirects itself — the exact bypass the module's own redirect walk exists to prevent
- Where: `Media_Playback/stream_resolve.py:165-208` (`_resolve_with_ytdlp`) vs `:102-143` (`_walk_redirects`);
  consumed at `UI/Console_Modules/video.py:1487,1504` → `VideoPlayerScreen` → `player_pipeline.py:258 "-i",
  self._source`.
- Evidence: `_walk_redirects` validates *every* hop and returns the terminal URL; `_resolve_with_ytdlp` calls
  `_validate_later_hop(final)` **once** and returns. `resolve_stream_url:239` then only calls `_probe_ranges`
  (`follow_redirects=False`, so it does not walk). **The module docstring at `:8-12` states the walk exists because
  "ffmpeg would otherwise follow these redirects internally, outside the policy (AC2)".** `player_pipeline.start()`
  passes no `-protocol_whitelist` and libavformat's http protocol follows redirects by default.
- Why it matters: a vendor-resolved CDN URL that 302s to `169.254.169.254` or an RFC1918 address is followed by
  ffmpeg with no policy check. **The direct-media branch is protected; the yt-dlp branch is not.**
- Recommended correction: run the yt-dlp output back through `_walk_redirects`-with-no-trust (it already exists and
  is already exercised), and/or pass ffmpeg `-follow_redirects 0`. Prefer the former — one call.
- Size: S · Confidence: **inferred** (ffmpeg redirect behaviour not executed here)
- Pinning test: `Tests/Media_Playback/test_stream_resolve.py::test_html_page_falls_to_ytdlp` asserts only the single
  untrusted check. It does **not** state the one-hop scope as a requirement — a partial assertion, not a decision.

### P1 [D1] — Reading-list export silently loses body content on any per-item DB error, and silently truncates or empties on filters it cannot honour
- Where: `Media/local_media_reading_service.py:5714-5720` (`_local_export_detail_row`) and `:2792-2842`
  (`export_reading_items`).
- Evidence: `:5715-5717` — `try: detail = self.get_media_detail(row["id"]) / except Exception: return dict(row)`.
  The search row carries no `content`/`analysis_content`, so with `include_text=True` **that item exports with an
  empty body, no error, no counter, no log line**. `:2810-2812` — `if normalized_statuses and "saved" not in
  normalized_statuses: rows = []`; `:2813-2814` — `elif favorite is True: rows = []`. "Export my archived items"
  produces a syntactically valid, **empty** JSONL/ZIP. `:2822-2828` — the `domain` filter is applied in Python
  **after** `search_media(limit=size, offset=...)`, so a domain-filtered export only sees matches inside the first
  `size` rows (default 1000).
- Why it matters: three separate ways an export the user believes is complete is silently not. The
  `except Exception` one is a data-path swallow on the only surface that produces an archival artefact.
- Recommended correction: (a) record the failure into the export response's error list; (b) **raise** for
  locally-unsupported `status`/`favorite` filters — the module already raises for unsupported `archive_mode` at
  `:2107`, so the convention exists two functions away; (c) push `domain` into the query or page until exhausted.
- Size: M · Confidence: verified by reading, not executed
- Already covered: partially TASK-32811.4 — not the same surface; file separately or extend .4 explicitly.

### P2 [D3] — A 9-table, 4-index schema is created ad hoc by a service module on every call, outside `DB/migrations/`, with no schema-version bump and no `VALID_TABLES` entry
- Where: `Media/local_media_reading_service.py:5817-5920` (`_ensure_local_reading_aux_schema`), invoked from **32
  call sites**; a sibling `_ensure_local_ingestion_schema` does the same for the ingestion tables.
- Evidence: `VALID_TABLES['media']` contains **none** of `local_reading_saved_searches`, `local_reading_note_links`,
  `local_reading_archives`, `local_reading_highlights`, `local_document_annotations`,
  `local_reading_digest_schedules`, `local_reading_digest_outputs`, `local_file_artifacts`.
  `scripts/check_schema_table_allowlist.py` scans only `DB/migrations/chachanotes_*.sql` + `ChaChaNotes_DB.py`, **so
  the media DB is outside its reach.** No `index_plan_pin_census.tsv` row for the 4 indexes.
- Why it matters: precisely the TASK-20971 defect class (stale allowlist → `validate_table_name` rejects) on the one
  DB the preflight guard cannot see; and the 4 indexes have never had a query plan captured with `sqlite_stat1`
  absent (CLAUDE.md gotcha 1). Cost of the re-run itself is small: **measured 0.051 ms/call on-disk WAL vs 0.0019 ms
  for the query it guards** — 27×, but cheap; the governance angle is the finding.
- Size: L · ADR: yes (new, or fold into the migrations README) · Confidence: verified

### P2 [D1] — `video_store._atomic_publish` renames without `fsync`; the shared helper it declined does fsync
- Where: `Video_Generation/video_store.py:683-730` — `staged.flush()` … size check … `os.replace` at `:680-682`.
  No `os.fsync(staged.fileno())`.
- Why it matters: `flush()` moves bytes to the page cache only. After a crash the target can exist with zero or
  partial length, and the store's model is "the file is the artefact". **The non-adoption of `atomic_file_ops` is
  deliberate and correct** (this site needs the portalocker root lease, the `expected_size` re-check and the
  `VideoPublicationGate`) — the *fsync* is what was missed with it.
- Size: S · Confidence: verified
- Already covered: TASK-32808.5 is Done and this site is correctly out of its scope — report as a gap .5 does not
  close, not a .5 regression.

### P2 [D4] — `Video_Generation/config.py` re-rolled boolean coercion beside the site TASK-32808.4 consolidated, and drifted: it does not accept integers
- Where: `Video_Generation/config.py:333-342` (`_coerce_bool`), used at `:407 confirm_cost_estimate` and
  `:424 minimax_video_allow_uploads`. Sibling `Image_Generation/config.py:553,570` **did** adopt
  `Utils.Utils.coerce_bool_flag`.
- Evidence: the shared helper matches `isinstance(value, (str, int))`; the video copy matches `isinstance(value,
  str)` only. So TOML `minimax_video_allow_uploads = 1` → shared: `True`; video copy: falls through to `default`
  (`False`).
- Why it matters: TASK-32808.4's stated goal was one vocabulary; **the sibling module that `_parse_list`/
  `_coerce_choice`/`_get_config_value`/`_read_*_toml` are all verbatim-duplicated into was not swept.** The drift is
  a silent config no-op on an outbound-upload gate (fail-closed, so not a security hole — but the user sets it and
  nothing happens).
- Size: S · Confidence: verified
- Already covered: **TASK-32808.4 is Done — it is now incomplete.**

### P2 [D2] — Digest-schedule dedupe and retention purge both `fetchall()` every output row ever written, and the purge's `IN (…)` becomes a permanent wedge past the SQLite variable limit
- Where: `Media/local_media_reading_service.py:6292-6310` and `:6312-6346`. Both run
  `SELECT … WHERE schedule_id = ?` with **no time predicate** and `.fetchall()`, then filter in Python; the purge
  then builds `DELETE … WHERE id IN (<len(expired_ids) placeholders>)` at `:6340-6345` **with no chunking**. Both on
  the per-tick path from `run_due_reading_digest_schedules:3336`.
- Why it matters: cost grows without bound with schedule history (a 15-minute schedule ≈ 35k rows parsed in Python
  per tick after a year). **Worse: once one schedule accumulates more expired outputs than
  `SQLITE_MAX_VARIABLE_NUMBER` (32,766 modern, 999 older), the `DELETE` raises `OperationalError: too many SQL
  variables` on every subsequent run — the schedule is permanently wedged and the store never shrinks again.**
- Recommended correction: push both predicates into SQL; the purge becomes a single bounded `DELETE` with no
  variable list at all. · Size: S · Confidence: verified by reading
- Already covered: partially TASK-32803.3 (timestamp side), not the unbounded-`IN` wedge.

### P2 [D3] — `Media_Creation/` ships a complete second SwarmUI client that bypasses the egress, format-validation and pixel-guard layer `Image_Generation/` was built to be; only one pure function of it is reachable
- Where: `Media_Creation/swarmui_client.py` (426), `image_generation_service.py` (442), consumed only by
  `Event_Handlers/Media_Creation_Events/swarmui_events.py`, **which nothing imports**.
- Evidence: `grep -rn "Media_Creation_Events|swarmui_events"` → only the package's own `__init__.py`. The one live
  entry point is `Chat/console_generate_image.py:436-447`, which imports `ImageGenerationService` solely to call the
  **static-shaped** `extract_context_from_messages(None, shaped)` (its own comment at `:415` says it avoids
  `__init__`'s side effects). `SwarmUIClient` builds a raw `aiohttp.ClientSession` (`:139-146`) with **no**
  `Utils/egress` call, **no** `validate_and_convert_image_output`, and **no** byte cap;
  `Image_Generation/adapters/swarmui_adapter.py` is the maintained implementation of the same protocol with all three.
- Why it matters: two clients for one backend, one with none of the hardening. Dead today, but it is the copy a
  future contributor will find first via `from tldw_chatbook.Media_Creation import SwarmUIClient`.
- Size: M · Confidence: verified
- Already covered: none — TASK-32807 `.1`-`.6` enumerate widgets, RAG, Event_Handlers, Tools/Settings, Chat,
  Utils/Widgets — **not `Media_Creation`**. (`.3` may sweep the events file but would leave the two modules orphaned.)

### P2 [D4] — The ComfyUI **video** adapter accepts a server-supplied `filename`/`subfolder` with none of the validation its **image** twin applies to the identical descriptor
- Where: `Video_Generation/adapters/comfyui_video_adapter.py:966-984` and `:1002-1013` vs
  `Image_Generation/adapters/comfyui_image_adapter.py:947-971` (`_safe_filename`/`_safe_subfolder`) and `:974-991`.
- Evidence: `grep -rn "_safe_filename|_safe_subfolder|_SAFE_NAME" tldw_chatbook/Video_Generation/` → **no matches**.
  The video path checks only `isinstance(filename, str)`, non-blank, `type == "output"`, and
  `Path(filename).suffix == expected`, then `urlencode(descriptor)` into `/view?`.
- Why it matters: not exploitable client-side today (percent-encoded into a query string; only `suffix` reaches a
  local path), but two adapters against the same protocol with two different trust postures is how the next copy
  gets the weaker one. `_safe_filename` is 8 lines. · Size: S · Confidence: verified

### P3 [D1] — `_ensure_local_reading_aux_schema` runs `executescript` inside `db.transaction()`; `executescript` implicitly COMMITs
- Evidence: AST walk for calls at `with … transaction()` depth > 0 → **`[]` (zero today)**.
  `Client_Media_DB_v2.transaction():1325-1357` is nesting-aware, so a nested call would have its **outer**
  transaction committed by `executescript` with no error, defeating the rollback the outer block relies on. Latent,
  not live — recorded because the module has 253 methods and 32 call sites to this function. · Size: S

### P3 [D1] — `save_reading_item` commits the media row and the read-it-later flag in two separate transactions
- Where: `Media/local_media_reading_service.py:2157-2175`. A failure between the two leaves the article saved but
  absent from the reading list — the user's action half-happened with no signal. · Size: S
- Already covered: TASK-32801 is marked all-Done; this site was not swept.

### P3 [D3] — `Media/local_media_reading_service.py` is a 6,752-line, 253-method single class with no size budget
- Evidence: AST counts; `grep -n "Media|Image_Gen|Video_Gen" Tests/Architecture/test_module_size_ratchet.py` → no
  output. Siblings `media_reading_scope_service.py` (3,773 / 205 methods) and `server_media_reading_service.py`
  (2,030 / 155) are likewise unbudgeted. **~16 responsibilities** countable from the method prefixes (media CRUD +
  trash, chunk navigation, highlights, annotations, reading progress, saved searches, note links, archives, digest
  schedules/outputs/cron, ingestion jobs + streaming, ingestion sources, file artifacts, URL download + scrape,
  PDF/EPUB/HTML extraction, TTS, export/import).
- Recommended correction: record the 16 against **TASK-32809.3**; digest scheduling, ingestion sources and text
  extraction are the three cleanest seams. · Size: L · Confidence: verified

## Candidate triage
**CONFIRMED:** `os_replace_no_atomic` `video_store.py:682` (P2). `except_exception_return`
`local_media_reading_service.py:5717` (P1 export). `fetchall_no_limit` 3 of 21 (`empty_media_trash:647`,
`:6297`, `:6324`). `_maybe_await` cluster → the D1 instance (P1 #2).
**RETIRED:** `os_replace_no_atomic` `local_media_reading_service.py:4334` — renames one tempfile to another tempfile
inside `mkstemp`'s dir purely to attach a suffix; the record carries `cleanup: True`. No durability owed.
`except_exception_pass` comfyui ×3 and `player_pipeline.py` ×4 — cleanup paths that re-raise or return the real
error. `except_exception_return` `Image_Generation/config.py:507` (deliberate lenient config parsing),
`comfyui_*_adapter` ×2 (best-effort remote-queue cleanup). The other 18 `fetchall_no_limit` rows — per-item-scoped,
already `LIMIT`-ed one line below the matched `COUNT(*)`, or user-facing lists the Library paginates.
`mutable_class_attr` ×11 (`supported_formats`) — class-level frozen-in-practice set literals read via `in`; not the
widget-reactive class. `raw_1024x1024` ×12 — default-dimension constants or byte-size arithmetic, no pixel-guard
meaning. `raw_mkdir` ×4 — `video_store.py` ×3 are preceded by `_ensure_safe_root`/`_is_safe_regular_file` lstat +
reparse + resolve checks, **stricter than `path_validation`**; the fourth is in the dead `Media_Creation` module.
`tempfile_no_secure` ×4 — all the secure APIs. `try_import_guard` ×6 — optional-dep guards that degrade correctly.
`function_body_import` ×89 — the ~50 `..tldw_api` imports in `server_media_reading_service.py` are the documented
server-client deferral; `Image_Generation/__init__.py:17,23` is the PEP-562 lazy re-export keeping Pillow off import
time. `legacy_markers` ×15 — prose. `strftime` ×3 — one is a human-readable digest title with `.astimezone(utc)`
applied first; two are in the dead module. The scope-scaffold clusters — owned by TASK-32808.6; Media's copies show
no drift.
**CONFIRMED, LOW:** `except_exception_pass` `local_media_reading_service.py:1590` — `_extract_pdf_text` swallows
every PyMuPDF failure and falls back to a raw byte-decode of the PDF, producing **binary garbage stored as media
content with no signal**. Almost certainly inside TASK-32811.5; not re-filed.

## D4 observations for repo-wide Phase 3
1. **`Image_Generation/config.py` ↔ `Video_Generation/config.py` are a forked pair, not two modules.**
   Verbatim/near-verbatim: `_parse_list` (also in both `adapter_registry.py`), `_coerce_choice`,
   `_get_config_value`, `_read_*_generation_toml`, and the `_coerce_bool`-vs-adopted split. **The drift reaches
   storage/wire** via `minimax_video_allow_uploads` (P2).
2. **The two `adapter_registry.py` are byte-identical in 5 of 6 functions.** No drift. One generic registry
   parameterised by the backend table deletes ~120 lines.
3. **The two `request_validation.py`** share `_positive_int_attr`, `_validate_int_bound`,
   `_validate_positive_finite_float` verbatim. No drift. Home: a shared `_bounds.py`.
4. **`_resolve_api_key` × 5** and **`_resolve_base_url` × 6** across the two adapter packages — the per-provider
   credential-precedence copies. Worth checking against CLAUDE.md's rule (`api_settings.<provider>.api_key` outranks
   env); I did not verify whether all five agree.
5. **`_extract_image_content` ×4 / `_extract_from_node` ×2 / `_extract_from_link_value` ×3 / `_extract_task_id` ×2 /
   `_extract_error_message` ×2** — the OpenAI-compatible response-shape walkers. **These feed `fetch_image_bytes`,
   so drift here is a security-relevant D1, not cosmetic.** Home: `openai_compatible_image_response.py`.
6. **`_tts_provider_for_model` / `_tts_internal_model_id` / `_default_tts_audio_generator`** are byte-identical
   between `Media/local_media_reading_service.py:5210,5219,5172` and
   `Audio_Services_Interop/local_audio_services_service.py:184,193,254`. Home: `TTS/`. Relevant to **task-32863**.
7. **`content_type_for_format`** is defined in `image_format_utils.py:130` **and** re-rolled privately as
   `_content_type_for_format` in `stable_diffusion_cpp_adapter.py:221` — **the same file that already imports
   `validate_and_convert_image_output` from `image_format_utils`.** One-line fix.
8. **Filename sanitizer drift — TASK-32808.2 input, and the direction matters.** Measured, all three against the
   same inputs:

   | input | `comfyui_image_adapter._safe_filename` | `Utils/text.py:17 sanitize_filename` | `Utils/file_extraction.py:614` |
   |---|---|---|---|
   | `..` | **reject** | `'..'` | `'..'` |
   | `../../etc/passwd` | **reject** | `'....etcpasswd'` | `'....etcpasswd'` |
   | `a\x00b.png` | **reject** | `'a\x00b.png'` | `'a\x00b.png'` |
   | `-rf.png` | **reject** | `'-rf.png'` | `'-rf.png'` |
   | `CON` / `NUL.png` | accept | `'CON'` | `'CON'` |
   | 300-char name | **reject** | unchanged | truncated to 100 |

   **The ComfyUI copy is the strictest of the three** (leading-alnum allowlist + explicit `.`/`..`/separator
   rejection + 255 cap); `Utils/text.py` is the weakest and passes `..`, NUL bytes and leading dashes through.
   **Consolidating onto `Utils/text.py` would be a regression on four separate classes.** No copy rejects reserved
   Windows device names. Recommended home for TASK-32808.2: a new `Utils/filename_safety.py` with **two** entry
   points — a *validator* (raise; ComfyUI's semantics; for names that must round-trip verbatim) and a *sanitizer*
   (rewrite; for names becoming local paths) — because collapsing those two contracts into one function is what
   produced the drift.

## Left UNVERIFIED
| Claim | Why | Command |
|---|---|---|
| ffmpeg follows the redirect from the yt-dlp-resolved URL, so the single-hop check is bypassable (P1 #4) | DO-NOT-RUN-THE-APP; needs a live redirecting host | `python -m http.server`-backed 302 to `169.254.169.254`, then `ffmpeg -hide_banner -loglevel debug -i http://127.0.0.1:PORT/redirect -f null -` and read the debug log for the second request |
| The 117 direct `_maybe_await` sites execute on the loop in a shipped flow (P1 #2) | requires driving the running TUI | `pytest Tests/Media/test_media_reading_scope_service.py -q` with an assertion that `asyncio.get_running_loop()` is not blocked, or instrument `save_reading_item` with `loop.call_soon`-latency timing under `app.run_test()` |
| Whether any real backend returns a URL whose bytes decode past 89 M pixels (P1 #1 *exploitability*; the guard gap is verified) | needs a hostile backend | `pytest Tests/Image_Generation/ -q` with a test stubbing `fetch_image_bytes` to return the 87 KB bomb and asserting `ImageGenerationError` |
| The `IN (…)` purge wedge past `SQLITE_MAX_VARIABLE_NUMBER` | needs 32,767 accumulated rows | `python -c "import sqlite3; print(sqlite3.connect(':memory:').getlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER))"`, then insert `limit+1` rows and call the purge |
| Whether `scrape_article_sync` has its own internal timeout (bounds the P1 #2 freeze) | out of slice | `grep -n "timeout" tldw_chatbook/Web_Scraping/Article_Extractor_Lib.py` |
| That deleting `Media_Creation/swarmui_client.py` + `image_generation_service.py` breaks nothing | grep cannot see through re-exports | `pytest --collect-only -q 2>&1 \| grep -c error` before/after, plus `rg -n "Media_Creation.swarmui_client\|Media_Creation.image_generation_service" --type py` |
