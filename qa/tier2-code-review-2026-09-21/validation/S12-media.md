# S12 validation — Media / Media_Creation / Media_Playback / Image_Generation / Video_Generation

Validated at worktree HEAD `d0face3ebe` against `qa/tier2-code-review-2026-09-21/slices/S12-media.md`.

## 1. P1 — Every image adapter's format-conversion path decodes backend bytes with no pixel cap
- Verdict: CONFIRMED
- Site now: `Image_Generation/adapters/image_format_utils.py:140-167 maybe_convert_format` (`Image.open(io.BytesIO(content))` → `.convert`/`.save`, no `.size` check anywhere in the function), reached from `validate_and_convert_image_output:173-194` which only calls `_enforce_max_bytes` (compressed-byte cap) before and after conversion — no pixel-count check at all.
- Proof: `grep -n "PILLOW_DECOMPRESSION" image_format_utils.py` → no hits; the guard (`PILLOW_DECOMPRESSION_WARNING_MAX_PIXELS`, `request_validation.py:37,320`) lives in a sibling module never imported by `image_format_utils.py`. Confirmed all 8 callers unchanged: `swarmui_adapter.py:62,77`, `fal_image_adapter.py:248`, `gemini_image_adapter.py:165`, `novita_image_adapter.py:52`, `modelstudio_image_adapter.py:83`, `openrouter_image_adapter.py:78`, `together_image_adapter.py:60`, `stable_diffusion_cpp_adapter.py:117` — all call `validate_and_convert_image_output`, none of which reaches the pixel guard.
- Note: none. (Did not re-run the 90M-pixel PNG reproduction; the code trace is decisive and unchanged since the review.)

## 2. P1 — 117 of 138 local-mode scope-service call sites run synchronous sqlite (one case: blocking HTTP) inline on the event loop
- Verdict: CONFIRMED
- Site now: `Media/media_reading_scope_service.py` — `_call_local_leaf(` count = 22 (21 uses + def, matches); `_maybe_await(` count = 119 (117 uses + def + 1 internal use inside `_call_local_leaf`, matches exactly). `_call_local_leaf` (`:190-197`) is `asyncio.to_thread`-based; the plain `_maybe_await` branch (`:125-...`) calls `fn(*args, **kwargs)` directly on the calling coroutine's loop.
- Proof: traced `save_reading_item` (`media_reading_scope_service.py:1589 await self._maybe_await(service.save_reading_item(...))`) → `local_media_reading_service.py:2091 save_reading_item` → `:4273 _default_url_article_scraper`, which (when a running loop is detected) submits to `ThreadPoolExecutor(max_workers=1)` then calls `future.result()` at `:4290` with **no timeout** — this blocks whatever loop is awaiting `_maybe_await`. Also traced `empty_media_trash` (`local_media_reading_service.py:642-649`): unbounded `SELECT id FROM Media WHERE deleted=0 AND is_trash=1` + `.fetchall()`, matching the review's line (`local_media_reading_service.py:642`, cited as `:1065` for the scope-service call site — both check out).
- Note: none.

## 3. P1 — `_clone_git_repository` is the only `subprocess.run` in the slice with no `timeout=`
- Verdict: CONFIRMED
- Site now: `Media/local_media_reading_service.py:4680-4682` — `subprocess.run(command, capture_output=True, text=True, check=False, env=clone_env)`, no `timeout=`.
- Proof: the other 3 `subprocess.run` sites in the slice all carry a timeout: `stream_resolve.py:186-188` (`timeout=YTDLP_TIMEOUT_SECONDS`), `player_pipeline.py:90-92` (`timeout=30`), `stable_diffusion_cpp_adapter.py:93-97` (config-driven timeout, confirmed further down in the function).
- Note: none.

## 4. P1 — yt-dlp stream branch egress-validates one hop, then hands ffmpeg a URL it can redirect through unchecked
- Verdict: CONFIRMED
- Site now: `Media_Playback/stream_resolve.py:164-207 _resolve_with_ytdlp` calls `_validate_later_hop(final)` exactly once (`:204`) and returns; `resolve_stream_url:239-240` then calls `_probe_ranges(final_url)` (`:146-161`), which itself uses `httpx.Client(follow_redirects=False, ...)` and only HEADs the URL once — no further redirect walk. `player_pipeline.py:249-266` builds the ffmpeg argv with `-i self._source` and no `-protocol_whitelist`/`-follow_redirects` flag anywhere in the file (`grep` → zero hits).
- Proof: read of `_walk_redirects` (`:100-141`, validates every hop) vs `_resolve_with_ytdlp` (validates only the terminal URL once) confirms the asymmetry the module's own docstring (`:8-19`) describes as the reason the walk exists.
- Note: none — confidence remains "inferred" per the review (ffmpeg's default-follows-redirects behavior is well-documented libavformat behavior, not executed live here).

## 5. P1 — Reading-list export silently loses body content on per-item DB errors, and silently truncates/empties on unhonoured filters
- Verdict: CONFIRMED
- Site now: `Media/local_media_reading_service.py:5714-5720 _local_export_detail_row` (`try: ... except Exception: return dict(row)`); `export_reading_items:2792-2842` — `:2810-2811` (`if normalized_statuses and "saved" not in normalized_statuses: rows = []`), `:2812-2813` (`elif favorite is True: rows = []`), `:2822-2828` (`domain` filter applied in Python **after** `search_media(limit=size, ...)`).
- Proof: read confirms all three mechanisms exactly as described; `search_media` is called with `limit=max(int(size), 1)` (default `size=1000`) before the domain filter runs, so a domain-filtered export only ever sees matches within the first page.
- Note: none.

## 6. P2 — A 9-table, 4-index schema created ad hoc outside `DB/migrations/`, no schema-version bump, no `VALID_TABLES` entry
- Verdict: CONFIRMED
- Site now: `Media/local_media_reading_service.py:5817-5921 _ensure_local_reading_aux_schema`, `grep -c "_ensure_local_reading_aux_schema("` → 32 call sites (matches). `scripts/check_schema_table_allowlist.py`'s own docstring (`:90`) states: *"Scope: chachanotes only, deliberately. The media and prompts ..."* — confirming the media DB is outside its reach.
- Proof: counted the actual `CREATE TABLE` statements in the function body: `local_reading_saved_searches`, `local_reading_note_links`, `local_reading_archives`, `local_reading_highlights`, `local_document_annotations`, `local_reading_digest_schedules`, `local_reading_digest_outputs`, `local_file_artifacts` — **8 tables**, plus 4 `CREATE INDEX IF NOT EXISTS` statements. None of the 8 names appear in `DB/sql_validation.py`.
- Note: the finding's title says "9-table" but its own Evidence section names only 8 tables, and I count exactly 8 `CREATE TABLE` statements in the function — the "9" in the title appears to be an off-by-one miscount (possibly conflating with the sibling `_ensure_local_ingestion_schema`, which is a separate function for a separate table set). Governance substance (ad hoc DDL outside migrations, invisible to the allowlist checker, no index-plan pin) is unaffected.

## 7. P2 — `video_store._atomic_publish` renames without `fsync`; the shared helper it declined does fsync
- Verdict: CONFIRMED
- Site now: `Video_Generation/video_store.py:683-730 _atomic_publish` — `staged.flush()` then size check then `_commit_sibling`→`os.replace` (`:681-682`); `grep -n "fsync" video_store.py` → zero hits anywhere in the file.
- Proof: `Utils/atomic_file_ops.py` exists and does call `os.fsync(f.fileno())` at two sites (`:103`, `:178`); `video_store.py` does not import or reference `atomic_file_ops` at all, confirming both "declined the helper" and "missed the fsync."
- Note: none.

## 8. P2 — `Video_Generation/config.py` re-rolled boolean coercion, drifted: doesn't accept integers
- Verdict: CONFIRMED
- Site now: `Video_Generation/config.py:333-342 _coerce_bool` (`isinstance(value, str)` only), used at `:407 confirm_cost_estimate` and `:413 minimax_video_allow_uploads`. Sibling `Image_Generation/config.py:554` imports and uses `Utils.Utils.coerce_bool_flag`.
- Proof: shared helper (`Utils/Utils.py:618-648 coerce_bool_flag`) matches `isinstance(value, (str, int))`; the video copy matches `isinstance(value, str)` only — confirmed by reading both bodies side by side. A TOML `minimax_video_allow_uploads = 1` (int) coerces to `True` via the shared helper's path but falls through to `default` via the video copy's path.
- Note: none.

## 9. P2 — Digest-schedule dedupe/purge both `fetchall()` every output row ever written; purge's `IN (…)` wedges past the SQLite variable limit
- Verdict: CONFIRMED
- Site now: `local_media_reading_service.py:6292-6309 _reading_digest_already_executed_for_minute` and `:6311-6341 _purge_expired_reading_digest_outputs` — both `SELECT ... WHERE schedule_id = ?` with no time predicate + `.fetchall()`, Python-side filtering; purge builds `f"DELETE FROM local_reading_digest_outputs WHERE id IN ({placeholders})"` with `placeholders = ", ".join("?" for _ in expired_ids)` — no chunking whatsoever.
- Proof: read confirms the unbounded `IN` list is built directly from `expired_ids` with no batching, so once `len(expired_ids)` exceeds `SQLITE_MAX_VARIABLE_NUMBER` the `DELETE` raises `OperationalError`.
- Note: none.

## 10. P2 — `Media_Creation/` ships a second SwarmUI client bypassing egress/format/pixel guards; only one pure function reachable
- Verdict: CONFIRMED
- Site now: `Media_Creation/swarmui_client.py` uses raw `aiohttp.ClientSession()` (`:119,142`) with no `egress`/`validate_and_convert_image_output` reference anywhere in the file.
- Proof: `grep -rn "Media_Creation_Events|swarmui_events" tldw_chatbook/ --include=*.py | grep -v /Tests/` → only the package's own `__init__.py` re-export; nothing else imports it. Live entry point confirmed: `Chat/console_generate_image.py:437,447` imports `ImageGenerationService` solely to call the static `extract_context_from_messages(None, shaped)`.
- Note: none.

## 11. P2 — ComfyUI video adapter accepts server-supplied `filename`/`subfolder` with none of the image twin's validation
- Verdict: CONFIRMED
- Site now: `Video_Generation/adapters/comfyui_video_adapter.py:966-984` — checks only `isinstance(filename, str)` + non-blank, `isinstance(subfolder, str)`, `output_type == "output"`, `Path(filename).suffix == expected_suffix`. `grep -n "_safe_filename|_safe_subfolder|_SAFE_NAME" Video_Generation/adapters/comfyui_video_adapter.py` → zero hits.
- Proof: `Image_Generation/adapters/comfyui_image_adapter.py:947-971` defines and uses `_safe_filename`/`_safe_subfolder` — confirmed present in the image twin, absent in the video copy.
- Note: none.

## 12. P3 — `_ensure_local_reading_aux_schema` runs `executescript` inside `db.transaction()`; `executescript` implicitly COMMITs
- Verdict: CONFIRMED
- Site now: `local_media_reading_service.py:5818-5819` — `with db.transaction() as conn: conn.executescript(...)`.
- Proof: standard sqlite3 semantics — `Cursor.executescript()` implicitly commits any open transaction before running. Review's own characterization (latent, zero live nested calls found) is unchanged; not independently re-run as an AST sweep, taken on the review's evidence plus this file's own read.
- Note: none.

## 13. P3 — `save_reading_item` commits the media row and read-it-later flag in two separate transactions
- Verdict: CONFIRMED
- Site now: `local_media_reading_service.py:2160-2176` — `db.add_media_with_keywords(...)` (its own commit) followed by a separate `db.save_media_to_read_it_later(...)` / `self.remove_from_read_it_later(...)` call, not wrapped in the same transaction.
- Note: none.

## 14. P3 — `local_media_reading_service.py` is a 6,752-line, 253-method single class with no size budget
- Verdict: CONFIRMED
- Site now: `wc -l` → `local_media_reading_service.py` 6752, `media_reading_scope_service.py` 3773, `server_media_reading_service.py` 2030 — all three line counts match exactly.
- Proof: `grep -in "media" Tests/Architecture/test_module_size_ratchet.py` → zero hits, confirming none of the three modules is in the size ratchet.
- Note: none (method count of 253 not independently re-counted via AST, taken from the review; the line-count match is exact enough to trust the rest).

TOTALS: confirmed=14 fixed=0 wrong=0 demoted=0 promoted=0
