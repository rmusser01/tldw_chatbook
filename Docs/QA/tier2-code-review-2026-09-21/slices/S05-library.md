# S05 — Library (business logic package)

**Coverage:** files read in full: 16 | sampled: 16 | mechanical only: 25 (of 57).
Full: `library_ingest_state.py`, `library_ingest_jobs.py`, `collections_capture_repository.py`,
`collections_offline_store.py`, `collections_legacy_recovery.py`, `library_collections_service.py`,
`library_rechunk_service.py`, `review_set_service.py`, `library_export_scope.py`, `library_rag_answer_service.py`,
`meeting_speaker_rename.py`, `ingest_preflight.py`, `library_browse_location.py`, `library_fts_query.py`,
`server_ingest_reconcile.py`, `__init__.py`; plus `DB/Library_Ingest_Jobs_DB.py` as out-of-slice evidence.
Sampled: 16 more (largest: `library_local_rag_search_service.py` ~1000/1837, `local_library_tool_service.py`
~700/1424, `local_media_chunk_tool_service.py` ~500/1218).
Mechanical only: 25, AST/grep-swept for the D1–D4 pattern set (internal-import resolution 0 unresolved;
keyword-arg-vs-signature check on every call into and out of `Library/` — 1 hit, the P0; `re.compile`-in-body 0).

## Findings

### P0 [D1] — Console builds **no** Library tool provider at all on the default configuration: the factory passes a constructor kwarg deleted on 2026-09-01, and the `TypeError` is swallowed into a warning
- Where: `Chat/console_runtime.py:675` (`collections_service=…`) vs
  `Library/local_library_tool_service.py:448-460`. Swallow sites:
  `Chat/console_chat_controller.py:27191-27198` and `:20767-20785`.
- Evidence: **independently re-verified by the lead — see `phase4-verification.md` "S05-P0" for the full chain.**
  `inspect.signature(LocalLibraryToolService.__init__)` has no `collections_service` and no `**kwargs`;
  constructing it as the factory does raises `TypeError`. The parameter was removed in `5dd1077df6` (2026-09-01,
  "retire generic containers from current surfaces"); the sibling factory
  `UI/Console_Modules/library_activity.py` **was** updated and no longer passes it. The broken factory is the one
  that runs: `console_runtime.ensure_chat_controller` (`:3647`) uses `kwargs.update(...)`, not `setdefault`, so it
  **overwrites** `chat_screen.py:10161`'s `library_provider_factory=self._library_activity.build_provider`. The
  failing branch is the default: `direct_library_tools` defaults to `True`
  (`settings_library_rag_defaults.py:94,126`), and `_library_provider_for_app` returns the `LibraryRagToolProvider`
  fallback only in the `if not …direct_library_tools:` branch — so nothing is returned at all.
  A repo-wide AST keyword-vs-signature sweep over every call into `Library/` found **exactly one** mismatch: this one.
- Why it matters: on stock config every Console agent run silently loses all 18 `library_*` tools, with no fallback.
  The only trace is one WARNING line.
- Recommended correction: delete the `collections_service=` argument at `console_runtime.py:675`. Better: delete
  `_library_provider_for_app` outright and let `ensure_chat_controller` keep the caller's factory —
  `library_activity.build_provider` is the maintained copy and additionally threads `capture_kwargs(turn_context)`
  into both providers, which the runtime copy drops. Canonical home:
  `UI/Console_Modules/library_activity.build_provider`.
- Size: S · ADR: no · Confidence: **verified**
- Pinning test: **none, and that is the cause.** `Tests/integration/test_console_library_control_integration.py:177`
  and `Tests/Chat/test_console_library_runtime_policy.py:101` both inject a **fake** `library_provider_factory`, so
  no test ever constructs the real one — the "mocked tests never catch these" shape exactly.
- Already covered: none.

### P1 [D1] — A re-chunk that yields zero chunks hard-DELETEs every stored chunk row for the item, inserts nothing, and reports `status: "rechunked"`
- Where: `Library/library_rechunk_service.py:301-314` (`_replace_chunk_rows`: unconditional `DELETE`, `if rows:` guard
  on the `INSERT`); receipt built at `:685-701`.
- Evidence: `improved_chunking_process("...", {"method":"sentences","max_size":3,"overlap":0})` → **0 chunks** on
  non-blank text, so `rechunk_one_item`'s `if not content.strip()` guard (`:535-538`) does not fire. Repro with a
  recording fake DB: `outcome: rechunked {'chunk_count': 0, …, 'spans_present': True}`, and the **only** SQL issued is
  `DELETE FROM UnvectorizedMediaChunks WHERE media_id = ? (7,)`. The delete is deliberately hard and deliberately
  writes no sync-log event (`:246-257`), so no other client can recover the rows.
- Why it matters: the item's derived chunk index is destroyed with a success receipt, and it is not self-healing —
  re-running produces 0 again, and `library_get_media_chunk` then degrades with `_RECHUNK_HINT` = *"no stored chunks —
  use library_rechunk_media to enable unit fetches"*, pointing the user at the action that emptied it. Reachable from
  the agent tool `library_rechunk_media` (flat `spec` override, `local_media_chunk_tool_service.py:1046-1155`) and
  from the batch via a stored/config `sentences` template. Secondary: `spans_present` is `all([])` → `True`, a
  vacuous claim.
- Recommended correction: in `_replace_chunk_rows`, refuse the replacement when `rows` is empty and have
  `rechunk_one_item` map it to `status: "skipped"` with `"chunking produced no chunks"` — the posture `:537` already
  takes for empty source content. Guard `spans_present` on a non-empty `written`.
- Size: S · ADR: no · Confidence: **verified**
- Pinning test: none in `Tests/Library/test_library_rechunk_service.py`.
- Already covered: none. Not P0 only because the trigger content is narrow and the rows are nominally derived.

### P1 [D1] — A cancelled capture extraction never settles: the service's cleanup reason is not in the repository's allowlist, the refusal is swallowed, and the row stays `processing` with a live 300 s lease and no Retry
- Where: `Library/collections_capture_service.py:322-335` (`reason="interrupted"`,
  `except CollectionsCaptureError: pass`) vs `collections_capture_repository.py:76-89`
  `_EXTRACTION_FAILURE_REASONS` and `:645-656` `fail_extraction`.
- Evidence: `'interrupted' in _EXTRACTION_FAILURE_REASONS` → `False` (the set is `{dependency_missing,
  empty_extraction, fetch_failed, invalid_url, network_error, redirect_limit, response_too_large, unknown,
  unsafe_url, unsupported_content}`). End-to-end repro on a real `LibraryCollectionsDB`:
  `fail_extraction('interrupted') REFUSED -> invalid_extraction_failure_reason`; `after cancel_extractions -> state:
  processing | last_fetch_error: None | owner token still held: True`. The value is legitimate *as stored data* —
  `interrupt_stale_extractions` (`repository.py:738-751`) writes `last_fetch_error = 'interrupted'` via raw SQL,
  bypassing the same allowlist. So this is an omission, not a policy.
- Why it matters: `app.py:10384` `_shutdown_collections_capture_runtime` → `cancel_extractions()`; the only repair is
  `app.py:10361` → `interrupt_stale_extractions()`, whose `WHERE … extraction_lease_expires_at IS NULL OR <= ?` skips
  a lease renewed within the last 300 s. A restart inside 5 minutes does not repair it. And `retry_extraction`
  (`repository.py:683-709`) requires `processing_state IN ('failed','interrupted')`, so a wedged row offers the user
  no action at all.
- Recommended correction: add `"interrupted"` to `_EXTRACTION_FAILURE_REASONS` (already a legal `last_fetch_error`
  and a legal `processing_state`); the swallow at `:333` then only covers a genuinely lost claim. Keep the `except`
  but log it.
- Size: S · ADR: no · Confidence: **verified**
- Pinning test: none. `Tests/Library/test_collections_capture_extraction.py` covers `interrupt_stale_extractions`
  (5 passed) but nothing exercises `cancel_extractions` / the `CancelledError` branch.
- Already covered: none.

### P2 [D1/D3] — `_run` uses `asyncio.run` inside a synchronous tool dispatcher; 4 of the 18 Library tools depend on it, and the failure mode is a scrubbed generic `storage_error`
- Where: `Library/local_library_tool_service.py:133-137`, used at `:626`, `:690`, `:692`; swallow at `:490-492`.
- Evidence: the async backends are `LocalPromptService.list_library_prompts`/`.search_library_prompts` and
  `LocalSkillsService.list_library_skills`/`.search_library_skills`. Repro:
  `off-loop: {'items': [], 'total': 0, …}` vs `on-loop: {'error': {'code': 'storage_error', 'message': 'The local
  Library store could not complete the operation.', 'retryable': True}}` + `RuntimeWarning: coroutine
  'list_library_skills' was never awaited`. **Not currently live**: tool calls run on a dedicated thread
  (`Agents/agent_service.py:1995 threading.Thread(target=_runner, name=f"tool-{tool_name}")`) with no running loop,
  so `asyncio.run` succeeds today.
- Why it matters: four agent tools are one dispatch-thread change away from silently reporting a retryable storage
  failure, and the cost today is a full event loop built and torn down per call. The
  `except Exception: return _storage_error_payload()` is what makes the failure indistinguishable from a real DB error.
- Recommended correction: a loop-aware bridge (`asyncio.run` only when `get_running_loop()` raises; otherwise
  `run_coroutine_threadsafe` or refuse explicitly). `local_media_chunk_tool_service.py:1153` documents the same
  "transient loop" bridge — one shared helper, not two. At minimum re-raise `RuntimeError` from `_run`.
- Size: S · ADR: no · Confidence: verified (behaviour and non-reachability both)

### P2 [D3] — `LibraryNoteSession._pending_save_requested` is write-only: 12 stores, 0 loads, repo-wide
- Where: `Library/library_notes_session.py:299, 428, 505, 550, 567, 576, 590, 638, 655, 703, 706, 1142`.
- Evidence: AST → `stores: 12 loads: 0`; `rg` across `tldw_chatbook/` and `Tests/` returns only the 12 assignments.
- Why it matters: the save-coalescing loop is actually driven by `has_newer_draft = current.draft_revision >
  draft_revision` (`:688`). `mutate`'s `if snapshot.saving: self._pending_save_requested = True` (`:504-505`) *looks*
  load-bearing and is not, so a future edit to the save chain will reason from a variable nothing consults.
- Size: S · Confidence: verified · Already covered: none (TASK-32807.6 is `Utils`/`Widgets`, not `Library` state)

### P2 [D2] — The review-set picker loads every set's full item list (N+1, unbounded per set) to render counts
- Where: `Library/review_set_service.py:452-455` (`SELECT * FROM review_set_items WHERE set_id = ? ORDER BY
  position`, no LIMIT), amplified by `:231-233` (`list_review_sets` calls `_read_review_set` once per set).
- Evidence: `list_review_sets(limit=200 default, clamped to 1000)` × `REVIEW_SET_CAP = 500` = up to 100,000 rows
  materialized into `ReviewSetItem` dataclasses. The one consumer,
  `UI/Screens/library_screen.py:31946-31965 _collect_review_set_picker_rows`, needs only per-set counts.
- Recommended correction: a `list_review_set_summaries()` returning header rows plus `COUNT(*)`/`SUM(done)` in one
  grouped query; keep `_read_review_set` for single-set navigation.
- Size: M · Confidence: inferred (read only, no measurement)

### P2 [D2] — `library_get_media_structure` materializes every chunk row of an item to compute three aggregates
- Where: `Library/local_media_chunk_tool_service.py:325-334` (`_chunk_rows`, `fetchall()`, no LIMIT), consumed only at
  `:401-417` as `len(chunk_rows)`, `{chunk_type}` and `{chunk_engine_version}` — all expressible as
  `SELECT COUNT(*), … GROUP BY chunk_type, chunk_engine_version`.
- Why it matters: per-call cost on an agent-reachable read tool, proportional to the largest item in the library
  (a 5,000-chunk document — the size `library_rechunk_service.py:261` calls out — builds 5,000 dicts per call).
- Size: S · Confidence: verified (single call site; sizes inferred)

### P3 [D4] — `_error_text` has drifted across its three copies: the Library copy honours `ERROR_CHAR_CAP`, the two Subscriptions copies overshoot it by 6 characters
- Where: `Library/library_rag_answer_service.py:354-364` (canonical) vs `Subscriptions/briefing_service.py:560-566`
  and `Subscriptions/briefing_cast.py:470-475`.
- Evidence: Library — `cut = ERROR_CHAR_CAP - len(_TRUNCATION_SUFFIX); message[:cut] + _TRUNCATION_SUFFIX` → exactly
  500 chars. Both Subscriptions copies — `message[:ERROR_CHAR_CAP] + " [...]"` → **506** chars. The sibling
  `_effective_max_tokens` / `_invoke_chat` copies in the same trio are byte-identical modulo constants and are
  *documented* duplication ("Copies … not imported: that function is private to its own module").
- Recommended correction: lift the trio into `Chat/one_shot_call.py` parameterised by the three constants.
  Library's `_error_text` is the correct body.
- Size: M · Confidence: verified · Already covered: partially by TASK-32808.9 (To Do); this trio is not named there

### P3 [D4] — `library_rechunk_service` writes canonical-shaped UTC timestamps by hand at two sites instead of `Utils/timestamps.utc_now_iso()`
- Where: `Library/library_rechunk_service.py:229` and `:259` —
  `datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"`. Both feed `Media.last_modified` and
  `UnvectorizedMediaChunks.last_modified`.
- Evidence: the shape matches `Utils/timestamps._CANONICAL_UTC_RE` exactly, so **no value drift** — but these are two
  storage writers TASK-32803.5 (**Done**) did not reach.
- Size: S · ADR: no (ADR-127 settles the shape) · Confidence: verified
- Already covered: TASK-32803.5 is Done — **this shows it insufficient.** *(Third independent sighting; see also
  S01-P3 and S11-P3.)*

### P3 [D3] — `meeting_speaker_rename` hand-writes the sync-log/FTS contract against four private `MediaDatabase` members because no public seam exists
- Where: `Library/meeting_speaker_rename.py:177` (`db._generate_uuid`), `:183`/`:201`/`:386` (`db._log_sync_event`),
  `:356` (`db._get_current_utc_timestamp_str`), `:390` (`db._update_fts_media`); same at
  `library_rechunk_service.py:292` and `:174` (function-body import of a private `Local_Ingestion` name).
- Evidence: all six names currently resolve. The module's own docstring (`:152-159`) states the cause: *"No public
  'add/update a Transcripts row' method exists anywhere on `MediaDatabase` … so this mirrors the sync contract … by
  hand."*
- Recommended correction: add `MediaDatabase.upsert_transcript(...)` (versioned + sync-logged) and call it.
  Canonical home: `DB/Client_Media_DB_v2.py`.
- Size: M · Confidence: verified

### P3 [D3] — `LocalLibraryCollectionsService` carries a constructor seam and four read methods with no production consumer
- Where: `Library/library_collections_service.py:181-186` (`id_factory`/`now_factory`/`_utc_now`), `:312-598`
  (`list_library_collections`, `locate_library_collection_page`, `search_library_collections`,
  `get_library_collection`).
- Evidence: `rg` for each method name across `tldw_chatbook/` returns **only** this module. `_id_factory`/
  `_now_factory` are referenced only by `Tests/Library/test_library_collections_service.py:62,63,91,111,121`. Every
  *write* method on the class raises `LegacyCollectionsReadOnlyError`, so nothing in production can reach them.
- Why it matters: ~290 lines kept alive by its own test — and two of the dead methods
  (`search_library_collections`, `get_library_collection`) **skip** the `_validate_collection_page_limit`/
  `_validate_collection_page_offset` guards their sibling applies, a boundary gap that becomes real the moment they
  are wired up.
- Size: M · Confidence: verified

### P3 [D2] — `count_export_scope` materializes every id in four sources to produce four integers
- Where: `Library/library_export_scope.py:150-164` — four `len(get_all_*_ids(...))` calls. For `kind="everything"`
  that is four full id scans per counts refresh; the count path never uses the ids.
- Recommended correction: `count_active_*` methods returning `SELECT COUNT(*)`; keep `get_all_*_ids` for
  `resolve_export_selections`.
- Size: M · Confidence: inferred

### P3 [D1] — `LibraryIngestJobRegistry.attach_remote` mutates the stored job in place, breaking the class's own replace-on-transition contract
- Where: `Library/library_ingest_jobs.py:958-968`. Every other mutator uses `dataclasses.replace` and reassigns
  `self._jobs[index]`; the module docstring (`:46-50`) states the contract. `iter_jobs_for_listeners` (`:1601-1613`)
  hands out the internal objects by reference.
- Why it matters: latent aliasing only — `jobs()`/`get_job()` copy, and the one consumer (`app.py:2758`) reads
  immediately. Still a contract violation in the module that documents the contract.
- Size: S · Confidence: verified

### P3 [D4] — `_first_present_text` is byte-identical in three Library modules; `_record_title` is the same body with a per-type fallback string
- Where: `library_conversations_state.py:187`/`:198`, `library_media_state.py:1301`/`:1312`,
  `library_media_viewer_state.py:142`. **No behavioural drift** — `_record_title` differs only in
  `"Untitled conversation"` vs `"Untitled media"`, which should be a parameter.
- Recommended correction: one `first_present_text(record, keys)` + `record_title(record, *, fallback)` in a small
  `Library/record_text.py` (all three consumers are in this package; `Utils/` would be over-promotion).
- Size: S · Confidence: verified

### P3 [D2] — `ingest_preflight._collect_files` pays 3–4 `stat` calls per entry
- Where: `Library/ingest_preflight.py:111-149` (`is_symlink()`, `is_dir()`, `is_file()` on bare `Path`s from
  `iterdir()`, each re-statting) plus `analyze_path:450-457` (`_statted_size` again per file). At the default
  `scan_limit=1000` that is ~4,000 syscalls per folder pre-flight.
- Recommended correction: `os.scandir`, taking the size from the cached `entry.stat()`.
- Size: S · Confidence: inferred
- Pinning test: `Tests/Library/…` cover the skip/truncate semantics, which the rewrite must preserve exactly.

### P3 [D3] — `library_artifacts_catalog` calls a private method on a `Chatbooks` service across the package boundary
- Where: `Library/library_artifacts_catalog.py:346` — `self.chatbook_service._is_console_saved_artifact(record)`.
  The only cross-*package* private-method call in the slice.
- Size: S · Confidence: verified

## Candidate triage
**RETIRED with evidence:** `inline_path_check_no_pv` `library_ingest_state.py:2707` — **the 2026-09-17 seed-list
entry.** `os.path.commonpath` there derives a **display label** for a batch header ("`inbox` — 6 files") inside
`build_ingest_queue_groups`; it performs no containment check, guards no filesystem access, and `ValueError` is
caught for the cross-drive case. `Utils/path_validation.py` has no "common ancestor of N paths" helper and would be
the wrong tool. Real path validation in this slice **does** use the helper: `ingest_preflight.py:29,422`,
`collections_legacy_recovery.py:20,528`, `library_browse_location.py:25`, `library_artifacts_catalog.py:293-301`.
`tempfile_no_secure` `collections_legacy_recovery.py:321` — `mkstemp` only on the labelled non-dirfd fallback, and
its descriptor is immediately identity-pinned (`S_ISREG`, `st_nlink == 1`, mode `0o600`); the preferred path uses
`os.open(O_CREAT|O_EXCL|O_NOFOLLOW, 0o600, dir_fd=…)` with a `secrets.token_hex(8)` leaf.
`os_replace_no_atomic` `collections_legacy_recovery.py:509` — bracketed by `_require_path_identity` before and after
plus `_require_parent_identity` on both sides. `fetchall_no_limit` (10 rows across
`collections_capture_repository`, `collections_legacy_recovery`, `library_collections_service`) — every row matched a
`SELECT COUNT(*)` beside a properly paged `LIMIT ? OFFSET ?` query. `fetchall_dynamic_sql` ×2 — the only
interpolation is `", ".join("?" …)` placeholders; all values bound. `lock_and_execute` — the lock guards a `set[str]`
membership test and is never held across a DB call. `strftime` (8 rows in notes state) — all display-local by design;
`library_notes_session.saved_status_message:446-451` actively *raises* on a naive datetime to prevent exactly the
TASK-32803 bug. `try_import_guard` ×4 — none guards an optional dependency. `legacy_markers` (40 rows) — domain
vocabulary (`chunk_engine_version IS NULL` = legacy rows) or retirement notes. `function_body_import` — 20 of 22 are
documented lazy seams, all resolving.
**CONFIRMED:** `fetchall_no_limit` `local_media_chunk_tool_service.py:334` and `review_set_service.py:452` (both
filed above); `except_exception_return` `local_library_tool_service:490` (the masking half of the `asyncio.run`
finding); `strftime` `library_rechunk_service:229,259` (filed); `raw_1024x1024` ×4 — retired as a defect but noted:
the four offline-quota constants are declared **twice** with identical values in
`collections_capture_repository.py:91-92` and `collections_offline_store.py:33-34`.

## D4 observations for repo-wide Phase 3
1. `Utils/timestamps.utc_now_iso()` — two hand-rolled canonical writers at `library_rechunk_service.py:229,259`.
   Re-run TASK-32803.5's sweep as `rg 'strftime\("%Y-%m-%dT%H:%M:%S\.%f"\)\[:-3\]'` repo-wide rather than by writer
   inventory.
2. `Utils/Utils.format_size_bytes` — adopted at `library_ingest_state.py:609-617` but as a **per-call function-body
   import**. TASK-32808.1 should count the un-adopted copy (`Tools/web_tool_impls.py:463`), not the
   adopted-but-lazily-imported one.
3. **One-shot LLM call trio** (`_error_text`/`_effective_max_tokens`/`_invoke_chat`), 3 copies, one with real drift —
   the highest-value D4 in the slice because the copies are already self-declared. Home: `Chat/one_shot_call.py`.
4. `_first_present_text`/`_record_title` — 3 and 2 copies, zero drift. Home: `Library/record_text.py`.
5. Offline-quota constants declared in two modules, the store passing its pair into the repository's defaults.
6. **Argument-key validation** — `_validate_argument_keys` byte-identical in `local_media_chunk_tool_service.py:253-265`
   and `local_library_tool_service.py:583-594`, and the second's docstring says "mirrors the shared dispatcher". Home:
   `library_tool_contract.py`. Same for the `_positive_limit`/`_offline_size`/`_validate_max_nodes`/
   `_validate_message_limit`/`_validate_collection_page_limit` family (5+ copies of "positive int, reject bool, clamp
   to max" across `Library/`, `Agents/session_todo_store.py:65`, `TTS/audio_cpp_artifact_catalog.py:136`,
   `Tools/workspace_tool_protocol.py:475`).
7. `_maybe_await` — the Library member is **named differently** (`library_rag_service.py:165
   _resolve_maybe_awaitable`), which is why a name-based census misses it. Flagged for the census method.
8. **Missing public seam, not a duplicate** — `MediaDatabase` has no public "write a Transcripts row with sync-log +
   FTS" method, so two Library modules reimplement the contract against private members. The
   "no helper exists, copies have not drifted *yet*" sub-case.

## Left UNVERIFIED
| Claim | Why | Command |
|---|---|---|
| The P0 leaves Console with **no** Library tools on a live run | DO-NOT-RUN-THE-APP; traced statically through `ensure_chat_controller` → `_resolve_library_provider_for_context` → both `except Exception` handlers | Launch Console with default config, send a message that should call a Library tool, then `rg 'library_provider_factory failed' ~/.local/share/tldw_cli/logs/*.log` |
| The P1 re-chunk wipe is reachable from a *stored* `sentences` template (the explicit-`spec` path is proven) | needs a seeded `ChunkingTemplates` row + a `Media.chunking_config` naming it | `pytest Tests/Library/test_library_rechunk_service.py` with a case that seeds the template and asserts `SELECT COUNT(*) FROM UnvectorizedMediaChunks WHERE media_id=?` is 0 after `rechunk_legacy_items` |
| The review-set picker cost in wall-clock terms | no measurement; sizes derived from `REVIEW_SET_CAP=500` × `_LIST_LIMIT_DEFAULT=200` | seed 200 sets × 500 items, time `svc.list_review_sets()` |
| Whether `interrupt_stale_extractions` runs anywhere but startup | `rg` returned exactly two sites; a dynamic-dispatch call by string would not appear | `rg -n 'interrupt_stale' tldw_chatbook/ && rg -n 'getattr\(.*repository.*interrupt' tldw_chatbook/` |
| That the 25 mechanical-only files hold no finding | not read; AST+grep swept clean, but a logic defect like the P1 wipe is only visible on a read | read `library_prompts_state.py` (2469), `library_notes_lasting_sync_state.py` (1225), `library_note_import_state.py` (1399) — the three largest with non-trivial state machines |
| Whether `MediaDatabase.get_connection()` (raw at `library_rechunk_service.py:140`, `local_media_chunk_tool_service.py:327,347,354`) is safe from the agent tool thread | `Library_Collections_DB` documents thread-local connections; `Client_Media_DB_v2` not confirmed, and these reads run on the per-tool thread | `sed -n '1182,1215p' tldw_chatbook/DB/Client_Media_DB_v2.py` — check for `threading.local()` / `check_same_thread` |
