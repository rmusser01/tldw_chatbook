# S05 — Library validation

## 1. P0 — Console builds no Library tool provider on default config; TypeError swallowed to a warning
- Verdict: CONFIRMED
- Site now: `tldw_chatbook/Chat/console_runtime.py:681` (`collections_service=getattr(app, "local_library_collections_service", None)`) vs `Library/local_library_tool_service.py:448-460` (`LocalLibraryToolService.__init__` has no `collections_service` param and no `**kwargs`). Swallow sites now at `console_chat_controller.py:20788-20806` and `:27212-27219` (shifted ~20 lines from the review's 20767-20785/27191-27198, consistent with 25 intervening commits — same code).
- Proof: `python -c "from tldw_chatbook.Library.local_library_tool_service import LocalLibraryToolService; LocalLibraryToolService(collections_service=None)"` → `TypeError: LocalLibraryToolService.__init__() got an unexpected keyword argument 'collections_service'`. `console_chat_controller.py:20788-20806` and `:27212-27219` both wrap the factory call in `except Exception: logger.opt(exception=True).warning("library_provider_factory failed; running without Library tools")`. `ensure_chat_controller` (`console_runtime.py:3624-3663`) uses `kwargs.update(...)` (not `setdefault`) to set `library_provider_factory=functools.partial(_library_provider_for_app, self._app)`, which overwrites `chat_screen.py:10161`'s `library_provider_factory=self._library_activity.build_provider`. `settings_library_rag_defaults.py:94` confirms `direct_library_tools: bool = True` default.
- Note: this is the single highest-severity finding across all 4 slices — independently reproduced via direct construction, not just static reading.

## 2. P1 — a re-chunk yielding zero chunks hard-DELETEs every stored chunk row, inserts nothing, reports "rechunked"
- Verdict: CONFIRMED
- Site now: `tldw_chatbook/Library/library_rechunk_service.py:296-306` (unconditional `DELETE FROM UnvectorizedMediaChunks WHERE media_id = ?`), `:307` (`if rows:` guards the `INSERT`), `_replace_chunk_rows` def at `:238`.
- Proof: direct read of `:296-309` — `with media_db.transaction() as conn: conn.execute("DELETE FROM UnvectorizedMediaChunks WHERE media_id = ?", (media_id,)); if rows: conn.executemany(...)`. `rechunk_one_item`'s empty-guard (`:536-538`, `if not content.strip(): return {"status": "skipped", ...}`) only catches empty *source* text, not zero-chunk *output*. `spans_present` computed as `all(... for chunk in written)` (`:697-700`) — vacuously `True` when `written` is empty, confirming the secondary claim.

## 3. P1 — a cancelled capture extraction never settles: "interrupted" is not in the repository's failure-reason allowlist
- Verdict: CONFIRMED
- Site now: `Library/collections_capture_service.py:322-333` (`reason="interrupted"` at :330, `except CollectionsCaptureError: pass` at :332-333) vs `collections_capture_repository.py:76-88` `_EXTRACTION_FAILURE_REASONS` (11 members, no `"interrupted"`) and `:645-661` `fail_extraction` (`raise CollectionsCaptureError("invalid_extraction_failure_reason")` at :656 when reason not in the set).
- Proof: direct read confirms the frozenset lacks `"interrupted"`, while `interrupt_stale_extractions` (`:738-751`) writes `last_fetch_error = 'interrupted'` via raw SQL, bypassing the same allowlist entirely — the value is legal as stored data but illegal through `fail_extraction`. `retry_extraction` (`:683-695`) docstring: "Requeue one failed or interrupted extraction" — confirms a wedged row (state stuck at `processing`) has no repair path.

## 4. P2 — `_run` uses `asyncio.run` inside a synchronous tool dispatcher; failure is scrubbed to a generic `storage_error`
- Verdict: CONFIRMED
- Site now: `Library/local_library_tool_service.py:133-137` (`_run`, `return asyncio.run(value)`), used at `:626,690,692,1160,1184,1217,1252` (review cited 626/690/692 — matches exactly, plus 4 more call sites not enumerated in the finding but present); swallow at `:490-492` (`except Exception: return _storage_error_payload()`).
- Proof: `grep -n "threading.Thread(target=_runner" tldw_chatbook/Agents/agent_service.py` → line 1995, confirming tool calls run on a dedicated thread with no running event loop today (so `asyncio.run` succeeds now, matching the review's "not currently live" framing).

## 5. P2 — `LibraryNoteSession._pending_save_requested` is write-only: 12 stores, 0 loads
- Verdict: CONFIRMED
- Site now: `Library/library_notes_session.py:299,428,505,550,567,576,590,638,655,703,706,1142` — exact line match to review, all assignments (`= True`/`= False`).
- Proof: `grep -n "_pending_save_requested" library_notes_session.py | grep -v " = "` → 0 hits (every occurrence is an assignment); `grep -rn "_pending_save_requested" tldw_chatbook/ Tests/` outside this file → 0 hits.

## 6. P2 — the review-set picker loads every set's full item list (N+1, unbounded) to render counts
- Verdict: CONFIRMED
- Site now: `Library/review_set_service.py:453` (`SELECT * FROM review_set_items WHERE set_id = ? ORDER BY position`, no LIMIT), `list_review_sets` (:210-232) calls `self._read_review_set(conn, row["set_id"])` once per row at :228-230. Consumer `UI/Screens/library_screen.py:31947 _collect_review_set_picker_rows` (review cited 31946-31965, off by one line — same function).
- Proof: direct read confirms the N+1 pattern and unbounded per-set query.

## 7. P2 — `library_get_media_structure` materializes every chunk row to compute three aggregates
- Verdict: CONFIRMED
- Site now: `Library/local_media_chunk_tool_service.py:325-334` (`_chunk_rows`, `fetchall()`, no LIMIT), consumed at `:401-417` only as `len(chunk_rows)`, a `{chunk_type}` set, and a `{chunk_engine_version}` set.
- Proof: direct read — all three aggregates are expressible as a single `GROUP BY` query.

## 8. P3 — `_error_text` has drifted across its three copies (Library respects its cap; Subscriptions overshoot theirs by 6 chars)
- Verdict: CONFIRMED
- Site now: `Library/library_rag_answer_service.py:354-364` (`cut = ERROR_CHAR_CAP - len(_TRUNCATION_SUFFIX)`, `ERROR_CHAR_CAP=500` at :99) vs `Subscriptions/briefing_service.py:558-567` and `Subscriptions/briefing_cast.py:465-474` (both `message[:ERROR_CHAR_CAP] + " [...]"`, no pre-subtraction).
- Proof: `python -c` with a 2000-char message confirms Library's output is exactly 500 chars (its own cap) while Subscriptions' is exactly `CAP + 6` chars.
- Note: the review's evidence line states Subscriptions' overshoot value as "**506** chars," implying `ERROR_CHAR_CAP=500` there. Current (and at-review-commit, via `git show 3722a85748:...briefing_service.py`) `ERROR_CHAR_CAP` in both Subscriptions files is **1000**, so the actual overshoot output is **1006** chars, not 506 — the review's illustrative number is off by an order of magnitude. The underlying defect (Subscriptions' shape overshoots its own `ERROR_CHAR_CAP` by exactly `len(" [...]")=6` characters, unlike Library's) is real and reproducible; only the review's worked-example arithmetic is wrong.

## 9. P3 — `library_rechunk_service` hand-writes canonical-shaped UTC timestamps instead of `Utils/timestamps.utc_now_iso()`
- Verdict: CONFIRMED
- Site now: `Library/library_rechunk_service.py:229,259` — `datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"` — exact line match to review.
- Proof: direct read, matches claim verbatim.

## 10. P3 — `meeting_speaker_rename` hand-writes the sync-log/FTS contract against four private `MediaDatabase` members
- Verdict: CONFIRMED
- Site now: `Library/meeting_speaker_rename.py:177` (`db._generate_uuid()`), `:183,201,386` (`db._log_sync_event`), `:356` (`db._get_current_utc_timestamp_str()`), `:390` (`db._update_fts_media`) — exact line match to review.
- Proof: direct read, matches claim verbatim.

## 11. P3 — `LocalLibraryCollectionsService` carries a dead constructor seam and four read methods with no production consumer
- Verdict: CONFIRMED
- Site now: `Library/library_collections_service.py:181-186` (`id_factory`/`now_factory`), `:312,368,453,532` (`list_library_collections`, `locate_library_collection_page`, `search_library_collections`, `get_library_collection`).
- Proof: `grep -rn "\.<method>(" tldw_chatbook/` for all four methods → 0 hits outside this module's own definitions. `_id_factory`/`_now_factory` referenced only by `Tests/Library/test_library_collections_service.py:62,63,91,111,121`.

## 12. P3 — `count_export_scope` materializes every id in four sources to produce four integers
- Verdict: CONFIRMED
- Site now: `Library/library_export_scope.py:136-163` — four `len(get_all_*_ids(...))` calls (`get_all_active_media_ids`, `get_all_conversation_ids`, `get_all_note_ids`, `get_all_active_prompt_ids`).
- Proof: direct read, matches claim verbatim; no `COUNT(*)`-style helper used.

## 13. P3 — `LibraryIngestJobRegistry.attach_remote` mutates the stored job in place, breaking the class's replace-on-transition contract
- Verdict: CONFIRMED
- Site now: `Library/library_ingest_jobs.py:935-966` — `job = self._jobs[index]` then `job.remote_job_id = ...` / `job.batch_id = ...` directly, no `dataclasses.replace`. Every other mutator (e.g. `:906-924`, `:999-1010`, `:1036-1046`) does `current = self._jobs[index]` → build `updated` via `dataclasses.replace` → `self._jobs[index] = updated`.
- Proof: direct read confirms `attach_remote` alone skips the copy-and-reassign pattern the module docstring (:43-50) documents as the contract.

## 14. P3 — `_first_present_text` is byte-identical in three Library modules
- Verdict: CONFIRMED
- Site now: `library_conversations_state.py:187-198`, `library_media_state.py:1301-1312`, `library_media_viewer_state.py:142`.
- Proof: `diff` of the three bodies shows zero drift except the parameter name (`record` vs `detail`) — semantically identical, matching the review's "no behavioural drift" claim.

## 15. P3 — `ingest_preflight._collect_files` pays 3-4 stat calls per entry
- Verdict: CONFIRMED
- Site now: `Library/ingest_preflight.py:75-135` (`_collect_files`: `entry.is_symlink()` at :112, `entry.is_dir()` at :117, `entry.is_file()` at :135, each a separate syscall on bare `Path` objects from `iterdir()`), `analyze_path` at `:356` uses `_statted_size` again per file (:438,451).
- Proof: direct read confirms the multi-stat pattern; line numbers shifted slightly (111-149→112-135, 450-457→448-451) but same code shape.

## 16. P3 — `library_artifacts_catalog` calls a private method on a `Chatbooks` service across the package boundary
- Verdict: CONFIRMED
- Site now: `Library/library_artifacts_catalog.py:346` — `self.chatbook_service._is_console_saved_artifact(record)` — exact line match to review.
- Proof: direct read, matches claim verbatim.

TOTALS: confirmed=16 fixed=0 wrong=0 demoted=0 promoted=0
