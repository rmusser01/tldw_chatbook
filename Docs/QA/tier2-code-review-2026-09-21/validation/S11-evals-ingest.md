# S11 validation — Evals + Local_Ingestion

Validated at worktree HEAD `d0face3ebe` against
`qa/tier2-code-review-2026-09-21/slices/S11-evals-ingest.md`.

## Reachability re-check (governs the three demotions below)
Independently re-ran the review's grep and traced further:
- `run_evaluation(` production callers: `Evals/ab_testing.py:182,192` (inside `ABTestRunner`, itself
  only constructed at `ab_testing.py:495` inside `ABTestOrchestrator.__init__`), and
  `eval_orchestrator.py:1178` (inside `quick_eval`, which itself has no caller).
- `grep -rn "ABTestOrchestrator(" tldw_chatbook/ --include=*.py | grep -v /Tests/` → **zero** hits (constructed nowhere in production; confirmed).
- `grep -rn "handle_start_evaluation" tldw_chatbook/` → **zero** hits anywhere in the tree (confirmed).
- `UI/Screens/evals_screen.py` — only reference to the orchestrator is `getattr(app_instance, "evaluation_orchestrator", None)` then `.db` (line 461-462); no `.run_evaluation`/`.quick_eval` call anywhere in the file.
- Checked one path the review didn't explicitly rule out: `Backup_Recovery/runtime_maintenance.py:332-336` references `Evals.eval_orchestrator.EvaluationOrchestrator` — but only as a `_bind(...)` recovery-participant registration (diagnostics/dirty-state tracking), never calling `.run_evaluation`.

**Verdict: the review's reachability claim is CONFIRMED CORRECT.** The whole legacy eval run stack
(`eval_runner.py`, `specialized_runners.py`, `ab_testing.py`'s `ABTestOrchestrator`, `quick_eval`,
`dataset_validator.py`, `dataset_loader.py`, `base_runner.py`, `ui_integration.py`) is unreachable from
the shipped app. The three P1→P2 demotions below are justified by this and are NOT reversed.

## Findings

## 1. P1 — Default install crashes with `AttributeError` on every PDF ingest instead of an actionable install message
- Verdict: CONFIRMED (reproduced independently)
- Site now: `Local_Ingestion/PDF_Processing_Lib.py:614-618` and `:696-700` (`except (RuntimeError, pymupdf.FileDataError, pymupdf.EmptyFileError)`), `pymupdf = None` at `:33-34`. `process_pdf` (`:353`) has no `PDF_PROCESSING_AVAILABLE` guard; only the two inner parsers (`:78`, `:236`) do.
- Proof: `sys.modules['pymupdf']=None; sys.modules['pymupdf4llm']=None` then fresh import + `m.process_pdf(b'not a real pdf', 'test.pdf')` → `RAISED AttributeError : 'NoneType' object has no attribute 'FileDataError'`. `grep -n "mobi" pyproject.toml` confirms `pymupdf`/`pymupdf4llm` are declared only in the `pdf` extra.
- Note: none.

## 2. P1 — Image-ingest path has no byte cap or decompression-bomb guard
- Verdict: CONFIRMED
- Site now: `Local_Ingestion/Image_Processing_Lib.py:132` (`Image.open`→`.convert("RGB")`), `:199-206` (`pixels = list(img.getdata())`), `process_image` (`:321`) reads `stat().st_size` at `:393` for metadata only, never gates on it.
- Proof: `grep -rn "MAX_IMAGE_PIXELS|DecompressionBomb" tldw_chatbook/` → hits in 8 other modules (`chat_image_events.py`, `visual_identity.py`, `console_chat_fork.py`, `character_expression_playback.py`, `Image_Generation/request_validation.py`, `comfyui_image_adapter.py`, `Actor_Packs/contracts.py`, `Petdex/sources.py`), **zero** in `Local_Ingestion/`.
- Note: none.

## 3. P1 — Default MOBI path reads the whole file into memory, then iterates byte-by-byte
- Verdict: CONFIRMED
- Site now: `Local_Ingestion/Book_Ingestion_Lib.py:2386-2410` — `binary_content = f.read()` then `for byte in binary_content:`.
- Proof: `grep -n "mobi" pyproject.toml` → zero matches anywhere in the file (package genuinely undeclared), so this fallback (`:2380` `except ImportError:`) is the path every user gets. Reached from `process_ebook` at `:331` (`elif file_extension in ['.mobi','.azw','.azw3']: result = process_mobi(...)`).
- Note: none.

## 4. P1 — Plaintext and HTML ingest read the entire file into memory with no ceiling
- Verdict: CONFIRMED
- Site now: `local_file_ingestion.py:1488-1490` (plaintext, `file_path.read_bytes()`), `:1514-1516`+`:1519` (HTML, then `BeautifulSoup(html_content, "html.parser")`).
- Proof: `config.py:5291-5294` `[media_processing]` defines only `max_audio_file_size_mb`/`max_video_file_size_mb`; no text/HTML/PDF/document/ebook/image equivalent anywhere in the section.
- Note: none.

## 5. P1 — A committed media row is reported as an ingest failure when the post-commit chunking-config UPDATE fails
- Verdict: CONFIRMED
- Site now: `local_file_ingestion.py:2039-2064` — `_persist()` (which commits via `add_media_with_keywords`) runs, then `_persist_chunking_template_columns` (`:1915-1929`, opens its own `media_db.transaction()`) runs inside the same outer `try`; the outer `except Exception` at `:2064-2067` raises `FileIngestionError` regardless of which of the two failed.
- Proof: read of both call sites — `media_id` is already non-`None` and committed by the time `_persist_chunking_template_columns` is invoked (guarded by `if (template_name or auto_decision) and media_id is not None:`), and its own `with media_db.transaction() as conn:` is a second, separate transaction from the first commit.
- Note: none — this remains inferred/read-only per the review's own confidence rating (no live reproduction attempted, consistent with the "Left UNVERIFIED" table).

## 6. P2 — The whole legacy evaluation run stack (~9,000 lines) is unreachable, including its `subprocess` sandbox *(headline finding)*
- Verdict: CONFIRMED
- Site now: unchanged module set/line counts (`eval_runner.py`, `specialized_runners.py`, `eval_templates.py`, `dataset_validator.py`, `dataset_loader.py`, `base_runner.py`, `ui_integration.py`, `eval_orchestrator._run_admitted_evaluation`).
- Proof: see the "Reachability re-check" section above — independently re-verified all four grep claims plus one path the review didn't explicitly rule out (`Backup_Recovery/runtime_maintenance.py`), which also does not construct/run it.
- Note: none.

## 7. P2 (demoted from P1) — Every eval run >1000 samples marked `failed` with false "results were lost"
- Verdict: CONFIRMED, demotion justified
- Site now: `eval_orchestrator.py:670` `stored_result_count = len(self.db.get_results_for_run(run_id))`; `DB/Evals_DB.py:2042-2046` `get_results_for_run(run_id, limit=1000, offset=0)` (alias for `get_run_results`).
- Proof: code read confirms the mismatch logic (`:683-691`) is unchanged; reachability re-check above confirms `_run_admitted_evaluation` (the only caller of this code path) is unreachable in production. Demotion to P2 stands.
- Note: none.

## 8. P2 (demoted) — Three more unpaginated reads of the same 1000-row API
- Verdict: CONFIRMED, demotion justified
- Site now: `eval_orchestrator.py:863-866` `export_results` (`results = self.get_run_results(run_id)`, no limit override), `:1065-1068` `get_run_status` (`samples_evaluated = len(results)`), `ab_testing.py:209-210` (`get_run_results(run_a_id)`/`(run_b_id)`).
- Proof: all three call `get_run_results`/`get_results_for_run` with the bare 1000 default; contrasted with three correctly-paginated call sites the review names (`character_probe/storage.py`, `word_bench/storage.py`, `skill_eval/storage.py`), spot-checked one (`skill_eval/storage.py`) which does carry an explicit hazard comment.
- Note: none.

## 9. P2 — `Evals/eval_templates.py` permanently shadowed by the `eval_templates/` package
- Verdict: CONFIRMED
- Site now: unchanged.
- Proof: `python -c "import tldw_chatbook.Evals.eval_templates as m; print(m.__file__)"` → `.../Evals/eval_templates/__init__.py`. All three production importers (`eval_orchestrator.py:974`, `task_loader.py:601,667`, `Widgets/template_selector.py:79`) do `from .eval_templates import get_eval_templates`, which resolves to the package.
- Note: none.

## 10. P2 — Falsy `[diarization]` config values silently replaced by defaults
- Verdict: CONFIRMED
- Site now: `Local_Ingestion/diarization_service.py:479-483` — `config[key] = get_cli_setting(f"diarization.{key}", default_value) or default_value`.
- Proof: `config.py` documents `vad_threshold`, `segment_overlap` (`:5454`), `min_speaker_duration` (`:5470`) etc. as legitimately-zero-settable values; the trailing `or default_value` fires on any successfully-resolved falsy value (e.g. user sets `segment_overlap = 0`).
- Note: none.

## 11. P2 — `XML_Ingestion.py` is on the open-XXE register and is entirely unreachable
- Verdict: CONFIRMED
- Site now: `Local_Ingestion/XML_Ingestion.py:4` `import xml.etree.ElementTree as ET`.
- Proof: `grep -rln "XML_Ingestion" tldw_chatbook/ --include=*.py | grep -v XML_Ingestion.py` → only a **comment** reference in `Chunking/auto_selection.py:135`, no import anywhere. `local_file_ingestion.py:1476` raises `"XML file processing is not yet implemented"`. `Tests/Subscriptions/test_watchlist_opml_entity_expansion.py:62` still lists this file in `_KNOWN_UNHARDENED`.
- Note: none.

## 12. P2 — `Evals/eval_runner.py` parses model output with stdlib ElementTree
- Verdict: CONFIRMED
- Site now: `Evals/eval_runner.py:1855-1862 _is_valid_xml` — `ET.fromstring(text.strip())`.
- Proof: `defusedxml` is an unconditional base dependency (`pyproject.toml:78`, inside the top-level `dependencies = [...]` list, not an extra) — confirming the recommended-correction's premise.
- Note: none.

## 13. P2 — `process_zip_of_epubs` extracts an untrusted ZIP with no member/size/ratio limit
- Verdict: CONFIRMED
- Site now: `Book_Ingestion_Lib.py:1605-1607` `zipfile.ZipFile(...).extractall(temp_dir_path_obj)`.
- Proof: `_MAX_EPUB_ARCHIVE_MEMBERS`/`_MAX_EPUB_MEMBER_BYTES`/`_MAX_EPUB_TOTAL_BYTES`/`_MAX_EPUB_COMPRESSION_RATIO` (`:113-117`) are enforced only in `_validate_epub_archive` (single-EPUB path); `grep -rn "process_zip_of_epubs" tldw_chatbook/ --include=*.py | grep -v /Tests/` → only the file's own `def` + a numbered doc comment, confirming no production caller (latent, not live, matching the review's own characterization).
- Note: none.

## 14. P2 — Dead modules and dead symbols, none referenced by production or tests
- Verdict: CONFIRMED (spot-checked, not exhaustively re-run)
- Site now: `Local_Ingestion/API_Endpoint_Sample.py`, `Evals/dataset_validator.py`, `Evals/ui_integration.py` — `grep -rln <modname>` outside the defining file → zero hits for all three. `Evals/dataset_loader.py` / `base_runner.py` — zero production importers; `specialized_runners.py:31` imports `BaseEvalRunner` from `.eval_runner`, not `.base_runner` (confirming production uses the live copy, not the dead one). Spot-checked `extract_images_from_pdf` (`Image_Processing_Lib.py:579`) — zero callers outside its own def, and its `NamedTemporaryFile(suffix=".png", delete=False)` at `:613` has no matching cleanup in the function.
- Note: none.

## 15. P2 — `_load_csv_dataset`/`_load_huggingface_dataset` drifted between the two `DatasetLoader` copies; live copy is weaker
- Verdict: CONFIRMED
- Site now: `eval_runner.py:397-408 _load_csv_dataset` (raw `open`+`csv.DictReader`, no try/except, no empty-rows check) vs `dataset_loader.py:264-287` (wraps in `try/except → DatasetLoadingError.invalid_format`, raises typed "No data rows found" when empty). HF-load error path: `eval_runner.py:519-522` raises bare `ValueError`; `dataset_loader.py:402-412` raises typed `DatasetLoadingError(is_retryable=True)`.
- Proof: side-by-side read of both bodies confirms both drift claims exactly as described.
- Note: none — the review's own "Correction to the 2026-09-17 seed list" note (off-by-one method boundary) was not independently re-verified but is plausible and doesn't affect this finding's substance.

## 16. P2 — Ten copies of a three-line `relocate` default, two in the same file
- Verdict: CONFIRMED
- Site now: all ten sites read and byte-identical bodies confirmed: `Evals/recovery.py:210,602`; `Writing_Interop/recovery.py:99`; `Study_Interop/recovery.py:151`; `Research_Interop/recovery.py:114`; `Notes/recovery.py:228`; `DB/recovery_operations.py:106`; `DB/recovery_core.py:276`; `Backup_Recovery/recovered_media.py:1503`; `Backup_Recovery/recovery_files.py:140`.
- Proof: each is `issues = self.validate(candidate); if issues: raise ValueError(issues[0])`, differing only in the leading comment.
- Note: none.

## 17. P3 — `MAX_FILE_SIZE_MB`/`CONVERSION_TIMEOUT_SECONDS` in the PDF library read the wrong dict, never used
- Verdict: CONFIRMED
- Site now: `PDF_Processing_Lib.py:66-67` `media_config.get("max_pdf_file_size_mb", 50)` against `media_config = {"pdf": {...}}` (`:54-65`) — top-level dict has only key `"pdf"`, so the `.get()` always falls through to the default.
- Proof: `grep -n "MAX_FILE_SIZE_MB|CONVERSION_TIMEOUT_SECONDS" PDF_Processing_Lib.py` → only the two definition lines, never referenced again in the file. No `[media_processing.pdf]` config section exists.
- Note: none.

## 18. P3 — Three `ffmpeg` invocations on the untrusted-media path have no `timeout=`
- Verdict: CONFIRMED
- Site now: `audio_processing.py:1332`, `video_processing.py:1140`, `transcription_service.py:1171` — all `subprocess.run(command, capture_output=True, text=True, check=True)` with no `timeout=` kwarg.
- Note: none.

## 19. P3 — `utc_now_iso()` re-rolled verbatim at the Library ingest writer seam; guard cannot see it
- Verdict: CONFIRMED
- Site now: `local_file_ingestion.py:1914` `datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"`; `:2032` `datetime.now().strftime("%Y-%m-%d")` (naive local time).
- Proof: `python scripts/check_timestamp_writers.py` → `OK` (0 flagged sites) — confirms its regex only catches `datetime.utcnow()` and naive `.now().isoformat()`, missing this `strftime(...)[:-3]+"Z"` shape entirely.
- Note: none.

## 20. P3 — `video_processing.py` mixes loguru and stdlib `logging` for one line
- Verdict: CONFIRMED
- Site now: `:11` `import logging`, `:17` `from loguru import logger`, `:33` `logging.warning("yt-dlp not available. ...")`.
- Proof: `grep -n "^\s*logging\." video_processing.py` → exactly one hit, confirming it's the only stdlib call in the file.
- Note: none.

## 21. P3 — `transcription_service.py`: 4,494 lines, 3,799 in one class
- Verdict: CONFIRMED
- Site now: `wc -l` → 4494 lines exactly; `_LegacyTranscriptionBackend` spans `:307-4106` (3799 lines), `TranscriptionService` at `:4106`, `ParakeetMLXStreamingTranscriber` at `:4384`.
- Note: none.

## 22. P3 — `_transcribe_with_remote_whisper` posts to a configured URL directly, buffers unbounded response
- Verdict: CONFIRMED
- Site now: `transcription_service.py:3211-3236` `requests.post(api_endpoint, files=..., headers={"Authorization": f"Bearer {api_key}"}, timeout=timeout)`, then `response.text` interpolated directly into the `TranscriptionError` message at `:3229-3230` with no cap. Default endpoint `http://localhost:8000/v1/audio/transcriptions` confirmed at `:3141-3142`.
- Note: none — matches review including its "inferred" confidence (egress-bypass exploitability not tested here).

TOTALS: confirmed=22 fixed=0 wrong=0 demoted=0 promoted=0
