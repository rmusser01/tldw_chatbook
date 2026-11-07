# S11 — Evals + Local_Ingestion

**Coverage:** files read in full: 9 | sampled (substantial sections + targeted greps): 18 | mechanical only: 47 (of 74).
Full: `local_file_ingestion.py` (2375), `Image_Processing_Lib.py` (659), `dataset_loader.py` (435), `base_runner.py`
(288), `ingest_parse_worker.py` (285), `XML_Ingestion.py` (214), `web_article_ingestion.py` (170), both `__init__.py`.
Sampled: `eval_runner.py` (~1400/3000), `specialized_runners.py` (~500/2997), `Book_Ingestion_Lib.py` (~600/2696),
`eval_orchestrator.py` (~750/1212), `PDF_Processing_Lib.py` (~700/1390), `transcription_service.py` (~250/4494 + full
structural map), `audio_processing.py`, `video_processing.py`, `diarization_service.py`, `Document_Processing_Lib.py`,
`task_loader.py`, `OCR_Backends.py`, `recovery.py`, `exporters.py`, `character_probe/storage.py`, `ab_testing.py`,
`dataset_validator.py`, `ui_integration.py`. Mechanical only: the 47 remaining, via an AST unreferenced-symbol scan.

> **LEAD CORRECTION (Phase 4).** The reviewer filed the 1000-row eval-result truncation at P1. I verified the code
> (confirmed) *and* the reachability (`grep -rn '\.run_evaluation(\|quick_eval\|ABTestOrchestrator\|handle_start_evaluation'`):
> `EvaluationOrchestrator.run_evaluation` is called only from `ab_testing.py:182,192`; `ABTestOrchestrator` is
> constructed nowhere; `quick_eval` has no caller; `handle_start_evaluation` does not exist in the tree.
> **The whole legacy eval run stack is unreachable from the shipped app**, so those three findings are **demoted to
> P2** — real defects that no user can currently hit. The reviewer's own P2 (the dead stack, ~9,000 lines, including a
> `subprocess` code-execution sandbox) is the finding that matters and is promoted to the top of this slice.

## Findings

### P1 [D1] — A default install crashes with `AttributeError: 'NoneType' object has no attribute 'FileDataError'` on every PDF ingest instead of saying "install tldw_chatbook[pdf]"
- Where: `Local_Ingestion/PDF_Processing_Lib.py:614-618` and `:696-700`
  (`except (RuntimeError, pymupdf.FileDataError, pymupdf.EmptyFileError)`), with `pymupdf = None` at `:33-36`.
  `process_pdf` (`:353`) has no `PDF_PROCESSING_AVAILABLE` guard — only the two inner parsers do (`:78`, `:236`).
- Evidence: **reproduced twice** (reviewer, then lead independently) in the worktree with `pymupdf`/`pymupdf4llm`
  blocked at import: `PDF_PROCESSING_AVAILABLE = False | pymupdf = None` →
  `RAISED AttributeError : 'NoneType' object has no attribute 'FileDataError'`.
  `pymupdf` is in the `pdf` extra only (`pyproject.toml:184-186`), not base `dependencies`.
- Why it matters: the inner parser raises the correct `ImportError("…pip install tldw_chatbook[pdf]")`, but Python
  evaluates the `except` tuple against `None` while handling it, replacing the actionable message.
  `parse_local_file_for_ingest` then wraps it as
  `FileIngestionError("Failed to ingest pdf file: 'NoneType' object has no attribute 'FileDataError'")` — what the
  Library queue shows the user, on the most common ingest type, on a stock install.
- Recommended correction: guard `process_pdf` at entry (mirroring `:78`), and bind the `pymupdf` ImportError fallback
  to a private sentinel exception class rather than `None`.
- Size: S · ADR: no · Confidence: **verified (reproduced)**
- Pinning test: none (`Tests/Utils/test_optional_deps.py:583-589` only asserts the flag is a bool)
- Already covered: none. TASK-32811.3 is about the dependency *diagnostic*, not this raise path.

### P1 [D1] — The image-ingest path has neither a byte cap nor a decompression-bomb guard, while eight other modules in the repo have both
- Where: `Local_Ingestion/Image_Processing_Lib.py:132` (`Image.open` → `.convert("RGB")`), `:184-222`
  (`extract_visual_features`: `pixels = list(img.getdata())` — one Python tuple per pixel — then a second `list(...)`
  at `:219`), `:401`. `process_image` (`:321`) reads `stat().st_size` at `:393` for metadata only and never gates on it.
- Evidence: `grep -rn "MAX_IMAGE_PIXELS|DecompressionBomb" tldw_chatbook` → guards present in
  `Event_Handlers/Chat_Events/chat_image_events.py:107-122`, `Character_Chat/visual_identity.py:1891-1928`,
  `Chat/console_chat_fork.py:800`, `Chat/character_expression_playback.py:134`,
  `Image_Generation/request_validation.py:36,337`, `Image_Generation/adapters/comfyui_image_adapter.py:1372`,
  `Actor_Packs/contracts.py:713-739`, `Petdex/sources.py:195` — and **zero** hits in `Local_Ingestion/`.
- Why it matters: PIL only *warns* above `MAX_IMAGE_PIXELS` and decodes anyway. TASK-32806.8's own arithmetic
  (96 MP → ~288 MB) applies here with no 10 MB size cap in front of it, and `list(img.getdata())` multiplies that
  again by ~40× as Python objects. Runs inside a spawned parse-pool worker.
- Recommended correction: escalate `DecompressionBombWarning` to an error and apply a size cap in `process_image`
  before the first `Image.open`, matching `chat_image_events.py:107-122`.
- Size: S · ADR: no · Confidence: verified (absence) / inferred (memory figures not measured here)
- Already covered: **NO — new.** TASK-32806.8 is scoped to "the chat attachment path" and names the five modules that
  already escalate; `Local_Ingestion` is not among them and is not in its ACs.

### P1 [D1] — The default MOBI path reads the whole file into memory and then iterates it byte-by-byte in Python
- Where: `Local_Ingestion/Book_Ingestion_Lib.py:2386-2410` — `binary_content = f.read()` then `for byte in
  binary_content:` building a per-character list.
- Evidence: the `mobi` package is not declared anywhere in `pyproject.toml`, so this "basic text extraction" fallback
  (`:2380`) is the path every user gets. Reached from `process_ebook` at `:332`.
- Why it matters: a 200 MB file named `.mobi` is fully buffered then walked one byte at a time at Python speed inside
  the parse worker. Unlike the EPUB path 2200 lines above, there is no admission check at all.
- Recommended correction: cap the read (the module already has `_MAX_EPUB_MEMBER_BYTES`/`_MAX_EPUB_TOTAL_BYTES` at
  `:110-113`) and replace the byte loop with a `bytes.translate`/`re` scan over a bounded buffer.
- Size: S · ADR: no · Confidence: verified (code + dependency absence)
- Already covered: none. TASK-32806.6 covers agent read tools and card/trajectory import, not `Local_Ingestion`.

### P1 [D1] — Plaintext and HTML ingest read the entire user-picked file into memory with no ceiling
- Where: `Local_Ingestion/local_file_ingestion.py:1488-1490` (plaintext) and `:1514-1516` (HTML, then
  `BeautifulSoup(html_content, "html.parser")` at `:1519`).
- Evidence: `detect_file_type` (`:408`) routes `.txt .md .markdown .rst .log .csv` to `plaintext`.
  `[media_processing]` (`config.py:5291-5294`) defines `max_audio_file_size_mb` and `max_video_file_size_mb` only;
  `audio_processing.py:217-222` and `video_processing.py:115` enforce theirs — no equivalent for
  text/HTML/PDF/document/ebook/image.
- Why it matters: a multi-GB `.log` is a routine thing to point an importer at. The read, then `_decode_ingest_text`'s
  decode (a second copy), then chunking, all in a spawned worker with no bound.
- Recommended correction: one `max_text_file_size_mb` under `[media_processing]`, symmetric with the two that exist.
- Size: S · ADR: no (config key only) · Confidence: verified

### P1 [D1] — A successfully committed media row is reported to the user as an ingest failure when the post-commit chunking-config UPDATE fails
- Where: `Local_Ingestion/local_file_ingestion.py:2039-2064` — `add_media_with_keywords` commits, then
  `_persist_chunking_template_columns` (`:1915-1930`) opens a **second** `media_db.transaction()`; both sit inside the
  same `try` whose `except Exception` (`:2067-2071`) raises `FileIngestionError`.
- Why it matters: the Library queue marks the job failed and offers Retry, while the media row exists. A user who
  retries gets a duplicate-or-overwrite decision on a row the UI told them was never written.
- Recommended correction: wrap only `_persist_chunking_template_columns` in its own handler that logs and appends a
  warning to the payload; the ingest stays successful once `media_id` is non-`None`.
- Size: S · ADR: no · Confidence: **inferred** (read only — see UNVERIFIED)
- Already covered: none (TASK-32801.1-.5 are all Done and none names this seam)

### P2 [D1] — The whole legacy evaluation run stack (~9,000 lines) is unreachable from the shipped app, including its `subprocess` code-execution sandbox  *(promoted by the lead to the slice's headline finding)*
- Where: `Evals/eval_runner.py` (3000), `Evals/specialized_runners.py` (2997), `Evals/eval_templates.py` (1298, also
  shadowed — below), `Evals/dataset_validator.py` (659), `Evals/dataset_loader.py` (435), `Evals/base_runner.py` (288),
  `Evals/ui_integration.py` (182), plus `eval_orchestrator._run_admitted_evaluation` (`:434-830`).
- Evidence (reviewer, re-run by the lead): `grep -rn '\.run_evaluation(\|quick_eval\|ABTestOrchestrator\|handle_start_evaluation'`
  → `run_evaluation` called only from `Evals/ab_testing.py:182,192` (`ABTestRunner`, constructed only by
  `ABTestOrchestrator` at `ab_testing.py:495`, which **nothing constructs**); `quick_eval` (`eval_orchestrator.py:1145`)
  has no caller; `handle_start_evaluation` — the handler `Docs/Evals-UI-Fix-1.md:94` documents as the entry point —
  **does not exist anywhere in the tree**. `UI/Screens/evals_screen.py:455-456` uses the orchestrator **only** for `.db`.
  Live eval paths are `skill_eval/`, `word_bench/`, `character_probe/` via `sample_bench`/`SkillEvalRunner`.
- Why it matters: it is the context that demotes three other findings here — and it means
  `CodeExecutionRunner._execute_code` (`specialized_runners.py:357-406`), which writes LLM-generated Python to a temp
  file and runs it with `sys.executable` behind an in-child "disable `eval`/`exec`/`open`/`__import__`" guard that
  `sys.modules['os']` walks straight past, is dead weight carrying an attack surface nothing needs.
- Recommended correction: a ruling first — revive or retire. If retire, the delete is large but clean;
  `backlog/decisions/031-bounded-evaluation-and-tool-worker-execution.md` is the place to record it. Note
  `Skills_Interop/skill_script_runner.py:7` documents itself as diverging from `specialized_runners.py`, so that file
  is a live *reference* even though its code is dead.
- Size: L · ADR: yes (`031-bounded-evaluation-and-tool-worker-execution.md`) · Confidence: verified
- Pinning test: `Tests/Evals/test_code_execution_security.py`, `test_specialized_runners.py`, `test_eval_runner.py`
  all exercise it — tests keep it green, not reachable.
- Already covered: none. TASK-32807.1-.6 name widget/RAG/Event_Handlers/Tools/Chat/Utils modules, not `Evals/`.

### P2 [D1] *(demoted from P1 by the lead — dead path)* — Every evaluation run of more than 1000 samples is marked `failed` with a false "results were lost" message
- Where: `Evals/eval_orchestrator.py:670` `stored_result_count = len(self.db.get_results_for_run(run_id))` → `:683-691`.
  `DB/Evals_DB.py:2043-2047` `get_results_for_run(run_id, limit=1000, offset=0)` → `get_run_results` (`LIMIT ? OFFSET ?`).
- Evidence: reviewer ran a real in-memory `EvalsDB`, 1005 `store_result` calls → `stored=1005 get_results_for_run
  len=1000` → run marked `failed` with `error = "Only stored 1000 of 1005 evaluation results"`. Lead re-read
  `Evals_DB.py:2043-2047` and `eval_orchestrator.py:668-692`: the code path is unambiguous.
- Recommended correction: add `EvalsDB.count_results_for_run(run_id)` (`SELECT COUNT(*)`) and use it at `:670`.
- Size: S · ADR: no · Confidence: verified (code) — **unreachable today**

### P2 [D1/D4] *(demoted from P1 — same dead path)* — Three more unpaginated reads of the same 1000-row API silently truncate exports, status counts, and A/B comparisons
- Where: `Evals/eval_orchestrator.py:866` (`export_results` writes ≤1000 of N results to the user's JSON/CSV),
  `:1065-1068` (`get_run_status` → `samples_evaluated` caps at 1000), `Evals/ab_testing.py:209-210`.
- Evidence: same `LIMIT 1000` default. The correct drain pattern already exists three times in this slice **and is
  documented as a hazard**: `Evals/character_probe/storage.py:522,978`, `Evals/word_bench/storage.py:435,544`,
  `Evals/skill_eval/storage.py:229-233` — all page explicitly with a comment naming the default.
- Why it matters: the export is the worst of the three — the file looks complete.
- Recommended correction: an `EvalsDB.iter_run_results(run_id)` generator so no caller re-rolls the loop.
- Size: M · Confidence: verified (limit) / inferred (per-site impact)

### P2 [D3] — `Evals/eval_templates.py` (1298 lines) is permanently shadowed by the `Evals/eval_templates/` package and can never execute
- Evidence: `python -c "import tldw_chatbook.Evals.eval_templates as m; print(m.__file__)"` →
  `.../Evals/eval_templates/__init__.py`. All three production importers (`eval_orchestrator.py:974`,
  `task_loader.py:601,667`, `Widgets/template_selector.py:79`) resolve to the package.
- Why it matters: a 53 KB file that looks like the template system, is packaged and shipped, and silently has no
  effect. Editing it produces no behaviour change — a trap, not just dead weight.
- Size: S · Confidence: verified

### P2 [D1] — Falsy values in the shipped `[diarization]` config section are silently replaced by defaults
- Where: `Local_Ingestion/diarization_service.py:479-483` —
  `config[key] = get_cli_setting(f"diarization.{key}", default_value) or default_value`.
- Evidence: `get_cli_setting` already honours its own default (`config.py:8542-8622`), so the trailing `or` only fires
  on a *successfully resolved falsy* value. Affected documented keys in `config.py`'s `[diarization]` block:
  `vad_threshold`, `vad_min_speech_duration`, `vad_min_silence_duration`, `segment_overlap`, `min_segment_duration`,
  `merge_threshold`, `min_speaker_duration` — each legitimately `0`, each documented with a comment inviting the user
  to tune it.
- Why it matters: a user who sets `segment_overlap = 0` gets 0.5 s of overlap, with no warning anywhere.
- Size: S · Confidence: verified

### P2 [D1] — `Local_Ingestion/XML_Ingestion.py` is on the repo's own open-XXE register and is also entirely unreachable
- Where: `Local_Ingestion/XML_Ingestion.py:4` (`import xml.etree.ElementTree as ET`), `:27`, `:100`, `:115`.
- Evidence: `Tests/Subscriptions/test_watchlist_opml_entity_expansion.py:41-88` maintains `_KNOWN_UNHARDENED`,
  explicitly "a REGISTER OF OPEN DEFECTS, not an allowlist", and lists this file. **New evidence:** no importer at all —
  `Tests/Chunking/test_callsite_characterization.py:113` and `Tests/Library/test_ingest_capabilities.py:963` both say so,
  and `local_file_ingestion.py:1474-1476` raises `"XML file processing is not yet implemented"`. Also carries a
  gradio-era signature (`import_file.name`, `:100`) and stdlib `logging` (`:14`, `:57`) in a loguru package.
- Recommended correction: delete the module and its `_KNOWN_UNHARDENED` entry in the same commit (the register's
  `test_known_unhardened_entries_are_still_unhardened` requires the pairing).
- Size: S · Confidence: verified

### P2 [D1] — `Evals/eval_runner.py` parses **model output** with stdlib ElementTree — the sharpest entry on the same register
- Where: `Evals/eval_runner.py:1855-1862` `_is_valid_xml` → `ET.fromstring(text.strip())`.
- Evidence: `Tests/Subscriptions/test_watchlist_opml_entity_expansion.py:55-60` names this exact file: *"Parses MODEL
  OUTPUT … prompt-injection reachable — the sharpest of the seven: a poisoned document in the corpus can choose the XML
  the model emits."* `except Exception: return False` at `:1861` does not help — a billion-laughs payload exhausts
  memory *during* the parse. Mitigated today only by the unreachability finding above.
- Recommended correction: `defusedxml.ElementTree` (already a **base** dependency, `pyproject.toml:78`), or delete with
  the rest of the legacy stack.
- Size: S · Confidence: verified

### P2 [D1] — `process_zip_of_epubs` extracts an untrusted ZIP with no member, size, or ratio limit, 1400 lines below the module's own admission gate
- Where: `Local_Ingestion/Book_Ingestion_Lib.py:1606-1607` `zipfile.ZipFile(...).extractall(temp_dir_path_obj)`.
- Evidence: the same module defines `_MAX_EPUB_ARCHIVE_MEMBERS=10_000`, `_MAX_EPUB_MEMBER_BYTES`,
  `_MAX_EPUB_TOTAL_BYTES`, `_MAX_EPUB_COMPRESSION_RATIO=200` (`:110-114`) and enforces all four in
  `_validate_epub_archive` (`:188-243`) for the single-EPUB path. The ZIP path calls none of them. Zip-slip itself is
  **not** reachable (CPython's `_extract_member` strips `..`/absolute components) — decompression-bomb only.
  `process_zip_of_epubs` has no production caller, so this is latent, not live.
- Size: S · Confidence: verified

### P2 [D3] — Dead modules and dead symbols in the slice, none referenced by production or tests
- Whole modules, zero references anywhere: `Local_Ingestion/API_Endpoint_Sample.py` (906),
  `Evals/dataset_validator.py` (659), `Evals/ui_integration.py` (182). Production-dead, tests-only:
  `Evals/dataset_loader.py` (435), `Evals/base_runner.py` (288) — production `specialized_runners.py:31` imports
  `BaseEvalRunner` from `.eval_runner`, not `.base_runner`.
- Dead symbols in live modules: `ab_testing.py:489 ABTestOrchestrator`; `eval_orchestrator.py:1145 quick_eval`,
  `:1201 create_task_template`; `eval_runner.py:2944 format_error_for_user`, `:2953 format_error_for_log`,
  `:2962 handle_batch_errors`; `specialized_runners.py:2892 get_specialized_runner`, `:2936 list_specialized_runners`,
  and classes `MathReasoningRunner` (`:2116`), `SummarizationRunner` (`:2374`), `DialogueRunner` (`:2623`) —
  `EvalRunner`'s dispatch (`:2664-2695`) constructs only Code/Safety/Multilingual/Creative/ResearchReport;
  `Image_Processing_Lib.py:579 extract_images_from_pdf`, `:642 simple_ocr`, `:547 process_image_batch`;
  `PDF_Processing_Lib.py:265 extract_metadata_from_pdf`, `:1298 process_pdf_task`;
  `Book_Ingestion_Lib.py:1560 process_zip_of_epubs`, `:1720 _process_markup_or_plain_text`, `:2282 ingest_folder`,
  `:2203 ingest_text_file`; `Document_Processing_Lib.py:708 process_document_with_docling`;
  `OCR_Backends.py:1325 get_ocr_manager`; `Evals/config_loader.py:285 reload_config`;
  `Evals/skill_eval/storage.py:84 list_skill_eval_benches`.
- Evidence: AST collection of top-level defs across both packages, cross-referenced against a tokenised `grep -rno` of
  all of `tldw_chatbook/` (excluding the defining line) and of `Tests/`; run twice (with and without intra-file uses)
  to bracket false positives.
- Why it matters: `extract_images_from_pdf` alone writes one `NamedTemporaryFile(delete=False)` per embedded PDF image
  (`:613-615`) and nothing ever deletes them — a leak waiting for its first caller.
- Size: M · Confidence: verified
- Already covered: **partially** — none of TASK-32807's six sub-tasks covers `Evals/` or `Local_Ingestion/`.

### P2 [D4] — `_load_csv_dataset` and `_load_huggingface_dataset` drifted between the two `DatasetLoader` copies, and the live copy is the weaker one
- Where: `Evals/eval_runner.py:180-546` vs `Evals/dataset_loader.py:36-435`.
- Evidence: `diff` of the two class bodies. Drift: (a) the standalone copy wraps the CSV read in
  `try/except → DatasetLoadingError.invalid_format` and raises a typed "No data rows found"
  (`dataset_loader.py:269-287`); the live copy (`eval_runner.py:403-405`) has neither — a non-UTF-8 CSV raises a raw
  `UnicodeDecodeError` and an empty CSV returns zero samples silently. (b) The standalone copy raises a typed
  `DatasetLoadingError(is_retryable=True)` for a failed HF load (`:402-412`); the live copy raises a bare `ValueError`
  (`eval_runner.py:520-523`), losing the retryability flag.
- Why it matters: invisible because `Tests/Evals/test_eval_runner.py:68-166` parametrizes both modules over four tests
  that exercise routing and HF-availability only, never CSV. The tests' names ("…_is_unchanged") say the repo *intends*
  these to stay identical; they have not.
- Size: S · Confidence: verified
- Pinning test: `Tests/Evals/test_eval_runner.py::test_local_dataset_routing_is_unchanged` (+3 siblings) — states parity
  as a requirement but does not cover the drifted branches.
- **Correction to the 2026-09-17 seed list:** it recorded `eval_runner.py:296 ≡ dataset_loader.py:163`. At `3722a857`
  the pair is `eval_runner.py:297` / `dataset_loader.py:163` and those bodies *are* identical apart from the docstring —
  but the seed stopped one method too early.

### P2 [D4] — Ten copies of a three-line `relocate` default, two of them in the same file
- Where: `Evals/recovery.py:210-215` and `:602-605`; `Writing_Interop/recovery.py:99`, `Study_Interop/recovery.py:151`,
  `Research_Interop/recovery.py:114`, `Notes/recovery.py:228`, `DB/recovery_operations.py:106`,
  `DB/recovery_core.py:276`, `Backup_Recovery/recovered_media.py:1503`, `Backup_Recovery/recovery_files.py:140`.
- Evidence: bodies byte-identical (`issues = self.validate(candidate); if issues: raise ValueError(issues[0])`); only
  the leading comment differs per owner. `Backup_Recovery/models.py:132-140` declares `OwnerAdapter` as a `Protocol`,
  so there is no shared implementation today. No behavioural drift — pure D4.
- Recommended correction: a concrete `ValidateOnlyRelocate` mixin beside the Protocol in `Backup_Recovery/models.py`.
- Size: M · Confidence: verified · Owner: possibly task-32808.9 (To Do)

### P3 [D1] — `MAX_FILE_SIZE_MB` and `CONVERSION_TIMEOUT_SECONDS` in the PDF library read the wrong dict and are never used
- Where: `Local_Ingestion/PDF_Processing_Lib.py:66-67` — `media_config.get("max_pdf_file_size_mb", 50)` against
  `media_config = {"pdf": {...}}` (`:54-65`), which has no such key. `grep` → only these two definition lines; no
  `[media_processing.pdf]` section exists.
- Why it matters: two constants that read as a PDF size cap and a conversion timeout, both inert. They make the
  *absence* of a PDF size cap look handled.
- Size: S · Confidence: verified

### P3 [D1] — The three `ffmpeg` invocations on the untrusted-media path have no `timeout=`
- Where: `Local_Ingestion/audio_processing.py:1332`, `video_processing.py:1140`, `transcription_service.py:1171` — all
  `subprocess.run(command, capture_output=True, text=True, check=True)`. All list-form, no `shell=True` (no injection),
  but none passes `timeout`.
- Why it matters: a crafted container that makes ffmpeg spin blocks the ingest worker indefinitely; the
  `LocalAudioProcessor._cancelled` flag (`:214`) is checked between files, not inside the call.
- Size: S · Confidence: verified

### P3 [D4] — `utc_now_iso()` re-rolled verbatim at the Library ingest writer seam, and the guard cannot see it
- Where: `Local_Ingestion/local_file_ingestion.py:1914`
  `datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"`; also `:2032`
  `ingestion_date=datetime.now().strftime("%Y-%m-%d")` — naive **local** time.
- Evidence: `Utils/timestamps.py:44-65` produces exactly that string via `to_utc_iso`/`utc_now_iso`. Five other verbatim
  copies repo-wide (`Library/library_rechunk_service.py:229,259`, `Chatbooks/server_chatbook_service.py:30`,
  `DB/Client_Media_DB_v2.py:2229`, `DB/Prompts_DB.py:1025`). `scripts/check_timestamp_writers.py` passes (`OK`) — its
  census covers `datetime.utcnow()` and `datetime.now().isoformat()`, not this `strftime` shape, and
  `scripts/timestamp_writer_census.tsv` has no `Local_Ingestion` or `Evals` row.
- Recommended correction: `utc_now_iso()` at `:1914`; decide the `ingestion_date` contract; widen
  `check_timestamp_writers.py` to catch the `strftime("…%f")[:-3]` shape so the ratchet stops missing six sites.
- Size: S · ADR: `173-…` · Confidence: verified
- Already covered: TASK-32803.5 is **Done** — this shows it insufficient. **Second independent sighting of the same
  guard blind spot (see S01-P3).**

### P3 [D3] — `video_processing.py` mixes loguru and stdlib `logging` for one line
- Where: `:11` (`import logging`), `:17` (`from loguru import logger`), `:33`
  (`logging.warning("yt-dlp not available. Video downloading will be disabled.")`) — the only stdlib call in the file.
- Why it matters: that warning bypasses the app's loguru sinks, so the one message telling a user why video download is
  unavailable never reaches the Logs window.
- Size: S · Confidence: verified

### P3 [D3] — `transcription_service.py`: 4,494 lines, 3,799 of them in one class
- Where: `_LegacyTranscriptionBackend` spans `:307`–`:4106`; `TranscriptionService` (`:4106`) and
  `ParakeetMLXStreamingTranscriber` (`:4384`) follow. No size-ratchet row (`grep -rn "transcription_service"
  scripts/*.tsv` → 0).
- Responsibilities in the one class: ffmpeg conversion, six provider backends (faster-whisper, parakeet-ONNX,
  parakeet-MLX, lightning-MLX, qwen2audio, remote-whisper), model download/caching, chunking, progress plumbing.
- Size: L · Confidence: verified · Already covered: TASK-32809.2 (In Progress) — check whether this file is in its list.

### P3 [D1] — `_transcribe_with_remote_whisper` posts to a configured URL with `requests` directly and buffers an unbounded response
- Where: `Local_Ingestion/transcription_service.py:3211-3236` — `requests.post(api_endpoint, files=...,
  headers={"Authorization": f"Bearer {api_key}"}, timeout=timeout)`, then `response.json()` / `response.text` with no
  cap; `response.text` is interpolated into the `TranscriptionError` message at `:3230-3232`.
- Evidence: every other outbound fetch in this slice goes through `Utils/egress.py`
  (`web_article_ingestion.py:73-92`, `audio_processing.py:265-303`).
- Why it matters: the endpoint defaults to `http://localhost:8000` (`:3142`), which the egress guard would likely
  block — so the bypass may be deliberate — but the missing response-size bound and the raw-body-into-an-error-message
  are defects either way.
- Size: S · Confidence: inferred

## Candidate triage (summary; full table in the slice transcript)
`DUP_VERBATIM` `load_dataset_samples`/`_load_local_dataset`/`_load_json_dataset` ×2 each — **confirmed** (docstring-only
differences); `relocate` ×10 — **confirmed**, no drift. Lazy `__getattr__` ×4+×2 — **retired** (PEP 562, documented at
`Local_Ingestion/__init__.py:1-27`). `dotted_section_setting` 35 rows — **retired**: `get_cli_setting`'s nested-tree
fallback (`config.py:8592-8602`) resolves these correctly; the PDF/document rows return hardcoded defaults only because
those config sections don't exist — but a **real bug found beside them** (`PDF_Processing_Lib.py:66-67`, P3).
`except_exception_pass` 9 rows — 8 **retired** (progress ticks, owned-client cleanup citing Qodo PR #2223, import
guards, temp unlink), 1 minor in dead code. `except_exception_return` 11 rows — **retired**, all typed predicates (the
`_is_valid_xml` one is a finding for a different reason). `function_body_import` 75 rows — **retired as a class**
(task-257 boot-cost deferral, documented in place). `get_cli_setting_hot` — **retired as hot** (once per init,
cache-backed) but a **different real bug at that line** (the `or default_value` tail, P2). `legacy_markers` 32 rows —
partially confirmed (`exporters.py:529,537` are dead legacy wrappers). `raw_1024x1024` 14 rows — **retired**, every row
is an arithmetic size *bound*, not a formatter. `raw_mkdir` 6 rows — **retired**. `strftime` 9 rows — 2 **confirmed**
(P3), 7 retired (display/timecodes/filenames). `tempfile_no_secure` 15 rows — **2 confirmed leaks**
(`Image_Processing_Lib.py:159` — the unlink at `:308-312` is skipped when `ocr_manager.process_image` raises, since the
outer handler at `:316` returns without cleanup; and `:613`, dead so latent), 13 retired. `try_import_guard` 96 rows —
**retired as a class**, with one exception found by reading not grepping: `PDF_Processing_Lib.py:28-37`'s
`pymupdf = None` fallback is what makes P1 #1 crash.

## D4 observations for repo-wide Phase 3
1. **Unpaginated reads of a paginated API.** `EvalsDB.get_run_results(limit=1000)` has 4 unpaginated callers and 3
   correctly-paginated ones that each carry a comment warning about the default. Canonical home: `iter_run_results()` +
   `count_results_for_run()` on `Evals_DB.py`. Worth a repo-wide sweep for other `limit=N`-defaulted accessors called bare.
2. **`ffmpeg` discovery re-rolled 3× with drifting fallback lists**: `transcription_service.py:1179-1210`
   (config → `shutil.which` → platform paths), `video_processing.py:1160-1174` (config → `which`, then raises),
   `audio_processing.py:1288-1303` (env var → three hardcoded Unix paths, **no `which`**). Canonical home:
   `Local_Ingestion/_ffmpeg.py::find_ffmpeg()`. The audio copy is the weakest and is the only one on the time-trim path.
3. **Per-file size caps.** Two exist and are enforced; six ingest types have none, and `PDF_Processing_Lib.py:66`
   defines an inert one. Canonical home: `ingest_size_cap(media_type)` next to `detect_file_type`.
4. **`relocate` mixin** — canonical home `Backup_Recovery/models.py`.
5. **Timestamp ratchet blind spot** — 6 files re-roll `utc_now_iso()` verbatim as `strftime(...)[:-3] + "Z"` while the
   guard reports OK. Widening one regex converts a manual sweep into a ratchet.
6. **Metrics labels carrying unbounded user data.** `PDF_Processing_Lib.py:84,186,242,260,348` and
   `Document_Processing_Lib.py:157` pass `labels={"file_path": <user path>}` and `labels={"error": str(e)}` —
   unbounded label cardinality, plus user paths and exception text in telemetry. Likely repo-wide.

## Left UNVERIFIED
| Claim | Why | Command |
|---|---|---|
| A failing `_persist_chunking_template_columns` reports a committed ingest as failed | read only | add a test monkeypatching it to raise, then assert `media_db.get_media_by_url(...)` is non-`None` while `persist_parsed_media` raised |
| Image decompression-bomb memory figures for `Local_Ingestion` | absence verified; MB numbers carried from TASK-32806.8's arithmetic | `tracemalloc` around `Image_Processing_Lib` on a generated ~96 MP flat PNG |
| MOBI byte-loop wall-clock / peak RSS | read only | `tracemalloc` around `Book_Ingestion_Lib.process_mobi` on a 200 MB fixture |
| ffmpeg hang without `timeout=` | read only | `subprocess.run([ffmpeg,"-f","lavfi","-i","testsrc","-t","999999",out])` under the audio path with a wall clock |
| Whether egress would block the remote-whisper localhost default | not tested | `guarded_fetch_requests('http://localhost:8000/v1/audio/transcriptions', max_bytes=1)` |
| `DUP_SHAPE` `capture` ×14 / `validate` ×3 | mostly outside the slice | `git grep -n "def capture(self, item: StorageItem"` then diff the bodies |
| Whether `transcription_service.py` is in TASK-32809.2's budget list | task file not opened | `cat "backlog/tasks/task-32809.2 - ...md"` |
| On-loop sqlite in the Evals screen workers | traced only `_run_bench_worker` into `sample_bench` | `git grep -n "EvalsDB\|self\._db\." tldw_chatbook/UI/Evals/sample_bench.py` and check for `asyncio.to_thread` |
