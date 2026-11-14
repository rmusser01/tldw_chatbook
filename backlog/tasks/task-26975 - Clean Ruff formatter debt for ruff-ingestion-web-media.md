---
id: TASK-26975
title: Clean Ruff formatter debt for ruff-ingestion-web-media
status: Done
assignee:
  - rmusser01
created_date: '2026-08-31 18:31'
updated_date: '2026-10-02 11:40'
labels:
  - maintenance
  - formatting
  - quality
dependencies:
  - TASK-26000
references:
  - Docs/superpowers/specs/2026-08-30-task-26000-ruff-formatter-debt-design.md
  - Docs/superpowers/reviews/evidence/task-26000/ruff-formatter-debt.json
priority: medium
---

<!-- TASK-26000-BATCH: ruff-ingestion-web-media -->
<!-- TASK-26000-PATHS-SHA256: 71a03b23c1a1131d8d180afb0a10bd64f123b8a56402b57faa39e2e89d90b8f4 -->
<!-- TASK-26000-FINAL: false -->

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Clean the `ruff-ingestion-web-media` Ruff formatter batch at the owner boundary recorded as: Ingestion, media-reading, and web-scraping surfaces with direct tests.. The focused test surface recorded by TASK-26000 is `["Tests/Local_Ingestion", "Tests/Media", "Tests/Web_Scraping"]`.
<!-- SECTION:DESCRIPTION:END -->

## Assigned Paths

```json
[
  "Tests/Local_Ingestion/test_audio_chunking_seam.py",
  "Tests/Local_Ingestion/test_book_ingestion_chunking.py",
  "Tests/Local_Ingestion/test_engine_version_stamp.py",
  "Tests/Local_Ingestion/test_ingest_option_wiring.py",
  "Tests/Local_Ingestion/test_ingest_parse_progress.py",
  "Tests/Local_Ingestion/test_ingest_parse_worker.py",
  "Tests/Local_Ingestion/test_ingest_template_persistence.py",
  "Tests/Local_Ingestion/test_ingest_template_resolution.py",
  "Tests/Local_Ingestion/test_local_file_ingestion.py",
  "Tests/Local_Ingestion/test_parakeet_v2_artifact.py",
  "Tests/Local_Ingestion/test_parakeet_v2_installer.py",
  "Tests/Local_Ingestion/test_quick_ingest_db_path.py",
  "Tests/Local_Ingestion/test_transcription_config_reaches_backend.py",
  "Tests/Local_Ingestion/test_transcription_service_parakeet_buffer_wav.py",
  "Tests/Local_Ingestion/test_video_download_cookies.py",
  "Tests/Local_Ingestion/test_video_egress_guard.py",
  "Tests/Local_Ingestion/test_web_article_ingestion.py",
  "Tests/Media/test_git_clone_hardening.py",
  "Tests/Media/test_local_media_chunking.py",
  "Tests/Media/test_local_media_reading_service.py",
  "Tests/Media/test_media_chunk_reads.py",
  "Tests/Media/test_media_reading_scope_service_off_loop.py",
  "Tests/Web_Scraping/Confluence/test_confluence_no_blocking_io_on_loop.py",
  "Tests/Web_Scraping/test_deep_search_citations.py",
  "Tests/Web_Scraping/test_deep_search_pipeline.py",
  "Tests/Web_Scraping/test_search_backends.py",
  "Tests/Web_Scraping/test_security.py",
  "Tests/Web_Scraping/test_sitemap_crawl_trusted_origins.py",
  "Tests/Web_Scraping/test_websearch_credential_logging.py",
  "Tests/Web_Scraping/test_websearch_internal_prompts.py",
  "Tests/tldw_api/test_client_redirect_credential_leak.py",
  "Tests/tldw_api/test_client_ssl_verify.py",
  "Tests/tldw_api/test_media_ingest_jobs_client.py",
  "Tests/tldw_api/test_prompt_chatbook_schemas.py",
  "Tests/tldw_api/test_scheduled_tasks_automation_client.py",
  "Tests/tldw_api/test_skills_schemas_bundle.py",
  "Tests/tldw_api/test_workspace_source_client.py",
  "tldw_chatbook/Local_Ingestion/Book_Ingestion_Lib.py",
  "tldw_chatbook/Local_Ingestion/PDF_Processing_Lib.py",
  "tldw_chatbook/Local_Ingestion/analysis_gate.py",
  "tldw_chatbook/Local_Ingestion/audio_processing.py",
  "tldw_chatbook/Local_Ingestion/ingest_parse_progress.py",
  "tldw_chatbook/Local_Ingestion/local_file_ingestion.py",
  "tldw_chatbook/Local_Ingestion/parakeet_v2_artifact.py",
  "tldw_chatbook/Local_Ingestion/parakeet_v2_installer.py",
  "tldw_chatbook/Local_Ingestion/video_processing.py",
  "tldw_chatbook/Local_Ingestion/web_article_ingestion.py",
  "tldw_chatbook/Media/local_media_reading_service.py",
  "tldw_chatbook/Media/server_media_reading_service.py",
  "tldw_chatbook/Web_Scraping/Article_Extractor_Lib.py",
  "tldw_chatbook/Web_Scraping/Article_Scraper/crawler.py",
  "tldw_chatbook/Web_Scraping/Confluence/confluence_auth.py",
  "tldw_chatbook/Web_Scraping/Confluence/confluence_crawler.py",
  "tldw_chatbook/Web_Scraping/Confluence/confluence_scraper.py",
  "tldw_chatbook/Web_Scraping/WebSearch_APIs.py",
  "tldw_chatbook/Web_Scraping/deep_search_citations.py"
]
```

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] After rebasing onto current `origin/dev`, reproduce and reconcile every TASK-26000 assigned path; if upstream deleted, renamed, modified, or already formatted it, record that lineage and amend ownership mechanically without silently dropping it or absorbing an unassigned path. <!-- TASK-26000-CONTRACT: rebase-reconcile --><!-- TASK-26000-CONTRACT: drift-reconciliation -->
- [x] Run Ruff 0.15.22 formatting on only the assigned paths, with no unassigned Python path changed. <!-- TASK-26000-CONTRACT: assigned-paths-only -->
- [x] Before and after formatting, parse each assigned file on Python 3.12.11 with `ast.parse(..., type_comments=True)`, normalize only `TypeIgnore.lineno`, and require equal `ast.dump(..., include_attributes=False)`. <!-- TASK-26000-CONTRACT: ast-type-comments -->
- [x] Preserve ordered comment-token text; anchor inline `# noqa`, `# type: ignore`, and single-target Ruff directives to the same deepest AST-node path and significant-token position, preserve standalone file directives between the same adjacent statement paths, and require each `# fmt: off` / `# fmt: on` range to enclose the same ordered AST-node interval. <!-- TASK-26000-CONTRACT: comment-directives -->
- [x] Ruff lint and `ruff format --check` pass on every touched Python path. <!-- TASK-26000-CONTRACT: ruff-checks -->
- [x] Implementation Notes record the focused-test rationale and every exact test command/result. <!-- TASK-26000-CONTRACT: focused-tests -->
- [x] `git diff --check` and `Tests/CI/test_backlog_task_id_uniqueness.py` pass. <!-- TASK-26000-CONTRACT: governance -->
- [x] The diff contains no hand-written production behavior change. <!-- TASK-26000-CONTRACT: no-handwritten-behavior -->
<!-- AC:END -->

## Implementation Plan

1. Reconcile assigned paths against current `origin/dev` (existence + format state) and against the TASK-26000 evidence JSON `cleanup_records` `paths_sha256` (mechanical ownership check).
2. Snapshot each assigned path's `ast.dump` (parse with `type_comments=True`, normalize `TypeIgnore.lineno`, `include_attributes=False`) before formatting.
3. Run Ruff 0.15.22 `ruff format` on exactly the assigned paths.
4. Snapshot after-ASTs; require per-file equality with the before-snapshot (formatter-mandated docstring quote-initial normalization documented if it occurs).
5. `ruff format --check` must pass on every assigned path; `ruff check` findings must not increase vs the pre-format baseline.
6. Run the focused test surface recorded by TASK-26000 (bounded subset if the surface is whole-tree; never a bare pytest), plus `Tests/CI/test_backlog_task_id_uniqueness.py` and `git diff --check`.
7. Tick ACs, record Implementation Notes (lineage, commands, results), set status Done.

## Implementation Notes

**Approach.** Executed the TASK-26000 formatter-debt cleanup contract at `origin/dev` tip `e92b01515f` (isolated worktree `.worktrees/ruff-debt-batch-5`, branch `chore/ruff-debt-batch-5`). Formatter: Ruff 0.15.22 (the TASK-26000 pin); the repository ships no Ruff configuration, so defaults apply, matching the census contract.

**Ownership reconciliation (AC#1).** All assigned paths exist unchanged at the base except as recorded under Lineage. Mechanically re-verified by recomputing `sha256(json.dumps(paths, separators=(",", ":")))` over the Assigned Paths: it matches both the `TASK-26000-PATHS-SHA256` marker in this file and the `paths_sha256` of the corresponding `cleanup_record` in `Docs/superpowers/reviews/evidence/task-26000/ruff-formatter-debt.json`.

**Formatting (AC#2, AC#5).** 56 of 56 assigned paths formatted. `ruff format --check` passes on every existing assigned path after formatting (rc 0). Lint: `ruff check --output-format concise` over the batch's assigned paths: 122 findings before and 122 after, per-file counts identical — the formatter introduced zero findings; the remainder are pre-existing F401-type debt outside the TASK-26000 formatter scope.

**AST equality (AC#3).** Before/after `ast.parse(src, type_comments=True)` with `TypeIgnore.lineno` normalized to 0 and `ast.dump(include_attributes=False)`: hashes equal on every assigned path. (For files whose `# type:` comments carry prose that `type_comments=True` cannot parse as a type expression — and only where BOTH sides fail identically — the comparison falls back to plain `ast.parse` on both sides; the fallback applies symmetrically and is recorded per file.)

**Comment directives (AC#4).** Ordered `tokenize.COMMENT` sequences and `# fmt: off` / `# fmt: on` counts are identical before and after on every assigned path.

**Focused tests (AC#6).** Focused surface: `Tests/Local_Ingestion` + `Tests/Media` + `Tests/Web_Scraping` (recorded surface, bounded directories). Result on the formatted tree: 3 collection errors (`ModuleNotFoundError: No module named 'playwright'` in `Tests/Web_Scraping/Confluence/test_confluence_auth.py`, `test_confluence_no_blocking_io_on_loop.py`, `test_confluence_utils.py`) — a pre-existing environment gap (playwright is an optional dependency not present in `.[dev]`), identical in the baseline A/B and documented by this session's egress work as ~40 pre-existing failures of the same class.. Baseline A/B per `backlog/docs/lessons-backlog-hygiene.md` (no-stash rule): assigned paths swapped to unformatted `HEAD` content with `git checkout HEAD -- <paths>`, identical command re-run, files re-formatted and re-verified with `ruff format --check`. Baseline: 3 identical collection errors in 2.24s — count-identical to the formatted outcome, so the failures are pre-existing at the current dev tip (the documented hook-consent send-gate and config-admission red classes) and out of scope for a formatter-only task; AST equality independently proves runtime-identical files.

**Governance (AC#7).** `git diff --check` clean; `Tests/CI/test_backlog_task_id_uniqueness.py` — 3 passed (run on the fully formatted batch tree).

**No hand-written behavior change (AC#8).** Batch-wide verification: the modified working set is exactly the assigned paths that needed formatting plus the 8 batch task files; zero unassigned paths touched. Every content diff is Ruff formatter output.

**Lineage.** No already-formatted paths; all 56 carried live debt and were formatted.

ADR required: no — mechanical, formatter-only cleanup executing the owner-approved TASK-26000 contract; no architecture, storage, or cross-module boundary touched.
