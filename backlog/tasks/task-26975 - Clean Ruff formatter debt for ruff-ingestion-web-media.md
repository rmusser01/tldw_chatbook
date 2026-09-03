---
id: TASK-26975
title: Clean Ruff formatter debt for ruff-ingestion-web-media
status: To Do
assignee: []
created_date: '2026-08-31 18:31'
updated_date: '2026-08-31 18:31'
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
- [ ] After rebasing onto current `origin/dev`, reproduce and reconcile every TASK-26000 assigned path; if upstream deleted, renamed, modified, or already formatted it, record that lineage and amend ownership mechanically without silently dropping it or absorbing an unassigned path. <!-- TASK-26000-CONTRACT: rebase-reconcile --><!-- TASK-26000-CONTRACT: drift-reconciliation -->
- [ ] Run Ruff 0.15.22 formatting on only the assigned paths, with no unassigned Python path changed. <!-- TASK-26000-CONTRACT: assigned-paths-only -->
- [ ] Before and after formatting, parse each assigned file on Python 3.12.11 with `ast.parse(..., type_comments=True)`, normalize only `TypeIgnore.lineno`, and require equal `ast.dump(..., include_attributes=False)`. <!-- TASK-26000-CONTRACT: ast-type-comments -->
- [ ] Preserve ordered comment-token text; anchor inline `# noqa`, `# type: ignore`, and single-target Ruff directives to the same deepest AST-node path and significant-token position, preserve standalone file directives between the same adjacent statement paths, and require each `# fmt: off` / `# fmt: on` range to enclose the same ordered AST-node interval. <!-- TASK-26000-CONTRACT: comment-directives -->
- [ ] Ruff lint and `ruff format --check` pass on every touched Python path. <!-- TASK-26000-CONTRACT: ruff-checks -->
- [ ] Implementation Notes record the focused-test rationale and every exact test command/result. <!-- TASK-26000-CONTRACT: focused-tests -->
- [ ] `git diff --check` and `Tests/CI/test_backlog_task_id_uniqueness.py` pass. <!-- TASK-26000-CONTRACT: governance -->
- [ ] The diff contains no hand-written production behavior change. <!-- TASK-26000-CONTRACT: no-handwritten-behavior -->
<!-- AC:END -->
