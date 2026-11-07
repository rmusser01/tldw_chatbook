---
id: TASK-760
title: 'quick_ingest() resolves a different media DB than the app opens'
status: Done
assignee: []
created_date: '2026-07-26 17:00'
labels:
  - config
  - bug
  - ingestion
dependencies:
  - TASK-658
---

## Description

`quick_ingest()` hand-rolls its media-DB path resolution instead of calling
`config.get_media_db_path()`, the accessor the app itself uses. When no custom
`[database] media_db_path` is configured -- the default for every user -- the
two resolve to different files, so anything ingested through `quick_ingest`
lands in a database the app never opens.

Measured on a default config:

- app (`get_media_db_path()`): `~/.local/share/tldw_cli/default_user/tldw_chatbook_media_v2.db`
- `quick_ingest()` fallback: `~/.local/share/tldw_cli/tldw_cli_media_v2.db`

They differ in both the per-user subdirectory and the filename stem.

This predates TASK-658, which only fixed the *custom*-path case (the config
value was being discarded entirely). It is currently latent: `quick_ingest`
is exported from `Local_Ingestion/__init__.py` but has no in-tree callers, so
nothing hits the divergence today. It becomes a silent data-loss bug the
moment something calls it.

## Acceptance Criteria

- [x] `quick_ingest()` opens the same database file the running app opens, for both a default config and a custom `[database] media_db_path`
- [x] The per-user data directory is honored rather than a hardcoded `~/.local/share/tldw_cli` literal
- [x] A test asserts the path `quick_ingest` uses equals `get_media_db_path()` under a default config
- [x] The existing TASK-658 regression tests still pass, or are updated with a documented reason if the fallback expectation legitimately changes


## Closeout audit 2026-10-11

Board-hygiene flip (cohort A, Jun-Jul 2026 status sweep). **Verdict: DONE** — evidence verified against origin/dev (a3c47c6c72).

quick_ingest() now calls config.get_media_db_path() (Local_Ingestion/local_file_ingestion.py:2629-2631) — the same accessor the app uses — instead of hand-rolling the media-DB path.
