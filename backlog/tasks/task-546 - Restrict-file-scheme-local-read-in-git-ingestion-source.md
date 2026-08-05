---
id: TASK-546
title: Restrict file:// scheme local read in git ingestion source
status: To Do
assignee: []
created_date: '2026-07-24 12:00'
labels: [security, media]
dependencies: [TASK-330]
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`_local_git_repository_path` and `_sync_git_repository_source_items` resolve a `file://` or no-scheme `repo_url` pointing at a real local directory and read it directly (skipping the clone), using `pathlib.Path(url).glob()` and similar direct filesystem operations. This is a secondary local-file-read vector not covered by the clone-time transport allowlist hardening in TASK-330. An ingestion source with `repo_url = "/etc"` or `repo_url = "file:///etc"` would read system directories without validation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] Local-path ingestion sources (`file://` or no-scheme URLs) are restricted by validating the resolved path against a configurable allowlist or sandbox root
- [ ] Paths outside the allowlist are rejected at ingestion time (fail-closed)
- [ ] Or explicit opt-in is required for local-path sources via config flag
- [ ] Unit test covers rejection of out-of-bounds paths (e.g. /etc, /tmp)
<!-- AC:END -->
