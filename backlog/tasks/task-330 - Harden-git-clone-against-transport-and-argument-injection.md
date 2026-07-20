---
id: TASK-330
title: Harden git clone against transport and argument injection
status: To Do
assignee: []
created_date: '2026-07-20 18:45'
labels: [security]
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`_clone_git_repository` (`Media/local_media_reading_service.py:3706-3716`, reached from `_sync_git_repository_source_items`) builds a `git clone` argv from an ingestion-source `repo_url` with no scheme validation and no `--` separator. `repo_url = "ext::sh -c '<cmd>'"` triggers git's ext transport and executes an arbitrary shell command (RCE); a `repo_url`/`ref` beginning with `-` (e.g. `--upload-pack=...`) is parsed as a git option (argument injection). Config is user-driven but can arrive via imported/shared library definitions, so it should be hardened. (No `shell=True`/`os.system` exists elsewhere — this is the one real shell vector found.)
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 `repo_url` scheme is allowlisted (https, optionally ssh/git@); `ext::` and other non-allowlisted transports are rejected
- [ ] #2 Leading-dash `repo_url`/`ref` values are rejected, and a `--` separator precedes the URL in the argv
- [ ] #3 The clone runs with `GIT_PROTOCOL_FROM_USER=0` / `GIT_ALLOW_PROTOCOL` restricting protocols
- [ ] #4 Tests cover an `ext::` payload and a leading-dash payload being rejected
<!-- AC:END -->
