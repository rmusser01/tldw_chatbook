---
id: TASK-1359
title: Support binary file fetching in web_fetch
status: In Progress
assignee:
  - '@claude'
created_date: '2026-08-05 06:04'
updated_date: '2026-08-07 21:50'
labels:
  - web-tools
dependencies:
  - TASK-1354
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
v1 rejects non-HTML content types. Add safe handling for common binaries (images, audio, archives): bounded temp download, metadata + safe preview/extraction where feasible, never execute downloaded content.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Content-type allowlist + size caps + zero on-disk persistence (no temp files; guarded by the existing no-persistence-import static test — amended 2026-08-07 from "temp-dir hygiene", superseded by the in-memory ruling in Docs/superpowers/specs/2026-08-07-web-fetch-binary-design.md),No execution of downloaded content,Clear result shape; tests
<!-- AC:END -->
