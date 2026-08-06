---
id: TASK-1359
title: Support binary file fetching in web_fetch
status: To Do
assignee: []
created_date: '2026-08-05 06:04'
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
- [ ] #1 Content-type allowlist + size caps + temp-dir hygiene,No execution of downloaded content,Clear result shape; tests
<!-- AC:END -->
