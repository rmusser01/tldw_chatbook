---
id: TASK-510
title: Address guarded_fetch buffer + retryable status classification
status: To Do
assignee: []
created_date: '2026-07-23 12:00'
labels: [web, followup]
dependencies: [task-328]
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`guarded_fetch_httpx` (and aiohttp/requests helpers) buffer the full body up to `max_bytes` before the caller can inspect status. An oversized (>cap) response with a RETRYABLE status (408/429/5xx) is misclassified as a permanent/oversize failure rather than a transient error worth retrying. Consider a small early-status peek or documenting this as a known residual behavior and its mitigation strategy.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] Either implement early-status inspection (peek at response before buffering body) OR document the residual as a known limitation with its impact and workarounds
- [ ] Retryable errors with oversized bodies are correctly classified or clearly noted as a future enhancement
<!-- AC:END -->
