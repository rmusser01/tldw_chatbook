---
id: TASK-590
title: Address guarded_fetch buffer + retryable status classification
status: Done
assignee:
  - '@zcode'
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
- [x] Either implement early-status inspection (peek at response before buffering body) OR document the residual as a known limitation with its impact and workarounds
- [x] Retryable errors with oversized bodies are correctly classified or clearly noted as a future enhancement
<!-- AC:END -->

## Implementation Plan

1. Verify the mechanism on current dev: where the guarded_fetch_* helpers raise on oversize, and what (if anything) carries the status to the caller.
2. Take the document branch of the AC: extend the egress module docstring's established "Non-goals / documented residual risk" section with the limitation, its impact, current mitigations, and the future enhancement path.

ADR required: no
ADR path: N/A
Reason: Documentation-only; the residual was an explicit non-goal of the TASK-328 design.

## Implementation Notes

Chose the document branch. Verified the mechanism first: each guarded_fetch_* helper streams the body and raises ``EgressFetchError("response exceeds N bytes")`` mid-body with the response status discarded, so an over-cap 408/429/5xx is indistinguishable from a permanent oversize failure at the call site.

Documented in ``tldw_chatbook/Utils/egress.py``'s module docstring, appended to its existing "Non-goals (documented residual risk)" list so all residual risks read in one place: limitation (buffer-then-classify), impact (retryable statuses surface as size failures), mitigations (treat EgressFetchError as retryable where a retry budget exists -- Subscriptions.watchlist_failure already classifies it into its retryable connection_failure bucket, pinned by test_watchlist_failure.py; or raise the caller's max_bytes), and the future enhancement (early status peek, attach status to the error).

Verification: docstring import-asserted; Tests/Utils/test_egress.py 106 passed (docstring-only change); ruff zero delta (6 pre-existing fixables at HEAD, unchanged).

Modified: tldw_chatbook/Utils/egress.py (module docstring only).