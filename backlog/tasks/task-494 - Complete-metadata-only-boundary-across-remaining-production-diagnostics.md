---
id: TASK-494
title: Complete metadata-only boundary across remaining production diagnostics
status: Done
assignee:
  - '@codex'
created_date: '2026-07-23 14:34'
updated_date: '2026-07-24 15:32'
labels:
  - security
  - privacy
  - logging
dependencies:
  - TASK-492
references:
  - backlog/decisions/022-local-private-data-boundary.md
documentation:
  - Docs/superpowers/specs/2026-07-23-local-privacy-containment-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Complete ADR-022 across every production diagnostic domain not covered by the provider/tool remediation so persistent application logs cannot retain user, model, credential, file-content, or response-body values.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Every repository-wide inventory entry not owned by TASK-492 is remediated or has a reviewed non-persistent/excluded classification.
- [x] #2 Normal and debug persistent diagnostics in the remaining domains exclude search queries, conversation/note/media/file content, fetched bodies, config values, credential fragments, request/response values, and arbitrary object representations.
- [x] #3 Error diagnostics use bounded categories, status, and exception types without raw exception, HTTP body, parser input, or result text.
- [x] #4 Approved operation identity, counts, lengths, status, duration, retry, and posture metadata remain available.
- [x] #5 A checked inventory/source guard detects new or changed production logging owners and persistent-sink topology so unclassified call sites fail review.
- [x] #6 Third-party warning/error records proven to contain a sentinel are filtered or disabled for the persistent file sink without suppressing safe UI diagnostics.
- [x] #7 Parameterized sentinel tests cover remaining RAG/search, ingestion, media/database, Notes/sync, subscription/web, and UI/application-orchestration domains through normal, debug, and error paths and the real rotating file sink.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/022-local-private-data-boundary.md
Reason: TASK-494 completes the accepted metadata-only persistence boundary without changing ADR-022.

1. Close direct Loguru file-sink and third-party record bypasses without suppressing UI diagnostics.
2. Exercise representative remaining domains through standard/Loguru normal, debug, and error sentinel paths against real rotation.
3. Review and regenerate the checked owner/topology inventory, then reconcile all acceptance criteria with verification evidence.

Detailed plan: Docs/superpowers/plans/2026-07-24-complete-metadata-only-diagnostics.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Completed the remaining ADR-022 diagnostic boundary. The sole persistent application file sink now requires Chatbook-owned metadata records and rejects all third-party records while preserving UI/terminal diagnostics. Legacy Metrics direct Loguru file sinks are disabled. The checked inventory assigns 422 owner files and records 1,008 TASK-492 calls, 7,143 TASK-494 calls, and five sink/topology sites with per-file digests.

Verification: 52 focused sentinel/inventory tests passed under /private/tmp/tldw-task492-494-sentinel-F0irEw/pytest; recursive scanning found zero sentinel/key fragments and all generated application/MCP files were 0600. Remaining RAG, ingestion, media/database, Notes/sync, subscription/web, DB, UI, logging, and utility tests produced 1,563 passes and 27 skips, with the macOS /var alias fixture corrected and its isolated rerun passing. Ruff, compileall, the inventory guard, and git diff --check passed.

Plan: Docs/superpowers/plans/2026-07-24-complete-metadata-only-diagnostics.md
ADR: backlog/decisions/022-local-private-data-boundary.md
<!-- SECTION:NOTES:END -->
