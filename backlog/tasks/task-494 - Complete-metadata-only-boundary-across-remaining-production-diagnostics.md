---
id: TASK-494
title: Complete metadata-only boundary across remaining production diagnostics
status: To Do
assignee: []
created_date: '2026-07-23 14:34'
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
- [ ] #1 Every repository-wide inventory entry not owned by TASK-492 is remediated or has a reviewed non-persistent/excluded classification.
- [ ] #2 Normal and debug persistent diagnostics in the remaining domains exclude search queries, conversation/note/media/file content, fetched bodies, config values, credential fragments, request/response values, and arbitrary object representations.
- [ ] #3 Error diagnostics use bounded categories, status, and exception types without raw exception, HTTP body, parser input, or result text.
- [ ] #4 Approved operation identity, counts, lengths, status, duration, retry, and posture metadata remain available.
- [ ] #5 A checked inventory/source guard detects new or changed production logging owners and persistent-sink topology so unclassified call sites fail review.
- [ ] #6 Third-party warning/error records proven to contain a sentinel are filtered or disabled for the persistent file sink without suppressing safe UI diagnostics.
- [ ] #7 Parameterized sentinel tests cover remaining RAG/search, ingestion, media/database, Notes/sync, subscription/web, and UI/application-orchestration domains through normal, debug, and error paths and the real rotating file sink.
<!-- AC:END -->
