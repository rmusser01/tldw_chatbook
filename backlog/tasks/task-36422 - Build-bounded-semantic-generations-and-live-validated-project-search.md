---
id: TASK-36422
title: Build bounded semantic generations and live-validated project search
status: To Do
assignee:
  - '@codex'
created_date: '2026-10-23 03:27'
updated_date: '2026-10-23 05:28'
labels: []
dependencies:
  - TASK-36421
references:
  - backlog/decisions/241-workspace-project-context-and-folder-bound-index.md
  - Docs/superpowers/plans/2026-10-20-workspace-project-semantic-index.md
type: feature
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Provide optional semantic project discovery with truthful readiness and keyword fallback under the existing folder authority.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Build pages are at most32chunks and1MiB, total retained/queued source is at most4MiB, and one checked keyword manifest produces immutable generation membership with atomic compatible Ready publication.
- [ ] #2 Interrupted/failed/quota-refused rebuild preserves the prior compatible Ready; stale scope, semantic revision, consent, destination or source cannot publish.
- [ ] #3 Semantic ranking applies exact run/current exclusions and policy before top-k; RRF60 merges bounded candidates through the existing100candidate/10hit/16KiB live identity/hash/permission disclosure boundary.
- [ ] #4 Semantic unavailable failures remain distinct from successful zero hits and always leave keyword-only search usable.
- [ ] #5 Durable job ownership, repeated cancellation, Clear/detach and shutdown retain real native borrowers through physical retirement; data cannot cross identical-text bindings.
- [ ] #6 Keyword lookup remains usable while actual embedding/provider work is held outside finite SQL units; binding-wide keyword/semantic job admission stays sequential and total retained/queued source across jobs stays at most4MiB.
<!-- AC:END -->
