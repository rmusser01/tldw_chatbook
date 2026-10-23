---
id: TASK-36423
title: Offer per-folder semantic consent and qualify optional search
status: To Do
assignee:
  - '@codex'
created_date: '2026-10-23 03:28'
labels: []
dependencies:
  - TASK-36422
references:
  - backlog/decisions/241-workspace-project-context-and-folder-bound-index.md
  - Docs/superpowers/plans/2026-10-20-workspace-project-semantic-index.md
type: feature
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make optional semantic project discovery understandable and explicitly approved per folder while keeping keyword operation independent.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Mounted Settings provides per-binding model/destination choice, disclosure approval, separate keyword/semantic readiness and coverage, and preserved dirty/stale-action fences without model work during compose or status.
- [ ] #2 Cancelled external disclosure causes zero provider calls; destination changes pause old semantic work before approval and stale semantic actions after Clear cannot affect a newer binding state.
- [ ] #3 Saved Library policies/scopes and keyword readiness stay unchanged through semantic enable/disable, missing dependencies, provider failure, restart and chat-provider changes.
- [ ] #4 A selected real local embedding backend is qualified in the real app for paraphrase discovery, refresh, failed rebuild, restart, disable and Clear/detach; no implicit model download or unapproved external project-text transmission occurs.
- [ ] #5 Actual bounded scale, storage peak, memory, query latency and cancellation evidence plus all limitations, docs, ADR linkage and independent reviews are retained; deterministic vectors alone never certify semantic quality.
<!-- AC:END -->
