---
id: TASK-36006
title: Manage optional keyword indexes in Settings and qualify release
status: In Progress
assignee:
  - '@codex'
created_date: '2026-10-21 14:32'
updated_date: '2026-10-21 21:57'
labels: []
dependencies:
  - TASK-36005
documentation:
  - Docs/superpowers/plans/2026-10-20-workspace-project-keyword-index.md
  - backlog/decisions/240-workspace-project-context-and-folder-bound-index.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let users explicitly manage folder-bound keyword indexing and understand coverage from Settings and Console.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Settings exposes truthful Off/Building/Partial/Ready/Paused/Interrupted/Error controls and processing choices using existing UI tokens; Console shows status and a Settings route.
- [ ] #2 Compose/status projections perform no I/O, stale actions have no effects, and late binding/job/epoch callbacks cannot overwrite another view.
- [ ] #3 Isolated app journeys cover build/query/pause/resume/restart/refresh/detach/Clear with preserved Library choices, no semantic controls, and a real production query pin that holds maintenance/restore replacement until physical retirement; isolated restore/rollback invalidates old scope and key.
- [ ] #4 Final service/store/tool qualification covers at least20,000 text files across20,000 nested/sibling folders plus a separate20,000-file single directory, physical retirement, permission races, missing embeddings and blocked sockets, with measured limits and updated architecture/user documentation.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read approved K5 brief, source-verified UI seams, design language/component patterns and Impeccable product context. Write mounted regressions before UI/service changes.
2. Add exact selected-binding management card and narrow admitted immutable service policy projection; no constructor/compose/event-loop I/O. Preserve scope/job CAS and mounted binding/epoch callbacks, explicit build/resume, truthful busy/partial/errors, existing tokens and navigation. Console projects status and routes to Workspaces Settings after safe modal dismissal.
3. Run targeted UI/policy GREEN and covering files. Qualify isolated app build/query/pause/resume/restart/refresh/detach/Clear, all saved Library choices, real production query storage pin through actual native retirement and isolated encrypted restore/rollback invalidation.
4. Qualify final service/store/tool path on at least20,000 text files across20,000 nested/sibling folders plus a separate20,000-file single directory. Retain stable private corpora, real marker response, machine/corpus/throughput/baseline+peak-extraRSS/whole-binding disk and sidecars/cancellation retirement/candidate p50p95/live-validation separately; missing embeddings and blocked sockets must not affect keyword operation. No nominal scanner-only substitution.
5. Update architecture/user docs and retained QA evidence, run targeted new project-index files and relevant UI governance/CSS checks, scoped lint/format/diff and self-review, then independent spec/quality gate.
ADR required: yes, existing accepted ADR.
ADR path: backlog/decisions/240-workspace-project-context-and-folder-bound-index.md.
Reason: direct implementation of accepted folder-bound storage/tool/management/revocation contracts, alongside ADR101/102/126/150/161/174; no new schema, dependency, semantic implementation, Library policy mutation or navigation API.
<!-- SECTION:PLAN:END -->
