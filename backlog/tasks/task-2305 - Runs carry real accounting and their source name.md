---
id: TASK-2305
title: Runs carry real accounting and their source name
status: To Do
assignee: []
created_date: '2026-08-04'
labels:
  - watchlists
  - bug
  - uat-2026-08-04
dependencies: []
priority: high
---

## Description (the why)

UAT: after a check that demonstrably harvested ~30 items, the Runs table
showed "Untitled · completed · Found 0 · Processed 0 · Filtered 0 · Errors 0
· Duration -". Both identity and accounting are broken: a history of
"Untitled" rows with zeroed stats is unusable and reads as if checks do
nothing.

UAT findings F32, F33 (high).

## Acceptance Criteria (the what)

- [ ] A run row names its source (and watchlist where applicable).
- [ ] Found/Processed/Filtered/Errors and Duration reflect what the run
      actually did (the ~30-item check shows ~30 found).
- [ ] A regression test asserts run accounting is populated from a real
      check against a stub feed.
