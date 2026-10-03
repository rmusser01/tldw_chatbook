---
id: TASK-34211
title: Build the in-repo merge queue for dev
status: In Progress
assignee:
  - '@claude'
created_date: '2026-10-03 14:06'
labels:
  - ci-throughput
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Under strict protection only one PR can merge into dev per CI cycle, yet every armed PR is re-synced and re-tested after each merge: 47% of required runs over 80 merged PRs (2026-09-21..28) were re-sync churn, about 740 runner-minutes a day. The owner asked (2026-10-03) for CI that pushes one PR at a time. Spec: Docs/superpowers/specs/2026-10-03-merge-queue-design.md. Mechanics verified by spike PR #2985.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Armed PRs merge into dev one at a time in arming order; PRs behind the front are never rebased, dispatched or commented on
- [ ] #2 The front PR is rebased with the built-in token, and its required check and other PR workflows run on the rebased head
- [ ] #3 Conflicting, twice-failed, blocked or stuck PRs are evicted with a comment; a single CI failure is retried once
- [ ] #4 The queue never enables auto-merge, merges or pushes (guard test)
- [ ] #5 MERGE_QUEUE unset/off has no effect; dry logs decisions with no side effects; on acts
- [ ] #6 CLAUDE.md and AGENTS.md carry the same mode-dependent merge rules
- [ ] #7 Re-sync runs per merged PR are measurable before and after with a committed script
<!-- AC:END -->
