---
id: TASK-36900
title: Keep Watchlists navigation responsive during article body reads
status: Done
created_date: 2026-10-25 19:33
labels:
- textual
- watchlists
- responsiveness
- book-audit
priority: medium
documentation:
- Docs/superpowers/qa/2026-10-25-textual-book-after-pr3247.md
assignee:
- '@codex'
updated_date: 2026-10-25 21:05
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let readers change articles and open enabled Watchlists sections while an earlier article body is still loading, without letting obsolete detail work change the current view.

### Evidence and bounded scope

Verified after PR #3247 merged, on dev `79da83a8dca98eab6ca854a7505b8abd264eb959`. With the actual private file-backed SubscriptionsDB.get_item_content boundary held before calling its original implementation, a later real ItemSelected event requesting C cannot begin its detail read and the reader stays on A. An enabled Sources button likewise cannot change section until release. During each approximately 0.6-second controlled hold, an independent App callback runs but a screen callback waits. After release, the requested article/section completes. Ordinary Pilot Enter/Down/Down and Sources navigation work. The database read already runs off the event loop: this is same-screen queue serialization, not evidence of an event-loop freeze, natural SQLite latency, data loss or stale final publication.

Affected source: tldw_chatbook/UI/Screens/watchlists_collections_screen.py:12177-12298,5659-5663; tldw_chatbook/UI/Watchlists_Modules/article_list.py:904-932,950-952,999-1011; tldw_chatbook/Subscriptions/local_watchlists_service.py:949-975.

Book basis: Chapter 7 PDF145-146/158 (printed136-137/149), independent queues and awaited handler serialization; Chapter 17 PDF335 (printed326), grouped workers and publication checks. Apply these through the existing destination owner rather than changing DB/service contracts.

### Existing ownership and decisions

TASK-15464 deliberately hydrates content before one reader publication while keeping body text out of list projections. Preserve that result. TASK-22450 owns stable snapshot/scope publication; TASK-16221 paging/scrolling; TASK-34425 structural sharing. TASK-33283 AC4 owns in-place pane updates, not responsiveness while an async body read is held. TASK-34967 owns revisit staleness caching. TASK-19559/19870 own the enumerated worker-group/refresh sites. Their complete criteria were checked. No matching open PR or remote-only owner was found.

Related tasks: [TASK-15464](./task-15464%20-%20Watchlists-items-feed---narrow-projection-and-indexed-effective-date-ordering.md), [TASK-22450](./task-22450%20-%20Stabilize-Watchlists-Feed-Items-snapshots-and-atomic-scope-commits.md), [TASK-16221](./task-16221%20-%20Make-Watchlists-Read-list-and-Content-independently-scrollable-with-explicit-pagination.md), [TASK-34425](./task-34425%20-%20Structural-sharing-for-ReaderItemSnapshot-pages.md), [TASK-33283](./task-33283%20-%20PERF-24-Watchlists-and-Schedules---incremental-panes-DB-off-the-event-loop.md), [TASK-34967](./task-34967%20-%20Gate-Watchlists-per-visit-section-loads-on-staleness.md), [TASK-19559](./task-19559%20-%20exclusive%3DTrue%20without%20group%3D%20makes%20every%20worker%20on%20a%20node%20cancel%20its%20siblings.md), [TASK-19870](./task-19870%20-%20Mutation-workers-refresh-inline-instead-of-dispatching-into-the-loader-group.md).

ADR required: no.
ADR paths: [ADR-042](../decisions/042-watchlists-reader-first-ia.md), [ADR-033](../decisions/033-application-session-state-ownership.md), [ADR-011](../decisions/011-chatbook-workbench-ui-system.md).
Reason: A contained message-handler/worker correction retaining existing screen ownership, service boundaries and stable snapshot policy.

[Post-merge audit and evidence](../../Docs/superpowers/qa/2026-10-25-textual-book-after-pr3247.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A later enabled article selection can reach its current reader while an older article body read remains held, without waiting for release of the obsolete read.
- [x] #2 Enabled Watchlists section navigation completes while an earlier article body read remains held.
- [x] #3 Late success or failure from superseded, departed-view, changed-backend/query/page or detached detail work cannot replace the current reader or mark an obsolete selection read through stale publication authority.
- [x] #4 Narrow list projections, shared row identity and one hydrated reader publication remain intact; mounted ordinary, overlapping, out-of-order, failure and teardown regressions pass with no background exceptions and all controlled holds released.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read the linked audit and existing ADRs; follow Docs/superpowers/plans/2026-10-25-quiz-start-and-reader-lifetimes.md.
2. Add mounted regressions and retain RED evidence before implementing the bounded owner-lifetime repair.
3. Run complete affected tests and targeted neighboring contracts, scoped lint/format, actual child report inspection and fresh Superpowers whole-branch review.
4. Register cases in CI, document evidence, mark criteria Done after required gates; create PR against dev and address CI/review feedback.
ADR required: no
ADR path: backlog/decisions/011-chatbook-workbench-ui-system.md; backlog/decisions/033-application-session-state-ownership.md; backlog/decisions/042-watchlists-reader-first-ia.md
Reason: bounded direct correction under existing ownership; no new service/storage contract.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Moved article hydration into a named/exclusive screen-owned wc_item_detail worker so ItemSelected releases the screen queue. Selection/snapshot/page/backend/section/attachment authority guards hydration, errors, publication and mark-read after arrival; narrow list projections and one shared-row publication remain intact. j/k/space now follow the pending list selection. Added 18 mounted production-screen/private SQLite regressions and adjusted two old pagination completion waits. Complete final Reader file: 18 passed, all 18 private child reports independently clean. Fresh Superpowers review found one Important keyboard-anchor issue; all three Pilot regressions failed before its two-line repair and passed in the full final file. Scoped lint/format and CI contracts152/152 pass; 26 new UI selectors collect, prior867 slots unchanged, floor893. Three existing E721s and local legacy setup/baseline failures are explicitly qualified in Docs/superpowers/qa/2026-10-25-textual-quiz-and-reader-lifetimes.md. Updated Docs/Architecture/subscriptions-watchlists.md. ADR required: no; direct implementation of existing ADR-011/033/042. Plan: Docs/superpowers/plans/2026-10-25-quiz-start-and-reader-lifetimes.md. No service/storage/dependency change, full-suite or physical-terminal claim.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
