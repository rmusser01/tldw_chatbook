---
id: TASK-1211
title: Retire the unreachable subscription scheduler and briefing island
status: To Do
assignee: []
created_date: '2026-07-27 22:15'
labels:
  - watchlists
  - cleanup
dependencies:
  - TASK-1210
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
About 5,100 LOC of subscription scheduling and briefing code is unreachable in the shipped app and
duplicates a live implementation. A runtime import trace (`import tldw_chatbook.app`, then inspect
`sys.modules`) confirms `textual_scheduler_worker` and `UI/Subscription_Modules` are never loaded,
and that `BriefingGenerator` — whose sole construction site is `textual_scheduler_worker.py:101` —
is never instantiated.

The modules form a closed import cycle (`textual_scheduler_worker` <-> `subscription_backend_controller`)
with no external entry point, which is why grep-based checks have repeatedly read them as live.
`SubscriptionBackendController` drives a `SubscriptionWindow` class that no longer exists anywhere
in the codebase.

Watchlist checking is now done by `Scheduling/scheduler/handlers/watchlist_check_handler.py`
delegating to `Subscriptions/monitoring_engine.py`. ADR-019 already accepted this migration and
deferred removal of the old scheduler to a follow-up; this is that follow-up.

Retiring this before building briefing generation avoids growing a second implementation of the
same feature beside a dead one.

Audit: `Docs/superpowers/research/2026-07-27-briefing-subsystem-revive-or-retire.md`
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The unreachable modules are removed - briefing_generator, briefing_templates, recursive_summarizer, aggregation_engine, rss_feed_generator, export_manager, distribution_manager, textual_scheduler_worker, website_monitor, and Subscriptions/scheduler.py
- [ ] #2 UI/Subscription_Modules/subscription_backend_controller.py is removed and dropped from the package __init__ exports
- [ ] #3 monitoring_engine.py is retained unchanged - it is live and used by WatchlistCheckHandler
- [ ] #4 The eager try-import of briefing_generator in Subscriptions/__init__.py is removed, along with the lazy re-exports of the retired symbols
- [ ] #5 Tests/Subscriptions/test_scheduler_deprecation.py is removed or rewritten, since the modules it warns about no longer exist
- [ ] #6 App startup imports no retired module - asserted by a test that inspects sys.modules after importing the app
- [ ] #7 ADR-019 is updated to record the removal as complete
- [ ] #8 Full test suite shows no new failures against the pre-change baseline
<!-- AC:END -->
