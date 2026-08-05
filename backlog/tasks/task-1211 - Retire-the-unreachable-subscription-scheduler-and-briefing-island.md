---
id: TASK-1211
title: Retire the unreachable subscription scheduler and briefing island
status: Done
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

**Scope correction found during implementation.** The island is larger than the audit measured. The
chain continues past `textual_scheduler_worker`, which is the sole importer of
`Event_Handlers/subscription_events.py`, which is in turn the sole importer of
`Event_Handlers/subscription_ingest_worker.py`. Removing the scheduler orphans both, so both are in
scope here — leaving them would create exactly the silent-orphan shape this cleanup exists to
remove. TASK-813's notes independently reached the same conclusion from the other direction:
`handle_add_subscription` has zero dispatchers and `#subscription-folder-input` is composed nowhere.

That brings the removal to ~7,750 LOC across 13 files.

`Subscriptions/content_processor.py` (590 LOC) also falls out of the graph once
`subscription_ingest_worker` goes, but is deliberately **left in place** and filed separately: it is
the LLM content-analysis component spec #2 needs, and its five entries in the internal prompt
catalog may carry user overrides. Deleting it now to re-add it for the briefing work would be churn,
and the override-migration question deserves its own decision.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The unreachable modules are removed - briefing_generator, briefing_templates, recursive_summarizer, aggregation_engine, rss_feed_generator, export_manager, distribution_manager, textual_scheduler_worker, website_monitor, and Subscriptions/scheduler.py
- [x] #2 UI/Subscription_Modules/subscription_backend_controller.py is removed and dropped from the package __init__ exports
- [x] #3 monitoring_engine.py is retained unchanged - it is live and used by WatchlistCheckHandler
- [x] #4 The eager try-import of briefing_generator in Subscriptions/__init__.py is removed, along with the lazy re-exports of the retired symbols
- [x] #5 Tests/Subscriptions/test_scheduler_deprecation.py is removed or rewritten, since the modules it warns about no longer exist
- [x] #6 App startup imports no retired module - asserted by a test that inspects sys.modules after importing the app
- [x] #7 ADR-019 is updated to record the removal as complete
- [x] #8 Full test suite shows no new failures against the pre-change baseline
- [x] #9 Event_Handlers/subscription_events.py and Event_Handlers/subscription_ingest_worker.py are removed - the scheduler's removal orphans both
- [x] #10 The two prompt specs describing removed modules (subscriptions.recursive_summarizer_system, subscriptions.briefing) are unregistered, since they document code that no longer exists
- [x] #11 content_processor.py is left in place and its newly-orphaned state is filed as a follow-up, not left silent
<!-- AC:END -->

## Implementation Notes

Deleted 8,148 lines across 13 modules; added 274 (a retirement guard test, plus doc and task
updates).

**The scope grew during the work, and the reason is worth keeping.** The audit measured the island
by walking outward from `BriefingGenerator`. Walking the other way — from `textual_scheduler_worker`
down — showed the chain continues: it is the only importer of `subscription_events.py`, which is the
only importer of `subscription_ingest_worker.py`. Deleting the scheduler orphans both. Stopping at
the audit's list would have left two files in exactly the state that made this island expensive to
diagnose: dead, but with importers a grep can point at.

**What was NOT removed, and why.** `monitoring_engine.py` is live — `WatchlistCheckHandler` calls it
— and it sat one import away from the deleted `website_monitor.py`, so that boundary is now pinned
by a test. `content_processor.py` falls out of the graph but is left in place (TASK-1220): it is the
content-analysis component the briefing work needs, and its five prompt specs are a user
customisation surface, so deleting it now would be churn plus an unhandled override migration.

**Two prompt specs had to go with their subjects.** `subscriptions.recursive_summarizer_system` and
`subscriptions.briefing` documented deleted files. A spec whose `used_in` points at a module that no
longer exists is worse than no spec — it offers a customisation surface for code that cannot run.
Their parity tests in `test_subscriptions_prompt_parity.py` went too; a test pinning a spec to a
source literal cannot outlive the source.

**Baseline discipline.** Pre-change: 6 failed / 608 passed. Post-change: 5 failed / 616 passed. All
5 remaining failures were reproduced on a clean `origin/dev` worktree
(`test_eval_db_operations_path`, `test_worker_local_citation_capture`) and are unrelated to
subscriptions. `Tests/Event_Handlers/test_worker_events_contract.py` fails collection on both
(`ImportError: cannot import name 'StreamDone'`) and was excluded from both runs.

**Live-verified.** Booted the app on a scratch profile, opened Watchlists, created a real source
through the form (`Summit Route`, `summitroute.com/blog/feed.xml`) and saw it land in the table as
`rss / active` with the Inspector selecting it. Deleting 8,148 lines changed nothing a user touches.

Two defects were found and filed rather than folded in: TASK-1220 (content_processor's fate) and
TASK-1221 (the Watchlists feature gate requires `markdown`, `schedule` and `feedparser`, none of
which anything imports — this commit orphaned the first, the other two predate it).

Removed: `Subscriptions/{briefing_generator,briefing_templates,recursive_summarizer,
aggregation_engine,rss_feed_generator,export_manager,distribution_manager,textual_scheduler_worker,
website_monitor,scheduler}.py`, `UI/Subscription_Modules/subscription_backend_controller.py`,
`Event_Handlers/{subscription_events,subscription_ingest_worker}.py`,
`Tests/Subscriptions/test_scheduler_deprecation.py`.
Modified: `Subscriptions/__init__.py`, `UI/Subscription_Modules/__init__.py`,
`Internal_Prompts/subscriptions_prompts.py`, `Subscriptions/SUB-Arch.md`,
`backlog/decisions/019-watchlist-scheduler-migration.md`,
`Tests/Internal_Prompts/test_subscriptions_{migration,prompt_parity}.py`.
Added: `Tests/Subscriptions/test_retired_modules_stay_retired.py`.
