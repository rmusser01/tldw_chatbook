---
id: TASK-1220
title: Decide content_processor's fate now that its last caller is gone
status: Done
assignee: []
created_date: '2026-07-28 00:20'
labels:
  - subscriptions
  - cleanup
dependencies:
  - TASK-1211
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Subscriptions/content_processor.py` (590 LOC) lost its only caller when TASK-1211 removed
`Event_Handlers/subscription_ingest_worker.py`. It is now reachable only through the eager re-export
in `Subscriptions/__init__.py`, which puts it in `sys.modules` at startup without anything ever
calling it — the same shape that let `BriefingGenerator` pass for live code for months.

It was left in place deliberately rather than deleted with the rest of the island, for two reasons
that need resolving rather than assuming:

1. It is the LLM content-analysis component (`_analyze_content`, `_build_analysis_prompt`,
   `KeywordExtractor`, `ContentSummarizer`). The briefing/podcast work in Watchlists spec #2 needs
   exactly this. Deleting it now and re-adding it shortly is churn.
2. Its five entries in the internal prompt catalog (`subscriptions.analysis_system`,
   `feed_analysis`, `url_change_analysis`, `podcast_analysis`, `generic_analysis`) are a
   user-facing customisation surface. If a user has overridden one, deleting the specs orphans
   their override — a migration question, not just a deletion.

Either wire it to something real or retire it with its prompt specs. What it must not do is sit in
the ambiguous state that made the original island so expensive to diagnose.

Note the pre-existing quirk documented in `Tests/Internal_Prompts/test_subscriptions_migration.py`:
`_analyze_content` is `async` but calls `chat_api_call` **without awaiting it**. If this module is
revived, that is a real bug to fix; if it is retired, it goes away.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A decision is recorded: content_processor is either wired to a live caller or removed
- [x] #2 If removed, its five prompt specs are unregistered and any persisted user overrides for those ids are handled explicitly rather than silently orphaned
- [x] #3 If retained, a production call path exists and is covered by a test that fails when the path is broken
- [x] #4 The eager re-export in Subscriptions/__init__.py no longer places an uncalled module in sys.modules at startup
- [x] #5 If retained, the un-awaited chat_api_call in _analyze_content is fixed (moot: retired, and the premise was wrong -- see notes)
<!-- AC:END -->

## Implementation Notes

**Decision: retire.** 1,115 LOC removed -- `content_processor.py` (590),
`Internal_Prompts/subscriptions_prompts.py` (151), and the two test files that covered them (374).

**What settled it was a user-facing finding the task did not know about.** `Settings > Internal
Prompts` lists specs grouped by subsystem and badges them as customizable. The five `subscriptions.*`
specs were in that list. Their only consumer was `ContentProcessor`, whose only caller went with the
retired ingest pipeline in TASK-1211 -- so a user could open Settings, carefully edit the
feed-analysis prompt, and change nothing at all. That is a false affordance, and a false affordance
is worse than an absent one. Keeping the module to preserve "customization" that cannot take effect
would have been preserving the appearance of a feature.

The competing argument -- that spec #2's briefing work needs LLM content analysis -- is real but is
the same reasoning that preserved the 8,148-line island through TASK-1211: keeping unreachable code
because something might want it later. Git history is a perfectly good archive, and spec #2 will
restore what it needs *with a caller*, deliberately.

**Correction to this task's own description.** It claimed `_analyze_content` is `async` but calls
`chat_api_call` "without awaiting it", and called that "a real bug to fix". `chat_api_call` is
declared `def`, not `async def` (`Chat/Chat_Functions.py:698`), so the call was correct. The real
smell was a blocking synchronous LLM call inside an async method, which would have stalled the event
loop. Moot now, but the description should not stand uncorrected.

**Blast radius handled.** `Tests/test_remaining_diagnostic_sentinel_matrix.py` used
`content_processor` as the representative module for the `subscriptions` logging domain; it now
names `monitoring_engine`, which is the module `WatchlistCheckHandler` actually calls. The
retirement guard from TASK-1211 gained both removed modules plus a new assertion that no
`subscriptions.*` prompt spec is registered, so Settings cannot start advertising inert prompts
again.

Any `[internal_prompts.subscriptions]` overrides in a user config become inert rather than
erroring -- unregistered ids are simply never looked up.

Removed: `Subscriptions/content_processor.py`, `Internal_Prompts/subscriptions_prompts.py`,
`Tests/Internal_Prompts/test_subscriptions_migration.py`,
`Tests/Internal_Prompts/test_subscriptions_prompt_parity.py`.
Modified: `Subscriptions/__init__.py`, `Internal_Prompts/__init__.py`,
`Tests/test_remaining_diagnostic_sentinel_matrix.py`,
`Tests/Subscriptions/test_retired_modules_stay_retired.py`.
