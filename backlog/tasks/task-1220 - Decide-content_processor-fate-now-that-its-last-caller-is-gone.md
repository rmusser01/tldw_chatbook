---
id: TASK-1220
title: Decide content_processor's fate now that its last caller is gone
status: To Do
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
- [ ] #1 A decision is recorded: content_processor is either wired to a live caller or removed
- [ ] #2 If removed, its five prompt specs are unregistered and any persisted user overrides for those ids are handled explicitly rather than silently orphaned
- [ ] #3 If retained, a production call path exists and is covered by a test that fails when the path is broken
- [ ] #4 The eager re-export in Subscriptions/__init__.py no longer places an uncalled module in sys.modules at startup
- [ ] #5 If retained, the un-awaited chat_api_call in _analyze_content is fixed
<!-- AC:END -->
