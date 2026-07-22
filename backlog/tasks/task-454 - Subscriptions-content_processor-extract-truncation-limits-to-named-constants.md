---
id: TASK-454
title: 'Subscriptions content_processor: extract truncation limits to named constants'
status: To Do
assignee: []
created_date: '2026-07-22 00:22'
labels:
  - internal-prompts
  - cleanup
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Subscriptions/content_processor.py's _build_analysis_prompt repeats the hardcoded truncation limits 5000 (custom prompt, rss/atom/json_feed, url_change, generic) and 3000 (podcast) across multiple branches. Pre-existing behavior carried over unchanged by PR #748 (Internal Prompts P2)'s migration to render_internal_prompt -- not introduced by that migration, and not a correctness bug, but the literals are error-prone to keep in sync if truncation policy is ever tuned. Flagged by qodo-code-review bot on PR #748; deferred as out-of-scope for a prompt-source migration.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The 5000-char and 3000-char truncation limits in _build_analysis_prompt are each defined once as named module or class constants
- [ ] #2 All 5 call sites (custom prompt / rss-atom-json_feed / url_change / podcast / generic) reference the constants instead of inline literals
- [ ] #3 Existing Subscriptions content_processor tests remain green
- [ ] #4 No behavior change: truncation limits and their per-type values are unchanged
<!-- AC:END -->
