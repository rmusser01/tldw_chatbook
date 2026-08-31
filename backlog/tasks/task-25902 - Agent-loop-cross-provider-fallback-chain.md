---
id: TASK-25902
title: 'Agent loop: cross-provider fallback chain'
status: To Do
assignee: []
created_date: '2026-08-31 15:08'
updated_date: '2026-08-31 15:11'
labels:
  - agents
  - reliability
dependencies:
  - TASK-25901
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
When a provider is exhausted or down, chatbook has no way to continue on another. Verified on origin/dev: the two fallback_ hits in the codebase are config-key fallbacks for model-name lookup (Chat/console_session_settings.py:725-727) and a use-Console-default in personas, neither of which is error-driven switching; a 429 retries the same key and then raises ChatRateLimitError (Chat/Chat_Functions.py:1180-1184). Builds on the retry classification from task-25901 - fallback is what happens when retry is exhausted or the error is credit/quota terminal.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 An ordered fallback provider list is configurable and consulted when the primary provider exhausts retries or returns a credit/quota-terminal error
- [ ] #2 Switching providers mid-run is visible in the transcript and the run log, never silent
- [ ] #3 The fallback attempt reuses the existing per-provider readiness check, so an unconfigured fallback is skipped rather than attempted and failed
- [ ] #4 Model-specific request shaping (tool schema, thinking, caching) is re-resolved for the fallback provider rather than carried over
- [ ] #5 With no fallback list configured, behavior is byte-identical to today
<!-- AC:END -->
