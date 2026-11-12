---
id: TASK-37422
title: >-
  Console: an unparseable [chat_defaults] streaming value drops the shipped
  default and falls back to the provider's own Off
status: To Do
assignee: []
created_date: '2026-10-29 10:14'
labels:
  - console
  - streaming
  - settings
  - p3
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Since TASK-33620.10 the shipped [chat_defaults] streaming = true makes every Console provider stream by default. The user's config file is merged over the shipped template, so a hand-edited value that Console cannot read, such as streaming = 0, replaces the shipped true; Console then ignores it and falls through to [api_settings.<provider>] streaming, which is false for OpenAI, Anthropic and nine other providers in the template and in older files. Such a user silently loses streaming for those providers while Settings > Console Behavior shows Streaming 'On' with the source word 'built-in' and the help 'used unless the provider sets its own'. This is the only way the built-in Streaming state can still appear, so CONSOLE_BUILT_IN_FALLBACK_FIELDS and the settings.md Global fallback defaults copy still describe it. Owner call: should an unreadable value fall back to the shipped default (and the built-in Streaming state and its copy go), or keep today's behaviour with clearer copy? Found by the independent review of TASK-33620.10 (2026-10-29).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The owner decides whether an unreadable [chat_defaults] streaming value falls back to the shipped default or to the provider's own value, and the decision is recorded in this task
- [ ] #2 A new Console chat and Settings > Console Behavior agree on the Streaming value for a config file holding streaming = 0, for OpenAI and for llama.cpp; a test pins both and fails before the change
- [ ] #3 If the shipped default wins, Streaming no longer appears in the built-in fallback fields and the User Guide settings page no longer describes a built-in Streaming value
<!-- AC:END -->
