---
id: TASK-717
title: TTS OpenAI backend reads nothing from config via get_cli_setting
status: To Do
assignee: []
created_date: '2026-07-26 08:00'
labels:
  - config
  - bug
  - tts
dependencies:
  - TASK-545
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`TTS/backends/openai.py:96,105,114` call `get_cli_setting("openai_api")`, `get_cli_setting("API")`, and `get_cli_setting("app_tts")` — each with no `key` argument at all. Unlike the sibling instances of this bug class (TASK-547/658/699/700), there isn't even a stray non-string second argument being mistaken for `default`; with no key supplied, `get_cli_setting` cannot resolve anything from the section and every one of these three calls always returns `None`. The OpenAI TTS backend therefore reads none of its intended configuration through these call sites.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] Each of the three `get_cli_setting` call sites in `TTS/backends/openai.py` is fixed to read the specific key it actually needs from the relevant section
- [ ] Configured values (e.g. API key or app-level TTS setting) are observably used by the OpenAI TTS backend instead of always falling through to `None`
- [ ] Unit test confirms a configured value at each fixed call site is used, and that the fallback only applies when the section/key is genuinely absent
<!-- AC:END -->
