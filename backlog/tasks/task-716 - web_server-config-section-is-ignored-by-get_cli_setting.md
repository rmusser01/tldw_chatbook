---
id: TASK-716
title: '[web_server] config section is ignored by get_cli_setting'
status: To Do
assignee: []
created_date: '2026-07-26 08:00'
labels:
  - config
  - bug
  - web-server
dependencies:
  - TASK-545
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Web_Server/serve.py:331` calls `get_cli_setting("web_server", default={})` — a bare section name with no `key` argument (only a keyword `default`). This hits the same bug class as TASK-547/TASK-658/TASK-715: `get_cli_setting`'s fallback branch returns the supplied default unconditionally when it isn't given a real key, so the `[web_server]` TOML section is never actually read. Confirmed against a real config where `[web_server]` exists and is populated — the values are discarded.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] `Web_Server/serve.py`'s read of `[web_server]` uses a call shape that actually returns the configured section, not an unconditional default
- [ ] Setting values under `[web_server]` in `config.toml` is observably honored by the web server
- [ ] Unit test confirms a configured `[web_server]` value is used, and that the hardcoded default only applies when the section/key is genuinely absent
<!-- AC:END -->
