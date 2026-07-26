---
id: TASK-703
title: Guard the get_cli_setting bare-section silent-default bug class
status: To Do
assignee: []
created_date: '2026-07-26 08:00'
labels:
  - config
  - bug
  - tech-debt
dependencies:
  - TASK-545
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`get_cli_setting(section, key=None, default=None)` silently returns `default` whenever it's called with a bare section name and either a non-string second positional argument (mistaken for `key`) or no key at all. This one shape of defect has now been found six separate times across five subsystems: TASK-547 (`Tools/tool_executor.py`, `[tools]` — deleted, not fixed, as System A was removed in TASK-545 P3), TASK-658 (`local_file_ingestion.py`, `[database]`), TASK-699 (`splash_screen.py` and `settings_splash_screen_viewer.py`, `[splash_screen]`, two instances), TASK-700 (`serve.py`, `[web_server]`), and TASK-701 (`TTS/backends/openai.py`, three instances with no key at all). Every one of these was a silent, non-crashing no-op: the caller always got the hardcoded default and never knew the real config section was never consulted.

`save_setting_to_cli_config(section, None, value)` has the same shape of defect in the opposite direction — it raises `KeyError: 'None'` rather than silently discarding data, but the root cause is the same: a caller reaching for a "write/read this whole section" call shape that the API doesn't actually support. TASK-545 P3 removed `save_setting_to_cli_config`'s only caller that hit this, but nothing in the function itself prevents the same mistake being reintroduced.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] `get_cli_setting` fails loudly (raises, warns, or otherwise cannot be silently misread) when called with a bare section name and a non-string second argument, instead of returning that argument as the default
- [ ] `get_cli_setting` fails loudly (or is given a documented, correct call shape) when called with a section name and no key at all, instead of unconditionally returning `default`
- [ ] `save_setting_to_cli_config` fails loudly or is documented against being called with `key=None` intending a section-level write, instead of raising an unhelpful `KeyError: 'None'`
- [ ] A lint, test, or repo-wide grep is added (or run and its result recorded) to confirm no further live instances of this call shape remain, and to catch new ones going forward
<!-- AC:END -->
