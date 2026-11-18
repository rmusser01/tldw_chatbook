---
id: TASK-41525
title: "A keyless settings save followed by quitting may write the shipped api_key_env_var back into config.toml"
status: To Do
assignee: []
created_date: 2026-11-17 05:27
labels:
- follow-up
- config
- settings
priority: high
references:
- tldw_chatbook/config.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the docs sweep (TASK-41101, round 15) in TASK-33001.7's fix round 1 notes: quitting after a keyless save wrote the shipped `api_key_env_var` back into `config.toml`. Not reproduced or traced; `config.py:1697` and `:7740` read `api_key_env_var`. Writing a default back is a silent config change and can mask the user's own setting.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The path is reproduced in a scratch profile (never the real one) or shown absent, with the notes recording which
- [ ] #2 If reproduced, a keyless save then quit leaves `config.toml` without a shipped `api_key_env_var` the user did not set, with a test red before the fix
<!-- AC:END -->
