---
id: TASK-40302
title: "Console readiness for a removed or skipped custom endpoint says it is not available in Console yet and shows a mangled id, while the send says to restore the entry"
status: To Do
created_date: 2026-11-12 19:36
labels:
- follow-up
- console
- providers
- copy
priority: low
references:
- tldw_chatbook/Chat/console_session_settings.py
- tldw_chatbook/Chat/console_provider_support.py
- tldw_chatbook/Chat/console_provider_gateway.py
- Docs/User_Guide/console.md
dependencies:
- TASK-39151
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 13 of the 2026-10-23 burn-down (security review r13s-rev1 of TASK-39151). Checked by reading the code at the round-13 stack top 8b633d0101, 2026-11-12; line numbers are that tree's.

**What the user sees.** After TASK-39151, a schemeless llama.cpp custom entry is skipped. A chat bound to it shows "Provider blocked: 'custom_ep:<slug>' is not available in Console yet." with the label "Unknown". That copy blames a missing feature and shows an internal id; the send path gives the real fix, "this custom endpoint is no longer configured. Restore its entry..." (gateway `:4436`, documented in `Docs/User_Guide/console.md:802-804`).

**Why.** When a `custom-ep:<slug>` entry was skipped or removed, `entry_for` returns None (`console_session_settings.py:1855`). `_canonical_chat_provider_id(settings.provider)` then runs without `app_config`, and `provider_config_key` mangles the id to `custom_ep:<slug>`, which misses the `custom-ep:` check in `console_provider_support.resolve_console_provider_identity`. The provider resolves as unknown, so readiness takes `provider_unsupported` (`:2017-2019`) and its copy (`:2078`). The skip reason is only logged.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Readiness for a `custom-ep:` id with no configured entry reads the send's 'no longer configured, restore its entry' copy and never shows the mangled id; a test is red before the fix
- [ ] #2 A provider Console truly does not support keeps the 'not available in Console yet' copy
<!-- AC:END -->
