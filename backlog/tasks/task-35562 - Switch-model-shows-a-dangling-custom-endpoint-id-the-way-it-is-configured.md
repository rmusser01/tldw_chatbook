---
id: TASK-35562
title: Switch model shows a dangling custom endpoint id the way it is configured
status: To Do
created_date: 2026-10-18 17:42
dependencies: []
labels:
- model-config-redesign
- follow-up
- console
priority: low
references:
- tldw_chatbook/Chat/provider_catalog.py
- tldw_chatbook/Widgets/Console/console_model_popover.py
- tldw_chatbook/Chat/console_session_settings.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in Phase 9's live capture phase (2026-10-18) and confirmed in triage to predate Phase 9: it reproduces at the phase base 111f8e48fc.

When the chat's provider is a custom endpoint id with no registry entry (for example `custom-ep:gone`), Switch model's title and its typed-id row show `custom_ep:gone`, with an underscore. The status chip shows `custom-ep:gone`, as configured.

The cause: `_canonical_chat_provider_id` returns the dangling id through `provider_config_key`, which turns '-' into '_'. Then `provider_display_name`'s fallback for an unmapped id returns that key unchanged. The underscore form exists only as a config-table spelling. Tests/Chat/test_provider_catalog.py already says a custom endpoint id must never be shown mangled.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every Switch model surface (title, rows, typed-id row) shows a dangling custom endpoint id in its configured dashed form, matching the status chip
- [ ] #2 A dangling id still reads as not ready (unsupported), as it does today
- [ ] #3 Tests cover the underscore input for the display name and for the switcher title
<!-- AC:END -->
