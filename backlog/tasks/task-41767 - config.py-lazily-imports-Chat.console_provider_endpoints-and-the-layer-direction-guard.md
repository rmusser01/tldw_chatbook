---
id: TASK-41767
title: "config.py lazily imports Chat.console_provider_endpoints, and the layer-direction guard is red on dev"
status: To Do
created_date: 2026-11-18 16:14
labels:
- follow-up
- architecture-review-2026-10-25
- architecture
- providers
priority: medium
references:
- tldw_chatbook/config.py
- tldw_chatbook/Chat/console_provider_endpoints.py
- Tests/Architecture/test_layer_direction.py
- Docs/Architecture/layer-map.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Tests/Architecture/test_layer_direction.py::test_no_new_upward_import` fails serially at dev 908e9840aa (2026-11-18). It reports one new upward import: `tldw_chatbook/config.py:3803 tldw_chatbook.config -> tldw_chatbook.Chat.console_provider_endpoints (kernel -> domain, lazy)`.

The function-local import of `keep_legacy_custom_2_endpoint` sits where `config.py` merges the user's config over the defaults. It came in ee8e4fa346 (2026-11-16, the TASK-41145 ruling on the legacy Custom OpenAI API 2 endpoint) and reached dev with #3606 on 2026-11-17. `Docs/Architecture/layer-map.md` puts `config.py` in the kernel layer, and the test counts a function-level import too.

It is one of five Tests/Architecture failures on dev's own baseline after the media series merged (47a05413cb, 2026-11-18). So every architecture-review PR subtracts it by hand, and while it is red the guard cannot catch the next upward import.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 No kernel-layer module imports Chat.console_provider_endpoints, and layer_direction_allowlist.tsv gains no row for it
- [ ] #2 The legacy Custom OpenAI API 2 endpoint keeps the behaviour the TASK-41145 ruling set, shown by its tests passing
- [ ] #3 test_no_new_upward_import passes serially on dev
<!-- AC:END -->
