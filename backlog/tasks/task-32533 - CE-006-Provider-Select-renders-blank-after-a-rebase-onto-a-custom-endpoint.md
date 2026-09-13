---
id: TASK-32533
title: 'CE-006: Provider Select renders blank after a rebase onto a custom endpoint'
status: Done
assignee: []
created_date: '2026-09-13 16:35'
updated_date: '2026-09-13 19:02'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Post-#2668 UAT re-run (Docs/superpowers/qa/console-custom-endpoints-uat-2026-09-13, captures 04b-06b): after creating/switching onto a custom-ep entry the modal's rebase completes without crashing, but the Provider Select renders blank and its dropdown lists no options until the modal is reopened. Cosmetic-but-confusing state right at the feature's moment of success.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Provider Select shows the entry display name immediately after post-Create auto-switch,Dropdown lists all providers incl. entries after the switch,Regression test drives the real rebase chain
<!-- AC:END -->

## Implementation Plan

1. Reproduce via the real chain (post `EndpointCreated` to a mounted modal wired with the real controller rebaser, entry registered only after mount) and confirm which control renders blank.
2. Add an option-refresh seam to `ConsoleProviderPicker` (it snapshots options at compose time) that preserves a still-known committed selection.
3. Refresh both provider adapters (visible picker + hidden legacy Select) in `_endpoint_created` before the selection assignment lands.
4. TDD: regression test asserts the rendered label, selected status, dropdown listing, and in-session entry switches; watch it fail first.

## Implementation Notes

- Root cause: the visible provider control is `ConsoleProviderPicker`, whose options/known-id maps are built once in `__init__`. `_endpoint_created` refreshed only the hidden legacy `Select` adapter's options, so `set_provider` on the picker dropped the unknown dashed entry id (`_value = None` -> blank field, "Choose a provider.") and the dropdown's stale option list never contained the new entries until the modal was reopened. The rebase itself landed correctly (hidden Select value, read-only Base URL, model).
- Fix: `ConsoleProviderPicker.set_options()` (new) rebuilds the option tuple/known-id/display-name/group maps via a shared `_index_options` classmethod and re-renders the committed label; `_endpoint_created` now refreshes the picker before assigning `provider_select.value`, so the `Select.Changed` -> `_switch_provider` -> `set_provider` chain resolves the entry label immediately. Ordering: options refresh (both adapters) -> value assignment, the same order compose uses.
- Modified files: `tldw_chatbook/Widgets/Console/console_provider_picker.py` (set_options + _index_options refactor), `tldw_chatbook/Widgets/Console/console_settings_modal.py` (_endpoint_created picker refresh), `Tests/UI/test_console_session_settings.py` (regression test).
- RED: `test_endpoint_created_switch_renders_entry_in_provider_picker` failed with `assert None == 'custom-ep:gpu-box'` on `picker.value` while the hidden Select held the entry id (exactly the UAT's blank-render state). GREEN after the fix; CE-001 neighbors and `Tests/Widgets/test_console_provider_picker.py` stay green.
