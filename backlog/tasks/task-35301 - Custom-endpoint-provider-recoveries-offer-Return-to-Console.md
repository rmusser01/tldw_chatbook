---
id: TASK-35301
title: 'Custom-endpoint provider recoveries offer Return to Console'
status: To Do
assignee: []
created_date: '2026-10-17 09:11'
labels:
- model-config-redesign
- follow-up
- console
- settings
dependencies:
- TASK-33008
references:
- tldw_chatbook/UI/Console_Modules/model_switcher.py
- tldw_chatbook/UI/Navigation/conversation_settings_navigation.py
- tldw_chatbook/UI/Settings_Modules/provider_return.py
- tldw_chatbook/Chat/console_onboarding_state.py
- Tests/UI/test_console_provider_return.py
- Docs/User_Guide/console.md
- Docs/User_Guide/settings.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Owner decision of 2026-10-17 on PR #3144 (Phase 8, merged as 111f8e48fc): TASK-33008's AC#3 was rescoped to the providers a return can name, and the rest of it is this task.

At origin/dev 111f8e48fc a Console chat on a named custom endpoint (`custom-ep:<slug>`) still opens Settings one way. `open_provider_setup` (UI/Console_Modules/model_switcher.py) builds a `ProviderSettingsNavigationTarget`, which raises ValueError for such an id; the `except ValueError` posts the plain NavigateToScreen context with no staged return (controller ruling R37, 2026-10-10). That covers both routes users take: the Get started card's Set up provider and the endpoint's NEEDS SETUP row in Switch model. The user fixes the endpoint in Custom endpoints and then has to find the chat by hand, which is the dead end Phase 8 closed for every built-in provider.

Today's behaviour is pinned on purpose: Tests/UI/test_console_provider_return.py's custom-endpoint case of `test_a_route_whose_named_field_is_disabled_arrives_where_it_can_act` and both cases of `test_a_custom_endpoint_row_with_no_field_to_focus_arrives_where_it_can_act` assert that no return shows. The copy says so too: `CONSOLE_SETUP_CARD_SUBTITLE` (Chat/console_onboarding_state.py) reads "except for custom endpoints", and settings.md reads "Settings remembers no way back for a custom endpoint". This task changes those pins and that copy.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 From a Console chat on a named custom endpoint, the Get started card's Set up provider and the endpoint's NEEDS SETUP row in Switch model open Settings with Return to Console offered, and one Enter there lands on the same chat; pinned by a test
- [ ] #2 Returning from a custom endpoint's NEEDS SETUP row reopens Switch model on that endpoint, as it does for a built-in provider
- [ ] #3 Arrival focus still lands on a control that acts (Edit this endpoint in Custom endpoints; Provider when the entry is gone), and the return stays reachable from there
- [ ] #4 The Get started card's subtitle, console.md and settings.md no longer exclude custom endpoints from the return
<!-- AC:END -->
