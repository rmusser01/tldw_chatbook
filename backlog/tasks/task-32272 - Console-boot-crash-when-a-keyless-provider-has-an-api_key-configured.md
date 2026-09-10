---
id: TASK-32272
title: Console boot crash when a keyless provider has an api_key configured
status: To Do
assignee: []
created_date: '2026-09-10 19:09'
labels:
  - console
  - provider-readiness
  - bug
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Console readiness validator raises 'Console credential source conflicts with its facet' when the credential facet is not_required but a stored key exists, and the app exits before rendering anything. Reproduced with [api_settings.custom] api_key on the keyless custom provider; introduced with the Console conversation-settings redesign (939dee8dc2, 2026-09-04). Keyless OpenAI-compatible servers commonly still accept a bearer token, so this configuration is legitimate. Source: live approval-card / MCP-permissions UX review 2026-09-10 against dev 3315241674 (snapshot .impeccable/critique/2026-09-10T17-31-53Z__hatbook-widgets-chat-widgets-chat-approval-card-py.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The app starts and Console reaches Ready when a keyless provider has an api_key configured.
- [ ] #2 Readiness reports the credential facet and its source consistently for keyless providers with and without a stored key, without raising.
- [ ] #3 A regression test covers a keyless provider with a stored key.
<!-- AC:END -->
