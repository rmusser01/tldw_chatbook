---
id: TASK-486
title: Redact custom-named credential query params in provider-test endpoint display
status: To Do
assignee: []
created_date: '2026-07-22 19:27'
labels:
  - settings
  - security
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-up from PR #781 review. _mask_url_userinfo masks endpoint userinfo passwords and redact_secret_text catches standard-named query params (api_key/token/secret/password), but a custom-named credential query param (e.g. ?mycred=SEKRET) in a provider endpoint still prints unredacted in the Test evidence. Same name-based-redaction gap class as the (now-fixed) env-var/userinfo cases, for query strings.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A custom-named credential query param in the provider endpoint is not printed verbatim in the provider-Test evidence
<!-- AC:END -->
