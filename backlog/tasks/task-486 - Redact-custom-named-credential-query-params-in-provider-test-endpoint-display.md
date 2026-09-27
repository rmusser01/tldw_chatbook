---
id: TASK-486
title: Redact custom-named credential query params in provider-test endpoint display
status: Done
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
- [x] #1 A custom-named credential query param in the provider endpoint is not printed verbatim in the provider-Test evidence
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Absorbed and closed by TASK-33002.2. The Provider Test result's Endpoint row now renders through `Chat/console_provider_endpoints.safe_endpoint_display`, which drops user info, the query string and the fragment. A custom-named credential query parameter such as `?mycred=SEKRET` therefore never reaches the rows or the toast. The positional `_mask_url_userinfo` helper is deleted, so there is one redaction path.

Tests in `Tests/UI/test_settings_provider_test_draft.py`:
- `test_findings_never_print_a_custom_named_credential_query_param` (unit: rows and toast, with no evidence, a failed probe and a reached probe)
- `test_custom_named_credential_query_param_never_reaches_rows_or_toast` (mounted: the probe still receives the typed URL)

Live capture: `qa/model-config-p2-2026-09-27/task-2/test-custom-credential-query-211x44.txt`.
<!-- SECTION:NOTES:END -->
