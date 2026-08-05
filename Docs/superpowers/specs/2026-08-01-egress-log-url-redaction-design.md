# Egress Log URL Redaction Design

**Status:** Approved
**Task:** TASK-1722

## Problem

The egress policy currently interpolates full request URLs into its blocked-request warning and disabled-policy debug log. Presigned URLs and other credential-bearing URLs can therefore write userinfo, paths, query tokens, or fragments to persistent logs.

## Decision

Add a private `_log_origin(url)` helper to `tldw_chatbook/Utils/egress.py`. It parses HTTP(S) URLs and returns only `scheme://host[:port]`, preserving IPv6 brackets. Invalid or unsupported URLs return the constant `<invalid-url>`.

Use the helper at both egress log sites. Request evaluation, exception messages, configuration, and public APIs remain unchanged.

## Verification

Focused tests will capture both warning and debug messages and prove that userinfo, path, query parameters, fragments, and their secret values are absent while the non-sensitive origin remains useful.

## Architecture Decision Record

- ADR required: no
- ADR path: N/A
- Reason: This is a localized security bug fix at an existing logging boundary and changes no storage, runtime, provider, or cross-module contract.
