---
id: TASK-1722
title: Redact credentials from egress block logs
status: Done
assignee:
  - '@codex'
created_date: '2026-08-01 16:06'
updated_date: '2026-08-01 18:38'
labels:
  - security
  - logging
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
tldw_chatbook/Utils/egress.py logs the full request URL when it blocks a fetch (_blocked, WARNING) and when the check is disabled (DEBUG). Any URL carrying a token in its query string -- presigned CDN/S3 URLs, which per-file artifact source maps are expected to use -- is written verbatim to the log file. Every egress caller is affected, not only model downloads. The parallel TASK-595 branch solved this with a _log_origin() helper that renders scheme://host:port only; port it with its test.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Egress log lines never contain a URL query string, fragment, or userinfo
- [x] #2 A blocked fetch of a URL with a token in the query string produces a log line asserted not to contain the token
- [x] #3 Redaction is applied at every egress log site, not only the block path
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add focused failing tests for blocked and disabled egress log paths using credential-bearing URLs.
2. Add a private origin-only log label helper and apply it at both URL log sites.
3. Regenerate and inspect the reviewed diagnostic inventory; only the existing egress.py call digest may change.
4. Run focused tests, the full egress test module, the diagnostic inventory gate, static checks, and diff review.
5. Check acceptance criteria, add implementation notes, and mark the task Done after verification.

ADR required: no
ADR path: N/A
Reason: Localized security bug fix at an existing logging boundary; no architecture or cross-module contract changes.

Approved design: Docs/superpowers/specs/2026-08-01-egress-log-url-redaction-design.md
Implementation plan: Docs/superpowers/plans/2026-08-01-egress-log-url-redaction.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented origin-only sanitization for both URL-bearing egress log sites. `_log_origin()` now renders only `scheme://host[:port]` (with IPv6 brackets) and returns `<invalid-url>` for malformed or unsupported input; request evaluation and public error behavior are unchanged. Added warning- and debug-path regression tests covering userinfo, path, query-token, and fragment removal. Refreshed only the reviewed `tldw_chatbook/Utils/egress.py` diagnostic digest; owner counts and sink topology are unchanged.

Verification: demonstrated both tests failing before the fix and passing afterward; post-rebase `Tests/Utils/test_egress.py` passed 72/72; the egress diagnostic owner and full sink topology match generated inventory; scoped Ruff (excluding pre-existing E402/F821 findings) and production-file format checks passed; commit/diff whitespace checks passed. The unrestricted full suite completed with 25,865 passed, 171 skipped, and four unrelated UI failures; serial reruns passed two and reproduced two deterministic latest-dev failures in unchanged Evals UI and Library worker-count sentinel paths. Latest dev also contains unrelated stale Voice V2 diagnostic inventory entries, which this task intentionally did not absorb.

ADR required: no. This is a localized logging-boundary security fix. Independent review found no Critical or Important issues; its whitespace finding was fixed before the final squash.

Modified: tldw_chatbook/Utils/egress.py, Tests/Utils/test_egress.py, Docs/security/production-diagnostic-inventory.json, approved design/plan documents, and this task file.
<!-- SECTION:NOTES:END -->
