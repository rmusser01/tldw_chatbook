---
id: TASK-1722
title: Redact credentials from egress block logs
status: Done
assignee:
  - '@codex'
created_date: '2026-08-01 16:06'
updated_date: '2026-08-03 20:43'
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
Merged as PR #1288 on 2026-08-03. Twist: the log-site redaction had ALREADY landed via PR #1173 (adopted branch B's _log_origin) after this task was filed -- the claim-time in-flight check only looks for OPEN PRs, so reproduce-before-implementing is the real guard. This PR closed the leak that remained: str()/repr() of EgressBlockedError and EgressFetchError embedded the full URL, and callers (Article_Extractor_Lib.py:344, acquisition.py, monitoring_engine.py) log str(exc) verbatim. Messages now render via _log_origin; the .url ATTRIBUTE keeps the full URL for programmatic consumers, pinned by test. Added the marker-absence test coverage the redaction never had (both log sites, both exceptions, never-raises over garbage input). Residual noted for a future sweep: Article_Extractor_Lib.py:344 logs its own raw url variable caller-side. TASK-1723 (trusted_private_origins) remains the other branch-B egress item, low priority.
<!-- SECTION:NOTES:END -->
