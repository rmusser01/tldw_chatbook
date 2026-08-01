---
id: TASK-1722
title: Redact credentials from egress block logs
status: To Do
assignee: []
created_date: '2026-08-01 16:06'
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
- [ ] #1 Egress log lines never contain a URL query string, fragment, or userinfo
- [ ] #2 A blocked fetch of a URL with a token in the query string produces a log line asserted not to contain the token
- [ ] #3 Redaction is applied at every egress log site, not only the block path
<!-- AC:END -->
