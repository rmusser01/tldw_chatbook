---
id: TASK-212
title: Loosen validate_url for URL ingest (long TLDs / IPv6 / IDN)
status: To Do
assignee: []
created_date: '2026-07-12 20:58'
labels:
  - library
  - ingest
  - follow-up
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
URL ingest (task 162) made Utils/input_validation.py:validate_url load-bearing at the Library ingest form. Its regex requires a 2-6 char TLD and has no IPv6/IDN support, so valid URLs like https://blog.example.software/post, https://[::1]/x, and Unicode-domain URLs are rejected at _submit_library_ingest_form even though the extractor would handle them. UX-only (no bad data flows), but user-visible false rejections.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A URL with a long TLD (e.g. .software) is accepted by the Library ingest form,IPv6 and IDN host handling is either supported or the rejection copy is made actionable,Existing validate_url callers are unaffected (no regression)
<!-- AC:END -->
