---
id: TASK-1358
title: Support PDF fetch in web_fetch without permanent ingestion
status: Done
assignee:
  - '@claude'
created_date: '2026-08-05 06:04'
updated_date: '2026-08-06'
labels:
  - web-tools
dependencies:
  - TASK-1354
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
web_fetch v1 rejects PDFs and routes users to media ingestion. Users need one-off PDF reads (papers, manuals) that do not write to the media DB.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 PDFs fetched via egress-guarded path and size-capped,Text extracted ephemerally (no media DB writes),Result truncated like HTML; tests with fixture PDFs
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Spec: `Docs/superpowers/specs/2026-08-06-web-crawl-pdf-fetch-design.md` §1; plan: `Docs/superpowers/plans/2026-08-06-web-crawl-pdf-fetch.md` (tasks 1–2, 6). PDF support inside `web_fetch` (owner ruling: same tool, not a sibling — the model never knows content type in advance), with a mid-stream cap decision so mislabeled PDFs work in one request.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Shipped on `feat/web-crawl-pdf-fetch` inside `tldw_chatbook/Tools/web_tool_impls.py` — no new tool, no MCP change; the agent-tool description now advertises PDF support.

- Detection: content-type `application/pdf` OR `%PDF-` magic sniff (≥5 buffered bytes; sniff wins over the declared type — mislabels as `text/html`/`octet-stream`/absent all work). The read cap is raised to `PDF_MAX_BYTES` (20 MB) MID-STREAM on detection: one request, no re-fetch.
- Over 20 MB → `[too-large]` refusal (a byte-truncated PDF is unparseable, so the ceiling refuses rather than truncates); truncation applies to the EXTRACTED TEXT at the caller's `max_bytes` with a marker recording pages processed, satisfying the "truncated like HTML" AC. The extraction page loop early-stops once over the cap.
- Ephemeral: `pymupdf.open(stream=...)`, nothing on disk, no media-DB import anywhere in the module. Structured refusals: `[missing-dep]` (pip install tldw_chatbook[pdf]), `[pdf-error]` encrypted/damaged, `[empty-content]` textless → points at media ingestion with OCR.
- Rode along: fetch cache keyed `(url, max_bytes)` (v1 quirk: a small-cap fetch poisoned later full-cap calls) + 256-entry bound with earliest-expiry eviction.
- Review trail: module-scope `importorskip` would have silently skipped 27 v1 SSRF/redirect tests in a `.[dev]` env (fixed round 1); the whole-branch review then caught sniffed-PDF cache poisoning via web_crawl's warm-write (fixed in the final wave). Deferred minors: TASK-2620.
- Tests: fixture PDFs generated in-test with pymupdf (valid/encrypted/textless/oversized/mislabeled/multi-chunk dribble), `Tests/Tools/test_web_tool_impls.py`; PDF tests skip individually when pymupdf is absent, v1 tests always run.
<!-- SECTION:NOTES:END -->
