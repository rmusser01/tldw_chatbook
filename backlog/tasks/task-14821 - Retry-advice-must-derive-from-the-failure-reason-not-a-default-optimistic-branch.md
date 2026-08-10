---
id: TASK-14821
title: >-
  Retry advice must derive from the failure reason, not a default optimistic branch
status: In Progress
assignee:
  - '@claude'
created_date: '2026-08-10 21:00'
labels:
  - library
  - ingest
  - ux
  - copy
priority: high
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
P1 of the 2026-08-10 re-critique. The `Show details` expansion contradicts the row it belongs to and recommends an action that cannot work.

The retry advisory's ELSE branch ("A retry can succeed if the failure was transient — a busy file or a network hiccup…") fires for every failure whose category isn't `parse_error` and whose message doesn't match `_MISSING_DEPENDENCY_RE` (which matches `No module named 'x'` / `x is not installed` / `pip install x`). A missing-OCR failure's own message reads "…install an OCR backend (docling, tesseract, easyocr, paddleocr, or docext)" — which that regex does NOT match — so the optimistic fallback fires. This makes it the COMMON case, not an edge case, and turns the Retry button into a trap for a deterministic failure.

Observed live on two different rows: `✗ failed · diagram.png · No text was found in diagram.png… install an OCR backend (…)` → Show details → `Category: write error` + the transient advisory. Identical on `report_draft.docx`.

Two further defects in the same expansion: `write error` is a mis-category (nothing was written — the failure was extraction producing no content), and `Category: <token>` is a raw internal token shown to users. The expansion also prints the raw ffmpeg banner TWICE (~40 lines including Homebrew Cellar paths and 7 libav version lines) under both `Details:` and `Underlying:`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 A failure's retry advice is derived from the same reason that produced its row line — a deterministic tooling/extraction failure is never described as possibly transient
- [ ] #2 The optimistic advisory appears only for genuinely retryable causes; the unknown case is silent rather than encouraging
- [ ] #3 A no-content extraction failure is categorised as such, not as a write error, and the category is user-readable rather than a raw token
- [ ] #4 Underlying tool output appears once, not duplicated between Details and Underlying
<!-- AC:END -->
