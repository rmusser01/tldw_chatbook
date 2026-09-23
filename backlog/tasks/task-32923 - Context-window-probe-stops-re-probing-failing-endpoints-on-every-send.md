---
id: TASK-32923
title: Context-window probe stops re-probing failing endpoints on every send
status: Done
assignee:
- '@claude'
created_date: 2026-09-23 15:43
labels:
- performance
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Sends await a serving-metadata probe (1 s timeout). Failures were cached for only 5 s, so an unreachable or slow local server added up to 1 s before nearly every message went out. OpenRouter's model list (measured 748,851 bytes) always exceeds the 256 KB cap, so its probe could only ever download 256 KB and fail -- on every send after 5 s.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A failed probe is remembered as long as a successful one (60 s)
- [x] #2 OpenRouter targets make no metadata request and still resolve their window from the model catalog
- [x] #3 Sends still wait for the first probe, so a local server's real n_ctx keeps governing the request budget
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. METADATA_FAILURE_TTL = METADATA_SUCCESS_TTL
2. Drop openrouter from the probe families
3. Tests with patched clock and a request-counting transport
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
`METADATA_FAILURE_TTL` now equals `METADATA_SUCCESS_TTL` (60 s), and `openrouter` is dropped from the probe families: its model list measured 748,851 bytes against the 256 KB cap, so the probe could only fail; `resolve_context_window` already resolves OpenRouter via the catalog/upstream split. The send path still awaits the first probe by design -- a local server's real `n_ctx` must govern the budget.

Files: `Chat/console_context_window.py`, `Tests/Chat/test_console_context_window.py`.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
