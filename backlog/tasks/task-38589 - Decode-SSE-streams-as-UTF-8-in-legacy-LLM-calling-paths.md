---
id: TASK-38589
title: Decode SSE streams as UTF-8 in legacy LLM calling paths
status: Done
created_date: 2026-11-02 14:27
labels:
- bug
- llm
priority: medium
updated_date: 2026-11-03 02:56
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Four legacy streaming paths use iter_lines with decode_unicode so endpoints that omit charset utf-8 get decoded as ISO-8859-1, silently corrupting every non-ASCII character from local llama.cpp or vLLM style servers. See bug-sweep-2026-11-02-dev.md section 1.7.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Legacy openai Responses Gemini and local engine paths decode bytes as UTF-8 explicitly
- [x] #2 An endpoint omitting charset produces correct non-ASCII output
- [x] #3 Regression test covers a charset-less content type
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
All four legacy SSE sites iterate bytes and decode UTF-8 explicitly with per-site line contracts preserved; charsetless-header tests were born red. Commit 05e5ea12ec.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
