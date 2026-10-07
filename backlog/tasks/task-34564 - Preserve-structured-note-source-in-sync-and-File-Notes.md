---
id: TASK-34564
title: Preserve structured note source in sync and File Notes
status: In Progress
created_date: 2026-10-07 02:12
assignee:
- '@codex'
labels:
- notes
- source-preservation
documentation:
- Docs/superpowers/plans/2026-09-27-structured-document-foundations.md
updated_date: 2026-10-07 02:16
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep accepted note source unchanged through sync and admit complete YAML, JSON and JSONL source in File Notes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Accepted note title and content remain unchanged, or are explicitly rejected without trimming, truncation or entity encoding.
- [x] #2 Structured file extensions expose their complete source while Markdown frontmatter behavior remains.
- [x] #3 Real file round trips preserve representation facts and retain conflict and excerpt safeguards.
- [ ] #4 Targeted regressions and touched-file static checks pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/194-structured-note-language-and-local-editor-validation.md
Reason: implement approved exact source transport under existing note/file authority.
1. Read F1 in Docs/superpowers/plans/2026-09-27-structured-document-foundations.md.
2. Add failing adapter and real file regressions.
3. Replace destructive acceptance with validation and admit structured files without frontmatter extraction.
4. Run targeted checks, independent review and scoped commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented ADR-194 F1: reject over-limit and forbidden controls without transforming accepted title/content; title/content limits match server ADR-031 (255/5,000,000 characters). Structured extensions preserve complete bodies; real round-trip tests retain BOM, CRLF, final newline, external conflict and excerpt safeguards. Targeted tests: 74 passed. Scoped static comparison: no introduced Ruff diagnostics or formatting debt; inherited diagnostics remain. Independent review pending; status retained In Progress.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
