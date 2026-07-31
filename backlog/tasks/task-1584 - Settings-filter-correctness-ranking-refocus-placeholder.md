---
id: task-1584
title: 'Settings filter correctness: ranking, refocus, placeholder'
status: To Do
assignee: []
created_date: '2026-07-31'
labels:
  - settings
  - ux
  - rescore-p2
dependencies: []
priority: medium
---

## Description (the why)

Three filter defects found live during the critique rescore capture:
(1) typing "rag" and pressing Enter opens Storage, not Library/RAG — the
substring match on "sto-RAG-e" ties on rank tier and wins on list index;
(2) refocusing the filter (via "/" or click) retains stale text with no
select-all, so a second "/" press types a literal slash into the focused
input and stale terms silently poison the next search;
(3) the placeholder says "Filter settings (/)" but results resolve to
categories (task-1564 made owned keys searchable, which helps, but the
Enter target is always a category).

## Acceptance Criteria (the what)

- [ ] A whole-word/prefix rank tier beats bare substring matches: "rag"
      opens Library/RAG, not Storage
- [ ] Refocusing the filter selects the existing text (typing replaces it)
      or clears it, so repeat searches never concatenate
- [ ] Placeholder copy matches behavior (e.g. "Filter categories (/)")
- [ ] Existing rank tiers (id/title, description, owned keys) keep their
      relative order
