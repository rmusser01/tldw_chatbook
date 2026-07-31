---
id: task-1641
title: 'Settings: field-level search'
status: Done
assignee:
  - '@claude'
created_date: '2026-07-31'
labels:
  - settings
  - ux
  - critique-r4
dependencies: []
priority: high
---

## Description (the why)

Critique round 4 P1: '/' matched category names, descriptions and owned config keys, so typing a setting's visible name ('threshold', 'density') found nothing. On a 22-category screen whose pitch is dense-but-findable, the operator had to know the taxonomy to change one known setting.

## Acceptance Criteria (the what)

- [x] Typing a field label surfaces its category (rank tier between description and owned keys)
- [x] The echo line names the field: 'Console Behavior › Threshold (chars)'
- [x] Enter opens the category AND focuses the matched field, firing its guidance
- [x] Existing rank-tier order is preserved

## Implementation Notes

FIELD_SEARCH_INDEX built at import from the real widget ids (Console Behavior, Appearance, Providers, Storage via STORAGE_FIELD_LABELS, RAG via _RAG_FIELD_GROUP_BY_ID); _top_field_match helper feeds the rank, the echo, and the post-open focus (call_after_refresh). Live-verified: 'threshold' → Console Behavior with the cursor in the field and 'Smallest pasted chunk…' guidance showing.
