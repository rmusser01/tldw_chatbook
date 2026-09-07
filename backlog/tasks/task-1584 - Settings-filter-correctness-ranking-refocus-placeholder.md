---
id: TASK-1584
title: 'Settings filter correctness: ranking, refocus, placeholder'
status: Done
assignee:
  - '@claude'
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

- [x] A whole-word/prefix rank tier beats bare substring matches: "rag"
      opens Library/RAG, not Storage
- [x] Refocusing the filter selects the existing text (typing replaces it)
      or clears it, so repeat searches never concatenate
- [x] Placeholder copy matches behavior (e.g. "Filter categories (/)")
- [x] Existing rank tiers (id/title, description, owned keys) keep their
      relative order

## Implementation Plan (the how)

1. RED tests: "rag" ranks Library/RAG first with Storage still findable;
   tier order pinned; "/" on the focused filter inserts no slash and the
   next keystroke replaces; placeholder assertion.
2. Word-boundary sub-tier in `_category_search_rank`; select-all on
   refocus; an Input subclass intercepting "/" (the screen's on_key never
   sees printable keys while an Input is focused); placeholder copy.

## Implementation Notes

`_category_search_rank` tiers rescaled 0-3: word-boundary primary match
(regex `(?<![a-z0-9])` before the escaped query) > substring primary >
description/status > owned keys — all consumers only checked `is not
None`, and the old id/title > description > owned ordering is preserved
and pinned by a test. Refocus: `_focus_category_search` now calls
`Input.select_all()`; the literal-slash trap needed
`SettingsCategorySearchInput(Input)` overriding `_on_key` because printable
keys are consumed by the focused Input before the screen's on_key —
every other Input keeps literal "/" typing (endpoint URLs). Placeholder:
"Filter categories (/)". TDD RED-first. Files:
`tldw_chatbook/UI/Screens/settings_screen.py`,
`Tests/UI/test_settings_configuration_hub.py`.
