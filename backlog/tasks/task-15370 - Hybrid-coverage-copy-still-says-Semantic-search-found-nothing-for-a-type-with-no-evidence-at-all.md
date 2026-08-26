---
id: TASK-15370
title: >-
  Hybrid coverage copy still says 'Semantic search found nothing' for a type
  with no evidence at all
status: To Do
assignee: []
created_date: '2026-08-11 16:02'
labels:
  - library
  - rag
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-14752 (landed inside TASK-15020's B1b) split the Library's scope-coverage diagnostic three ways -- covered / keyword_only / uncovered -- so a source whose rows came only from the engine's FTS leg now reads 'Keyword matches only from: X.' instead of being described as having produced nothing.

The residual: under a HYBRID profile, a selected type with NO rows at all still renders the original sentence, 'Semantic search found nothing from: X.' That is literally true and it was the whole truth before hybrid (only the semantic leg ran), but under hybrid the keyword leg ran and found nothing for that type either -- so the sentence names one of two legs and invites the reader to conclude the other might have helped.

The B1b review sharpened the case: now that the two sentences can appear together ('Semantic search found nothing from: Conversations. Keyword matches only from: Notes.'), the contrast actively implies that the first type failed only the SEMANTIC leg -- exactly the inference the second sentence exists to correct for the other type. The composition makes the wrong inference easier to draw than it was when the sentence stood alone.

Deliberately out of TASK-14752's scope (its AC covered only the keyword-sourced case) and left un-widened there rather than scope-crept. Filed with a number rather than left as report prose because 14752's own description records what happens otherwise: this exact pattern -- a known copy gap flagged in code and never given an id -- drifted across two arcs before it was filed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Under a hybrid profile, a selected source type with no rows from EITHER leg is described in a way that does not attribute the absence to the semantic leg alone
- [ ] #2 The semantic and plain profiles' existing sentence is unchanged (they genuinely ran only the semantic leg, so the current wording is exactly right there)
- [ ] #3 A test pins the hybrid no-evidence-at-all case alongside the existing keyword_only and uncovered pins in Tests/Library/test_library_rag_state.py
<!-- AC:END -->
