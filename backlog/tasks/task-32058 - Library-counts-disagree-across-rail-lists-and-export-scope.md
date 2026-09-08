---
id: TASK-32058
title: 'Library counts disagree across rail, lists and export scope'
status: In Progress
assignee: []
created_date: '2026-09-08 18:23'
updated_date: '2026-09-08 19:05'
labels:
  - library
  - export
  - skills
  - ux
  - critique-8
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Export 'Everything' reported '0 conversations' while the rail showed Conversations (6); one run saw 'No skills yet' against 'Skills (2)'; a skill import left the rail at (2) and the list unchanged until the row was re-entered. The solo-operator principle of explicit source authority is undermined when three surfaces count differently. Evidence: critique #8 snapshot `.impeccable/critique/2026-09-08T16-05-41Z__tldw-chatbook-ui-screens-library-screen-py.md` (dev 1c022378cb, 2026-09-08, dual-agent live review at 235x52/100x30/60x24 on a fresh and a seeded profile). Register row 9.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Rail counts, list rows and export-scope counts derive from one enumerator per source
- [ ] #2 Export 'Everything' counts every conversation the rail counts, or states the exclusion rule inline
- [ ] #3 A skill import updates the rail count and the list in place
- [ ] #4 A disagreement that cannot be reconciled renders as a callout with Retry rather than silent zeros
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Failing test: six conversations written by another client -- rail counts 6, export scope 'everything' counts 0.
2. Build get_all_conversation_ids from _conversation_search_filter(scope_type='all') so the enumerator IS the rail's query.
3. Failing test: a skill import updates the rail badge but not the mounted list.
4. Re-request the skills browse from the import receipt, as every other committed skills mutation does.
5. Docs stamps.
<!-- SECTION:PLAN:END -->
