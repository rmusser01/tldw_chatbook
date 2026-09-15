---
id: TASK-32620
title: >-
  Library Notes: Preview shows two titles for a frontmatter-titled note and runs
  a callout's type into its body
status: To Do
assignee: []
created_date: '2026-09-15 06:42'
labels:
  - library
  - notes
  - critique-4
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #4 (dev 77eb2601a6), assessor A P3 and A's docs table, persona Jordan, Obsidian workflow. Residual of task-32551 (PR #2683), which covered the body-starts-with-a-heading case.

What happened. A note whose title came from Obsidian frontmatter renders in Preview as 'Q3 planning — library review' left-aligned (the note title) directly above a CENTRED 'Library review' (the body's own heading) -- two title-shaped lines, the second one centred like a page banner (A cap 25). The de-duplication rule only fires when the heading matches the title exactly, which a frontmatter-titled Obsidian note never does.

Same capture, cosmetic: a warning callout renders as one run of text with its type and body joined and no colon or break, where the guide says the type becomes a rendered label (A cap 25; B saw the same shape at cap 27).

Cause PROVEN by capture.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A rendered heading is left-aligned, not centred
- [ ] #2 When a note's first heading differs from its title and the title came from frontmatter, the heading renders as an ordinary heading rather than a second title
- [ ] #3 A callout's type and body are visually separated as the guide describes
- [ ] #4 Both cases are pinned so the frontmatter path is covered alongside the filename path
<!-- AC:END -->
