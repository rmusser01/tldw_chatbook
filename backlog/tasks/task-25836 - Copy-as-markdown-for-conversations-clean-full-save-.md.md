---
id: TASK-25836
title: 'Copy-as-markdown for conversations (clean, full, save .md)'
status: In Progress
assignee:
  - '@Robert'
created_date: '2026-08-31 14:55'
updated_date: '2026-08-31 14:55'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add a Copy-as page to the conversation action menu: Clean markdown and Full transcript copy to the clipboard, Save .md writes a file. Works for persisted chats (DB read, paginated) and open native sessions (store messages_for_session), covering workspace and normal conversations alike.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The conversation menu root offers Copy as (page opener with disclosure glyph) opening Clean markdown / Full transcript / Save .md… / Back,Clean copies role headings plus verbatim user/assistant content with tool activity, thinking, and system noise skipped and images as placeholders,Full also renders tool rows and thinking as collapsed details blocks with citations under their message,Copy works for persisted rows (DB source) and open native session rows (store source) in both the grouped browser and the Workspaces tree,Save .md validates the path and writes the slug-named file,Empty chats disable the copy entries with a stated reason
<!-- AC:END -->

## Renumbering

TASK-19601 owner rule: this task originally took id 25714, which collided
with the older arrival ``task-25714 - Console-Context-rail-still-overflows-
below-140x40.md`` already on dev. Renumbered to TASK-25836 with this
provenance section; frontmatter, doc comments, and test references were
updated with it. Earlier commit messages on this branch still name
TASK-25714.

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Pure renderer Chat/console_conversation_markdown.py + golden tests (clean/full).\n2. Menu model: copy page + action ids + MenuPage literal + page_from_action + suffix list + ROOT_PAGE_HEIGHT 8 + updated pins.\n3. Targets carry native_session_id (browser opener attr + tree native: strip); screen routing for copy-markdown:clean/full (source pick: store vs paginated DB) and save-markdown (path validation + slug).\n4. Wiring tests; suites; lint; census; PR.
<!-- SECTION:PLAN:END -->
