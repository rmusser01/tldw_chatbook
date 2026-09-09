---
id: TASK-32067
title: >-
  Library Conversations reader: raw UUID and ISO timestamps; one-page pager
  painted
status: Done
assignee: []
created_date: '2026-09-08 18:25'
updated_date: '2026-09-08 19:25'
labels:
  - library
  - conversations
  - ux
  - critique-8
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The reader header reads 'Loaded bf20fab2-0474-… · 30 of 30 messages' and each message shows a raw ISO timestamp, while the list shows '27m'; the pager renders '1-6 of 6 · Page 1 of 1 / Already on the first page. · No more re… / ○ Previous ○ Next' on a one-page list, where Media hides it. Evidence: critique #8 snapshot `.impeccable/critique/2026-09-08T16-05-41Z__tldw-chatbook-ui-screens-library-screen-py.md` (dev 1c022378cb, 2026-09-08, dual-agent live review at 235x52/100x30/60x24 on a fresh and a seeded profile). Register row 18.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The reader identifies the conversation by title, not UUID, and shows relative or short timestamps
- [x] #2 Pagers are hidden (or inert and unlabelled) when there is a single page, consistently across Media, Conversations and Prompts
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. RED: the conversation reader status says 'Loaded <uuid>'; message headings carry raw ISO stamps; the Conversations and Prompts pagers paint 'Page 1 of 1' plus boundary reasons plus dead controls on a one-page list.
2. Status line names the loaded conversation by title (fallback: a neutral phrase, never the id).
3. Message headings use the list's own format_console_relative_age.
4. Give Conversations and Prompts the Media one-page pager rule (drop page_copy, the boundary reasons and the controls when pager.single_page and no Retry).
5. Extend the pinning test in test_library_entry_compose_once.py; GREEN; docs stamps.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
AC#1 (identity + stamps), both in the reader widget's presentation seam:
- the settled status line names the conversation by TITLE ('Loaded Design review notes · 30 of 30 messages · complete.'); it printed `state.loaded_id` -- a raw UUID that names nothing the user has seen. An untitled conversation gets 'this conversation', never the id.
- message headings carry the same compact age the Conversations list shows ('user · 27m'), through the list's own `format_console_relative_age`, instead of the stored ISO stamp. The state layer keeps the raw timestamp (it is validated identity metadata); only the heading formats it.

AC#2 (one-page pagers): Media's rule (task-28016 + task-31237) applied verbatim to Conversations and Prompts -- when `pager.single_page`, drop 'Page N of M', drop the 'Already on the first page.' / 'No more results.' reasons, and drop the Previous/Next controls, keeping the item range. A stale page's Retry still gets its row. `single_page` already existed on `LibraryPagerDisplay`; only the two canvases had never read it.

Not done here (out of scope, noted for a follow-up): the reader's ERROR and LOADING branches still append '(chat-a)' style ids to their copy -- e.g. 'Loading Beta review (chat-b); showing Alpha planning (chat-a)'. Those are different sentences with their own pinning tests, and the critique observed only the settled line.

Files: tldw_chatbook/Widgets/Library/library_conversation_reader.py, tldw_chatbook/Widgets/Library/library_conversations_canvas.py, tldw_chatbook/Widgets/Library/library_prompts_canvas.py, Tests/UI/test_library_crit8_polish_media.py, Tests/UI/test_library_entry_compose_once.py (pinning test updated to the title), Docs/User_Guide/library/media-and-conversations.md, Docs/User_Guide/library/prompts.md.
<!-- SECTION:NOTES:END -->
