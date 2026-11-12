---
id: TASK-40330
title: "The Library and Console guide pages say a handoff carries a title label, not the transcript, though it stages up to 3,000 characters of it"
status: To Do
created_date: 2026-11-12 19:36
labels:
- follow-up
- docs
- library
- privacy
priority: medium
references:
- Docs/User_Guide/library/media-and-conversations.md
- Docs/User_Guide/console/context-and-rag.md
- tldw_chatbook/UI/Screens/library_screen.py
- tldw_chatbook/UI/Library_Modules/library_media_controller.py
dependencies:
- TASK-38974
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 13 of the 2026-10-23 burn-down (docs stream, TASK-38974). Checked by reading the code at the round-13 stack top 8b633d0101, 2026-11-12; line numbers are that tree's.

**Why.** Users are told the model gets only a title, when it receives up to 3,000 characters of the conversation or media content. That is a privacy and context-budget fact people decide on, and the pages point at task-2376, which is Done.

**Evidence.**
- `media-and-conversations.md:778-779`: the reference "contains a title-based summary label, not the transcript text; real excerpts remain tracked in task-2376"; again at `:1017-1020`. `console/context-and-rag.md:989-990` says the same.
- The code: `library_screen.py:14279-14346` (`_selected_conversation_handoff_payload`) builds a "Conversation excerpt:" from the reader's speaker-labelled messages, capped at `LIBRARY_CONVERSATION_HANDOFF_EXCERPT_CHARS = 3_000` (`screen_constants.py:297`), setting `body_truncated` when cut. Media adds a "Content excerpt:" block (`library_media_controller.py:1845-1851`). The excerpts came in with d6b8ea2566 (2026-09-16), 41 minutes after the page sentence (f09a7786d4); `Tests/UI/test_library_handoff_excerpts.py` tests them.
- TASK-38974 pinned only "Use as source stages a reference to the selected conversation...", so correcting the sentence breaks no contract.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Both pages say that Use as source / Use in Console stages a bounded excerpt of the content (speaker-labelled for conversations, truncation reported), and neither calls task-2376 open
- [ ] #2 The controls docs contract pins the excerpt wording
<!-- AC:END -->
