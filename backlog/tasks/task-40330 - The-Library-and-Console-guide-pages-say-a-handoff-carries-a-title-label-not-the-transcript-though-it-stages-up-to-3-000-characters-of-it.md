---
id: TASK-40330
title: "The Library and Console guide pages say a handoff carries a title label, not the transcript, though it stages up to 3,000 characters of it"
status: Done
assignee: ['@claude']
updated_date: '2026-11-13 03:19'
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
- [x] #1 Both pages say that Use as source / Use in Console stages a bounded excerpt of the content (speaker-labelled for conversations, truncation reported), and neither calls task-2376 open
- [x] #2 The controls docs contract pins the excerpt wording
<!-- AC:END -->

## Implementation Plan

1. Re-check what a Library conversation "Use as source" and a media "Use in Console" stage at base 26b2e677dd, and what of it reaches the model on send (builder caps, the staging cap, send-time formatting) and what the user is told about truncation.
2. Pin the excerpt wording for both pages in `Tests/Docs/test_console_library_controls_docs.py` (replacing the "stages a reference" pin) and show it red on the current pages.
3. Rewrite the stale sentences in `library/media-and-conversations.md` (conversation reader paragraph, the Stage-a-conversation steps, the "Staging now actually reaches the model" quirk) and `console/context-and-rag.md` (staged sources paragraph); drop the task-2376 references.
4. Run the Docs tests through the wrapper; commit.

## Implementation Notes

Both pages now say what a handoff actually sends: a bounded excerpt of the content, with its limit, and neither mentions task-2376.

**Re-checked at base 26b2e677dd** (line numbers are this tree's):
- Conversation "Use as source" (`library_conversation_reader.py:216`): `library_screen.py:14279-14346` builds "Conversation excerpt:" from the reader's messages in order, each `sender: text`, cut at `LIBRARY_CONVERSATION_HANDOFF_EXCERPT_CHARS = 3_000` (`Library_Modules/screen_constants.py:297`; labels and separators count toward it), then title, ID, message count, workspace, updated, and sets `body_truncated` when cut. Later messages are dropped with no marker.
- Media "Use in Console": `library_media_controller.py:1845-1851` takes the first `LIBRARY_MEDIA_HANDOFF_EXCERPT_CHARS = 500` characters (`screen_constants.py:296`, not 3,000 as the task title implies), then title, ID, Type/Author/Keywords lines. `_library_media_console_representation` (`:1887`) feeds Info's "Use in Console sends:" line (`library_media_viewer.py:811`): "Complete stored text excerpt" / "Stored text excerpt (truncated)" / "Metadata only (no stored text)".
- Notes: `library_notes_controller.py:5277-5294` puts the body in the payload with no display summary.
- Staging: `chat_screen.py:16375-16380` uses the body for media/conversation (body or summary for others) and caps the snippet at 4,000 characters (`sanitize_string` truncates); the snippet becomes the evidence content unchanged (`local_citation_capture.py:411`), and send-time blocks are capped at 64 KB, so the excerpt reaches the model whole.
- Truncation: `body_truncated` is not copied into the staged launch, so for conversations nothing on screen (or in the model's context) says the excerpt was cut. The pages say so plainly rather than claim a report; media truncation is reported by the Info line. Proposed as a follow-up in the stream report.

**Changes.** `Docs/User_Guide/library/media-and-conversations.md`: the conversation reader paragraph, Stage-a-conversation step 2, and the quirk bullet (renamed "Staging sends an excerpt, not the whole item"). `Docs/User_Guide/console/context-and-rag.md`: the staged-sources paragraph (notes 4,000, media 500, conversations 3,000).

**Contract** (`Tests/Docs/test_console_library_controls_docs.py`): the "stages a reference" pin became "stages an excerpt"; both pages pin the 3,000-character speaker-labelled transcript and the 500-character media excerpt. Red before the page edits: both pages reported the missing fragments (2 failed, 19 passed). Green after: `Tests/Docs/` 32 passed.

**Ruling (coordinator, 2026-11-13):** AC#1's "truncation reported" is met by the pages saying plainly that nothing reports a conversation excerpt's truncation (media truncation is reported on Reader ▸ Info); the code gap (`body_truncated` not carried into the staged launch) is filed as a follow-up at closeout.
