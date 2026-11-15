---
id: TASK-41102
title: "A Library conversation handoff cut at 3,000 characters says so nowhere: Console's staged launch drops body_truncated"
status: To Do
created_date: 2026-11-15 02:20
labels:
- follow-up
- console
- library
- privacy
priority: medium
references:
- tldw_chatbook/UI/Screens/chat_screen.py
- tldw_chatbook/UI/Screens/library_screen.py
- tldw_chatbook/Chat/chat_handoff_models.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed in round 14 of the 2026-10-23 burn-down (closeout, 2026-11-15). Checked against the round-14 stack top c7d4a41717 (on dev beab5c1e4e); line numbers are that tree's.

**Why.** A user who sends "Use as source" on a long conversation, and the model answering, both take the excerpt for the whole conversation. Later messages are dropped with no marker. TASK-40330 made the pages say plainly that nothing reports it; the coordinator's ruling there (2026-11-13) filed the code gap as this follow-up.

**Evidence.**
- `library_screen.py:14294-14332` (`_selected_conversation_handoff_payload`) cuts the speaker-labelled excerpt at `LIBRARY_CONVERSATION_HANDOFF_EXCERPT_CHARS = 3_000` and sets `body_truncated`.
- `chat_screen.py:16358-16391` (`_stage_handoff_as_console_live_work`) builds `launch_payload` from the body (capped again at 4,000 by `sanitize_string`) and never copies `body_truncated`, so neither the staged-evidence strip, the Inspector tray nor the send-time context says the excerpt was cut.
- `ChatHandoffPayload.model_context_block` (`chat_handoff_models.py:148-176`) does print "Body truncated:", but the native Console lane does not use it.
- Media differs: Reader > Info says "Stored text excerpt (truncated)" (`library_media_controller.py:1887-1899`), but Console does not either.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 When a staged conversation or media excerpt was cut, Console's staged source says so where the user reviews it before sending
- [ ] #2 The context sent to the model for that source says the excerpt is partial
- [ ] #3 A test stages a conversation longer than the cap and checks both, red before the change
- [ ] #4 The Library and Console guide pages describe the new indication instead of saying nothing reports it
<!-- AC:END -->
