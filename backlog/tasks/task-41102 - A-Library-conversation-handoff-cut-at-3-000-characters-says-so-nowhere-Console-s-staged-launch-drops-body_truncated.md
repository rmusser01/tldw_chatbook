---
id: TASK-41102
title: "A Library conversation handoff cut at 3,000 characters says so nowhere: Console's staged launch drops body_truncated"
status: Done
assignee:
- '@claude'
created_date: 2026-11-15 02:20
updated_date: '2026-11-15 20:00'
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
- [x] #1 When a staged conversation or media excerpt was cut, Console's staged source says so where the user reviews it before sending
- [x] #2 The context sent to the model for that source says the excerpt is partial
- [x] #3 A test stages a conversation longer than the cap and checks both, red before the change
- [x] #4 The Library and Console guide pages describe the new indication instead of saying nothing reports it
<!-- AC:END -->

## Implementation Plan

1. Trace `body_truncated` from the Library conversation and media handoff payloads into Console's staged launch, the staged-evidence strip / Inspector tray, and the send-time context block.
2. Write a test that stages a conversation longer than the 3,000-character cap and checks both the review surface and the sent context say the excerpt is partial; show it red.
3. Carry the flag through the staged launch and mark the staged source and its sent context as partial.
4. Update `Docs/User_Guide/library/media-and-conversations.md` and `console/context-and-rag.md` (and their docs contract) to describe the indication.

## Implementation Notes

The premise held at base cc6006fcf5. `_stage_handoff_as_console_live_work` (`chat_screen.py`) built the staged evidence reference from the body and never read `body_truncated`, so nothing in Console or in the model's context marked a cut excerpt.

**Change.** The staging step now marks an excerpt as partial when either holds:
- the staged text is the payload's body and `body_truncated` is set (a Library conversation cut at 3,000 characters, or media at 500);
- the text is longer than the 4,000-character staging cap, which also cuts long notes and was just as silent.

A partial excerpt's reference title becomes "*title* (partial excerpt)". That one title feeds the staged-evidence strip row, the Inspector's Sources tray row, and the `[S1] ... — title` header of the context the model receives (`RAG_Search/local_citation_capture.py`). The title was used rather than an appended snippet marker because `EvidenceReference` cuts snippets at 4,000 characters (a trailing marker would be dropped), and because the tray hides the snippet until a row is expanded. The handoff payload, launch title and snippet are unchanged.

**Test (AC#3).** `Tests/UI/test_console_live_work_handoffs.py::test_a_cut_conversation_excerpt_reads_as_partial_before_and_on_send` builds the payload with Library's own builder from a conversation longer than the cap (`x` * 2994 plus a later message). It stages the payload in a mounted Console and checks the strip row, the tray row and the captured model context. Red at base (also re-run with the final test against base `chat_screen.py`): `assert 'Planning (partial excerpt)' in 'Planning — conversation'`. Green after.

**Docs (AC#4).** `library/media-and-conversations.md` (the conversation reader paragraph and the "Staging sends an excerpt" quirk) and `console/context-and-rag.md` (the staged-sources paragraph) now say a cut excerpt is staged and sent as "*title* (partial excerpt)", not that nothing reports it. `Tests/Docs/test_console_library_controls_docs.py` pins "title (partial excerpt)" on both pages: red with the page edits reverted (both pages missing it), then `Tests/Docs/` 32 passed.

**Covering runs (`-n 4`, load 95-133).** The live-work handoff, staged-evidence strip, staged context, staged-evidence sessions, cited-sources persisted, local citation capture and Library handoff excerpt files: 151 passed, 37 failed. With base `chat_screen.py`, 35 of those 37 failed again (RecoveryRequired in the plain Library excerpt file, citation-readiness and destination timeouts). The other two (`test_console_stage_then_navigate_then_stage_again_displays_the_newest_launch[False]`, a 2 s card wait, and `test_schedules_destination_routes_latest_digest_output_to_console`, a Schedules worker `NoMatches`) passed at HEAD on a rerun (3 passed with both parameters). Size ratchets: 231 passed; `chat_screen.py` is 24564 of its 24569 row.

**Lane.** UI census tail; `MINIMUM_FILES` 2148, no re-deal. Serial with the lane's flags (`-p no:xdist --timeout=180 --durations=0`, load 105-115): 71.2 s (setup 2.1 + call 68.3 + teardown 0.8).

Files: `tldw_chatbook/UI/Screens/chat_screen.py`, `Tests/UI/test_console_live_work_handoffs.py`, `Tests/Docs/test_console_library_controls_docs.py`, `Docs/User_Guide/library/media-and-conversations.md`, `Docs/User_Guide/console/context-and-rag.md`, `scripts/ui_pr_gate_census.txt`, `scripts/check_ui_pr_gate_census.py`.
