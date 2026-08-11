---
id: TASK-15423
title: >-
  Library "Open in Console" does not navigate or load the conversation
status: To Do
assignee: []
created_date: '2026-08-11 12:00'
labels:
  - library
  - console
  - bug
  - uat
priority: medium
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Observed twice during TTS UAT on `origin/dev` `82b595049` (2026-08-11), clean
profile, 235x52 tmux, conversation seeded directly into ChaChaNotes
(`add_conversation` + two `add_message`, no character).

Library ▸ Conversations ▸ conversation detail shows an "Open in Console"
button. Clicking it (button visibly takes focus) and pressing Enter did
nothing: the app stayed on the Library screen, and after manually switching to
Console the conversation was not loaded there. Reproduced in two app sessions —
one with the Console provider-setup gate active (no LLM configured) and one
with Console fully unlocked and a session live.

Two adjacent observations from the same UAT, possibly the same root or worth
splitting during triage:

- Console's rail "Search conversations" for the seeded title returned
  "0 matches" while Library search/browse found it fine — so Console could not
  reach the conversation by either route.
- No error, toast, or log line was observed for the failed open (in-app Logs
  Errors count did not change).

Needs verification of the intended contract first: if "Open in Console" is
expected to attach the DB conversation to a Console session, it silently does
nothing; if Console deliberately only lists its own session-created chats, the
button (and the search asymmetry) misleads.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] Activating "Open in Console" on a conversation either opens it in Console or surfaces a visible, accurate reason it cannot
- [ ] Console's conversation search finds DB conversations that Library finds, or the two surfaces' differing scopes are made explicit in the UI
<!-- AC:END -->

## Investigation Notes (2026-08-11, follow-up session)

<!-- SECTION:NOTES:BEGIN -->
Root-caused live; the original "silently does nothing" framing was WRONG in
one respect and right in a sharper one:

- **It is not silent.** The button's handler
  (`library_screen.py` `_open_selected_conversation_handoff`) posts a
  warning toast on every guard; the original UAT captures (~4-5s after the
  click) missed the transient toast. Caught live this session: "Copy or
  link blocked Library sources into the active workspace before using them
  in Console."
- **The real defect candidate is the all-or-nothing gate.**
  `build_library_workspace_depth_state` (`Workspaces/display_state.py`)
  computes `handoff_enabled = bool(rows) and blocked_count == 0` — ONE
  workspace-ineligible row anywhere in the visible Library disables
  "Open in Console" for EVERY conversation, including fully eligible ones.
  Reproduced deterministically: a Console-created conversation handed off
  fine (navigated to Console, staged as source context, auto-sent); after
  seeding ONE foreign-client conversation into the DB, the SAME action on
  ANY conversation is blocked. Per-item eligibility already exists
  (`row.active_context_eligible`), and the handoff label itself renders
  "N eligible, M blocked" — the UI acknowledges mixed states while the
  gate does not. The media handoff shares the same aggregate gate by
  documented intent ("identical workspace-staging policy"), so relaxing it
  to per-item gating is a workspace-policy decision for the owner, not a
  unilateral fix.
- The toast copy compounds it: a user clicking on an ELIGIBLE conversation
  is told about "blocked Library sources" collectively, with no hint their
  selected item is fine and some OTHER row is the blocker.
- "Open in Console" also does not open the transcript — it stages the
  conversation as SOURCE CONTEXT into a Console session and auto-sends a
  "Use this conversation as source context" turn. Whether that matches the
  label's promise is a second, smaller copy question.
- The Console-rail search asymmetry from the original report (Console's
  "Search conversations" found 0 for a conversation Library sees) remains
  unverified-in-mechanism and stays open.

Recommended AC revision for the implementing task, pending owner ruling:
gate the single-item action on the SELECTED row's eligibility, keep the
aggregate gate for bulk staging, and name the blocking row in the toast.
<!-- SECTION:NOTES:END -->
