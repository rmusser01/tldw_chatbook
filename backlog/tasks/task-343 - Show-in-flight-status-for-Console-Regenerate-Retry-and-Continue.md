---
id: TASK-343
title: Show in-flight status for Console Regenerate, Retry and Continue
status: To Do
assignee: []
created_date: '2026-07-20 14:21'
labels: [console, ux, keyboard]
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
With the newest assistant reply selected, pressing r produced no visible change for 75+ seconds of polling: header status stayed 'Ready', the old reply stayed untouched, no spinner/placeholder appeared, and the composer showed Send/Attach/Save (no Stop button — confirmed in j6-a23-after-r.png). The regenerated reply eventually replaced the message minutes later (new content + version '<' '>' chevrons in j6-a26-mystery-state.png). By contrast, the normal send path immediately appends the user message, streams the reply and shows Stop. During the silent window my extra keyboard presses activated unrelated late-arriving UI.

**Repro:** Open a conversation tab, select the last assistant message (k), press r. Watch header row, composer row and message area: nothing changes for over a minute against a slow local provider; the reply is later swapped wholesale.

**Verifier note:** Code-confirmed root cause: the 0.2s transcript sync timer is started ONLY in _submit_console_native_draft (chat_screen.py:8732, sole call site); _regenerate_console_message/_retry_console_message/_continue_console_message (10789-10820) never start it, so although ConsoleChatController.regenerate_message sets VALIDATING then STREAMING and streams into begin_variant_stream, no UI sync runs until completion — status chip stays Ready, Stop stays display:none, no placeholder. Present since the timer's introduction in PR #359 (git -S), so NEW not REGRESSION. Distinct from task-193 (send-path placeholder) and task-232 (mid-run gating). P1 stands: user cannot stop or even see a multi-minute run.

**Source:** Console UX expert review 2026-07-20 (finding j6-regenerate-zero-feedback; P1, verdict NEW, high confidence) — full report `Docs/superpowers/qa/console-ux-expert-review-2026-07-17/README.md`, app under review at origin/dev cad9e271d, J6 keyboard-only/small-terminal journey. Evidence: `j6-a23-select-assistant.png`, `j6-a23-after-r.png`, `j6-a24-regen-watch.png`, `j6-a26-mystery-state.png` (regression/P1 captures committed alongside the report; the full 339-capture set is on the review machine).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Regeneration shows the same in-flight affordances as send: an immediate placeholder or 'regenerating…' marker on the message, a Stop control, and a busy status
<!-- AC:END -->
