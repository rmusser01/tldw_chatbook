---
id: TASK-1364
title: >-
  Wire ChatInputTextArea (modal Enter, ghost text, history) into tabbed and
  default chat surfaces
status: To Do
assignee: []
created_date: '2026-08-05 15:07'
labels:
  - ui
  - chat
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-1348/1350 delivered ChatInputTextArea (modal Enter/Shift+Enter, generation-gated submit, ghost-text suggestions, JSONL history recall) but only wired it into Chat_Window_Enhanced. Default config uses enable_tabs=true and use_enhanced_window=false, so the tabbed ChatSession input (Widgets/Chat_Widgets/chat_session.py:108) and the legacy Chat_Window still use plain TextArea and the record_successful_send duck-typed hook no-ops there. Swap those surfaces to ChatInputTextArea (or a shared base) so the new input UX is active out of the box.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Tabbed chat session inputs use ChatInputTextArea with modal Enter working;Default (non-enhanced) chat window input uses ChatInputTextArea or is deliberately retired in favor of the enhanced window with the decision recorded;Prompt history recording works on all chat send surfaces;Existing tabbed chat tests pass
<!-- AC:END -->

## Superseded

Premise invalidated: dev has deleted the legacy chat surfaces entirely (Chat_Window_Enhanced.py, Chat_Window.py, chat_events.py, chat_streaming_events.py) — the tabbed/default surfaces this task targeted no longer exist. The Console (ChatScreen + ConsoleComposerBar + ConsoleTranscript) is the only live chat UI. Superseded by the console-port tasks created 2026-08-05 (composer history/ghost-text, transcript pruning, tool-call diff rows). The portable components (ChatInputTextArea, PromptHistory, chat_log_pruning, diff_widgets) remain on branch feat/toad-ui-improvements for reference; PromptHistory and diff_widgets are directly reusable.
