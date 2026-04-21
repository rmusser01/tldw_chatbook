# Chat-First Shell And Label Cleanup Design

Date: 2026-04-21
Status: Implemented on `codex/chat-first-shell-label-cleanup`
Scope: Shell UX, navigation labeling, chat context visibility, and handoff continuity

## Summary

tldw_chatbook has shifted to a chat-first shell where Chat is the primary work surface for agentic programming/control, preserved handoff context is visible at the shell level, and legacy navigation labels are tightened without breaking stable route IDs.

## Implemented Outcomes

This branch ships the following outcomes from the approved design:

- Global navigation uses `Library` in place of older `ccp`-style user-facing copy while keeping the `ccp` route stable.
- `Coding` remains routable for compatibility but is visually demoted out of the primary work cluster.
- Chat uses a combined shell bar above active chat content to show backend, scope, assistant identity, and session title.
- The compact model/runtime controls remain embedded in that shell bar.
- Restored chat state immediately repopulates the shell bar from saved tab/session data.
- Live tab lifecycle changes republish shell context on create, reuse, switch, close-next, and close-last flows.
- Fallback labels remain explicit when resolved workspace/persona labels are unavailable.

## Branch Reality Notes

- This branch does not include a dedicated `chat-task-surface` widget. The combined shell bar therefore mounts above current active chat content in the shipped layout.
- The design still assumes inline approvals and resume state belong in Chat, but those concerns remain separate from the shell-bar slice here.
- The earlier plan referenced `Tests/UI/test_chat_approvals_and_resume.py`; that file does not exist in this branch, so focused shell coverage landed in the existing chat UI modules plus `Tests/UI/test_chat_tab_container.py`.

## Verification

The implemented shell is covered by the branch-local focused suite:

- [Tests/UI/test_screen_navigation.py](/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/codex-chat-first-shell-label-cleanup/Tests/UI/test_screen_navigation.py)
- [Tests/UI/test_chat_shell_bar.py](/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/codex-chat-first-shell-label-cleanup/Tests/UI/test_chat_shell_bar.py)
- [Tests/UI/test_chat_window_enhanced.py](/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/codex-chat-first-shell-label-cleanup/Tests/UI/test_chat_window_enhanced.py)
- [Tests/UI/test_chat_screen_state.py](/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/codex-chat-first-shell-label-cleanup/Tests/UI/test_chat_screen_state.py)
- [Tests/UI/test_chat_tab_container.py](/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/codex-chat-first-shell-label-cleanup/Tests/UI/test_chat_tab_container.py)

## Acceptance Snapshot

The branch satisfies the shell-level intent of the original design:

- Chat is the clear top-level home for agentic programming/control work.
- Navigation labels and clustering now better match user mental models.
- The combined shell bar exposes active chat context without requiring recall from prior screens.
- Restored and tabbed sessions keep shell context visible and up to date.
- Quick model/runtime controls remain available without adding a second persistent strip.

## Deferred Follow-On

The next UX slices remain intentionally separate from this branch:

- add one end-to-end mounted Textual test for the live active-session bubble path
- promote more `Use in Chat` handoffs into reused chat sessions
- continue normalizing shell-level scope summaries across Notes, Study, Media, and future workspace-scoped shells
