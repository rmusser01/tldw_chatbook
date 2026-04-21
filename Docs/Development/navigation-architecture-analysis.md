# Navigation Architecture Analysis
## tldw_chatbook Screen Shell, April 21, 2026

## Current Direction

tldw_chatbook now defaults to screen-based navigation. The main architectural question is no longer whether the app should use screens or tabs; it is how to make the screen shell read as one coherent product instead of a stack of disconnected legacy modules.

The approved direction is chat-first:

- `Chat` is the default landing destination.
- Secondary destinations exist to browse, author, study, ingest, or configure supporting artifacts.
- Agentic programming/control belongs in Chat, with approvals, progress, failures, and resume state rendered inline.

## Current Shell Contract

The shared contract is now in place:

- `BaseAppScreen` provides the common wrapper, destination shell, and state save/restore seam.
- `NavigateToScreen` is the routing message used by the main navigation system.
- `MainNavigationBar` preserves stable route IDs while presenting updated user-facing labels and cluster order.
- The user-facing `ccp` label is now `Library`.
- The legacy `coding` route remains reachable for compatibility but is visually demoted out of the primary work cluster.

This is enough infrastructure to support the new IA without introducing another navigation system.

## Information Architecture Rules

The shell should follow a few strict rules:

- Only one top-level navigation system should be visible at a time.
- Destination shells may include local section switchers, but those switchers should not masquerade as global navigation.
- Cross-shell handoffs must preserve user intent: assistant identity, persona choice, runtime backend, workspace scope, and conversation continuity.
- Workspace/global transitions must be visible at the shell level, not buried inside embedded modules.
- Legacy route IDs may remain stable while user-facing labels and grouping evolve.

## Chat-First Product Model

Chat is now the primary work surface for:

- general conversation
- persona-guided assistance
- agentic programming and control
- inline approvals for privileged actions
- task continuity and resume state

That means the old dedicated `coding` destination should be treated as a compatibility surface, not the long-term primary entry point for programming workflows. New design and implementation work should prefer Chat-centered flows and reuse sessions instead of sending users into parallel control surfaces.

## Implemented Shell Outcomes

The current branch ships the first shared shell slice:

- `Library` replaces internal `ccp`-style copy in primary navigation while keeping the `ccp` route stable.
- `Coding` stays routable but is grouped with system/utility destinations instead of the primary work cluster.
- Chat mounts a combined shell bar above active chat content in both single-session and tabbed modes.
- The shell bar surfaces backend, scope, assistant identity, and session title from restored session state and from live tab lifecycle changes.
- The compact model/runtime controls remain embedded in the shell bar and still sync through the chat host.

## Migration Risks

The biggest UX risks in this migration are predictable:

- surfacing both global navigation and local module navigation at the same visual level
- losing workspace or persona context during `Use in Chat` handoffs
- leaving Chat visually empty while critical approvals or task status are hidden elsewhere
- preserving legacy routes but letting legacy labels dominate the information architecture

The current branch reduces those risks for Chat, but the same discipline still needs to be applied to the rest of the shells.

## Remaining Cleanup

The highest-value cleanup items after this branch are:

- add one end-to-end mounted Textual test for the live `ActiveSessionChanged` bubble path
- continue normalizing shell-level scope summaries across Notes, Study, Media, and future Chat handoff entry points
- keep route IDs stable while tightening labels, grouping, and destination ownership

## Verification Hooks

The architecture should stay grounded in focused verification:

- [Tests/UI/test_screen_navigation.py](/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/codex-chat-first-shell-label-cleanup/Tests/UI/test_screen_navigation.py) for routing and navigation clustering
- [Tests/UI/test_chat_shell_bar.py](/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/codex-chat-first-shell-label-cleanup/Tests/UI/test_chat_shell_bar.py) for shell-bar context, truncation, and compact control behavior
- [Tests/UI/test_chat_window_enhanced.py](/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/codex-chat-first-shell-label-cleanup/Tests/UI/test_chat_window_enhanced.py) for shell-bar mount position in the current chat UI
- [Tests/UI/test_chat_screen_state.py](/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/codex-chat-first-shell-label-cleanup/Tests/UI/test_chat_screen_state.py) for restore-time and live screen-side shell sync
- [Tests/UI/test_chat_tab_container.py](/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/codex-chat-first-shell-label-cleanup/Tests/UI/test_chat_tab_container.py) for tab reuse, switch, close-next, and close-last lifecycle publishing
- [Tests/UI/test_study_dashboard.py](/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/codex-chat-first-shell-label-cleanup/Tests/UI/test_study_dashboard.py) for shell-level study behavior
- [Tests/UI/test_notes_screen.py](/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/codex-chat-first-shell-label-cleanup/Tests/UI/test_notes_screen.py), [Tests/UI/test_search_rag_window.py](/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/codex-chat-first-shell-label-cleanup/Tests/UI/test_search_rag_window.py), [Tests/UI/test_media_window_v88_textual.py](/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/codex-chat-first-shell-label-cleanup/Tests/UI/test_media_window_v88_textual.py), and [Tests/UI/test_ingestion_ui_redesigned.py](/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/codex-chat-first-shell-label-cleanup/Tests/UI/test_ingestion_ui_redesigned.py) for wrapped destination shells
