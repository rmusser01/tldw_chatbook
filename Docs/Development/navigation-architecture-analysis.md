# Navigation Architecture Analysis
## tldw_chatbook Screen Shell, April 21, 2026

## Current Direction

tldw_chatbook now defaults to screen-based navigation. The main architectural question is no longer whether the app should use screens or tabs; it is how to make the screen shell read as one coherent product instead of a stack of disconnected legacy modules.

The approved direction is chat-first:

- `Chat` is the default landing destination.
- Secondary destinations exist to browse, author, study, ingest, or configure supporting artifacts.
- Agentic programming/control belongs in Chat, with approvals, progress, failures, and resume state rendered inline.

## Current Shell Contract

The shared contract is already in place:

- `BaseAppScreen` provides the common wrapper, destination shell, and state save/restore seam.
- `NavigateToScreen` is the routing message used by the main navigation system.
- The current screen set covers Chat, Study, STTS, Chatbooks, Subscriptions, Notes, Media, Ingest, Search, Evals, LLM, Settings, Customize, Logs, and Stats.
- `MainNavigationBar` still groups destinations visually by product area while preserving stable route IDs.

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

## Migration Risks

The biggest UX risks in this migration are predictable:

- surfacing both global navigation and local module navigation at the same visual level
- losing workspace or persona context during `Use in Chat` handoffs
- leaving Chat visually empty while critical approvals or task status are hidden elsewhere
- preserving legacy routes but letting legacy labels dominate the information architecture

The recent Study dashboard and inline Chat task-surface work reduce those risks, but the same discipline needs to be applied to the rest of the shells.

## Remaining Cleanup

The highest-value cleanup items are:

- remove or demote `Coding` from primary navigation once chat handoffs fully cover the programming flow
- rename `ccp` in user-facing copy so the shell matches the character/persona mental model
- normalize shell-level scope summaries across Notes, Study, Media, and future Chat handoff entry points
- keep route IDs stable while tightening labels, grouping, and destination ownership

## Runtime Policy Authority

The screen shell now has an explicit runtime-authority model:

- authoritative runtime state lives under `tldw_chatbook/runtime_policy/`
- `active_source` is app-authoritative; saved screen state and per-tab runtime fields are contextual metadata, not authority
- screen restore must never switch the active runtime source or silently replace the active server binding
- representative UI callers are expected to preflight against runtime policy before dispatching server-only or source-constrained actions
- raw shared-client construction is intentionally confined to `runtime_policy/bootstrap.py`

This matters because shell navigation is now carrying more cross-screen context than before: runtime backend, active server identity, workspace scope, assistant identity, and conversation reuse. Those values need one owner, and they need to survive restore/handoff without stale screen state taking precedence.

## Verification Hooks

The architecture should stay grounded in focused verification:

- [Tests/UI/test_screen_navigation.py](/Users/macbook-dev/Documents/GitHub/tldw_chatbook/Tests/UI/test_screen_navigation.py) for routing and primary shell wiring
- [Tests/UI/test_chat_approvals_and_resume.py](/Users/macbook-dev/Documents/GitHub/tldw_chatbook/Tests/UI/test_chat_approvals_and_resume.py) for inline approvals and continuity
- [Tests/UI/test_study_dashboard.py](/Users/macbook-dev/Documents/GitHub/tldw_chatbook/Tests/UI/test_study_dashboard.py) for shell-level study behavior
- [Tests/RuntimePolicy/test_runtime_policy_bootstrap.py](/Users/macbook-dev/Documents/GitHub/tldw_chatbook/Tests/RuntimePolicy/test_runtime_policy_bootstrap.py), [Tests/RuntimePolicy/test_runtime_policy_core.py](/Users/macbook-dev/Documents/GitHub/tldw_chatbook/Tests/RuntimePolicy/test_runtime_policy_core.py), and [Tests/RuntimePolicy/test_boundary_guards.py](/Users/macbook-dev/Documents/GitHub/tldw_chatbook/Tests/RuntimePolicy/test_boundary_guards.py) for runtime authority, hard-stop seams, and raw-client boundary confinement
- [Tests/UI/test_chat_screen_state.py](/Users/macbook-dev/Documents/GitHub/tldw_chatbook/Tests/UI/test_chat_screen_state.py), [Tests/UI/test_notes_screen.py](/Users/macbook-dev/Documents/GitHub/tldw_chatbook/Tests/UI/test_notes_screen.py), [Tests/UI/test_study_screen.py](/Users/macbook-dev/Documents/GitHub/tldw_chatbook/Tests/UI/test_study_screen.py), and [Tests/UI/test_ccp_screen.py](/Users/macbook-dev/Documents/GitHub/tldw_chatbook/Tests/UI/test_ccp_screen.py) for representative restore-precedence and UI preflight behavior
- [Tests/UI/test_notes_screen.py](/Users/macbook-dev/Documents/GitHub/tldw_chatbook/Tests/UI/test_notes_screen.py), [Tests/UI/test_search_rag_window.py](/Users/macbook-dev/Documents/GitHub/tldw_chatbook/Tests/UI/test_search_rag_window.py), [Tests/UI/test_media_window_v88_textual.py](/Users/macbook-dev/Documents/GitHub/tldw_chatbook/Tests/UI/test_media_window_v88_textual.py), and [Tests/UI/test_ingestion_ui_redesigned.py](/Users/macbook-dev/Documents/GitHub/tldw_chatbook/Tests/UI/test_ingestion_ui_redesigned.py) for wrapped destination shells
