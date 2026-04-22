# Chat-First Shell Migration

## New Top-Level IA

The application is converging on a single screen shell where Chat is the default landing destination and the rest of the product is organized as destination shells rather than peer tabs.

- `Chat` is the primary conversation surface for general assistance, agentic programming/control, inline approvals, and task continuity.
- `Notes`, `Media`, `Ingest`, `Search`, and `Subscriptions` are content/workflow shells that feed context into Chat.
- `Conversations, Characters & Prompts` (`ccp`) and `Study` are persona/learning shells that should preserve handoff context when they open or reuse Chat.
- `LLM`, `STTS`, `Evals`, `Settings`, `Customize`, `Logs`, and `Stats` remain utility/configuration shells.
- The legacy `coding` route remains available for compatibility while agentic programming workflows move into Chat.

## Legacy Route Aliases

Route IDs should stay stable even when labels or grouping evolve. Existing deep links and internal navigation may still target:

- `chat`
- `coding`
- `chatbooks`
- `notes`
- `media`
- `ingest`
- `search`
- `subscriptions`
- `ccp`
- `study`
- `llm`
- `stts`
- `evals`
- `tools_settings`
- `customize`
- `logs`
- `stats`

Presentation labels can change independently. The route contract should not.

## Coding Destination Deprecation

The product model is now chat-first for programming and control tasks.

- New agentic UX work should render progress, approvals, errors, and resume state inline in Chat.
- Coding-specific entry points should open or reuse a Chat session with coding context instead of fragmenting the experience across separate primary destinations.
- The standalone `coding` destination can remain as a compatibility surface during migration, but it should no longer define the mental model for agentic work.

## Workspace Scope Rules

Scope changes must be explicit and durable.

- When a flow enters workspace scope, show the workspace context at the shell level, not only inside embedded feature modules.
- Preserve `runtime_backend`, `scope_type`, `workspace_id`, and assistant/persona identity across `Use in Chat` handoffs.
- Do not silently reset workspace context when switching between dashboard and detail views.
- If workspace-scoped capabilities are unavailable in the current runtime, fail closed with explanatory empty states instead of partial or misleading data.
- Notes and Study are the reference patterns for shell-level scope summaries; future shells should follow the same contract.

## Testing Strategy

The migration should be defended by focused routing, shell, and wrapped-surface coverage:

- Screen wiring: [Tests/UI/test_screen_navigation.py](/Users/macbook-dev/Documents/GitHub/tldw_chatbook/Tests/UI/test_screen_navigation.py)
- Chat continuity: [Tests/UI/test_chat_approvals_and_resume.py](/Users/macbook-dev/Documents/GitHub/tldw_chatbook/Tests/UI/test_chat_approvals_and_resume.py), [Tests/UI/test_chat_screen_state.py](/Users/macbook-dev/Documents/GitHub/tldw_chatbook/Tests/UI/test_chat_screen_state.py), [Tests/UI/test_chat_window_enhanced.py](/Users/macbook-dev/Documents/GitHub/tldw_chatbook/Tests/UI/test_chat_window_enhanced.py)
- Study shell behavior: [Tests/UI/test_study_dashboard.py](/Users/macbook-dev/Documents/GitHub/tldw_chatbook/Tests/UI/test_study_dashboard.py)
- Wrapped feature surfaces: [Tests/UI/test_notes_screen.py](/Users/macbook-dev/Documents/GitHub/tldw_chatbook/Tests/UI/test_notes_screen.py), [Tests/UI/test_search_rag_window.py](/Users/macbook-dev/Documents/GitHub/tldw_chatbook/Tests/UI/test_search_rag_window.py), [Tests/UI/test_media_window_v88_textual.py](/Users/macbook-dev/Documents/GitHub/tldw_chatbook/Tests/UI/test_media_window_v88_textual.py), [Tests/UI/test_ingestion_ui_redesigned.py](/Users/macbook-dev/Documents/GitHub/tldw_chatbook/Tests/UI/test_ingestion_ui_redesigned.py), [Tests/UI/test_chat_window_enhanced_integration.py](/Users/macbook-dev/Documents/GitHub/tldw_chatbook/Tests/UI/test_chat_window_enhanced_integration.py)
