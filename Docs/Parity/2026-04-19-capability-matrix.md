# tldw_chatbook Capability Matrix

## Purpose

Crosswalk user-visible and interoperability-relevant capabilities between `tldw_server` and `tldw_chatbook`. This matrix is the primary prioritization artifact for deciding what to align first.

## Scoring Formula

```text
priority = impact*5 + alignment*4 + hermes*2 + blocking*3 - risk*3
```

Scores represent practical implementation priority under the current repo state, not abstract product importance. `Risk` includes current dirty-tree overlap and likely UI reconciliation cost.

## Status Legend

- `present`: capability exists in a materially usable local form
- `partial`: capability exists but is incomplete, outdated, or not aligned
- `absent`: no meaningful local support exists
- `legacy`: an older or duplicate local surface exists and may need replacement

## Matrix

| Domain | Capability | tldw_server Surface | tldw_chatbook Surface | Gap Type | Impact | Alignment | Hermes | Blocking | Risk | Priority | Phase | Notes |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---|
| Prompts / Chatbooks | Prompt library, prompt workflows, and chatbooks | `prompts.py`, `chatbooks.py`, `writing.py`, `writing_manuscripts.py` | `Prompt_Management/`, `DB/Prompts_DB.py`, `Chatbooks/`, `UI/Chatbooks_Window_Improved.py`, `UI/Screens/chatbooks_screen.py` | Already present but structurally incompatible | 4 | 5 | 1 | 5 | 2 | 51 | Phase 1 / 2 | Best first vertical: the prompt side already has strong UUID/version/delete overlap, the chatbook side already has import/export seams, and this avoids the current dirty chat UI overlap. |
| Notes | Notes, workspace notes, and notes graph | `notes.py`, `workspaces.py`, `notes_graph.py` | `UI/Notes_Window.py`, `UI/Screens/notes_screen.py`, `Notes/`, `DB/ChaChaNotes_DB.py`, `state/notes_state.py` | Missing interoperability support | 5 | 5 | 0 | 4 | 3 | 48 | Phase 1 / 2 | Strong offline-first fit and strong user value. Main work is explicit workspace and graph mapping, not inventing core note behavior. |
| Chat | Conversation sessions, history, and message flow | `chat.py`, `messages.py`, `chat_loop.py`, `chat_workflows.py` | `UI/Chat_Window_Enhanced.py`, `UI/Screens/chat_screen.py`, `Chat/`, `DB/ChaChaNotes_DB.py`, `app.py` | Missing interoperability support | 5 | 5 | 1 | 5 | 5 | 47 | Phase 1 / 2 | Core parity domain, but practical implementation risk is highest because current local work is already touching chat UI, navigation, and model-control surfaces. |
| Characters | Character catalog, chat sessions, and character messages | `characters_endpoint.py`, `character_chat_sessions.py`, `character_messages.py` | `UI/Conv_Char_Window.py`, `UI/Screens/ccp_screen.py`, `Character_Chat/Character_Chat_Lib.py`, `DB/ChaChaNotes_DB.py` | Missing interoperability support | 4 | 4 | 0 | 4 | 2 | 42 | Phase 1 / 2 | Strong local support already exists. This looks more like data-shape alignment than brand-new feature work and should likely follow conversation/session mapping. |
| Media | Files, ingestion sources, reading, and reading progress | `files.py`, `ingestion_sources.py`, `reading.py`, `media/reading_progress.py` | `UI/MediaWindow_v2.py`, `UI/MediaIngestWindowRebuilt.py`, `UI/Screens/media_screen.py`, `UI/Screens/media_ingest_screen.py`, `Local_Ingestion/`, `DB/Client_Media_DB_v2.py` | Already present but structurally incompatible | 4 | 4 | 0 | 3 | 3 | 36 | Phase 2 / 3 | Important for interoperability, but core content entities and packaging decisions should stabilize first. |
| RAG | Unified retrieval, chunking, templates, and chat documents | `rag_unified.py`, `chunking.py`, `chunking_templates.py`, `chat_documents.py` | `RAG_Search/`, `Embeddings/`, `UI/SearchRAGWindow.py`, `UI/Screens/search_screen.py` | Already present but structurally incompatible | 4 | 4 | 1 | 3 | 4 | 35 | Phase 3 | Valuable, but it depends on earlier decisions around document/media identity, chat-document linkage, and import/export compatibility. |
| Chat | Chat dictionaries, document context, and chat-side helpers | `chat_dictionaries.py`, `chat_documents.py`, `chat_grammars.py` | `Character_Chat/Chat_Dictionary_Lib.py`, `UI/CCP_Modules/`, `Tools/rag_search_tool.py`, `RAG_Search/` | Already present but structurally incompatible | 3 | 4 | 1 | 3 | 3 | 33 | Phase 2 | This should move with broader chat/document-context work instead of leading it. |
| MCP / Tools / Skills | MCP unified access, tools catalog, tool execution, and skills | `mcp_unified_endpoint.py`, `tools.py`, `skills.py` | `MCP/client.py`, `MCP/server.py`, `MCP/tools.py`, `MCP/resources.py`, `MCP/prompts.py`, `Tools/tool_executor.py`, `UI/Tools_Settings_Window.py` | Already present but structurally incompatible | 3 | 4 | 2 | 2 | 4 | 29 | Phase 3 / 4 | Local MCP is stdio-oriented today; server MCP is HTTP/WebSocket/JWT-oriented and likely needs a compatibility bridge before deeper parity work. |
| Evals / Study | Evals, datasets, benchmarks, flashcards, and quizzes | `evaluations/evaluations_unified.py`, `evaluations/evaluations_crud.py`, `evaluations/evaluations_datasets.py`, `evaluations/evaluations_benchmarks.py`, `flashcards.py`, `quizzes.py` | `UI/Evals/evals_window_v3.py`, `UI/Screens/evals_screen.py`, `UI/Study_Window.py`, `UI/Screens/study_screen.py`, `Evals/` | Missing interoperability support | 3 | 4 | 0 | 2 | 3 | 28 | Phase 3 | Useful, but should follow the core content and prompt packaging model so study artifacts are not aligned twice. |
| UX Overlay | Tool progress, session ergonomics, model controls, approvals, background-task visibility | `hermes-agent/RELEASE_v0.8.0.md`, `hermes-agent/model_tools.py`, `hermes-agent/cli.py`, `hermes-agent/hermes_cli/` | `UI/Chat_Window_Enhanced.py`, `UI/Screens/chat_screen.py`, `Widgets/`, `app.py` | Hermes-inspired UX enhancement only | 2 | 1 | 5 | 1 | 3 | 18 | Phase 4 | Keep secondary unless a Hermes-derived pattern directly improves a parity vertical already in flight. |
| Companion / Persona | Companion, persona, and personalization-adjacent workflows | `companion.py`, `persona.py`, `personalization.py` | `Character_Chat/`, `UI/Conv_Char_Window.py`, no obvious dedicated companion surface | Missing feature | 2 | 3 | 0 | 1 | 3 | 16 | Phase 4 | Secondary unless it unlocks a concrete offline workflow that is clearly distinct from ordinary character chat. |
