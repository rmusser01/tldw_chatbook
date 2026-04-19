# tldw_chatbook Capability Matrix

## Purpose

Crosswalk user-visible and interoperability-relevant capabilities between `tldw_server` and `tldw_chatbook`. This matrix is the primary prioritization artifact for deciding what to align first.

## Scoring Formula

```text
priority = impact*5 + alignment*4 + hermes*2 + blocking*3 - risk*3
```

## Status Legend

- `present`: capability exists in a materially usable local form
- `partial`: capability exists but is incomplete, outdated, or not aligned
- `absent`: no meaningful local support exists
- `legacy`: an older or duplicate local surface exists and may need replacement

## Matrix

| Domain | Capability | tldw_server Surface | tldw_chatbook Surface | Gap Type | Impact | Alignment | Hermes | Blocking | Risk | Priority | Phase | Notes |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---|
| Chat | Conversation sessions, history, and message flow | `chat.py`, `messages.py`, `chat_loop.py`, `chat_workflows.py` | `UI/Chat_Window_Enhanced.py`, `UI/Screens/chat_screen.py`, `Chat/`, `DB/ChaChaNotes_DB.py`, `app.py` | Missing interoperability support | - | - | - | - | - | - | Phase 1 / 2 | Local chat is mature, but the server has broader API/session semantics and newer chat-loop behavior. |
| Chat | Chat dictionaries, document context, and chat-side helpers | `chat_dictionaries.py`, `chat_documents.py`, `chat_grammars.py` | `Character_Chat/Chat_Dictionary_Lib.py`, `UI/CCP_Modules/`, `Tools/rag_search_tool.py`, `RAG_Search/` | Already present but structurally incompatible | - | - | - | - | - | - | Phase 2 | Dictionary and context features exist locally, but the server shape appears broader and more explicit. |
| Characters | Character catalog, chat sessions, and character messages | `characters_endpoint.py`, `character_chat_sessions.py`, `character_messages.py` | `UI/Conv_Char_Window.py`, `UI/Screens/ccp_screen.py`, `Character_Chat/Character_Chat_Lib.py`, `DB/ChaChaNotes_DB.py` | Missing interoperability support | - | - | - | - | - | - | Phase 1 / 2 | Strong local support exists; likely needs data-shape and workflow alignment more than feature invention. |
| Notes | Notes, workspace notes, and notes graph | `notes.py`, `workspaces.py`, `notes_graph.py` | `UI/Notes_Window.py`, `UI/Screens/notes_screen.py`, `Notes/`, `DB/ChaChaNotes_DB.py`, `state/notes_state.py` | Missing interoperability support | - | - | - | - | - | - | Phase 1 / 2 | Local notes are substantial and offline-first; workspace semantics need explicit mapping. |
| Media | Files, ingestion sources, reading, and reading progress | `files.py`, `ingestion_sources.py`, `reading.py`, `media/reading_progress.py` | `UI/MediaWindow_v2.py`, `UI/MediaIngestWindowRebuilt.py`, `UI/Screens/media_screen.py`, `UI/Screens/media_ingest_screen.py`, `Local_Ingestion/`, `DB/Client_Media_DB_v2.py` | Already present but structurally incompatible | - | - | - | - | - | - | Phase 2 / 3 | Media exists locally, but server-side ingestion/file abstractions are newer and broader. |
| RAG | Unified retrieval, chunking, templates, and chat documents | `rag_unified.py`, `chunking.py`, `chunking_templates.py`, `chat_documents.py` | `RAG_Search/`, `Embeddings/`, `UI/SearchRAGWindow.py`, `UI/Screens/search_screen.py` | Already present but structurally incompatible | - | - | - | - | - | - | Phase 3 | Local RAG is extensive, but likely diverges in pipelines, chunking templates, and retriever assumptions. |
| Prompts / Chatbooks | Prompt library, prompt workflows, and chatbooks | `prompts.py`, `chatbooks.py`, `writing.py`, `writing_manuscripts.py` | `Prompt_Management/`, `DB/Prompts_DB.py`, `Chatbooks/`, `UI/Chatbooks_Window_Improved.py`, `UI/Screens/chatbooks_screen.py` | Already present but structurally incompatible | - | - | - | - | - | - | Phase 1 / 2 | Chatbooks are a likely early interoperability seam because local import/export already exists. |
| Evals / Study | Evals, datasets, benchmarks, flashcards, and quizzes | `evaluations/evaluations_unified.py`, `evaluations/evaluations_crud.py`, `evaluations/evaluations_datasets.py`, `evaluations/evaluations_benchmarks.py`, `flashcards.py`, `quizzes.py` | `UI/Evals/evals_window_v3.py`, `UI/Screens/evals_screen.py`, `UI/Study_Window.py`, `UI/Screens/study_screen.py`, `Evals/` | Missing interoperability support | - | - | - | - | - | - | Phase 3 | Local evals and study surfaces exist, but server eval endpoints are broader and more structured. |
| MCP / Tools / Skills | MCP unified access, tools catalog, tool execution, and skills | `mcp_unified_endpoint.py`, `tools.py`, `skills.py` | `MCP/client.py`, `MCP/server.py`, `MCP/tools.py`, `MCP/resources.py`, `MCP/prompts.py`, `Tools/tool_executor.py`, `UI/Tools_Settings_Window.py` | Already present but structurally incompatible | - | - | - | - | - | - | Phase 3 / 4 | Local MCP is stdio-oriented today; server MCP is HTTP/WebSocket/JWT-oriented and needs a compatibility bridge. |
| Companion / Persona | Companion, persona, and personalization-adjacent workflows | `companion.py`, `persona.py`, `personalization.py` | `Character_Chat/`, `UI/Conv_Char_Window.py`, no obvious dedicated companion surface | Missing feature | - | - | - | - | - | - | Phase 4 | Likely secondary unless it unlocks a clearly valuable offline workflow. |
| UX Overlay | Tool progress, session ergonomics, model controls, approvals, background-task visibility | `hermes-agent/RELEASE_v0.8.0.md`, `hermes-agent/model_tools.py`, `hermes-agent/cli.py`, `hermes-agent/hermes_cli/` | `UI/Chat_Window_Enhanced.py`, `UI/Screens/chat_screen.py`, `Widgets/`, `app.py` | Hermes-inspired UX enhancement only | - | - | - | - | - | - | Phase 4 | Should remain secondary to parity and only promote when it materially improves `tldw_chatbook`. |
