# tldw_chatbook Data Compatibility Map

## Purpose

Track how local `tldw_chatbook` entities relate to `tldw_server` models and formats so future interoperability and sync work can build on stable assumptions.

## Entity Map

| Entity | Local Source | Server Source | ID Strategy | Timestamp Strategy | Metadata / Versioning | Delete Semantics | Import/Export Format | Sync-Safe Later? | Notes |
|---|---|---|---|---|---|---|---|---|---|
| Conversations | `DB/ChaChaNotes_DB.py`, `Chat/`, `UI/Chat_Window_Enhanced.py` | `chat.py`, `messages.py`, `chat_loop.py`, `chat_workflows.py` | TBD during matrix scoring | TBD during matrix scoring | TBD during matrix scoring | TBD during matrix scoring | Local DB plus possible chatbook export paths | TBD | Likely a Phase 1 compatibility seam because local and server chat both appear first-class. |
| Messages | `DB/ChaChaNotes_DB.py`, `Widgets/Chat_Widgets/chat_message_enhanced.py` | `messages.py`, `character_messages.py`, `chat.py` | TBD | TBD | TBD | TBD | Local DB rows, possible export in chatbooks | TBD | Need to compare tool-call/tool-result storage semantics explicitly. |
| Notes | `Notes/Notes_Library.py`, `Notes/sync_engine.py`, `DB/ChaChaNotes_DB.py` | `notes.py`, `workspaces.py`, `notes_graph.py` | TBD | TBD | TBD | TBD | Local note files and DB-backed notes | TBD | Workspace membership and graph edges are likely compatibility pressure points. |
| Characters | `Character_Chat/Character_Chat_Lib.py`, `Character_Chat/ccv3_parser.py`, `DB/ChaChaNotes_DB.py` | `characters_endpoint.py`, `character_chat_sessions.py`, `character_messages.py`, `persona.py` | TBD | TBD | TBD | TBD | Character cards plus local DB | TBD | Need to compare card format, placeholder handling, and session/message linkage. |
| Prompts | `Prompt_Management/`, `DB/Prompts_DB.py` | `prompts.py`, `writing.py`, `writing_manuscripts.py` | TBD | TBD | TBD | TBD | Local prompt DB and interop helpers | TBD | Prompt metadata/versioning likely matters for server alignment. |
| Chatbooks | `Chatbooks/chatbook_models.py`, `Chatbooks/chatbook_importer.py`, `Chatbooks/chatbook_creator.py` | `chatbooks.py` | TBD | TBD | TBD | TBD | Local chatbook manifests and packaged content | TBD | This is a strong early interoperability candidate because the local format already exists. |
| Media | `DB/Client_Media_DB_v2.py`, `Local_Ingestion/`, `UI/MediaWindow_v2.py` | `files.py`, `ingestion_sources.py`, `reading.py`, `media/reading_progress.py` | TBD | TBD | TBD | TBD | Local media DB and ingestion flows | TBD | Need to compare file IDs, derived artifacts, and reading-progress semantics. |
| Embeddings / Retrieval Artifacts | `Embeddings/`, `RAG_Search/`, `RAG_Search/simplified/` | `rag_unified.py`, `chunking.py`, `chunking_templates.py`, `chat_documents.py` | TBD | TBD | TBD | TBD | Local vector/FTS/chunking outputs | TBD | Chunk/template naming and collection layout likely differ. |
| Evaluations / Study Artifacts | `Evals/`, `UI/Evals/evals_window_v3.py`, `UI/Study_Window.py` | `evaluations/evaluations_unified.py`, `evaluations/evaluations_crud.py`, `evaluations/evaluations_datasets.py`, `flashcards.py`, `quizzes.py` | TBD | TBD | TBD | TBD | Local eval artifacts and study surfaces | TBD | Need to determine whether local study objects map cleanly onto server flashcard/quiz resources. |
