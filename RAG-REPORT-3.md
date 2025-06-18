# RAG Integration Report 3 - Chat Window RAG Settings Implementation

## Date: 2025-06-17

## Overview
This report documents the implementation of RAG (Retrieval-Augmented Generation) settings in the chat window's left sidebar, replacing the placeholder "Media Settings" section with functional RAG controls.

## Changes Performed

### 1. UI Modifications

#### 1.1 Settings Sidebar Update (`tldw_chatbook/Widgets/settings_sidebar.py`)
- **Replaced**: "Media Settings" placeholder section
- **With**: Comprehensive "RAG Settings" section containing:
  - Enable/disable toggles for RAG functionality
  - Source selection checkboxes
  - Parameter configuration inputs
  - Re-ranking options
  - Advanced chunking settings

#### 1.2 Sidebar Organization
- Split "Search & Tools Settings" into two separate sections:
  - "Tool Settings" - For LLM tool usage configuration
  - "Search & Templates" - For chat template search and application
- This improved organization and made room for dedicated RAG settings

### 2. Backend Implementation

#### 2.1 RAG Event Handler (`tldw_chatbook/Event_Handlers/Chat_Events/chat_rag_events.py`)
Created a new module with three main functions:

1. **`perform_plain_rag_search()`**
   - Implements BM25 (FTS5) search across selected sources
   - Supports searching: Media items, Conversations, and Notes
   - Includes optional re-ranking with FlashRank
   - Enforces context length limits to prevent overflow

2. **`perform_full_rag_pipeline()`**
   - Placeholder for future embeddings-based RAG
   - Currently falls back to plain RAG
   - Will eventually integrate with ChromaDB/embeddings

3. **`get_rag_context_for_chat()`**
   - Main interface function called from chat events
   - Reads UI settings and determines RAG approach
   - Returns formatted context string for LLM inclusion

#### 2.2 Chat Integration (`tldw_chatbook/Event_Handlers/Chat_Events/chat_events.py`)
- Modified `handle_chat_send_button_pressed()` to:
  - Import and call RAG functionality before sending message
  - Prepend RAG context to user message when enabled
  - Handle errors gracefully without interrupting chat flow

### 3. Search Tab Fixes
- Fixed crash when clicking Search tab by:
  - Creating `search_events.py` handler
  - Uncommenting search handlers in `app.py`
  - Adding proper error handling for missing dependencies

## Design Decisions

### 1. Dual RAG Modes
- **Full RAG**: For future implementation with embeddings and vector search
- **Plain RAG**: Immediate functionality using BM25/FTS5 search
- Rationale: Provides immediate value while allowing for future enhancement

### 2. Context Management
- **Character limit**: Prevents context from overwhelming the LLM's token limit
- **Smart truncation**: Cuts off at word boundaries with ellipsis
- **Metadata inclusion**: Optional to balance informativeness vs. context size

### 3. Source Flexibility
- **Multiple sources**: Media, Conversations, and Notes can be searched independently
- **Checkbox controls**: Users can enable/disable sources as needed
- **Unified results**: All sources are merged and ranked together

### 4. Re-ranking Architecture
- **Optional re-ranking**: Can be disabled for performance
- **Multiple providers**: Support for FlashRank (local) and Cohere (API)
- **Score-based sorting**: Results are ordered by relevance score

### 5. UI/UX Decisions
- **Collapsible section**: Keeps advanced options hidden by default
- **Sensible defaults**: Pre-configured for optimal performance
- **Clear labeling**: Each option has descriptive labels and placeholders

## Technical Considerations

### 1. Dependency Management
- Graceful handling of missing dependencies (FlashRank, embeddings)
- Features automatically disable if dependencies unavailable
- Clear error messages guide users to install requirements

### 2. Performance Optimization
- Asynchronous database queries prevent UI blocking
- Result limiting (top_k) prevents excessive processing
- Re-ranking only processes necessary results

### 3. Error Handling
- Try-catch blocks at all integration points
- Fallback behavior ensures chat continues working
- Detailed logging for debugging

## Configuration Options

### RAG Enable Options
| Setting | Default | Description |
|---------|---------|-------------|
| Perform RAG | False | Enable full RAG pipeline (future) |
| Perform Plain RAG | False | Enable BM25-only search |

### RAG Parameters
| Setting | Default | Description |
|---------|---------|-------------|
| Top K Results | 5 | Number of results to retrieve |
| Max Context Length | 10000 | Maximum characters for context |
| Enable Re-ranking | True | Apply re-ranking to results |
| Re-ranker Model | flashrank | Which re-ranking model to use |

### Advanced Options
| Setting | Default | Description |
|---------|---------|-------------|
| Chunk Size | 400 | Words per chunk |
| Chunk Overlap | 100 | Overlap between chunks |
| Include Metadata | True | Add metadata to context |

## Integration Points

1. **Chat Window**: RAG settings in left sidebar
2. **Chat Events**: RAG context injection before API call
3. **Database Layer**: Search functionality across all DBs
4. **Search Tab**: Foundation for dedicated RAG interface

## Testing

- Created comprehensive test scripts to verify:
  - UI elements properly created
  - Settings properly read
  - RAG functionality callable
  - No regression in existing features

## Future Enhancements

1. **Full RAG Pipeline**: Implement embeddings-based semantic search
2. **Search Tab RAG**: Dedicated interface for RAG exploration
3. **Caching**: Cache RAG results for repeated queries
4. **Analytics**: Track RAG usage and effectiveness
5. **Custom Embeddings**: Support for user-provided embedding models

## Conclusion

This implementation provides a solid foundation for RAG functionality in the chat interface. The modular design allows for easy enhancement while providing immediate value through BM25 search. The UI is intuitive and the backend is extensible, setting the stage for more advanced RAG features.