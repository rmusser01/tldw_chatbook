# RAG Module Import Analysis

## Files that Import from RAG_Search Module

### Core Application Files

1. **tldw_chatbook/Event_Handlers/Chat_Events/chat_rag_integration.py**
   - `from tldw_chatbook.RAG_Search.simplified import (...)`

2. **tldw_chatbook/Event_Handlers/Chat_Events/chat_rag_events.py**
   - `from ...RAG_Search.pipeline_builder_simple import execute_pipeline, BUILTIN_PIPELINES`
   - `from ...RAG_Search.pipeline_loader import get_pipeline_loader`
   - `from ...RAG_Search.simplified import RAGService, create_config_for_collection`
   - `from ...RAG_Search.pipeline_builder_simple import get_pipeline`
   - `from ...RAG_Search.pipeline_functions_simple import format_as_context`
   - `from ...RAG_Search.pipeline_types import SearchResult`

3. **tldw_chatbook/UI/SearchRAGWindow.py**
   - `from tldw_chatbook.RAG_Search.simplified import (...)`
   - `from tldw_chatbook.RAG_Search.Services.embeddings_service import EmbeddingsService`
   - `from ..RAG_Search.pipeline_integration import get_pipeline_manager`

4. **tldw_chatbook/UI/SearchRAGWindow_backup.py**
   - `from ..RAG_Search.Services import EmbeddingsService, ChunkingService, IndexingService`

5. **tldw_chatbook/Widgets/settings_sidebar.py**
   - `from ..RAG_Search.pipeline_integration import get_pipeline_manager`
   - `from ..RAG_Search.pipeline_builder_simple import get_pipeline, BUILTIN_PIPELINES`

6. **tldw_chatbook/Tools/rag_search_tool.py**
   - `from ..RAG_Search.simplified import RAGService, create_config_for_collection`

7. **tldw_chatbook/MCP/tools.py**
   - `from ..RAG_Search.simplified.search_service import SimplifiedRAGSearchService`

8. **tldw_chatbook/MCP/prompts.py**
   - `from ..RAG_Search.simplified.search_service import SimplifiedRAGSearchService`

### Ingestion Modules

9. **tldw_chatbook/Local_Ingestion/PDF_Processing_Lib.py**
   - `from ..RAG_Search.chunking_service import improved_chunking_process`

10. **tldw_chatbook/Local_Ingestion/Book_Ingestion_Lib.py**
    - `from ..RAG_Search.chunking_service import improved_chunking_process` (multiple occurrences)

11. **tldw_chatbook/Local_Ingestion/Image_Processing_Lib.py**
    - `from ..RAG_Search.chunking_service import improved_chunking_process`

12. **tldw_chatbook/Local_Ingestion/audio_processing.py**
    - `from ..RAG_Search.chunking_service import ChunkingService`

### Embeddings Integration

13. **tldw_chatbook/Embeddings/Chroma_Lib.py**
    - `from tldw_chatbook.RAG_Search.Services.embeddings_compat import EmbeddingFactoryCompat`

### Internal RAG Module Dependencies

14. **tldw_chatbook/RAG_Search/config_profiles.py**
    - `from .simplified.config import RAGConfig`
    - `from .reranker import RerankingConfig`

15. **tldw_chatbook/RAG_Search/pipeline_integration.py**
    - `from .simplified.config import RAGConfig`
    - `from .pipeline_loader import get_pipeline_loader, get_pipeline_function`

16. **tldw_chatbook/RAG_Search/query_expansion.py**
    - `from tldw_chatbook.RAG_Search.simplified.config import QueryExpansionConfig`

17. **tldw_chatbook/RAG_Search/reranker.py**
    - `from .simplified.vector_store import SearchResult, SearchResultWithCitations`

18. **tldw_chatbook/RAG_Search/parallel_processor.py**
    - `from .simplified.data_models import IndexingResult`

19. **tldw_chatbook/RAG_Search/enhanced_chunking_service.py**
    - `from .chunking_service import ChunkingService, ChunkingError`
    - `from .table_serializer import TableProcessor, serialize_table`

20. **tldw_chatbook/RAG_Search/pipeline_loader.py**
    - `from .pipeline_builder_simple import execute_pipeline`
    - `from .pipeline_builder_simple import BUILTIN_PIPELINES`

21. **tldw_chatbook/RAG_Search/pipeline_builder_simple.py**
    - `from .pipeline_types import SearchResult, StepType, PipelineContext`
    - `from .pipeline_functions_simple import (...)`

22. **tldw_chatbook/RAG_Search/__init__.py**
    - `from .chunking_service import ChunkingService`

### Simplified Submodule Internal Dependencies

23. **tldw_chatbook/RAG_Search/simplified/rag_service.py**
    - `from ..chunking_service import ChunkingService`

24. **tldw_chatbook/RAG_Search/simplified/enhanced_rag_service.py**
    - `from ..enhanced_chunking_service import EnhancedChunkingService`

25. **tldw_chatbook/RAG_Search/simplified/enhanced_rag_service_v2.py**
    - `from ..reranker import create_reranker, BaseReranker, RerankingConfig`

26. **tldw_chatbook/RAG_Search/simplified/enhanced_indexing_helpers.py**
    - `from ..enhanced_chunking_service import EnhancedChunkingService, StructuredChunk`

## Summary of Key RAG Modules in Use

### Most Used Modules:
1. **chunking_service.py** - Used by all ingestion modules and core RAG functionality
   - `ChunkingService` class
   - `improved_chunking_process` function

2. **simplified/** submodule - Main RAG interface
   - `RAGService`
   - `create_config_for_collection`
   - Configuration classes
   - Search service

3. **pipeline_builder_simple.py** and **pipeline_loader.py** - Pipeline execution
   - Used by chat events and UI components
   - `execute_pipeline`, `BUILTIN_PIPELINES`, `get_pipeline`

4. **Services/** submodule
   - `EmbeddingsService` - Used by UI and embeddings integration
   - `embeddings_compat` - Used by Chroma integration

5. **enhanced_chunking_service.py** - Advanced chunking features
   - Used internally by simplified RAG modules

6. **table_serializer.py** - Table processing
   - Used by enhanced chunking service

7. **reranker.py** - Search result reranking
   - Used by enhanced RAG service

8. **query_expansion.py** - Query enhancement
   - Has config dependency

9. **pipeline_integration.py** - Pipeline management
   - Used by UI components

### Files That Are Likely Unused:
Based on the import analysis, any RAG files not listed above appear to have no direct imports from the main application code. However, they might be:
- Imported dynamically
- Used by tests only
- Part of optional features
- Legacy code

### Test Files
Multiple test files import RAG modules, primarily in:
- `Tests/RAG/`
- `Tests/RAG_Search/`
- `Tests/test_enhanced_rag.py`
- `Tests/test_smoke.py`