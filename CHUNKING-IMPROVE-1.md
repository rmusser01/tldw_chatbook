# Template-Based Chunking with Per-Document Late Chunking Plan

## Overview
Implement a system that allows per-document chunking configurations with late chunking during RAG retrieval, plus parent document inclusion logic.

## Phase 1: Database Schema Extensions

### 1.1 Add Chunking Configuration Storage
- Add `chunking_config` JSON column to `Media` table
- Structure: `{"template": "template_name", "method": "hierarchical", "chunk_size": 400, "overlap": 100, "custom_params": {}}`
- Migration script to add columns with sensible defaults

### 1.2 Create Chunking Templates Table
```sql
CREATE TABLE ChunkingTemplates (
    id INTEGER PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    description TEXT,
    template_json TEXT NOT NULL,
    is_system BOOLEAN DEFAULT 0,
    created_at DATETIME,
    updated_at DATETIME
);
```

## Phase 2: Late Chunking Implementation

### 2.1 Modify RAG Pipeline
- Change from returning full documents to implementing late chunking
- Create `LateChunkingService` that:
  1. Retrieves matched documents
  2. Checks for per-document chunking config
  3. Falls back to default if none exists
  4. Performs chunking on-demand
  5. Caches chunks temporarily for session

### 2.2 Chunking Decision Flow
```python
def get_chunks_for_document(media_id, default_config):
    # 1. Check if document has custom config
    doc_config = get_chunking_config(media_id)
    
    # 2. Use document config or fall back to default
    config = doc_config or default_config
    
    # 3. Check if pre-chunked exists and matches config
    if has_matching_chunks(media_id, config):
        return get_existing_chunks(media_id)
    
    # 4. Perform late chunking
    return perform_late_chunking(media_id, config)
```

## Phase 3: Parent Document Inclusion (RAG Pipeline Feature)

### 3.1 RAG Pipeline Configuration
- Add to RAG pipeline config:
  ```python
  rag_config = {
      "include_parent_docs": True,
      "parent_size_threshold": 5000,  # chars
      "parent_inclusion_strategy": "size_based",  # or "always", "never"
      "max_context_size": 16000
  }
  ```
- Store in existing RAG config profiles

### 3.2 Context Assembly in Pipeline
```python
class ContextAssembler:
    def __init__(self, rag_config):
        self.include_parents = rag_config.get("include_parent_docs", False)
        self.parent_threshold = rag_config.get("parent_size_threshold", 5000)
        self.strategy = rag_config.get("parent_inclusion_strategy", "size_based")
        self.max_context = rag_config.get("max_context_size", 16000)
    
    def assemble_context(self, chunks):
        context = []
        size = 0
        
        # 1. Add matched chunks
        for chunk in chunks:
            if size + len(chunk.text) <= self.max_context:
                context.append(chunk)
                size += len(chunk.text)
        
        # 2. Try to add parent based on strategy
        if self.include_parents:
            for chunk in chunks:
                parent = get_parent_document(chunk.media_id)
                
                if self.strategy == "always":
                    add_parent = True
                elif self.strategy == "size_based":
                    add_parent = parent.size < self.parent_threshold
                else:  # "never"
                    add_parent = False
                
                if add_parent and size + parent.size <= self.max_context:
                    context.append(parent)
                    size += parent.size
                    break
        
        return context
```

### 3.3 Update RAG Config UI
- Add parent inclusion settings to RAG Search Window settings
- Options:
  - Enable/disable parent inclusion
  - Size threshold slider
  - Strategy dropdown (always/size_based/never)
  - Preview of how many docs would include parents

## Phase 4: UI Integration

### 4.1 Media Window Enhancement
- Add "Chunking Configuration" section in media details
- Template selector dropdown
- Advanced settings expandable section
- Preview chunking results button

### 4.2 RAG Search Window Updates
- Show chunk source indicator
- Display when parent document is included
- Settings panel for parent inclusion
- Real-time preview of context assembly

## Phase 5: Template Management

### 5.1 Built-in Templates
- "academic_paper": Structural chunking with abstract/section preservation
- "code_documentation": Code-aware chunking
- "conversational": Semantic chunking for chat logs
- "general": Default balanced approach

### 5.2 Custom Template Creation
- JSON editor for advanced users
- Visual template builder for common patterns
- Template inheritance support

## Implementation Order

1. **Database changes** - Add columns and tables
2. **Late chunking service** - Core functionality
3. **Update RAG pipeline** - Integrate late chunking
4. **Parent document logic in pipeline** - Add to RAG config
5. **Basic UI** - Configuration in media window
6. **RAG config UI** - Parent inclusion settings
7. **Template system** - Full template management

## Benefits

1. **Flexibility**: Each document can have optimal chunking
2. **Performance**: Only chunk what's needed, when needed
3. **Context**: Smart parent inclusion improves understanding
4. **Pipeline Control**: Parent inclusion is a retrieval-time decision
5. **User Control**: Fine-tune per document or use defaults
6. **No DB Changes for Parents**: Parent inclusion logic stays in the pipeline