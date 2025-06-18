# RAG Re-Architecture Implementation Plan

## Overview
This document outlines the detailed implementation plan for re-architecting the RAG (Retrieval-Augmented Generation) pipeline in the tldw_chatbook project, based on the approved design in RAG-REARCHITECT.md.

## Goals
1. Break down the monolithic 1604-line `Unified_RAG_v2.py` into modular components
2. Integrate with the existing TOML configuration system
3. Eliminate code duplication and improve maintainability
4. Fix resource leaks and performance issues
5. Create a testable, extensible architecture

## Implementation Phases

### Phase 1: Setup and Preparation (Days 1-2)
**Goal**: Create the new package structure without breaking existing functionality

#### Tasks:
1. **Create new package structure**
   ```
   tldw_chatbook/Services/rag_service/
   ├── __init__.py
   ├── app.py              # Main RAG application class
   ├── config.py           # Configuration management
   ├── retrieval.py        # Document retrieval logic
   ├── processing.py       # Document processing and ranking
   ├── generation.py       # Response generation
   ├── utils.py           # Utility functions
   └── tests/
       ├── __init__.py
       ├── test_config.py
       ├── test_retrieval.py
       ├── test_processing.py
       └── test_generation.py
   ```

2. **Set up comprehensive test suite**
   - Create test fixtures from existing RAG functionality
   - Write integration tests that verify current behavior
   - Set up test databases (in-memory SQLite)
   - Create mock data for testing

3. **Document current API surface**
   - Identify all public functions currently used by the app
   - Document their signatures and expected behavior
   - Create a compatibility checklist

### Phase 2: Core Infrastructure (Days 3-4)
**Goal**: Build the foundational components

#### Tasks:
1. **Implement `config.py`**
   - Create `RAGConfig` class that reads from TOML
   - Add validation for all configuration options
   - Support environment variable overrides
   - Implement configuration hot-reloading capability

2. **Implement `app.py` - RAGApplication class**
   ```python
   class RAGApplication:
       def __init__(self, config_path: Optional[Path] = None):
           self.config = RAGConfig.load(config_path)
           self.retriever = None
           self.processor = None
           self.generator = None
           self._initialize_components()
   ```

3. **Create base interfaces**
   - Define abstract base classes for retrieval strategies
   - Create interfaces for processing pipelines
   - Establish contracts for generation methods

### Phase 3: Migration - Retrieval Layer (Days 5-6)
**Goal**: Extract and refactor all retrieval logic

#### Tasks:
1. **Implement `retrieval.py`**
   - Extract database query functions
   - Create retrieval strategies:
     - `MediaDBRetriever`
     - `ChatHistoryRetriever`
     - `NotesRetriever`
     - `CharacterCardRetriever`
   - Implement the strategy pattern for retrieval

2. **Optimize database queries**
   - Push filtering to SQL level (fix character card loading)
   - Add proper indexing hints
   - Implement query result caching

3. **Add retrieval tests**
   - Unit tests for each retriever
   - Performance benchmarks
   - Edge case handling

### Phase 4: Migration - Processing Layer (Days 7-8)
**Goal**: Consolidate all document processing logic

#### Tasks:
1. **Implement `processing.py`**
   - Extract the unified `combine_and_rerank_results` function
   - Move embedding generation logic
   - Implement ranking strategies:
     - FlashRank integration
     - Custom scoring algorithms
   - Add result deduplication

2. **Fix resource management**
   - Replace file-based chat history with in-memory handling
   - Implement proper context managers
   - Add resource cleanup

3. **Add processing tests**
   - Test ranking algorithms
   - Verify deduplication
   - Test resource cleanup

### Phase 5: Migration - Generation Layer (Days 9-10)
**Goal**: Extract response generation logic

#### Tasks:
1. **Implement `generation.py`**
   - Extract LLM interaction code
   - Create generation strategies
   - Implement streaming support
   - Add error handling and retries

2. **Integrate with existing LLM infrastructure**
   - Use existing LLM call abstractions
   - Maintain compatibility with current prompt templates
   - Support all existing LLM providers

3. **Add generation tests**
   - Mock LLM responses
   - Test error handling
   - Verify streaming functionality

### Phase 6: Integration and Migration (Days 11-12)
**Goal**: Wire everything together and migrate existing code

#### Tasks:
1. **Update existing code to use new service**
   - Find all imports of `Unified_RAG_v2`
   - Update to use new `RAGApplication`
   - Maintain backward compatibility where needed

2. **Create migration guide**
   - Document API changes
   - Provide code examples
   - Create deprecation warnings

3. **Performance testing**
   - Benchmark old vs new implementation
   - Memory usage analysis
   - Response time comparisons

### Phase 7: Rollout and Monitoring (Days 13-14)
**Goal**: Deploy the new architecture safely

#### Tasks:
1. **Gradual rollout**
   - Add feature flags for new/old RAG
   - Start with non-critical paths
   - Monitor for issues

2. **Logging and metrics**
   - Add comprehensive logging
   - Track performance metrics
   - Set up alerts for anomalies

3. **Documentation**
   - Update developer documentation
   - Create architecture diagrams
   - Write troubleshooting guide

## Testing Strategy

### Unit Tests
- Each module gets comprehensive unit tests
- Mock external dependencies
- Test edge cases and error conditions

### Integration Tests
- Test complete RAG pipelines
- Use real (test) databases
- Verify end-to-end functionality

### Performance Tests
- Benchmark retrieval speed
- Measure memory usage
- Test with large datasets

### Compatibility Tests
- Ensure existing functionality works
- Test all supported databases
- Verify all RAG pipeline variants

## Risk Mitigation

### Technical Risks
1. **Breaking existing functionality**
   - Mitigation: Comprehensive test suite before changes
   - Feature flags for gradual rollout

2. **Performance regression**
   - Mitigation: Benchmark before and after
   - Keep old code available for rollback

3. **Data consistency issues**
   - Mitigation: Use transactions properly
   - Add data validation

### Process Risks
1. **Scope creep**
   - Mitigation: Stick to the plan
   - Defer enhancements to Phase 2

2. **Integration challenges**
   - Mitigation: Early integration tests
   - Close collaboration with team

## Success Criteria

1. **Functionality**: All existing RAG features work correctly
2. **Performance**: No regression in response times
3. **Maintainability**: Code coverage >80%, clear module boundaries
4. **Extensibility**: Easy to add new retrieval sources
5. **Resource Usage**: No memory leaks, proper cleanup

## Timeline Summary
- **Week 1**: Setup, Infrastructure, and Retrieval (Phases 1-3)
- **Week 2**: Processing, Generation, and Integration (Phases 4-6)
- **Week 3**: Rollout, monitoring, and stabilization (Phase 7)

Total estimated time: 14-15 working days

## Next Steps
1. Review and approve this plan
2. Set up the new package structure
3. Begin writing tests for current functionality
4. Start Phase 1 implementation

## Notes
- Keep the old `Unified_RAG_v2.py` during migration for reference
- Consider creating a `rag_service_v1` package for complete isolation
- Regular check-ins with the team to ensure alignment
- Document all decisions and deviations from the plan