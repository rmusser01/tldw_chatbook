# RAG Testing Scratch Pad

## Working Notes

### Current State Analysis

#### Files to Update:
1. `/Tests/RAG/simplified/test_rag_service_basic.py` - Main service tests
2. `/Tests/test_enhanced_rag.py` - Enhanced features test
3. `/Tests/RAG/test_rag_ui_integration.py` - UI integration
4. `/Tests/RAG/test_rag_dependencies.py` - Dependency checks
5. `/Tests/RAG/simplified/conftest.py` - Test fixtures

#### Key Changes Made:
1. Factory functions:
   - `create_rag_service()` - Main entry point
   - `create_rag_service_from_config()` - Config-based creation
   - Removed `create_rag_service_with_level()`

2. Profiles:
   - bm25_only
   - vector_only
   - hybrid_basic (default)
   - hybrid_enhanced
   - hybrid_full

### Code Snippets

#### Old pattern:
```python
service = RAGService(config=test_rag_config)
```

#### New pattern:
```python
service = create_rag_service(profile_name="hybrid_basic", config=config)
```

#### Profile testing:
```python
@pytest.mark.parametrize("profile", ["bm25_only", "vector_only", "hybrid_basic", "hybrid_enhanced", "hybrid_full"])
def test_profile_creation(profile):
    service = create_rag_service(profile_name=profile)
    assert isinstance(service, EnhancedRAGServiceV2)
```

### Issues Found

1. **Import Issue**: `create_rag_service_with_level` is imported in SearchRAGWindow but doesn't exist anymore
   - Fixed by updating to `create_rag_service`

2. **Test Fixtures**: Need to update conftest.py to use new factory
   - Fixed by updating mock_rag_service fixture to use create_rag_service_from_config

3. **Pipeline Integration**: chat_rag_events.py uses separate pipeline system, not V2 service
   - Fixed by updating get_or_initialize_rag_service to use create_rag_service factory
   - The pipeline system now properly initializes V2 service with profiles

### Test Execution Notes

#### Running specific tests:
```bash
# Run RAG tests only
pytest Tests/RAG -v

# Run with coverage
pytest Tests/RAG --cov=tldw_chatbook.RAG_Search

# Run specific test file
pytest Tests/RAG/simplified/test_rag_service_basic.py -v
```

### Debugging Notes

1. Check if V2 is always returned:
```python
def test_always_returns_v2():
    service = create_rag_service("bm25_only")
    assert type(service).__name__ == "EnhancedRAGServiceV2"
```

2. Profile feature verification:
```python
def test_profile_features(profile_name, expected_features):
    service = create_rag_service(profile_name)
    assert service.enable_parent_retrieval == expected_features["parent_retrieval"]
    assert service.enable_reranking == expected_features["reranking"]
```

### TODO Items

- [ ] Check if chat_rag_events needs updating
- [ ] Verify all imports are updated
- [ ] Test profile switching at runtime
- [ ] Check MCP integration

### Questions to Resolve

1. Should pipeline system use V2 service?
2. How to handle backward compatibility for old configs?
3. Should we keep the old service classes for compatibility?

### Performance Notes

- Profile loading seems fast
- No noticeable performance regression
- V2 with all features disabled ≈ base service performance