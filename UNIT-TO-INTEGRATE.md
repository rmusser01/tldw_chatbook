# Unit to Integration Test Conversion Guide

## Overview

This document provides a comprehensive analysis of all test files in the tldw_chatbook codebase and identifies which tests need to be converted from unit tests to proper integration tests (or vice versa). This analysis was conducted after discovering that many "integration" tests were actually unit tests due to extensive mocking.

## Summary of Issues

After analyzing 90+ test files:
- **~30 files** are incorrectly categorized (claim to be one type but are actually another)
- **~15 files** have already been addressed in previous work
- **~45 files** still need to be fixed or properly marked

## Files Requiring Changes

### Priority 1: Database Tests (Claim Unit, Are Integration)

These files claim to be unit tests but actually test real database operations:

#### **Tests/DB/test_pagination.py**
- **Current**: Claims unit test in docstring
- **Reality**: Uses real MediaDatabase with in-memory SQLite
- **Action**: 
  - Update docstring to say "Integration tests"
  - Add `pytestmark = pytest.mark.integration`
  - Remove unused mock imports

#### **Tests/DB/test_rag_indexing_db.py**
- **Current**: Docstring says "Unit tests for RAGIndexingDB"
- **Reality**: Tests real database operations
- **Action**:
  - Update docstring to "Integration tests"
  - Add `pytestmark = pytest.mark.integration`

#### **Tests/DB/test_search_history_db.py**
- **Current**: Docstring says "Unit tests"
- **Reality**: Uses real SearchHistoryDB
- **Action**:
  - Update docstring
  - Add integration marker

#### **Tests/Prompts_DB/test_prompts_db_pytest.py**
- **Current**: No explicit marking
- **Reality**: Tests real database
- **Action**: Add `pytestmark = pytest.mark.integration`

### Priority 2: Event Handler Tests (Use Heavy Mocking)

These files extensively mock the app and should be clearly marked as unit tests:

#### **Tests/Event_Handlers/Chat_Events/test_chat_events_sidebar.py**
- **Current**: No clear marking
- **Reality**: Mocks entire app instance
- **Action**: 
  - Add `pytestmark = pytest.mark.unit`
  - Add comment explaining why mocking is used
  - Consider creating integration version

#### **Tests/Event_Handlers/Chat_Events/test_chat_image_events.py**
- **Current**: No marking
- **Reality**: Uses MagicMock for app
- **Action**: Same as above

#### **Tests/Event_Handlers/Chat_Events/test_chat_streaming_events.py**
- **Current**: No marking
- **Reality**: Extensive app mocking
- **Action**: Same as above

### Priority 3: LLM Management Tests

#### **Tests/LLM_Management/test_llm_management_events.py**
- **Current**: No marking
- **Reality**: Mocks app and all UI components
- **Action**:
  - Add `pytestmark = pytest.mark.unit`
  - Create integration test with real app

#### **Tests/LLM_Management/test_mlx_lm.py**
- **Current**: No marking
- **Reality**: Heavy mocking of MLX components
- **Action**: Mark as unit test

### Priority 4: Mixed Test Files (Need Splitting)

#### **Tests/integration/test_core_functionality_without_optional_deps.py**
- **Current**: In integration folder
- **Reality**: Mix of real tests and mocked tests
- **Action**:
  - Split into two files:
    - `test_core_imports_unit.py` (for import tests)
    - `test_core_functionality_integration.py` (for real tests)

#### **Tests/integration/test_file_operations_with_validation.py**
- **Current**: In integration folder
- **Reality**: Mocks database methods
- **Action**:
  - Either convert to use real database
  - Or move to unit tests folder

### Priority 5: UI Tests (Need Proper Integration Versions)

All UI tests currently use heavy mocking. While this is appropriate for unit tests, we need integration versions:

#### Files needing integration versions:
- `Tests/UI/test_chat_window_tooltips.py`
- `Tests/UI/test_command_palette_basic.py`
- `Tests/UI/test_command_palette_providers.py`
- `Tests/UI/test_ingest_window.py`
- `Tests/UI/test_search_rag_window.py`
- `Tests/UI/test_tools_settings_window.py`

**Action for each**:
1. Keep current file as unit test (add marker)
2. Create `*_integration.py` version using real Textual app

### Priority 6: RAG Tests Missing Clear Categorization

#### Files needing markers:
- `Tests/RAG/test_cache_service.py` → Add unit marker (uses mocks)
- `Tests/RAG/test_chunking_service.py` → Add unit marker
- `Tests/RAG/test_embeddings_properties.py` → Add unit marker
- `Tests/RAG/test_embeddings_service.py` → Analyze and mark appropriately
- `Tests/RAG/test_full_rag.py` → Add integration marker
- `Tests/RAG/test_modular_rag.py` → Add integration marker
- `Tests/RAG/test_plain_rag.py` → Add integration marker

### Priority 7: Character Chat Tests

#### **Tests/Character_Chat/test_character_chat.py**
- **Current**: Mixed unit and integration tests
- **Reality**: Most use real database
- **Action**: 
  - Add `pytestmark = pytest.mark.integration`
  - Move mocked tests to separate unit test file

## Implementation Guidelines

### 1. For Database Tests
```python
# Change from:
"""Unit tests for XYZ database"""

# To:
"""Integration tests for XYZ database using real SQLite"""
pytestmark = pytest.mark.integration
```

### 2. For Mocked Tests
```python
# Add at module level:
"""Unit tests for XYZ using mocked dependencies"""
pytestmark = pytest.mark.unit
```

### 3. For Creating Integration Versions
```python
# Instead of:
mock_app = MagicMock()
mock_app.query_one.return_value = mock_widget

# Use:
async with app.run_test() as pilot:
    widget = app.query_one("#real-widget")
    await pilot.click(widget)
```

### 4. Test Organization
```
Tests/
├── unit/              # Tests with significant mocking
│   └── event_handlers/
│       └── test_chat_events_unit.py
├── integration/       # Tests with real components
│   └── event_handlers/
│       └── test_chat_events_integration.py
└── [module]/         # Mixed, but clearly marked
    └── test_*.py
```

## Validation Checklist

For each file:
- [ ] Has appropriate pytest marker (@pytest.mark.unit or @pytest.mark.integration)
- [ ] Docstring matches actual test type
- [ ] File location matches test type (or is clearly marked)
- [ ] Mocking is appropriate for the test type
- [ ] Integration tests use real components where possible

## Already Completed (Do Not Modify)

The following files have already been properly converted:
1. `Tests/Chat/test_chat_sidebar_media_search.py` ✓
2. `Tests/Chat/test_chat_unit_mocked_APIs.py` ✓
3. `Tests/Media_DB/test_sync_client.py` + `test_sync_client_integration.py` ✓
4. `Tests/Notes/test_notes_integration.py` ✓
5. `Tests/Event_Handlers/Chat_Events/test_chat_events_integration.py` ✓
6. `Tests/integration/test_chat_image_integration_real.py` ✓
7. `Tests/RAG_Search/test_embeddings_real_integration.py` ✓
8. `Tests/RAG/test_service_factory_integration.py` ✓
9. `Tests/RAG/test_memory_management_service_integration.py` ✓
10. `Tests/RAG/test_indexing_service_integration.py` ✓

## Estimated Effort

- **High Priority (DB tests)**: 2-3 hours (mostly marking/documentation)
- **Medium Priority (Event handlers, LLM)**: 4-6 hours (need integration versions)
- **Low Priority (UI tests)**: 8-10 hours (complex integration tests needed)
- **Total**: ~20 hours to complete all conversions

## Success Criteria

1. All test files have clear markers indicating their type
2. No test claims to be integration while using extensive mocking
3. No test claims to be unit while using real databases/services
4. Clear separation between unit and integration tests
5. Integration tests provide real confidence in component interactions

## Notes for Implementation

1. **Don't break existing tests** - Add markers and update docs first
2. **Create new files for integration versions** rather than modifying working unit tests
3. **Use the patterns established** in the already-completed files
4. **Test both with and without optional dependencies** where applicable
5. **Document why certain mocking is necessary** in unit tests

This document represents a complete analysis of the test suite as of the current state. Following this guide will result in a properly categorized and comprehensive test suite.