# test_optional_deps.py
# Test cases for optional dependency handling
#
import pytest
import sys
from unittest.mock import patch


def test_optional_deps_import():
    """Test that the optional dependencies module can be imported."""
    from tldw_chatbook.Utils.optional_deps import (
        DEPENDENCIES_AVAILABLE, check_dependency, get_safe_import, 
        require_dependency, create_unavailable_feature_handler
    )
    
    assert isinstance(DEPENDENCIES_AVAILABLE, dict)
    assert callable(check_dependency)
    assert callable(get_safe_import)
    assert callable(require_dependency)
    assert callable(create_unavailable_feature_handler)


def test_unavailable_feature_handler():
    """Test that unavailable feature handlers work correctly."""
    from tldw_chatbook.Utils.optional_deps import create_unavailable_feature_handler
    
    handler = create_unavailable_feature_handler('test_feature', 'pip install test')
    
    with pytest.raises(ImportError) as exc_info:
        handler()
    
    assert 'test_feature' in str(exc_info.value)
    assert 'pip install test' in str(exc_info.value)


def test_dependency_check_caching():
    """Test that dependency checks are cached."""
    from tldw_chatbook.Utils.optional_deps import check_dependency, DEPENDENCIES_AVAILABLE
    
    # Clear cache for this test
    test_key = 'test_nonexistent_module'
    if test_key in DEPENDENCIES_AVAILABLE:
        del DEPENDENCIES_AVAILABLE[test_key]
    
    # First call should check and cache the result
    result1 = check_dependency('nonexistent_module_for_testing', test_key)
    assert result1 is False
    assert DEPENDENCIES_AVAILABLE[test_key] is False
    
    # Second call should use cached result
    result2 = check_dependency('nonexistent_module_for_testing', test_key)
    assert result2 is False


def test_require_dependency_with_missing_module():
    """Test that require_dependency raises helpful error for missing modules."""
    from tldw_chatbook.Utils.optional_deps import require_dependency
    
    with pytest.raises(ImportError) as exc_info:
        require_dependency('nonexistent_module_for_testing')
    
    assert 'nonexistent_module_for_testing' in str(exc_info.value)
    assert 'pip install' in str(exc_info.value)


def test_get_safe_import_with_missing_module():
    """Test that get_safe_import returns None for missing modules."""
    from tldw_chatbook.Utils.optional_deps import get_safe_import
    
    result = get_safe_import('nonexistent_module_for_testing')
    assert result is None


def test_get_safe_import_with_existing_module():
    """Test that get_safe_import returns module for existing modules."""
    from tldw_chatbook.Utils.optional_deps import get_safe_import
    
    # Test with a built-in module that should always exist
    result = get_safe_import('os')
    assert result is not None
    assert hasattr(result, 'path')  # os module should have 'path' attribute


def test_embeddings_rag_deps_missing():
    """Test embeddings/RAG dependency checking when dependencies are missing."""
    # Mock the __import__ function to simulate missing modules
    modules_to_fail = {'torch', 'transformers', 'numpy', 'chromadb', 'sentence_transformers'}
    original_import = __builtins__['__import__']
    
    def mock_import(name, *args, **kwargs):
        if name in modules_to_fail:
            raise ModuleNotFoundError(f"No module named '{name}'")
        return original_import(name, *args, **kwargs)
    
    try:
        # Re-import and re-initialize the optional_deps module
        if 'tldw_chatbook.Utils.optional_deps' in sys.modules:
            del sys.modules['tldw_chatbook.Utils.optional_deps']
        
        # Patch the import
        __builtins__['__import__'] = mock_import
        
        from tldw_chatbook.Utils.optional_deps import DEPENDENCIES_AVAILABLE, check_embeddings_rag_deps, reset_dependency_checks
        
        # Reset the dependency checks to clear cached results
        reset_dependency_checks()
        
        # Should detect missing dependencies
        result = check_embeddings_rag_deps()
        assert result is False
        assert DEPENDENCIES_AVAILABLE.get('embeddings_rag', False) is False
        
    finally:
        # Restore original import
        __builtins__['__import__'] = original_import


def test_embeddings_lib_graceful_failure():
    """Test that Embeddings_Lib handles missing dependencies gracefully."""
    # This test verifies the module can be imported even if dependencies are missing
    try:
        # Clear optional dependency availability to simulate missing deps
        from tldw_chatbook.Utils.optional_deps import DEPENDENCIES_AVAILABLE
        original_rag_available = DEPENDENCIES_AVAILABLE.get('embeddings_rag')
        DEPENDENCIES_AVAILABLE['embeddings_rag'] = False
        
        # Test that EmbeddingFactory raises helpful error
        from tldw_chatbook.Embeddings.Embeddings_Lib import EmbeddingFactory
        
        with pytest.raises(ImportError) as exc_info:
            EmbeddingFactory({})
        
        assert 'embeddings/RAG dependencies' in str(exc_info.value)
        assert 'pip install' in str(exc_info.value)
        
    finally:
        # Restore original state
        if original_rag_available is not None:
            DEPENDENCIES_AVAILABLE['embeddings_rag'] = original_rag_available


def test_chroma_lib_graceful_failure():
    """Test that Chroma_Lib handles missing dependencies gracefully."""
    try:
        # Clear optional dependency availability to simulate missing deps
        from tldw_chatbook.Utils.optional_deps import DEPENDENCIES_AVAILABLE
        original_rag_available = DEPENDENCIES_AVAILABLE.get('embeddings_rag')
        DEPENDENCIES_AVAILABLE['embeddings_rag'] = False
        
        # Test that ChromaDBManager raises helpful error
        from tldw_chatbook.Embeddings.Chroma_Lib import ChromaDBManager
        
        with pytest.raises(ImportError) as exc_info:
            ChromaDBManager("test_user", {"test": "config"})
        
        assert 'embeddings/RAG dependencies' in str(exc_info.value)
        assert 'pip install' in str(exc_info.value)
        
    finally:
        # Restore original state
        if original_rag_available is not None:
            DEPENDENCIES_AVAILABLE['embeddings_rag'] = original_rag_available


if __name__ == "__main__":
    pytest.main([__file__])