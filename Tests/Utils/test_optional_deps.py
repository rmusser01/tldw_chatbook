# test_optional_deps.py
# Test cases for optional dependency handling
#
import pytest
import sys
import tomllib
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[2]


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


def test_optional_feature_metadata_exposes_recovery_commands_and_capability_tiers():
    """Optional extras expose source-honest recovery metadata for blocked features."""
    from tldw_chatbook.Utils.optional_deps import get_optional_feature_info

    rag = get_optional_feature_info("embeddings_rag")
    assert rag.extra == "embeddings_rag"
    assert rag.label == "Library/Search-RAG"
    assert rag.capability_tier == "advanced"
    assert rag.feature_area == "RAG and retrieval"
    assert rag.owner == "Library Search/RAG"
    assert rag.source_install_command == 'pip install -e ".[embeddings_rag]"'
    assert rag.package_install_command == 'pip install "tldw_chatbook[embeddings_rag]"'
    assert rag.recovery_action == "Settings > RAG"
    assert rag.unavailable_what == "Search/RAG queries"

    web = get_optional_feature_info("web")
    assert web.label == "Web server"
    assert web.feature_area == "Web access"
    assert web.owner == "Web/browser serving"
    assert web.source_install_command == 'pip install -e ".[web]"'


def test_optional_feature_metadata_covers_pyproject_extras():
    """Every declared optional extra has source-honest recovery metadata."""
    from tldw_chatbook.Utils.optional_deps import OPTIONAL_FEATURES, get_optional_feature_info

    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    declared_extras = set(pyproject["project"]["optional-dependencies"])

    assert set(OPTIONAL_FEATURES) == declared_extras
    for extra in declared_extras:
        info = get_optional_feature_info(extra)
        assert info.extra == extra
        assert info.owner != "optional dependency"
        assert info.feature_area
        assert info.recovery_action
        assert info.unavailable_what


def test_optional_feature_metadata_groups_release_capabilities_without_core_install():
    """The optional matrix separates baseline install from advanced feature groups."""
    from tldw_chatbook.Utils.optional_deps import (
        LOCAL_FIRST_BASELINE_INSTALL_COMMAND,
        optional_feature_groups_by_area,
    )

    groups = optional_feature_groups_by_area()
    assert LOCAL_FIRST_BASELINE_INSTALL_COMMAND == "pip install -e ."
    assert "RAG and retrieval" in groups
    assert "Media ingestion and transcription" in groups
    assert "MCP integration" in groups
    assert "Local inference" in groups
    assert "Web access" in groups
    assert "embeddings_rag" in groups["RAG and retrieval"]
    assert "mcp" in groups["MCP integration"]
    assert "web" in groups["Web access"]


def test_subscriptions_deps_require_beautifulsoup_for_route_import(monkeypatch):
    """The subscriptions route guard blocks imports when BeautifulSoup is absent."""
    from tldw_chatbook.Utils import optional_deps

    optional_deps.reset_dependency_checks()
    seen: list[tuple[str, str]] = []

    def fake_check_dependency(module_name: str, feature_name: str | None = None) -> bool:
        dependency_key = feature_name or module_name
        seen.append((module_name, dependency_key))
        is_available = module_name != "bs4"
        optional_deps.DEPENDENCIES_AVAILABLE[dependency_key] = is_available
        return is_available

    monkeypatch.setattr(optional_deps, "check_dependency", fake_check_dependency)

    assert optional_deps.check_subscriptions_deps() is False
    assert ("bs4", "beautifulsoup4") in seen
    assert optional_deps.DEPENDENCIES_AVAILABLE["subscriptions"] is False


def test_subscriptions_extra_declares_beautifulsoup_dependency():
    """The install extra matches the route guard's BeautifulSoup dependency."""
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert "beautifulsoup4" in pyproject["project"]["optional-dependencies"]["subscriptions"]


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


def test_pdf_processing_deps():
    """Test PDF processing dependency checking."""
    from tldw_chatbook.Utils.optional_deps import check_pdf_processing_deps, DEPENDENCIES_AVAILABLE
    
    # The result depends on what's installed, but function should always run
    result = check_pdf_processing_deps()
    assert isinstance(result, bool)
    assert 'pdf_processing' in DEPENDENCIES_AVAILABLE
    assert 'pymupdf' in DEPENDENCIES_AVAILABLE
    assert 'pymupdf4llm' in DEPENDENCIES_AVAILABLE
    assert 'docling' in DEPENDENCIES_AVAILABLE


def test_ebook_processing_deps():
    """Test e-book processing dependency checking."""
    from tldw_chatbook.Utils.optional_deps import check_ebook_processing_deps, DEPENDENCIES_AVAILABLE
    
    result = check_ebook_processing_deps()
    assert isinstance(result, bool)
    assert 'ebook_processing' in DEPENDENCIES_AVAILABLE
    assert 'ebooklib' in DEPENDENCIES_AVAILABLE
    assert 'html2text' in DEPENDENCIES_AVAILABLE
    assert 'defusedxml' in DEPENDENCIES_AVAILABLE


def test_web_scraping_deps():
    """Test web scraping dependency checking."""
    from tldw_chatbook.Utils.optional_deps import check_websearch_deps, DEPENDENCIES_AVAILABLE
    
    result = check_websearch_deps()
    assert isinstance(result, bool)
    assert 'websearch' in DEPENDENCIES_AVAILABLE
    assert 'websearch_core' in DEPENDENCIES_AVAILABLE


def test_ocr_deps():
    """Test OCR dependency checking."""
    from tldw_chatbook.Utils.optional_deps import check_ocr_deps, DEPENDENCIES_AVAILABLE
    
    result = check_ocr_deps()
    assert isinstance(result, bool)
    assert 'ocr_processing' in DEPENDENCIES_AVAILABLE
    # Individual dependencies are checked within the function
    # We just verify the main key is set


def test_local_llm_deps():
    """Test local LLM dependency checking."""
    from tldw_chatbook.Utils.optional_deps import check_local_llm_deps, DEPENDENCIES_AVAILABLE
    
    result = check_local_llm_deps()
    assert isinstance(result, bool)
    assert 'local_llm' in DEPENDENCIES_AVAILABLE
    assert 'vllm' in DEPENDENCIES_AVAILABLE
    assert 'onnxruntime' in DEPENDENCIES_AVAILABLE
    assert 'mlx_lm' in DEPENDENCIES_AVAILABLE


def test_tts_deps():
    """Test TTS dependency checking."""
    from tldw_chatbook.Utils.optional_deps import check_tts_deps, DEPENDENCIES_AVAILABLE
    
    result = check_tts_deps()
    assert isinstance(result, bool)
    assert 'tts_processing' in DEPENDENCIES_AVAILABLE
    # Individual dependencies are checked within the function
    # We just verify the main key is set


def test_stt_deps():
    """Test STT dependency checking."""
    from tldw_chatbook.Utils.optional_deps import check_stt_deps, DEPENDENCIES_AVAILABLE
    
    result = check_stt_deps()
    assert isinstance(result, bool)
    assert 'stt_processing' in DEPENDENCIES_AVAILABLE
    # Individual dependencies are checked within the function
    # We just verify the main key is set


def test_image_processing_deps():
    """Test image processing dependency checking."""
    from tldw_chatbook.Utils.optional_deps import check_image_processing_deps, DEPENDENCIES_AVAILABLE
    
    result = check_image_processing_deps()
    assert isinstance(result, bool)
    assert 'image_processing' in DEPENDENCIES_AVAILABLE
    assert 'pillow' in DEPENDENCIES_AVAILABLE
    assert 'textual_image' in DEPENDENCIES_AVAILABLE
    assert 'rich_pixels' in DEPENDENCIES_AVAILABLE


def test_mcp_deps():
    """Test MCP dependency checking."""
    from tldw_chatbook.Utils.optional_deps import check_mcp_deps, DEPENDENCIES_AVAILABLE
    
    result = check_mcp_deps()
    assert isinstance(result, bool)
    assert 'mcp' in DEPENDENCIES_AVAILABLE


def test_initialize_dependency_checks():
    """Test that initialize_dependency_checks calls all check functions."""
    from tldw_chatbook.Utils.optional_deps import initialize_dependency_checks, DEPENDENCIES_AVAILABLE

    # Clear all dependencies first — and restore afterwards: this module dict
    # is shared process-global state, and initialize_dependency_checks() only
    # repopulates the checked categories, not static registry keys (e.g.
    # 'svg_rendering'), so leaving it cleared poisons later tests.
    snapshot = dict(DEPENDENCIES_AVAILABLE)
    DEPENDENCIES_AVAILABLE.clear()
    try:
        # Initialize should populate all dependencies
        initialize_dependency_checks()

        # Check that all major categories are present with the correct keys
        assert 'embeddings_rag' in DEPENDENCIES_AVAILABLE
        assert 'websearch' in DEPENDENCIES_AVAILABLE  # Changed from web_scraping
        assert 'pdf_processing' in DEPENDENCIES_AVAILABLE
        assert 'ebook_processing' in DEPENDENCIES_AVAILABLE
        assert 'ocr_processing' in DEPENDENCIES_AVAILABLE  # Changed from ocr
        assert 'local_llm' in DEPENDENCIES_AVAILABLE
        assert 'tts_processing' in DEPENDENCIES_AVAILABLE  # Changed from tts
        assert 'stt_processing' in DEPENDENCIES_AVAILABLE  # Changed from stt
        assert 'image_processing' in DEPENDENCIES_AVAILABLE
        assert 'mcp' in DEPENDENCIES_AVAILABLE
    finally:
        DEPENDENCIES_AVAILABLE.update(snapshot)


def test_pdf_processing_lib_import_handling():
    """Test that PDF_Processing_Lib handles missing dependencies gracefully."""
    # Save original state
    from tldw_chatbook.Utils.optional_deps import DEPENDENCIES_AVAILABLE
    original_pdf_available = DEPENDENCIES_AVAILABLE.get('pdf_processing')
    
    try:
        # Simulate missing PDF dependencies
        DEPENDENCIES_AVAILABLE['pdf_processing'] = False
        DEPENDENCIES_AVAILABLE['pymupdf'] = False
        
        # Clear module cache to force re-import
        if 'tldw_chatbook.Local_Ingestion.PDF_Processing_Lib' in sys.modules:
            del sys.modules['tldw_chatbook.Local_Ingestion.PDF_Processing_Lib']
        
        # Import should work but functions should raise errors
        from tldw_chatbook.Local_Ingestion.PDF_Processing_Lib import PDF_PROCESSING_AVAILABLE
        
        # PDF_PROCESSING_AVAILABLE should reflect the actual import status
        # (not our mock, since the module was already imported)
        # So we just check it exists
        assert isinstance(PDF_PROCESSING_AVAILABLE, bool)
        
    finally:
        # Restore original state
        if original_pdf_available is not None:
            DEPENDENCIES_AVAILABLE['pdf_processing'] = original_pdf_available


def test_book_ingestion_lib_import_handling():
    """Test that Book_Ingestion_Lib handles missing dependencies gracefully."""
    # Save original state
    from tldw_chatbook.Utils.optional_deps import DEPENDENCIES_AVAILABLE
    original_ebook_available = DEPENDENCIES_AVAILABLE.get('ebook_processing')
    
    try:
        # Simulate missing ebook dependencies
        DEPENDENCIES_AVAILABLE['ebook_processing'] = False
        DEPENDENCIES_AVAILABLE['ebooklib'] = False
        
        # Clear module cache to force re-import
        if 'tldw_chatbook.Local_Ingestion.Book_Ingestion_Lib' in sys.modules:
            del sys.modules['tldw_chatbook.Local_Ingestion.Book_Ingestion_Lib']
        
        # Import should work but functions should handle missing deps
        from tldw_chatbook.Local_Ingestion.Book_Ingestion_Lib import EBOOK_PROCESSING_AVAILABLE
        
        # Check that the availability flag exists
        assert isinstance(EBOOK_PROCESSING_AVAILABLE, bool)
        
    finally:
        # Restore original state
        if original_ebook_available is not None:
            DEPENDENCIES_AVAILABLE['ebook_processing'] = original_ebook_available


if __name__ == "__main__":
    pytest.main([__file__])
