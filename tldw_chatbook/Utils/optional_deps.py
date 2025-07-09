# optional_deps.py
# Central module for checking availability of optional dependencies
#
import sys
from typing import Dict, Any, Optional, Callable
from loguru import logger

# Global flags for optional dependency availability
DEPENDENCIES_AVAILABLE = {
    'torch': False,
    'transformers': False,
    'numpy': False,
    'chromadb': False,
    'embeddings_rag': False,
    'websearch': False,
    'jieba': False,
    'fugashi': False,
    'flashrank': False,
    'sentence_transformers': False,
    'chunker': False,
    'chinese_chunking': False,
    'japanese_chunking': False,
    'token_chunking': False,
    'cohere': False,
    # Audio/Video processing
    'audio_processing': False,
    'video_processing': False,
    'faster_whisper': False,
    'yt_dlp': False,
    'soundfile': False,
    'scipy': False,
    'qwen2audio': False,
}

# Store actual modules for conditional use
MODULES = {}

# Store placeholder functions for unavailable features
PLACEHOLDERS = {}

def check_dependency(module_name: str, feature_name: Optional[str] = None) -> bool:
    """
    Check if a dependency is available and cache the result.
    
    Args:
        module_name: The module to import (e.g., 'torch')
        feature_name: Optional feature name for grouped dependencies
        
    Returns:
        bool: True if the dependency is available
    """
    if feature_name is None:
        feature_name = module_name
    
    try:
        module = __import__(module_name)
        MODULES[module_name] = module
        DEPENDENCIES_AVAILABLE[feature_name] = True
        logger.debug(f"✅ {module_name} dependency found. Feature '{feature_name}' is enabled.")
        return True
    except (ImportError, ModuleNotFoundError) as e:
        DEPENDENCIES_AVAILABLE[feature_name] = False
        logger.debug(f"⚠️ {module_name} dependency not found. Feature '{feature_name}' will be disabled. Reason: {e}")
        return False

def check_embeddings_rag_deps() -> bool:
    """Check all dependencies needed for embeddings and RAG functionality."""
    required_deps = ['torch', 'transformers', 'numpy', 'chromadb', 'sentence_transformers']
    all_available = True
    
    for dep in required_deps:
        if not check_dependency(dep):
            all_available = False
    
    DEPENDENCIES_AVAILABLE['embeddings_rag'] = all_available
    if all_available:
        logger.info("✅ All embeddings/RAG dependencies found. Features are enabled.")
    else:
        logger.warning("⚠️ Some embeddings/RAG dependencies missing. Features will be disabled.")
    
    return all_available

def check_websearch_deps() -> bool:
    """Check dependencies needed for web search functionality."""
    # Based on pyproject.toml websearch optional dependencies
    required_deps = ['lxml', 'bs4', 'pandas', 'playwright', 'trafilatura', 'langdetect', 'nltk', 'scikit-learn']
    optional_deps = ['playwright_stealth']  # This one was commented out
    
    essential_available = True
    for dep in ['lxml', 'bs4', 'trafilatura', 'langdetect']:  # Core web scraping deps
        if not check_dependency(dep, 'websearch_core'):
            essential_available = False
            break
    
    DEPENDENCIES_AVAILABLE['websearch'] = essential_available
    return essential_available

def check_chunker_deps() -> bool:
    """Check dependencies needed for enhanced chunking functionality."""
    # Core chunking deps that are always useful
    core_deps = ['langdetect', 'nltk']
    sklearn_available = check_dependency('sklearn', 'scikit-learn')
    all_core_available = all(check_dependency(dep) for dep in core_deps) and sklearn_available
    
    # Language-specific deps
    chinese_available = check_dependency('jieba', 'chinese_chunking')
    japanese_available = check_dependency('fugashi', 'japanese_chunking')
    token_available = check_dependency('transformers', 'token_chunking')
    
    # Overall chunker feature available if core deps are present
    chunker_available = all_core_available
    DEPENDENCIES_AVAILABLE['chunker'] = chunker_available
    DEPENDENCIES_AVAILABLE['chinese_chunking'] = chinese_available
    DEPENDENCIES_AVAILABLE['japanese_chunking'] = japanese_available
    DEPENDENCIES_AVAILABLE['token_chunking'] = token_available
    
    if chunker_available:
        logger.info("✅ Core chunking dependencies found.")
        enhanced_features = []
        if chinese_available:
            enhanced_features.append("Chinese")
        if japanese_available:
            enhanced_features.append("Japanese")
        if token_available:
            enhanced_features.append("Token-based")
        if enhanced_features:
            logger.info(f"✅ Enhanced chunking available for: {', '.join(enhanced_features)}")
    else:
        logger.warning("⚠️ Some core chunking dependencies missing.")
    
    return chunker_available

def get_safe_import(module_name: str, feature_name: Optional[str] = None):
    """
    Get a module if available, otherwise return None.
    
    Args:
        module_name: The module to import
        feature_name: Optional feature name for grouped dependencies
        
    Returns:
        The imported module or None if not available
    """
    if feature_name is None:
        feature_name = module_name
        
    if check_dependency(module_name, feature_name):
        return MODULES.get(module_name)
    return None

def require_dependency(module_name: str, feature_name: Optional[str] = None) -> Any:
    """
    Require a dependency and raise an informative error if not available.
    
    Args:
        module_name: The module to import
        feature_name: Optional feature name for grouped dependencies
        
    Returns:
        The imported module
        
    Raises:
        ImportError: If the dependency is not available
    """
    if feature_name is None:
        feature_name = module_name
        
    if not check_dependency(module_name, feature_name):
        raise ImportError(
            f"Required dependency '{module_name}' for feature '{feature_name}' is not available. "
            f"Please install the optional dependencies with: pip install tldw_chatbook[{feature_name}]"
        )
    
    return MODULES[module_name]

def check_audio_processing_deps() -> bool:
    """Check dependencies needed for audio processing functionality."""
    # Core audio processing dependencies
    core_deps = ['soundfile', 'scipy']
    transcription_deps = ['faster_whisper']
    download_deps = ['yt_dlp']
    
    # Check core deps
    core_available = all(check_dependency(dep) for dep in core_deps)
    
    # Check transcription deps
    faster_whisper_available = check_dependency('faster_whisper')
    
    # Check optional transcription backend
    qwen2audio_available = check_dependency('qwen2_audio', 'qwen2audio')
    
    # Check download capability
    yt_dlp_available = check_dependency('yt_dlp')
    
    # Overall audio processing available if core deps are present
    audio_available = core_available
    DEPENDENCIES_AVAILABLE['audio_processing'] = audio_available
    DEPENDENCIES_AVAILABLE['faster_whisper'] = faster_whisper_available
    DEPENDENCIES_AVAILABLE['qwen2audio'] = qwen2audio_available
    DEPENDENCIES_AVAILABLE['yt_dlp'] = yt_dlp_available
    
    if audio_available:
        logger.info("✅ Core audio processing dependencies found.")
        enhanced_features = []
        if faster_whisper_available:
            enhanced_features.append("faster-whisper transcription")
        if qwen2audio_available:
            enhanced_features.append("Qwen2Audio transcription")
        if yt_dlp_available:
            enhanced_features.append("YouTube/URL downloading")
        if enhanced_features:
            logger.info(f"✅ Enhanced audio features available: {', '.join(enhanced_features)}")
    else:
        logger.warning("⚠️ Some core audio processing dependencies missing.")
    
    return audio_available

def check_video_processing_deps() -> bool:
    """Check dependencies needed for video processing functionality."""
    # Video processing reuses audio processing capabilities
    audio_available = DEPENDENCIES_AVAILABLE.get('audio_processing', False)
    if not audio_available:
        # Check audio deps if not already checked
        check_audio_processing_deps()
        audio_available = DEPENDENCIES_AVAILABLE.get('audio_processing', False)
    
    # Additional video-specific deps (if any)
    # Currently video processing mainly relies on audio processing + ffmpeg (external)
    
    video_available = audio_available
    DEPENDENCIES_AVAILABLE['video_processing'] = video_available
    
    if video_available:
        logger.info("✅ Video processing dependencies found (via audio processing).")
    else:
        logger.warning("⚠️ Video processing unavailable due to missing audio dependencies.")
    
    return video_available

def create_unavailable_feature_handler(feature_name: str, suggestion: str = "") -> Callable:
    """
    Create a function that raises an informative error when a feature is unavailable.
    
    Args:
        feature_name: Name of the unavailable feature
        suggestion: Optional installation suggestion
        
    Returns:
        A function that raises an error with helpful information
    """
    def handler(*args, **kwargs):
        install_hint = f" Install with: {suggestion}" if suggestion else ""
        raise ImportError(
            f"Feature '{feature_name}' is not available due to missing dependencies.{install_hint}"
        )
    return handler

# Initialize dependency checks
def reset_dependency_checks():
    """Reset all dependency checks - useful for testing."""
    global DEPENDENCIES_AVAILABLE, MODULES
    DEPENDENCIES_AVAILABLE = {
        'torch': False,
        'transformers': False,
        'numpy': False,
        'chromadb': False,
        'embeddings_rag': False,
        'websearch': False,
        'jieba': False,
        'fugashi': False,
        'flashrank': False,
        'sentence_transformers': False,
        'chunker': False,
        'chinese_chunking': False,
        'japanese_chunking': False,
        'token_chunking': False,
        'cohere': False,
        # Audio/Video processing
        'audio_processing': False,
        'video_processing': False,
        'faster_whisper': False,
        'yt_dlp': False,
        'soundfile': False,
        'scipy': False,
        'qwen2audio': False,
    }
    MODULES = {}
    logger.debug("Reset dependency checks")

def initialize_dependency_checks():
    """Initialize all dependency checks at startup."""
    logger.info("Checking optional dependencies...")
    
    # Check core optional dependencies
    check_dependency('torch')
    check_dependency('transformers') 
    check_dependency('numpy')
    check_dependency('chromadb')
    check_dependency('sentence_transformers')
    check_dependency('jieba')
    check_dependency('fugashi')
    check_dependency('flashrank')
    check_dependency('cohere')
    
    # Check grouped features
    check_embeddings_rag_deps()
    check_websearch_deps()
    check_chunker_deps()
    check_audio_processing_deps()
    check_video_processing_deps()
    
    # Log summary
    enabled_features = [name for name, available in DEPENDENCIES_AVAILABLE.items() if available]
    disabled_features = [name for name, available in DEPENDENCIES_AVAILABLE.items() if not available]
    
    if enabled_features:
        logger.info(f"✅ Available features: {', '.join(enabled_features)}")
    if disabled_features:
        logger.info(f"⚠️ Disabled features: {', '.join(disabled_features)}")
    
    logger.info("Dependency check complete.")

# Auto-initialize when module is imported unless in test environment
import os
if 'PYTEST_CURRENT_TEST' not in os.environ:
    initialize_dependency_checks()
else:
    logger.debug("Skipping auto-initialization in test environment")