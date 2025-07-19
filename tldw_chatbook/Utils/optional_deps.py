# optional_deps.py
# Central module for checking availability of optional dependencies
#
import sys
import importlib.util
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
    'lightning_whisper_mlx': False,
    'parakeet_mlx': False,
    'yt_dlp': False,
    'soundfile': False,
    'scipy': False,
    'qwen2audio': False,
    # PDF processing
    'pdf_processing': False,
    'pymupdf': False,
    'pymupdf4llm': False,
    'docling': False,
    # E-book processing
    'ebook_processing': False,
    'ebooklib': False,
    'defusedxml': False,
    'html2text': False,
    'lxml': False,
    'beautifulsoup4': False,
    # Web scraping - additional
    'pandas': False,
    'playwright': False,
    'trafilatura': False,
    'aiohttp': False,
    # Local LLM
    'local_llm': False,
    'mlx_lm': False,
    'vllm': False,
    'onnxruntime': False,
    # MCP
    'mcp': False,
    # TTS
    'tts_processing': False,
    'kokoro_onnx': False,
    'pydub': False,
    'pyaudio': False,
    'av': False,
    # STT
    'stt_processing': False,
    'nemo_toolkit': False,
    # OCR
    'ocr_processing': False,
    'docext': False,
    'gradio_client': False,
    'openai': False,
    # Image processing
    'image_processing': False,
    'PIL': False,
    'pillow': False,
    'textual_image': False,
    'rich_pixels': False,
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
    # Check if already determined
    if 'embeddings_rag' in DEPENDENCIES_AVAILABLE:
        return DEPENDENCIES_AVAILABLE['embeddings_rag']
    
    # Don't actually import heavy deps unless explicitly checking
    # Just check if they would be importable
    required_deps = ['torch', 'transformers', 'numpy', 'chromadb', 'sentence_transformers']
    all_available = True
    
    for dep in required_deps:
        try:
            # Use importlib to check without actually importing
            import importlib.util
            spec = importlib.util.find_spec(dep)
            if spec is None:
                all_available = False
                logger.debug(f"⚠️ {dep} not found (lazy check)")
            else:
                logger.debug(f"✅ {dep} is available (lazy check)")
        except (ImportError, ModuleNotFoundError):
            all_available = False
            logger.debug(f"⚠️ {dep} not found (lazy check)")
    
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
    
    # Check lightning-whisper-mlx (macOS only)
    lightning_whisper_available = False
    if sys.platform == 'darwin':
        lightning_whisper_available = check_dependency('lightning_whisper_mlx')
    else:
        DEPENDENCIES_AVAILABLE['lightning_whisper_mlx'] = False
    
    # Check parakeet-mlx (macOS only)
    parakeet_mlx_available = False
    if sys.platform == 'darwin':
        parakeet_mlx_available = check_dependency('parakeet_mlx')
    else:
        DEPENDENCIES_AVAILABLE['parakeet_mlx'] = False
    
    # Check optional transcription backend
    qwen2audio_available = check_dependency('qwen2_audio', 'qwen2audio')
    
    # Check download capability
    yt_dlp_available = check_dependency('yt_dlp')
    
    # Overall audio processing available if core deps are present
    audio_available = core_available
    DEPENDENCIES_AVAILABLE['audio_processing'] = audio_available
    DEPENDENCIES_AVAILABLE['faster_whisper'] = faster_whisper_available
    DEPENDENCIES_AVAILABLE['lightning_whisper_mlx'] = lightning_whisper_available
    DEPENDENCIES_AVAILABLE['parakeet_mlx'] = parakeet_mlx_available
    DEPENDENCIES_AVAILABLE['qwen2audio'] = qwen2audio_available
    DEPENDENCIES_AVAILABLE['yt_dlp'] = yt_dlp_available
    
    if audio_available:
        logger.info("✅ Core audio processing dependencies found.")
        enhanced_features = []
        if parakeet_mlx_available:
            enhanced_features.append("parakeet-mlx (Real-time ASR for Apple Silicon)")
        if lightning_whisper_available:
            enhanced_features.append("lightning-whisper-mlx (Apple Silicon optimized)")
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

def check_pdf_processing_deps() -> bool:
    """Check dependencies needed for PDF processing functionality."""
    pymupdf_available = check_dependency('pymupdf')
    pymupdf4llm_available = check_dependency('pymupdf4llm')
    docling_available = check_dependency('docling')
    
    pdf_available = pymupdf_available
    DEPENDENCIES_AVAILABLE['pdf_processing'] = pdf_available
    
    if pdf_available:
        logger.info("✅ PDF processing dependencies found.")
        enhanced_features = []
        if pymupdf4llm_available:
            enhanced_features.append("PyMuPDF4LLM advanced extraction")
        if docling_available:
            enhanced_features.append("Docling document processing")
        if enhanced_features:
            logger.info(f"✅ Enhanced PDF features available: {', '.join(enhanced_features)}")
    else:
        logger.warning("⚠️ PDF processing dependencies missing.")
    
    return pdf_available

def check_ebook_processing_deps() -> bool:
    """Check dependencies needed for e-book processing functionality."""
    ebooklib_available = check_dependency('ebooklib')
    defusedxml_available = check_dependency('defusedxml')
    html2text_available = check_dependency('html2text')
    
    lxml_available = check_dependency('lxml')
    bs4_available = check_dependency('bs4', 'beautifulsoup4')
    
    ebook_available = ebooklib_available and defusedxml_available
    DEPENDENCIES_AVAILABLE['ebook_processing'] = ebook_available
    
    if ebook_available:
        logger.info("✅ E-book processing dependencies found.")
        enhanced_features = []
        if html2text_available:
            enhanced_features.append("HTML to text conversion")
        if lxml_available:
            enhanced_features.append("Advanced XML parsing")
        if bs4_available:
            enhanced_features.append("BeautifulSoup HTML parsing")
        if enhanced_features:
            logger.info(f"✅ Enhanced e-book features available: {', '.join(enhanced_features)}")
    else:
        logger.warning("⚠️ E-book processing dependencies missing.")
    
    return ebook_available

def check_ocr_deps() -> bool:
    """Check dependencies needed for OCR functionality."""
    docext_available = check_dependency('docext')
    gradio_client_available = check_dependency('gradio_client')
    openai_available = check_dependency('openai')
    
    ocr_available = docext_available or gradio_client_available or openai_available
    DEPENDENCIES_AVAILABLE['ocr_processing'] = ocr_available
    
    if ocr_available:
        logger.info("✅ OCR processing dependencies found.")
        available_backends = []
        if docext_available:
            available_backends.append("Docext")
        if gradio_client_available:
            available_backends.append("Gradio Client")
        if openai_available:
            available_backends.append("OpenAI Vision")
        logger.info(f"✅ Available OCR backends: {', '.join(available_backends)}")
    else:
        logger.warning("⚠️ OCR processing dependencies missing.")
    
    return ocr_available

def check_local_llm_deps() -> bool:
    """Check dependencies needed for local LLM functionality."""
    mlx_available = check_dependency('mlx_lm', 'mlx_lm')
    vllm_available = check_dependency('vllm')
    onnx_available = check_dependency('onnxruntime')
    
    local_llm_available = mlx_available or vllm_available or onnx_available
    DEPENDENCIES_AVAILABLE['local_llm'] = local_llm_available
    
    if local_llm_available:
        logger.info("✅ Local LLM dependencies found.")
        available_backends = []
        if mlx_available:
            available_backends.append("MLX-LM (Apple Silicon)")
        if vllm_available:
            available_backends.append("vLLM")
        if onnx_available:
            available_backends.append("ONNX Runtime")
        logger.info(f"✅ Available local LLM backends: {', '.join(available_backends)}")
    else:
        logger.warning("⚠️ Local LLM dependencies missing.")
    
    return local_llm_available

def check_tts_deps() -> bool:
    """Check dependencies needed for TTS functionality."""
    kokoro_available = check_dependency('kokoro_onnx', 'kokoro_onnx')
    pydub_available = check_dependency('pydub')
    pyaudio_available = check_dependency('pyaudio')
    av_available = check_dependency('av')
    
    tts_available = kokoro_available or (pydub_available and pyaudio_available)
    DEPENDENCIES_AVAILABLE['tts_processing'] = tts_available
    
    if tts_available:
        logger.info("✅ TTS processing dependencies found.")
        available_features = []
        if kokoro_available:
            available_features.append("Kokoro TTS")
        if pydub_available and pyaudio_available:
            available_features.append("Audio playback")
        if av_available:
            available_features.append("Advanced audio format support")
        logger.info(f"✅ Available TTS features: {', '.join(available_features)}")
    else:
        logger.warning("⚠️ TTS processing dependencies missing.")
    
    return tts_available

def check_stt_deps() -> bool:
    """Check dependencies needed for STT functionality."""
    nemo_available = check_dependency('nemo_toolkit')
    
    faster_whisper_available = DEPENDENCIES_AVAILABLE.get('faster_whisper', False)
    
    stt_available = nemo_available or faster_whisper_available
    DEPENDENCIES_AVAILABLE['stt_processing'] = stt_available
    
    if stt_available:
        logger.info("✅ STT processing dependencies found.")
        available_backends = []
        if nemo_available:
            available_backends.append("NVIDIA NeMo")
        if faster_whisper_available:
            available_backends.append("Faster Whisper")
        logger.info(f"✅ Available STT backends: {', '.join(available_backends)}")
    else:
        logger.warning("⚠️ STT processing dependencies missing.")
    
    return stt_available

def check_image_processing_deps() -> bool:
    """Check dependencies needed for image processing functionality."""
    try:
        import PIL
        pil_available = True
        DEPENDENCIES_AVAILABLE['PIL'] = True
        DEPENDENCIES_AVAILABLE['pillow'] = True
    except ImportError:
        pil_available = False
        DEPENDENCIES_AVAILABLE['PIL'] = False
        DEPENDENCIES_AVAILABLE['pillow'] = False
    
    textual_image_available = check_dependency('textual_image')
    rich_pixels_available = check_dependency('rich_pixels')
    
    image_available = pil_available
    DEPENDENCIES_AVAILABLE['image_processing'] = image_available
    
    if image_available:
        logger.info("✅ Image processing dependencies found.")
        enhanced_features = []
        if textual_image_available:
            enhanced_features.append("Terminal graphics protocol support")
        if rich_pixels_available:
            enhanced_features.append("Rich pixels rendering")
        if enhanced_features:
            logger.info(f"✅ Enhanced image features available: {', '.join(enhanced_features)}")
    else:
        logger.warning("⚠️ Image processing dependencies missing.")
    
    return image_available

def check_mcp_deps() -> bool:
    """Check dependencies needed for MCP functionality."""
    mcp_available = check_dependency('mcp')
    DEPENDENCIES_AVAILABLE['mcp'] = mcp_available
    
    if mcp_available:
        logger.info("✅ MCP (Model Context Protocol) dependencies found.")
    else:
        logger.warning("⚠️ MCP dependencies missing.")
    
    return mcp_available

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
        'lightning_whisper_mlx': False,
        'parakeet_mlx': False,
        'yt_dlp': False,
        'soundfile': False,
        'scipy': False,
        'qwen2audio': False,
        # PDF processing
        'pdf_processing': False,
        'pymupdf': False,
        'pymupdf4llm': False,
        'docling': False,
        # E-book processing
        'ebook_processing': False,
        'ebooklib': False,
        'defusedxml': False,
        'html2text': False,
        'lxml': False,
        'beautifulsoup4': False,
        # Web scraping - additional
        'pandas': False,
        'playwright': False,
        'trafilatura': False,
        'aiohttp': False,
        # Local LLM
        'local_llm': False,
        'mlx_lm': False,
        'vllm': False,
        'onnxruntime': False,
        # MCP
        'mcp': False,
        # TTS
        'tts_processing': False,
        'kokoro_onnx': False,
        'pydub': False,
        'pyaudio': False,
        'av': False,
        # STT
        'stt_processing': False,
        'nemo_toolkit': False,
        # OCR
        'ocr_processing': False,
        'docext': False,
        'gradio_client': False,
        'openai': False,
        # Image processing
        'image_processing': False,
        'PIL': False,
        'pillow': False,
        'textual_image': False,
        'rich_pixels': False,
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
    
    # Check new dependency groups
    check_pdf_processing_deps()
    check_ebook_processing_deps()
    check_ocr_deps()
    check_local_llm_deps()
    check_tts_deps()
    check_stt_deps()
    check_image_processing_deps()
    check_mcp_deps()
    
    # Log summary
    enabled_features = [name for name, available in DEPENDENCIES_AVAILABLE.items() if available]
    disabled_features = [name for name, available in DEPENDENCIES_AVAILABLE.items() if not available]
    
    if enabled_features:
        logger.info(f"✅ Available features: {', '.join(enabled_features)}")
    if disabled_features:
        logger.info(f"⚠️ Disabled features: {', '.join(disabled_features)}")
    
    logger.info("Dependency check complete.")

# Lazy initialization - dependencies will be checked on first use
import os

# Track if we've initialized
_initialized = False

def ensure_dependencies_checked():
    """Ensure dependencies have been checked at least once."""
    global _initialized
    if not _initialized and 'PYTEST_CURRENT_TEST' not in os.environ:
        initialize_dependency_checks()
        _initialized = True
    
# Check configuration for eager dependency checking
eager_check = False

# First check environment variable (highest priority)
if os.environ.get('TLDW_EAGER_DEPENDENCY_CHECK', '').lower() == 'true':
    eager_check = True
    logger.info("Eager dependency checking enabled via TLDW_EAGER_DEPENDENCY_CHECK environment variable")
else:
    # Check config file if not in test environment
    if 'PYTEST_CURRENT_TEST' not in os.environ:
        try:
            # Import config lazily to avoid circular dependencies
            from ..config import get_cli_setting
            if get_cli_setting("rag", "performance", {}).get("eager_dependency_check", False):
                eager_check = True
                logger.info("Eager dependency checking enabled via config file")
        except Exception as e:
            logger.debug(f"Could not check config for eager_dependency_check: {e}")

if eager_check:
    initialize_dependency_checks()
    _initialized = True
elif 'PYTEST_CURRENT_TEST' in os.environ:
    logger.debug("Skipping auto-initialization in test environment")
else:
    logger.info("Lazy dependency checking enabled (default). Dependencies will be checked on first use.")