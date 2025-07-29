# tldw_chatbook/Widgets/HuggingFace/__init__.py
"""
HuggingFace integration widgets for browsing and downloading GGUF models.
"""

from .model_browser_widget import HuggingFaceModelBrowser
from .model_search_widget import ModelSearchWidget
from .model_card_viewer import ModelCardViewer
from .download_manager import DownloadManager
from .local_models_widget import LocalModelsWidget

__all__ = [
    "HuggingFaceModelBrowser",
    "ModelSearchWidget", 
    "ModelCardViewer",
    "DownloadManager",
    "LocalModelsWidget"
]