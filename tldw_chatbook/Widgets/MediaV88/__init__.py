"""
MediaV88 Widget Components.

This module contains the rebuilt media UI components following Textual best practices.
"""

# Use OptionList version for better dropdown rendering
try:
    from .navigation_column_optionlist import NavigationColumn
except ImportError:
    # Fallback to original if OptionList version has issues
    from .navigation_column import NavigationColumn
from .search_bar import SearchBar
from .metadata_panel import MetadataPanel
from .content_viewer_tabs import ContentViewerTabs

__all__ = [
    'NavigationColumn',
    'SearchBar', 
    'MetadataPanel',
    'ContentViewerTabs',
]