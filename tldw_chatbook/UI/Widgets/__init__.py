# __init__.py
# Description: UI Widgets module
#
"""
UI Widgets
----------

Reusable widget components for the application.
"""

from .SmartContentTree import SmartContentTree, ContentNodeData, ContentSelectionChanged

# Optional imports
try:
    from .MindmapViewer import MindmapViewer, MindmapNodeSelected
    MINDMAP_AVAILABLE = True
except ImportError:
    MINDMAP_AVAILABLE = False
    MindmapViewer = None
    MindmapNodeSelected = None

__all__ = [
    'SmartContentTree',
    'ContentNodeData',
    'ContentSelectionChanged',
]

if MINDMAP_AVAILABLE:
    __all__.extend(['MindmapViewer', 'MindmapNodeSelected'])