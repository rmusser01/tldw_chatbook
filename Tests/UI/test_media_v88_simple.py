"""
Simple test to verify MediaWindowV88 works.
"""

import pytest
from unittest.mock import Mock

def test_media_window_imports():
    """Test that MediaWindowV88 can be imported."""
    from tldw_chatbook.UI.MediaWindowV88 import MediaWindowV88
    assert MediaWindowV88 is not None

def test_media_window_instantiation():
    """Test that MediaWindowV88 can be instantiated."""
    from tldw_chatbook.UI.MediaWindowV88 import MediaWindowV88
    
    # Create mock app
    mock_app = Mock()
    mock_app.media_db = Mock()
    mock_app.media_db.search_media_db = Mock(return_value=([], 0))
    mock_app.media_db.get_media_by_id = Mock(return_value=None)
    mock_app.notes_db = Mock()
    mock_app.app_config = {}
    mock_app.notify = Mock()
    mock_app.loguru_logger = Mock()
    mock_app._media_types_for_ui = ["All Media"]
    
    # Create window
    window = MediaWindowV88(mock_app)
    
    # Check basic properties
    assert window.app_instance == mock_app
    assert window.active_media_type is None
    assert window.selected_media_id is None

def test_component_imports():
    """Test that all components can be imported."""
    from tldw_chatbook.Widgets.Media import (
        MediaNavigationPanel,
        MediaSearchPanel,
        MediaListPanel,
        MediaViewerPanel,
    )
    
    assert MediaNavigationPanel is not None
    assert MediaSearchPanel is not None
    assert MediaListPanel is not None
    assert MediaViewerPanel is not None

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
